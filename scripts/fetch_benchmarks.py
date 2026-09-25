#!/usr/bin/env python3
"""Fetch public benchmark instances on demand.

Examples:
  python3 scripts/fetch_benchmarks.py --suite netlib --instance afiro
  python3 scripts/fetch_benchmarks.py --suite miplib-easy
  python3 scripts/fetch_benchmarks.py --suite miplib2017 --instance air05
  python3 scripts/fetch_benchmarks.py --all
  python3 scripts/fetch_benchmarks.py --list

Benchmark models are intentionally not stored in Git. Downloads come from the
official Netlib and MIPLIB hosts. Existing non-empty outputs are left alone
unless --force is supplied.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT / "benchmarks"
NETLIB_BASE = "https://www.netlib.org/lp/data"
MIPLIB_BASE = "https://miplib.zib.de/WebData/instances"
MIPLIB3_BASE = "https://miplib2010.zib.de/miplib3/miplib3"
UA = {"User-Agent": "SOR-benchmark-fetch/2.0 (academic benchmark download)"}

NETLIB = """25fv47 80bau3b adlittle afiro agg agg2 agg3 bandm beaconfd blend
bnl1 bnl2 boeing1 boeing2 bore3d brandy capri cycle czprob d2q06c d6cube
degen2 degen3 dfl001 e226 etamacro fffff800 finnis fit1d fit1p fit2d fit2p
forplan ganges gfrd-pnc greenbea greenbeb grow15 grow22 grow7 israel kb2 lotfi
maros maros-r7 modszk1 nesm perold pilot pilot.ja pilot.we pilot4 pilot87
pilotnov recipe sc105 sc205 sc50a sc50b scagr25 scagr7 scfxm1 scfxm2 scfxm3
scorpion scrs8 scsd1 scsd6 scsd8 sctap1 sctap2 sctap3 seba share1b share2b
shell ship04l ship04s ship08l ship08s ship12l ship12s sierra stair standata
standgub standmps stocfor1 stocfor2 tuff vtp.base wood1p woodw""".split()

MIPLIB_EASY = """flugpl gt2 blend2 p0201 markshare1 markshare2 pk1 gen-ip002
gen-ip054 n5-3 assign1-5-8 p0033 enigma stein27 lseu mod008 rgn vpm1 misc03
mod010""".split()

SUITES = ("netlib", "miplib-easy", "miplib2017")


def normalized_name(value: str) -> str:
    name = Path(value).name
    for suffix in (".mps.gz", ".mps"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"invalid instance name: {value!r}")
    return name


def miplib2017_names(metadata: Path) -> list[str]:
    if not metadata.is_file():
        raise RuntimeError(f"missing MIPLIB set file: {metadata}")
    names: list[str] = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(normalized_name(line))
    return names


def suite_names(suite: str, root: Path = DEFAULT_ROOT) -> list[str]:
    if suite == "netlib":
        return list(NETLIB)
    if suite == "miplib-easy":
        return list(MIPLIB_EASY)
    if suite == "miplib2017":
        return miplib2017_names(root / "miplib2017" / "benchmark-v2.test")
    raise ValueError(f"unknown suite: {suite}")


def select_instances(suite: str, requested: list[str], root: Path) -> list[str]:
    available = suite_names(suite, root)
    if not requested:
        return available
    wanted = [normalized_name(x) for x in requested]
    unknown = sorted(set(wanted) - set(available))
    if unknown:
        raise ValueError(
            f"unknown {suite} instance(s): {', '.join(unknown)}; use --list")
    # Keep command-line order while removing duplicates.
    return list(dict.fromkeys(wanted))


def download(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def looks_like_mps(data: bytes) -> bool:
    return any(line.lstrip().startswith(b"NAME") for line in data.splitlines())


def fetch_miplib(name: str, destination: Path, timeout: float, force: bool) -> None:
    if destination.is_file() and destination.stat().st_size and not force:
        print(f"  exists {destination}")
        return
    urls = [
        f"{MIPLIB_BASE}/{name}.mps.gz",
        f"{MIPLIB3_BASE}/{name}.mps.gz",
    ]
    model = b""
    errors: list[str] = []
    for url in urls:
        print(f"  fetch  {url}")
        try:
            payload = download(url, timeout)
            candidate = gzip.decompress(payload)
            if not looks_like_mps(candidate):
                raise RuntimeError("expanded content is not MPS")
            model = candidate
            break
        except Exception as error:  # try the other official archive
            errors.append(f"{url}: {error}")
    if not model:
        raise RuntimeError(
            f"could not download {name} from an official MIPLIB archive:\n  "
            + "\n  ".join(errors))
    atomic_write(destination, model)
    print(f"  wrote  {destination} ({len(model)} bytes)")


def build_emps(work: Path, timeout: float) -> Path:
    source = work / "emps.c"
    binary = work / "emps"
    atomic_write(source, download(f"{NETLIB_BASE}/emps.c", timeout))
    compiler = os.environ.get("CC", "cc")
    try:
        subprocess.run(
            [compiler, "-O2", "-o", str(binary), str(source)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as error:
        raise RuntimeError(f"C compiler {compiler!r} not found (set CC)") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"failed to compile Netlib emps.c: {detail}") from error
    return binary


def fetch_netlib(
    names: list[str], destination: Path, timeout: float, force: bool
) -> tuple[int, int]:
    pending = [
        name for name in names
        if force or not (destination / f"{name}.mps").is_file()
        or not (destination / f"{name}.mps").stat().st_size
    ]
    for name in names:
        if name not in pending:
            print(f"  exists {destination / (name + '.mps')}")
    if not pending:
        return len(names), 0

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sor-netlib-") as temp:
        work = Path(temp)
        emps = build_emps(work, timeout)
        for name in pending:
            url = f"{NETLIB_BASE}/{name}"
            print(f"  fetch  {url}")
            raw = work / name
            atomic_write(raw, download(url, timeout))
            result = subprocess.run(
                [str(emps), str(raw)], check=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            model = result.stdout
            if not looks_like_mps(model):
                raise RuntimeError(f"Netlib EMPS produced invalid MPS for {name}")
            output = destination / f"{name}.mps"
            atomic_write(output, model)
            print(f"  wrote  {output} ({len(model)} bytes)")
    return len(names), len(pending)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--suite", action="append", choices=SUITES,
        help="suite to fetch; repeat for multiple suites")
    result.add_argument(
        "--instance", action="append", default=[], metavar="NAME",
        help="only fetch this instance; repeatable (requires exactly one suite)")
    result.add_argument("--all", action="store_true", help="fetch every suite")
    result.add_argument("--list", action="store_true", help="list available names and exit")
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="benchmark root directory")
    result.add_argument("--timeout", type=float, default=60.0,
                        help="per-request timeout in seconds")
    result.add_argument("--force", action="store_true", help="replace existing models")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.timeout > 0:
        raise SystemExit("error: --timeout must be positive")
    suites = list(dict.fromkeys(args.suite or []))
    if args.all:
        suites = list(SUITES)
    if args.instance and len(suites) != 1:
        raise SystemExit("error: --instance requires exactly one --suite")
    if args.list:
        for suite in suites or SUITES:
            print(f"{suite} ({len(suite_names(suite, args.root))})")
            print("  " + " ".join(suite_names(suite, args.root)))
        return 0
    if not suites:
        raise SystemExit("error: select --suite NAME or --all (use --list to browse)")

    fetched = 0
    selected = 0
    for suite in suites:
        names = select_instances(suite, args.instance, args.root)
        selected += len(names)
        print(f"{suite}: {len(names)} instance(s)")
        if suite == "netlib":
            _, count = fetch_netlib(
                names, args.root / "netlib" / "mps", args.timeout, args.force)
            fetched += count
        else:
            output = args.root / suite / "mps"
            for name in names:
                before = output / f"{name}.mps"
                existed = before.is_file() and before.stat().st_size and not args.force
                fetch_miplib(name, before, args.timeout, args.force)
                fetched += 0 if existed else 1
    print(f"ready: {selected} selected, {fetched} downloaded under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
