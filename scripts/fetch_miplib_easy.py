#!/usr/bin/env python3
"""Fetch a small MIPLIB 'easy' subset for SIH demos.

Downloads public instances only (never vendors solver code).
Tries MIPLIB 2017 WebData first, then MIPLIB 3.0 archive for classics.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (name, preferred source). Sources: "2017" = miplib.zib.de WebData,
# "miplib3" = miplib2010.zib.de MIPLIB 3.0 archive.
# Curated for a from-scratch B&B demo - not a MIPLIB-2017 leaderboard claim.
EASY: list[tuple[str, str]] = [
    ("flugpl", "2017"),
    ("gt2", "2017"),
    ("blend2", "2017"),
    ("p0201", "2017"),
    ("markshare1", "2017"),
    ("markshare2", "2017"),
    ("pk1", "2017"),
    ("gen-ip002", "2017"),
    ("gen-ip054", "2017"),
    ("n5-3", "2017"),
    ("assign1-5-8", "2017"),
    ("p0033", "miplib3"),
    ("enigma", "miplib3"),
    ("stein27", "miplib3"),
    ("lseu", "miplib3"),
    ("mod008", "miplib3"),
    ("rgn", "miplib3"),
    ("vpm1", "miplib3"),
    ("misc03", "miplib3"),
    ("mod010", "miplib3"),
]

UA = {"User-Agent": "SOR-fetch/1.0 (+https://github.com; academic MIPLIB download)"}


def url_for(name: str, source: str) -> str:
    if source == "miplib3":
        return f"https://miplib2010.zib.de/miplib3/miplib3/{name}.mps.gz"
    return f"https://miplib.zib.de/WebData/instances/{name}.mps.gz"


def fetch_one(name: str, source: str, dest: Path, timeout: float = 60.0) -> bool:
    mps_path = dest / f"{name}.mps"
    if mps_path.exists() and mps_path.stat().st_size > 0:
        print(f"  exists {mps_path.name}")
        return True

    candidates = [source]
    # Fallbacks if preferred source fails.
    if source == "2017":
        candidates.append("miplib3")
    else:
        candidates.append("2017")

    last_err = ""
    for src in candidates:
        url = url_for(name, src)
        print(f"  fetch {url}")
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) < 2 or data[:2] != b"\x1f\x8b":
                last_err = f"not gzip (got {data[:12]!r})"
                print(f"  skip {src}: {last_err}")
                continue
            gz_path = dest / f"{name}.mps.gz"
            gz_path.write_bytes(data)
            with gzip.open(gz_path, "rb") as gzf, open(mps_path, "wb") as out:
                shutil.copyfileobj(gzf, out)
            gz_path.unlink(missing_ok=True)
            print(f"  wrote {mps_path} ({mps_path.stat().st_size} bytes) via {src}")
            return True
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            print(f"  FAIL {src}: {last_err}", file=sys.stderr)

    print(f"  FAIL {name}: {last_err}", file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "benchmarks" / "miplib-easy" / "mps"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dest = Path(args.outdir)
    dest.mkdir(parents=True, exist_ok=True)
    items = EASY[: args.limit] if args.limit else EASY
    ok = 0
    for name, source in items:
        if fetch_one(name, source, dest):
            ok += 1
    print(f"fetched {ok}/{len(items)} into {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
