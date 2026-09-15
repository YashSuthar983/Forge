#!/usr/bin/env python3
"""Build benchmarks/miplib-small from the local MIPLIB 2017 archive.

benchmarks/miplib-easy is twenty hand-picked instances, most of them MIPLIB 3.0
classics. That set is small enough to over-fit: a change can look like progress
because it happened to help two of the eight instances nobody proves. This
script cuts a wider slice out of the MIPLIB 2017 *benchmark* list -- still only
instances with a published optimum, so every run can be scored against ground
truth rather than against our own previous answer -- ordered by file size so the
result stays runnable in minutes rather than CPU-days.

It reads only what is already on disk (benchmarks/miplib2017/mps/*.mps.gz plus
the .solu file); it downloads nothing. Instances are gunzipped into
benchmarks/miplib-small/mps/, which .gitignore excludes for the same reason
benchmarks/miplib2017/mps/ is excluded.

  scripts/make_miplib_small.py --count 40
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "benchmarks" / "miplib2017"
DST = ROOT / "benchmarks" / "miplib-small"


def read_optima(solu: Path) -> dict[str, float]:
    """name -> proved optimal objective, for '=opt=' lines only.

    '=best=' lines are the best value anyone has found, not a proved optimum,
    so they cannot be used to score a proof and are skipped.
    """
    out: dict[str, float] = {}
    for line in solu.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "=opt=":
            try:
                out[parts[1]] = float(parts[2])
            except ValueError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=40,
                    help="how many instances to take (smallest first)")
    ap.add_argument("--max-kb", type=int, default=0,
                    help="also cap compressed size in KB (0 = no cap)")
    ap.add_argument("--test-list", default="benchmark-v2.test",
                    help="which MIPLIB .test list to draw from")
    args = ap.parse_args()

    solu = SRC / "miplib2017-v36.solu"
    tests = SRC / args.test_list
    mps_dir = SRC / "mps"
    for p in (solu, tests, mps_dir):
        if not p.exists():
            print(f"missing {p} -- see .gitignore for how to populate "
                  f"benchmarks/miplib2017/", file=sys.stderr)
            return 1

    optima = read_optima(solu)
    candidates: list[tuple[int, str, Path]] = []
    for line in tests.read_text().splitlines():
        fname = line.strip()
        if not fname:
            continue
        name = fname[:-7] if fname.endswith(".mps.gz") else fname
        src = mps_dir / fname
        if name not in optima or not src.exists():
            continue
        size = src.stat().st_size
        if args.max_kb and size > args.max_kb * 1024:
            continue
        candidates.append((size, name, src))
    candidates.sort()
    chosen = candidates[: args.count]
    if not chosen:
        print("no candidates matched", file=sys.stderr)
        return 1

    out_mps = DST / "mps"
    out_mps.mkdir(parents=True, exist_ok=True)
    ref_lines = ["# name,optimum  (MIPLIB 2017 v36 =opt= entries)"]
    for _, name, src in chosen:
        dst = out_mps / f"{name}.mps"
        if not dst.exists():
            with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
        ref_lines.append(f"{name},{optima[name]!r}")
    (DST / "reference.csv").write_text("\n".join(ref_lines) + "\n")

    total = sum(s for s, _, _ in chosen)
    print(f"wrote {len(chosen)} instances to {out_mps} "
          f"({total / 1024:.0f} KB compressed source)")
    print(f"reference optima -> {DST / 'reference.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
