#!/usr/bin/env python3
"""Thin wrapper around build/sor_milp_train for list-file workflows.

Prefer the C++ binary for fitting (reuses fit_sparse_sb /
fit_sc_milp_contrastive / fit_lifted_sb). This script only expands globs /
writes a path list.

  tools/milp_train.py --exe build/sor_milp_train \\
      --out-dir /tmp/milp_models benchmarks/milp-ablate-tiny/*.mps
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=ROOT / "build" / "sor_milp_train")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/sor_milp_models"))
    ap.add_argument("--time-limit", type=float, default=5.0)
    ap.add_argument("--max-nodes", type=int, default=500)
    ap.add_argument(
        "--only",
        choices=("both", "all", "sparse-sb", "sc-milp", "lifted", "planbb"),
        default="both",
    )
    ap.add_argument(
        "--sb-loss", choices=("ranking", "lasso"), default="ranking"
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("models", nargs="+", type=Path)
    args = ap.parse_args()

    if not args.exe.is_file():
        print(
            f"error: missing {args.exe} (build sor_milp_train first)",
            file=sys.stderr,
        )
        return 2

    paths = [p.resolve() for p in args.models if p.is_file()]
    if not paths:
        print("error: no readable model files", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for p in paths:
            fh.write(f"{p}\n")
        list_path = Path(fh.name)

    cmd = [
        str(args.exe),
        "--list",
        str(list_path),
        "--out-dir",
        str(args.out_dir),
        "--time-limit",
        str(args.time_limit),
        "--max-nodes",
        str(args.max_nodes),
        "--only",
        args.only,
        "--sb-loss",
        args.sb_loss,
    ]
    if args.verbose:
        cmd.append("--verbose")
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
