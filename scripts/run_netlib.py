#!/usr/bin/env python3
"""Sweep the pinned Netlib LP set with one SOR engine and summarise.

Reports, per instance: status, proof level, objective, primal violation, dual
residual, relative duality gap, iterations, wall time. The headline number is
how many instances reach Status=Optimal, which requires ProofLevel >=
ProvedOptimalFP and therefore a basis -- see sor_certify/finalize.cpp.

  python3 scripts/run_netlib.py --engine simplex --time-limit 60
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAT = {
    "rows_cols": re.compile(r"rows x cols:\s+(\d+) x (\d+)\s+nnz (\d+)"),
    "status": re.compile(r"^status:\s+(\S+)", re.M),
    "proof": re.compile(r"^proof_level:\s+(\S+)", re.M),
    "objective": re.compile(r"^objective:\s+(\S+)", re.M),
    "dual_bound": re.compile(r"^dual bound:\s+(\S+)", re.M),
    # The simplex prints "max primal viol" because it covers bound violations
    # too; PDHG prints "max row violation".
    "primal_viol": re.compile(r"^max (?:primal viol|row violation):\s+(\S+)", re.M),
    "dual_res": re.compile(r"^dual residual:\s+(\S+)", re.M),
    "gap": re.compile(r"^rel gap:\s+(\S+)", re.M),
    "iters": re.compile(r"^iterations:\s+(\d+)", re.M),
    "downgrade": re.compile(r"^downgrade:\s+(.+)$", re.M),
    "termination": re.compile(r"^termination:\s+(.+)$", re.M),
}


def grab(txt: str, key: str):
    m = PAT[key].search(txt)
    if not m:
        return None
    s = m.group(1)
    try:
        return float(s)
    except ValueError:
        return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="simplex")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--jsonl", default="")
    args = ap.parse_args()

    exe = ROOT / "build" / "sor_solve"
    if not exe.exists():
        print(f"missing {exe}; build first", file=sys.stderr)
        return 2

    mps_dir = ROOT / "benchmarks" / "netlib" / "mps"
    files = sorted(mps_dir.glob("*.mps"))
    if args.limit:
        files = files[: args.limit]

    rows = []
    print(f"{'instance':<12} {'rows':>6} {'cols':>6} {'nnz':>8} "
          f"{'status':<14} {'objective':>20} {'pviol':>9} {'dres':>9} "
          f"{'gap':>9} {'iters':>8} {'sec':>7}")
    print("-" * 128)

    for f in files:
        cmd = [str(exe), str(f), "--engine", args.engine,
               "--time-limit", str(args.time_limit)]
        if args.tol is not None:
            cmd += ["--tol", str(args.tol)]
        cmd += args.extra
        t0 = time.perf_counter()
        # The hard kill is a safety net well above the in-solver limit; the
        # solver is supposed to stop itself and report its best point.
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.time_limit + 120)
            wall = time.perf_counter() - t0
            txt = p.stdout
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - t0
            txt = ""

        rc = {
            "instance": f.stem,
            "status": grab(txt, "status") or "killed",
            "proof": grab(txt, "proof"),
            "objective": grab(txt, "objective"),
            "dual_bound": grab(txt, "dual_bound"),
            "primal_viol": grab(txt, "primal_viol"),
            "dual_res": grab(txt, "dual_res"),
            "gap": grab(txt, "gap"),
            "iterations": grab(txt, "iters"),
            "downgrade": grab(txt, "downgrade"),
            "termination": grab(txt, "termination"),
            "wall_s": round(wall, 3),
        }
        m = PAT["rows_cols"].search(txt)
        if m:
            rc["rows"], rc["cols"], rc["nnz"] = (int(m.group(i)) for i in (1, 2, 3))
        rows.append(rc)

        def fmt(v, w, spec=""):
            if v is None:
                return " " * (w - 1) + "-"
            return f"{v:>{w}{spec}}" if spec else f"{str(v):>{w}}"

        print(f"{rc['instance']:<12} {fmt(rc.get('rows'), 6)} {fmt(rc.get('cols'), 6)} "
              f"{fmt(rc.get('nnz'), 8)} {str(rc['status']):<14} "
              f"{fmt(rc['objective'], 20, '.10e')} {fmt(rc['primal_viol'], 9, '.1e')} "
              f"{fmt(rc['dual_res'], 9, '.1e')} {fmt(rc['gap'], 9, '.1e')} "
              f"{fmt(rc['iterations'], 8, '.0f')} {rc['wall_s']:>7.2f}", flush=True)

    n = len(rows)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[str(r["status"])] = by_status.get(str(r["status"]), 0) + 1
    opt = by_status.get("Optimal", 0)

    print("-" * 128)
    print(f"engine={args.engine}  instances={n}  OPTIMAL={opt}/{n}")
    for k in sorted(by_status, key=lambda s: -by_status[s]):
        print(f"  {k:<20} {by_status[k]}")
    solved = [r for r in rows if r["status"] == "Optimal"]
    if solved:
        tot = sum(r["wall_s"] for r in solved)
        print(f"  total wall on optimal instances: {tot:.2f}s  "
              f"(max {max(r['wall_s'] for r in solved):.2f}s on "
              f"{max(solved, key=lambda r: r['wall_s'])['instance']})")
    if args.jsonl:
        out = ROOT / args.jsonl
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
