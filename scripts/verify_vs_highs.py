#!/usr/bin/env python3
"""Compare SOR's Netlib results against HiGHS run as an EXTERNAL PROCESS.

HiGHS is never linked into libsor and never appears in a SOR certificate; it is
used here purely as an oracle, which is the default workflow in
sor/docs/clean_room_policy.md. The point of this script is that SOR's own
residuals cannot validate SOR: a self-consistent wrong basis has small residuals
too. Only an independent optimal value can catch that.

  python3 scripts/verify_vs_highs.py benchmarks/results/netlib_simplex.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--python", default=str(ROOT / "benchmarks" / ".venv-baseline" / "bin" / "python"))
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--rel-tol", type=float, default=1e-6)
    args = ap.parse_args()

    ours = {}
    with open(args.jsonl) as fh:
        for line in fh:
            r = json.loads(line)
            ours[r["instance"]] = r

    baseline = ROOT / "scripts" / "run_highs_baseline.py"
    mps_dir = ROOT / "benchmarks" / "netlib" / "mps"

    agree = disagree = no_ref = 0
    bad: list[str] = []
    print(f"{'instance':<12} {'SOR status':<14} {'SOR objective':>20} "
          f"{'HiGHS objective':>20} {'rel diff':>10}  verdict")
    print("-" * 100)

    for name in sorted(ours):
        mps = mps_dir / f"{name}.mps"
        try:
            p = subprocess.run([args.python, str(baseline), str(mps),
                                "--time-limit", str(args.time_limit)],
                               capture_output=True, text=True,
                               timeout=args.time_limit + 120)
            ref = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception as e:  # noqa: BLE001
            ref = {"status": "error", "error": str(e)}

        o = ours[name]
        ro = ref.get("objective")
        so = o.get("objective")
        verdict = "-"
        rel = None
        if ro is None or ref.get("status") not in ("optimal", "Optimal"):
            verdict = f"no reference ({ref.get('status')})"
            no_ref += 1
        elif o["status"] != "Optimal":
            verdict = f"SOR did not prove optimal ({o['status']})"
        elif so is None:
            verdict = "SOR objective unparsed"
            disagree += 1
            bad.append(name)
        else:
            rel = abs(so - ro) / (1.0 + abs(ro))
            if rel <= args.rel_tol:
                verdict = "AGREE"
                agree += 1
            else:
                verdict = "*** DISAGREE ***"
                disagree += 1
                bad.append(name)

        print(f"{name:<12} {o['status']:<14} "
              f"{(f'{so:.10e}' if so is not None else '-'):>20} "
              f"{(f'{ro:.10e}' if ro is not None else '-'):>20} "
              f"{(f'{rel:.2e}' if rel is not None else '-'):>10}  {verdict}", flush=True)

    print("-" * 100)
    print(f"agree={agree}  disagree={disagree}  no_reference={no_ref}  "
          f"(rel tol {args.rel_tol:g})")
    if bad:
        print("DISAGREEMENTS: " + ", ".join(bad))
    return 1 if disagree else 0


if __name__ == "__main__":
    raise SystemExit(main())
