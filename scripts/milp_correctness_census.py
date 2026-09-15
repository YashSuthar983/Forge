#!/usr/bin/env python3
"""MILP correctness census: SOR (Latest) vs HiGHS reference.

For every MPS in a suite this script runs SOR under product defaults and a
HiGHS reference (long budget), then checks soundness only — never speed:

  1. False Optimal: SOR claims Optimal while HiGHS exhibits a strictly
     better feasible point (min sense), or SOR's Optimal objective differs
     from a HiGHS-proved optimum.
  2. False dual: SOR's dual bound sits above a known feasible objective
     (the dual must lower-bound every feasible point).
  3. False Infeasible: SOR claims Infeasible while HiGHS finds a solution.

HiGHS runs via benchmarks/.venv-baseline (process-only; never linked).
Writes JSON artifacts and prints a verdict table.

Usage:
  scripts/milp_correctness_census.py --suite benchmarks/miplib-easy/mps \
      --sor build-milp/sor_solve -t 25 --ref-t 60 \
      --out build-milp/milp_correctness_census.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HIGHS_PY = ROOT / "benchmarks" / ".venv-baseline" / "bin" / "python"


def run_sor(exe: Path, model: Path, tl: float) -> dict:
    cmd = [str(exe), str(model), "--engine", "milp",
           "--milp-policy", "latest", "--time-limit", str(tl)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=tl + 60, check=False)
        out, rc = p.stdout, p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        rc = -9
    wall = time.perf_counter() - t0
    status, obj, dual = "CRASH", None, None
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 2 and f[0] == "status:":
            status = f[1]
        elif len(f) >= 2 and f[0] == "objective:":
            try:
                obj = float(f[1])
            except ValueError:
                pass
        elif len(f) >= 3 and f[0] == "dual" and f[1] == "bound:":
            try:
                dual = float(f[2])
            except ValueError:
                pass
    return {"status": status, "obj": obj, "dual": dual, "rc": rc,
            "wall": round(wall, 2), "term": ""}


HIGHS_SCRIPT = r"""
import sys, json, math
import highspy
h = highspy.Highs()
h.setOptionValue("output_flag", False)
h.setOptionValue("time_limit", float(sys.argv[2]))
h.readModel(sys.argv[1])
h.run()
info = h.getInfo()
ms = h.modelStatusToString(h.getModelStatus())
def fin(v):
    return v if v is not None and math.isfinite(v) else None
out = {
    "status": ms,
    "proved": ms in ("Optimal", "Infeasible"),
    "obj": fin(info.objective_function_value),
    "dual": fin(info.mip_dual_bound),
}
print(json.dumps(out))
"""


def run_highs(model: Path, tl: float) -> dict | None:
    if not HIGHS_PY.is_file():
        return None
    try:
        p = subprocess.run([str(HIGHS_PY), "-c", HIGHS_SCRIPT,
                            str(model), str(tl)],
                           capture_output=True, text=True, timeout=tl + 120,
                           check=False)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        return json.loads(line) if line else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
        return None


def check(sor: dict, ref: dict | None) -> tuple[str, str]:
    """Return (verdict, detail). Verdict OK or an issue label."""
    tol = 1e-4 * max(1.0, abs(sor.get("obj") or 0.0))
    if sor["status"] == "CRASH":
        return "CRASH", f"rc={sor['rc']}"
    if sor["status"] == "Infeasible":
        if ref and ref.get("obj") is not None and ref["status"] != "Infeasible":
            return "FALSE_INFEASIBLE", "HiGHS found a solution"
        return "OK", ""
    if sor["status"] not in ("Optimal", "Feasible"):
        return "OK", sor["status"]
    obj, dual = sor.get("obj"), sor.get("dual")
    if ref is None:
        return "OK", "no reference"
    # 1. False Optimal vs a proved reference optimum.
    if sor["status"] == "Optimal" and ref.get("proved") and \
            ref["status"] == "Optimal":
        if obj is None or abs(obj - ref["obj"]) > tol:
            return "FALSE_OPTIMAL", f"sor {obj} vs highs opt {ref['obj']}"
    # 2. Optimal while the reference has a strictly better incumbent.
    if sor["status"] == "Optimal" and ref.get("obj") is not None:
        if obj is not None and obj > ref["obj"] + tol:  # min sense
            return "FALSE_OPTIMAL", f"sor {obj} vs highs incumbent {ref['obj']}"
    # 3. Dual above a known feasible objective.
    if dual is not None and ref.get("obj") is not None:
        if dual > ref["obj"] + tol:
            return "FALSE_DUAL", f"dual {dual} above feasible {ref['obj']}"
    return "OK", ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sor", type=Path, default=ROOT / "build-milp" / "sor_solve")
    ap.add_argument("--suite", type=Path,
                    default=ROOT / "benchmarks" / "miplib-easy" / "mps")
    ap.add_argument("-t", "--time-limit", type=float, default=25.0)
    ap.add_argument("--ref-t", type=float, default=60.0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "build-milp" / "milp_correctness_census.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    models = sorted(args.suite.glob("*.mps"))
    if args.limit:
        models = models[: args.limit]
    if not models:
        print(f"error: no MPS in {args.suite}", file=sys.stderr)
        return 2

    rows, issues = [], []
    for m in models:
        sor = run_sor(args.sor, m, args.time_limit)
        ref = run_highs(m, args.ref_t)
        verdict, detail = check(sor, ref)
        row = {"name": m.stem, **sor,
               "ref": ({"status": ref["status"], "obj": ref["obj"],
                        "dual": ref["dual"], "proved": ref["proved"]}
                       if ref else None),
               "verdict": verdict, "detail": detail}
        rows.append(row)
        mark = "!!" if verdict != "OK" else "  "
        print(f"{mark} {m.stem:16s} {sor['status']:12s} "
              f"obj={sor.get('obj')} dual={sor.get('dual')} "
              f"[{verdict}] {detail}")
        if verdict != "OK":
            issues.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"rows": rows,
         "summary": {"n": len(rows), "issues": len(issues)}}, indent=2))
    print(f"\nverdict: {len(issues)} issue(s) / {len(rows)} instances "
          f"-> {args.out}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
