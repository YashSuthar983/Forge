#!/usr/bin/env python3
"""SOR benchmark harness.

Runs sor_solve and one or more EXTERNAL baseline solvers over a named instance
set, cross-checks objectives, and reports shifted geometric means.

Protocol rules this script enforces (sor/docs/master_spec.md §9):
  * every timeout, numerical failure, and disagreement is reported -- a table
    with no failure rows is a lie;
  * baselines run as separate processes, tagged "kind": "external_process",
    and never enter a SOR certificate;
  * hardware, versions, options, and per-instance rows go into the JSONL;
  * shifted geometric mean, not arithmetic mean.

Usage:
  scripts/run_bench.py --suite netlib --time-limit 20 -o benchmarks/results
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# sor_solve output parsing
# --------------------------------------------------------------------------
_PATTERNS = {
    "rows_cols": re.compile(r"rows x cols:\s+(\d+) x (\d+)\s+nnz (\d+)"),
    "status": re.compile(r"^status:\s+(\S+)", re.M),
    "proof": re.compile(r"^proof_level:\s+(\S+)", re.M),
    "objective": re.compile(r"^objective:\s+(\S+)", re.M),
    "dual_bound": re.compile(r"^dual bound:\s+(\S+)", re.M),
    "row_viol": re.compile(r"^max row violation:\s+(\S+)", re.M),
    "dual_res": re.compile(r"^dual residual:\s+(\S+)", re.M),
    "iters": re.compile(r"^iterations:\s+(\d+)", re.M),
    "total_ms": re.compile(r"^\s+total\s+(\S+)", re.M),
    "kernel_ms": re.compile(r"^\s+kernel\s+(\S+)", re.M),
}


def _f(text: str, key: str):
    m = _PATTERNS[key].search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


@dataclass
class Row:
    instance: str
    rows: int | None = None
    cols: int | None = None
    nnz: int | None = None

    sor_status: str = "not_run"
    sor_proof: str | None = None
    sor_objective: float | None = None
    sor_dual_bound: float | None = None
    sor_row_violation: float | None = None
    sor_dual_residual: float | None = None
    sor_iterations: int | None = None
    sor_wall_s: float | None = None
    sor_error: str | None = None

    base_status: str = "not_run"
    base_objective: float | None = None
    base_wall_s: float | None = None
    base_iterations: int | None = None
    base_error: str | None = None

    agree: bool | None = None
    rel_obj_gap: float | None = None
    verdict: str = "unknown"


def run_sor(exe: Path, mps: Path, time_limit: float, max_iter: int,
            tol: float, backend: str) -> tuple[dict, float, str | None]:
    cmd = [str(exe), str(mps), "--backend", backend,
           "--max-iter", str(max_iter), "--tol", str(tol)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=time_limit)
        wall = time.perf_counter() - t0
        return {"stdout": p.stdout, "stderr": p.stderr, "rc": p.returncode}, wall, None
    except subprocess.TimeoutExpired:
        return {}, time.perf_counter() - t0, f"wall timeout > {time_limit}s"
    except Exception as e:  # noqa: BLE001
        return {}, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def run_baseline(python: Path, script: Path, mps: Path,
                 time_limit: float) -> tuple[dict, float, str | None]:
    cmd = [str(python), str(script), str(mps), "--time-limit", str(time_limit)]
    t0 = time.perf_counter()
    try:
        # Generous outer margin: the inner limit should fire first.
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=time_limit + 60)
        wall = time.perf_counter() - t0
        try:
            return json.loads(p.stdout.strip().splitlines()[-1]), wall, None
        except Exception:
            return {}, wall, f"unparseable baseline output: {p.stdout[:200]!r} {p.stderr[:200]!r}"
    except subprocess.TimeoutExpired:
        return {}, time.perf_counter() - t0, f"baseline wall timeout > {time_limit + 60}s"
    except Exception as e:  # noqa: BLE001
        return {}, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def shifted_geomean(values: list[float], shift: float = 1.0) -> float | None:
    """Shifted geometric mean: exp(mean(log(v + shift))) - shift.

    Standard solver-benchmark aggregate (Mittelmann uses shift 10 for seconds).
    Robust to easy instances dominating, unlike an arithmetic mean.
    """
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    acc = sum(math.log(max(v, 0.0) + shift) for v in vals)
    return math.exp(acc / len(vals)) - shift


# Statuses that mean "SOR produced a usable primal point".
SOR_USABLE = {"Feasible", "Optimal"}
BASE_SOLVED = {"Optimal"}


def classify(r: Row, obj_tol: float, feas_tol: float) -> None:
    have_base = r.base_status in BASE_SOLVED and r.base_objective is not None
    have_sor = r.sor_status in SOR_USABLE and r.sor_objective is not None

    if have_base and r.sor_objective is not None:
        denom = 1.0 + abs(r.base_objective)
        r.rel_obj_gap = abs(r.sor_objective - r.base_objective) / denom
        r.agree = r.rel_obj_gap <= obj_tol

    feas_ok = (r.sor_row_violation is not None
               and r.sor_row_violation <= feas_tol)

    if r.sor_error:
        r.verdict = "sor_timeout" if "timeout" in r.sor_error else "sor_error"
    elif r.sor_status == "NumericalFailure":
        r.verdict = "sor_numerical_failure"
    elif not have_sor:
        # Interrupted / NoSolutionFound: did not reach tolerance.
        r.verdict = "sor_no_converge"
    elif not feas_ok:
        r.verdict = "sor_infeasible_point"
    elif r.agree is True:
        r.verdict = "match"
    elif r.agree is False:
        r.verdict = "objective_mismatch"
    else:
        r.verdict = "no_baseline_reference"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="netlib")
    ap.add_argument("--instances-dir", default=None)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--max-iter", type=int, default=200000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--obj-tol", type=float, default=1e-6)
    ap.add_argument("--feas-tol", type=float, default=1e-6)
    ap.add_argument("--limit", type=int, default=0, help="only first N instances")
    ap.add_argument("--sgm-shift", type=float, default=1.0)
    ap.add_argument("-o", "--outdir", default="benchmarks/results")
    args = ap.parse_args()

    inst_dir = Path(args.instances_dir or ROOT / "benchmarks" / args.suite / "mps")
    if not inst_dir.is_dir():
        print(f"error: instance dir not found: {inst_dir}", file=sys.stderr)
        return 2
    instances = sorted(inst_dir.glob("*.mps"))
    if args.limit:
        instances = instances[: args.limit]
    if not instances:
        print(f"error: no .mps files in {inst_dir}", file=sys.stderr)
        return 2

    exe = ROOT / "build" / "sor_solve"
    if not exe.exists():
        print(f"error: {exe} not built", file=sys.stderr)
        return 2

    venv_py = ROOT / "benchmarks" / ".venv-baseline" / "bin" / "python"
    base_script = ROOT / "scripts" / "run_highs_baseline.py"
    have_baseline = venv_py.exists() and base_script.exists()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    env = {
        "suite": args.suite,
        "n_instances": len(instances),
        "time_limit_s": args.time_limit,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "backend": args.backend,
        "obj_tol": args.obj_tol,
        "feas_tol": args.feas_tol,
        "sgm_shift": args.sgm_shift,
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "sor_engine": "pdhg (vanilla PDHG; no restarts/primal-weights/Halpern)",
        "baseline": "HiGHS via highspy, external process" if have_baseline
                    else "NONE AVAILABLE",
    }
    print(json.dumps(env, indent=2))
    print(f"\nrunning {len(instances)} instances "
          f"(time limit {args.time_limit}s each)...\n", flush=True)

    rows: list[Row] = []
    for i, mps in enumerate(instances, 1):
        r = Row(instance=mps.stem)

        raw, wall, err = run_sor(exe, mps, args.time_limit, args.max_iter,
                                 args.tol, args.backend)
        r.sor_wall_s = wall
        if err:
            r.sor_error = err
            r.sor_status = "timeout" if "timeout" in err else "error"
        else:
            txt = raw.get("stdout", "")
            m = _PATTERNS["rows_cols"].search(txt)
            if m:
                r.rows, r.cols, r.nnz = int(m[1]), int(m[2]), int(m[3])
            sm = _PATTERNS["status"].search(txt)
            r.sor_status = sm.group(1) if sm else "unparsed"
            pm = _PATTERNS["proof"].search(txt)
            r.sor_proof = pm.group(1) if pm else None
            r.sor_objective = _f(txt, "objective")
            r.sor_dual_bound = _f(txt, "dual_bound")
            r.sor_row_violation = _f(txt, "row_viol")
            r.sor_dual_residual = _f(txt, "dual_res")
            it = _f(txt, "iters")
            r.sor_iterations = int(it) if it is not None else None
            if r.sor_status in ("unparsed",):
                r.sor_error = (raw.get("stderr") or "")[:200]

        if have_baseline:
            bj, bwall, berr = run_baseline(venv_py, base_script, mps,
                                           args.time_limit)
            r.base_wall_s = bj.get("solve_s", bwall)
            if berr:
                r.base_status = "error"
                r.base_error = berr
            else:
                r.base_status = bj.get("status", "unknown")
                r.base_objective = bj.get("objective")
                r.base_iterations = bj.get("iterations")
                if r.rows is None:
                    r.rows, r.cols = bj.get("rows"), bj.get("cols")

        classify(r, args.obj_tol, args.feas_tol)
        rows.append(r)
        print(f"[{i:3d}/{len(instances)}] {r.instance:12s} "
              f"sor={r.sor_status:16s} base={r.base_status:12s} "
              f"verdict={r.verdict}", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl = outdir / f"{args.suite}-{stamp}.jsonl"
    with jsonl.open("w") as f:
        f.write(json.dumps({"record": "environment", **env}) + "\n")
        for r in rows:
            f.write(json.dumps({"record": "instance", **asdict(r)}) + "\n")

    # ---- summary ----------------------------------------------------------
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1

    matched = [r for r in rows if r.verdict == "match"]
    base_solved = [r for r in rows if r.base_status in BASE_SOLVED]

    summary = {
        "record": "summary",
        "n_instances": len(rows),
        "verdicts": counts,
        "sor_matched_baseline": len(matched),
        "baseline_solved": len(base_solved),
        "sgm_sor_wall_s_matched": shifted_geomean(
            [r.sor_wall_s for r in matched], args.sgm_shift),
        "sgm_base_wall_s_matched": shifted_geomean(
            [r.base_wall_s for r in matched], args.sgm_shift),
        "sgm_base_wall_s_all_solved": shifted_geomean(
            [r.base_wall_s for r in base_solved], args.sgm_shift),
    }
    with jsonl.open("a") as f:
        f.write(json.dumps(summary) + "\n")

    print("\n" + "=" * 72)
    print(f"instances                 : {len(rows)}")
    print(f"baseline solved to Optimal : {len(base_solved)}")
    print(f"SOR matched baseline obj   : {len(matched)}")
    print("\nverdict breakdown:")
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"  {k:26s} {counts[k]:4d}")
    if summary["sgm_sor_wall_s_matched"] is not None:
        print(f"\nSGM wall time on the {len(matched)} matched instances "
              f"(shift {args.sgm_shift}):")
        print(f"  SOR   : {summary['sgm_sor_wall_s_matched']:.3f} s")
        print(f"  HiGHS : {summary['sgm_base_wall_s_matched']:.3f} s")
    print(f"\nJSONL: {jsonl}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
