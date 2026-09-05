#!/usr/bin/env python3
"""Side-by-side solver comparison.

One tool for every comparison this project makes: give it some solvers and some
models, and it runs each solver on each model and prints the results next to
each other.

    # every SOR engine on one model
    scripts/compare.py examples/testlp.mps --solvers sor:simplex,sor:pdhg,sor:hpr

    # SOR against HiGHS on a whole suite
    scripts/compare.py benchmarks/netlib/mps --solvers sor:simplex,highs

    # a MILP, with a time limit and machine-readable output
    scripts/compare.py benchmarks/miplib-easy/mps --solvers sor:milp,highs \\
        --time-limit 30 --jsonl results.jsonl

Solver names are `sor:<engine>[:<backend>]` for this project, or one of the
external baselines: highs, cbc, scipy, scipy-ipm, gurobi. Externals run only if
their Python package is importable; they are reported as `unavailable`
otherwise, never silently dropped.

CLEAN ROOM: external solvers are used ONLY as independent reference points for
these measurements. They are never linked into SOR and never contribute to a
result SOR reports. See docs/clean_room_policy.md.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Depth-independent: walk up to whichever directory holds CMakeLists.txt, so
# this keeps working if the script is ever moved.
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "CMakeLists.txt").exists())

# SOR prints a human report; these pull the numbers back out of it.
_PAT = {
    "size":    re.compile(r"rows x cols:\s+(\d+) x (\d+)\s+nnz (\d+)"),
    "status":  re.compile(r"^status:\s+(\S+)", re.M),
    "proof":   re.compile(r"^proof_level:\s+(\S+)", re.M),
    "obj":     re.compile(r"^objective:\s+(\S+)", re.M),
    "viol":    re.compile(r"^max (?:primal viol|row violation):\s+(\S+)", re.M),
    "iters":   re.compile(r"^iterations:\s+(\d+)", re.M),
    "solve_ms": re.compile(r"^  total\s+(\S+)", re.M),
}


@dataclass
class Result:
    solver: str
    instance: str
    status: str = "notrun"
    objective: float | None = None
    seconds: float | None = None      # solver-internal where available
    wall_s: float | None = None       # full process wall
    iterations: int | None = None
    proof: str | None = None
    violation: float | None = None
    rows: int | None = None
    cols: int | None = None
    nnz: int | None = None
    error: str | None = None


def _num(text: str, key: str) -> float | None:
    m = _PAT[key].search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# SOR
# --------------------------------------------------------------------------
def run_sor(model: Path, engine: str, backend: str, time_limit: float,
            tol: float, exe: Path, label: str) -> Result:
    r = Result(solver=label, instance=model.name)
    flags: list[str] = []
    eng = engine
    # Two HPR variants are flags on the same engine rather than engines.
    if engine in ("hpr-full", "hpr-vanilla"):
        eng, flags = "hpr", [f"--{engine.replace('hpr-', 'hpr-')}"]
    cmd = [str(exe), str(model), "--engine", eng, "--backend", backend,
           "--tol", str(tol), "--time-limit", str(time_limit), *flags]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=time_limit + 30)
        r.wall_s = time.perf_counter() - t0
        out = p.stdout
        m = _PAT["status"].search(out)
        r.status = m.group(1) if m else "unparsed"
        m = _PAT["proof"].search(out)
        r.proof = m.group(1) if m else None
        r.objective = _num(out, "obj")
        r.violation = _num(out, "viol")
        it = _num(out, "iters")
        r.iterations = int(it) if it is not None else None
        ms = _num(out, "solve_ms")
        # Compare solver-internal time to solver-internal time; keep the full
        # process wall separately so startup and file I/O stay visible.
        r.seconds = ms / 1000.0 if ms is not None else r.wall_s
        sz = _PAT["size"].search(out)
        if sz:
            r.rows, r.cols, r.nnz = int(sz[1]), int(sz[2]), int(sz[3])
        if r.status == "unparsed":
            r.error = ((p.stderr or "") + out)[:200]
    except subprocess.TimeoutExpired:
        r.wall_s = r.seconds = time.perf_counter() - t0
        r.status = "timeout"
    except Exception as e:  # noqa: BLE001
        r.wall_s = r.seconds = time.perf_counter() - t0
        r.status = "error"
        r.error = f"{type(e).__name__}: {e}"
    return r


# --------------------------------------------------------------------------
# External baselines — separate processes / independent packages only.
# --------------------------------------------------------------------------
def run_highs(model: Path, time_limit: float, label: str) -> Result:
    r = Result(solver=label, instance=model.name)
    try:
        import highspy
    except ImportError:
        r.status = "unavailable"
        r.error = "highspy not installed (see benchmarks/requirements-baseline.txt)"
        return r
    try:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("time_limit", float(time_limit))
        # HiGHS otherwise drops Highs.log into the working directory.
        h.setOptionValue("log_to_console", False)
        h.readModel(str(model))
        t0 = time.perf_counter()
        h.run()
        r.seconds = r.wall_s = time.perf_counter() - t0
        r.status = str(h.getModelStatus()).replace("HighsModelStatus.k", "")
        r.objective = float(h.getObjectiveValue())
        try:
            r.iterations = int(h.getInfo().simplex_iteration_count)
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001 — a baseline crash must not kill the sweep
        r.status = "crash"
        r.error = f"{type(e).__name__}: {e}"
    return r


def run_scipy(model: Path, time_limit: float, label: str, method: str) -> Result:
    r = Result(solver=label, instance=model.name)
    try:
        import pulp
        from scipy.optimize import linprog
        import numpy as np
    except ImportError as e:
        r.status = "unavailable"
        r.error = f"{e.name} not installed"
        return r
    try:
        _, lp = pulp.LpProblem.fromMPS(str(model))
        cols = lp.variables()
        idx = {v.name: i for i, v in enumerate(cols)}
        c = np.zeros(len(cols))
        for v, coef in lp.objective.items():
            c[idx[v.name]] = float(coef)
        Aub, bub, Aeq, beq = [], [], [], []
        for con in lp.constraints.values():
            row = np.zeros(len(cols))
            for v, coef in con.items():
                row[idx[v.name]] = float(coef)
            rhs = -float(con.constant)
            if con.sense == pulp.LpConstraintEQ:
                Aeq.append(row); beq.append(rhs)
            elif con.sense == pulp.LpConstraintLE:
                Aub.append(row); bub.append(rhs)
            else:
                Aub.append(-row); bub.append(-rhs)
        bounds = [(v.lowBound, v.upBound) for v in cols]
        t0 = time.perf_counter()
        res = linprog(c, A_ub=Aub or None, b_ub=bub or None,
                      A_eq=Aeq or None, b_eq=beq or None,
                      bounds=bounds, method=method)
        r.seconds = r.wall_s = time.perf_counter() - t0
        r.status = "Optimal" if res.success else str(res.message)[:40]
        r.objective = float(res.fun) if res.fun is not None else None
    except Exception as e:  # noqa: BLE001
        r.status = "crash"
        r.error = f"{type(e).__name__}: {e}"
    return r


def run_cbc(model: Path, time_limit: float, label: str) -> Result:
    r = Result(solver=label, instance=model.name)
    try:
        import pulp
    except ImportError:
        r.status = "unavailable"
        r.error = "pulp not installed"
        return r
    try:
        _, lp = pulp.LpProblem.fromMPS(str(model))
        t0 = time.perf_counter()
        lp.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
        r.seconds = r.wall_s = time.perf_counter() - t0
        r.status = pulp.LpStatus[lp.status]
        r.objective = float(pulp.value(lp.objective))
    except Exception as e:  # noqa: BLE001
        r.status = "crash"
        r.error = f"{type(e).__name__}: {e}"
    return r


def run_gurobi(model: Path, time_limit: float, label: str) -> Result:
    r = Result(solver=label, instance=model.name)
    try:
        import gurobipy as gp
    except ImportError:
        r.status = "unavailable"
        r.error = "gurobipy not installed / no licence"
        return r
    try:
        env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
        m = gp.read(str(model), env)
        m.setParam("TimeLimit", float(time_limit)); m.setParam("Threads", 1)
        t0 = time.perf_counter()
        m.optimize()
        r.seconds = r.wall_s = time.perf_counter() - t0
        r.status = {2: "Optimal", 3: "Infeasible", 5: "Unbounded",
                    9: "TimeLimit"}.get(m.Status, f"code{m.Status}")
        if m.SolCount:
            r.objective = float(m.ObjVal)
    except Exception as e:  # noqa: BLE001
        r.status = "crash"
        r.error = f"{type(e).__name__}: {e}"
    return r


def dispatch(spec: str, model: Path, time_limit: float, tol: float,
             exe: Path) -> Result:
    """`sor:<engine>[:<backend>]`, or an external baseline name."""
    key = spec.strip().lower()
    if key.startswith("sor"):
        parts = key.split(":")
        engine = parts[1] if len(parts) > 1 else "simplex"
        backend = parts[2] if len(parts) > 2 else "cpu"
        return run_sor(model, engine, backend, time_limit, tol, exe, spec)
    if key == "highs":
        return run_highs(model, time_limit, spec)
    if key in ("scipy", "scipy-hi", "scipy-highs"):
        return run_scipy(model, time_limit, spec, "highs")
    if key in ("scipy-ipm", "scipy-interior"):
        return run_scipy(model, time_limit, spec, "highs-ipm")
    if key == "cbc":
        return run_cbc(model, time_limit, spec)
    if key == "gurobi":
        return run_gurobi(model, time_limit, spec)
    r = Result(solver=spec, instance=model.name, status="unknown-solver")
    r.error = f"unrecognised solver {spec!r}"
    return r


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def shifted_geomean(values: list[float], shift: float) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    acc = sum(math.log(max(v, 0.0) + shift) for v in vals)
    return math.exp(acc / len(vals)) - shift


def fmt_time(s: float | None) -> str:
    if s is None:
        return "-"
    return f"{s * 1000:.1f}ms" if s < 1.0 else f"{s:.3f}s"


def fmt_obj(v: float | None) -> str:
    return "-" if v is None else f"{v:.6g}"


def report(rows: list[Result], solvers: list[str], shift: float,
           obj_tol: float) -> None:
    by_inst: dict[str, dict[str, Result]] = {}
    for r in rows:
        by_inst.setdefault(r.instance, {})[r.solver] = r

    wid = max([len(i) for i in by_inst] + [8])
    colw = max([len(s) for s in solvers] + [18])

    print()
    header = f"{'instance':<{wid}}  " + "  ".join(f"{s:<{colw}}" for s in solvers)
    print(header)
    print("-" * len(header))
    for inst in sorted(by_inst):
        cells = []
        for s in solvers:
            r = by_inst[inst].get(s)
            cells.append("-" if r is None
                         else f"{r.status[:10]:<10} {fmt_time(r.seconds):>7}")
        print(f"{inst:<{wid}}  " + "  ".join(f"{c:<{colw}}" for c in cells))

    # Objective agreement, against the first solver that produced one.
    print()
    print(f"{'instance':<{wid}}  objective (first solver = reference)")
    print("-" * (wid + 44))
    mismatches = 0
    for inst in sorted(by_inst):
        ref = None
        for s in solvers:
            r = by_inst[inst].get(s)
            if r and r.objective is not None:
                ref = r.objective
                break
        line = f"{inst:<{wid}}  "
        for s in solvers:
            r = by_inst[inst].get(s)
            if r is None or r.objective is None:
                line += f"{'-':<{colw}}  "
                continue
            tag = ""
            if ref is not None and ref != 0:
                if abs(r.objective - ref) / max(abs(ref), 1e-12) > obj_tol:
                    tag = " MISMATCH"
                    mismatches += 1
            line += f"{fmt_obj(r.objective) + tag:<{colw}}  "
        print(line)

    print()
    print(f"{'solver':<{colw}} {'solved':>8} {'obj':>6} {'SGM(s)':>10} {'total(s)':>10}")
    print("-" * (colw + 38))
    for s in solvers:
        rs = [by_inst[i].get(s) for i in by_inst]
        rs = [r for r in rs if r is not None]
        ok = [r for r in rs if r.status.lower() in
              ("optimal", "optimal ", "1", "solved")]
        times = [r.seconds for r in ok if r.seconds is not None]
        total = sum(r.seconds for r in rs if r.seconds is not None)
        sgm = shifted_geomean(times, shift)
        print(f"{s:<{colw}} {len(ok):>4}/{len(rs):<3} "
              f"{sum(1 for r in rs if r.objective is not None):>6} "
              f"{(f'{sgm:.4f}' if sgm is not None else '-'):>10} {total:>10.2f}")
    if mismatches:
        print(f"\n{mismatches} objective mismatch(es) above the {obj_tol:g} "
              f"relative tolerance -- investigate before quoting any timing.")


def collect_models(paths: list[str], limit: int | None) -> list[Path]:
    models: list[Path] = []
    for p in paths:
        q = Path(p)
        if not q.is_absolute():
            q = (Path.cwd() / q) if (Path.cwd() / q).exists() else (ROOT / q)
        if q.is_dir():
            for pat in ("*.mps", "*.qps", "*.lp"):
                models.extend(sorted(q.rglob(pat)))
        elif q.exists():
            models.append(q)
        else:
            print(f"warning: no such model or directory: {p}", file=sys.stderr)
    seen, out = set(), []
    for m in models:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out[:limit] if limit else out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run several solvers on several models and show the "
                    "results side by side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("models", nargs="+",
                    help="model files, or directories to scan for .mps/.qps/.lp")
    ap.add_argument("--solvers", default="sor:simplex,highs",
                    help="comma-separated. sor:<engine>[:<backend>] "
                         "(simplex, dual, pdhg, hpr, hpr-full, milp, qp) "
                         "or highs, cbc, scipy, scipy-ipm, gurobi")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N models")
    ap.add_argument("--sgm-shift", type=float, default=1.0,
                    help="shift for the shifted geometric mean, in seconds")
    ap.add_argument("--obj-tol", type=float, default=1e-4,
                    help="relative tolerance for objective agreement")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="also append one JSON record per run to this file")
    ap.add_argument("--exe", type=Path, default=None,
                    help="sor_solve binary (default: <root>/build/sor_solve)")
    args = ap.parse_args()

    exe = args.exe or (ROOT / "build" / "sor_solve")
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    if any(s.lower().startswith("sor") for s in solvers) and not exe.exists():
        print(f"error: {exe} not found -- build first:\n"
              f"  cmake -S {ROOT} -B {ROOT}/build -DCMAKE_BUILD_TYPE=Release\n"
              f"  cmake --build {ROOT}/build -j", file=sys.stderr)
        return 2

    models = collect_models(args.models, args.limit)
    if not models:
        print("error: no models found", file=sys.stderr)
        return 2

    print(f"{len(models)} model(s) x {len(solvers)} solver(s), "
          f"time limit {args.time_limit:g}s")

    rows: list[Result] = []
    fh = args.jsonl.open("a") if args.jsonl else None
    try:
        for i, m in enumerate(models, 1):
            print(f"[{i}/{len(models)}] {m.name}", flush=True)
            for s in solvers:
                r = dispatch(s, m, args.time_limit, args.tol, exe)
                rows.append(r)
                print(f"    {s:<20} {r.status:<14} "
                      f"obj={fmt_obj(r.objective):>14} {fmt_time(r.seconds):>9}"
                      + (f"  ({r.error})" if r.error else ""), flush=True)
                if fh:
                    fh.write(json.dumps(asdict(r)) + "\n")
                    fh.flush()
    finally:
        if fh:
            fh.close()

    report(rows, solvers, args.sgm_shift, args.obj_tol)
    if args.jsonl:
        print(f"\nJSONL: {args.jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
