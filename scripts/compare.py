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
import os
import platform
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
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
    "dual":    re.compile(r"^dual bound:\s+(\S+)", re.M),
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
    # The MILP suites' half of gate rule 1 needs this: a dual bound past the
    # published optimum on the bounding side means the search pruned a region
    # that contained the optimum, which no proof or incumbent check can see.
    dual_bound: float | None = None
    seconds: float | None = None      # solver-internal where available
    wall_s: float | None = None       # full process wall
    iterations: int | None = None
    proof: str | None = None
    violation: float | None = None
    rows: int | None = None
    cols: int | None = None
    nnz: int | None = None
    error: str | None = None
    repetition: int | None = None
    samples_s: list[float] | None = None
    wall_samples_s: list[float] | None = None


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
def affined_command(cmd: list[str], cpu: int | None) -> list[str]:
    """Return a command pinned to one CPU when taskset is available."""
    if cpu is None:
        return cmd
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("--cpu requested but taskset is not available")
    return [taskset, "-c", str(cpu), *cmd]


def build_sor_command(model: Path, engine: str, backend: str,
                      time_limit: float, tol: float, exe: Path,
                      method: str | None, basis_update: str,
                      max_iter: int | None,
                      pricing: str = "choose",
                      dual_cost_perturbation: float = 0.0,
                      sor_extra: list[str] | None = None) -> list[str]:
    """Build the exact CLI command for one SOR solver specification.

    `sor_extra` is appended verbatim for SOR solvers. It exists so a tuning
    parameter can be swept and gated without this file growing a mirror of
    sor_solve's whole CLI; whatever is passed is recorded in the environment
    record, so a JSONL still says exactly what produced it."""
    flags: list[str] = []
    eng = engine
    # primal/dual are simplex methods, not standalone engines.
    if engine in ("auto", "primal", "dual"):
        eng, method = "simplex", engine
    # Two HPR variants are flags on the same engine rather than engines.
    if engine in ("hpr-full", "hpr-vanilla"):
        eng, flags = "hpr", [f"--{engine.replace('hpr-', 'hpr-')}"]
    cmd = [str(exe), str(model), "--engine", eng, "--backend", backend,
           "--tol", str(tol), "--time-limit", str(time_limit), *flags]
    if method is not None and eng in ("simplex", "milp"):
        cmd += ["--method", method]
    if eng in ("simplex", "milp"):
        cmd += ["--basis-update", basis_update, "--pricing", pricing]
        if dual_cost_perturbation > 0.0:
            cmd += ["--dual-cost-perturbation",
                    str(dual_cost_perturbation)]
    if max_iter is not None:
        cmd += ["--max-iter", str(max_iter)]
    if sor_extra:
        cmd += list(sor_extra)
    return cmd


def run_sor(model: Path, engine: str, backend: str, time_limit: float,
            tol: float, exe: Path, label: str, method: str | None = None,
            basis_update: str = "product", max_iter: int | None = None,
            cpu: int | None = None, pricing: str = "choose",
            dual_cost_perturbation: float = 0.0,
            sor_extra: list[str] | None = None) -> Result:
    r = Result(solver=label, instance=model.name)
    try:
        cmd = affined_command(
            build_sor_command(model, engine, backend, time_limit, tol, exe,
                              method, basis_update, max_iter, pricing,
                              dual_cost_perturbation, sor_extra), cpu)
    except RuntimeError as e:
        r.status, r.error = "error", str(e)
        return r
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
        r.dual_bound = _num(out, "dual")
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
def highs_worker(model: Path, time_limit: float) -> int:
    """Isolated HiGHS worker. Its only stdout is one JSON Result record."""
    r = Result(solver="highs", instance=model.name)
    try:
        import highspy
    except ImportError:
        r.status = "unavailable"
        r.error = "highspy not installed (see benchmarks/requirements-baseline.txt)"
        print(json.dumps(asdict(r)))
        return 0
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
    print(json.dumps(asdict(r)))
    return 0


def run_highs(model: Path, time_limit: float, label: str,
              cpu: int | None = None) -> Result:
    """Run HiGHS in a fresh process, matching SOR's process isolation."""
    r = Result(solver=label, instance=model.name)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_highs-worker",
           str(model), str(time_limit)]
    try:
        cmd = affined_command(cmd, cpu)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=time_limit + 30)
        outer_wall = time.perf_counter() - t0
        payload = json.loads(p.stdout.strip())
        r = Result(**payload)
        r.solver = label
        r.wall_s = outer_wall
        if p.returncode != 0 and r.status not in ("crash", "unavailable"):
            r.status = "crash"
            r.error = (p.stderr or f"worker exited {p.returncode}")[:200]
    except subprocess.TimeoutExpired:
        r.status, r.seconds, r.wall_s = "timeout", time_limit, time_limit
    except Exception as e:  # noqa: BLE001
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
             exe: Path, method: str | None = None,
             basis_update: str = "product", max_iter: int | None = None,
             cpu: int | None = None, pricing: str = "choose",
             dual_cost_perturbation: float = 0.0,
             sor_extra: list[str] | None = None) -> Result:
    """`sor:<engine>[:<backend>]`, or an external baseline name."""
    key = spec.strip().lower()
    if key.startswith("sor"):
        parts = key.split(":")
        engine = parts[1] if len(parts) > 1 else "simplex"
        backend = parts[2] if len(parts) > 2 else "cpu"
        return run_sor(model, engine, backend, time_limit, tol, exe, spec,
                       method, basis_update, max_iter, cpu, pricing,
                       dual_cost_perturbation, sor_extra)
    if key == "highs":
        return run_highs(model, time_limit, spec, cpu)
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


_SOR_PROOFS = {
    "ProvedOptimalFP", "ProvedOptimalExact", "ProvedOptimalCertified",
    "ProvedKKT", "ProvedGlobalEpsilon",
}


def is_optimal(r: Result) -> bool:
    return r.status.strip().lower() in ("optimal", "solved", "1")


def is_certified_success(r: Result) -> bool:
    """Optimal is success for SOR only when its independent proof gate passed."""
    if not is_optimal(r):
        return False
    return not r.solver.lower().startswith("sor") or r.proof in _SOR_PROOFS


def objectives_agree(a: float | None, b: float | None,
                     abs_tol: float, rel_tol: float) -> bool:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return False
    return abs(a - b) <= abs_tol + rel_tol * max(abs(a), abs(b))


def choose_reference(results: dict[str, Result], solvers: list[str],
                     requested: str | None) -> Result | None:
    """Choose a certified reference; never use an interrupted objective."""
    names = {name.lower(): name for name in results}
    preferred = requested
    if preferred is None and "highs" in names:
        preferred = names["highs"]
    if preferred is not None:
        key = names.get(preferred.lower())
        if key is None:
            return None
        candidate = results[key]
        return candidate if is_certified_success(candidate) and (
            candidate.objective is not None and math.isfinite(candidate.objective)) else None
    for name in solvers:
        candidate = results.get(name)
        if candidate and is_certified_success(candidate) and (
                candidate.objective is not None and math.isfinite(candidate.objective)):
            return candidate
    return None


def aggregate_repetitions(runs: list[Result]) -> Result:
    """Use a median representative, but surface any failed/flaky repetition."""
    if not runs:
        raise ValueError("cannot aggregate an empty run list")
    failed = [r for r in runs if not is_certified_success(r)]
    if failed:
        chosen = replace(failed[0])
    else:
        timed = [r for r in runs if r.seconds is not None]
        chosen = replace(min(timed, key=lambda r: abs(
            r.seconds - statistics.median(x.seconds for x in timed)))
                         if timed else runs[0])
    chosen.repetition = None
    chosen.samples_s = [r.seconds for r in runs if r.seconds is not None]
    chosen.wall_samples_s = [r.wall_s for r in runs if r.wall_s is not None]
    return chosen


def report(rows: list[Result], solvers: list[str], shift: float,
           abs_tol: float, rel_tol: float, time_limit: float,
           reference: str | None) -> dict[str, int]:
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

    # Objective agreement against an explicit or certified reference.
    print()
    ref_label = reference or ("highs" if "highs" in [s.lower() for s in solvers]
                              else "first certified solver")
    print(f"{'instance':<{wid}}  objective (reference: {ref_label})")
    print("-" * (wid + 44))
    mismatches = 0
    unchecked = 0
    agreements: dict[tuple[str, str], bool] = {}
    for inst in sorted(by_inst):
        ref = choose_reference(by_inst[inst], solvers, reference)
        if ref is None:
            unchecked += 1
        line = f"{inst:<{wid}}  "
        for s in solvers:
            r = by_inst[inst].get(s)
            if r is None or r.objective is None:
                line += f"{'-':<{colw}}  "
                continue
            tag = ""
            if ref is not None and is_certified_success(r):
                match = objectives_agree(r.objective, ref.objective,
                                         abs_tol, rel_tol)
                agreements[(inst, s)] = match
                if not match:
                    tag = " MISMATCH"
                    mismatches += 1
            elif is_certified_success(r):
                tag = " UNCHECKED"
            line += f"{fmt_obj(r.objective) + tag:<{colw}}  "
        print(line)

    print()
    print(f"{'solver':<{colw}} {'proved':>8} {'correct':>9} "
          f"{'pSGM(s)':>10} {'PAR2(s)':>10}")
    print("-" * (colw + 43))
    for s in solvers:
        rs = [by_inst[i].get(s) for i in by_inst]
        rs = [r for r in rs if r is not None]
        proved = [r for r in rs if is_certified_success(r)]
        correct = sum(1 for r in proved if agreements.get((r.instance, s), False))
        penalty = 2.0 * time_limit
        penalized = [r.seconds if (is_certified_success(r) and
                      agreements.get((r.instance, s), False) and
                      r.seconds is not None) else penalty for r in rs]
        sgm = shifted_geomean(penalized, shift)
        par2 = sum(penalized) / len(penalized) if penalized else None
        print(f"{s:<{colw}} {len(proved):>4}/{len(rs):<3} "
              f"{correct:>5}/{len(rs):<3} "
              f"{(f'{sgm:.4f}' if sgm is not None else '-'):>10} "
              f"{(f'{par2:.4f}' if par2 is not None else '-'):>10}")
    if mismatches:
        print(f"\n{mismatches} objective mismatch(es) above abs={abs_tol:g}, "
              f"rel={rel_tol:g}; timings are penalized.")
    if unchecked:
        print(f"\n{unchecked} instance(s) had no certified reference objective; "
              "timings are penalized.")
    unavailable = sum(r.status.lower() == "unavailable" for r in rows)
    return {"mismatches": mismatches, "unchecked": unchecked,
            "unavailable": unavailable}


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


def environment_record(args: argparse.Namespace, solvers: list[str],
                       models: list[Path], exe: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip())
    except Exception:  # noqa: BLE001
        commit, dirty = None, None
    return {
        "record": "environment",
        "commit": commit,
        "dirty": dirty,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cpu_affinity": args.cpu,
        "executable": str(exe.resolve()),
        "solvers": solvers,
        "models": len(models),
        "time_limit_s": args.time_limit,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "method": args.method,
        "pricing": args.pricing,
        "dual_cost_perturbation": args.dual_cost_perturbation,
        "basis_update": args.basis_update,
        "sor_extra_args": list(args.sor_arg),
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "seed": args.seed,
    }


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
    ap.add_argument("--method", choices=("auto", "primal", "dual"), default=None,
                    help="simplex method for sor:simplex/sor:milp; sor:primal and "
                         "sor:dual override it")
    ap.add_argument("--pricing", choices=("choose", "dantzig", "devex", "dse"),
                    default="choose", help="simplex pricing strategy")
    ap.add_argument("--dual-cost-perturbation", type=float, default=0.0,
                    help="experimental dual-simplex cost perturbation multiplier")
    ap.add_argument("--basis-update", choices=("product", "ft"), default="product")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="explicit solver iteration/node cap")
    ap.add_argument("--warmups", type=int, default=0)
    ap.add_argument("--repetitions", type=int, default=1,
                    help="measured repetitions; summary uses the median")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for solver-order randomization")
    ap.add_argument("--cpu", type=int, default=None,
                    help="pin child solver processes to this logical CPU")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N models")
    ap.add_argument("--sgm-shift", type=float, default=1.0,
                    help="shift for the shifted geometric mean, in seconds")
    ap.add_argument("--obj-tol", "--obj-rel-tol", dest="obj_rel_tol",
                    type=float, default=1e-4,
                    help="relative tolerance for objective agreement")
    ap.add_argument("--obj-abs-tol", type=float, default=1e-7,
                    help="absolute tolerance for objective agreement")
    ap.add_argument("--reference", default=None,
                    help="reference solver label (default: highs when selected, "
                         "otherwise first certified solver)")
    ap.add_argument("--allow-unavailable", action="store_true",
                    help="do not fail solely because a requested solver is unavailable")
    ap.add_argument("--allow-unchecked", action="store_true",
                    help="do not fail solely because no certified reference exists")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="also append one JSON record per run to this file")
    ap.add_argument("--exe", type=Path, default=None,
                    help="sor_solve binary (default: <root>/build/sor_solve)")
    ap.add_argument("--sor-arg", action="append", default=[],
                    metavar="FLAG",
                    help="extra flag passed verbatim to sor_solve; repeatable. "
                         "For sweeping a tuning parameter without teaching "
                         "this script every sor_solve option. Use the "
                         "=form for flags: "
                         "--sor-arg=--refactor-work-ratio --sor-arg=2.0")
    args = ap.parse_args()

    if args.time_limit <= 0 or args.repetitions <= 0 or args.warmups < 0:
        ap.error("time limit and repetitions must be positive; warmups cannot be negative")
    if args.max_iter is not None and args.max_iter <= 0:
        ap.error("--max-iter must be positive")
    if args.cpu is not None and args.cpu < 0:
        ap.error("--cpu cannot be negative")
    if (not math.isfinite(args.dual_cost_perturbation) or
            args.dual_cost_perturbation < 0):
        ap.error("--dual-cost-perturbation must be finite and nonnegative")

    exe = args.exe or (ROOT / "build" / "sor_solve")
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    if args.reference is not None and args.reference.lower() not in {
            s.lower() for s in solvers}:
        ap.error("--reference must name one of --solvers")
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
    if fh:
        fh.write(json.dumps(environment_record(args, solvers, models, exe)) + "\n")
        fh.flush()
    rng = random.Random(args.seed)
    try:
        for i, m in enumerate(models, 1):
            print(f"[{i}/{len(models)}] {m.name}", flush=True)
            order = list(solvers)
            rng.shuffle(order)
            for s in order:
                for _ in range(args.warmups):
                    warm = dispatch(s, m, args.time_limit, args.tol, exe,
                                    args.method, args.basis_update,
                                    args.max_iter, args.cpu, args.pricing,
                                    args.dual_cost_perturbation,
                                    args.sor_arg)
                    if warm.status.lower() == "unavailable":
                        break
                measured: list[Result] = []
                for rep in range(args.repetitions):
                    r = dispatch(s, m, args.time_limit, args.tol, exe,
                                 args.method, args.basis_update,
                                 args.max_iter, args.cpu, args.pricing,
                                 args.dual_cost_perturbation, args.sor_arg)
                    r.repetition = rep
                    measured.append(r)
                    if fh:
                        record = asdict(r)
                        record["record"] = "run"
                        fh.write(json.dumps(record) + "\n")
                        fh.flush()
                    if r.status.lower() == "unavailable":
                        break
                r = aggregate_repetitions(measured)
                rows.append(r)
                print(f"    {s:<20} {r.status:<14} "
                      f"obj={fmt_obj(r.objective):>14} {fmt_time(r.seconds):>9}"
                      + (f"  ({r.error})" if r.error else ""), flush=True)
    finally:
        if fh:
            fh.close()

    summary = report(rows, solvers, args.sgm_shift, args.obj_abs_tol,
                     args.obj_rel_tol, args.time_limit, args.reference)
    if args.jsonl:
        print(f"\nJSONL: {args.jsonl}")
    failed = summary["mismatches"] > 0
    failed = failed or (summary["unchecked"] > 0 and not args.allow_unchecked)
    failed = failed or (summary["unavailable"] > 0 and not args.allow_unavailable)
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--_highs-worker":
        sys.exit(highs_worker(Path(sys.argv[2]), float(sys.argv[3])))
    sys.exit(main())
