#!/usr/bin/env python3
"""SciPy linprog baseline — EXTERNAL PROCESS ONLY.

Builds the LP from an MPS via PuLP's reader (data only), then solves with
SciPy's legacy interior-point method — a different algorithm stack from HiGHS/CBC.

Never linked into libsor. Emits one JSON object on stdout.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def _pulp_to_scipy(problem, variables):
    """Convert a PuLP LP to SciPy linprog arguments (minimize form)."""
    import pulp

    names = list(variables.keys())
    n = len(names)
    index = {name: i for i, name in enumerate(names)}

    c = np.zeros(n)
    if problem.objective is not None:
        for v, coef in problem.objective.items():
            c[index[v.name]] = float(coef)
        if problem.sense == pulp.LpMaximize:
            c = -c

    bounds = []
    for name in names:
        v = variables[name]
        lo = -np.inf if v.lowBound is None else float(v.lowBound)
        hi = np.inf if v.upBound is None else float(v.upBound)
        bounds.append((lo, hi))

    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for constr in problem.constraints.values():
        row = np.zeros(n)
        for v, coef in constr.items():
            row[index[v.name]] = float(coef)
        const = -float(constr.constant)  # PuLP stores LHS - RHS = 0 form via constant
        # PuLP constraint: sum a_i x_i + constant {<=,>=,==} 0
        # so sum a_i x_i {<=,>=,==} -constant
        rhs = -float(constr.constant)
        sense = constr.sense
        if sense == pulp.LpConstraintEQ:
            A_eq.append(row)
            b_eq.append(rhs)
        elif sense == pulp.LpConstraintLE:
            A_ub.append(row)
            b_ub.append(rhs)
        elif sense == pulp.LpConstraintGE:
            A_ub.append(-row)
            b_ub.append(-rhs)
        else:
            raise ValueError(f"unknown sense {sense}")

    return {
        "c": c,
        "A_ub": np.asarray(A_ub) if A_ub else None,
        "b_ub": np.asarray(b_ub) if b_ub else None,
        "A_eq": np.asarray(A_eq) if A_eq else None,
        "b_eq": np.asarray(b_eq) if b_eq else None,
        "bounds": bounds,
        "maximize": problem.sense == pulp.LpMaximize,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mps")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--method", default="highs",
                    choices=["highs", "highs-ds", "highs-ipm", "interior-point", "simplex"])
    args = ap.parse_args()

    out = {
        "solver": f"scipy-{args.method}",
        "kind": "external_process",
        "instance": args.mps,
    }
    try:
        import pulp
        import scipy
        from scipy.optimize import linprog
    except ImportError as e:
        out["status"] = "unavailable"
        out["error"] = f"import failed: {e}"
        print(json.dumps(out))
        return 0

    out["version"] = f"scipy-{scipy.__version__}"
    try:
        t0 = time.perf_counter()
        variables, problem = pulp.LpProblem.fromMPS(args.mps)
        lp = _pulp_to_scipy(problem, variables)
        out["cols"] = len(variables)
        out["rows"] = len(problem.constraints)
        read_s = time.perf_counter() - t0

        opts = {"maxiter": 100000}
        # time_limit only honored by HiGHS methods in recent SciPy
        if args.method.startswith("highs"):
            opts["time_limit"] = args.time_limit

        t1 = time.perf_counter()
        res = linprog(
            lp["c"],
            A_ub=lp["A_ub"],
            b_ub=lp["b_ub"],
            A_eq=lp["A_eq"],
            b_eq=lp["b_eq"],
            bounds=lp["bounds"],
            method=args.method,
            options=opts,
        )
        solve_s = time.perf_counter() - t1

        status_map = {
            0: "Optimal",
            1: "IterationLimit",
            2: "Infeasible",
            3: "Unbounded",
            4: "NumericalFailure",
        }
        out["status"] = status_map.get(res.status, f"status_{res.status}")
        out["message"] = str(res.message)
        obj = float(res.fun) if res.fun is not None and np.isfinite(res.fun) else None
        if obj is not None and lp["maximize"]:
            obj = -obj
        out["objective"] = obj
        out["read_s"] = read_s
        out["solve_s"] = solve_s
        out["iterations"] = int(getattr(res, "nit", 0) or 0)
    except Exception as e:  # noqa: BLE001
        out["status"] = "crash"
        out["error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
