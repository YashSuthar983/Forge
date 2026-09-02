#!/usr/bin/env python3
"""Gurobi baseline runner — EXTERNAL PROCESS ONLY.

Invoked by run_compare.py as a subprocess. Gurobi is never linked into libsor
and its output never appears in a SOR certificate. Requires a valid Gurobi
license (commercial, academic, or free restricted license).

Emits one JSON object on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def _status_label(code: int) -> str:
    try:
        import gurobipy as gp
    except ImportError:
        return str(code)
    mapping = {
        gp.GRB.OPTIMAL: "Optimal",
        gp.GRB.INFEASIBLE: "Infeasible",
        gp.GRB.INF_OR_UNBD: "InfOrUnbd",
        gp.GRB.UNBOUNDED: "Unbounded",
        gp.GRB.CUTOFF: "Cutoff",
        gp.GRB.ITERATION_LIMIT: "IterationLimit",
        gp.GRB.NODE_LIMIT: "NodeLimit",
        gp.GRB.TIME_LIMIT: "TimeLimit",
        gp.GRB.SOLUTION_LIMIT: "SolutionLimit",
        gp.GRB.INTERRUPTED: "Interrupted",
        gp.GRB.NUMERIC: "Numeric",
        gp.GRB.SUBOPTIMAL: "Suboptimal",
        gp.GRB.INPROGRESS: "InProgress",
        gp.GRB.USER_OBJ_LIMIT: "UserObjLimit",
        gp.GRB.WORK_LIMIT: "WorkLimit",
        gp.GRB.MEM_LIMIT: "MemLimit",
    }
    return mapping.get(code, f"Status{code}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mps")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument(
        "--method",
        default="-1",
        help="Gurobi Method: -1=auto, 0=primal simplex, 1=dual simplex, "
             "2=barrier, 3=concurrent, 4=deterministic concurrent, 5=PDHG",
    )
    args = ap.parse_args()

    out = {"solver": "gurobi", "kind": "external_process", "instance": args.mps}
    try:
        import gurobipy as gp
    except ImportError as e:
        out["status"] = "unavailable"
        out["error"] = f"gurobipy not importable: {e}"
        print(json.dumps(out))
        return 0

    out["version"] = f"gurobi-{gp.gurobi.version()}"
    try:
        t0 = time.perf_counter()
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("LogToConsole", 0)
            env.start()
            with gp.read(args.mps, env=env) as model:
                out["rows"] = int(model.NumConstrs)
                out["cols"] = int(model.NumVars)
                read_s = time.perf_counter() - t0

                model.setParam("TimeLimit", args.time_limit)
                if args.method != "-1":
                    model.setParam("Method", int(args.method))

                t1 = time.perf_counter()
                model.optimize()
                solve_s = time.perf_counter() - t1

                out["status"] = _status_label(model.Status)
                out["read_s"] = read_s
                out["solve_s"] = solve_s
                if model.SolCount > 0:
                    out["objective"] = float(model.ObjVal)
                try:
                    out["iterations"] = int(model.IterCount)
                except Exception:
                    pass
                try:
                    out["runtime"] = float(model.Runtime)
                except Exception:
                    pass
    except gp.GurobiError as e:
        out["status"] = "gurobi_error"
        out["error"] = str(e)
    except Exception as e:  # noqa: BLE001
        out["status"] = "crash"
        out["error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
