#!/usr/bin/env python3
"""CBC baseline runner — EXTERNAL PROCESS ONLY (via PuLP's bundled CBC).

Never linked into libsor. Emits one JSON object on stdout.
"""
from __future__ import annotations

import argparse
import json
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mps")
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args()

    out = {"solver": "cbc", "kind": "external_process", "instance": args.mps}
    try:
        import pulp
    except ImportError as e:
        out["status"] = "unavailable"
        out["error"] = f"pulp not importable: {e}"
        print(json.dumps(out))
        return 0

    out["version"] = f"pulp-{pulp.__version__}+PULP_CBC_CMD"
    try:
        t0 = time.perf_counter()
        _vars, problem = pulp.LpProblem.fromMPS(args.mps)
        out["cols"] = len(_vars)
        out["rows"] = len(problem.constraints)
        read_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        status = problem.solve(
            pulp.PULP_CBC_CMD(msg=False, timeLimit=args.time_limit, options=["sec", str(args.time_limit)])
        )
        solve_s = time.perf_counter() - t1

        label = pulp.LpStatus.get(status, str(status))
        out["status"] = label
        out["read_s"] = read_s
        out["solve_s"] = solve_s
        obj = pulp.value(problem.objective)
        out["objective"] = float(obj) if obj is not None else None
    except Exception as e:  # noqa: BLE001
        out["status"] = "crash"
        out["error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
