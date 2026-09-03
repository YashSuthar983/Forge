#!/usr/bin/env python3
"""CBC baseline runner — EXTERNAL PROCESS ONLY (via PuLP's bundled CBC).

Never linked into libsor. Emits one JSON object on stdout.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from pathlib import Path


def classify_cbc_log(log: str, pulp_label: str) -> tuple[str, str | None]:
    """Normalize CBC's final termination reason.

    CBC may print an intermediate ``Optimal - objective value`` line when a
    feasible incumbent is found, then stop later on a time/node limit. PuLP's
    integer status alone can therefore say ``Optimal`` for an incomplete run.
    The terminal result line is authoritative here.
    """
    stop_patterns = (
        r"Result - Stopped on time limit",
        r"Result - Stopped on iterations",
        r"Result - Stopped on nodes",
        r"Result - User ctrl-c",
        r"Exiting on maximum time",
        r"Exiting on maximum nodes",
    )
    for pattern in stop_patterns:
        if re.search(pattern, log, re.IGNORECASE):
            return "TimeLimit", pattern
    if re.search(r"Result - Problem proven infeasible", log, re.IGNORECASE):
        return "Infeasible", "Problem proven infeasible"
    if re.search(r"Result - Problem proven unbounded", log, re.IGNORECASE):
        return "Unbounded", "Problem proven unbounded"
    if re.search(r"Result - Optimal solution found", log, re.IGNORECASE):
        return "Optimal", "Optimal solution found"
    return pulp_label, None


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
    log_path: str | None = None
    try:
        t0 = time.perf_counter()
        _vars, problem = pulp.LpProblem.fromMPS(args.mps)
        out["cols"] = len(_vars)
        out["rows"] = len(problem.constraints)
        read_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        with tempfile.NamedTemporaryFile(prefix="sor-cbc-", suffix=".log",
                                         delete=False) as log_file:
            log_path = log_file.name
        status = problem.solve(
            pulp.PULP_CBC_CMD(msg=False, timeLimit=args.time_limit,
                              options=["sec", str(args.time_limit)],
                              logPath=log_path)
        )
        solve_s = time.perf_counter() - t1

        label = pulp.LpStatus.get(status, str(status))
        log = Path(log_path).read_text(errors="replace") if log_path else ""
        normalized, termination = classify_cbc_log(log, label)
        out["status"] = normalized
        out["pulp_status"] = label
        if termination:
            out["termination"] = termination
        out["read_s"] = read_s
        out["solve_s"] = solve_s
        obj = pulp.value(problem.objective)
        out["objective"] = float(obj) if obj is not None else None
    except Exception as e:  # noqa: BLE001
        out["status"] = "crash"
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        if log_path:
            try:
                Path(log_path).unlink()
            except OSError:
                pass

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
