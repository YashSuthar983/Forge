#!/usr/bin/env python3
"""HiGHS baseline runner — EXTERNAL PROCESS ONLY.

Invoked by run_bench.py as a subprocess. HiGHS is never linked into libsor and
its output never appears in a SOR certificate. This is the "oracle, not source
browsing" workflow that sor/docs/clean_room_policy.md designates as the default:
run the reference solver as a separate binary and compare numbers.

Emits one JSON object on stdout.
"""
import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mps")
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args()

    out = {"solver": "highs", "kind": "external_process", "instance": args.mps}
    try:
        import highspy
    except ImportError as e:
        out["status"] = "unavailable"
        out["error"] = f"highspy not importable: {e}"
        print(json.dumps(out))
        return 0

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", args.time_limit)
    # SOR's current simplex is serial. Make the external oracle's resource
    # budget explicit rather than depending on a version-specific default.
    h.setOptionValue("threads", 1)
    out["version"] = str(h.version())

    try:
        t0 = time.perf_counter()
        st = h.readModel(args.mps)
        if str(st) not in ("HighsStatus.kOk", "HighsStatus.kWarning"):
            out["status"] = "read_error"
            out["error"] = str(st)
            print(json.dumps(out))
            return 0
        out["rows"] = int(h.getNumRow())
        out["cols"] = int(h.getNumCol())
        read_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        h.run()
        solve_s = time.perf_counter() - t1

        model_status = str(h.getModelStatus())
        out["status"] = model_status.replace("HighsModelStatus.k", "")
        out["objective"] = float(h.getObjectiveValue())
        out["read_s"] = read_s
        out["solve_s"] = solve_s
        info = h.getInfo()
        try:
            out["iterations"] = int(info.simplex_iteration_count)
        except Exception:
            pass
        try:
            out["primal_infeasibility"] = float(info.max_primal_infeasibility)
            out["dual_infeasibility"] = float(info.max_dual_infeasibility)
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001 - a baseline crash must not kill the sweep
        out["status"] = "crash"
        out["error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
