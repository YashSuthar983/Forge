#!/usr/bin/env python3
"""WP-K evidence: milp.policy=latest vs classical on a tiny frozen set.

Runs each instance twice (latest / classical), writes one JSONL record per
(instance, policy) with proved/gap/nodes/time. Defaults to a frozen trio of
tiny fixtures under benchmarks/milp-ablate-tiny/ (generated if missing), or
accepts --suite / explicit MPS paths.

For paper-facing MIPLIB claim-set tables (crash census + latest_stable arm),
prefer ``scripts/milp_paper_evidence.py``.

  scripts/milp_latest_ablate.py --exe build/sor_solve -t 5 \\
      --out /tmp/milp_ablate.jsonl

  scripts/milp_latest_ablate.py --exe build/sor_solve --suite benchmarks/miplib-easy \\
      --limit 3 -t 10 --out /tmp/easy3.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIELDS = {
    "status": re.compile(r"^status:\s+(\S+)"),
    "objective": re.compile(r"^objective:\s+(\S+)"),
    "dual_bound": re.compile(r"^dual bound:\s+(\S+)"),
    "gap": re.compile(r"^mip gap:\s+(\S+)"),
    "nodes": re.compile(r"^nodes:\s+(\d+)"),
    "warm_start_hits": re.compile(r"^warm_start_hits:\s+(\d+)"),
    "termination": re.compile(r"^termination:\s+(.+)$"),
}

# Tiny knapsack / frac-branch fixtures (same family as unit tests).
TINY_FIXTURES: dict[str, str] = {
    "knap3.mps": """\
NAME          KNAP3
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      -5             CAP       4
    X2        COST      -3             CAP       2
    X3        COST      -2             CAP       1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       5
BOUNDS
 UI BND       X1        1
 UI BND       X2        1
 UI BND       X3        1
ENDATA
""",
    "fracbr.mps": """\
NAME          FRACBR
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      -1             CAP       2
    X2        COST      -1             CAP       2
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       5
BOUNDS
 UI BND       X1        2
 UI BND       X2        2
ENDATA
""",
    "knap4.mps": """\
NAME          KNAP4
ROWS
 N  COST
 L  CAP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        COST      -10            CAP       5
    X2        COST      -6             CAP       3
    X3        COST      -4             CAP       2
    X4        COST      -2             CAP       1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS       CAP       7
BOUNDS
 UI BND       X1        1
 UI BND       X2        1
 UI BND       X3        1
 UI BND       X4        1
ENDATA
""",
}


def ensure_tiny_suite(dir_path: Path) -> list[Path]:
    dir_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, body in TINY_FIXTURES.items():
        p = dir_path / name
        if not p.exists():
            p.write_text(body)
        paths.append(p)
    return paths


def discover_suite(suite: Path, limit: int | None) -> list[Path]:
    if suite.is_file():
        return [suite]
    mps_dir = suite / "mps" if (suite / "mps").is_dir() else suite
    paths = sorted(mps_dir.glob("*.mps")) + sorted(mps_dir.glob("*.mps.gz"))
    if limit is not None:
        paths = paths[:limit]
    return paths


def parse_run(stdout: str, wall_s: float, rc: int) -> dict[str, object]:
    rec: dict[str, object] = {"wall_s": wall_s, "returncode": rc}
    for line in stdout.splitlines():
        for key, pat in FIELDS.items():
            m = pat.match(line)
            if m:
                rec.setdefault(key, m.group(1))
    for key in ("objective", "dual_bound", "gap"):
        if key in rec:
            try:
                rec[key] = float(rec[key])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                rec[key] = None
    if "nodes" in rec:
        try:
            rec["nodes"] = int(rec["nodes"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            rec["nodes"] = None
    if "warm_start_hits" in rec:
        try:
            rec["warm_start_hits"] = int(rec["warm_start_hits"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    status = str(rec.get("status", "NO_OUTPUT" if rc == 0 else "CRASH"))
    proved = status == "Optimal"
    gap = rec.get("gap")
    rec["proved"] = proved
    rec["gap"] = gap if isinstance(gap, float) and math.isfinite(gap) else None
    return rec


def run_one(
    exe: Path,
    model: Path,
    policy: str,
    time_limit: float,
    extra: list[str],
) -> dict[str, object]:
    cmd = [
        str(exe),
        str(model),
        "--engine",
        "milp",
        "--milp-policy",
        policy,
        "--time-limit",
        str(time_limit),
    ] + extra
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 60,
            check=False,
        )
        out, err, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        rc = -9
    wall = time.perf_counter() - t0
    rec = parse_run(out, wall, rc)
    # Surface hard failures so JSONL is usable on crashing Latest builds.
    if rc == -9 or rc == 124:
        rec["status"] = "TIMEOUT"
        rec["proved"] = False
        rec["crashed"] = True
    elif rc < 0 or rc == 139 or rc > 128:
        rec["status"] = "SEGFAULT"
        rec["proved"] = False
        rec["crashed"] = True
    elif "bad_alloc" in (out + err).lower():
        rec["status"] = "BAD_ALLOC"
        rec["proved"] = False
        rec["crashed"] = True
    elif rc != 0 and "status:" not in out:
        rec["status"] = "CRASH"
        rec["proved"] = False
        rec["crashed"] = True
    else:
        rec["crashed"] = False
    rec["model"] = model.name
    rec["path"] = str(model)
    rec["policy"] = policy
    rec["record"] = "instance"
    return rec


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_policy: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by_policy.setdefault(str(r["policy"]), []).append(r)

    summary: dict[str, object] = {"record": "summary", "policies": {}}
    for policy, items in by_policy.items():
        n = len(items)
        proved = sum(1 for r in items if r.get("proved"))
        gaps = [float(r["gap"]) for r in items if isinstance(r.get("gap"), float)]
        nodes = [int(r["nodes"]) for r in items if isinstance(r.get("nodes"), int)]
        times = [float(r["wall_s"]) for r in items if isinstance(r.get("wall_s"), float)]
        summary["policies"][policy] = {
            "n": n,
            "proved": proved,
            "proved_frac": proved / n if n else 0.0,
            "median_gap": sorted(gaps)[len(gaps) // 2] if gaps else None,
            "median_nodes": sorted(nodes)[len(nodes) // 2] if nodes else None,
            "median_wall_s": sorted(times)[len(times) // 2] if times else None,
            "sum_wall_s": sum(times) if times else 0.0,
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=ROOT / "build" / "sor_solve")
    ap.add_argument(
        "--suite",
        type=Path,
        default=None,
        help="directory or file; default = benchmarks/milp-ablate-tiny",
    )
    ap.add_argument("models", nargs="*", type=Path, help="explicit MPS paths")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-t", "--time-limit", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "milp_latest_ablate.jsonl")
    ap.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra flags after -- passed to both arms",
    )
    args = ap.parse_args(argv)

    exe = args.exe
    if not exe.is_file():
        print(f"error: missing solver binary {exe}", file=sys.stderr)
        return 2

    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.models:
        models = list(args.models)
    elif args.suite is not None:
        models = discover_suite(args.suite, args.limit)
    else:
        tiny = ROOT / "benchmarks" / "milp-ablate-tiny"
        models = ensure_tiny_suite(tiny)
        if args.limit is not None:
            models = models[: args.limit]

    if not models:
        print("error: no models to run", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"record": "summary", "policies": {}}
    with args.out.open("w", encoding="utf-8") as fh:
        for model in models:
            for policy in ("latest", "classical"):
                rec = run_one(exe, model, policy, args.time_limit, extra)
                rows.append(rec)
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                print(
                    f"{rec['model']:20s} policy={policy:10s} "
                    f"proved={rec['proved']} gap={rec.get('gap')} "
                    f"nodes={rec.get('nodes')} wall={rec['wall_s']:.2f}s",
                    flush=True,
                )
        summary = summarize(rows)
        fh.write(json.dumps(summary, sort_keys=True) + "\n")

    print(json.dumps(summary["policies"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
