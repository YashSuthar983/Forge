#!/usr/bin/env python3
"""Merge batched solver_accl results into an existing fair-compare JSONL."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_compare import shifted_geomean, is_solved  # noqa: E402

JULIA = os.environ.get("JULIA", str(Path.home() / ".juliaup/bin/julia"))
SOLVER_ACCL_ROOT = os.environ.get("SOLVER_ACCL_ROOT", "/tmp/solver_accl")


def batch(engine: str, inst_dir: Path, time_limit: float) -> dict[str, dict]:
    p = subprocess.run(
        [
            JULIA, f"--project={SOLVER_ACCL_ROOT}",
            str(ROOT / "scripts/batch_solver_accl_bench.jl"),
            str(inst_dir), "--engine", engine,
            "--time-limit", str(time_limit),
        ],
        capture_output=True, text=True,
        env={**os.environ, "SOLVER_ACCL_ROOT": SOLVER_ACCL_ROOT},
        timeout=3600,
    )
    if p.returncode != 0:
        print(p.stderr[-800:], file=sys.stderr)
    out: dict[str, dict] = {}
    for line in p.stdout.splitlines():
        if not line.strip().startswith("{"):
            continue
        j = json.loads(line)
        if j.get("loadable", True):
            out[j["instance"]] = j
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="existing compare-solver_accl-fair-*.jsonl with SOR rows")
    ap.add_argument("--time-limit", type=float, default=20.0)
    args = ap.parse_args()

    old = Path(args.input)
    inst_dir = ROOT / "benchmarks/netlib/mps"
    rows = []
    header_lines = []
    for line in old.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("record") == "instance":
            rows.append(rec)
        elif rec.get("record") in ("environment", "skipped"):
            header_lines.append(line)

    print("batch accl-simplex...", flush=True)
    accl_s = batch("simplex", inst_dir, args.time_limit)
    print("batch accl-pdhg...", flush=True)
    accl_p = batch("pdhg", inst_dir, args.time_limit)
    print(f"got {len(accl_s)} simplex, {len(accl_p)} pdhg", flush=True)

    obj_tol = 1e-4
    for row in rows:
        stem = row["instance"]
        for eng, data in (("simplex", accl_s), ("pdhg", accl_p)):
            j = data.get(stem, {})
            row["results"][f"accl-{eng}"] = {
                "name": f"accl-{eng}",
                "status": j.get("status", "not_run"),
                "objective": j.get("objective"),
                "wall_s": j.get("solve_s"),
                "iterations": j.get("iterations"),
                "error": j.get("error"),
            }
        ref = row.get("ref_objective")
        for n, rd in row["results"].items():
            if ref is None or rd.get("objective") is None:
                row["agreements"][n] = None
            else:
                g = abs(rd["objective"] - ref) / (1 + abs(ref))
                row["agreements"][n] = g <= obj_tol

    names = ["SOR-simplex", "SOR-pdhg", "SOR-hpr",
             "accl-simplex", "accl-pdhg", "highs"]
    solved = {n: 0 for n in names}
    match = {n: 0 for n in names}
    times = {n: [] for n in names}
    for row in rows:
        for n in names:
            rd = row["results"].get(n, {})
            if is_solved(rd.get("status", "")):
                solved[n] += 1
                if rd.get("wall_s") is not None:
                    times[n].append(rd["wall_s"])
            if row["agreements"].get(n) is True:
                match[n] += 1
    sgm = {n: shifted_geomean(times[n]) for n in names}

    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl = ROOT / f"benchmarks/results/compare-solver_accl-fair-{stamp}.jsonl"
    md = ROOT / f"benchmarks/results/compare-solver_accl-fair-{stamp}.md"
    with jsonl.open("w") as f:
        for line in header_lines:
            f.write(line + "\n")
        for row in rows:
            f.write(json.dumps({"record": "instance", **row}) + "\n")
        f.write(json.dumps({
            "record": "summary",
            "solved": solved,
            "obj_match_vs_highs": match,
            "sgm_wall_s": sgm,
        }) + "\n")

    lines = [
        "# SOR vs solver_accl (fair batch benchmark)",
        "",
        "- Competitor: [solver_accl](https://github.com/shreyas-omkar/solver_accl)",
        f"- Loadable instances: {len(rows)}/93 Netlib",
        f"- Time limit: {args.time_limit}s per solver",
        "- accl times: single warmed Julia process per engine",
        "",
        "## Summary (HiGHS reference)",
        "",
        "| Solver | Solved | Obj match | SGM time (s) |",
        "|--------|-------:|----------:|-------------:|",
    ]
    for n in names:
        s = f"{sgm[n]:.4f}" if sgm[n] is not None else "—"
        lines.append(f"| {n} | {solved[n]}/{len(rows)} | {match[n]}/{len(rows)} | {s} |")
    lines += ["", f"JSONL: `{jsonl}`"]
    md.write_text("\n".join(lines) + "\n")

    print("\nSUMMARY")
    for n in names:
        s = f"{sgm[n]:.4f}s" if sgm[n] is not None else "—"
        print(f"{n:18s} solved {solved[n]:2d}/{len(rows)} "
              f"match {match[n]:2d}/{len(rows)} sgm {s}")
    print(f"Markdown: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
