#!/usr/bin/env python3
"""Run SOR MILP (+ optional HiGHS external baseline) on miplib-easy subset.

Usage:
  python3 scripts/fetch_miplib_easy.py
  python3 scripts/run_miplib_easy.py --time-limit 30 --limit 10
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_sor(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        if line.startswith("status:"):
            out["status"] = line.split(":", 1)[1].strip().split()[0]
        elif line.startswith("proof_level:"):
            out["proof"] = line.split(":", 1)[1].strip().split()[0]
        elif line.startswith("objective:"):
            try:
                out["objective"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("mip gap:"):
            try:
                out["gap"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("nodes:"):
            try:
                out["nodes"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.strip().startswith("total"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out["total_ms"] = float(parts[1])
                except ValueError:
                    pass
    return out


def run_highs(mps: Path, time_limit: float) -> dict:
    script = ROOT / "scripts" / "run_highs_baseline.py"
    if not script.exists():
        return {"status": "skipped", "error": "no highs baseline script"}
    try:
        p = subprocess.run(
            [sys.executable, str(script), str(mps), "--time-limit", str(time_limit)],
            capture_output=True, text=True, timeout=time_limit + 60,
        )
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        return json.loads(line) if line else {"status": "error", "error": p.stderr[:200]}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances-dir",
                    default=str(ROOT / "benchmarks" / "miplib-easy" / "mps"))
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--max-nodes", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("-o", "--outdir", default=str(ROOT / "benchmarks" / "results"))
    args = ap.parse_args()

    inst_dir = Path(args.instances_dir)
    if not inst_dir.is_dir():
        print(f"error: {inst_dir} missing — run scripts/fetch_miplib_easy.py",
              file=sys.stderr)
        return 2
    instances = sorted(inst_dir.glob("*.mps"))
    if args.limit:
        instances = instances[: args.limit]
    if not instances:
        print(f"error: no .mps in {inst_dir}", file=sys.stderr)
        return 2

    exe = ROOT / "build" / "sor_solve"
    if not exe.exists():
        print(f"error: build sor_solve first ({exe})", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl = outdir / f"miplib-easy-{stamp}.jsonl"
    md = outdir / f"miplib-easy-{stamp}.md"

    rows = []
    for mps in instances:
        print(f"== {mps.name} ==")
        cmd = [str(exe), str(mps), "--engine", "milp",
               "--time-limit", str(args.time_limit),
               "--max-iter", str(args.max_nodes)]
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.time_limit + 30)
            wall = time.perf_counter() - t0
            sor = parse_sor(p.stdout)
            sor["wall_s"] = wall
            sor["rc"] = p.returncode
            if p.returncode not in (0, 4, 5) and "status" not in sor:
                sor["error"] = (p.stderr or p.stdout)[:300]
        except subprocess.TimeoutExpired:
            sor = {"status": "Timeout", "wall_s": args.time_limit, "error": "outer timeout"}

        base = {"status": "skipped"} if args.no_baseline else run_highs(mps, args.time_limit)
        agree = None
        if (sor.get("objective") is not None and
                base.get("status") == "Optimal" and base.get("objective") is not None):
            denom = 1.0 + abs(base["objective"])
            agree = abs(sor["objective"] - base["objective"]) / denom <= 1e-3

        row = {
            "instance": mps.stem,
            "sor": sor,
            "baseline": {"kind": "external_process", "solver": "highs", **base},
            "obj_agree": agree,
        }
        rows.append(row)
        with open(jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  SOR {sor.get('status')} obj={sor.get('objective')} "
              f"nodes={sor.get('nodes')}  HiGHS {base.get('status')} "
              f"agree={agree}")

    feasible = sum(1 for r in rows if r["sor"].get("status") in ("Feasible", "Optimal"))
    agree_n = sum(1 for r in rows if r["obj_agree"] is True)
    with open(md, "w") as f:
        f.write(f"# MIPLIB-easy subset\n\n")
        f.write(f"- Host: `{platform.node()}` · `{platform.platform()}`\n")
        f.write(f"- Time limit: {args.time_limit}s · max nodes: {args.max_nodes}\n")
        f.write(f"- SOR feasible/optimal: **{feasible}/{len(rows)}**\n")
        f.write(f"- Obj agree vs HiGHS (when both solved): **{agree_n}**\n\n")
        f.write("| Instance | SOR status | SOR obj | nodes | HiGHS | agree |\n")
        f.write("|----------|------------|---------|------:|-------|-------|\n")
        for r in rows:
            s, b = r["sor"], r["baseline"]
            f.write(
                f"| {r['instance']} | {s.get('status')} | {s.get('objective')} | "
                f"{s.get('nodes')} | {b.get('status')} | {r['obj_agree']} |\n"
            )
    print(f"\nwrote {jsonl}\nwrote {md}")
    print(f"summary: {feasible}/{len(rows)} feasible/optimal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
