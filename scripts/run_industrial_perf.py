#!/usr/bin/env python3
"""Generate a size ladder of industrial instances, then compare SOR vs open source.

Sizes are deliberately larger than the demo seed-42 toys so wall-time / scale
show up. Baselines are EXTERNAL PROCESSES only (HiGHS, CBC).

Usage:
  python3 scripts/run_industrial_perf.py
  python3 scripts/run_industrial_perf.py --time-limit 60 --sizes S,M,L
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
VENV_PY = ROOT / "benchmarks" / ".venv-baseline" / "bin" / "python"

# (tier, kind, engine, gen_args, baselines)
# Rows/cols grow roughly: S → demo+, M → thousands, L → 10k+, XL → bigger.
LADDER = [
    # Crude blending LP
    ("S",  "blend_lp",    "simplex", ["blend", "--crudes", "12", "--products", "8"],
     ["highs"]),
    ("M",  "blend_lp",    "simplex", ["blend", "--crudes", "60", "--products", "30"],
     ["highs"]),
    ("L",  "blend_lp",    "simplex", ["blend", "--crudes", "200", "--products", "80"],
     ["highs"]),
    ("XL", "blend_lp",    "simplex", ["blend", "--crudes", "500", "--products", "150"],
     ["highs"]),
    ("XXL", "blend_lp",   "simplex", ["blend", "--crudes", "1200", "--products", "300"],
     ["highs"]),
    ("HUGE", "blend_lp",  "simplex", ["blend", "--crudes", "2500", "--products", "500"],
     ["highs"]),
    # Refinery scheduling MILP (vars = 2 * periods * units)
    ("S",  "schedule_milp", "milp", ["schedule", "--periods", "24", "--units", "6"],
     ["highs", "cbc"]),
    ("M",  "schedule_milp", "milp", ["schedule", "--periods", "48", "--units", "12"],
     ["highs", "cbc"]),
    ("L",  "schedule_milp", "milp", ["schedule", "--periods", "96", "--units", "20"],
     ["highs", "cbc"]),
    ("XL", "schedule_milp", "milp", ["schedule", "--periods", "168", "--units", "30"],
     ["highs", "cbc"]),
    ("XXL", "schedule_milp", "milp", ["schedule", "--periods", "336", "--units", "40"],
     ["highs", "cbc"]),
    ("HUGE", "schedule_milp", "milp", ["schedule", "--periods", "672", "--units", "50"],
     ["highs", "cbc"]),
    # Economic dispatch QP
    ("S",  "dispatch_qp", "qp", ["dispatch", "--gens", "40"],
     ["highs-qp"]),
    ("M",  "dispatch_qp", "qp", ["dispatch", "--gens", "200"],
     ["highs-qp"]),
    ("L",  "dispatch_qp", "qp", ["dispatch", "--gens", "1000"],
     ["highs-qp"]),
    ("XL", "dispatch_qp", "qp", ["dispatch", "--gens", "4000"],
     ["highs-qp"]),
    ("XXL", "dispatch_qp", "qp", ["dispatch", "--gens", "8000"],
     ["highs-qp"]),
    ("HUGE", "dispatch_qp", "qp", ["dispatch", "--gens", "10000"],
     ["highs-qp"]),
]


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
        elif line.startswith("rows x cols:"):
            # "rows x cols:       8 x 7   nnz 26"
            parts = line.split(":", 1)[1].replace("x", " ").replace("nnz", " ").split()
            try:
                out["rows"] = int(parts[0])
                out["cols"] = int(parts[1])
                out["nnz"] = int(parts[2])
            except (ValueError, IndexError):
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


def obj_agree(a, b, tol=1e-3):
    if a is None or b is None:
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return abs(a - b) / (1.0 + abs(b)) <= tol


def run_external(script: Path, path: Path, time_limit: float) -> dict:
    py = str(VENV_PY if VENV_PY.exists() else sys.executable)
    try:
        p = subprocess.run(
            [py, str(script), str(path), "--time-limit", str(time_limit)],
            capture_output=True, text=True, timeout=time_limit + 300,
        )
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        return json.loads(line) if line else {"status": "error", "error": p.stderr[:200]}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


def gen_one(gen: Path, outdir: Path, seed: int, tier: str, kind: str,
            args: list[str]) -> Path:
    ext = ".qps" if kind.endswith("qp") else ".mps"
    path = outdir / f"{kind}_{tier.lower()}_s{seed}{ext}"
    cmd = [str(gen), *args, "--seed", str(seed), "-o", str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"sor_gen failed: {p.stderr or p.stdout}")
    return path


def fmt_s(x) -> str:
    if x is None:
        return "—"
    if x < 1e-3:
        return f"{x*1e6:.0f}µs"
    if x < 1:
        return f"{x*1e3:.2f}ms"
    return f"{x:.3f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--sizes", default="S,M,L,XL",
                    help="comma list of tiers to run")
    ap.add_argument("--kinds", default="blend_lp,schedule_milp,dispatch_qp")
    ap.add_argument("--baselines-only", default="",
                    help="comma list; keep only matching baselines "
                         "(prefix ok: 'highs' keeps highs + highs-qp)")
    ap.add_argument("--outdir-instances",
                    default=str(ROOT / "benchmarks" / "industrial-ladder"))
    ap.add_argument("-o", "--outdir", default=str(ROOT / "benchmarks" / "results"))
    args = ap.parse_args()

    wanted_sizes = {s.strip().upper() for s in args.sizes.split(",") if s.strip()}
    wanted_kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    allow = {x.strip() for x in args.baselines_only.split(",") if x.strip()}

    gen = ROOT / "build" / "sor_gen"
    exe = ROOT / "build" / "sor_solve"
    if not gen.exists() or not exe.exists():
        print("error: build sor_gen and sor_solve first", file=sys.stderr)
        return 2
    if not VENV_PY.exists():
        print(f"error: baseline venv missing: {VENV_PY}", file=sys.stderr)
        return 2

    scripts = {
        "highs": ROOT / "scripts" / "run_highs_baseline.py",
        "highs-qp": ROOT / "scripts" / "run_highs_qp_baseline.py",
        "cbc": ROOT / "scripts" / "run_cbc_baseline.py",
    }

    inst_dir = Path(args.outdir_instances)
    inst_dir.mkdir(parents=True, exist_ok=True)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl = outdir / f"industrial-perf-{stamp}.jsonl"
    md = outdir / f"industrial-perf-{stamp}.md"

    jobs = []
    for tier, kind, engine, gen_args, baselines in LADDER:
        if tier not in wanted_sizes or kind not in wanted_kinds:
            continue
        if allow:
            baselines = [b for b in baselines
                         if b in allow or any(b.startswith(a) for a in allow)]
        jobs.append((tier, kind, engine, gen_args, baselines))

    env = {
        "record": "env",
        "host": platform.node(),
        "platform": platform.platform(),
        "time_limit_s": args.time_limit,
        "max_nodes": args.max_nodes,
        "seed": args.seed,
        "n_jobs": len(jobs),
    }
    with open(jsonl, "w") as f:
        f.write(json.dumps(env) + "\n")

    rows = []
    print(f"Industrial size ladder · {len(jobs)} jobs · "
          f"limit {args.time_limit}s\n")

    for tier, kind, engine, gen_args, baselines in jobs:
        label = f"{kind}/{tier}"
        print(f"== {label}  gen={' '.join(gen_args)} ==")
        try:
            path = gen_one(gen, inst_dir, args.seed, tier, kind, gen_args)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL generate: {e}")
            continue

        max_iter = args.max_nodes if engine == "milp" else 500000
        cmd = [str(exe), str(path), "--engine", engine,
               "--time-limit", str(args.time_limit),
               "--max-iter", str(max_iter)]
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.time_limit + 300)
            wall = time.perf_counter() - t0
            sor = parse_sor(p.stdout)
            sor["wall_s"] = wall
            sor["rc"] = p.returncode
            if "status" not in sor:
                sor["status"] = "unparsed"
                sor["error"] = (p.stderr or p.stdout)[:300]
        except subprocess.TimeoutExpired:
            sor = {"status": "Timeout", "wall_s": args.time_limit}

        bases = {}
        agrees = {}
        speedups = {}
        for bn in baselines:
            br = run_external(scripts[bn], path, args.time_limit)
            bases[bn] = br
            agrees[bn] = obj_agree(sor.get("objective"), br.get("objective"))
            # Prefer solver-reported solve_s; fall back to unavailable.
            b_s = br.get("solve_s")
            s_s = sor.get("wall_s")
            if (b_s is not None and s_s is not None and s_s > 0
                    and math.isfinite(b_s) and math.isfinite(s_s)):
                speedups[bn] = b_s / s_s   # >1 ⇒ SOR faster than baseline solve
            else:
                speedups[bn] = None

        row = {
            "record": "instance",
            "tier": tier,
            "kind": kind,
            "engine": engine,
            "path": str(path),
            "gen_args": gen_args,
            "sor": sor,
            "baselines": bases,
            "obj_agree": agrees,
            "speedup_vs_baseline_solve": speedups,
        }
        rows.append(row)
        with open(jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")

        dim = f"{sor.get('rows')}x{sor.get('cols')} nnz={sor.get('nnz')}"
        print(f"  size {dim}")
        print(f"  SOR  {sor.get('status'):12s}  obj={sor.get('objective')}  "
              f"wall={fmt_s(sor.get('wall_s'))}  "
              f"solver_ms={sor.get('total_ms')}")
        for bn, br in bases.items():
            print(f"  {bn:8s} {str(br.get('status')):12s}  "
                  f"obj={br.get('objective')}  "
                  f"solve={fmt_s(br.get('solve_s'))}  "
                  f"agree={agrees.get(bn)}  "
                  f"speedup(SOR vs {bn} solve)={speedups.get(bn)}")
        print()

    # Markdown report with performance table
    with open(md, "w") as f:
        f.write("# Industrial size-ladder performance vs open source\n\n")
        f.write(f"- Host: `{platform.node()}` · `{platform.platform()}`\n")
        f.write(f"- Time limit: {args.time_limit}s · MILP max nodes: "
                f"{args.max_nodes} · seed {args.seed}\n")
        f.write("- Baselines: HiGHS (highspy, external), CBC (PuLP), "
                "HiGHS-QP for dispatch\n")
        f.write("- **speedup** = baseline `solve_s` / SOR wall_s "
                "(>1 means SOR wall faster than baseline solve)\n\n")

        f.write("## Summary table\n\n")
        f.write("| Tier | Kind | rows×cols | nnz | SOR status | SOR wall | "
                "HiGHS solve | speedup | agree | CBC solve | speedup | agree |\n")
        f.write("|------|------|-----------|----:|------------|---------:|"
                "------------|--------:|-------|-----------|--------:|-------|\n")
        for r in rows:
            s = r["sor"]
            bh = r["baselines"].get("highs") or r["baselines"].get("highs-qp") or {}
            bc = r["baselines"].get("cbc") or {}
            ah = r["obj_agree"].get("highs")
            if ah is None:
                ah = r["obj_agree"].get("highs-qp")
            sh = r["speedup_vs_baseline_solve"].get("highs")
            if sh is None:
                sh = r["speedup_vs_baseline_solve"].get("highs-qp")
            ac = r["obj_agree"].get("cbc")
            sc = r["speedup_vs_baseline_solve"].get("cbc")
            dim = f"{s.get('rows')}×{s.get('cols')}"
            f.write(
                f"| {r['tier']} | {r['kind']} | {dim} | {s.get('nnz')} | "
                f"{s.get('status')} | {fmt_s(s.get('wall_s'))} | "
                f"{fmt_s(bh.get('solve_s'))} | "
                f"{'—' if sh is None else f'{sh:.2f}×'} | {ah} | "
                f"{fmt_s(bc.get('solve_s'))} | "
                f"{'—' if sc is None else f'{sc:.2f}×'} | {ac} |\n"
            )

        f.write("\n## Objectives\n\n")
        f.write("| Tier | Kind | SOR obj | HiGHS obj | CBC obj |\n")
        f.write("|------|------|---------|-----------|----------|\n")
        for r in rows:
            s = r["sor"]
            bh = r["baselines"].get("highs") or r["baselines"].get("highs-qp") or {}
            bc = r["baselines"].get("cbc") or {}
            f.write(
                f"| {r['tier']} | {r['kind']} | {s.get('objective')} | "
                f"{bh.get('objective')} | {bc.get('objective')} |\n"
            )

    print(f"wrote {jsonl}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
