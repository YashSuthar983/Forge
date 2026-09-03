#!/usr/bin/env python3
"""Full open-source compare for newly implemented SOR features.

Suites:
  industrial — blend LP, schedule MILP, dispatch QP vs HiGHS (+ CBC on MILP)
  miplib     — miplib-easy MILP vs HiGHS + CBC
  all        — both

Baselines are EXTERNAL PROCESSES only (never linked into libsor).

Usage:
  python3 scripts/run_new_features_compare.py --suite all --time-limit 30
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
        elif line.startswith("dual bound:"):
            try:
                out["dual_bound"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("stationarity:"):
            try:
                out["stationarity"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.strip().startswith("total") and "ms" not in line.split()[0]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out["total_ms"] = float(parts[1])
                except ValueError:
                    pass
    return out


def run_external(script: Path, path: Path, time_limit: float,
                 extra: list[str] | None = None) -> dict:
    py = str(VENV_PY if VENV_PY.exists() else sys.executable)
    cmd = [py, str(script), str(path), "--time-limit", str(time_limit),
           *(extra or [])]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=time_limit + 90)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        if not line:
            return {"status": "error", "error": (p.stderr or "empty")[:300]}
        return json.loads(line)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


def run_sor(exe: Path, path: Path, engine: str, time_limit: float,
            max_iter: int) -> dict:
    cmd = [str(exe), str(path), "--engine", engine,
           "--time-limit", str(time_limit), "--max-iter", str(max_iter)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=time_limit + 45)
        wall = time.perf_counter() - t0
        sor = parse_sor(p.stdout)
        sor["wall_s"] = wall
        sor["rc"] = p.returncode
        if "status" not in sor:
            sor["error"] = (p.stderr or p.stdout)[:400]
            sor.setdefault("status", "unparsed")
        return sor
    except subprocess.TimeoutExpired:
        return {"status": "Timeout", "wall_s": time_limit, "error": "outer timeout"}


def obj_agree(a: float | None, b: float | None, tol: float = 1e-3) -> bool | None:
    if a is None or b is None:
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return abs(a - b) / (1.0 + abs(b)) <= tol


def is_ref_solved(st: str | None) -> bool:
    if not st:
        return False
    s = st.lower()
    return s in ("optimal", "optimalinfeas", "feasible") or s.endswith("optimal")


def industrial_jobs(examples: Path) -> list[dict]:
    return [
        {
            "name": "blend_lp",
            "path": examples / "crude_blending" / "blend_s42.mps",
            "engine": "simplex",
            "kind": "lp",
            "baselines": ["highs"],
        },
        {
            "name": "schedule_milp",
            "path": examples / "scheduling" / "schedule_s42.mps",
            "engine": "milp",
            "kind": "milp",
            "baselines": ["highs", "cbc"],
        },
        {
            "name": "dispatch_qp",
            "path": examples / "dispatch" / "dispatch_s42.qps",
            "engine": "qp",
            "kind": "qp",
            "baselines": ["highs-qp"],
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["industrial", "miplib", "all"],
                    default="all")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--max-nodes", type=int, default=10000)
    ap.add_argument("--limit", type=int, default=0,
                    help="limit MIPLIB instances (0 = all)")
    ap.add_argument("--baselines-only", default="",
                    help="comma list; keep only matching baselines "
                         "(prefix ok: 'highs' keeps highs + highs-qp)")
    ap.add_argument("--examples", default=str(ROOT / "examples"))
    ap.add_argument("--miplib-dir",
                    default=str(ROOT / "benchmarks" / "miplib-easy" / "mps"))
    ap.add_argument("-o", "--outdir", default=str(ROOT / "benchmarks" / "results"))
    args = ap.parse_args()
    allow = {x.strip() for x in args.baselines_only.split(",") if x.strip()}

    exe = ROOT / "build" / "sor_solve"
    if not exe.exists():
        print(f"error: build {exe} first", file=sys.stderr)
        return 2
    if not VENV_PY.exists():
        print(f"error: baseline venv missing: {VENV_PY}", file=sys.stderr)
        print("  python3 -m venv benchmarks/.venv-baseline && "
              "benchmarks/.venv-baseline/bin/pip install highspy pulp scipy",
              file=sys.stderr)
        return 2

    scripts = {
        "highs": ROOT / "scripts" / "run_highs_baseline.py",
        "highs-qp": ROOT / "scripts" / "run_highs_qp_baseline.py",
        "cbc": ROOT / "scripts" / "run_cbc_baseline.py",
    }

    def filter_bases(bases: list[str]) -> list[str]:
        if not allow:
            return bases
        return [b for b in bases
                if b in allow or any(b.startswith(a) for a in allow)]

    jobs: list[dict] = []
    if args.suite in ("industrial", "all"):
        for job in industrial_jobs(Path(args.examples)):
            job["baselines"] = filter_bases(job["baselines"])
            jobs.append(job)
    if args.suite in ("miplib", "all"):
        mdir = Path(args.miplib_dir)
        if not mdir.is_dir():
            print(f"error: {mdir} missing — run scripts/fetch_miplib_easy.py",
                  file=sys.stderr)
            return 2
        mps_list = sorted(mdir.glob("*.mps"))
        if args.limit:
            mps_list = mps_list[: args.limit]
        for mps in mps_list:
            jobs.append({
                "name": mps.stem,
                "path": mps,
                "engine": "milp",
                "kind": "milp",
                "baselines": filter_bases(["highs", "cbc"]),
            })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = args.suite
    jsonl = outdir / f"compare-new-{tag}-{stamp}.jsonl"
    md = outdir / f"compare-new-{tag}-{stamp}.md"

    env = {
        "record": "env",
        "suite": args.suite,
        "host": platform.node(),
        "platform": platform.platform(),
        "time_limit_s": args.time_limit,
        "max_nodes": args.max_nodes,
        "sor": str(exe),
        "baseline_python": str(VENV_PY),
        "n_jobs": len(jobs),
    }
    with open(jsonl, "w") as f:
        f.write(json.dumps(env) + "\n")

    rows: list[dict] = []
    for job in jobs:
        path = Path(job["path"])
        print(f"== {job['name']} ({job['kind']} / {job['engine']}) ==")
        if not path.exists():
            print(f"  SKIP missing {path}")
            continue
        max_iter = args.max_nodes if job["engine"] == "milp" else 200000
        sor = run_sor(exe, path, job["engine"], args.time_limit, max_iter)
        bases: dict[str, dict] = {}
        agrees: dict[str, bool | None] = {}
        for bname in job["baselines"]:
            bases[bname] = run_external(scripts[bname], path, args.time_limit)
            bst = bases[bname].get("status")
            agrees[bname] = (
                obj_agree(sor.get("objective"), bases[bname].get("objective"))
                if is_ref_solved(bst) or (bst or "").lower() == "optimal"
                else None
            )
            # HiGHS uses Optimal; CBC uses Optimal; normalize
            if (bst or "").lower() in ("optimal",) or (bst or "").endswith("Optimal"):
                agrees[bname] = obj_agree(
                    sor.get("objective"), bases[bname].get("objective"))

        row = {
            "record": "instance",
            "name": job["name"],
            "kind": job["kind"],
            "path": str(path),
            "sor": sor,
            "baselines": bases,
            "obj_agree": agrees,
        }
        rows.append(row)
        with open(jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")

        bits = [f"SOR {sor.get('status')} obj={sor.get('objective')} "
                f"wall={sor.get('wall_s')}"]
        for bn, br in bases.items():
            bits.append(f"{bn} {br.get('status')} obj={br.get('objective')} "
                        f"agree={agrees.get(bn)}")
        print("  " + " | ".join(bits))

    # Summary markdown
    n = len(rows)
    sor_opt = sum(1 for r in rows if r["sor"].get("status") == "Optimal")
    sor_feasible = sum(1 for r in rows
                       if r["sor"].get("status") == "Feasible")
    sor_ok = sor_opt + sor_feasible
    agree_highs = sum(1 for r in rows if r["obj_agree"].get("highs") is True
                      or r["obj_agree"].get("highs-qp") is True)
    agree_cbc = sum(1 for r in rows if r["obj_agree"].get("cbc") is True)

    with open(md, "w") as f:
        f.write(f"# New-features compare vs open source (`{args.suite}`)\n\n")
        f.write(f"- Host: `{platform.node()}` · `{platform.platform()}`\n")
        f.write(f"- Time limit: {args.time_limit}s · MILP max nodes: "
                f"{args.max_nodes}\n")
        f.write(f"- Baselines: HiGHS (external highspy), CBC (PuLP), "
                f"HiGHS-QP for dispatch\n")
        f.write(f"- SOR incumbent coverage (Optimal + Feasible): **{sor_ok}/{n}**\n")
        f.write(f"- SOR status split: **{sor_opt} Optimal** · "
                f"**{sor_feasible} Feasible (not proved optimal)**\n")
        f.write(f"- Exact objective agreement (relative tolerance 1e-3): "
                f"HiGHS **{agree_highs}** · CBC **{agree_cbc}**\n")
        f.write("- `Feasible` means a valid incumbent was found; it does not "
                "claim global optimality. `None` means the reference was "
                "time-limited or unavailable.\n\n")
        f.write("| Instance | kind | SOR | SOR obj | wall_s | HiGHS | "
                "HiGHS obj | agree | CBC | CBC obj | agree |\n")
        f.write("|----------|------|-----|---------|-------:|-------|"
                "----------|-------|-----|---------|-------|\n")
        for r in rows:
            s = r["sor"]
            bh = r["baselines"].get("highs") or r["baselines"].get("highs-qp") or {}
            bc = r["baselines"].get("cbc") or {}
            ah = r["obj_agree"].get("highs")
            if ah is None:
                ah = r["obj_agree"].get("highs-qp")
            ac = r["obj_agree"].get("cbc")
            f.write(
                f"| {r['name']} | {r['kind']} | {s.get('status')} | "
                f"{s.get('objective')} | {s.get('wall_s')} | "
                f"{bh.get('status')} | {bh.get('objective')} | {ah} | "
                f"{bc.get('status')} | {bc.get('objective')} | {ac} |\n"
            )

    summary = {
        "record": "summary",
        "sor_ok": sor_ok,
        "n": n,
        "agree_highs": agree_highs,
        "agree_cbc": agree_cbc,
        "jsonl": str(jsonl),
        "md": str(md),
    }
    with open(jsonl, "a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"\nwrote {jsonl}")
    print(f"wrote {md}")
    print(f"summary: SOR {sor_ok}/{n} · HiGHS agree {agree_highs} · "
          f"CBC agree {agree_cbc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
