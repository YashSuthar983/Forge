#!/usr/bin/env python3
"""Head-to-head benchmark: SOR vs shreyas-omkar/solver_accl (SovereignSolver).

Runs on every Netlib instance that solver_accl can load (x>=0, no finite
upper bounds, no two-sided row ranges). Uses HiGHS objective as reference.
"""
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_compare import (  # noqa: E402
    SOR_KEYS,
    SolverResult,
    display_name_for,
    fmt_obj,
    fmt_time,
    is_solved,
    rel_gap,
    run_external,
    run_sor,
    shifted_geomean,
)

JULIA = os.environ.get("JULIA", str(Path.home() / ".juliaup/bin/julia"))
SOLVER_ACCL_ROOT = os.environ.get("SOLVER_ACCL_ROOT", "/tmp/solver_accl")
BASELINE_JL = ROOT / "scripts" / "run_solver_accl_baseline.jl"


def run_solver_accl(mps: Path, time_limit: float, engine: str,
                    max_iter: int = 200_000, tol: float = 1e-6) -> SolverResult:
    name = f"accl-{engine}"
    r = SolverResult(name=name, kind="external_process")
    cmd = [
        JULIA, f"--project={SOLVER_ACCL_ROOT}", str(BASELINE_JL), str(mps),
        "--engine", engine, "--time-limit", str(time_limit),
        "--max-iter", str(max_iter), "--tol", str(tol),
    ]
    env = {**os.environ, "SOLVER_ACCL_ROOT": SOLVER_ACCL_ROOT}
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=time_limit + 120, env=env)
        r.wall_s = time.perf_counter() - t0
        lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip().startswith("{")]
        if not lines:
            r.status = "error"
            r.error = (p.stderr or p.stdout)[:240]
            return r
        j = json.loads(lines[-1])
        r.status = str(j.get("status", "unknown"))
        r.objective = j.get("objective")
        if j.get("solve_s") is not None:
            r.wall_s = float(j["solve_s"])
        if j.get("iterations") is not None:
            r.iterations = int(j["iterations"])
        if j.get("error"):
            r.error = str(j["error"])
        if j.get("rows") is not None:
            r.rows = int(j["rows"])
            r.cols = int(j.get("cols", 0))
    except subprocess.TimeoutExpired:
        r.wall_s = time.perf_counter() - t0
        r.status = "timeout"
    except Exception as e:  # noqa: BLE001
        r.wall_s = time.perf_counter() - t0
        r.status = "error"
        r.error = f"{type(e).__name__}: {e}"
    return r


def probe_loadable_instances(inst_dir: Path) -> tuple[list[Path], list[tuple[str, str]]]:
    probe = ROOT / "scripts" / "probe_solver_accl_loadable.jl"
    env = {**os.environ, "SOLVER_ACCL_ROOT": SOLVER_ACCL_ROOT}
    p = subprocess.run(
        [JULIA, f"--project={SOLVER_ACCL_ROOT}", str(probe), str(inst_dir)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    loadable, skipped = [], []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        j = json.loads(line)
        if j.get("loadable"):
            loadable.append(inst_dir / f"{j['instance']}.mps")
        else:
            skipped.append((j["instance"], j.get("reason", "")))
    return loadable, skipped


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="netlib")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--max-iter", type=int, default=200_000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--obj-tol", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-o", "--outdir", default="benchmarks/results")
    args = ap.parse_args()

    inst_dir = ROOT / "benchmarks" / args.suite / "mps"
    exe = ROOT / "build" / "sor_solve"
    venv_py = ROOT / "benchmarks" / ".venv-baseline" / "bin" / "python"
    highs_script = ROOT / "scripts" / "run_highs_baseline.py"

    if not exe.exists():
        print(f"error: {exe} not built", file=sys.stderr)
        return 2
    if not Path(JULIA).exists():
        print(f"error: julia not found at {JULIA}", file=sys.stderr)
        return 2

    all_instances = sorted(inst_dir.glob("*.mps"))
    if args.limit:
        all_instances = all_instances[: args.limit]

    print("Probing which instances solver_accl can load...", flush=True)
    loadable_all, skipped = probe_loadable_instances(inst_dir)
    loadable_set = {p.stem for p in loadable_all}
    instances = [m for m in all_instances if m.stem in loadable_set]
    print(f"  loadable: {len(instances)} / {len(all_instances)}", flush=True)

    solver_names = [
        "SOR-simplex", "SOR-pdhg", "SOR-hpr",
        "accl-simplex", "accl-pdhg", "highs",
    ]

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl = outdir / f"compare-solver_accl-{stamp}.jsonl"
    md = outdir / f"compare-solver_accl-{stamp}.md"

    env = {
        "suite": args.suite,
        "competitor": "https://github.com/shreyas-omkar/solver_accl",
        "n_all_instances": len(all_instances),
        "n_loadable": len(instances),
        "n_skipped": len(skipped),
        "time_limit_s": args.time_limit,
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "solvers": solver_names,
    }
    with jsonl.open("w") as f:
        f.write(json.dumps({"record": "environment", **env}) + "\n")
        f.write(json.dumps({"record": "skipped", "instances": skipped}) + "\n")

    rows = []
    for i, mps in enumerate(instances, 1):
        print(f"── [{i}/{len(instances)}] {mps.stem} ──", flush=True)
        results: dict[str, dict] = {}
        row_meta = {"rows": None, "cols": None, "nnz": None}

        for key in ("sor-simplex", "sor-pdhg", "sor-hpr"):
            display, eng, b_ov, extra = SOR_KEYS[key]
            res = run_sor(exe, mps, args.time_limit, args.max_iter,
                          args.tol, b_ov or args.backend, eng, extra, name=display)
            results[res.name] = asdict(res)
            if row_meta["rows"] is None and res.rows:
                row_meta.update(rows=res.rows, cols=res.cols, nnz=res.nnz)
            print(f"    {res.name:16s}  {res.status:16s}  {fmt_obj(res.objective):>12s}  "
                  f"{fmt_time(res.wall_s)}", flush=True)

        for eng in ("simplex", "pdhg"):
            res = run_solver_accl(mps, args.time_limit, eng,
                                  args.max_iter, args.tol)
            results[res.name] = asdict(res)
            print(f"    {res.name:16s}  {res.status:16s}  {fmt_obj(res.objective):>12s}  "
                  f"{fmt_time(res.wall_s)}", flush=True)

        hres = run_external(venv_py, highs_script, mps, args.time_limit, "highs")
        results["highs"] = asdict(hres)
        print(f"    {'highs':16s}  {hres.status:16s}  {fmt_obj(hres.objective):>12s}  "
              f"{fmt_time(hres.wall_s)}", flush=True)

        ref = hres.objective if is_solved(hres.status) else None
        agreements = {}
        for name, rd in results.items():
            if ref is None or rd.get("objective") is None:
                agreements[name] = None
            else:
                g = rel_gap(rd["objective"], ref)
                agreements[name] = g is not None and g <= args.obj_tol

        row = {
            "record": "instance",
            "instance": mps.stem,
            **row_meta,
            "ref_objective": ref,
            "results": results,
            "agreements": agreements,
        }
        rows.append(row)
        with jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")

    solved = {n: 0 for n in solver_names}
    match = {n: 0 for n in solver_names}
    times = {n: [] for n in solver_names}
    for row in rows:
        for n in solver_names:
            rd = row["results"].get(n, {})
            if is_solved(rd.get("status", "")):
                solved[n] += 1
                if rd.get("wall_s") is not None:
                    times[n].append(rd["wall_s"])
            if row["agreements"].get(n) is True:
                match[n] += 1

    sgm = {n: shifted_geomean(times[n]) for n in solver_names}
    summary = {
        "record": "summary",
        "n_instances": len(rows),
        "solved": solved,
        "obj_match_vs_highs": match,
        "sgm_wall_s": sgm,
    }
    with jsonl.open("a") as f:
        f.write(json.dumps(summary) + "\n")

    lines = [
        "# SOR vs solver_accl (SovereignSolver)",
        "",
        f"- Competitor: [shreyas-omkar/solver_accl](https://github.com/shreyas-omkar/solver_accl)",
        f"- Instances: **{len(rows)}** loadable / {len(all_instances)} Netlib "
        f"({len(skipped)} skipped — bounds/format unsupported by accl)",
        f"- Time limit: {args.time_limit}s per solver",
        f"- Host: `{env['host']}` · {env['cpu_count']} CPUs",
        "",
        "## Summary (vs HiGHS reference)",
        "",
        "| Solver | Solved | Obj match | SGM time (s) |",
        "|--------|-------:|----------:|-------------:|",
    ]
    for n in solver_names:
        s = f"{sgm[n]:.4f}" if sgm[n] is not None else "—"
        lines.append(f"| {n} | {solved[n]}/{len(rows)} | {match[n]}/{len(rows)} | {s} |")

    lines += ["", "## Skipped by solver_accl", ""]
    for name, reason in skipped[:20]:
        lines.append(f"- {name}: {reason}")
    if len(skipped) > 20:
        lines.append(f"- … and {len(skipped)-20} more")

    lines += ["", "## Per-instance", ""]
    hdr = "| Instance |"
    sep = "|----------|"
    for n in solver_names:
        hdr += f" {n} status | {n} obj | {n} time |"
        sep += "--------|--------:|---------:|"
    lines += [hdr, sep]
    for row in rows:
        size = f"{row['rows']}×{row['cols']}" if row.get("rows") else ""
        cells = f"| {row['instance']} | {size} |"
        for n in solver_names:
            rd = row["results"].get(n, {})
            cells += (f" {rd.get('status','—')} | {fmt_obj(rd.get('objective'))} | "
                      f"{fmt_time(rd.get('wall_s'))} |")
        lines.append(cells)

    lines += ["", f"JSONL: `{jsonl}`"]
    md.write_text("\n".join(lines) + "\n")

    print("\n" + "=" * 80)
    print(f"{'Solver':18s} {'Solved':>10s} {'Match':>10s} {'SGM':>10s}")
    for n in solver_names:
        s = f"{sgm[n]:.4f}s" if sgm[n] else "—"
        print(f"{n:18s} {solved[n]:4d}/{len(rows):<4d} {match[n]:4d}/{len(rows):<4d} {s:>10s}")
    print("=" * 80)
    print(f"Markdown: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
