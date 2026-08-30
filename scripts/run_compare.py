#!/usr/bin/env python3
"""Sequential side-by-side benchmark: SOR vs external competitors.

For each instance, solvers run ONE AFTER ANOTHER on the same machine
(same CPU, no parallel solver races). Baselines are external processes only —
never linked into libsor, never written into certificates.

Usage:
  python3 scripts/run_compare.py --suite netlib --time-limit 20
  python3 scripts/run_compare.py --suite netlib --limit 10   # smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_PATTERNS = {
    "rows_cols": re.compile(r"rows x cols:\s+(\d+) x (\d+)\s+nnz (\d+)"),
    "status": re.compile(r"^status:\s+(\S+)", re.M),
    "proof": re.compile(r"^proof_level:\s+(\S+)", re.M),
    "objective": re.compile(r"^objective:\s+(\S+)", re.M),
    "dual_bound": re.compile(r"^dual bound:\s+(\S+)", re.M),
    # The simplex prints "max primal viol" because it also covers bound
    # violations; PDHG prints "max row violation". Accept both, or the harness
    # silently records None for whichever engine it was not written for.
    "row_viol": re.compile(r"^max (?:primal viol|row violation):\s+(\S+)", re.M),
    "iters": re.compile(r"^iterations:\s+(\d+)", re.M),
}


def _f(text: str, key: str):
    m = _PATTERNS[key].search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


@dataclass
class SolverResult:
    name: str
    status: str = "not_run"
    objective: float | None = None
    wall_s: float | None = None
    iterations: int | None = None
    proof: str | None = None
    row_violation: float | None = None
    error: str | None = None
    kind: str = "external_process"


@dataclass
class InstanceRow:
    instance: str
    rows: int | None = None
    cols: int | None = None
    nnz: int | None = None
    results: dict = field(default_factory=dict)  # name -> SolverResult as dict
    ref_objective: float | None = None
    agreements: dict = field(default_factory=dict)  # name -> bool|None


def shifted_geomean(values: list[float], shift: float = 1.0) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    acc = sum(math.log(max(v, 0.0) + shift) for v in vals)
    return math.exp(acc / len(vals)) - shift


def run_sor(exe: Path, mps: Path, time_limit: float, max_iter: int,
            tol: float, backend: str, engine: str = "simplex") -> SolverResult:
    r = SolverResult(name="SOR", kind="ours")
    # --time-limit is passed so the solver stops itself and reports its best
    # point. Relying only on the subprocess timeout below throws the answer away.
    cmd = [str(exe), str(mps), "--engine", engine, "--backend", backend,
           "--max-iter", str(max_iter), "--tol", str(tol),
           "--time-limit", str(time_limit)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit)
        r.wall_s = time.perf_counter() - t0
        txt = p.stdout
        sm = _PATTERNS["status"].search(txt)
        r.status = sm.group(1) if sm else "unparsed"
        pm = _PATTERNS["proof"].search(txt)
        r.proof = pm.group(1) if pm else None
        r.objective = _f(txt, "objective")
        r.row_violation = _f(txt, "row_viol")
        it = _f(txt, "iters")
        r.iterations = int(it) if it is not None else None
        if r.status == "unparsed":
            r.error = ((p.stderr or "") + (p.stdout or ""))[:240]
    except subprocess.TimeoutExpired:
        r.wall_s = time.perf_counter() - t0
        r.status = "timeout"
        r.error = f"wall timeout > {time_limit}s"
    except Exception as e:  # noqa: BLE001
        r.wall_s = time.perf_counter() - t0
        r.status = "error"
        r.error = f"{type(e).__name__}: {e}"
    return r


def run_external(python: Path, script: Path, mps: Path,
                 time_limit: float, name: str,
                 extra: list[str] | None = None) -> SolverResult:
    r = SolverResult(name=name, kind="external_process")
    cmd = [str(python), str(script), str(mps), "--time-limit", str(time_limit)]
    if extra:
        cmd.extend(extra)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=time_limit + 90)
        wall = time.perf_counter() - t0
        try:
            j = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            r.status = "error"
            r.wall_s = wall
            r.error = f"unparseable: {p.stdout[:160]!r} {p.stderr[:160]!r}"
            return r
        r.status = str(j.get("status", "unknown"))
        r.objective = j.get("objective")
        r.wall_s = float(j.get("solve_s", wall))
        if j.get("iterations") is not None:
            r.iterations = int(j["iterations"])
        if j.get("error"):
            r.error = str(j["error"])
    except subprocess.TimeoutExpired:
        r.wall_s = time.perf_counter() - t0
        r.status = "timeout"
        r.error = f"wall timeout > {time_limit + 90}s"
    except Exception as e:  # noqa: BLE001
        r.wall_s = time.perf_counter() - t0
        r.status = "error"
        r.error = f"{type(e).__name__}: {e}"
    return r


def probe_size(exe: Path, mps: Path) -> tuple[int | None, int | None, int | None]:
    try:
        p = subprocess.run([str(exe), str(mps), "--max-iter", "0"],
                           capture_output=True, text=True, timeout=30)
        m = _PATTERNS["rows_cols"].search(p.stdout)
        if m:
            return int(m[1]), int(m[2]), int(m[3])
    except Exception:
        pass
    return None, None, None


def is_solved(status: str) -> bool:
    s = status.lower().replace("highsmodelstatus.k", "")
    return s in {"optimal", "feasible"} or s.endswith("optimal")


def is_usable_primal(status: str) -> bool:
    """Statuses where a primal objective is still comparable."""
    s = status.lower()
    return is_solved(s) or s in {"interrupted", "feasiblewithgap", "iterationlimit"}


def rel_gap(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b) / (1.0 + abs(b))


def fmt_obj(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.6g}"


def fmt_time(x: float | None) -> str:
    if x is None:
        return "—"
    if x < 0.001:
        return f"{x*1e6:.0f}µs"
    if x < 1:
        return f"{x*1e3:.1f}ms"
    return f"{x:.3f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="netlib")
    ap.add_argument("--instances-dir", default=None)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--max-iter", type=int, default=200000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--engine", default="simplex",
                    help="SOR engine: simplex (proves optimality) or pdhg")
    ap.add_argument("--obj-tol", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sgm-shift", type=float, default=1.0)
    ap.add_argument("--solvers", default="sor,highs,cbc,scipy-ipm",
                    help="comma list: sor,highs,cbc,scipy-ipm,scipy-simplex")
    ap.add_argument("-o", "--outdir", default="benchmarks/results")
    args = ap.parse_args()

    inst_dir = Path(args.instances_dir or ROOT / "benchmarks" / args.suite / "mps")
    if not inst_dir.is_dir():
        print(f"error: instance dir not found: {inst_dir}", file=sys.stderr)
        return 2
    instances = sorted(inst_dir.glob("*.mps"))
    if args.limit:
        instances = instances[: args.limit]
    if not instances:
        print(f"error: no .mps in {inst_dir}", file=sys.stderr)
        return 2

    exe = ROOT / "build" / "sor_solve"
    if not exe.exists():
        print(f"error: {exe} not built", file=sys.stderr)
        return 2

    venv_py = ROOT / "benchmarks" / ".venv-baseline" / "bin" / "python"
    if not venv_py.exists():
        print(f"error: baseline venv missing: {venv_py}", file=sys.stderr)
        return 2

    wanted = [s.strip().lower() for s in args.solvers.split(",") if s.strip()]
    scripts = {
        "highs": ROOT / "scripts" / "run_highs_baseline.py",
        "cbc": ROOT / "scripts" / "run_cbc_baseline.py",
        "scipy-ipm": (ROOT / "scripts" / "run_scipy_baseline.py",
                      ["--method", "interior-point"]),
        "scipy-simplex": (ROOT / "scripts" / "run_scipy_baseline.py",
                          ["--method", "simplex"]),
    }

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    env = {
        "suite": args.suite,
        "n_instances": len(instances),
        "time_limit_s": args.time_limit,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "backend": args.backend,
        "obj_tol": args.obj_tol,
        "solvers": wanted,
        "mode": "sequential_per_instance",
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "note": "Solvers run one-by-one per instance on the same hardware.",
    }
    print(json.dumps(env, indent=2), flush=True)
    print(f"\n{len(instances)} instances × {len(wanted)} solvers "
          f"(sequential, {args.time_limit}s limit each)\n", flush=True)

    rows: list[InstanceRow] = []
    solver_names = [s.upper() if s == "sor" else s for s in wanted]

    for i, mps in enumerate(instances, 1):
        row = InstanceRow(instance=mps.stem)
        print(f"── [{i}/{len(instances)}] {mps.stem} ──", flush=True)

        # Sequential: same order every time, one process at a time.
        for s in wanted:
            if s == "sor":
                res = run_sor(exe, mps, args.time_limit, args.max_iter,
                              args.tol, args.backend, args.engine)
                if row.rows is None:
                    m = None
                    # size already in SOR output path; re-probe cheaply if needed
                    row.rows, row.cols, row.nnz = probe_size(exe, mps)
            elif s in scripts:
                spec = scripts[s]
                if isinstance(spec, tuple):
                    script, extra = spec
                else:
                    script, extra = spec, None
                display = s
                res = run_external(venv_py, script, mps, args.time_limit,
                                   display, extra)
            else:
                res = SolverResult(name=s, status="unknown_solver",
                                   error=f"unknown solver key {s}")

            row.results[res.name] = asdict(res)
            tag = res.name
            print(f"    {tag:14s}  status={res.status:16s}  "
                  f"obj={fmt_obj(res.objective):>14s}  "
                  f"time={fmt_time(res.wall_s)}", flush=True)

        # Reference objective: prefer HiGHS, then CBC, then scipy
        ref = None
        for key in ("highs", "cbc", "scipy-ipm", "scipy-simplex"):
            rd = row.results.get(key)
            if rd and is_solved(rd["status"]) and rd.get("objective") is not None:
                ref = rd["objective"]
                break
        row.ref_objective = ref

        for name, rd in row.results.items():
            if ref is None or rd.get("objective") is None:
                row.agreements[name] = None
            elif not is_solved(rd["status"]) and name != "SOR":
                row.agreements[name] = None
            else:
                g = rel_gap(rd["objective"], ref)
                row.agreements[name] = (g is not None and g <= args.obj_tol)

        rows.append(row)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl = outdir / f"compare-{args.suite}-{stamp}.jsonl"
    md = outdir / f"compare-{args.suite}-{stamp}.md"

    with jsonl.open("w") as f:
        f.write(json.dumps({"record": "environment", **env}) + "\n")
        for row in rows:
            f.write(json.dumps({"record": "instance", **asdict(row)}) + "\n")

    # ---- aggregates ----
    names = []
    for s in wanted:
        names.append("SOR" if s == "sor" else s)

    solved_counts = {n: 0 for n in names}
    match_counts = {n: 0 for n in names}
    times = {n: [] for n in names}

    for row in rows:
        for n in names:
            rd = row.results.get(n)
            if not rd:
                continue
            if is_solved(rd["status"]):
                solved_counts[n] += 1
                if rd.get("wall_s") is not None:
                    times[n].append(rd["wall_s"])
            if row.agreements.get(n) is True:
                match_counts[n] += 1

    sgm = {n: shifted_geomean(times[n], args.sgm_shift) for n in names}

    summary = {
        "record": "summary",
        "n_instances": len(rows),
        "solved": solved_counts,
        "obj_match_vs_ref": match_counts,
        "sgm_wall_s": sgm,
    }
    with jsonl.open("a") as f:
        f.write(json.dumps(summary) + "\n")

    # ---- markdown side-by-side ----
    lines = []
    lines.append(f"# Side-by-side: {args.suite}")
    lines.append("")
    lines.append(f"- Mode: **sequential** (one solver at a time, same hardware)")
    lines.append(f"- Instances: {len(rows)}")
    lines.append(f"- Time limit: {args.time_limit}s / solver / instance")
    lines.append(f"- Obj agreement tol: {args.obj_tol} rel")
    lines.append(f"- Host: `{env['host']}` · `{env['platform']}` · {env['cpu_count']} CPUs")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Solver | Solved | Obj match (vs HiGHS/CBC ref) | SGM time (s) |")
    lines.append("|--------|-------:|-----------------------------:|-------------:|")
    for n in names:
        sgm_s = f"{sgm[n]:.4f}" if sgm[n] is not None else "—"
        lines.append(
            f"| {n} | {solved_counts[n]}/{len(rows)} | "
            f"{match_counts[n]}/{len(rows)} | {sgm_s} |"
        )
    lines.append("")
    lines.append("## Per-instance")
    lines.append("")

    # compact header: instance | for each solver: status / obj / time
    header = "| Instance | size |"
    sep = "|----------|-----:|"
    for n in names:
        header += f" {n} status | {n} obj | {n} time |"
        sep += "------------|--------:|---------:|"
    lines.append(header)
    lines.append(sep)

    for row in rows:
        size = ""
        if row.rows is not None:
            size = f"{row.rows}×{row.cols}"
        cells = f"| {row.instance} | {size} |"
        for n in names:
            rd = row.results.get(n, {})
            cells += (
                f" {rd.get('status', '—')} | "
                f"{fmt_obj(rd.get('objective'))} | "
                f"{fmt_time(rd.get('wall_s'))} |"
            )
        lines.append(cells)

    lines.append("")
    lines.append(f"JSONL: `{jsonl}`")
    md.write_text("\n".join(lines) + "\n")

    # ---- console summary table ----
    print("\n" + "=" * 78)
    print(f"{'Solver':14s} {'Solved':>10s} {'ObjMatch':>10s} {'SGM time':>12s}")
    print("-" * 78)
    for n in names:
        sgm_s = f"{sgm[n]:.4f}s" if sgm[n] is not None else "—"
        print(f"{n:14s} {solved_counts[n]:4d}/{len(rows):<4d} "
              f"{match_counts[n]:4d}/{len(rows):<4d} {sgm_s:>12s}")
    print("=" * 78)
    print(f"Markdown: {md}")
    print(f"JSONL:    {jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
