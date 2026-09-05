#!/usr/bin/env python3
"""Fair SOR vs solver_accl: Julia boots once per engine (warmed batch).

Unlike run_compare_solver_accl.py (cold Julia per instance), this script:
  1) runs SOR engines + HiGHS on every Netlib MPS
  2) runs accl-simplex in ONE Julia process over the suite
  3) runs accl-pdhg in ONE Julia process over the suite
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_compare import (  # noqa: E402
    SOR_KEYS,
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
SERVER_JL = ROOT / "scripts" / "accl_server.jl"


def _start_accl_server(warmup_mps: Path | None = None,
                       engine: str = "simplex",
                       max_iter: int = 200_000,
                       tol: float = 1e-6) -> subprocess.Popen:
    import select
    env = {**os.environ, "SOLVER_ACCL_ROOT": SOLVER_ACCL_ROOT}
    proc = subprocess.Popen(
        [JULIA, f"--project={SOLVER_ACCL_ROOT}", str(SERVER_JL)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, bufsize=1,
    )
    assert proc.stderr is not None
    # Wait until server finished JIT / include
    deadline = time.perf_counter() + 180
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"accl server exited early: {err[-400:]}")
        line = proc.stderr.readline()
        if not line:
            time.sleep(0.05)
            continue
        print(f"  [julia] {line.rstrip()}", flush=True)
        if "SOR_ACCL_SERVER_READY" in line:
            break
    else:
        proc.kill()
        raise RuntimeError("accl server failed to become ready")

    # One untimed warmup solve so method JIT isn't charged to the next instance
    if warmup_mps is not None and warmup_mps.is_file():
        assert proc.stdin is not None and proc.stdout is not None
        print(f"  [julia] warmup {warmup_mps.name} ({engine}) ...", flush=True)
        try:
            proc.stdin.write(f"SOLVE {warmup_mps} {engine} {max_iter} {tol}\n")
            proc.stdin.flush()
        except BrokenPipeError as e:
            _kill_accl_server(proc)
            raise RuntimeError(f"warmup write failed: {e}") from e
        ready, _, _ = select.select([proc.stdout], [], [], 60.0)
        if ready:
            _ = proc.stdout.readline()
            if proc.poll() is not None:
                _kill_accl_server(proc)
                raise RuntimeError("server died during warmup")
            print("  [julia] warmup done", flush=True)
        else:
            print("  [julia] warmup timed out — restarting clean", flush=True)
            _kill_accl_server(proc)
            return _start_accl_server(None, engine, max_iter, tol)
    return proc


def _kill_accl_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def _ensure_server(proc: subprocess.Popen | None,
                   warm: Path | None, engine: str,
                   max_iter: int, tol: float) -> subprocess.Popen:
    if proc is not None and proc.poll() is None:
        return proc
    if proc is not None:
        print("  server dead — restarting", flush=True)
        _kill_accl_server(proc)
    return _start_accl_server(warm, engine, max_iter, tol)


def batch_accl(engine: str, inst_dir: Path, time_limit: float,
               max_iter: int, tol: float,
               files: list[Path] | None = None) -> dict[str, dict]:
    """One warmed Julia server; Python enforces hard per-instance timeout.

    If the Julia process crashes on an instance (BrokenPipe / exit), that
    instance is recorded as ``crash`` and the server is restarted so the
    suite can continue.
    """
    import select

    files = list(files) if files is not None else sorted(inst_dir.glob("*.mps"))
    warm = next((p for p in files if p.stem in
                 ("afiro", "adlittle", "blend", "sc50a")),
                files[0] if files else None)

    print(f"batch accl-{engine}: warmed server, {len(files)} instances, "
          f"hard {time_limit}s/instance ...", flush=True)
    t0 = time.perf_counter()
    out: dict[str, dict] = {}
    proc: subprocess.Popen | None = _start_accl_server(
        warm, engine, max_iter, tol)

    try:
        for i, mps in enumerate(files, 1):
            stem = mps.stem
            print(f"  [{i}/{len(files)}] accl-{engine} {stem} ...", flush=True)
            recorded = False
            for attempt in range(2):  # one retry after crash/restart
                try:
                    proc = _ensure_server(proc, warm, engine, max_iter, tol)
                    assert proc.stdin is not None and proc.stdout is not None
                    cmd = f"SOLVE {mps} {engine} {max_iter} {tol}\n"
                    proc.stdin.write(cmd)
                    proc.stdin.flush()
                except (BrokenPipeError, RuntimeError) as e:
                    print(f"  write/start failed ({e}) — restart", flush=True)
                    _kill_accl_server(proc)
                    proc = None
                    if attempt == 0:
                        continue
                    out[stem] = {
                        "instance": stem, "engine": engine,
                        "loadable": True, "status": "crash",
                        "error": f"server broken pipe: {e}",
                    }
                    print("    -> crash (server)", flush=True)
                    recorded = True
                    break

                assert proc is not None and proc.stdout is not None
                ready, _, _ = select.select(
                    [proc.stdout], [], [], time_limit + 0.5)
                if not ready:
                    # Distinguish hang vs already-dead process
                    if proc.poll() is not None:
                        print("  server exited during solve — crash",
                              flush=True)
                        _kill_accl_server(proc)
                        proc = None
                        if attempt == 0:
                            continue
                        out[stem] = {
                            "instance": stem, "engine": engine,
                            "loadable": True, "status": "crash",
                            "error": "server exited during solve",
                        }
                        print("    -> crash", flush=True)
                        recorded = True
                        break
                    print(f"  HARD TIMEOUT {time_limit}s — killing server",
                          flush=True)
                    _kill_accl_server(proc)
                    proc = None
                    out[stem] = {
                        "instance": stem, "engine": engine,
                        "loadable": True, "status": "timeout",
                        "solve_s": time_limit,
                        "error": (f"per-instance limit {time_limit}s "
                                  "(server killed)"),
                    }
                    print("    -> timeout", flush=True)
                    recorded = True
                    break

                line = proc.stdout.readline()
                if proc.poll() is not None and not line:
                    print("  server died after solve — crash", flush=True)
                    _kill_accl_server(proc)
                    proc = None
                    if attempt == 0:
                        continue
                    out[stem] = {
                        "instance": stem, "engine": engine,
                        "loadable": True, "status": "crash",
                        "error": "server died with empty response",
                    }
                    print("    -> crash", flush=True)
                    recorded = True
                    break

                line = (line or "").strip()
                if not line.startswith("{"):
                    out[stem] = {
                        "instance": stem, "engine": engine,
                        "status": "error",
                        "error": f"bad response: {line[:200]}",
                    }
                    print(f"    -> error ({line[:60]})", flush=True)
                    recorded = True
                    break

                j = json.loads(line)
                out[stem] = j
                st = j.get("status") or (
                    "skipped" if not j.get("loadable", True) else "?"
                )
                print(f"    -> {st}  {j.get('solve_s', '')}", flush=True)
                recorded = True
                break

            if not recorded:
                out[stem] = {
                    "instance": stem, "engine": engine,
                    "status": "error", "error": "unrecorded",
                }
    finally:
        try:
            if proc is not None and proc.poll() is None and proc.stdin:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            _kill_accl_server(proc)

    wall = time.perf_counter() - t0
    n_ok = sum(1 for j in out.values() if j.get("loadable", True))
    n_skip = sum(1 for j in out.values() if not j.get("loadable", True))
    print(f"  accl-{engine}: {len(out)} rows ({n_ok} loadable-ish, "
          f"{n_skip} skipped), batch wall {wall:.1f}s", flush=True)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="netlib")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--max-iter", type=int, default=200_000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--obj-tol", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse-jsonl", default="",
                    help="reuse SOR/HiGHS rows from a prior fair JSONL; "
                         "only re-run Julia batches")
    ap.add_argument("-o", "--outdir", default="benchmarks/results")
    args = ap.parse_args()

    inst_dir = ROOT / "benchmarks" / args.suite / "mps"
    exe = ROOT / "build" / "sor_solve"
    venv_py = ROOT / "benchmarks" / ".venv-baseline" / "bin" / "python"
    highs_script = ROOT / "scripts" / "run_highs_baseline.py"

    if not Path(JULIA).exists():
        print(f"error: julia not found at {JULIA}", file=sys.stderr)
        return 2
    if not Path(SOLVER_ACCL_ROOT).is_dir():
        print(f"error: SOLVER_ACCL_ROOT missing: {SOLVER_ACCL_ROOT}",
              file=sys.stderr)
        return 2

    instances = sorted(inst_dir.glob("*.mps"))
    if args.limit:
        instances = instances[: args.limit]

    solver_names = [
        "SOR-simplex", "SOR-pdhg", "SOR-hpr",
        "accl-simplex", "accl-pdhg", "highs",
    ]

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl = outdir / f"compare-solver_accl-fair-{stamp}.jsonl"
    md = outdir / f"compare-solver_accl-fair-{stamp}.md"

    env = {
        "record": "environment",
        "mode": "fair_batch",
        "suite": args.suite,
        "competitor": "https://github.com/shreyas-omkar/solver_accl",
        "n_instances": len(instances),
        "time_limit_s": args.time_limit,
        "per_instance_hard_timeout": True,
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "solvers": solver_names,
        "note": "accl: single warmed Julia process per engine; "
                "hard per-instance InterruptException timeout",
    }

    rows: list[dict] = []
    if args.reuse_jsonl:
        reuse = Path(args.reuse_jsonl)
        if not reuse.is_file():
            print(f"error: --reuse-jsonl not found: {reuse}", file=sys.stderr)
            return 2
        print(f"=== [1/3] Reusing SOR+HiGHS from {reuse} ===", flush=True)
        for line in reuse.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("record") == "instance":
                # drop any prior accl columns; will refill
                rec.setdefault("results", {})
                rec["results"].pop("accl-simplex", None)
                rec["results"].pop("accl-pdhg", None)
                rows.append(rec)
        if args.limit:
            rows = rows[: args.limit]
        print(f"  loaded {len(rows)} instance rows", flush=True)
        with jsonl.open("w") as f:
            f.write(json.dumps(env) + "\n")
            for row in rows:
                f.write(json.dumps(row) + "\n")
    else:
        if not exe.exists():
            print(f"error: {exe} not built", file=sys.stderr)
            return 2
        with jsonl.open("w") as f:
            f.write(json.dumps(env) + "\n")

        print(f"=== [1/3] SOR + HiGHS on {len(instances)} instances ===",
              flush=True)
        for i, mps in enumerate(instances, 1):
            print(f"── [{i}/{len(instances)}] {mps.stem} ──", flush=True)
            results: dict[str, dict] = {}
            row_meta = {"rows": None, "cols": None, "nnz": None}

            for key in ("sor-simplex", "sor-pdhg", "sor-hpr"):
                display, eng, b_ov, extra = SOR_KEYS[key]
                res = run_sor(exe, mps, args.time_limit, args.max_iter,
                              args.tol, b_ov or args.backend, eng, extra,
                              name=display)
                results[res.name] = asdict(res)
                if row_meta["rows"] is None and res.rows:
                    row_meta.update(rows=res.rows, cols=res.cols, nnz=res.nnz)
                print(f"    {res.name:16s}  {res.status:16s}  "
                      f"{fmt_obj(res.objective):>12s}  {fmt_time(res.wall_s)}",
                      flush=True)

            hres = run_external(venv_py, highs_script, mps, args.time_limit,
                                "highs")
            results["highs"] = asdict(hres)
            print(f"    {'highs':16s}  {hres.status:16s}  "
                  f"{fmt_obj(hres.objective):>12s}  {fmt_time(hres.wall_s)}",
                  flush=True)

            ref = hres.objective if is_solved(hres.status) else None
            rows.append({
                "record": "instance",
                "instance": mps.stem,
                **row_meta,
                "ref_objective": ref,
                "results": results,
                "agreements": {},
            })
            with jsonl.open("a") as f:
                f.write(json.dumps(rows[-1]) + "\n")

    print("=== [2/3] Julia batch accl-simplex (boot once, hard per-inst) ===",
          flush=True)
    # Match the instance set used for SOR/HiGHS (respects --limit / reuse)
    accl_files = [inst_dir / f"{r['instance']}.mps" for r in rows]
    accl_files = [p for p in accl_files if p.is_file()]
    accl_s = batch_accl("simplex", inst_dir, args.time_limit,
                        args.max_iter, args.tol, files=accl_files)
    print("=== [3/3] Julia batch accl-pdhg (boot once, hard per-inst) ===",
          flush=True)
    accl_p = batch_accl("pdhg", inst_dir, args.time_limit,
                        args.max_iter, args.tol, files=accl_files)

    for row in rows:
        stem = row["instance"]
        for eng, data in (("simplex", accl_s), ("pdhg", accl_p)):
            j = data.get(stem, {})
            if not j:
                rd = {
                    "name": f"accl-{eng}",
                    "status": "not_run",
                    "objective": None,
                    "wall_s": None,
                    "error": "missing from batch",
                }
            elif not j.get("loadable", True):
                rd = {
                    "name": f"accl-{eng}",
                    "status": "skipped",
                    "objective": None,
                    "wall_s": None,
                    "error": j.get("reason"),
                }
            else:
                rd = {
                    "name": f"accl-{eng}",
                    "status": j.get("status", "unknown"),
                    "objective": j.get("objective"),
                    "wall_s": j.get("solve_s"),
                    "iterations": j.get("iterations"),
                    "error": j.get("error"),
                }
            row["results"][f"accl-{eng}"] = rd

        ref = row.get("ref_objective")
        for n, rd in row["results"].items():
            if ref is None or rd.get("objective") is None:
                row["agreements"][n] = None
            else:
                g = rel_gap(rd["objective"], ref)
                row["agreements"][n] = g is not None and g <= args.obj_tol

    # rewrite instance rows + summary
    with jsonl.open("w") as f:
        f.write(json.dumps(env) + "\n")
        for row in rows:
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
        "# SOR vs solver_accl (fair — Julia boots once per engine)",
        "",
        f"- Competitor: [solver_accl](https://github.com/shreyas-omkar/solver_accl)",
        f"- Instances: **{len(rows)}** / Netlib",
        f"- Time limit: {args.time_limit}s",
        "- accl times: **single warmed Julia process** per engine "
        "(solve_s only; hard per-instance timeout via InterruptException)",
        f"- Host: `{env['host']}` · {env['cpu_count']} CPUs",
        "",
        "## Summary (vs HiGHS reference)",
        "",
        "| Solver | Solved | Obj match | SGM time (s) |",
        "|--------|-------:|----------:|-------------:|",
    ]
    for n in solver_names:
        s = f"{sgm[n]:.4f}" if sgm[n] is not None else "—"
        lines.append(
            f"| {n} | {solved[n]}/{len(rows)} | {match[n]}/{len(rows)} | {s} |"
        )

    lines += ["", "## Per-instance", ""]
    hdr = "| Instance | size |"
    sep = "|----------|-----:|"
    for n in solver_names:
        hdr += f" {n} status | {n} obj | {n} time |"
        sep += "--------|--------:|---------:|"
    lines += [hdr, sep]
    for row in rows:
        size = (f"{row['rows']}×{row['cols']}" if row.get("rows") else "")
        cells = f"| {row['instance']} | {size} |"
        for n in solver_names:
            rd = row["results"].get(n, {})
            cells += (
                f" {rd.get('status', '—')} | {fmt_obj(rd.get('objective'))} | "
                f"{fmt_time(rd.get('wall_s'))} |"
            )
        lines.append(cells)

    lines += ["", f"JSONL: `{jsonl}`"]
    md.write_text("\n".join(lines) + "\n")

    print("\n" + "=" * 80)
    print(f"{'Solver':18s} {'Solved':>10s} {'Match':>10s} {'SGM':>10s}")
    for n in solver_names:
        s = f"{sgm[n]:.4f}s" if sgm[n] else "—"
        print(f"{n:18s} {solved[n]:4d}/{len(rows):<4d} "
              f"{match[n]:4d}/{len(rows):<4d} {s:>10s}")
    print("=" * 80)
    print(f"Markdown: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
