#!/usr/bin/env python3
"""CPU vs Vulkan crossover study on the synthetic LP ladder.

Times transfer-inclusive HPR solves. The sizes where the GPU loses are the
honest half of the result - Netlib sits entirely below the expected crossover.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LADDER = ["lp_1e5", "lp_5e5", "lp_1e6", "lp_2e6", "lp_5e6"]

_PAT = {
    "size": re.compile(r"rows x cols:\s+(\d+) x (\d+)\s+nnz (\d+)"),
    "status": re.compile(r"^status:\s+(\S+)", re.M),
    "obj": re.compile(r"^objective:\s+(\S+)", re.M),
    "pres": re.compile(r"^max row violation:\s+(\S+)", re.M),
    "dres": re.compile(r"^dual residual:\s+(\S+)", re.M),
    "iters": re.compile(r"^iterations:\s+(\d+)", re.M),
    "total": re.compile(r"^\s+total\s+([0-9.]+)", re.M),
    "h2d": re.compile(r"host->device\s+([0-9.]+)\s+\((\d+) bytes\)"),
    "d2h": re.compile(r"device->host\s+([0-9.]+)\s+\((\d+) bytes\)"),
    "backend": re.compile(r"^backend:\s+(\S+)", re.M),
}


def run_one(exe: Path, mps: Path, backend: str, max_iter: int, time_limit: float,
            tol: float) -> dict:
    cmd = [str(exe), str(mps), "--engine", "hpr", "--backend", backend,
           "--max-iter", str(max_iter), "--tol", str(tol),
           "--time-limit", str(time_limit)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=time_limit + 120)
    wall = time.perf_counter() - t0
    txt = p.stdout
    def g(k, cast=float):
        m = _PAT[k].search(txt)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except ValueError:
            return None
    size = _PAT["size"].search(txt)
    h2d = _PAT["h2d"].search(txt)
    d2h = _PAT["d2h"].search(txt)
    return {
        "backend": backend,
        "resolved_backend": (m.group(1) if (m := _PAT["backend"].search(txt)) else None),
        "status": g("status", str),
        "objective": g("obj"),
        "primal_res": g("pres"),
        "dual_res": g("dres"),
        "iterations": g("iters", int),
        "solver_total_ms": g("total"),
        "wall_s": wall,
        "h2d_bytes": int(h2d.group(2)) if h2d else None,
        "d2h_bytes": int(d2h.group(2)) if d2h else None,
        "rows": int(size.group(1)) if size else None,
        "cols": int(size.group(2)) if size else None,
        "nnz": int(size.group(3)) if size else None,
        "returncode": p.returncode,
        "stderr_tail": (p.stderr or "")[-400:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--backends", default="cpu,vulkan")
    ap.add_argument("-o", default="benchmarks/results")
    args = ap.parse_args()

    exe = ROOT / "build" / "sor_solve"
    inst = ROOT / "benchmarks" / "synthetic" / "mps"
    if not exe.exists():
        print("error: build/sor_solve missing", file=sys.stderr)
        return 2

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    rows = []
    print(f"crossover: {len(LADDER)} instances × {backends}  "
          f"max_iter={args.max_iter}  limit={args.time_limit}s\n", flush=True)

    for name in LADDER:
        mps = inst / f"{name}.mps"
        if not mps.exists():
            print(f"  skip {name} (missing)", flush=True)
            continue
        rec = {"instance": name}
        print(f"── {name} ──", flush=True)
        for b in backends:
            r = run_one(exe, mps, b, args.max_iter, args.time_limit, args.tol)
            rec[b] = r
            print(f"    {b:8s}  backend={r['resolved_backend']}  "
                  f"status={r['status']}  wall={r['wall_s']:.3f}s  "
                  f"iters={r['iterations']}  nnz={r['nnz']}  "
                  f"h2d={r['h2d_bytes']}  d2h={r['d2h_bytes']}", flush=True)
        if "cpu" in rec and "vulkan" in rec:
            tc, tg = rec["cpu"].get("wall_s"), rec["vulkan"].get("wall_s")
            if tc and tg and tg > 0:
                rec["speedup_gpu_over_cpu"] = tc / tg
                print(f"    speedup  {rec['speedup_gpu_over_cpu']:.2f}×  "
                      f"(>1 means GPU faster)", flush=True)
            oc, og = rec["cpu"].get("objective"), rec["vulkan"].get("objective")
            if oc is not None and og is not None and abs(oc) + 1 > 0:
                rec["obj_rel_diff"] = abs(oc - og) / (1.0 + abs(oc))
        rows.append(rec)

    outdir = ROOT / args.o
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl = outdir / f"crossover-{stamp}.jsonl"
    md = outdir / f"crossover-{stamp}.md"
    with jsonl.open("w") as f:
        f.write(json.dumps({"record": "env", "max_iter": args.max_iter,
                            "time_limit": args.time_limit}) + "\n")
        for r in rows:
            f.write(json.dumps({"record": "instance", **r}) + "\n")

    lines = ["# CPU vs Vulkan crossover (RX 5500M)", "",
             f"- max_iter={args.max_iter}  time_limit={args.time_limit}s",
             "- Times include host↔device transfer.",
             ""]
    lines.append("| Instance | nnz | CPU wall | Vulkan wall | GPU× | obj rel-diff | CPU D2H B | GPU D2H B |")
    lines.append("|----------|----:|---------:|------------:|-----:|-------------:|----------:|----------:|")
    for r in rows:
        cpu, vk = r.get("cpu", {}), r.get("vulkan", {})
        sp = r.get("speedup_gpu_over_cpu")
        sp_s = f"{sp:.2f}×" if sp else "-"
        rd = r.get("obj_rel_diff")
        rd_s = f"{rd:.2e}" if rd is not None else "-"
        nnz = cpu.get("nnz") or vk.get("nnz") or "-"
        lines.append(
            f"| {r['instance']} | {nnz} | "
            f"{cpu.get('wall_s', float('nan')):.3f}s | "
            f"{vk.get('wall_s', float('nan')):.3f}s | {sp_s} | {rd_s} | "
            f"{cpu.get('d2h_bytes') or 0} | {vk.get('d2h_bytes') or 0} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(f"\nMarkdown: {md}\nJSONL: {jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
