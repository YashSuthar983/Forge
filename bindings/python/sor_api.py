#!/usr/bin/env python3
"""SOR Python API — thin client over sor_solve / sor_check.

Part of the project (not the web demo).

  python3 -m python.sor_api --one-shot
  python3 scripts/sor_repl.py
  from python.sor_api import solve, check, show   # if PYTHONPATH=sor root
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo_out"
OUT.mkdir(exist_ok=True)

BIN = None
for cand in (ROOT / "build", ROOT / "build-native"):
    if (cand / "sor_solve").is_file():
        BIN = cand
        break
if BIN is None:
    raise SystemExit("build sor_solve first (cmake -S . -B build && cmake --build build -j)")

PRESETS = {
    "blend": ROOT / "examples/crude_blending/blend_s42.mps",
    "dispatch": ROOT / "examples/dispatch/dispatch_s42.qps",
    "schedule": ROOT / "examples/scheduling/schedule_s42.mps",
    "sparse": ROOT / "examples/sparse500.mps",
    "testlp": ROOT / "examples/testlp.mps",
}


def solve(
    model: str | Path,
    *,
    engine: str = "simplex",
    backend: str = "cpu",
    method: str | None = "auto",
    time_limit: float | None = None,
    max_iter: int | None = None,
    verbose: bool = False,
    solution_out: str | Path | None = None,
) -> dict:
    """Call sor_solve. Returns parsed fields + raw stdout."""
    model = PRESETS.get(str(model), Path(model))
    if not Path(model).is_file():
        raise FileNotFoundError(model)
    if solution_out is None:
        solution_out = OUT / f"repl_{Path(model).stem}.sol"
    argv = [
        str(BIN / "sor_solve"),
        str(model),
        "--engine",
        engine,
        "--backend",
        backend,
        "--solution-out",
        str(solution_out),
    ]
    if method and engine in ("simplex", "milp"):
        argv += ["--method", method]
    if time_limit is not None:
        argv += ["--time-limit", str(time_limit)]
    if max_iter is not None:
        argv += ["--max-iter", str(max_iter)]
    if verbose:
        argv.append("--verbose")
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(ROOT))
    fields = {}
    for line in proc.stdout.splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            k = k.strip().lower().replace(" ", "_")
            if k in (
                "status",
                "proof_level",
                "objective",
                "engine",
                "backend",
                "termination",
                "nodes",
                "iterations",
                "mip_gap",
                "dual_bound",
            ):
                fields[k] = v.strip()
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": " ".join(argv),
        "model": str(model),
        "solution": str(solution_out),
        **fields,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def check(model: str | Path, solution: str | Path) -> dict:
    model = PRESETS.get(str(model), Path(model))
    argv = [str(BIN / "sor_check"), str(model), str(solution)]
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(ROOT))
    return {
        "ok": proc.returncode == 0 and "VERIFIED" in proc.stdout,
        "verified": "VERIFIED" in proc.stdout,
        "command": " ".join(argv),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def show(r: dict) -> None:
    print("command:", r.get("command", ""))
    for k in ("status", "proof_level", "objective", "engine", "backend", "nodes", "iterations"):
        if k in r:
            print(f"  {k}: {r[k]}")
    if "verified" in r:
        print("  verified:", r["verified"])
    if r.get("stderr"):
        print("stderr:", r["stderr"][:400])


HELP = """
SOR Python API / REPL  (same binaries as CLI)

  solve("blend")
  solve("blend", engine="simplex", backend="cpu")
  solve("dispatch", engine="qp", method=None)
  solve("schedule", engine="milp", time_limit=30, verbose=True)
  solve("sparse", engine="hpr", backend="vulkan", method=None, max_iter=50000, time_limit=30)
  r = solve("blend"); check("blend", r["solution"])

  presets: blend | dispatch | schedule | sparse | testlp
  or path: solve("examples/testlp.mps")

  help_sor()   ·   quit with Ctrl-D / exit()
""".strip()


def help_sor() -> None:
    print(HELP)


def run_repl() -> None:
    print(HELP)
    print(f"binaries: {BIN}")
    print("Type solve(...)  —  basic API over sor_solve / sor_check.\n")
    import code

    code.interact(
        banner="",
        local={
            "solve": solve,
            "check": check,
            "show": show,
            "help_sor": help_sor,
            "PRESETS": PRESETS,
            "BIN": BIN,
            "OUT": OUT,
            "json": json,
        },
    )


def run_one_shot() -> None:
    print("=== API: solve blend ===")
    r = solve("blend")
    show(r)
    print("\n=== API: check ===")
    show(check("blend", r["solution"]))
    print("\n=== API: solve dispatch qp ===")
    show(solve("dispatch", engine="qp", method=None))
    print("\n=== API: solve schedule milp ===")
    show(solve("schedule", engine="milp", time_limit=30, method="auto"))
    print("\n=== API: solve sparse hpr vulkan ===")
    show(
        solve(
            "sparse",
            engine="hpr",
            backend="vulkan",
            method=None,
            max_iter=50000,
            time_limit=30,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one-shot":
        run_one_shot()
    else:
        run_repl()
