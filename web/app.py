#!/usr/bin/env python3
"""SOR demo console — thin FastAPI shell over sor_solve / sor_check / sor_gen.

Presentation surface for SIH26119. The product is the C++ engine; every solve
is a subprocess. Not a modelling environment.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lp_text import SAMPLE as LP_SAMPLE
from lp_text import LpTextError, lp_text_to_mps

LEARN_TEMPLATES: list[dict[str, str]] = [
    {
        "id": "toy_lp",
        "label": "Start here · tiny LP",
        "blurb": "Two variables, two inequalities. Best first lesson.",
        "text": """Maximize
  3 x + 4 y
Subject To
  wood:   x + 2 y <= 14
  metal:  3 x + y <= 18
Bounds
  x >= 0
  y >= 0
End
""",
    },
    {
        "id": "diet",
        "label": "Diet · meet nutrition",
        "blurb": "Minimize cost while hitting calorie and protein targets.",
        "text": """Minimize
  2 bread + 3 milk + 5 eggs
Subject To
  calories:  80 bread + 120 milk + 70 eggs >= 2000
  protein:   4 bread + 8 milk + 6 eggs >= 50
Bounds
  bread >= 0
  milk >= 0
  eggs >= 0
End
""",
    },
    {
        "id": "knapsack",
        "label": "Knapsack · integers",
        "blurb": "Pick whole items that fit a weight limit — introduces Binary / General.",
        "text": """Maximize
  10 laptop + 6 camera + 4 book
Subject To
  weight: 5 laptop + 3 camera + 1 book <= 8
Binary
  laptop
  camera
  book
End
""",
    },
    {
        "id": "blend_mini",
        "label": "Mini blend · refinery flavour",
        "blurb": "Mix two crudes into one product under a sulfur cap.",
        "text": """Maximize
  90 product - 40 light - 55 heavy
Subject To
  mass:     light + heavy - product = 0
  sulfur:   0.5 light + 2.5 heavy - 1.0 product <= 0
  demand:   product >= 100
Bounds
  light >= 0
  heavy >= 0
  product >= 0
End
""",
    },
]

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_BIN = Path(os.environ.get("SOR_BIN_DIR", ROOT / "build-native"))
DEFAULT_EXAMPLES = Path(os.environ.get("SOR_EXAMPLES", ROOT / "examples"))
DEFAULT_TIMEOUT = float(os.environ.get("SOR_WEB_TIMEOUT", "90"))
SESSIONS = Path(tempfile.gettempdir()) / "sor_web_sessions"
SESSIONS.mkdir(exist_ok=True)

PRESETS: dict[str, dict[str, Any]] = {
    "blend": {
        "label": "Crude blending",
        "kind": "lp",
        "path": DEFAULT_EXAMPLES / "crude_blending" / "blend_s42.mps",
        "engine": "simplex",
        "backend": "cpu",
        "blurb": "Pick crudes and products. Maximize margin under quality limits.",
    },
    "schedule": {
        "label": "Unit scheduling",
        "kind": "milp",
        "path": DEFAULT_EXAMPLES / "scheduling" / "schedule_s42.mps",
        "engine": "milp",
        "backend": "cpu",
        "blurb": "Turn units on/off across periods. Integer decisions.",
    },
    "dispatch": {
        "label": "Power dispatch",
        "kind": "qp",
        "path": DEFAULT_EXAMPLES / "dispatch" / "dispatch_s42.qps",
        "engine": "qp",
        "backend": "cpu",
        "blurb": "Meet demand at lowest quadratic generation cost.",
    },
    "sparse": {
        "label": "Large sparse LP",
        "kind": "lp",
        "path": DEFAULT_EXAMPLES / "sparse500.mps",
        "engine": "hpr",
        "backend": "cpu",
        "blurb": "First-order method — flip Advanced → GPU if you have Vulkan.",
    },
}

STATUS_RE = re.compile(r"^status:\s+(\S+)", re.M)
PROOF_RE = re.compile(r"^proof_level:\s+(\S+)", re.M)
OBJ_RE = re.compile(r"^objective:\s+([^\s]+)", re.M)
HUMAN_RE = re.compile(r"^ {19}(.+)$", re.M)
DOWNGRADE_RE = re.compile(r"^downgrade:\s+(.+)$", re.M)
ENGINE_RE = re.compile(r"^engine:\s+(\S+)", re.M)
BACKEND_RE = re.compile(r"^backend:\s+(.+)$", re.M)


def _bin(name: str) -> Path:
    p = DEFAULT_BIN / name
    if not p.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Binary not found: {p}. Build SOR or set SOR_BIN_DIR.",
        )
    return p


def _parse_solve_output(text: str) -> dict[str, Any]:
    def first(rx: re.Pattern[str]) -> str | None:
        m = rx.search(text)
        return m.group(1).strip() if m else None

    obj_raw = first(OBJ_RE)
    objective: float | None = None
    if obj_raw is not None:
        try:
            objective = float(obj_raw)
        except ValueError:
            objective = None

    return {
        "status": first(STATUS_RE),
        "proof_level": first(PROOF_RE),
        "human": first(HUMAN_RE),
        "downgrade": first(DOWNGRADE_RE),
        "objective": objective,
        "engine_line": first(ENGINE_RE),
        "backend_line": first(BACKEND_RE),
    }


async def _run(cmd: list[str], *, timeout: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "ok": False,
                "timed_out": True,
                "returncode": None,
                "wall_s": time.perf_counter() - t0,
                "stdout": "",
                "stderr": f"killed after {timeout:.0f}s timeout",
                "cmd": cmd,
            }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return {
        "ok": proc.returncode == 0,
        "timed_out": False,
        "returncode": proc.returncode,
        "wall_s": time.perf_counter() - t0,
        "stdout": out_b.decode("utf-8", errors="replace"),
        "stderr": err_b.decode("utf-8", errors="replace"),
        "cmd": cmd,
    }


app = FastAPI(title="SOR Demo Console", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    bins = {n: (DEFAULT_BIN / n).is_file() for n in ("sor_solve", "sor_check", "sor_gen")}
    presets: dict[str, dict[str, Any]] = {}
    for k, v in PRESETS.items():
        p = Path(v["path"])
        try:
            rel = str(p.relative_to(ROOT))
        except ValueError:
            rel = str(p)
        presets[k] = {
            "label": v["label"],
            "kind": v["kind"],
            "engine": v["engine"],
            "backend": v["backend"],
            "blurb": v["blurb"],
            "exists": p.is_file(),
            "path": rel,
        }

    return {
        "ok": all(bins.values()),
        "bin_dir": str(DEFAULT_BIN),
        "examples": str(DEFAULT_EXAMPLES),
        "timeout_s": DEFAULT_TIMEOUT,
        "bins": bins,
        "presets": presets,
        "note": "Demo UI over the SOR engine — CLI remains the PS interface.",
        "lp_sample": LP_SAMPLE,
        "learn_templates": LEARN_TEMPLATES,
    }


@app.post("/api/solve")
async def solve(
    preset: str | None = Form(default=None),
    engine: str = Form(default="simplex"),
    backend: str = Form(default="cpu"),
    method: str = Form(default="auto"),
    time_limit: float = Form(default=30.0),
    verbose: bool = Form(default=False),
    max_iter: int | None = Form(default=None),
    model_text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    if time_limit < 0 or time_limit > 300:
        raise HTTPException(400, "time_limit must be in [0, 300]")

    solve_bin = _bin("sor_solve")
    session = Path(tempfile.mkdtemp(prefix="sor_sol_", dir=SESSIONS))
    model_path: Path
    display_name: str
    text_meta: dict[str, Any] | None = None

    try:
        if model_text is not None and model_text.strip():
            try:
                mps, text_meta = lp_text_to_mps(model_text, name="WEBTEXT")
            except LpTextError as e:
                raise HTTPException(400, f"Equation parse error: {e}") from e
            model_path = session / "model.mps"
            model_path.write_text(mps)
            display_name = "Your equations"
            if engine in {"simplex", "auto"} or (
                engine == "simplex" and text_meta.get("suggested_engine") == "milp"
            ):
                # Auto-pick milp when integers present and client left LP engine.
                pass
            if text_meta.get("suggested_engine") == "milp" and engine == "simplex":
                engine = "milp"
        elif file is not None and file.filename:
            suffix = Path(file.filename).suffix.lower() or ".mps"
            if suffix not in {".mps", ".qps"}:
                raise HTTPException(400, "Upload .mps or .qps only")
            model_path = session / f"model{suffix}"
            data = await file.read()
            if len(data) > 40_000_000:
                raise HTTPException(400, "Upload too large (40 MB max)")
            model_path.write_bytes(data)
            display_name = file.filename
            if engine == "simplex" and suffix == ".qps":
                engine = "qp"
        elif preset:
            if preset not in PRESETS:
                raise HTTPException(400, f"Unknown preset '{preset}'")
            info = PRESETS[preset]
            src = Path(info["path"])
            if not src.is_file():
                raise HTTPException(404, f"Preset file missing: {src}")
            model_path = session / src.name
            shutil.copy2(src, model_path)
            display_name = str(info["label"])
        else:
            raise HTTPException(
                400, "Provide preset=…, model_text=…, or upload a model file"
            )

        sol_path = session / "out.sol"
        cmd = [
            str(solve_bin),
            str(model_path),
            "--engine",
            engine,
            "--backend",
            backend,
            "--method",
            method,
            "--solution-out",
            str(sol_path),
        ]
        if time_limit > 0:
            cmd += ["--time-limit", str(time_limit)]
        if verbose:
            cmd.append("--verbose")
        if max_iter is not None:
            cmd += ["--max-iter", str(max_iter)]

        timeout = max(time_limit + 15.0, 20.0) if time_limit > 0 else DEFAULT_TIMEOUT
        timeout = min(timeout, max(DEFAULT_TIMEOUT, time_limit + 15.0))

        result = await _run(cmd, timeout=timeout)
        parsed = _parse_solve_output(result["stdout"])
        if text_meta and text_meta.get("maximize") and parsed.get("objective") is not None:
            # Free MPS has no MAXIMIZE flag — we negated costs on write.
            parsed["objective"] = -float(parsed["objective"])

        check_ready = sol_path.is_file() and sol_path.stat().st_size > 0

        out = {
            **result,
            **parsed,
            "display_name": display_name,
            "model_name": model_path.name,
            "sol_token": session.name if check_ready else None,
            "check_ready": check_ready,
            "engine": engine,
            "backend": backend,
        }
        if text_meta:
            out["model_meta"] = text_meta
        return out
    except HTTPException:
        shutil.rmtree(session, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(session, ignore_errors=True)
        raise


@app.post("/api/check")
async def check(
    sol_token: str = Form(...),
    tol: float = Form(default=1e-7),
) -> dict[str, Any]:
    if "/" in sol_token or ".." in sol_token or not sol_token.startswith("sor_sol_"):
        raise HTTPException(400, "Invalid sol_token")
    session = SESSIONS / sol_token
    if not session.is_dir():
        raise HTTPException(404, "Solution expired — solve again")

    candidates = [p for p in session.iterdir() if p.suffix.lower() in {".mps", ".qps"}]
    if not candidates:
        raise HTTPException(404, "Model missing in session")
    model_file = candidates[0]
    sol_file = session / "out.sol"
    if not sol_file.is_file():
        raise HTTPException(404, "Solution file missing")

    check_bin = _bin("sor_check")
    cmd = [str(check_bin), str(model_file), str(sol_file), "--tol", str(tol)]
    result = await _run(cmd, timeout=60.0)
    passed = bool(result["ok"]) and "FAIL" not in result["stdout"]
    return {**result, "passed": passed}


@app.post("/api/gen")
async def gen(
    kind: str = Form(default="all"),
    seed: int = Form(default=42),
) -> dict[str, Any]:
    if kind not in {"blend", "schedule", "dispatch", "all"}:
        raise HTTPException(400, "kind must be blend|schedule|dispatch|all")
    gen_bin = _bin("sor_gen")
    outdir = DEFAULT_EXAMPLES
    outdir.mkdir(parents=True, exist_ok=True)

    if kind == "all":
        cmd = [str(gen_bin), "all", "--seed", str(seed), "--outdir", str(outdir)]
    else:
        targets = {
            "blend": outdir / "crude_blending" / f"blend_s{seed}.mps",
            "schedule": outdir / "scheduling" / f"schedule_s{seed}.mps",
            "dispatch": outdir / "dispatch" / f"dispatch_s{seed}.qps",
        }
        targets[kind].parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(gen_bin), kind, "--seed", str(seed), "-o", str(targets[kind])]

    return await _run(cmd, timeout=60.0)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("SOR_WEB_HOST", "127.0.0.1"),
        port=int(os.environ.get("SOR_WEB_PORT", "8765")),
        reload=False,
    )


if __name__ == "__main__":
    main()
