#!/usr/bin/env python3
"""FORGE demo console - thin FastAPI shell over sor_solve / sor_check.

Presentation surface for SIH26119. The product is the C++ engine; every solve
is a subprocess. Not a modelling environment.

Models are discovered on disk,
engines / backends are read from the binaries' own usage
text, and results are whatever the binaries print.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lp_text import LpTextError, lp_text_to_mps

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent
STATIC = WEB / "static"


def _env(name: str, default: str | None = None) -> str | None:
    """FORGE_<name>, falling back to the legacy SOR_<name>."""
    return os.environ.get(f"FORGE_{name}") or os.environ.get(f"SOR_{name}") or default


def _default_bin() -> Path:
    env = _env("BIN_DIR")
    if env:
        return Path(env)
    for cand in (ROOT / "build", ROOT / "build-native"):
        if (cand / "sor_solve").is_file():
            return cand
    return ROOT / "build"


BIN_DIR = _default_bin()
MODEL_DIRS = [
    Path(p) for p in (_env("MODEL_DIRS") or os.pathsep.join(
        [str(ROOT / "examples"), str(ROOT / "benchmarks")]
    )).split(os.pathsep) if p
]
TEMPLATE_DIR = Path(_env("TEMPLATE_DIR", str(WEB / "templates")))
MODEL_EXTS = tuple((_env("MODEL_EXTS") or ".mps,.qps,.qplib").split(","))
DEFAULT_TIMEOUT = float(_env("WEB_TIMEOUT", "90"))
DEFAULT_TIME_LIMIT = float(_env("WEB_TIME_LIMIT", "30"))
MAX_TIME_LIMIT = float(_env("WEB_MAX_TIME_LIMIT", "300"))
EDITOR_MAX_BYTES = int(_env("WEB_EDITOR_MAX_BYTES", "400000"))
UPLOAD_MAX_BYTES = int(_env("WEB_UPLOAD_MAX_BYTES", "40000000"))
# Model classes whose sor_check verdict is authoritative today (see the evidence
# report, section L: MILP dual/gap checks and QPS objectives are not supported yet).
CHECK_CLASSES = {c.strip().upper() for c in (_env("CHECK_CLASSES") or "LP").split(",") if c.strip()}

SESSIONS = Path(tempfile.gettempdir()) / "sor_web_sessions"
SESSIONS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Binary introspection
# ---------------------------------------------------------------------------


def _bin(name: str) -> Path:
    p = BIN_DIR / name
    if not p.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Binary not found: {p}. Build FORGE or set FORGE_BIN_DIR.",
        )
    return p


def _usage(name: str, *args: str) -> str:
    p = BIN_DIR / name
    if not p.is_file():
        return ""
    try:
        r = subprocess.run([str(p), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout + r.stderr


def _split_choices(text: str) -> list[str]:
    return [c.strip() for c in text.split("|") if c.strip()]


def _parse_choice_flag(help_text: str, flag: str) -> tuple[list[dict[str, str]], str | None]:
    """Collect `--flag NAME a (default) | b | c` plus `--flag x  description` lines."""
    out: list[dict[str, str]] = []
    default: str | None = None
    seen: set[str] = set()
    for line in help_text.splitlines():
        m = re.match(rf"^\s*{re.escape(flag)}\s+(\S+)\s+(.*)$", line)
        if not m:
            continue
        head, rest = m.group(1), m.group(2).strip()
        if head.isupper():  # `--engine NAME  a (default) | b | c`
            for choice in _split_choices(rest):
                cm = re.match(r"^([\w-]+)(?:\s*\((default)\))?", choice)
                if cm and cm.group(1) not in seen:
                    seen.add(cm.group(1))
                    out.append({"id": cm.group(1), "about": ""})
                    if cm.group(2):
                        default = cm.group(1)
        elif head not in seen:  # `--engine binquad  binary QP (.qplib): ...`
            seen.add(head)
            out.append({"id": head, "about": rest})
    # Continuation lines of a described choice belong to its description.
    lines = help_text.splitlines()
    for item in out:
        if not item["about"]:
            continue
        for i, line in enumerate(lines):
            if re.match(rf"^\s*{re.escape(flag)}\s+{re.escape(item['id'])}\s", line):
                j = i + 1
                while j < len(lines) and re.match(r"^\s{10,}\S", lines[j]):
                    item["about"] += " " + lines[j].strip()
                    j += 1
                break
    return out, default


_TINY_LP = """NAME PROBE
ROWS
 N obj
 L c1
COLUMNS
 x obj 1 c1 1
RHS
 rhs c1 1
ENDATA
"""


def _probe_backend(backend: str, engine: str) -> bool:
    """A backend is available when a tiny first-order solve on it is not Unsupported."""
    d = Path(tempfile.mkdtemp(prefix="sor_probe_", dir=SESSIONS))
    try:
        mps = d / "probe.mps"
        mps.write_text(_TINY_LP)
        # --no-presolve: otherwise presolve solves the probe before the backend runs.
        out = _usage("sor_solve", str(mps), "--engine", engine, "--backend", backend,
                     "--no-presolve", "--time-limit", "5")
        status = _parse_solve_output(out).get("status")
        return bool(status) and status not in {"Unsupported", "Error"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _git_info() -> dict[str, str] | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%h%x09%cd", "--date=short"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or "\t" not in r.stdout:
        return None
    commit, date = r.stdout.strip().split("\t", 1)
    return {"commit": commit, "date": date}


def _build_flags() -> dict[str, str]:
    cache = BIN_DIR / "CMakeCache.txt"
    flags: dict[str, str] = {}
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            m = re.match(r"^((?:SOR|FORGE)_[A-Z0-9_]+|CMAKE_BUILD_TYPE):[A-Z]+=(.*)$", line)
            if m:
                flags[m.group(1)] = m.group(2)
    return flags


@lru_cache(maxsize=1)
def capabilities() -> dict[str, Any]:
    help_text = _usage("sor_solve", "--help")
    engines, default_engine = _parse_choice_flag(help_text, "--engine")
    backends, default_backend = _parse_choice_flag(help_text, "--backend")
    methods, _ = _parse_choice_flag(help_text, "--method")
    engine_ids = {e["id"] for e in engines}
    probe_engine = next((e for e in ("hpr", "pdhg") if e in engine_ids), default_engine or "")
    build = _build_flags()
    for b in backends:
        b["available"] = b["id"] == default_backend or (
            bool(probe_engine) and _probe_backend(b["id"], probe_engine)
        )
        if not b["available"]:
            flag = next((k for k in build if k.endswith(f"_ENABLE_{b['id'].upper()}")), None)
            b["reason"] = (
                f"Disabled when the engine was built ({flag}={build[flag]})"
                if flag and build[flag].upper() in {"OFF", "0", "FALSE", "NO"}
                else "sor_solve reports this backend as Unsupported in this build"
            )
    return {
        "engines": engines,
        "default_engine": default_engine,
        "backends": backends,
        "default_backend": default_backend,
        "methods": methods,
        "has_threads": "--threads" in help_text,
        "build": build,
        "git": _git_info(),
    }


def pick_engine(info: dict[str, Any]) -> str | None:
    """Engine that actually solves this model class, chosen from what the binary offers.

    The CLI default (simplex) silently solves the LP relaxation of a MILP and
    ignores QUADOBJ, so the web UI must route by the model's own structure.
    """
    ids = {e["id"] for e in capabilities()["engines"]}
    if info.get("format") == "qplib" and "auto" in ids:
        return "auto"
    if info.get("quadratic"):
        return next((e for e in ("qp", "qpauto") if e in ids), None)
    if info.get("integers"):
        return "milp" if "milp" in ids else None
    return capabilities()["default_engine"]


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def _model_format(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    for ext in MODEL_EXTS:
        if name.endswith(ext):
            return ext.lstrip(".")
    return None


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _scan_mps(lines) -> dict[str, Any]:
    name = None
    section = None
    obj_rows: set[str] = set()
    rows = 0
    cols = 0
    nnz = 0
    ints: set[str] = set()
    in_int = False
    last_col = None
    quadratic = False
    sense = None
    for raw in lines:
        if not raw.strip() or raw.startswith("*"):
            continue
        if not raw[0].isspace():
            parts = raw.split()
            section = parts[0].upper()
            if section == "NAME" and len(parts) > 1:
                name = parts[1]
            if section == "OBJSENSE" and len(parts) > 1:
                sense = parts[1].upper()
            if section in {"QUADOBJ", "QMATRIX", "QSECTION", "QCMATRIX"}:
                quadratic = True
            continue
        parts = raw.split()
        if section == "ROWS" and len(parts) >= 2:
            if parts[0].upper() == "N":
                obj_rows.add(parts[1])
            else:
                rows += 1
        elif section == "COLUMNS" and parts:
            if len(parts) >= 3 and parts[1].strip("'").upper() == "MARKER":
                in_int = "INTORG" in raw.upper()
                continue
            col = parts[0]
            if col != last_col:
                cols += 1
                last_col = col
            if in_int:
                ints.add(col)
            for r in parts[1::2]:
                if r not in obj_rows:
                    nnz += 1
        elif section == "BOUNDS" and len(parts) >= 3:
            if parts[0].upper() in {"BV", "LI", "UI"}:
                ints.add(parts[2])
        elif section == "OBJSENSE" and parts:
            sense = parts[0].upper()
    return {
        "name": name, "rows": rows, "cols": cols, "nnz": nnz,
        "integers": len(ints), "quadratic": quadratic, "sense": sense,
    }


def _scan_qplib(lines) -> dict[str, Any]:
    body = [l.split("#", 1)[0].strip() for l in lines]
    body = [l for l in body if l]
    info: dict[str, Any] = {"name": body[0] if body else None}
    code = body[1].upper() if len(body) > 1 else ""
    info["qplib_class"] = code
    # QPLIB code = objective / variables / constraints; D, C, Q are all quadratic.
    quad_obj = code[:1] in {"D", "C", "Q"}
    info["quad_constraints"] = code[2:3] in {"D", "C", "Q"}
    info["quadratic"] = quad_obj or info["quad_constraints"]
    info["integers"] = 1 if code[1:2] in {"B", "M", "I", "G"} else 0
    nums = [int(x) for x in body[3:5] if x.lstrip("-").isdigit()]
    info["cols"] = nums[0] if nums else None
    info["rows"] = nums[1] if len(nums) > 1 else None
    info["sense"] = body[2].upper() if len(body) > 2 else None
    return info


def _classify(info: dict[str, Any]) -> str:
    if info.get("quad_constraints"):
        return "MIQCQP" if info.get("integers") else "QCQP"
    if info.get("quadratic"):
        return "MIQP" if info.get("integers") else "QP"
    return "MILP" if info.get("integers") else "LP"


_scan_cache: dict[tuple[str, int, int], dict[str, Any]] = {}


def scan_model(path: Path) -> dict[str, Any]:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key in _scan_cache:
        return _scan_cache[key]
    fmt = _model_format(path) or "mps"
    try:
        with _open_text(path) as f:
            info = _scan_qplib(f) if fmt == "qplib" else _scan_mps(f)
    except OSError as e:
        info = {"error": str(e)}
    info.update(format=fmt, bytes=st.st_size, gz=path.name.lower().endswith(".gz"))
    info["kind"] = _classify(info)
    info["engine"] = pick_engine(info)
    _scan_cache[key] = info
    return info


def _manifest_meta() -> dict[str, dict[str, Any]]:
    """Extra per-file facts from any sor_gen MANIFEST.json under the model dirs."""
    meta: dict[str, dict[str, Any]] = {}
    for d in MODEL_DIRS:
        for mf in d.rglob("MANIFEST.json") if d.is_dir() else []:
            try:
                data = json.loads(mf.read_text())
            except (OSError, ValueError):
                continue
            for inst in data.get("instances", []):
                p = inst.get("path")
                if not p:
                    continue
                full = (ROOT / p).resolve()
                extra = {k: v for k, v in inst.items() if k not in {"path", "kind"}}
                extra["generator"] = data.get("generator")
                meta[str(full)] = extra
    return meta


def _readme_blurb(d: Path) -> str | None:
    readme = d / "README.md"
    if not readme.is_file():
        return None
    for line in readme.read_text(errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith(("#", "|", "```", "-")):
            return re.sub(r"[*_`]", "", s)
    return None


def _humanize(name: str) -> str:
    s = re.sub(r"[_-]+", " ", name).strip()
    return s[:1].upper() + s[1:]


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _group_dir(root: Path, path: Path) -> Path:
    """Shallowest folder, walking down from the model root, that holds models itself.

    Keeps examples/ (and its per-case subfolders) as one group, while benchmarks/
    splits into one group per suite.
    """
    d = root
    for part in path.parent.relative_to(root).parts:
        if any(f.is_file() and _model_format(f) for f in d.iterdir()):
            return d
        d = d / part
    return path.parent


def discover_models() -> list[dict[str, Any]]:
    manifest = _manifest_meta()
    groups: dict[str, dict[str, Any]] = {}
    for root in MODEL_DIRS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or _model_format(path) is None:
                continue
            parent = _group_dir(root, path)
            gid = _rel(parent)
            g = groups.setdefault(gid, {
                "id": gid,
                "title": _humanize(parent.name),
                "about": _readme_blurb(parent) or (
                    _readme_blurb(parent.parent) if parent != root else None
                ),
                "models": [],
            })
            info = scan_model(path)
            g["models"].append({
                "id": _rel(path),
                "file": path.name,
                "title": path.name.split(".")[0],
                **info,
                "meta": manifest.get(str(path.resolve())),
            })
    out = list(groups.values())
    for g in out:
        g["models"].sort(key=lambda m: (m["kind"], m.get("nnz") or 0, m["file"]))
    root_order = {str(_rel(r)): i for i, r in enumerate(MODEL_DIRS)}
    out.sort(key=lambda g: (
        next((i for r, i in root_order.items() if g["id"] == r or g["id"].startswith(r + "/")), 99),
        g["id"] != next((r for r in root_order if g["id"] == r), None),
        g["id"],
    ))
    return out


def resolve_model(model_id: str) -> Path:
    """Map a model id (repo-relative path or session path) to a file, refusing escapes."""
    if model_id.startswith("session:"):
        rel = model_id[len("session:"):]
        path = (SESSIONS / rel).resolve()
        if SESSIONS.resolve() not in path.parents:
            raise HTTPException(400, "Invalid model id")
    else:
        path = (ROOT / model_id).resolve()
        if not any(r.resolve() in path.parents for r in MODEL_DIRS):
            raise HTTPException(400, "Model is outside the configured model directories")
    if not path.is_file() or _model_format(path) is None:
        raise HTTPException(404, f"Model not found: {model_id}")
    return path


def _read_for_editor(path: Path) -> tuple[str, bool]:
    with _open_text(path) as f:
        text = f.read(EDITOR_MAX_BYTES + 1)
    truncated = len(text) > EDITOR_MAX_BYTES
    return (text[:EDITOR_MAX_BYTES] if truncated else text), truncated


def _model_payload(model_id: str, path: Path) -> dict[str, Any]:
    text, truncated = _read_for_editor(path)
    return {
        "id": model_id,
        "file": path.name,
        "info": scan_model(path),
        "text": text,
        "truncated": truncated,
    }


def load_templates() -> list[dict[str, str]]:
    out = []
    if not TEMPLATE_DIR.is_dir():
        return out
    for p in sorted(TEMPLATE_DIR.glob("*.lp")):
        text = p.read_text(errors="replace")
        head = dict(re.findall(r"^#\s*(\w+):\s*(.+)$", text, re.M))
        try:
            _, meta = lp_text_to_mps(text, name="TEMPLATE")
            kind = "MILP" if meta.get("n_integer") else "LP"
        except LpTextError:
            kind = "LP"
        out.append({
            "id": p.stem,
            "title": head.get("title", _humanize(p.stem)),
            "about": head.get("about", ""),
            "kind": kind,
            "text": text,
        })
    return out


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

FIELD_RE = re.compile(r"^([A-Za-z][\w /()+.-]*?):\s+(.*\S)\s*$")
CONT_RE = re.compile(r"^\s{10,}(\S.*)$")


def _parse_solve_output(text: str) -> dict[str, Any]:
    """Every `key: value` line the solver prints before its timing block, in order."""
    header: list[list[str]] = []
    report: list[list[str]] = []
    warnings: list[str] = []
    human = None
    target = header
    total_ms = None
    in_timing = False
    for line in text.splitlines():
        if in_timing:
            m = re.match(r"^\s+total\s+([\d.eE+-]+)", line)
            if m:
                total_ms = float(m.group(1))
            continue
        if line.startswith("timing"):
            in_timing = True
            continue
        if line.lower().startswith(("warning:", "error:")):
            warnings.append(line.strip())
            continue
        m = FIELD_RE.match(line)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            if key == "status":
                target = report
            target.append([key, val])
            continue
        c = CONT_RE.match(line)
        if c and target and target[-1][0] == "proof_level" and human is None:
            human = c.group(1).strip()
        elif c and target:
            target[-1][1] += " " + c.group(1).strip()

    def get(key: str) -> str | None:
        return next((v for k, v in header + report if k == key), None)

    objective: float | None = None
    try:
        objective = float(get("objective")) if get("objective") is not None else None
    except ValueError:
        objective = None
    if objective is not None and objective != objective:  # nan
        objective = None
    return {
        "status": get("status"),
        "proof_level": get("proof_level"),
        "human": human,
        "objective": objective,
        "header": header,
        "report": [kv for kv in report if kv[0] not in {"status", "proof_level", "objective"}],
        "warnings": warnings,
        "solver_ms": total_ms,
    }


def _parse_check_output(text: str) -> dict[str, Any]:
    checks = [
        {"ok": ok == "pass", "name": name.strip(), "residual": res, "tol": tol}
        for ok, name, res, tol in re.findall(
            r"^(pass|FAIL)\s+(.+?)\s+residual=(\S+)\s+tol=(\S+)", text, re.M
        )
    ]
    verdict = next(
        (l.strip() for l in reversed(text.splitlines()) if re.match(r"^[A-Z]{4,}$", l.strip())),
        None,
    )
    warnings = [l.strip() for l in text.splitlines() if l.lower().startswith("warning:")]
    return {"checks": checks, "verdict": verdict, "warnings": warnings}


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


def _session_dir(token: str) -> Path:
    if "/" in token or ".." in token or not token.startswith("sor_sol_"):
        raise HTTPException(400, "Invalid sol_token")
    session = SESSIONS / token
    if not session.is_dir():
        raise HTTPException(404, "Solution expired - solve again")
    return session


def _new_session() -> Path:
    return Path(tempfile.mkdtemp(prefix="sor_sol_", dir=SESSIONS))


def _fix_suffix(path: Path) -> Path:
    """sor_solve's quadratic engines key on the .qps extension, whatever the content."""
    if path.suffix.lower() == ".mps" and scan_model(path).get("quadratic"):
        new = path.with_suffix(".qps")
        path.rename(new)
        return new
    return path


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

app = FastAPI(title="FORGE Demo Console", version="0.2.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    bins = {n: (BIN_DIR / n).is_file() for n in ("sor_solve", "sor_check")}
    caps = await asyncio.to_thread(capabilities) if bins["sor_solve"] else {}
    return {
        "ok": all(bins.values()),
        "bin_dir": str(BIN_DIR),
        "bins": bins,
        "capabilities": caps,
        "check_classes": sorted(CHECK_CLASSES),
        "time_limit": {"default": DEFAULT_TIME_LIMIT, "max": MAX_TIME_LIMIT},
        "templates": load_templates(),
    }


@app.get("/api/models")
async def models() -> dict[str, Any]:
    return {"groups": await asyncio.to_thread(discover_models)}


@app.get("/api/model")
async def get_model(id: str) -> dict[str, Any]:
    path = resolve_model(id)
    return await asyncio.to_thread(_model_payload, id, path)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    name = Path(file.filename or "model.mps").name
    if _model_format(Path(name)) is None:
        raise HTTPException(400, f"Unsupported file type - expected one of {', '.join(MODEL_EXTS)} (optionally .gz)")
    data = await file.read()
    if len(data) > UPLOAD_MAX_BYTES:
        raise HTTPException(400, f"Upload too large ({UPLOAD_MAX_BYTES // 1_000_000} MB max)")
    session = _new_session()
    (session / name).write_bytes(data)
    path = _fix_suffix(session / name)
    model_id = f"session:{session.name}/{path.name}"
    return await asyncio.to_thread(_model_payload, model_id, path)


@app.post("/api/lp-to-mps")
async def lp_to_mps(model_text: str = Form(...)) -> dict[str, Any]:
    try:
        mps, meta = lp_text_to_mps(model_text, name="WEBTEXT")
    except LpTextError as e:
        raise HTTPException(400, f"Equation parse error: {e}") from e
    info = _scan_mps(mps.splitlines())
    info["kind"] = _classify(info)
    info["engine"] = pick_engine(info)
    return {"mps": mps, "meta": meta, "info": info}


@app.post("/api/solve")
async def solve(
    model: str | None = Form(default=None),
    engine: str = Form(default=""),
    backend: str = Form(default=""),
    method: str = Form(default=""),
    threads: int | None = Form(default=None),
    time_limit: float = Form(default=DEFAULT_TIME_LIMIT),
    verbose: bool = Form(default=False),
    model_text: str | None = Form(default=None),
    mps_text: str | None = Form(default=None),
) -> dict[str, Any]:
    if time_limit < 0 or time_limit > MAX_TIME_LIMIT:
        raise HTTPException(400, f"time_limit must be in [0, {MAX_TIME_LIMIT:g}]")

    caps = capabilities()
    solve_bin = _bin("sor_solve")
    session = _new_session()

    try:
        if model_text is not None and model_text.strip():
            try:
                mps, _ = lp_text_to_mps(model_text, name="WEBTEXT")
            except LpTextError as e:
                raise HTTPException(400, f"Equation parse error: {e}") from e
            model_path = session / "model.mps"
            model_path.write_text(mps)
        elif mps_text is not None and mps_text.strip():
            raw = mps_text.strip() + "\n"
            suffix = ".qps" if re.search(r"^QUADOBJ|^QMATRIX|^QSECTION", raw, re.M | re.I) else ".mps"
            model_path = session / f"model{suffix}"
            model_path.write_text(raw)
        elif model:
            src = resolve_model(model)
            model_path = session / src.name
            try:
                model_path.symlink_to(src)
            except OSError:
                shutil.copy2(src, model_path)
        else:
            raise HTTPException(400, "Nothing to solve - pick a model, upload one, or write equations")

        display_path = model if model and model_path.is_symlink() and not model.startswith("session:") else model_path.name
        model_path = await asyncio.to_thread(_fix_suffix, model_path)
        info = await asyncio.to_thread(scan_model, model_path)
        engine_ids = {e["id"] for e in caps["engines"]}
        if not engine:
            engine = info["engine"] or caps["default_engine"] or ""
        if engine and engine not in engine_ids:
            raise HTTPException(400, f"Unknown engine '{engine}'")
        backend_ids = {b["id"] for b in caps["backends"]}
        if backend and backend not in backend_ids:
            raise HTTPException(400, f"Unknown backend '{backend}'")

        sol_path = session / "out.sol"
        cmd = [str(solve_bin), str(model_path), "--solution-out", str(sol_path)]
        if engine:
            cmd += ["--engine", engine]
        if backend:
            cmd += ["--backend", backend]
        if method:
            cmd += ["--method", method]
        if threads and caps["has_threads"]:
            cmd += ["--threads", str(threads)]
        if time_limit > 0:
            cmd += ["--time-limit", f"{time_limit:g}"]
        if verbose:
            cmd.append("--verbose")

        timeout = time_limit + 15.0 if time_limit > 0 else DEFAULT_TIMEOUT
        result = await _run(cmd, timeout=max(timeout, 20.0))
        parsed = _parse_solve_output(result["stdout"] + "\n" + result["stderr"])

        check_ready = sol_path.is_file() and sol_path.stat().st_size > 0
        return {
            **result,
            **parsed,
            # Show the command as a user would type it from the repo root.
            "cmd": [Path(cmd[0]).name, display_path,
                    *[sol_path.name if a == str(sol_path) else a for a in cmd[2:]]],
            "model_info": info,
            "sol_token": session.name if check_ready else None,
            "check_ready": check_ready,
            "check_authoritative": info["kind"] in CHECK_CLASSES,
            "engine": engine,
        }
    except Exception:
        shutil.rmtree(session, ignore_errors=True)
        raise


def _mps_names(path: Path) -> tuple[list[str], list[str]]:
    """(row names, column names) in file order; the objective (N) rows are excluded."""
    rows: list[str] = []
    cols: list[str] = []
    obj: set[str] = set()
    section = None
    last = None
    try:
        with _open_text(path) as fh:
            for raw in fh:
                if not raw.strip() or raw.startswith("*"):
                    continue
                if not raw[0].isspace():
                    section = raw.split()[0].upper()
                    continue
                p = raw.split()
                if section == "ROWS" and len(p) >= 2:
                    if p[0].upper() == "N":
                        obj.add(p[1])
                    else:
                        rows.append(p[1])
                elif section == "COLUMNS" and p:
                    if len(p) >= 3 and p[1].strip("'").upper() == "MARKER":
                        continue
                    if p[0] != last:
                        cols.append(p[0])
                        last = p[0]
                elif section in {"RHS", "RANGES", "BOUNDS"}:
                    break
    except OSError:
        pass
    return rows, cols


def _parse_sol(text: str) -> dict[str, Any]:
    """Parse the plain-text .sol: `key value` scalars and `key N v1 .. vN` vectors."""
    scalars: list[list[str]] = []
    vectors: dict[str, list[float]] = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        key = p[0]
        if len(p) >= 3 and p[1].isdigit() and int(p[1]) == len(p) - 2:
            try:
                vectors[key] = [float(v) for v in p[2:]]
                continue
            except ValueError:
                pass
        scalars.append([key, " ".join(p[1:])])
    return {"scalars": scalars, "vectors": vectors}


@app.get("/api/solution")
async def solution(sol_token: str) -> dict[str, Any]:
    session = _session_dir(sol_token)
    sol_file = session / "out.sol"
    if not sol_file.is_file():
        raise HTTPException(404, "Solution file missing")
    text = sol_file.read_text(errors="replace")
    parsed = _parse_sol(text)
    model_file = next((p for p in session.iterdir() if _model_format(p)), None)
    rows, cols = await asyncio.to_thread(_mps_names, model_file) if model_file else ([], [])
    return {
        **parsed,
        "row_names": rows,
        "col_names": cols,
        "bytes": len(text.encode()),
        "text": text if len(text) <= EDITOR_MAX_BYTES else text[:EDITOR_MAX_BYTES],
        "truncated": len(text) > EDITOR_MAX_BYTES,
    }


@app.post("/api/check")
async def check(
    sol_token: str = Form(...),
    tol: float | None = Form(default=None),
) -> dict[str, Any]:
    session = _session_dir(sol_token)

    candidates = [p for p in session.iterdir() if _model_format(p)]
    if not candidates:
        raise HTTPException(404, "Model missing in session")
    model_file = candidates[0]
    sol_file = session / "out.sol"
    if not sol_file.is_file():
        raise HTTPException(404, "Solution file missing")

    cmd = [str(_bin("sor_check")), str(model_file), str(sol_file)]
    if tol is not None:
        cmd += ["--tol", str(tol)]
    result = await _run(cmd, timeout=60.0)
    parsed = _parse_check_output(result["stdout"] + "\n" + result["stderr"])
    info = scan_model(model_file)
    return {
        **result,
        **parsed,
        "passed": bool(result["ok"]),
        "authoritative": info["kind"] in CHECK_CLASSES,
        "model_kind": info["kind"],
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app:app",
        host=_env("WEB_HOST", "127.0.0.1"),
        port=int(_env("WEB_PORT", "8765")),
        reload=False,
    )


if __name__ == "__main__":
    main()
