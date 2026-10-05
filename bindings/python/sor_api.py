#!/usr/bin/env python3
"""Typed Python client for the SOR command-line interface.

This remains a process boundary, but provides collision-free outputs, bounded
execution, cooperative cancellation, and stable typed results.
"""
from __future__ import annotations

import code
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "demo_out"
PRESETS = {
    "blend": ROOT / "examples/crude_blending/blend_s42.mps",
    "dispatch": ROOT / "examples/dispatch/dispatch_s42.qps",
    "schedule": ROOT / "examples/scheduling/schedule_s42.mps",
    "sparse": ROOT / "examples/sparse500.mps",
    "testlp": ROOT / "examples/testlp.mps",
}


class _ResultMapping(Mapping[str, Any]):
    """Compatibility bridge for callers that previously consumed dicts."""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
        return result

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class SolveResult(_ResultMapping):
    ok: bool
    returncode: int
    command: tuple[str, ...]
    model: Path
    solution: Path
    status: str | None = None
    proof_level: str | None = None
    objective: float | None = None
    engine: str | None = None
    backend: str | None = None
    termination: str | None = None
    nodes: int | None = None
    iterations: int | None = None
    mip_gap: float | None = None
    dual_bound: float | None = None
    timed_out: bool = False
    cancelled: bool = False
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class CheckResult(_ResultMapping):
    ok: bool
    verified: bool
    returncode: int
    command: tuple[str, ...]
    timed_out: bool = False
    cancelled: bool = False
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class _ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False


def _binary_dir(binary_dir: str | Path | None = None) -> Path:
    if binary_dir is not None:
        candidate = Path(binary_dir)
        if (candidate / "sor_solve").is_file() and \
                (candidate / "sor_check").is_file():
            return candidate
        raise FileNotFoundError(f"SOR binaries not found in {candidate}")
    candidates = []
    env_dir = os.environ.get("SOR_BINARY_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend((
        ROOT / "build",
        ROOT / "build-native",
        ROOT / "build-integration",
    ))
    for candidate in candidates:
        if (candidate / "sor_solve").is_file() and \
                (candidate / "sor_check").is_file():
            return candidate
    raise FileNotFoundError(
        "build SOR first (cmake -S . -B build && cmake --build build -j)")


def _model_path(model: str | Path) -> Path:
    path = PRESETS.get(str(model), Path(model))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _isolated_solution_path(model: Path) -> Path:
    OUT.mkdir(exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"sor-{model.stem}-", dir=OUT))
    return directory / "solution.sol"


def _run_process(argv: list[str], timeout_s: float | None,
                 cancel_event: threading.Event | None) -> _ProcessResult:
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("process_timeout must be positive")
    proc = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while True:
        if cancel_event is not None and cancel_event.is_set():
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            return _ProcessResult(proc.returncode, stdout, stderr,
                                  cancelled=True)
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            proc.kill()
            stdout, stderr = proc.communicate()
            return _ProcessResult(proc.returncode, stdout, stderr,
                                  timed_out=True)
        try:
            stdout, stderr = proc.communicate(
                timeout=0.1 if remaining is None else min(0.1, remaining))
            return _ProcessResult(proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            continue


def _parse_scalar(value: str, kind: type) -> Any:
    token = value.split()[0] if value.split() else ""
    try:
        return kind(token)
    except (ValueError, TypeError):
        return None


def solve(model: str | Path, *, engine: str = "simplex", backend: str = "cpu",
          method: str | None = "auto", time_limit: float | None = None,
          max_iter: int | None = None, verbose: bool = False,
          solution_out: str | Path | None = None,
          process_timeout: float | None = None,
          cancel_event: threading.Event | None = None,
          binary_dir: str | Path | None = None) -> SolveResult:
    """Solve one model and return typed fields.

    ``process_timeout`` bounds loading, solving, and result writing.
    ``cancel_event`` may be set from another thread to stop the child.
    """
    model_path = _model_path(model)
    bin_dir = _binary_dir(binary_dir)
    output = (Path(solution_out).resolve() if solution_out is not None
              else _isolated_solution_path(model_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(bin_dir / "sor_solve"), str(model_path), "--engine", engine,
            "--backend", backend, "--solution-out", str(output)]
    if method and engine in ("simplex", "milp"):
        argv += ["--method", method]
    if time_limit is not None:
        argv += ["--time-limit", str(time_limit)]
    if max_iter is not None:
        argv += ["--max-iter", str(max_iter)]
    if verbose:
        argv.append("--verbose")
    effective_timeout = (process_timeout if process_timeout is not None else
                         (time_limit + 30.0 if time_limit is not None else None))
    proc = _run_process(argv, effective_timeout, cancel_event)
    fields: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            fields[key.strip().lower().replace(" ", "_")] = value.strip()
    return SolveResult(
        ok=(proc.returncode == 0 and not proc.timed_out and not proc.cancelled),
        returncode=proc.returncode, command=tuple(argv), model=model_path,
        solution=output, status=fields.get("status"),
        proof_level=fields.get("proof_level"),
        objective=_parse_scalar(fields.get("objective", ""), float),
        engine=fields.get("engine"), backend=fields.get("backend"),
        termination=fields.get("termination"),
        nodes=_parse_scalar(fields.get("nodes", ""), int),
        iterations=_parse_scalar(fields.get("iterations", ""), int),
        mip_gap=_parse_scalar(fields.get("mip_gap", ""), float),
        dual_bound=_parse_scalar(fields.get("dual_bound", ""), float),
        timed_out=proc.timed_out, cancelled=proc.cancelled,
        stdout=proc.stdout, stderr=proc.stderr)


def check(model: str | Path, solution: str | Path, *, tol: float = 1e-7,
          process_timeout: float | None = 60.0,
          cancel_event: threading.Event | None = None,
          binary_dir: str | Path | None = None) -> CheckResult:
    model_path = _model_path(model)
    solution_path = Path(solution).resolve()
    if not solution_path.is_file():
        raise FileNotFoundError(solution_path)
    argv = [str(_binary_dir(binary_dir) / "sor_check"), str(model_path),
            str(solution_path), "--tol", str(tol)]
    proc = _run_process(argv, process_timeout, cancel_event)
    verified = proc.returncode == 0 and "VERIFIED" in proc.stdout
    return CheckResult(ok=verified and not proc.timed_out and not proc.cancelled,
                       verified=verified, returncode=proc.returncode,
                       command=tuple(argv), timed_out=proc.timed_out,
                       cancelled=proc.cancelled, stdout=proc.stdout,
                       stderr=proc.stderr)


def show(result: _ResultMapping) -> None:
    values = result.to_dict()
    print("command:", " ".join(values.get("command", [])))
    for key in ("status", "proof_level", "objective", "engine", "backend",
                "nodes", "iterations", "verified", "timed_out", "cancelled"):
        if values.get(key) is not None:
            print(f"  {key}: {values[key]}")
    if values.get("stderr"):
        print("stderr:", str(values["stderr"])[:400])


HELP = """
SOR Python API / REPL

  r = solve("blend", time_limit=30)
  show(r)
  show(check("blend", r.solution))
  cancel = threading.Event()
  solve("sparse", engine="hpr", time_limit=30, cancel_event=cancel)

  presets: blend | dispatch | schedule | sparse | testlp
""".strip()


def help_sor() -> None:
    print(HELP)


def run_repl() -> None:
    print(HELP)
    print(f"binaries: {_binary_dir()}")
    code.interact(banner="", local={"solve": solve, "check": check,
                  "show": show, "help_sor": help_sor, "PRESETS": PRESETS,
                  "OUT": OUT, "json": json, "threading": threading})


def run_one_shot() -> None:
    result = solve("blend")
    show(result)
    show(check("blend", result.solution))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one-shot":
        run_one_shot()
    else:
        run_repl()
