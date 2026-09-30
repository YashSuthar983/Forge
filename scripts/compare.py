#!/usr/bin/env python3
"""Side-by-side solver comparison.

One tool for every comparison this project makes: give it some solvers and some
models, and it runs each solver on each model and prints the results next to
each other.

    # every SOR engine on one model
    scripts/compare.py examples/testlp.mps --solvers sor:simplex,sor:pdhg,sor:hpr

    # SOR against HiGHS on a whole suite
    scripts/compare.py benchmarks/netlib/mps --solvers sor:simplex,highs

    # a MILP, with a time limit and machine-readable output
    scripts/compare.py benchmarks/miplib-easy/mps --solvers sor:milp,highs \\
        --time-limit 30 --jsonl results.jsonl

Solver names are `sor:<engine>[:<backend>]` for this project, or one of the
external baselines: highs, cbc, scipy, scipy-ipm, gurobi. Externals run only if
their Python package is importable; they are reported as `unavailable`
otherwise, never silently dropped.

CLEAN ROOM: external solvers are used ONLY as independent reference points for
these measurements. They are never linked into SOR and never contribute to a
result SOR reports. See docs/architecture.md §8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, replace, field
from pathlib import Path

# Depth-independent: walk up to whichever directory holds CMakeLists.txt, so
# this keeps working if the script is ever moved.
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "CMakeLists.txt").exists())

# SOR prints a human report; these pull the numbers back out of it.
_PAT = {
    "size":    re.compile(r"rows x cols:\s+(\d+) x (\d+)\s+nnz (\d+)"),
    "status":  re.compile(r"^status:\s+(\S+)", re.M),
    "proof":   re.compile(r"^proof_level:\s+(\S+)", re.M),
    "obj":     re.compile(r"^objective:\s+(\S+)", re.M),
    "dual":    re.compile(r"^dual bound:\s+(\S+)", re.M),
    "viol":    re.compile(r"^max (?:primal viol|row violation):\s+(\S+)", re.M),
    "iters":   re.compile(r"^iterations:\s+(\d+)", re.M),
    "solve_ms": re.compile(r"^  total\s+(\S+)", re.M),
    "stage_iters": re.compile(
        r"^(?:iterations:\s+\d+\s+\(|stage iterations:\s+)"
        r"FO\s+(\d+),\s+crossover\s+(\d+),\s+simplex\s+(\d+)\)?", re.M),
    "stage_ms": re.compile(
        r"^stage time \(ms\):\s+FO\s+(\S+),\s+crossover\s+(\S+),\s+simplex\s+(\S+)",
        re.M),
    "crossover": re.compile(
        r"^crossover:\s+attempted\s+(yes|no),\s+basis valid\s+(yes|no),\s+"
        r"cold fallback\s+(yes|no)", re.M),
    "checker_validation": re.compile(r"^validation:\s+(\S+)\s*$", re.M),
    "checker_verified": re.compile(r"^VERIFIED\s*$", re.M),
    "bab_threads": re.compile(r"^bab threads:\s+(\d+)\s*$", re.M),
    "mip_gap_tol": re.compile(r"^mip gap tolerance:\s+(\S+)\s*$", re.M),
}


@dataclass
class Result:
    solver: str
    instance: str
    status: str = "notrun"
    objective: float | None = None
    # The MILP suites' half of gate rule 1 needs this: a dual bound past the
    # published optimum on the bounding side means the search pruned a region
    # that contained the optimum, which no proof or incumbent check can see.
    dual_bound: float | None = None
    seconds: float | None = None      # solver-internal where available
    wall_s: float | None = None       # full process wall
    iterations: int | None = None
    fo_iterations: int | None = None
    crossover_iterations: int | None = None
    simplex_iterations: int | None = None
    fo_seconds: float | None = None
    crossover_seconds: float | None = None
    simplex_seconds: float | None = None
    crossover_attempted: bool | None = None
    crossover_basis_valid: bool | None = None
    crossover_cold_fallback: bool | None = None
    proof: str | None = None
    violation: float | None = None
    rows: int | None = None
    cols: int | None = None
    nnz: int | None = None
    error: str | None = None
    repetition: int | None = None
    samples_s: list[float] | None = None
    wall_samples_s: list[float] | None = None
    # Dispersion of the measured repetitions. A median with no dispersion
    # beside it cannot be audited: `mad_rel` above the protocol's 5% band is
    # the signal that a timing is host noise rather than a solver difference,
    # and the protocol says to rerun or reject the session rather than publish.
    median_s: float | None = None
    # Median of the process-wall samples, aggregated independently of the
    # solver-time median. See aggregate_repetitions().
    median_wall_s: float | None = None
    mad_s: float | None = None
    mad_rel: float | None = None
    noisy: bool | None = None
    # Claim runs retain the exact command and the exact solution emitted by a
    # measured repetition.  The independent checker consumes this file; it
    # must never obtain a fresh solution from a differently configured solve.
    command: list[str] | None = None
    configuration: dict | None = None
    solution_file: str | None = None
    solution_sha256: str | None = None
    model_sha256: str | None = None
    checker_command: list[str] | None = None
    checker_verified: bool | None = None
    checker_error: str | None = None
    checker_validation_scope: str | None = None
    checker_executable_sha256: str | None = None
    checker_returncode: int | None = None
    checker_stdout: str | None = None
    checker_stderr: str | None = None
    checker_timed_out: bool | None = None
    solver_returncode: int | None = None
    solver_stdout: str | None = None
    solver_stderr: str | None = None
    started_utc: str | None = None
    executable_sha256: str | None = None
    launch_affinity: list[int] | None = None
    effective_bab_threads: int | None = None
    mip_gap_tolerance: float | None = None
    reference_model_path: str | None = None
    reference_model_sha256: str | None = None
    # External-oracle identity is carried on every row.  In particular, the
    # source lane is valid only when this describes the native API runner and
    # the source tree/build it was linked against.
    solver_version: str | None = None
    build_identity: dict | None = None
    # Rerun selection is explicit in the artifact.  A superseded noisy
    # aggregate remains auditable but must not be selected by a loader.
    claim_selected: bool | None = None
    claim_superseded: bool | None = None
    claim_source: str | None = None


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_executable(path: Path, expected: str) -> None:
    """Abort a campaign if a solver binary changes during measurement."""
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"executable changed: {path}: {expected} -> {actual}")


def freeze_sor_binaries(exe: Path, destination: Path) -> tuple[Path, str]:
    """Keep the measured solver and checker beside the result artifact."""
    source_hash = sha256_file(exe)
    checker = exe.with_name("sor_check")
    checker_hash = sha256_file(checker)
    if source_hash is None or checker_hash is None:
        raise RuntimeError("SOR solver and checker must both exist before a campaign")
    destination.mkdir(parents=True, exist_ok=True)
    for source, expected in ((exe, source_hash), (checker, checker_hash)):
        target = destination / source.name
        if target.exists():
            verify_executable(target, expected)
        else:
            shutil.copy2(source, target)
            verify_executable(target, expected)
        verify_executable(source, expected)
    return destination / exe.name, source_hash


def _num(text: str, key: str) -> float | None:
    m = _PAT[key].search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# SOR
# --------------------------------------------------------------------------
def affined_command(cmd: list[str], cpu: int | None) -> list[str]:
    """Return a command pinned to one CPU when taskset is available."""
    if cpu is None:
        return cmd
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("--cpu requested but taskset is not available")
    return [taskset, "-c", str(cpu), *cmd]


def build_sor_command(model: Path, engine: str, backend: str,
                      time_limit: float, tol: float, exe: Path,
                      method: str | None, basis_update: str,
                      max_iter: int | None,
                      pricing: str = "choose",
                      dual_cost_perturbation: float = 0.0,
                      sor_extra: list[str] | None = None) -> list[str]:
    """Build the exact CLI command for one SOR solver specification.

    `sor_extra` is appended verbatim for SOR solvers. It exists so a tuning
    parameter can be swept and gated without this file growing a mirror of
    sor_solve's whole CLI; whatever is passed is recorded in the environment
    record, so a JSONL still says exactly what produced it."""
    flags: list[str] = []
    eng = engine
    # primal/dual are simplex methods, not standalone engines.
    if engine in ("auto", "primal", "dual"):
        eng, method = "simplex", engine
    # Two HPR variants are flags on the same engine rather than engines.
    if engine in ("hpr-full", "hpr-vanilla"):
        eng, flags = "hpr", [f"--{engine.replace('hpr-', 'hpr-')}"]
    cmd = [str(exe), str(model), "--engine", eng, "--backend", backend,
           "--tol", str(tol), "--time-limit", str(time_limit), *flags]
    if method is not None and eng in ("simplex", "milp"):
        cmd += ["--method", method]
    if eng in ("simplex", "milp"):
        # "default" means: do not pass the flag at all, so the run measures
        # whatever the engine ships as its default basis representation. The
        # harness used to hard-code "product" here, which silently pinned every
        # gate run to product form and would have hidden the 2026-09-17 switch
        # of the LP default to Forrest-Tomlin. Pass product/ft explicitly to
        # force one.
        if basis_update not in (None, "default"):
            cmd += ["--basis-update", basis_update]
        cmd += ["--pricing", pricing]
        if dual_cost_perturbation > 0.0:
            cmd += ["--dual-cost-perturbation",
                    str(dual_cost_perturbation)]
    if max_iter is not None:
        cmd += ["--max-iter", str(max_iter)]
    if sor_extra:
        cmd += list(sor_extra)
    return cmd


def run_solver_process(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Capture a solver and kill its whole process group at the hard limit.

    ``subprocess.run`` kills only its immediate child on timeout. If a solver
    has descendants holding the captured pipes open, its cleanup may wait
    well past the requested limit. A new session gives this run a dedicated
    process group and bounds the cleanup wait too.
    """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True)
    try:
        stdout, stderr = p.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, p.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            # A child stuck in uninterruptible kernel I/O must not hold the
            # harness forever. The group has received SIGKILL; report timeout.
            if p.stdout is not None:
                p.stdout.close()
            if p.stderr is not None:
                p.stderr.close()
            try:
                p.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        raise


def run_sor(model: Path, engine: str, backend: str, time_limit: float,
            tol: float, exe: Path, label: str, method: str | None = None,
            basis_update: str = "default", max_iter: int | None = None,
            cpu: int | None = None, pricing: str = "choose",
            dual_cost_perturbation: float = 0.0,
            sor_extra: list[str] | None = None,
            solution_out: Path | None = None) -> Result:
    r = Result(solver=label, instance=model.name)
    r.started_utc = datetime.now(timezone.utc).isoformat()
    r.executable_sha256 = sha256_file(exe)
    r.launch_affinity = ([cpu] if cpu is not None else
                         sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    try:
        extra = list(sor_extra or [])
        if engine == "milp" and cpu is not None and not any(
                arg == "--bab-threads" or arg.startswith("--bab-threads=")
                for arg in extra):
            # Affinity limits *where* workers run, not how many SOR starts.
            # SOR's auto setting can select eight Para-B&B workers inside a
            # one-CPU taskset, while the reference runs with one thread.
            extra += ["--bab-threads", "1"]
        base_cmd = build_sor_command(
            model, engine, backend, time_limit, tol, exe, method,
            basis_update, max_iter, pricing, dual_cost_perturbation, extra)
        if solution_out is not None:
            solution_out.parent.mkdir(parents=True, exist_ok=True)
            if solution_out.exists():
                r.status, r.error = "error", (
                    f"solution output {solution_out} already exists")
                return r
            base_cmd += ["--solution-out", str(solution_out)]
        cmd = affined_command(base_cmd, cpu)
        r.command = list(cmd)
    except RuntimeError as e:
        r.status, r.error = "error", str(e)
        return r
    t0 = time.perf_counter()
    try:
        p = run_solver_process(cmd, time_limit + 30)
        r.solver_returncode = p.returncode
        r.solver_stdout = p.stdout
        r.solver_stderr = p.stderr
        if r.executable_sha256 is not None:
            verify_executable(exe, r.executable_sha256)
        r.wall_s = time.perf_counter() - t0
        out = p.stdout
        m = _PAT["status"].search(out)
        r.status = m.group(1) if m else "unparsed"
        m = _PAT["proof"].search(out)
        r.proof = m.group(1) if m else None
        m = _PAT["bab_threads"].search(out)
        r.effective_bab_threads = int(m.group(1)) if m else None
        r.mip_gap_tolerance = _num(out, "mip_gap_tol")
        r.objective = _num(out, "obj")
        r.dual_bound = _num(out, "dual")
        r.violation = _num(out, "viol")
        it = _num(out, "iters")
        r.iterations = int(it) if it is not None else None
        stage_iters = _PAT["stage_iters"].search(out)
        if stage_iters:
            (r.fo_iterations, r.crossover_iterations,
             r.simplex_iterations) = map(int, stage_iters.groups())
        stage_ms = _PAT["stage_ms"].search(out)
        if stage_ms:
            (r.fo_seconds, r.crossover_seconds,
             r.simplex_seconds) = (float(x) / 1000.0 for x in stage_ms.groups())
        crossover = _PAT["crossover"].search(out)
        if crossover:
            (r.crossover_attempted, r.crossover_basis_valid,
             r.crossover_cold_fallback) = (
                value == "yes" for value in crossover.groups())
        ms = _num(out, "solve_ms")
        # Compare solver-internal time to solver-internal time; keep the full
        # process wall separately so startup and file I/O stay visible.
        r.seconds = ms / 1000.0 if ms is not None else r.wall_s
        sz = _PAT["size"].search(out)
        if sz:
            r.rows, r.cols, r.nnz = int(sz[1]), int(sz[2]), int(sz[3])
        if r.status == "unparsed":
            r.error = ((p.stderr or "") + out)[:200]
        if solution_out is not None:
            if solution_out.is_file():
                r.solution_file = str(solution_out.resolve())
                r.solution_sha256 = sha256_file(solution_out)
                checker = exe.with_name("sor_check")
                check_cmd = [str(checker), str(model), str(solution_out),
                             "--tol", str(tol)]
                if "--relax-integrality" in base_cmd:
                    check_cmd.append("--relax-integrality")
                if "--small-matrix-value" in base_cmd:
                    pos = base_cmd.index("--small-matrix-value")
                    check_cmd += ["--small-matrix-value", base_cmd[pos + 1]]
                r.checker_command = affined_command(check_cmd, cpu)
                r.checker_executable_sha256 = sha256_file(checker)
                if not checker.is_file():
                    r.checker_verified = False
                    r.checker_timed_out = False
                    r.checker_error = f"independent checker not found: {checker}"
                else:
                    try:
                        checked = subprocess.run(
                            r.checker_command, capture_output=True, text=True,
                            timeout=time_limit + 30)
                        r.checker_returncode = checked.returncode
                        r.checker_stdout = checked.stdout
                        r.checker_stderr = checked.stderr
                        r.checker_timed_out = False
                        scope = _PAT["checker_validation"].search(checked.stdout)
                        r.checker_validation_scope = (
                            scope.group(1) if scope else None)
                        r.checker_verified = bool(
                            checked.returncode == 0 and scope is not None and
                            _PAT["checker_verified"].search(checked.stdout))
                        if not r.checker_verified:
                            r.checker_error = "checker rejected or output was unparseable"
                    except subprocess.TimeoutExpired as e:
                        r.checker_timed_out = True
                        r.checker_verified = False
                        r.checker_stdout = (e.stdout or "") if isinstance(
                            e.stdout, str) else (e.stdout or b"").decode(errors="replace")
                        r.checker_stderr = (e.stderr or "") if isinstance(
                            e.stderr, str) else (e.stderr or b"").decode(errors="replace")
                        r.checker_error = "independent checker timed out"
                    except Exception as e:  # noqa: BLE001
                        r.checker_timed_out = False
                        r.checker_verified = False
                        r.checker_error = f"checker execution failed: {type(e).__name__}: {e}"
            else:
                r.checker_verified = False
                r.checker_timed_out = False
                r.checker_error = "measured solve did not write its requested solution"
                r.error = (r.error + "; " if r.error else "") + r.checker_error
    except subprocess.TimeoutExpired:
        r.wall_s = r.seconds = time.perf_counter() - t0
        r.status = "timeout"
    except Exception as e:  # noqa: BLE001
        r.wall_s = r.seconds = time.perf_counter() - t0
        r.status = "error"
        r.error = f"{type(e).__name__}: {e}"
    return r


# --------------------------------------------------------------------------
# External baselines - separate processes / independent packages only.
# --------------------------------------------------------------------------
def _set_option(h, name: str, value) -> None:
    """Set a HiGHS option, recording rather than swallowing a rejection.

    An option this HiGHS build does not recognise must not silently leave the
    baseline running a different configuration from the one the report claims.
    """
    try:
        status = h.setOptionValue(name, value)
    except Exception as e:  # noqa: BLE001
        _HIGHS_OPTION_NOTES.append(f"{name}: {type(e).__name__}: {e}")
        return
    if status is not None and "kOk" not in str(status) and str(status) != "0":
        _HIGHS_OPTION_NOTES.append(f"{name}: {status}")


_HIGHS_OPTION_NOTES: list[str] = []



def highs_worker(model: Path, time_limit: float, tol: float = 1e-7,
                 relax_integrality: bool = False,
                 small_matrix_value: float | None = None, seed: int = 0) -> int:
    """Isolated HiGHS worker ending its stdout with one JSON Result record.

    Every option here is part of the benchmark contract rather than a
    preference. The tolerance in particular: this worker used to run at
    HiGHS's defaults while SOR ran at whatever --tol asked for, so a sweep at
    1e-6 was comparing a loose SOR against a tight HiGHS and calling the
    difference performance.
    """
    _HIGHS_OPTION_NOTES.clear()
    r = Result(solver="highs", instance=model.name)
    r.started_utc = datetime.now(timezone.utc).isoformat()
    r.launch_affinity = (sorted(os.sched_getaffinity(0))
                         if hasattr(os, "sched_getaffinity") else None)
    try:
        import highspy
    except ImportError:
        r.status = "unavailable"
        r.error = "highspy not installed (see benchmarks/requirements-baseline.txt)"
        print(json.dumps(asdict(r)))
        return 0
    try:
        h = highspy.Highs()
        r.configuration = {
            "output_flag": False, "log_to_console": False,
            "threads": 1, "parallel": "off", "time_limit": float(time_limit),
            "primal_feasibility_tolerance": float(tol),
            "dual_feasibility_tolerance": float(tol),
            "random_seed": int(seed), "solver": "choose", "presolve": "choose",
            "run_crossover": "on", "solve_relaxation": bool(relax_integrality),
            "small_matrix_value": small_matrix_value,
        }
        _set_option(h, "output_flag", False)
        _set_option(h, "threads", 1)
        _set_option(h, "parallel", "off")
        _set_option(h, "time_limit", float(time_limit))
        # HiGHS otherwise drops Highs.log into the working directory.
        _set_option(h, "log_to_console", False)
        # Section 3.3: the same tolerance, seed, matrix-small-value policy and
        # solver/presolve choice on both sides, all recorded.
        for name in ("primal_feasibility_tolerance",
                     "dual_feasibility_tolerance"):
            _set_option(h, name, float(tol))
        if small_matrix_value is not None:
            _set_option(h, "small_matrix_value", float(small_matrix_value))
        _set_option(h, "random_seed", int(seed))
        _set_option(h, "solver", "choose")
        _set_option(h, "presolve", "choose")
        _set_option(h, "run_crossover", "on")
        if relax_integrality:
            _set_option(h, "solve_relaxation", True)
        # HiGHS dispatches its MPS reader by filename suffix. QPS uses the
        # MPS grammar with a quadratic section, but .qps is not accepted by
        # this reader. A byte-identical .mps alias preserves the model.
        alias_context = tempfile.TemporaryDirectory(prefix="sor-highs-qps-")
        try:
            read_path = model
            if model.suffix.lower() == ".qps":
                read_path = Path(alias_context.name) / (model.stem + ".mps")
                shutil.copyfile(model, read_path)
                source_hash = sha256_file(model)
                if source_hash != sha256_file(read_path):
                    raise RuntimeError("QPS reference alias differs from source bytes")
                r.reference_model_path = str(read_path)
                r.reference_model_sha256 = source_hash
            read_status = h.readModel(str(read_path))
        finally:
            alias_context.cleanup()
        if "kOk" not in str(read_status) and str(read_status) != "0":
            raise RuntimeError(f"readModel rejected model: {read_status}")
        run_time_before = float(h.getRunTime())
        t0 = time.perf_counter()
        h.run()
        r.wall_s = time.perf_counter() - t0
        r.seconds = float(h.getRunTime()) - run_time_before
        r.status = str(h.getModelStatus()).replace("HighsModelStatus.k", "")
        r.objective = float(h.getObjectiveValue())
        r.solver_version = str(h.version())
        r.build_identity = {
            "kind": "official-highspy-wheel",
            "runtime_version": r.solver_version,
            "module": str(getattr(highspy, "__file__", "")),
        }
        try:
            r.iterations = int(h.getInfo().simplex_iteration_count)
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001 - a baseline crash must not kill the sweep
        r.status = "crash"
        r.error = f"{type(e).__name__}: {e}"
    if _HIGHS_OPTION_NOTES and not r.error:
        r.error = "options not applied: " + "; ".join(_HIGHS_OPTION_NOTES)
    print(json.dumps(asdict(r)))
    return 0


def run_highs(model: Path, time_limit: float, label: str,
              cpu: int | None = None, tol: float = 1e-7,
              relax_integrality: bool = False,
              small_matrix_value: float | None = None, seed: int = 0) -> Result:
    """Run HiGHS in a fresh process, matching SOR's process isolation."""
    r = Result(solver=label, instance=model.name)
    r.started_utc = datetime.now(timezone.utc).isoformat()
    r.launch_affinity = ([cpu] if cpu is not None else
                         sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_highs-worker",
           str(model), str(time_limit), str(tol),
           "1" if relax_integrality else "0",
           "" if small_matrix_value is None else repr(small_matrix_value),
           str(seed)]
    r.command = list(cmd)
    try:
        cmd = affined_command(cmd, cpu)
        r.command = list(cmd)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=time_limit + 30)
        outer_wall = time.perf_counter() - t0
        worker_returncode = p.returncode
        worker_stdout = p.stdout
        worker_stderr = p.stderr
        payload = None
        # With console logging enabled to match the source CLI lane, HiGHS
        # writes its normal report before (or, depending on C stdio flushing,
        # around) our JSON record. Locate the worker record explicitly.
        for line in reversed(p.stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("solver") == "highs":
                payload = candidate
                break
        if payload is None:
            raise ValueError("HiGHS wheel worker emitted no result record")
        r = Result(**payload)
        r.solver = label
        r.wall_s = outer_wall
        r.command = list(cmd)
        r.solver_returncode = worker_returncode
        r.solver_stdout = worker_stdout
        r.solver_stderr = worker_stderr
        if p.returncode != 0 and r.status not in ("crash", "unavailable"):
            r.status = "crash"
            r.error = (p.stderr or f"worker exited {p.returncode}")[:200]
    except subprocess.TimeoutExpired:
        r.status, r.seconds, r.wall_s = "timeout", time_limit, time_limit
    except Exception as e:  # noqa: BLE001
        r.status = "crash"
        r.error = f"{type(e).__name__}: {e}"
    return r


def run_highs_source(model: Path, time_limit: float, label: str,
                     executable: Path | None, cpu: int | None = None,
                     tol: float = 1e-7, relax_integrality: bool = False,
                     small_matrix_value: float | None = None,
                     seed: int = 0) -> Result:
    """Run the native source-HiGHS API runner.

    The old implementation parsed the human CLI's two-decimal ``HiGHS run
    time`` line.  Sub-centisecond models consequently acquired a primary
    reference time of exactly zero.  Public source timings now have one
    accepted representation: the versioned JSON contract emitted by
    tools/highs_source_runner, whose solve interval uses Highs::getRunTime().
    """
    r = Result(solver=label, instance=model.name)
    if executable is None:
        r.status, r.error = "unavailable", "no --highs-source executable"
        return r
    r.configuration = {
        "output_flag": False, "log_to_console": False,
        "threads": 1, "parallel": "off", "time_limit": float(time_limit),
        "primal_feasibility_tolerance": float(tol),
        "dual_feasibility_tolerance": float(tol), "random_seed": int(seed),
        "solver": "choose", "presolve": "choose", "run_crossover": "on",
        "solve_relaxation": bool(relax_integrality),
        "small_matrix_value": small_matrix_value,
    }
    try:
        cmd = [str(executable), "--model", str(model),
               "--time-limit", repr(float(time_limit)),
               "--tol", repr(float(tol)), "--seed", str(int(seed))]
        if relax_integrality:
            cmd.append("--relax-integrality")
        if small_matrix_value is not None:
            cmd += ["--small-matrix-value", repr(float(small_matrix_value))]
        cmd = affined_command(cmd, cpu)
        r.command = list(cmd)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=time_limit + 30)
        r.wall_s = time.perf_counter() - t0
        lines = [line for line in p.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(
                "native source runner must emit exactly one JSON record; "
                "human-readable HiGHS CLI output is not timing evidence")
        payload = json.loads(lines[0])
        if not isinstance(payload, dict) or \
                payload.get("schema") != "sor-highs-source-runner-v1" or \
                payload.get("kind") != "solve_result":
            raise ValueError("source executable did not emit the native runner schema")
        if payload.get("read_status") != "kOk" or \
                payload.get("run_status") != "kOk" or \
                payload.get("options_applied") is not True:
            raise ValueError("source runner rejected the model, solve, or an option")
        if payload.get("configuration") != r.configuration:
            raise ValueError("source runner configuration disagrees with requested options")
        seconds = payload.get("solve_seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(
                "source runner solve_seconds must be finite and positive; "
                "a rounded CLI value such as 0.00 is never valid")
        objective = payload.get("objective")
        if type(objective) not in (int, float) or not math.isfinite(objective):
            raise ValueError("source runner objective is not finite")
        identity = payload.get("build_identity")
        version = payload.get("highs_version")
        if not isinstance(identity, dict) or identity.get("kind") != \
                "sor-highs-source-api-runner" or not isinstance(version, str):
            raise ValueError("source runner omitted versioned build identity")
        r.status = str(payload.get("model_status") or "unparsed").replace(" ", "")
        r.objective = float(objective)
        r.seconds = float(seconds)
        r.iterations = (int(payload["simplex_iterations"])
                        if type(payload.get("simplex_iterations")) is int else None)
        r.solver_version = version
        r.build_identity = dict(identity)
        r.build_identity["executable_sha256"] = sha256_file(executable)
        if p.returncode != 0:
            r.status = "crash"
            r.error = f"source runner exited {p.returncode}: {(p.stderr or '')[:200]}"
        elif p.stderr.strip():
            r.error = "source runner emitted unexpected stderr: " + p.stderr.strip()[:200]
    except subprocess.TimeoutExpired:
        r.status, r.seconds, r.wall_s = "timeout", time_limit, time_limit
    except Exception as e:  # noqa: BLE001
        r.status = "crash"
        r.error = f"{type(e).__name__}: {e}"
    return r


def run_scipy(model: Path, time_limit: float, label: str, method: str) -> Result:
    r = Result(solver=label, instance=model.name)
    try:
        import pulp
        from scipy.optimize import linprog
        import numpy as np
    except ImportError as e:
        r.status = "unavailable"
        r.error = f"{e.name} not installed"
        return r
    try:
        _, lp = pulp.LpProblem.fromMPS(str(model))
        cols = lp.variables()
        idx = {v.name: i for i, v in enumerate(cols)}
        c = np.zeros(len(cols))
        for v, coef in lp.objective.items():
            c[idx[v.name]] = float(coef)
        Aub, bub, Aeq, beq = [], [], [], []
        for con in lp.constraints.values():
            row = np.zeros(len(cols))
            for v, coef in con.items():
                row[idx[v.name]] = float(coef)
            rhs = -float(con.constant)
            if con.sense == pulp.LpConstraintEQ:
                Aeq.append(row); beq.append(rhs)
            elif con.sense == pulp.LpConstraintLE:
                Aub.append(row); bub.append(rhs)
            else:
                Aub.append(-row); bub.append(-rhs)
        bounds = [(v.lowBound, v.upBound) for v in cols]
        t0 = time.perf_counter()
        res = linprog(c, A_ub=Aub or None, b_ub=bub or None,
                      A_eq=Aeq or None, b_eq=beq or None,
                      bounds=bounds, method=method)
        r.seconds = r.wall_s = time.perf_counter() - t0
        r.status = "Optimal" if res.success else str(res.message)[:40]
        r.objective = float(res.fun) if res.fun is not None else None
    except Exception as e:  # noqa: BLE001
        r.status = "crash"
        r.error = f"{type(e).__name__}: {e}"
    return r


def run_cbc(model: Path, time_limit: float, label: str) -> Result:
    r = Result(solver=label, instance=model.name)
    try:
        import pulp
    except ImportError:
        r.status = "unavailable"
        r.error = "pulp not installed"
        return r
    try:
        _, lp = pulp.LpProblem.fromMPS(str(model))
        t0 = time.perf_counter()
        lp.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
        r.seconds = r.wall_s = time.perf_counter() - t0
        r.status = pulp.LpStatus[lp.status]
        r.objective = float(pulp.value(lp.objective))
    except Exception as e:  # noqa: BLE001
        r.status = "crash"
        r.error = f"{type(e).__name__}: {e}"
    return r


def run_gurobi(model: Path, time_limit: float, label: str) -> Result:
    r = Result(solver=label, instance=model.name)
    try:
        import gurobipy as gp
    except ImportError:
        r.status = "unavailable"
        r.error = "gurobipy not installed / no licence"
        return r
    try:
        env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
        m = gp.read(str(model), env)
        m.setParam("TimeLimit", float(time_limit)); m.setParam("Threads", 1)
        t0 = time.perf_counter()
        m.optimize()
        r.seconds = r.wall_s = time.perf_counter() - t0
        r.status = {2: "Optimal", 3: "Infeasible", 5: "Unbounded",
                    9: "TimeLimit"}.get(m.Status, f"code{m.Status}")
        if m.SolCount:
            r.objective = float(m.ObjVal)
    except Exception as e:  # noqa: BLE001
        r.status = "crash"
        r.error = f"{type(e).__name__}: {e}"
    return r


def dispatch(spec: str, model: Path, time_limit: float, tol: float,
             exe: Path, method: str | None = None,
             basis_update: str = "default", max_iter: int | None = None,
             cpu: int | None = None, pricing: str = "choose",
             dual_cost_perturbation: float = 0.0,
             sor_extra: list[str] | None = None,
             relax_integrality: bool = False,
             small_matrix_value: float | None = None,
             seed: int = 0, highs_source: Path | None = None,
             solution_out: Path | None = None) -> Result:
    """`sor:<engine>[:<backend>]`, or an external baseline name."""
    key = spec.strip().lower()
    if key.startswith("sor"):
        parts = key.split(":")
        engine = parts[1] if len(parts) > 1 else "simplex"
        backend = parts[2] if len(parts) > 2 else "cpu"
        extra = list(sor_extra or [])
        # Handed to Agent 2 as a required CLI flag; the option field already
        # exists as io::MpsReadOptions::relax_integrality.
        if relax_integrality:
            extra.append("--relax-integrality")
        if small_matrix_value is not None:
            extra += ["--small-matrix-value", repr(small_matrix_value)]
        return run_sor(model, engine, backend, time_limit, tol, exe, spec,
                       method, basis_update, max_iter, cpu, pricing,
                       dual_cost_perturbation, extra, solution_out)
    if key in ("highs", "highs-wheel"):
        return run_highs(model, time_limit, spec, cpu, tol, relax_integrality,
                         small_matrix_value, seed)
    if key == "highs-source":
        return run_highs_source(model, time_limit, spec, highs_source, cpu, tol,
                                relax_integrality, small_matrix_value, seed)
    if key in ("scipy", "scipy-hi", "scipy-highs"):
        return run_scipy(model, time_limit, spec, "highs")
    if key in ("scipy-ipm", "scipy-interior"):
        return run_scipy(model, time_limit, spec, "highs-ipm")
    if key == "cbc":
        return run_cbc(model, time_limit, spec)
    if key == "gurobi":
        return run_gurobi(model, time_limit, spec)
    r = Result(solver=spec, instance=model.name, status="unknown-solver")
    r.error = f"unrecognised solver {spec!r}"
    return r


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def shifted_geomean(values: list[float], shift: float) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    acc = sum(math.log(max(v, 0.0) + shift) for v in vals)
    return math.exp(acc / len(vals)) - shift


# --------------------------------------------------------------------------
# Shifted geometric mean used by comparison reports.
#
# The metric is the RATIO OF SHIFTED GEOMETRIC MEANS:
#
#     SGM(t) = exp(mean(log(t + 1s))) - 1s        ratio = SGM(SOR) / SGM(HiGHS)
#
# It is NOT the geometric mean of per-model ratios. Those are different
# statistics and they disagree violently: on Netlib-93 (2026-09-10) the ratio
# of SGMs was 1.4307 while the geomean of ratios was 0.8684 -- one fails the
# 0.95 gate by 50%, the other appears to pass it. The geomean of ratios weights
# a 0.4 ms model exactly like an 11 s one, so it reports the small end of the
# suite; the ratio of SGMs is dominated by the models that actually cost time.
# A claim was published off the wrong one. Hence this function.
# --------------------------------------------------------------------------
PAR2_MULTIPLIER = 2.0


def par2_seconds(result: "Result | None", time_limit: float) -> float:
    """Scored time for one result: PAR-2 unless it is a certified success.

    Plan §3.4 charges twice the time limit for "timeouts, wrong answers,
    invalid certificates, or unchecked results". Dropping such a model instead
    flatters the aggregate exactly where the solver did worst -- an unproved
    pilot87 still costs 13 s, and excluding it reports a number the run did not
    earn.
    """
    if result is None:
        return PAR2_MULTIPLIER * time_limit
    if not is_certified_success(result):
        return PAR2_MULTIPLIER * time_limit
    seconds = result.seconds
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return PAR2_MULTIPLIER * time_limit
    return seconds


def noise_band(result: "Result | None") -> float:
    """Half-width of the measured noise for one result, in seconds.

    §3.4: "Treat a timing as a win only when its difference exceeds the
    recorded noise band; otherwise count it as a tie." Without this a 0.2%
    difference on a 1 ms model counts as a win, and the win rate becomes a
    measurement of the host rather than of the solver.
    """
    if result is None or result.mad_s is None or not math.isfinite(result.mad_s):
        return 0.0
    return max(result.mad_s, 0.0)


def noise_aware_outcome(cand: "Result | None", ref: "Result | None",
                        time_limit: float) -> str:
    """'win' | 'loss' | 'tie', with ties absorbing the combined noise band."""
    c = par2_seconds(cand, time_limit)
    r = par2_seconds(ref, time_limit)
    band = noise_band(cand) + noise_band(ref)
    if c < r - band:
        return "win"
    if c > r + band:
        return "loss"
    return "tie"


def reference_is_valid(r: "Result | None") -> tuple[bool, str]:
    """Is this oracle row usable as the denominator of a published ratio?

    An unavailable, errored, crashed, uncertified or misconfigured oracle is
    not a reference. `Result.error` in particular must veto certification: the
    HiGHS worker reports rejected options through it, so a run where
    `small_matrix_value` or the tolerance was silently not applied would
    otherwise be treated as a valid baseline for a claim.
    """
    if r is None:
        return False, "no reference row"
    if r.error:
        return False, f"reference reported an error: {str(r.error)[:80]}"
    if r.status and r.status.lower() in ("unavailable", "crash", "error",
                                         "timeout", "notrun", "unparsed"):
        return False, f"reference status {r.status!r}"
    if not is_certified_success(r):
        return False, f"reference not a certified success (status {r.status!r})"
    if r.objective is None or not math.isfinite(r.objective):
        return False, "reference objective is not finite"
    if r.seconds is None or not math.isfinite(r.seconds) or r.seconds <= 0.0:
        return False, "reference time is not finite and positive"
    return True, ""


@dataclass
class ClaimGateResult:
    suite: str = ""
    tolerance: float = 0.0
    # PASS / FAIL / INCOMPLETE. INCOMPLETE means the evidence needed to decide
    # was not present -- which is never a pass.
    status: str = "INCOMPLETE"
    scored: int = 0
    expected: int | None = None
    sgm_candidate: float | None = None
    sgm_reference: float | None = None
    sgm_ratio: float | None = None
    wins: int = 0
    losses: int = 0
    ties: int = 0
    win_rate: float = 0.0
    candidate_certified: int = 0
    reference_certified: int = 0
    objective_mismatches: int = 0
    par2_penalised: int = 0
    noisy_pairs: int = 0
    wall_candidate: float = 0.0
    wall_baseline: float | None = None
    wall_ratio: float | None = None
    tail_violations: list = field(default_factory=list)
    missing_instances: list = field(default_factory=list)
    invalid_references: list = field(default_factory=list)
    checker_rejections: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    incompleteness: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def evaluate_public_claim(by_instance: dict, candidate: str, reference: str,
                          time_limit: float, shift: float = 1.0,
                          abs_tol: float = 1e-7, rel_tol: float = 1e-7,
                          baseline_solver: dict | None = None,
                          baseline_wall: dict | None = None,
                          allow: dict | None = None,
                          expected_instances=None,
                          claim_mode: bool = False,
                          suite: str = "", tolerance: float = 0.0,
                          candidate_checks: dict | None = None
                          ) -> ClaimGateResult:
    """Score a suite against the oracle and enforce every §3.5 gate.

    FAILS CLOSED. In `claim_mode` the evidence a gate needs must be PRESENT:
    the complete instance set from the frozen manifest, a pinned SOR baseline
    for both solver time and process wall, and the committed allow-list. If any
    is missing the verdict is INCOMPLETE, never PASS -- a gate that silently
    skips itself when its inputs are absent is worse than no gate, because it
    reports success.

    `baseline_solver` and `baseline_wall` are deliberately SEPARATE dicts. The
    2x tail rule compares solver time against baseline solver time; the
    process-wall rule compares wall against baseline wall. Feeding one dict to
    both compares different quantities and silently mislabels whichever it is
    not.
    """
    out = ClaimGateResult(suite=suite, tolerance=tolerance)
    cand_times: list[float] = []
    ref_times: list[float] = []

    # ---- claim-mode evidence requirements (fail closed) -----------------
    if claim_mode:
        if expected_instances is None:
            out.incompleteness.append(
                "no frozen manifest instance set supplied: cannot verify the "
                "run covered the corpus it claims")
        if baseline_solver is None:
            out.incompleteness.append(
                "no pinned SOR baseline: the 2x per-model tail rule cannot run")
        if baseline_wall is None:
            out.incompleteness.append(
                "no pinned SOR process-wall baseline: the wall "
                "non-regression rule cannot run")
        if allow is None:
            out.incompleteness.append(
                "no committed allow-list: waivers cannot be distinguished from "
                "unexplained regressions")
        if candidate_checks is None:
            out.incompleteness.append(
                "no measured-solution checker results supplied")
    allow = allow or {}

    if expected_instances is not None:
        expected_set = set(expected_instances)
        missing = sorted(expected_set - set(by_instance))
        if missing:
            out.missing_instances = missing
            out.incompleteness.append(
                f"{len(missing)} manifest instance(s) absent from the run, "
                f"first: {', '.join(missing[:5])}")
        if claim_mode:
            for label, baseline in (("solver", baseline_solver),
                                    ("wall", baseline_wall)):
                if baseline is None:
                    continue
                absent = sorted(expected_set - set(baseline))
                unexpected = sorted(set(baseline) - expected_set)
                invalid = sorted(k for k, v in baseline.items()
                                 if type(v) not in (int, float) or
                                 not math.isfinite(v) or v <= 0.0)
                if absent:
                    out.incompleteness.append(
                        f"baseline {label} coverage misses {absent[:5]}")
                if unexpected:
                    out.incompleteness.append(
                        f"baseline {label} has unexpected instances {unexpected[:5]}")
                if invalid:
                    out.incompleteness.append(
                        f"baseline {label} has non-positive/non-finite values {invalid[:5]}")

    # In claim mode the manifest's scoring-eligible set is the entire scoring
    # universe. Import-valid but non-scoring manifest entries must not leak in
    # merely because the sweep happened to contain them.
    universe = (sorted(set(expected_instances)) if expected_instances is not None
                else sorted(by_instance))
    for inst in universe:
        pair = by_instance.get(inst, {})
        c = pair.get(candidate)
        r = pair.get(reference)

        ok, why = reference_is_valid(r)
        if not ok:
            # An unusable oracle is INCOMPLETE evidence, not a skipped model.
            out.invalid_references.append((inst, why))
            continue

        out.scored += 1
        out.reference_certified += 1
        check_ok = True
        check_why = ""
        if candidate_checks is not None:
            verdict = candidate_checks.get(inst)
            if verdict is None:
                check_ok, check_why = False, "no checker result"
            elif isinstance(verdict, tuple):
                check_ok, check_why = bool(verdict[0]), str(verdict[1])
            else:
                check_ok = bool(verdict)
                check_why = "checker rejected" if not check_ok else ""
            if not check_ok:
                out.checker_rejections.append((inst, check_why))
        if c is not None and is_certified_success(c) and check_ok:
            out.candidate_certified += 1
        if (c is not None and c.noisy) or r.noisy:
            out.noisy_pairs += 1

        # Correctness is decided BEFORE timing is aggregated: a candidate whose
        # objective disagrees with the oracle is wrong, and a wrong answer is
        # charged PAR-2 rather than contributing its fast real time.
        correct = True
        if c is None or not is_certified_success(c) or not check_ok:
            correct = False
        elif not objectives_agree(c.objective, r.objective, abs_tol, rel_tol):
            correct = False
            out.objective_mismatches += 1

        cs = par2_seconds(c, time_limit) if correct else PAR2_MULTIPLIER * time_limit
        rs = par2_seconds(r, time_limit)
        if not correct:
            out.par2_penalised += 1
        cand_times.append(cs)
        ref_times.append(rs)

        cw = (c.median_wall_s if c is not None and c.median_wall_s
              else (c.wall_s if c is not None and c.wall_s else cs))
        out.wall_candidate += cw if correct else PAR2_MULTIPLIER * time_limit

        outcome = (noise_aware_outcome(c, r, time_limit) if correct else "loss")
        if outcome == "win":
            out.wins += 1
        elif outcome == "loss":
            out.losses += 1
        else:
            out.ties += 1

        if baseline_solver is not None:
            base = baseline_solver.get(inst)
            if base is None:
                out.incompleteness.append(f"{inst}: no baseline solver time")
            elif base > 0.0 and cs / base > 2.0:
                ratio = cs / base
                waiver = allow.get(inst)
                waived = (isinstance(waiver, dict) and
                           isinstance(waiver.get("reason"), str) and
                           bool(waiver["reason"].strip()))
                bound = waiver.get("max_work_ratio") if waived else None
                if bound is not None:
                    waived = (type(bound) in (int, float) and
                              math.isfinite(bound) and bound > 0.0 and
                              ratio <= bound)
                if not waived:
                    out.tail_violations.append((inst, ratio))

    if out.invalid_references:
        out.incompleteness.append(
            f"{len(out.invalid_references)} model(s) have no usable "
            f"{reference} reference, first: "
            + ", ".join(f"{i} ({w})" for i, w in out.invalid_references[:3]))

    if expected_instances is not None:
        out.expected = len(set(expected_instances))
        if out.scored != out.expected:
            out.incompleteness.append(
                f"scored {out.scored} of {out.expected} manifest instances")

    if not cand_times:
        out.incompleteness.append("no mutually comparable models")
        out.status = "INCOMPLETE"
        return out

    out.sgm_candidate = shifted_geomean(cand_times, shift)
    out.sgm_reference = shifted_geomean(ref_times, shift)
    if out.sgm_reference:
        out.sgm_ratio = out.sgm_candidate / out.sgm_reference
    out.win_rate = 100.0 * out.wins / out.scored

    if baseline_wall is not None:
        # Even malformed evidence with an unexpected row must never change the
        # displayed metric.  Claim mode already rejects that coverage above;
        # the number itself is nevertheless defined only over the manifest's
        # scoring universe, not over every key present in the JSONL file.
        total = sum(baseline_wall[inst] for inst in universe
                    if type(baseline_wall.get(inst)) in (int, float) and
                    math.isfinite(baseline_wall[inst]) and
                    baseline_wall[inst] > 0.0)
        out.wall_baseline = total or None
        if out.wall_baseline:
            out.wall_ratio = out.wall_candidate / out.wall_baseline

    # ---- the gates, in order of severity --------------------------------
    if out.candidate_certified < out.scored:
        out.failures.append(
            f"proofs: {out.candidate_certified}/{out.scored} certified "
            f"({out.scored - out.candidate_certified} uncertified, PAR-2 charged)")
    if out.objective_mismatches:
        out.failures.append(
            f"correctness: {out.objective_mismatches} certified objective(s) "
            f"disagree with {reference} (PAR-2 charged)")
    if out.checker_rejections:
        out.failures.append(
            f"independent checker rejected or did not check "
            f"{len(out.checker_rejections)} measured result(s) (PAR-2 charged)")
    if out.sgm_ratio is None or out.sgm_ratio > 0.95:
        out.failures.append(
            f"shifted SGM ratio {out.sgm_ratio:.4f} > 0.95"
            if out.sgm_ratio is not None else "shifted SGM ratio unavailable")
    if out.win_rate < 60.0:
        out.failures.append(
            f"noise-aware win rate {out.win_rate:.1f}% < 60% "
            f"({out.wins}W/{out.losses}L/{out.ties}T)")
    if out.tail_violations:
        worst = max(out.tail_violations, key=lambda t: t[1])
        out.failures.append(
            f"{len(out.tail_violations)} model(s) over 2x the pinned baseline "
            f"without a waiver, worst {worst[0]} {worst[1]:.2f}x")
    if claim_mode and out.wall_ratio is None:
        out.incompleteness.append("process-wall ratio could not be computed")
    elif out.wall_ratio is not None and out.wall_ratio > 1.0:
        out.failures.append(
            f"process-wall sum {out.wall_ratio:.4f}x its own pinned baseline")

    if out.incompleteness:
        out.status = "INCOMPLETE"
    elif out.failures:
        out.status = "FAIL"
    else:
        out.status = "PASS"
    return out


def fmt_time(s: float | None) -> str:
    if s is None:
        return "-"
    return f"{s * 1000:.1f}ms" if s < 1.0 else f"{s:.3f}s"


def fmt_obj(v: float | None) -> str:
    return "-" if v is None else f"{v:.6g}"


_SOR_PROOFS = {
    "ProvedOptimalFP", "ProvedOptimalExact", "ProvedOptimalCertified",
    "ProvedKKT", "ProvedGlobalEpsilon",
}


def is_optimal(r: Result) -> bool:
    return r.status.strip().lower() in ("optimal", "solved", "1")


def is_certified_success(r: Result) -> bool:
    """Fail closed for claims that the checker can validate end to end.

    ``milp_incumbent`` deliberately does not qualify: sor_check verifies the
    original-model incumbent, integrality and objective, but it does not replay
    the search tree or independently validate the final global bound.
    """
    if not is_optimal(r):
        return False
    if not r.solver.lower().startswith("sor"):
        return True
    # Incumbent and primal-point checks, including QP, do not replay an
    # optimality proof. Admit only scopes that independently check it.
    return (r.proof in _SOR_PROOFS and r.checker_verified is True and
            r.checker_validation_scope in ("lp_optimality_f64", "qp_kkt_f64"))


def parse_solu_file(path: Path) -> dict[str, tuple[str, float | None]]:
    """Parse a MIPLIB .solu file: '=opt=|=best=|=unkn=  name  [value]' per
    line. Maps instance name -> (tag, value); value is None for =unkn= (and
    for any line missing a trailing number, rather than raising)."""
    out: dict[str, tuple[str, float | None]] = {}
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            tag, name = parts[0], parts[1]
            if tag not in ("=opt=", "=best=", "=unkn="):
                continue
            value = None
            if len(parts) >= 3:
                try:
                    value = float(parts[2])
                except ValueError:
                    value = None
            out[name] = (tag, value)
    return out


def solu_lookup(solu: dict[str, tuple[str, float | None]],
                instance: str) -> tuple[str, float | None] | None:
    """MPS files carry a path and extension the .solu file does not."""
    stem = Path(instance).stem
    return solu.get(stem)


def objectives_agree(a: float | None, b: float | None,
                     abs_tol: float, rel_tol: float) -> bool:
    """Does candidate objective `a` agree with REFERENCE objective `b`?

    The band is max(abs_tol, rel_tol * (1 + |b|)) -- scaled by the reference,
    not by max(|a|, |b|), so a candidate cannot widen its own tolerance by
    being wrong in the large direction.

    This used to be `abs_tol + rel_tol * max(|a|, |b|)` at rel_tol 1e-4, which
    is a looser agreement than either solver's own optimality tolerance: a
    forced-dual run certified pilot.mps to 1.45e-05 and the harness called it
    a match. At 1e-7 the harness is no longer the weakest link in the claim.
    """
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return False
    return abs(a - b) <= max(abs_tol, rel_tol * (1.0 + abs(b)))


def choose_reference(results: dict[str, Result], solvers: list[str],
                     requested: str | None) -> Result | None:
    """Choose a certified reference; never use an interrupted objective."""
    names = {name.lower(): name for name in results}
    preferred = requested
    if preferred is None and "highs" in names:
        preferred = names["highs"]
    if preferred is not None:
        key = names.get(preferred.lower())
        if key is None:
            return None
        candidate = results[key]
        return candidate if is_certified_success(candidate) and (
            candidate.objective is not None and math.isfinite(candidate.objective)) else None
    for name in solvers:
        candidate = results.get(name)
        if candidate and is_certified_success(candidate) and (
                candidate.objective is not None and math.isfinite(candidate.objective)):
            return candidate
    return None


# Section 3.4 of the execution plan: a model/solver pair whose MAD exceeds
# this fraction of its median is rerun, and a host that stays above it is
# rejected rather than published.
NOISE_BAND = 0.05


def aggregate_repetitions(runs: list[Result]) -> Result:
    """Use a median representative, but surface any failed/flaky repetition."""
    if not runs:
        raise ValueError("cannot aggregate an empty run list")
    failed = [r for r in runs if not is_certified_success(r)]
    if failed:
        chosen = replace(failed[0])
    else:
        timed = [r for r in runs if r.seconds is not None]
        chosen = replace(min(timed, key=lambda r: abs(
            r.seconds - statistics.median(x.seconds for x in timed)))
                         if timed else runs[0])
    chosen.repetition = None
    chosen.samples_s = [r.seconds for r in runs if r.seconds is not None]
    chosen.wall_samples_s = [r.wall_s for r in runs if r.wall_s is not None]
    if chosen.wall_samples_s:
        # The process wall is aggregated INDEPENDENTLY. Keeping the wall time
        # that happened to accompany the median solver-time sample reports one
        # repetition's wall, not the median wall -- and the two orderings
        # differ whenever process startup varies, which is exactly the regime
        # (millisecond models) where the wall matters most.
        chosen.median_wall_s = statistics.median(chosen.wall_samples_s)
        chosen.wall_s = chosen.median_wall_s
    if chosen.samples_s:
        median = statistics.median(chosen.samples_s)
        # Median absolute deviation, not stdev: one descheduled repetition
        # should not be allowed to describe the other four.
        mad = statistics.median([abs(x - median) for x in chosen.samples_s])
        chosen.median_s = median
        chosen.mad_s = mad
        chosen.mad_rel = (mad / median) if median > 0 else 0.0
        chosen.noisy = chosen.mad_rel > NOISE_BAND
    return chosen


def report(rows: list[Result], solvers: list[str], shift: float,
           abs_tol: float, rel_tol: float, time_limit: float,
           reference: str | None,
           solu: dict[str, tuple[str, float | None]] | None = None
           ) -> dict[str, int]:
    by_inst: dict[str, dict[str, Result]] = {}
    for r in rows:
        by_inst.setdefault(r.instance, {})[r.solver] = r

    wid = max([len(i) for i in by_inst] + [8])
    colw = max([len(s) for s in solvers] + [18])

    print()
    header = f"{'instance':<{wid}}  " + "  ".join(f"{s:<{colw}}" for s in solvers)
    print(header)
    print("-" * len(header))
    for inst in sorted(by_inst):
        cells = []
        for s in solvers:
            r = by_inst[inst].get(s)
            cells.append("-" if r is None
                         else f"{r.status[:10]:<10} {fmt_time(r.seconds):>7}")
        print(f"{inst:<{wid}}  " + "  ".join(f"{c:<{colw}}" for c in cells))

    # Objective agreement against an explicit or certified reference.
    print()
    ref_label = reference or ("highs" if "highs" in [s.lower() for s in solvers]
                              else "first certified solver")
    print(f"{'instance':<{wid}}  objective (reference: {ref_label})")
    print("-" * (wid + 44))
    mismatches = 0
    unchecked = 0
    agreements: dict[tuple[str, str], bool] = {}
    for inst in sorted(by_inst):
        ref = choose_reference(by_inst[inst], solvers, reference)
        if ref is None:
            unchecked += 1
        line = f"{inst:<{wid}}  "
        for s in solvers:
            r = by_inst[inst].get(s)
            if r is None or r.objective is None:
                line += f"{'-':<{colw}}  "
                continue
            tag = ""
            if ref is not None and is_certified_success(r):
                match = objectives_agree(r.objective, ref.objective,
                                         abs_tol, rel_tol)
                agreements[(inst, s)] = match
                if not match:
                    tag = " MISMATCH"
                    mismatches += 1
            elif is_certified_success(r):
                tag = " UNCHECKED"
            line += f"{fmt_obj(r.objective) + tag:<{colw}}  "
        print(line)

    # Primal/dual gap against the published MIPLIB reference (--solu), when
    # given: independent of any solver-vs-solver agreement above, and the
    # only one of these two tables that says anything when only one solver
    # ran. =unkn= instances are printed as "no ref" rather than skipped, so a
    # thin table doesn't read as "everything agreed".
    if solu:
        print()
        print(f"{'instance':<{wid}}  gap vs .solu")
        print("-" * (wid + 44))
        for inst in sorted(by_inst):
            entry = solu_lookup(solu, inst)
            line = f"{inst:<{wid}}  "
            for s in solvers:
                r = by_inst[inst].get(s)
                if r is None or r.objective is None:
                    line += f"{'-':<{colw}}  "
                    continue
                if entry is None or entry[1] is None:
                    line += f"{'no ref':<{colw}}  "
                    continue
                tag, ref_val = entry
                gap = abs(r.objective - ref_val) / (1.0 + abs(ref_val))
                cell = f"{gap:.2e}"
                # A gap on an =opt= reference that isn't itself a certified
                # proof is a primal gap only -- the dual side is whatever the
                # solver's own dual_bound says, printed separately, never
                # implied by matching the known optimum.
                if tag == "=opt=" and is_certified_success(r) and gap > rel_tol:
                    cell += " MISMATCH"
                line += f"{cell:<{colw}}  "
            print(line)

    print()
    print(f"{'solver':<{colw}} {'proved':>8} {'correct':>9} "
          f"{'pSGM(s)':>10} {'PAR2(s)':>10}")
    print("-" * (colw + 43))
    for s in solvers:
        rs = [by_inst[i].get(s) for i in by_inst]
        rs = [r for r in rs if r is not None]
        proved = [r for r in rs if is_certified_success(r)]
        correct = sum(1 for r in proved if agreements.get((r.instance, s), False))
        penalty = 2.0 * time_limit
        penalized = [r.seconds if (is_certified_success(r) and
                      agreements.get((r.instance, s), False) and
                      r.seconds is not None) else penalty for r in rs]
        sgm = shifted_geomean(penalized, shift)
        par2 = sum(penalized) / len(penalized) if penalized else None
        print(f"{s:<{colw}} {len(proved):>4}/{len(rs):<3} "
              f"{correct:>5}/{len(rs):<3} "
              f"{(f'{sgm:.4f}' if sgm is not None else '-'):>10} "
              f"{(f'{par2:.4f}' if par2 is not None else '-'):>10}")
    # Ordinary compare.py runs deliberately lack the frozen manifest, pinned
    # baseline, allow-list, measured-solution checker and dual HiGHS lanes
    # required by claim_run.py.  They may show the same metric, but must never
    # print a public PASS that can be mistaken for protocol-complete evidence.
    ref_label = None
    if reference is not None:
        ref_label = next((x for x in solvers if x.lower() == reference.lower()), None)
    if ref_label is None:
        ref_label = next((x for x in solvers if x.lower() == "highs"), None)
    cand_label = next((x for x in solvers if x.lower().startswith("sor")), None)
    if ref_label and cand_label and ref_label != cand_label:
        gate = evaluate_public_claim(by_inst, cand_label, ref_label,
                                     time_limit=time_limit, shift=shift,
                                     abs_tol=abs_tol, rel_tol=rel_tol)
        print()
        print(f"PROVISIONAL COMPARISON  {cand_label} vs {ref_label}   "
              f"(SGM(t)=exp(mean(log(t+{shift:g}s)))-{shift:g}s, PAR-2 charged)")
        print(f"  shifted SGM ratio : {gate.sgm_ratio:.4f}"
              f"   [SOR {gate.sgm_candidate:.4f}s / ref {gate.sgm_reference:.4f}s]"
              f"   gate <= 0.95"
              if gate.sgm_ratio is not None else "  shifted SGM ratio : n/a")
        print(f"  noise-aware wins  : {gate.win_rate:.1f}%"
              f"   ({gate.wins}W / {gate.losses}L / {gate.ties}T of {gate.scored})"
              f"   gate >= 60%")
        print(f"  certified         : {gate.candidate_certified}/{gate.scored}"
              f"   PAR-2 charged: {gate.par2_penalised}"
              f"   noisy pairs: {gate.noisy_pairs}")
        print(f"  VERDICT           : PROVISIONAL "
              f"{'PASS' if gate.passed else 'FAIL'}")
        print("  claim status      : INCOMPLETE (use claim_run.py with all evidence)")
        for f in gate.failures:
            print(f"      - {f}")

    if mismatches:
        print(f"\n{mismatches} objective mismatch(es) above abs={abs_tol:g}, "
              f"rel={rel_tol:g}; timings are penalized.")
    if unchecked:
        print(f"\n{unchecked} instance(s) had no certified reference objective; "
              "timings are penalized.")
    unavailable = sum(r.status.lower() == "unavailable" for r in rows)
    return {"mismatches": mismatches, "unchecked": unchecked,
            "unavailable": unavailable}


def collect_models(paths: list[str], limit: int | None) -> list[Path]:
    models: list[Path] = []
    for p in paths:
        q = Path(p)
        if not q.is_absolute():
            q = (Path.cwd() / q) if (Path.cwd() / q).exists() else (ROOT / q)
        if q.is_dir():
            # `.gz` is discovered alongside the plain forms: MIPLIB ships
            # every model compressed, and a suite that silently skipped them
            # would report a smaller corpus than the manifest claims.
            for pat in ("*.mps", "*.qps", "*.lp",
                        "*.mps.gz", "*.qps.gz", "*.lp.gz"):
                models.extend(sorted(q.rglob(pat)))
        elif q.exists():
            models.append(q)
        else:
            print(f"warning: no such model or directory: {p}", file=sys.stderr)
    seen, out = set(), []
    for m in models:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out[:limit] if limit else out


def duplicate_model_names(models: list[Path]) -> dict[str, list[Path]]:
    """Find distinct paths that would share one benchmark instance identity.

    MIPLIB's local archive can contain the same 240 compressed models in a
    parent directory and a nested directory. A recursive run over the parent
    otherwise counts 480 instances and can overwrite solution filenames.
    """
    by_name: dict[str, list[Path]] = {}
    for model in models:
        name = model.name[:-3] if model.name.endswith(".gz") else model.name
        identity = Path(name).stem
        by_name.setdefault(identity, []).append(model)
    return {name: paths for name, paths in by_name.items() if len(paths) > 1}


def read_clocksource() -> str | None:
    """The kernel clocksource, which decides what a timing run even measures.

    This is not trivia. SOR calls the clock ~20 times per simplex pivot; HiGHS
    calls it about once per solve. Under `tsc` a call is ~25 ns and the
    instrumentation is invisible. Under `hpet` it is ~1400 ns, which taxes SOR
    roughly 28 us PER PIVOT and HiGHS not at all -- enough to turn a 0.28 ms
    model into 0.99 ms and hand HiGHS 30 wins it did not earn. Measured on this
    host 2026-09-10, after a reboot silently dropped tsc from
    available_clocksource. Repetitions and MAD cannot detect it: the bias is
    perfectly stable, so every repetition agrees.
    """
    try:
        return (Path("/sys/devices/system/clocksource/clocksource0"
                     "/current_clocksource").read_text().strip())
    except OSError:
        return None


def environment_record(args: argparse.Namespace, solvers: list[str],
                       models: list[Path], exe: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
        status_text = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout
        dirty = bool(status_text.strip())
        tracked_patch = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=ROOT,
            capture_output=True, check=True).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT,
            capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception:  # noqa: BLE001
        commit, dirty, tracked_patch, untracked = None, None, b"", []
    untracked_hashes = {
        name: sha256_file(ROOT / name) for name in sorted(untracked)
        if (ROOT / name).is_file()
    }
    workspace_identity = tracked_patch + json.dumps(
        untracked_hashes, sort_keys=True).encode()
    cache = exe.parent / "CMakeCache.txt"
    cmake_configuration: dict[str, str] = {}
    if cache.is_file():
        wanted = {"CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER",
                  "CMAKE_CXX_COMPILER_VERSION", "CMAKE_CXX_FLAGS",
                  "CMAKE_CXX_FLAGS_RELEASE"}
        for line in cache.read_text(errors="replace").splitlines():
            if "=" not in line or line.startswith(("#", "//")):
                continue
            key_type, value = line.split("=", 1)
            key = key_type.split(":", 1)[0]
            if key in wanted:
                cmake_configuration[key] = value
    return {
        "record": "environment",
        "commit": commit,
        "dirty": dirty,
        "workspace_sha256": sha256_bytes(workspace_identity),
        "untracked_sha256": untracked_hashes,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cpu_affinity": args.cpu,
        "executable": str(exe.resolve()),
        "executable_sha256": sha256_file(exe),
        "cmake_configuration": cmake_configuration,
        "solvers": solvers,
        "models": len(models),
        "model_sha256": {str(m.resolve()): sha256_file(m) for m in models},
        "time_limit_s": args.time_limit,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "method": args.method,
        "pricing": args.pricing,
        "dual_cost_perturbation": args.dual_cost_perturbation,
        "basis_update": args.basis_update,
        "sor_extra_args": list(args.sor_arg),
        "relax_integrality": args.relax_integrality,
        "small_matrix_value": args.small_matrix_value,
        "obj_abs_tol": args.obj_abs_tol,
        "obj_rel_tol": args.obj_rel_tol,
        "noise_band": NOISE_BAND,
        "clocksource": read_clocksource(),
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "highs_source": (str(args.highs_source.resolve())
                         if args.highs_source is not None else None),
        "highs_source_sha256": (sha256_file(args.highs_source)
                                if args.highs_source is not None else None),
        "solutions_dir": (str(args.solutions_dir.resolve())
                          if args.solutions_dir is not None else None),
    }


def build_parser() -> argparse.ArgumentParser:
    """The CLI, separated from main() so the shipped defaults are testable.

    The tolerance defaults are part of the benchmark contract rather than
    ergonomics, and they have been wrong before, so they get a test.
    """
    ap = argparse.ArgumentParser(
        description="Run several solvers on several models and show the "
                    "results side by side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("models", nargs="+",
                    help="model files, or directories to scan for .mps/.qps/.lp")
    ap.add_argument("--solvers", default="sor:simplex,highs",
                    help="comma-separated. sor:<engine>[:<backend>] "
                         "(simplex, dual, pdhg, hpr, hpr-full, milp, qp) "
                         "or highs, cbc, scipy, scipy-ipm, gurobi")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--tol", type=float, default=1e-7,
                    help="feasibility/optimality tolerance; the primary "
                         "public gate is 1e-7. Use --tol 1e-6 for the "
                         "Netlib historical continuity lane.")
    ap.add_argument("--method", choices=("auto", "primal", "dual"), default=None,
                    help="simplex method for sor:simplex/sor:milp; sor:primal and "
                         "sor:dual override it")
    ap.add_argument("--pricing", choices=("choose", "dantzig", "devex", "dse"),
                    default="choose", help="simplex pricing strategy")
    ap.add_argument("--dual-cost-perturbation", type=float, default=0.0,
                    help="experimental dual-simplex cost perturbation multiplier")
    ap.add_argument("--basis-update", choices=("default", "product", "ft"),
                    default="default",
                    help="default: let the solver choose (recorded as such); "
                         "product/ft force one representation")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="explicit solver iteration/node cap")
    ap.add_argument("--warmups", type=int, default=0)
    ap.add_argument("--repetitions", type=int, default=1,
                    help="measured repetitions; summary uses the median")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for solver-order randomization")
    ap.add_argument("--cpu", type=int, default=None,
                    help="pin child solver processes to this logical CPU")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N models")
    ap.add_argument("--sgm-shift", type=float, default=1.0,
                    help="shift for the shifted geometric mean, in seconds")
    ap.add_argument("--obj-tol", "--obj-rel-tol", dest="obj_rel_tol",
                    type=float, default=1e-7,
                    help="relative tolerance for objective agreement, scaled "
                         "by (1 + |reference objective|)")
    ap.add_argument("--obj-abs-tol", type=float, default=1e-7,
                    help="absolute tolerance for objective agreement")
    ap.add_argument("--reference", default=None,
                    help="reference solver label (default: highs when selected, "
                         "otherwise first certified solver)")
    ap.add_argument("--solu", type=Path, default=None,
                    help="MIPLIB .solu file (=opt=/=best=/=unkn= per instance); "
                         "when given, an extra table reports the primal gap "
                         "against the published value, independent of any "
                         "solver-vs-solver reference comparison")
    ap.add_argument("--allow-unavailable", action="store_true",
                    help="do not fail solely because a requested solver is unavailable")
    ap.add_argument("--allow-unchecked", action="store_true",
                    help="do not fail solely because no certified reference exists")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="also append one JSON record per run to this file")
    ap.add_argument("--solutions-dir", type=Path, default=None,
                    help="write each measured SOR repetition's solution here; "
                         "the directory must not already exist")
    ap.add_argument("--highs-source", type=Path, default=None,
                    help="source-built HiGHS CLI used by the highs-source lane")
    ap.add_argument("--exe", type=Path, default=None,
                    help="sor_solve binary (default: <root>/build/sor_solve)")
    ap.add_argument("--relax-integrality", action="store_true",
                    help="solve the LP relaxation of an integer model. Both "
                         "solvers read the SAME original file: HiGHS with "
                         "solve_relaxation=true, SOR with relax_integrality. "
                         "Never compare against an MPS another solver rewrote.")
    ap.add_argument("--small-matrix-value", type=float, default=None,
                    help="drop matrix coefficients at or below this magnitude, "
                         "on BOTH solvers, so they receive the same matrix. "
                         "Unset means neither solver is told to filter -- it is "
                         "never applied to one side alone. Tier-2A requires "
                         "1e-9 here, which needs sor_solve's matching CLI flag.")
    ap.add_argument("--sor-arg", action="append", default=[],
                    metavar="FLAG",
                    help="extra flag passed verbatim to sor_solve; repeatable. "
                         "For sweeping a tuning parameter without teaching "
                         "this script every sor_solve option. Use the "
                         "=form for flags: "
                         "--sor-arg=--refactor-work-ratio --sor-arg=2.0")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    if (not math.isfinite(args.time_limit) or args.time_limit <= 0 or
            args.repetitions <= 0 or args.warmups < 0):
        ap.error("time limit and repetitions must be positive; warmups cannot be negative")
    if not math.isfinite(args.tol) or args.tol <= 0:
        ap.error("--tol must be finite and positive")
    if not math.isfinite(args.sgm_shift) or args.sgm_shift <= 0:
        ap.error("--sgm-shift must be finite and positive")
    if (not math.isfinite(args.obj_abs_tol) or args.obj_abs_tol < 0 or
            not math.isfinite(args.obj_rel_tol) or args.obj_rel_tol < 0):
        ap.error("objective tolerances must be finite and nonnegative")
    if args.max_iter is not None and args.max_iter <= 0:
        ap.error("--max-iter must be positive")
    if args.cpu is not None and args.cpu < 0:
        ap.error("--cpu cannot be negative")
    if (not math.isfinite(args.dual_cost_perturbation) or
            args.dual_cost_perturbation < 0):
        ap.error("--dual-cost-perturbation must be finite and nonnegative")
    if (args.small_matrix_value is not None and
            (not math.isfinite(args.small_matrix_value) or
             args.small_matrix_value <= 0)):
        ap.error("--small-matrix-value must be finite and positive")

    exe = args.exe or (ROOT / "build" / "sor_solve")
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    if args.reference is not None and args.reference.lower() not in {
            s.lower() for s in solvers}:
        ap.error("--reference must name one of --solvers")
    if any(s.lower().startswith("sor") for s in solvers) and not exe.exists():
        print(f"error: {exe} not found -- build first:\n"
              f"  cmake -S {ROOT} -B {ROOT}/build -DCMAKE_BUILD_TYPE=Release\n"
              f"  cmake --build {ROOT}/build -j", file=sys.stderr)
        return 2
    if any(s.lower() == "highs-source" for s in solvers):
        if args.highs_source is None:
            ap.error("the highs-source lane requires --highs-source")
        if not args.highs_source.is_file() or not os.access(args.highs_source, os.X_OK):
            ap.error("--highs-source must be a regular executable file")

    if args.jsonl is not None and args.jsonl.exists():
        print(f"error: result file {args.jsonl} already exists; refusing to "
              "mix campaigns", file=sys.stderr)
        return 2

    models = collect_models(args.models, args.limit)
    if not models:
        print("error: no models found", file=sys.stderr)
        return 2
    duplicates = duplicate_model_names(models)
    if duplicates:
        first = next(iter(duplicates.items()))
        print(f"error: {len(duplicates)} duplicate model identities; "
              f"{first[0]} appears at {first[1][0]} and {first[1][1]}. "
              "Choose one copy of each model.", file=sys.stderr)
        return 2

    if args.solutions_dir is not None:
        if args.solutions_dir.exists():
            print(f"error: solution directory {args.solutions_dir} already exists; "
                  "refusing to overwrite measured evidence", file=sys.stderr)
            return 2
        args.solutions_dir.mkdir(parents=True)

    source_exe = exe
    source_hash = None
    binary_temp = None
    if any(s.lower().startswith("sor") for s in solvers):
        if args.jsonl is None:
            binary_temp = tempfile.TemporaryDirectory(prefix="sor-compare-binaries-")
            binary_dir = Path(binary_temp.name)
        else:
            binary_dir = args.jsonl.parent / (args.jsonl.stem + "-binaries")
        exe, source_hash = freeze_sor_binaries(source_exe, binary_dir)
        frozen_hash = sha256_file(exe)
        source_checker = source_exe.with_name("sor_check")
        source_checker_hash = sha256_file(source_checker)
        frozen_checker = exe.with_name("sor_check")
        frozen_checker_hash = sha256_file(frozen_checker)
    else:
        frozen_hash = None
        source_checker_hash = frozen_checker_hash = None

    print(f"{len(models)} model(s) x {len(solvers)} solver(s), "
          f"time limit {args.time_limit:g}s")
    clock = read_clocksource()
    if clock is not None and clock != "tsc":
        print(f"\n*** WARNING: kernel clocksource is {clock!r}, not 'tsc'. ***\n"
              f"    clock_gettime is ~50x more expensive here, and SOR calls it\n"
              f"    ~20x per pivot while HiGHS calls it once per solve. Every\n"
              f"    SOR/HiGHS ratio from this host is biased AGAINST SOR, most\n"
              f"    severely on small models, and repeating the run cannot\n"
              f"    detect it. Do not publish these timings.\n",
              file=sys.stderr)

    rows: list[Result] = []
    # The build metadata lives beside the original executable, while every
    # command below runs the frozen copy. Record both identities explicitly.
    environment = environment_record(args, solvers, models, source_exe)
    environment["executable"] = str(exe.resolve())
    environment["executable_sha256"] = frozen_hash if frozen_hash else sha256_file(exe)
    environment["source_executable"] = str(source_exe.resolve())
    environment["source_executable_sha256"] = source_hash
    environment["checker_executable_sha256"] = frozen_checker_hash
    fh = args.jsonl.open("x") if args.jsonl else None
    if fh:
        fh.write(json.dumps(environment) + "\n")
        fh.flush()
    rng = random.Random(args.seed)
    try:
        for i, m in enumerate(models, 1):
            print(f"[{i}/{len(models)}] {m.name}", flush=True)
            order = list(solvers)
            rng.shuffle(order)
            for s in order:
                for _ in range(args.warmups):
                    if source_hash is not None:
                        verify_executable(source_exe, source_hash)
                        verify_executable(exe, frozen_hash)
                        verify_executable(source_checker, source_checker_hash)
                        verify_executable(frozen_checker, frozen_checker_hash)
                    warm = dispatch(s, m, args.time_limit, args.tol, exe,
                                    args.method, args.basis_update,
                                    args.max_iter, args.cpu, args.pricing,
                                    args.dual_cost_perturbation,
                                    args.sor_arg, args.relax_integrality,
                                    args.small_matrix_value, args.seed,
                                    args.highs_source)
                    if source_hash is not None:
                        verify_executable(source_exe, source_hash)
                        verify_executable(exe, frozen_hash)
                        verify_executable(source_checker, source_checker_hash)
                        verify_executable(frozen_checker, frozen_checker_hash)
                    if warm.status.lower() == "unavailable":
                        break
                measured: list[Result] = []
                for rep in range(args.repetitions):
                    if source_hash is not None:
                        verify_executable(source_exe, source_hash)
                        verify_executable(exe, frozen_hash)
                        verify_executable(source_checker, source_checker_hash)
                        verify_executable(frozen_checker, frozen_checker_hash)
                    solution_out = None
                    if (args.solutions_dir is not None and
                            s.lower().startswith("sor")):
                        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", m.name)
                        safe_solver = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
                        solution_out = (args.solutions_dir /
                                        f"{i:04d}-{safe_model}--{safe_solver}--"
                                        f"rep{rep}.sol")
                    r = dispatch(s, m, args.time_limit, args.tol, exe,
                                 args.method, args.basis_update,
                                 args.max_iter, args.cpu, args.pricing,
                                 args.dual_cost_perturbation, args.sor_arg,
                                 args.relax_integrality,
                                 args.small_matrix_value, args.seed,
                                 args.highs_source, solution_out)
                    if source_hash is not None:
                        verify_executable(source_exe, source_hash)
                        verify_executable(exe, frozen_hash)
                        verify_executable(source_checker, source_checker_hash)
                        verify_executable(frozen_checker, frozen_checker_hash)
                    r.repetition = rep
                    r.model_sha256 = sha256_file(m)
                    measured.append(r)
                    if fh:
                        record = asdict(r)
                        record["record"] = "run"
                        fh.write(json.dumps(record) + "\n")
                        fh.flush()
                    if r.status.lower() == "unavailable":
                        break
                r = aggregate_repetitions(measured)
                rows.append(r)
                if fh:
                    # The raw repetitions above are the evidence; this is the
                    # scored representative, and the only place the median and
                    # its MAD are recorded. Reading a sweep must not require
                    # re-deriving the aggregation the summary actually used.
                    agg = asdict(r)
                    agg["record"] = "aggregate"
                    fh.write(json.dumps(agg) + "\n")
                    fh.flush()
                print(f"    {s:<20} {r.status:<14} "
                      f"obj={fmt_obj(r.objective):>14} {fmt_time(r.seconds):>9}"
                      + (f"  ({r.error})" if r.error else ""), flush=True)
    finally:
        if fh:
            fh.close()
        if binary_temp is not None:
            binary_temp.cleanup()

    solu = parse_solu_file(args.solu) if args.solu is not None else None
    summary = report(rows, solvers, args.sgm_shift, args.obj_abs_tol,
                     args.obj_rel_tol, args.time_limit, args.reference, solu)
    if args.jsonl:
        print(f"\nJSONL: {args.jsonl}")
    failed = summary["mismatches"] > 0
    failed = failed or (summary["unchecked"] > 0 and not args.allow_unchecked)
    failed = failed or (summary["unavailable"] > 0 and not args.allow_unavailable)
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) == 8 and sys.argv[1] == "--_highs-worker":
        sys.exit(highs_worker(Path(sys.argv[2]), float(sys.argv[3]),
                              float(sys.argv[4]), sys.argv[5] == "1",
                              float(sys.argv[6]) if sys.argv[6] else None,
                              int(sys.argv[7])))
    sys.exit(main())
