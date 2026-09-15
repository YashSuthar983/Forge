#!/usr/bin/env python3
"""Run a suite under the full §3.4 claim protocol, or refuse to run at all.

    scripts/claim_run.py --manifest benchmarks/manifests/netlib-v1.json \
        --exe build/sor_solve --tol 1e-7 --time-limit 60 --cpu 10

A "claim run" is a measurement whose numbers may be published. Everything here
exists because a number that looked publishable was published three times in
one session and was wrong every time: once from an unpinned single repetition,
once from a host whose kernel had silently dropped the TSC clocksource, and
once from the wrong summary statistic. Each of those would have been refused by
one of the checks below, before any time was spent measuring.

The script does four things and nothing else:

  1. PREFLIGHT. Refuses on a dirty tree, a non-tsc clocksource, a non-performance
     governor, low MemAvailable, swap activity, a busy SMT sibling, or competing
     load. A refusal is the point: a rejected session costs ten minutes, a
     published wrong number costs a day.
  2. MANIFEST VERIFICATION. Re-checksums every model and confirms the count and
     the scoring-eligibility flags before running, so a claim always names the
     exact corpus it was measured on.
  3. MEASUREMENT. One warm-up and five measured repetitions, pinned, with an
     automatic single rerun of any model/solver pair whose MAD/median exceeds
     5%, and rejection of the session if it stays noisy.
  4. SCORING. Through compare.py's public claim gate -- the same code a report
     quotes -- never a locally re-derived statistic.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "compare.py").exists())


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


compare = _load("sor_compare", "scripts/compare.py")
protocol = _load("sor_claim_protocol", "scripts/claim_protocol.py")

MIN_MEM_AVAILABLE_GIB = 12.0
MAX_COMPETING_LOAD = 0.5          # loadavg(1m) excluding our own run
NOISE_BAND = 0.05                 # MAD/median
MAX_SMT_SIBLING_BUSY_PERCENT = 5.0


# --------------------------------------------------------------------------
# 1. Preflight
# --------------------------------------------------------------------------
def read_first(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def meminfo_gib(key: str) -> float | None:
    txt = read_first("/proc/meminfo")
    if not txt:
        return None
    m = re.search(rf"^{key}:\s+(\d+) kB", txt, re.M)
    return int(m.group(1)) / (1024 * 1024) if m else None


def vmstat_value(key: str) -> int | None:
    txt = read_first("/proc/vmstat")
    if not txt:
        return None
    m = re.search(rf"^{key} (\d+)", txt, re.M)
    return int(m.group(1)) if m else None


def smt_siblings(cpu: int) -> list[int]:
    txt = read_first(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    if not txt:
        return []
    out: list[int] = []
    for part in txt.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [c for c in out if c != cpu]


def cpu_busy_percent(cpu: int, window: float = 0.4) -> float | None:
    """Rough per-CPU utilisation over a short window, from /proc/stat."""
    def sample() -> tuple[int, int] | None:
        txt = read_first("/proc/stat")
        if not txt:
            return None
        m = re.search(rf"^cpu{cpu} (.+)$", txt, re.M)
        if not m:
            return None
        f = [int(x) for x in m.group(1).split()]
        idle = f[3] + (f[4] if len(f) > 4 else 0)
        return sum(f), idle
    a = sample()
    if a is None:
        return None
    time.sleep(window)
    b = sample()
    if b is None:
        return None
    dt, di = b[0] - a[0], b[1] - a[1]
    return 100.0 * (1.0 - di / dt) if dt > 0 else None


def competing_processes(threshold: float = 5.0) -> list[dict] | None:
    """Processes above `threshold` percent CPU, excluding this run's own tree.

    Load average is a one-minute smoothed number: it is blind to a browser
    that just woke up and it lags a process that just exited. A claim protocol
    needs to know what is running NOW.
    """
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pcpu,pid,comm", "--sort=-pcpu"],
            capture_output=True, text=True, timeout=20)
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        out = proc.stdout
    except (OSError, subprocess.SubprocessError):
        return None
    mine = {os.getpid(), os.getppid()}
    busy = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pcpu, pid = float(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pcpu < threshold or pid in mine:
            continue
        busy.append({"pcpu": pcpu, "pid": pid, "comm": parts[2].strip()})
    return busy


def throttle_counters(
        root: Path = Path("/sys/devices/system/cpu")) -> dict | None:
    """Per-core thermal/power throttle counters, where the kernel exposes them.

    A throttling EVENT is a change in these between the start and end of a
    session. A single reading cannot tell you whether the CPU throttled during
    the run, which is the thing §3.4 actually forbids.
    """
    paths: list[Path] = []
    for name in ("core_throttle_count", "package_throttle_count"):
        paths.extend(sorted(root.glob(f"cpu*/thermal_throttle/{name}")))
    if not paths:
        return None
    out: dict = {}
    for path in paths:
        try:
            value = path.read_text().strip()
        except OSError:
            # One unreadable counter makes the state unknowable: silently
            # dropping it could hide the only counter that changed.
            return None
        if not value.isdigit():
            return None
        out[str(path)] = int(value)
    return out or None


def host_snapshot(cpu: int | None) -> dict:
    """The volatile host state, sampled at one instant."""
    return {
        "loadavg": os.getloadavg(),
        "mem_available_gib": meminfo_gib("MemAvailable"),
        "pswpin": vmstat_value("pswpin"),
        "pswpout": vmstat_value("pswpout"),
        "temperature_c": cpu_temperature(),
        "throttle": throttle_counters(),
        "competing_processes": competing_processes(),
        "governor": read_first(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
        if cpu is not None else None,
    }


def snapshot_delta(before: dict, after: dict) -> dict:
    """What CHANGED during the session -- swap traffic and throttling events."""
    d: dict = {}
    for key in ("pswpin", "pswpout"):
        a, b = before.get(key), after.get(key)
        d[key + "_delta"] = (b - a) if (a is not None and b is not None) else None
    tb, ta = before.get("throttle"), after.get("throttle")
    if not isinstance(tb, dict) or not tb or not isinstance(ta, dict) or not ta \
            or set(tb) != set(ta):
        d["throttle_events"] = None
    else:
        d["throttle_events"] = {
            k: ta[k] - tb[k] for k in ta if ta[k] - tb[k] > 0}
    d["max_temperature_c"] = max(
        [v for v in (before.get("temperature_c"), after.get("temperature_c"))
         if v is not None] or [None])
    return d


def verify_build(exe: Path, info: dict) -> list[str]:
    """The primary comparison build is Release + -O3 + NDEBUG + -march=native."""
    problems: list[str] = []
    if not exe.exists():
        return [f"executable {exe} does not exist"]
    flags = info.get("cmake_flags") or {}
    if not flags:
        return [f"no CMakeCache.txt beside {exe}: build configuration unknown"]
    build_type = (flags.get("CMAKE_BUILD_TYPE") or "").strip()
    if build_type.lower() != "release":
        problems.append(f"CMAKE_BUILD_TYPE={build_type!r}, expected Release")
    release_flags = (flags.get("CMAKE_CXX_FLAGS_RELEASE") or "")
    if "-O3" not in release_flags:
        problems.append(f"release flags {release_flags!r} lack -O3")
    if "NDEBUG" not in release_flags:
        problems.append(f"release flags {release_flags!r} lack NDEBUG")
    if (flags.get("SOR_NATIVE_ARCH") or "").upper() != "ON":
        problems.append(
            f"SOR_NATIVE_ARCH={flags.get('SOR_NATIVE_ARCH')!r}: the primary "
            f"comparison build is -march=native (plan 3.3)")
    return problems


def cpu_temperature() -> float | None:
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        t = read_first(str(zone / "temp"))
        if t and t.isdigit():
            v = int(t) / 1000.0
            if 20.0 < v < 120.0:
                return v
    return None


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = run("git", "status", "--porcelain")
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD")}


def compiler_info(exe: Path) -> dict:
    cc = shutil.which("c++") or shutil.which("g++")
    ver = None
    if cc:
        try:
            ver = subprocess.run([cc, "--version"], capture_output=True,
                                 text=True).stdout.splitlines()[0]
        except (OSError, IndexError):
            pass
    flags = native = None
    cache = None
    for parent in (exe.parent,):
        c = parent / "CMakeCache.txt"
        if c.exists():
            cache = c.read_text()
            break
    if cache:
        def grab(key):
            m = re.search(rf"^{key}:[^=]*=(.*)$", cache, re.M)
            return m.group(1) if m else None
        flags = {k: grab(k) for k in
                 ("CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS",
                  "CMAKE_CXX_FLAGS_RELEASE", "SOR_NATIVE_ARCH",
                  "SOR_DETERMINISTIC_FP", "CMAKE_CXX_COMPILER")}
        native = grab("SOR_NATIVE_ARCH")
    return {"compiler": ver, "cmake_flags": flags, "native_arch": native}


def sha256_of(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def wheel_highs_info(python: str) -> tuple[dict, list[str]]:
    info: dict = {"python": python, "version": None, "module": None,
                  "module_sha256": None, "core": None,
                  "core_sha256": None, "distribution_version": None}
    problems: list[str] = []
    py = Path(python)
    if not py.is_file() or not os.access(py, os.X_OK):
        return info, [f"wheel interpreter {python!r} is not a regular executable file"]
    try:
        code = (
            "import importlib.metadata as m,json,highspy,highspy._core as c;"
            "h=highspy.Highs();"
            "print(json.dumps({'version':h.version(),"
            "'module':highspy.__file__,"
            "'core':c.__file__,"
            "'distribution_version':m.version('highspy')}))")
        p = subprocess.run([python, "-c", code], capture_output=True,
                           text=True, timeout=60)
        if p.returncode != 0:
            problems.append(f"wheel highspy probe exited {p.returncode}: {p.stderr[:160]}")
            return info, problems
        payload = json.loads(p.stdout)
        info.update(payload)
        module = Path(str(info["module"]))
        info["module_sha256"] = sha256_of(module)
        core = Path(str(info["core"]))
        info["core_sha256"] = sha256_of(core)
        if not re.fullmatch(r"v?\d+\.\d+\.\d+(?:[-+].+)?", str(info["version"])):
            problems.append(f"wheel reported an unrecognised HiGHS version {info['version']!r}")
        if str(info["distribution_version"]).lstrip("v") != str(info["version"]).lstrip("v"):
            problems.append("highspy distribution and runtime versions disagree")
        if info["module_sha256"] is None:
            problems.append("wheel highspy module could not be hashed")
        if info["core_sha256"] is None:
            problems.append("wheel highspy native extension could not be hashed")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError,
            KeyError, TypeError) as e:
        problems.append(f"wheel highspy configuration is unknown: {type(e).__name__}: {e}")
    return info, problems


def source_highs_info(executable: Path, sor_env: dict | None = None
                      ) -> tuple[dict, list[str]]:
    info: dict = {"path": str(executable), "sha256": None, "version": None,
                  "identity": None}
    problems: list[str] = []
    if not executable.exists():
        return info, [f"--highs-source {executable} does not exist"]
    if not executable.is_file():
        return info, [f"--highs-source {executable} is not a regular file"]
    if not os.access(executable, os.X_OK):
        return info, [f"--highs-source {executable} is not executable"]
    try:
        with executable.open("rb") as fh:
            magic = fh.read(4)
    except OSError as e:
        return info, [f"--highs-source cannot be read: {e}"]
    native_magics = (b"\x7fELF", b"MZ", b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
                     b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe")
    if not any(magic.startswith(prefix) for prefix in native_magics):
        problems.append(
            "--highs-source is not a native executable produced by a source build")
    info["sha256"] = sha256_of(executable)
    if info["sha256"] is None:
        problems.append("source HiGHS executable could not be hashed")
    try:
        probe = subprocess.run([str(executable), "--identity"],
                               capture_output=True, text=True, timeout=30)
        lines = [line for line in probe.stdout.splitlines() if line.strip()]
        if probe.returncode != 0 or probe.stderr.strip() or len(lines) != 1:
            problems.append(
                "--highs-source did not produce one clean native-runner identity record")
            return info, problems
        payload = json.loads(lines[0])
        if not isinstance(payload, dict) or \
                payload.get("schema") != "sor-highs-source-runner-v1" or \
                payload.get("kind") != "identity":
            problems.append("--highs-source is not the native source-HiGHS API runner")
            return info, problems
        identity = payload.get("build_identity")
        info["identity"] = identity
        info["version"] = payload.get("highs_version")
        if not isinstance(identity, dict) or identity.get("kind") != \
                "sor-highs-source-api-runner" or identity.get("runner_version") != 1:
            problems.append("source runner build identity is missing or unsupported")
            return info, problems
        if identity.get("build_type") != "Release":
            problems.append("source HiGHS runner is not Release")
        if identity.get("optimization") != "-O3":
            problems.append("source HiGHS runner is not compiled with -O3")
        if identity.get("ndebug") is not True:
            problems.append("source HiGHS runner lacks NDEBUG")
        if identity.get("native_arch") is not True:
            problems.append("source HiGHS runner lacks -march=native")
        if identity.get("highs_source_dirty") is not False:
            problems.append("source HiGHS checkout is dirty or its state is unknown")
        commit = identity.get("highs_source_commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            problems.append("source HiGHS pinned commit is missing or invalid")
        githash = identity.get("highs_githash")
        if not isinstance(githash, str) or not githash or \
                not isinstance(commit, str) or not commit.startswith(githash):
            problems.append("linked HiGHS githash does not match its pinned source commit")
        if not re.fullmatch(r"v?\d+\.\d+\.\d+(?:[-+].+)?",
                            str(info.get("version"))):
            problems.append(f"source runner reported invalid version {info.get('version')!r}")
        sor_compiler = ((sor_env or {}).get("cmake_flags") or {}).get(
            "CMAKE_CXX_COMPILER")
        runner_compiler = identity.get("compiler")
        if not sor_compiler or not runner_compiler:
            problems.append("source/SOR compiler configuration is unknown")
        elif Path(str(sor_compiler)).resolve() != Path(str(runner_compiler)).resolve():
            problems.append(
                f"source HiGHS compiler {runner_compiler} does not match SOR {sor_compiler}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError,
            TypeError, KeyError) as e:
        problems.append(f"source HiGHS probe failed: {type(e).__name__}: {e}")
    return info, problems


def preflight(args, exe: Path) -> tuple[dict, list[str]]:
    """Everything a published number depends on, and every reason to refuse."""
    env: dict = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "git": git_state(),
        "executable": str(exe),
        "executable_sha256": sha256_of(exe),
        "clocksource": read_first(
            "/sys/devices/system/clocksource/clocksource0/current_clocksource"),
        "available_clocksource": read_first(
            "/sys/devices/system/clocksource/clocksource0/available_clocksource"),
        "governor": read_first(
            f"/sys/devices/system/cpu/cpu{args.cpu}/cpufreq/scaling_governor")
        if args.cpu is not None else None,
        "smt_active": read_first("/sys/devices/system/cpu/smt/active"),
        "cpu_count": os.cpu_count(),
        "affinity": args.cpu,
        "taskset": shutil.which("taskset"),
        "mem_available_gib": meminfo_gib("MemAvailable"),
        "swap_total_gib": meminfo_gib("SwapTotal"),
        "pswpin": vmstat_value("pswpin"),
        "pswpout": vmstat_value("pswpout"),
        "temperature_c": cpu_temperature(),
        "loadavg": os.getloadavg(),
        "throttle_before": throttle_counters(),
        "competing_processes": competing_processes(),
        "seed": args.seed,
        "tolerance": args.tol,
        "time_limit_s": args.time_limit,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "noise_band": NOISE_BAND,
    }
    env.update(compiler_info(exe))
    wheel_info, wheel_problems = wheel_highs_info(args.python)
    env["wheel_highs"] = wheel_info
    env["wheel_highs_version"] = wheel_info.get("version")
    source_info: dict = {"path": None, "sha256": None, "version": None}
    source_problems: list[str] = []
    if getattr(args, "highs_source", None) is not None:
        source_info, source_problems = source_highs_info(
            Path(args.highs_source), env)
    env["source_highs"] = source_info
    env["source_highs_sha256"] = source_info.get("sha256")
    env["source_highs_version"] = source_info.get("version")
    env["development_overrides"] = [
        name for name in ("allow_dirty", "allow_noisy")
        if getattr(args, name, False)]
    if args.cpu is not None:
        sib = smt_siblings(args.cpu)
        env["smt_siblings"] = sib
        env["sibling_busy_percent"] = {c: cpu_busy_percent(c) for c in sib}

    refusals: list[str] = []
    if not args.allow_dirty and env["git"]["dirty"] is not False:
        state = "unknown" if env["git"]["dirty"] is None else "dirty"
        refusals.append(
            f"working tree state is {state}; a claim must name an exact commit")
    if env["clocksource"] != "tsc":
        refusals.append(
            f"clocksource is {env['clocksource']!r}, not 'tsc' -- clock_gettime "
            f"is ~50x more expensive and SOR reads it far more often than HiGHS, "
            f"so every ratio is biased and repetitions cannot detect it")
    if args.cpu is None:
        refusals.append("no CPU affinity requested; claim runs require --cpu")
    elif env["taskset"] is None:
        refusals.append("taskset is unavailable; CPU affinity cannot be enforced")
    else:
        try:
            if args.cpu not in os.sched_getaffinity(0):
                refusals.append(f"cpu{args.cpu} is outside this process's allowed affinity")
        except (AttributeError, OSError):
            refusals.append("current CPU affinity state is unknown")
    if env["governor"] != "performance":
        refusals.append(f"cpufreq governor is {env['governor']!r}, not 'performance'")
    mem = env["mem_available_gib"]
    if mem is None:
        refusals.append("MemAvailable is unknown")
    elif mem < MIN_MEM_AVAILABLE_GIB:
        refusals.append(f"MemAvailable {mem:.1f} GiB < {MIN_MEM_AVAILABLE_GIB} GiB")
    for cpu, busy in (env.get("sibling_busy_percent") or {}).items():
        if busy is None:
            refusals.append(f"SMT sibling cpu{cpu} activity is unknown")
        elif busy > MAX_SMT_SIBLING_BUSY_PERCENT:
            refusals.append(
                f"SMT sibling cpu{cpu} is {busy:.1f}% busy; protocol maximum "
                f"is {MAX_SMT_SIBLING_BUSY_PERCENT:.1f}%")
    load1 = env["loadavg"][0]
    if load1 > MAX_COMPETING_LOAD + 1.0:
        refusals.append(f"1-minute load average {load1:.2f} indicates competing work")
    if env.get("competing_processes") is None:
        refusals.append("competing process state is unknown")
    else:
        for proc in env["competing_processes"]:
            refusals.append(
                f"competing process above 5% CPU: {proc['comm']} "
                f"(pid {proc['pid']}, {proc['pcpu']:.0f}%)")
    if env.get("throttle_before") is None:
        refusals.append(
            "thermal/power throttle counters are unsupported, missing, or unreadable")
    build_problems = verify_build(exe, env)
    refusals.extend(build_problems)
    # Item 14: the wheel is the SECONDARY lane. Without a source-built HiGHS
    # to cross-check against, reference generation is incomplete and no number
    # measured against it may be published.
    if getattr(args, "highs_source", None) is None:
        refusals.append(
            "no source-built HiGHS (--highs-source): the official wheel is the "
            "secondary reproducibility lane, so oracle/reference generation is "
            "INCOMPLETE and this run cannot support a published claim")
    else:
        refusals.extend(source_problems)
    refusals.extend(wheel_problems)
    source_version = source_info.get("version")
    wheel_version = wheel_info.get("version")
    if source_version is not None and wheel_version is not None and \
            str(source_version).lstrip("v") != str(wheel_version).lstrip("v"):
        refusals.append(
            f"source HiGHS version {source_version} does not match wheel {wheel_version}")
    if args.repetitions != 5 or args.warmups != 1:
        refusals.append(
            f"claim protocol is 1 warm-up + 5 repetitions; got "
            f"{args.warmups} + {args.repetitions}")
    return env, refusals


# --------------------------------------------------------------------------
# 2. Manifest verification
# --------------------------------------------------------------------------
def verify_manifest(path: Path) -> tuple[list[Path], list[str], dict]:
    doc = json.loads(path.read_text())
    problems: list[str] = []
    models: list[Path] = []
    eligible = 0
    for entry in doc["models"]:
        p = ROOT / entry["path"]
        if not p.exists():
            problems.append(f"missing: {entry['path']}")
            continue
        got = sha256_of(p)
        if got != entry["sha256"]:
            problems.append(
                f"checksum changed: {entry['name']} "
                f"expected {entry['sha256'][:12]} got {(got or '?')[:12]}")
            continue
        if entry.get("scoring_eligible"):
            eligible += 1
        models.append(p)
    if len(doc["models"]) != doc.get("model_count"):
        problems.append(
            f"model_count {doc.get('model_count')} != {len(doc['models'])} entries")
    if eligible != doc.get("scoring_eligible_count"):
        problems.append(
            f"scoring_eligible_count {doc.get('scoring_eligible_count')} "
            f"!= {eligible} verified")
    if eligible == 0:
        problems.append(
            "no model is scoring-eligible: this manifest has no reference "
            "answers, so a run against it cannot support a claim")
    return models, problems, doc


# --------------------------------------------------------------------------
# 3/4. Measure and score
# --------------------------------------------------------------------------
def sweep(args, models: list[Path], exe: Path, out: Path,
          *, solvers: str | None = None,
          solutions_dir: Path | None = None) -> int:
    cmd = [args.python, str(ROOT / "scripts" / "compare.py")]
    cmd += [str(m) for m in models]
    cmd += ["--solvers", solvers or args.claim_solvers, "--tol", str(args.tol),
            "--time-limit", str(args.time_limit), "--exe", str(exe),
            "--warmups", str(args.warmups),
            "--repetitions", str(args.repetitions),
            "--seed", str(args.seed), "--jsonl", str(out), "--allow-unchecked"]
    if args.highs_source is not None:
        cmd += ["--highs-source", str(args.highs_source)]
    if solutions_dir is not None:
        cmd += ["--solutions-dir", str(solutions_dir)]
    if args.cpu is not None:
        cmd += ["--cpu", str(args.cpu)]
    if args.method is not None:
        cmd += ["--method", args.method]
    cmd += ["--pricing", args.pricing, "--basis-update", args.basis_update,
            "--dual-cost-perturbation", str(args.dual_cost_perturbation)]
    if args.max_iter is not None:
        cmd += ["--max-iter", str(args.max_iter)]
    if args.relax_integrality:
        cmd.append("--relax-integrality")
    if args.small_matrix_value is not None:
        cmd += ["--small-matrix-value", str(args.small_matrix_value)]
    cmd += [f"--sor-arg={value}" for value in args.sor_arg]
    return subprocess.run(cmd, cwd=ROOT).returncode


SCHEMA_VERSION = protocol.PROTOCOL_VERSION


def independent_check(checker: Path, model: Path, measured: compare.Result,
                      tol: float) -> dict:
    """Re-verify one model with SOR's INDEPENDENT checker, on the ORIGINAL model.

    A proof string in the solver's own stdout is the solver's opinion of
    itself. `sor_check` reloads the original unscaled MPS, recomputes row and
    column residuals and the objective from the solution file, and exits
    nonzero if the claim does not hold. That is what "independently checked"
    has to mean before a number is published.
    """
    out = {"checked": False, "checker_status": None, "residual_ok": None,
           "certificate_ok": None, "model_sha256": sha256_of(model),
           "solution_file": measured.solution_file,
           "solution_sha256": measured.solution_sha256,
           "measured_command": measured.command, "checker_command": None,
           "output": None, "detail": None}
    if not checker.is_file() or not os.access(checker, os.X_OK):
        out["detail"] = f"checker {checker} not built"
        return out
    if not measured.solution_file or not measured.solution_sha256:
        out["detail"] = "selected measured repetition has no captured solution"
        return out
    sol = Path(measured.solution_file)
    if not sol.is_file():
        out["detail"] = f"measured solution {sol} is missing"
        return out
    if sha256_of(sol) != measured.solution_sha256:
        out["detail"] = "measured solution SHA-256 no longer matches its run record"
        return out
    cmd = measured.command or []
    if "--solution-out" not in cmd:
        out["detail"] = "measured command did not request the recorded solution"
        return out
    try:
        recorded = Path(cmd[cmd.index("--solution-out") + 1]).resolve()
    except (IndexError, ValueError, OSError):
        out["detail"] = "measured command has a malformed --solution-out"
        return out
    if recorded != sol.resolve():
        out["detail"] = "measured command and recorded solution path disagree"
        return out
    try:
        checker_command = [str(checker), str(model), str(sol),
                           "--tol", str(tol)]
        # These flags alter how the original file becomes the mathematical
        # model. Preserve them in command order (sor_solve's last occurrence
        # wins) while intentionally ignoring algorithm choices: the checker
        # verifies the recorded point/certificate and never runs an engine.
        index = 0
        while index < len(cmd):
            arg = cmd[index]
            if arg in ("--relax-integrality", "--fixed-mps", "--free-mps"):
                checker_command.append(arg)
            elif arg == "--small-matrix-value":
                if index + 1 >= len(cmd):
                    out["detail"] = "measured command has malformed --small-matrix-value"
                    return out
                checker_command.extend((arg, cmd[index + 1]))
                index += 1
            index += 1
        out["checker_command"] = checker_command
        c = subprocess.run(checker_command,
                           capture_output=True, text=True, timeout=600)
        text = (c.stdout or "") + "\n" + (c.stderr or "")
        out["output"] = text.strip()
        out["checked"] = True
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        has_failure = any(line.startswith("FAIL") for line in lines)
        verified_marker = bool(lines) and lines[-1] == "VERIFIED"
        accepted = c.returncode == 0 and verified_marker and not has_failure
        out["checker_status"] = "VERIFIED" if accepted else "REJECTED"
        out["residual_ok"] = accepted
        out["certificate_ok"] = accepted
        out["detail"] = text.strip().splitlines()[-1] if text.strip() else None
    except (OSError, subprocess.SubprocessError) as e:
        out["detail"] = f"{type(e).__name__}: {e}"
    return out


class BaselineError(ValueError):
    pass


def load_baselines(path: Path | None, **requirements) -> protocol.BaselineData:
    """Validate the baseline completely before returning any numeric value."""
    if path is None:
        raise BaselineError("no pinned baseline supplied")
    data, problems = protocol.validate_baseline(path, **requirements)
    if problems:
        raise BaselineError("; ".join(problems))
    return data


@dataclass(frozen=True)
class AllowListSnapshot:
    entries: dict[str, dict]
    path: str
    sha256: str
    schema_version: int


def load_allow(suite: str, manifest_instances: set[str],
               path: Path | None = None
               ) -> tuple[AllowListSnapshot | None, list[str]]:
    """Load one fully validated, content-addressed allow-list snapshot."""
    path = path or (ROOT / "benchmarks" / "results" / "gate-allow.json")
    problems: list[str] = []
    if not path.is_file():
        return None, [f"allow-list {path} is missing or not a regular file"]
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return None, [f"allow-list {path} cannot be read: {e}"]
    if not isinstance(doc, dict):
        return None, ["allow-list top level is not an object"]
    schema = doc.get("_schema")
    if not isinstance(schema, dict) or schema.get("version") != 1:
        problems.append("allow-list schema/version is missing or unsupported")
    if not isinstance(suite, str) or not suite.strip() or suite not in doc:
        problems.append(f"allow-list contains no known suite {suite!r}")
    entries = doc.get(suite)
    if not isinstance(entries, dict):
        problems.append(f"allow-list suite {suite!r} is not an object")
        entries = {}
    validated: dict[str, dict] = {}
    for instance, entry in entries.items():
        prefix = f"allow-list {suite}.{instance}"
        if instance not in manifest_instances:
            problems.append(f"{prefix} is not a known manifest instance")
        if not isinstance(entry, dict):
            problems.append(f"{prefix} is not an object")
            continue
        unexpected = sorted(set(entry) - {"reason", "max_work_ratio"})
        if unexpected:
            problems.append(f"{prefix} has unknown field(s) {unexpected}")
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f"{prefix} has no nonempty reason")
        ratio = entry.get("max_work_ratio")
        if ratio is not None and (type(ratio) not in (int, float) or
                                  not math.isfinite(ratio) or ratio <= 0.0):
            problems.append(f"{prefix}.max_work_ratio is not finite and positive")
        if (not unexpected and isinstance(reason, str) and reason.strip() and
                (ratio is None or (type(ratio) in (int, float) and
                                   math.isfinite(ratio) and ratio > 0.0)) and
                instance in manifest_instances):
            validated[instance] = dict(entry)
    digest = sha256_of(path)
    if digest is None:
        problems.append("allow-list could not be hashed")
    if problems:
        return None, problems
    return AllowListSnapshot(validated, str(path.resolve()), digest, 1), []


def load_aggregates(path: Path) -> dict:
    grouped: dict[tuple[str, str], list[compare.Result]] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} line {line_no}: invalid JSON: {e}") from e
        if not isinstance(rec, dict):
            raise ValueError(f"{path} line {line_no}: record is not an object")
        if rec.get("record") != "aggregate":
            continue
        rec = dict(rec)
        rec.pop("record", None)
        try:
            result = compare.Result(**rec)
        except (KeyError, TypeError) as e:
            raise ValueError(f"{path} line {line_no}: invalid aggregate: {e}") from e
        if not result.instance or not result.solver:
            raise ValueError(f"{path} line {line_no}: aggregate lacks instance/solver")
        grouped.setdefault((result.instance, result.solver), []).append(result)
    by: dict = {}
    for (instance, solver), results in grouped.items():
        selected = [r for r in results
                    if not r.claim_superseded and r.claim_selected is not False]
        if len(selected) != 1:
            raise ValueError(
                f"{path}: {instance}/{solver} has {len(selected)} selected "
                f"aggregate records (expected exactly one)")
        by.setdefault(instance, {})[solver] = selected[0]
    return by


def noisy_pairs(by: dict) -> list[tuple[str, str]]:
    return sorted((inst, solver) for inst, pair in by.items()
                  for solver, result in pair.items() if result.noisy)


def mark_originals_superseded(path: Path,
                              pairs: set[tuple[str, str]]) -> None:
    """Mark the original noisy aggregate in place before appending replacements."""
    rewritten: list[str] = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        key = (str(rec.get("instance")), str(rec.get("solver")))
        if rec.get("record") == "aggregate" and key in pairs and \
                not rec.get("claim_source"):
            rec["claim_superseded"] = True
            rec["claim_selected"] = False
            rec["claim_source"] = "initial"
        rewritten.append(json.dumps(rec))
    tmp = path.with_name(path.name + ".rewrite")
    tmp.write_text("\n".join(rewritten) + "\n")
    os.replace(tmp, path)


def highs_lanes_agree(by: dict, instances: set[str], abs_tol: float,
                      rel_tol: float, *, source_version: str | None = None,
                      wheel_version: str | None = None) -> list[str]:
    problems: list[str] = []
    for inst in sorted(instances):
        pair = by.get(inst, {})
        source = pair.get("highs-source")
        wheel = pair.get("highs-wheel")
        sok, swhy = compare.reference_is_valid(source)
        wok, wwhy = compare.reference_is_valid(wheel)
        if not sok:
            problems.append(f"{inst}: source HiGHS invalid ({swhy})")
            continue
        if not wok:
            problems.append(f"{inst}: wheel HiGHS invalid ({wwhy})")
            continue
        source_config_problem = protocol.highs_configuration_problem(
            source.configuration)
        wheel_config_problem = protocol.highs_configuration_problem(
            wheel.configuration)
        if source_config_problem or wheel_config_problem:
            problems.append(
                f"{inst}: source/wheel configuration is unknown "
                f"(source: {source_config_problem or 'ok'}; "
                f"wheel: {wheel_config_problem or 'ok'})")
            continue
        if source.configuration != wheel.configuration:
            problems.append(f"{inst}: source/wheel configuration mismatch")
            continue
        if not isinstance(source.build_identity, dict) or \
                source.build_identity.get("kind") != "sor-highs-source-api-runner":
            problems.append(f"{inst}: source runner build identity is missing")
        if source_version is not None and source.solver_version != source_version:
            problems.append(
                f"{inst}: source row version {source.solver_version!r} "
                f"does not match preflight {source_version!r}")
        if wheel_version is not None and wheel.solver_version != wheel_version:
            problems.append(
                f"{inst}: wheel row version {wheel.solver_version!r} "
                f"does not match preflight {wheel_version!r}")
        if source.status.strip().lower() != wheel.status.strip().lower():
            problems.append(
                f"{inst}: source/wheel status mismatch "
                f"{source.status!r} vs {wheel.status!r}")
        elif not compare.objectives_agree(source.objective, wheel.objective,
                                          abs_tol, rel_tol):
            problems.append(
                f"{inst}: source/wheel objective mismatch "
                f"{source.objective!r} vs {wheel.objective!r}")
    return problems


def append_claim_record(path: Path, role: str, **payload) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps({"record": role,
                             "claim_schema_version": SCHEMA_VERSION,
                             "claim_role": role, **payload}) + "\n")


def fmt_optional(value: float | None, digits: int = 4) -> str:
    return ("n/a" if not isinstance(value, (int, float)) or
            not math.isfinite(value) else f"{value:.{digits}f}")


def apply_protocol_outcome(gate: compare.ClaimGateResult,
                           incompleteness: list[str]) -> None:
    gate.incompleteness.extend(incompleteness)
    if gate.incompleteness:
        gate.status = "INCOMPLETE"
    elif gate.failures:
        gate.status = "FAIL"
    else:
        gate.status = "PASS"


def evaluate_baseline_eligibility(
        by: dict, candidate: str, expected: set[str],
        candidate_checks: dict[str, tuple[bool, str]],
        *, lane_problems: list[str], protocol_problems: list[str]
        ) -> dict:
    """Decide whether a run is valid baseline evidence, independent of speed."""
    correctness: list[str] = []
    evidence: list[str] = list(lane_problems) + list(protocol_problems)
    actual = set(by)
    if actual != expected:
        evidence.append(
            f"aggregate coverage mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    wanted_lanes = {candidate, "highs-source", "highs-wheel"}
    for inst in sorted(expected):
        pair = by.get(inst, {})
        missing = wanted_lanes - set(pair)
        unexpected = set(pair) - wanted_lanes
        if missing:
            evidence.append(f"{inst}: missing aggregate lane(s) {sorted(missing)}")
        if unexpected:
            evidence.append(f"{inst}: unexpected aggregate lane(s) {sorted(unexpected)}")
        measured = pair.get(candidate)
        source = pair.get("highs-source")
        source_ok, source_why = compare.reference_is_valid(source)
        if measured is None or not compare.is_certified_success(measured):
            correctness.append(f"{inst}: candidate status/proof is not certified")
        elif not source_ok:
            evidence.append(f"{inst}: source oracle invalid ({source_why})")
        elif not compare.objectives_agree(
                measured.objective, source.objective, 1e-7, 1e-7):
            correctness.append(f"{inst}: candidate/source objective mismatch")
        checked, why = candidate_checks.get(
            inst, (False, "no measured-solution checker result"))
        if not checked:
            correctness.append(f"{inst}: independent checker rejected ({why})")
        for lane in wanted_lanes:
            result = pair.get(lane)
            if result is not None and result.noisy:
                evidence.append(f"{inst}/{lane}: timing remains noisy")
    status = "FAIL" if correctness else "INCOMPLETE" if evidence else "PASS"
    return {
        "status": status,
        "eligible": status == "PASS",
        "expected_instances": len(expected),
        "correctness_failures": correctness,
        "evidence_failures": evidence,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--cpu", type=int, default=None)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solvers", default="sor:simplex",
                    help="exactly one SOR candidate lane; claim_run adds the "
                         "required highs-source and highs-wheel lanes")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--python", default=str(
        ROOT / "benchmarks" / ".venv-baseline" / "bin" / "python"))
    ap.add_argument("--skip-checker", action="store_true",
                    help="development only; a claim requires independent checking")
    ap.add_argument("--highs-source", type=Path, default=None,
                    help="source-built HiGHS executable for the PRIMARY oracle "
                         "lane. The wheel is the secondary reproducibility "
                         "lane, not a replacement: without this the reference "
                         "is labelled INCOMPLETE.")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="pinned SOR baseline JSONL supplying per-model solver "
                         "time and process wall for the tail and wall rules")
    ap.add_argument(
        "--establish-baseline", action="store_true",
        help="produce correctness/protocol-eligible baseline evidence without "
             "requiring an older baseline; never emits a public PASS")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="development only; never for a published number")
    ap.add_argument("--allow-noisy", action="store_true",
                    help="development only; never for a published number")
    ap.add_argument("--method", choices=("auto", "primal", "dual"), default=None)
    ap.add_argument("--pricing", choices=("choose", "dantzig", "devex", "dse"),
                    default="choose")
    ap.add_argument("--basis-update", choices=("product", "ft"), default="product")
    ap.add_argument("--dual-cost-perturbation", type=float, default=0.0)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--relax-integrality", action="store_true")
    ap.add_argument("--small-matrix-value", type=float, default=None)
    ap.add_argument("--sor-arg", action="append", default=[])
    args = ap.parse_args()

    if not math.isfinite(args.tol) or args.tol <= 0:
        ap.error("--tol must be finite and positive")
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        ap.error("--time-limit must be finite and positive")
    if args.cpu is not None and args.cpu < 0:
        ap.error("--cpu cannot be negative")
    if (not math.isfinite(args.dual_cost_perturbation) or
            args.dual_cost_perturbation < 0):
        ap.error("--dual-cost-perturbation must be finite and nonnegative")
    if args.max_iter is not None and args.max_iter <= 0:
        ap.error("--max-iter must be positive")
    if (args.small_matrix_value is not None and
            (not math.isfinite(args.small_matrix_value) or
             args.small_matrix_value <= 0)):
        ap.error("--small-matrix-value must be finite and positive")

    args.manifest = (args.manifest if args.manifest.is_absolute()
                     else (ROOT / args.manifest)).resolve()
    if args.baseline is not None:
        args.baseline = (args.baseline if args.baseline.is_absolute()
                         else (ROOT / args.baseline)).resolve()
    if args.highs_source is not None:
        args.highs_source = (args.highs_source if args.highs_source.is_absolute()
                             else (ROOT / args.highs_source)).resolve()
    exe = args.exe if args.exe.is_absolute() else ROOT / args.exe
    exe = exe.resolve()
    requested_solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    candidates = [s for s in requested_solvers if s.lower().startswith("sor")]
    solver_problems: list[str] = []
    if len(candidates) != 1:
        solver_problems.append(
            f"claim mode requires exactly one SOR candidate lane; got {candidates}")
        cand = candidates[0] if candidates else "sor:simplex"
    else:
        cand = candidates[0]
    unexpected_solvers = [s for s in requested_solvers
                          if not s.lower().startswith("sor") and
                          s.lower() not in ("highs", "highs-source", "highs-wheel")]
    if unexpected_solvers:
        solver_problems.append(
            f"claim mode does not permit extra solver lanes: {unexpected_solvers}")
    args.claim_solvers = f"{cand},highs-source,highs-wheel"

    # The output is chosen but NOT created, and an existing file is never
    # destroyed: preflight may refuse, and a refusal that has already deleted
    # the previous run's evidence has made things worse, not safer.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (ROOT / "benchmarks" / "results" /
                       f"claim-{args.manifest.stem}-{args.tol:g}-{stamp}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"error: {out} already exists; refusing to overwrite evidence",
              file=sys.stderr)
        return 2
    solutions_dir = out.with_name(out.stem + ".solutions")
    rerun = out.with_name(out.stem + ".rerun.jsonl")
    rerun_solutions = out.with_name(out.stem + ".rerun-solutions")
    for evidence_path in (solutions_dir, rerun, rerun_solutions):
        if evidence_path.exists():
            print(f"error: {evidence_path} already exists; refusing to reuse "
                  "rerun/solution evidence", file=sys.stderr)
            return 2

    print("=" * 72)
    run_kind = "BASELINE ESTABLISHMENT" if args.establish_baseline else "CLAIM RUN"
    print(f"{run_kind}  manifest={args.manifest.name}  tol={args.tol:g}")
    print("=" * 72)

    try:
        models, problems, doc = verify_manifest(args.manifest)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"MANIFEST PROBLEM: {e}", file=sys.stderr)
        return 2
    problems.extend(solver_problems)
    print(f"manifest: {len(models)}/{doc.get('model_count')} models verified, "
          f"{doc.get('scoring_eligible_count')} scoring-eligible")
    for p in problems:
        print(f"  MANIFEST PROBLEM: {p}")
    env, refusals = preflight(args, exe)
    eligible = {e["name"] for e in doc["models"] if e.get("scoring_eligible")}
    eligible_models = [m for m in models if m.name in eligible]
    baseline_data: protocol.BaselineData | None = None
    if not args.establish_baseline:
        try:
            baseline_data = load_baselines(
                args.baseline, expected_instances=eligible,
                manifest_sha256=sha256_of(args.manifest), tolerance=args.tol,
                source_highs_sha256=env.get("source_highs_sha256"),
                source_highs_version=env.get("source_highs_version"),
                wheel_highs_version=env.get("wheel_highs_version"),
                require_accept=True)
        except BaselineError as e:
            refusals.append(f"pinned baseline rejected: {e}")
    manifest_instances = {str(entry.get("name")) for entry in doc["models"]}
    allow_snapshot, allow_problems = load_allow(
        str(doc.get("suite", "")), manifest_instances)
    refusals.extend(allow_problems)
    if allow_snapshot is not None:
        env.update({
            "allow_list_path": allow_snapshot.path,
            "allow_list_sha256": allow_snapshot.sha256,
            "allow_list_schema_version": allow_snapshot.schema_version,
        })
    print(f"host: clocksource={env['clocksource']} governor={env['governor']} "
          f"mem={fmt_optional(env['mem_available_gib'], 1)}GiB "
          f"load={env['loadavg'][0]:.2f} "
          f"temp={env['temperature_c']}")
    print(f"build: native_arch={env.get('native_arch')} commit={(env['git']['commit'] or '?')[:12]} "
          f"dirty={env['git']['dirty']} exe_sha={(env['executable_sha256'] or '?')[:12]}")
    for r in refusals:
        print(f"  REFUSAL: {r}")

    if problems or refusals:
        print("\nCLAIM RUN REFUSED. Nothing was measured.")
        print("A refused session costs minutes; a published wrong number costs a day.")
        return 2

    assert allow_snapshot is not None
    if not args.establish_baseline:
        assert baseline_data is not None
    append_claim_record(
        out, "preflight", status="PASS", host_preflight_passed=True,
        manifest_sha256=sha256_of(args.manifest), manifest=str(args.manifest),
        build_native_arch=(env.get("cmake_flags") or {}).get("SOR_NATIVE_ARCH"),
        candidate_solver=cand,
        run_mode=("baseline_establishment" if args.establish_baseline
                  else "public_claim"), **env)

    before = host_snapshot(args.cpu)
    sweep_rc = sweep(args, eligible_models, exe, out,
                     solutions_dir=solutions_dir)
    if sweep_rc != 0:
        # A partial sweep must never yield a passing subset.
        print(f"\nSWEEP FAILED (exit {sweep_rc}). A nonzero sweep cannot "
              f"produce a claim: the surviving rows are a subset chosen by "
              f"whatever went wrong.", file=sys.stderr)
        append_claim_record(out, "gate", status="INCOMPLETE",
                            failures=[f"sweep exited {sweep_rc}"])
        return 3

    by = load_aggregates(out)
    noisy = noisy_pairs(by)
    still: list[tuple[str, str]] = []
    rerun_incomplete: list[str] = []
    if noisy:
        labels = [f"{inst}/{solver}" for inst, solver in noisy]
        print(f"\n{len(noisy)} noisy pair(s) above {NOISE_BAND:.0%} MAD/median; "
              f"rerunning exactly: {', '.join(labels[:6])}"
              + (" ..." if len(labels) > 6 else ""))
        model_by_name = {m.name: m for m in eligible_models}
        for index, (inst, solver) in enumerate(noisy):
            model = model_by_name.get(inst)
            if model is None:
                rerun_incomplete.append(f"no manifest model for noisy pair {inst}/{solver}")
                continue
            rc = sweep(args, [model], exe, rerun, solvers=solver,
                       solutions_dir=rerun_solutions / f"pair-{index:04d}")
            if rc != 0:
                rerun_incomplete.append(f"rerun {inst}/{solver} exited {rc}")
        if not rerun.is_file():
            rerun_incomplete.append("rerun produced no output artifact")
        if rerun_incomplete:
            append_claim_record(out, "rerun", status="INCOMPLETE",
                                failures=rerun_incomplete, pairs=labels)
            print("\nSESSION REJECTED: " + "; ".join(rerun_incomplete))
            return 3
        try:
            replacement = load_aggregates(rerun)
        except (OSError, ValueError, TypeError, KeyError) as e:
            append_claim_record(out, "rerun", status="INCOMPLETE",
                                failures=[f"rerun artifact unreadable: {e}"],
                                pairs=labels)
            print(f"\nSESSION REJECTED: rerun artifact unreadable: {e}")
            return 3
        actual = {(inst, solver) for inst, pair in replacement.items()
                  for solver in pair}
        requested = set(noisy)
        if actual != requested:
            rerun_incomplete.append(
                f"rerun coverage mismatch: requested {sorted(requested)}, got {sorted(actual)}")
        if rerun_incomplete:
            append_claim_record(out, "rerun", status="INCOMPLETE",
                                failures=rerun_incomplete, pairs=labels)
            print("\nSESSION REJECTED: " + "; ".join(rerun_incomplete))
            return 3
        mark_originals_superseded(out, requested)
        append_claim_record(
            out, "rerun", status="PASS",
            reason="MAD/median above the noise band", pairs=labels,
            rerun_artifact=str(rerun), rerun_sha256=sha256_of(rerun))
        with out.open("a") as fh:
            for line in rerun.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("record") == "run":
                    rec["claim_source"] = "rerun"
                    fh.write(json.dumps(rec) + "\n")
                elif rec.get("record") == "aggregate":
                    rec["claim_superseded"] = False
                    rec["claim_selected"] = True
                    rec["claim_source"] = "rerun"
                    fh.write(json.dumps(rec) + "\n")
        for inst, solver in requested:
            by.setdefault(inst, {})[solver] = replacement[inst][solver]
        still = noisy_pairs({inst: {solver: by[inst][solver]}
                             for inst, solver in requested})
        if still and not args.allow_noisy:
            print(f"\nSESSION REJECTED: {len(still)} pair(s) still noisy after "
                  f"one rerun: {', '.join(f'{i}/{s}' for i, s in still)}")
            print("Per plan 3.4 a host that stays noisy is rejected, not published.")
            return 3

    # ---- host state AFTER the session: swap traffic and throttling events
    # are DELTAS. A single reading cannot say whether the CPU throttled during
    # the run, which is the thing the protocol actually forbids.
    after = host_snapshot(args.cpu)
    delta = snapshot_delta(before, after)
    session_problems: list[str] = []
    if delta.get("pswpin_delta") is None:
        session_problems.append("postflight swap-in state is unknown")
    elif delta.get("pswpin_delta"):
        session_problems.append(f"swapped IN {delta['pswpin_delta']} pages during the run")
    if delta.get("pswpout_delta") is None:
        session_problems.append("postflight swap-out state is unknown")
    elif delta.get("pswpout_delta"):
        session_problems.append(f"swapped OUT {delta['pswpout_delta']} pages during the run")
    if delta.get("throttle_events") is None:
        session_problems.append("postflight throttle state is unknown")
    if delta.get("throttle_events"):
        session_problems.append(
            f"thermal throttling during the run: {delta['throttle_events']}")
    if after.get("competing_processes") is None:
        session_problems.append("postflight process state is unknown")
    else:
        for proc in after["competing_processes"]:
            session_problems.append(
                f"competing process appeared during the run: {proc['comm']} "
                f"({proc['pcpu']:.0f}%)")
    if after.get("mem_available_gib") is None:
        session_problems.append("postflight memory state is unknown")
    elif after["mem_available_gib"] < MIN_MEM_AVAILABLE_GIB:
        session_problems.append(
            f"postflight MemAvailable {after['mem_available_gib']:.1f} GiB "
            f"< {MIN_MEM_AVAILABLE_GIB} GiB")
    if after.get("governor") != "performance":
        session_problems.append(
            f"postflight governor is {after.get('governor')!r}, not 'performance'")
    append_claim_record(out, "postflight",
                        status="PASS" if not session_problems else "INCOMPLETE",
                        host_before=before, host_after=after, delta=delta,
                        session_problems=session_problems)
    for sp in session_problems:
        print(f"  SESSION PROBLEM: {sp}")

    # Check the SELECTED MEASURED solution before any score is computed.
    scoring_by = {inst: by[inst] for inst in eligible if inst in by}
    checker = exe.parent / "sor_check"
    verified: dict[str, dict] = {}
    if not args.skip_checker:
        print(f"\nindependent checker: {checker}")
        for m in eligible_models:
            measured = scoring_by.get(m.name, {}).get(cand)
            if measured is None:
                verified[m.name] = {"checked": False, "checker_status": None,
                                    "residual_ok": False, "certificate_ok": False,
                                    "detail": "candidate aggregate is missing"}
            else:
                verified[m.name] = independent_check(checker, m, measured, args.tol)
        rejected = [k for k, v in verified.items()
                    if v.get("checked") and v.get("checker_status") != "VERIFIED"]
        unchecked = [k for k, v in verified.items() if not v["checked"]]
        print(f"  verified {sum(1 for v in verified.values() if v.get('checker_status') == 'VERIFIED')}"
              f"/{len(verified)}   rejected {len(rejected)}   unchecked {len(unchecked)}")
        append_claim_record(
            out, "independent_check",
            status=("FAIL" if rejected else
                    "INCOMPLETE" if unchecked else "PASS"),
            checker=str(checker), checker_sha256=sha256_of(checker),
            results=verified)
        if rejected:
            print("  checker rejection(s) will be scored as invalid with PAR-2: "
                  + ", ".join(rejected[:5]))
        if unchecked:
            session_problems.append(
                f"{len(unchecked)} model(s) could not be independently checked")
    else:
        session_problems.append(
            "independent checker skipped: results are self-reported only")
        append_claim_record(out, "independent_check", status="INCOMPLETE",
                            checker=None, results={})

    candidate_checks = {
        inst: (bool(v.get("checked") and v.get("checker_status") == "VERIFIED" and
                    v.get("residual_ok") and v.get("certificate_ok")),
               str(v.get("detail") or "checker did not verify"))
        for inst, v in verified.items()
    }
    lane_problems = highs_lanes_agree(
        scoring_by, eligible, 1e-7, 1e-7,
        source_version=env.get("source_highs_version"),
        wheel_version=env.get("wheel_highs_version"))
    protocol_incomplete = list(session_problems)
    for override in env["development_overrides"]:
        protocol_incomplete.append(
            f"development override --{override.replace('_', '-')} was used")
    if still:
        protocol_incomplete.append(
            f"{len(still)} rerun pair(s) remained noisy under --allow-noisy")

    eligibility = evaluate_baseline_eligibility(
        scoring_by, cand, eligible, candidate_checks,
        lane_problems=lane_problems,
        protocol_problems=protocol_incomplete)
    append_claim_record(
        out, "baseline_eligibility", **eligibility,
        suite=doc.get("suite"), tolerance=args.tol,
        candidate_solver=cand, manifest=str(args.manifest),
        manifest_sha256=sha256_of(args.manifest),
        allow_list_path=allow_snapshot.path,
        allow_list_sha256=allow_snapshot.sha256,
        allow_list_schema_version=allow_snapshot.schema_version,
        source_highs_sha256=env.get("source_highs_sha256"),
        source_highs_version=env.get("source_highs_version"),
        wheel_highs_version=env.get("wheel_highs_version"))
    print()
    print(f"BASELINE ELIGIBILITY : {eligibility['status']}")
    for failure in eligibility["correctness_failures"]:
        print(f"      - FAIL: {failure}")
    for failure in eligibility["evidence_failures"]:
        print(f"      - INCOMPLETE: {failure}")

    if args.establish_baseline:
        # This deliberately does not receive an older SOR baseline.  It is a
        # performance comparison against the source oracle, not a public claim
        # and not a prerequisite for accepting correct development evidence.
        gate = compare.evaluate_public_claim(
            scoring_by, cand, "highs-source", time_limit=args.time_limit,
            shift=1.0, expected_instances=eligible, claim_mode=False,
            suite=doc.get("suite", ""), tolerance=args.tol,
            candidate_checks=candidate_checks)
        apply_protocol_outcome(gate, list(lane_problems) + protocol_incomplete)
        print()
        print("=" * 72)
        print(f"PROVISIONAL PERFORMANCE  {doc.get('suite')} @ {args.tol:g}")
        print("=" * 72)
        print(f"  shifted SGM ratio : {fmt_optional(gate.sgm_ratio)}")
        print(f"  noise-aware wins  : {gate.win_rate:.1f}%")
        print(f"  VERDICT           : {gate.status} (not a public claim)")
        append_claim_record(
            out, "performance", status=gate.status, public=False,
            suite=doc.get("suite"), tolerance=args.tol,
            candidate_solver=cand, reference_solver="highs-source",
            sgm_ratio=gate.sgm_ratio, sgm_candidate=gate.sgm_candidate,
            sgm_reference=gate.sgm_reference, wins=gate.wins,
            losses=gate.losses, ties=gate.ties, win_rate=gate.win_rate,
            certified=gate.candidate_certified, scored=gate.scored,
            par2_penalised=gate.par2_penalised,
            checker_rejections=gate.checker_rejections,
            incompleteness=gate.incompleteness, failures=gate.failures,
            manifest=str(args.manifest),
            manifest_sha256=sha256_of(args.manifest),
            allow_list_path=allow_snapshot.path,
            allow_list_sha256=allow_snapshot.sha256,
            allow_list_schema_version=allow_snapshot.schema_version,
            source_highs_sha256=env.get("source_highs_sha256"),
            source_highs_version=env.get("source_highs_version"),
            wheel_highs_version=env.get("wheel_highs_version"))
        print(f"\nJSONL: {out}")
        return 0 if eligibility["eligible"] else 1

    assert baseline_data is not None
    gate = compare.evaluate_public_claim(
        scoring_by, cand, "highs-source", time_limit=args.time_limit, shift=1.0,
        baseline_solver=baseline_data.solver, baseline_wall=baseline_data.wall,
        allow=allow_snapshot.entries, expected_instances=eligible,
        claim_mode=True, suite=doc.get("suite", ""), tolerance=args.tol,
        candidate_checks=candidate_checks)

    protocol_incomplete = list(lane_problems) + protocol_incomplete
    if gate.scored != len(eligible):
        protocol_incomplete.append(
            f"scored {gate.scored} != scoring_eligible_count {len(eligible)}")
    apply_protocol_outcome(gate, protocol_incomplete)
    print()
    print("=" * 72)
    print(f"PUBLIC CLAIM GATE  {doc.get('suite')} @ {args.tol:g}")
    print("=" * 72)
    print(f"  shifted SGM ratio : {fmt_optional(gate.sgm_ratio)}   gate <= 0.95")
    print(f"  noise-aware wins  : {gate.win_rate:.1f}%   "
          f"({gate.wins}W/{gate.losses}L/{gate.ties}T)   gate >= 60%")
    print(f"  certified         : {gate.candidate_certified}/{gate.scored}   "
          f"PAR-2 charged {gate.par2_penalised}")
    print(f"  wall candidate    : {fmt_optional(gate.wall_candidate)} s")
    print(f"  wall baseline     : {fmt_optional(gate.wall_baseline)} s")
    print(f"  wall ratio        : {fmt_optional(gate.wall_ratio)}")
    print(f"  tail violations   : {gate.tail_violations or 'none'}")
    print(f"  baseline          : sha={(baseline_data.sha256 or '?')[:12]} "
          f"version={baseline_data.version}")
    print(f"  source HiGHS      : sha={(env.get('source_highs_sha256') or '?')[:12]} "
          f"version={env.get('source_highs_version')}")
    print(f"  wheel HiGHS       : version={env.get('wheel_highs_version')}")
    print(f"  VERDICT           : {gate.status}")
    for f in gate.failures:
        print(f"      - FAIL: {f}")
    for f in gate.incompleteness:
        print(f"      - INCOMPLETE: {f}")
    append_claim_record(
        out, "gate", status=gate.status, suite=doc.get("suite"),
        tolerance=args.tol, candidate_solver=cand,
        reference_solver="highs-source", secondary_reference="highs-wheel",
        sgm_ratio=gate.sgm_ratio, sgm_candidate=gate.sgm_candidate,
        sgm_reference=gate.sgm_reference, wins=gate.wins, losses=gate.losses,
        ties=gate.ties, win_rate=gate.win_rate,
        certified=gate.candidate_certified, scored=gate.scored,
        par2_penalised=gate.par2_penalised, passed=gate.passed,
        checker_rejections=gate.checker_rejections,
        wall_candidate=gate.wall_candidate, wall_baseline=gate.wall_baseline,
        wall_ratio=gate.wall_ratio, tail_violations=gate.tail_violations,
        incompleteness=gate.incompleteness,
        missing_instances=gate.missing_instances,
        invalid_references=gate.invalid_references,
        manifest=str(args.manifest), manifest_sha256=sha256_of(args.manifest),
        baseline=str(args.baseline), baseline_sha256=baseline_data.sha256,
        baseline_version=baseline_data.version,
        source_highs_sha256=env.get("source_highs_sha256"),
        source_highs_version=env.get("source_highs_version"),
        wheel_highs_version=env.get("wheel_highs_version"),
        allow_list_path=allow_snapshot.path,
        allow_list_sha256=allow_snapshot.sha256,
        allow_list_schema_version=allow_snapshot.schema_version,
        highs_lane_failures=lane_problems, failures=gate.failures)
    print(f"\nJSONL: {out}")
    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())
