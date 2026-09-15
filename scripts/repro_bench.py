#!/usr/bin/env python3
"""Run compare.py under controlled conditions and report robust statistics.

    scripts/repro_bench.py benchmarks/netlib/mps --solvers sor:simplex \\
        --repetitions 5 --warmups 1 --cpu 2

    scripts/repro_bench.py examples/testlp.mps --dry-run    # just show the command

This changes NO solver behaviour. It is a wrapper that

  1. records the machine state that decides whether a timing is reproducible
     at all (CPU governor, turbo, SMT, ASLR, load average, thermal throttling),
     and refuses to run when that state is known to produce noisy numbers;
  2. drives scripts/compare.py with pinning, warmups and repetitions;
  3. reads the resulting JSONL back and reports a MEDIAN and a MAD per
     (solver, instance), instead of a single sample.

WHY MAD. The mean and standard deviation of a benchmark timing are dominated by
the occasional 10x outlier from an unrelated process waking up. The median and
the median absolute deviation are not: half the samples have to move before
either does. `robust_cv` below is 1.4826*MAD/median, which for normal noise
estimates the same thing as the coefficient of variation but survives outliers.
A run whose robust_cv is above --stable-threshold is reported as UNSTABLE,
which means "do not draw a conclusion from this number", not "the solver
regressed".

The environment probes are Linux sysfs paths. On any other platform they read
as unknown and the corresponding checks are skipped rather than guessed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "CMakeLists.txt").exists())
SCRIPTS = Path(__file__).resolve().parent

_SPEC = importlib.util.spec_from_file_location(
    "sor_compare_runs", SCRIPTS / "compare_runs.py")
assert _SPEC is not None and _SPEC.loader is not None
compare_runs = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = compare_runs
_SPEC.loader.exec_module(compare_runs)

compare = compare_runs.compare


# --------------------------------------------------------------------------
# Command construction
# --------------------------------------------------------------------------
def build_compare_command(models: list[str], *, solvers: str, time_limit: float,
                          repetitions: int, warmups: int, cpu: int | None,
                          jsonl: Path, tol: float | None = None,
                          method: str | None = None,
                          pricing: str | None = None,
                          basis_update: str | None = None,
                          max_iter: int | None = None,
                          seed: int | None = None,
                          limit: int | None = None,
                          allow_unchecked: bool = False,
                          allow_unavailable: bool = False,
                          python: str | None = None,
                          script: Path | None = None) -> list[str]:
    """The exact compare.py invocation this wrapper will run.

    Kept separate from running it so the command is testable, and so --dry-run
    prints something the user can paste and re-run by hand.
    """
    cmd = [python or sys.executable, str(script or (SCRIPTS / "compare.py"))]
    cmd += list(models)
    cmd += ["--solvers", solvers,
            "--time-limit", str(time_limit),
            "--repetitions", str(repetitions),
            "--warmups", str(warmups),
            "--jsonl", str(jsonl)]
    if cpu is not None:
        cmd += ["--cpu", str(cpu)]
    if tol is not None:
        cmd += ["--tol", str(tol)]
    if method is not None:
        cmd += ["--method", method]
    if pricing is not None:
        cmd += ["--pricing", pricing]
    if basis_update is not None:
        cmd += ["--basis-update", basis_update]
    if max_iter is not None:
        cmd += ["--max-iter", str(max_iter)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if allow_unchecked:
        cmd.append("--allow-unchecked")
    if allow_unavailable:
        cmd.append("--allow-unavailable")
    return cmd


# --------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------
def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def capture_environment(sysroot: Path = Path("/"), cpu: int | None = None
                        ) -> dict[str, object]:
    """Machine state that decides whether a timing is reproducible.

    `sysroot` exists so the probes can be pointed at a fixture tree in tests;
    in production it is "/". Anything unreadable comes back as None, which the
    checks below treat as "unknown", never as "fine".
    """
    sysfs = sysroot / "sys/devices/system/cpu"
    proc = sysroot / "proc"

    governors = sorted({
        g for g in (_read(p) for p in sorted(sysfs.glob("cpu*/cpufreq/scaling_governor")))
        if g})

    # Both spellings exist: intel_pstate exposes no_turbo (1 = turbo OFF),
    # acpi-cpufreq exposes boost (1 = boost ON).
    no_turbo = _read(sysfs / "intel_pstate/no_turbo")
    boost = _read(sysfs / "cpufreq/boost")
    if no_turbo is not None:
        turbo = no_turbo == "0"
    elif boost is not None:
        turbo = boost == "1"
    else:
        turbo = None

    loadavg_text = _read(proc / "loadavg")
    loadavg = None
    if loadavg_text:
        try:
            loadavg = float(loadavg_text.split()[0])
        except (ValueError, IndexError):
            loadavg = None

    siblings = None
    if cpu is not None:
        siblings_text = _read(sysfs / f"cpu{cpu}/topology/thread_siblings_list")
        if siblings_text:
            siblings = siblings_text

    model = None
    cpuinfo = _read(proc / "cpuinfo")
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break

    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_model": model,
        "cpu_count": os.cpu_count(),
        "governors": governors or None,
        "turbo_enabled": turbo,
        "smt_control": _read(sysfs / "smt/control"),
        "isolated_cpus": _read(sysfs / "isolated") or None,
        "thread_siblings_of_pinned_cpu": siblings,
        "aslr": _read(proc / "sys/kernel/randomize_va_space"),
        "loadavg_1min": loadavg,
        "pinned_cpu": cpu,
        "taskset_available": shutil.which("taskset") is not None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


# Conditions that make a wall-clock measurement not reproducible. Each is
# (key, predicate on the environment, message).
def stability_warnings(env: dict[str, object], cpu: int | None,
                       load_limit: float = 0.5) -> list[str]:
    out: list[str] = []
    govs = env.get("governors")
    if isinstance(govs, list) and any(g != "performance" for g in govs):
        out.append(
            f"CPU governor is {sorted(set(govs))}, not ['performance']: clock "
            f"ramping adds run-to-run variance. "
            f"sudo cpupower frequency-set -g performance")
    if env.get("turbo_enabled") is True:
        out.append(
            "turbo/boost is enabled: sustained runs clock down as the package "
            "heats, so later instances are systematically slower")
    load = env.get("loadavg_1min")
    if isinstance(load, (int, float)) and load > load_limit:
        out.append(f"1-minute load average is {load:g} (> {load_limit:g}): "
                   f"another process is competing for CPU")
    if cpu is None:
        out.append("no --cpu given: the scheduler may migrate the solver "
                   "between cores mid-run")
    elif not env.get("taskset_available"):
        out.append("--cpu was given but taskset is not installed, so nothing "
                   "is actually pinned")
    if cpu is not None and env.get("smt_control") == "on":
        sib = env.get("thread_siblings_of_pinned_cpu")
        out.append(f"SMT is on and CPU {cpu} shares a core with {sib}: a "
                   f"process on the sibling steals execution resources")
    if env.get("aslr") not in (None, "0"):
        out.append(f"ASLR is {env.get('aslr')}: layout changes move the "
                   f"cache-conflict pattern between runs (minor, but real)")
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
# 1.4826 * MAD estimates the standard deviation of normally distributed data,
# which is what makes robust_cv comparable to a coefficient of variation.
MAD_TO_SIGMA = 1.4826


def median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def mad(values: list[float]) -> float | None:
    """Median absolute deviation from the median."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    med = statistics.median(vals)
    return statistics.median([abs(v - med) for v in vals])


def robust_cv(values: list[float]) -> float | None:
    """1.4826*MAD / median. None when the median is 0 (no scale to divide by)."""
    med, dispersion = median(values), mad(values)
    if med is None or dispersion is None or med <= 0.0:
        return None
    return MAD_TO_SIGMA * dispersion / med


@dataclass
class Stats:
    solver: str
    instance: str
    status: str
    n: int
    median_s: float | None = None
    mad_s: float | None = None
    robust_cv: float | None = None
    min_s: float | None = None
    max_s: float | None = None
    samples: list[float] = field(default_factory=list)
    certified: bool = True
    stable: bool | None = None      # None = below the noise floor, not judged


def summarize(run: "compare_runs.Run", solver: str, threshold: float,
              floor_s: float) -> list[Stats]:
    """Per-instance robust statistics from one loaded sweep.

    An instance whose median is below `floor_s` is not judged stable or
    unstable: at a few milliseconds the measurement is mostly process startup,
    and its relative spread says nothing about the solver.
    """
    out: list[Stats] = []
    for instance, result in sorted(run.results[solver].items()):
        samples = [s for s in (result.samples_s or []) if s is not None]
        st = Stats(solver=solver, instance=instance, status=result.status,
                   n=len(samples), samples=samples,
                   certified=compare.is_certified_success(result))
        if samples:
            st.median_s = median(samples)
            st.mad_s = mad(samples)
            st.robust_cv = robust_cv(samples)
            st.min_s = min(samples)
            st.max_s = max(samples)
            if st.median_s is not None and st.median_s >= floor_s:
                st.stable = (st.robust_cv is not None
                             and st.robust_cv <= threshold)
        out.append(st)
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def report(stats: list[Stats], env: dict[str, object], warnings: list[str],
           threshold: float, floor_s: float) -> None:
    print("\nENVIRONMENT")
    for key in ("cpu_model", "cpu_count", "governors", "turbo_enabled",
                "smt_control", "isolated_cpus", "pinned_cpu", "aslr",
                "loadavg_1min", "git_commit", "git_dirty"):
        print(f"  {key:<16} {env.get(key)}")

    print("\nREPRODUCIBILITY")
    if warnings:
        for w in warnings:
            print(f"  WARN  {w}")
    else:
        print("  no known sources of run-to-run variance")

    print("\nPER-INSTANCE (median +/- MAD over repetitions)")
    if not stats:
        print("  no results")
        return
    wid = max(len(s.instance) for s in stats)
    print(f"  {'instance':<{wid}}  {'n':>3}  {'median':>10}  {'MAD':>10}  "
          f"{'robust cv':>9}  verdict")
    for s in stats:
        cv = "-" if s.robust_cv is None else f"{100 * s.robust_cv:8.2f}%"
        verdict = ("UNCERTIFIED" if not s.certified else
                   "-" if s.stable is None else
                   "stable" if s.stable else "UNSTABLE")
        print(f"  {s.instance:<{wid}}  {s.n:>3}  "
              f"{compare.fmt_time(s.median_s):>10}  "
              f"{compare.fmt_time(s.mad_s):>10}  {cv:>9}  {verdict}")

    judged = [s for s in stats if s.stable is not None]
    unstable = [s for s in judged if not s.stable]
    below = [s for s in stats if s.stable is None and s.certified]
    print(f"\n  {len(judged) - len(unstable)}/{len(judged)} judged instances "
          f"within {100 * threshold:g}% robust cv; {len(unstable)} unstable; "
          f"{len(below)} below the {1000 * floor_s:g}ms noise floor "
          f"(not judged)")
    if unstable:
        print("  unstable: " + ", ".join(s.instance for s in unstable))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Reproducible benchmark runner around scripts/compare.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--solvers", default="sor:simplex")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--repetitions", type=int, default=5,
                    help="measured repetitions per (solver, instance)")
    ap.add_argument("--warmups", type=int, default=1,
                    help="unmeasured runs before the measured ones")
    ap.add_argument("--cpu", type=int, default=None,
                    help="pin every solver process to this logical CPU")
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--method", choices=("auto", "primal", "dual"), default=None)
    ap.add_argument("--pricing", choices=("choose", "dantzig", "devex", "dse"),
                    default=None)
    ap.add_argument("--basis-update", choices=("product", "ft"), default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-unchecked", action="store_true")
    ap.add_argument("--allow-unavailable", action="store_true")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="where compare.py writes its records "
                         "(default: repro_<timestamp>.jsonl in the cwd)")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the environment record and statistics here")
    ap.add_argument("--stable-threshold", type=float, default=0.05,
                    help="robust cv above which an instance is UNSTABLE")
    ap.add_argument("--noise-floor-ms", type=float, default=5.0,
                    help="instances faster than this are not judged")
    ap.add_argument("--load-limit", type=float, default=0.5,
                    help="1-minute load average above which to warn")
    ap.add_argument("--allow-unstable-machine", action="store_true",
                    help="run even when the machine state is known to be noisy")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the environment and the command, run nothing")
    args = ap.parse_args(argv)

    if args.repetitions < 1:
        ap.error("--repetitions must be at least 1")
    if args.warmups < 0:
        ap.error("--warmups cannot be negative")
    if args.time_limit <= 0:
        ap.error("--time-limit must be positive")
    if not 0.0 <= args.stable_threshold:
        ap.error("--stable-threshold cannot be negative")
    if args.noise_floor_ms < 0:
        ap.error("--noise-floor-ms cannot be negative")
    if args.cpu is not None and args.cpu < 0:
        ap.error("--cpu cannot be negative")
    # A median is only meaningful with a few samples; MAD is 0 for n < 3 by
    # construction, which would look like perfect stability.
    if args.repetitions < 3 and not args.dry_run:
        print(f"note: --repetitions {args.repetitions} makes the MAD "
              f"uninformative; 5 or more is the useful range", file=sys.stderr)

    jsonl = args.jsonl or Path(f"repro_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    env = capture_environment(cpu=args.cpu)
    warnings = stability_warnings(env, args.cpu, args.load_limit)
    cmd = build_compare_command(
        args.models, solvers=args.solvers, time_limit=args.time_limit,
        repetitions=args.repetitions, warmups=args.warmups, cpu=args.cpu,
        jsonl=jsonl, tol=args.tol, method=args.method, pricing=args.pricing,
        basis_update=args.basis_update, max_iter=args.max_iter, seed=args.seed,
        limit=args.limit, allow_unchecked=args.allow_unchecked,
        allow_unavailable=args.allow_unavailable)

    if args.dry_run:
        report([], env, warnings, args.stable_threshold,
               args.noise_floor_ms / 1000.0)
        print("\nCOMMAND")
        print("  " + " ".join(cmd))
        return 0

    if warnings and not args.allow_unstable_machine:
        print("error: the machine is not in a reproducible state:",
              file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
        print("re-run with --allow-unstable-machine to measure anyway",
              file=sys.stderr)
        return 2

    print("running: " + " ".join(cmd), flush=True)
    completed = subprocess.run(cmd)

    try:
        run = compare_runs.load_run(jsonl)
    except compare_runs.RunFileError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    all_stats: list[Stats] = []
    for solver in run.solvers:
        all_stats += summarize(run, solver, args.stable_threshold,
                               args.noise_floor_ms / 1000.0)
    report(all_stats, env, warnings, args.stable_threshold,
           args.noise_floor_ms / 1000.0)
    print(f"\nJSONL: {jsonl}")

    if args.json is not None:
        with args.json.open("w") as fh:
            json.dump({"tool": "repro_bench",
                       "environment": env,
                       "warnings": warnings,
                       "command": cmd,
                       "compare_exit_code": completed.returncode,
                       "stable_threshold": args.stable_threshold,
                       "noise_floor_s": args.noise_floor_ms / 1000.0,
                       "instances": [vars(s) for s in all_stats]}, fh, indent=2)
            fh.write("\n")
        print(f"JSON:  {args.json}")

    unstable = [s for s in all_stats if s.stable is False]
    if completed.returncode != 0:
        return completed.returncode
    return 1 if unstable else 0


if __name__ == "__main__":
    sys.exit(main())
