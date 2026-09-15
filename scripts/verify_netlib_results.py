#!/usr/bin/env python3
"""Independently re-check the benchmark results stored in benchmarks/results/.

    scripts/verify_netlib_results.py                       # every netlib file
    scripts/verify_netlib_results.py benchmarks/results/netlib_simplex.jsonl
    scripts/verify_netlib_results.py --json failures.json

A stored result file is a claim. This re-derives, from the files themselves and
from the models on disk, whether that claim holds:

    coverage    every instance in the suite appears exactly once
    status      a status is present and non-empty
    proof       a SOR result carries a proof level, and an Optimal SOR result
                carries one from the accepted set (an "Optimal" with no proof
                is precisely the failure mode certification exists to catch)
    objective   the objective agrees with an independent reference
    violation   the reported primal/row violation is within tolerance

It reads and reports; it never rewrites a result file.

THE REFERENCE. The honest reference is HiGHS. When highspy is importable it is
run live. When it is not, the reference is assembled from the HiGHS runs
ALREADY STORED in benchmarks/results/, and any instance where two stored HiGHS
runs disagree is reported as `reference-disagreement` rather than silently
picking one. Every finding records which reference it used, so a conclusion
drawn from stored HiGHS is never mistaken for a fresh independent solve. With
neither available, objective checking is reported as `no-reference` instead of
being skipped in silence.

THREE STORED SCHEMAS. These files were written by three generations of the
harness, and all three are still on disk:
    * {"record": "instance", "results": {"SOR": {...}, "highs": {...}}}
    * {"record": "run", "solver": ..., ...}          (current compare.py)
    * flat per-instance records with no "record" key  (netlib_simplex.jsonl)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "CMakeLists.txt").exists())

# Proof levels that make an "Optimal" trustworthy. Kept in step with
# compare.py's _SOR_PROOFS -- imported rather than copied where possible.
try:
    import importlib.util
    _SPEC = importlib.util.spec_from_file_location(
        "sor_compare_verify", Path(__file__).resolve().parent / "compare.py")
    assert _SPEC is not None and _SPEC.loader is not None
    _compare = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _compare
    _SPEC.loader.exec_module(_compare)
    ACCEPTED_PROOFS = set(_compare._SOR_PROOFS)
except Exception:  # noqa: BLE001 - stay usable if compare.py moves
    ACCEPTED_PROOFS = {"ProvedOptimalFP", "ProvedOptimalExact",
                       "ProvedOptimalCertified", "ProvedKKT",
                       "ProvedGlobalEpsilon"}

# Names a stored file uses for this project's own solver.
OURS = {"sor", "sor:simplex", "sor:dual", "sor:primal", "sor:milp", "sor:pdhg",
        "sor:hpr", "sor:qp", "ours"}
# Names that denote an independent reference solver.
REFERENCE_SOLVERS = {"highs", "gurobi", "cbc", "scipy", "scipy-ipm"}

# A file covering at least this fraction of the suite is treated as claiming
# the whole suite, so its gaps are real. Below it, the file is a partial run.
FULL_SUITE_FRACTION = 0.9


def is_ours(solver: str) -> bool:
    s = solver.strip().lower()
    return s in OURS or s.startswith("sor")


def normalize_instance(name: str) -> str:
    """`25fv47.mps` and `25fv47` are the same instance."""
    stem = Path(str(name)).name
    for suffix in (".mps", ".qps", ".lp", ".MPS", ".QPS", ".LP"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


@dataclass
class Observation:
    """One (file, instance, solver) result, normalized across schemas."""
    file: str
    instance: str
    solver: str
    status: str | None
    proof: str | None
    objective: float | None
    violation: float | None
    iterations: int | None = None
    error: str | None = None
    schema: str = ""

    @property
    def ours(self) -> bool:
        return is_ours(self.solver)

    def optimal(self) -> bool:
        return (self.status or "").strip().lower() in ("optimal", "solved", "1")


@dataclass
class Failure:
    kind: str
    file: str
    instance: str
    solver: str
    detail: str
    observed: object = None
    expected: object = None
    reference_source: str | None = None


@dataclass
class FileReport:
    path: str
    schema: str
    instances: int = 0
    observations: int = 0
    failures: list[Failure] = field(default_factory=list)


# --------------------------------------------------------------------------
# Loading the three schemas
# --------------------------------------------------------------------------
def _f(value: object) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v


def _i(value: object) -> int | None:
    v = _f(value)
    return None if v is None else int(v)


def load_file(path: Path) -> tuple[list[Observation], str, list[Failure]]:
    """Read one result file. Returns (observations, schema, parse failures)."""
    out: list[Observation] = []
    problems: list[Failure] = []
    schemas: set[str] = set()
    try:
        text = path.read_text()
    except OSError as e:
        problems.append(Failure("unreadable-file", str(path), "", "", str(e)))
        return out, "unknown", problems

    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(Failure("malformed-json", str(path), "", "",
                                    f"line {line_no}: {e}"))
            continue
        # environment/summary records describe the run, not a result.
        if not isinstance(rec, dict) or rec.get("record") in ("environment",
                                                              "summary"):
            continue

        if rec.get("record") == "instance" and isinstance(rec.get("results"), dict):
            schemas.add("nested")
            instance = normalize_instance(rec.get("instance", ""))
            for solver, r in rec["results"].items():
                if not isinstance(r, dict):
                    continue
                out.append(Observation(
                    file=str(path), instance=instance, solver=str(solver),
                    status=r.get("status"), proof=r.get("proof"),
                    objective=_f(r.get("objective")),
                    # This generation called it row_violation.
                    violation=_f(r.get("row_violation",
                                       r.get("primal_viol",
                                             r.get("violation")))),
                    iterations=_i(r.get("iterations")),
                    error=r.get("error"), schema="nested"))
        elif rec.get("record") == "run" or "solver" in rec:
            schemas.add("run")
            out.append(Observation(
                file=str(path), instance=normalize_instance(rec.get("instance", "")),
                solver=str(rec.get("solver", "")), status=rec.get("status"),
                proof=rec.get("proof"), objective=_f(rec.get("objective")),
                violation=_f(rec.get("violation")),
                iterations=_i(rec.get("iterations")),
                error=rec.get("error"), schema="run"))
        elif "instance" in rec:
            # Flat, single-solver-per-line: these files record SOR only.
            schemas.add("flat")
            out.append(Observation(
                file=str(path), instance=normalize_instance(rec["instance"]),
                solver="sor", status=rec.get("status"), proof=rec.get("proof"),
                objective=_f(rec.get("objective")),
                violation=_f(rec.get("primal_viol", rec.get("row_violation"))),
                iterations=_i(rec.get("iterations")),
                error=rec.get("downgrade"), schema="flat"))
        else:
            problems.append(Failure("unrecognized-record", str(path), "", "",
                                    f"line {line_no}: keys {sorted(rec)[:6]}"))

    schema = "+".join(sorted(schemas)) if schemas else "empty"
    return out, schema, problems


# --------------------------------------------------------------------------
# Reference objectives
# --------------------------------------------------------------------------
def live_highs_available() -> bool:
    try:
        import highspy  # noqa: F401
        highspy.Highs()  # the shared library must actually load
    except Exception:  # noqa: BLE001
        return False
    return True


def solve_with_highs(model: Path, time_limit: float) -> float | None:
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(time_limit))
    h.readModel(str(model))
    h.run()
    if "kOptimal" not in str(h.getModelStatus()):
        return None
    return float(h.getObjectiveValue())


def build_stored_reference(observations: list[Observation], abs_tol: float,
                           rel_tol: float) -> tuple[dict[str, float],
                                                    dict[str, str],
                                                    list[Failure]]:
    """Reference objectives from the reference-solver runs already on disk.

    Disagreement between two stored runs of the same reference solver is
    reported, not averaged away: it means one of the stored files is wrong and
    neither can be trusted as the reference for that instance.
    """
    seen: dict[str, list[Observation]] = {}
    for o in observations:
        if o.solver.strip().lower() in REFERENCE_SOLVERS and o.optimal() \
                and o.objective is not None and math.isfinite(o.objective):
            seen.setdefault(o.instance, []).append(o)

    reference: dict[str, float] = {}
    source: dict[str, str] = {}
    failures: list[Failure] = []
    for instance, obs in sorted(seen.items()):
        first = obs[0]
        disagreeing = [o for o in obs
                       if not objectives_agree(o.objective, first.objective,
                                               abs_tol, rel_tol)]
        if disagreeing:
            failures.append(Failure(
                "reference-disagreement", disagreeing[0].file, instance,
                first.solver,
                f"stored {first.solver} objectives disagree across files: "
                f"{first.objective!r} in {Path(first.file).name} vs "
                f"{disagreeing[0].objective!r} in {Path(disagreeing[0].file).name}",
                observed=disagreeing[0].objective, expected=first.objective))
            continue
        reference[instance] = first.objective  # type: ignore[assignment]
        source[instance] = f"stored:{first.solver}:{Path(first.file).name}"
    return reference, source, failures


def objectives_agree(a: float | None, b: float | None, abs_tol: float,
                     rel_tol: float) -> bool:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return False
    return abs(a - b) <= abs_tol + rel_tol * max(abs(a), abs(b))


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def verify_file(path: Path, observations: list[Observation], schema: str,
                expected: set[str], reference: dict[str, float],
                ref_source: dict[str, str], *, abs_tol: float, rel_tol: float,
                viol_tol: float, check_coverage: bool) -> FileReport:
    report = FileReport(path=str(path), schema=schema)
    ours = [o for o in observations if o.ours]
    report.observations = len(observations)
    report.instances = len({o.instance for o in observations})

    # Coverage: every expected instance exactly once, per solver.
    #
    # benchmarks/results/ holds a mix of full sweeps and deliberately partial
    # ones (5-instance smoke runs, single-instance repros). Reporting 88
    # missing-instance findings against a 5-instance smoke file is not a
    # finding, it is noise that buries the real ones -- so a file that clearly
    # is not a full-suite claim gets ONE `partial-file` note instead, and only
    # files that do claim the whole suite are held to it.
    if check_coverage and expected:
        by_solver: dict[str, list[str]] = {}
        for o in observations:
            by_solver.setdefault(o.solver, []).append(o.instance)
        for solver, names in sorted(by_solver.items()):
            counts: dict[str, int] = {}
            for n in names:
                counts[n] = counts.get(n, 0) + 1
            for name, count in sorted(counts.items()):
                if count > 1:
                    report.failures.append(Failure(
                        "duplicate-instance", str(path), name, solver,
                        f"appears {count} times", observed=count, expected=1))
            present = set(names)
            missing = expected - present
            if missing and len(present & expected) < FULL_SUITE_FRACTION * len(expected):
                report.failures.append(Failure(
                    "partial-file", str(path), "", solver,
                    f"covers {len(present & expected)} of {len(expected)} suite "
                    f"instances; not a full-suite result, so its coverage is "
                    f"not held against the suite",
                    observed=len(present & expected), expected=len(expected)))
            else:
                for name in sorted(missing):
                    report.failures.append(Failure(
                        "missing-instance", str(path), name, solver,
                        "expected by the suite but absent from this file"))
            for name in sorted(present - expected):
                report.failures.append(Failure(
                    "unexpected-instance", str(path), name, solver,
                    "present in the file but not in the suite directory"))

    for o in ours:
        if o.status is None or not str(o.status).strip():
            report.failures.append(Failure(
                "status-missing", str(path), o.instance, o.solver,
                "no status recorded", observed=o.status))
            continue

        # A proof is only owed by a result that CLAIMS something. A timeout or
        # an error has nothing to prove, and demanding a proof level from it
        # would bury the real finding under expected noise.
        if o.optimal() and (o.proof is None or not str(o.proof).strip()):
            report.failures.append(Failure(
                "proof-missing", str(path), o.instance, o.solver,
                f"status {o.status!r} with no proof level", observed=o.proof))
        elif o.optimal() and o.proof not in ACCEPTED_PROOFS:
            report.failures.append(Failure(
                "unproved-optimal", str(path), o.instance, o.solver,
                f"status Optimal but proof {o.proof!r} is not in the accepted set",
                observed=o.proof, expected=sorted(ACCEPTED_PROOFS)))

        # A violation bound is only owed by a result that claims a usable
        # point. An Interrupted run stopped mid-iteration and its iterate is
        # not expected to be feasible -- dfl001 at the time limit reports a
        # violation of 161, which is a description of a timeout, not a defect.
        if o.optimal() and o.violation is not None and (
                not math.isfinite(o.violation) or o.violation > viol_tol):
            report.failures.append(Failure(
                "violation-too-large", str(path), o.instance, o.solver,
                f"primal/row violation {o.violation:g} exceeds {viol_tol:g}",
                observed=o.violation, expected=viol_tol))

        # Objectives are only comparable for a result claiming optimality.
        if not o.optimal():
            continue
        if o.instance not in reference:
            report.failures.append(Failure(
                "no-reference", str(path), o.instance, o.solver,
                "no independent reference objective is available for this "
                "instance, so its objective is unverified",
                observed=o.objective))
            continue
        if not objectives_agree(o.objective, reference[o.instance],
                                abs_tol, rel_tol):
            report.failures.append(Failure(
                "objective-mismatch", str(path), o.instance, o.solver,
                f"objective {o.objective!r} disagrees with reference "
                f"{reference[o.instance]!r}",
                observed=o.objective, expected=reference[o.instance],
                reference_source=ref_source.get(o.instance)))
    return report


def suite_instances(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {normalize_instance(p.name) for p in directory.glob("*.mps")}


def default_files() -> list[Path]:
    results = ROOT / "benchmarks" / "results"
    return sorted(p for p in results.glob("*.jsonl")
                  if "netlib" in p.name.lower())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify stored Netlib benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("files", nargs="*", type=Path,
                    help="result JSONL files (default: benchmarks/results/*netlib*.jsonl)")
    ap.add_argument("--suite-dir", type=Path,
                    default=ROOT / "benchmarks" / "netlib" / "mps",
                    help="directory whose .mps files define the expected suite")
    ap.add_argument("--obj-rel-tol", type=float, default=1e-6)
    ap.add_argument("--obj-abs-tol", type=float, default=1e-6)
    ap.add_argument("--violation-tol", type=float, default=1e-6)
    ap.add_argument("--no-coverage", action="store_true",
                    help="skip the missing/duplicate/unexpected instance checks")
    ap.add_argument("--live-highs", action="store_true",
                    help="solve with HiGHS now instead of using stored runs")
    ap.add_argument("--highs-time-limit", type=float, default=60.0)
    ap.add_argument("--json", type=Path, default=None,
                    help="write the machine-readable failure list here")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the summary")
    args = ap.parse_args(argv)

    for name in ("obj_rel_tol", "obj_abs_tol", "violation_tol"):
        v = getattr(args, name)
        if not math.isfinite(v) or v < 0.0:
            ap.error(f"--{name.replace('_', '-')} must be finite and nonnegative")

    files = args.files or default_files()
    if not files:
        print("error: no result files found", file=sys.stderr)
        return 2

    expected = suite_instances(args.suite_dir)
    all_obs: list[Observation] = []
    loaded: list[tuple[Path, list[Observation], str]] = []
    parse_failures: list[Failure] = []
    for path in files:
        obs, schema, problems = load_file(path)
        parse_failures += problems
        loaded.append((path, obs, schema))
        all_obs += obs

    # Reference objectives.
    reference, ref_source, ref_failures = build_stored_reference(
        all_obs, args.obj_abs_tol, args.obj_rel_tol)
    live_used = False
    if args.live_highs:
        if not live_highs_available():
            print("error: --live-highs requested but highspy is not usable "
                  "(see benchmarks/requirements-baseline.txt)", file=sys.stderr)
            return 2
        live_used = True
        for instance in sorted({o.instance for o in all_obs}):
            model = args.suite_dir / f"{instance}.mps"
            if not model.is_file():
                continue
            value = solve_with_highs(model, args.highs_time_limit)
            if value is not None:
                reference[instance] = value
                ref_source[instance] = "live:highs"

    reports = [verify_file(path, obs, schema, expected, reference, ref_source,
                           abs_tol=args.obj_abs_tol, rel_tol=args.obj_rel_tol,
                           viol_tol=args.violation_tol,
                           check_coverage=not args.no_coverage)
               for path, obs, schema in loaded]

    failures = list(parse_failures) + list(ref_failures)
    for r in reports:
        failures += r.failures

    kinds: dict[str, int] = {}
    for f in failures:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1

    if not args.quiet:
        print(f"reference: {'live HiGHS' if live_used else 'stored reference-solver runs'}"
              f" -- {len(reference)} instance(s) covered")
        print(f"suite: {len(expected)} instance(s) in {args.suite_dir}")
        print()
        for r in reports:
            status = "OK" if not r.failures else f"{len(r.failures)} finding(s)"
            print(f"{Path(r.path).name:<48} {r.schema:<8} "
                  f"{r.instances:>4} inst  {status}")
        if failures:
            print("\nFINDINGS")
            for kind in sorted(kinds):
                hits = [f for f in failures if f.kind == kind]
                print(f"\n  {kind}: {kinds[kind]}")
                for f in hits[:20]:
                    where = Path(f.file).name if f.file else "-"
                    print(f"    {where}  {f.instance or '-':<14} "
                          f"{f.solver or '-':<12} {f.detail}")
                if len(hits) > 20:
                    print(f"    ... and {len(hits) - 20} more")

    print(f"\n{len(failures)} finding(s) across {len(files)} file(s)")
    for kind in sorted(kinds):
        print(f"  {kind:<24} {kinds[kind]}")

    if args.json is not None:
        with args.json.open("w") as fh:
            json.dump({
                "tool": "verify_netlib_results",
                "reference": "live-highs" if live_used else "stored",
                "reference_instances": len(reference),
                "suite_dir": str(args.suite_dir),
                "suite_instances": sorted(expected),
                "tolerances": {"objective_rel": args.obj_rel_tol,
                               "objective_abs": args.obj_abs_tol,
                               "violation": args.violation_tol},
                "files": [{"path": r.path, "schema": r.schema,
                           "instances": r.instances,
                           "observations": r.observations,
                           "failures": len(r.failures)} for r in reports],
                "counts": kinds,
                "failures": [asdict(f) for f in failures],
            }, fh, indent=2)
            fh.write("\n")
        print(f"\nJSON: {args.json}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
