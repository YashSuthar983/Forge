#!/usr/bin/env python3
"""Diff two compare.py JSONL runs: what broke, what changed, what got faster.

    scripts/compare_runs.py before.jsonl after.jsonl
    scripts/compare_runs.py before.jsonl after.jsonl --solver sor:simplex --top 15
    scripts/compare_runs.py before.jsonl after.jsonl --json diff.json

CORRECTNESS COMES FIRST. Timing is only comparable between two results that
both passed SOR's own proof gate, so this tool reports, and exits nonzero on:

    proof regressions     certified in the baseline, not in the candidate
    status mismatches     e.g. Optimal -> timeout, or Optimal -> Infeasible
    objective mismatches  both certified, but disagreeing beyond tolerance
    missing models        present in one run and not the other

Only after that does it compare performance, and only over the instances that
are certified on BOTH sides -- an instance that stopped being solved is a
correctness finding, never a speedup.

The definitions of "certified", "optimal" and "objectives agree", and the
median-of-repetitions aggregation, are IMPORTED from compare.py rather than
restated, so this tool cannot drift from the harness that produced the data.

APPENDED FILES: compare.py opens its --jsonl in append mode, so one file can
hold several sweeps, each introduced by an "environment" record. By default
the LAST sweep in each file is used; --segment picks another.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "CMakeLists.txt").exists())

_SPEC = importlib.util.spec_from_file_location(
    "sor_compare", Path(__file__).resolve().parent / "compare.py")
assert _SPEC is not None and _SPEC.loader is not None
compare = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = compare
_SPEC.loader.exec_module(compare)

Result = compare.Result
_RESULT_FIELDS = {f.name for f in dataclasses.fields(Result)}


class RunFileError(ValueError):
    """A JSONL file this tool cannot interpret."""


@dataclass
class Run:
    """One sweep out of one JSONL file."""
    path: Path
    environment: dict[str, object]
    # results[solver label][instance] -> the aggregated Result
    results: dict[str, dict[str, Result]]
    metadata: list[dict[str, object]] = dataclasses.field(default_factory=list)
    segments: int = 1
    segment: int = -1

    @property
    def solvers(self) -> list[str]:
        return sorted(self.results)

    def label(self) -> str:
        commit = self.environment.get("commit")
        short = f" @{str(commit)[:8]}" if commit else ""
        dirty = "+dirty" if self.environment.get("dirty") else ""
        return f"{self.path.name}{short}{dirty}"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_run(path: Path, segment: int = -1) -> Run:
    """Read one JSONL file and return the requested sweep from it."""
    try:
        text = path.read_text()
    except OSError as e:
        raise RunFileError(f"{path}: {e}") from e

    # Only compare.py's environment record starts a sweep. Claim protocol
    # metadata (preflight, postflight, checks, reruns and the final gate) is
    # attached to that sweep and never creates an empty trailing segment.
    # Legacy claim artifacts encoded those records as environments with a
    # claim_role; recognise them explicitly as metadata too.
    segments: list[tuple[dict[str, object], list[dict[str, object]],
                         list[dict[str, object]]]] = []
    pending_metadata: list[dict[str, object]] = []
    metadata_kinds = {"metadata", "preflight", "postflight",
                      "independent_check", "baseline_eligibility",
                      "performance", "gate", "rerun", "gate_accept"}
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise RunFileError(f"{path} line {line_no}: {e}") from e
        if not isinstance(rec, dict):
            raise RunFileError(f"{path} line {line_no}: record is not an object")
        kind = rec.get("record")
        legacy_metadata = (kind == "environment" and
                           (rec.get("claim_role") is not None or
                            rec.get("gate_accept") is not None or
                            rec.get("gate_accept_voided") is not None))
        if kind in metadata_kinds or legacy_metadata:
            if segments:
                segments[-1][2].append(rec)
            else:
                pending_metadata.append(rec)
        elif kind == "environment":
            segments.append((rec, [], pending_metadata))
            pending_metadata = []
        elif kind in ("run", "aggregate"):
            if not segments:
                segments.append(({}, [], pending_metadata))
                pending_metadata = []
            segments[-1][1].append(rec)
        else:
            raise RunFileError(
                f"{path} line {line_no}: unknown record type {kind!r}")

    if pending_metadata and segments:
        segments[-1][2].extend(pending_metadata)

    if not segments:
        raise RunFileError(f"{path}: no records")
    try:
        env, records, metadata = segments[segment]
    except IndexError as e:
        raise RunFileError(
            f"{path}: --segment {segment} out of range "
            f"({len(segments)} sweep(s) in the file)") from e

    # Group repetitions, then aggregate each group the way compare.py's own
    # summary does, so a median here means the same thing it did there.
    grouped: dict[tuple[str, str], list[Result]] = {}
    aggregates: dict[tuple[str, str], list[Result]] = {}
    for line_no, rec in enumerate(records, 1):
        unknown = set(rec) - _RESULT_FIELDS - {"record"}
        if unknown:
            raise RunFileError(
                f"{path}: run record has unknown field(s) "
                f"{sorted(unknown)} -- written by a different compare.py?")
        payload = {k: v for k, v in rec.items() if k in _RESULT_FIELDS}
        try:
            r = Result(**payload)
        except TypeError as e:
            raise RunFileError(f"{path}: bad run record: {e}") from e
        if not r.solver or not r.instance:
            raise RunFileError(f"{path}: run record without solver/instance")
        target = aggregates if rec.get("record") == "aggregate" else grouped
        target.setdefault((r.solver, r.instance), []).append(r)

    results: dict[str, dict[str, Result]] = {}
    for key in sorted(set(grouped) | set(aggregates)):
        solver, instance = key
        active = [r for r in aggregates.get(key, [])
                  if not r.claim_superseded and r.claim_selected is not False]
        if len(active) > 1:
            raise RunFileError(
                f"{path}: {solver}/{instance} has {len(active)} selected "
                "aggregate records")
        if active:
            selected = active[0]
        elif aggregates.get(key):
            raise RunFileError(
                f"{path}: {solver}/{instance} has no selected aggregate")
        else:
            selected = compare.aggregate_repetitions(grouped[key])
        results.setdefault(solver, {})[instance] = selected

    if not results:
        raise RunFileError(f"{path}: sweep contains no run records")
    return Run(path=path, environment=env, results=results, metadata=metadata,
               segments=len(segments), segment=segment)


def pick_solver(run: Run, requested: str | None, which: str) -> str:
    if requested is not None:
        for name in run.solvers:
            if name.lower() == requested.lower():
                return name
        raise RunFileError(
            f"{run.path}: no solver {requested!r} in this sweep; "
            f"it has {run.solvers}")
    if len(run.solvers) == 1:
        return run.solvers[0]
    raise RunFileError(
        f"{run.path}: the {which} sweep has several solvers {run.solvers}; "
        f"say which one with --solver (or --{which}-solver)")


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
@dataclass
class Row:
    """One instance, side by side."""
    instance: str
    base: Result | None
    cand: Result | None
    time_ratio: float | None = None     # candidate / baseline, shifted
    iter_ratio: float | None = None
    findings: list[str] = dataclasses.field(default_factory=list)

    @property
    def comparable(self) -> bool:
        """Both sides certified, so their timings mean the same thing."""
        return (self.base is not None and self.cand is not None
                and compare.is_certified_success(self.base)
                and compare.is_certified_success(self.cand))


def _status(r: Result | None) -> str:
    return "-" if r is None else r.status.strip().lower()


def shifted_ratio(base: float | None, cand: float | None,
                  shift: float) -> float | None:
    """(cand + shift) / (base + shift). The shift is what makes a ratio
    meaningful when either side is zero or in the sub-millisecond noise."""
    if base is None or cand is None:
        return None
    if not (math.isfinite(base) and math.isfinite(cand)):
        return None
    denom = max(base, 0.0) + shift
    if denom <= 0.0:
        return None
    return (max(cand, 0.0) + shift) / denom


def compare_rows(base: dict[str, Result], cand: dict[str, Result],
                 time_shift: float, iter_shift: float,
                 obj_abs_tol: float, obj_rel_tol: float) -> list[Row]:
    rows: list[Row] = []
    for instance in sorted(set(base) | set(cand)):
        b, c = base.get(instance), cand.get(instance)
        row = Row(instance=instance, base=b, cand=c)

        if b is None:
            row.findings.append("missing-in-baseline")
        elif c is None:
            row.findings.append("missing-in-candidate")
        else:
            b_ok = compare.is_certified_success(b)
            c_ok = compare.is_certified_success(c)
            if b_ok and not c_ok:
                row.findings.append("proof-regression")
            elif c_ok and not b_ok:
                row.findings.append("proof-improvement")
            if _status(b) != _status(c):
                row.findings.append("status-mismatch")
            if b_ok and c_ok and not compare.objectives_agree(
                    b.objective, c.objective, obj_abs_tol, obj_rel_tol):
                row.findings.append("objective-mismatch")

        if row.comparable:
            assert row.base is not None and row.cand is not None
            row.time_ratio = shifted_ratio(row.base.seconds, row.cand.seconds,
                                           time_shift)
            row.iter_ratio = shifted_ratio(
                None if row.base.iterations is None else float(row.base.iterations),
                None if row.cand.iterations is None else float(row.cand.iterations),
                iter_shift)
        rows.append(row)
    return rows


def _total(rows: list[Row], side: str, attr: str) -> float:
    total = 0.0
    for row in rows:
        r = getattr(row, side)
        v = getattr(r, attr, None) if r is not None else None
        if v is not None and math.isfinite(float(v)):
            total += float(v)
    return total


def summarize(rows: list[Row], sgm_shift: float) -> dict[str, object]:
    """Aggregates. Every performance number is over the comparable rows only."""
    comparable = [r for r in rows if r.comparable]
    findings: dict[str, int] = {}
    for row in rows:
        for f in row.findings:
            findings[f] = findings.get(f, 0) + 1

    base_times = [r.base.seconds for r in comparable if r.base.seconds is not None]
    cand_times = [r.cand.seconds for r in comparable if r.cand.seconds is not None]
    base_sgm = compare.shifted_geomean(base_times, sgm_shift)
    cand_sgm = compare.shifted_geomean(cand_times, sgm_shift)

    ratios = [r.time_ratio for r in comparable if r.time_ratio is not None]
    iter_ratios = [r.iter_ratio for r in comparable if r.iter_ratio is not None]

    return {
        "instances": len(rows),
        "comparable": len(comparable),
        "findings": findings,
        "baseline_total_s": _total(comparable, "base", "seconds"),
        "candidate_total_s": _total(comparable, "cand", "seconds"),
        "baseline_total_wall_s": _total(comparable, "base", "wall_s"),
        "candidate_total_wall_s": _total(comparable, "cand", "wall_s"),
        "baseline_total_iterations": int(_total(comparable, "base", "iterations")),
        "candidate_total_iterations": int(_total(comparable, "cand", "iterations")),
        "baseline_sgm_s": base_sgm,
        "candidate_sgm_s": cand_sgm,
        "sgm_ratio": (None if not base_sgm or base_sgm <= 0.0 or cand_sgm is None
                      else cand_sgm / base_sgm),
        "geomean_time_ratio": geomean(ratios),
        "geomean_iteration_ratio": geomean(iter_ratios),
    }


def geomean(values: list[float]) -> float | None:
    """Plain geometric mean. Inputs are already shifted ratios, so they are
    strictly positive and need no second shift."""
    vals = [v for v in values if v is not None and v > 0.0 and math.isfinite(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def ratio_str(v: float | None) -> str:
    if v is None:
        return "-"
    if v < 1.0:
        return f"{v:.3f}x ({(1.0 / v):.2f}x faster)"
    return f"{v:.3f}x ({v:.2f}x slower)" if v > 1.0 else "1.000x"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
FINDING_ORDER = ["proof-regression", "status-mismatch", "objective-mismatch",
                 "missing-in-candidate", "missing-in-baseline",
                 "proof-improvement"]

# Findings that mean the two runs do not describe the same solved problem set.
FAILING_FINDINGS = {"proof-regression", "status-mismatch", "objective-mismatch"}
MISSING_FINDINGS = {"missing-in-candidate", "missing-in-baseline"}


def _cell(r: Result | None) -> str:
    if r is None:
        return "absent"
    proof = "" if r.proof is None else f"/{r.proof}"
    return f"{r.status}{proof} {compare.fmt_time(r.seconds)}"


def report(rows: list[Row], summary: dict[str, object], base: Run, cand: Run,
           base_solver: str, cand_solver: str, top: int) -> None:
    print(f"baseline : {base.label()}  solver={base_solver}  "
          f"instances={len(base.results[base_solver])}"
          + (f"  [sweep {base.segment} of {base.segments}]"
             if base.segments > 1 else ""))
    print(f"candidate: {cand.label()}  solver={cand_solver}  "
          f"instances={len(cand.results[cand_solver])}"
          + (f"  [sweep {cand.segment} of {cand.segments}]"
             if cand.segments > 1 else ""))
    print(f"{summary['instances']} instance(s) in the union, "
          f"{summary['comparable']} certified on both sides")

    findings = summary["findings"]
    assert isinstance(findings, dict)
    print("\nCORRECTNESS")
    if not findings:
        print("  no differences")
    for name in FINDING_ORDER:
        if name not in findings:
            continue
        hits = [r for r in rows if name in r.findings]
        print(f"  {name}: {findings[name]}")
        wid = max(len(r.instance) for r in hits)
        for r in hits:
            line = f"    {r.instance:<{wid}}  {_cell(r.base)}  ->  {_cell(r.cand)}"
            if name == "objective-mismatch":
                line += (f"   obj {compare.fmt_obj(r.base.objective)}"
                         f" -> {compare.fmt_obj(r.cand.objective)}")
            print(line)
    for name in sorted(set(findings) - set(FINDING_ORDER)):
        print(f"  {name}: {findings[name]}")

    print("\nPERFORMANCE (certified on both sides only)")
    if not summary["comparable"]:
        print("  nothing to compare")
        return
    print(f"  {'':<22} {'baseline':>12} {'candidate':>12}   ratio")
    for label, bk, ck in (
            ("total solve time (s)", "baseline_total_s", "candidate_total_s"),
            ("total wall time (s)", "baseline_total_wall_s",
             "candidate_total_wall_s"),
            ("total iterations", "baseline_total_iterations",
             "candidate_total_iterations")):
        b, c = summary[bk], summary[ck]
        assert isinstance(b, (int, float)) and isinstance(c, (int, float))
        r = (c / b) if b else None
        print(f"  {label:<22} {b:>12.4g} {c:>12.4g}   "
              f"{('-' if r is None else f'{r:.3f}x')}")
    b_sgm, c_sgm = summary["baseline_sgm_s"], summary["candidate_sgm_s"]
    sgm_ratio = summary["sgm_ratio"]
    b_txt = "-" if b_sgm is None else f"{b_sgm:.4g}"
    c_txt = "-" if c_sgm is None else f"{c_sgm:.4g}"
    r_txt = "-" if sgm_ratio is None else f"{sgm_ratio:.3f}x"
    print(f"  {'shifted geomean (s)':<22} {b_txt:>12} {c_txt:>12}   {r_txt}")
    print(f"\n  geomean of per-instance time ratios:      "
          f"{ratio_str(summary['geomean_time_ratio'])}")
    print(f"  geomean of per-instance iteration ratios: "
          f"{ratio_str(summary['geomean_iteration_ratio'])}")

    ranked = sorted((r for r in rows if r.time_ratio is not None),
                    key=lambda r: r.time_ratio or 0.0, reverse=True)
    slower = [r for r in ranked if r.time_ratio and r.time_ratio > 1.0][:top]
    faster = [r for r in reversed(ranked)
              if r.time_ratio and r.time_ratio < 1.0][:top]
    for title, subset in (("TOP REGRESSIONS", slower),
                          ("TOP IMPROVEMENTS", faster)):
        print(f"\n{title}")
        if not subset:
            print("  none")
            continue
        wid = max(len(r.instance) for r in subset)
        for r in subset:
            print(f"  {r.instance:<{wid}}  {compare.fmt_time(r.base.seconds):>9}"
                  f" -> {compare.fmt_time(r.cand.seconds):>9}   "
                  f"{r.time_ratio:.3f}x"
                  + (f"   iters {r.base.iterations} -> {r.cand.iterations}"
                     if r.base.iterations is not None
                     and r.cand.iterations is not None else ""))


def to_json(rows: list[Row], summary: dict[str, object], base: Run, cand: Run,
            base_solver: str, cand_solver: str) -> dict[str, object]:
    return {
        "tool": "compare_runs",
        "baseline": {"path": str(base.path), "solver": base_solver,
                     "environment": base.environment},
        "candidate": {"path": str(cand.path), "solver": cand_solver,
                      "environment": cand.environment},
        "summary": summary,
        "instances": [
            {"instance": r.instance,
             "findings": r.findings,
             "comparable": r.comparable,
             "time_ratio": r.time_ratio,
             "iteration_ratio": r.iter_ratio,
             "baseline": None if r.base is None else dataclasses.asdict(r.base),
             "candidate": None if r.cand is None else dataclasses.asdict(r.cand)}
            for r in rows],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare two compare.py JSONL runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--solver", default=None,
                    help="solver label to compare in both files")
    ap.add_argument("--baseline-solver", default=None,
                    help="solver label in the baseline file (overrides --solver)")
    ap.add_argument("--candidate-solver", default=None,
                    help="solver label in the candidate file (overrides --solver)")
    ap.add_argument("--segment", type=int, default=-1,
                    help="which sweep to read from an appended file "
                         "(0-based, negative counts from the end; default -1)")
    ap.add_argument("--top", type=int, default=10,
                    help="how many regressions/improvements to list")
    ap.add_argument("--time-shift", type=float, default=0.01,
                    help="seconds added to both sides of a time ratio, so a "
                         "near-zero baseline cannot manufacture a huge ratio")
    ap.add_argument("--iteration-shift", type=float, default=1.0,
                    help="iterations added to both sides of an iteration ratio")
    ap.add_argument("--sgm-shift", type=float, default=1.0,
                    help="shift for the shifted geometric mean, in seconds")
    ap.add_argument("--obj-rel-tol", type=float, default=1e-7,
                    help="claim-facing objective agreement: max(abs_tol, rel_tol*(1+|reference|)). 1e-7 is the public gate; use the named 1e-6 continuity command for the historical lane.")
    ap.add_argument("--obj-abs-tol", type=float, default=1e-7)
    ap.add_argument("--allow-missing", action="store_true",
                    help="do not fail solely because an instance is in one run "
                         "and not the other")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the full comparison here")
    args = ap.parse_args(argv)

    if args.top < 0:
        ap.error("--top cannot be negative")
    for name in ("time_shift", "iteration_shift", "sgm_shift",
                 "obj_rel_tol", "obj_abs_tol"):
        v = getattr(args, name)
        if not math.isfinite(v) or v < 0.0:
            ap.error(f"--{name.replace('_', '-')} must be finite and nonnegative")

    try:
        base = load_run(args.baseline, args.segment)
        cand = load_run(args.candidate, args.segment)
        base_solver = pick_solver(base, args.baseline_solver or args.solver,
                                  "baseline")
        cand_solver = pick_solver(cand, args.candidate_solver or args.solver,
                                  "candidate")
    except RunFileError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    rows = compare_rows(base.results[base_solver], cand.results[cand_solver],
                        args.time_shift, args.iteration_shift,
                        args.obj_abs_tol, args.obj_rel_tol)
    summary = summarize(rows, args.sgm_shift)
    report(rows, summary, base, cand, base_solver, cand_solver, args.top)

    if args.json is not None:
        with args.json.open("w") as fh:
            json.dump(to_json(rows, summary, base, cand, base_solver,
                              cand_solver), fh, indent=2)
            fh.write("\n")
        print(f"\nJSON: {args.json}")

    findings = summary["findings"]
    assert isinstance(findings, dict)
    failed = any(f in findings for f in FAILING_FINDINGS)
    if not args.allow_missing:
        failed = failed or any(f in findings for f in MISSING_FINDINGS)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
