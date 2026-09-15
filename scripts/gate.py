#!/usr/bin/env python3
"""The merge gate: does this tree still deserve its baseline?

    scripts/gate.py                              # netlib, default engine path
    scripts/gate.py --suite netlib --tag dual --method dual
    scripts/gate.py --determinism --tests        # the full pre-merge run
    scripts/gate.py --accept --reason "WS1 phase-1 subproblem method"

WS6 of PERFORMANCE_PLAN_20260908.md. Every workstream in that plan merges on a
green run of this script, which is the only reason it exists: several past
regressions -- the Forrest-Tomlin slowdown, dual phase-1 re-entry, the cut
filter that quietly lost gt2's proof, and pilotnov's proof under the
unconditional entering-column shift -- were all found by a human noticing a
number, one at a time, days later.

WHAT IT ENFORCES, in order of severity:

    1. Correctness, in three halves (the arithmetic is not the point). Against
       the BASELINE: a model certified there must still be certified, its
       status must not change, and two certified objectives must agree.
       Against the REFERENCE: a certified objective must also agree with the
       external solver's -- see evaluate_reference_objectives(). Against the
       PUBLISHED OPTIMUM, where a suite commits one: incumbent, dual bound and
       status must bracket it -- see evaluate_known_optima(), which is the only
       rule that says anything at all about the 39 of 40 miplib-small
       instances that never finish. The last two exist because the first is
       blind to an answer that is wrong in both sweeps. None can ever be
       waived: there is no allow-list entry for a lost proof or a wrong number.
    2. Per-model work. No model may take more than +10% pivots (MILP: nodes)
       against the baseline unless benchmarks/results/gate-allow.json carries
       an entry for it, with a ratio bound and a written reason.
    3. Aggregates. The geometric time ratio (G2) and the geometric pivot ratio
       (G5) may not regress by more than 2%.
    4. The HiGHS reference. G2 measured against a committed HiGHS sweep may not
       regress by more than 5%, and no model HiGHS solves in >= 50 ms may lose
       more than 15% of its time while its pivot count barely moves -- that is
       a per-pivot cost regression, which rules 2 and 3 are blind to because
       they compare SOR against SOR. Waivable only with a reason that carries
       the bucket profile, so the next reader knows where the time went.

Optionally (--determinism) it also runs the suite twice and requires identical
pivot counts and BIT-IDENTICAL objectives, because rule 4 of the plan's
correctness section is that nothing in the solver may depend on timing.

This script owns no definitions. "Certified", "objectives agree", the
median-of-repetitions aggregation, the shifted ratios and the aggregate
geomeans all come from compare.py / compare_runs.py, so the gate cannot drift
from the harness that produced its numbers or from the tool humans read the
diff with.

BASELINES live in benchmarks/results/ and are committed. They move only under
--accept, which demands a --reason and records it in the file, so `git log -p`
on a baseline answers "who said this regression was acceptable, and why".
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "CMakeLists.txt").exists())
SCRIPTS = Path(__file__).resolve().parent
RESULTS = ROOT / "benchmarks" / "results"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


compare = _load("sor_compare", SCRIPTS / "compare.py")
runs = _load("sor_compare_runs", SCRIPTS / "compare_runs.py")
protocol = _load("sor_claim_protocol", SCRIPTS / "claim_protocol.py")
# read_reference / model_minimizes / check_sound already exist and are already
# careful (sense from the MODEL, per-entry tolerances for the MIPLIB 3.0 tables
# that truncate to six significant digits). The gate reuses them rather than
# growing a second, subtly different copy.
miplib = _load("sor_miplib_eval", SCRIPTS / "miplib_eval.py")


# --------------------------------------------------------------------------
# Suites
#
# A note on the MILP suites: most of their instances stop at the time limit
# rather than proving, and a time-limited run's node count is a timing
# artifact. Those rows are not certified on both sides, so they never enter a
# ratio -- what the gate protects there is the set of instances that PROVE, and
# the objectives of the ones that do. Do not run --determinism against a
# time-limited MILP suite and conclude the solver is nondeterministic: raise
# the limit until the instances you care about prove, or gate them by proof
# alone.
# --------------------------------------------------------------------------
SUITES: dict[str, dict[str, object]] = {
    "netlib": {
        "models": "benchmarks/netlib/mps",
        "solver": "sor:simplex",
        "time_limit": 300.0,
        "work": "pivots",
    },
    "miplib-small": {
        "models": "benchmarks/miplib-small/mps",
        "solver": "sor:milp",
        "time_limit": 30.0,
        "work": "nodes",
    },
    "miplib-easy": {
        "models": "benchmarks/miplib-easy/mps",
        "solver": "sor:milp",
        "time_limit": 60.0,
        "work": "nodes",
    },
}

# Correctness findings from compare_runs. These are never waivable.
FATAL_FINDINGS = ("proof-regression", "status-mismatch", "objective-mismatch",
                  "missing-in-candidate")


def baseline_path(suite: str, tag: str | None) -> Path:
    stem = f"baseline-{suite}" + (f"-{tag}" if tag else "")
    return RESULTS / f"{stem}.jsonl"


def allow_path() -> Path:
    return RESULTS / "gate-allow.json"


def default_reference(suite: str) -> Path | None:
    """Newest committed HiGHS sweep for this suite, if one exists."""
    hits = sorted(RESULTS.glob(f"reference-highs-{suite}-*.jsonl"))
    return hits[-1] if hits else None


def load_reference(path: Path, label: str) -> dict:
    """instance -> seconds, for the reference solver in a committed sweep."""
    try:
        run = runs.load_run(path)
    except runs.RunFileError as e:
        raise SystemExit(f"gate: {e}")
    if label not in run.results:
        raise SystemExit(f"gate: {path} has no solver {label!r}; "
                         f"it has {run.solvers}")
    return {k: v for k, v in run.results[label].items()}


def measure_g2(rows: list, reference: dict, time_limit: float = 60.0,
               shift: float = 1.0) -> float | None:
    """The PUBLIC metric: SGM(candidate) / SGM(reference).

    This was a geometric mean of per-model ratios. That is a different
    statistic and it disagreed with the public metric by 50% on Netlib-93
    (0.8684 against the true 1.4307), because it weights a 0.4 ms model exactly
    like an 11 s one. The definition now comes from compare.py so this gate and
    a published report cannot disagree.

    Note this is the only rule in gate.py that looks at HiGHS. Rules 1-3 are
    internal SOR-vs-SOR regression checks against the pinned baseline and are
    deliberately kept separate: a change can be a legitimate internal
    regression while the public claim still holds, and vice versa.
    """
    cand_times, ref_times = [], []
    for row in rows:
        ref = reference.get(row.instance)
        if ref is None:
            continue
        cand_times.append(compare.par2_seconds(row.cand, time_limit))
        ref_times.append(compare.par2_seconds(ref, time_limit))
    if not cand_times:
        return None
    cand_sgm = compare.shifted_geomean(cand_times, shift)
    ref_sgm = compare.shifted_geomean(ref_times, shift)
    if not ref_sgm:
        return None
    return cand_sgm / ref_sgm


def recorded_g2_of(path: Path) -> float | None:
    """The public SGM ratio stamped into a baseline by its last --accept."""
    try:
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                stamp = rec.get("gate_accept")
                if rec.get("record") == "gate_accept" and \
                        isinstance(rec.get("payload"), dict):
                    stamp = rec["payload"]
                if isinstance(stamp, dict) and "g2_vs_reference" in stamp:
                    return float(stamp["g2_vs_reference"])
                break
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def load_allow(suite: str, tag: str | None,
               known_instances: set[str] | None = None) -> dict[str, dict]:
    """Committed per-model waivers for rule 2. Never consulted for rule 1."""
    path = allow_path()
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"gate: {path}: {e}")
    if not isinstance(doc, dict):
        raise SystemExit(f"gate: {path}: top level must be an object")
    schema = doc.get("_schema")
    if not isinstance(schema, dict) or schema.get("version") != 1:
        raise SystemExit(f"gate: {path}: unsupported or missing schema version")
    key = suite + (f"-{tag}" if tag else "")
    if key not in doc:
        raise SystemExit(f"gate: {path}: no entry for known suite {key!r}")
    entries = doc.get(key)
    if not isinstance(entries, dict):
        raise SystemExit(f"gate: {path}: {key} must be an object")
    for model, entry in entries.items():
        if (not isinstance(entry, dict) or
                not isinstance(entry.get("reason"), str) or
                not entry["reason"].strip()):
            raise SystemExit(
                f"gate: {path}: {key}.{model} needs a nonempty \"reason\"")
        unknown = sorted(set(entry) - {"reason", "max_work_ratio"})
        if unknown:
            raise SystemExit(
                f"gate: {path}: {key}.{model} has unknown fields {unknown}")
        if known_instances is not None and model not in known_instances:
            raise SystemExit(
                f"gate: {path}: {key}.{model} is not a known suite instance")
        ratio = entry.get("max_work_ratio")
        if ratio is not None and not (type(ratio) in (int, float) and
                                      math.isfinite(ratio) and ratio > 0):
            raise SystemExit(
                f"gate: {path}: {key}.{model}.max_work_ratio must be positive")
    return entries


# --------------------------------------------------------------------------
# Running the sweep
# --------------------------------------------------------------------------
def run_sweep(args: argparse.Namespace, suite: dict[str, object],
              out: Path) -> None:
    """One compare.py sweep into a fresh JSONL. compare.py's own exit code is
    about ITS checks (a missing reference solver, a mismatch against HiGHS);
    the gate makes its own decisions from the data, so a nonzero exit here is
    reported but not by itself fatal -- except when it produced no file."""
    if out.exists():
        out.unlink()
    cmd = [sys.executable, str(SCRIPTS / "compare.py"),
           str(ROOT / str(suite["models"])),
           "--solvers", args.solver or str(suite["solver"]),
           "--time-limit", str(args.time_limit or suite["time_limit"]),
           "--jsonl", str(out),
           "--allow-unchecked", "--allow-unavailable"]
    if args.method:
        cmd += ["--method", args.method]
    if args.cpu is not None:
        cmd += ["--cpu", str(args.cpu)]
    if args.exe:
        cmd += ["--exe", str(args.exe)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    # The =form, because these values usually start with '-' and argparse on
    # the far side would read them as its own options otherwise.
    cmd += [f"--sor-arg={extra}" for extra in args.sor_arg]
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if not out.exists():
        raise SystemExit(f"gate: sweep produced no {out} "
                         f"(compare.py exited {proc.returncode})")


def sole_solver(run: "runs.Run", which: str) -> str:
    return runs.pick_solver(run, None, which)


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------
def evaluate_reference(rows: list, reference: dict, allow: dict[str, dict],
                       g2_bound: float, model_time_bound: float,
                       model_work_slack: float, min_reference_seconds: float,
                       recorded_g2: float | None
                       ) -> tuple[list[str], list[str]]:
    """Rule 4: compare against a fixed external solver, not against ourselves.

    Rules 2 and 3 measure SOR against SOR, so a change that costs time without
    moving a single pivot -- the whole class of per-pivot-cost regressions --
    is invisible to them. This one is not: a model whose pivots barely moved
    but whose time grew is exactly that, and it is reported per model rather
    than only in an aggregate where one model's loss hides inside 92 others."""
    failures: list[str] = []
    notes: list[str] = []

    ratios = []
    for row in rows:
        if row.cand is None or not row.comparable:
            continue
        ref = reference.get(row.instance)
        if ref is None or not ref.seconds or not row.cand.seconds:
            continue
        ratios.append(row.cand.seconds / ref.seconds)
    g2 = runs.geomean(ratios)
    if g2 is not None:
        notes.append(f"G2 vs reference: {g2:.4f}x over {len(ratios)} model(s)")
        # The bar is a REGRESSION bar, not an absolute one: SOR is allowed to
        # be slower than HiGHS (that is the whole point of the plan), it is not
        # allowed to get slower than it was when the baseline was accepted.
        if recorded_g2 is None:
            notes.append("no G2 recorded in the baseline yet, so rule 4's "
                         "aggregate half is informational on this run")
        elif g2 > recorded_g2 * g2_bound:
            failures.append(
                f"G2 against the HiGHS reference {g2:.4f}x, over the "
                f"{recorded_g2:.4f}x recorded at the last --accept by more "
                f"than {(g2_bound - 1.0) * 100:.0f}%")

    for row in rows:
        if row.base is None or row.cand is None or not row.comparable:
            continue
        ref = reference.get(row.instance)
        if ref is None or not ref.seconds or ref.seconds < min_reference_seconds:
            continue
        if not row.base.seconds or not row.cand.seconds:
            continue
        time_ratio = row.cand.seconds / row.base.seconds
        if time_ratio <= model_time_bound:
            continue
        # Only a per-pivot cost regression: if the work moved, rule 2 owns it.
        if row.iter_ratio is None or abs(row.iter_ratio - 1.0) > model_work_slack:
            continue
        detail = (f"{row.instance}: {row.base.seconds * 1000:.1f} -> "
                  f"{row.cand.seconds * 1000:.1f} ms ({time_ratio:.3f}x) on "
                  f"{row.base.iterations} -> {row.cand.iterations} pivots "
                  f"(HiGHS {ref.seconds * 1000:.1f} ms)")
        entry = allow.get(row.instance)
        if entry is None:
            failures.append(detail + " -- per-pivot cost regression, and not "
                                     "in " + allow_path().name)
        elif "ms" not in entry["reason"]:
            failures.append(
                detail + " -- its allow-list reason carries no bucket profile; "
                         "say where the time went (factorization/PRICE/FTRAN/"
                         "BTRAN/ratio/update ms)")
        else:
            notes.append(detail + f" -- {entry['reason']}")
    return failures, notes


def evaluate_known_optima(rows: list, models_dir: Path, tol: float
                          ) -> tuple[list[str], list[str]]:
    """Rule 1 for the MILP suites, against the published optimum.

    The netlib half of rule 1 has an external oracle because a committed HiGHS
    sweep exists. The MILP suites have something better sitting unused next to
    their models -- benchmarks/<suite>/reference.csv, the MIPLIB `=opt=`
    values -- and it does not need a timing run to be trustworthy.

    It also protects far more than a proof check can. On miplib-small at 30 s
    exactly ONE instance of 40 is certified, so gating by proof alone leaves 39
    unexamined; every one of them still has an incumbent and a dual bound that
    must bracket the published optimum. check_sound() tests all three
    impossibilities: an incumbent better than the optimum, a dual bound past it
    on the bounding side, and Infeasible on an instance that has a solution.

    This is what caught `pg`: certified ProvedGlobalEpsilon at 7250 against a
    published optimum of -8674.34, because the time limit expired inside the
    root node's LP and the abandoned subtree left the dual bound seeded at the
    incumbent. Unwaivable, like the rest of rule 1.
    """
    failures: list[str] = []
    notes: list[str] = []
    ref = miplib.read_reference(models_dir.parent / "reference.csv")
    if not ref:
        notes.append(f"no reference.csv beside {models_dir}, so the "
                     f"published-optimum half of rule 1 is not enforced")
        return failures, notes
    checked = 0
    for row in rows:
        if row.cand is None:
            continue
        name = row.instance
        stem = name[:-4] if name.endswith(".mps") else name
        entry = ref.get(stem)
        if entry is None:
            continue
        checked += 1
        # check_sound() tests `status == "Optimal"` exactly, so normalise here
        # rather than depend on how a given sweep spelled it -- the whole point
        # of this rule is that it fires on a claim of optimality.
        status = "Optimal" if compare.is_optimal(row.cand) else row.cand.status
        rec = {"status": status, "objective": row.cand.objective,
               "dual_bound": getattr(row.cand, "dual_bound", None)}
        msg = miplib.check_sound(
            rec, entry, tol, miplib.model_minimizes(models_dir / name))
        if msg:
            failures.append(
                f"{name}: {msg} [status={row.cand.status} "
                f"proof={row.cand.proof or '-'}] -- unsound against the "
                f"published optimum, which no allow-list entry can waive")
    notes.append(f"checked against published optima: {checked} model(s), "
                 f"{len(failures)} unsound")
    return failures, notes


def evaluate_reference_objectives(rows: list, reference: dict,
                                  obj_abs_tol: float, obj_rel_tol: float
                                  ) -> tuple[list[str], list[str]]:
    """Rule 1's second half: is the answer right, not merely unchanged?

    Every other rule in this file compares SOR against SOR. That makes one
    whole class of defect invisible: an answer that is wrong in the baseline
    AND in the candidate reads as "no differences", and the gate goes green
    forever. It is not hypothetical. At --tol 1e-6 the forced-dual path
    returned pilot.mps as -557.48163926 with proof ProvedOptimalFP, against
    HiGHS's -557.4897292744 -- a 1.5e-05 relative error, certified, on a
    baseline every gate run had called clean.

    An optimal objective is unique even where the optimal vertex is not, so a
    disagreement here is a defect in one of the two solvers and not a matter of
    taste. Unwaivable, like the rest of rule 1: gate-allow.json is about how
    much WORK a model may cost, never about what answer it may return.
    """
    failures: list[str] = []
    notes: list[str] = []
    checked = 0
    for row in rows:
        if row.cand is None or not compare.is_certified_success(row.cand):
            continue
        ref = reference.get(row.instance)
        if ref is None or not compare.is_certified_success(ref):
            continue
        if ref.objective is None or row.cand.objective is None:
            continue
        checked += 1
        if compare.objectives_agree(row.cand.objective, ref.objective,
                                    obj_abs_tol, obj_rel_tol):
            continue
        denom = max(abs(row.cand.objective), abs(ref.objective), 1.0)
        rel = abs(row.cand.objective - ref.objective) / denom
        failures.append(
            f"{row.instance}: certified objective {row.cand.objective!r} "
            f"({row.cand.proof or '-'}) disagrees with the reference "
            f"{ref.objective!r} by {rel:.2e} relative -- a certified wrong "
            f"answer, which no allow-list entry can waive")
    notes.append(f"objectives checked against the reference: {checked} "
                 f"model(s), {len(failures)} disagreement(s)")
    return failures, notes


def evaluate(rows: list, allow: dict[str, dict],
             work_name: str, work_bound: float, aggregate_bound: float,
             aggregate_min: int = 10) -> tuple[list[str], list[str]]:
    """Return (failures, waived). Failures are what make the gate red."""
    failures: list[str] = []
    waived: list[str] = []

    for row in rows:
        fatal = [f for f in row.findings if f in FATAL_FINDINGS]
        for f in fatal:
            base = "-" if row.base is None else row.base.status
            cand = "-" if row.cand is None else row.cand.status
            proof = "-" if row.cand is None else (row.cand.proof or "-")
            failures.append(
                f"{row.instance}: {f} (baseline {base}, candidate {cand}/{proof})")

    waived_models: set[str] = set()
    for row in rows:
        if row.iter_ratio is None or row.iter_ratio <= work_bound:
            continue
        entry = allow.get(row.instance)
        b = None if row.base is None else row.base.iterations
        c = None if row.cand is None else row.cand.iterations
        detail = (f"{row.instance}: {work_name} {b} -> {c} "
                  f"({row.iter_ratio:.3f}x)")
        if entry is None:
            failures.append(detail + f", over the {work_bound:.2f}x bar and "
                                     f"not in {allow_path().name}")
            continue
        bound = entry.get("max_work_ratio")
        if bound is not None and row.iter_ratio > float(bound):
            failures.append(
                detail + f", over its own allow-list bound {float(bound):.3f}x")
        else:
            waived.append(detail + f" -- {entry['reason']}")
            waived_models.add(row.instance)

    # The aggregate bars describe a SUITE, and they are computed over the rows
    # the allow-list did NOT waive. Otherwise a waiver would be worth nothing:
    # the same accepted regression would come back as a G5 failure, and the
    # only way to merge would be to widen the aggregate bar for everything
    # else. What stops that from hiding death by a thousand cuts is that each
    # waiver costs a written reason in a committed file.
    scored = [r for r in rows if r.comparable
              and r.instance not in waived_models]
    if len(scored) >= aggregate_min:
        for attr, label in (("time_ratio", "G2 geometric time ratio"),
                            ("iter_ratio", f"G5 geometric {work_name} ratio")):
            values = [getattr(r, attr) for r in scored
                      if getattr(r, attr) is not None]
            value = runs.geomean(values)
            if value is None:
                continue
            if value > aggregate_bound:
                failures.append(
                    f"{label} {value:.4f}x over {len(scored)} instance(s), "
                    f"over the {aggregate_bound:.2f}x aggregate bar")
    return failures, waived


def check_determinism(a: "runs.Run", b: "runs.Run", solver: str,
                      work_name: str) -> list[str]:
    """Two sweeps of the same tree must agree on every pivot and every
    objective BIT. A float that differs in its last place means some choice in
    the solver depended on something other than the model."""
    left, right = a.results[solver], b.results[solver]
    failures: list[str] = []
    for instance in sorted(set(left) | set(right)):
        x, y = left.get(instance), right.get(instance)
        if x is None or y is None:
            failures.append(f"{instance}: present in only one of the two runs")
            continue
        if x.iterations != y.iterations:
            failures.append(
                f"{instance}: {work_name} {x.iterations} vs {y.iterations}")
        if x.status != y.status:
            failures.append(f"{instance}: status {x.status} vs {y.status}")
        ox, oy = x.objective, y.objective
        if (ox is None) != (oy is None):
            failures.append(f"{instance}: objective {ox} vs {oy}")
        elif ox is not None and oy is not None and ox.hex() != oy.hex():
            failures.append(
                f"{instance}: objective bits {ox.hex()} vs {oy.hex()}")
    return failures


# --------------------------------------------------------------------------
# Side quests the plan puts in the gate
# --------------------------------------------------------------------------
def run_ctest(build: Path, label: str) -> list[str]:
    if not (build / "CTestTestfile.cmake").exists():
        return [f"{label}: {build} is not a configured build directory"]
    print(f"\n$ ctest --test-dir {build}", flush=True)
    proc = subprocess.run(["ctest", "--output-on-failure", "-j4"], cwd=build,
                          capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-15:]
    print("\n".join(tail))
    return [] if proc.returncode == 0 else [
        f"{label}: ctest exited {proc.returncode} in {build}"]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def write_markdown(path: Path, rows: list, summary: dict, base, cand,
                   base_solver: str, cand_solver: str,
                   failures: list[str], waived: list[str], top: int) -> None:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runs.report(rows, summary, base, cand, base_solver, cand_solver, top)
    body = buf.getvalue()
    verdict = "FAIL" if failures else "PASS"
    lines = [f"# gate: {verdict}", "",
             f"generated {_dt.datetime.now().isoformat(timespec='seconds')}",
             ""]
    if failures:
        lines += ["## Failures", ""] + [f"- {f}" for f in failures] + [""]
    if waived:
        lines += ["## Waived by the allow-list", ""] + \
                 [f"- {w}" for w in waived] + [""]
    lines += ["## Diff", "", "```text", body.rstrip(), "```", ""]
    path.write_text("\n".join(lines))
    print(f"\nmarkdown: {path}")


def refuse_unfit_baseline(candidate: Path) -> list[str]:
    """Reasons this sweep must not become a committed baseline.

    A baseline is what every later regression is measured against, so it must
    be REPRODUCIBLE. That means the metadata proving it is reproducible has to
    be PRESENT -- absence is a refusal, not a pass. An earlier version checked
    only for explicitly bad values, so a sweep that recorded nothing at all
    sailed through, which is precisely the sweep you cannot reproduce.
    """
    _, problems = protocol.validate_baseline(candidate, require_accept=False)
    return problems


def accept(candidate: Path, baseline: Path, reason: str, suite: str,
           tag: str | None, g2: float | None = None) -> None:
    """Promote a sweep to the committed baseline.

    The protocol check lives HERE, not at the call sites, because there is more
    than one call site and a guard that can be reached around is not a guard.
    """
    unfit = refuse_unfit_baseline(candidate)
    if unfit:
        print("\ngate: refusing to make this sweep a baseline:", file=sys.stderr)
        for u in unfit:
            print(f"  - {u}", file=sys.stderr)
        print("  A baseline defines what 'no regression' means for every later "
              "run. Re-measure under the claim protocol "
              "(scripts/claim_run.py).", file=sys.stderr)
        raise SystemExit(1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    accepted_utc = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    payload = {"reason": reason,
               "suite": suite + (f"-{tag}" if tag else ""),
               "accepted_utc": accepted_utc,
               "baseline_version": accepted_utc,
               "claim_protocol_version": protocol.PROTOCOL_VERSION}
    # Rule 4 compares the next run against this, so a baseline that moves
    # carries the reference ratio it was accepted at.
    if g2 is not None:
        payload["g2_vs_reference"] = round(float(g2), 6)
    stamp = {"record": "gate_accept", "payload": payload}
    text = candidate.read_text()
    # The accept stamp rides in front of the sweep's own environment record,
    # so `head -2` on a baseline shows both why it moved and what produced it.
    baseline.write_text(json.dumps(stamp) + "\n" + text)
    print(f"accepted: {baseline}\n  reason: {reason}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the merge gate for one benchmark suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--suite", choices=sorted(SUITES), default="netlib")
    ap.add_argument("--tag", default=None,
                    help="baseline suffix, for a second baseline of the same "
                         "suite (e.g. --tag dual --method dual)")
    ap.add_argument("--solver", default=None, help="override the suite solver")
    ap.add_argument("--method", choices=("auto", "primal", "dual"), default=None)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--cpu", type=int, default=None,
                    help="pin child solvers to this logical CPU")
    ap.add_argument("--exe", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N models (for smoke-testing the gate)")
    ap.add_argument("--sor-arg", action="append", default=[], metavar="FLAG",
                    help="extra sor_solve flag for the candidate sweep; "
                         "repeatable. This is how a tuning parameter is put "
                         "in front of the gate before it becomes a default. "
                         "Use the =form for flags: "
                         "--sor-arg=--refactor-work-ratio --sor-arg=2.0")
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="where to write the candidate sweep "
                         "(default: benchmarks/results/gate-<suite>.jsonl)")
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--work-bar", type=float, default=1.10,
                    help="per-model pivot/node ratio bar (default 1.10)")
    ap.add_argument("--aggregate-bar", type=float, default=1.02,
                    help="G2/G5 aggregate ratio bar (default 1.02)")
    ap.add_argument("--reference", type=Path, default=None,
                    help="committed HiGHS sweep for rule 4 (default: newest "
                         "benchmarks/results/reference-highs-<suite>-*.jsonl)")
    ap.add_argument("--reference-solver", default="highs")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip rule 4 entirely")
    ap.add_argument("--reference-g2-bar", type=float, default=1.05)
    ap.add_argument("--reference-model-time-bar", type=float, default=1.15,
                    help="per-model time ratio bar for rule 4 (default 1.15)")
    ap.add_argument("--reference-work-slack", type=float, default=0.05,
                    help="how far pivots may move and still count as "
                         "'unchanged work' for rule 4 (default 5%%)")
    ap.add_argument("--reference-min-seconds", type=float, default=0.05,
                    help="rule 4 only polices models the reference solves in "
                         "at least this long (default 50 ms)")
    ap.add_argument("--aggregate-min-instances", type=int, default=10,
                    help="skip the aggregate bars below this many comparable "
                         "instances, where a geomean is just one model "
                         "(default 10)")
    ap.add_argument("--sgm-shift", type=float, default=1.0)
    ap.add_argument("--iter-shift", type=float, default=100.0,
                    help="shift for pivot ratios, so a 12 -> 15 pivot model "
                         "does not read as a 25%% regression")
    ap.add_argument("--obj-rel-tol", type=float, default=1e-7,
                    help="relative objective agreement, scaled by "
                         "(1 + |reference objective|). 1e-7 is the public "
                         "gate; the former 1e-6 default was looser than "
                         "either solver's own optimality tolerance.")
    ap.add_argument("--obj-abs-tol", type=float, default=1e-7,
                    help="claim-facing objective agreement: max(abs_tol, rel_tol*(1+|reference|)). 1e-7 is the public gate; use the named 1e-6 continuity command for the historical lane.")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--determinism", action="store_true",
                    help="also sweep twice and require identical pivots and "
                         "bit-identical objectives")
    ap.add_argument("--tests", action="store_true",
                    help="also run ctest in --build")
    ap.add_argument("--build", type=Path, default=ROOT / "build")
    ap.add_argument("--sanitize", action="store_true",
                    help="also run ctest in --asan-build")
    ap.add_argument("--asan-build", type=Path, default=ROOT / "build-asan")
    ap.add_argument("--reuse", type=Path, default=None,
                    help="skip the sweep and gate this existing JSONL")
    ap.add_argument("--accept", action="store_true",
                    help="write the candidate over the baseline")
    ap.add_argument("--reason", default=None,
                    help="why the baseline is allowed to move (required "
                         "with --accept)")
    args = ap.parse_args(argv)

    if args.accept and not args.reason:
        ap.error("--accept requires --reason: the baseline records why it moved")
    if args.work_bar < 1.0 or args.aggregate_bar < 1.0:
        ap.error("bars are ratios and cannot be below 1.0")

    suite = SUITES[args.suite]
    models_dir = ROOT / str(suite["models"])
    if not models_dir.exists():
        print(f"gate: {models_dir} does not exist", file=sys.stderr)
        return 2

    baseline = args.baseline or baseline_path(args.suite, args.tag)
    candidate = args.reuse or args.jsonl or (
        RESULTS / f"gate-{args.suite}{'-' + args.tag if args.tag else ''}.jsonl")
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.reuse is None:
        run_sweep(args, suite, candidate)

    if not baseline.exists():
        if not args.accept:
            print(f"\ngate: no baseline at {baseline}.\n"
                  f"      Establish one from a tree you trust:\n"
                  f"        scripts/gate.py --suite {args.suite}"
                  + (f" --tag {args.tag}" if args.tag else "")
                  + " --accept --reason \"...\"", file=sys.stderr)
            return 2
        accept(candidate, baseline, args.reason, args.suite, args.tag)
        return 0

    try:
        base_run = runs.load_run(baseline)
        cand_run = runs.load_run(candidate)
    except runs.RunFileError as e:
        print(f"gate: {e}", file=sys.stderr)
        return 2

    try:
        base_solver = sole_solver(base_run, "baseline")
        cand_solver = sole_solver(cand_run, "candidate")
    except runs.RunFileError as e:
        print(f"gate: {e}", file=sys.stderr)
        return 2

    work_name = str(suite["work"])
    rows = runs.compare_rows(
        base_run.results[base_solver], cand_run.results[cand_solver],
        args.sgm_shift, args.iter_shift, args.obj_abs_tol, args.obj_rel_tol)
    summary = runs.summarize(rows, args.sgm_shift)
    known_instances = {p.name for p in compare.collect_models(
        [str(models_dir)], None)}
    allow = load_allow(args.suite, args.tag, known_instances)
    failures, waived = evaluate(rows, allow, work_name, args.work_bar,
                                args.aggregate_bar,
                                args.aggregate_min_instances)

    # Rule 1 against the published optimum, where one is committed next to the
    # models. Runs before the reference block because it needs no sweep, no
    # timing and no baseline -- only the answer and the literature.
    optima_failures, optima_notes = evaluate_known_optima(
        rows, models_dir, args.obj_rel_tol)
    failures += optima_failures

    reference_path = args.reference or default_reference(args.suite)
    reference_notes: list[str] = []
    measured_g2: float | None = None
    if args.no_reference:
        pass
    elif reference_path is None or not reference_path.exists():
        print(f"\nnote: no HiGHS reference for suite {args.suite}; rule 4 is "
              f"not enforced. Produce one with\n"
              f"      benchmarks/.venv-baseline/bin/python scripts/compare.py "
              f"<models> --solvers sor:simplex,highs --jsonl "
              f"benchmarks/results/reference-highs-{args.suite}-<date>.jsonl")
    else:
        ref = load_reference(reference_path, args.reference_solver)
        measured_g2 = measure_g2(rows, ref)
        of, obj_notes = evaluate_reference_objectives(
            rows, ref, args.obj_abs_tol, args.obj_rel_tol)
        rf, time_notes = evaluate_reference(
            rows, ref, allow, args.reference_g2_bar,
            args.reference_model_time_bar, args.reference_work_slack,
            args.reference_min_seconds, recorded_g2_of(baseline))
        failures += of + rf
        reference_notes = obj_notes + time_notes
        print(f"\nreference: {reference_path.name} "
              f"({args.reference_solver}, {len(ref)} models)")
        for n in reference_notes:
            print(f"  {n}")
    # Coverage, stated out loud. Rules 2 and 3 compare only rows that are
    # certified on BOTH sides, so on a suite where most instances stop at the
    # time limit the gate is protecting far less than its instance count
    # suggests -- and a green run on 1 of 40 instances should not read the same
    # as a green run on 93 of 93.
    total = int(summary.get("instances") or 0)
    comparable = int(summary.get("comparable") or 0)
    scored = comparable - len(waived)
    print(f"\ncoverage: {comparable} of {total} instance(s) certified on both "
          f"sides; the per-model and aggregate rules apply to those only")
    for n in optima_notes:
        print(f"  {n}")
    if total and comparable * 2 < total:
        print(f"warning: fewer than half the instances are comparable. On a "
              f"time-limited suite this gate mostly checks that nothing "
              f"stopped proving; it does not police node counts or the "
              f"primal gap of the instances that hit the limit.")
    if scored < args.aggregate_min_instances:
        print(f"note: {scored} scored instance(s) after waivers -- below "
              f"--aggregate-min-instances {args.aggregate_min_instances}, so "
              f"the G2/G5 bars are not enforced on this run")

    if args.determinism:
        print("\n--- determinism: second sweep ---", flush=True)
        second = candidate.with_name(candidate.stem + "-repeat.jsonl")
        run_sweep(args, suite, second)
        try:
            repeat_run = runs.load_run(second)
            failures += [f"determinism: {f}" for f in check_determinism(
                cand_run, repeat_run, cand_solver, work_name)]
        except runs.RunFileError as e:
            failures.append(f"determinism: {e}")

    if args.tests:
        failures += run_ctest(args.build, "tests")
    if args.sanitize:
        failures += run_ctest(args.asan_build, "sanitizers")

    runs.report(rows, summary, base_run, cand_run, base_solver, cand_solver,
                args.top)

    print("\n" + "=" * 70)
    if waived:
        print(f"WAIVED ({len(waived)}), by {allow_path().name}:")
        for w in waived:
            print(f"  {w}")
    if failures:
        print(f"GATE FAIL -- {len(failures)} finding(s):")
        for f in failures:
            print(f"  {f}")
    else:
        print("GATE PASS")
    print("=" * 70)

    markdown = args.markdown or candidate.with_suffix(".md")
    write_markdown(markdown, rows, summary, base_run, cand_run,
                   base_solver, cand_solver, failures, waived, args.top)

    if args.accept:
        if failures:
            print("\ngate: refusing to accept a candidate that fails "
                  "correctness or its own bars; fix it, or widen the bars "
                  "explicitly and say so in --reason", file=sys.stderr)
            fatal = [f for f in failures
                     if any(k in f for k in FATAL_FINDINGS)]
            if fatal:
                return 1
        accept(candidate, baseline, args.reason, args.suite, args.tag,
               measured_g2)
        return 0

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
