#!/usr/bin/env python3
"""Regression tests for the merge gate's rule table.

gate.py decides what may land, so the one thing that must not be taken on
trust is the gate itself. The contract under test:

  * correctness findings (a lost proof, a status change, disagreeing certified
    objectives, a model that stopped being run) fail the gate and CANNOT be
    waived by the allow-list, however generous the entry;
  * the per-model work bar is shifted, so a 12 -> 15 pivot model is not a 25%
    regression;
  * a waiver covers exactly what it says it covers, and takes its model out of
    the aggregates too -- otherwise the accepted regression comes straight back
    as a G5 failure and the waiver buys nothing;
  * the aggregate bars stand down on a handful of instances, where a geomean is
    one model with a louder voice rather than an aggregate;
  * determinism compares objective BITS, not objectives.

No solver is built or run: every case is a synthetic sweep in a temp dir.
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sor_gate", ROOT / "scripts" / "gate.py")
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)

runs = gate.runs


class BaselineAcceptanceSafetyTests(unittest.TestCase):
    def test_force_accept_bypass_does_not_exist(self):
        self.assertNotIn("force", inspect.signature(gate.accept).parameters)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                gate.main(["--force-accept"])


def record(instance: str, *, status: str = "optimal",
           proof: str | None = "ProvedOptimalFP", objective: float = 1.0,
           seconds: float = 1.0, iterations: int = 1000) -> dict:
    return {"record": "run", "solver": "sor:simplex", "instance": instance,
            "status": status, "proof": proof, "objective": objective,
            "seconds": seconds, "wall_s": seconds + 0.01,
            "iterations": iterations}


def steady(n: int, **kw) -> list[dict]:
    """n unchanging models, so a case about ONE model is not also a case about
    the aggregate."""
    return [record(f"steady{i}", **kw) for i in range(n)]


class GateCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, path: Path, records: list[dict]) -> None:
        lines = [json.dumps({"record": "environment", "commit": "0" * 40,
                             "solvers": ["sor:simplex"]})]
        lines += [json.dumps(r) for r in records]
        path.write_text("\n".join(lines) + "\n")

    def judge(self, base: list[dict], cand: list[dict],
              allow: dict | None = None, work_bar: float = 1.10,
              aggregate_bar: float = 1.02,
              aggregate_min: int = 10) -> tuple[list[str], list[str]]:
        b, c = self.tmp / "b.jsonl", self.tmp / "c.jsonl"
        self.write(b, base)
        self.write(c, cand)
        rows = runs.compare_rows(
            runs.load_run(b).results["sor:simplex"],
            runs.load_run(c).results["sor:simplex"],
            1.0, 100.0, 1e-9, 1e-6)
        return gate.evaluate(rows, allow or {}, "pivots", work_bar,
                             aggregate_bar, aggregate_min)


class TestCorrectnessRules(GateCase):
    def test_identical_sweeps_pass(self):
        rs = [record("a"), record("b", iterations=2000, seconds=2.0)]
        failures, waived = self.judge(rs, rs)
        self.assertEqual(failures, [])
        self.assertEqual(waived, [])

    def test_lost_proof_fails(self):
        failures, _ = self.judge(
            [record("a")], [record("a", status="interrupted", proof=None)])
        self.assertTrue(any("proof-regression" in f for f in failures),
                        failures)

    def test_no_allow_list_entry_can_waive_a_lost_proof(self):
        # The entire reason rule 1 is evaluated separately from rule 2.
        generous = {"a": {"reason": "we really want this merged",
                          "max_work_ratio": 1000.0}}
        failures, _ = self.judge(
            [record("a")], [record("a", status="interrupted", proof=None)],
            allow=generous)
        self.assertTrue(any("proof-regression" in f for f in failures),
                        failures)

    def test_status_change_fails(self):
        failures, _ = self.judge(
            [record("a")], [record("a", status="infeasible", proof=None)])
        self.assertTrue(
            any("status-mismatch" in f or "proof-regression" in f
                for f in failures), failures)

    def test_disagreeing_certified_objectives_fail(self):
        failures, _ = self.judge([record("a", objective=1.0)],
                                 [record("a", objective=1.5)])
        self.assertTrue(any("objective-mismatch" in f for f in failures),
                        failures)

    def test_a_model_that_stopped_being_run_fails(self):
        failures, _ = self.judge([record("a"), record("b")], [record("a")])
        self.assertTrue(any("missing-in-candidate" in f for f in failures),
                        failures)

    def test_adding_a_model_is_not_a_regression(self):
        failures, _ = self.judge([record("a")], [record("a"), record("b")])
        self.assertEqual(failures, [])

    def test_getting_faster_never_fails(self):
        base = [record(f"m{i}", seconds=2.0, iterations=5000) for i in range(12)]
        cand = [record(f"m{i}", seconds=0.5, iterations=1000) for i in range(12)]
        failures, _ = self.judge(base, cand)
        self.assertEqual(failures, [])


class TestWorkBar(GateCase):
    def test_a_tiny_model_does_not_trip_the_bar(self):
        # 12 -> 15 pivots is +25% and completely meaningless.
        base = steady(12) + [record("tiny", iterations=12)]
        cand = steady(12) + [record("tiny", iterations=15)]
        failures, _ = self.judge(base, cand)
        self.assertEqual(failures, [])

    def test_a_real_regression_trips_the_bar(self):
        base = steady(12) + [record("big", iterations=1000)]
        cand = steady(12) + [record("big", iterations=1400)]
        failures, _ = self.judge(base, cand)
        self.assertTrue(any("pivots 1000 -> 1400" in f for f in failures),
                        failures)


class TestAllowList(GateCase):
    def setUp(self):
        super().setUp()
        self.base = steady(12) + [record("m", iterations=1000, seconds=1.0)]
        self.cand = steady(12) + [record("m", iterations=1400, seconds=1.4)]

    def test_an_in_bound_waiver_passes_and_is_reported(self):
        entry = {"m": {"reason": "measured, WS3 owns it", "max_work_ratio": 1.5}}
        failures, waived = self.judge(self.base, self.cand, allow=entry)
        self.assertEqual(failures, [])
        self.assertEqual(len(waived), 1)
        self.assertIn("WS3 owns it", waived[0])

    def test_a_waiver_does_not_cover_more_than_it_says(self):
        entry = {"m": {"reason": "stale bound", "max_work_ratio": 1.05}}
        failures, _ = self.judge(self.base, self.cand, allow=entry)
        self.assertTrue(
            any("over its own allow-list bound" in f for f in failures),
            failures)

    def test_a_waiver_with_no_bound_waives_outright(self):
        failures, waived = self.judge(
            self.base, self.cand, allow={"m": {"reason": "known, tracked"}})
        self.assertEqual(failures, [])
        self.assertEqual(len(waived), 1)

    def test_a_waived_model_leaves_the_aggregates(self):
        # Otherwise the waiver buys nothing: the accepted regression returns as
        # a G5 failure and the only way to merge is to widen the aggregate bar
        # for every other model too.
        base = steady(12) + [record("m", iterations=1000, seconds=1.0)]
        cand = steady(12) + [record("m", iterations=9000, seconds=9.0)]
        failures, _ = self.judge(base, cand)
        self.assertTrue(failures, "unwaived, the outlier must fail something")

        entry = {"m": {"reason": "WS3 owns this one, tracked in the plan"}}
        failures, waived = self.judge(base, cand, allow=entry)
        self.assertEqual(failures, [])
        self.assertEqual(len(waived), 1)

    def test_an_entry_without_a_reason_is_rejected(self):
        path = self.tmp / "gate-allow.json"
        original = gate.allow_path
        gate.allow_path = lambda: path
        self.addCleanup(lambda: setattr(gate, "allow_path", original))

        path.write_text(json.dumps({"_schema": {"version": 1},
                                    "netlib": {"m": {"max_work_ratio": 2.0}},
                                    "miplib-small": {}}))
        with self.assertRaises(SystemExit):
            gate.load_allow("netlib", None)

        path.write_text(json.dumps({"_schema": {"version": 1},
                                    "netlib": {"m": {"reason": "ok"}},
                                    "miplib-small": {}}))
        self.assertEqual(gate.load_allow("netlib", None)["m"]["reason"], "ok")
        self.assertEqual(gate.load_allow("miplib-small", None), {})

    def test_the_committed_allow_file_parses(self):
        # It ships with the repo; a typo in it would fail every gate run.
        for suite, tag in (("netlib", None), ("netlib", "dual"),
                           ("miplib-small", None)):
            self.assertIsInstance(gate.load_allow(suite, tag), dict)


class TestAggregateBars(GateCase):
    def test_an_across_the_board_slowdown_fails_g2(self):
        base = [record(f"m{i}", seconds=1.0) for i in range(12)]
        cand = [record(f"m{i}", seconds=1.11) for i in range(12)]
        failures, _ = self.judge(base, cand)
        self.assertTrue(any("G2 geometric time ratio" in f for f in failures),
                        failures)

    def test_a_pivot_rise_under_the_per_model_bar_still_fails_g5(self):
        base = [record(f"m{i}", iterations=10000) for i in range(12)]
        cand = [record(f"m{i}", iterations=10800) for i in range(12)]
        failures, _ = self.judge(base, cand)
        self.assertFalse(any("pivots 10000 -> 10800" in f for f in failures),
                         "each model is individually under the per-model bar")
        self.assertTrue(any("G5 geometric pivots ratio" in f for f in failures),
                        failures)

    def test_the_bars_stand_down_on_a_handful_of_instances(self):
        base = [record(f"m{i}", seconds=1.0) for i in range(3)]
        cand = [record(f"m{i}", seconds=1.5) for i in range(3)]
        self.assertEqual(self.judge(base, cand)[0], [])
        self.assertTrue(
            any("G2" in f for f in self.judge(base, cand, aggregate_min=1)[0]))


class TestReferenceRule(GateCase):
    """Rule 4. Rules 2 and 3 compare SOR against SOR, so a change that costs
    time without moving a pivot is invisible to them. This is the rule that
    sees it, and it needs an external fixed point to see it against."""

    def refs(self, **kw):
        return {k: gate.runs.compare.Result(solver="highs", instance=k,
                                            status="optimal", seconds=v)
                for k, v in kw.items()}

    def judge_ref(self, base, cand, reference, allow=None, g2_bar=1.05,
                  time_bar=1.15, work_slack=0.05, min_seconds=0.05,
                  recorded_g2=None):
        b, c = self.tmp / "b.jsonl", self.tmp / "c.jsonl"
        self.write(b, base)
        self.write(c, cand)
        rows = runs.compare_rows(
            runs.load_run(b).results["sor:simplex"],
            runs.load_run(c).results["sor:simplex"],
            1.0, 100.0, 1e-9, 1e-6)
        return gate.evaluate_reference(rows, reference, allow or {}, g2_bar,
                                       time_bar, work_slack, min_seconds,
                                       recorded_g2)

    def test_g2_regression_against_the_recorded_ratio_fails(self):
        base = [record(f"m{i}", seconds=1.0) for i in range(6)]
        cand = [record(f"m{i}", seconds=1.5) for i in range(6)]
        ref = self.refs(**{f"m{i}": 1.0 for i in range(6)})
        # Recorded at 1.0x; now 1.5x, far over the 5% regression bar.
        failures, _ = self.judge_ref(base, cand, ref, recorded_g2=1.0)
        self.assertTrue(any("G2 against the HiGHS reference" in f
                            for f in failures), failures)

    def test_being_slower_than_highs_is_not_itself_a_failure(self):
        # SOR is allowed to be slower than HiGHS; it is not allowed to get
        # slower than it was. Same 1.5x, but that is what was recorded.
        base = [record(f"m{i}", seconds=1.5) for i in range(6)]
        cand = [record(f"m{i}", seconds=1.5) for i in range(6)]
        ref = self.refs(**{f"m{i}": 1.0 for i in range(6)})
        failures, _ = self.judge_ref(base, cand, ref, recorded_g2=1.5)
        self.assertEqual(failures, [])

    def test_no_recorded_ratio_makes_only_the_aggregate_informational(self):
        # A baseline accepted before rule 4 existed has no G2 stamp. The
        # AGGREGATE half then has nothing to compare against and says so; the
        # per-model half does not depend on it and still applies, so the models
        # here sit under the reference-time floor to isolate the aggregate.
        base = [record(f"m{i}", seconds=0.002) for i in range(6)]
        cand = [record(f"m{i}", seconds=0.018) for i in range(6)]
        ref = self.refs(**{f"m{i}": 0.001 for i in range(6)})
        failures, notes = self.judge_ref(base, cand, ref, recorded_g2=None)
        self.assertEqual(failures, [])
        self.assertTrue(any("no G2 recorded" in n for n in notes), notes)

        # ...and with a stamp, the same sweep fails on the aggregate.
        failures, _ = self.judge_ref(base, cand, ref, recorded_g2=2.0)
        self.assertTrue(any("G2 against the HiGHS reference" in f
                            for f in failures), failures)

    def test_per_pivot_cost_regression_is_caught(self):
        # Time up 40%, pivots unchanged: rule 2 sees nothing, rule 4 must.
        base = [record("m", seconds=1.0, iterations=1000)]
        cand = [record("m", seconds=1.4, iterations=1000)]
        failures, _ = self.judge_ref(base, cand, self.refs(m=0.5),
                                     recorded_g2=10.0)
        self.assertTrue(any("per-pivot cost regression" in f
                            for f in failures), failures)

    def test_a_model_whose_work_moved_is_left_to_rule_2(self):
        base = [record("m", seconds=1.0, iterations=1000)]
        cand = [record("m", seconds=1.4, iterations=1400)]
        failures, _ = self.judge_ref(base, cand, self.refs(m=0.5),
                                     recorded_g2=10.0)
        self.assertEqual(failures, [])

    def test_models_the_reference_solves_quickly_are_exempt(self):
        # Below the floor a 40% time ratio is timer noise, not a finding.
        base = [record("m", seconds=0.002, iterations=1000)]
        cand = [record("m", seconds=0.003, iterations=1000)]
        failures, _ = self.judge_ref(base, cand, self.refs(m=0.001),
                                     recorded_g2=10.0)
        self.assertEqual(failures, [])

    def test_a_waiver_must_carry_a_bucket_profile(self):
        base = [record("m", seconds=1.0, iterations=1000)]
        cand = [record("m", seconds=1.4, iterations=1000)]
        ref = self.refs(m=0.5)

        vague = {"m": {"reason": "known, we will look at it"}}
        failures, _ = self.judge_ref(base, cand, ref, allow=vague,
                                     recorded_g2=10.0)
        self.assertTrue(any("no bucket profile" in f for f in failures),
                        failures)

        profiled = {"m": {"reason": "WS4 owns it: BTRAN 250 ms of 400 ms, "
                                    "factorization 90 ms"}}
        failures, notes = self.judge_ref(base, cand, ref, allow=profiled,
                                         recorded_g2=10.0)
        self.assertEqual(failures, [])
        self.assertTrue(any("WS4 owns it" in n for n in notes), notes)

    def test_the_committed_reference_loads(self):
        path = gate.default_reference("netlib")
        self.assertIsNotNone(path, "netlib HiGHS reference should be committed")
        ref = gate.load_reference(path, "highs")
        self.assertGreaterEqual(len(ref), 90)
        self.assertTrue(all(r.seconds is not None for r in ref.values()))
        self.assertTrue(all(r.objective is not None for r in ref.values()),
                        "rule 1's reference half needs objectives in the sweep")


class TestKnownOptima(GateCase):
    """Rule 1 against the published optimum. This is the only rule that says
    anything about the miplib-small instances that never finish -- 39 of 40 at
    a 30 s limit -- because it needs no proof, only an answer."""

    MPS = ("NAME          T\nROWS\n N  COST\n L  R1\nCOLUMNS\n"
           "    MARK0000  'MARKER'                 'INTORG'\n"
           "    X1        COST      -5             R1        4\n"
           "    MARK0001  'MARKER'                 'INTEND'\n"
           "RHS\n    RHS       R1        5\n"
           "BOUNDS\n UI BND       X1        1\nENDATA\n")

    def models_dir(self, optimum: str = "-5.0") -> Path:
        d = self.tmp / "suite" / "mps"
        d.mkdir(parents=True, exist_ok=True)
        (d / "t.mps").write_text(self.MPS)
        (d.parent / "reference.csv").write_text(
            f"# name,optimum\nt,{optimum}\n")
        return d

    def judge_opt(self, cand: list[dict], models_dir: Path, tol: float = 1e-6):
        b, c = self.tmp / "b.jsonl", self.tmp / "c.jsonl"
        self.write(b, cand)
        self.write(c, cand)
        rows = runs.compare_rows(
            runs.load_run(b).results["sor:simplex"],
            runs.load_run(c).results["sor:simplex"], 1.0, 100.0, 1e-9, tol)
        return gate.evaluate_known_optima(rows, models_dir, tol)

    def test_a_certified_wrong_optimum_fails(self):
        # pg.mps: ProvedGlobalEpsilon at 7250 against a published -8674.34,
        # because the time limit expired inside the root node's LP.
        failures, _ = self.judge_opt(
            [record("t.mps", objective=7250.0)], self.models_dir("-8674.342607"))
        self.assertTrue(any("claimed Optimal" in f for f in failures), failures)
        self.assertTrue(any("no allow-list entry can waive" in f
                            for f in failures), failures)

    def test_an_incumbent_better_than_the_optimum_fails(self):
        # Needs no proof at all -- this is the half that covers the instances
        # that stop at the time limit.
        failures, _ = self.judge_opt(
            [record("t.mps", status="feasible", proof="FeasibleWithGap",
                    objective=-9.0)], self.models_dir("-5.0"))
        self.assertTrue(any("beats published optimum" in f for f in failures),
                        failures)

    def test_a_dual_bound_past_the_optimum_fails(self):
        rec = record("t.mps", status="feasible", proof="FeasibleWithGap",
                     objective=-4.0)
        rec["dual_bound"] = -3.0   # minimizing: a lower bound above the optimum
        failures, _ = self.judge_opt([rec], self.models_dir("-5.0"))
        self.assertTrue(any("crossed past optimum" in f for f in failures),
                        failures)

    def test_a_sound_time_limited_run_passes(self):
        rec = record("t.mps", status="feasible", proof="FeasibleWithGap",
                     objective=-4.0)
        rec["dual_bound"] = -7.0
        failures, notes = self.judge_opt([rec], self.models_dir("-5.0"))
        self.assertEqual(failures, [])
        self.assertTrue(any("1 model(s), 0 unsound" in n for n in notes), notes)

    def test_a_suite_without_reference_csv_is_not_judged(self):
        d = self.tmp / "bare" / "mps"
        d.mkdir(parents=True)
        (d / "t.mps").write_text(self.MPS)
        failures, notes = self.judge_opt([record("t.mps", objective=7250.0)], d)
        self.assertEqual(failures, [])
        self.assertTrue(any("no reference.csv" in n for n in notes), notes)

    def test_the_committed_milp_references_load(self):
        for suite in ("miplib-small", "miplib-easy"):
            path = ROOT / "benchmarks" / suite / "reference.csv"
            if not path.exists():
                continue
            ref = gate.miplib.read_reference(path)
            self.assertGreater(len(ref), 0, suite)


class TestReferenceObjectives(GateCase):
    """Rule 1, second half. Everything else here compares SOR to SOR, so an
    answer that is wrong in BOTH sweeps reads as "no differences"."""

    def refs(self, **kw):
        return {k: gate.runs.compare.Result(
            solver="highs", instance=k, status="optimal", objective=v,
            seconds=1.0) for k, v in kw.items()}

    def judge_obj(self, cand, reference, abs_tol=1e-9, rel_tol=1e-6):
        b, c = self.tmp / "b.jsonl", self.tmp / "c.jsonl"
        self.write(b, cand)
        self.write(c, cand)
        rows = runs.compare_rows(
            runs.load_run(b).results["sor:simplex"],
            runs.load_run(c).results["sor:simplex"],
            1.0, 100.0, abs_tol, rel_tol)
        return gate.evaluate_reference_objectives(rows, reference,
                                                  abs_tol, rel_tol)

    def test_an_answer_wrong_in_both_sweeps_is_caught(self):
        # pilot.mps, forced dual, --tol 1e-6: certified ProvedOptimalFP at
        # -557.48163926 against HiGHS's -557.4897292744. Identical in the
        # baseline and the candidate, so every other rule called it clean.
        cand = [record("pilot.mps", objective=-557.48163926)]
        failures, _ = self.judge_obj(
            cand, self.refs(**{"pilot.mps": -557.4897292744025}))
        self.assertTrue(any("certified wrong answer" in f for f in failures),
                        failures)
        self.assertTrue(any("1.45e-05 relative" in f for f in failures),
                        failures)

    def test_agreement_within_tolerance_passes(self):
        # The same model at --tol 1e-7, which is what HiGHS itself uses.
        cand = [record("pilot.mps", objective=-557.48972928)]
        failures, notes = self.judge_obj(
            cand, self.refs(**{"pilot.mps": -557.4897292744025}))
        self.assertEqual(failures, [])
        self.assertTrue(any("1 model(s), 0 disagreement" in n for n in notes),
                        notes)

    def test_an_uncertified_candidate_is_not_judged_on_its_objective(self):
        # A run that hit the time limit has no claim to defend.
        cand = [record("m", status="interrupted", proof=None, objective=0.0)]
        failures, _ = self.judge_obj(cand, self.refs(m=-557.0))
        self.assertEqual(failures, [])

    def test_a_reference_that_did_not_solve_is_not_an_oracle(self):
        cand = [record("m", objective=1.0)]
        ref = {"m": gate.runs.compare.Result(
            solver="highs", instance="m", status="TimeLimit",
            objective=999.0, seconds=1.0)}
        failures, _ = self.judge_obj(cand, ref)
        self.assertEqual(failures, [])

    def test_no_allow_list_entry_can_waive_a_wrong_answer(self):
        # evaluate_reference_objectives takes no allow-list at all: the
        # signature is the guarantee, so this test is about the signature.
        import inspect
        params = inspect.signature(
            gate.evaluate_reference_objectives).parameters
        self.assertNotIn("allow", params)


class TestDeterminism(GateCase):
    def load_pair(self, left: list[dict], right: list[dict]):
        a, b = self.tmp / "d1.jsonl", self.tmp / "d2.jsonl"
        self.write(a, left)
        self.write(b, right)
        return runs.load_run(a), runs.load_run(b)

    def test_identical_sweeps_are_deterministic(self):
        a, b = self.load_pair([record("m")], [record("m")])
        self.assertEqual(
            gate.check_determinism(a, b, "sor:simplex", "pivots"), [])

    def test_one_ulp_of_objective_drift_is_caught(self):
        a, b = self.load_pair([record("m", objective=1.0)],
                              [record("m", objective=1.0 + 2.0 ** -52)])
        out = gate.check_determinism(a, b, "sor:simplex", "pivots")
        self.assertTrue(any("objective bits" in f for f in out), out)

    def test_a_pivot_difference_is_caught(self):
        a, b = self.load_pair([record("m", iterations=100)],
                              [record("m", iterations=101)])
        out = gate.check_determinism(a, b, "sor:simplex", "pivots")
        self.assertTrue(any("pivots 100 vs 101" in f for f in out), out)


if __name__ == "__main__":
    unittest.main()
