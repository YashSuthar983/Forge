#!/usr/bin/env python3
"""Regression tests for the JSONL run-comparison tool.

The contract under test is that a timing difference is only ever reported
between two results that are BOTH certified, and that everything else -- a
missing model, a timeout, a result that lost its proof, a zero time or a zero
iteration count -- is reported as such instead of turning into a ratio.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sor_compare_runs", ROOT / "scripts" / "compare_runs.py")
assert SPEC is not None and SPEC.loader is not None
cr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cr
SPEC.loader.exec_module(cr)

compare = cr.compare


def run_record(instance: str, solver: str = "sor:simplex", *,
               status: str = "Optimal", proof: str | None = "ProvedOptimalFP",
               objective: float | None = 1.0, seconds: float | None = 1.0,
               wall_s: float | None = None, iterations: int | None = 100,
               repetition: int | None = 0, **extra: object) -> dict:
    rec = {"record": "run", "solver": solver, "instance": instance,
           "status": status, "proof": proof, "objective": objective,
           "seconds": seconds,
           "wall_s": seconds if wall_s is None else wall_s,
           "iterations": iterations, "repetition": repetition}
    rec.update(extra)
    return rec


def env_record(**extra: object) -> dict:
    rec = {"record": "environment", "commit": "abcdef1234", "dirty": False,
           "solvers": ["sor:simplex"], "models": 1}
    rec.update(extra)
    return rec


class JsonlFile:
    """A temporary .jsonl file built from records."""

    def __init__(self, records: list[dict], name: str = "run.jsonl") -> None:
        self.records, self.name = records, name

    def __enter__(self) -> Path:
        self._dir = tempfile.TemporaryDirectory()
        p = Path(self._dir.name) / self.name
        p.write_text("".join(json.dumps(r) + "\n" for r in self.records))
        return p

    def __exit__(self, *exc: object) -> None:
        self._dir.cleanup()


def compare_records(base: list[dict], cand: list[dict], **kw: float
                    ) -> tuple[list[cr.Row], dict[str, object]]:
    """Load two record lists and compare their single solver."""
    with JsonlFile([env_record()] + base, "base.jsonl") as bp:
        with JsonlFile([env_record()] + cand, "cand.jsonl") as cp:
            b, c = cr.load_run(bp), cr.load_run(cp)
            rows = cr.compare_rows(
                b.results[cr.pick_solver(b, None, "baseline")],
                c.results[cr.pick_solver(c, None, "candidate")],
                kw.get("time_shift", 0.01), kw.get("iteration_shift", 1.0),
                kw.get("obj_abs_tol", 1e-7), kw.get("obj_rel_tol", 1e-4))
            return rows, cr.summarize(rows, kw.get("sgm_shift", 1.0))


def findings(rows: list[cr.Row]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in rows:
        for f in r.findings:
            out.setdefault(f, []).append(r.instance)
    return out


class LoadTests(unittest.TestCase):
    def test_environment_and_runs(self) -> None:
        with JsonlFile([env_record(commit="deadbeefcafe"),
                        run_record("a.mps"), run_record("b.mps")]) as p:
            run = cr.load_run(p)
        self.assertEqual(run.environment["commit"], "deadbeefcafe")
        self.assertEqual(run.solvers, ["sor:simplex"])
        self.assertEqual(sorted(run.results["sor:simplex"]), ["a.mps", "b.mps"])
        self.assertIn("deadbeef", run.label())

    def test_blank_lines_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text(json.dumps(env_record()) + "\n\n"
                         + json.dumps(run_record("a.mps")) + "\n\n")
            self.assertEqual(len(cr.load_run(p).results["sor:simplex"]), 1)

    def test_repetitions_are_aggregated_to_a_median(self) -> None:
        reps = [run_record("a.mps", seconds=s, repetition=i)
                for i, s in enumerate((1.0, 5.0, 9.0))]
        with JsonlFile([env_record()] + reps) as p:
            run = cr.load_run(p)
        r = run.results["sor:simplex"]["a.mps"]
        self.assertEqual(r.seconds, 5.0)
        self.assertEqual(sorted(r.samples_s or []), [1.0, 5.0, 9.0])

    def test_a_failed_repetition_wins_over_the_median(self) -> None:
        # compare.py's aggregation deliberately surfaces flakiness rather than
        # averaging it away; this tool must inherit exactly that.
        reps = [run_record("a.mps", seconds=1.0, repetition=0),
                run_record("a.mps", status="timeout", proof=None,
                           objective=None, seconds=30.0, repetition=1)]
        with JsonlFile([env_record()] + reps) as p:
            run = cr.load_run(p)
        self.assertEqual(run.results["sor:simplex"]["a.mps"].status, "timeout")

    def test_several_solvers_in_one_file(self) -> None:
        with JsonlFile([env_record(), run_record("a.mps"),
                        run_record("a.mps", solver="highs", proof=None)]) as p:
            run = cr.load_run(p)
        self.assertEqual(run.solvers, ["highs", "sor:simplex"])

    def test_appended_sweeps_default_to_the_last(self) -> None:
        records = [env_record(commit="1111111111"), run_record("a.mps", seconds=9.0),
                   env_record(commit="2222222222"), run_record("a.mps", seconds=2.0)]
        with JsonlFile(records) as p:
            last = cr.load_run(p)
            first = cr.load_run(p, segment=0)
        self.assertEqual(last.segments, 2)
        self.assertEqual(last.environment["commit"], "2222222222")
        self.assertEqual(last.results["sor:simplex"]["a.mps"].seconds, 2.0)
        self.assertEqual(first.results["sor:simplex"]["a.mps"].seconds, 9.0)

    def test_claim_metadata_round_trips_without_empty_final_segment(self) -> None:
        records = [
            {"record": "preflight", "claim_role": "preflight",
             "status": "PASS"},
            env_record(commit="feedface"),
            run_record("a.mps"),
            {"record": "aggregate", **{
                k: v for k, v in run_record("a.mps").items()
                if k != "record"}},
            {"record": "postflight", "claim_role": "postflight",
             "status": "PASS"},
            {"record": "independent_check", "status": "PASS"},
            {"record": "baseline_eligibility", "status": "PASS",
             "eligible": True},
            {"record": "performance", "status": "FAIL", "public": False},
            {"record": "gate", "claim_role": "gate", "status": "PASS"},
        ]
        with JsonlFile(records) as p:
            run = cr.load_run(p)
        self.assertEqual(run.segments, 1)
        self.assertEqual(run.environment["commit"], "feedface")
        self.assertEqual(run.results["sor:simplex"]["a.mps"].seconds, 1.0)
        self.assertEqual([m["record"] for m in run.metadata],
                         ["preflight", "postflight", "independent_check",
                          "baseline_eligibility", "performance", "gate"])

    def test_rerun_selected_aggregate_replaces_superseded_original(self) -> None:
        initial = run_record("a.mps", seconds=9.0)
        initial["record"] = "aggregate"
        initial.update(claim_superseded=True, claim_selected=False,
                       claim_source="initial")
        replacement = run_record("a.mps", seconds=2.0)
        replacement["record"] = "aggregate"
        replacement.update(claim_superseded=False, claim_selected=True,
                           claim_source="rerun")
        with JsonlFile([env_record(), initial,
                        {"record": "rerun", "status": "PASS"},
                        replacement]) as p:
            run = cr.load_run(p)
        selected = run.results["sor:simplex"]["a.mps"]
        self.assertEqual(selected.seconds, 2.0)
        self.assertEqual(selected.claim_source, "rerun")

    def test_file_without_an_environment_record_is_one_sweep(self) -> None:
        with JsonlFile([run_record("a.mps")]) as p:
            run = cr.load_run(p)
        self.assertEqual(run.segments, 1)
        self.assertEqual(run.environment, {})

    def test_segment_out_of_range(self) -> None:
        with JsonlFile([env_record(), run_record("a.mps")]) as p:
            with self.assertRaises(cr.RunFileError) as ctx:
                cr.load_run(p, segment=5)
        self.assertIn("out of range", str(ctx.exception))

    def test_malformed_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text(json.dumps(env_record()) + "\n{not json\n")
            with self.assertRaises(cr.RunFileError) as ctx:
                cr.load_run(p)
        self.assertIn("line 2", str(ctx.exception))

    def test_unknown_field_is_refused(self) -> None:
        # Silently dropping a field a newer compare.py started writing would
        # make this tool quietly compare the wrong thing.
        with JsonlFile([env_record(),
                        run_record("a.mps", nodes=17)]) as p:
            with self.assertRaises(cr.RunFileError) as ctx:
                cr.load_run(p)
        self.assertIn("nodes", str(ctx.exception))

    def test_record_without_instance(self) -> None:
        rec = run_record("a.mps")
        rec["instance"] = ""
        with JsonlFile([env_record(), rec]) as p:
            with self.assertRaises(cr.RunFileError):
                cr.load_run(p)

    def test_sweep_with_no_runs(self) -> None:
        with JsonlFile([env_record()]) as p:
            with self.assertRaises(cr.RunFileError) as ctx:
                cr.load_run(p)
        self.assertIn("no run records", str(ctx.exception))

    def test_empty_file(self) -> None:
        with JsonlFile([]) as p:
            with self.assertRaises(cr.RunFileError) as ctx:
                cr.load_run(p)
        self.assertIn("no records", str(ctx.exception))

    def test_missing_file(self) -> None:
        with self.assertRaises(cr.RunFileError):
            cr.load_run(Path("/no/such/run.jsonl"))


class SolverSelectionTests(unittest.TestCase):
    def _run(self, solvers: list[str]) -> cr.Run:
        recs = [run_record("a.mps", solver=s, proof=None if s == "highs"
                           else "ProvedOptimalFP") for s in solvers]
        with JsonlFile([env_record()] + recs) as p:
            return cr.load_run(p)

    def test_single_solver_is_chosen_automatically(self) -> None:
        self.assertEqual(
            cr.pick_solver(self._run(["sor:simplex"]), None, "baseline"),
            "sor:simplex")

    def test_ambiguous_file_requires_an_explicit_choice(self) -> None:
        run = self._run(["sor:simplex", "highs"])
        with self.assertRaises(cr.RunFileError) as ctx:
            cr.pick_solver(run, None, "baseline")
        self.assertIn("--solver", str(ctx.exception))
        self.assertEqual(cr.pick_solver(run, "highs", "baseline"), "highs")

    def test_solver_name_is_case_insensitive(self) -> None:
        self.assertEqual(
            cr.pick_solver(self._run(["sor:simplex"]), "SOR:SIMPLEX", "baseline"),
            "sor:simplex")

    def test_unknown_solver_names_what_is_available(self) -> None:
        with self.assertRaises(cr.RunFileError) as ctx:
            cr.pick_solver(self._run(["sor:simplex"]), "sor:pdhg", "baseline")
        self.assertIn("sor:simplex", str(ctx.exception))


class FindingTests(unittest.TestCase):
    def test_clean_comparison_has_no_findings(self) -> None:
        rows, summary = compare_records([run_record("a.mps")],
                                        [run_record("a.mps")])
        self.assertEqual(summary["findings"], {})
        self.assertTrue(rows[0].comparable)

    def test_missing_models_in_either_direction(self) -> None:
        rows, _ = compare_records(
            [run_record("only_base.mps"), run_record("both.mps")],
            [run_record("only_cand.mps"), run_record("both.mps")])
        f = findings(rows)
        self.assertEqual(f["missing-in-candidate"], ["only_base.mps"])
        self.assertEqual(f["missing-in-baseline"], ["only_cand.mps"])
        # A model present on one side only can never be comparable.
        self.assertEqual([r.instance for r in rows if r.comparable], ["both.mps"])

    def test_unproved_optimal_is_a_proof_regression(self) -> None:
        # Optimal but with a proof level outside the accepted set: the status
        # matches, so only the proof gate catches this.
        rows, summary = compare_records(
            [run_record("a.mps", proof="ProvedOptimalFP")],
            [run_record("a.mps", proof="BoundOnly")])
        self.assertEqual(rows[0].findings, ["proof-regression"])
        self.assertFalse(rows[0].comparable)
        self.assertIsNone(rows[0].time_ratio)
        self.assertEqual(summary["comparable"], 0)

    def test_missing_proof_field_is_a_proof_regression_for_sor(self) -> None:
        rows, _ = compare_records([run_record("a.mps")],
                                  [run_record("a.mps", proof=None)])
        self.assertIn("proof-regression", rows[0].findings)

    def test_external_solver_needs_no_proof(self) -> None:
        # HiGHS never reports a SOR proof level; requiring one would make every
        # baseline comparison uncomparable.
        rows, summary = compare_records(
            [run_record("a.mps", solver="highs", proof=None)],
            [run_record("a.mps", solver="highs", proof=None)])
        self.assertEqual(rows[0].findings, [])
        self.assertEqual(summary["comparable"], 1)

    def test_regaining_a_proof_is_an_improvement_not_a_regression(self) -> None:
        rows, _ = compare_records([run_record("a.mps", proof="BoundOnly")],
                                  [run_record("a.mps", proof="ProvedOptimalFP")])
        self.assertEqual(rows[0].findings, ["proof-improvement"])

    def test_timeout_is_both_a_proof_regression_and_a_status_mismatch(self) -> None:
        rows, summary = compare_records(
            [run_record("a.mps", seconds=1.0)],
            [run_record("a.mps", status="timeout", proof=None, objective=None,
                        seconds=30.0, iterations=None)])
        self.assertEqual(rows[0].findings, ["proof-regression", "status-mismatch"])
        self.assertFalse(rows[0].comparable)
        # The 30s timeout must not be added to any total.
        self.assertEqual(summary["candidate_total_s"], 0.0)

    def test_timeout_on_both_sides_is_not_a_finding(self) -> None:
        timed_out = run_record("a.mps", status="timeout", proof=None,
                               objective=None, seconds=30.0, iterations=None)
        rows, summary = compare_records([timed_out], [dict(timed_out)])
        self.assertEqual(rows[0].findings, [])
        self.assertFalse(rows[0].comparable)   # still not a timing datapoint
        self.assertEqual(summary["comparable"], 0)

    def test_status_case_and_spacing_do_not_count_as_a_mismatch(self) -> None:
        rows, _ = compare_records([run_record("a.mps", status="Optimal")],
                                  [run_record("a.mps", status=" optimal ")])
        self.assertEqual(rows[0].findings, [])

    def test_objective_mismatch(self) -> None:
        rows, _ = compare_records([run_record("a.mps", objective=100.0)],
                                  [run_record("a.mps", objective=101.0)])
        self.assertEqual(rows[0].findings, ["objective-mismatch"])

    def test_objective_agreement_is_relative(self) -> None:
        rows, _ = compare_records([run_record("a.mps", objective=1e6)],
                                  [run_record("a.mps", objective=1e6 + 1e-3)])
        self.assertEqual(rows[0].findings, [])

    def test_objective_is_not_compared_when_one_side_is_uncertified(self) -> None:
        # An interrupted run's objective is an incumbent, not an optimum;
        # flagging it as a mismatch would be noise on top of the real finding.
        rows, _ = compare_records(
            [run_record("a.mps", objective=100.0)],
            [run_record("a.mps", status="Interrupted", proof="BoundOnly",
                        objective=999.0)])
        self.assertNotIn("objective-mismatch", rows[0].findings)
        self.assertIn("proof-regression", rows[0].findings)

    def test_missing_objective_on_both_sides_is_a_mismatch(self) -> None:
        # objectives_agree() is false for None, so two certified results that
        # both failed to report an objective are still reported rather than
        # silently accepted.
        rows, _ = compare_records([run_record("a.mps", objective=None)],
                                  [run_record("a.mps", objective=None)])
        self.assertEqual(rows[0].findings, ["objective-mismatch"])


class RatioTests(unittest.TestCase):
    def test_shifted_ratio_basics(self) -> None:
        self.assertAlmostEqual(cr.shifted_ratio(1.0, 2.0, 0.0), 2.0)
        self.assertAlmostEqual(cr.shifted_ratio(2.0, 1.0, 0.0), 0.5)
        self.assertIsNone(cr.shifted_ratio(None, 1.0, 0.01))
        self.assertIsNone(cr.shifted_ratio(1.0, None, 0.01))
        self.assertIsNone(cr.shifted_ratio(float("inf"), 1.0, 0.01))

    def test_zero_baseline_time_does_not_divide_by_zero(self) -> None:
        rows, _ = compare_records([run_record("a.mps", seconds=0.0)],
                                  [run_record("a.mps", seconds=0.0)])
        self.assertEqual(rows[0].time_ratio, 1.0)

    def test_zero_baseline_with_nonzero_candidate_is_bounded_by_the_shift(self) -> None:
        # Without the shift this is a division by zero; with it, the ratio is
        # (0.01 + 0.01) / 0.01 = 2, which is an honest statement about two
        # timings that are both in the measurement noise.
        rows, _ = compare_records([run_record("a.mps", seconds=0.0)],
                                  [run_record("a.mps", seconds=0.01)])
        self.assertAlmostEqual(rows[0].time_ratio, 2.0)

    def test_zero_shift_and_zero_baseline_yields_no_ratio(self) -> None:
        rows, _ = compare_records([run_record("a.mps", seconds=0.0)],
                                  [run_record("a.mps", seconds=1.0)],
                                  time_shift=0.0)
        self.assertIsNone(rows[0].time_ratio)

    def test_zero_iterations_on_both_sides(self) -> None:
        rows, _ = compare_records([run_record("a.mps", iterations=0)],
                                  [run_record("a.mps", iterations=0)])
        self.assertEqual(rows[0].iter_ratio, 1.0)

    def test_zero_baseline_iterations(self) -> None:
        # (5 + 1) / (0 + 1) = 6, not an error and not infinity.
        rows, _ = compare_records([run_record("a.mps", iterations=0)],
                                  [run_record("a.mps", iterations=5)])
        self.assertAlmostEqual(rows[0].iter_ratio, 6.0)

    def test_absent_iteration_counts_produce_no_iteration_ratio(self) -> None:
        rows, _ = compare_records([run_record("a.mps", iterations=None)],
                                  [run_record("a.mps", iterations=7)])
        self.assertIsNone(rows[0].iter_ratio)
        self.assertIsNotNone(rows[0].time_ratio)

    def test_absent_times_produce_no_time_ratio(self) -> None:
        rows, summary = compare_records([run_record("a.mps", seconds=None)],
                                        [run_record("a.mps", seconds=None)])
        self.assertIsNone(rows[0].time_ratio)
        self.assertTrue(rows[0].comparable)
        self.assertEqual(summary["baseline_total_s"], 0.0)


class SummaryTests(unittest.TestCase):
    def test_totals_only_count_comparable_rows(self) -> None:
        rows, summary = compare_records(
            [run_record("ok.mps", seconds=2.0, iterations=10),
             run_record("lost.mps", seconds=4.0, iterations=20)],
            [run_record("ok.mps", seconds=3.0, iterations=15),
             run_record("lost.mps", status="timeout", proof=None,
                        objective=None, seconds=60.0, iterations=None)])
        self.assertEqual(summary["comparable"], 1)
        self.assertEqual(summary["baseline_total_s"], 2.0)
        self.assertEqual(summary["candidate_total_s"], 3.0)
        self.assertEqual(summary["baseline_total_iterations"], 10)
        self.assertEqual(summary["candidate_total_iterations"], 15)
        self.assertEqual(summary["findings"],
                         {"proof-regression": 1, "status-mismatch": 1})
        self.assertEqual(len(rows), 2)

    def test_geomean_of_ratios(self) -> None:
        rows, summary = compare_records(
            [run_record("a.mps", seconds=1.0), run_record("b.mps", seconds=1.0)],
            [run_record("a.mps", seconds=4.0), run_record("b.mps", seconds=0.25)],
            time_shift=0.0)
        self.assertAlmostEqual(summary["geomean_time_ratio"], 1.0)
        self.assertEqual(sorted(r.time_ratio for r in rows), [0.25, 4.0])

    def test_sgm_ratio(self) -> None:
        _, summary = compare_records([run_record("a.mps", seconds=1.0)],
                                     [run_record("a.mps", seconds=3.0)])
        # shifted geomean with shift 1: (1+1)-1 = 1 vs (3+1)-1 = 3.
        self.assertAlmostEqual(summary["baseline_sgm_s"], 1.0)
        self.assertAlmostEqual(summary["candidate_sgm_s"], 3.0)
        self.assertAlmostEqual(summary["sgm_ratio"], 3.0)

    def test_no_comparable_rows_leaves_aggregates_empty(self) -> None:
        _, summary = compare_records(
            [run_record("a.mps")],
            [run_record("a.mps", status="timeout", proof=None,
                        objective=None, seconds=9.0)])
        self.assertEqual(summary["comparable"], 0)
        self.assertIsNone(summary["baseline_sgm_s"])
        self.assertIsNone(summary["sgm_ratio"])
        self.assertIsNone(summary["geomean_time_ratio"])

    def test_geomean_helper_ignores_nonpositive_values(self) -> None:
        self.assertIsNone(cr.geomean([]))
        self.assertIsNone(cr.geomean([0.0, -1.0]))
        self.assertAlmostEqual(cr.geomean([1.0, 4.0]), 2.0)


class CliTests(unittest.TestCase):
    def _run(self, base: list[dict], cand: list[dict],
             *args: str) -> tuple[int, str, str]:
        with JsonlFile([env_record()] + base, "base.jsonl") as bp:
            with JsonlFile([env_record()] + cand, "cand.jsonl") as cp:
                out, err = io.StringIO(), io.StringIO()
                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = out, err
                try:
                    code = cr.main([str(bp), str(cp), *args])
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
                return code, out.getvalue(), err.getvalue()

    def test_clean_run_exits_zero(self) -> None:
        code, out, _ = self._run([run_record("a.mps", seconds=2.0)],
                                 [run_record("a.mps", seconds=1.0)])
        self.assertEqual(code, 0)
        self.assertIn("no differences", out)
        self.assertIn("TOP IMPROVEMENTS", out)
        self.assertIn("a.mps", out.split("TOP IMPROVEMENTS")[1])

    def test_proof_regression_exits_one(self) -> None:
        code, out, _ = self._run([run_record("a.mps")],
                                 [run_record("a.mps", proof="BoundOnly")])
        self.assertEqual(code, 1)
        self.assertIn("proof-regression: 1", out)

    def test_missing_model_exits_one_unless_allowed(self) -> None:
        base, cand = [run_record("a.mps"), run_record("b.mps")], [run_record("a.mps")]
        code, out, _ = self._run(base, cand)
        self.assertEqual(code, 1)
        self.assertIn("missing-in-candidate", out)
        self.assertEqual(self._run(base, cand, "--allow-missing")[0], 0)

    def test_proof_improvement_alone_does_not_fail(self) -> None:
        code, out, _ = self._run([run_record("a.mps", proof="BoundOnly")],
                                 [run_record("a.mps")])
        self.assertEqual(code, 0)
        self.assertIn("proof-improvement: 1", out)

    def test_report_survives_having_nothing_comparable(self) -> None:
        code, out, _ = self._run(
            [run_record("a.mps")],
            [run_record("a.mps", status="timeout", proof=None,
                        objective=None, seconds=9.0)])
        self.assertEqual(code, 1)
        self.assertIn("nothing to compare", out)

    def test_ambiguous_solver_is_a_usage_error(self) -> None:
        code, _, err = self._run(
            [run_record("a.mps"), run_record("a.mps", solver="highs", proof=None)],
            [run_record("a.mps")])
        self.assertEqual(code, 2)
        self.assertIn("--solver", err)

    def test_solver_flag_selects_one_side_each(self) -> None:
        code, out, _ = self._run(
            [run_record("a.mps"), run_record("a.mps", solver="highs", proof=None)],
            [run_record("a.mps")], "--baseline-solver", "sor:simplex")
        self.assertEqual(code, 0)
        self.assertIn("solver=sor:simplex", out)

    def test_unreadable_file_exits_two(self) -> None:
        with JsonlFile([env_record(), run_record("a.mps")]) as p:
            out, err = io.StringIO(), io.StringIO()
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = out, err
            try:
                code = cr.main([str(p), "/no/such/file.jsonl"])
            finally:
                sys.stdout, sys.stderr = old_out, old_err
        self.assertEqual(code, 2)
        self.assertIn("error:", err.getvalue())

    def test_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "diff.json"
            code, _, _ = self._run(
                [run_record("a.mps", seconds=2.0), run_record("gone.mps")],
                [run_record("a.mps", seconds=1.0)],
                "--json", str(out_path), "--allow-missing")
            self.assertEqual(code, 0)
            payload = json.loads(out_path.read_text())
        self.assertEqual(payload["tool"], "compare_runs")
        self.assertEqual(payload["summary"]["comparable"], 1)
        by_inst = {i["instance"]: i for i in payload["instances"]}
        self.assertEqual(by_inst["gone.mps"]["findings"], ["missing-in-candidate"])
        self.assertIsNone(by_inst["gone.mps"]["candidate"])
        self.assertAlmostEqual(by_inst["a.mps"]["time_ratio"], 1.01 / 2.01)

    def test_negative_shift_is_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run([run_record("a.mps")], [run_record("a.mps")],
                      "--time-shift", "-1")
        self.assertNotEqual(ctx.exception.code, 0)

    def test_nan_shift_is_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run([run_record("a.mps")], [run_record("a.mps")],
                      "--sgm-shift", "nan")
        self.assertNotEqual(ctx.exception.code, 0)

    def test_negative_top_is_rejected(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self._run([run_record("a.mps")], [run_record("a.mps")], "--top", "-2")
        self.assertNotEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
