#!/usr/bin/env python3
"""Regression tests for the comparison harness's correctness contract."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import math
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sor_compare", ROOT / "scripts" / "compare.py")
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


class CommandTests(unittest.TestCase):
    def test_solver_timeout_kills_descendants_holding_output_pipes(self) -> None:
        child = "import time; time.sleep(4)"
        parent = ("import subprocess,sys,time; "
                  f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
                  "time.sleep(4)")
        start = time.monotonic()
        with self.assertRaises(compare.subprocess.TimeoutExpired):
            compare.run_solver_process([sys.executable, "-c", parent], 0.2)
        self.assertLess(time.monotonic() - start, 2.0)

    def test_recursive_duplicate_models_fail_before_writing_results(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nested = root / "nested"
            nested.mkdir()
            (root / "same.mps.gz").write_bytes(b"model")
            (nested / "same.mps.gz").write_bytes(b"model")
            output = root / "result.jsonl"
            solutions = root / "solutions"
            argv = ["compare.py", str(root), "--solvers", "highs",
                    "--jsonl", str(output), "--solutions-dir", str(solutions)]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(compare.main(), 2)
            self.assertIn("duplicate model identities", err.getvalue())
            self.assertFalse(output.exists())
            self.assertFalse(solutions.exists())

    def test_existing_jsonl_is_not_appended(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result.jsonl"
            path.write_text("old evidence\n")
            model = Path(td) / "model.mps"
            model.write_text("NAME X\nENDATA\n")
            argv = ["compare.py", str(model), "--solvers", "highs",
                    "--jsonl", str(path)]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(compare.main(), 2)
            self.assertEqual(path.read_text(), "old evidence\n")
            self.assertIn("refusing to mix campaigns", err.getvalue())

    def test_frozen_binaries_are_copied_and_detect_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()
            (source / "sor_solve").write_bytes(b"solver-1")
            (source / "sor_check").write_bytes(b"checker-1")
            frozen, digest = compare.freeze_sor_binaries(
                source / "sor_solve", root / "evidence")
            self.assertEqual(frozen.read_bytes(), b"solver-1")
            self.assertEqual(digest, compare.sha256_file(frozen))
            (source / "sor_solve").write_bytes(b"solver-2")
            with self.assertRaisesRegex(RuntimeError, "executable changed"):
                compare.verify_executable(source / "sor_solve", digest)
            with self.assertRaisesRegex(RuntimeError, "executable changed"):
                compare.freeze_sor_binaries(source / "sor_solve", root / "evidence")

    def test_qps_reference_reads_identical_mps_alias(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "problem.qps"
            model.write_bytes(b"NAME TEST\nQUADOBJ\n X X 1\nENDATA\n")
            observed = []
            class FakeHighs:
                def setOptionValue(self, *_args):
                    return "HighsStatus.kOk"
                def readModel(self, path):
                    alias = Path(path)
                    observed.append((alias.suffix, alias.read_bytes()))
                    return "HighsStatus.kOk"
                def getRunTime(self):
                    return 0.0
                def run(self):
                    pass
                def getModelStatus(self):
                    return "HighsModelStatus.kOptimal"
                def getObjectiveValue(self):
                    return 1.0
                def version(self):
                    return "test"
                def getInfo(self):
                    return type("Info", (), {"simplex_iteration_count": 0})()
            with mock.patch.dict(sys.modules, {"highspy": type("Module", (), {"Highs": FakeHighs})()}):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    compare.highs_worker(model, 1.0)
            row = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(observed, [(".mps", model.read_bytes())])
            self.assertEqual(row["status"], "Optimal")
            self.assertEqual(row["reference_model_sha256"], compare.sha256_file(model))

    def test_qp_primal_check_never_counts_as_optimality_proof(self) -> None:
        r = compare.Result(solver="sor:qp", instance="x.qps",
                           status="Optimal", proof="ProvedKKT",
                           checker_verified=True,
                           checker_validation_scope="qp_primal_point")
        self.assertFalse(compare.is_certified_success(r))
        r.checker_validation_scope = "qp_kkt_f64"
        self.assertTrue(compare.is_certified_success(r))

    def test_pinned_milp_uses_one_worker_unless_explicitly_overridden(self) -> None:
        fake = compare.subprocess.CompletedProcess(
            [], 0, stdout="status: Feasible\n", stderr="")
        with mock.patch.object(compare, "run_solver_process", return_value=fake):
            serial = compare.run_sor(
                Path("model.mps"), "milp", "cpu", 1.0, 1e-7,
                Path("sor_solve"), "sor:milp", cpu=0)
            self.assertEqual(serial.command[serial.command.index("--bab-threads") + 1], "1")
            parallel = compare.run_sor(
                Path("model.mps"), "milp", "cpu", 1.0, 1e-7,
                Path("sor_solve"), "sor:milp", cpu=0,
                sor_extra=["--bab-threads", "3"])
            self.assertEqual(parallel.command.count("--bab-threads"), 1)
            self.assertEqual(parallel.command[parallel.command.index("--bab-threads") + 1], "3")

    def test_dual_is_a_simplex_method(self) -> None:
        cmd = compare.build_sor_command(
            Path("model.mps"), "dual", "cpu", 30.0, 1e-6,
            Path("sor_solve"), None, "product", 200_000)
        self.assertEqual(cmd[cmd.index("--engine") + 1], "simplex")
        self.assertEqual(cmd[cmd.index("--method") + 1], "dual")
        self.assertEqual(cmd[cmd.index("--basis-update") + 1], "product")
        self.assertEqual(cmd[cmd.index("--pricing") + 1], "choose")
        self.assertEqual(cmd[cmd.index("--max-iter") + 1], "200000")

    def test_explicit_alias_overrides_global_method(self) -> None:
        cmd = compare.build_sor_command(
            Path("model.mps"), "primal", "cpu", 1.0, 1e-7,
            Path("sor_solve"), "dual", "ft", None)
        self.assertEqual(cmd[cmd.index("--method") + 1], "primal")
        self.assertEqual(cmd[cmd.index("--basis-update") + 1], "ft")

    def test_hpr_variant_maps_to_engine_and_flag(self) -> None:
        cmd = compare.build_sor_command(
            Path("model.mps"), "hpr-full", "cpu", 1.0, 1e-7,
            Path("sor_solve"), None, "product", None)
        self.assertEqual(cmd[cmd.index("--engine") + 1], "hpr")
        self.assertIn("--hpr-full", cmd)
        self.assertNotIn("--basis-update", cmd)
        self.assertNotIn("--pricing", cmd)

    def test_pricing_is_forwarded(self) -> None:
        cmd = compare.build_sor_command(
            Path("model.mps"), "simplex", "cpu", 1.0, 1e-7,
            Path("sor_solve"), "dual", "product", None, "dse")
        self.assertEqual(cmd[cmd.index("--pricing") + 1], "dse")

    def test_cost_perturbation_is_opt_in_and_forwarded(self) -> None:
        base = compare.build_sor_command(
            Path("model.mps"), "simplex", "cpu", 1.0, 1e-7,
            Path("sor_solve"), "dual", "product", None)
        self.assertNotIn("--dual-cost-perturbation", base)
        perturbed = compare.build_sor_command(
            Path("model.mps"), "simplex", "cpu", 1.0, 1e-7,
            Path("sor_solve"), "dual", "product", None, "choose", 1.5)
        self.assertEqual(
            perturbed[perturbed.index("--dual-cost-perturbation") + 1], "1.5")


class CorrectnessTests(unittest.TestCase):
    def result(self, solver: str, objective: float = 0.0, *,
               status: str = "Optimal", proof: str | None = None,
               seconds: float = 0.1) -> object:
        return compare.Result(solver=solver, instance="x.mps", status=status,
                              objective=objective, proof=proof, seconds=seconds,
                              checker_verified=(True if solver.startswith("sor")
                                                else None),
                              checker_validation_scope=("lp_optimality_f64"
                                                        if solver.startswith("sor") else None))

    def test_zero_objective_uses_absolute_tolerance(self) -> None:
        # At the shipped defaults (abs == rel == 1e-7) the band at a zero
        # reference is exactly 1e-7: max(1e-7, 1e-7 * (1 + 0)).
        self.assertTrue(compare.objectives_agree(0.0, 5e-8, 1e-7, 1e-7))
        self.assertTrue(compare.objectives_agree(9e-8, 0.0, 1e-7, 1e-7))
        self.assertFalse(compare.objectives_agree(0.0, 2e-7, 1e-7, 1e-7))
        self.assertFalse(compare.objectives_agree(2e-7, 0.0, 1e-7, 1e-7))
        # The absolute term is a FLOOR, so a tiny reference can never shrink
        # the band below it however small rel_tol is.
        self.assertTrue(compare.objectives_agree(0.0, 9e-8, 1e-7, 1e-30))

    def test_band_is_scaled_by_the_reference_not_the_candidate(self) -> None:
        # A candidate must not be able to widen its own tolerance by being
        # wrong in the large direction. Reference 1.0 gives a 2e-7 band; the
        # same pair with the roles swapped gives a band of 1e6 * 1e-7 and
        # would agree, which is exactly the asymmetry being ruled out.
        self.assertFalse(compare.objectives_agree(1e6, 1.0, 1e-7, 1e-7))
        self.assertTrue(compare.objectives_agree(1.0 + 1e-7, 1.0, 1e-7, 1e-7))
        self.assertFalse(compare.objectives_agree(1.0 + 1e-5, 1.0, 1e-7, 1e-7))

    def test_default_tolerances_are_the_primary_gate(self) -> None:
        # The 1e-4 relative agreement this harness used to ship was looser
        # than either solver's optimality tolerance, so the harness itself
        # could certify a wrong seventh digit. Guard the defaults.
        parser_defaults = {a.dest: a.default
                           for a in compare.build_parser()._actions}
        self.assertEqual(parser_defaults["tol"], 1e-7)
        self.assertEqual(parser_defaults["obj_rel_tol"], 1e-7)
        self.assertEqual(parser_defaults["obj_abs_tol"], 1e-7)

    def test_nonfinite_objectives_never_agree(self) -> None:
        self.assertFalse(compare.objectives_agree(float("nan"), 0.0, 1e-7, 1e-6))
        self.assertFalse(compare.objectives_agree(float("inf"), 0.0, 1e-7, 1e-6))
        self.assertFalse(compare.objectives_agree(None, 0.0, 1e-7, 1e-6))

    def test_sor_optimal_requires_proof(self) -> None:
        self.assertFalse(compare.is_certified_success(self.result("sor:simplex")))
        self.assertTrue(compare.is_certified_success(
            self.result("sor:simplex", proof="ProvedOptimalFP")))
        unchecked = self.result("sor:simplex", proof="ProvedOptimalFP")
        unchecked.checker_verified = None
        self.assertFalse(compare.is_certified_success(unchecked))
        rejected = self.result("sor:simplex", proof="ProvedOptimalFP")
        rejected.checker_verified = False
        self.assertFalse(compare.is_certified_success(rejected))
        incumbent_only = self.result("sor:milp", proof="ProvedGlobalEpsilon")
        incumbent_only.checker_validation_scope = "milp_incumbent"
        self.assertFalse(compare.is_certified_success(incumbent_only))
        self.assertTrue(compare.is_certified_success(self.result("highs")))

    def test_stage_attribution_is_machine_parseable(self) -> None:
        text = (
            "iterations:        123 (FO 100, crossover 20, simplex 3)\n"
            "stage time (ms):    FO 1.250, crossover 0.500, simplex 0.125\n"
            "crossover:         attempted yes, basis valid no, cold fallback yes\n"
        )
        iters = compare._PAT["stage_iters"].search(text)
        times = compare._PAT["stage_ms"].search(text)
        flags = compare._PAT["crossover"].search(text)
        self.assertEqual(iters.groups(), ("100", "20", "3"))
        self.assertEqual(times.groups(), ("1.250", "0.500", "0.125"))
        self.assertEqual(flags.groups(), ("yes", "no", "yes"))

    def test_reference_is_not_first_solver_objective(self) -> None:
        interrupted = self.result("sor:simplex", 999.0, status="Interrupted")
        highs = self.result("highs", 0.0)
        results = {"sor:simplex": interrupted, "highs": highs}
        self.assertIs(compare.choose_reference(
            results, ["sor:simplex", "highs"], None), highs)

    def test_unavailable_explicit_reference_is_not_replaced(self) -> None:
        ours = self.result("sor:simplex", proof="ProvedOptimalFP")
        highs = self.result("highs", status="unavailable")
        self.assertIsNone(compare.choose_reference(
            {"sor:simplex": ours, "highs": highs},
            ["sor:simplex", "highs"], "highs"))

    def test_failed_repetition_cannot_be_hidden_by_median(self) -> None:
        good = self.result("sor:simplex", proof="ProvedOptimalFP")
        bad = self.result("sor:simplex", status="timeout", seconds=2.0)
        aggregate = compare.aggregate_repetitions([good, bad, good])
        self.assertEqual(aggregate.status, "timeout")
        self.assertEqual(len(aggregate.samples_s), 3)

    def test_report_penalizes_wrong_objective_and_timeout(self) -> None:
        ours = self.result("sor:simplex", 1.0, proof="ProvedOptimalFP")
        highs = self.result("highs", 0.0)
        timeout = compare.Result(solver="slow", instance="x.mps",
                                 status="timeout", seconds=1.0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary = compare.report(
                [ours, highs, timeout], ["sor:simplex", "highs", "slow"],
                1.0, 1e-7, 1e-6, 2.0, "highs")
        self.assertEqual(summary["mismatches"], 1)
        self.assertIn("4.0000", output.getvalue())  # PAR-2 penalty, not 1 second

    def test_ordinary_report_is_provisional_not_a_public_claim(self) -> None:
        ours = self.result("sor:simplex", proof="ProvedOptimalFP")
        highs = self.result("highs")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            compare.report([ours, highs], ["sor:simplex", "highs"],
                           1.0, 1e-7, 1e-7, 2.0, "highs")
        text = output.getvalue()
        self.assertIn("PROVISIONAL COMPARISON", text)
        self.assertNotIn("PUBLIC CLAIM GATE", text)
        self.assertIn("claim status      : INCOMPLETE", text)


class CheckerExecutionTests(unittest.TestCase):
    SOLVER_OUTPUT = (
        "status:            Optimal\n"
        "proof_level:       ProvedOptimalFP\n"
        "objective:         1.0000000000e+00\n"
        "dual bound:        1.0000000000e+00\n"
        "max primal viol:   0.000e+00\n"
        "iterations:        1\n"
        "timing (ms)\n"
        "  total                 1.000\n"
    )

    def run_case(self, checker_result=None, *, make_solution=True,
                 make_checker=True):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exe = root / "sor_solve"
            checker = root / "sor_check"
            model = root / "model.mps"
            solution = root / "result.sol"
            exe.write_bytes(b"solver")
            model.write_text("NAME X\nENDATA\n")
            if make_checker:
                checker.write_bytes(b"checker")
            solver_result = compare.subprocess.CompletedProcess(
                [str(exe)], 0, stdout=self.SOLVER_OUTPUT, stderr="")
            def solver_mock(*args, **kwargs):
                if make_solution:
                    solution.write_text("solution")
                return solver_result
            def checker_mock(*args, **kwargs):
                if isinstance(checker_result, BaseException):
                    raise checker_result
                return checker_result
            with mock.patch.object(compare, "run_solver_process",
                                   side_effect=solver_mock):
                with mock.patch.object(compare.subprocess, "run",
                                       side_effect=checker_mock):
                    return compare.run_sor(
                        model, "dual", "cpu", 1.0, 1e-7, exe, "sor:dual",
                        method=None, basis_update="default", max_iter=None,
                        cpu=0, pricing="choose", dual_cost_perturbation=0.0,
                        sor_extra=[], solution_out=solution)

    def test_checker_success_requires_parseable_scope_and_records_evidence(self):
        checked = compare.subprocess.CompletedProcess(
            ["sor_check"], 0,
            stdout="validation: lp_optimality_f64\nVERIFIED\n", stderr="")
        result = self.run_case(checked)
        self.assertTrue(result.checker_verified)
        self.assertEqual(result.checker_validation_scope, "lp_optimality_f64")
        self.assertEqual(result.checker_returncode, 0)
        self.assertEqual(result.checker_stdout, checked.stdout)
        self.assertIsNotNone(result.checker_executable_sha256)
        self.assertTrue(compare.is_certified_success(result))

    def test_missing_solution_fails_validation(self):
        result = self.run_case(make_solution=False)
        self.assertFalse(result.checker_verified)
        self.assertIn("did not write", result.checker_error)
        self.assertFalse(compare.is_certified_success(result))

    def test_missing_checker_fails_validation(self):
        result = self.run_case(make_checker=False)
        self.assertFalse(result.checker_verified)
        self.assertIn("not found", result.checker_error)

    def test_checker_failure_is_distinct_from_solver_failure(self):
        checked = compare.subprocess.CompletedProcess(
            ["sor_check"], 1,
            stdout="validation: lp_optimality_f64\nREJECTED\n",
            stderr="dual residual")
        result = self.run_case(checked)
        self.assertEqual(result.status, "Optimal")
        self.assertEqual(result.solver_returncode, 0)
        self.assertEqual(result.solver_stdout, self.SOLVER_OUTPUT)
        self.assertEqual(result.solver_stderr, "")
        self.assertFalse(result.checker_verified)
        self.assertEqual(result.checker_returncode, 1)
        self.assertEqual(result.checker_stderr, "dual residual")

    def test_checker_unverified_is_not_recorded_as_rejected_or_verified(self):
        checked = compare.subprocess.CompletedProcess(
            ["sor_check"], 3,
            stdout=("validation: milp_infeasibility_unverified\n"
                    "UNVERIFIED\n"), stderr="")
        result = self.run_case(checked)
        self.assertIsNone(result.checker_verified)
        self.assertEqual(result.checker_validation_scope,
                         "milp_infeasibility_unverified")
        self.assertIn("outcome unverified", result.checker_error)
        self.assertFalse(compare.is_certified_success(result))

    def test_unparseable_checker_output_fails_closed(self):
        checked = compare.subprocess.CompletedProcess(
            ["sor_check"], 0, stdout="VERIFIED\n", stderr="")
        result = self.run_case(checked)
        self.assertFalse(result.checker_verified)
        self.assertIsNone(result.checker_validation_scope)

    def test_checker_timeout_fails_closed_and_retains_output(self):
        timeout = compare.subprocess.TimeoutExpired(
            ["sor_check"], 31, output="partial stdout", stderr="partial stderr")
        result = self.run_case(timeout)
        self.assertFalse(result.checker_verified)
        self.assertTrue(result.checker_timed_out)
        self.assertEqual(result.checker_stdout, "partial stdout")
        self.assertEqual(result.checker_stderr, "partial stderr")




class PublicClaimGateTests(unittest.TestCase):
    """The gate that a published claim must pass.

    These exist because a claim WAS published off the wrong statistic: on
    Netlib-93 the geometric mean of per-model ratios read 0.8684 (apparent
    PASS) while the true metric, the ratio of shifted geometric means, read
    1.4307 (FAIL by 50%). Every test here is written so that the old
    interpretation fails it.
    """

    def result(self, solver, instance, seconds, *, proof="ProvedOptimalFP",
               status="Optimal", objective=1.0, mad=0.0, wall=None):
        return compare.Result(solver=solver, instance=instance, status=status,
                              objective=objective, proof=proof,
                              seconds=seconds, wall_s=wall if wall is not None else seconds,
                              mad_s=mad, median_s=seconds, noisy=False,
                              checker_verified=(True if solver.startswith("sor")
                                                else None),
                              checker_validation_scope=("lp_optimality_f64"
                                                        if solver.startswith("sor") else None))

    def suite(self, rows):
        """rows: [(instance, sor_seconds, highs_seconds)]"""
        by = {}
        for inst, cs, hs in rows:
            by[inst] = {"sor:simplex": self.result("sor:simplex", inst, cs),
                        "highs": self.result("highs", inst, hs)}
        return by

    # ---- the formula itself -------------------------------------------
    def test_shifted_geomean_is_exp_mean_log_minus_shift(self):
        vals = [0.5, 2.0, 8.0]
        got = compare.shifted_geomean(vals, 1.0)
        want = math.exp(sum(math.log(v + 1.0) for v in vals) / 3) - 1.0
        self.assertAlmostEqual(got, want, places=12)

    def test_sgm_of_identical_values_is_that_value(self):
        self.assertAlmostEqual(compare.shifted_geomean([3.0] * 5, 1.0), 3.0, places=12)

    # ---- THE regression: ratio-of-SGMs, not geomean-of-ratios ----------
    def test_gate_uses_ratio_of_sgms_not_geomean_of_ratios(self):
        # Constructed to reproduce the real failure: SOR wins many tiny models
        # and loses one big one. The geomean of per-model ratios says 0.5 (a
        # comfortable PASS); the ratio of SGMs says well above 1 (a FAIL).
        rows = [(f"tiny{i}.mps", 0.0005, 0.001) for i in range(20)]
        rows.append(("big.mps", 40.0, 2.0))
        by = self.suite(rows)

        ratios = [0.5] * 20 + [20.0]
        geomean_of_ratios = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        self.assertLess(geomean_of_ratios, 0.95)      # the OLD number "passes"

        g = compare.evaluate_public_claim(by, "sor:simplex", "highs",
                                          time_limit=60.0)
        self.assertGreater(g.sgm_ratio, 1.0)          # the TRUE number fails
        self.assertFalse(g.passed)
        self.assertTrue(any("SGM ratio" in f for f in g.failures))

    def test_sgm_ratio_is_dominated_by_expensive_models(self):
        # Making the one expensive model slower must move the metric; making a
        # sub-millisecond model slower must barely move it. That asymmetry is
        # the entire point of the shift.
        base = self.suite([("tiny.mps", 0.001, 0.001), ("big.mps", 10.0, 10.0)])
        g0 = compare.evaluate_public_claim(base, "sor:simplex", "highs", time_limit=60.0)
        slow_big = self.suite([("tiny.mps", 0.001, 0.001), ("big.mps", 20.0, 10.0)])
        g1 = compare.evaluate_public_claim(slow_big, "sor:simplex", "highs", time_limit=60.0)
        slow_tiny = self.suite([("tiny.mps", 0.002, 0.001), ("big.mps", 10.0, 10.0)])
        g2 = compare.evaluate_public_claim(slow_tiny, "sor:simplex", "highs", time_limit=60.0)
        self.assertGreater(g1.sgm_ratio - g0.sgm_ratio, 0.10)
        self.assertLess(g2.sgm_ratio - g0.sgm_ratio, 0.001)

    # ---- PAR-2 ---------------------------------------------------------
    def test_uncertified_model_is_par2_charged_not_dropped(self):
        by = self.suite([("ok.mps", 1.0, 1.0)])
        by["bad.mps"] = {
            "sor:simplex": self.result("sor:simplex", "bad.mps", 0.001,
                                       proof="FeasibleWithGap", status="Feasible"),
            "highs": self.result("highs", "bad.mps", 1.0)}
        g = compare.evaluate_public_claim(by, "sor:simplex", "highs", time_limit=60.0)
        self.assertEqual(g.scored, 2)             # not dropped
        self.assertEqual(g.par2_penalised, 1)
        self.assertEqual(g.candidate_certified, 1)
        # 2 x 60 s charged, so it cannot count as a win despite 1 ms measured.
        self.assertEqual(g.wins, 0)
        self.assertFalse(g.passed)
        self.assertTrue(any("proofs" in f for f in g.failures))

    def test_missing_candidate_row_is_par2_not_ignored(self):
        by = {"gone.mps": {"highs": self.result("highs", "gone.mps", 1.0)}}
        g = compare.evaluate_public_claim(by, "sor:simplex", "highs", time_limit=60.0)
        self.assertEqual(g.scored, 1)
        self.assertEqual(g.par2_penalised, 1)

    # ---- noise-aware wins ----------------------------------------------
    def test_difference_inside_the_noise_band_is_a_tie(self):
        by = {"a.mps": {
            "sor:simplex": self.result("sor:simplex", "a.mps", 0.99, mad=0.05),
            "highs": self.result("highs", "a.mps", 1.00, mad=0.05)}}
        g = compare.evaluate_public_claim(by, "sor:simplex", "highs", time_limit=60.0)
        self.assertEqual((g.wins, g.losses, g.ties), (0, 0, 1))

    def test_difference_outside_the_noise_band_is_a_win(self):
        by = {"a.mps": {
            "sor:simplex": self.result("sor:simplex", "a.mps", 0.50, mad=0.01),
            "highs": self.result("highs", "a.mps", 1.00, mad=0.01)}}
        g = compare.evaluate_public_claim(by, "sor:simplex", "highs", time_limit=60.0)
        self.assertEqual((g.wins, g.losses, g.ties), (1, 0, 0))

    # ---- the remaining gates -------------------------------------------
    def test_objective_disagreement_fails_correctness(self):
        by = {"a.mps": {
            "sor:simplex": self.result("sor:simplex", "a.mps", 0.1, objective=1.0),
            "highs": self.result("highs", "a.mps", 1.0, objective=1.0 + 1e-3)}}
        g = compare.evaluate_public_claim(by, "sor:simplex", "highs", time_limit=60.0)
        self.assertEqual(g.objective_mismatches, 1)
        self.assertTrue(any("correctness" in f for f in g.failures))

    def test_tail_rule_flags_unwaived_2x_regression(self):
        by = self.suite([("a.mps", 3.0, 1.0)])
        g = compare.evaluate_public_claim(by, "sor:simplex", "highs",
                                          time_limit=60.0,
                                          baseline_solver={"a.mps": 1.0})
        self.assertEqual(len(g.tail_violations), 1)
        self.assertTrue(any("2x the pinned baseline" in f for f in g.failures))

    def test_tail_rule_respects_a_committed_waiver(self):
        by = self.suite([("a.mps", 3.0, 1.0)])
        g = compare.evaluate_public_claim(by, "sor:simplex", "highs",
                                          time_limit=60.0,
                                          baseline_solver={"a.mps": 1.0},
                                          allow={"a.mps": {"reason": "documented"}})
        self.assertEqual(g.tail_violations, [])

    def test_process_wall_regression_fails(self):
        by = self.suite([("a.mps", 1.0, 1.0)])
        g = compare.evaluate_public_claim(by, "sor:simplex", "highs",
                                          time_limit=60.0,
                                          baseline_solver={"a.mps": 0.5},
                                          baseline_wall={"a.mps": 0.5})
        self.assertTrue(any("process-wall" in f for f in g.failures))

    def test_a_genuinely_winning_suite_passes_every_gate(self):
        rows = [(f"m{i}.mps", 0.40, 1.00) for i in range(10)]
        by = self.suite(rows)
        g = compare.evaluate_public_claim(by, "sor:simplex", "highs",
                                          time_limit=60.0,
                                          baseline_solver={f"m{i}.mps": 0.50 for i in range(10)},
                                          baseline_wall={f"m{i}.mps": 0.50 for i in range(10)})
        self.assertLessEqual(g.sgm_ratio, 0.95)
        self.assertGreaterEqual(g.win_rate, 60.0)
        self.assertTrue(g.passed, g.failures)



class FailClosedTests(unittest.TestCase):
    """Every way the gate could report success without the evidence to justify it.

    A gate that skips itself when its inputs are absent is worse than no gate,
    because it prints PASS.
    """

    def r(self, solver, inst, seconds, *, proof="ProvedOptimalFP",
          status="Optimal", objective=1.0, error=None, wall=None):
        return compare.Result(solver=solver, instance=inst, status=status,
                              objective=objective, proof=proof, seconds=seconds,
                              wall_s=wall if wall is not None else seconds,
                              median_wall_s=wall if wall is not None else seconds,
                              median_s=seconds, mad_s=0.0, noisy=False,
                              error=error,
                              checker_verified=(True if solver.startswith("sor")
                                                else None),
                              checker_validation_scope=("lp_optimality_f64"
                                                        if solver.startswith("sor") else None))

    def suite(self, rows):
        return {i: {"sor:simplex": self.r("sor:simplex", i, c),
                    "highs": self.r("highs", i, h)} for i, c, h in rows}

    def winning(self, n=5):
        return self.suite([(f"m{i}.mps", 0.4, 1.0) for i in range(n)])

    def full_evidence(self, n=5):
        return dict(expected_instances={f"m{i}.mps" for i in range(n)},
                    baseline_solver={f"m{i}.mps": 0.5 for i in range(n)},
                    baseline_wall={f"m{i}.mps": 0.5 for i in range(n)},
                    allow={},
                    candidate_checks={f"m{i}.mps": (True, "VERIFIED")
                                      for i in range(n)})

    def test_complete_evidence_can_pass(self):
        g = compare.evaluate_public_claim(
            self.winning(), "sor:simplex", "highs", time_limit=60.0,
            claim_mode=True, **self.full_evidence())
        self.assertEqual(g.status, "PASS", g.incompleteness + g.failures)

    def test_missing_baseline_is_incomplete_not_pass(self):
        ev = self.full_evidence(); ev.pop("baseline_solver")
        g = compare.evaluate_public_claim(
            self.winning(), "sor:simplex", "highs", time_limit=60.0,
            claim_mode=True, **ev)
        self.assertEqual(g.status, "INCOMPLETE")
        self.assertFalse(g.passed)

    def test_missing_wall_baseline_is_incomplete(self):
        ev = self.full_evidence(); ev.pop("baseline_wall")
        g = compare.evaluate_public_claim(
            self.winning(), "sor:simplex", "highs", time_limit=60.0,
            claim_mode=True, **ev)
        self.assertEqual(g.status, "INCOMPLETE")

    def test_missing_allow_list_is_incomplete(self):
        ev = self.full_evidence(); ev.pop("allow")
        g = compare.evaluate_public_claim(
            self.winning(), "sor:simplex", "highs", time_limit=60.0,
            claim_mode=True, **ev)
        self.assertEqual(g.status, "INCOMPLETE")

    def test_partial_suite_is_incomplete(self):
        # Ran 3 of the manifest's 5 models: a subset must never pass.
        by = self.suite([(f"m{i}.mps", 0.4, 1.0) for i in range(3)])
        ev = self.full_evidence()
        g = compare.evaluate_public_claim(
            by, "sor:simplex", "highs", time_limit=60.0, claim_mode=True, **ev)
        self.assertEqual(g.status, "INCOMPLETE")
        self.assertEqual(len(g.missing_instances), 2)

    def test_missing_reference_row_is_incomplete_not_skipped(self):
        by = self.winning()
        del by["m0.mps"]["highs"]
        g = compare.evaluate_public_claim(
            by, "sor:simplex", "highs", time_limit=60.0, claim_mode=True,
            **self.full_evidence())
        self.assertEqual(g.status, "INCOMPLETE")
        self.assertTrue(g.invalid_references)

    def test_errored_reference_cannot_certify(self):
        # The HiGHS worker reports rejected options through Result.error; such
        # a row is a misconfigured oracle, not a reference.
        by = self.winning()
        by["m0.mps"]["highs"] = self.r("highs", "m0.mps", 1.0,
                                       error="options not applied: solver")
        g = compare.evaluate_public_claim(
            by, "sor:simplex", "highs", time_limit=60.0, claim_mode=True,
            **self.full_evidence())
        self.assertEqual(g.status, "INCOMPLETE")
        self.assertTrue(any("error" in w for _, w in g.invalid_references))

    def test_unavailable_reference_cannot_certify(self):
        by = self.winning()
        by["m0.mps"]["highs"] = self.r("highs", "m0.mps", 1.0,
                                       status="unavailable", proof=None)
        g = compare.evaluate_public_claim(
            by, "sor:simplex", "highs", time_limit=60.0, claim_mode=True,
            **self.full_evidence())
        self.assertEqual(g.status, "INCOMPLETE")

    def test_wrong_objective_is_par2_not_its_fast_real_time(self):
        # The candidate is fast AND wrong. It must not contribute 0.001 s.
        by = self.winning()
        by["m0.mps"]["sor:simplex"] = self.r("sor:simplex", "m0.mps", 0.001,
                                             objective=99.0)
        g = compare.evaluate_public_claim(
            by, "sor:simplex", "highs", time_limit=60.0, claim_mode=True,
            **self.full_evidence())
        self.assertEqual(g.objective_mismatches, 1)
        self.assertEqual(g.par2_penalised, 1)
        self.assertEqual(g.status, "FAIL")
        # 2 x 60 s dominates: the ratio must be far above 1 despite four wins.
        self.assertGreater(g.sgm_ratio, 1.0)
        self.assertEqual(g.wins, 4)

    def test_rejected_measured_solution_is_par2_before_scoring(self):
        by = self.winning()
        ev = self.full_evidence()
        ev["candidate_checks"]["m0.mps"] = (False, "dual residual rejected")
        g = compare.evaluate_public_claim(
            by, "sor:simplex", "highs", time_limit=60.0,
            claim_mode=True, **ev)
        self.assertEqual(g.par2_penalised, 1)
        self.assertEqual(g.candidate_certified, 4)
        self.assertEqual(g.status, "FAIL")
        self.assertTrue(any("independent checker" in f for f in g.failures))
        self.assertGreater(g.sgm_ratio, 1.0)

    def test_claim_scores_only_manifest_eligible_instances(self):
        by = self.winning()
        by["import-only.mps"] = {
            "sor:simplex": self.r("sor:simplex", "import-only.mps", 120.0,
                                   status="timeout", proof=None),
            "highs": self.r("highs", "import-only.mps", 0.01)}
        g = compare.evaluate_public_claim(
            by, "sor:simplex", "highs", time_limit=60.0,
            claim_mode=True, **self.full_evidence())
        self.assertEqual(g.scored, 5)
        self.assertNotIn("import-only.mps", g.missing_instances)
        self.assertEqual(g.status, "PASS", g.incompleteness + g.failures)

    def test_baseline_coverage_and_values_are_exact(self):
        ev = self.full_evidence()
        ev["baseline_solver"]["unexpected.mps"] = 1.0
        ev["baseline_wall"]["m0.mps"] = float("nan")
        g = compare.evaluate_public_claim(
            self.winning(), "sor:simplex", "highs", time_limit=60.0,
            claim_mode=True, **ev)
        self.assertEqual(g.status, "INCOMPLETE")
        self.assertTrue(any("unexpected" in x for x in g.incompleteness))
        self.assertTrue(any("non-positive/non-finite" in x
                            for x in g.incompleteness))

    def test_wall_sum_uses_only_the_declared_suite(self):
        ev = self.full_evidence()
        ev["baseline_wall"]["unexpected.mps"] = 1000.0
        g = compare.evaluate_public_claim(
            self.winning(), "sor:simplex", "highs", time_limit=60.0,
            claim_mode=True, **ev)
        self.assertEqual(g.status, "INCOMPLETE")
        self.assertAlmostEqual(g.wall_baseline, 2.5)
        self.assertAlmostEqual(g.wall_ratio, 0.8)

    def test_tail_uses_solver_time_and_wall_uses_wall(self):
        # Solver time fine vs its baseline, wall doubled vs its baseline.
        by = {"a.mps": {
            "sor:simplex": self.r("sor:simplex", "a.mps", 0.40, wall=4.0),
            "highs": self.r("highs", "a.mps", 1.0, wall=1.0)}}
        g = compare.evaluate_public_claim(
            by, "sor:simplex", "highs", time_limit=60.0, claim_mode=True,
            expected_instances={"a.mps"},
            baseline_solver={"a.mps": 0.5},   # 0.40/0.5 = 0.8 -> no tail hit
            baseline_wall={"a.mps": 1.0},     # 4.0/1.0 = 4.0 -> wall regressed
            allow={})
        self.assertEqual(g.tail_violations, [])
        self.assertGreater(g.wall_ratio, 1.0)
        self.assertTrue(any("process-wall" in f for f in g.failures))

    def test_non_claim_mode_does_not_demand_the_evidence(self):
        # Exploratory sweeps must stay usable without a manifest or baseline.
        g = compare.evaluate_public_claim(
            self.winning(), "sor:simplex", "highs", time_limit=60.0)
        self.assertEqual(g.status, "PASS")


class NativeSourceRunnerTests(unittest.TestCase):
    def payload(self, seconds=0.0013123456789012345):
        return {
            "schema": "sor-highs-source-runner-v1",
            "kind": "solve_result",
            "read_status": "kOk",
            "run_status": "kOk",
            "options_applied": True,
            "model_status": "Optimal",
            "objective": 1.25,
            "solve_seconds": seconds,
            "simplex_iterations": 3,
            "highs_version": "1.15.1",
            "configuration": {
                "output_flag": False, "log_to_console": False,
                "threads": 1, "parallel": "off", "time_limit": 60.0,
                "primal_feasibility_tolerance": 1e-7,
                "dual_feasibility_tolerance": 1e-7,
                "random_seed": 0, "solver": "choose", "presolve": "choose",
                "run_crossover": "on", "solve_relaxation": False,
                "small_matrix_value": None,
            },
            "build_identity": {
                "kind": "sor-highs-source-api-runner", "runner_version": 1,
            },
        }

    def test_full_precision_api_time_is_primary(self):
        completed = compare.subprocess.CompletedProcess(
            ["runner"], 0, stdout=json.dumps(self.payload()) + "\n", stderr="")
        with mock.patch.object(compare.subprocess, "run", return_value=completed):
            result = compare.run_highs_source(
                Path("afiro.mps"), 60.0, "highs-source", Path("runner"))
        self.assertEqual(result.status, "Optimal")
        self.assertEqual(result.seconds, 0.0013123456789012345)
        self.assertEqual(result.solver_version, "1.15.1")
        self.assertIsNone(result.error)

    def test_cli_centisecond_zero_is_never_reference_evidence(self):
        cli = ("Model status        : Optimal\nObjective value     : 1.25\n"
               "HiGHS run time      : 0.00\n")
        completed = compare.subprocess.CompletedProcess(
            ["highs"], 0, stdout=cli, stderr="")
        with mock.patch.object(compare.subprocess, "run", return_value=completed):
            result = compare.run_highs_source(
                Path("afiro.mps"), 60.0, "highs-source", Path("highs"))
        self.assertEqual(result.status, "crash")
        self.assertIsNone(result.seconds)
        self.assertIn("native", result.error)

    def test_empty_allow_entry_cannot_waive_three_x_tail(self):
        candidate = compare.Result(
            solver="sor:simplex", instance="a.mps", status="Optimal",
            proof="ProvedOptimalFP", objective=1.0, seconds=3.0, wall_s=3.0,
            checker_verified=True,
            checker_validation_scope="lp_optimality_f64")
        reference = compare.Result(
            solver="highs", instance="a.mps", status="Optimal",
            objective=1.0, seconds=4.0, wall_s=4.0)
        gate = compare.evaluate_public_claim(
            {"a.mps": {"sor:simplex": candidate, "highs": reference}},
            "sor:simplex", "highs", 60.0,
            baseline_solver={"a.mps": 1.0}, allow={"a.mps": {}})
        self.assertEqual(gate.tail_violations, [("a.mps", 3.0)])

if __name__ == "__main__":
    unittest.main()
