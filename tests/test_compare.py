#!/usr/bin/env python3
"""Regression tests for the comparison harness's correctness contract."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sor_compare", ROOT / "scripts" / "compare.py")
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


class CommandTests(unittest.TestCase):
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
                              objective=objective, proof=proof, seconds=seconds)

    def test_zero_objective_uses_absolute_tolerance(self) -> None:
        self.assertTrue(compare.objectives_agree(0.0, 5e-8, 1e-7, 1e-6))
        self.assertFalse(compare.objectives_agree(0.0, 2e-7, 1e-7, 1e-6))

    def test_nonfinite_objectives_never_agree(self) -> None:
        self.assertFalse(compare.objectives_agree(float("nan"), 0.0, 1e-7, 1e-6))
        self.assertFalse(compare.objectives_agree(float("inf"), 0.0, 1e-7, 1e-6))
        self.assertFalse(compare.objectives_agree(None, 0.0, 1e-7, 1e-6))

    def test_sor_optimal_requires_proof(self) -> None:
        self.assertFalse(compare.is_certified_success(self.result("sor:simplex")))
        self.assertTrue(compare.is_certified_success(
            self.result("sor:simplex", proof="ProvedOptimalFP")))
        self.assertTrue(compare.is_certified_success(self.result("highs")))

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


if __name__ == "__main__":
    unittest.main()
