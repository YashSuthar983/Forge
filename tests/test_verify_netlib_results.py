#!/usr/bin/env python3
"""Tests for the stored-benchmark-result verifier.

The verifier's job is to fail loudly on a result that cannot be trusted, and to
stay quiet about results that are merely incomplete. Both halves are tested
here: every failure kind is provoked, and the cases that must NOT be reported
(a timeout with no proof, an interrupted run with a large violation, a
deliberately partial smoke file) are pinned down too, because a verifier that
cries wolf on those is one nobody reads.
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
    "sor_verify_netlib", ROOT / "scripts" / "verify_netlib_results.py")
assert SPEC is not None and SPEC.loader is not None
v = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v
SPEC.loader.exec_module(v)


def sor_result(instance="a", *, status="Optimal", proof="ProvedOptimalFP",
               objective=1.0, violation=1e-12, solver="SOR"):
    return {"name": solver, "status": status, "objective": objective,
            "proof": proof, "row_violation": violation, "iterations": 10,
            "error": None}


def highs_result(objective=1.0, status="Optimal"):
    return {"name": "highs", "status": status, "objective": objective,
            "proof": None, "row_violation": None, "iterations": 9,
            "error": None}


def nested_record(instance, results):
    return {"record": "instance", "instance": instance, "rows": 1, "cols": 1,
            "nnz": 1, "results": results}


def flat_record(instance, *, status="Optimal", proof="ProvedOptimalFP",
                objective=1.0, primal_viol=1e-12):
    return {"instance": instance, "status": status, "proof": proof,
            "objective": objective, "primal_viol": primal_viol,
            "iterations": 10, "downgrade": None}


def run_record(instance, *, solver="sor:simplex", status="Optimal",
               proof="ProvedOptimalFP", objective=1.0, violation=1e-12):
    return {"record": "run", "solver": solver, "instance": instance,
            "status": status, "proof": proof, "objective": objective,
            "violation": violation, "iterations": 10}


class Fixture:
    """A temp directory holding a suite of .mps files and result files."""

    def __init__(self, instances=("a", "b")) -> None:
        self.instances = instances

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.suite = self.root / "mps"
        self.suite.mkdir()
        for name in self.instances:
            (self.suite / f"{name}.mps").write_text("NAME\nROWS\nENDATA\n")
        return self

    def __exit__(self, *exc: object) -> None:
        self._dir.cleanup()

    def write(self, name: str, records: list[dict]) -> Path:
        p = self.root / name
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        return p


def kinds(failures) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in failures:
        out[f.kind] = out.get(f.kind, 0) + 1
    return out


def check(fx: Fixture, path: Path, *, reference=None, ref_source=None,
          coverage=True, obj_tol=1e-6, viol_tol=1e-6):
    obs, schema, problems = v.load_file(path)
    if reference is None:
        reference, ref_source, _ = v.build_stored_reference(obs, obj_tol, obj_tol)
    report = v.verify_file(path, obs, schema,
                           {n for n in fx.instances}, reference,
                           ref_source or {}, abs_tol=obj_tol, rel_tol=obj_tol,
                           viol_tol=viol_tol, check_coverage=coverage)
    return report, problems


class LoadTests(unittest.TestCase):
    def test_nested_schema(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                {"record": "environment", "suite": "netlib"},
                nested_record("a", {"SOR": sor_result(), "highs": highs_result()}),
                {"record": "summary", "n_instances": 1},
            ])
            obs, schema, problems = v.load_file(p)
        self.assertEqual(schema, "nested")
        self.assertEqual(problems, [])
        self.assertEqual(len(obs), 2)
        self.assertEqual({o.solver for o in obs}, {"SOR", "highs"})
        self.assertEqual(obs[0].violation, 1e-12)

    def test_flat_schema(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [flat_record("a"), flat_record("b")])
            obs, schema, _ = v.load_file(p)
        self.assertEqual(schema, "flat")
        self.assertEqual([o.instance for o in obs], ["a", "b"])
        # A flat file records this project's solver only.
        self.assertTrue(all(o.ours for o in obs))
        self.assertEqual(obs[0].violation, 1e-12)

    def test_run_schema(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [run_record("a"), run_record("a", solver="highs",
                                                                proof=None)])
            obs, schema, _ = v.load_file(p)
        self.assertEqual(schema, "run")
        self.assertTrue(obs[0].ours)
        self.assertFalse(obs[1].ours)

    def test_environment_and_summary_records_are_not_results(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [{"record": "environment", "host": "x"},
                                     {"record": "summary", "solved": {}}])
            obs, _, problems = v.load_file(p)
        self.assertEqual(obs, [])
        self.assertEqual(problems, [])

    def test_malformed_json_is_reported_not_fatal(self) -> None:
        with Fixture() as fx:
            p = fx.root / "r.jsonl"
            p.write_text(json.dumps(flat_record("a")) + "\n{oops\n"
                         + json.dumps(flat_record("b")) + "\n")
            obs, _, problems = v.load_file(p)
        self.assertEqual(len(obs), 2)          # the good lines still load
        self.assertEqual(kinds(problems), {"malformed-json": 1})

    def test_unrecognized_record(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [{"totally": "unknown"}])
            obs, _, problems = v.load_file(p)
        self.assertEqual(obs, [])
        self.assertEqual(kinds(problems), {"unrecognized-record": 1})

    def test_unreadable_file(self) -> None:
        obs, schema, problems = v.load_file(Path("/no/such/results.jsonl"))
        self.assertEqual(obs, [])
        self.assertEqual(kinds(problems), {"unreadable-file": 1})

    def test_instance_names_are_normalized(self) -> None:
        for raw, want in (("25fv47", "25fv47"), ("25fv47.mps", "25fv47"),
                          ("a/b/afiro.mps", "afiro"), ("x.QPS", "x"),
                          ("model.lp", "model")):
            self.assertEqual(v.normalize_instance(raw), want)

    def test_solver_classification(self) -> None:
        for name in ("sor", "SOR", "sor:simplex", "SOR-simplex", "ours"):
            self.assertTrue(v.is_ours(name), name)
        for name in ("highs", "gurobi", "cbc", "scipy"):
            self.assertFalse(v.is_ours(name), name)


class ReferenceTests(unittest.TestCase):
    def test_reference_comes_from_stored_external_runs(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(), "highs": highs_result(5.0)})])
            obs, _, _ = v.load_file(p)
        ref, source, fails = v.build_stored_reference(obs, 1e-9, 1e-9)
        self.assertEqual(ref, {"a": 5.0})
        self.assertIn("stored:highs", source["a"])
        self.assertEqual(fails, [])

    def test_agreeing_stored_runs_are_not_a_disagreement(self) -> None:
        with Fixture() as fx:
            p1 = fx.write("r1.jsonl", [nested_record("a", {"highs": highs_result(5.0)})])
            p2 = fx.write("r2.jsonl", [nested_record("a", {"highs": highs_result(5.0 + 1e-12)})])
            obs = v.load_file(p1)[0] + v.load_file(p2)[0]
        ref, _, fails = v.build_stored_reference(obs, 1e-6, 1e-6)
        self.assertEqual(fails, [])
        self.assertAlmostEqual(ref["a"], 5.0)

    def test_disagreeing_stored_runs_are_reported_and_yield_no_reference(self) -> None:
        # This is the case that matters: if two stored HiGHS runs disagree,
        # neither can serve as the reference, and picking one silently would
        # make the verifier's own verdict arbitrary.
        with Fixture() as fx:
            p1 = fx.write("r1.jsonl", [nested_record("a", {"highs": highs_result(5.0)})])
            p2 = fx.write("r2.jsonl", [nested_record("a", {"highs": highs_result(7.0)})])
            obs = v.load_file(p1)[0] + v.load_file(p2)[0]
        ref, _, fails = v.build_stored_reference(obs, 1e-9, 1e-9)
        self.assertEqual(kinds(fails), {"reference-disagreement": 1})
        self.assertNotIn("a", ref)

    def test_non_optimal_external_run_is_not_a_reference(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [nested_record(
                "a", {"highs": highs_result(5.0, status="TimeLimit")})])
            obs, _, _ = v.load_file(p)
        ref, _, _ = v.build_stored_reference(obs, 1e-9, 1e-9)
        self.assertEqual(ref, {})

    def test_objectives_agree_is_relative(self) -> None:
        self.assertTrue(v.objectives_agree(1e6, 1e6 + 1.0, 0.0, 1e-5))
        self.assertFalse(v.objectives_agree(1e6, 1e6 + 1.0, 0.0, 1e-9))
        self.assertFalse(v.objectives_agree(None, 1.0, 1e-9, 1e-9))
        self.assertFalse(v.objectives_agree(float("nan"), 1.0, 1e-9, 1e-9))
        self.assertFalse(v.objectives_agree(float("inf"), float("inf"), 1e-9, 1e-9))


class VerifyTests(unittest.TestCase):
    def test_a_clean_full_file_has_no_findings(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(), "highs": highs_result(1.0)}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result(1.0)})])
            report, problems = check(fx, p)
        self.assertEqual(report.failures, [])
        self.assertEqual(problems, [])
        self.assertEqual(report.instances, 2)

    def test_missing_status(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(status=""),
                                    "highs": highs_result()}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result()})])
            report, _ = check(fx, p)
        self.assertEqual(kinds(report.failures), {"status-missing": 1})

    def test_optimal_without_a_proof(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(proof=None),
                                    "highs": highs_result()}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result()})])
            report, _ = check(fx, p)
        self.assertEqual(kinds(report.failures), {"proof-missing": 1})

    def test_optimal_with_an_unaccepted_proof(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(proof="BoundOnly"),
                                    "highs": highs_result()}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result()})])
            report, _ = check(fx, p)
        self.assertEqual(kinds(report.failures), {"unproved-optimal": 1})

    def test_timeout_owes_no_proof_and_no_violation_bound(self) -> None:
        # dfl001 at the time limit reports violation 161 with proof None. That
        # describes a timeout, not a defect, and must not be reported.
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(status="Interrupted",
                                                      proof=None, violation=161.6),
                                    "highs": highs_result()}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result()})])
            report, _ = check(fx, p)
        self.assertEqual(report.failures, [])

    def test_timeout_objective_is_not_compared(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(status="timeout", proof=None,
                                                      objective=999.0),
                                    "highs": highs_result(1.0)}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result(1.0)})])
            report, _ = check(fx, p)
        self.assertEqual(report.failures, [])

    def test_objective_mismatch(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(objective=2.0),
                                    "highs": highs_result(1.0)}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result(1.0)})])
            report, _ = check(fx, p)
        self.assertEqual(kinds(report.failures), {"objective-mismatch": 1})
        f = report.failures[0]
        self.assertEqual(f.observed, 2.0)
        self.assertEqual(f.expected, 1.0)
        self.assertIn("stored:highs", f.reference_source or "")

    def test_violation_too_large_on_an_optimal_claim(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(violation=1e-3),
                                    "highs": highs_result()}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result()})])
            report, _ = check(fx, p)
        self.assertEqual(kinds(report.failures), {"violation-too-large": 1})

    def test_non_finite_violation_is_reported(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(violation=float("inf")),
                                    "highs": highs_result()}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result()})])
            report, _ = check(fx, p)
        self.assertEqual(kinds(report.failures), {"violation-too-large": 1})

    def test_no_reference_for_an_optimal_claim(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [flat_record("a"), flat_record("b")])
            report, _ = check(fx, p)   # flat files carry no external solver
        self.assertEqual(kinds(report.failures), {"no-reference": 2})

    def test_duplicate_instance(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [flat_record("a"), flat_record("a"),
                                     flat_record("b")])
            report, _ = check(fx, p, reference={"a": 1.0, "b": 1.0}, ref_source={})
        self.assertEqual(kinds(report.failures).get("duplicate-instance"), 1)

    def test_missing_instance_in_a_file_that_claims_the_suite(self) -> None:
        # 2 of 3 expected is above the 90%... no: it is below, so build a
        # bigger suite where one absence still leaves a full-suite claim.
        with Fixture(instances=tuple(f"i{n}" for n in range(20))) as fx:
            records = [flat_record(f"i{n}") for n in range(19)]   # 19 of 20
            p = fx.write("r.jsonl", records)
            report, _ = check(fx, p, reference={f"i{n}": 1.0 for n in range(20)},
                              ref_source={})
        self.assertEqual(kinds(report.failures).get("missing-instance"), 1)
        self.assertNotIn("partial-file", kinds(report.failures))

    def test_a_partial_smoke_file_is_one_note_not_many_failures(self) -> None:
        with Fixture(instances=tuple(f"i{n}" for n in range(20))) as fx:
            p = fx.write("r.jsonl", [flat_record("i0"), flat_record("i1")])
            report, _ = check(fx, p, reference={f"i{n}": 1.0 for n in range(20)},
                              ref_source={})
        counts = kinds(report.failures)
        self.assertEqual(counts.get("partial-file"), 1)
        self.assertNotIn("missing-instance", counts)

    def test_unexpected_instance(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [flat_record("a"), flat_record("b"),
                                     flat_record("ghost")])
            report, _ = check(fx, p, reference={"a": 1.0, "b": 1.0, "ghost": 1.0},
                              ref_source={})
        self.assertEqual(kinds(report.failures).get("unexpected-instance"), 1)

    def test_coverage_can_be_switched_off(self) -> None:
        with Fixture(instances=tuple(f"i{n}" for n in range(20))) as fx:
            p = fx.write("r.jsonl", [flat_record("i0")])
            report, _ = check(fx, p, reference={"i0": 1.0}, ref_source={},
                              coverage=False)
        self.assertEqual(report.failures, [])

    def test_external_solver_results_are_not_held_to_sor_rules(self) -> None:
        # HiGHS never reports a proof level; requiring one would flag every
        # reference run in every file.
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"highs": highs_result(1.0)}),
                nested_record("b", {"highs": highs_result(1.0)})])
            report, _ = check(fx, p)
        self.assertEqual(report.failures, [])


class CliTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = v.main(argv)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        return code, out.getvalue(), err.getvalue()

    def test_clean_file_exits_zero(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(), "highs": highs_result(1.0)}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result(1.0)})])
            code, out, _ = self._run([str(p), "--suite-dir", str(fx.suite)])
        self.assertEqual(code, 0)
        self.assertIn("0 finding(s)", out)

    def test_bad_file_exits_one_and_lists_the_finding(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(proof="BoundOnly"),
                                    "highs": highs_result(1.0)}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result(1.0)})])
            code, out, _ = self._run([str(p), "--suite-dir", str(fx.suite)])
        self.assertEqual(code, 1)
        self.assertIn("unproved-optimal", out)

    def test_machine_readable_output(self) -> None:
        with Fixture() as fx:
            p = fx.write("r.jsonl", [
                nested_record("a", {"SOR": sor_result(objective=2.0),
                                    "highs": highs_result(1.0)}),
                nested_record("b", {"SOR": sor_result(), "highs": highs_result(1.0)})])
            out_path = fx.root / "failures.json"
            code, _, _ = self._run([str(p), "--suite-dir", str(fx.suite),
                                    "--json", str(out_path), "--quiet"])
            payload = json.loads(out_path.read_text())
        self.assertEqual(code, 1)
        self.assertEqual(payload["tool"], "verify_netlib_results")
        self.assertEqual(payload["reference"], "stored")
        self.assertEqual(payload["counts"], {"objective-mismatch": 1})
        f = payload["failures"][0]
        self.assertEqual(f["instance"], "a")
        self.assertEqual(f["kind"], "objective-mismatch")
        self.assertEqual(f["observed"], 2.0)
        self.assertEqual(sorted(payload["suite_instances"]), ["a", "b"])

    def test_no_files_found(self) -> None:
        code, _, err = self._run(["/no/such/dir/nothing.jsonl", "--quiet"])
        # The file is named explicitly, so it is loaded and reported unreadable
        # rather than treated as "no files".
        self.assertEqual(code, 1)

    def test_negative_tolerance_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self._run(["x.jsonl", "--obj-rel-tol", "-1"])
        with self.assertRaises(SystemExit):
            self._run(["x.jsonl", "--violation-tol", "nan"])

    def test_live_highs_flag_fails_cleanly_when_unavailable(self) -> None:
        if v.live_highs_available():
            self.skipTest("highspy is usable here, so the failure path is moot")
        with Fixture() as fx:
            p = fx.write("r.jsonl", [nested_record("a", {"SOR": sor_result()})])
            code, _, err = self._run([str(p), "--suite-dir", str(fx.suite),
                                      "--live-highs"])
        self.assertEqual(code, 2)
        self.assertIn("highspy", err)


if __name__ == "__main__":
    unittest.main()
