#!/usr/bin/env python3
"""Unit tests for scripts/freeze_manifest.py and scripts/ablate.py.

Register with sor_add_python_test(test_freeze_manifest) in tests/CMakeLists.txt.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

# tests/ has its own CMakeLists.txt, so the marker must be one only the
# repository root carries.
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "scripts" / "compare.py").exists())


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


freeze = _load("sor_freeze", ROOT / "scripts" / "freeze_manifest.py")
ablate = _load("sor_ablate", ROOT / "scripts" / "ablate.py")


class ChecksumTests(unittest.TestCase):
    def test_sha256_matches_hashlib(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.mps"
            f.write_bytes(b"NAME T\nROWS\n N OBJ\nENDATA\n")
            self.assertEqual(freeze.sha256_of(f),
                             hashlib.sha256(f.read_bytes()).hexdigest())

    def test_sha256_is_content_not_name(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.mps", Path(d) / "b.mps"
            a.write_bytes(b"same")
            b.write_bytes(b"same")
            self.assertEqual(freeze.sha256_of(a), freeze.sha256_of(b))


class ReferenceTests(unittest.TestCase):
    def write(self, d, records):
        p = Path(d) / "run.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        return p

    def test_reads_only_the_oracle_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.write(d, [
                {"record": "environment"},
                {"record": "run", "solver": "sor:simplex", "instance": "x.mps",
                 "status": "Optimal", "objective": 1.0},
                {"record": "run", "solver": "highs", "instance": "x.mps",
                 "status": "Optimal", "objective": 2.0},
            ])
            ref = freeze.load_reference(p)
            # The reference is the ORACLE's answer. Taking SOR's own objective
            # would make the manifest agree with SOR by construction.
            self.assertEqual(ref["x.mps"]["objective"], 2.0)

    def test_missing_file_is_not_an_error(self):
        self.assertEqual(freeze.load_reference(Path("/nonexistent.jsonl")), {})
        self.assertEqual(freeze.load_reference(None), {})

    def test_malformed_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "run.jsonl"
            p.write_text('not json\n' + json.dumps(
                {"record": "run", "solver": "highs", "instance": "y.mps",
                 "status": "Optimal", "objective": 3.0}) + "\n")
            self.assertEqual(freeze.load_reference(p)["y.mps"]["objective"], 3.0)


class ScoringEligibilityTests(unittest.TestCase):
    def test_frozen_netlib_manifest_is_complete(self):
        path = ROOT / "benchmarks" / "manifests" / "netlib-v1.json"
        if not path.exists():
            self.skipTest("netlib manifest not frozen in this tree")
        doc = json.loads(path.read_text())
        self.assertEqual(doc["suite"], "netlib")
        self.assertEqual(doc["model_count"], 93)
        # Every model must carry a checksum and a shape, or the manifest does
        # not actually pin the corpus a claim was measured on.
        for e in doc["models"]:
            self.assertEqual(len(e["sha256"]), 64, e["name"])
            self.assertTrue(e["rows"] and e["cols"] and e["nnz"], e["name"])
            # Scoring eligibility must imply a reference answer exists.
            if e["scoring_eligible"]:
                self.assertIsNotNone(e["reference_objective"], e["name"])
                self.assertIsNotNone(e["reference_status"], e["name"])

    def test_checksums_are_unique_per_model(self):
        path = ROOT / "benchmarks" / "manifests" / "netlib-v1.json"
        if not path.exists():
            self.skipTest("netlib manifest not frozen in this tree")
        doc = json.loads(path.read_text())
        sums = [e["sha256"] for e in doc["models"]]
        self.assertEqual(len(sums), len(set(sums)))


class AblationScoringTests(unittest.TestCase):
    """ablate.py must DELEGATE to compare.py's public gate, not re-derive it.

    It previously computed its own geometric mean of per-model ratios, which is
    a different statistic from the public metric and disagreed with it by 50%
    on Netlib. These tests pin the delegation and the key names the table reads.
    """

    def run_file(self, d, rows):
        p = Path(d) / "arm.jsonl"
        recs = []
        for inst, sor_s, highs_s, proof in rows:
            recs.append({"record": "aggregate", "solver": "sor:simplex",
                         "instance": inst,
                         "status": "Optimal" if proof else "Feasible",
                         "proof": proof, "median_s": sor_s, "seconds": sor_s,
                         "wall_s": sor_s, "objective": 1.0,
                         "iterations": 10, "noisy": False, "mad_s": 0.0})
            recs.append({"record": "aggregate", "solver": "highs",
                         "instance": inst, "status": "Optimal",
                         "median_s": highs_s, "seconds": highs_s,
                         "wall_s": highs_s, "objective": 1.0,
                         "iterations": 10, "noisy": False, "mad_s": 0.0})
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        return p

    def test_reports_the_ratio_of_sgms(self):
        import math
        with tempfile.TemporaryDirectory() as d:
            p = self.run_file(d, [("a.mps", 1.0, 2.0, "ProvedOptimalFP"),
                                  ("b.mps", 4.0, 2.0, "ProvedOptimalFP")])
            r = ablate.score(p, time_limit=60.0)
            sgm = lambda v: math.exp(sum(math.log(x + 1.0) for x in v) / len(v)) - 1.0
            self.assertAlmostEqual(r["sgm_ratio"], sgm([1.0, 4.0]) / sgm([2.0, 2.0]),
                                   places=10)
            # The geomean of per-model ratios here is exactly 1.0; the true
            # metric is not. If these ever coincide the test is not testing.
            self.assertNotAlmostEqual(r["sgm_ratio"], 1.0, places=3)
            self.assertEqual(r["models"], 2)
            self.assertEqual(r["proved"], 2)

    def test_uncertified_model_is_par2_charged_not_dropped(self):
        # The failure mode this guards: pilot87 losing its proof while still
        # costing ~13 s. Scoring only the proved models drops it entirely and
        # reports a number the run did not earn.
        with tempfile.TemporaryDirectory() as d:
            p = self.run_file(d, [("ok.mps", 1.0, 1.0, "ProvedOptimalFP"),
                                  ("unproved.mps", 0.001, 1.0, None)])
            r = ablate.score(p, time_limit=60.0)
            self.assertEqual(r["models"], 2)      # not dropped
            self.assertEqual(r["par2"], 1)
            self.assertEqual(r["proved"], 1)
            self.assertEqual(r["wins"], 0)        # charged 2 x 60 s
            self.assertGreater(r["sgm_ratio"], 1.0)
            self.assertFalse(r["passed"])

    def test_gate_verdict_is_surfaced(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_file(d, [(f"m{i}.mps", 0.4, 1.0, "ProvedOptimalFP")
                                  for i in range(10)])
            r = ablate.score(p, time_limit=60.0)
            self.assertLessEqual(r["sgm_ratio"], 0.95)
            self.assertGreaterEqual(r["win_rate"], 60.0)
            self.assertTrue(r["passed"], r["failures"])

    def test_borrowed_reference_fills_a_failed_oracle_row(self):
        with tempfile.TemporaryDirectory() as d:
            good = self.run_file(d, [("a.mps", 1.0, 2.0, "ProvedOptimalFP")])
            ref = ablate.load_reference_results(good)
            self.assertIn("a.mps", ref)
            # An arm whose HiGHS row is unavailable must still be scorable.
            p = Path(d) / "arm2.jsonl"
            p.write_text(json.dumps(
                {"record": "aggregate", "solver": "sor:simplex",
                 "instance": "a.mps", "status": "Optimal",
                 "proof": "ProvedOptimalFP", "median_s": 1.0, "seconds": 1.0,
                 "wall_s": 1.0, "objective": 1.0, "iterations": 10,
                 "noisy": False, "mad_s": 0.0}) + "\n")
            r = ablate.score(p, reference=ref, time_limit=60.0)
            self.assertEqual(r["models"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
