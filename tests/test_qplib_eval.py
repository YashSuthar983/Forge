#!/usr/bin/env python3
"""scripts/qplib_eval.py -- the independent QPLIB evaluator -- on the
hand-written QCQP in tests/data (values worked by hand in test_qcqp.cpp)."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qplib_eval", ROOT / "scripts" / "qplib_eval.py")
ev = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ev
SPEC.loader.exec_module(ev)
DATA = ROOT / "tests" / "data"


class QcqpTiny(unittest.TestCase):
    def setUp(self):
        self.p = ev.read(str(DATA / "qcqp_tiny.qplib"))

    def test_structure(self):
        p = self.p
        self.assertEqual((p["n"], p["m"], p["kind"]), (4, 3, "QGQ"))
        self.assertEqual(len(p["qc"]), 8)
        self.assertEqual(p["typ"], [0, 2, 1, 0])
        self.assertEqual(p["vnames"], {3: "w4"})
        self.assertEqual(p["leftover"], 0)

    def test_published_point(self):
        x, objvar = ev.read_point(self.p, str(DATA / "qcqp_tiny.sol"))
        self.assertEqual(x, [0.5, 1.0, 2.0, -1.0])
        self.assertEqual(objvar, 4.5)
        obj, viol = ev.evaluate(self.p, x)
        self.assertEqual(obj, 4.5)
        self.assertEqual(viol["qrows"], 0.5)   # row 3: -4 against <= -4.5
        self.assertEqual(viol["rows"], 0.5)
        self.assertEqual(viol["bounds"], 0.0)

    def test_binary_is_zero_one_whatever_the_listed_bounds(self):
        # var 2 is binary with listed bounds [0, inf): 2.0 violates by 1.
        _, viol = ev.evaluate(self.p, [0.5, 2.0, 2.0, -1.0])
        self.assertEqual(viol["bounds"], 1.0)

    def test_integrality(self):
        _, viol = ev.evaluate(self.p, [0.5, 1.0, 2.25, -1.0])
        self.assertEqual(viol["integrality"], 0.25)


if __name__ == "__main__":
    unittest.main()
