#!/usr/bin/env python3
"""Product-boundary tests for the typed Python process API."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sor_api", ROOT / "bindings/python/sor_api.py")
assert SPEC is not None and SPEC.loader is not None
api = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = api
SPEC.loader.exec_module(api)


class PythonApiTests(unittest.TestCase):
    def test_typed_solve_has_isolated_output_and_independent_check(self) -> None:
        first = api.solve("testlp", process_timeout=10.0)
        second = api.solve("testlp", process_timeout=10.0)
        self.assertIsInstance(first, api.SolveResult)
        self.assertTrue(first.ok, first.stderr)
        self.assertEqual(first.status, "Optimal")
        self.assertIsInstance(first.objective, float)
        self.assertNotEqual(first.solution, second.solution)
        self.assertTrue(first.solution.is_file())
        checked = api.check("testlp", first.solution, process_timeout=10.0)
        self.assertIsInstance(checked, api.CheckResult)
        self.assertTrue(checked.verified, checked.stderr + checked.stdout)
        # Existing dict-style clients remain source-compatible.
        self.assertEqual(first["status"], "Optimal")
        self.assertEqual(first["solution"], str(first.solution))

    def test_timeout_kills_child(self) -> None:
        result = api._run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            0.05, None)
        self.assertTrue(result.timed_out)

    def test_pre_cancelled_event_stops_child(self) -> None:
        cancel = threading.Event()
        cancel.set()
        result = api._run_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            5.0, cancel)
        self.assertTrue(result.cancelled)


if __name__ == "__main__":
    unittest.main()
