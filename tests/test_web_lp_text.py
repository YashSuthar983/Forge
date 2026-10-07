#!/usr/bin/env python3
"""The web console's LP-text -> MPS conversion must not round the model."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lp_text", ROOT / "web/lp_text.py")
assert SPEC is not None and SPEC.loader is not None
lp_text = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lp_text
SPEC.loader.exec_module(lp_text)


def mps_numbers(mps: str) -> list[float]:
    """Every numeric field of the COLUMNS, RHS and BOUNDS records."""
    out: list[float] = []
    section = ""
    for line in mps.splitlines():
        if line and not line[0].isspace():
            section = line.split()[0]
            continue
        if section not in {"COLUMNS", "RHS", "BOUNDS"} or "MARKER" in line:
            continue
        for token in line.split():
            try:
                out.append(float(token))
            except ValueError:
                pass
    return out


class WebLpTextTests(unittest.TestCase):
    def test_numbers_survive_conversion_exactly(self) -> None:
        # Ten significant digits used to turn 0.12345678901234 into
        # 0.123456789 and 1.0000000000001 into 1.
        text = """Minimize
  0.12345678901234 x + 1.0000000000001 y
Subject To
  c1: 0.3333333333333333 x + y >= 2.0000000000000004
Bounds
  x <= 7.123456789012345
  y >= -0.1000000000000001
End
"""
        mps, _ = lp_text.lp_text_to_mps(text)
        numbers = mps_numbers(mps)
        for value in (0.12345678901234, 1.0000000000001, 0.3333333333333333,
                      2.0000000000000004, 7.123456789012345, -0.1000000000000001):
            self.assertIn(value, numbers, mps)


if __name__ == "__main__":
    unittest.main()
