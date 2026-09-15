#!/usr/bin/env python3
"""Unit tests for scripts/milp_latest_ablate.py (no solver required)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "milp_latest_ablate", ROOT / "scripts" / "milp_latest_ablate.py"
)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_parse_and_summary() -> None:
    sample = """\
status:            Optimal
objective:         -8.0000000000e+00
dual bound:        -8.0000000000e+00
mip gap:           0.000e+00
nodes:             3
warm_start_hits:   2 / 2 attempts (node LP dual warm; --basis-update product|ft)
termination:       tree exhausted
"""
    rec = MOD.parse_run(sample, wall_s=0.12, rc=0)
    assert rec["status"] == "Optimal"
    assert rec["proved"] is True
    assert rec["gap"] == 0.0
    assert rec["nodes"] == 3
    assert rec["warm_start_hits"] == 2

    rows = [
        {
            "policy": "latest",
            "proved": True,
            "gap": 0.0,
            "nodes": 3,
            "wall_s": 0.1,
        },
        {
            "policy": "latest",
            "proved": False,
            "gap": 0.2,
            "nodes": 10,
            "wall_s": 0.5,
        },
        {
            "policy": "classical",
            "proved": True,
            "gap": 0.0,
            "nodes": 5,
            "wall_s": 0.2,
        },
    ]
    summary = MOD.summarize(rows)
    assert summary["record"] == "summary"
    assert summary["policies"]["latest"]["proved"] == 1
    assert summary["policies"]["classical"]["proved"] == 1
    assert summary["policies"]["latest"]["n"] == 2


def test_ensure_tiny_suite() -> None:
    with tempfile.TemporaryDirectory() as td:
        paths = MOD.ensure_tiny_suite(Path(td))
        assert len(paths) == 3
        for p in paths:
            assert p.is_file()
            assert "ENDATA" in p.read_text()


def main() -> None:
    test_parse_and_summary()
    test_ensure_tiny_suite()
    print("ok")


if __name__ == "__main__":
    main()
