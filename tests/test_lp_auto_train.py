#!/usr/bin/env python3
"""Determinism, leakage and promotion-gate tests for LP Auto fitting."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lp_auto_train", ROOT / "tools" / "lp_auto_train.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def features() -> list[dict[str, float | str]]:
    return [
        {
            "id": f"model-{index:03d}",
            "rows": float(index + 1),
            "cols": float(2 * index + 3),
            "nonzeros": float(5 * index + 7),
            "density": 0.01 + index / 10000.0,
        }
        for index in range(100)
    ]


def outcome(model_id: str) -> dict[str, object]:
    index = int(model_id.rsplit("-", 1)[1])
    simplex = {"correct": True, "wall_s": 10.0, "timed_out": False}
    auto_wall = 1.0 if index < 50 else 12.0
    routes = {
        split: {"correct": True, "wall_s": auto_wall, "timed_out": False}
        for split in MODULE.ALLOWED_SPLITS
    }
    return {"id": model_id, "simplex": simplex, "auto": routes}


def check_tree(tree: dict[str, object], depth: int = 0) -> None:
    assert depth <= 4
    assert int(tree["sample_count"]) >= 10
    if tree["kind"] == "branch":
        check_tree(tree["left"], depth + 1)  # type: ignore[arg-type]
        check_tree(tree["right"], depth + 1)  # type: ignore[arg-type]


def main() -> None:
    rows = features()
    frozen = MODULE.freeze(rows, "fixed-seed")
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    assert frozen == MODULE.freeze(shuffled, "fixed-seed")
    assert len(frozen["holdout_ids"]) == 20
    assert len(frozen["training_ids"]) == 80
    assert not (set(frozen["holdout_ids"]) & set(frozen["training_ids"]))
    assert all(len(stratum["holdout_ids"]) == 4 for stratum in frozen["strata"])

    training_outcomes = [outcome(model_id) for model_id in frozen["training_ids"]]
    artifact = MODULE.train(rows, training_outcomes, frozen, 30.0)
    artifact_again = MODULE.train(shuffled, list(reversed(training_outcomes)), frozen, 30.0)
    assert artifact == artifact_again
    assert artifact["promoted"] is False
    assert artifact["budget_split"] == "60/25/15"
    assert artifact["training_manifest_hash"] == frozen["training_manifest_hash"]
    assert artifact["holdout_manifest_hash"] == frozen["holdout_manifest_hash"]
    assert artifact["training_metrics"]["zero_correctness_loss"] is True
    assert artifact["training_metrics"]["tail_2x_green"] is True
    check_tree(artifact["tree"])

    # The useful small-model region is routed to Auto; the slower large-model
    # region remains simplex.  This also proves ties/defaults prefer simplex.
    assert MODULE._route(artifact["tree"], {  # pylint: disable=protected-access
        key: float(value) for key, value in rows[10].items() if key != "id"
    }) == "hpr-crossover-simplex"
    assert MODULE._route(artifact["tree"], {  # pylint: disable=protected-access
        key: float(value) for key, value in rows[90].items() if key != "id"
    }) == "simplex"

    holdout_outcomes = [outcome(model_id) for model_id in frozen["holdout_ids"]]
    holdout = MODULE.evaluate_holdout(
        rows, holdout_outcomes, frozen, artifact, 30.0
    )
    assert holdout["model_count"] == 20
    assert len(holdout["results"]) == 20
    assert holdout["zero_correctness_loss"] is True
    assert holdout["tail_2x_green"] is True
    assert holdout["training_manifest_hash"] == frozen["training_manifest_hash"]
    assert holdout["holdout_manifest_hash"] == frozen["holdout_manifest_hash"]

    # Fitting refuses even one holdout result rather than silently filtering
    # it.  Holdout evaluation must be an explicit later promotion step.
    leaked = training_outcomes + [outcome(frozen["holdout_ids"][0])]
    try:
        MODULE.train(rows, leaked, frozen, 30.0)
    except ValueError as exc:
        assert "holdout" in str(exc)
    else:
        raise AssertionError("holdout leakage was accepted")

    # A different feature corpus cannot reuse the frozen hashes.
    changed = json.loads(json.dumps(rows))
    changed[0]["nonzeros"] = float(changed[0]["nonzeros"]) + 1.0
    try:
        MODULE.train(changed, training_outcomes, frozen, 30.0)
    except ValueError as exc:
        assert "feature manifest" in str(exc)
    else:
        raise AssertionError("changed feature corpus was accepted")

    print("test_lp_auto_train: PASS")


if __name__ == "__main__":
    main()
