#!/usr/bin/env python3
"""Freeze and fit an unpromoted serial-LP Auto routing candidate.

The split is deliberately a separate operation that consumes features only.
Training then refuses an outcomes file containing any frozen holdout model, so
holdout outcomes cannot accidentally influence fitting.  The emitted artifact
is evidence for review; this tool never edits the compiled routing table and
never marks a candidate promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ALLOWED_SPLITS = ("60/25/15", "70/20/10", "80/15/5")
FEATURE_NAMES = (
    "rows",
    "cols",
    "nonzeros",
    "density",
    "row_degree_mean",
    "row_degree_max",
    "col_degree_mean",
    "col_degree_max",
    "fixed_variables",
    "boxed_variables",
    "one_sided_variables",
    "free_variables",
    "equality_rows",
    "ranged_rows",
    "one_sided_rows",
    "coefficient_spread",
    "objective_density",
)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: object required")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical(value))


def _feature_row(row: dict[str, Any]) -> dict[str, Any]:
    model_id = str(row.get("id", ""))
    if not model_id:
        raise ValueError("every feature row needs a nonempty id")
    features: dict[str, float] = {}
    for name in FEATURE_NAMES:
        if name not in row:
            continue
        value = float(row[name])
        if not math.isfinite(value):
            raise ValueError(f"{model_id}: feature {name} is not finite")
        features[name] = value
    for required in ("rows", "cols", "nonzeros"):
        if required not in features:
            raise ValueError(f"{model_id}: missing pre-solve feature {required}")
    return {"id": model_id, **features}


def freeze(features: Iterable[dict[str, Any]], seed: str = "20260910") -> dict[str, Any]:
    rows = sorted((_feature_row(row) for row in features), key=lambda r: r["id"])
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("feature ids must be unique")
    if len(rows) < 5:
        raise ValueError("at least five models are required for a 20% holdout")

    # Five rank-quantile strata retain the model-size distribution.  Hamilton
    # allocation makes the total exactly round(20%) even for uneven strata.
    by_size = sorted(
        rows,
        key=lambda r: (r["rows"] + r["cols"], r["nonzeros"], r["id"]),
    )
    strata: list[list[dict[str, Any]]] = [[] for _ in range(5)]
    for rank, row in enumerate(by_size):
        strata[min(4, rank * 5 // len(by_size))].append(row)
    target = max(1, round(0.20 * len(rows)))
    exact = [0.20 * len(stratum) for stratum in strata]
    quotas = [math.floor(value) for value in exact]
    remaining = target - sum(quotas)
    order = sorted(
        range(5), key=lambda i: (-(exact[i] - quotas[i]), i)
    )
    for index in order[:remaining]:
        quotas[index] += 1

    holdout: list[str] = []
    stratum_records: list[dict[str, Any]] = []
    for index, stratum in enumerate(strata):
        ranked = sorted(
            stratum,
            key=lambda r: (
                hashlib.sha256(f"{seed}\0{r['id']}".encode()).hexdigest(),
                r["id"],
            ),
        )
        selected = sorted(row["id"] for row in ranked[: quotas[index]])
        holdout.extend(selected)
        stratum_records.append(
            {"index": index, "model_count": len(stratum), "holdout_ids": selected}
        )

    holdout = sorted(holdout)
    holdout_set = set(holdout)
    training = sorted(model_id for model_id in ids if model_id not in holdout_set)
    feature_payload = sorted(rows, key=lambda r: r["id"])
    return {
        "schema": "sor-lp-auto-freeze-v1",
        "seed": seed,
        "holdout_fraction": 0.20,
        "feature_manifest_hash": _hash(feature_payload),
        "training_manifest_hash": _hash(training),
        "holdout_manifest_hash": _hash(holdout),
        "training_ids": training,
        "holdout_ids": holdout,
        "strata": stratum_records,
    }


def _score(run: dict[str, Any], limit_s: float) -> float:
    wall = float(run.get("wall_s", math.inf))
    correct = bool(run.get("correct", False))
    timed_out = bool(run.get("timed_out", False))
    if not correct or timed_out or not math.isfinite(wall) or wall < 0.0:
        return 2.0 * limit_s
    return min(wall, 2.0 * limit_s)


def _route_run(row: dict[str, Any], split: str) -> dict[str, Any]:
    routes = row.get("auto")
    if not isinstance(routes, dict) or split not in routes:
        raise ValueError(f"{row.get('id')}: missing auto result for split {split}")
    run = routes[split]
    if not isinstance(run, dict):
        raise ValueError(f"{row.get('id')}: invalid auto result for split {split}")
    return run


@dataclass(frozen=True)
class Sample:
    model_id: str
    features: dict[str, float]
    simplex: dict[str, Any]
    auto: dict[str, dict[str, Any]]


def _leaf(samples: list[Sample], split: str, limit_s: float) -> dict[str, Any]:
    simplex_loss = sum(math.log1p(_score(s.simplex, limit_s)) for s in samples)
    auto_runs = [_route_run({"id": s.model_id, "auto": s.auto}, split) for s in samples]
    zero_loss = all(
        not bool(s.simplex.get("correct", False)) or bool(run.get("correct", False))
        for s, run in zip(samples, auto_runs)
    )
    tail_green = all(
        _score(run, limit_s) <= 2.0 * max(_score(s.simplex, limit_s), 1e-12)
        for s, run in zip(samples, auto_runs)
    )
    auto_loss = sum(math.log1p(_score(run, limit_s)) for run in auto_runs)
    # Equality deliberately routes to simplex.
    use_auto = zero_loss and tail_green and auto_loss < simplex_loss - 1e-15
    return {
        "kind": "leaf",
        "route": "hpr-crossover-simplex" if use_auto else "simplex",
        "sample_count": len(samples),
        "loss": auto_loss if use_auto else simplex_loss,
        "rationale": (
            "lower PAR-2 shifted-log loss with zero correctness loss and 2x tail green"
            if use_auto
            else "simplex tie/default or Auto correctness/tail gate failed"
        ),
    }


def _fit_tree(
    samples: list[Sample], split: str, limit_s: float, depth: int = 0
) -> dict[str, Any]:
    incumbent = _leaf(samples, split, limit_s)
    if depth >= 4 or len(samples) < 20:
        return incumbent
    best: tuple[float, str, float, list[Sample], list[Sample], dict[str, Any], dict[str, Any]] | None = None
    feature_names = sorted(set.intersection(*(set(s.features) for s in samples)))
    for feature in feature_names:
        values = sorted(set(s.features[feature] for s in samples))
        for lo, hi in zip(values, values[1:]):
            threshold = (lo + hi) / 2.0
            left = [s for s in samples if s.features[feature] <= threshold]
            right = [s for s in samples if s.features[feature] > threshold]
            if len(left) < 10 or len(right) < 10:
                continue
            left_leaf = _leaf(left, split, limit_s)
            right_leaf = _leaf(right, split, limit_s)
            loss = left_leaf["loss"] + right_leaf["loss"]
            key = (loss, feature, threshold)
            if best is None or key < best[:3]:
                best = (loss, feature, threshold, left, right, left_leaf, right_leaf)
    if best is None or best[0] >= incumbent["loss"] - 1e-15:
        return incumbent
    _, feature, threshold, left, right, _, _ = best
    left_tree = _fit_tree(left, split, limit_s, depth + 1)
    right_tree = _fit_tree(right, split, limit_s, depth + 1)
    return {
        "kind": "branch",
        "feature": feature,
        "threshold": threshold,
        "sample_count": len(samples),
        "loss": left_tree["loss"] + right_tree["loss"],
        "left": left_tree,
        "right": right_tree,
    }


def _route(tree: dict[str, Any], features: dict[str, float]) -> str:
    while tree["kind"] == "branch":
        tree = tree["left"] if features[tree["feature"]] <= tree["threshold"] else tree["right"]
    return str(tree["route"])


def _metrics(samples: list[Sample], tree: dict[str, Any], split: str,
             limit_s: float) -> dict[str, Any]:
    scores: list[float] = []
    zero_loss = True
    tail_green = True
    auto_count = 0
    for sample in samples:
        use_auto = _route(tree, sample.features) != "simplex"
        run = _route_run({"id": sample.model_id, "auto": sample.auto}, split) if use_auto else sample.simplex
        if use_auto:
            auto_count += 1
        zero_loss &= not bool(sample.simplex.get("correct", False)) or bool(run.get("correct", False))
        tail_green &= _score(run, limit_s) <= 2.0 * max(_score(sample.simplex, limit_s), 1e-12)
        scores.append(_score(run, limit_s))
    return {
        "shifted_sgm_s": math.exp(sum(math.log1p(v) for v in scores) / len(scores)) - 1.0,
        "zero_correctness_loss": zero_loss,
        "tail_2x_green": tail_green,
        "auto_route_count": auto_count,
        "model_count": len(samples),
    }


def train(features: Iterable[dict[str, Any]], outcomes: Iterable[dict[str, Any]],
          frozen: dict[str, Any], limit_s: float) -> dict[str, Any]:
    if frozen.get("schema") != "sor-lp-auto-freeze-v1":
        raise ValueError("unsupported freeze schema")
    feature_rows = {_feature_row(row)["id"]: _feature_row(row) for row in features}
    if _hash(sorted(feature_rows.values(), key=lambda r: r["id"])) != frozen["feature_manifest_hash"]:
        raise ValueError("feature manifest differs from frozen split")
    training_ids = set(frozen["training_ids"])
    holdout_ids = set(frozen["holdout_ids"])
    if len(training_ids) < 10:
        raise ValueError("at least ten training models are required")
    outcome_rows: dict[str, dict[str, Any]] = {}
    for row in outcomes:
        model_id = str(row.get("id", ""))
        if model_id in holdout_ids:
            raise ValueError("training outcomes contain a frozen holdout model")
        if model_id not in training_ids:
            raise ValueError(f"outcome for non-training model {model_id}")
        if model_id in outcome_rows:
            raise ValueError(f"duplicate outcome {model_id}")
        outcome_rows[model_id] = row
    if set(outcome_rows) != training_ids:
        raise ValueError("outcomes must cover exactly the frozen training set")
    if not math.isfinite(limit_s) or limit_s <= 0.0:
        raise ValueError("time limit must be positive and finite")

    samples: list[Sample] = []
    for model_id in sorted(training_ids):
        feature_row = feature_rows[model_id]
        outcome = outcome_rows[model_id]
        simplex = outcome.get("simplex")
        auto = outcome.get("auto")
        if not isinstance(simplex, dict) or not isinstance(auto, dict):
            raise ValueError(f"{model_id}: simplex and auto results required")
        samples.append(Sample(
            model_id,
            {k: float(v) for k, v in feature_row.items() if k != "id"},
            simplex,
            auto,
        ))

    candidates: list[tuple[float, int, str, dict[str, Any], dict[str, Any]]] = []
    for split in ALLOWED_SPLITS:
        tree = _fit_tree(samples, split, limit_s)
        metrics = _metrics(samples, tree, split, limit_s)
        if metrics["zero_correctness_loss"] and metrics["tail_2x_green"]:
            # If objective values tie, prefer the split with more simplex
            # reserve (15 > 10 > 5), then lexical determinism.
            simplex_reserve = int(split.split("/")[2])
            candidates.append((metrics["shifted_sgm_s"], -simplex_reserve,
                               split, tree, metrics))
    if not candidates:
        raise ValueError("no budget split satisfies correctness and 2x tail gates")
    _, _, split, tree, metrics = min(candidates, key=lambda item: item[:3])
    body = {
        "schema": "sor-lp-auto-candidate-v1",
        "promoted": False,
        "budget_split": split,
        "allowed_budget_splits": list(ALLOWED_SPLITS),
        "training_manifest_hash": frozen["training_manifest_hash"],
        "holdout_manifest_hash": frozen["holdout_manifest_hash"],
        "feature_manifest_hash": frozen["feature_manifest_hash"],
        "time_limit_s": limit_s,
        "training_metrics": metrics,
        "tree": tree,
        "route_rationale": "deterministic depth<=4/min-leaf-10 PAR-2 shifted-SGM fit; holdout not evaluated",
    }
    body["rule_table_version"] = "candidate-unpromoted-" + _hash(body)[:16]
    return body


def evaluate_holdout(
    features: Iterable[dict[str, Any]], outcomes: Iterable[dict[str, Any]],
    frozen: dict[str, Any], candidate: dict[str, Any], limit_s: float
) -> dict[str, Any]:
    feature_rows = {_feature_row(row)["id"]: _feature_row(row) for row in features}
    if _hash(sorted(feature_rows.values(), key=lambda r: r["id"])) != frozen["feature_manifest_hash"]:
        raise ValueError("feature manifest differs from frozen split")
    for key in ("training_manifest_hash", "holdout_manifest_hash", "feature_manifest_hash"):
        if candidate.get(key) != frozen.get(key):
            raise ValueError(f"candidate {key} differs from frozen split")
    split = str(candidate.get("budget_split", ""))
    if split not in ALLOWED_SPLITS:
        raise ValueError("candidate has an invalid budget split")
    holdout_ids = set(frozen["holdout_ids"])
    rows: dict[str, dict[str, Any]] = {}
    for row in outcomes:
        model_id = str(row.get("id", ""))
        if model_id not in holdout_ids:
            raise ValueError(f"holdout outcomes contain non-holdout model {model_id}")
        if model_id in rows:
            raise ValueError(f"duplicate holdout outcome {model_id}")
        rows[model_id] = row
    if set(rows) != holdout_ids:
        raise ValueError("outcomes must cover exactly the frozen holdout")

    results: list[dict[str, Any]] = []
    candidate_scores: list[float] = []
    simplex_scores: list[float] = []
    zero_loss = True
    tail_green = True
    for model_id in sorted(holdout_ids):
        row = rows[model_id]
        simplex = row.get("simplex")
        if not isinstance(simplex, dict):
            raise ValueError(f"{model_id}: simplex result required")
        features_for_model = {
            key: float(value) for key, value in feature_rows[model_id].items()
            if key != "id"
        }
        selected_route = _route(candidate["tree"], features_for_model)
        selected = simplex
        if selected_route != "simplex":
            selected = _route_run(row, split)
        selected_score = _score(selected, limit_s)
        simplex_score = _score(simplex, limit_s)
        selected_correct = bool(selected.get("correct", False))
        simplex_correct = bool(simplex.get("correct", False))
        zero_loss &= not simplex_correct or selected_correct
        tail_green &= selected_score <= 2.0 * max(simplex_score, 1e-12)
        candidate_scores.append(selected_score)
        simplex_scores.append(simplex_score)
        results.append({
            "id": model_id,
            "selected_route": selected_route,
            "selected_result": selected,
            "simplex_result": simplex,
            "selected_par2_s": selected_score,
            "simplex_par2_s": simplex_score,
        })
    candidate_sgm = math.exp(
        sum(math.log1p(value) for value in candidate_scores) / len(candidate_scores)
    ) - 1.0
    simplex_sgm = math.exp(
        sum(math.log1p(value) for value in simplex_scores) / len(simplex_scores)
    ) - 1.0
    return {
        "schema": "sor-lp-auto-holdout-v1",
        "rule_table_version": candidate["rule_table_version"],
        "training_manifest_hash": frozen["training_manifest_hash"],
        "holdout_manifest_hash": frozen["holdout_manifest_hash"],
        "budget_split": split,
        "model_count": len(results),
        "candidate_shifted_sgm_s": candidate_sgm,
        "simplex_shifted_sgm_s": simplex_sgm,
        "zero_correctness_loss": zero_loss,
        "tail_2x_green": tail_green,
        "beats_simplex": candidate_sgm < simplex_sgm,
        # Other promotion gates (Netlib, sanitizers, claim protocol) live
        # outside this artifact, so this is intentionally not a promotion bit.
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--features", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.add_argument("--seed", default="20260910")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--features", type=Path, required=True)
    train_parser.add_argument("--outcomes", type=Path, required=True)
    train_parser.add_argument("--freeze", type=Path, required=True)
    train_parser.add_argument("--time-limit", type=float, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = sub.add_parser("evaluate-holdout")
    evaluate_parser.add_argument("--features", type=Path, required=True)
    evaluate_parser.add_argument("--outcomes", type=Path, required=True)
    evaluate_parser.add_argument("--freeze", type=Path, required=True)
    evaluate_parser.add_argument("--candidate", type=Path, required=True)
    evaluate_parser.add_argument("--time-limit", type=float, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        _write_json(args.output, freeze(_read_jsonl(args.features), args.seed))
    elif args.command == "train":
        frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
        artifact = train(_read_jsonl(args.features), _read_jsonl(args.outcomes),
                         frozen, args.time_limit)
        _write_json(args.output, artifact)
    else:
        frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        report = evaluate_holdout(
            _read_jsonl(args.features), _read_jsonl(args.outcomes), frozen,
            candidate, args.time_limit)
        _write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
