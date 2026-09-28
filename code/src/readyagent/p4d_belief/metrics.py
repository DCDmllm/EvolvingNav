"""Belief, persistence, calibration and offline-search metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            result += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    return float(result)


def binary_ece(probability: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = (probability > low) & (probability <= high)
        if selected.any():
            result += selected.mean() * abs(probability[selected].mean() - target[selected].mean())
    return float(result)


def evaluate_probabilities(
    probabilities: np.ndarray,
    targets: np.ndarray,
    last_indices: np.ndarray,
    region_instance_ids: tuple[str, ...],
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    target_probability = probabilities[np.arange(len(targets)), targets].clip(min=1e-12)
    ordering = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.empty(len(targets), dtype=np.float64)
    reciprocal_ranks = np.empty(len(targets), dtype=np.float64)
    success_at_1 = np.empty(len(targets), dtype=np.float64)
    success_at_3 = np.empty(len(targets), dtype=np.float64)
    success_at_5 = np.empty(len(targets), dtype=np.float64)
    room_checks = np.empty(len(targets), dtype=np.int64)
    for row, order in enumerate(ordering):
        score = probabilities[row, targets[row]]
        greater = int(np.sum(probabilities[row] > score + 1e-12))
        tied = int(np.sum(np.isclose(probabilities[row], score, rtol=0.0, atol=1e-12)))
        first_rank = greater + 1
        last_rank = greater + tied
        ranks[row] = (first_rank + last_rank) / 2.0
        reciprocal_ranks[row] = np.mean(1.0 / np.arange(first_rank, last_rank + 1))
        for cutoff, destination in (
            (1, success_at_1), (3, success_at_3), (5, success_at_5)
        ):
            destination[row] = np.clip((cutoff - greater) / max(tied, 1), 0.0, 1.0)
        # Room checks use a documented, stable catalog-order tiebreak because an
        # exact expected number of unique rooms under tied permutations is not
        # represented by the packed benchmark.
        rank = int(np.flatnonzero(order == targets[row])[0]) + 1
        rooms: set[str] = set()
        for candidate in order[:rank]:
            rooms.add(region_instance_ids[int(candidate)])
        room_checks[row] = len(rooms)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    persist_target = targets == last_indices
    persist_probability = probabilities[np.arange(len(targets)), last_indices]
    predicted_persist = persist_probability >= 0.5
    persistence_auc = (
        float(roc_auc_score(persist_target, persist_probability))
        if len(np.unique(persist_target)) == 2
        else float("nan")
    )
    return {
        "records": int(len(targets)),
        "top1": float(success_at_1.mean()),
        "top3": float(success_at_3.mean()),
        "success_at_5": float(success_at_5.mean()),
        "mrr": float(reciprocal_ranks.mean()),
        "nll": float(-np.log(target_probability).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece": expected_calibration_error(probabilities, targets),
        "mean_target_rank": float(ranks.mean()),
        "mean_receptacles_checked": float(ranks.mean()),
        "mean_rooms_checked": float(room_checks.mean()),
        "persistence_accuracy": float(np.mean(predicted_persist == persist_target)),
        "persistence_f1": float(f1_score(persist_target, predicted_persist, zero_division=0)),
        "persistence_auroc": persistence_auc,
        "persistence_brier": float(np.square(persist_probability - persist_target).mean()),
        "persistence_ece": binary_ece(persist_probability, persist_target),
        "mean_persistence_probability": float(persist_probability.mean()),
    }


def aggregate_seed_metrics(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_results:
        return {}
    aggregate: dict[str, Any] = {"seeds": len(seed_results)}
    for subset in seed_results[0]:
        rows = [result[subset] for result in seed_results if subset in result]
        aggregate[subset] = {}
        for key, value in rows[0].items():
            if isinstance(value, (int, float)) and key != "records":
                values = np.asarray([row[key] for row in rows], dtype=np.float64)
                finite = values[np.isfinite(values)]
                aggregate[subset][key] = {
                    "mean": float(finite.mean()) if len(finite) else float("nan"),
                    "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                }
            elif key == "records":
                aggregate[subset][key] = value
    return aggregate
