"""Leakage-safe classical baselines for current-state prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(counts: np.ndarray) -> np.ndarray:
    return counts / counts.sum(axis=-1, keepdims=True).clip(min=1e-12)


@dataclass
class ClassicalBaselines:
    state_count: int
    category_count: int
    smoothing: float = 0.25

    def fit(
        self,
        *,
        target_category: np.ndarray,
        target_state: np.ndarray,
        last_state: np.ndarray,
        weekday: np.ndarray,
        time_sin_cos: np.ndarray,
    ) -> None:
        self.object_counts = np.full(
            (self.category_count, self.state_count), self.smoothing, dtype=np.float64
        )
        self.markov_counts = np.full(
            (self.category_count, self.state_count, self.state_count),
            self.smoothing,
            dtype=np.float64,
        )
        self.time_counts = np.full(
            (self.category_count, 7, 4, self.state_count), self.smoothing, dtype=np.float64
        )
        hour = _time_quarter(time_sin_cos)
        np.add.at(self.object_counts, (target_category, target_state), 1.0)
        np.add.at(self.markov_counts, (target_category, last_state, target_state), 1.0)
        np.add.at(self.time_counts, (target_category, weekday, hour, target_state), 1.0)
        self.object_probabilities = _normalize(self.object_counts)
        self.markov_probabilities = _normalize(self.markov_counts)
        self.time_probabilities = _normalize(self.time_counts)

    def predict(
        self,
        method: str,
        *,
        target_category: np.ndarray,
        last_state: np.ndarray,
        weekday: np.ndarray,
        time_sin_cos: np.ndarray,
    ) -> np.ndarray:
        if method == "last_seen":
            probabilities = np.zeros((len(last_state), self.state_count), dtype=np.float64)
            probabilities[np.arange(len(last_state)), last_state] = 1.0
            return probabilities
        if method == "object_frequency":
            return self.object_probabilities[target_category]
        if method == "markov":
            return self.markov_probabilities[target_category, last_state]
        if method == "time_frequency":
            quarter = _time_quarter(time_sin_cos)
            return self.time_probabilities[target_category, weekday, quarter]
        raise ValueError(method)


def _time_quarter(time_sin_cos: np.ndarray) -> np.ndarray:
    angle = np.arctan2(time_sin_cos[:, 0], time_sin_cos[:, 1])
    fraction = np.mod(angle, 2 * np.pi) / (2 * np.pi)
    return np.floor(fraction * 4).astype(np.int64).clip(0, 3)
