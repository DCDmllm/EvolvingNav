"""Validation-only logistic calibration of online detector recall."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "coverage", "range_m", "angle_cos", "projected_pixels",
    "depth_quality", "category_recall",
)


@dataclass(frozen=True)
class DetectionCalibrator:
    intercept: float
    coefficients: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, validation_rows: list[dict]) -> "DetectionCalibrator":
        if not validation_rows or {int(row["detected"]) for row in validation_rows} != {0, 1}:
            raise ValueError("calibration requires positive and negative validation observations")
        features = np.asarray(
            [[float(row[key]) for key in FEATURES] for row in validation_rows], dtype=float
        )
        if not np.isfinite(features).all():
            raise ValueError("calibration features must be finite")
        scaler = StandardScaler().fit(features)
        classifier = LogisticRegression(max_iter=1000).fit(
            scaler.transform(features), [int(row["detected"]) for row in validation_rows]
        )
        return cls(
            float(classifier.intercept_[0]), tuple(classifier.coef_[0].tolist()),
            tuple(scaler.mean_.tolist()), tuple(scaler.scale_.tolist()),
        )

    def predict(self, online_features: dict) -> float:
        values = np.asarray([float(online_features[key]) for key in FEATURES])
        standardized = (values - self.means) / self.scales
        logit = self.intercept + float(np.dot(self.coefficients, standardized))
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "intercept": self.intercept, "coefficients": self.coefficients,
            "means": self.means, "scales": self.scales,
            "features": FEATURES,
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DetectionCalibrator":
        data = json.loads(path.read_text(encoding="utf-8"))
        if tuple(data["features"]) != FEATURES:
            raise ValueError("calibrator feature schema mismatch")
        return cls(
            float(data["intercept"]), tuple(data["coefficients"]),
            tuple(data["means"]), tuple(data["scales"]),
        )
