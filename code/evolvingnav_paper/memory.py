"""Causal, time-valid 4D entity memory (paper Equations 3, 17 and 18)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


def backproject(u: float, v: float, depth_m: float, intrinsics: np.ndarray,
                camera_to_world: np.ndarray) -> np.ndarray:
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth must be positive and finite")
    ray = np.linalg.solve(np.asarray(intrinsics, dtype=float), [u, v, 1.0])
    camera_point = np.r_[depth_m * ray, 1.0]
    world = np.asarray(camera_to_world, dtype=float) @ camera_point
    return world[:3] / world[3]


@dataclass
class EntityVersion:
    valid_from: float
    valid_to: float | None
    state_id: int
    confidence: float
    feature: tuple[float, ...] = ()
    evidence_handles: list[str] = field(default_factory=list)
    world_points: list[tuple[float, float, float]] = field(default_factory=list)
    evidence_times: list[float] = field(default_factory=list)
    evidence_confidences: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class NegativeFrame:
    evidence_id: str
    state_id: int
    timestamp: float
    pose_xyz: tuple[float, float, float]


class VersionedMemory:
    def __init__(self) -> None:
        self.entities: dict[str, list[EntityVersion]] = {}
        self.used_evidence: set[str] = set()
        self.negative_frames: list[NegativeFrame] = []

    def record_negative(self, evidence_id: str, state_id: int, timestamp: float,
                        pose_xyz) -> None:
        if evidence_id in self.used_evidence:
            return
        self.negative_frames.append(NegativeFrame(
            evidence_id, state_id, timestamp, tuple(float(x) for x in pose_xyz)
        ))
        self.used_evidence.add(evidence_id)

    def evidence_at(self, cutoff: float) -> list[NegativeFrame]:
        return [frame for frame in self.negative_frames if frame.timestamp <= cutoff]

    def observe(self, entity_id: str, state_id: int, timestamp: float,
                confidence: float, evidence_id: str, world_point: np.ndarray,
                feature: tuple[float, ...] = ()) -> None:
        if evidence_id in self.used_evidence:
            return
        versions = self.entities.setdefault(entity_id, [])
        if versions and timestamp < versions[-1].valid_from:
            raise ValueError("causal memory cannot accept an older observation")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be a probability")
        point = tuple(float(x) for x in world_point)
        if versions and versions[-1].state_id == state_id:
            current = versions[-1]
            current.confidence = max(current.confidence, confidence)
            current.evidence_handles.append(evidence_id)
            current.world_points.append(point)
            current.evidence_times.append(timestamp)
            current.evidence_confidences.append(confidence)
        else:
            if versions:
                versions[-1].valid_to = timestamp
            versions.append(EntityVersion(timestamp, None, state_id, confidence,
                                          feature, [evidence_id], [point],
                                          [timestamp], [confidence]))
        self.used_evidence.add(evidence_id)

    @staticmethod
    def _causal(version: EntityVersion, cutoff: float) -> EntityVersion:
        indices = [i for i, time in enumerate(version.evidence_times) if time <= cutoff]
        return replace(
            version,
            valid_to=version.valid_to if version.valid_to is not None and version.valid_to <= cutoff else None,
            confidence=max(version.evidence_confidences[i] for i in indices),
            evidence_handles=[version.evidence_handles[i] for i in indices],
            world_points=[version.world_points[i] for i in indices],
            evidence_times=[version.evidence_times[i] for i in indices],
            evidence_confidences=[version.evidence_confidences[i] for i in indices],
        )

    def at(self, entity_id: str, timestamp: float) -> EntityVersion | None:
        for version in reversed(self.entities.get(entity_id, [])):
            if version.valid_from <= timestamp and (version.valid_to is None or timestamp < version.valid_to):
                return self._causal(version, timestamp)
        return None

    def history(self, entity_id: str, cutoff: float) -> list[EntityVersion]:
        return [self._causal(v, cutoff)
                for v in self.entities.get(entity_id, []) if v.valid_from <= cutoff]
