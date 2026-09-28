"""High-level Habitat action adapter; evaluator-private fields never enter Agent."""

from __future__ import annotations

import habitat_sim
import numpy as np

from evolvingnav_paper.agent import ViewEvidence
from evolvingnav_paper.coverage import (
    camera_forward, camera_transform, candidate_surface_samples, depth_quality,
    heading_quaternion,
    visible_sample_ids,
)
from evolvingnav_paper.habitat_utils import set_agent
from evolvingnav_paper.memory import backproject


class HabitatAgentWorld:
    def __init__(self, backend, viewpoints: dict[int, dict], state_centers: dict[int, list],
                 start_xyz, start_xyzw, *, speed_mps: float = 1.0,
                 calibrator=None, category_recall: float = 0.8,
                 known_states: set[int] | None = None,
                 motion_schedule: list[dict] | None = None,
                 surface_points: dict[int, list] | None = None) -> None:
        self.backend = backend
        self.viewpoints = viewpoints
        self.state_centers = state_centers
        self.position = np.asarray(start_xyz, dtype=float)
        self.rotation = list(start_xyzw)
        self.speed_mps = speed_mps
        self.calibrator = calibrator
        self.category_recall = category_recall
        self.frame = 0
        self.private_inspections: list[dict] = []
        self.last_detection = None
        surface_points = surface_points or {}
        self.samples = {state: candidate_surface_samples(
            center, place_points=surface_points.get(state))
                        for state, center in state_centers.items()}
        self.frontiers = set(viewpoints) - (known_states or set())
        self.motion_schedule = sorted(motion_schedule or [], key=lambda row: row["time_s"])
        self.world_time_s = 0.0
        self._next_motion = 0

    def advance_time(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("world time cannot move backwards")
        self.world_time_s += seconds
        while (self._next_motion < len(self.motion_schedule)
               and self.motion_schedule[self._next_motion]["time_s"] <= self.world_time_s):
            self.backend.move_target(self.motion_schedule[self._next_motion])
            self._next_motion += 1

    def distance(self, goal) -> float:
        return self.backend.distance(self.position, goal)

    def sample_count(self, state: int) -> int:
        return len(self.samples[state])

    def expected_new_detection(self, state: int, uncovered: float) -> float:
        center = np.asarray(self.state_centers[state], dtype=float)
        viewpoint = self.viewpoints[state]
        distance = float(np.linalg.norm(center - np.asarray(viewpoint["position_xyz"])))
        direction = center - np.asarray(viewpoint["position_xyz"])
        angle = max(0.0, float(np.dot(
            direction / max(distance, 1e-9),
            camera_forward(viewpoint["rotation_xyzw"]),
        )))
        if self.calibrator is None:
            return self.category_recall * uncovered
        return self.calibrator.predict({
            "coverage": uncovered,
            "range_m": distance,
            "angle_cos": angle,
            "projected_pixels": int(25 * uncovered),
            "depth_quality": 1.0,
            "category_recall": self.category_recall,
        })

    def move_chunk(self, goal, max_distance: float) -> tuple[float, float]:
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(self.position, dtype=np.float32)
        path.requested_end = np.asarray(goal, dtype=np.float32)
        if not self.backend.real.pathfinder.find_path(path):
            return 0.0, 0.0
        remaining = min(float(max_distance), float(path.geodesic_distance))
        points = [np.asarray(point, dtype=float) for point in path.points]
        new_position = points[0]
        for point in points[1:]:
            length = float(np.linalg.norm(point - new_position))
            if length >= remaining:
                new_position = new_position + (point - new_position) * (remaining / length)
                break
            remaining -= length
            new_position = point
        displacement = min(float(max_distance), float(path.geodesic_distance))
        heading = new_position - self.position
        if np.linalg.norm(heading[[0, 2]]) > 1e-6:
            self.rotation = heading_quaternion(heading)
        self.position = new_position
        set_agent(self.backend.real.get_agent(0), self.position.tolist(), self.rotation)
        self.advance_time(displacement / self.speed_mps)
        return displacement, displacement / self.speed_mps

    def inspect(self, state: int) -> tuple[bool, list[ViewEvidence]]:
        viewpoint = self.viewpoints[state]
        self.position = np.asarray(viewpoint["position_xyz"], dtype=float)
        self.rotation = viewpoint["rotation_xyzw"]
        private = self.backend.inspect(state, self.position.tolist())
        self.private_inspections.append({"state_id": state, **private})
        observation = self.backend.last_observation
        if observation is None:
            raise RuntimeError("Habitat did not produce an RGB-D frame")
        self.last_detection = None
        if self.backend.last_detections:
            strongest = max(self.backend.last_detections, key=lambda item: item.confidence)
            depth = observation["depth"]
            pixels = np.argwhere(strongest.mask & np.isfinite(depth) & (depth > 0))
            if len(pixels):
                v, u = np.median(pixels, axis=0)
                depth_m = float(np.median(depth[pixels[:, 0], pixels[:, 1]]))
                height, width = depth.shape
                focal = width / (2 * np.tan(np.deg2rad(79.0) / 2))
                intrinsics = np.array([[focal, 0, width / 2],
                                       [0, focal, height / 2], [0, 0, 1]], dtype=float)
                point = backproject(u, v, depth_m, intrinsics,
                                    camera_transform(self.position, self.rotation))
                self.last_detection = {
                    "world_point": point,
                    "confidence": strongest.confidence,
                    "evidence_id": f"frame-{self.frame + 1}:positive",
                }
        return bool(private["detected"]), self._evidence_from_depth(observation["depth"])

    def observe_chunk(self) -> list[ViewEvidence]:
        if self.backend.detector is None:
            return []
        observation = self.backend.real.get_sensor_observations()
        depth = np.asarray(observation["depth"])
        covered_any = any(visible_sample_ids(
            samples, self.position, self.rotation, depth, 79.0
        ) for samples in self.samples.values())
        if not covered_any:
            return []
        detected = self.backend.detector(
            np.asarray(observation["rgb"]), depth, self.backend.target_category
        )
        return [] if detected else self._evidence_from_depth(depth)

    def _evidence_from_depth(self, depth: np.ndarray) -> list[ViewEvidence]:
        self.frame += 1
        evidence = []
        for candidate, samples in self.samples.items():
            covered = visible_sample_ids(
                samples, self.position, self.rotation, depth, 79.0
            )
            if covered:
                fraction = len(covered) / len(samples)
                if self.calibrator is None:
                    detection_probability = self.category_recall * fraction
                else:
                    direction = np.asarray(self.state_centers[candidate]) - self.position
                    distance = float(np.linalg.norm(direction))
                    features = {
                        "coverage": fraction,
                        "range_m": distance,
                        "angle_cos": max(0.0, float(np.dot(
                            direction / max(distance, 1e-9), camera_forward(self.rotation)))),
                        "projected_pixels": len(covered),
                        "depth_quality": depth_quality(depth),
                        "category_recall": self.category_recall,
                    }
                    detection_probability = self.calibrator.predict(features)
                evidence.append(ViewEvidence(
                    f"frame-{self.frame}:state-{candidate}", candidate, covered,
                    detection_probability, fraction, tuple(self.position.tolist()),
                ))
        return evidence

    def explore(self, budget_m: float) -> tuple[dict[int, tuple[object, float]], float, float]:
        reachable = [
            (self.distance(self.viewpoints[state]["position_xyz"]), state)
            for state in self.frontiers
        ]
        reachable = [(distance, state) for distance, state in reachable
                     if np.isfinite(distance) and distance <= budget_m]
        if not reachable:
            return {}, 0.0, 0.0
        distance, state = min(reachable)
        self.frontiers.remove(state)
        goal = self.viewpoints[state]["position_xyz"]
        displacement, duration = self.move_chunk(goal, distance)
        self.rotation = self.viewpoints[state]["rotation_xyzw"]
        set_agent(self.backend.real.get_agent(0), self.position.tolist(), self.rotation)
        observations = self.backend.real.get_sensor_observations()
        detected = (
            bool(self.backend.detector(
                np.asarray(observations["rgb"]), np.asarray(observations["depth"]),
                self.backend.target_category,
            )) if self.backend.detector is not None else False
        )
        return {state: (goal, 1.0 if detected else 0.5)}, duration, displacement
