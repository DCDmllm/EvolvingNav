"""Habitat navigation and evaluator-private visual inspection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import habitat_sim
import magnum as mn
import numpy as np

from evolvingnav_paper.habitat_utils import make_simulator, path_distance, sensor_specs, set_agent


def visible_fraction_from_masks(actual: np.ndarray, target_only: np.ndarray, semantic_id: int) -> float:
    """Visible target pixels divided by its unobstructed projection at the same pose."""
    projected = int(np.count_nonzero(target_only == semantic_id))
    if projected == 0:
        return 0.0
    visible = int(np.count_nonzero(actual == semantic_id))
    return min(1.0, visible / projected)


class HabitatInspectionBackend:
    def __init__(
        self, hssd_root: Path, scene_id: str, navmesh: Path,
        public_viewpoints: dict[int, dict], *, gpu: int = 0,
        detector: Callable[[np.ndarray, np.ndarray, str], bool] | None = None,
    ) -> None:
        self.hssd_root = hssd_root
        self.scene_id = scene_id
        self.navmesh = navmesh
        self.gpu = gpu
        self.public_viewpoints = public_viewpoints
        self.detector = detector
        self.real = None
        self.object_id = None
        self.truth = None
        self.projected_pixels = {}

    def distance(self, start, goal) -> float:
        if self.real is None:
            raise RuntimeError("prepare an episode before requesting paths")
        return path_distance(self.real.pathfinder, start, goal)

    def _place_target(self, sim, truth: dict, target: dict) -> int:
        config_path = self.hssd_root / target["template_handle"]
        manager = sim.get_object_template_manager()
        manager.load_configs(str(config_path.resolve()))
        handles = manager.get_template_handles(target["template_hash"])
        if not handles:
            raise RuntimeError(f"target template unavailable: {config_path}")
        obj = sim.get_rigid_object_manager().add_object_by_template_handle(handles[-1])
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.semantic_id = self.semantic_id
        obj.translation = mn.Vector3(*truth["target_position_xyz"])
        obj.rotation = mn.Quaternion.rotation(
            mn.Rad((self.semantic_id % 12) * math.pi / 6), mn.Vector3.y_axis()
        )
        sim.perform_discrete_collision_detection()
        return obj.object_id

    def prepare(self, truth: dict, target: dict, candidate_ids: list[int]) -> None:
        self.clear()
        if self.real is not None:
            self.real.close()
            self.real = None
        self.truth = truth
        self.semantic_id = int(target["semantic_instance_id"])
        self.target_category = str(target["category_canonical"])
        config = habitat_sim.SimulatorConfiguration()
        config.scene_id = "NONE"
        config.create_renderer = True
        config.enable_physics = True
        config.gpu_device_id = self.gpu
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = sensor_specs(320, 240)
        target_only = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))
        try:
            self._place_target(target_only, truth, target)
            self.projected_pixels = {}
            for state in candidate_ids:
                viewpoint = self.public_viewpoints[state]
                set_agent(
                    target_only.get_agent(0), viewpoint["position_xyz"],
                    viewpoint["rotation_xyzw"],
                )
                semantic = np.asarray(target_only.get_sensor_observations()["semantic"])
                self.projected_pixels[state] = int(np.count_nonzero(semantic == self.semantic_id))
        finally:
            target_only.close()
        self.real = make_simulator(
            self.hssd_root, self.scene_id, self.navmesh, self.gpu, 320, 240
        )
        self.object_id = self._place_target(self.real, truth, target)

    def inspect(self, state: int, position) -> dict:
        if self.truth is None:
            raise RuntimeError("prepare an episode before inspection")
        viewpoint = self.public_viewpoints[state]
        set_agent(self.real.get_agent(0), position, viewpoint["rotation_xyzw"])
        observations = self.real.get_sensor_observations()
        actual = np.asarray(observations["semantic"])
        actual_pixels = int(np.count_nonzero(actual == self.semantic_id))
        projected_pixels = self.projected_pixels[state]
        visible_fraction = min(1.0, actual_pixels / projected_pixels) if projected_pixels else 0.0
        goal_distance = min(
            self.distance(position, goal["position_xyz"])
            for goal in self.truth["valid_goal_viewpoints"]
        )
        detected = (
            self.detector(
                np.asarray(observations["rgb"]),
                np.asarray(observations["depth"]),
                self.target_category,
            )
            if self.detector is not None else actual_pixels > 0
        )
        return {
            "detected": detected,
            "visible_fraction": visible_fraction,
            "distance_to_valid_goal_m": goal_distance,
            "visible_target_pixels": actual_pixels,
            "projected_target_pixels": projected_pixels,
        }

    def clear(self) -> None:
        if self.real is not None and self.object_id is not None:
            self.real.get_rigid_object_manager().remove_object_by_id(self.object_id)
        self.object_id = None
        self.truth = None

    def close(self) -> None:
        self.clear()
        if self.real is not None:
            self.real.close()
            self.real = None
