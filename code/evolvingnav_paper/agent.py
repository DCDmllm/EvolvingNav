"""Event-driven EvolvingNav controller (paper Equations 11–15, 20–22)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from evolvingnav_paper.filter import BeliefFilter, EvidenceLedger
from evolvingnav_paper.policy import candidate_utility


@dataclass(frozen=True)
class ViewEvidence:
    evidence_id: str
    state_id: int
    surface_samples: frozenset[int]
    detection_probability: float
    covered_fraction: float = 1.0
    pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AgentConfig:
    max_inspections: int = 10
    max_path_m: float = 100.0
    chunk_m: float = 2.0
    speed_mps: float = 1.0
    inspection_s: float = 1.0
    lambda_time: float = 0.05
    lambda_inspect: float = 0.25
    lambda_scan: float = 0.25
    explore_cost: float = 2.0
    discovery_probability: float = 0.5
    speed_ema_alpha: float = 0.8
    unknown_state: int | None = None


@dataclass
class AgentResult:
    found: bool = False
    actions: list[str] = field(default_factory=list)
    inspections: list[int] = field(default_factory=list)
    path_m: float = 0.0
    elapsed_s: float = 0.0
    posterior: dict[int, float] = field(default_factory=dict)
    termination: str = ""
    evidence_trace: list[dict] = field(default_factory=list)


class World(Protocol):
    def distance(self, goal) -> float: ...
    def move_chunk(self, goal, max_distance: float) -> tuple[float, float]: ...
    def inspect(self, state: int) -> tuple[bool, list[ViewEvidence]]: ...
    def explore(self, budget_m: float) -> tuple[dict[int, tuple[object, float]], float, float]: ...


class Agent:
    def __init__(self, prior: dict[int, float], goals: dict[int, object], world: World,
                 transition, config: AgentConfig | None = None,
                 sample_count: dict[int, int] | None = None, controller=None,
                 memory=None, target_id: str | None = None,
                 time_origin_s: float = 0.0) -> None:
        self.config = config or AgentConfig()
        self.filter = BeliefFilter(prior, transition)
        self.goals = {state: goal for state, goal in goals.items() if state in prior}
        self.world = world
        self.ledger = EvidenceLedger(sample_count=sample_count)
        self.speed = self.config.speed_mps
        self.controller = controller
        self.memory = memory
        self.target_id = target_id
        self.time_origin_s = time_origin_s

    def _return_probability(self, state: int, eta: float) -> float:
        if state not in self.filter.posterior:
            return 0.0
        forecast = self.filter.arrival(eta)
        # Equation 21 excludes probability already at this state.
        return max(0.0, forecast.get(state, 0.0) - self.filter.posterior.get(state, 0.0)
                   * self.filter.transition.matrix(list(self.filter.posterior), eta)[
                       list(self.filter.posterior).index(state), list(self.filter.posterior).index(state)])

    def _choose(self, result: AgentResult) -> int | str | None:
        scored: list[tuple[float, int | str]] = []
        for state, goal in self.goals.items():
            distance = self.world.distance(goal)
            if not math.isfinite(distance) or result.path_m + distance > self.config.max_path_m:
                continue
            eta = distance / self.speed + self.config.inspection_s
            arrival = self.filter.arrival(eta)
            return_probability = self._return_probability(state, eta)
            if not self.ledger.eligible(
                state, belief=self.filter.posterior.get(state, 0.0),
                return_probability=return_probability, new_coverage=0.0,
                dynamic=self.filter.transition.dynamic,
                now_s=result.elapsed_s,
            ):
                continue
            uncovered = max(0.0, 1.0 - self.ledger.coverage(state))
            new_detection = (
                self.world.expected_new_detection(state, uncovered)
                if hasattr(self.world, "expected_new_detection") else uncovered
            )
            utility = candidate_utility(
                arrival_probability=arrival.get(state, 0.0),
                new_detection_probability=new_detection,
                distance_m=distance, eta_s=eta,
                lambda_time=self.config.lambda_time,
                lambda_inspect=self.config.lambda_inspect,
            )
            scored.append((utility, state))
        unknown = self.config.unknown_state
        if unknown is not None and self.filter.posterior.get(unknown, 0.0) > 0:
            utility = (self.filter.posterior[unknown] * self.config.discovery_probability
                       / (self.config.explore_cost + self.config.lambda_scan))
            scored.append((utility, "EXPLORE"))
        if not scored:
            return None
        preferred = max(scored, key=lambda row: (row[0], -row[1] if isinstance(row[1], int) else 0))
        if self.controller is not None:
            labels = [f"NAVIGATE_TO({action})" if isinstance(action, int) else action
                      for _, action in scored]
            chosen = self.controller.choose(labels, {
                "belief": self.filter.posterior,
                "utilities": dict(zip(labels, [utility for utility, _ in scored], strict=True)),
                "elapsed_s": result.elapsed_s, "path_m": result.path_m,
            })
            for utility, action in scored:
                label = f"NAVIGATE_TO({action})" if isinstance(action, int) else action
                if label == chosen and utility >= preferred[0] - 1e-9:
                    return action
        return preferred[1]

    def _admit_negative(self, evidence: list[ViewEvidence], now_s: float,
                        result: AgentResult) -> None:
        for item in evidence:
            new_coverage = self.ledger.admit(item)
            if new_coverage > 0:
                probability = min(1.0, item.detection_probability * new_coverage
                                  / max(item.covered_fraction, 1e-9))
                before = self.filter.posterior.get(item.state_id, 0.0)
                applied = self.filter.negative(
                    {item.state_id: probability},
                    f"{item.evidence_id}:{item.state_id}",
                )
                if applied:
                    result.evidence_trace.append({
                        "evidence_id": item.evidence_id,
                        "state_id": item.state_id,
                        "time_s": now_s,
                        "new_coverage": new_coverage,
                        "detection_probability": probability,
                        "prior": before,
                        "posterior": self.filter.posterior.get(item.state_id, 0.0),
                    })
                if applied and self.memory is not None:
                    self.memory.record_negative(
                        item.evidence_id, item.state_id,
                        self.time_origin_s + now_s, item.pose_xyz
                    )

    def run(self) -> AgentResult:
        result = AgentResult()
        for _ in range(self.config.max_inspections + len(self.goals) * 20):
            selected = self._choose(result)
            if selected is None:
                result.termination = "search_exhausted"
                unknown = self.config.unknown_state
                unknown_mass = self.filter.posterior.get(unknown, 0.0) if unknown is not None else 0.0
                searchable = sum(self.filter.posterior.get(state, 0.0) for state in self.goals)
                if unknown_mass < 0.05 and searchable < 0.05:
                    result.actions.append("NOT_FOUND")
                break
            if selected == "EXPLORE":
                result.actions.append("EXPLORE")
                discovered, duration, displacement = self.world.explore(
                    self.config.max_path_m - result.path_m
                )
                self.filter.advance(duration)
                result.elapsed_s += duration
                result.path_m += displacement
                if not discovered:
                    result.termination = "exploration_exhausted"
                    break
                unknown = self.config.unknown_state
                mass = self.filter.posterior[unknown]
                total_weight = sum(weight for _, weight in discovered.values())
                allocated = min(mass, total_weight)
                self.filter.posterior[unknown] -= allocated
                for state, (goal, probability) in discovered.items():
                    self.goals[state] = goal
                    if hasattr(self.world, "sample_count"):
                        self.ledger.sample_count[state] = self.world.sample_count(state)
                    if hasattr(self.filter.transition, "add_candidate"):
                        self.filter.transition.add_candidate(state)
                    self.filter.posterior[state] = self.filter.posterior.get(state, 0.0) + (
                        allocated * probability / total_weight)
                continue
            state = int(selected)
            goal = self.goals[state]
            result.actions.append(f"NAVIGATE_TO({state})")
            while self.world.distance(goal) > 1e-4:
                remaining = self.config.max_path_m - result.path_m
                if remaining <= 0:
                    result.termination = "path_budget_exhausted"
                    result.posterior = self.filter.posterior.copy()
                    return result
                displacement, duration = self.world.move_chunk(goal, min(self.config.chunk_m, remaining))
                if displacement <= 0 or duration <= 0:
                    result.termination = "navigation_blocked"
                    result.posterior = self.filter.posterior.copy()
                    return result
                self.filter.advance(duration)
                result.path_m += displacement
                result.elapsed_s += duration
                self.speed = (self.config.speed_ema_alpha * self.speed
                              + (1 - self.config.speed_ema_alpha) * displacement / duration)
                if hasattr(self.world, "observe_chunk"):
                    self._admit_negative(self.world.observe_chunk(), result.elapsed_s, result)
                if self._choose(result) != state:
                    break
            else:
                result.actions.append(f"INSPECT({state})")
                if hasattr(self.world, "advance_time"):
                    self.world.advance_time(self.config.inspection_s)
                self.filter.advance(self.config.inspection_s)
                result.elapsed_s += self.config.inspection_s
                detected, evidence = self.world.inspect(state)
                result.inspections.append(state)
                self.ledger.mark_inspected(state, now_s=result.elapsed_s)
                if detected:
                    detection = getattr(self.world, "last_detection", None)
                    if (self.memory is not None and self.target_id is not None
                            and detection is not None):
                        self.memory.observe(
                            self.target_id, state, self.time_origin_s + result.elapsed_s,
                            detection["confidence"], detection["evidence_id"],
                            detection["world_point"],
                        )
                    result.found = True
                    result.actions.append("STOP")
                    break
                self._admit_negative(evidence, result.elapsed_s, result)
                if len(result.inspections) >= self.config.max_inspections:
                    result.termination = "inspection_budget_exhausted"
                    break
        result.posterior = self.filter.posterior.copy()
        return result
