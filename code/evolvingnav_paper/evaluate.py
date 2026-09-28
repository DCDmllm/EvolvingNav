"""Evaluator-only truth access and high-level N1/N2 scoring."""

from __future__ import annotations

import math
from typing import Callable


def evaluate_search(
    episode: dict,
    truth: dict,
    belief: dict[int, float],
    *,
    start,
    goals: dict,
    distance: Callable,
    inspect: Callable,
    task: str = "n2",
) -> dict:
    from evolvingnav_paper.policy import rank_candidates

    if task not in {"n1", "n2"}:
        raise ValueError(task)
    true_state = int(truth["current_state_id"])
    oracle_m = float(truth["oracle_shortest_path_m"])
    max_inspections = 1 if task == "n1" else int(episode["episode_budget"]["max_candidate_inspections"])
    max_path = float(episode["episode_budget"]["max_path_length_m"])
    current, travelled, inspected, actions, evidence = start, 0.0, [], [], []
    remaining = set(belief) & set(goals)
    success = False
    for _ in range(max_inspections):
        costs = {state: float(distance(current, goals[state])) for state in remaining}
        reachable = {state: belief[state] for state in remaining if math.isfinite(costs[state])}
        if not reachable:
            break
        state = rank_candidates(reachable, costs, task=task)[0]
        goal = goals[state]
        leg = costs[state]
        if not math.isfinite(leg) or travelled + leg > max_path:
            break
        travelled += leg
        inspected.append(state)
        actions.extend((f"NAVIGATE_TO({state})", f"INSPECT({state})"))
        current = goal
        observation = inspect(state, goal)
        evidence.append({"state_id": state, **observation})
        if observation["detected"]:
            actions.append("STOP")
            success = (
                float(observation["distance_to_valid_goal_m"])
                <= float(episode["success_spec"]["max_geodesic_distance_m"])
                and float(observation["visible_fraction"]) >= 0.20
            )
            break
        remaining.remove(state)
    return {
        "base_episode_id": episode["base_episode_id"],
        "task": task,
        "success": success,
        "inspections": len(inspected),
        "inspection_order": inspected,
        "inspection_evidence": evidence,
        "actions": actions,
        "path_m": round(travelled, 6),
        "spl": round(float(success) * oracle_m / max(travelled, oracle_m, 1e-9), 6),
        "true_state_id": true_state,
    }
