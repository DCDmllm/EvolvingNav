"""Run P4D-HSSD navigation episodes with belief or closed-loop Agent control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evolvingnav_paper.backend import HabitatInspectionBackend
from evolvingnav_paper.agent import Agent, AgentConfig
from evolvingnav_paper.calibration import DetectionCalibrator
from evolvingnav_paper.controller import LunaToolController
from evolvingnav_paper.evaluate import evaluate_search
from evolvingnav_paper.memory import VersionedMemory
from evolvingnav_paper.perception import GroundedSAMInspector
from evolvingnav_paper.policy import load_belief, model_input_batch, pack_public_query, predict_public
from evolvingnav_paper.transition import IdentityTransition
from evolvingnav_paper.transition_model import NeuralTransition, TransitionHead
from evolvingnav_paper.world import HabitatAgentWorld

CODE_ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("n1", "n2", "n3", "n4"), default="n3")
    parser.add_argument("--agent", action="store_true", help="Run the event-driven Agent for N1/N2 as well")
    parser.add_argument("--controller", choices=("utility", "luna"), default="utility")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--world", choices=("routine", "random", "static"), default="routine")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transition-checkpoint", type=Path)
    parser.add_argument("--inspection", choices=("semantic-oracle", "grounded-sam"), default="grounded-sam")
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--navmesh-root", type=Path, required=True)
    parser.add_argument("--grounding-dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--perception-config", type=Path, default=CODE_ROOT / "configs/perception.yaml")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.task == "n4" and args.transition_checkpoint is None:
        parser.error("N4 requires --transition-checkpoint")
    return args


def main() -> int:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    episodes = []
    for episode in rows(args.tasks / f"public/episodes_{args.task}.jsonl"):
        if episode["world_variant"] == args.world:
            episodes.append(episode)
        if len(episodes) == args.limit:
            break
    if len(episodes) != args.limit:
        raise ValueError(f"only found {len(episodes)} matching episodes")
    wanted_queries = {episode["query_id"] for episode in episodes}
    queries = {row["query_id"]: row for row in rows(args.tasks / "public/query_inputs.jsonl") if row["query_id"] in wanted_queries}
    active_checkpoint = args.transition_checkpoint if args.task == "n4" else args.checkpoint
    model, schema = load_belief(active_checkpoint, args.dataset)
    transition_head = None
    if args.task == "n4":
        import torch

        checkpoint = torch.load(active_checkpoint, map_location="cpu", weights_only=True)
        transition_head = TransitionHead(checkpoint["model_config"]["hidden_dim"])
        transition_head.load_state_dict(checkpoint["transition_head"])
        transition_head.eval()
    with np.load(args.dataset / "records/packed/train.npz", allow_pickle=False) as train:
        features = {key: train[key] for key in (
            "candidate_region_category_id", "candidate_receptacle_category_id",
            "candidate_center_xyz", "candidate_is_unknown",
        )}
    catalog = json.loads((args.tasks / "catalogs/candidate_states_navigation.json").read_text())
    public_viewpoints = {
        int(row["state_id"]): row["navigation_viewpoint"]
        for row in catalog["states"] if row.get("navigation_eligible")
    }
    public_goals = {state: row["position_xyz"] for state, row in public_viewpoints.items()}
    all_centers = {
        int(row["state_id"]): row["state_center"] for row in catalog["states"]
    }
    state_centers = {
        int(row["state_id"]): row["state_center"]
        for row in catalog["states"] if row.get("navigation_eligible")
    }
    surface_points = {
        int(row["state_id"]): [slot["point"] for slot in row.get("sampled_place_points", [])]
        for row in rows(args.tasks / "catalogs/receptacles.jsonl")
    }
    objects = {
        row["instance_uuid"]: row
        for row in rows(args.tasks / "catalogs/object_instances.jsonl")
    }

    decisions = []
    query_batches = []
    for episode in episodes:
        query = queries[episode["query_id"]]
        packed = pack_public_query(query, schema, features)
        query_batches.append(model_input_batch(packed, schema))
        belief = predict_public(model, schema, packed)
        candidates = {
            int(state): belief[int(state)]
            for state in episode["public_refs"]["candidate_state_ids"]
        }
        decisions.append({
            "base_episode_id": episode["base_episode_id"], "query_id": episode["query_id"],
            "task": args.task, "belief": candidates,
        })

    wanted = {episode["base_episode_id"] for episode in episodes}
    private = {
        row["base_episode_id"]: row["evaluation_private"]
        for row in rows(args.tasks / "private/evaluation_gt.jsonl")
        if row["base_episode_id"] in wanted
    }
    scene_ids = {episode["scene_id"] for episode in episodes}
    if len(scene_ids) != 1:
        raise ValueError("--tasks must select episodes from one scene")
    scene_id = next(iter(scene_ids))
    navmesh = args.navmesh_root / f"{scene_id}.navmesh"
    inspector = (
        GroundedSAMInspector(
            args.perception_config,
            dino_model=args.grounding_dino_model,
            sam_model=args.sam2_model,
        )
        if args.inspection == "grounded-sam" else None
    )
    backend = HabitatInspectionBackend(
        args.hssd_root, scene_id, navmesh, public_viewpoints,
        detector=inspector,
    )
    calibrator = DetectionCalibrator.load(args.calibration) if args.calibration else None
    controller = LunaToolController() if args.controller == "luna" else None
    scores = []
    try:
        for episode, decision in zip(episodes, decisions, strict=True):
            truth = private[episode["base_episode_id"]]
            backend.prepare(
                truth, objects[episode["target"]["object_id"]],
                [state for state in decision["belief"] if state in public_goals],
                dynamic=args.task == "n4",
            )
            if args.task in {"n3", "n4"} or args.agent:
                query = queries[episode["query_id"]]
                target_id = query["input"]["target"]["instance_uuid"]
                memory = VersionedMemory()
                for index, event in enumerate(query["input"].get("target_history", [])):
                    if event["event_type"] == "positive_observation":
                        observed_state = int(event["observed_state_id"])
                        memory.observe(
                            target_id, observed_state, float(event["timestamp_s"]),
                            float(event["detector_confidence"])
                            * float(event["instance_match_confidence"]),
                            f"{query['query_id']}:history:{index}",
                            np.asarray(all_centers[observed_state]),
                        )
                motion_schedule = None
                if args.task == "n4":
                    motion_schedule = truth.get("target_motion_schedule")
                    if motion_schedule is None:
                        raise ValueError("N4 private evaluation record requires target_motion_schedule")
                    required_motion = {
                        "time_s", "target_position_xyz", "current_state_id",
                        "valid_goal_viewpoints",
                    }
                    if any(required_motion - set(event) for event in motion_schedule):
                        raise ValueError("N4 target_motion_schedule has incomplete motion events")
                world = HabitatAgentWorld(
                    backend, public_viewpoints,
                    state_centers,
                    episode["agent_start"]["position_xyz"],
                    episode["agent_start"]["rotation_xyzw"],
                    calibrator=calibrator,
                    known_states=set(decision["belief"]),
                    motion_schedule=motion_schedule,
                    surface_points=surface_points,
                )
                unknown_ids = [int(i) for i, flag in enumerate(features["candidate_is_unknown"]) if flag]
                can_explore = "EXPLORE" in episode["public_refs"].get("action_space", [])
                transition = (NeuralTransition(model, transition_head, query_batches[len(scores)])
                              if args.task == "n4" else IdentityTransition())
                agent = Agent(
                    decision["belief"], public_goals, world, transition,
                    AgentConfig(
                        max_inspections=1 if args.task == "n1" else int(
                            episode["episode_budget"]["max_candidate_inspections"]),
                        max_path_m=float(episode["episode_budget"]["max_path_length_m"]),
                        chunk_m=2.0,
                        unknown_state=unknown_ids[0] if can_explore and unknown_ids else None,
                    ),
                    sample_count={state: 25 for state in decision["belief"]},
                    controller=controller,
                    memory=memory, target_id=target_id,
                    time_origin_s=float(query["input"]["query"]["query_time_s"]),
                )
                result = agent.run()
                final = world.private_inspections[-1] if world.private_inspections else None
                success = bool(
                    result.found and final is not None
                    and final["distance_to_valid_goal_m"]
                    <= float(episode["success_spec"]["max_geodesic_distance_m"])
                    and final["visible_fraction"] >= 0.20
                )
                oracle_m = float(truth["oracle_shortest_path_m"])
                score = {
                    "base_episode_id": episode["base_episode_id"],
                    "task": args.task, "success": success,
                    "inspections": len(result.inspections),
                    "inspection_order": result.inspections,
                    "inspection_evidence": world.private_inspections,
                    "actions": result.actions, "path_m": round(result.path_m, 6),
                    "spl": round(float(success) * oracle_m / max(result.path_m, oracle_m, 1e-9), 6),
                    "true_state_id": int(backend.truth["current_state_id"]),
                    "posterior": result.posterior,
                    "termination": result.termination,
                    "evidence_trace": result.evidence_trace,
                }
            else:
                score = evaluate_search(
                    episode, truth, decision["belief"],
                    start=episode["agent_start"]["position_xyz"], goals=public_goals,
                    distance=backend.distance, inspect=backend.inspect, task=args.task,
                )
            scores.append(score)
            backend.clear()
    finally:
        backend.close()
        if inspector is not None:
            inspector.close()

    args.output.mkdir(parents=True)
    with (args.output / "policy.jsonl").open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    with (args.output / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(json.dumps(score, ensure_ascii=False) + "\n")
    summary = {
        "task": args.task, "world": args.world, "episodes": len(scores),
        "successes": sum(row["success"] for row in scores),
        "sr": sum(row["success"] for row in scores) / len(scores),
        "spl": sum(row["spl"] for row in scores) / len(scores),
        "track": f"high_level_{'event_agent' if args.task in {'n3', 'n4'} or args.agent else 'ranked'}_{args.inspection}",
        "min_visible_fraction": 0.20,
        "dataset": str(args.tasks), "checkpoint": str(args.checkpoint),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
