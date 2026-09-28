"""Run P4D-HSSD navigation episodes through the N1/N2 high-level track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evolvingnav_paper.backend import HabitatInspectionBackend
from evolvingnav_paper.evaluate import evaluate_search
from evolvingnav_paper.perception import GroundedSAMInspector
from evolvingnav_paper.policy import load_belief, pack_public_query, predict_public

CODE_ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("n1", "n2"), default="n2")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--world", choices=("routine", "random", "static"), default="routine")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inspection", choices=("semantic-oracle", "grounded-sam"), default="grounded-sam")
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--navmesh-root", type=Path, required=True)
    parser.add_argument("--grounding-dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--perception-config", type=Path, default=CODE_ROOT / "configs/perception.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
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
    model, schema = load_belief(args.checkpoint, args.dataset)
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
    objects = {
        row["instance_uuid"]: row
        for row in rows(args.tasks / "catalogs/object_instances.jsonl")
    }

    decisions = []
    for episode in episodes:
        query = queries[episode["query_id"]]
        packed = pack_public_query(query, schema, features)
        belief = predict_public(model, schema, packed)
        candidates = {
            int(state): belief[int(state)]
            for state in episode["public_refs"]["candidate_state_ids"]
            if int(state) in public_goals
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
    scores = []
    try:
        for episode, decision in zip(episodes, decisions, strict=True):
            truth = private[episode["base_episode_id"]]
            backend.prepare(
                truth, objects[episode["target"]["object_id"]], list(decision["belief"])
            )
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
        "track": f"high_level_public_viewpoints_{args.inspection}",
        "min_visible_fraction": 0.20,
        "dataset": str(args.tasks), "checkpoint": str(args.checkpoint),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
