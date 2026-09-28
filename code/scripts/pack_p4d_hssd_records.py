#!/usr/bin/env python3
"""Pack P4D JSONL query records into leakage-safe NumPy training tensors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MAX_HISTORY = 64
MAX_CONTEXT_HISTORY = 16


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-history", type=int, default=MAX_HISTORY)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pack_split(
    records: list[dict[str, Any]],
    *,
    max_history: int,
    category_to_id: dict[str, int],
    state_count: int,
    candidate_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    count = len(records)
    event_type = np.zeros((count, max_history), dtype=np.int8)
    event_time_days = np.zeros((count, max_history), dtype=np.float32)
    delta_time_log1p = np.zeros((count, max_history), dtype=np.float32)
    observed_state_id = np.full((count, max_history), -1, dtype=np.int16)
    candidate_state_id = np.full((count, max_history), -1, dtype=np.int16)
    # detector_conf, identity_conf, visible_fraction, frustum, unoccluded, neg_strength
    evidence_features = np.zeros((count, max_history, 6), dtype=np.float32)
    history_mask = np.zeros((count, max_history), dtype=np.bool_)
    candidate_state_ids = np.full((count, state_count), -1, dtype=np.int16)
    candidate_mask = np.zeros((count, state_count), dtype=np.bool_)
    target_category_id = np.zeros(count, dtype=np.int16)
    context_time_days = np.zeros((count, MAX_CONTEXT_HISTORY), dtype=np.float32)
    context_category_counts = np.zeros(
        (count, MAX_CONTEXT_HISTORY, len(category_to_id)), dtype=np.float32
    )
    context_observation_features = np.zeros(
        (count, MAX_CONTEXT_HISTORY, 4), dtype=np.float32
    )
    context_mask = np.zeros((count, MAX_CONTEXT_HISTORY), dtype=np.bool_)
    query_time_days = np.zeros(count, dtype=np.float32)
    query_time_of_day_sin_cos = np.zeros((count, 2), dtype=np.float32)
    query_weekday_id = np.zeros(count, dtype=np.int8)
    elapsed_since_last_positive_days = np.zeros(count, dtype=np.float32)
    meta_world_variant_id = np.zeros(count, dtype=np.int8)
    meta_grounding_quality_id = np.zeros(count, dtype=np.int8)
    y_current_state = np.zeros(count, dtype=np.int16)
    y_moved = np.zeros(count, dtype=np.bool_)
    y_transition_occurred = np.zeros(count, dtype=np.bool_)
    y_returned_to_last = np.zeros(count, dtype=np.bool_)
    record_ids = []
    instance_ids = []
    world_ids = {"routine": 0, "random": 1, "static": 2}
    grounding_ids = {
        "exact": 0,
        "role_equivalent": 1,
        "fallback": 2,
        "counterfactual": 3,
        "no_transition": 4,
    }

    for row_index, record in enumerate(records):
        model_input = record["input"]
        history = model_input["target_history"][-max_history:]
        offset = max_history - len(history)
        for history_index, event in enumerate(history, start=offset):
            history_mask[row_index, history_index] = True
            event_time_days[row_index, history_index] = event["timestamp_s"] / 86400.0
            delta_time_log1p[row_index, history_index] = math.log1p(
                max(0.0, event["delta_t_from_previous_s"])
            )
            if event["event_type"] == "positive_observation":
                event_type[row_index, history_index] = 1
                observed_state_id[row_index, history_index] = event["observed_state_id"]
                evidence_features[row_index, history_index, :3] = (
                    event["detector_confidence"],
                    event["instance_match_confidence"],
                    event["visible_fraction_estimate"],
                )
            elif event["event_type"] == "candidate_inspection":
                event_type[row_index, history_index] = 2
                candidate_state_id[row_index, history_index] = event["candidate_state_id"]
                evidence_features[row_index, history_index, 3:] = (
                    event["surface_in_frustum_fraction"],
                    event["surface_unoccluded_fraction"],
                    event["negative_evidence_strength"],
                )
            else:
                raise ValueError(f"unknown event type: {event['event_type']}")
        candidates = model_input["candidate_state_ids"]
        if len(candidates) > state_count:
            raise ValueError(f"candidate overflow: {record['record_id']}")
        candidate_state_ids[row_index, : len(candidates)] = candidates
        candidate_mask[row_index, : len(candidates)] = True
        target = model_input["target"]
        target_category_id[row_index] = category_to_id[target["category"]]
        query_time_s = model_input["query"]["query_time_s"]
        query_time_days[row_index] = query_time_s / 86400.0
        phase = 2.0 * math.pi * (query_time_s % 86400.0) / 86400.0
        query_time_of_day_sin_cos[row_index] = (math.sin(phase), math.cos(phase))
        query_weekday_id[row_index] = int(query_time_s // 86400) % 7
        context_history = model_input.get(
            "observable_context_history", []
        )[-MAX_CONTEXT_HISTORY:]
        context_offset = MAX_CONTEXT_HISTORY - len(context_history)
        for context_index, context in enumerate(
            context_history, start=context_offset
        ):
            context_mask[row_index, context_index] = True
            context_time_days[row_index, context_index] = (
                context["timestamp_s"] / 86400.0
            )
            for category, value in context["observed_category_counts"].items():
                context_category_counts[
                    row_index, context_index, category_to_id[category]
                ] = float(value)
            context_observation_features[row_index, context_index] = (
                float(context["detected_object_count"]),
                float(context["observed_change_count"]),
                float(len(context["inspected_candidate_state_ids"])),
                float(len(context["observed_region_ids"])),
            )
        elapsed_since_last_positive_days[row_index] = (
            model_input["history_summary"]["elapsed_since_last_positive_s"] / 86400.0
        )
        meta_world_variant_id[row_index] = world_ids[record["world_variant"]]
        grounding = record["supervision"].get(
            "last_transition_grounding_quality", "no_transition"
        )
        meta_grounding_quality_id[row_index] = grounding_ids.get(
            grounding, grounding_ids["no_transition"]
        )
        y_current_state[row_index] = record["supervision"]["current_state_id"]
        y_moved[row_index] = record["supervision"]["moved_since_last_positive"]
        y_transition_occurred[row_index] = record["supervision"].get(
            "transition_occurred_since_last_positive", y_moved[row_index]
        )
        y_returned_to_last[row_index] = record["supervision"].get(
            "returned_to_last_state", False
        )
        record_ids.append(record["record_id"])
        instance_ids.append(target["instance_uuid"])

    return {
        "event_type": event_type,
        "event_time_days": event_time_days,
        "delta_time_log1p": delta_time_log1p,
        "observed_state_id": observed_state_id,
        "candidate_state_id": candidate_state_id,
        "evidence_features": evidence_features,
        "history_mask": history_mask,
        "candidate_state_ids": candidate_state_ids,
        "candidate_mask": candidate_mask,
        "target_category_id": target_category_id,
        "context_time_days": context_time_days,
        "context_category_counts": context_category_counts,
        "context_observation_features": context_observation_features,
        "context_mask": context_mask,
        "query_time_days": query_time_days,
        "query_time_of_day_sin_cos": query_time_of_day_sin_cos,
        "query_weekday_id": query_weekday_id,
        "elapsed_since_last_positive_days": elapsed_since_last_positive_days,
        "meta_world_variant_id": meta_world_variant_id,
        "meta_grounding_quality_id": meta_grounding_quality_id,
        "y_current_state": y_current_state,
        "y_moved": y_moved,
        "y_transition_occurred": y_transition_occurred,
        "y_returned_to_last": y_returned_to_last,
        "record_id": np.asarray(record_ids),
        "instance_uuid": np.asarray(instance_ids),
        **candidate_features,
    }


def main() -> int:
    args = arguments()
    root = args.root.resolve()
    objects = load_jsonl(root / "object_pool/objects.jsonl")
    categories = sorted({row["category_canonical"] for row in objects})
    category_to_id = {category: index for index, category in enumerate(categories)}
    state_space = json.loads((root / "scene/candidate_states.json").read_text())
    state_count = len(state_space["states"])
    all_records = {
        split: load_jsonl(root / f"records/{split}_queries.jsonl")
        for split in ("train", "val", "test")
    }
    region_categories = sorted(
        {row["region_category"] for row in state_space["states"]}
    )
    receptacle_categories = sorted(
        {row["receptacle_category"] for row in state_space["states"]}
    )
    region_to_id = {value: index for index, value in enumerate(region_categories)}
    receptacle_to_id = {
        value: index for index, value in enumerate(receptacle_categories)
    }
    candidate_features = {
        "candidate_region_category_id": np.asarray(
            [region_to_id[row["region_category"]] for row in state_space["states"]],
            dtype=np.int16,
        ),
        "candidate_receptacle_category_id": np.asarray(
            [
                receptacle_to_id[row["receptacle_category"]]
                for row in state_space["states"]
            ],
            dtype=np.int16,
        ),
        "candidate_center_xyz": np.asarray(
            [row["state_center"] or [0.0, 0.0, 0.0] for row in state_space["states"]],
            dtype=np.float32,
        ),
        "candidate_is_unknown": np.asarray(
            [row["region_id"] == "unknown" for row in state_space["states"]],
            dtype=np.bool_,
        ),
    }
    summaries = {}
    output_dir = root / "records/packed"
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        records = all_records[split]
        tensors = pack_split(
            records,
            max_history=args.max_history,
            category_to_id=category_to_id,
            state_count=state_count,
            candidate_features=candidate_features,
        )
        numeric = [value for value in tensors.values() if value.dtype.kind in "biufc"]
        if not all(np.isfinite(value).all() for value in numeric):
            raise ValueError(f"non-finite tensor in {split}")
        if not np.all((tensors["y_current_state"] >= 0) & (tensors["y_current_state"] < state_count)):
            raise ValueError(f"invalid state label in {split}")
        if not np.all(tensors["history_mask"].any(axis=1)):
            raise ValueError(f"empty history in {split}")
        positive_mask = tensors["event_type"] == 1
        if not np.all(positive_mask.any(axis=1)):
            raise ValueError(f"missing positive event in {split}")
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, **tensors)
        summaries[split] = {
            "records": len(records),
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "history_length_min": int(tensors["history_mask"].sum(axis=1).min()),
            "history_length_max": int(tensors["history_mask"].sum(axis=1).max()),
            "moved_fraction": float(tensors["y_moved"].mean()),
            "transition_occurred_fraction": float(
                tensors["y_transition_occurred"].mean()
            ),
            "returned_to_last_fraction": float(
                tensors["y_returned_to_last"].mean()
            ),
        }
    schema = {
        "schema_version": "p4d_packed_tensor_v1.2",
        "max_history": args.max_history,
        "state_count_including_unknown": state_count,
        "category_to_id": category_to_id,
        "region_category_to_id": region_to_id,
        "receptacle_category_to_id": receptacle_to_id,
        "metadata_world_variant_to_id": {"routine": 0, "random": 1, "static": 2},
        "metadata_grounding_quality_to_id": {
            "exact": 0,
            "role_equivalent": 1,
            "fallback": 2,
            "counterfactual": 3,
            "no_transition": 4,
        },
        "event_type_to_id": {"padding": 0, "positive_observation": 1, "candidate_inspection": 2},
        "evidence_feature_order": [
            "detector_confidence",
            "instance_match_confidence",
            "visible_fraction_estimate",
            "surface_in_frustum_fraction",
            "surface_unoccluded_fraction",
            "negative_evidence_strength",
        ],
        "observable_context_feature_order": [
            "detected_object_count",
            "observed_change_count",
            "inspected_candidate_state_count",
            "observed_region_count",
        ],
        "label_tensors": [
            "y_current_state",
            "y_moved",
            "y_transition_occurred",
            "y_returned_to_last",
        ],
        "metadata_tensors_not_for_model_input": [
            "meta_world_variant_id",
            "meta_grounding_quality_id",
            "record_id",
            "instance_uuid",
        ],
        "access_rule": "input tensors derive from query.input; y_* derives from supervision; meta_* is only for filtering and reporting",
        "pytorch_dtype_note": "cast int16 state/category ID arrays to torch.long before embedding, one_hot, or cross_entropy",
        "splits": summaries,
        "validation": {
            "finite_numeric_tensors": True,
            "valid_label_ranges": True,
            "nonempty_history": True,
            "positive_anchor_present": True,
        },
    }
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(schema, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
