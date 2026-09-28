"""Leakage-safe loader for packed P4D-HSSD query records."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


WORLD_NAMES = {0: "routine", 1: "random", 2: "static"}
QUALITY_NAMES = {0: "exact", 1: "fallback", 2: "counterfactual", 3: "no_transition"}

# This is the complete allowlist passed to neural models. In particular, no
# meta_world_variant_id, meta_grounding_quality_id, activity or supervision field
# is included.
MODEL_INPUT_KEYS = (
    "event_type",
    "event_time_days",
    "observed_state_id",
    "candidate_state_id",
    "evidence_features",
    "history_mask",
    "candidate_state_ids",
    "candidate_mask",
    "target_category_id",
    "query_time_days",
    "query_time_of_day_sin_cos",
    "query_weekday_id",
    "elapsed_since_last_positive_days",
)

OPTIONAL_MODEL_INPUT_KEYS = (
    "context_time_days",
    "context_category_counts",
    "context_observation_features",
    "context_mask",
)

DERIVED_MODEL_INPUT_KEYS = ("target_instance_id",)


@dataclass(frozen=True)
class Catalog:
    """Static scene catalog used by the shared semantic candidate encoder."""

    region_category: torch.Tensor
    receptacle_category: torch.Tensor
    center_xyz: torch.Tensor
    is_unknown: torch.Tensor
    region_instance_ids: tuple[str, ...]
    schema: dict[str, Any]

    @property
    def state_count(self) -> int:
        return int(self.region_category.shape[0])


class PackedQueries(Dataset[dict[str, torch.Tensor]]):
    """Selected packed records with metadata kept outside model inputs."""

    def __init__(self, packed_path: Path, indices: np.ndarray | None = None) -> None:
        data = np.load(packed_path, allow_pickle=False)
        self.arrays = {key: data[key] for key in data.files}
        instance_values = sorted(str(value) for value in np.unique(self.arrays["instance_uuid"]))
        instance_to_id = {value: index for index, value in enumerate(instance_values)}
        self.target_instance_id = np.asarray(
            [instance_to_id[str(value)] for value in self.arrays["instance_uuid"]],
            dtype=np.int64,
        )
        total = len(self.arrays["y_current_state"])
        self.indices = (
            np.arange(total, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.last_state = derive_last_state(self.arrays)
        self.target_candidate_index = map_state_to_candidate_index(
            self.arrays["candidate_state_ids"], self.arrays["y_current_state"]
        )
        self.last_candidate_index = map_state_to_candidate_index(
            self.arrays["candidate_state_ids"], self.last_state
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        result: dict[str, torch.Tensor] = {}
        for key in MODEL_INPUT_KEYS:
            result[key] = torch.as_tensor(self.arrays[key][index])
        for key in OPTIONAL_MODEL_INPUT_KEYS:
            if key in self.arrays:
                result[key] = torch.as_tensor(self.arrays[key][index])
        result["target_instance_id"] = torch.tensor(
            self.target_instance_id[index], dtype=torch.long
        )
        result["last_state"] = torch.tensor(self.last_state[index], dtype=torch.long)
        result["last_candidate_index"] = torch.tensor(
            self.last_candidate_index[index], dtype=torch.long
        )
        result["target_candidate_index"] = torch.tensor(
            self.target_candidate_index[index], dtype=torch.long
        )
        result["y_current_state"] = torch.tensor(
            self.arrays["y_current_state"][index], dtype=torch.long
        )
        result["y_persist"] = torch.tensor(
            self.last_state[index] == self.arrays["y_current_state"][index],
            dtype=torch.float32,
        )
        result["source_index"] = torch.tensor(index, dtype=torch.long)
        return result


def load_catalog(dataset_root: Path) -> Catalog:
    packed = dataset_root / "records/packed"
    schema = json.loads((packed / "feature_schema.json").read_text(encoding="utf-8"))
    with np.load(packed / "train.npz", allow_pickle=False) as data:
        region = torch.from_numpy(data["candidate_region_category_id"].astype(np.int64))
        receptacle = torch.from_numpy(
            data["candidate_receptacle_category_id"].astype(np.int64)
        )
        center = torch.from_numpy(data["candidate_center_xyz"].astype(np.float32))
        unknown = torch.from_numpy(data["candidate_is_unknown"].astype(np.bool_))
        instance_values = sorted(str(value) for value in np.unique(data["instance_uuid"]))
    schema["instance_uuid_to_id"] = {
        value: index for index, value in enumerate(instance_values)
    }
    candidate_json = json.loads(
        (dataset_root / "scene/candidate_states.json").read_text(encoding="utf-8")
    )
    states = sorted(candidate_json["states"], key=lambda row: row["state_id"])
    if len(states) != len(region):
        raise ValueError("candidate catalog and packed arrays have different lengths")
    region_ids = tuple(str(row["region_id"]) for row in states)
    return Catalog(region, receptacle, center, unknown, region_ids, schema)


def derive_last_state(arrays: dict[str, np.ndarray]) -> np.ndarray:
    positive = (arrays["event_type"] == 1) & arrays["history_mask"]
    positions = np.where(positive, np.arange(positive.shape[1])[None, :], -1)
    last_index = positions.max(axis=1)
    if np.any(last_index < 0):
        raise ValueError("every query must contain a positive observation anchor")
    last_state = arrays["observed_state_id"][np.arange(len(last_index)), last_index]
    if np.any(last_state < 0):
        raise ValueError("positive observation has an invalid state")
    return last_state.astype(np.int64)


def map_state_to_candidate_index(candidates: np.ndarray, states: np.ndarray) -> np.ndarray:
    matches = candidates == np.asarray(states)[:, None]
    if not np.all(matches.any(axis=1)):
        raise ValueError("target or last state is absent from candidate set")
    return matches.argmax(axis=1).astype(np.int64)


def subset_indices(
    arrays: dict[str, np.ndarray],
    *,
    world: str,
    quality: str = "all",
) -> np.ndarray:
    reverse_world = {value: key for key, value in WORLD_NAMES.items()}
    if world == "main":
        # Main training/validation excludes semantically-fallback transitions:
        # Routine exact/no-transition plus all Static records.
        keep = (
            ((arrays["meta_world_variant_id"] == reverse_world["routine"])
             & np.isin(arrays["meta_grounding_quality_id"], (0, 3)))
            | (arrays["meta_world_variant_id"] == reverse_world["static"])
        )
    else:
        keep = arrays["meta_world_variant_id"] == reverse_world[world]
    if quality == "exact":
        # Exact Routine includes unchanged queries, for which grounding is not
        # applicable and is explicitly labelled no_transition.
        keep &= np.isin(arrays["meta_grounding_quality_id"], (0, 3))
    elif quality == "fallback":
        keep &= arrays["meta_grounding_quality_id"] == 1
    elif quality != "all":
        raise ValueError(f"unknown quality subset: {quality}")
    return np.flatnonzero(keep)


def load_arrays(dataset_root: Path, split: str) -> dict[str, np.ndarray]:
    path = dataset_root / "records/packed" / f"{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def audit_training_contract(dataset_root: Path) -> dict[str, Any]:
    """Fail fast on label, chronology, candidate and leakage assumptions."""

    report: dict[str, Any] = {
        "splits": {},
        "model_input_keys": list(
            MODEL_INPUT_KEYS + OPTIONAL_MODEL_INPUT_KEYS + DERIVED_MODEL_INPUT_KEYS
        ),
    }
    forbidden = {
        "meta_world_variant_id",
        "meta_grounding_quality_id",
        "activity_type_id",
        "activity_history",
        "y_current_state",
        "y_moved",
    }
    report["forbidden_keys_absent_from_model_input"] = not bool(
        forbidden.intersection(
            MODEL_INPUT_KEYS + OPTIONAL_MODEL_INPUT_KEYS + DERIVED_MODEL_INPUT_KEYS
        )
    )
    total_disagreement = 0
    scheduled_events: dict[str, list[float]] = {}
    schedule_path = dataset_root / "schedules/routine_events.jsonl"
    if schedule_path.exists():
        for line in schedule_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            scheduled_events.setdefault(str(event["instance_uuid"]), []).append(
                float(event["event_time_s"]) / 86400.0
            )
        for timestamps in scheduled_events.values():
            timestamps.sort()
    total_return_to_last = 0
    for split in ("train", "val", "test"):
        arrays = load_arrays(dataset_root, split)
        last = derive_last_state(arrays)
        target_index = map_state_to_candidate_index(
            arrays["candidate_state_ids"], arrays["y_current_state"]
        )
        last_index = map_state_to_candidate_index(arrays["candidate_state_ids"], last)
        valid_time = arrays["event_time_days"] <= arrays["query_time_days"][:, None] + 1e-6
        valid_time |= ~arrays["history_mask"]
        persist = last == arrays["y_current_state"]
        disagreement = int(np.sum(arrays["y_moved"] != ~persist))
        total_disagreement += disagreement
        routine = arrays["meta_world_variant_id"] == 0
        routine_indices = np.flatnonzero(routine)
        hidden_event_count = np.zeros(len(routine_indices), dtype=np.int64)
        for output_index, record_index in enumerate(routine_indices):
            positive = (
                (arrays["event_type"][record_index] == 1)
                & arrays["history_mask"][record_index]
            )
            last_time = float(
                arrays["event_time_days"][record_index, np.flatnonzero(positive)[-1]]
            )
            query_time = float(arrays["query_time_days"][record_index])
            timestamps = scheduled_events.get(str(arrays["instance_uuid"][record_index]), [])
            hidden_event_count[output_index] = bisect.bisect_right(
                timestamps, query_time + 1e-6
            ) - bisect.bisect_right(timestamps, last_time + 1e-6)
        routine_persist = persist[routine_indices]
        returns = int(np.sum((hidden_event_count > 0) & routine_persist))
        total_return_to_last += returns
        report["splits"][split] = {
            "records": int(len(last)),
            "target_present": bool(np.all(target_index >= 0)),
            "last_present": bool(np.all(last_index >= 0)),
            "no_future_history": bool(np.all(valid_time)),
            "persist_fraction": float(np.mean(persist)),
            "moved_vs_nonpersist_disagreement": disagreement,
            "routine_no_hidden_event_since_last_positive": int(
                np.sum(hidden_event_count == 0)
            ),
            "routine_event_and_return_to_last": returns,
            "routine_event_and_changed": int(
                np.sum((hidden_event_count > 0) & ~routine_persist)
            ),
            "unknown_targets": int(
                np.sum(arrays["y_current_state"] == int(arrays["candidate_is_unknown"].argmax()))
            ),
        }
    report["moved_label_semantics"] = (
        "y_moved is net current-state difference from last positive, not whether any "
        "hidden transition occurred"
    )
    report["routine_return_to_last_records"] = total_return_to_last
    report["passed"] = bool(
        report["forbidden_keys_absent_from_model_input"]
        and all(
            row["target_present"] and row["last_present"] and row["no_future_history"]
            for row in report["splits"].values()
        )
    )
    return report
