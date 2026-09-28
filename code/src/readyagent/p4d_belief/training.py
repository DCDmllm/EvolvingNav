"""Training, inference and experiment orchestration utilities."""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from readyagent.p4d_belief.baselines import ClassicalBaselines
from readyagent.p4d_belief.data import (
    Catalog,
    PackedQueries,
    derive_last_state,
    load_arrays,
    subset_indices,
)
from readyagent.p4d_belief.metrics import aggregate_seed_metrics, evaluate_probabilities
from readyagent.p4d_belief.models import (
    DirectPointer,
    GRUDirectPointer,
    ModelConfig,
    P4DBelief,
    bayesian_update,
    state_nll,
)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 12
    gradient_clip: float = 1.0
    num_workers: int = 0
    routine_train_fraction: float = 0.5
    validation_subset: str = "main"
    unknown_train_boost: float = 1.0


EVALUATION_SUBSETS = {
    "routine_all": ("routine", "all"),
    "routine_exact": ("routine", "exact"),
    "routine_fallback": ("routine", "fallback"),
    "random": ("random", "all"),
    "static": ("static", "all"),
}

SPECIAL_EVALUATION_SUBSETS = (
    "routine_unknown",
    "routine_returned",
)


def evaluation_subset_indices(
    arrays: dict[str, np.ndarray], subset_name: str
) -> np.ndarray:
    if subset_name in EVALUATION_SUBSETS:
        world, quality = EVALUATION_SUBSETS[subset_name]
        return subset_indices(arrays, world=world, quality=quality)
    routine = arrays["meta_world_variant_id"] == 0
    if subset_name == "routine_unknown":
        unknown_candidates = np.flatnonzero(arrays["candidate_is_unknown"])
        if not len(unknown_candidates):
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(
            routine & (arrays["y_current_state"] == int(unknown_candidates[0]))
        )
    if subset_name == "routine_returned":
        if "y_returned_to_last" not in arrays:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(routine & arrays["y_returned_to_last"].astype(bool))
    raise ValueError(f"unknown evaluation subset: {subset_name}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_model(name: str, catalog: Catalog, config: ModelConfig) -> torch.nn.Module:
    if name == "p4d":
        return P4DBelief(catalog, config)
    if name == "direct":
        return DirectPointer(catalog, config)
    if name == "gru":
        return GRUDirectPointer(catalog, config)
    raise ValueError(name)


def main_training_indices(
    arrays: dict[str, np.ndarray],
    routine_fraction: float = 0.5,
    unknown_boost: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= routine_fraction <= 1.0:
        raise ValueError("routine_fraction must be in [0, 1]")
    if unknown_boost < 1.0:
        raise ValueError("unknown_boost must be >= 1")
    indices = subset_indices(arrays, world="main")
    world = arrays["meta_world_variant_id"][indices]
    routine = world == 0
    weights = np.zeros(len(indices), dtype=np.float64)
    if routine_fraction > 0.0:
        routine_weights = np.ones(int(routine.sum()), dtype=np.float64)
        unknown_candidates = np.flatnonzero(arrays["candidate_is_unknown"])
        if len(unknown_candidates):
            routine_targets = arrays["y_current_state"][indices[routine]]
            routine_weights[routine_targets == int(unknown_candidates[0])] *= unknown_boost
        weights[routine] = routine_fraction * routine_weights / routine_weights.sum()
    if routine_fraction < 1.0:
        weights[~routine] = (1.0 - routine_fraction) / (~routine).sum()
    return indices, weights.astype(np.float64)


def validation_indices(
    arrays: dict[str, np.ndarray], validation_subset: str
) -> np.ndarray:
    if validation_subset == "main":
        return subset_indices(arrays, world="main")
    if validation_subset in (*EVALUATION_SUBSETS, *SPECIAL_EVALUATION_SUBSETS):
        return evaluation_subset_indices(arrays, validation_subset)
    raise ValueError(f"unknown validation subset: {validation_subset}")


@torch.no_grad()
def validation_nll(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    records = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        probabilities = model(batch)["probabilities"]
        loss = state_nll(probabilities, batch["target_candidate_index"])
        total_loss += float(loss) * len(probabilities)
        records += len(probabilities)
    return total_loss / max(records, 1)


def train_model(
    *,
    name: str,
    seed: int,
    dataset_root: Path,
    catalog: Catalog,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    output_dir: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    seed_everything(seed)
    train_path = dataset_root / "records/packed/train.npz"
    val_path = dataset_root / "records/packed/val.npz"
    train_arrays = load_arrays(dataset_root, "train")
    train_indices, weights = main_training_indices(
        train_arrays,
        train_config.routine_train_fraction,
        train_config.unknown_train_boost,
    )
    val_arrays = load_arrays(dataset_root, "val")
    val_indices = validation_indices(val_arrays, train_config.validation_subset)
    train_dataset = PackedQueries(train_path, train_indices)
    val_dataset = PackedQueries(val_path, val_indices)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=sampler,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.eval_batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(name, catalog, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []
    stale = 0
    start = time.time()
    for epoch in range(train_config.max_epochs):
        model.train()
        running = 0.0
        seen = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(batch)["probabilities"]
            loss = state_nll(probabilities, batch["target_candidate_index"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss for {name}, seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
            optimizer.step()
            running += float(loss.detach()) * len(probabilities)
            seen += len(probabilities)
        train_nll = running / seen
        val_nll = validation_nll(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_nll": train_nll, "val_nll": val_nll})
        if val_nll < best_val - 1e-5:
            best_val = val_nll
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= train_config.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    torch.save(
        {
            "model": best_state,
            "model_name": name,
            "seed": seed,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "best_epoch": best_epoch,
            "best_val_nll": best_val,
        },
        checkpoint_path,
    )
    training_result = {
        "model": name,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_nll": best_val,
        "epochs_run": len(history),
        "wall_seconds": time.time() - start,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    (output_dir / "training.json").write_text(
        json.dumps(training_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return model, training_result


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    dataset: PackedQueries,
    *,
    batch_size: int,
    device: torch.device,
    hold_out_last_inspection: bool = False,
    apply_update: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    lasts: list[np.ndarray] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        checked: torch.Tensor | None = None
        strength: torch.Tensor | None = None
        coverage: torch.Tensor | None = None
        eligible: torch.Tensor | None = None
        if hold_out_last_inspection:
            inspection = (batch["event_type"] == 2) & batch["history_mask"].bool()
            positions = torch.where(
                inspection,
                torch.arange(inspection.shape[1], device=device)[None, :],
                -1,
            )
            held_index = positions.max(dim=1).values
            eligible = held_index >= 0
            safe_index = held_index.clamp_min(0)
            rows = torch.arange(len(held_index), device=device)
            checked = batch["candidate_state_id"][rows, safe_index].long().clamp_min(0)
            evidence = batch["evidence_features"][rows, safe_index].float()
            coverage = (evidence[:, 3] * evidence[:, 4]).clamp(0.0, 1.0)
            strength = evidence[:, 5].clamp(0.0, 1.0)
            batch["history_mask"] = batch["history_mask"].clone()
            batch["history_mask"][rows[eligible], held_index[eligible]] = False
        output = model(batch)
        belief = output["probabilities"]
        if apply_update and checked is not None and eligible is not None:
            updated = bayesian_update(
                belief,
                checked,
                observed=torch.zeros_like(eligible),
                confidence=strength,
                coverage=coverage,
            )
            belief = torch.where(eligible[:, None], updated, belief)
        probabilities.append(belief.cpu().numpy())
        targets.append(batch["target_candidate_index"].cpu().numpy())
        lasts.append(batch["last_candidate_index"].cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(targets), np.concatenate(lasts)


def evaluate_model(
    model: torch.nn.Module,
    *,
    dataset_root: Path,
    catalog: Catalog,
    split: str,
    device: torch.device,
    batch_size: int,
    include_update: bool,
) -> dict[str, dict[str, Any]]:
    packed_path = dataset_root / "records/packed" / f"{split}.npz"
    arrays = load_arrays(dataset_root, split)
    result: dict[str, dict[str, Any]] = {}
    for subset_name in (*EVALUATION_SUBSETS, *SPECIAL_EVALUATION_SUBSETS):
        indices = evaluation_subset_indices(arrays, subset_name)
        if not len(indices):
            continue
        dataset = PackedQueries(packed_path, indices)
        probabilities, targets, lasts = predict_model(
            model, dataset, batch_size=batch_size, device=device
        )
        result[subset_name] = evaluate_probabilities(
            probabilities, targets, lasts, catalog.region_instance_ids
        )
        if include_update:
            held_prior, held_targets, held_lasts = predict_model(
                model,
                dataset,
                batch_size=batch_size,
                device=device,
                hold_out_last_inspection=True,
                apply_update=False,
            )
            updated, _, _ = predict_model(
                model,
                dataset,
                batch_size=batch_size,
                device=device,
                hold_out_last_inspection=True,
                apply_update=True,
            )
            result[f"{subset_name}_heldout_prior"] = evaluate_probabilities(
                held_prior, held_targets, held_lasts, catalog.region_instance_ids
            )
            result[f"{subset_name}_bayes_update"] = evaluate_probabilities(
                updated, held_targets, held_lasts, catalog.region_instance_ids
            )
    return result


def evaluate_classical_baselines(
    dataset_root: Path, catalog: Catalog, split: str = "test"
) -> dict[str, dict[str, dict[str, Any]]]:
    train = load_arrays(dataset_root, "train")
    train_indices = subset_indices(train, world="main")
    train_last = derive_last_state(train)
    baseline = ClassicalBaselines(
        state_count=catalog.state_count,
        category_count=len(catalog.schema["category_to_id"]),
    )
    baseline.fit(
        target_category=train["target_category_id"][train_indices].astype(np.int64),
        target_state=train["y_current_state"][train_indices].astype(np.int64),
        last_state=train_last[train_indices],
        weekday=train["query_weekday_id"][train_indices].astype(np.int64),
        time_sin_cos=train["query_time_of_day_sin_cos"][train_indices],
    )
    test = load_arrays(dataset_root, split)
    test_last = derive_last_state(test)
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    for method in ("last_seen", "object_frequency", "time_frequency", "markov"):
        method_results: dict[str, dict[str, Any]] = {}
        for subset_name in (*EVALUATION_SUBSETS, *SPECIAL_EVALUATION_SUBSETS):
            indices = evaluation_subset_indices(test, subset_name)
            if not len(indices):
                continue
            probabilities = baseline.predict(
                method,
                target_category=test["target_category_id"][indices].astype(np.int64),
                last_state=test_last[indices],
                weekday=test["query_weekday_id"][indices].astype(np.int64),
                time_sin_cos=test["query_time_of_day_sin_cos"][indices],
            )
            targets = test["y_current_state"][indices].astype(np.int64)
            method_results[subset_name] = evaluate_probabilities(
                probabilities, targets, test_last[indices], catalog.region_instance_ids
            )
        all_results[method] = method_results
    return all_results


def summarize_neural(seed_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {name: aggregate_seed_metrics(results) for name, results in seed_results.items()}
