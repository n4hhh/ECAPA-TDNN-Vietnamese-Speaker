"""Run exactly one cached-Fbank ECAPA/AAM epoch and fixed validation v1."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import speechbrain
import torch
import torch.nn.functional as F

from scripts.smoke_test_aam_training_cuda import load_trainable_encoder
from src.aam_training import (
    AAMSoftmax,
    PRETRAINED_MODEL_ID,
    SEED,
    apply_batchnorm_policy,
    assert_batchnorm_running_state_exact,
    batchnorm_running_state,
    build_adamw_optimizer,
    capture_rng_state,
    create_train_only_smoke_dataset,
    file_sha256,
    iter_microbatches,
    restore_rng_state,
    round_robin_reorder,
    to_cpu_tree,
    validate_optimizer_coverage,
    values_exactly_equal,
)
from src.cached_fbank_dataset import (
    CachedFbankDataset,
    create_cached_fbank_dataloader,
    create_cached_fbank_training_dataloader,
)
from src.cached_fbank_samplers import HybridShardAwareSpeakerBatchSampler
from src.ecapa_one_epoch_pilot import (
    AAM_CONFIGURATION,
    ACCUMULATION_STEPS,
    ACTIVE_SHARD_WINDOW,
    AMP_POLICY,
    BASELINE_INTERPOLATED_EER,
    BATCHNORM_POLICY,
    EXPECTED_TRIAL_SHA256,
    LOGICAL_BATCH_SIZE,
    LOGGING_INTERVAL,
    NUM_LOGICAL_BATCHES,
    OPTIMIZER_CONFIGURATION,
    PHYSICAL_MICROBATCH_SIZE,
    PILOT_RUNTIME_SCHEMA_NAME,
    PILOT_RUNTIME_SCHEMA_VERSION,
    PILOT_SCHEMA_NAME,
    PILOT_SCHEMA_VERSION,
    QUALITY_TOLERANCE,
    ROLLING_CHECKPOINT_STEPS,
    SAMPLER_CONFIGURATION,
    SAMPLES_PER_SPEAKER,
    SPEAKERS_PER_BATCH,
    TOTAL_SAMPLE_SELECTIONS,
    LoggingWindow,
    OneEpochController,
    append_jsonl,
    atomic_json,
    atomic_save_pilot_checkpoint,
    baseline_comparison,
    checkpoint_state_roundtrip_exact,
    hash_explicit_paths,
    hash_mapping_digest,
    load_pilot_checkpoint,
    require_trial_hash,
    resume_cursor,
    rolling_checkpoint_due,
    select_best_checkpoint,
    summarize_finite_values,
    validate_pilot_checkpoint,
    validate_runtime_payload,
    validate_training_log_record,
    validate_validation_metrics_payload,
    validation_forward_batch,
)
from src.verification_baseline import (
    numeric_summary,
    score_trials,
    validate_saved_artifacts,
)
from src.verification_metrics import calculate_eer
from src.verification_trials import (
    read_trials,
    validate_trials,
    validate_trials_against_metadata,
)


OUTPUT_RELATIVE = Path("outputs/ecapa_aam_one_epoch_pilot_v1")
OUTPUT_DIR = PROJECT_ROOT / OUTPUT_RELATIVE
TRAIN_LOG_PATH = OUTPUT_DIR / "training_log.jsonl"
LOSS_PATH = OUTPUT_DIR / "logical_losses_epoch_000.json"
LAST_CHECKPOINT_PATH = OUTPUT_DIR / "last.pt"
EPOCH_CHECKPOINT_PATH = OUTPUT_DIR / "epoch_000.pt"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "best.pt"
EMBEDDING_PATH = OUTPUT_DIR / "validation_embeddings_epoch_000.pt"
SCORE_PATH = OUTPUT_DIR / "validation_scores_epoch_000.pt"
METRICS_PATH = OUTPUT_DIR / "validation_metrics_epoch_000.json"
VALIDATION_RUNTIME_PATH = OUTPUT_DIR / "validation_runtime_epoch_000.json"
RUNTIME_PATH = OUTPUT_DIR / "pilot_runtime.json"
FAILURE_PATH = OUTPUT_DIR / "pilot_failure.json"

CACHE_RELATIVE = Path("outputs/fbank_cache_v1")
CACHE_DIR = PROJECT_ROOT / CACHE_RELATIVE
TRAIN_INDEX_RELATIVE = CACHE_RELATIVE / "train_feature_index_v1.csv"
VALIDATION_INDEX_RELATIVE = CACHE_RELATIVE / "validation_feature_index_v1.csv"
CACHE_CONFIG_RELATIVE = CACHE_RELATIVE / "fbank_cache_config_v1.json"
VALIDATION_MANIFEST_RELATIVE = Path(
    "manifests/portable/validation_manifest_v1.csv"
)
TRIAL_RELATIVE = Path("manifests/verification/validation_trials_v1.csv")
TRIAL_CONFIG_RELATIVE = Path(
    "manifests/verification/validation_trials_config_v1.json"
)
VALIDATION_ROWS = 8504
VALIDATION_SPEAKERS = 100
POSITIVE_TRIALS = 9764
NEGATIVE_TRIALS = 9764
TOTAL_TRIALS = 19528
VALIDATION_BATCH_SIZE = 32
TRAIN_LRU_SHARDS = 8

BASELINE_METRICS = {
    "interpolated_eer": BASELINE_INTERPOLATED_EER,
    "interpolated_eer_percentage": BASELINE_INTERPOLATED_EER * 100.0,
    "empirical_threshold": 0.31165990233421326,
    "empirical_far": 0.11665301106104056,
    "empirical_frr": 0.11665301106104056,
    "empirical_average_error": 0.11665301106104056,
    "same_speaker_scores": {
        "mean": 0.4905432510233207,
        "standard_deviation": 0.1512383760047422,
        "median": 0.5050629079341888,
    },
    "different_speaker_scores": {
        "mean": 0.16336830629613902,
        "standard_deviation": 0.11727968425816838,
        "median": 0.15320874005556107,
    },
    "score_range": [-0.20446446537971497, 0.935106635093689],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_error(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    for spelling in (str(PROJECT_ROOT), str(PROJECT_ROOT).lower()):
        message = message.replace(spelling, ".")
    return message


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def ensure_fresh_runtime_targets() -> None:
    targets = (
        TRAIN_LOG_PATH,
        LOSS_PATH,
        LAST_CHECKPOINT_PATH,
        EPOCH_CHECKPOINT_PATH,
        BEST_CHECKPOINT_PATH,
        EMBEDDING_PATH,
        SCORE_PATH,
        METRICS_PATH,
        VALIDATION_RUNTIME_PATH,
        RUNTIME_PATH,
        FAILURE_PATH,
    )
    existing = [
        path.resolve().relative_to(PROJECT_ROOT).as_posix()
        for path in targets
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "pilot runtime targets already exist; refusing to overwrite: "
            + ", ".join(existing)
        )


def relative_string(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def strict_shard_relative(value: str, split: str) -> str:
    pure = PurePosixPath(value)
    if (
        split not in {"train", "validation"}
        or len(pure.parts) != 2
        or pure.parts[0] != split
        or len(pure.parts[1]) != len("shard_00000.pt")
        or not pure.parts[1].startswith("shard_")
        or not pure.parts[1].endswith(".pt")
        or not pure.parts[1][6:11].isdigit()
    ):
        raise ValueError(f"non-allowlisted {split} shard path: {value!r}")
    return value


def read_validation_shards_from_index() -> tuple[str, ...]:
    path = PROJECT_ROOT / VALIDATION_INDEX_RELATIVE
    shards: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "feature_shard_path",
            "final_split",
            "speaker_label",
            "relative_audio_path",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("validation cache index schema is incomplete")
        for raw in reader:
            rows += 1
            if raw["final_split"] != "validation" or raw["speaker_label"] != "-1":
                raise ValueError("validation cache index split/label invariant failed")
            relative = strict_shard_relative(
                raw["feature_shard_path"], "validation"
            )
            shards.add((CACHE_RELATIVE / Path(*PurePosixPath(relative).parts)).as_posix())
    if rows != VALIDATION_ROWS or not shards:
        raise ValueError("validation cache index count/shard invariant failed")
    return tuple(sorted(shards))


def validate_sampler_plan(
    dataset: CachedFbankDataset, batches: Sequence[Sequence[int]]
) -> dict[str, Any]:
    if len(batches) != NUM_LOGICAL_BATCHES:
        raise ValueError("epoch 0 sampler must contain exactly 1,000 batches")
    rows_identity = dataset.rows
    selections = 0
    selected_indexes: Counter[int] = Counter()
    selected_speakers: Counter[str] = Counter()
    for position, batch in enumerate(batches):
        if len(batch) != LOGICAL_BATCH_SIZE or len(set(batch)) != LOGICAL_BATCH_SIZE:
            raise ValueError(
                f"sampler batch {position} is not 32 unique Dataset indexes"
            )
        if any(type(index) is not int or not 0 <= index < len(dataset) for index in batch):
            raise ValueError(f"sampler batch {position} contains an invalid index")
        speakers = [dataset.rows[index].speaker_id for index in batch]
        labels = [dataset.rows[index].speaker_label for index in batch]
        counts = Counter(speakers)
        if (
            len(counts) != SPEAKERS_PER_BATCH
            or set(counts.values()) != {SAMPLES_PER_SPEAKER}
            or any(not 0 <= label <= 487 for label in labels)
        ):
            raise ValueError(f"sampler batch {position} is not exact P16K2")
        for index in batch:
            selected_indexes[index] += 1
            selected_speakers[dataset.rows[index].speaker_id] += 1
        selections += len(batch)
    if selections != TOTAL_SAMPLE_SELECTIONS:
        raise ValueError("sampler plan does not contain exactly 32,000 selections")
    if dataset.rows is not rows_identity:
        raise AssertionError("sampler plan mutated the Dataset rows")
    canonical = json.dumps([list(batch) for batch in batches], separators=(",", ":"))
    return {
        "logical_batches": len(batches),
        "sample_selections": selections,
        "unique_dataset_indexes_selected": len(selected_indexes),
        "repeated_sample_selections": sum(
            count - 1 for count in selected_indexes.values()
        ),
        "speakers_selected": len(selected_speakers),
        "speaker_selection_minimum": min(selected_speakers.values()),
        "speaker_selection_median": statistics.median(selected_speakers.values()),
        "speaker_selection_maximum": max(selected_speakers.values()),
        "plan_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "all_batches_exact_p16k2": True,
        "duplicate_dataset_index_within_batch": False,
    }


def selected_train_shards(
    dataset: CachedFbankDataset, batches: Sequence[Sequence[int]]
) -> tuple[str, ...]:
    relative_shards = {
        strict_shard_relative(dataset.rows[index].feature_shard_path, "train")
        for batch in batches
        for index in batch
    }
    return tuple(
        sorted(
            (CACHE_RELATIVE / Path(*PurePosixPath(relative).parts)).as_posix()
            for relative in relative_shards
        )
    )


def protected_path_set(
    train_shards: Sequence[str], validation_shards: Sequence[str]
) -> tuple[str, ...]:
    core = (
        CACHE_CONFIG_RELATIVE.as_posix(),
        TRAIN_INDEX_RELATIVE.as_posix(),
        VALIDATION_INDEX_RELATIVE.as_posix(),
        VALIDATION_MANIFEST_RELATIVE.as_posix(),
        TRIAL_RELATIVE.as_posix(),
        TRIAL_CONFIG_RELATIVE.as_posix(),
    )
    paths = tuple(sorted(set(core) | set(train_shards) | set(validation_shards)))
    if len(paths) != len(set(paths)):
        raise AssertionError("protected path set contains duplicates")
    return paths


def train_identity(
    protected_hashes: Mapping[str, str], train_shards: Sequence[str]
) -> dict[str, Any]:
    allowed = {
        CACHE_CONFIG_RELATIVE.as_posix(),
        TRAIN_INDEX_RELATIVE.as_posix(),
        *train_shards,
    }
    selected = {
        path: digest for path, digest in protected_hashes.items() if path in allowed
    }
    if set(selected) != allowed:
        raise ValueError("train checkpoint identity is incomplete")
    return {
        "files": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(selected.items())
        ],
        "set_sha256": hash_mapping_digest(selected),
    }


def validate_logical_batch(
    dataset: CachedFbankDataset,
    batch: Mapping[str, Any],
    planned_indexes: Sequence[int],
) -> dict[str, Any]:
    reordered = round_robin_reorder(
        batch,
        speakers_per_batch=SPEAKERS_PER_BATCH,
        samples_per_speaker=SAMPLES_PER_SPEAKER,
    )
    features = reordered["fbank"]
    labels = reordered["speaker_label"]
    indexes = reordered["dataset_index"]
    speakers = reordered["speaker_id"]
    paths = reordered["relative_audio_path"]
    if (
        tuple(features.shape) != (LOGICAL_BATCH_SIZE, 301, 80)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not bool(torch.isfinite(features).all().item())
    ):
        raise ValueError("logical cached features must be finite CPU [32,301,80]")
    if (
        labels.dtype != torch.long
        or tuple(labels.shape) != (LOGICAL_BATCH_SIZE,)
        or bool(((labels < 0) | (labels > 487)).any().item())
    ):
        raise ValueError("logical train labels must be int64 in 0..487")
    if sorted(indexes.tolist()) != sorted(planned_indexes):
        raise ValueError("DataLoader batch identity differs from sampler plan")
    if reordered["final_split"] != ["train"] * LOGICAL_BATCH_SIZE:
        raise ValueError("logical batch contains non-train data")
    if len(set(indexes.tolist())) != LOGICAL_BATCH_SIZE:
        raise ValueError("logical batch contains duplicate Dataset indexes")
    label_to_speaker: dict[int, str] = {}
    for position, index in enumerate(indexes.tolist()):
        row = dataset.rows[index]
        observed = (
            int(labels[position].item()),
            speakers[position],
            paths[position],
        )
        expected = (row.speaker_label, row.speaker_id, row.relative_audio_path)
        if observed != expected:
            raise ValueError("logical batch aligned metadata changed during reordering")
        prior = label_to_speaker.setdefault(observed[0], observed[1])
        if prior != observed[1]:
            raise ValueError("speaker label maps to multiple speaker IDs")
    microbatches = list(iter_microbatches(reordered, PHYSICAL_MICROBATCH_SIZE))
    if len(microbatches) != ACCUMULATION_STEPS or any(
        len(set(microbatch["speaker_id"])) != PHYSICAL_MICROBATCH_SIZE
        for microbatch in microbatches
    ):
        raise ValueError("round-robin physical microbatches are not four speakers")
    return reordered


def finite_gradient_norm(
    parameters: Sequence[tuple[str, torch.nn.Parameter]], group_name: str
) -> float:
    total: torch.Tensor | None = None
    count = 0
    for name, parameter in parameters:
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            raise ValueError(f"{group_name} parameter {name!r} has no gradient")
        square_sum = gradient.detach().float().square().sum()
        total = square_sum if total is None else total + square_sum
        count += 1
    if count == 0 or total is None:
        raise ValueError(f"{group_name} has no trainable gradients")
    numeric = float(total.item())
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{group_name} gradients are non-finite or zero")
    return math.sqrt(numeric)


def optimizer_all_parameter_steps(
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    result: list[int] = []
    for group in optimizer.param_groups:
        parameters = list(group["params"])
        if not parameters:
            raise ValueError("optimizer group is empty")
        steps: list[int] = []
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            if not isinstance(state, Mapping) or "step" not in state:
                raise ValueError("an AdamW parameter has no step counter")
            step = state["step"]
            numeric = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
            steps.append(numeric)
        if len(set(steps)) != 1:
            raise RuntimeError("AdamW parameter step counters disagree within a group")
        result.append(steps[0])
    if len(result) != 2:
        raise ValueError("optimizer must have exactly two representative steps")
    return result[0], result[1]


def create_training_objects(device: torch.device) -> dict[str, Any]:
    mean_var_norm, embedding_model, loader_proof = load_trainable_encoder(device)
    aam = AAMSoftmax(seed=SEED).to(device)
    if tuple(aam.weight.shape) != (488, 192):
        raise AssertionError("AAM weight shape invariant failed")
    optimizer = build_adamw_optimizer(embedding_model, aam)
    validate_optimizer_coverage(optimizer, embedding_model, aam)
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128.0)
    if not scaler.is_enabled():
        raise RuntimeError("GradScaler must be enabled")
    if any(parameter.requires_grad for parameter in mean_var_norm.parameters()):
        raise AssertionError("mean_var_norm must be excluded from optimization")
    return {
        "mean_var_norm": mean_var_norm,
        "embedding_model": embedding_model,
        "aam": aam,
        "optimizer": optimizer,
        "scaler": scaler,
        "loader_proof": loader_proof,
    }


def run_optimizer_update(
    objects: Mapping[str, Any],
    logical_batch: Mapping[str, Any],
    *,
    global_step: int,
    optimizer_hook_count: dict[str, int],
    stage: dict[str, str],
) -> dict[str, Any]:
    mean_var_norm = objects["mean_var_norm"]
    embedding_model = objects["embedding_model"]
    aam = objects["aam"]
    optimizer = objects["optimizer"]
    scaler = objects["scaler"]
    device = next(embedding_model.parameters()).device

    stage["value"] = f"train step {global_step}: reapply BatchNorm policy"
    apply_batchnorm_policy(embedding_model)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    stage["value"] = f"train step {global_step}: optimizer zero_grad"
    optimizer.zero_grad(set_to_none=True)
    logical_loss = 0.0
    microbatch_count = 0
    for number, microbatch in enumerate(
        iter_microbatches(logical_batch, PHYSICAL_MICROBATCH_SIZE), start=1
    ):
        microbatch_count += 1
        stage["value"] = f"train step {global_step}: microbatch {number} transfer"
        features_cpu = microbatch["fbank"]
        labels_cpu = microbatch["speaker_label"]
        if (
            tuple(features_cpu.shape) != (PHYSICAL_MICROBATCH_SIZE, 301, 80)
            or features_cpu.dtype != torch.float32
            or features_cpu.device.type != "cpu"
            or not bool(torch.isfinite(features_cpu).all().item())
        ):
            raise ValueError("physical cached features are invalid")
        features = features_cpu.to(device=device, dtype=torch.float32)
        labels = labels_cpu.to(device=device, dtype=torch.long)
        lengths = torch.ones(
            PHYSICAL_MICROBATCH_SIZE, device=device, dtype=torch.float32
        )
        stage["value"] = f"train step {global_step}: microbatch {number} normalize"
        with torch.no_grad():
            normalized = mean_var_norm(features, lengths)
        if (
            tuple(normalized.shape) != (PHYSICAL_MICROBATCH_SIZE, 301, 80)
            or not bool(torch.isfinite(normalized).all().item())
        ):
            raise ValueError("normalized train features are invalid")
        stage["value"] = f"train step {global_step}: microbatch {number} ECAPA"
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            raw_embedding = embedding_model(normalized, lengths)
        if (
            tuple(raw_embedding.shape)
            != (PHYSICAL_MICROBATCH_SIZE, 1, 192)
            or not bool(torch.isfinite(raw_embedding).all().item())
        ):
            raise ValueError("raw train embedding is invalid")
        embedding = raw_embedding.squeeze(1)
        if tuple(embedding.shape) != (PHYSICAL_MICROBATCH_SIZE, 192):
            raise ValueError("squeezed train embedding is invalid")
        stage["value"] = f"train step {global_step}: microbatch {number} AAM/loss"
        with torch.cuda.amp.autocast(enabled=False):
            logits = aam(embedding.float(), labels)
            loss = (
                F.cross_entropy(logits.float(), labels, reduction="sum")
                / LOGICAL_BATCH_SIZE
            )
        if (
            logits.dtype != torch.float32
            or tuple(logits.shape) != (PHYSICAL_MICROBATCH_SIZE, 488)
            or not bool(torch.isfinite(logits).all().item())
            or loss.dtype != torch.float32
            or not bool(torch.isfinite(loss).item())
        ):
            raise ValueError("AAM logits or logical loss are invalid")
        logical_loss += float(loss.detach().item())
        stage["value"] = f"train step {global_step}: microbatch {number} backward"
        scaler.scale(loss).backward()
        del (
            features,
            labels,
            lengths,
            normalized,
            raw_embedding,
            embedding,
            logits,
            loss,
        )
    if microbatch_count != ACCUMULATION_STEPS:
        raise AssertionError("logical update did not process exactly eight microbatches")
    stage["value"] = f"train step {global_step}: unscale/gradient validation"
    scaler.unscale_(optimizer)
    ecapa_gradient_norm = finite_gradient_norm(
        list(embedding_model.named_parameters()), "ECAPA"
    )
    aam_gradient_norm = finite_gradient_norm(
        list(aam.named_parameters()), "AAM classifier"
    )
    stage["value"] = f"train step {global_step}: GradScaler/AdamW update"
    hook_before = optimizer_hook_count["value"]
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if optimizer_hook_count["value"] != hook_before + 1:
        raise RuntimeError("GradScaler skipped the AdamW optimizer update")
    ecapa_state_step, aam_state_step = optimizer_all_parameter_steps(optimizer)
    if (ecapa_state_step, aam_state_step) != (global_step, global_step):
        raise RuntimeError("AdamW state counters did not advance exactly once")
    scale_after = float(scaler.get_scale())
    if (
        not math.isfinite(scale_before)
        or not math.isfinite(scale_after)
        or scale_after <= 0.0
        or scale_after < scale_before
    ):
        raise RuntimeError("GradScaler scale indicates an invalid/skipped update")
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    if not math.isfinite(logical_loss) or duration <= 0.0:
        raise RuntimeError("logical loss or step duration is invalid")
    return {
        "logical_loss": logical_loss,
        "ecapa_gradient_norm": ecapa_gradient_norm,
        "aam_gradient_norm": aam_gradient_norm,
        "grad_scaler_scale_before": scale_before,
        "grad_scaler_scale_after": scale_after,
        "duration_seconds": duration,
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "cuda_max_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "optimizer_step_hook_count": optimizer_hook_count["value"],
        "ecapa_adamw_state_step": ecapa_state_step,
        "aam_adamw_state_step": aam_state_step,
    }


def build_checkpoint(
    objects: Mapping[str, Any],
    *,
    train_cache_identity: Mapping[str, Any],
    losses: Sequence[float],
    global_step: int,
    reason: str,
    validation_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    next_epoch, next_position = resume_cursor(global_step)
    checkpoint = {
        "schema_name": PILOT_SCHEMA_NAME,
        "schema_version": PILOT_SCHEMA_VERSION,
        "pretrained_model_identifier": PRETRAINED_MODEL_ID,
        "embedding_model_state_dict": to_cpu_tree(
            objects["embedding_model"].state_dict()
        ),
        "mean_var_norm_state_dict": to_cpu_tree(
            objects["mean_var_norm"].state_dict()
        ),
        "aam_classifier_state_dict": to_cpu_tree(objects["aam"].state_dict()),
        "optimizer_state_dict": to_cpu_tree(objects["optimizer"].state_dict()),
        "grad_scaler_state_dict": to_cpu_tree(objects["scaler"].state_dict()),
        "epoch": 0,
        "next_epoch": next_epoch,
        "next_logical_batch_position": next_position,
        "global_optimizer_step": global_step,
        "sampler_configuration": dict(SAMPLER_CONFIGURATION),
        "physical_microbatch_size": PHYSICAL_MICROBATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "aam_configuration": dict(AAM_CONFIGURATION),
        "optimizer_configuration": [dict(group) for group in OPTIMIZER_CONFIGURATION],
        "batchnorm_policy": dict(BATCHNORM_POLICY),
        "amp_policy": dict(AMP_POLICY),
        "train_cache_identity": dict(train_cache_identity),
        "rng_state": capture_rng_state(),
        "completed_training_loss_summary": summarize_finite_values(
            losses, "completed logical losses"
        ),
        "checkpoint_creation_reason": reason,
        "validation_selection": (
            dict(validation_selection) if validation_selection is not None else None
        ),
    }
    validate_pilot_checkpoint(checkpoint)
    return checkpoint


def save_and_validate_checkpoint(
    checkpoint: Mapping[str, Any],
    path: Path,
    *,
    expected_step: int,
    expected_reason: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    atomic_save_pilot_checkpoint(checkpoint, path)
    loaded = load_pilot_checkpoint(path)
    if (
        loaded["global_optimizer_step"] != expected_step
        or loaded["checkpoint_creation_reason"] != expected_reason
    ):
        raise RuntimeError("checkpoint readable metadata does not match expectation")
    expected_cursor = resume_cursor(expected_step)
    if (
        loaded["next_epoch"],
        loaded["next_logical_batch_position"],
    ) != expected_cursor:
        raise RuntimeError("checkpoint readable resume cursor is wrong")
    result = {
        "relative_path": relative_string(path),
        "size_bytes": path.stat().st_size,
        "global_optimizer_step": expected_step,
        "next_epoch": expected_cursor[0],
        "next_logical_batch_position": expected_cursor[1],
        "creation_reason": expected_reason,
        "atomic_save": True,
        "readable_and_schema_valid": True,
        "write_and_validation_seconds": time.perf_counter() - started,
    }
    del loaded
    gc.collect()
    return result


def fresh_epoch_roundtrip(
    checkpoint: Mapping[str, Any],
    batchnorm_baseline: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mean_var_norm, embedding_model, loader_proof = load_trainable_encoder(
        torch.device("cpu")
    )
    aam = AAMSoftmax(seed=SEED)
    optimizer = build_adamw_optimizer(embedding_model, aam)
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128.0)
    embedding_model.load_state_dict(
        checkpoint["embedding_model_state_dict"], strict=True
    )
    mean_var_norm.load_state_dict(
        checkpoint["mean_var_norm_state_dict"], strict=True
    )
    aam.load_state_dict(checkpoint["aam_classifier_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
    apply_batchnorm_policy(embedding_model)
    assert_batchnorm_running_state_exact(embedding_model, batchnorm_baseline)
    checks = checkpoint_state_roundtrip_exact(
        checkpoint,
        embedding_model=embedding_model,
        mean_var_norm=mean_var_norm,
        aam_classifier=aam,
        optimizer=optimizer,
        grad_scaler=scaler,
    )
    restore_rng_state(checkpoint["rng_state"])
    rng_restored_exact = values_exactly_equal(
        capture_rng_state(), checkpoint["rng_state"]
    )
    if not rng_restored_exact:
        raise AssertionError("checkpoint RNG state was not restored exactly")
    optimizer.zero_grad(set_to_none=True)
    result = {
        **checks,
        "fresh_objects_constructed": True,
        "batchnorm_policy_reapplied": True,
        "batchnorm_baseline_exact": True,
        "rng_restored_exact": True,
        "optimizer_steps_taken_after_load": 0,
        "epoch_0_complete": True,
        "global_optimizer_step": 1000,
        "next_epoch": 1,
        "next_logical_batch_position": 0,
        "loader_proof": loader_proof,
    }
    del aam, optimizer, scaler
    gc.collect()
    return {
        "mean_var_norm": mean_var_norm,
        "embedding_model": embedding_model,
    }, result


def read_validation_manifest() -> list[dict[str, str]]:
    path = PROJECT_ROOT / VALIDATION_MANIFEST_RELATIVE
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (
        len(rows) != VALIDATION_ROWS
        or len({row["relative_audio_path"] for row in rows}) != VALIDATION_ROWS
        or len({row["speaker_id"] for row in rows}) != VALIDATION_SPEAKERS
        or any(
            row["final_split"] != "validation" or row["speaker_label"] != "-1"
            for row in rows
        )
    ):
        raise ValueError("approved validation manifest invariants failed")
    return rows


def validate_trial_config() -> dict[str, Any]:
    path = PROJECT_ROOT / TRIAL_CONFIG_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "version": 1,
        "seed": SEED,
        "rows": VALIDATION_ROWS,
        "speakers": VALIDATION_SPEAKERS,
        "positive_trials": POSITIVE_TRIALS,
        "negative_trials": NEGATIVE_TRIALS,
        "trial_sha256": EXPECTED_TRIAL_SHA256,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("immutable validation trial config invariant failed")
    return payload


def extract_and_score_validation(
    objects: Mapping[str, Any],
    *,
    device: torch.device,
    epoch_checkpoint_sha256: str,
    protected_hashes: Mapping[str, str],
    planned_validation_shards: Sequence[str],
    batchnorm_baseline: Mapping[str, torch.Tensor],
    stage: dict[str, str],
) -> tuple[dict[str, Any], set[str]]:
    mean_var_norm = objects["mean_var_norm"]
    embedding_model = objects["embedding_model"]
    stage["value"] = "validation: BatchNorm preflight and eval transition"
    assert_batchnorm_running_state_exact(embedding_model, batchnorm_baseline)
    mean_var_norm.to(device)
    embedding_model.to(device)
    mean_var_norm.eval()
    embedding_model.eval()
    assert_batchnorm_running_state_exact(embedding_model, batchnorm_baseline)
    parameters_before = to_cpu_tree(embedding_model.state_dict())

    stage["value"] = "validation: approved metadata and fixed trials"
    dataset = CachedFbankDataset(
        CACHE_DIR,
        "validation",
        max_cached_shards=TRAIN_LRU_SHARDS,
        validate_finite=False,
    )
    manifest_rows = read_validation_manifest()
    if (
        len(dataset) != VALIDATION_ROWS
        or len({row.speaker_id for row in dataset.rows}) != VALIDATION_SPEAKERS
        or any(
            row.speaker_label != -1 or row.final_split != "validation"
            for row in dataset.rows
        )
    ):
        raise ValueError("validation Dataset count/split/label invariant failed")
    expected_metadata = [
        (row["relative_audio_path"], row["speaker_id"], row["filename_group"])
        for row in manifest_rows
    ]
    actual_metadata = [
        (row.relative_audio_path, row.speaker_id, row.filename_group)
        for row in dataset.rows
    ]
    if actual_metadata != expected_metadata:
        raise ValueError("validation Dataset order differs from approved manifest")
    trial_config = validate_trial_config()
    trial_hash = require_trial_hash(PROJECT_ROOT / TRIAL_RELATIVE)
    trials = read_trials(PROJECT_ROOT / TRIAL_RELATIVE)
    expected_speakers = {row.speaker_id for row in dataset.rows}
    validate_trials(trials, expected_speakers)
    validate_trials_against_metadata(
        trials,
        [(row.relative_audio_path, row.speaker_id) for row in dataset.rows],
        expected_speakers,
    )
    positives = sum(trial.target == 1 for trial in trials)
    negatives = sum(trial.target == 0 for trial in trials)
    if (
        len(trials) != TOTAL_TRIALS
        or positives != POSITIVE_TRIALS
        or negatives != NEGATIVE_TRIALS
    ):
        raise ValueError("fixed validation trial counts are invalid")

    loader = create_cached_fbank_dataloader(
        dataset,
        VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    embeddings: list[torch.Tensor] = []
    paths: list[str] = []
    speakers: list[str] = []
    actual_shards: set[str] = set()
    expected_next_index = 0
    torch.cuda.reset_peak_memory_stats(device)
    extraction_started = time.perf_counter()
    stage["value"] = "validation: embedding extraction"
    for batch_number, batch in enumerate(loader, start=1):
        features = batch["fbank"]
        indexes = batch["dataset_index"].tolist()
        expected_indexes = list(
            range(expected_next_index, expected_next_index + len(indexes))
        )
        if indexes != expected_indexes:
            raise ValueError("validation traversal is not deterministic sequential order")
        expected_next_index += len(indexes)
        if (
            batch["final_split"] != ["validation"] * len(indexes)
            or batch["speaker_label"].tolist() != [-1] * len(indexes)
        ):
            raise ValueError("validation batch contains invalid split/labels")
        for index in indexes:
            shard = strict_shard_relative(
                dataset.rows[index].feature_shard_path, "validation"
            )
            actual_shards.add(
                (CACHE_RELATIVE / Path(*PurePosixPath(shard).parts)).as_posix()
            )
        batch_embeddings = validation_forward_batch(
            mean_var_norm,
            embedding_model,
            features,
            device=device,
        )
        embeddings.append(batch_embeddings)
        paths.extend(batch["relative_audio_path"])
        speakers.extend(batch["speaker_id"])
        stage["value"] = f"validation: embedding extraction batch {batch_number}"
    torch.cuda.synchronize(device)
    extraction_elapsed = time.perf_counter() - extraction_started
    stored = torch.cat(embeddings, dim=0)
    if (
        expected_next_index != VALIDATION_ROWS
        or tuple(stored.shape) != (VALIDATION_ROWS, 192)
        or stored.dtype != torch.float32
        or stored.device.type != "cpu"
        or not bool(torch.isfinite(stored).all().item())
        or paths != [row.relative_audio_path for row in dataset.rows]
        or speakers != [row.speaker_id for row in dataset.rows]
        or actual_shards != set(planned_validation_shards)
    ):
        raise ValueError("validation embedding count/order/shard invariant failed")
    if not values_exactly_equal(
        to_cpu_tree(embedding_model.state_dict()), parameters_before
    ):
        raise RuntimeError("embedding model state changed during validation")
    embedding_artifact = {
        "schema_name": "ecapa_aam_pilot_validation_embeddings",
        "schema_version": 1,
        "checkpoint": relative_string(EPOCH_CHECKPOINT_PATH),
        "checkpoint_sha256": epoch_checkpoint_sha256,
        "trial_csv_sha256": trial_hash,
        "trial_config_sha256": protected_hashes[TRIAL_CONFIG_RELATIVE.as_posix()],
        "validation_manifest_sha256": protected_hashes[
            VALIDATION_MANIFEST_RELATIVE.as_posix()
        ],
        "validation_index_sha256": protected_hashes[
            VALIDATION_INDEX_RELATIVE.as_posix()
        ],
        "relative_audio_paths": paths,
        "speaker_ids": speakers,
        "embeddings": stored,
        "embedding_shape": [VALIDATION_ROWS, 192],
        "embedding_dtype": "float32",
    }
    stage["value"] = "validation: save/reload embeddings"
    atomic_torch_save(embedding_artifact, EMBEDDING_PATH)
    loaded_embeddings = torch.load(
        EMBEDDING_PATH, map_location="cpu", weights_only=False
    )
    expected_embedding_identity = {
        "schema_name": "ecapa_aam_pilot_validation_embeddings",
        "schema_version": 1,
        "checkpoint": relative_string(EPOCH_CHECKPOINT_PATH),
        "checkpoint_sha256": epoch_checkpoint_sha256,
        "trial_csv_sha256": trial_hash,
        "trial_config_sha256": protected_hashes[TRIAL_CONFIG_RELATIVE.as_posix()],
        "validation_manifest_sha256": protected_hashes[
            VALIDATION_MANIFEST_RELATIVE.as_posix()
        ],
        "validation_index_sha256": protected_hashes[
            VALIDATION_INDEX_RELATIVE.as_posix()
        ],
        "embedding_shape": [VALIDATION_ROWS, 192],
        "embedding_dtype": "float32",
    }
    if (
        any(
            loaded_embeddings.get(key) != value
            for key, value in expected_embedding_identity.items()
        )
        or loaded_embeddings["relative_audio_paths"] != paths
        or loaded_embeddings["speaker_ids"] != speakers
        or not torch.equal(loaded_embeddings["embeddings"], stored)
    ):
        raise RuntimeError("validation embedding artifact roundtrip failed")

    stage["value"] = "validation: fixed-trial cosine scoring"
    scoring_started = time.perf_counter()
    scores = score_trials(stored, paths, trials)
    targets = torch.tensor([trial.target for trial in trials], dtype=torch.long)
    scoring_elapsed = time.perf_counter() - scoring_started
    if tuple(scores.shape) != (TOTAL_TRIALS,) or not bool(torch.isfinite(scores).all()):
        raise ValueError("fixed validation scores are invalid")
    trial_ids = [trial.trial_id for trial in trials]
    score_artifact = {
        "schema_name": "ecapa_aam_pilot_validation_scores",
        "schema_version": 1,
        "checkpoint": relative_string(EPOCH_CHECKPOINT_PATH),
        "checkpoint_sha256": epoch_checkpoint_sha256,
        "trial_csv_sha256": trial_hash,
        "trial_config_sha256": protected_hashes[TRIAL_CONFIG_RELATIVE.as_posix()],
        "trial_ids": trial_ids,
        "targets": targets,
        "scores": scores,
        "threshold_semantics": "accept same speaker when score >= threshold",
    }
    stage["value"] = "validation: save/reload scores"
    atomic_torch_save(score_artifact, SCORE_PATH)
    loaded_scores = torch.load(SCORE_PATH, map_location="cpu", weights_only=False)
    expected_score_identity = {
        "schema_name": "ecapa_aam_pilot_validation_scores",
        "schema_version": 1,
        "checkpoint": relative_string(EPOCH_CHECKPOINT_PATH),
        "checkpoint_sha256": epoch_checkpoint_sha256,
        "trial_csv_sha256": trial_hash,
        "trial_config_sha256": protected_hashes[TRIAL_CONFIG_RELATIVE.as_posix()],
        "threshold_semantics": "accept same speaker when score >= threshold",
    }
    validate_saved_artifacts(
        loaded_embeddings,
        loaded_scores,
        trials,
        paths,
        speakers,
    )
    if (
        any(
            loaded_scores.get(key) != value
            for key, value in expected_score_identity.items()
        )
        or loaded_scores["trial_ids"] != trial_ids
        or not torch.equal(loaded_scores["targets"], targets)
        or not torch.equal(loaded_scores["scores"], scores)
    ):
        raise RuntimeError("validation score/trial identity roundtrip failed")

    stage["value"] = "validation: tied-score-safe EER metrics"
    metric_started = time.perf_counter()
    metric = calculate_eer(scores.tolist(), targets.tolist())
    metric_elapsed = time.perf_counter() - metric_started
    same = numeric_summary(scores[targets == 1].tolist())
    different = numeric_summary(scores[targets == 0].tolist())
    metric_payload = {
        **dataclasses.asdict(metric),
        "same_speaker_scores": same,
        "different_speaker_scores": different,
        "score_range": [float(scores.min()), float(scores.max())],
        "score_count": len(scores),
        "positive_trials": positives,
        "negative_trials": negatives,
        "threshold_semantics": "accept same speaker when score >= threshold",
        "trial_csv_sha256": trial_hash,
        "trial_config_version": trial_config["version"],
        "metric_implementation": "grouped_sorted_eer_v2",
        "metric_complexity": "O(N log N) sort plus O(N) grouped scan",
    }
    validate_validation_metrics_payload(metric_payload)
    atomic_json(METRICS_PATH, metric_payload)
    reloaded_metric_payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    validate_validation_metrics_payload(reloaded_metric_payload)
    if reloaded_metric_payload != metric_payload:
        raise RuntimeError("validation metric JSON roundtrip failed")
    validation_runtime = {
        "schema_name": "ecapa_aam_pilot_validation_runtime",
        "schema_version": 1,
        "validation_rows": VALIDATION_ROWS,
        "validation_speakers": VALIDATION_SPEAKERS,
        "batch_size": VALIDATION_BATCH_SIZE,
        "num_workers": 0,
        "embedding_shape": [VALIDATION_ROWS, 192],
        "embedding_dtype": "torch.float32",
        "embedding_finite": True,
        "extraction_seconds": extraction_elapsed,
        "utterances_per_second": VALIDATION_ROWS / extraction_elapsed,
        "scoring_seconds": scoring_elapsed,
        "metric_seconds": metric_elapsed,
        "score_count": TOTAL_TRIALS,
        "positive_trials": positives,
        "negative_trials": negatives,
        "cuda_peak_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "cuda_peak_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "aam_classifier_calls": 0,
        "pretrained_classifier_calls": 0,
        "waveform_frontend_calls": 0,
        "compute_feature_frontend_calls": 0,
        "inference_mode": True,
        "sequential_traversal": True,
        "augmentation": False,
        "trial_path_ownership_validated": True,
        "actual_validation_shards": len(actual_shards),
    }
    atomic_json(VALIDATION_RUNTIME_PATH, validation_runtime)
    if (
        json.loads(VALIDATION_RUNTIME_PATH.read_text(encoding="utf-8"))
        != validation_runtime
    ):
        raise RuntimeError("validation runtime JSON roundtrip failed")
    return {
        "metrics": metric_payload,
        "runtime": validation_runtime,
        "artifacts": {
            "embeddings": relative_string(EMBEDDING_PATH),
            "scores": relative_string(SCORE_PATH),
            "metrics": relative_string(METRICS_PATH),
            "runtime": relative_string(VALIDATION_RUNTIME_PATH),
        },
    }, actual_shards


def run(args: argparse.Namespace, stage: dict[str, str]) -> dict[str, Any]:
    stage["value"] = "preflight: interpreter/device"
    expected_python = (PROJECT_ROOT / ".venv-cuda/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError("pilot must run with .venv-cuda\\Scripts\\python.exe")
    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or device.index not in (None, 0)
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("pilot requires CUDA device cuda:0")
    ensure_fresh_runtime_targets()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    task_started_utc = utc_now()
    task_started = time.perf_counter()

    stage["value"] = "preflight: approved train Dataset"
    dataset, train_guard = create_train_only_smoke_dataset(
        project_root=PROJECT_ROOT,
        cache_dir=CACHE_RELATIVE,
        split="train",
        index_filename="train_feature_index_v1.csv",
        max_cached_shards=TRAIN_LRU_SHARDS,
        validate_finite=False,
    )
    train_speakers = {row.speaker_id for row in dataset.rows}
    train_labels = {row.speaker_label for row in dataset.rows}
    validation_manifest_preflight = read_validation_manifest()
    validation_speakers_preflight = {
        row["speaker_id"] for row in validation_manifest_preflight
    }
    if (
        len(dataset) != 31998
        or len(train_speakers) != 488
        or train_labels != set(range(488))
        or len(validation_manifest_preflight) != VALIDATION_ROWS
        or len(validation_speakers_preflight) != VALIDATION_SPEAKERS
        or not train_speakers.isdisjoint(validation_speakers_preflight)
    ):
        raise ValueError(
            "approved train/validation count, label, or speaker-disjointness "
            "preflight failed"
        )
    rows_before = dataset.rows
    sampler = HybridShardAwareSpeakerBatchSampler(
        dataset,
        speakers_per_batch=SPEAKERS_PER_BATCH,
        samples_per_speaker=SAMPLES_PER_SPEAKER,
        active_shard_window=ACTIVE_SHARD_WINDOW,
        num_batches=NUM_LOGICAL_BATCHES,
        seed=SEED,
        # Materialize the metadata-only plan first, then stat/hash/load exactly
        # the selected shard allowlist below.
        validate_shard_existence=False,
    )
    sampler.set_epoch(0)
    stage["value"] = "preflight: materialize deterministic epoch-0 sampler plan"
    planned_batches = list(iter(sampler))
    sampler_audit = validate_sampler_plan(dataset, planned_batches)
    sampler_stats = dict(sampler.last_epoch_stats)
    train_shards = selected_train_shards(dataset, planned_batches)
    validation_shards = read_validation_shards_from_index()
    protected_paths = protected_path_set(train_shards, validation_shards)
    stage["value"] = "preflight: hash protected train/validation inputs"
    protected_before = hash_explicit_paths(PROJECT_ROOT, protected_paths)
    if require_trial_hash(PROJECT_ROOT / TRIAL_RELATIVE) != EXPECTED_TRIAL_SHA256:
        raise AssertionError("fixed trial hash check did not return approved value")
    train_cache_identity = train_identity(protected_before, train_shards)

    stage["value"] = "preflight: dedicated DataLoader generator"
    global_rng_before_loader = torch.get_rng_state().clone()
    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED + 1000)
    loader = create_cached_fbank_training_dataloader(
        dataset,
        planned_batches,  # type: ignore[arg-type]
        num_workers=0,
        generator=loader_generator,
    )
    iterator = iter(loader)
    if not torch.equal(global_rng_before_loader, torch.get_rng_state()):
        raise RuntimeError("DataLoader construction consumed the model RNG")

    stage["value"] = "training: initial model/AAM/optimizer construction"
    objects = create_training_objects(device)
    loader_proof = objects["loader_proof"]
    batchnorm_baseline = batchnorm_running_state(objects["embedding_model"])
    if not batchnorm_baseline:
        raise RuntimeError("embedding model has no BatchNorm running buffers")
    optimizer_hook_count = {"value": 0}

    def optimizer_post_hook(
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        optimizer_hook_count["value"] += 1

    hook_handle = objects["optimizer"].register_step_post_hook(optimizer_post_hook)
    controller = OneEpochController()
    controller.start(0)
    losses: list[float] = []
    durations: list[float] = []
    ecapa_gradient_norms: list[float] = []
    aam_gradient_norms: list[float] = []
    scaler_scales: list[float] = []
    logging_history: list[dict[str, Any]] = []
    rolling_results: list[dict[str, Any]] = []
    actual_train_shards: set[str] = set()
    window = LoggingWindow()
    torch.cuda.reset_peak_memory_stats(device)
    training_started_utc = utc_now()
    training_started = time.perf_counter()

    for position in range(NUM_LOGICAL_BATCHES):
        stage["value"] = f"training: load logical batch {position + 1}"
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise RuntimeError("training DataLoader ended before 1,000 batches") from error
        logical_batch = validate_logical_batch(
            dataset, batch, planned_batches[position]
        )
        for index in logical_batch["dataset_index"].tolist():
            relative = strict_shard_relative(
                dataset.rows[index].feature_shard_path, "train"
            )
            actual_train_shards.add(
                (CACHE_RELATIVE / Path(*PurePosixPath(relative).parts)).as_posix()
            )
        global_step = position + 1
        update = run_optimizer_update(
            objects,
            logical_batch,
            global_step=global_step,
            optimizer_hook_count=optimizer_hook_count,
            stage=stage,
        )
        controller.record_update(position)
        losses.append(update["logical_loss"])
        durations.append(update["duration_seconds"])
        ecapa_gradient_norms.append(update["ecapa_gradient_norm"])
        aam_gradient_norms.append(update["aam_gradient_norm"])
        scaler_scales.append(update["grad_scaler_scale_after"])
        window.add(update["logical_loss"], update["duration_seconds"])

        rolling_path: str | None = None
        if rolling_checkpoint_due(global_step):
            stage["value"] = f"training: rolling checkpoint at step {global_step}"
            assert_batchnorm_running_state_exact(
                objects["embedding_model"], batchnorm_baseline
            )
            checkpoint = build_checkpoint(
                objects,
                train_cache_identity=train_cache_identity,
                losses=losses,
                global_step=global_step,
                reason="rolling",
            )
            rolling_result = save_and_validate_checkpoint(
                checkpoint,
                LAST_CHECKPOINT_PATH,
                expected_step=global_step,
                expected_reason="rolling",
            )
            assert_batchnorm_running_state_exact(
                objects["embedding_model"], batchnorm_baseline
            )
            rolling_result["batchnorm_baseline_exact"] = True
            rolling_results.append(rolling_result)
            rolling_path = relative_string(LAST_CHECKPOINT_PATH)
            del checkpoint
            gc.collect()

        if global_step % LOGGING_INTERVAL == 0:
            window_summary = window.emit(global_step)
            record = {
                "schema_name": "ecapa_aam_pilot_training_log",
                "schema_version": 1,
                "epoch": 0,
                "global_optimizer_step": global_step,
                "logical_batch_position": position,
                "current_logical_loss": update["logical_loss"],
                "mean_logical_loss_latest_window": window_summary[
                    "mean_logical_loss"
                ],
                "ecapa_learning_rate": float(
                    objects["optimizer"].param_groups[0]["lr"]
                ),
                "aam_learning_rate": float(
                    objects["optimizer"].param_groups[1]["lr"]
                ),
                "grad_scaler_scale": update["grad_scaler_scale_after"],
                "ecapa_gradient_norm": update["ecapa_gradient_norm"],
                "aam_gradient_norm": update["aam_gradient_norm"],
                "step_duration_seconds": update["duration_seconds"],
                "rolling_mean_step_duration_seconds": window_summary[
                    "mean_step_duration_seconds"
                ],
                "logical_samples_per_second": (
                    LOGICAL_BATCH_SIZE / update["duration_seconds"]
                ),
                "cuda_memory_allocated_bytes": update[
                    "cuda_memory_allocated_bytes"
                ],
                "cuda_memory_reserved_bytes": update[
                    "cuda_memory_reserved_bytes"
                ],
                "cuda_max_memory_allocated_bytes": update[
                    "cuda_max_memory_allocated_bytes"
                ],
                "cuda_max_memory_reserved_bytes": update[
                    "cuda_max_memory_reserved_bytes"
                ],
                "rolling_checkpoint_path": rolling_path,
            }
            validate_training_log_record(record)
            append_jsonl(TRAIN_LOG_PATH, record)
            logging_history.append(record)
            print(
                json.dumps(
                    {
                        "step": global_step,
                        "loss": update["logical_loss"],
                        "mean_loss_50": window_summary["mean_logical_loss"],
                        "seconds": update["duration_seconds"],
                        "checkpoint": rolling_path,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del batch, logical_batch, update

    try:
        extra_batch = next(iterator)
    except StopIteration:
        extra_batch = None
    if extra_batch is not None:
        raise RuntimeError("training DataLoader produced more than 1,000 batches")
    epoch_result = controller.finish()
    training_ended_utc = utc_now()
    training_duration = time.perf_counter() - training_started
    hook_handle.remove()
    if (
        optimizer_hook_count["value"] != NUM_LOGICAL_BATCHES
        or len(losses) != NUM_LOGICAL_BATCHES
        or actual_train_shards != set(train_shards)
        or dataset.rows is not rows_before
        or sampler.epoch != 0
    ):
        raise RuntimeError("one-epoch training count/shard/mutation invariant failed")
    assert_batchnorm_running_state_exact(
        objects["embedding_model"], batchnorm_baseline
    )
    if set(ROLLING_CHECKPOINT_STEPS) != {
        result["global_optimizer_step"] for result in rolling_results
    }:
        raise RuntimeError("rolling checkpoint boundaries are incomplete")
    loss_summary = summarize_finite_values(losses, "logical losses")
    gradient_summary = {
        "ecapa": summarize_finite_values(
            ecapa_gradient_norms, "ECAPA gradient norms"
        ),
        "aam": summarize_finite_values(aam_gradient_norms, "AAM gradient norms"),
    }
    duration_summary = summarize_finite_values(durations, "step durations")
    scaler_summary = {
        **summarize_finite_values(scaler_scales, "GradScaler scales"),
        "unique_scales": sorted(set(scaler_scales)),
        "skipped_updates": 0,
        "optimizer_step_hook_count": optimizer_hook_count["value"],
    }
    atomic_json(
        LOSS_PATH,
        {
            "schema_name": "ecapa_aam_pilot_logical_losses",
            "schema_version": 1,
            "epoch": 0,
            "count": len(losses),
            "losses": losses,
        },
    )

    stage["value"] = "checkpoint: immutable epoch checkpoint"
    epoch_checkpoint = build_checkpoint(
        objects,
        train_cache_identity=train_cache_identity,
        losses=losses,
        global_step=NUM_LOGICAL_BATCHES,
        reason="epoch_complete",
    )
    epoch_checkpoint_result = save_and_validate_checkpoint(
        epoch_checkpoint,
        EPOCH_CHECKPOINT_PATH,
        expected_step=NUM_LOGICAL_BATCHES,
        expected_reason="epoch_complete",
    )
    epoch_checkpoint_result["batchnorm_baseline_exact"] = True
    epoch_checkpoint_loaded = load_pilot_checkpoint(EPOCH_CHECKPOINT_PATH)
    epoch_checkpoint_sha256 = file_sha256(EPOCH_CHECKPOINT_PATH)

    stage["value"] = "checkpoint: release training objects"
    objects["optimizer"].zero_grad(set_to_none=True)
    del iterator, loader, planned_batches, sampler, dataset, objects, epoch_checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    stage["value"] = "checkpoint: full fresh-object epoch roundtrip"
    validation_objects, roundtrip_result = fresh_epoch_roundtrip(
        epoch_checkpoint_loaded, batchnorm_baseline
    )
    if roundtrip_result["optimizer_steps_taken_after_load"] != 0:
        raise RuntimeError("epoch checkpoint roundtrip took an optimizer step")

    stage["value"] = "validation: release checkpoint-only tensors"
    assert_batchnorm_running_state_exact(
        validation_objects["embedding_model"], batchnorm_baseline
    )
    validation_result, actual_validation_shards = extract_and_score_validation(
        validation_objects,
        device=device,
        epoch_checkpoint_sha256=epoch_checkpoint_sha256,
        protected_hashes=protected_before,
        planned_validation_shards=validation_shards,
        batchnorm_baseline=batchnorm_baseline,
        stage=stage,
    )
    pilot_metrics = validation_result["metrics"]
    comparison = baseline_comparison(
        pilot_metrics, BASELINE_METRICS, QUALITY_TOLERANCE
    )
    model_quality_result = comparison["model_quality_result"]
    best_candidate = select_best_checkpoint(
        [
            {
                "name": "epoch_000.pt",
                "interpolated_eer": pilot_metrics["interpolated_eer"],
                "empirical_average_error": pilot_metrics[
                    "empirical_average_error"
                ],
                "order": 0,
            }
        ]
    )
    if best_candidate["name"] != "epoch_000.pt":
        raise RuntimeError("one-checkpoint pilot did not select epoch_000 internally")
    selection = {
        "selected_checkpoint": "epoch_000.pt",
        "selection_scope": "best among trained pilot checkpoints",
        "selection_metric": "lowest interpolated validation EER",
        "tie_break_1": "lowest empirical average error",
        "tie_break_2": "earlier checkpoint",
        "pilot_metrics": pilot_metrics,
        "baseline_is_comparison_reference_only": True,
        "improved_relative_to_pretrained": model_quality_result == "IMPROVED",
        "model_quality_result": model_quality_result,
    }
    stage["value"] = "checkpoint: best validation checkpoint"
    best_checkpoint = dict(epoch_checkpoint_loaded)
    best_checkpoint["checkpoint_creation_reason"] = "best_validation"
    best_checkpoint["validation_selection"] = selection
    best_checkpoint_result = save_and_validate_checkpoint(
        best_checkpoint,
        BEST_CHECKPOINT_PATH,
        expected_step=NUM_LOGICAL_BATCHES,
        expected_reason="best_validation",
    )
    loaded_best = load_pilot_checkpoint(BEST_CHECKPOINT_PATH)
    state_keys = (
        "embedding_model_state_dict",
        "mean_var_norm_state_dict",
        "aam_classifier_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
    )
    if not all(
        values_exactly_equal(
            loaded_best[key], epoch_checkpoint_loaded[key]
        )
        for key in state_keys
    ):
        raise RuntimeError("best checkpoint state differs from epoch_000")
    best_checkpoint_result["state_matches_epoch_000"] = True

    stage["value"] = "postflight: protected hashes"
    protected_after = hash_explicit_paths(PROJECT_ROOT, protected_paths)
    trial_hash_after = require_trial_hash(PROJECT_ROOT / TRIAL_RELATIVE)
    mismatches = {
        path: {"before": protected_before[path], "after": protected_after[path]}
        for path in protected_before
        if protected_before[path] != protected_after[path]
    }
    if (
        mismatches
        or set(protected_before) != set(protected_after)
        or actual_validation_shards != set(validation_shards)
        or trial_hash_after != EXPECTED_TRIAL_SHA256
    ):
        raise RuntimeError("protected input preservation check failed")

    training_pass = all(
        (
            epoch_result["optimizer_updates"] == NUM_LOGICAL_BATCHES,
            epoch_result["logical_sample_selections"] == TOTAL_SAMPLE_SELECTIONS,
            epoch_result["epoch_1_started"] is False,
            loss_summary["count"] == NUM_LOGICAL_BATCHES,
            gradient_summary["ecapa"]["minimum"] > 0.0,
            gradient_summary["aam"]["minimum"] > 0.0,
            scaler_summary["skipped_updates"] == 0,
            len(logging_history) == NUM_LOGICAL_BATCHES // LOGGING_INTERVAL,
            actual_train_shards == set(train_shards),
        )
    )
    checkpoint_pass = all(
        (
            len(rolling_results) == len(ROLLING_CHECKPOINT_STEPS),
            epoch_checkpoint_result["readable_and_schema_valid"],
            all(
                value is True
                for key, value in roundtrip_result.items()
                if key.endswith("_exact") or key.endswith("_valid")
            ),
            roundtrip_result["optimizer_steps_taken_after_load"] == 0,
            best_checkpoint_result["state_matches_epoch_000"],
        )
    )
    validation_pass = all(
        (
            validation_result["runtime"]["validation_rows"] == VALIDATION_ROWS,
            validation_result["runtime"]["score_count"] == TOTAL_TRIALS,
            validation_result["runtime"]["aam_classifier_calls"] == 0,
            validation_result["runtime"]["sequential_traversal"],
        )
    )
    scope_compliance_pass = all(
        (
            not mismatches,
            trial_hash_after == EXPECTED_TRIAL_SHA256,
            epoch_result["epoch_1_started"] is False,
            actual_train_shards == set(train_shards),
            actual_validation_shards == set(validation_shards),
        )
    )
    technical_pass = training_pass and checkpoint_pass and validation_pass
    overall_pass = technical_pass and scope_compliance_pass
    if not overall_pass:
        raise RuntimeError("derived pilot PASS gates did not all succeed")

    task_ended_utc = utc_now()
    result = {
        "schema_name": PILOT_RUNTIME_SCHEMA_NAME,
        "schema_version": PILOT_RUNTIME_SCHEMA_VERSION,
        "technical_pass": technical_pass,
        "training_pass": training_pass,
        "checkpoint_pass": checkpoint_pass,
        "validation_pass": validation_pass,
        "scope_compliance_pass": scope_compliance_pass,
        "overall_pass": overall_pass,
        "model_quality_result": model_quality_result,
        "task_started_utc": task_started_utc,
        "task_ended_utc": task_ended_utc,
        "task_duration_seconds": time.perf_counter() - task_started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "speechbrain": speechbrain.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
            "interpreter": ".venv-cuda/Scripts/python.exe",
        },
        "training_configuration": {
            "pretrained_model_identifier": PRETRAINED_MODEL_ID,
            "epoch": 0,
            "logical_batches": NUM_LOGICAL_BATCHES,
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "sample_selections": TOTAL_SAMPLE_SELECTIONS,
            "physical_microbatch_size": PHYSICAL_MICROBATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "num_workers": 0,
            "max_cached_shards": TRAIN_LRU_SHARDS,
            "augmentation": False,
            "scheduler": False,
            "warmup": False,
            "gradient_clipping": False,
            "early_stopping": False,
        },
        "sampler_configuration": dict(SAMPLER_CONFIGURATION),
        "sampler_audit": sampler_audit,
        "sampler_runtime_statistics": {
            "queue_cycles": int(sampler_stats.get("queue_cycles", 0)),
            "unique_selected_indexes": int(
                sampler_stats.get("unique_selected_indexes", 0)
            ),
            "repeated_selections": int(
                sampler_stats.get("repeated_selections", 0)
            ),
        },
        "optimizer_configuration": [dict(group) for group in OPTIMIZER_CONFIGURATION],
        "aam_configuration": dict(AAM_CONFIGURATION),
        "batchnorm_policy": dict(BATCHNORM_POLICY),
        "amp_policy": dict(AMP_POLICY),
        "training_summary": {
            **epoch_result,
            "training_started_utc": training_started_utc,
            "training_ended_utc": training_ended_utc,
            "training_duration_seconds": training_duration,
            "loss": loss_summary,
            "gradient_norms": gradient_summary,
            "step_durations": duration_summary,
            "grad_scaler": scaler_summary,
            "batchnorm_running_buffers_compared": len(batchnorm_baseline),
            "batchnorm_running_buffers_bit_exact": True,
            "cuda_peak_memory_allocated_bytes": int(
                max(record["cuda_max_memory_allocated_bytes"] for record in logging_history)
            ),
            "cuda_peak_memory_reserved_bytes": int(
                max(record["cuda_max_memory_reserved_bytes"] for record in logging_history)
            ),
            "logging_records": logging_history,
            "logging_record_count": len(logging_history),
            "logical_losses_artifact": relative_string(LOSS_PATH),
            "training_log_artifact": relative_string(TRAIN_LOG_PATH),
            "actual_train_shards_loaded": len(actual_train_shards),
            "forbidden_call_proof": {
                "waveform_frontend_calls": 0,
                "compute_feature_frontend_calls": 0,
                "pretrained_classifier_calls": 0,
                "loader": loader_proof,
            },
        },
        "checkpoint_summary": {
            "rolling_checkpoints": rolling_results,
            "epoch_checkpoint": epoch_checkpoint_result,
            "epoch_checkpoint_sha256": epoch_checkpoint_sha256,
            "epoch_roundtrip": roundtrip_result,
            "best_checkpoint": best_checkpoint_result,
        },
        "validation_summary": validation_result,
        "baseline_metrics": BASELINE_METRICS,
        "pilot_metrics": pilot_metrics,
        "metric_differences": comparison,
        "best_checkpoint_decision": selection,
        "protected_hashes": {
            "files_compared": len(protected_before),
            "train_shards": len(train_shards),
            "validation_shards": len(validation_shards),
            "before_set_sha256": hash_mapping_digest(protected_before),
            "after_set_sha256": hash_mapping_digest(protected_after),
            "unchanged": True,
            "mismatches": mismatches,
            "before": protected_before,
            "after": protected_after,
        },
        "validation_trial_hash": trial_hash_after,
        "train_only_guard": train_guard,
        "final_test_accessed": False,
        "recursive_listing_used": False,
        "waveform_accessed": False,
        "epoch_1_started": False,
        "commit_or_push_performed": False,
        "files_created": [
            relative_string(path)
            for path in (
                TRAIN_LOG_PATH,
                LOSS_PATH,
                LAST_CHECKPOINT_PATH,
                EPOCH_CHECKPOINT_PATH,
                BEST_CHECKPOINT_PATH,
                EMBEDDING_PATH,
                SCORE_PATH,
                METRICS_PATH,
                VALIDATION_RUNTIME_PATH,
                RUNTIME_PATH,
            )
        ],
        "deferred": [
            "epoch 1 and all later training epochs",
            "hyperparameter tuning",
            "augmentation",
            "scheduler, warmup, gradient clipping, and early stopping",
            "every final-test operation",
            "full multi-epoch fine-tuning",
            "commit and push",
        ],
    }
    validate_runtime_payload(result)
    atomic_json(RUNTIME_PATH, result)
    return result


def main() -> None:
    args = parse_args()
    stage = {"value": "startup"}
    try:
        result = run(args, stage)
    except BaseException as error:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_name": "ecapa_aam_one_epoch_pilot_failure",
            "schema_version": 1,
            "technical_pass": False,
            "overall_pass": False,
            "exact_stage": stage["value"],
            "error": safe_error(error),
            "cuda_oom": "out of memory" in str(error).lower(),
            "microbatch_2_fallback_attempted": False,
            "epoch_1_started": False,
            "final_test_accessed": False,
            "recursive_listing_used": False,
            "created_utc": utc_now(),
        }
        atomic_json(FAILURE_PATH, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        raise
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "model_quality_result": result["model_quality_result"],
                "optimizer_updates": result["training_summary"][
                    "optimizer_updates"
                ],
                "validation_rows": result["validation_summary"]["runtime"][
                    "validation_rows"
                ],
                "score_count": result["validation_summary"]["runtime"][
                    "score_count"
                ],
                "pilot_interpolated_eer": result["pilot_metrics"][
                    "interpolated_eer"
                ],
                "runtime": relative_string(RUNTIME_PATH),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("ONE-EPOCH ECAPA/AAM PILOT: PASS", flush=True)


if __name__ == "__main__":
    main()
