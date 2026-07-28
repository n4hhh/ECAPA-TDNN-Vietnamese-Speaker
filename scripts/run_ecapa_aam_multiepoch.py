"""Resume the epoch-0 ECAPA/AAM pilot through epoch 4 with fixed validation."""

from __future__ import annotations

import argparse
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
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import speechbrain
import torch

from scripts.run_ecapa_aam_one_epoch_pilot import (
    BASELINE_METRICS,
    CACHE_CONFIG_RELATIVE,
    CACHE_DIR,
    CACHE_RELATIVE,
    NEGATIVE_TRIALS,
    POSITIVE_TRIALS,
    TOTAL_TRIALS,
    TRAIN_INDEX_RELATIVE,
    TRAIN_LRU_SHARDS,
    TRIAL_CONFIG_RELATIVE,
    TRIAL_RELATIVE,
    VALIDATION_BATCH_SIZE,
    VALIDATION_INDEX_RELATIVE,
    VALIDATION_MANIFEST_RELATIVE,
    VALIDATION_ROWS,
    VALIDATION_SPEAKERS,
    create_training_objects,
    finite_gradient_norm,
    optimizer_all_parameter_steps,
    read_validation_manifest,
    read_validation_shards_from_index,
    selected_train_shards,
    strict_shard_relative,
    validate_logical_batch,
    validate_sampler_plan,
    validate_trial_config,
)
from src.aam_training import (
    PRETRAINED_MODEL_ID,
    SEED,
    apply_batchnorm_policy,
    assert_batchnorm_running_state_exact,
    batchnorm_running_state,
    capture_rng_state,
    file_sha256,
    restore_rng_state,
    to_cpu_tree,
    values_exactly_equal,
)
from src.cached_fbank_dataset import (
    CachedFbankDataset,
    create_cached_fbank_dataloader,
    create_cached_fbank_training_dataloader,
)
from src.cached_fbank_samplers import HybridShardAwareSpeakerBatchSampler
from src.ecapa_multiepoch import (
    BASE_LRS,
    FIRST_RESUMED_EPOCH,
    INITIAL_BEST_EER,
    INITIAL_EMPIRICAL_THRESHOLD,
    LAST_RESUMED_EPOCH,
    MAX_RESUMED_STEPS,
    MIN_DELTA,
    PATIENCE,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    START_GLOBAL_STEP,
    STEPS_PER_EPOCH,
    EarlyStoppingState,
    ResumedCosineScheduler,
    reject_final_test_path,
    start_epoch,
    validate_epoch0_migration,
    validate_multiepoch_checkpoint,
)
from src.ecapa_one_epoch_pilot import (
    AAM_CONFIGURATION,
    AMP_POLICY,
    BATCHNORM_POLICY,
    EXPECTED_TRIAL_SHA256,
    OPTIMIZER_CONFIGURATION,
    SAMPLER_CONFIGURATION,
    atomic_json,
    load_pilot_checkpoint,
    require_trial_hash,
    summarize_finite_values,
    validate_validation_metrics_payload,
    validation_forward_batch,
)
from src.verification_baseline import numeric_summary, score_trials
from src.verification_metrics import calculate_eer
from src.verification_trials import (
    read_trials,
    validate_trials,
    validate_trials_against_metadata,
)


OUTPUT_RELATIVE = Path("outputs/ecapa_aam_multiepoch_v1")
OUTPUT_DIR = PROJECT_ROOT / OUTPUT_RELATIVE
LAST_CHECKPOINT_PATH = OUTPUT_DIR / "last.pt"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "best.pt"
RUNTIME_PATH = OUTPUT_DIR / "runtime.json"
FAILURE_PATH = OUTPUT_DIR / "failure.json"
TRAIN_LOG_PATH = OUTPUT_DIR / "training_log.jsonl"
START_CHECKPOINT_RELATIVE = Path(
    "outputs/ecapa_aam_one_epoch_pilot_v1/epoch_000.pt"
)
START_CHECKPOINT_PATH = PROJECT_ROOT / START_CHECKPOINT_RELATIVE
START_METRICS_RELATIVE = Path(
    "outputs/ecapa_aam_one_epoch_pilot_v1/validation_metrics_epoch_000.json"
)
START_METRICS_PATH = PROJECT_ROOT / START_METRICS_RELATIVE
EXPECTED_START_SHA256 = (
    "e82ba006aef4a505244769f137af897a94bbb601a26ffe2b68fbfec135201b67"
)
ROLLING_INTERVAL = 250
LOGGING_INTERVAL = 50

TRAINING_CONFIGURATION = {
    "first_resumed_epoch": 1,
    "last_resumed_epoch": 4,
    "logical_batches_per_epoch": 1000,
    "logical_batch_size": 32,
    "physical_microbatch_size": 4,
    "gradient_accumulation_steps": 8,
    "sampler": "HybridShardAwareSpeakerBatchSampler",
    "speakers_per_batch": 16,
    "samples_per_speaker": 2,
    "active_shard_window": 8,
    "seed": 20260727,
    "cached_fbank_shape": [301, 80],
    "validation_rows": 8504,
    "validation_trials": 19528,
    "scheduler_total_steps": 4000,
    "scheduler_minimum_factor": 0.1,
    "early_stopping_patience": 2,
    "early_stopping_min_delta": 0.0001,
    "augmentation": False,
    "gradient_clipping": False,
    "num_workers": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_string(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def safe_error(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    return message.replace(str(PROJECT_ROOT), ".").replace(
        str(PROJECT_ROOT).lower(), "."
    )


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


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def validate_runtime_paths() -> None:
    paths = (
        CACHE_CONFIG_RELATIVE,
        TRAIN_INDEX_RELATIVE,
        VALIDATION_INDEX_RELATIVE,
        VALIDATION_MANIFEST_RELATIVE,
        TRIAL_RELATIVE,
        TRIAL_CONFIG_RELATIVE,
        START_CHECKPOINT_RELATIVE,
        START_METRICS_RELATIVE,
        OUTPUT_RELATIVE,
    )
    for path in paths:
        reject_final_test_path(path.as_posix())


def hash_allowed_paths(relative_paths: Sequence[str]) -> dict[str, str]:
    """Hash only explicit approved paths, with containment and quarantine guards."""
    hashes: dict[str, str] = {}
    root = PROJECT_ROOT.resolve()
    for value in relative_paths:
        relative = reject_final_test_path(value)
        if relative in hashes:
            raise ValueError(f"duplicate protected path: {relative!r}")
        path = (PROJECT_ROOT / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"protected path escapes the project: {relative!r}") from error
        if not path.is_file():
            raise FileNotFoundError(f"missing protected input: {relative!r}")
        hashes[relative] = file_sha256(path)
    return hashes


def hash_mapping_digest(hashes: Mapping[str, str]) -> str:
    canonical = json.dumps(
        dict(sorted(hashes.items())),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_start_state() -> tuple[dict[str, Any], dict[str, Any], str]:
    start_hash = file_sha256(START_CHECKPOINT_PATH)
    if start_hash != EXPECTED_START_SHA256:
        raise RuntimeError("start checkpoint SHA-256 differs from the approved pilot")
    checkpoint = load_pilot_checkpoint(START_CHECKPOINT_PATH)
    metrics = json.loads(START_METRICS_PATH.read_text(encoding="utf-8"))
    validate_validation_metrics_payload(metrics)
    validate_epoch0_migration(checkpoint, metrics)
    if require_trial_hash(PROJECT_ROOT / TRIAL_RELATIVE) != EXPECTED_TRIAL_SHA256:
        raise RuntimeError("fixed validation trial SHA-256 changed")
    optimizer_steps = {
        int(
            state["step"].item()
            if isinstance(state["step"], torch.Tensor)
            else state["step"]
        )
        for state in checkpoint["optimizer_state_dict"]["state"].values()
    }
    if optimizer_steps != {START_GLOBAL_STEP}:
        raise RuntimeError("epoch-0 AdamW counters are not exactly 1000")
    if [group["lr"] for group in checkpoint["optimizer_state_dict"]["param_groups"]] != [
        BASE_LRS[0],
        BASE_LRS[1],
    ]:
        raise RuntimeError("epoch-0 optimizer learning rates changed")
    return checkpoint, metrics, start_hash


def initial_epoch_metrics(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "epoch": 0,
            "train_loss": 5.100674975889735,
            "validation": dict(metrics),
            "ending_ecapa_lr": BASE_LRS[0],
            "ending_aam_lr": BASE_LRS[1],
            "best": True,
            "patience_counter": 0,
            "optimizer_steps": 1000,
        }
    ]


def build_checkpoint(
    objects: Mapping[str, Any],
    scheduler: ResumedCosineScheduler,
    early_stopping: EarlyStoppingState,
    *,
    epoch: int,
    next_epoch: int,
    next_position: int,
    pending_validation_epoch: int | None,
    epoch_metrics: Sequence[Mapping[str, Any]],
    completed_epochs: Sequence[int],
    current_epoch_statistics: Mapping[str, Any],
    protected_hashes_before: Mapping[str, str],
    start_hash: str,
    reason: str,
    stop_reason: str | None,
) -> dict[str, Any]:
    global_step = START_GLOBAL_STEP + scheduler.completed_resumed_steps
    checkpoint = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
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
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "next_epoch": next_epoch,
        "next_logical_batch_position": next_position,
        "pending_validation_epoch": pending_validation_epoch,
        "global_optimizer_step": global_step,
        "resumed_optimizer_steps": scheduler.completed_resumed_steps,
        "sampler_state": {
            **dict(SAMPLER_CONFIGURATION),
            "epoch": next_epoch if pending_validation_epoch is None else pending_validation_epoch,
            "next_logical_batch_position": next_position,
        },
        "rng_state": capture_rng_state(),
        "early_stopping_state": early_stopping.state_dict(),
        "batchnorm_policy": dict(BATCHNORM_POLICY),
        "batchnorm_reference_state": to_cpu_tree(
            objects["batchnorm_reference_state"]
        ),
        "training_configuration": dict(TRAINING_CONFIGURATION),
        "aam_configuration": dict(AAM_CONFIGURATION),
        "optimizer_configuration": [dict(group) for group in OPTIMIZER_CONFIGURATION],
        "amp_policy": dict(AMP_POLICY),
        "start_checkpoint": {
            "path": START_CHECKPOINT_RELATIVE.as_posix(),
            "sha256": start_hash,
            "epoch": 0,
            "global_optimizer_step": 1000,
        },
        "epoch_metrics": [dict(value) for value in epoch_metrics],
        "completed_epochs": list(completed_epochs),
        "current_epoch_statistics": dict(current_epoch_statistics),
        "protected_hashes_before": dict(protected_hashes_before),
        "checkpoint_creation_reason": reason,
        "stop_reason": stop_reason,
        "invalid_or_skipped_updates": 0,
        "scheduler_migration": {
            "source_schema": "speaker_verification_ecapa_aam_one_epoch_pilot",
            "source_had_scheduler_state": False,
            "initialized_completed_resumed_steps": 0,
        },
    }
    validate_multiepoch_checkpoint(checkpoint)
    return checkpoint


def save_checkpoint(checkpoint: Mapping[str, Any], path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    atomic_torch_save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    validate_multiepoch_checkpoint(loaded)
    identity = (
        loaded["global_optimizer_step"],
        loaded["next_epoch"],
        loaded["next_logical_batch_position"],
        loaded["pending_validation_epoch"],
    )
    expected = (
        checkpoint["global_optimizer_step"],
        checkpoint["next_epoch"],
        checkpoint["next_logical_batch_position"],
        checkpoint["pending_validation_epoch"],
    )
    if identity != expected:
        raise RuntimeError("checkpoint readable cursor differs after atomic save")
    result = {
        "path": relative_string(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "global_optimizer_step": checkpoint["global_optimizer_step"],
        "resumed_optimizer_steps": checkpoint["resumed_optimizer_steps"],
        "next_epoch": checkpoint["next_epoch"],
        "next_logical_batch_position": checkpoint[
            "next_logical_batch_position"
        ],
        "pending_validation_epoch": checkpoint["pending_validation_epoch"],
        "seconds": time.perf_counter() - started,
        "atomic": True,
        "readable": True,
    }
    del loaded
    gc.collect()
    return result


def load_objects_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device
) -> tuple[dict[str, Any], ResumedCosineScheduler, EarlyStoppingState]:
    objects = create_training_objects(device)
    objects["embedding_model"].load_state_dict(
        checkpoint["embedding_model_state_dict"], strict=True
    )
    objects["mean_var_norm"].load_state_dict(
        checkpoint["mean_var_norm_state_dict"], strict=True
    )
    objects["aam"].load_state_dict(
        checkpoint["aam_classifier_state_dict"], strict=True
    )
    objects["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])
    objects["scaler"].load_state_dict(checkpoint["grad_scaler_state_dict"])
    scheduler = ResumedCosineScheduler(
        objects["optimizer"],
        completed_resumed_steps=0,
    )
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    early_stopping = EarlyStoppingState.from_state_dict(
        checkpoint["early_stopping_state"]
    )
    objects["batchnorm_reference_state"] = to_cpu_tree(
        checkpoint["batchnorm_reference_state"]
    )
    apply_batchnorm_policy(objects["embedding_model"])
    assert_batchnorm_running_state_exact(
        objects["embedding_model"], objects["batchnorm_reference_state"]
    )
    restore_rng_state(checkpoint["rng_state"])
    if not values_exactly_equal(capture_rng_state(), checkpoint["rng_state"]):
        raise RuntimeError("checkpoint RNG state did not restore exactly")
    ecapa_step, aam_step = optimizer_all_parameter_steps(objects["optimizer"])
    if (ecapa_step, aam_step) != (
        checkpoint["global_optimizer_step"],
        checkpoint["global_optimizer_step"],
    ):
        raise RuntimeError("loaded AdamW counters disagree with checkpoint")
    return objects, scheduler, early_stopping


def migrate_start_checkpoint(
    pilot: Mapping[str, Any],
    metrics: Mapping[str, Any],
    start_hash: str,
    device: torch.device,
    protected_hashes: Mapping[str, str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    ResumedCosineScheduler,
    EarlyStoppingState,
]:
    objects = create_training_objects(device)
    objects["embedding_model"].load_state_dict(
        pilot["embedding_model_state_dict"], strict=True
    )
    objects["mean_var_norm"].load_state_dict(
        pilot["mean_var_norm_state_dict"], strict=True
    )
    objects["aam"].load_state_dict(pilot["aam_classifier_state_dict"], strict=True)
    objects["optimizer"].load_state_dict(pilot["optimizer_state_dict"])
    objects["scaler"].load_state_dict(pilot["grad_scaler_state_dict"])
    apply_batchnorm_policy(objects["embedding_model"])
    objects["batchnorm_reference_state"] = batchnorm_running_state(
        objects["embedding_model"]
    )
    early_stopping = EarlyStoppingState()
    scheduler = ResumedCosineScheduler(objects["optimizer"])
    if scheduler.factor != 1.0 or scheduler.lrs != BASE_LRS:
        raise RuntimeError("migrated scheduler did not initialize at factor 1.0")
    restore_rng_state(pilot["rng_state"])
    checkpoint = build_checkpoint(
        objects,
        scheduler,
        early_stopping,
        epoch=0,
        next_epoch=1,
        next_position=0,
        pending_validation_epoch=None,
        epoch_metrics=initial_epoch_metrics(metrics),
        completed_epochs=[0],
        current_epoch_statistics={},
        protected_hashes_before=protected_hashes,
        start_hash=start_hash,
        reason="epoch_0_scheduler_migration",
        stop_reason=None,
    )
    return checkpoint, objects, scheduler, early_stopping


def run_optimizer_update(
    objects: Mapping[str, Any],
    logical_batch: Mapping[str, Any],
    *,
    global_step: int,
    optimizer_hook_count: dict[str, int],
) -> dict[str, Any]:
    mean_var_norm = objects["mean_var_norm"]
    embedding_model = objects["embedding_model"]
    aam = objects["aam"]
    optimizer = objects["optimizer"]
    scaler = objects["scaler"]
    device = next(embedding_model.parameters()).device
    apply_batchnorm_policy(embedding_model)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logical_loss = 0.0
    microbatch_count = 0
    from src.aam_training import iter_microbatches
    import torch.nn.functional as F

    for microbatch in iter_microbatches(logical_batch, 4):
        microbatch_count += 1
        features = microbatch["fbank"].to(device=device, dtype=torch.float32)
        labels = microbatch["speaker_label"].to(device=device, dtype=torch.long)
        lengths = torch.ones(4, device=device, dtype=torch.float32)
        with torch.no_grad():
            normalized = mean_var_norm(features, lengths)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            raw_embedding = embedding_model(normalized, lengths)
        if tuple(raw_embedding.shape) != (4, 1, 192):
            raise ValueError("ECAPA embedding shape changed")
        with torch.cuda.amp.autocast(enabled=False):
            logits = aam(raw_embedding.squeeze(1).float(), labels)
            loss = F.cross_entropy(logits.float(), labels, reduction="sum") / 32
        if (
            tuple(logits.shape) != (4, 488)
            or logits.dtype != torch.float32
            or loss.dtype != torch.float32
            or not bool(torch.isfinite(logits).all().item())
            or not bool(torch.isfinite(loss).item())
        ):
            raise ValueError("non-finite AAM logits or loss")
        logical_loss += float(loss.detach().item())
        scaler.scale(loss).backward()
        del features, labels, lengths, normalized, raw_embedding, logits, loss
    if microbatch_count != 8:
        raise RuntimeError("logical update did not contain eight microbatches")
    scaler.unscale_(optimizer)
    ecapa_gradient_norm = finite_gradient_norm(
        list(embedding_model.named_parameters()), "ECAPA"
    )
    aam_gradient_norm = finite_gradient_norm(
        list(aam.named_parameters()), "AAM classifier"
    )
    before_hook = optimizer_hook_count["value"]
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if optimizer_hook_count["value"] != before_hook + 1:
        raise RuntimeError("GradScaler skipped an optimizer update")
    scale_after = float(scaler.get_scale())
    if not math.isfinite(scale_after) or scale_after <= 0 or scale_after < scale_before:
        raise RuntimeError("GradScaler scale indicates a skipped/invalid update")
    ecapa_step, aam_step = optimizer_all_parameter_steps(optimizer)
    if (ecapa_step, aam_step) != (global_step, global_step):
        raise RuntimeError("AdamW counters did not advance exactly once")
    assert_batchnorm_running_state_exact(
        embedding_model, objects["batchnorm_reference_state"]
    )
    torch.cuda.synchronize(device)
    return {
        "logical_loss": logical_loss,
        "ecapa_gradient_norm": ecapa_gradient_norm,
        "aam_gradient_norm": aam_gradient_norm,
        "grad_scaler_scale": scale_after,
        "duration_seconds": time.perf_counter() - started,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def validate_epoch(
    objects: Mapping[str, Any],
    *,
    epoch: int,
    device: torch.device,
    planned_validation_shards: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    mean_var_norm = objects["mean_var_norm"]
    embedding_model = objects["embedding_model"]
    assert_batchnorm_running_state_exact(
        embedding_model, objects["batchnorm_reference_state"]
    )
    model_state_before = to_cpu_tree(embedding_model.state_dict())
    mean_var_norm.eval()
    embedding_model.eval()
    dataset = CachedFbankDataset(
        CACHE_DIR,
        "validation",
        max_cached_shards=TRAIN_LRU_SHARDS,
        validate_finite=False,
    )
    manifest = read_validation_manifest()
    if (
        len(dataset) != VALIDATION_ROWS
        or [(row.relative_audio_path, row.speaker_id) for row in dataset.rows]
        != [(row["relative_audio_path"], row["speaker_id"]) for row in manifest]
    ):
        raise RuntimeError("validation Dataset differs from the approved manifest")
    trial_config = validate_trial_config()
    trial_hash = require_trial_hash(PROJECT_ROOT / TRIAL_RELATIVE)
    trials = read_trials(PROJECT_ROOT / TRIAL_RELATIVE)
    speakers = {row.speaker_id for row in dataset.rows}
    validate_trials(trials, speakers)
    validate_trials_against_metadata(
        trials,
        [(row.relative_audio_path, row.speaker_id) for row in dataset.rows],
        speakers,
    )
    positives = sum(trial.target == 1 for trial in trials)
    negatives = sum(trial.target == 0 for trial in trials)
    if (
        trial_hash != EXPECTED_TRIAL_SHA256
        or len(trials) != TOTAL_TRIALS
        or positives != POSITIVE_TRIALS
        or negatives != NEGATIVE_TRIALS
    ):
        raise RuntimeError("fixed validation trials changed")
    loader = create_cached_fbank_dataloader(
        dataset, VALIDATION_BATCH_SIZE, shuffle=False, num_workers=0
    )
    embeddings: list[torch.Tensor] = []
    paths: list[str] = []
    actual_shards: set[str] = set()
    next_index = 0
    torch.cuda.reset_peak_memory_stats(device)
    extraction_started = time.perf_counter()
    for batch in loader:
        indexes = batch["dataset_index"].tolist()
        if indexes != list(range(next_index, next_index + len(indexes))):
            raise RuntimeError("validation traversal is not sequential")
        next_index += len(indexes)
        if (
            batch["speaker_label"].tolist() != [-1] * len(indexes)
            or batch["final_split"] != ["validation"] * len(indexes)
        ):
            raise RuntimeError("validation batch split/labels changed")
        for index in indexes:
            shard = strict_shard_relative(
                dataset.rows[index].feature_shard_path, "validation"
            )
            actual_shards.add(
                (CACHE_RELATIVE / Path(*PurePosixPath(shard).parts)).as_posix()
            )
        embeddings.append(
            validation_forward_batch(
                mean_var_norm,
                embedding_model,
                batch["fbank"],
                device=device,
            )
        )
        paths.extend(batch["relative_audio_path"])
    torch.cuda.synchronize(device)
    extraction_seconds = time.perf_counter() - extraction_started
    stored = torch.cat(embeddings)
    if (
        next_index != VALIDATION_ROWS
        or tuple(stored.shape) != (VALIDATION_ROWS, 192)
        or stored.dtype != torch.float32
        or not bool(torch.isfinite(stored).all())
        or actual_shards != set(planned_validation_shards)
    ):
        raise RuntimeError("validation embedding invariants failed")
    scoring_started = time.perf_counter()
    scores = score_trials(stored, paths, trials)
    targets = torch.tensor([trial.target for trial in trials], dtype=torch.long)
    scoring_seconds = time.perf_counter() - scoring_started
    metric_started = time.perf_counter()
    metric = calculate_eer(scores.tolist(), targets.tolist())
    metric_seconds = time.perf_counter() - metric_started
    payload = {
        **dataclasses.asdict(metric),
        "same_speaker_scores": numeric_summary(scores[targets == 1].tolist()),
        "different_speaker_scores": numeric_summary(scores[targets == 0].tolist()),
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
    validate_validation_metrics_payload(payload)
    metrics_path = OUTPUT_DIR / f"validation_metrics_epoch_{epoch:03d}.json"
    atomic_json(metrics_path, payload)
    if json.loads(metrics_path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("validation metric JSON roundtrip failed")
    if not values_exactly_equal(
        model_state_before, to_cpu_tree(embedding_model.state_dict())
    ):
        raise RuntimeError("embedding model state changed during validation")
    assert_batchnorm_running_state_exact(
        embedding_model, objects["batchnorm_reference_state"]
    )
    runtime = {
        "validation_rows": VALIDATION_ROWS,
        "validation_speakers": VALIDATION_SPEAKERS,
        "score_count": TOTAL_TRIALS,
        "positive_trials": positives,
        "negative_trials": negatives,
        "extraction_seconds": extraction_seconds,
        "scoring_seconds": scoring_seconds,
        "metric_seconds": metric_seconds,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "aam_classifier_calls": 0,
        "sequential_traversal": True,
        "trial_path_ownership_validated": True,
        "metrics_path": relative_string(metrics_path),
    }
    del loader, dataset, embeddings, stored, scores, targets
    gc.collect()
    torch.cuda.empty_cache()
    return payload, runtime, actual_shards


def core_protected_paths(validation_shards: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                CACHE_CONFIG_RELATIVE.as_posix(),
                TRAIN_INDEX_RELATIVE.as_posix(),
                VALIDATION_INDEX_RELATIVE.as_posix(),
                VALIDATION_MANIFEST_RELATIVE.as_posix(),
                TRIAL_RELATIVE.as_posix(),
                TRIAL_CONFIG_RELATIVE.as_posix(),
                START_CHECKPOINT_RELATIVE.as_posix(),
                START_METRICS_RELATIVE.as_posix(),
                *validation_shards,
            }
        )
    )


def prepare_train_epoch(
    dataset: CachedFbankDataset,
    epoch: int,
    early_stopping: EarlyStoppingState,
) -> tuple[
    HybridShardAwareSpeakerBatchSampler,
    list[list[int]],
    dict[str, Any],
    tuple[str, ...],
]:
    sampler = HybridShardAwareSpeakerBatchSampler(
        dataset,
        speakers_per_batch=16,
        samples_per_speaker=2,
        active_shard_window=8,
        num_batches=1000,
        seed=SEED,
        validate_shard_existence=False,
    )
    start_epoch(epoch, early_stopping, sampler)
    planned = list(iter(sampler))
    audit = validate_sampler_plan(dataset, planned)
    shards = selected_train_shards(dataset, planned)
    return sampler, planned, audit, shards


def train_one_epoch(
    objects: Mapping[str, Any],
    scheduler: ResumedCosineScheduler,
    early_stopping: EarlyStoppingState,
    *,
    dataset: CachedFbankDataset,
    epoch: int,
    start_position: int,
    prior_statistics: Mapping[str, Any],
    epoch_metrics: Sequence[Mapping[str, Any]],
    completed_epochs: Sequence[int],
    protected_hashes_before: dict[str, str],
    start_hash: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    sampler, planned, sampler_audit, train_shards = prepare_train_epoch(
        dataset, epoch, early_stopping
    )
    epoch_hashes = hash_allowed_paths(train_shards)
    for path, digest in epoch_hashes.items():
        prior = protected_hashes_before.setdefault(path, digest)
        if prior != digest:
            raise RuntimeError("protected train input changed before epoch start")
    generator = torch.Generator().manual_seed(SEED + 1000 + epoch)
    loader = create_cached_fbank_training_dataloader(
        dataset,
        planned[start_position:],  # type: ignore[arg-type]
        num_workers=0,
        generator=generator,
    )
    losses = list(prior_statistics.get("losses", []))
    ecapa_gradients = list(prior_statistics.get("ecapa_gradient_norms", []))
    aam_gradients = list(prior_statistics.get("aam_gradient_norms", []))
    durations = list(prior_statistics.get("step_durations", []))
    if not all(len(values) == start_position for values in (
        losses,
        ecapa_gradients,
        aam_gradients,
        durations,
    )):
        raise RuntimeError("rolling checkpoint statistics do not match resume position")
    starting_lrs = prior_statistics.get("starting_lrs", list(scheduler.lrs))
    actual_shards: set[str] = set()
    hook_count = {"value": 0}

    def optimizer_post_hook(
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hook_count["value"] += 1

    hook = objects["optimizer"].register_step_post_hook(optimizer_post_hook)
    epoch_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for position, batch in enumerate(loader, start=start_position):
        logical_batch = validate_logical_batch(dataset, batch, planned[position])
        for index in logical_batch["dataset_index"].tolist():
            shard = strict_shard_relative(
                dataset.rows[index].feature_shard_path, "train"
            )
            actual_shards.add(
                (CACHE_RELATIVE / Path(*PurePosixPath(shard).parts)).as_posix()
            )
        global_step = START_GLOBAL_STEP + scheduler.completed_resumed_steps + 1
        update = run_optimizer_update(
            objects,
            logical_batch,
            global_step=global_step,
            optimizer_hook_count=hook_count,
        )
        scheduler.step()
        losses.append(update["logical_loss"])
        ecapa_gradients.append(update["ecapa_gradient_norm"])
        aam_gradients.append(update["aam_gradient_norm"])
        durations.append(update["duration_seconds"])
        current_statistics = {
            "epoch": epoch,
            "starting_lrs": list(starting_lrs),
            "losses": losses,
            "ecapa_gradient_norms": ecapa_gradients,
            "aam_gradient_norms": aam_gradients,
            "step_durations": durations,
        }
        resumed_position = position + 1
        if scheduler.completed_resumed_steps % ROLLING_INTERVAL == 0:
            next_epoch = epoch if resumed_position < STEPS_PER_EPOCH else epoch + 1
            next_position = resumed_position if resumed_position < STEPS_PER_EPOCH else 0
            pending = None if resumed_position < STEPS_PER_EPOCH else epoch
            checkpoint = build_checkpoint(
                objects,
                scheduler,
                early_stopping,
                epoch=epoch,
                next_epoch=next_epoch,
                next_position=next_position,
                pending_validation_epoch=pending,
                epoch_metrics=epoch_metrics,
                completed_epochs=completed_epochs,
                current_epoch_statistics=current_statistics,
                protected_hashes_before=protected_hashes_before,
                start_hash=start_hash,
                reason="rolling",
                stop_reason=None,
            )
            save_checkpoint(checkpoint, LAST_CHECKPOINT_PATH)
            del checkpoint
        if scheduler.completed_resumed_steps % LOGGING_INTERVAL == 0:
            avg50 = statistics.fmean(losses[-50:])
            record = {
                "epoch": epoch,
                "global_optimizer_step": global_step,
                "resumed_optimizer_steps": scheduler.completed_resumed_steps,
                "logical_batch_position": resumed_position,
                "loss": update["logical_loss"],
                "avg50": avg50,
                "ecapa_lr": scheduler.lrs[0],
                "aam_lr": scheduler.lrs[1],
                "ecapa_gradient_norm": update["ecapa_gradient_norm"],
                "aam_gradient_norm": update["aam_gradient_norm"],
                "grad_scaler_scale": update["grad_scaler_scale"],
                "cuda_reserved_mb": update["cuda_reserved_bytes"] / (1024.0**2),
            }
            append_jsonl(TRAIN_LOG_PATH, record)
            print(
                "TRAIN "
                f"epoch={epoch} step={global_step} batch={resumed_position}/1000 "
                f"loss={update['logical_loss']:.8f} avg50={avg50:.8f} "
                f"ecapa_lr={scheduler.lrs[0]:.12g} aam_lr={scheduler.lrs[1]:.12g} "
                f"ecapa_grad={update['ecapa_gradient_norm']:.8f} "
                f"aam_grad={update['aam_gradient_norm']:.8f} "
                f"scaler={update['grad_scaler_scale']:.1f} "
                f"reserved_mb={update['cuda_reserved_bytes'] / (1024.0**2):.2f}",
                flush=True,
            )
        del batch, logical_batch, update
    hook.remove()
    if len(losses) != STEPS_PER_EPOCH:
        raise RuntimeError("epoch did not complete exactly 1,000 optimizer updates")
    if hook_count["value"] != STEPS_PER_EPOCH - start_position:
        raise RuntimeError("optimizer post-hook count differs from resumed work")
    if start_position == 0 and actual_shards != set(train_shards):
        raise RuntimeError("complete epoch loaded a different train-shard set")
    summary = {
        "epoch": epoch,
        "optimizer_steps": STEPS_PER_EPOCH,
        "resumed_from_batch_position": start_position,
        "new_optimizer_updates": STEPS_PER_EPOCH - start_position,
        "train_loss": statistics.fmean(losses),
        "loss_summary": summarize_finite_values(losses, "epoch losses"),
        "ecapa_gradient_summary": summarize_finite_values(
            ecapa_gradients, "epoch ECAPA gradients"
        ),
        "aam_gradient_summary": summarize_finite_values(
            aam_gradients, "epoch AAM gradients"
        ),
        "step_duration_summary": summarize_finite_values(
            durations, "epoch step durations"
        ),
        "starting_ecapa_lr": float(starting_lrs[0]),
        "starting_aam_lr": float(starting_lrs[1]),
        "ending_ecapa_lr": scheduler.lrs[0],
        "ending_aam_lr": scheduler.lrs[1],
        "duration_seconds": time.perf_counter() - epoch_started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "sampler_audit": sampler_audit,
        "sampler_epoch": sampler.epoch,
        "batchnorm_running_buffers_bit_exact": True,
        "invalid_or_skipped_updates": 0,
    }
    current_statistics = {
        "epoch": epoch,
        "starting_lrs": list(starting_lrs),
        "losses": losses,
        "ecapa_gradient_norms": ecapa_gradients,
        "aam_gradient_norms": aam_gradients,
        "step_durations": durations,
    }
    del loader, sampler, planned
    gc.collect()
    return summary, current_statistics, train_shards


def run(args: argparse.Namespace) -> dict[str, Any]:
    expected_python = (PROJECT_ROOT / ".venv-cuda/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError("multi-epoch run requires .venv-cuda\\Scripts\\python.exe")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in (None, 0) or not torch.cuda.is_available():
        raise RuntimeError("multi-epoch run requires cuda:0")
    validate_runtime_paths()
    pilot, epoch0_metrics, start_hash = load_start_state()
    validation_shards = read_validation_shards_from_index()
    protected_core = hash_allowed_paths(core_protected_paths(validation_shards))
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_started_utc = utc_now()
    task_started = time.perf_counter()
    rolling_results: list[dict[str, Any]] = []
    validation_results: list[dict[str, Any]] = []
    checkpoint_results: list[dict[str, Any]] = []

    if LAST_CHECKPOINT_PATH.exists():
        checkpoint = torch.load(
            LAST_CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        validate_multiepoch_checkpoint(checkpoint)
        if checkpoint["start_checkpoint"]["sha256"] != start_hash:
            raise RuntimeError("resumed checkpoint points to a different epoch-0 start")
        for path, digest in checkpoint["protected_hashes_before"].items():
            if file_sha256(PROJECT_ROOT / Path(*PurePosixPath(path).parts)) != digest:
                raise RuntimeError("protected input changed before checkpoint resume")
        objects, scheduler, early_stopping = load_objects_from_checkpoint(
            checkpoint, device
        )
    else:
        existing = [
            path
            for path in (
                BEST_CHECKPOINT_PATH,
                RUNTIME_PATH,
                FAILURE_PATH,
                TRAIN_LOG_PATH,
            )
            if path.exists()
        ]
        existing.extend(OUTPUT_DIR.glob("epoch_*.pt"))
        if existing:
            raise FileExistsError(
                "multi-epoch output exists without last.pt; refusing ambiguous restart"
            )
        (
            checkpoint,
            objects,
            scheduler,
            early_stopping,
        ) = migrate_start_checkpoint(
            pilot,
            epoch0_metrics,
            start_hash,
            device,
            protected_core,
        )
        migration_last = save_checkpoint(checkpoint, LAST_CHECKPOINT_PATH)
        migration_best = save_checkpoint(checkpoint, BEST_CHECKPOINT_PATH)
        checkpoint_results.extend([migration_last, migration_best])

    epoch_metrics = [dict(value) for value in checkpoint["epoch_metrics"]]
    completed_epochs = list(checkpoint["completed_epochs"])
    protected_hashes_before = dict(checkpoint["protected_hashes_before"])
    next_epoch = checkpoint["next_epoch"]
    next_position = checkpoint["next_logical_batch_position"]
    pending_validation_epoch = checkpoint["pending_validation_epoch"]
    current_statistics = dict(checkpoint["current_epoch_statistics"])
    if completed_epochs[0] != 0 or 0 in completed_epochs[1:]:
        raise RuntimeError("epoch 0 completion history is invalid")
    if early_stopping.should_stop and checkpoint["stop_reason"] is None:
        raise RuntimeError("checkpoint would start after early stopping")
    if checkpoint["stop_reason"] is not None:
        stop_reason = checkpoint["stop_reason"]
    else:
        stop_reason = None

    dataset = CachedFbankDataset(
        CACHE_DIR,
        "train",
        max_cached_shards=TRAIN_LRU_SHARDS,
        validate_finite=False,
    )
    if (
        len(dataset) != 31998
        or len({row.speaker_id for row in dataset.rows}) != 488
        or {row.speaker_label for row in dataset.rows} != set(range(488))
    ):
        raise RuntimeError("approved train Dataset invariants failed")
    validation_speakers = {row["speaker_id"] for row in read_validation_manifest()}
    if not {row.speaker_id for row in dataset.rows}.isdisjoint(validation_speakers):
        raise RuntimeError("train and validation speakers are not disjoint")

    while stop_reason is None:
        if pending_validation_epoch is None:
            if next_epoch > LAST_RESUMED_EPOCH:
                stop_reason = "max_epoch"
                break
            epoch = next_epoch
            train_summary, current_statistics, train_shards = train_one_epoch(
                objects,
                scheduler,
                early_stopping,
                dataset=dataset,
                epoch=epoch,
                start_position=next_position,
                prior_statistics=current_statistics,
                epoch_metrics=epoch_metrics,
                completed_epochs=completed_epochs,
                protected_hashes_before=protected_hashes_before,
                start_hash=start_hash,
                device=device,
            )
            pending_validation_epoch = epoch
            next_epoch = epoch + 1
            next_position = 0
        else:
            epoch = pending_validation_epoch
            train_summary = {
                "epoch": epoch,
                "optimizer_steps": STEPS_PER_EPOCH,
                "train_loss": statistics.fmean(current_statistics["losses"]),
                "loss_summary": summarize_finite_values(
                    current_statistics["losses"], "resumed epoch losses"
                ),
                "ecapa_gradient_summary": summarize_finite_values(
                    current_statistics["ecapa_gradient_norms"],
                    "resumed epoch ECAPA gradients",
                ),
                "aam_gradient_summary": summarize_finite_values(
                    current_statistics["aam_gradient_norms"],
                    "resumed epoch AAM gradients",
                ),
                "step_duration_summary": summarize_finite_values(
                    current_statistics["step_durations"],
                    "resumed epoch step durations",
                ),
                "starting_ecapa_lr": current_statistics["starting_lrs"][0],
                "starting_aam_lr": current_statistics["starting_lrs"][1],
                "ending_ecapa_lr": scheduler.lrs[0],
                "ending_aam_lr": scheduler.lrs[1],
                "duration_seconds": sum(current_statistics["step_durations"]),
                "sampler_epoch": epoch,
                "batchnorm_running_buffers_bit_exact": True,
                "invalid_or_skipped_updates": 0,
            }

        validation_metrics, validation_runtime, actual_validation_shards = validate_epoch(
            objects,
            epoch=epoch,
            device=device,
            planned_validation_shards=validation_shards,
        )
        if actual_validation_shards != set(validation_shards):
            raise RuntimeError("validation loaded an unexpected shard")
        improved = early_stopping.observe(
            epoch, validation_metrics["interpolated_eer"]
        )
        completed_epochs.append(epoch)
        epoch_record = {
            **train_summary,
            "validation": validation_metrics,
            "validation_runtime": validation_runtime,
            "best": improved,
            "patience_counter": early_stopping.patience_counter,
        }
        epoch_metrics.append(epoch_record)
        pending_validation_epoch = None
        current_statistics = {}
        if early_stopping.should_stop:
            stop_reason = "early_stopping"
        elif epoch == LAST_RESUMED_EPOCH:
            stop_reason = "max_epoch"
        checkpoint = build_checkpoint(
            objects,
            scheduler,
            early_stopping,
            epoch=epoch,
            next_epoch=epoch + 1,
            next_position=0,
            pending_validation_epoch=None,
            epoch_metrics=epoch_metrics,
            completed_epochs=completed_epochs,
            current_epoch_statistics={},
            protected_hashes_before=protected_hashes_before,
            start_hash=start_hash,
            reason="epoch_complete",
            stop_reason=stop_reason,
        )
        epoch_path = OUTPUT_DIR / f"epoch_{epoch:03d}.pt"
        epoch_result = save_checkpoint(checkpoint, epoch_path)
        last_result = save_checkpoint(checkpoint, LAST_CHECKPOINT_PATH)
        checkpoint_results.extend([epoch_result, last_result])
        if improved:
            best_checkpoint = dict(checkpoint)
            best_checkpoint["checkpoint_creation_reason"] = "best_validation"
            best_result = save_checkpoint(best_checkpoint, BEST_CHECKPOINT_PATH)
            checkpoint_results.append(best_result)
            del best_checkpoint
        validation_results.append(validation_runtime)
        print(
            "EPOCH_RESULT "
            f"epoch={epoch} train_loss={train_summary['train_loss']:.10f} "
            f"val_eer={validation_metrics['interpolated_eer']:.15f} "
            f"threshold={validation_metrics['empirical_threshold']:.15f} "
            f"best_epoch={early_stopping.best_epoch} "
            f"best_eer={early_stopping.best_eer:.15f} "
            f"patience={early_stopping.patience_counter}",
            flush=True,
        )
        next_epoch = epoch + 1
        next_position = 0
        del checkpoint
        gc.collect()

    protected_after = hash_allowed_paths(tuple(sorted(protected_hashes_before)))
    mismatches = {
        path: {
            "before": protected_hashes_before[path],
            "after": protected_after[path],
        }
        for path in protected_hashes_before
        if protected_hashes_before[path] != protected_after[path]
    }
    if mismatches:
        raise RuntimeError("protected input hash changed during multi-epoch run")
    if scheduler.completed_resumed_steps != len(completed_epochs[1:]) * 1000:
        raise RuntimeError("completed epochs and resumed optimizer steps disagree")
    best_checkpoint = torch.load(
        BEST_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    validate_multiepoch_checkpoint(best_checkpoint)
    if (
        best_checkpoint["epoch"] != early_stopping.best_epoch
        or best_checkpoint["early_stopping_state"]["best_eer"]
        != early_stopping.best_eer
    ):
        raise RuntimeError("best.pt does not identify the selected best epoch/EER")
    print(
        "TRAINING_STOP "
        f"reason={stop_reason} best_epoch={early_stopping.best_epoch} "
        f"best_eer={early_stopping.best_eer:.15f}",
        flush=True,
    )
    task_duration = time.perf_counter() - task_started
    validation_history = [
        record["validation_runtime"]
        for record in epoch_metrics
        if isinstance(record.get("validation_runtime"), Mapping)
    ]
    peak_allocated = max(
        [record.get("cuda_peak_allocated_bytes", 0) for record in epoch_metrics]
        + [record.get("cuda_peak_allocated_bytes", 0) for record in validation_history]
        + [record.get("cuda_peak_allocated_bytes", 0) for record in validation_results]
    )
    peak_reserved = max(
        [record.get("cuda_peak_reserved_bytes", 0) for record in epoch_metrics]
        + [record.get("cuda_peak_reserved_bytes", 0) for record in validation_history]
        + [record.get("cuda_peak_reserved_bytes", 0) for record in validation_results]
    )
    result = {
        "schema_name": "ecapa_aam_multiepoch_runtime",
        "schema_version": 1,
        "technical_pass": True,
        "model_quality_result": (
            "IMPROVED"
            if early_stopping.best_eer < INITIAL_BEST_EER
            else "NO_NEW_IMPROVEMENT"
        ),
        "task_started_utc": task_started_utc,
        "task_ended_utc": utc_now(),
        "task_duration_seconds": task_duration,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "speechbrain": speechbrain.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        },
        "start_checkpoint": {
            "path": START_CHECKPOINT_RELATIVE.as_posix(),
            "sha256": start_hash,
            "epoch": 0,
            "global_optimizer_step": 1000,
            "validation_eer": INITIAL_BEST_EER,
            "empirical_threshold": INITIAL_EMPIRICAL_THRESHOLD,
        },
        "completed_epochs": completed_epochs,
        "stop_reason": stop_reason,
        "best_epoch": early_stopping.best_epoch,
        "best_eer": early_stopping.best_eer,
        "best_checkpoint": {
            "path": relative_string(BEST_CHECKPOINT_PATH),
            "sha256": file_sha256(BEST_CHECKPOINT_PATH),
        },
        "epoch_metrics": epoch_metrics,
        "scheduler": {
            **scheduler.state_dict(),
            "start_factor": 1.0,
            "end_factor_at_4000": 0.1,
            "minimum_lrs": [1.0e-6, 1.0e-4],
            "migration_from_epoch_0_recorded": True,
        },
        "early_stopping": early_stopping.state_dict(),
        "checkpoint_results": checkpoint_results,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "batchnorm_running_buffers_compared": len(
            objects["batchnorm_reference_state"]
        ),
        "batchnorm_running_buffers_bit_exact": True,
        "validation_trial_hash": require_trial_hash(
            PROJECT_ROOT / TRIAL_RELATIVE
        ),
        "protected_hashes": {
            "files_compared": len(protected_hashes_before),
            "before_set_sha256": hash_mapping_digest(protected_hashes_before),
            "after_set_sha256": hash_mapping_digest(protected_after),
            "mismatches": mismatches,
            "unchanged": True,
            "before": protected_hashes_before,
            "after": protected_after,
        },
        "final_test_accessed": False,
        "invalid_or_skipped_updates": 0,
        "oom": False,
        "commit_or_push_performed": False,
        "training_configuration": TRAINING_CONFIGURATION,
        "baseline_metrics": BASELINE_METRICS,
    }
    atomic_json(RUNTIME_PATH, result)
    return result


def main() -> None:
    try:
        run(parse_args())
    except BaseException as error:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_name": "ecapa_aam_multiepoch_failure",
            "schema_version": 1,
            "error": safe_error(error),
            "cuda_oom": "out of memory" in str(error).lower(),
            "final_test_accessed": False,
            "created_utc": utc_now(),
        }
        atomic_json(FAILURE_PATH, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        raise


if __name__ == "__main__":
    main()
