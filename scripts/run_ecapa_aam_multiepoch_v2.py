#!/usr/bin/env python
"""Resume VieSpeaker2.0 ECAPA/AAM training through epoch four at most."""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import speechbrain  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.run_ecapa_aam_one_epoch_v2 import (  # noqa: E402
    CACHE_DIR,
    TRAIN_MANIFEST,
    VALIDATION_MANIFEST,
    atomic_json,
    create_training_objects,
    file_sha256,
    prepare_logical_batch,
    register_optimizer_hook,
    run_validation,
    verify_upstream_files,
)
from src.aam_training import (  # noqa: E402
    PRETRAINED_MODEL_ID,
    apply_batchnorm_policy,
    capture_rng_state,
    iter_microbatches,
    restore_rng_state,
    to_cpu_tree,
    values_exactly_equal,
)
from src.cached_fbank_dataset import (  # noqa: E402
    CachedFbankDataset,
    create_cached_fbank_training_dataloader,
)
from src.cached_fbank_samplers import (  # noqa: E402
    HybridShardAwareSpeakerBatchSampler,
)
from src.ecapa_multiepoch_v2 import (  # noqa: E402
    BASE_LRS,
    AMP_OVERFLOW_POLICY,
    BATCHNORM_POLICY,
    EARLY_STOPPING_PATIENCE,
    EPOCH_PLAN_HASHES,
    FIRST_RESUMED_EPOCH,
    HISTORICAL_FAILURE_BINDINGS,
    IMPROVEMENT_TOLERANCE,
    INITIAL_BEST_EER,
    INITIAL_BEST_EPOCH,
    INITIAL_EMPIRICAL_AVERAGE_ERROR,
    INITIAL_EMPIRICAL_THRESHOLD,
    LAST_RESUMED_EPOCH,
    MULTIEPOCH_TRAINING_CONFIGURATION,
    RECOVERY_SCHEMA_VERSION,
    RECOVERY_START_CHECKPOINT_PATH,
    RECOVERY_START_CHECKPOINT_SHA256,
    RUNTIME_SCHEMA_NAME,
    RUNTIME_SCHEMA_VERSION,
    SAMPLER_BINDING,
    SCHEDULER_CONFIGURATION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    START_CHECKPOINT_PATH,
    START_CHECKPOINT_SHA256,
    START_GLOBAL_STEP,
    START_REPORT_PATH,
    START_REPORT_SHA256,
    STEPS_PER_EPOCH,
    TOTAL_RESUMED_UPDATES,
    UPSTREAM_BINDINGS,
    EarlyStoppingState,
    ExactResumedCosineScheduler,
    audit_epoch_plan,
    classify_recoverable_overflow,
    gradient_diagnostics,
    logical_batch_identity,
    migrate_checkpoint_for_amp_recovery,
    reject_final_test_path,
    retry_scale,
    stop_reason_after_validation,
    validate_epoch0_migration,
    validate_failed_attempt_invariants,
    validate_multiepoch_checkpoint,
)
from src.ecapa_one_epoch_v2 import (  # noqa: E402
    ACCUMULATION_STEPS,
    AAM_CONFIGURATION,
    EMBEDDING_DIM,
    FEATURE_SHAPE,
    LOGICAL_BATCH_SIZE,
    MIXED_PRECISION_CONFIGURATION,
    NUM_CLASSES,
    OPTIMIZER_CONFIGURATION,
    PHYSICAL_MICROBATCH_SIZE,
    SEED,
    assert_batchnorm_reference_exact,
    assert_finite_tensor_tree,
    assert_module_parameters_finite,
    batchnorm_reference_state,
    finite_gradient_norm,
    optimizer_parameter_steps,
    summarize_values,
)
from src.training_readiness_v2 import (  # noqa: E402
    plan_sha256,
    validate_train_manifest_alignment,
)
from src.verification_metrics import calculate_eer  # noqa: E402
from src.verification_v2 import (  # noqa: E402
    empirical_confusion,
    read_validation_manifest,
)


OUTPUT_RELATIVE = Path("outputs/ecapa_aam_multiepoch_v2")
OUTPUT_DIR = REPO_ROOT / OUTPUT_RELATIVE
LAST_PATH = OUTPUT_DIR / "last.pt"
BEST_PATH = OUTPUT_DIR / "best.pt"
TRAIN_LOG_PATH = OUTPUT_DIR / "training_scalars.jsonl"
RUNTIME_PATH = OUTPUT_DIR / "runtime_v2.json"
FAILURE_PATH = OUTPUT_DIR / "failure_v2.json"
STDOUT_PATH = OUTPUT_DIR / "production.stdout.log"
STDERR_PATH = OUTPUT_DIR / "production.stderr.log"
REPORT_JSON = REPO_ROOT / "reports/ecapa_aam_multiepoch_v2.json"
REPORT_MD = REPO_ROOT / "reports/ecapa_aam_multiepoch_v2.md"
RECOVERY_COPY_PATH = OUTPUT_DIR / "recovery_start_last.pt"
RECONCILIATION_PATH = OUTPUT_DIR / "reconciliation_recovery_v2.json"
RECOVERY_RUNTIME_PATH = OUTPUT_DIR / "runtime_recovery_v2.json"
RECOVERY_FAILURE_PATH = OUTPUT_DIR / "failure_recovery_v2.json"
RECOVERY_STDOUT_PATH = OUTPUT_DIR / "recovery.stdout.log"
RECOVERY_STDERR_PATH = OUTPUT_DIR / "recovery.stderr.log"
RECOVERY_REPORT_JSON = REPO_ROOT / "reports/ecapa_aam_multiepoch_v2_recovery.json"
RECOVERY_REPORT_MD = REPO_ROOT / "reports/ecapa_aam_multiepoch_v2_recovery.md"
FINAL_REPORT_JSON = REPO_ROOT / "reports/ecapa_aam_multiepoch_v2_final.json"
FINAL_REPORT_MD = REPO_ROOT / "reports/ecapa_aam_multiepoch_v2_final.md"
ROLLING_INTERVAL = 1000
LOGGING_INTERVAL = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--amp-overflow-retries", required=True, type=int)
    parser.add_argument("--amp-overflow-scale-factor", required=True, type=float)
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
        stream.flush()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def truncate_training_log(
    resumed_optimizer_steps: int,
    path: Path = TRAIN_LOG_PATH,
) -> dict[str, Any]:
    if not path.exists():
        return {"removed_count": 0, "removed_resumed_steps": [], "retained_count": 0}
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [int(record["resumed_optimizer_steps"]) for record in records]
    if steps != sorted(set(steps)):
        raise RuntimeError("training scalar log has duplicate or decreasing steps")
    retained = [
        record
        for record in records
        if record["resumed_optimizer_steps"] <= resumed_optimizer_steps
    ]
    removed_records = records[len(retained) :]
    if any(
        int(record["resumed_optimizer_steps"]) <= resumed_optimizer_steps
        for record in removed_records
    ):
        raise RuntimeError("scalar reconciliation is not a suffix truncation")
    if removed_records:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in retained:
                stream.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"))
                )
                stream.write("\n")
        os.replace(temporary, path)
    return {
        "removed_count": len(removed_records),
        "removed_resumed_steps": [
            int(record["resumed_optimizer_steps"]) for record in removed_records
        ],
        "retained_count": len(retained),
    }


def protected_hashes() -> dict[str, str]:
    bindings = {
        item["path"]: item["sha256"] for item in UPSTREAM_BINDINGS.values()
    }
    bindings[START_CHECKPOINT_PATH] = START_CHECKPOINT_SHA256
    bindings[START_REPORT_PATH] = START_REPORT_SHA256
    bindings.update(
        {
            item["path"]: item["sha256"]
            for item in HISTORICAL_FAILURE_BINDINGS.values()
        }
    )
    actual: dict[str, str] = {}
    for path, expected in sorted(bindings.items()):
        reject_final_test_path(path)
        digest = file_sha256(REPO_ROOT / Path(path))
        if digest != expected:
            raise RuntimeError(f"protected input identity changed: {path}")
        actual[path] = digest
    return actual


def build_plans(
    dataset: CachedFbankDataset,
) -> tuple[dict[int, list[list[int]]], dict[int, dict[str, Any]]]:
    plans: dict[int, list[list[int]]] = {}
    audits: dict[int, dict[str, Any]] = {}
    sampler = HybridShardAwareSpeakerBatchSampler(
        dataset,
        speakers_per_batch=16,
        samples_per_speaker=2,
        active_shard_window=8,
        num_batches=STEPS_PER_EPOCH,
        seed=SEED,
        validate_shard_existence=False,
    )
    for epoch in range(FIRST_RESUMED_EPOCH, LAST_RESUMED_EPOCH + 1):
        sampler.set_epoch(epoch)
        plan = list(sampler)
        digest = plan_sha256(epoch, plan)
        if digest != EPOCH_PLAN_HASHES[epoch]:
            raise RuntimeError(f"epoch {epoch} sampler plan identity changed")
        audit = audit_epoch_plan(dataset, plan, epoch=epoch)
        audit["sha256"] = digest
        plans[epoch] = plan
        audits[epoch] = audit
    return plans, audits


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    expected_python = (REPO_ROOT / ".venv-cuda/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError("production run requires .venv-cuda")
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("production run requires cuda:0")
    if (
        args.amp_overflow_retries
        != AMP_OVERFLOW_POLICY["maximum_retries_per_logical_batch"]
        or args.amp_overflow_scale_factor
        != AMP_OVERFLOW_POLICY["scale_reduction_factor"]
    ):
        raise ValueError("CLI AMP overflow policy differs from the approved policy")
    resume = Path(args.resume.replace("\\", "/"))
    if resume.as_posix() != RECOVERY_START_CHECKPOINT_PATH:
        raise ValueError("resume path is not the approved recovery last.pt")
    if file_sha256(REPO_ROOT / START_REPORT_PATH) != START_REPORT_SHA256:
        raise RuntimeError("epoch-zero report SHA-256 changed")
    upstream = verify_upstream_files()
    before = protected_hashes()
    epoch_one_digest = HISTORICAL_FAILURE_BINDINGS["epoch_001"]["sha256"]
    if (
        not BEST_PATH.exists()
        or file_sha256(BEST_PATH) != epoch_one_digest
        or file_sha256(epoch_artifact_paths(1)["checkpoint"])
        != epoch_one_digest
    ):
        raise RuntimeError("provisional epoch-one best identity changed")
    start_checkpoint = torch.load(
        REPO_ROOT / START_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    start_report = json.loads(
        (REPO_ROOT / START_REPORT_PATH).read_text(encoding="utf-8")
    )
    validate_epoch0_migration(start_checkpoint, start_report)
    dataset = CachedFbankDataset(
        CACHE_DIR,
        "train",
        max_cached_shards=8,
        validate_finite=False,
    )
    alignment = validate_train_manifest_alignment(dataset, TRAIN_MANIFEST)
    if (
        len(dataset) != 95009
        or alignment["aligned_rows"] != 95009
        or len({row.speaker_id for row in dataset.rows}) != NUM_CLASSES
        or {row.speaker_label for row in dataset.rows} != set(range(NUM_CLASSES))
    ):
        raise RuntimeError("approved v2 train Dataset invariants changed")
    validation_manifest = read_validation_manifest(VALIDATION_MANIFEST)
    if len(validation_manifest) != 15355:
        raise RuntimeError("approved validation manifest count changed")
    if not {row.speaker_id for row in dataset.rows}.isdisjoint(
        {row.speaker_id for row in validation_manifest}
    ):
        raise RuntimeError("train and validation speakers overlap")
    plans, plan_audits = build_plans(dataset)
    if plan_audits[1]["sha256"] != SAMPLER_BINDING["epoch_1_plan_sha256"]:
        raise RuntimeError("approved epoch-one plan did not reproduce")
    if not LAST_PATH.exists():
        raise FileNotFoundError("approved recovery last.pt is missing")
    existing_last = torch.load(LAST_PATH, map_location="cpu", weights_only=False)
    validate_multiepoch_checkpoint(existing_last)
    last_digest = file_sha256(LAST_PATH)
    if existing_last["schema_version"] == SCHEMA_VERSION:
        if last_digest != RECOVERY_START_CHECKPOINT_SHA256:
            raise RuntimeError("approved recovery last.pt SHA-256 changed")
        recovery_restart = False
    else:
        if (
            not RECOVERY_COPY_PATH.exists()
            or file_sha256(RECOVERY_COPY_PATH) != RECOVERY_START_CHECKPOINT_SHA256
        ):
            raise RuntimeError("schema-v3 restart lacks the exact recovery origin copy")
        recovery_restart = True
    recovered_failure = json.loads(FAILURE_PATH.read_text(encoding="utf-8"))
    return {
        "start_checkpoint": start_checkpoint,
        "start_report": start_report,
        "upstream": upstream,
        "protected_before": before,
        "dataset": dataset,
        "plans": plans,
        "plan_audits": plan_audits,
        "existing_last": existing_last,
        "recovered_failure": recovered_failure,
        "recovery_restart": recovery_restart,
    }


def reconcile_recovery_artifacts(
    checkpoint: Mapping[str, Any],
    *,
    restart: bool,
) -> dict[str, Any]:
    """Bind the atomic cursor and discard only uncheckpointed scalar suffix rows."""
    if restart:
        if not RECONCILIATION_PATH.exists():
            raise RuntimeError("recovery restart lacks reconciliation evidence")
        reconciliation = json.loads(
            RECONCILIATION_PATH.read_text(encoding="utf-8")
        )
        if (
            reconciliation.get("result") != "PASS"
            or reconciliation.get("recovery_checkpoint_sha256")
            != RECOVERY_START_CHECKPOINT_SHA256
        ):
            raise RuntimeError("recovery reconciliation evidence is invalid")
        return reconciliation
    if checkpoint["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("initial reconciliation requires the historical schema")
    if file_sha256(LAST_PATH) != RECOVERY_START_CHECKPOINT_SHA256:
        raise RuntimeError("recovery checkpoint changed before reconciliation")
    historical_report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    historical_artifact_hashes = historical_report["artifact_hashes"]
    committed_artifacts = {
        "logical_losses_epoch_001.json": epoch_artifact_paths(1)["losses"],
        "validation_embeddings_epoch_001.pt": epoch_artifact_paths(1)[
            "embeddings"
        ],
        "validation_scores_epoch_001.pt": epoch_artifact_paths(1)["scores"],
        "validation_metrics_epoch_001.json": epoch_artifact_paths(1)["metrics"],
    }
    committed_artifact_hashes: dict[str, str] = {}
    for name, path in committed_artifacts.items():
        digest = file_sha256(path)
        if digest != historical_artifact_hashes[name]:
            raise RuntimeError(f"committed epoch-one artifact changed: {name}")
        committed_artifact_hashes[name] = digest
    scalar_log_before = file_sha256(TRAIN_LOG_PATH)
    if scalar_log_before != historical_artifact_hashes["training_scalars.jsonl"]:
        raise RuntimeError("historical scalar log identity changed before reconciliation")
    atomic_copy(LAST_PATH, RECOVERY_COPY_PATH)
    if file_sha256(RECOVERY_COPY_PATH) != RECOVERY_START_CHECKPOINT_SHA256:
        raise RuntimeError("recovery checkpoint copy failed identity verification")
    scalar_reconciliation = truncate_training_log(5000)
    if scalar_reconciliation["removed_resumed_steps"] != [
        5100,
        5200,
        5300,
        5400,
        5500,
    ]:
        raise RuntimeError("unexpected uncheckpointed scalar suffix")
    absent_epoch_two = [
        relative(path)
        for path in epoch_artifact_paths(2).values()
        if path.exists()
    ]
    if absent_epoch_two:
        raise RuntimeError(
            f"unexpected epoch-two artifacts before recovery: {absent_epoch_two}"
        )
    reconciliation = {
        "schema_name": "viespeaker2_multiepoch_amp_recovery_reconciliation",
        "schema_version": 1,
        "result": "PASS",
        "recovery_checkpoint_path": relative(RECOVERY_COPY_PATH),
        "recovery_checkpoint_sha256": RECOVERY_START_CHECKPOINT_SHA256,
        "approved_cursor": {
            "phase": "training",
            "epoch": 2,
            "completed_batch_position": 2031,
            "global_optimizer_step": 7969,
            "resumed_optimizer_steps": 5000,
        },
        "scalar_log": scalar_reconciliation,
        "scalar_log_sha256_before": scalar_log_before,
        "scalar_log_sha256_after": file_sha256(TRAIN_LOG_PATH),
        "committed_epoch_one_artifact_hashes": committed_artifact_hashes,
        "epoch_001_checkpoint_sha256": file_sha256(
            epoch_artifact_paths(1)["checkpoint"]
        ),
        "epoch_two_artifacts_present_before_recovery": absent_epoch_two,
        "historical_failure_bindings": copy.deepcopy(
            HISTORICAL_FAILURE_BINDINGS
        ),
        "historical_artifacts_modified": False,
    }
    atomic_json(RECONCILIATION_PATH, reconciliation)
    return reconciliation


def initial_validation_history(start_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "epoch": 0,
            "training": {
                "optimizer_steps": STEPS_PER_EPOCH,
                "global_optimizer_step": START_GLOBAL_STEP,
                "train_loss": start_report["training"]["loss"]["mean"],
            },
            "metrics": copy.deepcopy(start_report["validation"]["metrics"]),
            "runtime": copy.deepcopy(start_report["validation"]["runtime"]),
            "artifacts": copy.deepcopy(start_report["validation"]["artifacts"]),
            "best": True,
            "bad_epoch_count": 0,
        }
    ]


def initial_best_state() -> dict[str, Any]:
    return {
        "epoch": INITIAL_BEST_EPOCH,
        "interpolated_eer": INITIAL_BEST_EER,
        "empirical_average_error": INITIAL_EMPIRICAL_AVERAGE_ERROR,
        "empirical_threshold": INITIAL_EMPIRICAL_THRESHOLD,
        "checkpoint_path": START_CHECKPOINT_PATH,
        "validation_source_checkpoint_path": START_CHECKPOINT_PATH,
        "validation_source_checkpoint_sha256": START_CHECKPOINT_SHA256,
    }


def create_generator(state: torch.Tensor) -> torch.Generator:
    generator = torch.Generator()
    generator.set_state(state)
    return generator


def load_state_into_objects(
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    dict[str, Any],
    ExactResumedCosineScheduler,
    EarlyStoppingState,
    torch.Generator,
]:
    objects = create_training_objects(device)
    objects["embedding_model"].load_state_dict(
        checkpoint["embedding_model_state"], strict=True
    )
    objects["mean_var_norm"].load_state_dict(
        checkpoint["mean_var_norm_state"], strict=True
    )
    objects["aam"].load_state_dict(checkpoint["aam_state"], strict=True)
    objects["optimizer"].load_state_dict(checkpoint["optimizer_state"])
    objects["scaler"].load_state_dict(checkpoint["grad_scaler_state"])
    scheduler = ExactResumedCosineScheduler(
        objects["optimizer"],
        completed_updates=checkpoint["resumed_optimizer_steps"],
    )
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    early = EarlyStoppingState.from_state_dict(
        checkpoint["early_stopping_state"]
    )
    generator = create_generator(checkpoint["dataloader_generator_state"])
    apply_batchnorm_policy(objects["embedding_model"])
    assert_batchnorm_reference_exact(
        objects["embedding_model"], checkpoint["batchnorm_reference_state"]
    )
    restore_rng_state(checkpoint["rng_state"])
    if not values_exactly_equal(capture_rng_state(), checkpoint["rng_state"]):
        raise RuntimeError("checkpoint RNG state did not restore exactly")
    if not torch.equal(
        generator.get_state(), checkpoint["dataloader_generator_state"]
    ):
        raise RuntimeError("DataLoader generator state did not restore exactly")
    if optimizer_parameter_steps(objects["optimizer"]) != (
        checkpoint["global_optimizer_step"],
        checkpoint["global_optimizer_step"],
    ):
        raise RuntimeError("restored optimizer counters disagree")
    return objects, scheduler, early, generator


def migrate_start(
    start: Mapping[str, Any],
    start_report: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    dict[str, Any],
    ExactResumedCosineScheduler,
    EarlyStoppingState,
    torch.Generator,
    Mapping[str, torch.Tensor],
]:
    objects = create_training_objects(device)
    objects["embedding_model"].load_state_dict(
        start["embedding_model_state"], strict=True
    )
    objects["mean_var_norm"].load_state_dict(
        start["mean_var_norm_state"], strict=True
    )
    objects["aam"].load_state_dict(start["aam_state"], strict=True)
    objects["optimizer"].load_state_dict(start["optimizer_state"])
    objects["scaler"].load_state_dict(start["grad_scaler_state"])
    scheduler = ExactResumedCosineScheduler(objects["optimizer"])
    early = EarlyStoppingState()
    generator = create_generator(start["dataloader_generator_state"])
    apply_batchnorm_policy(objects["embedding_model"])
    bn_reference = to_cpu_tree(start["batchnorm_reference_state"])
    assert_batchnorm_reference_exact(objects["embedding_model"], bn_reference)
    restore_rng_state(start["rng_state"])
    if scheduler.next_update_index != 0 or scheduler.current_lrs != BASE_LRS:
        raise RuntimeError("scheduler migration did not initialize at update zero")
    if optimizer_parameter_steps(objects["optimizer"]) != (
        START_GLOBAL_STEP,
        START_GLOBAL_STEP,
    ):
        raise RuntimeError("migrated optimizer counters changed")
    return objects, scheduler, early, generator, bn_reference


def build_checkpoint(
    objects: Mapping[str, Any],
    scheduler: ExactResumedCosineScheduler,
    early: EarlyStoppingState,
    generator: torch.Generator,
    bn_reference: Mapping[str, torch.Tensor],
    *,
    current_epoch: int,
    completed_batch_position: int,
    phase: str,
    completed_epochs: Sequence[int],
    validation_history: Sequence[Mapping[str, Any]],
    best_state: Mapping[str, Any],
    current_epoch_statistics: Mapping[str, Any],
    overflow_state: Mapping[str, Any],
    reason: str,
    stop_reason: str | None,
) -> dict[str, Any]:
    resumed = scheduler.scheduler_completed_updates
    if phase == "training":
        cursor = {
            "next_epoch": current_epoch,
            "next_batch_position": completed_batch_position,
        }
    else:
        cursor = {"next_epoch": current_epoch + 1, "next_batch_position": 0}
    checkpoint = {
        "schema_name": SCHEMA_NAME,
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "recovery_origin": {
            "path": RECOVERY_START_CHECKPOINT_PATH,
            "sha256": RECOVERY_START_CHECKPOINT_SHA256,
            "historical_failure_bindings": copy.deepcopy(
                HISTORICAL_FAILURE_BINDINGS
            ),
        },
        "overflow_policy": copy.deepcopy(AMP_OVERFLOW_POLICY),
        "overflow_state": copy.deepcopy(dict(overflow_state)),
        "original_epoch_zero_checkpoint": {
            "path": START_CHECKPOINT_PATH,
            "sha256": START_CHECKPOINT_SHA256,
            "completed_epoch": 0,
            "global_optimizer_step": START_GLOBAL_STEP,
        },
        "epoch_zero_report": {
            "path": START_REPORT_PATH,
            "sha256": START_REPORT_SHA256,
        },
        "upstream_bindings": copy.deepcopy(UPSTREAM_BINDINGS),
        "sampler_binding": copy.deepcopy(SAMPLER_BINDING),
        "epoch_plan_hashes": {
            str(key): value for key, value in EPOCH_PLAN_HASHES.items()
        },
        "model_source": PRETRAINED_MODEL_ID,
        "feature_shape": list(FEATURE_SHAPE),
        "class_count": NUM_CLASSES,
        "label_range": [0, NUM_CLASSES - 1],
        "aam_configuration": copy.deepcopy(AAM_CONFIGURATION),
        "optimizer_configuration": copy.deepcopy(OPTIMIZER_CONFIGURATION),
        "training_configuration": copy.deepcopy(
            MULTIEPOCH_TRAINING_CONFIGURATION
        ),
        "mixed_precision_configuration": copy.deepcopy(
            MIXED_PRECISION_CONFIGURATION
        ),
        "scheduler_configuration": copy.deepcopy(SCHEDULER_CONFIGURATION),
        "embedding_model_state": to_cpu_tree(
            objects["embedding_model"].state_dict()
        ),
        "mean_var_norm_state": to_cpu_tree(
            objects["mean_var_norm"].state_dict()
        ),
        "aam_state": to_cpu_tree(objects["aam"].state_dict()),
        "optimizer_state": to_cpu_tree(objects["optimizer"].state_dict()),
        "grad_scaler_state": to_cpu_tree(objects["scaler"].state_dict()),
        "scheduler_state": scheduler.state_dict(),
        "batchnorm_policy": copy.deepcopy(BATCHNORM_POLICY),
        "batchnorm_reference_state": to_cpu_tree(bn_reference),
        "rng_state": capture_rng_state(),
        "dataloader_generator_state": generator.get_state().clone(),
        "current_epoch": current_epoch,
        "completed_batch_position": completed_batch_position,
        "global_optimizer_step": START_GLOBAL_STEP + resumed,
        "resumed_optimizer_steps": resumed,
        "next_cursor": cursor,
        "phase": phase,
        "completed_epochs": list(completed_epochs),
        "validation_history": copy.deepcopy(list(validation_history)),
        "best_state": copy.deepcopy(dict(best_state)),
        "early_stopping_state": early.state_dict(),
        "current_epoch_statistics": copy.deepcopy(
            dict(current_epoch_statistics)
        ),
        "checkpoint_creation_reason": reason,
        "stop_reason": stop_reason,
        "epoch_zero_retrained": False,
        "invalid_or_skipped_updates": 0,
    }
    validate_multiepoch_checkpoint(checkpoint)
    return checkpoint


def save_checkpoint(checkpoint: Mapping[str, Any], path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    atomic_torch_save(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    validate_multiepoch_checkpoint(loaded)
    if (
        loaded["phase"],
        loaded["global_optimizer_step"],
        loaded["next_cursor"],
    ) != (
        checkpoint["phase"],
        checkpoint["global_optimizer_step"],
        checkpoint["next_cursor"],
    ):
        raise RuntimeError("checkpoint readback cursor changed")
    result = {
        "path": relative(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "phase": checkpoint["phase"],
        "global_optimizer_step": checkpoint["global_optimizer_step"],
        "resumed_optimizer_steps": checkpoint["resumed_optimizer_steps"],
        "current_epoch": checkpoint["current_epoch"],
        "next_cursor": checkpoint["next_cursor"],
        "reason": checkpoint["checkpoint_creation_reason"],
        "write_and_readback_seconds": time.perf_counter() - started,
        "atomic": True,
        "readable": True,
    }
    del loaded
    gc.collect()
    return result


def epoch_artifact_paths(epoch: int) -> dict[str, Path]:
    return {
        "checkpoint": OUTPUT_DIR / f"epoch_{epoch:03d}.pt",
        "losses": OUTPUT_DIR / f"logical_losses_epoch_{epoch:03d}.json",
        "embeddings": OUTPUT_DIR
        / f"validation_embeddings_epoch_{epoch:03d}.pt",
        "scores": OUTPUT_DIR / f"validation_scores_epoch_{epoch:03d}.pt",
        "metrics": OUTPUT_DIR / f"validation_metrics_epoch_{epoch:03d}.json",
    }


def _transaction_snapshot(
    objects: Mapping[str, Any],
    scheduler: ExactResumedCosineScheduler,
    *,
    optimizer_hook_count: Mapping[str, int],
    global_optimizer_step: int,
    batch_position: int,
    logical_loss_count: int,
    batch_identity: str,
) -> dict[str, Any]:
    parameters = list(objects["embedding_model"].parameters()) + list(
        objects["aam"].parameters()
    )
    return {
        "optimizer_steps": optimizer_parameter_steps(objects["optimizer"]),
        "optimizer_hook_count": int(optimizer_hook_count["value"]),
        "parameter_versions": [parameter._version for parameter in parameters],
        "scheduler_completed_updates": scheduler.scheduler_completed_updates,
        "global_optimizer_step": global_optimizer_step,
        "batch_position": batch_position,
        "logical_loss_count": logical_loss_count,
        "batch_identity": batch_identity,
        "learning_rates": list(scheduler.current_lrs),
    }


def _amp_attempt(
    objects: Mapping[str, Any],
    logical_batch: Mapping[str, Any],
    *,
    expected_step: int,
    optimizer_hook_count: dict[str, int],
    batchnorm_reference: Mapping[str, torch.Tensor],
    stage: dict[str, str],
) -> dict[str, Any]:
    mean_var_norm = objects["mean_var_norm"]
    embedding_model = objects["embedding_model"]
    aam = objects["aam"]
    optimizer = objects["optimizer"]
    scaler = objects["scaler"]
    device = next(embedding_model.parameters()).device
    apply_batchnorm_policy(embedding_model)
    assert_module_parameters_finite(
        [embedding_model, aam], "pre-attempt trainable"
    )
    assert_finite_tensor_tree(optimizer.state_dict(), "pre-attempt optimizer")
    optimizer.zero_grad(set_to_none=True)
    logical_loss = 0.0
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for number, microbatch in enumerate(
        iter_microbatches(logical_batch, PHYSICAL_MICROBATCH_SIZE), start=1
    ):
        stage["value"] = (
            f"optimizer step {expected_step}: microbatch {number}/"
            f"{ACCUMULATION_STEPS}"
        )
        features = microbatch["fbank"].to(device=device, dtype=torch.float32)
        if not bool(torch.isfinite(features).all().item()):
            raise RuntimeError("cached training Fbank is non-finite")
        labels = microbatch["speaker_label"].to(device=device, dtype=torch.long)
        lengths = torch.ones(
            PHYSICAL_MICROBATCH_SIZE, device=device, dtype=torch.float32
        )
        with torch.no_grad():
            normalized = mean_var_norm(features, lengths)
        if (
            tuple(normalized.shape)
            != (PHYSICAL_MICROBATCH_SIZE, *FEATURE_SHAPE)
            or not bool(torch.isfinite(normalized).all().item())
        ):
            raise RuntimeError("normalized training Fbank is invalid")
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            raw = embedding_model(normalized, lengths)
        if (
            tuple(raw.shape)
            != (PHYSICAL_MICROBATCH_SIZE, 1, EMBEDDING_DIM)
            or not bool(torch.isfinite(raw).all().item())
        ):
            raise RuntimeError("ECAPA training embedding is invalid")
        embedding = raw.squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            logits = aam(embedding.float(), labels)
            loss = (
                F.cross_entropy(logits.float(), labels, reduction="sum")
                / LOGICAL_BATCH_SIZE
            )
        if (
            tuple(logits.shape) != (PHYSICAL_MICROBATCH_SIZE, NUM_CLASSES)
            or logits.dtype != torch.float32
            or loss.dtype != torch.float32
            or not bool(torch.isfinite(logits).all().item())
            or not bool(torch.isfinite(loss).item())
        ):
            raise RuntimeError("AAM logits/loss are invalid")
        logical_loss += float(loss.detach().item())
        scaler.scale(loss).backward()
        del features, labels, lengths, normalized, raw, embedding, logits, loss
    named_parameters = list(embedding_model.named_parameters()) + [
        (f"aam.{name}", parameter) for name, parameter in aam.named_parameters()
    ]
    scaled_diagnostics = gradient_diagnostics(named_parameters)
    scaler.unscale_(optimizer)
    unscaled_diagnostics = gradient_diagnostics(named_parameters)
    scale_before = float(scaler.get_scale())
    if not unscaled_diagnostics["all_finite"]:
        assert_batchnorm_reference_exact(embedding_model, batchnorm_reference)
        return {
            "recoverable": classify_recoverable_overflow(
                cached_features_finite=True,
                normalized_features_finite=True,
                embeddings_finite=True,
                logits_finite=True,
                unscaled_loss_finite=math.isfinite(logical_loss),
                model_parameters_finite=True,
                optimizer_state_finite=True,
                post_unscale_diagnostics=unscaled_diagnostics,
                optimizer_step_called=False,
                counters_advanced=False,
            ),
            "logical_loss": logical_loss,
            "scale_before": scale_before,
            "scaled_gradient_diagnostics": scaled_diagnostics,
            "unscaled_gradient_diagnostics": unscaled_diagnostics,
            "optimizer_step_called": False,
            "duration_seconds": time.perf_counter() - started,
        }
    ecapa_gradient_norm = finite_gradient_norm(
        list(embedding_model.named_parameters()),
        "ECAPA",
        require_positive=False,
    )
    aam_gradient_norm = finite_gradient_norm(
        list(aam.named_parameters()), "AAM", require_positive=False
    )
    hook_before = optimizer_hook_count["value"]
    scaler.step(optimizer)
    if optimizer_hook_count["value"] != hook_before + 1:
        raise RuntimeError("GradScaler did not execute exactly one optimizer step")
    scaler.update()
    if optimizer_parameter_steps(optimizer) != (expected_step, expected_step):
        raise RuntimeError("AdamW state counters did not advance exactly")
    assert_batchnorm_reference_exact(embedding_model, batchnorm_reference)
    assert_module_parameters_finite(
        [embedding_model, aam], "post-update trainable"
    )
    assert_finite_tensor_tree(optimizer.state_dict(), "post-update optimizer")
    torch.cuda.synchronize(device)
    if not math.isfinite(logical_loss) or logical_loss <= 0.0:
        raise RuntimeError("logical loss is invalid")
    return {
        "recoverable": False,
        "logical_loss": logical_loss,
        "ecapa_gradient_norm": ecapa_gradient_norm,
        "aam_gradient_norm": aam_gradient_norm,
        "grad_scaler_scale": float(scaler.get_scale()),
        "scale_before": scale_before,
        "scaled_gradient_diagnostics": scaled_diagnostics,
        "unscaled_gradient_diagnostics": unscaled_diagnostics,
        "optimizer_step_called": True,
        "duration_seconds": time.perf_counter() - started,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run_optimizer_update_with_amp_recovery(
    objects: Mapping[str, Any],
    logical_batch: Mapping[str, Any],
    *,
    planned_indexes: Sequence[int],
    scheduler: ExactResumedCosineScheduler,
    expected_step: int,
    optimizer_hook_count: dict[str, int],
    batchnorm_reference: Mapping[str, torch.Tensor],
    stage: dict[str, str],
    generator: torch.Generator,
    epoch: int,
    batch_position: int,
    logical_loss_count: int,
    overflow_state: Mapping[str, Any],
) -> dict[str, Any]:
    identity = logical_batch_identity(logical_batch, planned_indexes)
    rng_before = capture_rng_state()
    generator_before = generator.get_state().clone()
    transaction_before = _transaction_snapshot(
        objects,
        scheduler,
        optimizer_hook_count=optimizer_hook_count,
        global_optimizer_step=expected_step - 1,
        batch_position=batch_position,
        logical_loss_count=logical_loss_count,
        batch_identity=identity,
    )
    first = _amp_attempt(
        objects,
        logical_batch,
        expected_step=expected_step,
        optimizer_hook_count=optimizer_hook_count,
        batchnorm_reference=batchnorm_reference,
        stage=stage,
    )
    if first["optimizer_step_called"]:
        first["overflow_event"] = None
        first["batch_identity"] = identity
        return first
    if not first["recoverable"]:
        raise RuntimeError("non-finite gradient attempt is not recoverable")
    transaction_after = _transaction_snapshot(
        objects,
        scheduler,
        optimizer_hook_count=optimizer_hook_count,
        global_optimizer_step=expected_step - 1,
        batch_position=batch_position,
        logical_loss_count=logical_loss_count,
        batch_identity=identity,
    )
    validate_failed_attempt_invariants(transaction_before, transaction_after)
    assert_batchnorm_reference_exact(
        objects["embedding_model"], batchnorm_reference
    )
    original_scale = first["scale_before"]
    reduced_scale = retry_scale(original_scale)
    objects["optimizer"].zero_grad(set_to_none=True)
    objects["scaler"].update(new_scale=reduced_scale)
    restore_rng_state(rng_before)
    generator.set_state(generator_before)
    if (
        not values_exactly_equal(capture_rng_state(), rng_before)
        or not torch.equal(generator.get_state(), generator_before)
    ):
        raise RuntimeError("pre-attempt RNG state did not restore exactly")
    retry = _amp_attempt(
        objects,
        logical_batch,
        expected_step=expected_step,
        optimizer_hook_count=optimizer_hook_count,
        batchnorm_reference=batchnorm_reference,
        stage=stage,
    )
    event = {
        "epoch": epoch,
        "logical_batch_position": batch_position,
        "attempted_global_optimizer_step": expected_step,
        "batch_identity": identity,
        "original_scale": original_scale,
        "retry_scale": reduced_scale,
        "failed_attempt": {
            "scaled_gradient_diagnostics": first[
                "scaled_gradient_diagnostics"
            ],
            "unscaled_gradient_diagnostics": first[
                "unscaled_gradient_diagnostics"
            ],
            "optimizer_step_called": False,
            "transaction_invariants_exact": True,
            "rng_restored_exact": True,
        },
        "retry": {
            "optimizer_step_called": bool(retry["optimizer_step_called"]),
            "unscaled_gradient_diagnostics": retry[
                "unscaled_gradient_diagnostics"
            ],
        },
    }
    if not retry["optimizer_step_called"]:
        event["result"] = "FAILED"
        failure = {
            "schema_name": "viespeaker2_ecapa_aam_multiepoch_recovery_failure",
            "schema_version": 1,
            "phase": "recovery_failed",
            "error": "single approved half-scale retry also overflowed",
            "overflow_event": event,
            "overflow_counters": {
                "overflow_attempt_count": int(
                    overflow_state["overflow_attempt_count"]
                )
                + 1,
                "recovered_overflow_count": int(
                    overflow_state["recovered_overflow_count"]
                ),
                "failed_overflow_count": int(
                    overflow_state["failed_overflow_count"]
                )
                + 1,
                "successful_optimizer_updates": int(
                    overflow_state["successful_optimizer_updates"]
                ),
            },
            "preserved_last_checkpoint": relative(LAST_PATH),
            "final_test_accessed": False,
        }
        atomic_json(RECOVERY_FAILURE_PATH, failure)
        raise RuntimeError("single approved half-scale retry also overflowed")
    event["result"] = "RECOVERED"
    retry["overflow_event"] = event
    retry["batch_identity"] = identity
    retry["duration_seconds"] += first["duration_seconds"]
    return retry


def train_epoch(
    *,
    objects: Mapping[str, Any],
    scheduler: ExactResumedCosineScheduler,
    early: EarlyStoppingState,
    generator: torch.Generator,
    bn_reference: Mapping[str, torch.Tensor],
    dataset: CachedFbankDataset,
    plan: Sequence[Sequence[int]],
    plan_audit: Mapping[str, Any],
    epoch: int,
    start_position: int,
    prior_statistics: Mapping[str, Any],
    completed_epochs: Sequence[int],
    validation_history: Sequence[Mapping[str, Any]],
    best_state: Mapping[str, Any],
    device: torch.device,
    checkpoint_results: list[dict[str, Any]],
    overflow_state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_epoch_plan(dataset, plan, epoch=epoch)
    if plan_sha256(epoch, plan) != EPOCH_PLAN_HASHES[epoch]:
        raise RuntimeError("epoch plan changed immediately before optimizer work")
    losses = list(prior_statistics.get("losses", []))
    ecapa_gradients = list(prior_statistics.get("ecapa_gradient_norms", []))
    aam_gradients = list(prior_statistics.get("aam_gradient_norms", []))
    durations = list(prior_statistics.get("step_durations_seconds", []))
    multipliers = list(prior_statistics.get("lr_multipliers", []))
    ecapa_lrs = list(prior_statistics.get("ecapa_lrs", []))
    aam_lrs = list(prior_statistics.get("aam_lrs", []))
    collections = (
        losses,
        ecapa_gradients,
        aam_gradients,
        durations,
        multipliers,
        ecapa_lrs,
        aam_lrs,
    )
    if any(len(values) != start_position for values in collections):
        raise RuntimeError("rolling statistics disagree with the resume position")
    loader_generator = torch.Generator()
    loader_generator.set_state(generator.get_state())
    loader = create_cached_fbank_training_dataloader(
        dataset,
        plan[start_position:],  # type: ignore[arg-type]
        num_workers=0,
        generator=loader_generator,
    )
    counter, hook = register_optimizer_hook(objects["optimizer"])
    stage = {"value": f"epoch {epoch} training"}
    torch.cuda.reset_peak_memory_stats(device)
    epoch_started = time.perf_counter()
    for position, batch in enumerate(loader, start=start_position):
        logical_batch = prepare_logical_batch(batch, plan[position])
        update_index, multiplier, lrs = scheduler.apply_for_next_update()
        expected_update_index = (epoch - 1) * STEPS_PER_EPOCH + position
        if update_index != expected_update_index:
            raise RuntimeError("scheduler position disagrees with epoch cursor")
        expected_global_step = START_GLOBAL_STEP + update_index + 1
        update = run_optimizer_update_with_amp_recovery(
            objects,
            logical_batch,
            planned_indexes=plan[position],
            scheduler=scheduler,
            expected_step=expected_global_step,
            optimizer_hook_count=counter,
            batchnorm_reference=bn_reference,
            stage=stage,
            generator=generator,
            epoch=epoch,
            batch_position=position + 1,
            logical_loss_count=len(losses),
            overflow_state=overflow_state,
        )
        scheduler.mark_successful_update(update_index)
        if scheduler.scheduler_completed_updates != update_index + 1:
            raise RuntimeError("scheduler did not advance exactly once")
        losses.append(update["logical_loss"])
        ecapa_gradients.append(update["ecapa_gradient_norm"])
        aam_gradients.append(update["aam_gradient_norm"])
        durations.append(update["duration_seconds"])
        multipliers.append(multiplier)
        ecapa_lrs.append(lrs[0])
        aam_lrs.append(lrs[1])
        completed_position = position + 1
        overflow_state["successful_optimizer_updates"] += 1
        overflow_state["recovery_session_successful_updates"] += 1
        overflow_state["last_successful_batch_identity"] = update[
            "batch_identity"
        ]
        overflow_event = update["overflow_event"]
        if overflow_event is not None:
            overflow_state["overflow_attempt_count"] += 1
            overflow_state["recovered_overflow_count"] += 1
            overflow_state["recovered_events"].append(overflow_event)
        historical = overflow_state["historical_failure"]
        if epoch == 2 and completed_position == 2532:
            historical["replay_batch_identity"] = update["batch_identity"]
            historical["replay_result"] = (
                "reproduced_recovered"
                if overflow_event is not None
                else "not_reproduced"
            )
        current = {
            "epoch": epoch,
            "losses": losses,
            "ecapa_gradient_norms": ecapa_gradients,
            "aam_gradient_norms": aam_gradients,
            "step_durations_seconds": durations,
            "lr_multipliers": multipliers,
            "ecapa_lrs": ecapa_lrs,
            "aam_lrs": aam_lrs,
        }
        if scheduler.scheduler_completed_updates % ROLLING_INTERVAL == 0:
            rolling = build_checkpoint(
                objects,
                scheduler,
                early,
                generator,
                bn_reference,
                current_epoch=epoch,
                completed_batch_position=completed_position,
                phase=(
                    "training"
                    if completed_position < STEPS_PER_EPOCH
                    else "validation_pending"
                ),
                completed_epochs=completed_epochs,
                validation_history=validation_history,
                best_state=best_state,
                current_epoch_statistics=current,
                overflow_state=overflow_state,
                reason="rolling_1000_resumed_updates",
                stop_reason=None,
            )
            checkpoint_results.append(save_checkpoint(rolling, LAST_PATH))
            del rolling
        if overflow_event is not None:
            recovered = build_checkpoint(
                objects,
                scheduler,
                early,
                generator,
                bn_reference,
                current_epoch=epoch,
                completed_batch_position=completed_position,
                phase=(
                    "training"
                    if completed_position < STEPS_PER_EPOCH
                    else "validation_pending"
                ),
                completed_epochs=completed_epochs,
                validation_history=validation_history,
                best_state=best_state,
                current_epoch_statistics=current,
                overflow_state=overflow_state,
                reason="recovered_amp_overflow_atomic",
                stop_reason=None,
            )
            checkpoint_results.append(save_checkpoint(recovered, LAST_PATH))
            del recovered
            print(
                "AMP_OVERFLOW_RECOVERED "
                f"epoch={epoch} batch={completed_position} "
                f"global_step={expected_global_step} "
                f"scale={overflow_event['original_scale']}->"
                f"{overflow_event['retry_scale']}",
                flush=True,
            )
        if (
            scheduler.scheduler_completed_updates % LOGGING_INTERVAL == 0
            or completed_position == STEPS_PER_EPOCH
        ):
            record = {
                "epoch": epoch,
                "global_optimizer_step": expected_global_step,
                "resumed_optimizer_steps": scheduler.scheduler_completed_updates,
                "logical_batch_position": completed_position,
                "loss": update["logical_loss"],
                "average_last_50_loss": statistics.fmean(losses[-50:]),
                "scheduler_update_index": update_index,
                "lr_multiplier": multiplier,
                "ecapa_lr": lrs[0],
                "aam_lr": lrs[1],
                "ecapa_gradient_norm": update["ecapa_gradient_norm"],
                "aam_gradient_norm": update["aam_gradient_norm"],
                "grad_scaler_scale": update["grad_scaler_scale"],
                "cuda_reserved_bytes": update["cuda_reserved_bytes"],
            }
            append_jsonl(TRAIN_LOG_PATH, record)
            print(
                "TRAIN "
                f"epoch={epoch} batch={completed_position}/{STEPS_PER_EPOCH} "
                f"global_step={expected_global_step} resumed_step="
                f"{scheduler.scheduler_completed_updates} loss="
                f"{update['logical_loss']:.8f} multiplier={multiplier:.12g} "
                f"ecapa_lr={lrs[0]:.12g} aam_lr={lrs[1]:.12g}",
                flush=True,
            )
        del batch, logical_batch, update
    hook.remove()
    if (
        len(losses) != STEPS_PER_EPOCH
        or counter["value"] != STEPS_PER_EPOCH - start_position
    ):
        raise RuntimeError("epoch did not complete exactly 2,969 updates")
    summary = {
        "epoch": epoch,
        "optimizer_steps": STEPS_PER_EPOCH,
        "optimizer_updates": STEPS_PER_EPOCH,
        "resumed_from_batch_position": start_position,
        "new_optimizer_updates": STEPS_PER_EPOCH - start_position,
        "global_optimizer_step": START_GLOBAL_STEP
        + scheduler.scheduler_completed_updates,
        "resumed_optimizer_steps": scheduler.scheduler_completed_updates,
        "loss": summarize_values(losses),
        "ecapa_gradient_norm": summarize_values(ecapa_gradients),
        "aam_gradient_norm": summarize_values(aam_gradients),
        "step_duration_seconds": summarize_values(durations),
        "lr_multiplier": summarize_values(multipliers),
        "ecapa_lr": summarize_values(ecapa_lrs),
        "aam_lr": summarize_values(aam_lrs),
        "duration_seconds": time.perf_counter() - epoch_started,
        "cuda_peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "cuda_peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "sampler_audit": dict(plan_audit),
        "batchnorm_buffers_bit_exact": True,
        "invalid_or_skipped_updates": 0,
    }
    current = {
        "epoch": epoch,
        "losses": losses,
        "ecapa_gradient_norms": ecapa_gradients,
        "aam_gradient_norms": aam_gradients,
        "step_durations_seconds": durations,
        "lr_multipliers": multipliers,
        "ecapa_lrs": ecapa_lrs,
        "aam_lrs": aam_lrs,
        "summary": summary,
    }
    loss_path = epoch_artifact_paths(epoch)["losses"]
    atomic_json(
        loss_path,
        {
            "schema_name": "viespeaker2_multiepoch_logical_losses",
            "schema_version": 2,
            "epoch": epoch,
            "count": len(losses),
            "global_steps": list(
                range(
                    START_GLOBAL_STEP + (epoch - 1) * STEPS_PER_EPOCH + 1,
                    START_GLOBAL_STEP + epoch * STEPS_PER_EPOCH + 1,
                )
            ),
            "logical_losses": losses,
            "sha256_binding": EPOCH_PLAN_HASHES[epoch],
        },
    )
    gc.collect()
    return summary, current


def recompute_metrics_from_scores(score_path: Path) -> dict[str, Any]:
    stored = torch.load(score_path, map_location="cpu", weights_only=False)
    scores = stored["scores"]
    targets = stored["targets"]
    if (
        tuple(scores.shape) != (20000,)
        or tuple(targets.shape) != (20000,)
        or not bool(torch.isfinite(scores).all().item())
    ):
        raise RuntimeError("selected validation score artifact is invalid")
    eer = calculate_eer(scores.tolist(), targets.tolist())
    confusion = empirical_confusion(
        scores.tolist(), targets.tolist(), eer.empirical_threshold
    )
    return {
        "interpolated_eer": eer.interpolated_eer,
        "interpolated_threshold": eer.interpolated_threshold,
        "empirical_threshold": eer.empirical_threshold,
        "empirical_far": eer.empirical_far,
        "empirical_frr": eer.empirical_frr,
        "empirical_average_error": eer.empirical_average_error,
        "empirical_confusion": confusion,
    }


def fresh_checkpoint_audit(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    validate_multiepoch_checkpoint(checkpoint)
    before_steps = optimizer_parameter_steps_from_state(
        checkpoint["optimizer_state"]
    )
    objects, scheduler, early, generator = load_state_into_objects(
        checkpoint, device
    )
    after_steps = optimizer_parameter_steps(objects["optimizer"])
    if before_steps != after_steps:
        raise RuntimeError("fresh checkpoint audit changed optimizer counters")
    assert_batchnorm_reference_exact(
        objects["embedding_model"], checkpoint["batchnorm_reference_state"]
    )
    result = {
        "path": relative(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "phase": checkpoint["phase"],
        "current_epoch": checkpoint["current_epoch"],
        "global_optimizer_step": checkpoint["global_optimizer_step"],
        "resumed_optimizer_steps": checkpoint["resumed_optimizer_steps"],
        "scheduler_completed_updates": scheduler.scheduler_completed_updates,
        "best_epoch": early.best_epoch,
        "best_eer": early.best_eer,
        "rng_exact": values_exactly_equal(
            capture_rng_state(), checkpoint["rng_state"]
        ),
        "generator_exact": torch.equal(
            generator.get_state(), checkpoint["dataloader_generator_state"]
        ),
        "optimizer_steps_before_after": [list(before_steps), list(after_steps)],
        "optimizer_steps_during_audit": 0,
        "batchnorm_buffers_bit_exact": True,
    }
    del objects, scheduler, early, generator, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def optimizer_parameter_steps_from_state(
    state: Mapping[str, Any],
) -> tuple[int, int]:
    values = {
        int(item["step"].item() if isinstance(item["step"], torch.Tensor) else item["step"])
        for item in state["state"].values()
    }
    if len(values) != 1:
        raise RuntimeError("optimizer state has inconsistent counters")
    step = next(iter(values))
    return step, step


def markdown_report(report: Mapping[str, Any]) -> str:
    rows = []
    for record in report["validation_history"]:
        training = record["training"]
        metrics = record["metrics"]
        rows.append(
            f"| {record['epoch']} | {training['optimizer_steps']} | "
            f"{training['global_optimizer_step']} | "
            f"{metrics['interpolated_eer']:.10f} | "
            f"{metrics['empirical_threshold']:.12f} | "
            f"{'yes' if record['best'] else 'no'} | "
            f"{record['bad_epoch_count']} |"
        )
    return f"""# VieSpeaker2.0 resumed multi-epoch ECAPA-AAM

Result: **PASS**. Epoch zero was not retrained. Training stopped after epoch
{report['stop_epoch']} with reason `{report['stop_reason']}`.

## Epoch history

| epoch | updates | global step | validation EER | empirical threshold | selected best | bad epochs |
|---:|---:|---:|---:|---:|:---:|---:|
{chr(10).join(rows)}

## Selection

- Best epoch: {report['best']['epoch']}
- Best checkpoint: `{report['best']['checkpoint_path']}`
- Best checkpoint SHA-256: `{report['best']['checkpoint_sha256']}`
- Validation EER: {report['best']['interpolated_eer']}
- Locked validation empirical threshold: {report['best']['empirical_threshold']}

## Scheduler

`{SCHEDULER_CONFIGURATION['formula']}` was applied before each successful
resumed optimizer update, with no warmup. Stored completed updates:
{report['scheduler']['scheduler_completed_updates']}.

## Preservation

All approved input identities remained unchanged. No source audio or cache
artifact was modified. The final-test manifest and all final-test/excluded
content remained quarantined and were not opened, statted, hashed, parsed, or
loaded. No commit or push occurred.
"""


def recovery_markdown(report: Mapping[str, Any]) -> str:
    replay = report["historical_failure_replay"]
    counters = report["overflow_counters"]
    event_lines = [
        "- Epoch {epoch}, batch {logical_batch_position}, global step "
        "{attempted_global_optimizer_step}: scale {original_scale} -> "
        "{retry_scale}; affected `{affected}`; result `{result}`.".format(
            **event,
            affected=", ".join(
                event["failed_attempt"]["unscaled_gradient_diagnostics"][
                    "affected_parameter_names"
                ]
            ),
        )
        for event in report["recovered_events"]
    ]
    if not event_lines:
        event_lines = ["- No recoverable AMP overflow occurred during replay."]
    return f"""# VieSpeaker2.0 AMP overflow recovery

Result: **PASS**.

The exact atomic checkpoint `{report['recovery_checkpoint']['path']}` with
SHA-256 `{report['recovery_checkpoint']['sha256']}` was preserved and resumed
from epoch 2, batch position 2,031.

## Historical replay

The historical failed batch result was `{replay['replay_result']}`. Its replay
batch identity was `{replay['replay_batch_identity']}`.

## Overflow events

{chr(10).join(event_lines)}

Attempts: {counters['overflow_attempt_count']}; recovered:
{counters['recovered_overflow_count']}; unrecovered:
{counters['failed_overflow_count']}. Failed attempts advanced no training
counters and no logical batch was skipped.

The historical failed-run reports, runtime, failure record, and epoch-one
checkpoint remained byte-identical. Final-test content remained quarantined.
"""


def execute(preflight_state: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    task_started = time.perf_counter()
    start = preflight_state["start_checkpoint"]
    start_report = preflight_state["start_report"]
    dataset = preflight_state["dataset"]
    plans = preflight_state["plans"]
    plan_audits = preflight_state["plan_audits"]
    checkpoint_results: list[dict[str, Any]] = []
    validation_runtime: list[dict[str, Any]] = []
    reconciliation = reconcile_recovery_artifacts(
        preflight_state["existing_last"],
        restart=preflight_state["recovery_restart"],
    )
    removed_scalar_records = reconciliation["scalar_log"]["removed_count"]

    seed_everything(SEED)
    checkpoint = preflight_state["existing_last"]
    if checkpoint["schema_version"] == SCHEMA_VERSION:
        checkpoint = migrate_checkpoint_for_amp_recovery(checkpoint)
        checkpoint_results.append(save_checkpoint(checkpoint, LAST_PATH))
    (
        objects,
        scheduler,
        early,
        generator,
    ) = load_state_into_objects(checkpoint, device)
    bn_reference = to_cpu_tree(checkpoint["batchnorm_reference_state"])
    completed_epochs = list(checkpoint["completed_epochs"])
    validation_history = copy.deepcopy(checkpoint["validation_history"])
    best_state = copy.deepcopy(checkpoint["best_state"])
    current_statistics = copy.deepcopy(checkpoint["current_epoch_statistics"])
    overflow_state = copy.deepcopy(checkpoint["overflow_state"])
    phase = checkpoint["phase"]
    current_epoch = checkpoint["current_epoch"]
    completed_position = checkpoint["completed_batch_position"]

    while phase != "stopped":
        if phase == "epoch_complete":
            epoch = current_epoch + 1
            if epoch > LAST_RESUMED_EPOCH:
                raise RuntimeError("epoch-complete state exceeded the approved maximum")
            current_epoch = epoch
            completed_position = 0
            current_statistics = {}
            phase = "training"
        if phase == "training":
            epoch = current_epoch
            start_position = completed_position
            summary, current_statistics = train_epoch(
                objects=objects,
                scheduler=scheduler,
                early=early,
                generator=generator,
                bn_reference=bn_reference,
                dataset=dataset,
                plan=plans[epoch],
                plan_audit=plan_audits[epoch],
                epoch=epoch,
                start_position=start_position,
                prior_statistics=current_statistics,
                completed_epochs=completed_epochs,
                validation_history=validation_history,
                best_state=best_state,
                device=device,
                checkpoint_results=checkpoint_results,
                overflow_state=overflow_state,
            )
            completed_position = STEPS_PER_EPOCH
            phase = "validation_pending"
            pending = build_checkpoint(
                objects,
                scheduler,
                early,
                generator,
                bn_reference,
                current_epoch=epoch,
                completed_batch_position=STEPS_PER_EPOCH,
                phase=phase,
                completed_epochs=completed_epochs,
                validation_history=validation_history,
                best_state=best_state,
                current_epoch_statistics=current_statistics,
                overflow_state=overflow_state,
                reason="epoch_training_complete_validation_pending",
                stop_reason=None,
            )
            pending_result = save_checkpoint(pending, LAST_PATH)
            checkpoint_results.append(pending_result)
        elif phase == "validation_pending":
            epoch = current_epoch
            if len(current_statistics.get("losses", [])) != STEPS_PER_EPOCH:
                raise RuntimeError("validation-pending state lacks complete training statistics")
            summary = current_statistics["summary"]
            pending_result = {
                "path": relative(LAST_PATH),
                "sha256": file_sha256(LAST_PATH),
            }
        else:
            raise RuntimeError(f"unsupported recovery phase: {phase}")

        paths = epoch_artifact_paths(epoch)
        validation = run_validation(
            objects,
            device=device,
            checkpoint_sha256=pending_result["sha256"],
            bn_reference=bn_reference,
            epoch=epoch,
            checkpoint_path=LAST_PATH,
            embedding_path=paths["embeddings"],
            score_path=paths["scores"],
            metrics_path=paths["metrics"],
        )
        metrics = validation["metrics"]
        empirical_average = float(metrics["empirical_average_error"])
        improved = early.observe(
            epoch,
            interpolated_eer=float(metrics["interpolated_eer"]),
            empirical_average_error=empirical_average,
            empirical_threshold=float(metrics["empirical_threshold"]),
        )
        if improved:
            best_state = {
                "epoch": epoch,
                "interpolated_eer": float(metrics["interpolated_eer"]),
                "empirical_average_error": empirical_average,
                "empirical_threshold": float(metrics["empirical_threshold"]),
                "checkpoint_path": relative(paths["checkpoint"]),
                "validation_source_checkpoint_path": relative(LAST_PATH),
                "validation_source_checkpoint_sha256": pending_result["sha256"],
            }
        completed_epochs.append(epoch)
        validation_record = {
            "epoch": epoch,
            "training": summary,
            "metrics": metrics,
            "runtime": validation["runtime"],
            "artifacts": validation["artifacts"],
            "best": improved,
            "bad_epoch_count": early.bad_epoch_count,
        }
        validation_history.append(validation_record)
        validation_runtime.append(validation["runtime"])
        stop_reason = stop_reason_after_validation(epoch, early)
        phase = "stopped" if stop_reason else "epoch_complete"
        current_statistics = {}
        completed = build_checkpoint(
            objects,
            scheduler,
            early,
            generator,
            bn_reference,
            current_epoch=epoch,
            completed_batch_position=STEPS_PER_EPOCH,
            phase=phase,
            completed_epochs=completed_epochs,
            validation_history=validation_history,
            best_state=best_state,
            current_epoch_statistics={},
            overflow_state=overflow_state,
            reason=(
                "final_stop_after_validation"
                if stop_reason
                else "epoch_validation_complete"
            ),
            stop_reason=stop_reason,
        )
        epoch_result = save_checkpoint(completed, paths["checkpoint"])
        checkpoint_results.append(epoch_result)
        if improved:
            atomic_copy(paths["checkpoint"], BEST_PATH)
            if file_sha256(BEST_PATH) != epoch_result["sha256"]:
                raise RuntimeError("best checkpoint copy differs from selected epoch")
            selected = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
            validate_multiepoch_checkpoint(selected)
            checkpoint_results.append(
                {
                    "path": relative(BEST_PATH),
                    "sha256": file_sha256(BEST_PATH),
                    "size_bytes": BEST_PATH.stat().st_size,
                    "phase": selected["phase"],
                    "global_optimizer_step": selected["global_optimizer_step"],
                    "reason": "best_validation_exact_epoch_copy",
                    "atomic": True,
                    "readable": True,
                }
            )
        last_result = save_checkpoint(completed, LAST_PATH)
        checkpoint_results.append(last_result)
        print(
            "EPOCH_RESULT "
            f"epoch={epoch} eer={metrics['interpolated_eer']:.12f} "
            f"threshold={metrics['empirical_threshold']:.12f} "
            f"best_epoch={early.best_epoch} bad_epochs={early.bad_epoch_count} "
            f"phase={phase}",
            flush=True,
        )
        current_epoch = epoch
        completed_position = STEPS_PER_EPOCH

    if scheduler.scheduler_completed_updates != (
        len(completed_epochs) - 1
    ) * STEPS_PER_EPOCH:
        raise RuntimeError("completed epoch history disagrees with scheduler updates")
    final_last = torch.load(LAST_PATH, map_location="cpu", weights_only=False)
    final_best = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
    validate_multiepoch_checkpoint(final_last)
    validate_multiepoch_checkpoint(final_best)
    if final_last["phase"] != "stopped":
        raise RuntimeError("final last.pt is not stopped")
    best_epoch = final_last["best_state"]["epoch"]
    if best_epoch > 0:
        selected_epoch_path = epoch_artifact_paths(best_epoch)["checkpoint"]
        if file_sha256(BEST_PATH) != file_sha256(selected_epoch_path):
            raise RuntimeError("best.pt is not the exact selected epoch checkpoint")
        selected_score_path = epoch_artifact_paths(best_epoch)["scores"]
    else:
        selected_score_path = (
            REPO_ROOT
            / "outputs/ecapa_aam_one_epoch_v2/validation_scores_epoch_000.pt"
        )
    recomputed = recompute_metrics_from_scores(selected_score_path)
    selected_history = next(
        record for record in validation_history if record["epoch"] == best_epoch
    )
    selected_metrics = selected_history["metrics"]
    for key in (
        "interpolated_eer",
        "empirical_threshold",
        "empirical_far",
        "empirical_frr",
        "empirical_average_error",
    ):
        if recomputed[key] != selected_metrics[key]:
            raise RuntimeError(f"selected best metric recomputation failed: {key}")
    if recomputed["empirical_confusion"] != selected_metrics["empirical_confusion"]:
        raise RuntimeError("selected empirical threshold did not reproduce confusion")
    last_audit = fresh_checkpoint_audit(LAST_PATH, device)
    best_audit = fresh_checkpoint_audit(BEST_PATH, device)
    protected_after = protected_hashes()
    if protected_after != preflight_state["protected_before"]:
        raise RuntimeError("protected input changed during multiepoch training")
    if any(
        record["training"]["optimizer_steps"] != STEPS_PER_EPOCH
        for record in validation_history[1:]
    ):
        raise RuntimeError("a resumed epoch has the wrong update count")
    if [record["epoch"] for record in validation_history] != completed_epochs:
        raise RuntimeError("validation did not run exactly once per completed epoch")

    stop_reason = final_last["stop_reason"]
    stop_epoch = final_last["current_epoch"]
    task_duration = time.perf_counter() - task_started
    peak_allocated = max(
        [
            record["training"].get("cuda_peak_allocated_bytes", 0)
            for record in validation_history[1:]
        ]
        + [
            record["runtime"].get("cuda_peak_allocated_bytes", 0)
            for record in validation_history[1:]
        ]
    )
    peak_reserved = max(
        [
            record["training"].get("cuda_peak_reserved_bytes", 0)
            for record in validation_history[1:]
        ]
        + [
            record["runtime"].get("cuda_peak_reserved_bytes", 0)
            for record in validation_history[1:]
        ]
    )
    runtime = {
        "schema_name": RUNTIME_SCHEMA_NAME,
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "result": "PASS",
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
        "migration": {
            "source_path": START_CHECKPOINT_PATH,
            "source_sha256": START_CHECKPOINT_SHA256,
            "source_epoch": 0,
            "source_global_optimizer_step": START_GLOBAL_STEP,
            "source_scheduler_state": False,
            "initialized_scheduler_completed_updates": 0,
            "epoch_zero_retrained": False,
            "recovery_checkpoint_path": relative(RECOVERY_COPY_PATH),
            "recovery_checkpoint_sha256": RECOVERY_START_CHECKPOINT_SHA256,
            "recovery_cursor": reconciliation["approved_cursor"],
        },
        "epoch_plan_hashes": {
            str(key): value for key, value in EPOCH_PLAN_HASHES.items()
        },
        "validation_history": validation_history,
        "completed_resumed_epochs": completed_epochs[1:],
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "scheduler": final_last["scheduler_state"],
        "early_stopping": final_last["early_stopping_state"],
        "best": {
            **final_last["best_state"],
            "checkpoint_path": relative(BEST_PATH),
            "checkpoint_sha256": file_sha256(BEST_PATH),
        },
        "final_cursor": final_last["next_cursor"],
        "final_global_optimizer_step": final_last["global_optimizer_step"],
        "final_resumed_optimizer_steps": final_last["resumed_optimizer_steps"],
        "checkpoint_results": checkpoint_results,
        "fresh_object_audit": {"last": last_audit, "best": best_audit},
        "selected_metric_recomputation": recomputed,
        "protected_hashes_before": preflight_state["protected_before"],
        "protected_hashes_after": protected_after,
        "protected_inputs_unchanged": True,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "invalid_or_skipped_updates": 0,
        "overflow_policy": copy.deepcopy(AMP_OVERFLOW_POLICY),
        "overflow_state": copy.deepcopy(final_last["overflow_state"]),
        "batchnorm_buffers_bit_exact": True,
        "oom": False,
        "epoch_zero_retrained": False,
        "epoch_after_stop_started": False,
        "final_test_accessed": False,
        "commit_or_push_performed": False,
        "interruption_recovery": {
            "occurred": True,
            "failure": preflight_state["recovered_failure"],
            "reconciliation": reconciliation,
            "resumed_from_phase": preflight_state["existing_last"]["phase"],
            "resumed_from_global_optimizer_step": preflight_state[
                "existing_last"
            ]["global_optimizer_step"],
            "discarded_uncheckpointed_scalar_records": removed_scalar_records,
            "hyperparameters_changed": False,
            "execution_policy_change_only": "one half-scale AMP gradient retry",
        },
    }
    atomic_json(RECOVERY_RUNTIME_PATH, runtime)
    report = {
        "schema_name": "viespeaker2_ecapa_aam_multiepoch_report",
        "schema_version": 2,
        "result": "PASS",
        "approved_starting_state": runtime["migration"],
        "approved_upstream_bindings": UPSTREAM_BINDINGS,
        "sampler_binding": SAMPLER_BINDING,
        "epoch_plan_hashes": runtime["epoch_plan_hashes"],
        "training_configuration": MULTIEPOCH_TRAINING_CONFIGURATION,
        "aam_configuration": AAM_CONFIGURATION,
        "optimizer_configuration": OPTIMIZER_CONFIGURATION,
        "mixed_precision_configuration": MIXED_PRECISION_CONFIGURATION,
        "scheduler_configuration": SCHEDULER_CONFIGURATION,
        "scheduler": runtime["scheduler"],
        "validation_history": validation_history,
        "completed_resumed_epochs": runtime["completed_resumed_epochs"],
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "early_stopping": runtime["early_stopping"],
        "best": runtime["best"],
        "final_cursor": runtime["final_cursor"],
        "final_global_optimizer_step": runtime["final_global_optimizer_step"],
        "checkpoint_hashes": {
            "last.pt": file_sha256(LAST_PATH),
            "best.pt": file_sha256(BEST_PATH),
            **{
                f"epoch_{epoch:03d}.pt": file_sha256(
                    epoch_artifact_paths(epoch)["checkpoint"]
                )
                for epoch in completed_epochs[1:]
            },
        },
        "artifact_hashes": {
            "runtime_recovery_v2.json": file_sha256(RECOVERY_RUNTIME_PATH),
            "reconciliation_recovery_v2.json": file_sha256(
                RECONCILIATION_PATH
            ),
            "training_scalars.jsonl": file_sha256(TRAIN_LOG_PATH),
            **{
                f"logical_losses_epoch_{epoch:03d}.json": file_sha256(
                    epoch_artifact_paths(epoch)["losses"]
                )
                for epoch in completed_epochs[1:]
            },
            **{
                f"validation_embeddings_epoch_{epoch:03d}.pt": file_sha256(
                    epoch_artifact_paths(epoch)["embeddings"]
                )
                for epoch in completed_epochs[1:]
            },
            **{
                f"validation_scores_epoch_{epoch:03d}.pt": file_sha256(
                    epoch_artifact_paths(epoch)["scores"]
                )
                for epoch in completed_epochs[1:]
            },
            **{
                f"validation_metrics_epoch_{epoch:03d}.json": file_sha256(
                    epoch_artifact_paths(epoch)["metrics"]
                )
                for epoch in completed_epochs[1:]
            },
        },
        "runtime_summary": {
            "task_duration_seconds": task_duration,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "fresh_object_audit": runtime["fresh_object_audit"],
        },
        "epoch_zero_comparison": {
            "eer": INITIAL_BEST_EER,
            "selected_best_delta": runtime["best"]["interpolated_eer"]
            - INITIAL_BEST_EER,
        },
        "pretrained_comparison": {
            "eer": 0.1257,
            "selected_best_delta": runtime["best"]["interpolated_eer"] - 0.1257,
        },
        "protected_inputs_unchanged": True,
        "final_test_accessed": False,
        "commit_or_push_performed": False,
        "interruption_recovery": runtime["interruption_recovery"],
        "overflow_policy": runtime["overflow_policy"],
        "overflow_state": runtime["overflow_state"],
    }
    recovery_report = {
        "schema_name": "viespeaker2_ecapa_aam_multiepoch_recovery_report",
        "schema_version": 1,
        "result": "PASS",
        "historical_failure_identity": copy.deepcopy(
            HISTORICAL_FAILURE_BINDINGS
        ),
        "recovery_checkpoint": {
            "path": relative(RECOVERY_COPY_PATH),
            "sha256": RECOVERY_START_CHECKPOINT_SHA256,
            "cursor": reconciliation["approved_cursor"],
        },
        "artifact_reconciliation": reconciliation,
        "overflow_policy": copy.deepcopy(AMP_OVERFLOW_POLICY),
        "historical_failure_replay": copy.deepcopy(
            final_last["overflow_state"]["historical_failure"]
        ),
        "overflow_counters": {
            key: final_last["overflow_state"][key]
            for key in (
                "overflow_attempt_count",
                "recovered_overflow_count",
                "failed_overflow_count",
                "successful_optimizer_updates",
                "recovery_session_successful_updates",
            )
        },
        "recovered_events": copy.deepcopy(
            final_last["overflow_state"]["recovered_events"]
        ),
        "counter_invariants": {
            "failed_attempts_advanced_zero": True,
            "recovered_events_advanced_once": True,
            "logical_batches_skipped": 0,
        },
        "final_test_accessed": False,
    }
    atomic_json(RECOVERY_REPORT_JSON, recovery_report)
    atomic_text(RECOVERY_REPORT_MD, recovery_markdown(recovery_report))
    atomic_json(FINAL_REPORT_JSON, report)
    atomic_text(FINAL_REPORT_MD, markdown_report(report))
    print(
        "TRAINING_STOP "
        f"reason={stop_reason} stop_epoch={stop_epoch} "
        f"best_epoch={runtime['best']['epoch']} "
        f"best_eer={runtime['best']['interpolated_eer']:.12f}",
        flush=True,
    )
    return runtime


def main() -> None:
    args = parse_args()
    preflight_state = preflight(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (
        RECOVERY_STDOUT_PATH.open("a", encoding="utf-8", buffering=1) as stdout,
        RECOVERY_STDERR_PATH.open("a", encoding="utf-8", buffering=1) as stderr,
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        try:
            execute(preflight_state, torch.device(args.device))
        except BaseException as error:
            failure = {
                "schema_name": "viespeaker2_ecapa_aam_multiepoch_recovery_failure",
                "schema_version": 1,
                "error_type": type(error).__name__,
                "error": str(error),
                "cuda_oom": "out of memory" in str(error).lower(),
                "final_test_accessed": False,
            }
            if not RECOVERY_FAILURE_PATH.exists():
                atomic_json(RECOVERY_FAILURE_PATH, failure)
            print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
            raise


if __name__ == "__main__":
    main()
