"""Fail-closed helpers for the fixed one-epoch ECAPA/AAM pilot.

This module is intentionally dependency-light and contains no dataset discovery,
waveform frontend, training loop, or multi-epoch orchestration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from src.aam_training import (
    PRETRAINED_MODEL_ID,
    capture_rng_state,
    file_sha256,
    values_exactly_equal,
)


PILOT_SCHEMA_NAME = "speaker_verification_ecapa_aam_one_epoch_pilot"
PILOT_SCHEMA_VERSION = 1
PILOT_RUNTIME_SCHEMA_NAME = "ecapa_aam_one_epoch_pilot_runtime"
PILOT_RUNTIME_SCHEMA_VERSION = 1
EXPECTED_TRIAL_SHA256 = (
    "3badacbe16aa82537b618efe1d494241680fcf309c5bf7cc8103a25c172e7f6f"
)
SEED = 20260727
EPOCH = 0
NUM_LOGICAL_BATCHES = 1000
LOGICAL_BATCH_SIZE = 32
TOTAL_SAMPLE_SELECTIONS = 32000
SPEAKERS_PER_BATCH = 16
SAMPLES_PER_SPEAKER = 2
ACTIVE_SHARD_WINDOW = 8
PHYSICAL_MICROBATCH_SIZE = 4
ACCUMULATION_STEPS = 8
ROLLING_CHECKPOINT_STEPS = (250, 500, 750, 1000)
LOGGING_INTERVAL = 50
QUALITY_TOLERANCE = 1e-12
BASELINE_INTERPOLATED_EER = 0.11665301106104056
BASELINE_EMPIRICAL_AVERAGE_ERROR = 0.11665301106104056

SAMPLER_CONFIGURATION = {
    "name": "HybridShardAwareSpeakerBatchSampler",
    "seed": SEED,
    "epoch": EPOCH,
    "speakers_per_batch": SPEAKERS_PER_BATCH,
    "samples_per_speaker": SAMPLES_PER_SPEAKER,
    "active_shard_window": ACTIVE_SHARD_WINDOW,
    "logical_batch_size": LOGICAL_BATCH_SIZE,
    "num_logical_batches": NUM_LOGICAL_BATCHES,
    "sample_selections": TOTAL_SAMPLE_SELECTIONS,
}
AAM_CONFIGURATION = {
    "embedding_dim": 192,
    "num_classes": 488,
    "weight_shape": [488, 192],
    "bias": False,
    "margin_radians": 0.2,
    "scale": 30.0,
    "math_dtype": "float32",
}
OPTIMIZER_CONFIGURATION = [
    {
        "name": "embedding_model",
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
    },
    {
        "name": "aam_classifier",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    },
]
BATCHNORM_POLICY = {
    "embedding_model_training": True,
    "batchnorm_modules_eval": True,
    "batchnorm_affine_trainable": True,
    "running_buffers_bit_exactly_frozen": True,
}
AMP_POLICY = {
    "enabled": True,
    "ecapa_autocast_dtype": "float16",
    "aam_math_dtype": "float32",
    "loss_dtype": "float32",
    "grad_scaler_initial_scale": 128.0,
}

APPROVED_INPUT_PATHS = {
    "outputs/fbank_cache_v1/fbank_cache_config_v1.json",
    "outputs/fbank_cache_v1/train_feature_index_v1.csv",
    "outputs/fbank_cache_v1/validation_feature_index_v1.csv",
    "manifests/portable/validation_manifest_v1.csv",
    "manifests/verification/validation_trials_v1.csv",
    "manifests/verification/validation_trials_config_v1.json",
}


def _finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a Python number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{description} must be finite")
    return numeric


def validate_scalar_tree(value: Any, description: str = "payload") -> None:
    """Require JSON-compatible Python scalars and finite numeric values."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{description} contains a non-string key")
            validate_scalar_tree(item, f"{description}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_scalar_tree(item, f"{description}[{index}]")
        return
    raise TypeError(
        f"{description} contains non-serializable {type(value).__name__}"
    )


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    validate_scalar_tree(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    os.replace(temporary, target)


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    validate_scalar_tree(record, "JSONL record")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


TRAINING_LOG_KEYS = {
    "schema_name",
    "schema_version",
    "epoch",
    "global_optimizer_step",
    "logical_batch_position",
    "current_logical_loss",
    "mean_logical_loss_latest_window",
    "ecapa_learning_rate",
    "aam_learning_rate",
    "grad_scaler_scale",
    "ecapa_gradient_norm",
    "aam_gradient_norm",
    "step_duration_seconds",
    "rolling_mean_step_duration_seconds",
    "logical_samples_per_second",
    "cuda_memory_allocated_bytes",
    "cuda_memory_reserved_bytes",
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
    "rolling_checkpoint_path",
}


def validate_training_log_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != TRAINING_LOG_KEYS:
        raise ValueError("training log record has incorrect fields")
    if (
        record["schema_name"] != "ecapa_aam_pilot_training_log"
        or record["schema_version"] != 1
        or record["epoch"] != EPOCH
    ):
        raise ValueError("training log schema/epoch is invalid")
    step = record["global_optimizer_step"]
    if (
        type(step) is not int
        or not LOGGING_INTERVAL <= step <= NUM_LOGICAL_BATCHES
        or step % LOGGING_INTERVAL != 0
        or record["logical_batch_position"] != step - 1
    ):
        raise ValueError("training log step/cadence is invalid")
    finite_nonnegative = (
        "current_logical_loss",
        "mean_logical_loss_latest_window",
    )
    for key in finite_nonnegative:
        if _finite_number(record[key], key) < 0.0:
            raise ValueError(f"training log {key} must be non-negative")
    positive_numeric = (
        "grad_scaler_scale",
        "ecapa_gradient_norm",
        "aam_gradient_norm",
        "step_duration_seconds",
        "rolling_mean_step_duration_seconds",
        "logical_samples_per_second",
    )
    for key in positive_numeric:
        if _finite_number(record[key], key) <= 0.0:
            raise ValueError(f"training log {key} must be positive")
    if record["ecapa_learning_rate"] != 1e-5 or record["aam_learning_rate"] != 1e-3:
        raise ValueError("training log optimizer learning rates are invalid")
    for key in (
        "cuda_memory_allocated_bytes",
        "cuda_memory_reserved_bytes",
        "cuda_max_memory_allocated_bytes",
        "cuda_max_memory_reserved_bytes",
    ):
        if type(record[key]) is not int or record[key] < 0:
            raise ValueError(f"training log {key} must be a non-negative integer")
    checkpoint = record["rolling_checkpoint_path"]
    if step in ROLLING_CHECKPOINT_STEPS:
        if (
            not isinstance(checkpoint, str)
            or not checkpoint.endswith("/last.pt")
            or Path(checkpoint).is_absolute()
        ):
            raise ValueError("rolling checkpoint log path is invalid")
    elif checkpoint is not None:
        raise ValueError("non-boundary log record contains a checkpoint path")
    validate_scalar_tree(record, "training log record")


@dataclass
class OneEpochController:
    """A hard stop that can represent only epoch zero and 1,000 updates."""

    epochs_started: list[int] = field(default_factory=list)
    epochs_completed: list[int] = field(default_factory=list)
    successful_updates: int = 0
    finished: bool = False

    def start(self, epoch: int) -> None:
        if self.finished or self.epochs_started:
            raise RuntimeError("the one-epoch pilot cannot start another epoch")
        if epoch != EPOCH:
            raise ValueError("the one-epoch pilot may start only epoch 0")
        self.epochs_started.append(epoch)

    def record_update(self, logical_batch_position: int) -> None:
        if self.epochs_started != [EPOCH] or self.finished:
            raise RuntimeError("epoch 0 is not active")
        if logical_batch_position != self.successful_updates:
            raise ValueError("logical batch positions must be contiguous from zero")
        if self.successful_updates >= NUM_LOGICAL_BATCHES:
            raise RuntimeError("the one-epoch update limit has already been reached")
        self.successful_updates += 1

    def finish(self) -> dict[str, Any]:
        if self.finished:
            raise RuntimeError("epoch 0 was already finalized")
        if self.epochs_started != [EPOCH]:
            raise RuntimeError("epoch 0 was not started exactly once")
        if self.successful_updates != NUM_LOGICAL_BATCHES:
            raise RuntimeError(
                f"epoch 0 has {self.successful_updates} updates, "
                f"expected {NUM_LOGICAL_BATCHES}"
            )
        self.epochs_completed.append(EPOCH)
        self.finished = True
        return {
            "epochs_started": [EPOCH],
            "epochs_completed": [EPOCH],
            "optimizer_updates": NUM_LOGICAL_BATCHES,
            "logical_sample_selections": TOTAL_SAMPLE_SELECTIONS,
            "epoch_1_started": False,
            "next_epoch": 1,
            "next_logical_batch_position": 0,
        }


def resume_cursor(global_optimizer_step: int) -> tuple[int, int]:
    if not isinstance(global_optimizer_step, int):
        raise TypeError("global optimizer step must be an integer")
    if not 0 <= global_optimizer_step <= NUM_LOGICAL_BATCHES:
        raise ValueError("global optimizer step is outside epoch 0")
    if global_optimizer_step == NUM_LOGICAL_BATCHES:
        return 1, 0
    return EPOCH, global_optimizer_step


def rolling_checkpoint_due(global_optimizer_step: int) -> bool:
    return global_optimizer_step in ROLLING_CHECKPOINT_STEPS


@dataclass
class LoggingWindow:
    interval: int = LOGGING_INTERVAL
    losses: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)

    def add(self, loss: float, duration: float) -> None:
        loss_value = _finite_number(loss, "logical loss")
        duration_value = _finite_number(duration, "step duration")
        if duration_value <= 0.0:
            raise ValueError("step duration must be positive")
        if len(self.losses) >= self.interval:
            raise RuntimeError("logging window must be emitted before adding more data")
        self.losses.append(loss_value)
        self.durations.append(duration_value)

    def emit(self, global_optimizer_step: int) -> dict[str, float | int]:
        if (
            global_optimizer_step % self.interval != 0
            or len(self.losses) != self.interval
            or len(self.durations) != self.interval
        ):
            raise RuntimeError("logging window is not complete at an emission step")
        result = {
            "window_size": self.interval,
            "window_start_step": global_optimizer_step - self.interval + 1,
            "window_end_step": global_optimizer_step,
            "mean_logical_loss": sum(self.losses) / self.interval,
            "mean_step_duration_seconds": sum(self.durations) / self.interval,
        }
        self.losses.clear()
        self.durations.clear()
        return result


def summarize_finite_values(values: Sequence[float], description: str) -> dict[str, Any]:
    if not values:
        raise ValueError(f"{description} is empty")
    numeric = [_finite_number(value, description) for value in values]
    return {
        "count": len(numeric),
        "first": numeric[0],
        "final": numeric[-1],
        "mean": sum(numeric) / len(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def checkpoint_selection_key(candidate: Mapping[str, Any]) -> tuple[float, float, int]:
    return (
        _finite_number(candidate.get("interpolated_eer"), "interpolated EER"),
        _finite_number(
            candidate.get("empirical_average_error"), "empirical average error"
        ),
        int(candidate.get("order")),
    )


def select_best_checkpoint(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("at least one trained checkpoint is required")
    names = [candidate.get("name") for candidate in candidates]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("checkpoint candidates require non-empty names")
    if len(names) != len(set(names)):
        raise ValueError("checkpoint candidate names must be unique")
    return min(candidates, key=checkpoint_selection_key)


def classify_model_quality(
    pilot_interpolated_eer: float,
    baseline_interpolated_eer: float = BASELINE_INTERPOLATED_EER,
    tolerance: float = QUALITY_TOLERANCE,
) -> str:
    pilot = _finite_number(pilot_interpolated_eer, "pilot interpolated EER")
    baseline = _finite_number(
        baseline_interpolated_eer, "baseline interpolated EER"
    )
    tolerance = _finite_number(tolerance, "numerical tolerance")
    if tolerance < 0.0:
        raise ValueError("numerical tolerance must be non-negative")
    difference = pilot - baseline
    if difference < -tolerance:
        return "IMPROVED"
    if difference > tolerance:
        return "DEGRADED"
    return "UNCHANGED_WITHIN_NUMERICAL_TOLERANCE"


def baseline_comparison(
    pilot_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    tolerance: float = QUALITY_TOLERANCE,
) -> dict[str, Any]:
    pilot = _finite_number(
        pilot_metrics.get("interpolated_eer"), "pilot interpolated EER"
    )
    baseline = _finite_number(
        baseline_metrics.get("interpolated_eer"), "baseline interpolated EER"
    )
    if baseline <= 0.0:
        raise ValueError("baseline interpolated EER must be positive")
    signed = pilot - baseline
    result = {
        "pilot_interpolated_eer": pilot,
        "baseline_interpolated_eer": baseline,
        "signed_eer_difference": signed,
        "absolute_eer_difference": abs(signed),
        "percentage_point_difference": signed * 100.0,
        "relative_eer_change": signed / baseline,
        "numerical_tolerance": tolerance,
        "model_quality_result": classify_model_quality(
            pilot, baseline, tolerance
        ),
    }
    validate_scalar_tree(result)
    return result


def validate_relative_artifact_path(value: Any, description: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{description} must be a string")
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or Path(value).is_absolute()
        or ".." in pure.parts
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"unsafe {description}: {value!r}")
    lowered = [part.lower() for part in pure.parts]
    if any(part in {"test", "final_test", "final-test", "part"} for part in lowered):
        raise ValueError(f"{description} references a quarantined final-test path")
    filename = lowered[-1]
    if filename.startswith(("test_", "final_test_", "part_")):
        raise ValueError(f"{description} references a quarantined final-test file")
    return value


def validate_approved_input_path(value: str) -> str:
    value = validate_relative_artifact_path(value, "input path")
    if value in APPROVED_INPUT_PATHS:
        return value
    pure = PurePosixPath(value)
    if (
        len(pure.parts) == 4
        and pure.parts[:2] == ("outputs", "fbank_cache_v1")
        and pure.parts[2] in {"train", "validation"}
        and pure.parts[3].startswith("shard_")
        and pure.parts[3].endswith(".pt")
        and len(pure.parts[3]) == len("shard_00000.pt")
        and pure.parts[3][6:11].isdigit()
    ):
        return value
    raise ValueError(f"input path is outside the pilot allowlist: {value!r}")


def hash_explicit_paths(
    project_root: str | Path, relative_paths: Sequence[str]
) -> dict[str, str]:
    root = Path(project_root).resolve()
    if not relative_paths or len(relative_paths) != len(set(relative_paths)):
        raise ValueError("protected paths must be a non-empty unique sequence")
    result: dict[str, str] = {}
    for relative in sorted(relative_paths):
        approved = validate_approved_input_path(relative)
        path = (root / Path(*PurePosixPath(approved).parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("protected path escapes the project root") from error
        result[approved] = file_sha256(path)
    return result


def hash_mapping_digest(hashes: Mapping[str, str]) -> str:
    if not hashes:
        raise ValueError("hash mapping is empty")
    for path, digest in hashes.items():
        validate_approved_input_path(path)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("hash mapping contains a malformed SHA-256")
    canonical = json.dumps(dict(sorted(hashes.items())), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require_trial_hash(path: str | Path) -> str:
    digest = file_sha256(path)
    if digest != EXPECTED_TRIAL_SHA256:
        raise ValueError(
            f"immutable validation trial hash mismatch: {digest}"
        )
    return digest


CHECKPOINT_KEYS = {
    "schema_name",
    "schema_version",
    "pretrained_model_identifier",
    "embedding_model_state_dict",
    "mean_var_norm_state_dict",
    "aam_classifier_state_dict",
    "optimizer_state_dict",
    "grad_scaler_state_dict",
    "epoch",
    "next_epoch",
    "next_logical_batch_position",
    "global_optimizer_step",
    "sampler_configuration",
    "physical_microbatch_size",
    "accumulation_steps",
    "aam_configuration",
    "optimizer_configuration",
    "batchnorm_policy",
    "amp_policy",
    "train_cache_identity",
    "rng_state",
    "completed_training_loss_summary",
    "checkpoint_creation_reason",
    "validation_selection",
}


def _validate_sha256(value: Any, description: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} is not a lowercase SHA-256")


def validate_pilot_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != CHECKPOINT_KEYS:
        raise ValueError("pilot checkpoint has incorrect top-level keys")
    if (
        checkpoint["schema_name"] != PILOT_SCHEMA_NAME
        or checkpoint["schema_version"] != PILOT_SCHEMA_VERSION
        or checkpoint["pretrained_model_identifier"] != PRETRAINED_MODEL_ID
    ):
        raise ValueError("pilot checkpoint schema/model identifier is invalid")
    for key in (
        "embedding_model_state_dict",
        "mean_var_norm_state_dict",
        "aam_classifier_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
    ):
        if not isinstance(checkpoint[key], Mapping):
            raise ValueError(f"{key} must be a mapping")
    aam_state = checkpoint["aam_classifier_state_dict"]
    weight = aam_state.get("weight")
    if not isinstance(weight, Tensor) or tuple(weight.shape) != (488, 192):
        raise ValueError("AAM checkpoint weight must have shape [488, 192]")
    step = checkpoint["global_optimizer_step"]
    if not isinstance(step, int) or not 1 <= step <= NUM_LOGICAL_BATCHES:
        raise ValueError("checkpoint global step is outside epoch 0")
    optimizer_state = checkpoint["optimizer_state_dict"]
    parameter_states = optimizer_state.get("state")
    if not isinstance(parameter_states, Mapping) or not parameter_states:
        raise ValueError("checkpoint AdamW state must be non-empty")
    for parameter_state in parameter_states.values():
        if not isinstance(parameter_state, Mapping) or "step" not in parameter_state:
            raise ValueError("checkpoint AdamW parameter state is malformed")
        state_step = parameter_state["step"]
        if isinstance(state_step, Tensor):
            if state_step.numel() != 1:
                raise ValueError("checkpoint AdamW step tensor is malformed")
            state_step = int(state_step.item())
        elif isinstance(state_step, (int, float)):
            state_step = int(state_step)
        else:
            raise ValueError("checkpoint AdamW step value is malformed")
        if state_step != step:
            raise ValueError("checkpoint AdamW step does not match global step")
    if checkpoint["epoch"] != EPOCH:
        raise ValueError("checkpoint completed/training epoch must be 0")
    expected_next_epoch, expected_position = resume_cursor(step)
    if (
        checkpoint["next_epoch"] != expected_next_epoch
        or checkpoint["next_logical_batch_position"] != expected_position
    ):
        raise ValueError("checkpoint resume cursor is invalid")
    reason = checkpoint["checkpoint_creation_reason"]
    if reason not in {"rolling", "epoch_complete", "best_validation"}:
        raise ValueError("invalid checkpoint creation reason")
    if reason == "rolling" and step not in ROLLING_CHECKPOINT_STEPS:
        raise ValueError("rolling checkpoint was created at an invalid step")
    if reason in {"epoch_complete", "best_validation"} and step != NUM_LOGICAL_BATCHES:
        raise ValueError("completed/best checkpoint must represent step 1,000")
    if checkpoint["sampler_configuration"] != SAMPLER_CONFIGURATION:
        raise ValueError("checkpoint sampler configuration is invalid")
    if (
        checkpoint["physical_microbatch_size"] != PHYSICAL_MICROBATCH_SIZE
        or checkpoint["accumulation_steps"] != ACCUMULATION_STEPS
    ):
        raise ValueError("checkpoint physical/logical batch configuration is invalid")
    if checkpoint["aam_configuration"] != AAM_CONFIGURATION:
        raise ValueError("checkpoint AAM configuration is invalid")
    if checkpoint["optimizer_configuration"] != OPTIMIZER_CONFIGURATION:
        raise ValueError("checkpoint optimizer configuration is invalid")
    if checkpoint["batchnorm_policy"] != BATCHNORM_POLICY:
        raise ValueError("checkpoint BatchNorm policy is invalid")
    if checkpoint["amp_policy"] != AMP_POLICY:
        raise ValueError("checkpoint AMP policy is invalid")
    identity = checkpoint["train_cache_identity"]
    if not isinstance(identity, Mapping) or set(identity) != {"files", "set_sha256"}:
        raise ValueError("checkpoint train-cache identity is malformed")
    files = identity["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint train-cache identity is empty")
    seen: set[str] = set()
    mapping: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError("checkpoint train-cache file identity is malformed")
        path = validate_approved_input_path(item["path"])
        if "validation" in PurePosixPath(path).parts:
            raise ValueError("checkpoint train-cache identity contains validation data")
        if path in seen:
            raise ValueError("checkpoint train-cache identity contains duplicate paths")
        seen.add(path)
        _validate_sha256(item["sha256"], "train-cache file hash")
        mapping[path] = item["sha256"]
    _validate_sha256(identity["set_sha256"], "train-cache set hash")
    if hash_mapping_digest(mapping) != identity["set_sha256"]:
        raise ValueError("checkpoint train-cache set hash is invalid")
    rng = checkpoint["rng_state"]
    if not isinstance(rng, Mapping) or set(rng) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("checkpoint RNG state is malformed")
    if not isinstance(rng["torch_cpu"], Tensor) or rng["torch_cpu"].dtype != torch.uint8:
        raise ValueError("checkpoint CPU RNG state is malformed")
    if not isinstance(rng["torch_cuda"], list) or any(
        not isinstance(value, Tensor) or value.dtype != torch.uint8
        for value in rng["torch_cuda"]
    ):
        raise ValueError("checkpoint CUDA RNG state is malformed")
    summary = checkpoint["completed_training_loss_summary"]
    required_summary = {"count", "first", "final", "mean", "minimum", "maximum"}
    if not isinstance(summary, Mapping) or set(summary) != required_summary:
        raise ValueError("checkpoint loss summary is malformed")
    if summary["count"] != step:
        raise ValueError("checkpoint loss count does not match global step")
    for key in required_summary - {"count"}:
        _finite_number(summary[key], f"checkpoint loss {key}")
    if not (
        summary["minimum"]
        <= summary["first"]
        <= summary["maximum"]
        and summary["minimum"]
        <= summary["final"]
        <= summary["maximum"]
        and summary["minimum"]
        <= summary["mean"]
        <= summary["maximum"]
    ):
        raise ValueError("checkpoint loss summary relationships are invalid")
    selection = checkpoint["validation_selection"]
    if reason == "best_validation":
        if not isinstance(selection, Mapping):
            raise ValueError("best checkpoint requires validation selection metadata")
        if (
            selection.get("selected_checkpoint") != "epoch_000.pt"
            or selection.get("selection_scope")
            != "best among trained pilot checkpoints"
            or selection.get("baseline_is_comparison_reference_only") is not True
            or selection.get("model_quality_result")
            not in {
                "IMPROVED",
                "UNCHANGED_WITHIN_NUMERICAL_TOLERANCE",
                "DEGRADED",
            }
        ):
            raise ValueError("best checkpoint selection metadata is invalid")
        validate_validation_metrics_payload(selection.get("pilot_metrics"))
    elif selection is not None:
        raise ValueError("non-best checkpoint must not contain validation selection")


def atomic_save_pilot_checkpoint(
    checkpoint: Mapping[str, Any], path: str | Path
) -> None:
    validate_pilot_checkpoint(checkpoint)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(dict(checkpoint), temporary)
    os.replace(temporary, target)


def load_pilot_checkpoint(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.stat().st_size <= 0:
        raise ValueError("checkpoint file is missing or empty")
    checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    validate_pilot_checkpoint(checkpoint)
    return checkpoint


def checkpoint_state_roundtrip_exact(
    checkpoint: Mapping[str, Any],
    *,
    embedding_model: nn.Module,
    mean_var_norm: nn.Module,
    aam_classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_scaler: Any,
) -> dict[str, bool]:
    checks = {
        "embedding_model_state_exact": values_exactly_equal(
            embedding_model.state_dict(), checkpoint["embedding_model_state_dict"]
        ),
        "mean_var_norm_state_exact": values_exactly_equal(
            mean_var_norm.state_dict(), checkpoint["mean_var_norm_state_dict"]
        ),
        "aam_classifier_state_exact": values_exactly_equal(
            aam_classifier.state_dict(), checkpoint["aam_classifier_state_dict"]
        ),
        "optimizer_state_exact": values_exactly_equal(
            optimizer.state_dict(), checkpoint["optimizer_state_dict"]
        ),
        "grad_scaler_state_exact": values_exactly_equal(
            grad_scaler.state_dict(), checkpoint["grad_scaler_state_dict"]
        ),
        "counters_and_schema_valid": True,
    }
    validate_pilot_checkpoint(checkpoint)
    if not all(checks.values()):
        raise AssertionError(f"fresh-object checkpoint roundtrip failed: {checks}")
    return checks


def validate_validation_metrics_payload(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("validation metrics payload must be a mapping")
    required = {
        "interpolated_eer",
        "interpolated_eer_percentage",
        "interpolated_threshold",
        "interpolated_far",
        "interpolated_frr",
        "eer_kind",
        "eer_threshold_kind",
        "empirical_threshold",
        "empirical_far",
        "empirical_frr",
        "empirical_far_frr_gap",
        "empirical_average_error",
        "same_speaker_scores",
        "different_speaker_scores",
        "score_range",
        "score_count",
        "positive_trials",
        "negative_trials",
        "threshold_semantics",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"validation metrics are missing fields: {sorted(missing)}")
    eer = _finite_number(payload["interpolated_eer"], "interpolated EER")
    if not 0.0 <= eer <= 1.0:
        raise ValueError("interpolated EER is outside [0, 1]")
    if not math.isclose(
        _finite_number(
            payload["interpolated_eer_percentage"],
            "interpolated EER percentage",
        ),
        eer * 100.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("interpolated EER percentage is inconsistent")
    for key in (
        "interpolated_threshold",
        "interpolated_far",
        "interpolated_frr",
        "empirical_threshold",
        "empirical_far",
        "empirical_frr",
        "empirical_far_frr_gap",
        "empirical_average_error",
    ):
        _finite_number(payload[key], key)
    score_range = payload["score_range"]
    if (
        not isinstance(score_range, list)
        or len(score_range) != 2
        or any(not -1.000001 <= _finite_number(value, "score range") <= 1.000001 for value in score_range)
        or score_range[0] > score_range[1]
    ):
        raise ValueError("score range is invalid")
    if payload["threshold_semantics"] != "accept same speaker when score >= threshold":
        raise ValueError("threshold semantics are invalid")
    if (
        payload["eer_kind"] != "linearly_interpolated_roc_crossing"
        or payload["eer_threshold_kind"] != "interpolated_non_empirical"
    ):
        raise ValueError("interpolated EER/threshold designation is invalid")
    for key in (
        "interpolated_eer",
        "interpolated_far",
        "interpolated_frr",
        "empirical_far",
        "empirical_frr",
        "empirical_far_frr_gap",
        "empirical_average_error",
    ):
        if not 0.0 <= float(payload[key]) <= 1.0:
            raise ValueError(f"{key} is outside [0, 1]")
    if not math.isclose(
        float(payload["empirical_far_frr_gap"]),
        abs(float(payload["empirical_far"]) - float(payload["empirical_frr"])),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("empirical FAR/FRR gap is inconsistent")
    if not math.isclose(
        float(payload["empirical_average_error"]),
        (float(payload["empirical_far"]) + float(payload["empirical_frr"])) / 2.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("empirical average error is inconsistent")
    if (
        payload["score_count"] != 19528
        or payload["positive_trials"] != 9764
        or payload["negative_trials"] != 9764
    ):
        raise ValueError("validation score/trial counts are invalid")
    for key in ("same_speaker_scores", "different_speaker_scores"):
        summary = payload[key]
        if not isinstance(summary, Mapping) or summary.get("count") != 9764:
            raise ValueError(f"{key} summary is invalid")
        for field in ("minimum", "mean", "standard_deviation", "median", "maximum"):
            _finite_number(summary.get(field), f"{key}.{field}")
    validate_scalar_tree(payload, "validation metrics")


def validate_runtime_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_name",
        "schema_version",
        "technical_pass",
        "training_pass",
        "checkpoint_pass",
        "validation_pass",
        "scope_compliance_pass",
        "overall_pass",
        "model_quality_result",
        "training_summary",
        "checkpoint_summary",
        "validation_summary",
        "protected_hashes",
        "final_test_accessed",
        "recursive_listing_used",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"runtime payload is missing fields: {sorted(missing)}")
    if (
        payload["schema_name"] != PILOT_RUNTIME_SCHEMA_NAME
        or payload["schema_version"] != PILOT_RUNTIME_SCHEMA_VERSION
    ):
        raise ValueError("runtime payload schema is invalid")
    for key in (
        "technical_pass",
        "training_pass",
        "checkpoint_pass",
        "validation_pass",
        "scope_compliance_pass",
        "overall_pass",
        "final_test_accessed",
        "recursive_listing_used",
    ):
        if type(payload[key]) is not bool:
            raise ValueError(f"runtime {key} must be a bool")
    expected_technical = all(
        payload[key]
        for key in ("training_pass", "checkpoint_pass", "validation_pass")
    )
    if payload["technical_pass"] != expected_technical:
        raise ValueError("technical_pass is inconsistent")
    if payload["overall_pass"] != (
        payload["technical_pass"] and payload["scope_compliance_pass"]
    ):
        raise ValueError("overall_pass is inconsistent")
    if payload["final_test_accessed"] or payload["recursive_listing_used"]:
        raise ValueError("runtime payload records a scope violation")
    if payload["model_quality_result"] not in {
        "IMPROVED",
        "UNCHANGED_WITHIN_NUMERICAL_TOLERANCE",
        "DEGRADED",
    }:
        raise ValueError("runtime model-quality result is invalid")
    training = payload["training_summary"]
    if (
        not isinstance(training, Mapping)
        or training.get("optimizer_updates") != NUM_LOGICAL_BATCHES
        or training.get("logical_sample_selections") != TOTAL_SAMPLE_SELECTIONS
        or training.get("epoch_1_started") is not False
        or not isinstance(training.get("loss"), Mapping)
        or training["loss"].get("count") != NUM_LOGICAL_BATCHES
        or training.get("logging_record_count")
        != NUM_LOGICAL_BATCHES // LOGGING_INTERVAL
    ):
        raise ValueError("runtime training summary is incomplete or inconsistent")
    checkpoints = payload["checkpoint_summary"]
    if (
        not isinstance(checkpoints, Mapping)
        or not isinstance(checkpoints.get("rolling_checkpoints"), list)
        or [
            item.get("global_optimizer_step")
            for item in checkpoints["rolling_checkpoints"]
        ]
        != list(ROLLING_CHECKPOINT_STEPS)
        or checkpoints.get("epoch_checkpoint", {}).get("global_optimizer_step")
        != NUM_LOGICAL_BATCHES
        or checkpoints.get("epoch_roundtrip", {}).get(
            "optimizer_steps_taken_after_load"
        )
        != 0
        or checkpoints.get("best_checkpoint", {}).get(
            "state_matches_epoch_000"
        )
        is not True
    ):
        raise ValueError("runtime checkpoint summary is incomplete or inconsistent")
    validation = payload["validation_summary"]
    validation_runtime = (
        validation.get("runtime") if isinstance(validation, Mapping) else None
    )
    if (
        not isinstance(validation_runtime, Mapping)
        or validation_runtime.get("validation_rows") != 8504
        or validation_runtime.get("validation_speakers") != 100
        or validation_runtime.get("score_count") != 19528
        or validation_runtime.get("aam_classifier_calls") != 0
    ):
        raise ValueError("runtime validation summary is incomplete or inconsistent")
    protected = payload["protected_hashes"]
    if (
        not isinstance(protected, Mapping)
        or protected.get("unchanged") is not True
        or protected.get("mismatches") != {}
        or type(protected.get("files_compared")) is not int
        or protected["files_compared"] < 6
    ):
        raise ValueError("runtime protected-hash summary is invalid")
    if payload.get("validation_trial_hash") != EXPECTED_TRIAL_SHA256:
        raise ValueError("runtime validation trial hash is invalid")
    validate_scalar_tree(payload, "runtime payload")


def validation_forward_batch(
    mean_var_norm: nn.Module,
    embedding_model: nn.Module,
    features: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    """Run only the fixed validation encoder path; no classifier argument exists."""
    if mean_var_norm.training or embedding_model.training:
        raise ValueError("validation modules must be in eval mode")
    if (
        not isinstance(features, Tensor)
        or features.ndim != 3
        or tuple(features.shape[1:]) != (301, 80)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not bool(torch.isfinite(features).all().item())
    ):
        raise ValueError("validation Fbank must be finite CPU float32 [B, 301, 80]")
    batch_size = features.shape[0]
    lengths = torch.ones(batch_size, device=device, dtype=torch.float32)
    with torch.inference_mode():
        device_features = features.to(device=device, dtype=torch.float32)
        normalized = mean_var_norm(device_features, lengths)
        if (
            tuple(normalized.shape) != (batch_size, 301, 80)
            or not bool(torch.isfinite(normalized).all().item())
        ):
            raise ValueError("validation normalized features are invalid")
        raw = embedding_model(normalized, lengths)
        if (
            tuple(raw.shape) != (batch_size, 1, 192)
            or not bool(torch.isfinite(raw).all().item())
        ):
            raise ValueError("validation raw embeddings are invalid")
        embeddings = raw.squeeze(1).to(device="cpu", dtype=torch.float32)
    if (
        tuple(embeddings.shape) != (batch_size, 192)
        or not bool(torch.isfinite(embeddings).all().item())
    ):
        raise ValueError("validation scoring embeddings are invalid")
    return embeddings


def default_rng_state() -> dict[str, Any]:
    """Small indirection used by checkpoint construction and synthetic tests."""
    return capture_rng_state()
