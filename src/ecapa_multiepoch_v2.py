"""Fail-closed policy helpers for resumed VieSpeaker2.0 ECAPA/AAM training."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

import torch

from src.ecapa_one_epoch_v2 import (
    AAM_CONFIGURATION,
    BATCHNORM_POLICY,
    FEATURE_SHAPE,
    MIXED_PRECISION_CONFIGURATION,
    NUM_CLASSES,
    NUM_LOGICAL_BATCHES,
    OPTIMIZER_CONFIGURATION,
    SAMPLER_BINDING,
    TRAINING_CONFIGURATION,
    UPSTREAM_BINDINGS,
    validate_checkpoint_v2,
)


SCHEMA_NAME = "viespeaker2_ecapa_aam_multiepoch"
SCHEMA_VERSION = 2
RECOVERY_SCHEMA_VERSION = 3
RUNTIME_SCHEMA_NAME = "viespeaker2_ecapa_aam_multiepoch_runtime"
RUNTIME_SCHEMA_VERSION = 2
START_CHECKPOINT_PATH = "outputs/ecapa_aam_one_epoch_v2/best.pt"
START_CHECKPOINT_SHA256 = (
    "19cfd3482172ce482d21b65915fd347aeeac4ee1e51bc61e59484f46a31a38db"
)
START_REPORT_PATH = "reports/ecapa_aam_one_epoch_v2.json"
START_REPORT_SHA256 = (
    "ffea2e9b706404a75de40ce5fb68579d2890b351e38243a79d74c425f261afb3"
)
START_GLOBAL_STEP = 2969
FIRST_RESUMED_EPOCH = 1
LAST_RESUMED_EPOCH = 4
STEPS_PER_EPOCH = NUM_LOGICAL_BATCHES
TOTAL_RESUMED_UPDATES = 4 * STEPS_PER_EPOCH
FINAL_UPDATE_INDEX = TOTAL_RESUMED_UPDATES - 1
BASE_LRS = (1.0e-5, 1.0e-3)
MINIMUM_MULTIPLIER = 0.1
MINIMUM_LRS = tuple(value * MINIMUM_MULTIPLIER for value in BASE_LRS)
SCHEDULER_FORMULA = "0.1 + 0.9 * 0.5 * (1 + cos(pi * r / 11875))"
SCHEDULER_VERSION = 2
INITIAL_BEST_EPOCH = 0
INITIAL_BEST_EER = 0.064
INITIAL_EMPIRICAL_AVERAGE_ERROR = 0.064
INITIAL_EMPIRICAL_THRESHOLD = 0.16190975904464722
IMPROVEMENT_TOLERANCE = 1.0e-12
EARLY_STOPPING_PATIENCE = 2
PHASES = {"training", "validation_pending", "epoch_complete", "stopped"}
STOP_REASONS = {"early_stopping", "max_epoch"}
EPOCH_PLAN_HASHES = {
    1: "c5ff9c7fca4c88541c47738706b1852293e6727da9bb420350e75379179bba0f",
    2: "a6ecc945b67ec425efd6939045587f3f0313f1f1b934c89ebcba3a6a16d6705d",
    3: "1d3319abf5965382664993880b2d3e7e75c9441aa3be65cbf71359279dbc3988",
    4: "583e0a1bfb4800e5a44028d05aa6278ab1e9dc12dac095ba85492643e524c33a",
}
RECOVERY_START_CHECKPOINT_PATH = "outputs/ecapa_aam_multiepoch_v2/last.pt"
RECOVERY_START_CHECKPOINT_SHA256 = (
    "3b8d613012ee8340a30ad264729c83ea97501b793e6c97e1d0ed78555e5c96a0"
)
HISTORICAL_FAILURE_BINDINGS = {
    "runtime": {
        "path": "outputs/ecapa_aam_multiepoch_v2/runtime_v2.json",
        "sha256": "b99b68b7d2c5885a1c4a8d9228786d12a3ca5c602e7760394e45d084bd10adcb",
    },
    "failure_record": {
        "path": "outputs/ecapa_aam_multiepoch_v2/failure_v2.json",
        "sha256": "fe8d940fed333624b5475f4a1442e82c2dbcd9da8f7f2d7bacb50fddaacf47ae",
    },
    "json_report": {
        "path": "reports/ecapa_aam_multiepoch_v2.json",
        "sha256": "a6a1ab26830c0f9fcd31b97347d57259a09d9e1e1099817bb661efb4b11008bc",
    },
    "markdown_report": {
        "path": "reports/ecapa_aam_multiepoch_v2.md",
        "sha256": "1b76649637b837cc42663dcdfeb7f5f79cc60c16d3763f91865b740455e7de9e",
    },
    "epoch_001": {
        "path": "outputs/ecapa_aam_multiepoch_v2/epoch_001.pt",
        "sha256": "a7820ec84eec75acaca595219a81603d7d9056382024d004617613150271711b",
    },
}
AMP_OVERFLOW_POLICY = {
    "schema_name": "viespeaker2_amp_gradient_overflow_recovery",
    "schema_version": 1,
    "maximum_retries_per_logical_batch": 1,
    "scale_reduction_factor": 0.5,
    "minimum_retry_scale": 1.0,
    "retry_same_materialized_logical_batch": True,
    "restore_pre_attempt_rng": True,
    "advance_counters_on_failed_attempt": False,
}

MULTIEPOCH_TRAINING_CONFIGURATION = {
    **TRAINING_CONFIGURATION,
    "first_resumed_epoch": FIRST_RESUMED_EPOCH,
    "last_resumed_epoch": LAST_RESUMED_EPOCH,
    "planned_resumed_updates": TOTAL_RESUMED_UPDATES,
    "augmentation": False,
    "scheduler": "explicit_per_update_cosine_v2",
    "warmup": False,
    "gradient_clipping": False,
    "early_stopping": {
        "monitor": "validation interpolated EER",
        "patience": EARLY_STOPPING_PATIENCE,
        "improvement_tolerance": IMPROVEMENT_TOLERANCE,
    },
}

SCHEDULER_CONFIGURATION = {
    "name": "explicit_resumed_cosine",
    "version": SCHEDULER_VERSION,
    "formula": SCHEDULER_FORMULA,
    "index_semantics": "apply multiplier for zero-based resumed update r before optimizer update",
    "total_planned_resumed_updates": TOTAL_RESUMED_UPDATES,
    "first_update_index": 0,
    "final_update_index": FINAL_UPDATE_INDEX,
    "minimum_multiplier": MINIMUM_MULTIPLIER,
    "base_learning_rates": list(BASE_LRS),
    "minimum_learning_rates": list(MINIMUM_LRS),
    "warmup": False,
}


def multiplier_for_update(update_index: int) -> float:
    """Return the exact multiplier applied before resumed update ``update_index``."""
    if not isinstance(update_index, int):
        raise TypeError("resumed update index must be an integer")
    if not 0 <= update_index <= FINAL_UPDATE_INDEX:
        raise ValueError("resumed update index is outside the approved schedule")
    return MINIMUM_MULTIPLIER + 0.9 * 0.5 * (
        1.0 + math.cos(math.pi * update_index / FINAL_UPDATE_INDEX)
    )


class ExactResumedCosineScheduler:
    """Explicit scheduler whose stored position is the successful update count."""

    _STATE_KEYS = {
        "schema_name",
        "schema_version",
        "formula",
        "scheduler_completed_updates",
        "last_applied_update_index",
        "last_applied_multiplier",
        "current_group_learning_rates",
        "total_planned_resumed_updates",
        "base_learning_rates",
        "minimum_multiplier",
    }

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        completed_updates: int = 0,
    ) -> None:
        if len(optimizer.param_groups) != 2:
            raise ValueError("scheduler requires exactly two optimizer groups")
        if not 0 <= completed_updates <= TOTAL_RESUMED_UPDATES:
            raise ValueError("scheduler completed-update count is invalid")
        self.optimizer = optimizer
        self.scheduler_completed_updates = completed_updates
        self.last_applied_update_index: int | None = (
            completed_updates - 1 if completed_updates else None
        )
        self.last_applied_multiplier: float | None = (
            multiplier_for_update(completed_updates - 1)
            if completed_updates
            else None
        )
        expected_multiplier = self.last_applied_multiplier or 1.0
        self._set_lrs(expected_multiplier)
        self._pending_update_index: int | None = None

    @property
    def current_lrs(self) -> tuple[float, float]:
        return tuple(float(group["lr"]) for group in self.optimizer.param_groups)  # type: ignore[return-value]

    @property
    def next_update_index(self) -> int:
        return self.scheduler_completed_updates

    def _set_lrs(self, multiplier: float) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, BASE_LRS):
            group["lr"] = base_lr * multiplier

    def apply_for_next_update(self) -> tuple[int, float, tuple[float, float]]:
        if self.scheduler_completed_updates >= TOTAL_RESUMED_UPDATES:
            raise RuntimeError("scheduler cannot advance past its approved horizon")
        if self._pending_update_index is not None:
            raise RuntimeError("scheduler position is already pending an optimizer update")
        index = self.scheduler_completed_updates
        multiplier = multiplier_for_update(index)
        self._set_lrs(multiplier)
        self._pending_update_index = index
        return index, multiplier, self.current_lrs

    def mark_successful_update(self, update_index: int) -> None:
        if (
            self._pending_update_index != update_index
            or update_index != self.scheduler_completed_updates
        ):
            raise RuntimeError("scheduler update completion is out of position")
        self.last_applied_update_index = update_index
        self.last_applied_multiplier = multiplier_for_update(update_index)
        self.scheduler_completed_updates += 1
        self._pending_update_index = None

    def state_dict(self) -> dict[str, Any]:
        if self._pending_update_index is not None:
            raise RuntimeError("cannot checkpoint an unconfirmed scheduler update")
        return {
            "schema_name": "viespeaker2_explicit_resumed_cosine",
            "schema_version": SCHEDULER_VERSION,
            "formula": SCHEDULER_FORMULA,
            "scheduler_completed_updates": self.scheduler_completed_updates,
            "last_applied_update_index": self.last_applied_update_index,
            "last_applied_multiplier": self.last_applied_multiplier,
            "current_group_learning_rates": list(self.current_lrs),
            "total_planned_resumed_updates": TOTAL_RESUMED_UPDATES,
            "base_learning_rates": list(BASE_LRS),
            "minimum_multiplier": MINIMUM_MULTIPLIER,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        validate_scheduler_state(state)
        completed = int(state["scheduler_completed_updates"])
        self.scheduler_completed_updates = completed
        self.last_applied_update_index = state["last_applied_update_index"]
        self.last_applied_multiplier = state["last_applied_multiplier"]
        self._pending_update_index = None
        expected_lrs = tuple(float(value) for value in state["current_group_learning_rates"])
        self._set_lrs(self.last_applied_multiplier or 1.0)
        if self.current_lrs != expected_lrs:
            raise ValueError("restored scheduler learning rates are not exact")


def validate_scheduler_state(state: Mapping[str, Any]) -> None:
    if set(state) != ExactResumedCosineScheduler._STATE_KEYS:
        raise ValueError("scheduler state keys are invalid")
    if (
        state["schema_name"] != "viespeaker2_explicit_resumed_cosine"
        or state["schema_version"] != SCHEDULER_VERSION
        or state["formula"] != SCHEDULER_FORMULA
        or state["total_planned_resumed_updates"] != TOTAL_RESUMED_UPDATES
        or tuple(state["base_learning_rates"]) != BASE_LRS
        or state["minimum_multiplier"] != MINIMUM_MULTIPLIER
    ):
        raise ValueError("scheduler configuration changed")
    completed = state["scheduler_completed_updates"]
    if not isinstance(completed, int) or not 0 <= completed <= TOTAL_RESUMED_UPDATES:
        raise ValueError("scheduler completed-update count is invalid")
    expected_index = completed - 1 if completed else None
    expected_multiplier = (
        multiplier_for_update(expected_index) if expected_index is not None else None
    )
    if state["last_applied_update_index"] != expected_index:
        raise ValueError("scheduler last update index is inconsistent")
    if state["last_applied_multiplier"] != expected_multiplier:
        raise ValueError("scheduler last multiplier is inconsistent")
    multiplier = expected_multiplier or 1.0
    expected_lrs = tuple(base * multiplier for base in BASE_LRS)
    if tuple(state["current_group_learning_rates"]) != expected_lrs:
        raise ValueError("scheduler learning rates are inconsistent")


@dataclass
class EarlyStoppingState:
    best_epoch: int = INITIAL_BEST_EPOCH
    best_eer: float = INITIAL_BEST_EER
    best_empirical_average_error: float = INITIAL_EMPIRICAL_AVERAGE_ERROR
    best_threshold: float = INITIAL_EMPIRICAL_THRESHOLD
    bad_epoch_count: int = 0

    def observe(
        self,
        epoch: int,
        *,
        interpolated_eer: float,
        empirical_average_error: float,
        empirical_threshold: float,
    ) -> bool:
        if not FIRST_RESUMED_EPOCH <= epoch <= LAST_RESUMED_EPOCH:
            raise ValueError("validation epoch is outside the resumed range")
        for name, value in (
            ("interpolated EER", interpolated_eer),
            ("empirical average error", empirical_average_error),
            ("empirical threshold", empirical_threshold),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        lower_eer = interpolated_eer < self.best_eer - IMPROVEMENT_TOLERANCE
        tied_eer = abs(interpolated_eer - self.best_eer) <= IMPROVEMENT_TOLERANCE
        lower_empirical = (
            empirical_average_error
            < self.best_empirical_average_error - IMPROVEMENT_TOLERANCE
        )
        improved = lower_eer or (tied_eer and lower_empirical)
        if improved:
            self.best_epoch = epoch
            self.best_eer = float(interpolated_eer)
            self.best_empirical_average_error = float(empirical_average_error)
            self.best_threshold = float(empirical_threshold)
            self.bad_epoch_count = 0
        else:
            self.bad_epoch_count += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epoch_count >= EARLY_STOPPING_PATIENCE

    def state_dict(self) -> dict[str, Any]:
        return {
            "monitor": "validation interpolated EER",
            "patience": EARLY_STOPPING_PATIENCE,
            "improvement_tolerance": IMPROVEMENT_TOLERANCE,
            "best_epoch": self.best_epoch,
            "best_eer": self.best_eer,
            "best_empirical_average_error": self.best_empirical_average_error,
            "best_threshold": self.best_threshold,
            "bad_epoch_count": self.bad_epoch_count,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EarlyStoppingState":
        if (
            state.get("monitor") != "validation interpolated EER"
            or state.get("patience") != EARLY_STOPPING_PATIENCE
            or state.get("improvement_tolerance") != IMPROVEMENT_TOLERANCE
        ):
            raise ValueError("early-stopping configuration changed")
        result = cls(
            best_epoch=int(state["best_epoch"]),
            best_eer=float(state["best_eer"]),
            best_empirical_average_error=float(
                state["best_empirical_average_error"]
            ),
            best_threshold=float(state["best_threshold"]),
            bad_epoch_count=int(state["bad_epoch_count"]),
        )
        if (
            not 0 <= result.best_epoch <= LAST_RESUMED_EPOCH
            or not 0.0 <= result.best_eer <= 1.0
            or not 0.0 <= result.best_empirical_average_error <= 1.0
            or not math.isfinite(result.best_threshold)
            or result.bad_epoch_count < 0
        ):
            raise ValueError("early-stopping state is invalid")
        return result


def validate_epoch0_migration(
    checkpoint: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    validate_checkpoint_v2(checkpoint)
    if (
        checkpoint["epoch"] != 0
        or checkpoint["next_cursor"]
        != {"next_epoch": 1, "next_batch_position": 0}
        or checkpoint["completed_batch_position"] != STEPS_PER_EPOCH - 1
        or checkpoint["global_optimizer_step"] != START_GLOBAL_STEP
        or checkpoint["epoch_1_started"] is not False
        or checkpoint["class_count"] != NUM_CLASSES
        or tuple(checkpoint["aam_state"]["weight"].shape) != (NUM_CLASSES, 192)
    ):
        raise ValueError("epoch-zero checkpoint state differs from the migration contract")
    selection = checkpoint["validation_selection"]
    if (
        selection["epoch_0_interpolated_eer"] != INITIAL_BEST_EER
        or selection["epoch_0_empirical_threshold"]
        != INITIAL_EMPIRICAL_THRESHOLD
    ):
        raise ValueError("epoch-zero validation selection changed")
    if (
        report.get("result") != "PASS"
        or report.get("training", {}).get("epoch_1_started") is not False
        or report.get("validation", {}).get("metrics", {}).get("interpolated_eer")
        != INITIAL_BEST_EER
    ):
        raise ValueError("epoch-zero report does not bind the approved start")


def audit_epoch_plan(
    dataset: Any,
    plan: Sequence[Sequence[int]],
    *,
    epoch: int,
) -> dict[str, Any]:
    if epoch not in EPOCH_PLAN_HASHES or len(plan) != STEPS_PER_EPOCH:
        raise ValueError("epoch plan count or epoch identity is invalid")
    represented: set[str] = set()
    for batch in plan:
        if len(batch) != 32 or len(set(batch)) != 32:
            raise ValueError("epoch plan contains a malformed logical batch")
        speakers: dict[str, int] = {}
        duplicate_groups: set[str] = set()
        for index in batch:
            if not isinstance(index, int) or not 0 <= index < len(dataset):
                raise ValueError("epoch plan contains an invalid Dataset index")
            row = dataset.rows[index]
            if (
                row.final_split != "train"
                or not 0 <= row.speaker_label < NUM_CLASSES
            ):
                raise ValueError("epoch plan selects a non-train identity")
            represented.add(row.speaker_id)
            speakers[row.speaker_id] = speakers.get(row.speaker_id, 0) + 1
            if row.duplicate_group:
                if row.duplicate_group in duplicate_groups:
                    raise ValueError("epoch plan contains a duplicate-group conflict")
                duplicate_groups.add(row.duplicate_group)
        if len(speakers) != 16 or set(speakers.values()) != {2}:
            raise ValueError("epoch plan is not exact P16K2")
    if len(represented) != NUM_CLASSES:
        raise ValueError("epoch plan does not represent every train speaker")
    return {
        "epoch": epoch,
        "logical_batches": len(plan),
        "logical_selections": sum(len(batch) for batch in plan),
        "represented_speakers": len(represented),
        "p16k2": True,
        "unique_indexes_per_batch": True,
        "duplicate_group_conflicts": 0,
        "train_only": True,
    }


def expected_cursor(resumed_updates: int) -> dict[str, int]:
    if not isinstance(resumed_updates, int) or not 0 <= resumed_updates <= TOTAL_RESUMED_UPDATES:
        raise ValueError("resumed update count is invalid")
    return {
        "next_epoch": FIRST_RESUMED_EPOCH + resumed_updates // STEPS_PER_EPOCH,
        "next_batch_position": resumed_updates % STEPS_PER_EPOCH,
    }


def logical_batch_identity(
    batch: Mapping[str, Any], planned_indexes: Sequence[int]
) -> str:
    """Hash portable batch metadata without serializing cached feature tensors."""
    required = (
        "dataset_index",
        "speaker_label",
        "speaker_id",
        "relative_audio_path",
        "duplicate_group",
    )
    if any(key not in batch for key in required):
        raise ValueError("logical batch identity metadata is incomplete")
    dataset_indexes = [int(value) for value in batch["dataset_index"].tolist()]
    labels = [int(value) for value in batch["speaker_label"].tolist()]
    payload = {
        "planned_indexes": [int(value) for value in planned_indexes],
        "dataset_indexes": dataset_indexes,
        "speaker_labels": labels,
        "speaker_ids": list(batch["speaker_id"]),
        "relative_audio_paths": list(batch["relative_audio_path"]),
        "duplicate_groups": list(batch["duplicate_group"]),
    }
    planned = [int(value) for value in planned_indexes]
    if (
        len(dataset_indexes) != len(planned)
        or len(set(dataset_indexes)) != len(dataset_indexes)
        or sorted(dataset_indexes) != sorted(planned)
    ):
        raise ValueError("logical batch identity disagrees with the sampler plan")
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def gradient_diagnostics(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    affected: list[str] = []
    total_elements = finite_elements = nan_elements = positive_inf = negative_inf = 0
    parameters_with_gradients = 0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"parameter {name!r} has no gradient")
        parameters_with_gradients += 1
        gradient = parameter.grad.detach()
        total_elements += gradient.numel()
        finite = int(torch.isfinite(gradient).sum().item())
        nan = int(torch.isnan(gradient).sum().item())
        pos_inf = int(torch.isposinf(gradient).sum().item())
        neg_inf = int(torch.isneginf(gradient).sum().item())
        finite_elements += finite
        nan_elements += nan
        positive_inf += pos_inf
        negative_inf += neg_inf
        if finite != gradient.numel():
            affected.append(name)
    if parameters_with_gradients == 0:
        raise RuntimeError("no trainable gradients were produced")
    return {
        "parameters_with_gradients": parameters_with_gradients,
        "total_elements": total_elements,
        "finite_elements": finite_elements,
        "nonfinite_elements": total_elements - finite_elements,
        "nan_elements": nan_elements,
        "positive_inf_elements": positive_inf,
        "negative_inf_elements": negative_inf,
        "affected_parameter_names": affected,
        "all_finite": finite_elements == total_elements,
    }


def classify_recoverable_overflow(
    *,
    cached_features_finite: bool,
    normalized_features_finite: bool,
    embeddings_finite: bool,
    logits_finite: bool,
    unscaled_loss_finite: bool,
    model_parameters_finite: bool,
    optimizer_state_finite: bool,
    post_unscale_diagnostics: Mapping[str, Any],
    optimizer_step_called: bool,
    counters_advanced: bool,
) -> bool:
    forward_valid = all(
        (
            cached_features_finite,
            normalized_features_finite,
            embeddings_finite,
            logits_finite,
            unscaled_loss_finite,
            model_parameters_finite,
            optimizer_state_finite,
        )
    )
    return (
        forward_valid
        and not bool(post_unscale_diagnostics["all_finite"])
        and not optimizer_step_called
        and not counters_advanced
    )


def retry_scale(original_scale: float) -> float:
    if not math.isfinite(original_scale) or original_scale <= 0.0:
        raise ValueError("original GradScaler scale is invalid")
    reduced = original_scale * AMP_OVERFLOW_POLICY["scale_reduction_factor"]
    if (
        not math.isfinite(reduced)
        or reduced >= original_scale
        or reduced < AMP_OVERFLOW_POLICY["minimum_retry_scale"]
    ):
        raise RuntimeError("AMP overflow retry scale violates the approved floor")
    return reduced


def validate_failed_attempt_invariants(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    required = {
        "optimizer_steps",
        "optimizer_hook_count",
        "parameter_versions",
        "scheduler_completed_updates",
        "global_optimizer_step",
        "batch_position",
        "logical_loss_count",
        "batch_identity",
        "learning_rates",
    }
    if set(before) != required or set(after) != required:
        raise ValueError("overflow transaction snapshot keys are invalid")
    if before != after:
        changed = sorted(key for key in required if before[key] != after[key])
        raise RuntimeError(
            f"failed AMP attempt advanced transactional state: {changed}"
        )


def initial_overflow_state() -> dict[str, Any]:
    return {
        "overflow_attempt_count": 0,
        "recovered_overflow_count": 0,
        "failed_overflow_count": 0,
        "successful_optimizer_updates": 5000,
        "recovery_session_successful_updates": 0,
        "last_successful_batch_identity": None,
        "recovered_events": [],
        "historical_failure": {
            "epoch": 2,
            "logical_batch_position": 2532,
            "attempted_global_optimizer_step": 8470,
            "affected_parameter_name": "blocks.0.norm.norm.weight",
            "saved_batch_identity_available": False,
            "replay_result": "pending",
            "replay_batch_identity": None,
        },
    }


def migrate_checkpoint_for_amp_recovery(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    validate_multiepoch_checkpoint(checkpoint)
    if (
        checkpoint["schema_version"] != SCHEMA_VERSION
        or checkpoint["phase"] != "training"
        or checkpoint["current_epoch"] != 2
        or checkpoint["completed_batch_position"] != 2031
        or checkpoint["global_optimizer_step"] != 7969
        or checkpoint["resumed_optimizer_steps"] != 5000
    ):
        raise ValueError("checkpoint is not the approved AMP recovery cursor")
    migrated = copy.deepcopy(dict(checkpoint))
    migrated["schema_version"] = RECOVERY_SCHEMA_VERSION
    migrated["recovery_origin"] = {
        "path": RECOVERY_START_CHECKPOINT_PATH,
        "sha256": RECOVERY_START_CHECKPOINT_SHA256,
        "historical_failure_bindings": copy.deepcopy(
            HISTORICAL_FAILURE_BINDINGS
        ),
    }
    migrated["overflow_policy"] = copy.deepcopy(AMP_OVERFLOW_POLICY)
    migrated["overflow_state"] = initial_overflow_state()
    migrated["checkpoint_creation_reason"] = "amp_overflow_policy_migration"
    validate_multiepoch_checkpoint(migrated)
    return migrated


def _has_absolute_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_has_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_absolute_path(item) for item in value)
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/")
    return bool(re.match(r"^[A-Za-z]:/", normalized)) or normalized.startswith("/")


def reject_final_test_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("runtime path must be a non-empty string")
    pure = PurePosixPath(value.replace("\\", "/"))
    lowered = [part.lower().replace("-", "_") for part in pure.parts]
    if any(
        part == "test"
        or part == "final_test"
        or part.startswith("test_")
        or part.endswith("_test")
        or "final_test" in part
        for part in lowered
    ):
        raise ValueError(f"final-test path is forbidden: {value!r}")
    return pure.as_posix()


def recovery_action(checkpoint: Mapping[str, Any]) -> str:
    """Return the only permitted action for a validated runtime phase."""
    validate_multiepoch_checkpoint(checkpoint)
    return {
        "training": "resume_training",
        "validation_pending": "resume_validation_only",
        "epoch_complete": "start_next_epoch",
        "stopped": "stop",
    }[checkpoint["phase"]]


def stop_reason_after_validation(
    epoch: int, early_stopping: EarlyStoppingState
) -> str | None:
    if not FIRST_RESUMED_EPOCH <= epoch <= LAST_RESUMED_EPOCH:
        raise ValueError("validated epoch is outside the resumed range")
    if early_stopping.should_stop:
        return "early_stopping"
    if epoch == LAST_RESUMED_EPOCH:
        return "max_epoch"
    return None


CHECKPOINT_REQUIRED_KEYS = {
    "schema_name",
    "schema_version",
    "original_epoch_zero_checkpoint",
    "epoch_zero_report",
    "upstream_bindings",
    "sampler_binding",
    "epoch_plan_hashes",
    "model_source",
    "feature_shape",
    "class_count",
    "label_range",
    "aam_configuration",
    "optimizer_configuration",
    "training_configuration",
    "mixed_precision_configuration",
    "scheduler_configuration",
    "embedding_model_state",
    "mean_var_norm_state",
    "aam_state",
    "optimizer_state",
    "grad_scaler_state",
    "scheduler_state",
    "batchnorm_policy",
    "batchnorm_reference_state",
    "rng_state",
    "dataloader_generator_state",
    "current_epoch",
    "completed_batch_position",
    "global_optimizer_step",
    "resumed_optimizer_steps",
    "next_cursor",
    "phase",
    "completed_epochs",
    "validation_history",
    "best_state",
    "early_stopping_state",
    "current_epoch_statistics",
    "checkpoint_creation_reason",
    "stop_reason",
    "epoch_zero_retrained",
    "invalid_or_skipped_updates",
}

RECOVERY_CHECKPOINT_KEYS = {
    "recovery_origin",
    "overflow_policy",
    "overflow_state",
}


def validate_overflow_state(state: Mapping[str, Any], resumed: int) -> None:
    required = {
        "overflow_attempt_count",
        "recovered_overflow_count",
        "failed_overflow_count",
        "successful_optimizer_updates",
        "recovery_session_successful_updates",
        "last_successful_batch_identity",
        "recovered_events",
        "historical_failure",
    }
    if set(state) != required:
        raise ValueError("AMP overflow state keys are invalid")
    for key in (
        "overflow_attempt_count",
        "recovered_overflow_count",
        "failed_overflow_count",
        "successful_optimizer_updates",
        "recovery_session_successful_updates",
    ):
        if not isinstance(state[key], int) or state[key] < 0:
            raise ValueError(f"AMP overflow counter {key!r} is invalid")
    if (
        state["overflow_attempt_count"]
        != state["recovered_overflow_count"] + state["failed_overflow_count"]
        or len(state["recovered_events"]) != state["recovered_overflow_count"]
        or state["successful_optimizer_updates"] != resumed
        or state["recovery_session_successful_updates"]
        != resumed - 5000
    ):
        raise ValueError("AMP overflow counters disagree with the checkpoint cursor")
    identity = state["last_successful_batch_identity"]
    if identity is not None and (
        not isinstance(identity, str)
        or re.fullmatch(r"[0-9a-f]{64}", identity) is None
    ):
        raise ValueError("last successful logical-batch identity is invalid")
    historical = state["historical_failure"]
    required_historical = {
        "epoch",
        "logical_batch_position",
        "attempted_global_optimizer_step",
        "affected_parameter_name",
        "saved_batch_identity_available",
        "replay_result",
        "replay_batch_identity",
    }
    if set(historical) != required_historical:
        raise ValueError("historical overflow replay state keys are invalid")
    if (
        historical["epoch"] != 2
        or historical["logical_batch_position"] != 2532
        or historical["attempted_global_optimizer_step"] != 8470
        or historical["affected_parameter_name"]
        != "blocks.0.norm.norm.weight"
        or historical["saved_batch_identity_available"] is not False
        or historical["replay_result"]
        not in {
            "pending",
            "not_reproduced",
            "reproduced_recovered",
            "reproduced_failed",
        }
    ):
        raise ValueError("historical overflow replay state is invalid")


def validate_multiepoch_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    missing = CHECKPOINT_REQUIRED_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"multi-epoch checkpoint is missing keys: {sorted(missing)}")
    schema_version = checkpoint["schema_version"]
    if (
        checkpoint["schema_name"] != SCHEMA_NAME
        or schema_version not in {SCHEMA_VERSION, RECOVERY_SCHEMA_VERSION}
    ):
        raise ValueError("multi-epoch checkpoint schema is invalid")
    if schema_version == SCHEMA_VERSION:
        if RECOVERY_CHECKPOINT_KEYS & set(checkpoint):
            raise ValueError("legacy checkpoint unexpectedly contains recovery state")
    else:
        missing_recovery = RECOVERY_CHECKPOINT_KEYS - set(checkpoint)
        if missing_recovery:
            raise ValueError(
                f"recovery checkpoint is missing keys: {sorted(missing_recovery)}"
            )
        if checkpoint["recovery_origin"] != {
            "path": RECOVERY_START_CHECKPOINT_PATH,
            "sha256": RECOVERY_START_CHECKPOINT_SHA256,
            "historical_failure_bindings": HISTORICAL_FAILURE_BINDINGS,
        }:
            raise ValueError("AMP recovery origin binding changed")
        if checkpoint["overflow_policy"] != AMP_OVERFLOW_POLICY:
            raise ValueError("AMP overflow recovery policy changed")
    if _has_absolute_path(checkpoint):
        raise ValueError("multi-epoch checkpoint contains an absolute path")
    if checkpoint["original_epoch_zero_checkpoint"] != {
        "path": START_CHECKPOINT_PATH,
        "sha256": START_CHECKPOINT_SHA256,
        "completed_epoch": 0,
        "global_optimizer_step": START_GLOBAL_STEP,
    }:
        raise ValueError("original epoch-zero checkpoint binding changed")
    if checkpoint["epoch_zero_report"] != {
        "path": START_REPORT_PATH,
        "sha256": START_REPORT_SHA256,
    }:
        raise ValueError("epoch-zero report binding changed")
    if (
        checkpoint["upstream_bindings"] != UPSTREAM_BINDINGS
        or checkpoint["sampler_binding"] != SAMPLER_BINDING
        or checkpoint["epoch_plan_hashes"]
        != {str(key): value for key, value in EPOCH_PLAN_HASHES.items()}
        or checkpoint["feature_shape"] != FEATURE_SHAPE
        or checkpoint["class_count"] != NUM_CLASSES
        or checkpoint["label_range"] != [0, NUM_CLASSES - 1]
        or checkpoint["aam_configuration"] != AAM_CONFIGURATION
        or checkpoint["optimizer_configuration"] != OPTIMIZER_CONFIGURATION
        or checkpoint["training_configuration"]
        != MULTIEPOCH_TRAINING_CONFIGURATION
        or checkpoint["mixed_precision_configuration"]
        != MIXED_PRECISION_CONFIGURATION
        or checkpoint["scheduler_configuration"] != SCHEDULER_CONFIGURATION
        or checkpoint["batchnorm_policy"] != BATCHNORM_POLICY
    ):
        raise ValueError("multi-epoch configuration or upstream binding changed")
    validate_scheduler_state(checkpoint["scheduler_state"])
    resumed = checkpoint["resumed_optimizer_steps"]
    if schema_version == RECOVERY_SCHEMA_VERSION:
        validate_overflow_state(checkpoint["overflow_state"], resumed)
    if (
        checkpoint["scheduler_state"]["scheduler_completed_updates"] != resumed
        or checkpoint["global_optimizer_step"] != START_GLOBAL_STEP + resumed
        or checkpoint["epoch_zero_retrained"] is not False
        or checkpoint["invalid_or_skipped_updates"] != 0
    ):
        raise ValueError("optimizer/scheduler counters are inconsistent")
    optimizer_steps = {
        int(value["step"].item() if isinstance(value["step"], torch.Tensor) else value["step"])
        for value in checkpoint["optimizer_state"]["state"].values()
    }
    if optimizer_steps != {checkpoint["global_optimizer_step"]}:
        raise ValueError("AdamW state counters disagree with the global step")
    if tuple(checkpoint["aam_state"]["weight"].shape) != (NUM_CLASSES, 192):
        raise ValueError("AAM checkpoint shape changed")
    phase = checkpoint["phase"]
    epoch = checkpoint["current_epoch"]
    position = checkpoint["completed_batch_position"]
    cursor = checkpoint["next_cursor"]
    if phase not in PHASES:
        raise ValueError("checkpoint phase is invalid")
    if epoch == 0:
        if not (
            phase == "epoch_complete"
            and resumed == 0
            and position == STEPS_PER_EPOCH
            and cursor == {"next_epoch": 1, "next_batch_position": 0}
        ):
            raise ValueError("migrated epoch-zero phase/cursor is invalid")
    elif phase == "training":
        expected = (epoch - 1) * STEPS_PER_EPOCH + position
        if not (
            FIRST_RESUMED_EPOCH <= epoch <= LAST_RESUMED_EPOCH
            and 0 <= position < STEPS_PER_EPOCH
            and resumed == expected
            and cursor == {"next_epoch": epoch, "next_batch_position": position}
        ):
            raise ValueError("training phase/cursor is invalid")
    else:
        expected = epoch * STEPS_PER_EPOCH
        if not (
            FIRST_RESUMED_EPOCH <= epoch <= LAST_RESUMED_EPOCH
            and position == STEPS_PER_EPOCH
            and resumed == expected
            and cursor == {"next_epoch": epoch + 1, "next_batch_position": 0}
        ):
            raise ValueError("completed training phase/cursor is invalid")
    completed = checkpoint["completed_epochs"]
    history_epochs = [record["epoch"] for record in checkpoint["validation_history"]]
    expected_completed = list(range(0, epoch + 1))
    if phase in {"training", "validation_pending"} and epoch > 0:
        expected_completed = list(range(0, epoch))
    if completed != expected_completed or history_epochs != completed:
        raise ValueError("completed epoch and validation history disagree")
    early = EarlyStoppingState.from_state_dict(checkpoint["early_stopping_state"])
    best = checkpoint["best_state"]
    if (
        best["epoch"] != early.best_epoch
        or best["interpolated_eer"] != early.best_eer
        or best["empirical_average_error"]
        != early.best_empirical_average_error
        or best["empirical_threshold"] != early.best_threshold
    ):
        raise ValueError("best checkpoint selection state is inconsistent")
    stop_reason = checkpoint["stop_reason"]
    if phase == "stopped":
        if stop_reason not in STOP_REASONS:
            raise ValueError("stopped checkpoint has no valid stop reason")
        if stop_reason == "early_stopping" and not early.should_stop:
            raise ValueError("early-stopped checkpoint has insufficient bad epochs")
        if stop_reason == "max_epoch" and epoch != LAST_RESUMED_EPOCH:
            raise ValueError("max-epoch checkpoint stopped at the wrong epoch")
    elif stop_reason is not None:
        raise ValueError("non-stopped checkpoint records a stop reason")
