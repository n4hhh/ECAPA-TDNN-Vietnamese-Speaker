"""State and policy helpers for resumed multi-epoch ECAPA/AAM fine-tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

import torch


SCHEMA_NAME = "speaker_verification_ecapa_aam_multiepoch"
SCHEMA_VERSION = 1
START_GLOBAL_STEP = 1000
FIRST_RESUMED_EPOCH = 1
LAST_RESUMED_EPOCH = 4
STEPS_PER_EPOCH = 1000
MAX_RESUMED_STEPS = 4000
INITIAL_BEST_EPOCH = 0
INITIAL_BEST_EER = 0.0646251536255633
INITIAL_EMPIRICAL_THRESHOLD = 0.1744520664215088
MIN_DELTA = 0.0001
PATIENCE = 2
BASE_LRS = (1.0e-5, 1.0e-3)
MIN_LRS = (1.0e-6, 1.0e-4)


def cosine_factor(completed_resumed_steps: int, total_steps: int = MAX_RESUMED_STEPS) -> float:
    """Return the fixed multiplicative cosine factor for completed updates."""
    if not isinstance(completed_resumed_steps, int):
        raise TypeError("completed_resumed_steps must be an integer")
    if not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    progress = min(1.0, max(0.0, completed_resumed_steps / total_steps))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


class ResumedCosineScheduler:
    """Two-group, optimizer-step cosine schedule with an explicit portable state."""

    _STATE_KEYS = {
        "schema_name",
        "schema_version",
        "completed_resumed_steps",
        "total_steps",
        "base_lrs",
        "last_factor",
        "last_lrs",
    }

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        completed_resumed_steps: int = 0,
        total_steps: int = MAX_RESUMED_STEPS,
        base_lrs: tuple[float, float] = BASE_LRS,
    ) -> None:
        if len(optimizer.param_groups) != 2:
            raise ValueError("cosine scheduler requires exactly two optimizer groups")
        if len(base_lrs) != 2 or any(not math.isfinite(value) or value <= 0 for value in base_lrs):
            raise ValueError("base_lrs must contain two finite positive values")
        if not 0 <= completed_resumed_steps <= total_steps:
            raise ValueError("completed scheduler steps are outside the schedule")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.base_lrs = tuple(float(value) for value in base_lrs)
        self.completed_resumed_steps = completed_resumed_steps
        self._apply()

    @property
    def factor(self) -> float:
        return cosine_factor(self.completed_resumed_steps, self.total_steps)

    @property
    def lrs(self) -> tuple[float, float]:
        return tuple(float(group["lr"]) for group in self.optimizer.param_groups)  # type: ignore[return-value]

    def _apply(self) -> None:
        factor = self.factor
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        if self.completed_resumed_steps >= self.total_steps:
            raise RuntimeError("cosine scheduler cannot step beyond its fixed horizon")
        self.completed_resumed_steps += 1
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "resumed_cosine_lr",
            "schema_version": 1,
            "completed_resumed_steps": self.completed_resumed_steps,
            "total_steps": self.total_steps,
            "base_lrs": list(self.base_lrs),
            "last_factor": self.factor,
            "last_lrs": list(self.lrs),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != self._STATE_KEYS:
            raise ValueError("scheduler checkpoint keys are invalid")
        if state["schema_name"] != "resumed_cosine_lr" or state["schema_version"] != 1:
            raise ValueError("scheduler checkpoint schema is invalid")
        if state["total_steps"] != self.total_steps:
            raise ValueError("scheduler total-step horizon changed")
        if tuple(state["base_lrs"]) != self.base_lrs:
            raise ValueError("scheduler base learning rates changed")
        completed = state["completed_resumed_steps"]
        if not isinstance(completed, int) or not 0 <= completed <= self.total_steps:
            raise ValueError("scheduler completed-step count is invalid")
        expected_factor = cosine_factor(completed, self.total_steps)
        expected_lrs = tuple(base * expected_factor for base in self.base_lrs)
        if not math.isclose(float(state["last_factor"]), expected_factor, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("scheduler factor is inconsistent with its step count")
        if any(
            not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
            for actual, expected in zip(state["last_lrs"], expected_lrs)
        ):
            raise ValueError("scheduler learning rates are inconsistent with its step count")
        self.completed_resumed_steps = completed
        self._apply()


@dataclass
class EarlyStoppingState:
    best_epoch: int = INITIAL_BEST_EPOCH
    best_eer: float = INITIAL_BEST_EER
    patience_counter: int = 0

    def observe(self, epoch: int, validation_eer: float) -> bool:
        if not FIRST_RESUMED_EPOCH <= epoch <= LAST_RESUMED_EPOCH:
            raise ValueError("validation epoch is outside the resumed range")
        if not math.isfinite(validation_eer) or not 0.0 <= validation_eer <= 1.0:
            raise ValueError("validation EER must be finite and in [0,1]")
        improved = validation_eer < self.best_eer - MIN_DELTA
        if improved:
            self.best_epoch = epoch
            self.best_eer = float(validation_eer)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience_counter >= PATIENCE

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_eer": self.best_eer,
            "patience_counter": self.patience_counter,
            "patience": PATIENCE,
            "min_delta": MIN_DELTA,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "EarlyStoppingState":
        if state.get("patience") != PATIENCE or state.get("min_delta") != MIN_DELTA:
            raise ValueError("early-stopping configuration changed")
        result = cls(
            best_epoch=int(state["best_epoch"]),
            best_eer=float(state["best_eer"]),
            patience_counter=int(state["patience_counter"]),
        )
        if (
            not INITIAL_BEST_EPOCH <= result.best_epoch <= LAST_RESUMED_EPOCH
            or not math.isfinite(result.best_eer)
            or result.patience_counter < 0
        ):
            raise ValueError("early-stopping checkpoint state is invalid")
        return result


def start_epoch(epoch: int, early_stopping: EarlyStoppingState, sampler: Any) -> None:
    """Fail closed before an epoch begins, then set the sampler epoch exactly once."""
    if epoch == 0:
        raise RuntimeError("epoch 0 is complete and must never be retrained")
    if not FIRST_RESUMED_EPOCH <= epoch <= LAST_RESUMED_EPOCH:
        raise RuntimeError("requested epoch is outside the approved resumed range")
    if early_stopping.should_stop:
        raise RuntimeError("no epoch may start after early stopping triggers")
    sampler.set_epoch(epoch)
    if getattr(sampler, "epoch", None) != epoch:
        raise RuntimeError("sampler.set_epoch did not take effect")


def validate_resume_counters(
    *,
    global_optimizer_step: int,
    resumed_optimizer_steps: int,
    next_epoch: int,
    next_logical_batch_position: int,
) -> None:
    if global_optimizer_step != START_GLOBAL_STEP + resumed_optimizer_steps:
        raise ValueError("global and resumed optimizer-step counters disagree")
    if not 0 <= resumed_optimizer_steps <= MAX_RESUMED_STEPS:
        raise ValueError("resumed optimizer-step counter is invalid")
    if not FIRST_RESUMED_EPOCH <= next_epoch <= LAST_RESUMED_EPOCH + 1:
        raise ValueError("next epoch is outside the approved cursor range")
    if not 0 <= next_logical_batch_position < STEPS_PER_EPOCH:
        raise ValueError("next logical batch position is invalid")
    expected = (next_epoch - FIRST_RESUMED_EPOCH) * STEPS_PER_EPOCH + next_logical_batch_position
    if resumed_optimizer_steps != expected:
        raise ValueError("epoch/batch cursor disagrees with resumed optimizer steps")


def reject_final_test_path(value: str) -> str:
    """Reject final-test-like paths before any filesystem operation."""
    if not isinstance(value, str) or not value:
        raise ValueError("runtime path must be a non-empty string")
    pure = PurePosixPath(value.replace("\\", "/"))
    lowered = [part.lower().replace("-", "_") for part in pure.parts]
    forbidden = any(
        part == "test"
        or part == "final_test"
        or part.startswith("test_")
        or part.endswith("_test")
        or "final_test" in part
        for part in lowered
    )
    if forbidden:
        raise ValueError(f"final-test path is forbidden: {value!r}")
    return pure.as_posix()


def validate_epoch0_migration(
    checkpoint: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> None:
    """Validate the one and only scheduler-less checkpoint migration."""
    expected = {
        "epoch": 0,
        "next_epoch": 1,
        "next_logical_batch_position": 0,
        "global_optimizer_step": START_GLOBAL_STEP,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("epoch-0 checkpoint counters do not match the migration contract")
    if "scheduler_state_dict" in checkpoint:
        raise ValueError("epoch-0 migration is only for a scheduler-less pilot checkpoint")
    if validation_metrics.get("interpolated_eer") != INITIAL_BEST_EER:
        raise ValueError("epoch-0 validation EER does not match the migration contract")
    if validation_metrics.get("empirical_threshold") != INITIAL_EMPIRICAL_THRESHOLD:
        raise ValueError("epoch-0 empirical threshold does not match the migration contract")
    if validation_metrics.get("score_count") != 19528:
        raise ValueError("epoch-0 fixed-trial score count is invalid")


CHECKPOINT_REQUIRED_KEYS = {
    "schema_name",
    "schema_version",
    "pretrained_model_identifier",
    "embedding_model_state_dict",
    "mean_var_norm_state_dict",
    "aam_classifier_state_dict",
    "optimizer_state_dict",
    "grad_scaler_state_dict",
    "scheduler_state_dict",
    "epoch",
    "next_epoch",
    "next_logical_batch_position",
    "pending_validation_epoch",
    "global_optimizer_step",
    "resumed_optimizer_steps",
    "sampler_state",
    "rng_state",
    "early_stopping_state",
    "batchnorm_policy",
    "batchnorm_reference_state",
    "training_configuration",
    "aam_configuration",
    "optimizer_configuration",
    "amp_policy",
    "start_checkpoint",
    "epoch_metrics",
    "completed_epochs",
    "current_epoch_statistics",
    "protected_hashes_before",
    "checkpoint_creation_reason",
    "stop_reason",
    "invalid_or_skipped_updates",
    "scheduler_migration",
}


def validate_multiepoch_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    missing = CHECKPOINT_REQUIRED_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"multi-epoch checkpoint is missing keys: {sorted(missing)}")
    if checkpoint["schema_name"] != SCHEMA_NAME or checkpoint["schema_version"] != SCHEMA_VERSION:
        raise ValueError("multi-epoch checkpoint schema is invalid")
    validate_resume_counters(
        global_optimizer_step=checkpoint["global_optimizer_step"],
        resumed_optimizer_steps=checkpoint["resumed_optimizer_steps"],
        next_epoch=checkpoint["next_epoch"],
        next_logical_batch_position=checkpoint["next_logical_batch_position"],
    )
    scheduler = checkpoint["scheduler_state_dict"]
    if scheduler.get("completed_resumed_steps") != checkpoint["resumed_optimizer_steps"]:
        raise ValueError("checkpoint scheduler and optimizer-step counters disagree")
    early = EarlyStoppingState.from_state_dict(checkpoint["early_stopping_state"])
    if checkpoint["next_epoch"] <= LAST_RESUMED_EPOCH and early.should_stop and checkpoint["stop_reason"] is None:
        raise ValueError("early-stopped checkpoint must record its stop reason")
    if checkpoint["pending_validation_epoch"] is not None:
        pending = checkpoint["pending_validation_epoch"]
        if pending != checkpoint["next_epoch"] - 1 or checkpoint["next_logical_batch_position"] != 0:
            raise ValueError("pending-validation checkpoint cursor is invalid")
    if checkpoint["invalid_or_skipped_updates"] != 0:
        raise ValueError("checkpoint records an invalid or skipped optimizer update")
    if checkpoint["scheduler_migration"] != {
        "source_schema": "speaker_verification_ecapa_aam_one_epoch_pilot",
        "source_had_scheduler_state": False,
        "initialized_completed_resumed_steps": 0,
    }:
        raise ValueError("epoch-0 scheduler migration record is invalid")
