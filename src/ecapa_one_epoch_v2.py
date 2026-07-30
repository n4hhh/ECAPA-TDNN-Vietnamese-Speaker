"""Fail-closed helpers for the single approved VieSpeaker2.0 epoch-zero run."""

from __future__ import annotations

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
    values_exactly_equal,
)


SCHEMA_NAME = "viespeaker2_ecapa_aam_one_epoch"
SCHEMA_VERSION = 2
RUNTIME_SCHEMA_NAME = "viespeaker2_ecapa_aam_one_epoch_runtime"
RUNTIME_SCHEMA_VERSION = 2
SEED = 20260729
EPOCH = 0
NUM_CLASSES = 1347
EMBEDDING_DIM = 192
FEATURE_SHAPE = [301, 80]
NUM_LOGICAL_BATCHES = 2969
LOGICAL_BATCH_SIZE = 32
TOTAL_SELECTIONS = 95008
SPEAKERS_PER_BATCH = 16
SAMPLES_PER_SPEAKER = 2
ACTIVE_SHARD_WINDOW = 8
PHYSICAL_MICROBATCH_SIZE = 4
ACCUMULATION_STEPS = 8
ROLLING_CHECKPOINT_STEPS = (500, 1000, 1500, 2000, 2500, 2969)
LOGGING_STEPS = tuple(range(100, 2901, 100)) + (2969,)
QUALITY_TOLERANCE = 1e-12
BASELINE_INTERPOLATED_EER = 0.1257

UPSTREAM_BINDINGS = {
    "portable_package_identity": {
        "path": "manifests/portable_v2/portable_manifests_v2_identity.json",
        "sha256": "29f374366c917fb6c54dcd44eeeff59ccc46a732b270188e7157a586e4d9e7c5",
    },
    "train_manifest": {
        "path": "manifests/portable_v2/train_manifest_v2.csv",
        "sha256": "f76aa0321f5f9a2714b2bad9f4b9ab0fd155075f26b50397f79931c8a4bd552b",
    },
    "validation_manifest": {
        "path": "manifests/portable_v2/validation_manifest_v2.csv",
        "sha256": "9c85332cbcd3e33818055c526c0bc54e5b86e7a2c4ed8b3c869433412c24a6fc",
    },
    "train_label_mapping": {
        "path": "manifests/portable_v2/speaker_to_label_v2.json",
        "sha256": "9d4e9015d25f023b8466f7932c296faece937104b17120bdb85c45ad10623cd8",
    },
    "cache_config": {
        "path": "outputs/fbank_cache_v2/fbank_cache_config_v2.json",
        "sha256": "ec71959ec64361038991e760e772d11bf1779e1b505a892dff364cf45aaeb018",
    },
    "cache_identity": {
        "path": "outputs/fbank_cache_v2/fbank_cache_identity_v2.json",
        "sha256": "1a2d6af777311f687e887575bfaf20915ed0409fd2e05b2f1ac232d43cd0b8c8",
    },
    "train_cache_index": {
        "path": "outputs/fbank_cache_v2/train_feature_index_v2.csv",
        "sha256": "e20c320fc5842502a26684023bb307a7b2afa27a14a3cf1130fdffe31e85d4b9",
    },
    "validation_cache_index": {
        "path": "outputs/fbank_cache_v2/validation_feature_index_v2.csv",
        "sha256": "1d3a95e95aaa5b13e6614c077fbbdf10f7c208c79970c2a85bdd461193b9f585",
    },
    "training_sampler_config": {
        "path": "configs/v2/training_sampler_v2.json",
        "sha256": "e59d3794371095acddcee17218ff99fef4393cb15a7f7176c2d7680120436899",
    },
    "validation_trial_csv": {
        "path": "manifests/verification_v2/validation_trials_v2.csv",
        "sha256": "11bec5ff0a0a4ca4930e2664bdc391388a9a677795afaefe5de3fee2d0d39e3d",
    },
    "validation_trial_config": {
        "path": "manifests/verification_v2/validation_trials_config_v2.json",
        "sha256": "9e725ce006ae522f0f0274739b75e9aee7ebb302db331fead73c79cdc326822e",
    },
    "validation_trial_identity": {
        "path": "manifests/verification_v2/validation_trials_identity_v2.json",
        "sha256": "09b55236ad3f80537e1517a5efa7ad1d7a7efc3454f6b56d9ba62af83cc6bf73",
    },
    "pretrained_baseline_identity": {
        "path": "outputs/pretrained_ecapa_validation_baseline_v2/baseline_identity_v2.json",
        "sha256": "5000ddabe804f4cfd7c905c3ed51bc8fd1d0273822ca2091019da923f387974c",
    },
}

SAMPLER_BINDING = {
    "name": "HybridShardAwareSpeakerBatchSampler",
    "config_sha256": UPSTREAM_BINDINGS["training_sampler_config"]["sha256"],
    "combined_plan_sha256": "b11a97fd45f11b8f71fa980ebd26cb335bac0f3bcdfb781e8f87ced535606e4f",
    "epoch_0_plan_sha256": "010e80f042ae91f1c890045d12833005359833466ba6d5a4b5fc70e5c8f04a68",
    "epoch_1_plan_sha256": "c5ff9c7fca4c88541c47738706b1852293e6727da9bb420350e75379179bba0f",
    "seed": SEED,
    "epoch": EPOCH,
    "speakers_per_batch": SPEAKERS_PER_BATCH,
    "samples_per_speaker": SAMPLES_PER_SPEAKER,
    "logical_batch_size": LOGICAL_BATCH_SIZE,
    "active_shard_window": ACTIVE_SHARD_WINDOW,
    "num_logical_batches": NUM_LOGICAL_BATCHES,
    "logical_selections": TOTAL_SELECTIONS,
    "duplicate_group_safe": True,
}

TRAINING_CONFIGURATION = {
    "epoch": EPOCH,
    "logical_batches": NUM_LOGICAL_BATCHES,
    "logical_batch_size": LOGICAL_BATCH_SIZE,
    "physical_microbatch_size": PHYSICAL_MICROBATCH_SIZE,
    "gradient_accumulation_microbatches": ACCUMULATION_STEPS,
    "logical_selections": TOTAL_SELECTIONS,
    "workers": 0,
    "dataset_lru_shards": 8,
    "dataset_validate_finite": False,
    "augmentation": False,
    "scheduler": False,
    "warmup": False,
    "gradient_clipping": False,
    "early_stopping": False,
}
AAM_CONFIGURATION = {
    "embedding_dimension": EMBEDDING_DIM,
    "classes": NUM_CLASSES,
    "weight_shape": [NUM_CLASSES, EMBEDDING_DIM],
    "bias": False,
    "margin": 0.2,
    "scale": 30.0,
    "initialization": "deterministic_xavier_uniform",
    "seed": SEED,
    "math_dtype": "float32",
}
OPTIMIZER_CONFIGURATION = {
    "name": "AdamW",
    "groups": [
        {"name": "embedding_model", "learning_rate": 1e-5, "weight_decay": 1e-4},
        {"name": "aam_classifier", "learning_rate": 1e-3, "weight_decay": 1e-4},
    ],
    "scheduler": None,
    "gradient_clipping": None,
}
MIXED_PRECISION_CONFIGURATION = {
    "enabled": True,
    "ecapa_autocast_dtype": "float16",
    "aam_logits_loss_dtype": "float32",
    "grad_scaler_initial_scale": 128.0,
}
BATCHNORM_POLICY = {
    "embedding_model_training": True,
    "batchnorm_modules_eval": True,
    "batchnorm_affine_trainable": True,
    "running_buffers_bit_exactly_frozen": True,
    "mean_var_norm_optimized": False,
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_relative_path(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{description} must be a string")
    pure = PurePosixPath(value.replace("\\", "/"))
    lowered = [part.lower().replace("-", "_") for part in pure.parts]
    if (
        not value
        or pure.is_absolute()
        or Path(value).is_absolute()
        or ".." in pure.parts
        or ":" in value
        or any(
            part == "test"
            or part == "final_test"
            or part.startswith("test_")
            or part.endswith("_test")
            or "final_test" in part
            for part in lowered
        )
    ):
        raise ValueError(f"unsafe or quarantined {description}: {value!r}")
    return pure.as_posix()


def validate_upstream_bindings(bindings: Mapping[str, Any]) -> None:
    if bindings != UPSTREAM_BINDINGS:
        raise ValueError("approved v2 upstream bindings changed")
    for name, binding in bindings.items():
        if set(binding) != {"path", "sha256"}:
            raise ValueError(f"{name} binding has invalid fields")
        validate_relative_path(binding["path"], f"{name} path")
        if not _is_sha256(binding["sha256"]):
            raise ValueError(f"{name} binding has invalid SHA-256")


def validate_sampler_binding(binding: Mapping[str, Any]) -> None:
    if binding != SAMPLER_BINDING:
        raise ValueError("approved v2 sampler binding changed")


@dataclass
class OneEpochV2Controller:
    epochs_started: list[int] = field(default_factory=list)
    epochs_completed: list[int] = field(default_factory=list)
    optimizer_updates: int = 0
    finished: bool = False

    def start(self, epoch: int) -> None:
        if self.finished or self.epochs_started:
            raise RuntimeError("the v2 one-epoch run cannot start another epoch")
        if epoch != EPOCH:
            raise ValueError("only epoch 0 is approved")
        self.epochs_started.append(epoch)

    def record_update(self, batch_position: int) -> None:
        if self.epochs_started != [EPOCH] or self.finished:
            raise RuntimeError("epoch 0 is not active")
        if batch_position != self.optimizer_updates:
            raise ValueError("batch positions must be contiguous from zero")
        if self.optimizer_updates >= NUM_LOGICAL_BATCHES:
            raise RuntimeError("the approved epoch is already complete")
        self.optimizer_updates += 1

    def finish(self) -> dict[str, Any]:
        if self.finished or self.optimizer_updates != NUM_LOGICAL_BATCHES:
            raise RuntimeError("epoch 0 cannot finish at the current update count")
        self.epochs_completed.append(EPOCH)
        self.finished = True
        return {
            "epochs_started": [0],
            "epochs_completed": [0],
            "optimizer_updates": NUM_LOGICAL_BATCHES,
            "logical_selections": TOTAL_SELECTIONS,
            "next_epoch": 1,
            "next_batch_position": 0,
            "epoch_1_started": False,
        }


def resume_cursor(global_optimizer_step: int) -> dict[str, int]:
    if not isinstance(global_optimizer_step, int) or not (
        1 <= global_optimizer_step <= NUM_LOGICAL_BATCHES
    ):
        raise ValueError("global optimizer step is outside epoch 0")
    if global_optimizer_step == NUM_LOGICAL_BATCHES:
        return {"next_epoch": 1, "next_batch_position": 0}
    return {"next_epoch": 0, "next_batch_position": global_optimizer_step}


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("summary values must be finite and non-empty")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "first": numeric[0],
        "final": numeric[-1],
        "mean": sum(numeric) / len(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def validate_logical_batch(
    batch: Mapping[str, Any], planned_indexes: Sequence[int],
) -> dict[str, Any]:
    required = {
        "fbank",
        "dataset_index",
        "speaker_label",
        "speaker_id",
        "relative_audio_path",
        "final_split",
        "duplicate_group",
    }
    if required - set(batch):
        raise ValueError("logical batch is missing v2 metadata")
    features = batch["fbank"]
    indexes = batch["dataset_index"]
    labels = batch["speaker_label"]
    speakers = batch["speaker_id"]
    groups = batch["duplicate_group"]
    if (
        not isinstance(features, Tensor)
        or tuple(features.shape) != (LOGICAL_BATCH_SIZE, *FEATURE_SHAPE)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not bool(torch.isfinite(features).all().item())
    ):
        raise ValueError("logical Fbank batch must be finite CPU float32 [32,301,80]")
    if (
        not isinstance(indexes, Tensor)
        or indexes.dtype != torch.long
        or tuple(indexes.shape) != (LOGICAL_BATCH_SIZE,)
        or len(set(indexes.tolist())) != LOGICAL_BATCH_SIZE
        or sorted(indexes.tolist()) != sorted(planned_indexes)
    ):
        raise ValueError("logical Dataset indexes are invalid")
    if (
        not isinstance(labels, Tensor)
        or labels.dtype != torch.long
        or tuple(labels.shape) != (LOGICAL_BATCH_SIZE,)
        or bool(((labels < 0) | (labels >= NUM_CLASSES)).any().item())
    ):
        raise ValueError("logical labels must be int64 in 0..1346")
    if (
        not isinstance(speakers, Sequence)
        or len(speakers) != LOGICAL_BATCH_SIZE
        or len(set(speakers)) != SPEAKERS_PER_BATCH
        or set(speakers.count(speaker) for speaker in set(speakers))
        != {SAMPLES_PER_SPEAKER}
        or batch["final_split"] != ["train"] * LOGICAL_BATCH_SIZE
    ):
        raise ValueError("logical batch is not exact train-only P16K2")
    for speaker in set(speakers):
        speaker_groups = [
            group
            for current, group in zip(speakers, groups)
            if current == speaker and group
        ]
        if len(speaker_groups) != len(set(speaker_groups)):
            raise ValueError("logical batch has a duplicate-group conflict")
    return dict(batch)


def batchnorm_reference_state(module: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for module_name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(child, buffer_name, None)
                if value is not None:
                    state[f"{module_name}.{buffer_name}"] = value.detach().clone()
    if not state:
        raise ValueError("embedding model has no BatchNorm running buffers")
    return state


def assert_batchnorm_reference_exact(
    module: nn.Module, reference: Mapping[str, Tensor],
) -> None:
    actual = {
        f"{module_name}.{buffer_name}": getattr(child, buffer_name)
        for module_name, child in module.named_modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
        for buffer_name in ("running_mean", "running_var", "num_batches_tracked")
        if getattr(child, buffer_name, None) is not None
    }
    if set(actual) != set(reference):
        raise RuntimeError("BatchNorm running-buffer keys changed")
    device = next(iter(actual.values())).device
    exact = torch.ones((), dtype=torch.bool, device=device)
    for name, value in actual.items():
        expected = reference[name].to(device=value.device)
        exact.logical_and_(torch.eq(value, expected).all())
    if not bool(exact.item()):
        raise RuntimeError("BatchNorm running buffers changed")


def finite_gradient_norm(
    parameters: Sequence[tuple[str, nn.Parameter]],
    description: str,
    *,
    require_positive: bool = True,
) -> float:
    total: Tensor | None = None
    count = 0
    for name, parameter in parameters:
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"{description} parameter {name!r} has no gradient")
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError(
                f"{description} parameter {name!r} gradient is non-finite"
            )
        contribution = gradient.double().square().sum()
        total = contribution if total is None else total + contribution
        count += 1
    if total is None or count == 0:
        raise RuntimeError(f"{description} has no gradients")
    numeric = float(total.item())
    if not math.isfinite(numeric):
        raise RuntimeError(f"{description} gradients are non-finite")
    if require_positive and numeric <= 0.0:
        raise RuntimeError(f"{description} gradients are zero")
    return math.sqrt(numeric)


def assert_finite_tensor_tree(value: Any, description: str) -> None:
    tensors: list[Tensor] = []

    def collect(item: Any) -> None:
        if isinstance(item, Tensor):
            if item.is_floating_point() or item.is_complex():
                tensors.append(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    if tensors and not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise RuntimeError(f"{description} contains NaN or Inf")


def assert_module_parameters_finite(
    modules: Sequence[nn.Module], description: str,
) -> None:
    parameters = [
        parameter
        for module in modules
        for parameter in module.parameters()
    ]
    if not parameters:
        raise RuntimeError(f"{description} has no parameters")
    device = parameters[0].device
    finite = torch.ones((), dtype=torch.bool, device=device)
    for parameter in parameters:
        if parameter.device != device:
            raise RuntimeError(f"{description} parameters span multiple devices")
        finite.logical_and_(torch.isfinite(parameter).all())
    if not bool(finite.item()):
        raise RuntimeError(f"{description} parameters contain NaN or Inf")


def optimizer_parameter_steps(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    group_steps: list[int] = []
    for group in optimizer.param_groups:
        steps: set[int] = set()
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not isinstance(state, Mapping) or "step" not in state:
                raise RuntimeError("AdamW parameter has no step state")
            value = state["step"]
            steps.add(int(value.item()) if isinstance(value, Tensor) else int(value))
        if len(steps) != 1:
            raise RuntimeError("AdamW steps disagree within a parameter group")
        group_steps.append(steps.pop())
    if len(group_steps) != 2:
        raise RuntimeError("AdamW must have exactly two parameter groups")
    return group_steps[0], group_steps[1]


def require_successful_scaler_update(
    *, hook_before: int, hook_after: int, scale_before: float, scale_after: float,
) -> None:
    if (
        hook_after != hook_before + 1
        or not math.isfinite(scale_before)
        or not math.isfinite(scale_after)
        or scale_before <= 0.0
        or scale_after <= 0.0
        or scale_after < scale_before
    ):
        raise RuntimeError("GradScaler skipped or invalidated the optimizer update")


def classify_baseline_comparison(
    epoch_eer: float,
    baseline_eer: float = BASELINE_INTERPOLATED_EER,
    tolerance: float = QUALITY_TOLERANCE,
) -> dict[str, float | str]:
    if (
        not math.isfinite(epoch_eer)
        or not math.isfinite(baseline_eer)
        or not math.isfinite(tolerance)
        or baseline_eer <= 0.0
        or tolerance < 0.0
    ):
        raise ValueError("baseline comparison inputs are invalid")
    signed = epoch_eer - baseline_eer
    if signed < -tolerance:
        classification = "IMPROVED"
    elif signed > tolerance:
        classification = "REGRESSED"
    else:
        classification = "UNCHANGED"
    return {
        "epoch_0_interpolated_eer": epoch_eer,
        "pretrained_interpolated_eer": baseline_eer,
        "signed_eer_difference": signed,
        "absolute_eer_difference": abs(signed),
        "percentage_point_difference": signed * 100.0,
        "relative_eer_change": signed / baseline_eer,
        "numerical_tolerance": tolerance,
        "classification": classification,
    }


CHECKPOINT_KEYS = {
    "schema_name",
    "schema_version",
    "model_source",
    "upstream_bindings",
    "sampler_binding",
    "feature_shape",
    "embedding_dimension",
    "class_count",
    "label_range",
    "training_configuration",
    "aam_configuration",
    "optimizer_configuration",
    "mixed_precision_configuration",
    "batchnorm_policy",
    "epoch",
    "completed_batch_position",
    "global_optimizer_step",
    "next_cursor",
    "embedding_model_state",
    "mean_var_norm_state",
    "aam_state",
    "optimizer_state",
    "grad_scaler_state",
    "rng_state",
    "dataloader_generator_state",
    "batchnorm_reference_state",
    "loss_summary",
    "creation_reason",
    "validation_selection",
    "stop_reason",
    "epoch_1_started",
}


def validate_checkpoint_v2(checkpoint: Mapping[str, Any]) -> None:
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != CHECKPOINT_KEYS:
        raise ValueError("v2 checkpoint has incorrect top-level keys")
    if (
        checkpoint["schema_name"] != SCHEMA_NAME
        or checkpoint["schema_version"] != SCHEMA_VERSION
        or checkpoint["model_source"] != PRETRAINED_MODEL_ID
    ):
        raise ValueError("v2 checkpoint schema/model source is invalid")
    validate_upstream_bindings(checkpoint["upstream_bindings"])
    validate_sampler_binding(checkpoint["sampler_binding"])
    if (
        checkpoint["feature_shape"] != FEATURE_SHAPE
        or checkpoint["embedding_dimension"] != EMBEDDING_DIM
        or checkpoint["class_count"] != NUM_CLASSES
        or checkpoint["label_range"] != [0, NUM_CLASSES - 1]
        or checkpoint["training_configuration"] != TRAINING_CONFIGURATION
        or checkpoint["aam_configuration"] != AAM_CONFIGURATION
        or checkpoint["optimizer_configuration"] != OPTIMIZER_CONFIGURATION
        or checkpoint["mixed_precision_configuration"]
        != MIXED_PRECISION_CONFIGURATION
        or checkpoint["batchnorm_policy"] != BATCHNORM_POLICY
        or checkpoint["epoch"] != 0
        or checkpoint["epoch_1_started"] is not False
    ):
        raise ValueError("v2 checkpoint configuration changed")
    step = checkpoint["global_optimizer_step"]
    if not isinstance(step, int) or not 1 <= step <= NUM_LOGICAL_BATCHES:
        raise ValueError("v2 checkpoint optimizer step is invalid")
    if checkpoint["completed_batch_position"] != step - 1:
        raise ValueError("v2 checkpoint completed batch position is invalid")
    if checkpoint["next_cursor"] != resume_cursor(step):
        raise ValueError("v2 checkpoint next cursor is invalid")
    reason = checkpoint["creation_reason"]
    if reason == "rolling":
        if step not in ROLLING_CHECKPOINT_STEPS:
            raise ValueError("rolling checkpoint step is invalid")
    elif reason not in {"epoch_complete", "best_validation"} or step != NUM_LOGICAL_BATCHES:
        raise ValueError("completed checkpoint reason/step is invalid")
    for key in (
        "embedding_model_state",
        "mean_var_norm_state",
        "aam_state",
        "optimizer_state",
        "grad_scaler_state",
        "rng_state",
        "batchnorm_reference_state",
    ):
        if not isinstance(checkpoint[key], Mapping):
            raise ValueError(f"{key} must be a mapping")
    weight = checkpoint["aam_state"].get("weight")
    if not isinstance(weight, Tensor) or tuple(weight.shape) != (
        NUM_CLASSES,
        EMBEDDING_DIM,
    ):
        raise ValueError("v2 AAM checkpoint weight shape is invalid")
    optimizer = checkpoint["optimizer_state"]
    if not isinstance(optimizer.get("state"), Mapping) or not optimizer["state"]:
        raise ValueError("v2 AdamW state is empty")
    optimizer_steps = set()
    for state in optimizer["state"].values():
        value = state.get("step") if isinstance(state, Mapping) else None
        if isinstance(value, Tensor):
            value = int(value.item())
        elif isinstance(value, (int, float)):
            value = int(value)
        else:
            raise ValueError("v2 AdamW step state is malformed")
        optimizer_steps.add(value)
    if optimizer_steps != {step}:
        raise ValueError("v2 AdamW steps do not match the global step")
    rng = checkpoint["rng_state"]
    if set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("v2 checkpoint RNG state is malformed")
    generator_state = checkpoint["dataloader_generator_state"]
    if not isinstance(generator_state, Tensor) or generator_state.dtype != torch.uint8:
        raise ValueError("v2 DataLoader generator state is malformed")
    summary = checkpoint["loss_summary"]
    if (
        not isinstance(summary, Mapping)
        or set(summary)
        != {"count", "first", "final", "mean", "minimum", "maximum"}
        or summary["count"] != step
    ):
        raise ValueError("v2 checkpoint loss summary is invalid")
    selection = checkpoint["validation_selection"]
    if reason == "best_validation":
        if (
            not isinstance(selection, Mapping)
            or selection.get("selected_checkpoint") != "epoch_000.pt"
            or selection.get("classification")
            not in {"IMPROVED", "UNCHANGED", "REGRESSED"}
        ):
            raise ValueError("v2 best-checkpoint validation selection is invalid")
    elif selection is not None:
        raise ValueError("non-best v2 checkpoint has validation selection")
    expected_stop = (
        "epoch_0_complete" if step == NUM_LOGICAL_BATCHES else "in_progress_epoch_0"
    )
    if checkpoint["stop_reason"] != expected_stop:
        raise ValueError("v2 checkpoint stop reason is invalid")


def atomic_save_checkpoint(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    validate_checkpoint_v2(checkpoint)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(dict(checkpoint), temporary)
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_checkpoint_v2(checkpoint)
    return checkpoint


def checkpoint_roundtrip_exact(
    checkpoint: Mapping[str, Any],
    *,
    embedding_model: nn.Module,
    mean_var_norm: nn.Module,
    aam: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    dataloader_generator: torch.Generator,
) -> dict[str, bool]:
    checks = {
        "embedding_model_state_exact": values_exactly_equal(
            embedding_model.state_dict(), checkpoint["embedding_model_state"]
        ),
        "mean_var_norm_state_exact": values_exactly_equal(
            mean_var_norm.state_dict(), checkpoint["mean_var_norm_state"]
        ),
        "aam_state_exact": values_exactly_equal(
            aam.state_dict(), checkpoint["aam_state"]
        ),
        "optimizer_state_exact": values_exactly_equal(
            optimizer.state_dict(), checkpoint["optimizer_state"]
        ),
        "grad_scaler_state_exact": values_exactly_equal(
            scaler.state_dict(), checkpoint["grad_scaler_state"]
        ),
        "dataloader_generator_state_exact": torch.equal(
            dataloader_generator.get_state().cpu(),
            checkpoint["dataloader_generator_state"].cpu(),
        ),
        "schema_and_cursor_valid": True,
    }
    validate_checkpoint_v2(checkpoint)
    if not all(checks.values()):
        raise RuntimeError(f"fresh-object checkpoint roundtrip failed: {checks}")
    return checks


def default_rng_state() -> dict[str, Any]:
    return capture_rng_state()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
