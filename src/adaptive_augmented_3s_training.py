"""Production ECAPA/AAM fine-tuning for the adaptive_augmented_3s_v1 package."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.aam_training import (
    AAMSoftmax,
    apply_batchnorm_policy,
    assert_batchnorm_running_state_exact,
    batchnorm_running_state,
    build_adamw_optimizer,
    capture_rng_state,
    iter_microbatches,
    restore_rng_state,
    round_robin_reorder,
    to_cpu_tree,
    values_exactly_equal,
)
from src.adaptive_augmented_3s_training_input import (
    PRODUCTION_CACHE_IDENTITY,
    create_train_input,
    create_validation_input,
)
from src.adaptive_augmented_3s_verification import (
    TRIAL_FIELDS,
    ValidationTrial,
    read_validation_manifest,
    sha256_file,
    trials_csv_bytes,
    validate_validation_trials,
)
from src.speechbrain_frontend import SpeechBrainECAPAFrontend
from src.verification_baseline import score_trials
from src.verification_metrics import calculate_eer


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "outputs/fbank_cache_adaptive_augmented_3s_v1"
CONFIG_PATH = ROOT / "configs/adaptive_augmented_3s_training_v1.json"
TRIAL_PATH = ROOT / "manifests/verification/adaptive_augmented_3s_v1_validation_trials.csv"
TRIAL_IDENTITY_PATH = (
    ROOT / "manifests/verification/adaptive_augmented_3s_v1_validation_trials_identity.json"
)
VALIDATION_MANIFEST_PATH = ROOT / "manifests/adaptive_augmented_3s_v1_validation_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/ecapa_aam_adaptive_augmented_3s_v1"
CHECKPOINT_SCHEMA = "adaptive_augmented_3s_ecapa_aam_training"
CHECKPOINT_VERSION = 1
AMP_OVERFLOW_RETRY_LIMIT = 1


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def configuration() -> tuple[dict[str, Any], str]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    required = {
        "schema_version", "training_version", "cache", "sampler", "model", "aam",
        "optimizer", "amp", "scheduler", "early_stopping", "validation",
        "checkpoint_interval_updates",
    }
    if set(value) != required or value["schema_version"] != 1:
        raise ValueError("production training configuration schema is invalid")
    if value["training_version"] != "adaptive_augmented_3s_v1":
        raise ValueError("production training configuration version is invalid")
    sampler, scheduler = value["sampler"], value["scheduler"]
    if (
        sampler != {
            "name": "HybridShardAwareSpeakerBatchSampler", "seed": 20260729,
            "speakers_per_batch": 16, "samples_per_speaker": 2,
            "active_shard_window": 8, "batches_per_epoch": 1520,
        }
        or scheduler["max_epochs"] != 10
        or scheduler["steps_per_epoch"] != 1520
        or scheduler["total_steps"] != 15200
        or scheduler["warmup_steps"] != 0
        or value["cache"]["identity_sha256"] != PRODUCTION_CACHE_IDENTITY
        or value["cache"]["train_rows"] != 48640
        or value["cache"]["validation_rows"] != 6076
        or value["cache"]["feature_shape"] != [301, 80]
        or value["model"] != {
            "source": "speechbrain/spkrec-ecapa-voxceleb", "embedding_dim": 192,
        }
        or value["aam"] != {"num_classes": 488, "margin": 0.2, "scale": 30.0}
        or value["optimizer"] != {
            "name": "AdamW", "ecapa_lr": 1e-5, "aam_lr": 1e-3,
            "weight_decay": 1e-4,
        }
        or value["amp"] != {
            "enabled": True, "ecapa_dtype": "float16", "grad_scaler_initial_scale": 128.0,
        }
        or value["validation"]["trial_count"] != 20000
        or value["early_stopping"] != {"patience": 2, "min_improvement_eer": 0.0001}
    ):
        raise ValueError("production training configuration disagrees with the approved contract")
    return value, hashlib.sha256(canonical_json(value)).hexdigest()


class CosineScheduler:
    """Cosine schedule whose state is an explicit count of successful updates."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: Mapping[str, Any]) -> None:
        self.optimizer = optimizer
        self.total_steps = int(config["total_steps"])
        self.minimum_factor = float(config["minimum_factor"])
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        self.completed_steps = 0
        if self.total_steps < 1 or not 0.0 <= self.minimum_factor <= 1.0:
            raise ValueError("invalid cosine scheduler configuration")
        self._apply()

    def factor(self) -> float:
        progress = min(1.0, self.completed_steps / self.total_steps)
        return self.minimum_factor + (1.0 - self.minimum_factor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def _apply(self) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * self.factor()

    def step(self) -> None:
        if self.completed_steps >= self.total_steps:
            raise RuntimeError("cosine scheduler cannot exceed its configured horizon")
        self.completed_steps += 1
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "minimum_factor": self.minimum_factor,
            "base_lrs": list(self.base_lrs),
            "completed_steps": self.completed_steps,
            "last_factor": self.factor(),
            "last_lrs": [float(group["lr"]) for group in self.optimizer.param_groups],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {
            "total_steps", "minimum_factor", "base_lrs", "completed_steps",
            "last_factor", "last_lrs",
        }:
            raise ValueError("scheduler checkpoint schema is invalid")
        if (
            state["total_steps"] != self.total_steps
            or float(state["minimum_factor"]) != self.minimum_factor
            or tuple(state["base_lrs"]) != self.base_lrs
            or not isinstance(state["completed_steps"], int)
            or not 0 <= state["completed_steps"] <= self.total_steps
        ):
            raise ValueError("scheduler checkpoint does not match the production schedule")
        self.completed_steps = state["completed_steps"]
        expected_factor = self.factor()
        expected_lrs = tuple(base * expected_factor for base in self.base_lrs)
        if not (
            math.isclose(float(state["last_factor"]), expected_factor, abs_tol=1e-15)
            and all(
                math.isclose(float(actual), expected, abs_tol=1e-15)
                for actual, expected in zip(state["last_lrs"], expected_lrs)
            )
        ):
            raise ValueError("scheduler checkpoint learning rates are inconsistent")
        self._apply()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_binding() -> dict[str, str]:
    root = ROOT / "pretrained_models/spkrec-ecapa-voxceleb"
    files = {name: root / name for name in ("hyperparams.yaml", "embedding_model.ckpt")}
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError("required local pretrained ECAPA files are missing")
    return {
        "source": "speechbrain/spkrec-ecapa-voxceleb",
        **{name: sha256_file(path) for name, path in files.items()},
    }


def validate_bound_files(bindings: Mapping[str, Any], expected_names: set[str]) -> None:
    if set(bindings) != expected_names:
        raise ValueError("approved identity binding names changed")
    for name, item in bindings.items():
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError(f"malformed identity binding: {name}")
        relative = item["path"]
        if not isinstance(relative, str) or not isinstance(item["sha256"], str):
            raise ValueError(f"malformed identity binding values: {name}")
        pure = PurePosixPath(relative)
        if (
            not relative or pure.is_absolute() or ".." in pure.parts or "\\" in relative
            or any("final" in part.lower() or part.lower() == "test" for part in pure.parts)
        ):
            raise ValueError(f"identity binding is not train/validation-safe: {name}")
        if sha256_file(ROOT.joinpath(*pure.parts)) != item["sha256"]:
            raise ValueError(f"identity binding changed on disk: {name}")


def input_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    cache_identity_path = CACHE_ROOT / "fbank_cache_identity_adaptive_augmented_3s_v1.json"
    cache_identity = json.loads(cache_identity_path.read_text(encoding="utf-8"))
    if (
        cache_identity.get("identity_sha256") != config["cache"]["identity_sha256"]
        or cache_identity.get("config_sha256") != config["cache"]["config_sha256"]
        or sha256_file(CACHE_ROOT / cache_identity.get("config_path", ""))
        != config["cache"]["config_sha256"]
        or cache_identity.get("included_splits") != ["train", "validation"]
        or cache_identity.get("final_test_cache_absent") is not True
    ):
        raise ValueError("cache identity is not the approved train/validation cache")
    validate_bound_files(
        cache_identity.get("input_bindings", {}),
        {"dataset_identity", "speaker_to_label", "split_identity", "train_manifest", "validation_manifest"},
    )
    trial_identity = json.loads(TRIAL_IDENTITY_PATH.read_text(encoding="utf-8"))
    if (
        trial_identity.get("identity_sha256") != config["validation"]["trial_identity_sha256"]
        or trial_identity.get("trial_csv_sha256") != config["validation"]["trial_csv_sha256"]
        or trial_identity.get("final_test_access") is not False
        or trial_identity.get("trial_counts", {}).get("total") != config["validation"]["trial_count"]
        or sha256_file(TRIAL_PATH) != config["validation"]["trial_csv_sha256"]
    ):
        raise ValueError("validation trial identity is not approved")
    validate_bound_files(
        trial_identity.get("input_bindings", {}),
        {"dataset_identity", "split_identity", "validation_manifest"},
    )
    return {
        "cache_identity_sha256": cache_identity["identity_sha256"],
        "cache_config_sha256": cache_identity["config_sha256"],
        "validation_trial_identity_sha256": trial_identity["identity_sha256"],
        "validation_trial_csv_sha256": trial_identity["trial_csv_sha256"],
        "model": model_binding(),
    }


def read_trials(config: Mapping[str, Any]) -> tuple[ValidationTrial, ...]:
    rows = read_validation_manifest(VALIDATION_MANIFEST_PATH)
    if len(rows) != config["cache"]["validation_rows"]:
        raise ValueError("validation manifest row count is not approved")
    with TRIAL_PATH.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRIAL_FIELDS:
            raise ValueError("validation trial CSV schema is invalid")
        trials = tuple(
            ValidationTrial(
                raw["trial_id"], raw["left_audio_path"], raw["right_audio_path"],
                raw["left_speaker_id"], raw["right_speaker_id"], int(raw["target"]),
            )
            for raw in reader
        )
    validate_validation_trials(trials, rows)
    if len(trials) != config["validation"]["trial_count"] or trials_csv_bytes(trials) != TRIAL_PATH.read_bytes():
        raise ValueError("validation trial CSV is not the frozen approved package")
    return trials


def create_production_train_input(config: Mapping[str, Any]):
    sampler = config["sampler"]
    return create_train_input(
        CACHE_ROOT,
        speakers_per_batch=sampler["speakers_per_batch"],
        samples_per_speaker=sampler["samples_per_speaker"],
        active_shard_window=sampler["active_shard_window"], seed=sampler["seed"],
    )


def create_objects(
    device: torch.device, config: Mapping[str, Any]
) -> tuple[dict[str, Any], CosineScheduler]:
    frontend = SpeechBrainECAPAFrontend(device="cpu")
    mean_var_norm = frontend.classifier.mods.mean_var_norm
    embedding_model = frontend.classifier.mods.embedding_model
    classifier_parameter_ids = {
        id(parameter) for parameter in frontend.classifier.mods.classifier.parameters()
    }
    del frontend
    mean_var_norm.eval().to(device)
    for parameter in mean_var_norm.parameters():
        parameter.requires_grad_(False)
    embedding_model.to(device)
    apply_batchnorm_policy(embedding_model)
    aam = AAMSoftmax(
        embedding_dim=config["model"]["embedding_dim"],
        num_classes=config["aam"]["num_classes"], margin=config["aam"]["margin"],
        scale=config["aam"]["scale"], seed=config["sampler"]["seed"],
    ).to(device)
    optimizer = build_adamw_optimizer(
        embedding_model, aam, embedding_lr=config["optimizer"]["ecapa_lr"],
        classifier_lr=config["optimizer"]["aam_lr"], weight_decay=config["optimizer"]["weight_decay"],
    )
    if classifier_parameter_ids & {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }:
        raise AssertionError("pretrained VoxCeleb classifier entered the optimizer")
    scaler = torch.cuda.amp.GradScaler(
        enabled=config["amp"]["enabled"], init_scale=config["amp"]["grad_scaler_initial_scale"],
    )
    if not scaler.is_enabled() or any(parameter.requires_grad for parameter in mean_var_norm.parameters()):
        raise AssertionError("production AMP or normalization policy is invalid")
    return {
        "mean_var_norm": mean_var_norm, "embedding_model": embedding_model,
        "aam": aam, "optimizer": optimizer, "scaler": scaler,
        "batchnorm_reference_state": batchnorm_running_state(embedding_model),
    }, CosineScheduler(optimizer, config["scheduler"])


def validate_logical_batch(batch: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    sampler = config["sampler"]
    logical = round_robin_reorder(
        batch, speakers_per_batch=sampler["speakers_per_batch"],
        samples_per_speaker=sampler["samples_per_speaker"],
    )
    if (
        tuple(logical["fbank"].shape) != (32, 301, 80)
        or logical["fbank"].dtype != torch.float32
        or logical["speaker_label"].dtype != torch.long
        or len(set(logical["dataset_index"].tolist())) != 32
        or logical["final_split"] != ["train"] * 32
        or bool(((logical["speaker_label"] < 0) | (logical["speaker_label"] >= 488)).any())
    ):
        raise ValueError("logical train batch violates the production contract")
    if len(list(iter_microbatches(logical, 4))) != 8:
        raise AssertionError("logical batch does not produce eight physical microbatches")
    return logical


def optimizer_update(
    objects: Mapping[str, Any], logical_batch: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, float | int]:
    embedding_model, mean_var_norm = objects["embedding_model"], objects["mean_var_norm"]
    aam, optimizer, scaler = objects["aam"], objects["optimizer"], objects["scaler"]
    device = next(embedding_model.parameters()).device
    apply_batchnorm_policy(embedding_model)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    backwards = 0
    for microbatch in iter_microbatches(logical_batch, 4):
        features = microbatch["fbank"].to(device=device, dtype=torch.float32)
        labels = microbatch["speaker_label"].to(device=device, dtype=torch.long)
        lengths = torch.ones(4, device=device, dtype=torch.float32)
        with torch.no_grad():
            normalized = mean_var_norm(features, lengths)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            embedding = embedding_model(normalized, lengths).squeeze(1)
        with torch.cuda.amp.autocast(enabled=False):
            logits = aam(embedding.float(), labels)
            loss = F.cross_entropy(logits.float(), labels, reduction="sum") / 32
        if tuple(logits.shape) != (4, 488) or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("invalid ECAPA/AAM microbatch output")
        scaler.scale(loss).backward()
        total_loss += float(loss.detach())
        backwards += 1
    if backwards != 8:
        raise AssertionError("one optimizer update must contain eight backward passes")
    scaler.unscale_(optimizer)
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    try:
        found_inf_per_device = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
    except (AttributeError, KeyError) as error:
        raise RuntimeError("GradScaler did not expose optimizer overflow state") from error
    optimizer_updated = not any(
        bool(found_inf.detach().item()) for found_inf in found_inf_per_device.values()
    )
    scaler.update()
    scale_after = float(scaler.get_scale())
    if not optimizer_updated:
        if scale_after >= scale_before:
            raise RuntimeError("GradScaler reported overflow without reducing its scale")
        optimizer.zero_grad(set_to_none=True)
    assert_batchnorm_running_state_exact(embedding_model, objects["batchnorm_reference_state"])
    return {
        "loss": total_loss, "backward_calls": backwards, "scaler_scale": scale_after,
        "optimizer_updated": int(optimizer_updated),
    }


def validation_eer(
    objects: Mapping[str, Any], config: Mapping[str, Any], trials: Sequence[ValidationTrial]
) -> dict[str, float]:
    _, loader = create_validation_input(CACHE_ROOT, batch_size=config["validation"]["batch_size"])
    mean_var_norm, embedding_model = objects["mean_var_norm"], objects["embedding_model"]
    device = next(embedding_model.parameters()).device
    mean_var_norm.eval()
    embedding_model.eval()
    embeddings: list[torch.Tensor] = []
    paths: list[str] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                features = batch["fbank"].to(device=device, dtype=torch.float32)
                lengths = torch.ones(features.shape[0], device=device, dtype=torch.float32)
                value = embedding_model(mean_var_norm(features, lengths), lengths).squeeze(1).cpu()
                if tuple(value.shape) != (features.shape[0], 192) or not bool(torch.isfinite(value).all()):
                    raise RuntimeError("invalid validation embedding")
                embeddings.append(value)
                paths.extend(batch["relative_audio_path"])
    finally:
        apply_batchnorm_policy(embedding_model)
    assert_batchnorm_running_state_exact(embedding_model, objects["batchnorm_reference_state"])
    joined = torch.cat(embeddings)
    if tuple(joined.shape) != (config["cache"]["validation_rows"], 192):
        raise RuntimeError("validation cache row count changed")
    scores = score_trials(joined, paths, trials)
    result = calculate_eer(scores.tolist(), [trial.target for trial in trials])
    return {
        "eer": float(result.interpolated_eer),
        "empirical_threshold": float(result.empirical_threshold),
        "score_count": int(scores.numel()),
    }


def checkpoint_payload(
    objects: Mapping[str, Any], scheduler: CosineScheduler, *, config_hash: str,
    binding: Mapping[str, Any], epoch: int, next_epoch: int, next_position: int,
    best_eer: float | None, best_epoch: int | None, patience_counter: int,
    completed_epochs: Sequence[int], reason: str,
) -> dict[str, Any]:
    if not 0 <= next_epoch <= 10 or not 0 <= next_position < 1520:
        raise ValueError("checkpoint cursor is outside the production schedule")
    return {
        "schema": CHECKPOINT_SCHEMA, "version": CHECKPOINT_VERSION,
        "configuration_sha256": config_hash, "input_binding": dict(binding),
        "embedding_model_state_dict": to_cpu_tree(objects["embedding_model"].state_dict()),
        "mean_var_norm_state_dict": to_cpu_tree(objects["mean_var_norm"].state_dict()),
        "aam_state_dict": to_cpu_tree(objects["aam"].state_dict()),
        "optimizer_state_dict": to_cpu_tree(objects["optimizer"].state_dict()),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": to_cpu_tree(objects["scaler"].state_dict()),
        "epoch": epoch, "next_epoch": next_epoch,
        "next_logical_batch_position": next_position,
        "sampler_state": {"seed": 20260729, "epoch": next_epoch},
        "global_optimizer_steps": scheduler.completed_steps,
        "best_validation_eer": best_eer, "best_epoch": best_epoch,
        "early_stopping": {"patience_counter": patience_counter, "patience": 2, "min_improvement_eer": 0.0001},
        "completed_epochs": list(completed_epochs),
        "batchnorm_reference_state": to_cpu_tree(objects["batchnorm_reference_state"]),
        "rng_state": capture_rng_state(), "reason": reason,
        "final_test_access_count": 0,
    }


def validate_checkpoint(checkpoint: Mapping[str, Any], config_hash: str, binding: Mapping[str, Any]) -> None:
    required = {
        "schema", "version", "configuration_sha256", "input_binding",
        "embedding_model_state_dict", "mean_var_norm_state_dict", "aam_state_dict",
        "optimizer_state_dict", "scheduler_state_dict", "grad_scaler_state_dict",
        "epoch", "next_epoch", "next_logical_batch_position", "sampler_state",
        "global_optimizer_steps", "best_validation_eer", "best_epoch", "early_stopping",
        "completed_epochs", "batchnorm_reference_state", "rng_state", "reason",
        "final_test_access_count",
    }
    if (
        set(checkpoint) != required or checkpoint["schema"] != CHECKPOINT_SCHEMA
        or checkpoint["version"] != CHECKPOINT_VERSION
        or checkpoint["configuration_sha256"] != config_hash
        or checkpoint["input_binding"] != binding
        or checkpoint["final_test_access_count"] != 0
    ):
        raise ValueError("checkpoint does not belong to this production experiment")
    if (
        checkpoint["sampler_state"] != {"seed": 20260729, "epoch": checkpoint["next_epoch"]}
        or checkpoint["global_optimizer_steps"] != checkpoint["scheduler_state_dict"]["completed_steps"]
        or checkpoint["early_stopping"] != {
            "patience_counter": checkpoint["early_stopping"]["patience_counter"],
            "patience": 2, "min_improvement_eer": 0.0001,
        }
    ):
        raise ValueError("checkpoint sampler, scheduler, or early-stopping state is invalid")
    expected_steps = checkpoint["next_epoch"] * 1520 + checkpoint["next_logical_batch_position"]
    if checkpoint["global_optimizer_steps"] != expected_steps:
        raise ValueError("checkpoint cursor does not reproduce its scheduler position")


def atomic_save(checkpoint: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(checkpoint), temporary)
    os.replace(temporary, path)


def assert_optimizer_step(optimizer: torch.optim.Optimizer, expected: int) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            step = state.get("step") if isinstance(state, Mapping) else None
            numeric = int(step.item()) if isinstance(step, torch.Tensor) else step
            if numeric != expected:
                raise RuntimeError("optimizer checkpoint step disagrees with scheduler position")


def load_checkpoint(
    path: Path, device: torch.device, config: Mapping[str, Any], config_hash: str,
    binding: Mapping[str, Any],
) -> tuple[dict[str, Any], CosineScheduler, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, config_hash, binding)
    objects, scheduler = create_objects(device, config)
    objects["embedding_model"].load_state_dict(checkpoint["embedding_model_state_dict"], strict=True)
    objects["mean_var_norm"].load_state_dict(checkpoint["mean_var_norm_state_dict"], strict=True)
    objects["aam"].load_state_dict(checkpoint["aam_state_dict"], strict=True)
    objects["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    objects["scaler"].load_state_dict(checkpoint["grad_scaler_state_dict"])
    assert_optimizer_step(objects["optimizer"], scheduler.completed_steps)
    if not values_exactly_equal(objects["scaler"].state_dict(), checkpoint["grad_scaler_state_dict"]):
        raise RuntimeError("GradScaler checkpoint state did not restore exactly")
    objects["batchnorm_reference_state"] = checkpoint["batchnorm_reference_state"]
    apply_batchnorm_policy(objects["embedding_model"])
    assert_batchnorm_running_state_exact(objects["embedding_model"], objects["batchnorm_reference_state"])
    restore_rng_state(checkpoint["rng_state"])
    if not values_exactly_equal(capture_rng_state(), checkpoint["rng_state"]):
        raise RuntimeError("checkpoint RNG state did not restore exactly")
    return objects, scheduler, {
        "next_epoch": checkpoint["next_epoch"],
        "next_logical_batch_position": checkpoint["next_logical_batch_position"],
        "best_validation_eer": checkpoint["best_validation_eer"],
        "best_epoch": checkpoint["best_epoch"],
        "early_stopping": checkpoint["early_stopping"],
        "completed_epochs": checkpoint["completed_epochs"],
    }


def save_state(
    output_dir: Path, name: str, objects: Mapping[str, Any], scheduler: CosineScheduler,
    *, config_hash: str, binding: Mapping[str, Any], epoch: int, next_epoch: int,
    next_position: int, best_eer: float | None, best_epoch: int | None,
    patience_counter: int, completed_epochs: Sequence[int], reason: str,
) -> Path:
    path = output_dir / name
    atomic_save(checkpoint_payload(
        objects, scheduler, config_hash=config_hash, binding=binding, epoch=epoch,
        next_epoch=next_epoch, next_position=next_position, best_eer=best_eer,
        best_epoch=best_epoch, patience_counter=patience_counter,
        completed_epochs=completed_epochs, reason=reason,
    ), path)
    return path


def run_training(
    *, output_dir: Path, device: str = "cuda:0", resume: Path | None = None,
    max_updates: int | None = None, validate_after_partial: bool = False,
) -> dict[str, Any]:
    config, config_hash = configuration()
    binding = input_binding(config)
    trials = read_trials(config)
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production training requires an available CUDA device")
    if resume is None:
        if (output_dir / "last.pt").exists():
            raise FileExistsError("output already has last.pt; use --resume to continue it")
        seed_everything(config["sampler"]["seed"])
        objects, scheduler = create_objects(resolved_device, config)
        state = {"next_epoch": 0, "next_logical_batch_position": 0,
                 "best_validation_eer": None, "best_epoch": None,
                 "early_stopping": {"patience_counter": 0}, "completed_epochs": []}
    else:
        objects, scheduler, state = load_checkpoint(
            resume, resolved_device, config, config_hash, binding
        )
    _, sampler, _ = create_production_train_input(config)
    updates = 0
    for epoch in range(state["next_epoch"], config["scheduler"]["max_epochs"]):
        sampler.set_epoch(epoch)
        planned = list(sampler)
        if len(planned) != config["sampler"]["batches_per_epoch"]:
            raise RuntimeError("sampler epoch length differs from the approved schedule")
        start = state["next_logical_batch_position"] if epoch == state["next_epoch"] else 0
        _, _, loader = create_production_train_input(config)
        loader = torch.utils.data.DataLoader(
            loader.dataset, batch_sampler=planned[start:], num_workers=0,
            collate_fn=loader.collate_fn,
        )
        for position, batch in enumerate(loader, start=start):
            logical = validate_logical_batch(batch, config)
            overflow_events = 0
            while True:
                update = optimizer_update(objects, logical, config)
                if update["optimizer_updated"]:
                    break
                overflow_events += 1
                if overflow_events > AMP_OVERFLOW_RETRY_LIMIT:
                    save_state(output_dir, "last.pt", objects, scheduler, config_hash=config_hash,
                        binding=binding, epoch=epoch, next_epoch=epoch, next_position=position,
                        best_eer=state["best_validation_eer"], best_epoch=state["best_epoch"],
                        patience_counter=state["early_stopping"]["patience_counter"],
                        completed_epochs=state["completed_epochs"], reason="amp_overflow_retry_exhausted")
                    raise RuntimeError(
                        "AMP overflow retry limit exceeded at "
                        f"epoch={epoch}, logical_batch={position}, "
                        f"scaler_scale={update['scaler_scale']}, "
                        f"retry_count={AMP_OVERFLOW_RETRY_LIMIT}"
                    )
            scheduler.step()
            updates += 1
            next_epoch = epoch if position + 1 < len(planned) else epoch + 1
            next_position = position + 1 if next_epoch == epoch else 0
            state.update({"next_epoch": next_epoch, "next_logical_batch_position": next_position})
            if scheduler.completed_steps % config["checkpoint_interval_updates"] == 0:
                save_state(output_dir, "last.pt", objects, scheduler, config_hash=config_hash,
                    binding=binding, epoch=epoch, next_epoch=next_epoch, next_position=next_position,
                    best_eer=state["best_validation_eer"], best_epoch=state["best_epoch"],
                    patience_counter=state["early_stopping"]["patience_counter"],
                    completed_epochs=state["completed_epochs"], reason="rolling")
            if max_updates is not None and updates >= max_updates:
                save_state(output_dir, "last.pt", objects, scheduler, config_hash=config_hash,
                    binding=binding, epoch=epoch, next_epoch=next_epoch, next_position=next_position,
                    best_eer=state["best_validation_eer"], best_epoch=state["best_epoch"],
                    patience_counter=state["early_stopping"]["patience_counter"],
                    completed_epochs=state["completed_epochs"], reason="partial")
                if validate_after_partial:
                    state["partial_validation"] = validation_eer(objects, config, trials)
                return {"updates": updates, "scheduler_steps": scheduler.completed_steps,
                        "last_checkpoint": str(output_dir / "last.pt"), **state, "last_loss": update["loss"]}
        metrics = validation_eer(objects, config, trials)
        state["completed_epochs"] = [*state["completed_epochs"], epoch]
        best_eer = state["best_validation_eer"]
        improved = best_eer is None or metrics["eer"] < best_eer - config["early_stopping"]["min_improvement_eer"]
        if improved:
            state["best_validation_eer"], state["best_epoch"] = metrics["eer"], epoch
            state["early_stopping"]["patience_counter"] = 0
            save_state(output_dir, "best.pt", objects, scheduler, config_hash=config_hash,
                binding=binding, epoch=epoch, next_epoch=epoch + 1, next_position=0,
                best_eer=state["best_validation_eer"], best_epoch=state["best_epoch"],
                patience_counter=0, completed_epochs=state["completed_epochs"], reason="best_validation_eer")
        else:
            state["early_stopping"]["patience_counter"] += 1
        state.update({"next_epoch": epoch + 1, "next_logical_batch_position": 0,
                      "last_validation": metrics})
        save_state(output_dir, "last.pt", objects, scheduler, config_hash=config_hash,
            binding=binding, epoch=epoch, next_epoch=epoch + 1, next_position=0,
            best_eer=state["best_validation_eer"], best_epoch=state["best_epoch"],
            patience_counter=state["early_stopping"]["patience_counter"],
            completed_epochs=state["completed_epochs"], reason="epoch_complete")
        if state["early_stopping"]["patience_counter"] >= config["early_stopping"]["patience"]:
            break
    return {"updates": updates, "scheduler_steps": scheduler.completed_steps,
            "last_checkpoint": str(output_dir / "last.pt"), **state}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run production training")
    parser.add_argument("--dry-run", action="store_true", help="run one update, save, load, then resume one update")
    parser.add_argument("--validate-after-dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.run == args.dry_run:
        parser.error("specify exactly one of --run or --dry-run")
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    if args.dry_run:
        if args.output_dir is None:
            parser.error("--dry-run requires an explicit disposable --output-dir")
        first = run_training(output_dir=output_dir, device=args.device, max_updates=1,
                             validate_after_partial=args.validate_after_dry_run)
        second = run_training(output_dir=output_dir, device=args.device,
                              resume=output_dir / "last.pt", max_updates=1)
        print(json.dumps({"dry_run_first": first, "dry_run_resumed": second}, indent=2, sort_keys=True))
    else:
        print(json.dumps(run_training(output_dir=output_dir, device=args.device, resume=args.resume), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
