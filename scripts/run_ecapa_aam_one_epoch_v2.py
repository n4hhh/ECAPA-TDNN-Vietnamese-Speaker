#!/usr/bin/env python
"""Run the single approved VieSpeaker2.0 ECAPA/AAM epoch zero and validation."""

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
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import speechbrain  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.aam_training import (  # noqa: E402
    AAMSoftmax,
    PRETRAINED_MODEL_ID,
    apply_batchnorm_policy,
    build_adamw_optimizer,
    capture_rng_state,
    cpu_clone_state_dict,
    iter_microbatches,
    parameter_delta,
    restore_rng_state,
    round_robin_reorder,
    to_cpu_tree,
    validate_optimizer_coverage,
    values_exactly_equal,
)
from src.cached_fbank_dataset import (  # noqa: E402
    CachedFbankDataset,
    create_cached_fbank_dataloader,
    create_cached_fbank_training_dataloader,
)
from src.cached_fbank_samplers import (  # noqa: E402
    HybridShardAwareSpeakerBatchSampler,
)
from src.ecapa_one_epoch_pilot import append_jsonl, atomic_json  # noqa: E402
from src.ecapa_one_epoch_v2 import (  # noqa: E402
    AAM_CONFIGURATION,
    ACCUMULATION_STEPS,
    ACTIVE_SHARD_WINDOW,
    BASELINE_INTERPOLATED_EER,
    BATCHNORM_POLICY,
    EMBEDDING_DIM,
    FEATURE_SHAPE,
    LOGGING_STEPS,
    LOGICAL_BATCH_SIZE,
    MIXED_PRECISION_CONFIGURATION,
    NUM_CLASSES,
    NUM_LOGICAL_BATCHES,
    OPTIMIZER_CONFIGURATION,
    PHYSICAL_MICROBATCH_SIZE,
    QUALITY_TOLERANCE,
    ROLLING_CHECKPOINT_STEPS,
    RUNTIME_SCHEMA_NAME,
    RUNTIME_SCHEMA_VERSION,
    SAMPLER_BINDING,
    SAMPLES_PER_SPEAKER,
    SEED,
    SPEAKERS_PER_BATCH,
    TOTAL_SELECTIONS,
    TRAINING_CONFIGURATION,
    UPSTREAM_BINDINGS,
    OneEpochV2Controller,
    assert_batchnorm_reference_exact,
    assert_finite_tensor_tree,
    assert_module_parameters_finite,
    atomic_save_checkpoint,
    batchnorm_reference_state,
    checkpoint_roundtrip_exact,
    classify_baseline_comparison,
    finite_gradient_norm,
    load_checkpoint,
    optimizer_parameter_steps,
    require_successful_scaler_update,
    resume_cursor,
    summarize_values,
    validate_checkpoint_v2,
    validate_logical_batch as validate_v2_logical_batch,
    validate_sampler_binding,
    validate_upstream_bindings,
)
from src.speechbrain_frontend import SpeechBrainECAPAFrontend  # noqa: E402
from src.training_readiness_v2 import (  # noqa: E402
    plan_sha256,
    validate_train_manifest_alignment,
)
from src.verification_baseline import numeric_summary, score_trials  # noqa: E402
from src.verification_metrics import calculate_eer  # noqa: E402
from src.verification_v2 import (  # noqa: E402
    empirical_confusion,
    read_trials_csv,
    read_validation_manifest,
    validate_validation_trials,
)


OUTPUT_RELATIVE = Path("outputs/ecapa_aam_one_epoch_v2")
OUTPUT_DIR = REPO_ROOT / OUTPUT_RELATIVE
PREFLIGHT_PATH = OUTPUT_DIR / "cuda_preflight_v2.json"
TRAIN_LOG_PATH = OUTPUT_DIR / "training_log_epoch_000.jsonl"
LOSS_PATH = OUTPUT_DIR / "logical_losses_epoch_000.json"
LAST_PATH = OUTPUT_DIR / "last.pt"
EPOCH_PATH = OUTPUT_DIR / "epoch_000.pt"
BEST_PATH = OUTPUT_DIR / "best.pt"
EMBEDDING_PATH = OUTPUT_DIR / "validation_embeddings_epoch_000.pt"
SCORE_PATH = OUTPUT_DIR / "validation_scores_epoch_000.pt"
METRICS_PATH = OUTPUT_DIR / "validation_metrics_epoch_000.json"
RUNTIME_PATH = OUTPUT_DIR / "runtime_v2.json"
FAILURE_PATH = OUTPUT_DIR / "failure_v2.json"
REPORT_JSON = REPO_ROOT / "reports/ecapa_aam_one_epoch_v2.json"
REPORT_MD = REPO_ROOT / "reports/ecapa_aam_one_epoch_v2.md"
CACHE_DIR = REPO_ROOT / "outputs/fbank_cache_v2"
TRAIN_MANIFEST = REPO_ROOT / "manifests/portable_v2/train_manifest_v2.csv"
VALIDATION_MANIFEST = REPO_ROOT / "manifests/portable_v2/validation_manifest_v2.csv"
TRIAL_PATH = REPO_ROOT / "manifests/verification_v2/validation_trials_v2.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def ensure_fresh_targets() -> None:
    targets = (
        PREFLIGHT_PATH,
        TRAIN_LOG_PATH,
        LOSS_PATH,
        LAST_PATH,
        EPOCH_PATH,
        BEST_PATH,
        EMBEDDING_PATH,
        SCORE_PATH,
        METRICS_PATH,
        RUNTIME_PATH,
        FAILURE_PATH,
        REPORT_JSON,
        REPORT_MD,
    )
    existing = [relative(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing v2 epoch artifacts: "
            + ", ".join(existing)
        )


def verify_upstream_files() -> dict[str, dict[str, str]]:
    validate_upstream_bindings(UPSTREAM_BINDINGS)
    verified: dict[str, dict[str, str]] = {}
    for name, binding in UPSTREAM_BINDINGS.items():
        path = REPO_ROOT / binding["path"]
        actual = file_sha256(path)
        if actual != binding["sha256"]:
            raise ValueError(
                f"approved identity mismatch for {name}: {actual}"
            )
        verified[name] = dict(binding)
    sampler_config = json.loads(
        (REPO_ROOT / UPSTREAM_BINDINGS["training_sampler_config"]["path"])
        .read_text(encoding="utf-8")
    )
    if (
        sampler_config["sampler_plan"]["sampler_plan_identity_sha256"]
        != SAMPLER_BINDING["combined_plan_sha256"]
        or sampler_config["sampler_plan"]["epoch_plan_sha256"]["0"]
        != SAMPLER_BINDING["epoch_0_plan_sha256"]
        or sampler_config["sampler_plan"]["epoch_plan_sha256"]["1"]
        != SAMPLER_BINDING["epoch_1_plan_sha256"]
        or sampler_config["parameters"]["batches_per_epoch"]
        != NUM_LOGICAL_BATCHES
        or sampler_config["parameters"]["logical_selections_per_epoch"]
        != TOTAL_SELECTIONS
    ):
        raise ValueError("approved sampler config/plan binding mismatch")
    validate_sampler_binding(SAMPLER_BINDING)
    baseline = json.loads(
        (REPO_ROOT / "reports/pretrained_ecapa_validation_baseline_v2.json")
        .read_text(encoding="utf-8")
    )
    if baseline["metrics"]["interpolated_eer"] != BASELINE_INTERPOLATED_EER:
        raise ValueError("approved pretrained baseline EER changed")
    return verified


def create_train_dataset_and_plans() -> tuple[
    CachedFbankDataset, list[list[int]], list[list[int]], dict[str, Any]
]:
    dataset = CachedFbankDataset(
        CACHE_DIR,
        "train",
        max_cached_shards=8,
        validate_finite=False,
    )
    alignment = validate_train_manifest_alignment(dataset, TRAIN_MANIFEST)
    labels = {row.speaker_label for row in dataset.rows}
    if (
        len(dataset) != 95009
        or len({row.speaker_id for row in dataset.rows}) != NUM_CLASSES
        or labels != set(range(NUM_CLASSES))
        or alignment["aligned_rows"] != len(dataset)
    ):
        raise ValueError("approved train Dataset count/label/alignment failed")

    plans: list[list[list[int]]] = []
    reports: list[dict[str, Any]] = []
    for epoch in (0, 1):
        sampler = HybridShardAwareSpeakerBatchSampler(
            dataset,
            speakers_per_batch=SPEAKERS_PER_BATCH,
            samples_per_speaker=SAMPLES_PER_SPEAKER,
            active_shard_window=ACTIVE_SHARD_WINDOW,
            num_batches=NUM_LOGICAL_BATCHES,
            seed=SEED,
        )
        sampler.set_epoch(epoch)
        batches = list(sampler)
        digest = plan_sha256(epoch, batches)
        expected = SAMPLER_BINDING[f"epoch_{epoch}_plan_sha256"]
        if digest != expected:
            raise ValueError(f"approved epoch-{epoch} plan hash mismatch")
        plans.append(batches)
        reports.append({
            "epoch": epoch,
            "plan_sha256": digest,
            "batches": len(batches),
            "logical_selections": sum(map(len, batches)),
            "duplicate_group_rejections": sampler.last_epoch_stats[
                "duplicate_group_rejections"
            ],
        })
    return dataset, plans[0], plans[1], {
        "alignment": alignment,
        "epochs": reports,
    }


def load_trainable_modules(
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    frontend = SpeechBrainECAPAFrontend(device="cpu")
    encoder = frontend.classifier
    module_names = tuple(sorted(encoder.mods.keys()))
    if not {"mean_var_norm", "embedding_model"}.issubset(module_names):
        raise RuntimeError("pretrained checkpoint lacks approved modules")
    mean_var_norm = encoder.mods.mean_var_norm
    embedding_model = encoder.mods.embedding_model
    del encoder, frontend
    gc.collect()
    mean_var_norm.to(device).eval()
    for parameter in mean_var_norm.parameters():
        parameter.requires_grad_(False)
    embedding_model.to(device)
    batchnorm_names = apply_batchnorm_policy(embedding_model)
    if not batchnorm_names:
        raise RuntimeError("embedding model has no BatchNorm modules")
    return mean_var_norm, embedding_model, {
        "checkpoint_module_names": list(module_names),
        "retained_modules": ["mean_var_norm", "embedding_model"],
        "compute_features_retained": False,
        "pretrained_classifier_retained": False,
        "batchnorm_module_count": len(batchnorm_names),
    }


def create_training_objects(device: torch.device) -> dict[str, Any]:
    mean_var_norm, embedding_model, proof = load_trainable_modules(device)
    aam = AAMSoftmax(
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        margin=0.2,
        scale=30.0,
        seed=SEED,
    ).to(device)
    if tuple(aam.weight.shape) != (NUM_CLASSES, EMBEDDING_DIM):
        raise RuntimeError("dynamic v2 AAM weight shape is invalid")
    optimizer = build_adamw_optimizer(
        embedding_model,
        aam,
        embedding_lr=1e-5,
        classifier_lr=1e-3,
        weight_decay=1e-4,
    )
    validate_optimizer_coverage(optimizer, embedding_model, aam)
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128.0)
    if not scaler.is_enabled():
        raise RuntimeError("GradScaler must be enabled")
    if any(parameter.requires_grad for parameter in mean_var_norm.parameters()):
        raise RuntimeError("mean_var_norm must not be optimized")
    return {
        "mean_var_norm": mean_var_norm,
        "embedding_model": embedding_model,
        "aam": aam,
        "optimizer": optimizer,
        "scaler": scaler,
        "module_proof": proof,
    }


def prepare_logical_batch(
    batch: Mapping[str, Any], planned: Sequence[int],
) -> dict[str, Any]:
    reordered = round_robin_reorder(
        batch,
        speakers_per_batch=SPEAKERS_PER_BATCH,
        samples_per_speaker=SAMPLES_PER_SPEAKER,
    )
    validate_v2_logical_batch(reordered, planned)
    microbatches = list(iter_microbatches(reordered, PHYSICAL_MICROBATCH_SIZE))
    if (
        len(microbatches) != ACCUMULATION_STEPS
        or any(
            len(set(microbatch["speaker_id"])) != PHYSICAL_MICROBATCH_SIZE
            for microbatch in microbatches
        )
    ):
        raise ValueError("physical microbatch round-robin structure is invalid")
    return reordered


def run_optimizer_update(
    objects: Mapping[str, Any],
    logical_batch: Mapping[str, Any],
    *,
    expected_step: int,
    optimizer_hook_count: dict[str, int],
    batchnorm_reference: Mapping[str, torch.Tensor],
    stage: dict[str, str],
    require_positive_gradients: bool = True,
) -> dict[str, Any]:
    mean_var_norm = objects["mean_var_norm"]
    embedding_model = objects["embedding_model"]
    aam = objects["aam"]
    optimizer = objects["optimizer"]
    scaler = objects["scaler"]
    device = next(embedding_model.parameters()).device
    apply_batchnorm_policy(embedding_model)
    optimizer.zero_grad(set_to_none=True)
    logical_loss = 0.0
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for number, microbatch in enumerate(
        iter_microbatches(logical_batch, PHYSICAL_MICROBATCH_SIZE),
        start=1,
    ):
        stage["value"] = (
            f"optimizer step {expected_step}: microbatch {number}/"
            f"{ACCUMULATION_STEPS}"
        )
        features = microbatch["fbank"].to(device=device, dtype=torch.float32)
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
            tuple(raw.shape) != (PHYSICAL_MICROBATCH_SIZE, 1, EMBEDDING_DIM)
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
    scaler.unscale_(optimizer)
    ecapa_gradient_norm = finite_gradient_norm(
        list(embedding_model.named_parameters()),
        "ECAPA",
        require_positive=require_positive_gradients,
    )
    aam_gradient_norm = finite_gradient_norm(
        list(aam.named_parameters()),
        "AAM",
        require_positive=require_positive_gradients,
    )
    hook_before = optimizer_hook_count["value"]
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    require_successful_scaler_update(
        hook_before=hook_before,
        hook_after=optimizer_hook_count["value"],
        scale_before=scale_before,
        scale_after=scale_after,
    )
    state_steps = optimizer_parameter_steps(optimizer)
    if state_steps != (expected_step, expected_step):
        raise RuntimeError("AdamW state counters did not advance exactly")
    assert_batchnorm_reference_exact(embedding_model, batchnorm_reference)
    assert_module_parameters_finite(
        [embedding_model, aam], "post-update trainable"
    )
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    if not math.isfinite(logical_loss) or logical_loss <= 0.0:
        raise RuntimeError("logical loss is invalid")
    return {
        "logical_loss": logical_loss,
        "ecapa_gradient_norm": ecapa_gradient_norm,
        "aam_gradient_norm": aam_gradient_norm,
        "grad_scaler_scale": scale_after,
        "duration_seconds": duration,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "cuda_peak_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
    }


def register_optimizer_hook(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, int], Any]:
    counter = {"value": 0}

    def hook(
        _optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        counter["value"] += 1

    return counter, optimizer.register_step_post_hook(hook)


def run_cuda_preflight(
    dataset: CachedFbankDataset,
    epoch0_plan: Sequence[Sequence[int]],
    device: torch.device,
    stage: dict[str, str],
) -> dict[str, Any]:
    stage["value"] = "CUDA preflight: fresh objects"
    objects = create_training_objects(device)
    bn_reference = batchnorm_reference_state(objects["embedding_model"])
    ecapa_before = cpu_clone_state_dict(objects["embedding_model"])
    aam_before = cpu_clone_state_dict(objects["aam"])
    generator = torch.Generator().manual_seed(SEED + 1000)
    loader = create_cached_fbank_training_dataloader(
        dataset,
        list(epoch0_plan[:2]),  # type: ignore[arg-type]
        num_workers=0,
        generator=generator,
    )
    counter, handle = register_optimizer_hook(objects["optimizer"])
    updates = []
    torch.cuda.reset_peak_memory_stats(device)
    for position, batch in enumerate(loader):
        logical = prepare_logical_batch(batch, epoch0_plan[position])
        updates.append(
            run_optimizer_update(
                objects,
                logical,
                expected_step=position + 1,
                optimizer_hook_count=counter,
                batchnorm_reference=bn_reference,
                stage=stage,
            )
        )
    handle.remove()
    if len(updates) != 2 or counter["value"] != 2:
        raise RuntimeError("CUDA preflight did not complete exactly two updates")
    assert_finite_tensor_tree(
        objects["optimizer"].state_dict(), "preflight optimizer state"
    )
    ecapa_delta = parameter_delta(objects["embedding_model"], ecapa_before)
    aam_delta = parameter_delta(objects["aam"], aam_before)
    assert_batchnorm_reference_exact(objects["embedding_model"], bn_reference)
    result = {
        "result": "PASS",
        "fresh_objects": True,
        "train_only": True,
        "logical_batches": 2,
        "optimizer_updates": 2,
        "microbatch_size": PHYSICAL_MICROBATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "epoch_0_plan_sha256": SAMPLER_BINDING["epoch_0_plan_sha256"],
        "losses": [update["logical_loss"] for update in updates],
        "ecapa_parameter_delta": ecapa_delta,
        "aam_parameter_delta": aam_delta,
        "batchnorm_buffers_exact": True,
        "grad_scaler_skipped_updates": 0,
        "optimizer_state_finite": True,
        "cuda_peak_allocated_bytes": max(
            update["cuda_peak_allocated_bytes"] for update in updates
        ),
        "cuda_peak_reserved_bytes": max(
            update["cuda_peak_reserved_bytes"] for update in updates
        ),
        "validation_accessed": False,
        "production_checkpoint_created": False,
    }
    atomic_json(PREFLIGHT_PATH, result)
    objects["optimizer"].zero_grad(set_to_none=True)
    del loader, objects, bn_reference, ecapa_before, aam_before
    gc.collect()
    torch.cuda.empty_cache()
    return result


def build_checkpoint(
    objects: Mapping[str, Any],
    *,
    global_step: int,
    losses: Sequence[float],
    generator: torch.Generator,
    bn_reference: Mapping[str, torch.Tensor],
    reason: str,
    validation_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "schema_name": "viespeaker2_ecapa_aam_one_epoch",
        "schema_version": 2,
        "model_source": PRETRAINED_MODEL_ID,
        "upstream_bindings": {
            name: dict(binding) for name, binding in UPSTREAM_BINDINGS.items()
        },
        "sampler_binding": dict(SAMPLER_BINDING),
        "feature_shape": list(FEATURE_SHAPE),
        "embedding_dimension": EMBEDDING_DIM,
        "class_count": NUM_CLASSES,
        "label_range": [0, NUM_CLASSES - 1],
        "training_configuration": dict(TRAINING_CONFIGURATION),
        "aam_configuration": dict(AAM_CONFIGURATION),
        "optimizer_configuration": {
            "name": OPTIMIZER_CONFIGURATION["name"],
            "groups": [
                dict(group) for group in OPTIMIZER_CONFIGURATION["groups"]
            ],
            "scheduler": None,
            "gradient_clipping": None,
        },
        "mixed_precision_configuration": dict(
            MIXED_PRECISION_CONFIGURATION
        ),
        "batchnorm_policy": dict(BATCHNORM_POLICY),
        "epoch": 0,
        "completed_batch_position": global_step - 1,
        "global_optimizer_step": global_step,
        "next_cursor": resume_cursor(global_step),
        "embedding_model_state": to_cpu_tree(
            objects["embedding_model"].state_dict()
        ),
        "mean_var_norm_state": to_cpu_tree(
            objects["mean_var_norm"].state_dict()
        ),
        "aam_state": to_cpu_tree(objects["aam"].state_dict()),
        "optimizer_state": to_cpu_tree(objects["optimizer"].state_dict()),
        "grad_scaler_state": to_cpu_tree(objects["scaler"].state_dict()),
        "rng_state": capture_rng_state(),
        "dataloader_generator_state": generator.get_state().cpu().clone(),
        "batchnorm_reference_state": to_cpu_tree(dict(bn_reference)),
        "loss_summary": summarize_values(losses),
        "creation_reason": reason,
        "validation_selection": (
            dict(validation_selection) if validation_selection is not None else None
        ),
        "stop_reason": (
            "epoch_0_complete"
            if global_step == NUM_LOGICAL_BATCHES
            else "in_progress_epoch_0"
        ),
        "epoch_1_started": False,
    }
    validate_checkpoint_v2(checkpoint)
    return checkpoint


def save_checkpoint(
    checkpoint: Mapping[str, Any], path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    atomic_save_checkpoint(checkpoint, path)
    loaded = load_checkpoint(path)
    if (
        loaded["global_optimizer_step"]
        != checkpoint["global_optimizer_step"]
        or loaded["creation_reason"] != checkpoint["creation_reason"]
    ):
        raise RuntimeError("checkpoint readback identity mismatch")
    return {
        "path": relative(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "global_optimizer_step": loaded["global_optimizer_step"],
        "next_cursor": loaded["next_cursor"],
        "creation_reason": loaded["creation_reason"],
        "atomic_and_readable": True,
        "seconds": time.perf_counter() - started,
    }


def fresh_object_roundtrip(
    checkpoint: Mapping[str, Any],
    dataset: CachedFbankDataset,
) -> tuple[dict[str, Any], dict[str, Any]]:
    objects = create_training_objects(torch.device("cpu"))
    generator = torch.Generator()
    objects["embedding_model"].load_state_dict(
        checkpoint["embedding_model_state"], strict=True
    )
    objects["mean_var_norm"].load_state_dict(
        checkpoint["mean_var_norm_state"], strict=True
    )
    objects["aam"].load_state_dict(checkpoint["aam_state"], strict=True)
    objects["optimizer"].load_state_dict(checkpoint["optimizer_state"])
    objects["scaler"].load_state_dict(checkpoint["grad_scaler_state"])
    generator.set_state(checkpoint["dataloader_generator_state"])
    apply_batchnorm_policy(objects["embedding_model"])
    assert_batchnorm_reference_exact(
        objects["embedding_model"],
        checkpoint["batchnorm_reference_state"],
    )
    checks = checkpoint_roundtrip_exact(
        checkpoint,
        embedding_model=objects["embedding_model"],
        mean_var_norm=objects["mean_var_norm"],
        aam=objects["aam"],
        optimizer=objects["optimizer"],
        scaler=objects["scaler"],
        dataloader_generator=generator,
    )
    restore_rng_state(checkpoint["rng_state"])
    if not values_exactly_equal(capture_rng_state(), checkpoint["rng_state"]):
        raise RuntimeError("fresh-object RNG restoration failed")
    epoch1_sampler = HybridShardAwareSpeakerBatchSampler(
        dataset,
        speakers_per_batch=SPEAKERS_PER_BATCH,
        samples_per_speaker=SAMPLES_PER_SPEAKER,
        active_shard_window=ACTIVE_SHARD_WINDOW,
        num_batches=NUM_LOGICAL_BATCHES,
        seed=SEED,
    )
    epoch1_sampler.set_epoch(1)
    regenerated_epoch1_plan = list(epoch1_sampler)
    regenerated_epoch1 = plan_sha256(1, regenerated_epoch1_plan)
    if regenerated_epoch1 != SAMPLER_BINDING["epoch_1_plan_sha256"]:
        raise RuntimeError("fresh-object epoch-1 plan regeneration failed")
    if checkpoint["next_cursor"] != {
        "next_epoch": 1,
        "next_batch_position": 0,
    }:
        raise RuntimeError("fresh-object checkpoint cursor is invalid")
    result = {
        **checks,
        "fresh_speechbrain_constructed": True,
        "fresh_aam_constructed": True,
        "fresh_adamw_constructed": True,
        "fresh_grad_scaler_constructed": True,
        "rng_restored_exact": True,
        "batchnorm_policy_reapplied": True,
        "batchnorm_buffers_exact": True,
        "next_epoch": 1,
        "next_batch_position": 0,
        "global_optimizer_step": NUM_LOGICAL_BATCHES,
        "epoch_1_plan_sha256": regenerated_epoch1,
        "optimizer_steps_after_load": 0,
        "epoch_1_started": False,
    }
    validation_objects = {
        "mean_var_norm": objects["mean_var_norm"],
        "embedding_model": objects["embedding_model"],
    }
    del objects["aam"], objects["optimizer"], objects["scaler"]
    gc.collect()
    return validation_objects, result


def run_validation(
    objects: Mapping[str, Any],
    *,
    device: torch.device,
    checkpoint_sha256: str,
    bn_reference: Mapping[str, torch.Tensor],
    epoch: int = 0,
    checkpoint_path: Path = EPOCH_PATH,
    embedding_path: Path = EMBEDDING_PATH,
    score_path: Path = SCORE_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, Any]:
    dataset = CachedFbankDataset(
        CACHE_DIR,
        "validation",
        max_cached_shards=2,
        validate_finite=False,
    )
    manifest = read_validation_manifest(VALIDATION_MANIFEST)
    manifest_metadata = [
        (row.audio_path, row.speaker_id, row.duplicate_group)
        for row in manifest
    ]
    dataset_metadata = [
        (row.relative_audio_path, row.speaker_id, row.duplicate_group)
        for row in dataset.rows
    ]
    if (
        len(dataset) != 15355
        or len({row.speaker_id for row in dataset.rows}) != 100
        or any(
            row.speaker_label != -1 or row.final_split != "validation"
            for row in dataset.rows
        )
        or dataset_metadata != manifest_metadata
    ):
        raise ValueError("approved validation Dataset invariants failed")
    trials = read_trials_csv(TRIAL_PATH)
    speakers = {row.speaker_id for row in dataset.rows}
    validate_validation_trials(
        trials,
        expected_speakers=speakers,
        expected_positive=10000,
        expected_negative=10000,
        expected_positive_per_speaker=100,
        expected_negative_participation=200,
    )
    ownership = {
        row.relative_audio_path: row.speaker_id for row in dataset.rows
    }
    for trial in trials:
        if (
            ownership.get(trial.left_audio_path) != trial.left_speaker_id
            or ownership.get(trial.right_audio_path) != trial.right_speaker_id
        ):
            raise ValueError("fixed trial path ownership failed")

    mean_var_norm = objects["mean_var_norm"].to(device).eval()
    embedding_model = objects["embedding_model"].to(device).eval()
    assert_batchnorm_reference_exact(embedding_model, bn_reference)
    state_before = to_cpu_tree(embedding_model.state_dict())
    loader = create_cached_fbank_dataloader(
        dataset, 64, shuffle=False, num_workers=0
    )
    embeddings: list[torch.Tensor] = []
    paths: list[str] = []
    output_speakers: list[str] = []
    expected_index = 0
    torch.cuda.reset_peak_memory_stats(device)
    validation_started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            indexes = batch["dataset_index"].tolist()
            if indexes != list(range(expected_index, expected_index + len(indexes))):
                raise RuntimeError("validation cache order is not sequential")
            expected_index += len(indexes)
            features = batch["fbank"].to(device=device, dtype=torch.float32)
            lengths = torch.ones(
                features.shape[0], device=device, dtype=torch.float32
            )
            normalized = mean_var_norm(features, lengths)
            raw = embedding_model(normalized, lengths)
            if (
                tuple(normalized.shape) != tuple(features.shape)
                or tuple(raw.shape) != (features.shape[0], 1, EMBEDDING_DIM)
                or not bool(torch.isfinite(raw).all().item())
            ):
                raise RuntimeError("validation encoder output is invalid")
            flat = raw.squeeze(1).to(device="cpu", dtype=torch.float32)
            embeddings.append(flat)
            paths.extend(batch["relative_audio_path"])
            output_speakers.extend(batch["speaker_id"])
    torch.cuda.synchronize(device)
    validation_seconds = time.perf_counter() - validation_started
    stored = torch.cat(embeddings)
    if (
        expected_index != 15355
        or tuple(stored.shape) != (15355, EMBEDDING_DIM)
        or stored.dtype != torch.float32
        or stored.device.type != "cpu"
        or not bool(torch.isfinite(stored).all().item())
        or paths != [row.relative_audio_path for row in dataset.rows]
        or output_speakers != [row.speaker_id for row in dataset.rows]
    ):
        raise RuntimeError("validation embedding count/order invariant failed")
    assert_batchnorm_reference_exact(embedding_model, bn_reference)
    if not values_exactly_equal(
        to_cpu_tree(embedding_model.state_dict()), state_before
    ):
        raise RuntimeError("embedding model changed during validation")

    embedding_artifact = {
        "schema_name": "viespeaker2_validation_embeddings",
        "schema_version": 2,
        "epoch": epoch,
        "checkpoint_path": relative(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_index_sha256": UPSTREAM_BINDINGS[
            "validation_cache_index"
        ]["sha256"],
        "validation_manifest_sha256": UPSTREAM_BINDINGS[
            "validation_manifest"
        ]["sha256"],
        "trial_csv_sha256": UPSTREAM_BINDINGS[
            "validation_trial_csv"
        ]["sha256"],
        "relative_audio_paths": paths,
        "speaker_ids": output_speakers,
        "embeddings": stored,
        "shape": [15355, EMBEDDING_DIM],
        "dtype": "float32",
        "device": "cpu",
        "order": "validation_cache_index_order",
    }
    atomic_torch_save(embedding_artifact, embedding_path)
    loaded_embeddings = torch.load(
        embedding_path, map_location="cpu", weights_only=False
    )
    if (
        loaded_embeddings["relative_audio_paths"] != paths
        or loaded_embeddings["speaker_ids"] != output_speakers
        or not torch.equal(loaded_embeddings["embeddings"], stored)
    ):
        raise RuntimeError("validation embedding artifact roundtrip failed")

    scoring_started = time.perf_counter()
    scores = score_trials(stored, paths, trials)
    scoring_seconds = time.perf_counter() - scoring_started
    targets = torch.tensor([trial.target for trial in trials], dtype=torch.long)
    score_artifact = {
        "schema_name": "viespeaker2_validation_scores",
        "schema_version": 2,
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "trial_csv_sha256": UPSTREAM_BINDINGS[
            "validation_trial_csv"
        ]["sha256"],
        "trial_ids": [trial.trial_id for trial in trials],
        "targets": targets,
        "scores": scores,
        "score_kind": "cosine_similarity",
        "threshold_semantics": "accept same speaker when score >= threshold",
    }
    atomic_torch_save(score_artifact, score_path)
    loaded_scores = torch.load(
        score_path, map_location="cpu", weights_only=False
    )
    if (
        loaded_scores["trial_ids"] != score_artifact["trial_ids"]
        or not torch.equal(loaded_scores["targets"], targets)
        or not torch.equal(loaded_scores["scores"], scores)
    ):
        raise RuntimeError("validation score artifact roundtrip failed")

    metric_started = time.perf_counter()
    eer = calculate_eer(scores.tolist(), targets.tolist())
    confusion = empirical_confusion(
        loaded_scores["scores"].tolist(),
        loaded_scores["targets"].tolist(),
        eer.empirical_threshold,
    )
    same = numeric_summary(scores[targets == 1].tolist())
    different = numeric_summary(scores[targets == 0].tolist())
    metric_seconds = time.perf_counter() - metric_started
    if (
        abs(confusion["far"] - eer.empirical_far) > 1e-15
        or abs(confusion["frr"] - eer.empirical_frr) > 1e-15
    ):
        raise RuntimeError("stored-score confusion disagrees with empirical EER")
    metrics = {
        **dataclasses.asdict(eer),
        "empirical_confusion": confusion,
        "same_speaker_scores": same,
        "different_speaker_scores": different,
        "score_range": [float(scores.min()), float(scores.max())],
        "score_count": len(scores),
        "positive_trials": int(targets.sum().item()),
        "negative_trials": int((targets == 0).sum().item()),
        "threshold_semantics": "accept same speaker when score >= threshold",
        "metric_implementation": "grouped_sorted_tie_aware_o_n_log_n",
    }
    atomic_json(metrics_path, metrics)
    return {
        "metrics": metrics,
        "runtime": {
            "validation_rows": 15355,
            "validation_speakers": 100,
            "batch_size": 64,
            "workers": 0,
            "embedding_shape": [15355, EMBEDDING_DIM],
            "embedding_dtype": "float32",
            "embedding_device": "cpu",
            "validation_seconds": validation_seconds,
            "embedding_throughput": 15355 / validation_seconds,
            "scoring_seconds": scoring_seconds,
            "metric_seconds": metric_seconds,
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "cuda_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
            "aam_calls": 0,
            "compute_features_calls": 0,
            "pretrained_classifier_calls": 0,
            "augmentation": False,
            "sequential": True,
        },
        "artifacts": {
            "embedding_path": relative(embedding_path),
            "embedding_sha256": file_sha256(embedding_path),
            "score_path": relative(score_path),
            "score_sha256": file_sha256(score_path),
            "metrics_path": relative(metrics_path),
            "metrics_sha256": file_sha256(metrics_path),
        },
    }


def markdown_report(result: Mapping[str, Any]) -> str:
    metrics = result["validation"]["metrics"]
    confusion = metrics["empirical_confusion"]
    comparison = result["baseline_comparison"]
    return f"""# VieSpeaker2.0 ECAPA-AAM epoch 0

Result: **PASS**. Exactly 2,969 optimizer updates completed; epoch 1 did not start.

## Training

- Cached Fbank `[B,301,80] -> mean_var_norm -> embedding_model -> [B,1,192] -> [B,192]`
- P=16, K=2, logical batch 32; physical microbatch 4; accumulation 8
- AdamW ECAPA/AAM learning rates `1e-5 / 1e-3`; no scheduler or gradient clipping
- Training duration: {result['training']['duration_seconds']:.6f} seconds
- Loss first/final/mean: {result['training']['loss']['first']} / {result['training']['loss']['final']} / {result['training']['loss']['mean']}

## Validation

- Embeddings: `[15355,192]` float32 CPU; trials: 10,000 positive / 10,000 negative
- Interpolated EER: {metrics['interpolated_eer']} ({metrics['interpolated_eer_percentage']:.6f}%)
- Empirical threshold: {metrics['empirical_threshold']}
- Empirical FAR / FRR: {metrics['empirical_far']} / {metrics['empirical_frr']}
- TP / TN / FP / FN: {confusion['tp']} / {confusion['tn']} / {confusion['fp']} / {confusion['fn']}

## Baseline comparison

- Pretrained EER: {comparison['pretrained_interpolated_eer']}
- Signed difference: {comparison['signed_eer_difference']}
- Relative change: {comparison['relative_eer_change']}
- Classification: **{comparison['classification']}**

## Checkpoint and stop

- Approved future-resume checkpoint: `outputs/ecapa_aam_one_epoch_v2/best.pt`
- Best checkpoint SHA-256: `{result['checkpoints']['best']['sha256']}`
- Final cursor: epoch 1, batch position 0, global step 2,969
- Epoch 1 started: false
- Final test remained quarantined
"""


def run(args: argparse.Namespace, stage: dict[str, str]) -> dict[str, Any]:
    expected_python = (REPO_ROOT / ".venv-cuda/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError("v2 epoch must run with .venv-cuda Python")
    device = torch.device(args.device)
    if (
        str(device) != "cuda:0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() < 1
    ):
        raise RuntimeError("v2 epoch requires exactly cuda:0")
    ensure_fresh_targets()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    total_started = time.perf_counter()

    stage["value"] = "identity preflight"
    protected_before = verify_upstream_files()
    dataset, epoch0_plan, epoch1_plan, sampler_audit = (
        create_train_dataset_and_plans()
    )
    rows_identity = dataset.rows

    stage["value"] = "two-update train-only CUDA preflight"
    preflight = run_cuda_preflight(dataset, epoch0_plan, device, stage)
    if preflight["result"] != "PASS":
        raise RuntimeError("CUDA capacity preflight failed")

    stage["value"] = "fresh production object construction"
    if plan_sha256(0, epoch0_plan) != SAMPLER_BINDING["epoch_0_plan_sha256"]:
        raise RuntimeError("epoch-0 plan changed before the first production step")
    seed_everything(SEED)
    objects = create_training_objects(device)
    bn_reference = batchnorm_reference_state(objects["embedding_model"])
    generator = torch.Generator().manual_seed(SEED + 1000)
    loader = create_cached_fbank_training_dataloader(
        dataset,
        epoch0_plan,  # type: ignore[arg-type]
        num_workers=0,
        generator=generator,
    )
    iterator = iter(loader)
    counter, hook_handle = register_optimizer_hook(objects["optimizer"])
    controller = OneEpochV2Controller()
    controller.start(0)
    losses: list[float] = []
    durations: list[float] = []
    ecapa_norms: list[float] = []
    aam_norms: list[float] = []
    scaler_scales: list[float] = []
    rolling: list[dict[str, Any]] = []
    recent_losses: deque[float] = deque(maxlen=100)
    training_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for position in range(NUM_LOGICAL_BATCHES):
        stage["value"] = f"production epoch 0: load batch {position}"
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise RuntimeError("production DataLoader ended early") from error
        logical = prepare_logical_batch(batch, epoch0_plan[position])
        global_step = position + 1
        update = run_optimizer_update(
            objects,
            logical,
            expected_step=global_step,
            optimizer_hook_count=counter,
            batchnorm_reference=bn_reference,
            stage=stage,
        )
        controller.record_update(position)
        losses.append(update["logical_loss"])
        durations.append(update["duration_seconds"])
        ecapa_norms.append(update["ecapa_gradient_norm"])
        aam_norms.append(update["aam_gradient_norm"])
        scaler_scales.append(update["grad_scaler_scale"])
        recent_losses.append(update["logical_loss"])

        checkpoint_path: str | None = None
        if global_step in ROLLING_CHECKPOINT_STEPS:
            stage["value"] = f"rolling checkpoint step {global_step}"
            checkpoint = build_checkpoint(
                objects,
                global_step=global_step,
                losses=losses,
                generator=generator,
                bn_reference=bn_reference,
                reason="rolling",
            )
            rolling_result = save_checkpoint(checkpoint, LAST_PATH)
            rolling.append(rolling_result)
            checkpoint_path = relative(LAST_PATH)
            del checkpoint
            gc.collect()

        if global_step in LOGGING_STEPS:
            record = {
                "schema_name": "viespeaker2_epoch0_scalar_training_log",
                "schema_version": 2,
                "epoch": 0,
                "global_optimizer_step": global_step,
                "logical_batch_position": position,
                "logical_loss": update["logical_loss"],
                "mean_logical_loss_latest_window": (
                    sum(recent_losses) / len(recent_losses)
                ),
                "ecapa_gradient_norm": update["ecapa_gradient_norm"],
                "aam_gradient_norm": update["aam_gradient_norm"],
                "ecapa_learning_rate": float(
                    objects["optimizer"].param_groups[0]["lr"]
                ),
                "aam_learning_rate": float(
                    objects["optimizer"].param_groups[1]["lr"]
                ),
                "grad_scaler_scale": update["grad_scaler_scale"],
                "elapsed_training_seconds": time.perf_counter()
                - training_started,
                "step_duration_seconds": update["duration_seconds"],
                "logical_samples_per_second": LOGICAL_BATCH_SIZE
                / update["duration_seconds"],
                "cuda_allocated_bytes": update["cuda_allocated_bytes"],
                "cuda_reserved_bytes": update["cuda_reserved_bytes"],
                "cuda_peak_allocated_bytes": update[
                    "cuda_peak_allocated_bytes"
                ],
                "cuda_peak_reserved_bytes": update[
                    "cuda_peak_reserved_bytes"
                ],
                "rolling_checkpoint_path": checkpoint_path,
            }
            append_jsonl(TRAIN_LOG_PATH, record)
            print(json.dumps(record, sort_keys=True), flush=True)
        del batch, logical, update

    try:
        extra = next(iterator)
    except StopIteration:
        extra = None
    if extra is not None:
        raise RuntimeError("production DataLoader exceeded 2,969 batches")
    hook_handle.remove()
    epoch_summary = controller.finish()
    training_seconds = time.perf_counter() - training_started
    training_peak_allocated = int(torch.cuda.max_memory_allocated(device))
    training_peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if (
        counter["value"] != NUM_LOGICAL_BATCHES
        or len(losses) != NUM_LOGICAL_BATCHES
        or dataset.rows is not rows_identity
    ):
        raise RuntimeError("production epoch count/mutation invariant failed")
    assert_batchnorm_reference_exact(objects["embedding_model"], bn_reference)
    if [item["global_optimizer_step"] for item in rolling] != list(
        ROLLING_CHECKPOINT_STEPS
    ):
        raise RuntimeError("rolling checkpoint schedule is incomplete")
    atomic_json(
        LOSS_PATH,
        {
            "schema_name": "viespeaker2_epoch0_logical_losses",
            "schema_version": 2,
            "epoch": 0,
            "count": len(losses),
            "losses": losses,
        },
    )

    stage["value"] = "epoch_000 checkpoint"
    epoch_checkpoint = build_checkpoint(
        objects,
        global_step=NUM_LOGICAL_BATCHES,
        losses=losses,
        generator=generator,
        bn_reference=bn_reference,
        reason="epoch_complete",
    )
    epoch_result = save_checkpoint(epoch_checkpoint, EPOCH_PATH)
    epoch_loaded = load_checkpoint(EPOCH_PATH)
    epoch_sha256 = epoch_result["sha256"]

    objects["optimizer"].zero_grad(set_to_none=True)
    del iterator, loader, objects, epoch_checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    stage["value"] = "fresh-object checkpoint roundtrip"
    validation_objects, roundtrip = fresh_object_roundtrip(
        epoch_loaded, dataset
    )
    if roundtrip["optimizer_steps_after_load"] != 0:
        raise RuntimeError("roundtrip executed an optimizer step")

    stage["value"] = "fixed validation"
    validation = run_validation(
        validation_objects,
        device=device,
        checkpoint_sha256=epoch_sha256,
        bn_reference=epoch_loaded["batchnorm_reference_state"],
    )
    comparison = classify_baseline_comparison(
        validation["metrics"]["interpolated_eer"],
        BASELINE_INTERPOLATED_EER,
        QUALITY_TOLERANCE,
    )
    selection = {
        "selected_checkpoint": "epoch_000.pt",
        "selection_scope": "best trained v2 checkpoint available in this task",
        "selection_metric": "interpolated validation EER",
        "classification": comparison["classification"],
        "epoch_0_interpolated_eer": validation["metrics"][
            "interpolated_eer"
        ],
        "epoch_0_empirical_threshold": validation["metrics"][
            "empirical_threshold"
        ],
        "pretrained_baseline_is_comparison_only": True,
    }
    stage["value"] = "best checkpoint"
    best_checkpoint = dict(epoch_loaded)
    best_checkpoint["creation_reason"] = "best_validation"
    best_checkpoint["validation_selection"] = selection
    validate_checkpoint_v2(best_checkpoint)
    best_result = save_checkpoint(best_checkpoint, BEST_PATH)
    best_loaded = load_checkpoint(BEST_PATH)
    state_keys = (
        "embedding_model_state",
        "mean_var_norm_state",
        "aam_state",
        "optimizer_state",
        "grad_scaler_state",
    )
    if not all(
        values_exactly_equal(best_loaded[key], epoch_loaded[key])
        for key in state_keys
    ):
        raise RuntimeError("best checkpoint states differ from epoch_000")
    best_result["states_match_epoch_000"] = True

    stage["value"] = "protected input postflight"
    protected_after = verify_upstream_files()
    if protected_after != protected_before:
        raise RuntimeError("approved input hashes changed")

    training = {
        **epoch_summary,
        "duration_seconds": training_seconds,
        "loss": summarize_values(losses),
        "ecapa_gradient_norms": summarize_values(ecapa_norms),
        "aam_gradient_norms": summarize_values(aam_norms),
        "grad_scaler": {
            **summarize_values(scaler_scales),
            "unique_scales": sorted(set(scaler_scales)),
            "skipped_updates": 0,
        },
        "step_duration": summarize_values(durations),
        "logging_records": len(LOGGING_STEPS),
        "batchnorm_buffers_compared": len(bn_reference),
        "batchnorm_buffers_bit_exact": True,
        "cuda_peak_allocated_bytes": int(
            training_peak_allocated
        ),
        "cuda_peak_reserved_bytes": int(
            training_peak_reserved
        ),
        "training_log_sha256": file_sha256(TRAIN_LOG_PATH),
        "logical_losses_sha256": file_sha256(LOSS_PATH),
    }
    runtime = {
        "schema_name": RUNTIME_SCHEMA_NAME,
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "result": "PASS",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "speechbrain": speechbrain.__version__,
            "device": "cuda:0",
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        },
        "upstream_bindings": protected_after,
        "sampler_binding": dict(SAMPLER_BINDING),
        "training_configuration": dict(TRAINING_CONFIGURATION),
        "aam_configuration": dict(AAM_CONFIGURATION),
        "optimizer_configuration": OPTIMIZER_CONFIGURATION,
        "mixed_precision_configuration": MIXED_PRECISION_CONFIGURATION,
        "batchnorm_policy": BATCHNORM_POLICY,
        "cuda_preflight": preflight,
        "sampler_audit": sampler_audit,
        "training": training,
        "checkpoints": {
            "rolling": rolling,
            "last": {
                "path": relative(LAST_PATH),
                "sha256": file_sha256(LAST_PATH),
            },
            "epoch": epoch_result,
            "best": best_result,
            "fresh_object_roundtrip": roundtrip,
        },
        "validation": validation,
        "baseline_comparison": comparison,
        "total_runtime_seconds": time.perf_counter() - total_started,
        "final_cursor": {
            "next_epoch": 1,
            "next_batch_position": 0,
            "global_optimizer_step": NUM_LOGICAL_BATCHES,
        },
        "stop_reason": "epoch_0_complete_validation_complete_stop",
        "epoch_1_started": False,
        "final_test_accessed": False,
        "source_audio_accessed": False,
        "commit_or_push_performed": False,
    }
    atomic_json(RUNTIME_PATH, runtime)
    report = {
        **runtime,
        "runtime_path": relative(RUNTIME_PATH),
        "runtime_sha256": file_sha256(RUNTIME_PATH),
        "tracked_report_identity": {
            "upstream_bindings": protected_after,
            "sampler_config_sha256": SAMPLER_BINDING["config_sha256"],
            "epoch_0_plan_sha256": SAMPLER_BINDING["epoch_0_plan_sha256"],
            "validation_trial_sha256": UPSTREAM_BINDINGS[
                "validation_trial_csv"
            ]["sha256"],
            "pretrained_baseline_identity_sha256": UPSTREAM_BINDINGS[
                "pretrained_baseline_identity"
            ]["sha256"],
            "training_configuration": dict(TRAINING_CONFIGURATION),
            "last_checkpoint_sha256": file_sha256(LAST_PATH),
            "epoch_checkpoint_sha256": file_sha256(EPOCH_PATH),
            "best_checkpoint_sha256": file_sha256(BEST_PATH),
            "validation_embedding_sha256": validation["artifacts"][
                "embedding_sha256"
            ],
            "validation_score_sha256": validation["artifacts"]["score_sha256"],
            "epoch_0_interpolated_eer": validation["metrics"][
                "interpolated_eer"
            ],
            "epoch_0_empirical_threshold": validation["metrics"][
                "empirical_threshold"
            ],
            "model_quality_classification": comparison["classification"],
            "final_cursor": runtime["final_cursor"],
            "stop_reason": runtime["stop_reason"],
            "timestamps_in_identity": False,
            "absolute_paths_in_identity": False,
        },
    }
    atomic_json(REPORT_JSON, report)
    REPORT_MD.write_text(
        markdown_report(report),
        encoding="utf-8",
        newline="",
    )
    return report


def safe_error(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    return message.replace(str(REPO_ROOT), ".")


def main() -> None:
    args = parse_args()
    stage = {"value": "startup"}
    try:
        result = run(args, stage)
    except BaseException as error:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_name": "viespeaker2_ecapa_aam_one_epoch_failure",
            "schema_version": 2,
            "result": "FAIL",
            "stage": stage["value"],
            "error": safe_error(error),
            "cuda_oom": "out of memory" in str(error).lower(),
            "microbatch_fallback_attempted": False,
            "production_configuration_changed": False,
            "epoch_1_started": False,
            "final_test_accessed": False,
        }
        atomic_json(FAILURE_PATH, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        raise
    print(json.dumps({
        "result": result["result"],
        "optimizer_updates": result["training"]["optimizer_updates"],
        "interpolated_eer": result["validation"]["metrics"][
            "interpolated_eer"
        ],
        "empirical_threshold": result["validation"]["metrics"][
            "empirical_threshold"
        ],
        "classification": result["baseline_comparison"]["classification"],
        "best_checkpoint_sha256": result["checkpoints"]["best"]["sha256"],
        "epoch_1_started": result["epoch_1_started"],
    }, indent=2, sort_keys=True), flush=True)
    print("VIESPEAKER2 EPOCH 0: PASS", flush=True)


if __name__ == "__main__":
    main()
