"""Run the approved cached-Fbank AAM training smoke test or capacity probe."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import speechbrain
import torch

from src.aam_training import (
    AAMSoftmax,
    APPROVED_TRAIN_INDEX_FILENAME,
    PRETRAINED_MODEL_ID,
    SEED,
    aggregate_gradient_norm,
    apply_batchnorm_policy,
    assert_batchnorm_running_state_exact,
    atomic_save_checkpoint,
    batch_at_position,
    batchnorm_parameter_names,
    batchnorm_running_state,
    build_adamw_optimizer,
    capture_rng_state,
    cpu_clone_state_dict,
    create_train_only_smoke_dataset,
    file_sha256,
    iter_microbatches,
    parameter_delta,
    restore_rng_state,
    round_robin_reorder,
    scaled_cross_entropy_sum,
    to_cpu_tree,
    validate_checkpoint_v1,
    validate_optimizer_coverage,
    values_exactly_equal,
)
from src.cached_fbank_dataset import (
    CachedFbankDataset,
    create_cached_fbank_training_dataloader,
)
from src.cached_fbank_samplers import HybridShardAwareSpeakerBatchSampler
from src.speechbrain_frontend import SpeechBrainECAPAFrontend


LOGICAL_BATCH_SIZE = 32
MAIN_MICROBATCH_SIZE = 2
MAIN_ACCUMULATION_STEPS = 16
CHECKPOINT_PATH = Path("outputs/aam_softmax_training_smoke_v1_1/checkpoint_v1.pt")
MAIN_RESULT_PATH = Path("reports/aam_softmax_training_smoke_v1_1_main_runtime.json")
PROBE_RESULT_PATH = Path("reports/aam_softmax_training_smoke_v1_1_probe_runtime.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("main", "probe"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/fbank_cache_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--main-result", type=Path, default=MAIN_RESULT_PATH)
    return parser.parse_args()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline=""
    )
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_sampler(dataset: CachedFbankDataset) -> HybridShardAwareSpeakerBatchSampler:
    sampler = HybridShardAwareSpeakerBatchSampler(
        dataset,
        speakers_per_batch=16,
        samples_per_speaker=2,
        active_shard_window=8,
        seed=SEED,
    )
    sampler.set_epoch(0)
    return sampler


def batch_identity(dataset: CachedFbankDataset, indexes: list[int]) -> dict[str, Any]:
    return {
        "dataset_indexes": indexes,
        "speaker_ids": [dataset.rows[index].speaker_id for index in indexes],
        "relative_audio_paths": [
            dataset.rows[index].relative_audio_path for index in indexes
        ],
    }


def load_logical_batch(
    dataset: CachedFbankDataset, indexes: list[int], position: int
) -> dict[str, Any]:
    generator = torch.Generator()
    generator.manual_seed(SEED + position)
    loader = create_cached_fbank_training_dataloader(
        dataset, [indexes], num_workers=0, generator=generator  # type: ignore[arg-type]
    )
    batch = next(iter(loader))
    return round_robin_reorder(batch, speakers_per_batch=16, samples_per_speaker=2)


def validate_logical_batch(
    batch: Mapping[str, Any], expected_indexes: list[int], microbatch_size: int
) -> dict[str, Any]:
    features = batch["fbank"]
    labels = batch["speaker_label"]
    indexes = batch["dataset_index"]
    speakers = batch["speaker_id"]
    final_splits = batch["final_split"]
    if (
        tuple(features.shape) != (32, 301, 80)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
    ):
        raise AssertionError("cached logical Fbank batch must be CPU float32 [32, 301, 80]")
    if not bool(torch.isfinite(features).all().item()):
        raise AssertionError("cached logical Fbank batch contains NaN or Inf")
    if labels.dtype != torch.long or tuple(labels.shape) != (32,):
        raise AssertionError("logical labels must be int64 [32]")
    if bool(((labels < 0) | (labels > 487)).any().item()):
        raise AssertionError("logical labels are outside 0..487")
    if not isinstance(final_splits, list) or final_splits != ["train"] * 32:
        raise AssertionError("logical batch contains a non-train row")
    if len(set(speakers)) != 16:
        raise AssertionError("logical batch does not contain exactly 16 speakers")
    counts = {speaker: speakers.count(speaker) for speaker in set(speakers)}
    if set(counts.values()) != {2}:
        raise AssertionError("logical batch does not contain exactly two samples per speaker")
    if len(set(indexes.tolist())) != 32:
        raise AssertionError("logical batch contains a duplicate Dataset index")
    if sorted(indexes.tolist()) != sorted(expected_indexes):
        raise AssertionError("logical batch identities differ from sampler output")
    microbatches = list(iter_microbatches(batch, microbatch_size))
    expected_microbatches = LOGICAL_BATCH_SIZE // microbatch_size
    if len(microbatches) != expected_microbatches:
        raise AssertionError("incorrect number of physical microbatches")
    return {
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "feature_device": str(features.device),
        "feature_finite": True,
        "label_shape": list(labels.shape),
        "label_dtype": str(labels.dtype),
        "label_min": int(labels.min().item()),
        "label_max": int(labels.max().item()),
        "all_final_split_train": True,
        "dataset_indexes_unique": True,
        "speakers": 16,
        "samples_per_speaker": 2,
        "physical_microbatches": expected_microbatches,
        "distinct_speakers_per_microbatch": [
            len(set(microbatch["speaker_id"])) for microbatch in microbatches
        ],
    }


def load_trainable_encoder(
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    # Load the SpeechBrain checkpoint on CPU, retain only the two approved modules,
    # and move those modules to CUDA. The compute_features and pretrained
    # classifier modules therefore have no runtime call path in this script.
    frontend = SpeechBrainECAPAFrontend(device="cpu")
    encoder = frontend.classifier
    module_names = tuple(sorted(encoder.mods.keys()))
    if "mean_var_norm" not in module_names or "embedding_model" not in module_names:
        raise RuntimeError("SpeechBrain encoder is missing approved training modules")
    mean_var_norm = encoder.mods.mean_var_norm
    embedding_model = encoder.mods.embedding_model
    del encoder, frontend
    gc.collect()
    mean_var_norm.to(device)
    mean_var_norm.eval()
    for parameter in mean_var_norm.parameters():
        parameter.requires_grad_(False)
    embedding_model.to(device)
    batchnorm_names = apply_batchnorm_policy(embedding_model)
    if not batchnorm_names:
        raise AssertionError("ECAPA embedding model contains no BatchNorm modules")
    return mean_var_norm, embedding_model, {
        "checkpoint_module_names": list(module_names),
        "retained_modules": ["mean_var_norm", "embedding_model"],
        "compute_features_retained": False,
        "pretrained_classifier_retained": False,
        "batchnorm_module_count": len(batchnorm_names),
    }


def create_training_objects(device: torch.device) -> dict[str, Any]:
    mean_var_norm, embedding_model, loader_proof = load_trainable_encoder(device)
    aam = AAMSoftmax(seed=SEED).to(device)
    optimizer = build_adamw_optimizer(embedding_model, aam)
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128.0)
    validate_optimizer_coverage(optimizer, embedding_model, aam)
    if any(parameter.requires_grad for parameter in mean_var_norm.parameters()):
        raise AssertionError("mean_var_norm parameters must not require gradients")
    return {
        "mean_var_norm": mean_var_norm,
        "embedding_model": embedding_model,
        "aam": aam,
        "optimizer": optimizer,
        "scaler": scaler,
        "loader_proof": loader_proof,
    }


def run_optimizer_step(
    objects: Mapping[str, Any],
    logical_batch: Mapping[str, Any],
    *,
    microbatch_size: int,
    optimizer_step: int,
    stage: dict[str, str],
) -> dict[str, Any]:
    mean_var_norm = objects["mean_var_norm"]
    embedding_model = objects["embedding_model"]
    aam = objects["aam"]
    optimizer = objects["optimizer"]
    scaler = objects["scaler"]
    device = next(embedding_model.parameters()).device

    apply_batchnorm_policy(embedding_model)
    if not all(parameter.requires_grad for parameter in embedding_model.parameters()):
        raise AssertionError("an intended ECAPA parameter is frozen")
    batchnorm_before = batchnorm_running_state(embedding_model)
    embedding_before = cpu_clone_state_dict(embedding_model)
    aam_before = cpu_clone_state_dict(aam)
    excluded = batchnorm_parameter_names(embedding_model)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    stage["value"] = f"optimizer step {optimizer_step}: zero_grad"
    optimizer.zero_grad(set_to_none=True)
    logical_loss_sum = 0.0
    microbatch_shapes: list[dict[str, list[int]]] = []

    for number, microbatch in enumerate(iter_microbatches(logical_batch, microbatch_size)):
        stage["value"] = f"optimizer step {optimizer_step}: microbatch {number + 1} transfer"
        features_cpu = microbatch["fbank"]
        labels_cpu = microbatch["speaker_label"]
        if (
            tuple(features_cpu.shape) != (microbatch_size, 301, 80)
            or features_cpu.dtype != torch.float32
            or features_cpu.device.type != "cpu"
            or not bool(torch.isfinite(features_cpu).all().item())
        ):
            raise AssertionError("physical input must be finite CPU float32 [M, 301, 80]")
        features = features_cpu.to(device=device, dtype=torch.float32)
        labels = labels_cpu.to(device=device, dtype=torch.long)
        lengths = torch.ones(microbatch_size, device=device, dtype=torch.float32)

        stage["value"] = f"optimizer step {optimizer_step}: microbatch {number + 1} mean_var_norm"
        with torch.no_grad():
            normalized = mean_var_norm(features, lengths)
        if tuple(normalized.shape) != (microbatch_size, 301, 80):
            raise AssertionError("mean_var_norm changed the feature shape")
        if not bool(torch.isfinite(normalized).all().item()):
            raise AssertionError("normalized features contain NaN or Inf")

        stage["value"] = f"optimizer step {optimizer_step}: microbatch {number + 1} ECAPA forward"
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            raw_embedding = embedding_model(normalized, lengths)
        if tuple(raw_embedding.shape) != (microbatch_size, 1, 192):
            raise AssertionError("raw ECAPA embedding must have shape [M, 1, 192]")
        if not bool(torch.isfinite(raw_embedding).all().item()):
            raise AssertionError("raw ECAPA embedding contains NaN or Inf")
        embedding = raw_embedding.squeeze(1)
        if tuple(embedding.shape) != (microbatch_size, 192):
            raise AssertionError("training embedding must have shape [M, 192]")

        stage["value"] = f"optimizer step {optimizer_step}: microbatch {number + 1} AAM/loss"
        with torch.cuda.amp.autocast(enabled=False):
            logits = aam(embedding.float(), labels)
            loss = scaled_cross_entropy_sum(logits, labels, LOGICAL_BATCH_SIZE)
        if tuple(logits.shape) != (microbatch_size, 488):
            raise AssertionError("AAM logits must have shape [M, 488]")
        if loss.dtype != torch.float32 or not bool(torch.isfinite(loss).item()):
            raise AssertionError("scaled loss must be finite float32")
        logical_loss_sum += float(
            torch.nn.functional.cross_entropy(
                logits.detach(), labels, reduction="sum"
            ).item()
        )
        microbatch_shapes.append({
            "input": list(features.shape),
            "normalized": list(normalized.shape),
            "raw_embedding": list(raw_embedding.shape),
            "embedding": list(embedding.shape),
            "logits": list(logits.shape),
        })
        stage["value"] = f"optimizer step {optimizer_step}: microbatch {number + 1} backward"
        scaler.scale(loss).backward()
        del features, labels, lengths, normalized, raw_embedding, embedding, logits, loss

    stage["value"] = f"optimizer step {optimizer_step}: unscale and gradient validation"
    scaler.unscale_(optimizer)
    ecapa_gradient_norm = aggregate_gradient_norm(
        embedding_model.named_parameters(), "ECAPA embedding_model"
    )
    aam_gradient_norm = aggregate_gradient_norm(
        aam.named_parameters(), "AAM classifier"
    )
    stage["value"] = f"optimizer step {optimizer_step}: GradScaler/AdamW step"
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))

    stage["value"] = f"optimizer step {optimizer_step}: post-step verification"
    assert_batchnorm_running_state_exact(embedding_model, batchnorm_before)
    ecapa_delta = parameter_delta(embedding_model, embedding_before, exclude=excluded)
    aam_delta = parameter_delta(aam, aam_before)
    return {
        "optimizer_step": optimizer_step,
        "logical_loss": logical_loss_sum / LOGICAL_BATCH_SIZE,
        "ecapa_gradient_norm": ecapa_gradient_norm,
        "aam_gradient_norm": aam_gradient_norm,
        "ecapa_parameter_delta": ecapa_delta,
        "aam_parameter_delta": aam_delta,
        "batchnorm_running_buffers_compared": len(batchnorm_before),
        "batchnorm_running_buffers_exactly_unchanged": True,
        "grad_scaler_scale_before": scale_before,
        "grad_scaler_scale_after": scale_after,
        "duration_seconds": duration,
        "max_memory_allocated_bytes": allocated,
        "max_memory_reserved_bytes": reserved,
        "microbatch_shapes": microbatch_shapes,
    }


def train_data_identity(
    cache_dir: Path, selected_shards: list[str]
) -> dict[str, Any]:
    candidates = [
        cache_dir / "fbank_cache_config_v1.json",
        cache_dir / "train_feature_index_v1.csv",
    ]
    for relative in selected_shards:
        pure = Path(*relative.split("/"))
        if (
            len(pure.parts) != 2
            or pure.parts[0] != "train"
            or pure.name != relative.split("/")[-1]
        ):
            raise ValueError(f"non-allowlisted train shard identity: {relative!r}")
        candidates.append(cache_dir / pure)
    files = []
    for path in candidates:
        resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        files.append({"path": relative, "sha256": file_sha256(resolved)})
    return {"files": files}


def build_checkpoint(
    objects: Mapping[str, Any],
    identity: Mapping[str, Any],
    next_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_name": "speaker_verification_aam_training_smoke",
        "schema_version": 1,
        "pretrained_model_identifier": PRETRAINED_MODEL_ID,
        "embedding_model_state_dict": cpu_clone_state_dict(objects["embedding_model"]),
        "mean_var_norm_state_dict": cpu_clone_state_dict(objects["mean_var_norm"]),
        "aam_classifier_state_dict": cpu_clone_state_dict(objects["aam"]),
        "optimizer_state_dict": to_cpu_tree(objects["optimizer"].state_dict()),
        "grad_scaler_state_dict": to_cpu_tree(objects["scaler"].state_dict()),
        "epoch": 0,
        "next_logical_batch_position": 1,
        "global_optimizer_step": 1,
        "sampler": {
            "name": "HybridShardAwareSpeakerBatchSampler",
            "seed": SEED,
            "epoch": 0,
            "speakers_per_batch": 16,
            "samples_per_speaker": 2,
            "active_shard_window": 8,
            "logical_batch_size": 32,
        },
        "physical_microbatch_size": 2,
        "accumulation_steps": 16,
        "aam": {
            "embedding_dim": 192,
            "num_classes": 488,
            "margin_radians": 0.2,
            "scale": 30.0,
            "initialization_seed": SEED,
        },
        "optimizer_groups": [
            {"name": "embedding_model", "learning_rate": 1e-5, "weight_decay": 1e-4},
            {"name": "aam_classifier", "learning_rate": 1e-3, "weight_decay": 1e-4},
        ],
        "batchnorm_policy": {
            "embedding_model_training": True,
            "batchnorm_modules_eval": True,
            "batchnorm_affine_trainable": True,
            "running_buffers_exactly_frozen": True,
        },
        "amp_policy": {
            "enabled": True,
            "ecapa_autocast_dtype": "float16",
            "aam_math_dtype": "float32",
            "loss_dtype": "float32",
            "grad_scaler_initial_scale": 128.0,
        },
        "train_data_identity": dict(identity),
        "next_logical_batch_identity": dict(next_identity),
        "rng_state": capture_rng_state(),
    }


def load_checkpoint_into_fresh_objects(
    checkpoint: Mapping[str, Any], device: torch.device
) -> tuple[dict[str, Any], dict[str, bool]]:
    validate_checkpoint_v1(checkpoint)
    objects = create_training_objects(device)
    objects["embedding_model"].load_state_dict(
        checkpoint["embedding_model_state_dict"], strict=True
    )
    objects["mean_var_norm"].load_state_dict(
        checkpoint["mean_var_norm_state_dict"], strict=True
    )
    objects["aam"].load_state_dict(checkpoint["aam_classifier_state_dict"], strict=True)
    objects["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])
    objects["scaler"].load_state_dict(checkpoint["grad_scaler_state_dict"])
    apply_batchnorm_policy(objects["embedding_model"])
    checks = {
        "embedding_model_state_exact": values_exactly_equal(
            cpu_clone_state_dict(objects["embedding_model"]),
            checkpoint["embedding_model_state_dict"],
        ),
        "mean_var_norm_state_exact": values_exactly_equal(
            cpu_clone_state_dict(objects["mean_var_norm"]),
            checkpoint["mean_var_norm_state_dict"],
        ),
        "aam_state_exact": values_exactly_equal(
            cpu_clone_state_dict(objects["aam"]),
            checkpoint["aam_classifier_state_dict"],
        ),
        "optimizer_state_exact": values_exactly_equal(
            to_cpu_tree(objects["optimizer"].state_dict()),
            checkpoint["optimizer_state_dict"],
        ),
        "grad_scaler_state_exact": values_exactly_equal(
            objects["scaler"].state_dict(), checkpoint["grad_scaler_state_dict"]
        ),
        "configuration_and_counters_valid": True,
        "batchnorm_policy_reapplied": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"checkpoint state roundtrip failed: {checks}")
    restore_rng_state(checkpoint["rng_state"])
    return objects, checks


def environment_payload(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "speechbrain": speechbrain.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
    }


def run_main(args: argparse.Namespace, result_path: Path, stage: dict[str, str]) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("the main smoke test requires cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    seed_everything(SEED)
    stage["value"] = "main: train Dataset and sampler construction"
    dataset, initial_train_guard = create_train_only_smoke_dataset(
        project_root=PROJECT_ROOT,
        cache_dir=args.cache_dir,
        split="train",
        index_filename=APPROVED_TRAIN_INDEX_FILENAME,
        max_cached_shards=8,
        validate_finite=False,
    )
    cache_dir = dataset.cache_dir
    sampler = create_sampler(dataset)
    first_indexes = batch_at_position(sampler, 0)
    expected_second_indexes = batch_at_position(create_sampler(dataset), 1)
    expected_second_identity = batch_identity(dataset, expected_second_indexes)
    selected_main_shards = sorted({
        dataset.rows[index].feature_shard_path
        for index in first_indexes + expected_second_indexes
    })
    identity = train_data_identity(cache_dir, selected_main_shards)
    first_batch = load_logical_batch(dataset, first_indexes, 0)
    first_validation = validate_logical_batch(
        first_batch, first_indexes, MAIN_MICROBATCH_SIZE
    )

    stage["value"] = "main: initial SpeechBrain/AAM/optimizer construction"
    objects = create_training_objects(device)
    loader_proof_step1 = objects["loader_proof"]
    step1 = run_optimizer_step(
        objects,
        first_batch,
        microbatch_size=MAIN_MICROBATCH_SIZE,
        optimizer_step=1,
        stage=stage,
    )

    stage["value"] = "main: checkpoint construction and atomic save"
    checkpoint = build_checkpoint(objects, identity, expected_second_identity)
    checkpoint_path = args.checkpoint
    if checkpoint_path.is_absolute():
        checkpoint_path.resolve().relative_to(PROJECT_ROOT)
    else:
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    atomic_save_checkpoint(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_v1(loaded)
    whole_checkpoint_roundtrip_exact = values_exactly_equal(checkpoint, loaded)
    if not whole_checkpoint_roundtrip_exact:
        raise AssertionError("atomic checkpoint file does not exactly match saved state")
    checkpoint_size = checkpoint_path.stat().st_size

    stage["value"] = "main: release step-1 training objects"
    del objects, first_batch, dataset, sampler, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    stage["value"] = "main: fresh reconstruction and checkpoint load"
    fresh_objects, roundtrip_checks = load_checkpoint_into_fresh_objects(loaded, device)
    loader_proof_resume = fresh_objects["loader_proof"]

    stage["value"] = "main: deterministic resumed batch positioning"
    resumed_dataset, resumed_train_guard = create_train_only_smoke_dataset(
        project_root=PROJECT_ROOT,
        cache_dir=cache_dir,
        split="train",
        index_filename=APPROVED_TRAIN_INDEX_FILENAME,
        max_cached_shards=8,
        validate_finite=False,
    )
    resumed_sampler = create_sampler(resumed_dataset)
    resumed_indexes = batch_at_position(
        resumed_sampler, loaded["next_logical_batch_position"]
    )
    resumed_identity = batch_identity(resumed_dataset, resumed_indexes)
    if resumed_identity != loaded["next_logical_batch_identity"]:
        raise AssertionError("resumed logical batch identity differs from checkpoint expectation")
    if resumed_indexes != expected_second_indexes:
        raise AssertionError("resumed logical batch differs from originally expected batch 2")
    second_batch = load_logical_batch(resumed_dataset, resumed_indexes, 1)
    second_validation = validate_logical_batch(
        second_batch, resumed_indexes, MAIN_MICROBATCH_SIZE
    )
    step2 = run_optimizer_step(
        fresh_objects,
        second_batch,
        microbatch_size=MAIN_MICROBATCH_SIZE,
        optimizer_step=2,
        stage=stage,
    )
    global_optimizer_step = int(loaded["global_optimizer_step"]) + 1
    if global_optimizer_step != 2:
        raise AssertionError("resumed global optimizer step did not become 2")

    result = {
        "result": "PASS",
        "mode": "main_microbatch_2",
        "environment": environment_payload(device),
        "scope": {
            "train_cache_only": True,
            "logical_batches_consumed": 2,
            "optimizer_steps": 2,
            "physical_microbatch_size": 2,
            "accumulation_steps": 16,
            "num_workers": 0,
            "augmentation": False,
            "validation_access": False,
            "final_test_access": False,
            "recursive_listing_used": False,
        },
        "train_only_guards": {
            "initial_dataset": initial_train_guard,
            "resumed_dataset": resumed_train_guard,
        },
        "train_shards_loaded": selected_main_shards,
        "model_path": (
            "cached Fbank [B, 301, 80] -> mean_var_norm -> embedding_model "
            "-> [B, 1, 192] -> squeeze(1) -> [B, 192]"
        ),
        "forbidden_call_proof": {
            "compute_features_calls": 0,
            "pretrained_classifier_calls": 0,
            "step1_loader": loader_proof_step1,
            "resume_loader": loader_proof_resume,
        },
        "train_data_identity": identity,
        "sampler": loaded["sampler"],
        "aam": loaded["aam"],
        "optimizer_groups": loaded["optimizer_groups"],
        "batchnorm_policy": loaded["batchnorm_policy"],
        "amp_policy": loaded["amp_policy"],
        "logical_batch_1": first_validation,
        "step_1": step1,
        "checkpoint": {
            "relative_path": checkpoint_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": checkpoint_size,
            "atomic_save": True,
            "whole_payload_roundtrip_exact": whole_checkpoint_roundtrip_exact,
            "schema_valid": True,
            "state_roundtrip": roundtrip_checks,
        },
        "resume": {
            "fresh_objects_constructed": True,
            "rng_restored": True,
            "sampler_recreated": True,
            "saved_next_position": 1,
            "resumed_identity_exact": True,
            "originally_expected_second_batch_exact": True,
            "global_optimizer_step": global_optimizer_step,
        },
        "logical_batch_2": second_validation,
        "step_2": step2,
        "main_max_memory_allocated_bytes": max(
            step1["max_memory_allocated_bytes"], step2["max_memory_allocated_bytes"]
        ),
        "main_max_memory_reserved_bytes": max(
            step1["max_memory_reserved_bytes"], step2["max_memory_reserved_bytes"]
        ),
    }
    stage["value"] = "main: write result"
    atomic_json(result_path, result)
    return result


def run_probe(args: argparse.Namespace, result_path: Path, stage: dict[str, str]) -> dict[str, Any]:
    main_result_path = (
        args.main_result if args.main_result.is_absolute() else PROJECT_ROOT / args.main_result
    )
    main_result = json.loads(main_result_path.read_text(encoding="utf-8"))
    if main_result.get("result") != "PASS":
        raise RuntimeError("microbatch-4 probe requires a successful main result")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in (None, 0) or not torch.cuda.is_available():
        raise RuntimeError("the capacity probe requires cuda:0")
    seed_everything(SEED)
    stage["value"] = "probe: train Dataset and first sampler batch"
    dataset, train_guard = create_train_only_smoke_dataset(
        project_root=PROJECT_ROOT,
        cache_dir=args.cache_dir,
        split="train",
        index_filename=APPROVED_TRAIN_INDEX_FILENAME,
        max_cached_shards=8,
        validate_finite=False,
    )
    indexes = batch_at_position(create_sampler(dataset), 0)
    selected_probe_shards = sorted({
        dataset.rows[index].feature_shard_path for index in indexes
    })
    identity = train_data_identity(dataset.cache_dir, selected_probe_shards)
    logical_batch = load_logical_batch(dataset, indexes, 0)
    validation = validate_logical_batch(logical_batch, indexes, 4)
    stage["value"] = "probe: fresh SpeechBrain/AAM/optimizer construction"
    objects = create_training_objects(device)
    loader_proof = objects["loader_proof"]
    step = run_optimizer_step(
        objects, logical_batch, microbatch_size=4, optimizer_step=1, stage=stage
    )
    result = {
        "result": "PASS",
        "mode": "microbatch_4_capacity_probe",
        "environment": environment_payload(device),
        "scope": {
            "train_cache_only": True,
            "logical_batches_consumed": 1,
            "optimizer_steps": 1,
            "physical_microbatch_size": 4,
            "accumulation_steps": 8,
            "num_workers": 0,
            "checkpoint_or_resume": False,
            "validation_access": False,
            "final_test_access": False,
            "recursive_listing_used": False,
        },
        "train_only_guard": train_guard,
        "train_data_identity": identity,
        "train_shards_loaded": selected_probe_shards,
        "forbidden_call_proof": {
            "compute_features_calls": 0,
            "pretrained_classifier_calls": 0,
            "loader": loader_proof,
        },
        "logical_batch": validation,
        "step": step,
    }
    stage["value"] = "probe: write result"
    atomic_json(result_path, result)
    return result


def safe_error(error: BaseException) -> str:
    return str(error).replace(str(PROJECT_ROOT), ".").replace(str(PROJECT_ROOT).lower(), ".")


def main() -> None:
    args = parse_args()
    result_path = args.result_json or (
        MAIN_RESULT_PATH if args.mode == "main" else PROBE_RESULT_PATH
    )
    if not result_path.is_absolute():
        result_path = PROJECT_ROOT / result_path
    stage = {"value": "startup"}
    try:
        result = (
            run_main(args, result_path, stage)
            if args.mode == "main"
            else run_probe(args, result_path, stage)
        )
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            outcome = "FAIL" if args.mode == "main" else "OOM"
            payload = {
                "result": outcome,
                "mode": args.mode,
                "exact_stage": stage["value"],
                "error": safe_error(error),
            }
            atomic_json(result_path, payload)
            print(
                f"{'MAIN MICROBATCH-2' if args.mode == 'main' else 'MICROBATCH-4 CAPACITY PROBE'}: "
                f"{outcome} (CUDA OOM at {stage['value']})",
                flush=True,
            )
            raise SystemExit(1 if args.mode == "main" else 0)
        raise
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(
        "MAIN MICROBATCH-2 SMOKE TEST: PASS"
        if args.mode == "main"
        else "MICROBATCH-4 CAPACITY PROBE: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
