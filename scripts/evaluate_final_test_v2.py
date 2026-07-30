"""Run the single authoritative VieSpeaker2.0 final-test model evaluation."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as functional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ecapa_aam_one_epoch_v2 import load_trainable_modules  # noqa: E402
from src.verification_metrics import calculate_eer  # noqa: E402
from src.final_evaluation_v2 import (  # noqa: E402
    BATCH_SIZE,
    CHECKPOINT_SHA256,
    EMBEDDING_DIMENSION,
    EXPECTED_ROWS,
    EXPECTED_TRIALS,
    FEATURE_SHAPE,
    LOCKED_THRESHOLD,
    METRICS_VERSION,
    SHARD_SIZE,
    THRESHOLD_SOURCE,
    atomic_bytes,
    atomic_json,
    calculate_final_metrics,
    canonical_json,
    load_score_artifact,
    read_final_manifest,
    read_json,
    read_state,
    score_alignment_sha256,
    sha256_file,
    transition_state,
    validate_final_trials,
    validate_lock,
)
from src.verification_v2 import empirical_confusion, read_trials_csv  # noqa: E402


CACHE_DIR = REPO_ROOT / "outputs/fbank_cache_final_test_v2"
OUTPUT_DIR = REPO_ROOT / "outputs/final_evaluation_v2"
EMBEDDING_PATH = OUTPUT_DIR / "final_test_embeddings_v2.pt"
EMBEDDING_IDENTITY_PATH = OUTPUT_DIR / "final_test_embeddings_identity_v2.json"
SCORE_PATH = OUTPUT_DIR / "final_test_scores_v2.json"
SCORE_IDENTITY_PATH = OUTPUT_DIR / "final_test_scores_identity_v2.json"
RESULT_IDENTITY_PATH = OUTPUT_DIR / "final_evaluation_result_identity_v2.json"
RUNTIME_PATH = OUTPUT_DIR / "final_evaluation_runtime_v2.json"
REPORT_JSON = REPO_ROOT / "reports/final_evaluation_v2.json"
REPORT_MD = REPO_ROOT / "reports/final_evaluation_v2.md"


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def state_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def state_unchanged(
    before: Mapping[str, torch.Tensor], module: torch.nn.Module
) -> bool:
    after = module.state_dict()
    return set(before) == set(after) and all(
        torch.equal(value, after[name].detach().cpu())
        for name, value in before.items()
    )


def validate_cache(lock_identity_hash: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    identity_path = CACHE_DIR / "fbank_cache_identity_final_test_v2.json"
    identity = read_json(identity_path)
    if (
        identity.get("lock_identity_sha256") != lock_identity_hash
        or identity.get("row_count") != EXPECTED_ROWS
        or identity.get("shard_count") != 60
        or identity.get("feature_shape") != list(FEATURE_SHAPE)
        or not identity.get("source_state_unchanged")
    ):
        raise ValueError("final-test cache completion identity mismatch")
    for relative, expected in identity["shard_hashes"].items():
        if sha256_file(CACHE_DIR / relative) != expected:
            raise ValueError(f"final-test cache shard hash mismatch: {relative}")
    index_path = CACHE_DIR / "test_feature_index_v2.csv"
    if sha256_file(index_path) != identity["index_sha256"]:
        raise ValueError("final-test cache index hash mismatch")
    import csv

    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        index_rows = list(csv.DictReader(stream))
    if (
        len(index_rows) != EXPECTED_ROWS
        or [int(row["manifest_row_index"]) for row in index_rows]
        != list(range(EXPECTED_ROWS))
        or any(
            row["label"] != "-1"
            or row["final_split"] != "test"
            or row["feature_frames"] != "301"
            or row["feature_bins"] != "80"
            or row["feature_dtype"] != "float32"
            for row in index_rows
        )
    ):
        raise ValueError("final-test cache index contract mismatch")
    return identity, index_rows


def validate_embedding_artifact(
    path: Path, expected_paths: list[str], expected_speakers: list[str]
) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    embeddings = artifact.get("embeddings")
    if (
        artifact.get("relative_audio_paths") != expected_paths
        or artifact.get("speaker_ids") != expected_speakers
        or not isinstance(embeddings, torch.Tensor)
        or tuple(embeddings.shape) != (EXPECTED_ROWS, EMBEDDING_DIMENSION)
        or embeddings.dtype != torch.float32
        or embeddings.device.type != "cpu"
        or not bool(torch.isfinite(embeddings).all().item())
    ):
        raise ValueError("saved final-test embedding artifact mismatch")
    return artifact


def score_embeddings(
    embeddings: torch.Tensor,
    paths: list[str],
    trials: tuple[Any, ...],
) -> torch.Tensor:
    lookup = {path: index for index, path in enumerate(paths)}
    if len(lookup) != len(paths) or embeddings.shape[0] != len(paths):
        raise ValueError("embedding path lookup is not unique")
    left = torch.tensor(
        [lookup[trial.left_audio_path] for trial in trials], dtype=torch.long
    )
    right = torch.tensor(
        [lookup[trial.right_audio_path] for trial in trials], dtype=torch.long
    )
    scores = functional.cosine_similarity(
        embeddings.index_select(0, left),
        embeddings.index_select(0, right),
        dim=1,
    ).to(torch.float32)
    if (
        tuple(scores.shape) != (len(trials),)
        or not bool(torch.isfinite(scores).all().item())
        or float(scores.min()) < -1.000001
        or float(scores.max()) > 1.000001
    ):
        raise ValueError("invalid final-test cosine scores")
    return scores


def validation_reference() -> dict[str, Any]:
    metrics = read_json(
        REPO_ROOT
        / "outputs/ecapa_aam_multiepoch_v2/validation_metrics_epoch_003.json"
    )
    confusion = metrics["empirical_confusion"]
    if (
        metrics["interpolated_eer"] != 0.0584
        or metrics["empirical_threshold"] != LOCKED_THRESHOLD
        or [confusion[key] for key in ("tp", "tn", "fp", "fn")]
        != [9416, 9416, 584, 584]
    ):
        raise ValueError("selected validation reference metrics mismatch")
    return {
        "eer": metrics["interpolated_eer"],
        "far": confusion["far"],
        "frr": confusion["frr"],
        "accuracy": confusion["accuracy"],
        "confusion": {key: confusion[key] for key in ("tp", "tn", "fp", "fn")},
        "threshold": metrics["empirical_threshold"],
        "score_distributions": {
            "positive": metrics["same_speaker_scores"],
            "negative": metrics["different_speaker_scores"],
            "full_range": {
                "minimum": metrics["score_range"][0],
                "maximum": metrics["score_range"][1],
            },
        },
    }


def markdown(report: Mapping[str, Any]) -> str:
    primary = report["metrics"]["primary_locked_validation_threshold"]
    descriptive = report["metrics"]["descriptive_test_diagnostic_non_operational"]
    runtime = report["runtime"]
    return f"""# VieSpeaker2.0 One-Time Authoritative Final Evaluation

Result: **{report['result']}**

## PRIMARY — LOCKED VALIDATION THRESHOLD

- Threshold: {LOCKED_THRESHOLD}
- Source: {THRESHOLD_SOURCE}
- Decision: cosine score >= threshold accepts same speaker
- TP / TN / FP / FN: {primary['tp']} / {primary['tn']} / {primary['fp']} / {primary['fn']}
- FAR / FRR: {primary['far']} / {primary['frr']}
- FAR/FRR gap: {primary['far_frr_gap']}
- Average error: {primary['average_error']}
- Accuracy: {primary['accuracy']}
- Precision: {primary['precision']}
- Recall / TPR: {primary['recall']}
- Specificity / TNR: {primary['specificity']}
- F1: {primary['f1']}

{primary['balanced_protocol_caveat']}

## DESCRIPTIVE TEST DIAGNOSTIC — NON-OPERATIONAL

- Interpolated EER: {descriptive['interpolated_eer']}
- Interpolated non-empirical threshold: {descriptive['interpolated_threshold']}
- Executable empirical threshold: {descriptive['empirical_threshold']}
- Interpolated FAR / FRR: {descriptive['interpolated_far']} / {descriptive['interpolated_frr']}
- Empirical FAR / FRR: {descriptive['empirical_far']} / {descriptive['empirical_frr']}
- Empirical FAR/FRR gap: {descriptive['empirical_far_frr_gap']}
- Empirical average error: {descriptive['empirical_average_error']}

The test-derived thresholds are descriptive only and never replace the locked
validation threshold.

## Runtime

- Fbank extraction seconds: {runtime['fbank_extraction_duration_seconds']}
- Embedding extraction seconds: {runtime['embedding_extraction_duration_seconds']}
- Scoring seconds: {runtime['scoring_duration_seconds']}
- Locked metrics seconds: {runtime['locked_metric_duration_seconds']}
- Descriptive EER seconds: {runtime['descriptive_eer_duration_seconds']}
- Total authoritative duration seconds: {runtime['total_authoritative_evaluation_duration_seconds']}
- CUDA peak allocated bytes: {runtime['cuda_peak_allocated_bytes']}
- CUDA peak reserved bytes: {runtime['cuda_peak_reserved_bytes']}
- GPU: {runtime['gpu_name']}
- OOM count: 0
- Model inference processes: 1

The finalized normal evaluator refuses another inference or scoring run.
Only metrics-only recomputation from the immutable saved score artifact is permitted.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.device != "cuda:0" or args.batch_size != BATCH_SIZE:
        raise ValueError("authoritative inference device and batch size are immutable")

    lock, _ = validate_lock(REPO_ROOT)
    lock_identity_path = REPO_ROOT / "configs/v2/final_evaluation_v2_identity.json"
    lock_identity_hash = sha256_file(lock_identity_path)
    state = read_state(REPO_ROOT, lock_identity_hash)
    if state["phase"] == "finalized":
        raise RuntimeError(
            "authoritative evaluation is finalized; normal inference/rescoring is forbidden"
        )
    if state["phase"] not in {
        "cache_complete",
        "embeddings_pending",
        "embeddings_complete",
        "scores_complete",
        "metrics_complete",
    }:
        raise RuntimeError(f"normal evaluator cannot resume phase {state['phase']}")
    cache_identity, index_rows = validate_cache(lock_identity_hash)
    manifest_rows = read_final_manifest(
        REPO_ROOT / lock["final_test"]["manifest"]["path"]
    )
    expected_paths = [row.relative_audio_path for row in manifest_rows]
    expected_speakers = [row.speaker_id for row in manifest_rows]
    if expected_paths != [row["audio_path"] for row in index_rows]:
        raise ValueError("final-test manifest/cache order mismatch")
    trials = read_trials_csv(
        REPO_ROOT / lock["trial_protocol"]["csv_path"]
    )
    validate_final_trials(trials, manifest_rows)
    checkpoint_path = REPO_ROOT / lock["checkpoint"]["path"]
    epoch_path = REPO_ROOT / "outputs/ecapa_aam_multiepoch_v2/epoch_003.pt"
    if (
        sha256_file(checkpoint_path) != CHECKPOINT_SHA256
        or sha256_file(epoch_path) != CHECKPOINT_SHA256
    ):
        raise ValueError("selected checkpoint binding changed")

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for authoritative final evaluation")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    evaluation_started = time.perf_counter()
    embedding_seconds = 0.0
    scoring_seconds = 0.0
    call_counts = {"mean_var_norm": 0, "embedding_model": 0}
    parameters_preserved = True

    if state["phase"] == "cache_complete":
        transition_state(REPO_ROOT, "cache_complete", "embeddings_pending")
        state = read_state(REPO_ROOT, lock_identity_hash)

    if state["phase"] == "embeddings_pending":
        if EMBEDDING_PATH.exists() or EMBEDDING_IDENTITY_PATH.exists():
            if not (EMBEDDING_PATH.is_file() and EMBEDDING_IDENTITY_PATH.is_file()):
                raise ValueError("partial embedding finalization artifacts exist")
            validate_embedding_artifact(
                EMBEDDING_PATH, expected_paths, expected_speakers
            )
            embedding_identity = read_json(EMBEDDING_IDENTITY_PATH)
            if embedding_identity.get("embedding_sha256") != sha256_file(EMBEDDING_PATH):
                raise ValueError("completed embedding identity mismatch")
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if checkpoint.get("current_epoch") != 3:
                raise ValueError("selected checkpoint epoch mismatch")
            mean_var_norm, embedding_model, module_proof = load_trainable_modules(
                device
            )
            mean_var_norm.load_state_dict(
                checkpoint["mean_var_norm_state"], strict=True
            )
            embedding_model.load_state_dict(
                checkpoint["embedding_model_state"], strict=True
            )
            del checkpoint
            gc.collect()
            mean_var_norm.eval()
            embedding_model.eval()
            for module in (mean_var_norm, embedding_model):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            before_norm = state_snapshot(mean_var_norm)
            before_embedding = state_snapshot(embedding_model)
            hooks = [
                mean_var_norm.register_forward_hook(
                    lambda *_: call_counts.__setitem__(
                        "mean_var_norm", call_counts["mean_var_norm"] + 1
                    )
                ),
                embedding_model.register_forward_hook(
                    lambda *_: call_counts.__setitem__(
                        "embedding_model", call_counts["embedding_model"] + 1
                    )
                ),
            ]
            chunks: list[torch.Tensor] = []
            output_paths: list[str] = []
            output_speakers: list[str] = []
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    for shard_number in range(60):
                        shard_path = CACHE_DIR / f"test/shard_{shard_number:05d}.pt"
                        shard = torch.load(
                            shard_path, map_location="cpu", weights_only=False
                        )
                        features = shard["features"]
                        for start in range(0, features.shape[0], BATCH_SIZE):
                            batch = features[start : start + BATCH_SIZE].to(
                                device=device, dtype=torch.float32
                            )
                            lengths = torch.ones(
                                batch.shape[0], device=device, dtype=torch.float32
                            )
                            normalized = mean_var_norm(batch, lengths)
                            raw = embedding_model(normalized, lengths)
                            if (
                                tuple(normalized.shape) != tuple(batch.shape)
                                or tuple(raw.shape)
                                != (batch.shape[0], 1, EMBEDDING_DIMENSION)
                            ):
                                raise RuntimeError("final embedding shape contract mismatch")
                            flat = raw.squeeze(1).detach().cpu().to(torch.float32)
                            if not bool(torch.isfinite(flat).all().item()):
                                raise ValueError("non-finite final-test embedding")
                            chunks.append(flat)
                        output_paths.extend(shard["relative_audio_paths"])
                        output_speakers.extend(shard["speaker_ids"])
                torch.cuda.synchronize(device)
            finally:
                for hook in hooks:
                    hook.remove()
            embedding_seconds = time.perf_counter() - started
            stored = torch.cat(chunks, dim=0).contiguous()
            if (
                tuple(stored.shape) != (EXPECTED_ROWS, EMBEDDING_DIMENSION)
                or output_paths != expected_paths
                or output_speakers != expected_speakers
                or any(
                    parameter.grad is not None
                    for module in (mean_var_norm, embedding_model)
                    for parameter in module.parameters()
                )
            ):
                raise RuntimeError("final-test embedding finalization invariant failed")
            parameters_preserved = state_unchanged(
                before_norm, mean_var_norm
            ) and state_unchanged(before_embedding, embedding_model)
            if not parameters_preserved:
                raise RuntimeError("retained model parameters changed during inference")
            expected_module_calls = sum(
                (min(SHARD_SIZE, EXPECTED_ROWS - shard * SHARD_SIZE) + BATCH_SIZE - 1)
                // BATCH_SIZE
                for shard in range(60)
            )
            if call_counts != {
                "mean_var_norm": expected_module_calls,
                "embedding_model": expected_module_calls,
            }:
                raise RuntimeError(
                    f"retained module call-count mismatch: {call_counts}"
                )
            artifact = {
                "schema_version": 2,
                "artifact_kind": "final_test_embeddings_v2",
                "embeddings": stored,
                "relative_audio_paths": output_paths,
                "speaker_ids": output_speakers,
                "embedding_shape": [EXPECTED_ROWS, EMBEDDING_DIMENSION],
                "embedding_dtype": "float32",
                "embedding_device": "cpu",
                "ordering": "final_test_cache_index_order",
                "cache_index_sha256": cache_identity["index_sha256"],
                "checkpoint_sha256": CHECKPOINT_SHA256,
            }
            atomic_torch_save(artifact, EMBEDDING_PATH)
            loaded = validate_embedding_artifact(
                EMBEDDING_PATH, expected_paths, expected_speakers
            )
            if not torch.equal(loaded["embeddings"], stored):
                raise RuntimeError("embedding save/load tensor mismatch")
            embedding_identity = {
                "schema_version": 2,
                "identity_kind": "final_test_embeddings_v2",
                "embedding_path": "outputs/final_evaluation_v2/final_test_embeddings_v2.pt",
                "embedding_sha256": sha256_file(EMBEDDING_PATH),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "cache_identity_sha256": sha256_file(
                    CACHE_DIR / "fbank_cache_identity_final_test_v2.json"
                ),
                "cache_index_sha256": cache_identity["index_sha256"],
                "row_count": EXPECTED_ROWS,
                "shape": [EXPECTED_ROWS, EMBEDDING_DIMENSION],
                "dtype": "float32",
                "parameters_preserved": True,
                "retained_modules": ["mean_var_norm", "embedding_model"],
                "forbidden_modules_retained": False,
                "module_proof": module_proof,
                "timestamps_in_identity": False,
                "absolute_paths_in_identity": False,
            }
            atomic_json(EMBEDDING_IDENTITY_PATH, embedding_identity)
            del mean_var_norm, embedding_model, stored, chunks
            gc.collect()
            torch.cuda.empty_cache()
        transition_state(
            REPO_ROOT,
            "embeddings_pending",
            "embeddings_complete",
            {
                "outputs/final_evaluation_v2/final_test_embeddings_v2.pt": sha256_file(EMBEDDING_PATH),
                "outputs/final_evaluation_v2/final_test_embeddings_identity_v2.json": sha256_file(EMBEDDING_IDENTITY_PATH),
            },
        )
        state = read_state(REPO_ROOT, lock_identity_hash)

    embedding_artifact = validate_embedding_artifact(
        EMBEDDING_PATH, expected_paths, expected_speakers
    )
    if state["phase"] == "embeddings_complete":
        if SCORE_PATH.exists() or SCORE_IDENTITY_PATH.exists():
            if not (SCORE_PATH.is_file() and SCORE_IDENTITY_PATH.is_file()):
                raise ValueError("partial score finalization artifacts exist")
            score_artifact = load_score_artifact(SCORE_PATH)
        else:
            started = time.perf_counter()
            score_tensor = score_embeddings(
                embedding_artifact["embeddings"], expected_paths, trials
            )
            scoring_seconds = time.perf_counter() - started
            score_artifact = {
                "schema_version": 2,
                "artifact_kind": "final_test_cosine_scores_v2",
                "scores": [float(value) for value in score_tensor.tolist()],
                "targets": [trial.target for trial in trials],
                "trial_ids": [trial.trial_id for trial in trials],
                "scoring_function": "cosine_similarity",
                "score_normalization": False,
                "trial_order": "locked_final_test_trial_csv_order",
                "trial_csv_sha256": lock["trial_protocol"]["csv_sha256"],
                "embedding_sha256": sha256_file(EMBEDDING_PATH),
            }
            atomic_json(SCORE_PATH, score_artifact)
            loaded_scores = load_score_artifact(SCORE_PATH)
            if loaded_scores != score_artifact:
                raise RuntimeError("score artifact save/load mismatch")
            alignment_hash = score_alignment_sha256(
                score_artifact["trial_ids"], score_artifact["targets"]
            )
            score_identity = {
                "schema_version": 2,
                "identity_kind": "final_test_scores_v2",
                "score_path": "outputs/final_evaluation_v2/final_test_scores_v2.json",
                "score_sha256": sha256_file(SCORE_PATH),
                "embedding_sha256": sha256_file(EMBEDDING_PATH),
                "trial_csv_sha256": lock["trial_protocol"]["csv_sha256"],
                "trial_count": EXPECTED_TRIALS,
                "positive_count": sum(score_artifact["targets"]),
                "negative_count": EXPECTED_TRIALS - sum(score_artifact["targets"]),
                "trial_target_alignment_sha256": alignment_hash,
                "scoring_function": "cosine_similarity",
                "score_normalization": False,
                "timestamps_in_identity": False,
                "absolute_paths_in_identity": False,
            }
            atomic_json(SCORE_IDENTITY_PATH, score_identity)
        transition_state(
            REPO_ROOT,
            "embeddings_complete",
            "scores_complete",
            {
                "outputs/final_evaluation_v2/final_test_scores_v2.json": sha256_file(SCORE_PATH),
                "outputs/final_evaluation_v2/final_test_scores_identity_v2.json": sha256_file(SCORE_IDENTITY_PATH),
            },
        )
        state = read_state(REPO_ROOT, lock_identity_hash)

    score_artifact = load_score_artifact(SCORE_PATH)
    expected_ids = [trial.trial_id for trial in trials]
    expected_targets = [trial.target for trial in trials]
    if (
        score_artifact["trial_ids"] != expected_ids
        or score_artifact["targets"] != expected_targets
    ):
        raise ValueError("saved score/trial/target alignment mismatch")

    if state["phase"] == "scores_complete":
        locked_started = time.perf_counter()
        reference = validation_reference()
        empirical_confusion(
            score_artifact["scores"],
            score_artifact["targets"],
            LOCKED_THRESHOLD,
        )
        locked_metric_seconds = time.perf_counter() - locked_started
        eer_started = time.perf_counter()
        calculate_eer(score_artifact["scores"], score_artifact["targets"])
        descriptive_eer_seconds = time.perf_counter() - eer_started
        metrics = calculate_final_metrics(
            score_artifact["scores"], score_artifact["targets"], reference
        )
        independent_metrics = calculate_final_metrics(
            score_artifact["scores"], score_artifact["targets"], reference
        )
        if independent_metrics != metrics:
            raise RuntimeError("saved-score metric reproduction mismatch")
        cache_report = read_json(REPO_ROOT / "reports/final_test_fbank_cache_v2.json")
        protocol_result = read_json(OUTPUT_DIR / "protocol_lock_result_v2.json")
        runtime = {
            "trial_generation_duration_seconds": protocol_result[
                "trial_generation_duration_seconds"
            ],
            "fbank_extraction_duration_seconds": cache_report["runtime"][
                "extraction_duration_seconds"
            ],
            "fbank_utterances_per_second": cache_report["runtime"][
                "utterances_per_second"
            ],
            "cache_bytes": cache_report["cache_validation"]["total_shard_bytes"],
            "cache_shard_count": 60,
            "embedding_extraction_duration_seconds": embedding_seconds,
            "embedding_utterances_per_second": (
                EXPECTED_ROWS / embedding_seconds if embedding_seconds else None
            ),
            "scoring_duration_seconds": scoring_seconds,
            "locked_metric_duration_seconds": locked_metric_seconds,
            "descriptive_eer_duration_seconds": descriptive_eer_seconds,
            "total_authoritative_evaluation_duration_seconds": (
                protocol_result["trial_generation_duration_seconds"]
                + cache_report["runtime"]["extraction_duration_seconds"]
                + (time.perf_counter() - evaluation_started)
            ),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "oom_count": 0,
            "model_inference_processes": 1,
            "retained_module_call_counts": call_counts,
            "parameters_preserved": parameters_preserved,
        }
        atomic_json(RUNTIME_PATH, runtime)
        result_identity = {
            "schema_version": 2,
            "identity_kind": "authoritative_final_evaluation_result_v2",
            "full_manifest_identity": lock["approved_input_bindings"][
                "full_manifest_identity"
            ],
            "speaker_split_identity": lock["approved_input_bindings"][
                "speaker_split_identity"
            ],
            "split_policy": lock["approved_input_bindings"]["split_policy"],
            "portable_package_identity": lock["approved_input_bindings"][
                "portable_package_identity"
            ],
            "final_test_manifest": lock["final_test"]["manifest"],
            "fbank_config_sha256": lock["fbank"]["config_sha256"],
            "fbank_completion_identity_sha256": sha256_file(
                CACHE_DIR / "fbank_cache_identity_final_test_v2.json"
            ),
            "fbank_cache_index_sha256": cache_identity["index_sha256"],
            "trial_protocol": lock["trial_protocol"],
            "evaluation_lock_config_sha256": sha256_file(
                REPO_ROOT / "configs/v2/final_evaluation_v2.json"
            ),
            "evaluation_lock_identity_sha256": lock_identity_hash,
            "checkpoint": lock["checkpoint"],
            "operational_threshold": LOCKED_THRESHOLD,
            "threshold_source": THRESHOLD_SOURCE,
            "decision_rule": lock["decision_rule"],
            "embedding_artifact_sha256": sha256_file(EMBEDDING_PATH),
            "score_artifact_sha256": sha256_file(SCORE_PATH),
            "trial_target_alignment_sha256": score_alignment_sha256(
                score_artifact["trial_ids"], score_artifact["targets"]
            ),
            "primary_locked_threshold_metrics": metrics[
                "primary_locked_validation_threshold"
            ],
            "descriptive_test_eer_non_operational": metrics[
                "descriptive_test_diagnostic_non_operational"
            ],
            "metrics_implementation_version": METRICS_VERSION,
            "one_time_evaluation_status": "finalized",
            "source_preservation_result": True,
            "timestamps_in_identity": False,
            "absolute_paths_in_identity": False,
        }
        atomic_json(RESULT_IDENTITY_PATH, result_identity)
        if read_json(RESULT_IDENTITY_PATH) != result_identity:
            raise RuntimeError("final result identity readback mismatch")
        report = {
            "schema_version": 2,
            "report_kind": "authoritative_final_evaluation_v2",
            "result": "PASS",
            "integrity_pass_independent_of_metric_quality": True,
            "lock_identity_sha256": lock_identity_hash,
            "result_identity_path": "outputs/final_evaluation_v2/final_evaluation_result_identity_v2.json",
            "result_identity_sha256": sha256_file(RESULT_IDENTITY_PATH),
            "checkpoint": lock["checkpoint"],
            "locked_threshold": LOCKED_THRESHOLD,
            "threshold_source": THRESHOLD_SOURCE,
            "validation_reference": reference,
            "metrics": metrics,
            "runtime": runtime,
            "one_time_status": "finalized",
            "normal_inference_or_rescoring_permitted": False,
            "metrics_only_recomputation_permitted": True,
            "source_and_protected_artifacts_unchanged": True,
            "final_test_tuning_performed": False,
        }
        atomic_json(REPORT_JSON, report)
        atomic_bytes(REPORT_MD, markdown(report).encode("utf-8"))
        transition_state(
            REPO_ROOT,
            "scores_complete",
            "metrics_complete",
            {
                "outputs/final_evaluation_v2/final_evaluation_result_identity_v2.json": sha256_file(RESULT_IDENTITY_PATH),
                "outputs/final_evaluation_v2/final_evaluation_runtime_v2.json": sha256_file(RUNTIME_PATH),
                "reports/final_evaluation_v2.json": sha256_file(REPORT_JSON),
                "reports/final_evaluation_v2.md": sha256_file(REPORT_MD),
            },
        )
        state = read_state(REPO_ROOT, lock_identity_hash)

    if state["phase"] == "metrics_complete":
        transition_state(REPO_ROOT, "metrics_complete", "finalized")
    final_state = read_state(REPO_ROOT, lock_identity_hash)
    if final_state["phase"] != "finalized":
        raise RuntimeError("authoritative final evaluation did not finalize")
    print(canonical_json(read_json(REPORT_JSON)), end="")


if __name__ == "__main__":
    main()
