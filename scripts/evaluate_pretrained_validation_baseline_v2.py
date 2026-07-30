#!/usr/bin/env python
"""Evaluate the untouched pretrained ECAPA model on fixed v2 validation trials."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import speechbrain  # noqa: E402
import torch  # noqa: E402

from src.cached_fbank_dataset import (  # noqa: E402
    CachedFbankDataset,
    create_cached_fbank_dataloader,
)
from src.speechbrain_frontend import SpeechBrainECAPAFrontend  # noqa: E402
from src.training_readiness_v2 import (  # noqa: E402
    APPROVED_INPUTS,
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_file,
    verify_approved_inputs,
)
from src.verification_baseline import numeric_summary, score_trials  # noqa: E402
from src.verification_metrics import calculate_eer  # noqa: E402
from src.verification_v2 import (  # noqa: E402
    empirical_confusion,
    read_trials_csv,
    read_validation_manifest,
    validate_validation_trials,
)


EXPECTED_ROWS = 15355
EXPECTED_SPEAKERS = 100
EXPECTED_TRIALS = 20000


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def parameters_equal(
    before: dict[str, torch.Tensor], module: torch.nn.Module,
) -> bool:
    return len(before) == len(tuple(module.named_parameters())) and all(
        name in before and torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in module.named_parameters()
    )


def markdown_report(result: dict[str, object]) -> bytes:
    metrics = result["metrics"]
    empirical = result["empirical_metrics"]
    same = result["score_distributions"]["same_speaker"]
    different = result["score_distributions"]["different_speaker"]
    runtime = result["runtime"]
    lines = [
        "# VieSpeaker2.0 pretrained ECAPA validation baseline",
        "",
        f"Result: **{result['result']}**",
        "",
        f"Untouched pretrained interpolated validation EER: **{metrics['interpolated_eer_percentage']:.6f}%**.",
        "",
        "## Fixed evaluation",
        "",
        f"- Validation embeddings: {result['embedding']['shape']} float32 CPU",
        f"- Trials: {result['trials']['positive']} positive / {result['trials']['negative']} negative",
        "- Score: cosine similarity",
        "- EER: grouped O(N log N) tie-aware operating points",
        "- Threshold semantics: accept same speaker when score >= threshold",
        "",
        "## Model invariants",
        "",
        "- Path: cached Fbank [B,T,80] -> mean_var_norm -> embedding_model -> [B,1,192] -> squeeze -> [B,192]",
        "- No feature transpose, compute_features call, pretrained classifier call, gradient, optimizer, AAM-Softmax, or parameter update",
        f"- Parameters unchanged: {result['model_integrity']['parameters_unchanged']}",
        "",
        "## EER and executable threshold",
        "",
        f"- Interpolated EER: {metrics['interpolated_eer']} ({metrics['interpolated_eer_percentage']:.6f}%)",
        f"- Interpolated threshold: {metrics['interpolated_threshold']}",
        f"- Interpolated FAR / FRR: {metrics['interpolated_far']} / {metrics['interpolated_frr']}",
        f"- Empirical threshold: {metrics['empirical_threshold']}",
        f"- Empirical FAR / FRR: {empirical['far']} / {empirical['frr']}",
        f"- TP / TN / FP / FN: {empirical['tp']} / {empirical['tn']} / {empirical['fp']} / {empirical['fn']}",
        f"- Accuracy / precision / recall / F1: {empirical['accuracy']} / {empirical['precision']} / {empirical['recall']} / {empirical['f1']}",
        "",
        "## Score distributions",
        "",
        f"- Same-speaker: `{json.dumps(same, sort_keys=True)}`",
        f"- Different-speaker: `{json.dumps(different, sort_keys=True)}`",
        "",
        "## CUDA runtime",
        "",
        f"- Device: {result['environment']['device']} ({result['environment']['gpu']})",
        f"- Embedding extraction: {runtime['embedding_seconds']:.6f}s ({runtime['utterances_per_second']:.6f} utterances/s)",
        f"- Trial scoring: {runtime['scoring_seconds']:.6f}s",
        f"- Metrics: {runtime['metrics_seconds']:.6f}s",
        f"- Peak allocated/reserved VRAM: {runtime['peak_allocated_vram_bytes']} / {runtime['peak_reserved_vram_bytes']} bytes",
        "",
        "Tensor artifacts and runtime identity/configuration remain under the ignored "
        "`outputs/pretrained_ecapa_validation_baseline_v2/` directory.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pretrained_ecapa_validation_baseline_v2"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size != 64:
        raise ValueError("approved baseline batch size is exactly 64")
    if args.device != "cuda:0":
        raise ValueError("approved baseline device is exactly cuda:0")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream = verify_approved_inputs(REPO_ROOT)
    sampler_path = REPO_ROOT / "configs/v2/training_sampler_v2.json"
    sampler = json.loads(sampler_path.read_text(encoding="utf-8"))
    sampler_binding = {
        "path": "configs/v2/training_sampler_v2.json",
        "sha256": sha256_file(sampler_path),
        "sampler_plan_identity_sha256": sampler["sampler_plan"][
            "sampler_plan_identity_sha256"
        ],
    }
    trial_dir = REPO_ROOT / "manifests/verification_v2"
    trial_config_path = trial_dir / "validation_trials_config_v2.json"
    trial_identity_path = trial_dir / "validation_trials_identity_v2.json"
    trial_csv_path = trial_dir / "validation_trials_v2.csv"
    trial_identity = json.loads(trial_identity_path.read_text(encoding="utf-8"))
    trial_binding = {
        "config_path": "manifests/verification_v2/validation_trials_config_v2.json",
        "config_sha256": sha256_file(trial_config_path),
        "csv_path": "manifests/verification_v2/validation_trials_v2.csv",
        "csv_sha256": sha256_file(trial_csv_path),
        "identity_path": "manifests/verification_v2/validation_trials_identity_v2.json",
        "identity_sha256": sha256_file(trial_identity_path),
    }
    if (
        trial_binding["config_sha256"] != trial_identity["trial_config_sha256"]
        or trial_binding["csv_sha256"] != trial_identity["trial_csv_sha256"]
        or trial_identity["training_sampler_binding"] != sampler_binding
    ):
        raise ValueError("validation trial identity binding mismatch")
    guarded_hashes_before = {
        **{name: value["sha256"] for name, value in upstream.items()},
        "training_sampler_config": sampler_binding["sha256"],
        "validation_trial_config": trial_binding["config_sha256"],
        "validation_trial_csv": trial_binding["csv_sha256"],
        "validation_trial_identity": trial_binding["identity_sha256"],
    }

    dataset = CachedFbankDataset(
        REPO_ROOT / "outputs/fbank_cache_v2",
        "validation",
        max_cached_shards=2,
        validate_finite=False,
    )
    if (
        len(dataset) != EXPECTED_ROWS
        or len({row.speaker_id for row in dataset.rows}) != EXPECTED_SPEAKERS
        or any(
            row.speaker_label != -1 or row.final_split != "validation"
            for row in dataset.rows
        )
    ):
        raise ValueError("validation cache invariants failed")
    manifest_rows = read_validation_manifest(
        REPO_ROOT / APPROVED_INPUTS["validation_manifest"]["path"]
    )
    manifest_metadata = [
        (row.audio_path, row.speaker_id, row.duplicate_group)
        for row in manifest_rows
    ]
    cache_metadata = [
        (row.relative_audio_path, row.speaker_id, row.duplicate_group)
        for row in dataset.rows
    ]
    if manifest_metadata != cache_metadata:
        raise ValueError("validation manifest/cache order or metadata mismatch")

    trials = read_trials_csv(trial_csv_path)
    speakers = {row.speaker_id for row in dataset.rows}
    validate_validation_trials(
        trials,
        expected_speakers=speakers,
        expected_positive=10000,
        expected_negative=10000,
        expected_positive_per_speaker=100,
        expected_negative_participation=200,
    )
    path_owner = {
        row.relative_audio_path: row.speaker_id for row in dataset.rows
    }
    for trial in trials:
        if (
            path_owner.get(trial.left_audio_path) != trial.left_speaker_id
            or path_owner.get(trial.right_audio_path) != trial.right_speaker_id
        ):
            raise ValueError("trial path/speaker ownership mismatch")
    if len(trials) != EXPECTED_TRIALS:
        raise ValueError("unexpected validation trial count")

    device = torch.device(args.device)
    frontend = SpeechBrainECAPAFrontend(device=device)
    frontend.eval()
    encoder = frontend.classifier
    encoder.eval()
    before = parameter_snapshot(encoder)
    call_counts = {"compute_features": 0, "classifier": 0}

    def count_compute_features(
        _module: torch.nn.Module, _inputs: object, _output: object,
    ) -> None:
        call_counts["compute_features"] += 1

    def count_classifier(
        _module: torch.nn.Module, _inputs: object, _output: object,
    ) -> None:
        call_counts["classifier"] += 1

    hooks = [
        encoder.mods.compute_features.register_forward_hook(count_compute_features),
        encoder.mods.classifier.register_forward_hook(count_classifier),
    ]
    loader = create_cached_fbank_dataloader(
        dataset, 64, shuffle=False, num_workers=0
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    embeddings: list[torch.Tensor] = []
    paths: list[str] = []
    output_speakers: list[str] = []
    embedding_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader):
                features = batch["fbank"].to(device=device, dtype=torch.float32)
                lengths = torch.ones(
                    features.shape[0], device=device, dtype=torch.float32
                )
                normalized = encoder.mods.mean_var_norm(features, lengths)
                raw = encoder.mods.embedding_model(normalized, lengths)
                if tuple(normalized.shape) != tuple(features.shape):
                    raise RuntimeError("mean_var_norm changed cached feature shape")
                if tuple(raw.shape) != (features.shape[0], 1, 192):
                    raise RuntimeError(
                        f"batch {batch_number} has wrong raw embedding shape"
                    )
                flat = raw.squeeze(1).detach().to(device="cpu", dtype=torch.float32)
                if tuple(flat.shape) != (features.shape[0], 192):
                    raise RuntimeError("squeezed embedding shape is invalid")
                if not bool(torch.isfinite(flat).all().item()):
                    raise ValueError("non-finite validation embedding")
                embeddings.append(flat)
                paths.extend(batch["relative_audio_path"])
                output_speakers.extend(batch["speaker_id"])
        torch.cuda.synchronize(device)
    finally:
        for hook in hooks:
            hook.remove()
    embedding_seconds = time.perf_counter() - embedding_started
    stored = torch.cat(embeddings, dim=0)
    if (
        tuple(stored.shape) != (EXPECTED_ROWS, 192)
        or stored.dtype != torch.float32
        or stored.device.type != "cpu"
        or paths != [row.relative_audio_path for row in dataset.rows]
        or output_speakers != [row.speaker_id for row in dataset.rows]
    ):
        raise RuntimeError("stored validation embedding invariant failed")
    if call_counts != {"compute_features": 0, "classifier": 0}:
        raise RuntimeError("forbidden pretrained module was called")
    unchanged = parameters_equal(before, encoder)
    if not unchanged:
        raise RuntimeError("pretrained parameters changed")
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("pretrained parameter gradient was created")

    embedding_path = output_dir / "validation_embeddings_v2.pt"
    embedding_artifact = {
        "schema_version": 2,
        "artifact_kind": "pretrained_ecapa_validation_embeddings_v2",
        "embeddings": stored,
        "relative_audio_paths": paths,
        "speaker_ids": output_speakers,
        "model_source": SpeechBrainECAPAFrontend.SOURCE,
        "feature_shape": [301, 80],
        "raw_embedding_shape": [EXPECTED_ROWS, 1, 192],
        "embedding_shape": [EXPECTED_ROWS, 192],
        "embedding_dtype": "float32",
        "embedding_device": "cpu",
        "ordering": "validation_cache_index_order",
        "validation_cache_index_sha256": upstream[
            "validation_cache_index"
        ]["sha256"],
    }
    atomic_torch_save(embedding_artifact, embedding_path)
    reloaded_embeddings = torch.load(
        embedding_path, map_location="cpu", weights_only=False
    )
    embedding_difference = float(
        (reloaded_embeddings["embeddings"] - stored).abs().max().item()
    )
    if embedding_difference != 0.0:
        raise RuntimeError("embedding save/load validation failed")

    scoring_started = time.perf_counter()
    scores = score_trials(
        reloaded_embeddings["embeddings"],
        reloaded_embeddings["relative_audio_paths"],
        trials,
    )
    scoring_seconds = time.perf_counter() - scoring_started
    targets = torch.tensor(
        [trial.target for trial in trials], dtype=torch.long
    )
    score_path = output_dir / "validation_trial_scores_v2.pt"
    score_artifact = {
        "schema_version": 2,
        "artifact_kind": "pretrained_ecapa_validation_trial_scores_v2",
        "trial_ids": [trial.trial_id for trial in trials],
        "targets": targets,
        "scores": scores,
        "score_kind": "cosine_similarity",
        "trial_csv_sha256": trial_binding["csv_sha256"],
    }
    atomic_torch_save(score_artifact, score_path)
    reloaded_scores = torch.load(
        score_path, map_location="cpu", weights_only=False
    )
    score_difference = float(
        (reloaded_scores["scores"] - scores).abs().max().item()
    )
    if score_difference != 0.0 or tuple(scores.shape) != (EXPECTED_TRIALS,):
        raise RuntimeError("score save/load/count validation failed")

    metrics_started = time.perf_counter()
    eer = calculate_eer(scores.tolist(), targets.tolist())
    confusion = empirical_confusion(
        scores.tolist(), targets.tolist(), eer.empirical_threshold
    )
    same_summary = numeric_summary(scores[targets == 1].tolist())
    different_summary = numeric_summary(scores[targets == 0].tolist())
    metrics_seconds = time.perf_counter() - metrics_started
    if (
        abs(confusion["far"] - eer.empirical_far) > 1e-15
        or abs(confusion["frr"] - eer.empirical_frr) > 1e-15
    ):
        raise RuntimeError("empirical confusion and EER metrics disagree")

    runtime_config = {
        "schema_version": 2,
        "artifact_kind": "pretrained_ecapa_validation_baseline_config_v2",
        "model_source": SpeechBrainECAPAFrontend.SOURCE,
        "device": "cuda:0",
        "batch_size": 64,
        "num_workers": 0,
        "shuffle": False,
        "dataset_max_cached_shards": 2,
        "dataset_validate_finite": False,
        "pipeline": [
            "cached_fbank_[B,T,80]",
            "mean_var_norm",
            "embedding_model_[B,1,192]",
            "squeeze_dim_1_[B,192]",
            "cosine_similarity",
        ],
        "upstream_bindings": upstream,
        "training_sampler_binding": sampler_binding,
        "validation_trial_binding": trial_binding,
        "timestamps_in_identity": False,
    }
    runtime_config_path = output_dir / "baseline_config_v2.json"
    atomic_write_bytes(runtime_config_path, canonical_json_bytes(runtime_config))
    baseline_identity = {
        "schema_version": 2,
        "identity_kind": "pretrained_ecapa_validation_baseline_v2",
        "model_source": SpeechBrainECAPAFrontend.SOURCE,
        "upstream_bindings": upstream,
        "training_sampler_binding": sampler_binding,
        "validation_trial_binding": trial_binding,
        "runtime_config_sha256": sha256_file(runtime_config_path),
        "embedding_artifact": {
            "path": "outputs/pretrained_ecapa_validation_baseline_v2/validation_embeddings_v2.pt",
            "sha256": sha256_file(embedding_path),
            "shape": [EXPECTED_ROWS, 192],
            "dtype": "float32",
            "device": "cpu",
        },
        "score_artifact": {
            "path": "outputs/pretrained_ecapa_validation_baseline_v2/validation_trial_scores_v2.pt",
            "sha256": sha256_file(score_path),
            "count": EXPECTED_TRIALS,
            "dtype": "float32",
        },
        "timestamps_in_identity": False,
    }
    baseline_identity_path = output_dir / "baseline_identity_v2.json"
    atomic_write_bytes(
        baseline_identity_path, canonical_json_bytes(baseline_identity)
    )
    runtime = {
        "embedding_seconds": embedding_seconds,
        "utterances_per_second": EXPECTED_ROWS / embedding_seconds,
        "scoring_seconds": scoring_seconds,
        "metrics_seconds": metrics_seconds,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
    }
    runtime_path = output_dir / "baseline_runtime_v2.json"
    atomic_write_bytes(runtime_path, canonical_json_bytes(runtime))

    guarded_hashes_after = {
        **{
            name: sha256_file(REPO_ROOT / value["path"])
            for name, value in APPROVED_INPUTS.items()
        },
        "training_sampler_config": sha256_file(sampler_path),
        "validation_trial_config": sha256_file(trial_config_path),
        "validation_trial_csv": sha256_file(trial_csv_path),
        "validation_trial_identity": sha256_file(trial_identity_path),
    }
    if guarded_hashes_after != guarded_hashes_before:
        raise RuntimeError("an immutable approved input changed during baseline")

    result = {
        "schema_version": 2,
        "report_kind": "pretrained_ecapa_validation_baseline_v2",
        "result": "PASS",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "speechbrain": speechbrain.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
        "approved_input_hashes": guarded_hashes_after,
        "training_sampler_binding": sampler_binding,
        "validation_trial_binding": trial_binding,
        "baseline_identity_sha256": sha256_file(baseline_identity_path),
        "embedding": {
            "shape": [EXPECTED_ROWS, 192],
            "dtype": "float32",
            "device": "cpu",
            "save_load_maximum_absolute_difference": embedding_difference,
        },
        "trials": {
            "total": EXPECTED_TRIALS,
            "positive": int(targets.sum().item()),
            "negative": int((targets == 0).sum().item()),
            "score_save_load_maximum_absolute_difference": score_difference,
        },
        "model_integrity": {
            "parameters_frozen": frontend.pretrained_parameters_frozen,
            "parameters_unchanged": unchanged,
            "parameter_gradients_absent": True,
            "compute_features_call_count": call_counts["compute_features"],
            "pretrained_classifier_call_count": call_counts["classifier"],
            "feature_transpose_used": False,
            "optimizer_created": False,
            "aam_softmax_used": False,
            "training_started": False,
        },
        "metrics": dataclasses.asdict(eer),
        "metric_implementation": "grouped_sorted_tie_aware_o_n_log_n",
        "empirical_metrics": confusion,
        "score_distributions": {
            "same_speaker": same_summary,
            "different_speaker": different_summary,
        },
        "runtime": runtime,
        "artifacts": {
            "embedding_sha256": baseline_identity["embedding_artifact"]["sha256"],
            "scores_sha256": baseline_identity["score_artifact"]["sha256"],
            "runtime_config_sha256": baseline_identity["runtime_config_sha256"],
            "baseline_identity_sha256": sha256_file(baseline_identity_path),
            "runtime_sha256": sha256_file(runtime_path),
        },
        "source_and_cache_preservation": {
            "approved_hashes_unchanged": True,
            "source_audio_opened": False,
            "cache_artifacts_modified": False,
        },
        "final_test_quarantine": {
            "manifest_opened": False,
            "audio_opened": False,
            "features_extracted": False,
            "trials_generated": False,
            "evaluation_run": False,
        },
    }
    atomic_write_bytes(
        REPO_ROOT / "reports/pretrained_ecapa_validation_baseline_v2.json",
        canonical_json_bytes(result),
    )
    atomic_write_bytes(
        REPO_ROOT / "reports/pretrained_ecapa_validation_baseline_v2.md",
        markdown_report(result),
    )
    print(json.dumps({
        "result": "PASS",
        "interpolated_eer_percentage": eer.interpolated_eer_percentage,
        "empirical_threshold": eer.empirical_threshold,
        "baseline_identity_sha256": result["baseline_identity_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
