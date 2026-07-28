"""Extract pretrained ECAPA validation embeddings, cosine scores, and EER."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import speechbrain
import torch

from src.cached_fbank_dataset import CachedFbankDataset, create_cached_fbank_dataloader
from src.speechbrain_frontend import SpeechBrainECAPAFrontend
from src.verification_baseline import numeric_summary, score_trials
from src.verification_metrics import calculate_eer
from src.verification_trials import read_trials, validate_trials, validate_trials_against_metadata

EXPECTED_ROWS = 8504


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def parameters_equal(before: dict[str, torch.Tensor], module: torch.nn.Module) -> bool:
    return all(torch.equal(before[name], parameter.detach().cpu()) for name, parameter in module.named_parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/fbank_cache_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/portable/validation_manifest_v1.csv"))
    parser.add_argument("--trials", type=Path, default=Path("manifests/verification/validation_trials_v1.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation_pretrained_ecapa_baseline_v1"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/pretrained_ecapa_validation_baseline_v1.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/pretrained_ecapa_validation_baseline_v1.md"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    index_path = args.cache_dir / "validation_feature_index_v1.csv"
    config_path = args.cache_dir / "fbank_cache_config_v1.json"
    guarded = (args.manifest, index_path, config_path)
    hashes_before = {path.as_posix(): sha256(path) for path in guarded}

    dataset = CachedFbankDataset(args.cache_dir, "validation", max_cached_shards=2, validate_finite=False)
    if len(dataset) != EXPECTED_ROWS or len({row.speaker_id for row in dataset.rows}) != 100:
        raise ValueError("validation Dataset count invariant failed")
    if any(row.speaker_label != -1 or row.final_split != "validation" for row in dataset.rows):
        raise ValueError("validation label/split invariant failed")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    expected_metadata = [(row["relative_audio_path"], row["speaker_id"], row["filename_group"]) for row in manifest_rows]
    actual_metadata = [(row.relative_audio_path, row.speaker_id, row.filename_group) for row in dataset.rows]
    if expected_metadata != actual_metadata or len(set(path for path, _, _ in actual_metadata)) != EXPECTED_ROWS:
        raise ValueError("Dataset order/metadata does not exactly match approved validation manifest")
    trials = read_trials(args.trials)
    validate_trials(trials, {row.speaker_id for row in dataset.rows})
    validate_trials_against_metadata(
        trials,
        [(row.relative_audio_path, row.speaker_id) for row in dataset.rows],
        {row.speaker_id for row in dataset.rows},
    )
    if sum(t.target == 1 for t in trials) != sum(t.target == 0 for t in trials):
        raise ValueError("trial class counts differ")

    device = torch.device(args.device)
    frontend = SpeechBrainECAPAFrontend(device=device)
    frontend.eval()
    encoder = frontend.classifier
    encoder.eval()
    before = parameter_snapshot(encoder)
    loader = create_cached_fbank_dataloader(dataset, args.batch_size, shuffle=False, num_workers=0)
    first = next(iter(loader))
    preflight_features = first["fbank"][:4]
    lengths = torch.ones(4, dtype=torch.float32, device=device)
    with torch.inference_mode():
        features = preflight_features.to(device, torch.float32)
        normalized = encoder.mods.mean_var_norm(features, lengths)
        raw_embedding = encoder.mods.embedding_model(normalized, lengths)
        scoring_embedding = raw_embedding[:, 0, :]
    preflight = {
        "individual_fbank_shape": [301, 80], "batch_fbank_shape": list(features.shape),
        "normalized_shape": list(normalized.shape), "raw_embedding_shape": list(raw_embedding.shape),
        "scoring_embedding_shape": list(scoring_embedding.shape), "lengths_shape": list(lengths.shape),
        "lengths_dtype": str(lengths.dtype), "lengths_all_one": bool(torch.equal(lengths, torch.ones_like(lengths))),
        "dtype": str(features.dtype), "device": str(features.device),
        "all_finite": all(bool(torch.isfinite(value).all()) for value in (features, normalized, raw_embedding)),
        "inference_mode": not raw_embedding.requires_grad,
        "parameters_unchanged": parameters_equal(before, encoder),
        "compute_features_called": False, "classifier_head_called": False, "transposed": False,
    }
    required_shapes = ([4, 301, 80], [4, 301, 80], [4, 1, 192], [4, 192])
    if tuple(preflight[key] for key in ("batch_fbank_shape", "normalized_shape", "raw_embedding_shape", "scoring_embedding_shape")) != required_shapes:
        raise RuntimeError("preflight shape invariant failed")
    if not all((preflight["all_finite"], preflight["inference_mode"], preflight["parameters_unchanged"])):
        raise RuntimeError("preflight integrity check failed")
    print("PREFLIGHT PASS: " + json.dumps(preflight, sort_keys=True), flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    embeddings, paths, speakers = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch_features = batch["fbank"].to(device, torch.float32)
            batch_lengths = torch.ones(batch_features.shape[0], dtype=torch.float32, device=device)
            normalized_batch = encoder.mods.mean_var_norm(batch_features, batch_lengths)
            raw = encoder.mods.embedding_model(normalized_batch, batch_lengths)
            if tuple(raw.shape) != (batch_features.shape[0], 1, 192):
                raise RuntimeError("raw embedding shape invariant failed")
            flat = raw[:, 0, :].detach().to("cpu", torch.float32)
            if not bool(torch.isfinite(flat).all()):
                raise ValueError("non-finite embedding")
            embeddings.append(flat)
            paths.extend(batch["relative_audio_path"])
            speakers.extend(batch["speaker_id"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    stored = torch.cat(embeddings)
    if tuple(stored.shape) != (EXPECTED_ROWS, 192) or paths != [row.relative_audio_path for row in dataset.rows]:
        raise ValueError("stored embedding count/shape/order invariant failed")
    if len(paths) != len(set(paths)) or speakers != [row.speaker_id for row in dataset.rows]:
        raise ValueError("stored embedding path/speaker mapping invariant failed")
    unchanged = parameters_equal(before, encoder)
    if not unchanged:
        raise RuntimeError("model parameters changed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = args.output_dir / "validation_embeddings_v1.pt"
    score_path = args.output_dir / "validation_trial_scores_v1.pt"
    baseline_config_path = args.output_dir / "validation_baseline_config_v1.json"
    created = datetime.now(timezone.utc).isoformat()
    metadata = {
        "relative_audio_paths": paths, "speaker_ids": speakers, "embeddings": stored,
        "model_id": SpeechBrainECAPAFrontend.SOURCE, "torch_version": torch.__version__,
        "speechbrain_version": speechbrain.__version__, "validation_manifest_sha256": sha256(args.manifest),
        "cache_index_sha256": sha256(index_path), "trial_manifest_sha256": sha256(args.trials),
        "created_utc": created, "feature_shape": [301, 80], "embedding_dtype": "float32",
        "normalized": False, "embedding_definition": "direct embedding_model output after mean_var_norm",
        "score_method": "cosine_similarity",
    }
    atomic_torch_save(metadata, embedding_path)
    reloaded = torch.load(embedding_path, map_location="cpu", weights_only=False)
    embedding_difference = float((reloaded["embeddings"] - stored).abs().max())
    if embedding_difference != 0.0:
        raise RuntimeError("embedding save/load difference is nonzero")
    scores = score_trials(reloaded["embeddings"], reloaded["relative_audio_paths"], trials)
    targets = torch.tensor([trial.target for trial in trials], dtype=torch.long)
    score_artifact = {"trial_ids": [trial.trial_id for trial in trials], "targets": targets, "scores": scores}
    atomic_torch_save(score_artifact, score_path)
    reloaded_scores = torch.load(score_path, map_location="cpu", weights_only=False)
    score_difference = float((reloaded_scores["scores"] - scores).abs().max())
    if score_difference != 0.0 or len(scores) != len(trials):
        raise RuntimeError("score save/load/count invariant failed")
    eer = calculate_eer(scores.tolist(), targets.tolist())
    same = numeric_summary(scores[targets == 1].tolist())
    different = numeric_summary(scores[targets == 0].tolist())
    hashes_after = {path.as_posix(): sha256(path) for path in guarded}
    if hashes_before != hashes_after:
        raise RuntimeError("immutable validation input changed")
    result = {
        "overall_result": "PASS", "created_utc": created,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "speechbrain": speechbrain.__version__,
                        "cuda_available": torch.cuda.is_available(), "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "input_hashes": hashes_after, "trial_manifest_sha256": sha256(args.trials),
        "validation_rows": len(dataset), "validation_speakers": 100,
        "positive_trials": int(targets.sum()), "negative_trials": int((targets == 0).sum()),
        "preflight": preflight, "embedding_shape": list(stored.shape), "embedding_dtype": str(stored.dtype),
        "batch_size": args.batch_size, "num_workers": 0, "elapsed_seconds": elapsed,
        "utterances_per_second": len(dataset) / elapsed,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "parameters_unchanged": unchanged, "embedding_save_load_max_difference": embedding_difference,
        "score_count": len(scores), "score_range": [float(scores.min()), float(scores.max())],
        "score_save_load_max_difference": score_difference, "same_speaker_scores": same,
        "different_speaker_scores": different, "metrics": dataclasses.asdict(eer),
        "metric_implementation_version": "grouped_sorted_eer_v2",
        "trial_path_speaker_ownership_validated": True,
        "eer": eer.eer, "eer_percentage": eer.eer_percentage,
        "eer_threshold": eer.eer_threshold, "far": eer.far, "frr": eer.frr,
        "eer_kind": eer.eer_kind, "eer_threshold_kind": eer.eer_threshold_kind,
        "interpolated_eer": eer.interpolated_eer,
        "interpolated_eer_percentage": eer.interpolated_eer_percentage,
        "interpolated_threshold": eer.interpolated_threshold,
        "interpolated_far": eer.interpolated_far, "interpolated_frr": eer.interpolated_frr,
        "empirical_threshold": eer.empirical_threshold, "empirical_far": eer.empirical_far,
        "empirical_frr": eer.empirical_frr,
        "empirical_far_frr_gap": eer.empirical_far_frr_gap,
        "empirical_average_error": eer.empirical_average_error,
        "empirical_threshold_semantics": eer.empirical_threshold_semantics,
        "threshold_semantics": "accept same speaker when score >= threshold",
    }
    baseline_config_path.write_text(json.dumps({k: result[k] for k in (
        "created_utc", "input_hashes", "trial_manifest_sha256", "batch_size", "num_workers",
        "threshold_semantics")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = f"""# Pretrained ECAPA validation baseline v1

## Executive result

**PASS.** Pretrained interpolated validation EER is **{eer.interpolated_eer_percentage:.6f}%**.

## Environment and inputs

- Python {platform.python_version()}; torch {torch.__version__}; SpeechBrain {speechbrain.__version__}
- Device: {device}; GPU: {result['environment']['gpu']}
- Validation: {len(dataset)} utterances / 100 speakers / labels all -1 / 34 referenced shards
- Input hashes: `{json.dumps(hashes_after, sort_keys=True)}`
- Trial manifest SHA-256: `{result['trial_manifest_sha256']}`

## Model path and preflight

- Path: cached Fbank -> `encoder.mods.mean_var_norm` -> `encoder.mods.embedding_model`
- Fbank `{preflight['batch_fbank_shape']}`; normalized `{preflight['normalized_shape']}`
- Raw embedding `{preflight['raw_embedding_shape']}`; scoring embedding `{preflight['scoring_embedding_shape']}`
- Lengths `[4]`, float32, all 1.0; finite; inference mode; parameters unchanged
- No transpose, `compute_features`, classifier head, gradient, or update

## Extraction and scoring

- Stored embeddings: `{list(stored.shape)}`, float32 CPU, direct unnormalized embedding-model output
- Batch size {args.batch_size}; workers 0; elapsed {elapsed:.6f}s; throughput {len(dataset) / elapsed:.6f} utterances/s
- Peak VRAM allocated/reserved: {result['peak_allocated_vram_bytes']} / {result['peak_reserved_vram_bytes']} bytes
- Embedding save/load maximum difference: {embedding_difference}
- Trials: {int(targets.sum())} positive / {int((targets == 0).sum())} negative; scores: {len(scores)}
- Cosine range: {result['score_range']}; score save/load maximum difference: {score_difference}
- Same-speaker summary: `{json.dumps(same, sort_keys=True)}`
- Different-speaker summary: `{json.dumps(different, sort_keys=True)}`

## Interpolated result

- Interpolated EER fraction: {eer.interpolated_eer}
- Interpolated EER percentage: {eer.interpolated_eer_percentage}
- Interpolated threshold: {eer.interpolated_threshold}
- Interpolated FAR / FRR: {eer.interpolated_far} / {eer.interpolated_frr}
- This is a linearly interpolated ROC crossing and may not be an executable empirical point.

## Executable empirical operating point

- Threshold: {eer.empirical_threshold}
- Actual FAR / FRR: {eer.empirical_far} / {eer.empirical_frr}
- FAR/FRR gap: {eer.empirical_far_frr_gap}
- Average error: {eer.empirical_average_error}
- Semantics: {eer.empirical_threshold_semantics}

## Validation, files, and limitations

- All path ownership, mappings/counts, source hashes, finite checks, cosine bounds, empirical recomputation, parameter integrity, and save/load checks passed.
- Generated tensors remain under ignored `outputs/validation_pretrained_ecapa_baseline_v1/`.
- This is an unfine-tuned pretrained baseline on fixed validation trials, not a model-quality comparison.
- Deferred: training, AAM-Softmax, checkpoints, augmentation, and every final-test operation.

## Overall result

**PASS**
"""
    args.report_md.write_text(md, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
