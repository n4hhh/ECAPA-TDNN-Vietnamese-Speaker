"""Recompute validation metrics from immutable saved baseline artifacts only."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.verification_baseline import validate_saved_artifacts
from src.verification_metrics import calculate_eer, verification_operating_points
from src.verification_trials import read_trials, validate_trials_against_metadata

EXPECTED_ROWS = 8504
EXPECTED_SPEAKERS = 100
OLD_EER = 0.11665301106104053
OLD_THRESHOLD = 0.31165990233421326


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def read_validation_metadata(manifest: Path, index: Path) -> tuple[list[str], list[str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    with index.open("r", encoding="utf-8-sig", newline="") as stream:
        index_rows = list(csv.DictReader(stream))
    if len(manifest_rows) != EXPECTED_ROWS or len(index_rows) != EXPECTED_ROWS:
        raise ValueError("validation metadata must contain exactly 8,504 rows")
    paths = [row["relative_audio_path"] for row in manifest_rows]
    speakers = [row["speaker_id"] for row in manifest_rows]
    if len(set(paths)) != EXPECTED_ROWS:
        raise ValueError("duplicate validation manifest path")
    if len(set(speakers)) != EXPECTED_SPEAKERS:
        raise ValueError("validation metadata must contain exactly 100 speakers")
    if any(row["speaker_label"] != "-1" or row["final_split"] != "validation" for row in manifest_rows):
        raise ValueError("invalid validation manifest label or split")
    indexed = {row["relative_audio_path"]: row for row in index_rows}
    if len(indexed) != EXPECTED_ROWS or set(indexed) != set(paths):
        raise ValueError("validation index paths disagree with manifest")
    for row in manifest_rows:
        cached = indexed[row["relative_audio_path"]]
        if (
            cached["speaker_id"] != row["speaker_id"]
            or cached["speaker_label"] != "-1"
            or cached["final_split"] != "validation"
            or cached["filename_group"] != row["filename_group"]
        ):
            raise ValueError(f"validation index metadata mismatch: {row['relative_audio_path']}")
    return paths, speakers


def render_markdown(result: dict) -> str:
    metric = result["metrics"]
    return f"""# Pretrained ECAPA validation baseline v1

## Executive result

**PASS.** The saved-score, optimized interpolated validation EER is **{metric['interpolated_eer_percentage']:.12f}%**. No embedding extraction or model inference was run for this metrics patch.

## Preserved baseline and inputs

- Validation: {result['validation_rows']} utterances / {result['validation_speakers']} speakers / labels all `-1`
- Embeddings: `{result['embedding_shape']}`, float32 CPU
- Trials: {result['positive_trials']} positive / {result['negative_trials']} negative / {result['score_count']} scores
- Score range: `{result['score_range']}`
- Protected SHA-256: `{json.dumps(result['protected_hashes'], sort_keys=True)}`
- Trial paths were validated against authoritative validation path ownership.

## Interpolated result

- Interpolated EER fraction: {metric['interpolated_eer']}
- Interpolated EER percentage: {metric['interpolated_eer_percentage']}
- Interpolated threshold: {metric['interpolated_threshold']}
- Interpolated FAR / FRR: {metric['interpolated_far']} / {metric['interpolated_frr']}
- Kind: `{metric['eer_kind']}`; threshold kind: `{metric['eer_threshold_kind']}`

The interpolated threshold describes a linearly interpolated ROC crossing. With discrete scores it may not be an empirical operating point, so directly applying it can produce errors different from the interpolated FAR/FRR.

## Executable empirical operating point

- Selection: minimize `abs(FAR-FRR)`, then average error, then prefer the higher threshold
- Empirical threshold: {metric['empirical_threshold']}
- Actual FAR / FRR: {metric['empirical_far']} / {metric['empirical_frr']}
- FAR/FRR gap: {metric['empirical_far_frr_gap']}
- Average error: {metric['empirical_average_error']}
- Semantics: `{metric['empirical_threshold_semantics']}`
- FAR/FRR were independently recomputed from every saved score using the stated threshold rule.

## Regression and performance

- Previous / optimized interpolated EER: {result['old_interpolated_eer']} / {metric['interpolated_eer']}
- Absolute EER difference: {result['interpolated_eer_absolute_difference']}
- Previous / optimized interpolated threshold: {result['old_interpolated_threshold']} / {metric['interpolated_threshold']}
- Metric runtime: {result['optimized_metric_runtime_seconds']:.9f} seconds
- Unique scores: {result['unique_score_count']}
- Complexity: `O(N log N)` sorting plus `O(N)` tied-group scan

## Original extraction context

- Original extraction elapsed: {result['elapsed_seconds']} seconds; throughput: {result['utterances_per_second']} utterances/s
- Original peak allocated/reserved VRAM: {result['peak_allocated_vram_bytes']} / {result['peak_reserved_vram_bytes']} bytes
- Embedding and score save/load differences: {result['embedding_save_load_max_difference']} / {result['score_save_load_max_difference']}
- No SpeechBrain/ECAPA import, inference, scoring, parameter access, CUDA use, WAV access, or final-test access occurred in this patch.

## Overall result

**PASS**
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/portable/validation_manifest_v1.csv"))
    parser.add_argument("--cache-index", type=Path, default=Path("outputs/fbank_cache_v1/validation_feature_index_v1.csv"))
    parser.add_argument("--cache-config", type=Path, default=Path("outputs/fbank_cache_v1/fbank_cache_config_v1.json"))
    parser.add_argument("--trials", type=Path, default=Path("manifests/verification/validation_trials_v1.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/validation_pretrained_ecapa_baseline_v1/validation_embeddings_v1.pt"))
    parser.add_argument("--scores", type=Path, default=Path("outputs/validation_pretrained_ecapa_baseline_v1/validation_trial_scores_v1.pt"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/pretrained_ecapa_validation_baseline_v1.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/pretrained_ecapa_validation_baseline_v1.md"))
    args = parser.parse_args()
    protected = (args.trials, args.embeddings, args.scores, args.manifest, args.cache_index, args.cache_config)
    hashes_before = {path.as_posix(): sha256(path) for path in protected}

    paths, speakers = read_validation_metadata(args.manifest, args.cache_index)
    trials = read_trials(args.trials)
    path_owner_pairs = list(zip(paths, speakers))
    validate_trials_against_metadata(trials, path_owner_pairs, set(speakers))
    embedding_artifact = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    score_artifact = torch.load(args.scores, map_location="cpu", weights_only=False)
    embeddings, scores = validate_saved_artifacts(
        embedding_artifact, score_artifact, trials, paths, speakers
    )
    targets = score_artifact["targets"].tolist()
    started = time.perf_counter()
    metric = calculate_eer(scores.tolist(), targets)
    runtime = time.perf_counter() - started
    points = verification_operating_points(scores.tolist(), targets)
    if abs(metric.interpolated_eer - OLD_EER) > 1e-15:
        raise RuntimeError("optimized interpolated EER does not reproduce the approved result")
    hashes_after = {path.as_posix(): sha256(path) for path in protected}
    if hashes_before != hashes_after:
        raise RuntimeError("protected artifact changed before report writing")

    old = json.loads(args.report_json.read_text(encoding="utf-8"))
    result = dict(old)
    result.update({
        "schema_version": 2,
        "metric_implementation_version": "grouped_sorted_eer_v2",
        "metric_complexity": "O(N log N) sort plus O(N) grouped scan",
        "metrics_only_recalculation": True,
        "embedding_extraction_performed": False,
        "model_inference_performed": False,
        "trial_path_speaker_ownership_validated": True,
        "protected_hashes": hashes_after,
        "optimized_metric_runtime_seconds": runtime,
        "unique_score_count": len(points) - 1,
        "old_interpolated_eer": OLD_EER,
        "old_interpolated_threshold": OLD_THRESHOLD,
        "interpolated_eer_absolute_difference": abs(metric.interpolated_eer - OLD_EER),
        "metrics": dataclasses.asdict(metric),
        "eer": metric.eer,
        "eer_percentage": metric.eer_percentage,
        "eer_threshold": metric.eer_threshold,
        "far": metric.far,
        "frr": metric.frr,
        "eer_kind": metric.eer_kind,
        "eer_threshold_kind": metric.eer_threshold_kind,
        "interpolated_eer": metric.interpolated_eer,
        "interpolated_eer_percentage": metric.interpolated_eer_percentage,
        "interpolated_threshold": metric.interpolated_threshold,
        "interpolated_far": metric.interpolated_far,
        "interpolated_frr": metric.interpolated_frr,
        "empirical_threshold": metric.empirical_threshold,
        "empirical_far": metric.empirical_far,
        "empirical_frr": metric.empirical_frr,
        "empirical_far_frr_gap": metric.empirical_far_frr_gap,
        "empirical_average_error": metric.empirical_average_error,
        "empirical_threshold_semantics": metric.empirical_threshold_semantics,
        "metrics_patch_created_utc": datetime.now(timezone.utc).isoformat(),
        "overall_result": "PASS",
    })
    atomic_text(args.report_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_text(args.report_md, render_markdown(result))
    final_hashes = {path.as_posix(): sha256(path) for path in protected}
    if final_hashes != hashes_before:
        raise RuntimeError("protected artifact changed during report writing")
    print(json.dumps({
        "overall_result": "PASS", "metrics": dataclasses.asdict(metric),
        "runtime_seconds": runtime, "trial_count": len(trials),
        "unique_score_count": len(points) - 1, "protected_hashes": final_hashes,
        "embedding_extraction_performed": False, "model_inference_performed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
