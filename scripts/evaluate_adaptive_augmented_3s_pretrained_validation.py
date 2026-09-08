"""Evaluate untouched pretrained ECAPA on frozen adaptive_augmented_3s validation trials."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adaptive_augmented_3s_verification import (
    TRIAL_FIELDS,
    ValidationTrial,
    read_validation_manifest,
    sha256_file,
    trials_csv_bytes,
    validate_validation_trials,
)
from src.cached_fbank_dataset import CachedFbankDataset, create_cached_fbank_dataloader
from src.speechbrain_frontend import SpeechBrainECAPAFrontend
from src.verification_baseline import numeric_summary, score_trials
from src.verification_metrics import calculate_eer


CACHE_ROOT = REPO_ROOT / "outputs/fbank_cache_adaptive_augmented_3s_v1"
TRIAL_CSV = REPO_ROOT / "manifests/verification/adaptive_augmented_3s_v1_validation_trials.csv"
TRIAL_IDENTITY = REPO_ROOT / "manifests/verification/adaptive_augmented_3s_v1_validation_trials_identity.json"
RUNTIME_DIR = REPO_ROOT / "outputs/pretrained_ecapa_validation_baseline_adaptive_augmented_3s_v1"
SCORES_PATH = REPO_ROOT / "reports/pretrained_ecapa_validation_baseline_adaptive_augmented_3s_v1_scores.csv"
SUMMARY_PATH = REPO_ROOT / "reports/pretrained_ecapa_validation_baseline_adaptive_augmented_3s_v1.json"
CACHE_IDENTITY = "019e2734ad6e1b62b230c3938befa2d3620a70af895778952622f9ab2bf7446b"
TRIAL_IDENTITY_HASH = "5897ab00009f8cdd30d27d5028089a6fd358d66b2d76f4d13d1bd79282f78d67"
TRIAL_CSV_HASH = "58d6526904a51a64e1eb2cedf06648ddfbf3154f1dcf79fa398a1cd584bb915e"


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_exact_write(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace incompatible baseline artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def read_trials() -> tuple[ValidationTrial, ...]:
    rows = read_validation_manifest(
        REPO_ROOT / "manifests/adaptive_augmented_3s_v1_validation_manifest.csv"
    )
    with TRIAL_CSV.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRIAL_FIELDS:
            raise ValueError("trial CSV schema is invalid")
        trials = tuple(
            ValidationTrial(
                raw["trial_id"], raw["left_audio_path"], raw["right_audio_path"],
                raw["left_speaker_id"], raw["right_speaker_id"], int(raw["target"]),
            )
            for raw in reader
        )
    validate_validation_trials(trials, rows)
    if trials_csv_bytes(trials) != TRIAL_CSV.read_bytes():
        raise ValueError("trial CSV is not canonical")
    return trials


def model_binding() -> dict[str, str]:
    root = REPO_ROOT / "pretrained_models/spkrec-ecapa-voxceleb"
    files = {name: root / name for name in ("hyperparams.yaml", "embedding_model.ckpt")}
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError("required local pretrained ECAPA files are missing")
    return {"source": SpeechBrainECAPAFrontend.SOURCE, **{name: sha256_file(path) for name, path in files.items()}}


def embedding_artifact(path: Path, expected: dict[str, Any]) -> torch.Tensor | None:
    if not path.is_file():
        return None
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or artifact.get("binding") != expected:
        raise ValueError("existing embedding artifact identity is incompatible")
    embeddings = artifact.get("embeddings")
    paths = artifact.get("relative_audio_paths")
    if (
        not isinstance(embeddings, torch.Tensor) or tuple(embeddings.shape) != (6076, 192)
        or embeddings.dtype != torch.float32 or embeddings.device.type != "cpu"
        or not bool(torch.isfinite(embeddings).all()) or not isinstance(paths, list)
        or len(paths) != 6076 or len(paths) != len(set(paths))
    ):
        raise ValueError("existing embedding artifact is malformed")
    if paths != expected["validation_paths"]:
        raise ValueError("existing embedding artifact order differs from validation cache")
    return embeddings


def generate_embeddings(dataset: CachedFbankDataset, binding: dict[str, Any], batch_size: int) -> tuple[torch.Tensor, float, str]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    frontend = SpeechBrainECAPAFrontend(device=device)
    frontend.eval()
    if not frontend.pretrained_parameters_frozen:
        raise RuntimeError("pretrained parameters are not frozen")
    calls = {"compute_features": 0, "classifier": 0}
    hooks = [
        frontend.classifier.mods.compute_features.register_forward_hook(lambda *_: calls.__setitem__("compute_features", calls["compute_features"] + 1)),
        frontend.classifier.mods.classifier.register_forward_hook(lambda *_: calls.__setitem__("classifier", calls["classifier"] + 1)),
    ]
    embeddings: list[torch.Tensor] = []
    started = time.perf_counter()
    try:
        loader = create_cached_fbank_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for batch in loader:
                features = batch["fbank"]
                lengths = torch.ones(features.shape[0], dtype=torch.float32)
                normalized = frontend.mean_var_norm(features, lengths)
                embedding = frontend.embedding_model(normalized, lengths).squeeze(1).cpu()
                if tuple(embedding.shape) != (features.shape[0], 192) or not bool(torch.isfinite(embedding).all()):
                    raise RuntimeError("invalid pretrained validation embedding batch")
                embeddings.append(embedding)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
    finally:
        for hook in hooks:
            hook.remove()
    result = torch.cat(embeddings)
    if tuple(result.shape) != (6076, 192) or calls != {"compute_features": 0, "classifier": 0}:
        raise RuntimeError("pretrained inference path violated the baseline contract")
    artifact = {"binding": binding, "embeddings": result, "relative_audio_paths": binding["validation_paths"]}
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    target = RUNTIME_DIR / "validation_embeddings.pt"
    temporary = target.with_name(target.name + ".tmp")
    torch.save(artifact, temporary)
    os.replace(temporary, target)
    return result, time.perf_counter() - started, device


def scores_csv(trials: tuple[ValidationTrial, ...], scores: torch.Tensor) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("trial_id", "target", "score"), lineterminator="\n")
    writer.writeheader()
    for trial, score in zip(trials, scores.tolist()):
        writer.writerow({"trial_id": trial.trial_id, "target": trial.target, "score": repr(float(score))})
    return stream.getvalue().encode("utf-8")


def metrics_from_scores(payload: bytes, trials: tuple[ValidationTrial, ...]) -> tuple[dict[str, Any], list[float]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    rows = list(reader)
    if len(rows) != len(trials):
        raise ValueError("saved score count is invalid")
    scores = [float(row["score"]) for row in rows]
    if [row["trial_id"] for row in rows] != [trial.trial_id for trial in trials] or [int(row["target"]) for row in rows] != [trial.target for trial in trials]:
        raise ValueError("saved score trial alignment is invalid")
    if not all(torch.isfinite(torch.tensor(scores)).tolist()):
        raise ValueError("saved scores are non-finite")
    targets = [trial.target for trial in trials]
    eer = calculate_eer(scores, targets)
    same = [score for score, target in zip(scores, targets) if target == 1]
    different = [score for score, target in zip(scores, targets) if target == 0]
    return {
        "eer": dataclasses.asdict(eer),
        "genuine_scores": numeric_summary(same),
        "impostor_scores": numeric_summary(different),
    }, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if sha256_file(TRIAL_CSV) != TRIAL_CSV_HASH:
        raise ValueError("frozen validation trial CSV hash mismatch")
    trial_identity = json.loads(TRIAL_IDENTITY.read_text(encoding="utf-8"))
    if trial_identity.get("identity_sha256") != TRIAL_IDENTITY_HASH:
        raise ValueError("frozen validation trial identity mismatch")
    trials = read_trials()
    dataset = CachedFbankDataset(CACHE_ROOT, "validation", max_cached_shards=8, validate_finite=False)
    if dataset.identity is None or dataset.identity.get("identity_sha256") != CACHE_IDENTITY:
        raise ValueError("validation cache identity mismatch")
    paths = [row.relative_audio_path for row in dataset.rows]
    if len(dataset) != 6076 or len(paths) != len(set(paths)):
        raise ValueError("validation cache population is invalid")
    binding = {
        "cache_identity": CACHE_IDENTITY,
        "cache_identity_file_sha256": sha256_file(CACHE_ROOT / "fbank_cache_identity_adaptive_augmented_3s_v1.json"),
        "trial_identity": TRIAL_IDENTITY_HASH,
        "trial_csv_sha256": TRIAL_CSV_HASH,
        "model": model_binding(),
        "validation_paths": paths,
    }
    cached = embedding_artifact(RUNTIME_DIR / "validation_embeddings.pt", binding)
    if cached is None:
        embeddings, elapsed, device = generate_embeddings(dataset, binding, args.batch_size)
        embedding_status = "generated"
    else:
        embeddings, elapsed, device = cached, 0.0, "reused_cpu_artifact"
        embedding_status = "reused"
    scores = score_trials(embeddings, paths, trials)
    if tuple(scores.shape) != (20000,) or not bool(torch.isfinite(scores).all()):
        raise RuntimeError("trial score contract failed")
    score_payload = scores_csv(trials, scores)
    atomic_exact_write(SCORES_PATH, score_payload)
    metrics, recalculated_scores = metrics_from_scores(SCORES_PATH.read_bytes(), trials)
    if recalculated_scores != scores.tolist():
        raise RuntimeError("metrics-only score roundtrip differs from in-memory scores")
    summary = {
        "schema_version": 1,
        "identity_kind": "adaptive_augmented_3s_pretrained_validation_baseline",
        "input_bindings": {
            "dataset_identity": "8fd9fcc0b802d56d4a96080e53180e73e317e57165c25f8d81b77ce4f18d421d",
            "split_identity": "6aa9f029cec5cfd76e00bcf4eebad92d4f61aeb39467055009ce0e06ab22885c",
            **{key: value for key, value in binding.items() if key != "validation_paths"},
        },
        "embedding": {"count": 6076, "shape": [6076, 192], "dtype": "float32", "ordering": "validation_cache_index_order"},
        "trials": {"count": 20000, "genuine": 10000, "impostor": 10000},
        "scoring_method": "torch.nn.functional.cosine_similarity",
        "metric_implementation": "src.verification_metrics.calculate_eer",
        "score_csv_path": "reports/pretrained_ecapa_validation_baseline_adaptive_augmented_3s_v1_scores.csv",
        "score_csv_sha256": sha256_file(SCORES_PATH),
        "metrics": metrics,
        "final_test_access_count": 0,
        "timestamps_in_identity": False,
    }
    unsigned = dict(summary)
    summary["identity_sha256"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    atomic_exact_write(SUMMARY_PATH, canonical_json(summary))
    print(json.dumps({
        "embedding_status": embedding_status, "device": device, "embedding_seconds": elapsed,
        "score_csv_sha256": summary["score_csv_sha256"], "baseline_identity_sha256": summary["identity_sha256"],
        "eer": metrics["eer"]["interpolated_eer"], "empirical_threshold": metrics["eer"]["empirical_threshold"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
