"""One-shot final-test evaluation, bound to frozen inputs and best.pt only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import torch

from src.adaptive_augmented_3s_verification import ValidationRow, ValidationTrial, validate_validation_trials
from src.speechbrain_frontend import SpeechBrainECAPAFrontend
from src.verification_baseline import score_trials
from src.verification_metrics import calculate_eer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/adaptive_augmented_3s_final_evaluation_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved(relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
        raise ValueError(f"unsafe bound path: {relative}")
    return ROOT / Path(*pure.parts)


def read_config() -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("identity_kind") != "adaptive_augmented_3s_final_evaluation":
        raise ValueError("final evaluation configuration is invalid")
    if Path(value["checkpoint"]["path"]).name != "best.pt":
        raise ValueError("final evaluation accepts only the selected best.pt")
    return value


def read_manifest(path: Path) -> tuple[ValidationRow, ...]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ("relative_audio_path", "speaker_id", "speaker_label", "final_split"):
            raise ValueError("final-test manifest schema is invalid")
        for line, raw in enumerate(reader, 2):
            audio, speaker = raw["relative_audio_path"].strip(), raw["speaker_id"].strip()
            pure = PurePosixPath(audio)
            if (not audio or "\\" in audio or pure.is_absolute() or ".." in pure.parts or pure.parent.name != speaker
                    or raw["speaker_label"].strip() != "-1" or raw["final_split"].strip() != "final_test"):
                raise ValueError(f"final-test manifest row {line} is invalid")
            rows.append(ValidationRow(audio, speaker))
    if len(rows) != 6087 or len({row.audio_path for row in rows}) != 6087 or len({row.speaker_id for row in rows}) != 61:
        raise ValueError("final-test manifest population is invalid")
    return tuple(rows)


def read_trials(path: Path, rows: tuple[ValidationRow, ...]) -> tuple[ValidationTrial, ...]:
    trials = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for raw in reader:
            trials.append(ValidationTrial(raw["trial_id"], raw["left_audio_path"], raw["right_audio_path"], raw["left_speaker_id"], raw["right_speaker_id"], int(raw["target"])))
    validate_validation_trials(trials, rows)
    return tuple(trials)


def validate_bound_inputs() -> tuple[dict, tuple[ValidationRow, ...], tuple[ValidationTrial, ...]]:
    config = read_config()
    for key in ("checkpoint", "final_test_manifest"):
        bound = config[key]
        if sha256_file(resolved(bound["path"])) != bound["sha256"]:
            raise ValueError(f"{key} hash mismatch")
    manifest = read_manifest(resolved(config["final_test_manifest"]["path"]))
    trials_cfg = config["trials"]
    if sha256_file(resolved(trials_cfg["path"])) != trials_cfg["sha256"] or sha256_file(resolved(trials_cfg["identity_path"])) != trials_cfg["identity_file_sha256"]:
        raise ValueError("frozen trial package hash mismatch")
    identity = json.loads(resolved(trials_cfg["identity_path"]).read_text(encoding="utf-8"))
    if identity.get("trial_csv_sha256") != trials_cfg["sha256"] or identity.get("input_split") != "final_test":
        raise ValueError("trial identity does not bind the final-test CSV")
    trials = read_trials(resolved(trials_cfg["path"]), manifest)
    cache_cfg = config["cache"]
    if sha256_file(resolved(cache_cfg["identity_path"])) != cache_cfg["identity_file_sha256"]:
        raise ValueError("final-test cache identity hash mismatch")
    cache_identity = json.loads(resolved(cache_cfg["identity_path"]).read_text(encoding="utf-8"))
    if cache_identity.get("identity_sha256") != cache_cfg["identity_sha256"] or cache_identity.get("included_splits") != ["final_test"] or cache_identity.get("total_cached_utterances") != 6087:
        raise ValueError("cache is not an isolated final-test cache")
    return config, manifest, trials


def run_evaluation(device: str) -> dict:
    config, manifest, trials = validate_bound_inputs()
    checkpoint = torch.load(resolved(config["checkpoint"]["path"]), map_location="cpu", weights_only=False)
    frontend = SpeechBrainECAPAFrontend(device="cpu")
    norm, embedding = frontend.classifier.mods.mean_var_norm, frontend.classifier.mods.embedding_model
    norm.load_state_dict(checkpoint["mean_var_norm_state_dict"], strict=True); embedding.load_state_dict(checkpoint["embedding_model_state_dict"], strict=True)
    norm.eval().to(device); embedding.eval().to(device)
    cache_root = resolved(config["cache"]["path"]); paths, vectors = [], []
    for shard in sorted((cache_root / "final_test").glob("shard_*.pt")):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        features = payload["features"].to(device=device, dtype=torch.float32); lengths = torch.ones(features.shape[0], device=device)
        with torch.inference_mode(): vectors.append(embedding(norm(features, lengths), lengths).squeeze(1).cpu())
        paths.extend(payload["relative_audio_paths"])
    embeddings = torch.cat(vectors)
    if tuple(embeddings.shape) != (6087, 192) or set(paths) != {row.audio_path for row in manifest}:
        raise ValueError("final-test embedding population is invalid")
    scores = score_trials(embeddings, paths, trials); eer = calculate_eer(scores.tolist(), [trial.target for trial in trials])
    return {"checkpoint": config["checkpoint"], "score_count": int(scores.numel()), "eer": float(eer.interpolated_eer), "eer_threshold_descriptive_only": float(eer.empirical_threshold), "deployment_threshold_policy": config["threshold_policy"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify frozen bindings without embeddings or scoring")
    parser.add_argument("--run", action="store_true", help="perform the later authorized one-shot evaluation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.check == args.run:
        parser.error("specify exactly one of --check or --run")
    if args.check:
        config, rows, trials = validate_bound_inputs(); print(json.dumps({"validated_final_test_rows": len(rows), "validated_trials": len(trials), "checkpoint": config["checkpoint"]["path"], "embeddings_computed": 0, "scores_computed": 0}, sort_keys=True))
    else:
        print(json.dumps(run_evaluation(args.device), sort_keys=True))


if __name__ == "__main__":
    main()
