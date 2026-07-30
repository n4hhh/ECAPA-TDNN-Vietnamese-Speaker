"""Locked, one-time VieSpeaker2.0 final-evaluation helpers.

This module deliberately has no torch, SpeechBrain, CUDA, waveform, or cache
imports so the metrics-only command can safely reuse it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from src.verification_metrics import calculate_eer
from src.verification_v2 import (
    ValidationManifestRow,
    ValidationTrial,
    empirical_confusion,
    generate_validation_trials,
    read_trials_csv,
    trials_csv_bytes,
    validate_validation_trials,
)


SEED = 20260729
EXPECTED_ROWS = 15355
EXPECTED_SPEAKERS = 100
EXPECTED_POSITIVE = 10000
EXPECTED_NEGATIVE = 10000
EXPECTED_TRIALS = 20000
POSITIVE_PER_SPEAKER = 100
NEGATIVE_ROUNDS = 200
LOCKED_THRESHOLD = 0.16545939445495605
CHECKPOINT_SHA256 = (
    "ba9d989c1b6a922f3f392cd297fb771bba05319d3fad99df740ac889d2836d6f"
)
FINAL_MANIFEST_SHA256 = (
    "14c782fa36d9c23040dd7a7fce26d91b9a57a230bdeebd19658dbcb2ee8364eb"
)
FEATURE_SHAPE = (301, 80)
EMBEDDING_DIMENSION = 192
SHARD_SIZE = 256
BATCH_SIZE = 64
PHASES = (
    "protocol_locked",
    "cache_complete",
    "embeddings_pending",
    "embeddings_complete",
    "scores_complete",
    "metrics_complete",
    "finalized",
    "failed",
)
ALLOWED_TRANSITIONS = {
    "protocol_locked": {"cache_complete", "failed"},
    "cache_complete": {"embeddings_pending", "failed"},
    "embeddings_pending": {"embeddings_complete", "failed"},
    "embeddings_complete": {"scores_complete", "failed"},
    "scores_complete": {"metrics_complete", "failed"},
    "metrics_complete": {"finalized", "failed"},
    "failed": set(),
    "finalized": set(),
}
METRICS_VERSION = "viespeaker2_grouped_tie_aware_eer_and_locked_confusion_v2"
TRIAL_ALGORITHM_VERSION = "viespeaker2_validation_sha256_round_robin_v2"
ONE_TIME_POLICY_VERSION = "viespeaker2_authoritative_final_evaluation_once_v2"
DECISION_RULE = "cosine_score >= threshold => accept as same speaker"
THRESHOLD_SOURCE = "selected epoch-3 validation threshold"

APPROVED_INPUTS: dict[str, dict[str, str]] = {
    "full_manifest": {
        "path": "manifests/v2/full_manifest_v2.csv",
        "sha256": "26a0157abce3bb00ce5f0ca16f9b964e180f72484e53f9d2602577f5ce4acf8f",
    },
    "full_manifest_identity": {
        "path": "manifests/v2/full_manifest_v2_identity.json",
        "sha256": "f7b6c1cbc95b8a0d20b596b4841de7ebc26e69d6bd7c130801937dab681c544e",
    },
    "speaker_split": {
        "path": "splits/v2/speaker_split_v2.csv",
        "sha256": "cbbcdcd4d3561ff2470a6cd713187cbb612939e643e5bc8cf4ed25504539f87e",
    },
    "speaker_split_identity": {
        "path": "splits/v2/speaker_split_v2_identity.json",
        "sha256": "87d2a542ae1716f5d143e27835bf0478413e672cfa131462c0410808498c3c67",
    },
    "split_policy": {
        "path": "splits/v2/split_policy_v2.json",
        "sha256": "3e414836b4fe307841810cffa56c3f0040d0c662d265d276d73a7d00a16a5d10",
    },
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
    "final_test_manifest": {
        "path": "manifests/portable_v2/test_manifest_v2.csv",
        "sha256": FINAL_MANIFEST_SHA256,
    },
    "train_validation_fbank_config": {
        "path": "outputs/fbank_cache_v2/fbank_cache_config_v2.json",
        "sha256": "ec71959ec64361038991e760e772d11bf1779e1b505a892dff364cf45aaeb018",
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
    "final_training_report": {
        "path": "reports/ecapa_aam_multiepoch_v2_final.json",
        "sha256": "",
    },
    "checkpoint": {
        "path": "outputs/ecapa_aam_multiepoch_v2/best.pt",
        "sha256": CHECKPOINT_SHA256,
    },
    "epoch_003_checkpoint": {
        "path": "outputs/ecapa_aam_multiepoch_v2/epoch_003.pt",
        "sha256": CHECKPOINT_SHA256,
    },
}


@dataclass(frozen=True)
class FinalManifestRow:
    relative_audio_path: str
    speaker_id: str
    speaker_label: int
    final_split: str
    filename_group: str
    duplicate_group: str
    manifest_version: str
    manifest_row_index: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json(value).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def safe_portable_path(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or "\\" in value
        or ":" in value
        or ".." in pure.parts
        or value != pure.as_posix()
        or len(pure.parts) < 2
    ):
        raise ValueError(f"unsafe portable path: {value!r}")
    return value


def assert_no_absolute_paths(value: Any) -> None:
    def walk(item: Any) -> Iterable[str]:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                yield str(key)
                yield from walk(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                yield from walk(nested)
        elif isinstance(item, str):
            yield item

    for text in walk(value):
        lowered = text.lower()
        if (
            "e:\\viespeaker" in lowered
            or "e:/viespeaker" in lowered
            or lowered.startswith("c:\\users\\")
        ):
            raise ValueError("absolute local path persisted in identity content")


def verify_approved_inputs(repo_root: Path) -> dict[str, dict[str, str]]:
    verified: dict[str, dict[str, str]] = {}
    for name, binding in APPROVED_INPUTS.items():
        path = repo_root / binding["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing approved input {name}: {path}")
        actual = sha256_file(path)
        expected = binding["sha256"]
        if expected and actual != expected:
            raise ValueError(f"approved input hash mismatch for {name}: {actual}")
        verified[name] = {"path": binding["path"], "sha256": actual}
    if verified["checkpoint"]["sha256"] != verified["epoch_003_checkpoint"]["sha256"]:
        raise ValueError("best.pt is not byte-identical to epoch_003.pt")
    report = read_json(repo_root / APPROVED_INPUTS["final_training_report"]["path"])
    best = report.get("best", {})
    if (
        report.get("result") != "PASS"
        or best.get("epoch") != 3
        or best.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or best.get("empirical_threshold") != LOCKED_THRESHOLD
    ):
        raise ValueError("final-training checkpoint/threshold binding mismatch")
    return verified


def read_final_manifest(path: Path) -> tuple[FinalManifestRow, ...]:
    rows: list[FinalManifestRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = (
            "relative_audio_path",
            "speaker_id",
            "speaker_label",
            "final_split",
            "filename_group",
            "duplicate_group",
            "manifest_version",
        )
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError("final-test manifest schema mismatch")
        for index, raw in enumerate(reader):
            audio_path = safe_portable_path(raw["relative_audio_path"].strip())
            speaker_id = raw["speaker_id"].strip()
            if (
                PurePosixPath(audio_path).parent.name != speaker_id
                or raw["speaker_label"].strip() != "-1"
                or raw["final_split"].strip() != "test"
                or raw["manifest_version"].strip() != "v2"
                or audio_path in seen
            ):
                raise ValueError(f"invalid final-test manifest row {index + 2}")
            seen.add(audio_path)
            rows.append(
                FinalManifestRow(
                    relative_audio_path=audio_path,
                    speaker_id=speaker_id,
                    speaker_label=-1,
                    final_split="test",
                    filename_group=raw["filename_group"].strip(),
                    duplicate_group=raw["duplicate_group"].strip(),
                    manifest_version="v2",
                    manifest_row_index=index,
                )
            )
    counts = Counter(row.speaker_id for row in rows)
    duplicate_rows = [row for row in rows if row.duplicate_group]
    if (
        len(rows) != EXPECTED_ROWS
        or len(counts) != EXPECTED_SPEAKERS
        or min(counts.values()) < 20
        or len({row.duplicate_group for row in duplicate_rows}) != 2
        or len(duplicate_rows) != 4
    ):
        raise ValueError("authoritative final-test manifest facts mismatch")
    return tuple(rows)


def read_manifest_speakers(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["speaker_id"].strip() for row in csv.DictReader(stream)}


def validate_split_disjointness(
    rows: Sequence[FinalManifestRow],
    train_manifest: Path,
    validation_manifest: Path,
    split_csv: Path,
) -> dict[str, Any]:
    test = {row.speaker_id for row in rows}
    train = read_manifest_speakers(train_manifest)
    validation = read_manifest_speakers(validation_manifest)
    with split_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        split_rows = list(csv.DictReader(stream))
    excluded = {
        row["speaker_id"].strip()
        for row in split_rows
        if row["final_split"].strip() == "excluded"
    }
    if test & train or test & validation or test & excluded:
        raise ValueError("final-test speaker partition is not disjoint")
    approved_test = {
        row["speaker_id"].strip()
        for row in split_rows
        if row["final_split"].strip() == "test"
    }
    if test != approved_test:
        raise ValueError("final-test speakers disagree with approved split")
    return {
        "test_speakers": len(test),
        "train_overlap": 0,
        "validation_overlap": 0,
        "excluded_overlap": 0,
        "approved_split_exact": True,
    }


def generate_final_trials(
    rows: Sequence[FinalManifestRow], seed: int = SEED,
) -> tuple[tuple[ValidationTrial, ...], dict[str, Any]]:
    base_rows = tuple(
        ValidationManifestRow(
            audio_path=row.relative_audio_path,
            speaker_id=row.speaker_id,
            duplicate_group=row.duplicate_group,
        )
        for row in rows
    )
    generated, metadata = generate_validation_trials(
        base_rows,
        seed=seed,
        positive_per_speaker=POSITIVE_PER_SPEAKER,
        negative_rounds=NEGATIVE_ROUNDS,
    )
    trials = tuple(
        ValidationTrial(
            trial_id=f"final-test-v2-{index:05d}",
            left_audio_path=trial.left_audio_path,
            right_audio_path=trial.right_audio_path,
            left_speaker_id=trial.left_speaker_id,
            right_speaker_id=trial.right_speaker_id,
            target=trial.target,
        )
        for index, trial in enumerate(generated)
    )
    validate_final_trials(trials, rows)
    return trials, metadata


def validate_final_trials(
    trials: Sequence[ValidationTrial], rows: Sequence[FinalManifestRow]
) -> None:
    speakers = {row.speaker_id for row in rows}
    validate_validation_trials(
        trials,
        expected_speakers=speakers,
        expected_positive=EXPECTED_POSITIVE,
        expected_negative=EXPECTED_NEGATIVE,
        expected_positive_per_speaker=POSITIVE_PER_SPEAKER,
        expected_negative_participation=NEGATIVE_ROUNDS,
    )
    owners = {row.relative_audio_path: row.speaker_id for row in rows}
    duplicates = {row.relative_audio_path: row.duplicate_group for row in rows}
    for trial in trials:
        safe_portable_path(trial.left_audio_path)
        safe_portable_path(trial.right_audio_path)
        if (
            owners.get(trial.left_audio_path) != trial.left_speaker_id
            or owners.get(trial.right_audio_path) != trial.right_speaker_id
            or (trial.target == 1)
            != (trial.left_speaker_id == trial.right_speaker_id)
        ):
            raise ValueError("final trial ownership/target mismatch")
        if (
            trial.target == 1
            and duplicates[trial.left_audio_path]
            and duplicates[trial.left_audio_path] == duplicates[trial.right_audio_path]
        ):
            raise ValueError("positive trial contains duplicate-group conflict")


def final_trials_csv_bytes(trials: Sequence[ValidationTrial]) -> bytes:
    return trials_csv_bytes(trials)


def build_trial_config(
    verified: Mapping[str, Mapping[str, str]], seed: int
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "identity_kind": "final_test_trials_config_v2",
        "input_split": "test",
        "seed": seed,
        "speaker_count": EXPECTED_SPEAKERS,
        "trial_algorithm_version": TRIAL_ALGORITHM_VERSION,
        "validation_policy_reference": {
            "config_path": verified["validation_trial_config"]["path"],
            "config_sha256": verified["validation_trial_config"]["sha256"],
            "identity_path": verified["validation_trial_identity"]["path"],
            "identity_sha256": verified["validation_trial_identity"]["sha256"],
        },
        "positive_protocol": {
            "pairs_per_speaker": POSITIVE_PER_SPEAKER,
            "selection": "stable_sha256_ranked_canonical_unordered_path_pairs",
            "same_nonempty_duplicate_group_pair_forbidden": True,
        },
        "negative_protocol": {
            "rounds": NEGATIVE_ROUNDS,
            "participations_per_speaker": NEGATIVE_ROUNDS,
            "schedule": "two_99_round_cycles_plus_first_2_rounds_of_third_cycle",
            "utterance_selection": "least_used_then_stable_sha256_tie_break",
        },
        "trial_counts": {
            "positive": EXPECTED_POSITIVE,
            "negative": EXPECTED_NEGATIVE,
            "total": EXPECTED_TRIALS,
        },
        "trial_order": "all_hash_ranked_positives_then_round_robin_negatives",
        "pair_uniqueness": "canonical_unordered_relative_audio_path_pair",
        "final_test_manifest_binding": verified["final_test_manifest"],
        "timestamps_in_identity": False,
        "absolute_paths_in_artifacts": False,
    }


def build_trial_identity(
    verified: Mapping[str, Mapping[str, str]],
    config_hash: str,
    csv_hash: str,
    seed: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "identity_kind": "final_test_trials_v2",
        "seed": seed,
        "speaker_count": EXPECTED_SPEAKERS,
        "trial_algorithm_version": TRIAL_ALGORITHM_VERSION,
        "trial_counts": {
            "positive": EXPECTED_POSITIVE,
            "negative": EXPECTED_NEGATIVE,
            "total": EXPECTED_TRIALS,
        },
        "trial_csv_path": "manifests/verification_v2/final_test_trials_v2.csv",
        "trial_csv_sha256": csv_hash,
        "trial_config_path": (
            "manifests/verification_v2/final_test_trials_config_v2.json"
        ),
        "trial_config_sha256": config_hash,
        "final_test_manifest_binding": verified["final_test_manifest"],
        "validation_policy_binding": {
            "csv": verified["validation_trial_csv"],
            "config": verified["validation_trial_config"],
            "identity": verified["validation_trial_identity"],
        },
        "generation_metadata": dict(metadata),
        "deterministic_reproduction": {
            "different_pythonhashseed_byte_identical": True,
            "csv_config_identity_byte_identical": True,
        },
        "timestamps_in_identity": False,
        "absolute_paths_in_identity": False,
    }


def build_cache_config(
    verified: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "identity_kind": "final_test_fbank_cache_config_v2",
        "cache_version": "final_test_v2",
        "dataset_root_logical_identity": "VieSpeaker2.0/augmented_dataset",
        "manifest_binding": verified["final_test_manifest"],
        "approved_input_bindings": dict(verified),
        "included_splits": ["test"],
        "expected_rows": EXPECTED_ROWS,
        "expected_speakers": EXPECTED_SPEAKERS,
        "expected_labels": [-1],
        "model_source": "speechbrain/spkrec-ecapa-voxceleb",
        "reference_fbank_config": verified["train_validation_fbank_config"],
        "feature_stage": "raw_compute_features_before_mean_var_norm",
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": "float32",
        "raw_pre_normalization": True,
        "transposed": False,
        "sample_rate_hz": 16000,
        "waveform_samples": 48000,
        "device": "cuda:0",
        "extraction_batch_size": BATCH_SIZE,
        "shard_size": SHARD_SIZE,
        "expected_shards": math.ceil(EXPECTED_ROWS / SHARD_SIZE),
        "index_path": "outputs/fbank_cache_final_test_v2/test_feature_index_v2.csv",
        "shard_path_pattern": "test/shard_<zero_padded_5_digit_number>.pt",
        "source_snapshot": "manifest_derived_path_size_mtime_ns_before_and_after",
        "preprocessing": {
            "resampling": False,
            "channel_conversion": False,
            "vad": False,
            "normalization": False,
            "cropping": False,
            "padding": False,
            "augmentation": False,
            "feature_transposition": False,
        },
        "timestamps_in_identity": False,
        "absolute_paths_in_artifacts": False,
    }


def build_lock_config(
    verified: Mapping[str, Mapping[str, str]],
    *,
    trial_csv_hash: str,
    trial_config_hash: str,
    trial_identity_hash: str,
    fbank_config_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "identity_kind": "final_evaluation_lock_config_v2",
        "one_time_policy_version": ONE_TIME_POLICY_VERSION,
        "dataset_root_logical_identity": "VieSpeaker2.0/augmented_dataset",
        "approved_input_bindings": dict(verified),
        "checkpoint": {
            "path": "outputs/ecapa_aam_multiepoch_v2/best.pt",
            "sha256": CHECKPOINT_SHA256,
            "byte_identical_epoch_003": True,
            "selected_epoch": 3,
        },
        "operational_threshold": LOCKED_THRESHOLD,
        "threshold_source": THRESHOLD_SOURCE,
        "decision_rule": DECISION_RULE,
        "trial_protocol": {
            "seed": SEED,
            "algorithm_version": TRIAL_ALGORITHM_VERSION,
            "csv_path": "manifests/verification_v2/final_test_trials_v2.csv",
            "csv_sha256": trial_csv_hash,
            "config_path": (
                "manifests/verification_v2/final_test_trials_config_v2.json"
            ),
            "config_sha256": trial_config_hash,
            "identity_path": (
                "manifests/verification_v2/final_test_trials_identity_v2.json"
            ),
            "identity_sha256": trial_identity_hash,
            "expected_positive": EXPECTED_POSITIVE,
            "expected_negative": EXPECTED_NEGATIVE,
            "expected_total": EXPECTED_TRIALS,
        },
        "final_test": {
            "manifest": verified["final_test_manifest"],
            "expected_rows": EXPECTED_ROWS,
            "expected_speakers": EXPECTED_SPEAKERS,
        },
        "fbank": {
            "config_path": "configs/v2/final_test_fbank_cache_v2.json",
            "config_sha256": fbank_config_hash,
            "feature_shape": list(FEATURE_SHAPE),
            "feature_dtype": "float32",
            "shard_size": SHARD_SIZE,
            "extraction_batch_size": BATCH_SIZE,
        },
        "inference": {
            "device": "cuda:0",
            "batch_size": BATCH_SIZE,
            "workers": 0,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "retained_modules": ["mean_var_norm", "embedding_model"],
        },
        "expected_outputs": {
            "cache_root": "outputs/fbank_cache_final_test_v2",
            "evaluation_root": "outputs/final_evaluation_v2",
            "embedding": (
                "outputs/final_evaluation_v2/final_test_embeddings_v2.pt"
            ),
            "scores": "outputs/final_evaluation_v2/final_test_scores_v2.json",
            "result_identity": (
                "outputs/final_evaluation_v2/final_evaluation_result_identity_v2.json"
            ),
            "report_json": "reports/final_evaluation_v2.json",
            "report_markdown": "reports/final_evaluation_v2.md",
        },
        "metrics_implementation_version": METRICS_VERSION,
        "timestamps_in_identity": False,
        "absolute_paths_in_artifacts": False,
    }


def build_lock_identity(config_hash: str, lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "identity_kind": "final_evaluation_lock_identity_v2",
        "lock_config_path": "configs/v2/final_evaluation_v2.json",
        "lock_config_sha256": config_hash,
        "one_time_policy_version": ONE_TIME_POLICY_VERSION,
        "evaluation_id": sha256_bytes(
            f"{config_hash}\x1f{ONE_TIME_POLICY_VERSION}".encode("utf-8")
        ),
        "protocol_locked": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_epoch": 3,
        "operational_threshold": LOCKED_THRESHOLD,
        "threshold_source": THRESHOLD_SOURCE,
        "trial_identity_sha256": lock["trial_protocol"]["identity_sha256"],
        "fbank_config_sha256": lock["fbank"]["config_sha256"],
        "timestamps_in_identity": False,
        "absolute_paths_in_identity": False,
    }


def validate_lock(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = repo_root / "configs/v2/final_evaluation_v2.json"
    identity_path = repo_root / "configs/v2/final_evaluation_v2_identity.json"
    lock = read_json(config_path)
    identity = read_json(identity_path)
    if (
        sha256_file(config_path) != identity.get("lock_config_sha256")
        or lock.get("one_time_policy_version") != ONE_TIME_POLICY_VERSION
        or lock.get("operational_threshold") != LOCKED_THRESHOLD
        or lock.get("threshold_source") != THRESHOLD_SOURCE
        or lock.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
        or lock.get("checkpoint", {}).get("selected_epoch") != 3
    ):
        raise ValueError("immutable final-evaluation lock mismatch")
    for key in ("csv", "config", "identity"):
        binding = lock["trial_protocol"]
        path = repo_root / binding[f"{key}_path"]
        if sha256_file(path) != binding[f"{key}_sha256"]:
            raise ValueError(f"locked trial {key} hash mismatch")
    fbank_path = repo_root / lock["fbank"]["config_path"]
    if sha256_file(fbank_path) != lock["fbank"]["config_sha256"]:
        raise ValueError("locked Fbank configuration hash mismatch")
    assert_no_absolute_paths(lock)
    assert_no_absolute_paths(identity)
    return lock, identity


def state_path(repo_root: Path) -> Path:
    return repo_root / "outputs/final_evaluation_v2/runtime_state_v2.json"


def initialize_state(
    repo_root: Path, lock_identity_hash: str, evaluation_id: str
) -> dict[str, Any]:
    path = state_path(repo_root)
    if path.exists():
        raise FileExistsError("final-evaluation runtime state already exists")
    state = {
        "schema_version": 2,
        "phase": "protocol_locked",
        "lock_identity_sha256": lock_identity_hash,
        "evaluation_id": evaluation_id,
        "completed_artifacts": {},
        "one_time_policy_version": ONE_TIME_POLICY_VERSION,
    }
    atomic_json(path, state)
    return state


def read_state(
    repo_root: Path, lock_identity_hash: str | None = None
) -> dict[str, Any]:
    state = read_json(state_path(repo_root))
    if state.get("phase") not in PHASES:
        raise ValueError("unknown final-evaluation state phase")
    if (
        lock_identity_hash is not None
        and state.get("lock_identity_sha256") != lock_identity_hash
    ):
        raise ValueError("runtime state is bound to another lock identity")
    for relative, expected in state.get("completed_artifacts", {}).items():
        path = repo_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"completed artifact hash mismatch: {relative}")
    return state


def transition_state(
    repo_root: Path,
    expected_phase: str,
    new_phase: str,
    artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    state = read_state(repo_root)
    if state["phase"] != expected_phase:
        raise RuntimeError(
            f"phase transition expected {expected_phase}, got {state['phase']}"
        )
    if new_phase not in ALLOWED_TRANSITIONS[expected_phase]:
        raise RuntimeError(f"forbidden phase transition {expected_phase}->{new_phase}")
    completed = dict(state.get("completed_artifacts", {}))
    if artifacts:
        for relative, expected_hash in artifacts.items():
            if sha256_file(repo_root / relative) != expected_hash:
                raise ValueError(f"new completed artifact hash mismatch: {relative}")
            completed[relative] = expected_hash
    state["phase"] = new_phase
    state["completed_artifacts"] = completed
    atomic_json(state_path(repo_root), state)
    return state


def load_score_artifact(path: Path) -> dict[str, Any]:
    artifact = read_json(path)
    scores = artifact.get("scores")
    targets = artifact.get("targets")
    trial_ids = artifact.get("trial_ids")
    if (
        not isinstance(scores, list)
        or not isinstance(targets, list)
        or not isinstance(trial_ids, list)
        or len(scores) != EXPECTED_TRIALS
        or len(targets) != EXPECTED_TRIALS
        or len(trial_ids) != EXPECTED_TRIALS
        or any(not math.isfinite(float(score)) for score in scores)
        or any(float(score) < -1.000001 or float(score) > 1.000001 for score in scores)
        or any(target not in (0, 1) for target in targets)
        or len(set(trial_ids)) != EXPECTED_TRIALS
    ):
        raise ValueError("invalid saved final-test score artifact")
    return artifact


def score_alignment_sha256(
    trial_ids: Sequence[str], targets: Sequence[int]
) -> str:
    digest = hashlib.sha256()
    for trial_id, target in zip(trial_ids, targets):
        digest.update(f"{trial_id}\x1f{target}\n".encode("utf-8"))
    return digest.hexdigest()


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    if not numeric or any(not math.isfinite(value) for value in numeric):
        raise ValueError("distribution requires finite values")
    return {
        "count": len(numeric),
        "minimum": min(numeric),
        "mean": statistics.fmean(numeric),
        "standard_deviation": statistics.pstdev(numeric),
        "median": statistics.median(numeric),
        "maximum": max(numeric),
    }


def calculate_final_metrics(
    scores: Sequence[float],
    targets: Sequence[int],
    validation_reference: Mapping[str, Any],
) -> dict[str, Any]:
    locked = empirical_confusion(scores, targets, LOCKED_THRESHOLD)
    independent = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for score, target in zip(scores, targets):
        accepted = float(score) >= LOCKED_THRESHOLD
        key = (
            "tp" if target == 1 and accepted else
            "fn" if target == 1 else
            "fp" if accepted else "tn"
        )
        independent[key] += 1
    if any(locked[key] != independent[key] for key in independent):
        raise RuntimeError("independent locked-threshold confusion mismatch")
    locked["far_frr_gap"] = abs(float(locked["far"]) - float(locked["frr"]))
    locked["average_error"] = (float(locked["far"]) + float(locked["frr"])) / 2
    locked["threshold_source"] = THRESHOLD_SOURCE
    locked["decision_rule"] = DECISION_RULE
    locked["independent_confusion_reproduction"] = True
    locked["balanced_protocol_caveat"] = (
        "Accuracy, precision, and F1 describe this balanced 50/50 trial "
        "protocol and do not represent real-world same/different-speaker prevalence."
    )

    eer = calculate_eer(scores, targets)
    descriptive = {
        **asdict(eer),
        "label": "DESCRIPTIVE TEST DIAGNOSTIC - NON-OPERATIONAL",
        "operational": False,
        "must_not_replace_locked_threshold": True,
    }
    positives = [float(s) for s, target in zip(scores, targets) if target == 1]
    negatives = [float(s) for s, target in zip(scores, targets) if target == 0]
    distributions = {
        "positive": distribution(positives),
        "negative": distribution(negatives),
        "full_range": {"minimum": min(scores), "maximum": max(scores)},
    }
    validation_eer = float(validation_reference["eer"])
    validation_far = float(validation_reference["far"])
    validation_frr = float(validation_reference["frr"])
    validation_accuracy = float(validation_reference["accuracy"])
    final_eer = float(eer.interpolated_eer)
    generalization = {
        "final_test_descriptive_eer_minus_validation_eer": final_eer
        - validation_eer,
        "absolute_eer_gap": abs(final_eer - validation_eer),
        "percentage_point_eer_gap": (final_eer - validation_eer) * 100.0,
        "relative_eer_change": (
            (final_eer - validation_eer) / validation_eer
            if validation_eer
            else None
        ),
        "locked_far_minus_validation_far": float(locked["far"]) - validation_far,
        "locked_frr_minus_validation_frr": float(locked["frr"]) - validation_frr,
        "locked_accuracy_minus_validation_accuracy": float(locked["accuracy"])
        - validation_accuracy,
    }
    for name in ("positive", "negative"):
        reference = validation_reference["score_distributions"][name]
        for statistic in ("mean", "standard_deviation", "median"):
            generalization[f"{name}_{statistic}_shift"] = (
                float(distributions[name][statistic]) - float(reference[statistic])
            )
    generalization["full_score_range_comparison"] = {
        "validation": validation_reference["score_distributions"]["full_range"],
        "final_test": distributions["full_range"],
    }
    return {
        "metrics_implementation_version": METRICS_VERSION,
        "primary_locked_validation_threshold": locked,
        "descriptive_test_diagnostic_non_operational": descriptive,
        "score_distributions": distributions,
        "generalization_analysis": generalization,
    }


def validate_trial_artifacts(repo_root: Path) -> tuple[ValidationTrial, ...]:
    rows = read_final_manifest(
        repo_root / APPROVED_INPUTS["final_test_manifest"]["path"]
    )
    path = repo_root / "manifests/verification_v2/final_test_trials_v2.csv"
    trials = read_trials_csv(path)
    validate_final_trials(trials, rows)
    return trials

