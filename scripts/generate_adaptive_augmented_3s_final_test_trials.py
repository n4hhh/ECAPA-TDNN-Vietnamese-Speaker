"""Freeze the one-shot adaptive_augmented_3s_v1 final-test trial package."""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive_augmented_3s_verification import (
    ValidationRow, ValidationTrial, generate_validation_trials, sha256_bytes,
    sha256_file, trials_csv_bytes, validate_validation_trials,
)

MANIFEST = ROOT / "manifests/adaptive_augmented_3s_v1_final_test_manifest.csv"
OUTPUT = ROOT / "manifests/verification"
CSV = OUTPUT / "adaptive_augmented_3s_v1_final_test_trials.csv"
IDENTITY = OUTPUT / "adaptive_augmented_3s_v1_final_test_trials_identity.json"
MANIFEST_SHA256 = "73c1ce536b66266ef620de06f8e4fdca7777576934e032f5540aec444034a912"


def atomic_exact(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise FileExistsError(f"frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_replace(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def read_rows() -> tuple[ValidationRow, ...]:
    rows: list[ValidationRow] = []
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ("relative_audio_path", "speaker_id", "speaker_label", "final_split"):
            raise ValueError("final-test manifest schema is invalid")
        for line, row in enumerate(reader, 2):
            path, speaker = row["relative_audio_path"].strip(), row["speaker_id"].strip()
            pure = PurePosixPath(path)
            if (not path or "\\" in path or pure.is_absolute() or ".." in pure.parts
                    or pure.parent.name != speaker or row["speaker_label"].strip() != "-1"
                    or row["final_split"].strip() != "final_test"):
                raise ValueError(f"final-test manifest row {line} is invalid")
            rows.append(ValidationRow(path, speaker))
    if len(rows) != 6087 or len({row.audio_path for row in rows}) != 6087 or len({row.speaker_id for row in rows}) != 61:
        raise ValueError("final-test manifest does not match the approved population")
    return tuple(rows)


def main() -> None:
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("approved final-test manifest hash mismatch")
    rows = read_rows()
    generated = generate_validation_trials(rows, seed=2026)
    trials = tuple(replace(trial, trial_id=f"adaptive-augmented-3s-v1-final-test-{index:05d}") for index, trial in enumerate(generated))
    validate_validation_trials(trials, rows)
    payload = trials_csv_bytes(trials)
    genuine_participation = Counter(trial.left_speaker_id for trial in trials if trial.target == 1)
    impostor_endpoints = Counter()
    for trial in trials:
        if trial.target == 0:
            impostor_endpoints.update((trial.left_speaker_id, trial.right_speaker_id))
    identity = {
        "schema_version": 1, "identity_kind": "adaptive_augmented_3s_final_test_trials",
        "package_version": "adaptive_augmented_3s_v1", "generation_algorithm": "hash_ranked_genuine_and_balanced_impostor_endpoints_v1",
        "seed": 2026, "input_split": "final_test", "final_test_manifest": {"path": MANIFEST.relative_to(ROOT).as_posix(), "sha256": MANIFEST_SHA256},
        "speaker_count": 61, "trial_counts": {"genuine": 10000, "impostor": 10000, "total": 20000},
        "trial_csv_path": CSV.relative_to(ROOT).as_posix(), "trial_csv_sha256": sha256_bytes(payload),
        "trial_schema": ["trial_id", "left_audio_path", "right_audio_path", "left_speaker_id", "right_speaker_id", "target"],
        "pair_identity": "canonical_unordered_dataset_relative_audio_paths", "frozen": True, "timestamps_in_identity": False,
        "genuine_trial_participation": dict(sorted(genuine_participation.items())),
        "impostor_endpoint_participation": dict(sorted(impostor_endpoints.items())),
    }
    identity["identity_sha256"] = sha256_bytes(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    atomic_exact(CSV, payload)
    atomic_replace(IDENTITY, (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"trials": len(trials), "trial_csv_sha256": identity["trial_csv_sha256"], "trial_identity_sha256": identity["identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
