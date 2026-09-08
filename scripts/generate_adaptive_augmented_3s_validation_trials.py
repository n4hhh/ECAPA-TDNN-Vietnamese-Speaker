"""Generate the frozen adaptive_augmented_3s_v1 validation trial package."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adaptive_augmented_3s_verification import (
    generate_validation_trials,
    read_validation_manifest,
    sha256_bytes,
    sha256_file,
    trials_csv_bytes,
)


OUTPUT_DIR = Path("manifests/verification")
CSV_NAME = "adaptive_augmented_3s_v1_validation_trials.csv"
IDENTITY_NAME = "adaptive_augmented_3s_v1_validation_trials_identity.json"
INPUTS = {
    "dataset_identity": ("manifests/adaptive_augmented_3s_v1_dataset_identity.json", "8fd9fcc0b802d56d4a96080e53180e73e317e57165c25f8d81b77ce4f18d421d"),
    "split_identity": ("splits/adaptive_augmented_3s_v1_split_identity.json", "6aa9f029cec5cfd76e00bcf4eebad92d4f61aeb39467055009ce0e06ab22885c"),
    "validation_manifest": ("manifests/adaptive_augmented_3s_v1_validation_manifest.csv", "3ed60cf7d2041d564476de52808d375c3bb9cb84e5f01689bdf5a18ced98ca69"),
}


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_exact_write(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace incompatible frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary artifact: {temporary}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    bindings: dict[str, dict[str, str]] = {}
    for name, (relative, expected) in INPUTS.items():
        path = REPO_ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"approved {name} hash mismatch: {actual}")
        bindings[name] = {"path": relative, "sha256": actual}
    rows = read_validation_manifest(REPO_ROOT / INPUTS["validation_manifest"][0])
    if len(rows) != 6076 or len({row.speaker_id for row in rows}) != 61:
        raise ValueError("validation population does not match the approved package")
    trials = generate_validation_trials(rows, seed=2026)
    csv_payload = trials_csv_bytes(trials)
    if trials_csv_bytes(generate_validation_trials(rows, seed=2026)) != csv_payload:
        raise RuntimeError("deterministic trial regeneration failed")
    identity = {
        "schema_version": 1,
        "identity_kind": "adaptive_augmented_3s_validation_trials",
        "package_version": "adaptive_augmented_3s_v1",
        "generation_algorithm": "hash_ranked_genuine_and_balanced_impostor_endpoints_v1",
        "seed": 2026,
        "input_split": "validation",
        "input_bindings": bindings,
        "speaker_count": 61,
        "trial_counts": {"genuine": 10000, "impostor": 10000, "total": 20000},
        "trial_csv_path": f"manifests/verification/{CSV_NAME}",
        "trial_csv_sha256": sha256_bytes(csv_payload),
        "trial_schema": ["trial_id", "left_audio_path", "right_audio_path", "left_speaker_id", "right_speaker_id", "target"],
        "pair_identity": "canonical_unordered_dataset_relative_audio_paths",
        "final_test_access": False,
        "timestamps_in_identity": False,
    }
    identity_without_hash = dict(identity)
    identity["identity_sha256"] = sha256_bytes(
        json.dumps(identity_without_hash, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    atomic_exact_write(output_dir / CSV_NAME, csv_payload)
    atomic_exact_write(output_dir / IDENTITY_NAME, canonical_json(identity))
    print(json.dumps({
        "trial_csv_sha256": identity["trial_csv_sha256"],
        "trial_identity_sha256": identity["identity_sha256"],
        "trials": len(trials),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
