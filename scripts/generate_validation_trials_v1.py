"""Validate approved validation metadata and generate deterministic trials v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.verification_baseline import numeric_summary
from src.verification_trials import Utterance, generate_trials, sha256_bytes, trials_csv_text

EXPECTED_ROWS = 8504
EXPECTED_SPEAKERS = 100


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_metadata(manifest: Path, index: Path, cache_dir: Path) -> list[Utterance]:
    manifest_rows, index_rows = read_csv(manifest), read_csv(index)
    if len(manifest_rows) != EXPECTED_ROWS or len(index_rows) != EXPECTED_ROWS:
        raise ValueError("validation row count is not 8,504")
    by_path = {row["relative_audio_path"]: row for row in manifest_rows}
    indexed = {row["relative_audio_path"]: row for row in index_rows}
    if len(by_path) != EXPECTED_ROWS or len(indexed) != EXPECTED_ROWS or set(by_path) != set(indexed):
        raise ValueError("manifest/cache paths are duplicated or disagree")
    speakers = {row["speaker_id"] for row in manifest_rows}
    if len(speakers) != EXPECTED_SPEAKERS:
        raise ValueError("validation speaker count is not 100")
    positions: set[tuple[str, int]] = set()
    for path, row in by_path.items():
        cached = indexed[path]
        if row["speaker_label"] != "-1" or cached["speaker_label"] != "-1":
            raise ValueError("validation labels must all be -1")
        if row["final_split"] != "validation" or cached["final_split"] != "validation":
            raise ValueError("non-validation split found")
        if row["speaker_id"] != cached["speaker_id"]:
            raise ValueError("speaker ID mismatch")
        if row.get("filename_group") and cached.get("filename_group") != row["filename_group"]:
            raise ValueError("filename_group mismatch")
        shard = cached["feature_shard_path"]
        position = int(cached["feature_index"])
        if position < 0 or position >= 256 or (shard, position) in positions:
            raise ValueError("invalid or duplicate feature index")
        positions.add((shard, position))
        if not (cache_dir / Path(*Path(shard).parts)).is_file():
            raise FileNotFoundError(f"missing validation shard: {shard}")
    counts = Counter(row["speaker_id"] for row in manifest_rows)
    if min(counts.values()) < 2:
        raise ValueError("every validation speaker must have at least two utterances")
    return [Utterance(row["relative_audio_path"], row["speaker_id"]) for row in manifest_rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/portable/validation_manifest_v1.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/fbank_cache_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("manifests/verification"))
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--positive-cap", type=int, default=100)
    args = parser.parse_args()
    index = args.cache_dir / "validation_feature_index_v1.csv"
    config_path = args.cache_dir / "fbank_cache_config_v1.json"
    hashes_before = {str(path.as_posix()): file_hash(path) for path in (args.manifest, index, config_path)}
    rows = validate_metadata(args.manifest, index, args.cache_dir)
    trials = generate_trials(rows, args.seed, args.positive_cap)
    text = trials_csv_text(trials)
    if text != trials_csv_text(generate_trials(rows, args.seed, args.positive_cap)):
        raise RuntimeError("trial generation is nondeterministic")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_path = args.output_dir / "validation_trials_v1.csv"
    config_output = args.output_dir / "validation_trials_config_v1.json"
    trial_path.write_text(text, encoding="utf-8", newline="")
    positives = Counter(t.left_speaker_id for t in trials if t.target == 1)
    negative_participation = Counter(
        speaker for t in trials if t.target == 0 for speaker in (t.left_speaker_id, t.right_speaker_id)
    )
    speaker_pairs = Counter(tuple(sorted((t.left_speaker_id, t.right_speaker_id))) for t in trials if t.target == 0)
    config = {
        "version": 1, "seed": args.seed, "positive_cap_per_speaker": args.positive_cap,
        "positive_policy": "all unordered pairs; seeded sample without replacement; cap 100 per speaker",
        "negative_policy": "iteratively select minimum-participation distinct speakers with seeded stable ties; seeded utterances; reject duplicate unordered path pairs",
        "rows": len(rows), "speakers": len(set(row.speaker_id for row in rows)),
        "positive_trials": sum(t.target == 1 for t in trials),
        "negative_trials": sum(t.target == 0 for t in trials),
        "positive_per_speaker": dict(sorted(positives.items())),
        "positive_statistics": numeric_summary(list(positives.values())),
        "speakers_capped_below_100": sum(value < args.positive_cap for value in positives.values()),
        "negative_participation": dict(sorted(negative_participation.items())),
        "negative_participation_statistics": numeric_summary(list(negative_participation.values())),
        "negative_participation_cv": (
            numeric_summary(list(negative_participation.values()))["standard_deviation"]
            / numeric_summary(list(negative_participation.values()))["mean"]
        ),
        "unique_speaker_pairs": len(speaker_pairs),
        "speaker_pair_trials_min": min(speaker_pairs.values()),
        "speaker_pair_trials_max": max(speaker_pairs.values()),
        "trial_sha256": sha256_bytes(text.encode("utf-8")),
        "input_hashes": hashes_before,
        "schema": list(trials[0].__dict__),
    }
    config_output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes_after = {str(path.as_posix()): file_hash(path) for path in (args.manifest, index, config_path)}
    if hashes_before != hashes_after:
        raise RuntimeError("immutable input hash changed")
    print(json.dumps(config, indent=2, sort_keys=True))
    print("VALIDATION TRIAL GENERATION: PASS")


if __name__ == "__main__":
    main()
