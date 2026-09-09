"""Build the isolated final-test raw SpeechBrain Fbank cache; never score it."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_adaptive_augmented_3s_fbank_cache as base

FINAL_MANIFEST = "manifests/adaptive_augmented_3s_v1_final_test_manifest.csv"
FINAL_SHA256 = "73c1ce536b66266ef620de06f8e4fdca7777576934e032f5540aec444034a912"

base.SPLITS = ("final_test",)
base.EXPECTED_ROWS = {"final_test": 6087}
base.EXPECTED_SPEAKERS = {"final_test": 61}
base.CACHE_VERSION = "adaptive_augmented_3s_v1_final_test"
base.CONFIG_FILENAME = "fbank_cache_config_adaptive_augmented_3s_v1_final_test.json"
base.IDENTITY_FILENAME = "fbank_cache_identity_adaptive_augmented_3s_v1_final_test.json"
base.RUNTIME_FILENAME = "fbank_cache_runtime_adaptive_augmented_3s_v1_final_test.json"
base.INDEX_FILENAMES = {"final_test": "final_test_feature_index_adaptive_augmented_3s_v1.csv"}


def validate_inputs():
    path = ROOT / FINAL_MANIFEST
    actual = base.sha256_file(path)
    if actual != FINAL_SHA256:
        raise ValueError("approved final-test manifest hash mismatch")
    rows, seen = [], set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != base.PORTABLE_FIELDS:
            raise ValueError("final-test manifest schema is invalid")
        for index, raw in enumerate(reader):
            relative = base.safe_relative_path(raw["relative_audio_path"].strip(), "audio path")
            speaker = raw["speaker_id"].strip()
            if (not speaker or relative in seen or PurePosixPath(relative).parent.name != speaker
                    or raw["speaker_label"].strip() != "-1" or raw["final_split"].strip() != "final_test"):
                raise ValueError(f"final-test manifest row {index + 2} is invalid")
            seen.add(relative)
            rows.append(base.SourceRow(relative, speaker, -1, "final_test", index))
    if len(rows) != 6087 or len({row.speaker_id for row in rows}) != 61:
        raise ValueError("final-test manifest count does not match approval")
    return {"final_test_manifest": {"path": FINAL_MANIFEST, "sha256": actual}}, {"final_test": rows}


def main() -> None:
    parser = base.argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--shard-size", default=512, type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch size and shard size must be positive")
    dataset_root, cache_root = args.dataset_root.resolve(strict=True), args.cache_root.resolve()
    if cache_root.parent != ROOT / "outputs":
        raise ValueError("final-test cache root must be directly under outputs/")
    bindings, rows = validate_inputs()
    config = base.cache_config(bindings, args.shard_size, args.batch_size)
    config.update({"identity_kind": "adaptive_augmented_3s_final_test_fbank_cache_config", "included_splits": ["final_test"], "expected_rows": {"final_test": 6087}, "evaluation_label": -1, "final_test_cache_only": True})
    config.pop("train_class_count", None); config.pop("train_label_range", None); config.pop("validation_label", None)
    base.check_no_absolute_dataset_root(config, dataset_root)
    indexes = {"final_test": base.index_text(rows["final_test"], "final_test", args.shard_size)}
    complete = base.prepare_plan(cache_root, base.canonical_json(config), indexes)
    frontend = base.SpeechBrainECAPAFrontend(device=args.device)
    source_before = base.source_state(dataset_root, rows)
    extraction = {"final_test": {"written": 0, "reused": 0, "total": 0}} if complete else base.extract_cache(frontend, dataset_root, cache_root, rows, args.shard_size, args.batch_size)
    validation = base.validate_complete_cache(cache_root, rows, indexes, args.shard_size)
    if source_before != base.source_state(dataset_root, rows):
        raise RuntimeError("source audio changed during final-test cache construction")
    validation["final_test_cached_entries"] = validation["total_cached_utterances"]
    identity = base.build_identity(bindings, cache_root / base.CONFIG_FILENAME, cache_root, validation)
    identity.update({"identity_kind": "adaptive_augmented_3s_final_test_fbank_cache", "final_test_cache_absent": False, "final_test_cache_only": True})
    identity["identity_sha256"] = base.canonical_digest({key: value for key, value in identity.items() if key != "identity_sha256"})
    base.atomic_text(cache_root / base.IDENTITY_FILENAME, base.canonical_json(identity))
    compatibility = base.fresh_vs_cached(frontend, dataset_root, cache_root, rows, args.shard_size)
    first_payload = base.torch.load(cache_root / "final_test" / "shard_00000.pt", map_location="cpu", weights_only=False)
    first_features = first_payload["features"][:4]
    if (tuple(first_features.shape) != (4, 301, 80) or first_features.dtype != base.torch.float32
            or not bool(base.torch.isfinite(first_features).all().item())
            or first_payload["final_split"] != "final_test" or first_payload["speaker_labels"][:4].tolist() != [-1] * 4):
        raise ValueError("final-test cache mini-batch check failed")
    dataset_check = {"final_test": {"batch_shape": list(first_features.shape), "labels": first_payload["speaker_labels"][:4].tolist()}}
    print(json.dumps({"identity_sha256": identity["identity_sha256"], "cache_count": validation["total_cached_utterances"], "extraction": extraction, "compatibility": compatibility, "dataset_check": dataset_check}, sort_keys=True))


if __name__ == "__main__":
    main()
