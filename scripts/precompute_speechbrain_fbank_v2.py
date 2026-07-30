"""Build and fully validate the approved VieSpeaker2.0 train/validation Fbank cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import speechbrain
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.speechbrain_frontend import SpeechBrainECAPAFrontend


SPLITS = ("train", "validation")
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE_HZ = 16_000
FEATURE_BINS = 80
CACHE_SCHEMA_VERSION = 2
SHARD_SCHEMA_VERSION = 2
CONFIG_FILENAME = "fbank_cache_config_v2.json"
IDENTITY_FILENAME = "fbank_cache_identity_v2.json"
RUNTIME_RESULT_FILENAME = "fbank_cache_runtime_result_v2.json"
INDEX_FILENAMES = {
    "train": "train_feature_index_v2.csv",
    "validation": "validation_feature_index_v2.csv",
}
LOG_FILENAMES = {"extraction.stdout.log", "extraction.stderr.log"}
INDEX_FIELDS = (
    "audio_path",
    "speaker_id",
    "label",
    "final_split",
    "shard_path",
    "within_shard_index",
    "feature_frames",
    "feature_bins",
    "feature_dtype",
    "manifest_row_index",
    "filename_group",
    "duplicate_group",
    "manifest_version",
)
V2_SHARD_KEYS = {
    "schema_version",
    "features",
    "speaker_labels",
    "speaker_ids",
    "relative_audio_paths",
    "manifest_row_indices",
    "final_split",
}
APPROVED_INPUTS = {
    "full_manifest": (
        "manifests/v2/full_manifest_v2.csv",
        "26a0157abce3bb00ce5f0ca16f9b964e180f72484e53f9d2602577f5ce4acf8f",
    ),
    "full_manifest_identity": (
        "manifests/v2/full_manifest_v2_identity.json",
        "f7b6c1cbc95b8a0d20b596b4841de7ebc26e69d6bd7c130801937dab681c544e",
    ),
    "speaker_split": (
        "splits/v2/speaker_split_v2.csv",
        "cbbcdcd4d3561ff2470a6cd713187cbb612939e643e5bc8cf4ed25504539f87e",
    ),
    "speaker_split_identity": (
        "splits/v2/speaker_split_v2_identity.json",
        "87d2a542ae1716f5d143e27835bf0478413e672cfa131462c0410808498c3c67",
    ),
    "split_policy": (
        "splits/v2/split_policy_v2.json",
        "3e414836b4fe307841810cffa56c3f0040d0c662d265d276d73a7d00a16a5d10",
    ),
    "portable_package_identity": (
        "manifests/portable_v2/portable_manifests_v2_identity.json",
        "29f374366c917fb6c54dcd44eeeff59ccc46a732b270188e7157a586e4d9e7c5",
    ),
    "train_manifest": (
        "manifests/portable_v2/train_manifest_v2.csv",
        "f76aa0321f5f9a2714b2bad9f4b9ab0fd155075f26b50397f79931c8a4bd552b",
    ),
    "validation_manifest": (
        "manifests/portable_v2/validation_manifest_v2.csv",
        "9c85332cbcd3e33818055c526c0bc54e5b86e7a2c4ed8b3c869433412c24a6fc",
    ),
    "train_label_mapping": (
        "manifests/portable_v2/speaker_to_label_v2.json",
        "9d4e9015d25f023b8466f7932c296faece937104b17120bdb85c45ad10623cd8",
    ),
}


@dataclass(frozen=True)
class SourceRow:
    audio_path: str
    speaker_id: str
    label: int
    final_split: str
    filename_group: str
    duplicate_group: str
    manifest_version: str
    manifest_row_index: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to replace stale temporary file: {temporary}")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to replace stale temporary file: {temporary}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def safe_portable_path(value: str, description: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in value
        or Path(value).is_absolute()
    ):
        raise ValueError(f"unsafe {description}: {value!r}")
    return value


def reject_forbidden_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(
            f"split must be one of {SPLITS}; final-test and excluded cache access is forbidden"
        )


def require_sha256(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing approved {description}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"approved {description} identity mismatch: expected {expected}, got {actual}"
        )
    return actual


def read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _require_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ValueError(f"{description} mismatch: expected {expected!r}, got {actual!r}")


def validate_approved_input_hashes(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, (relative_path, expected_hash) in APPROVED_INPUTS.items():
        path = project_root / Path(*PurePosixPath(relative_path).parts)
        actual_hash = require_sha256(path, expected_hash, name.replace("_", " "))
        bindings[name] = {"path": relative_path, "sha256": actual_hash}
    return bindings


def read_portable_manifest(
    path: Path,
    split: str,
    *,
    expected_rows: int,
    label_range: tuple[int, int],
    label_mapping: dict[str, int],
) -> list[SourceRow]:
    reject_forbidden_split(split)
    required = {
        "relative_audio_path",
        "speaker_id",
        "speaker_label",
        "final_split",
        "filename_group",
        "duplicate_group",
        "manifest_version",
    }
    rows: list[SourceRow] = []
    seen_paths: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for manifest_row_index, raw in enumerate(reader):
            line = manifest_row_index + 2
            try:
                audio_path = safe_portable_path(
                    raw["relative_audio_path"].strip(), "portable audio path"
                )
                speaker_id = raw["speaker_id"].strip()
                label = int(raw["speaker_label"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line}: malformed manifest row: {error}") from error
            final_split = raw["final_split"].strip()
            filename_group = raw["filename_group"].strip()
            duplicate_group = raw["duplicate_group"].strip()
            manifest_version = raw["manifest_version"].strip()
            if final_split != split:
                raise ValueError(
                    f"{path}:{line}: expected split {split!r}, got {final_split!r}"
                )
            if not speaker_id or not filename_group or manifest_version != "v2":
                raise ValueError(f"{path}:{line}: invalid speaker/provenance/version metadata")
            if PurePosixPath(audio_path).parent.name != speaker_id:
                raise ValueError(
                    f"{path}:{line}: direct-parent speaker identity disagrees with speaker_id"
                )
            if audio_path in seen_paths:
                raise ValueError(f"{path}:{line}: duplicate audio path {audio_path!r}")
            seen_paths.add(audio_path)
            if split == "train":
                if not label_range[0] <= label <= label_range[1]:
                    raise ValueError(
                        f"{path}:{line}: train label outside {label_range[0]}..{label_range[1]}"
                    )
                if label_mapping.get(speaker_id) != label:
                    raise ValueError(
                        f"{path}:{line}: speaker label disagrees with approved mapping"
                    )
            elif label != -1:
                raise ValueError(f"{path}:{line}: validation label must be -1")
            rows.append(
                SourceRow(
                    audio_path=audio_path,
                    speaker_id=speaker_id,
                    label=label,
                    final_split=final_split,
                    filename_group=filename_group,
                    duplicate_group=duplicate_group,
                    manifest_version=manifest_version,
                    manifest_row_index=manifest_row_index,
                )
            )
    if len(rows) != expected_rows:
        raise ValueError(
            f"{split} manifest has {len(rows)} rows, expected authoritative {expected_rows}"
        )
    return rows


def validate_approved_inputs(
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, list[SourceRow]]]:
    bindings = validate_approved_input_hashes(project_root)
    full_identity = read_json(
        project_root / APPROVED_INPUTS["full_manifest_identity"][0],
        "full-manifest identity",
    )
    split_identity = read_json(
        project_root / APPROVED_INPUTS["speaker_split_identity"][0],
        "speaker-split identity",
    )
    split_policy = read_json(
        project_root / APPROVED_INPUTS["split_policy"][0], "split policy"
    )
    portable_identity = read_json(
        project_root / APPROVED_INPUTS["portable_package_identity"][0],
        "portable-package identity",
    )
    _require_equal(
        full_identity.get("manifest_sha256"),
        bindings["full_manifest"]["sha256"],
        "full-manifest binding",
    )
    for identity_name, identity in (
        ("speaker-split", split_identity),
        ("portable-package", portable_identity),
    ):
        _require_equal(
            identity.get("full_manifest_sha256"),
            bindings["full_manifest"]["sha256"],
            f"{identity_name} full-manifest binding",
        )
        _require_equal(
            identity.get("full_manifest_identity_sha256"),
            bindings["full_manifest_identity"]["sha256"],
            f"{identity_name} full-manifest-identity binding",
        )
        _require_equal(
            identity.get("split_csv_sha256"),
            bindings["speaker_split"]["sha256"],
            f"{identity_name} speaker-split binding",
        )
        _require_equal(
            identity.get("split_policy_sha256"),
            bindings["split_policy"]["sha256"],
            f"{identity_name} split-policy binding",
        )
        _require_equal(
            identity.get("train_manifest_sha256"),
            bindings["train_manifest"]["sha256"],
            f"{identity_name} train-manifest binding",
        )
        _require_equal(
            identity.get("validation_manifest_sha256"),
            bindings["validation_manifest"]["sha256"],
            f"{identity_name} validation-manifest binding",
        )
        _require_equal(
            identity.get("speaker_to_label_sha256"),
            bindings["train_label_mapping"]["sha256"],
            f"{identity_name} label-mapping binding",
        )
    _require_equal(
        split_policy.get("full_manifest_sha256"),
        bindings["full_manifest"]["sha256"],
        "split-policy full-manifest binding",
    )
    _require_equal(
        split_policy.get("full_manifest_identity_sha256"),
        bindings["full_manifest_identity"]["sha256"],
        "split-policy full-manifest-identity binding",
    )
    if portable_identity.get("final_test_quarantine") is not True:
        raise ValueError("portable identity does not preserve final-test quarantine")

    row_counts_raw = portable_identity.get("row_counts")
    speaker_counts_raw = portable_identity.get("speaker_counts")
    label_range_raw = portable_identity.get("train_label_range")
    if not isinstance(row_counts_raw, dict) or not isinstance(speaker_counts_raw, dict):
        raise ValueError("portable identity is missing authoritative row/speaker counts")
    if (
        not isinstance(label_range_raw, list)
        or len(label_range_raw) != 2
        or not all(isinstance(value, int) for value in label_range_raw)
    ):
        raise ValueError("portable identity has an invalid train label range")
    row_counts = {split: row_counts_raw.get(split) for split in SPLITS}
    speaker_counts = {split: speaker_counts_raw.get(split) for split in SPLITS}
    if not all(isinstance(value, int) and value > 0 for value in row_counts.values()):
        raise ValueError("portable identity has invalid train/validation row counts")
    if not all(
        isinstance(value, int) and value > 0 for value in speaker_counts.values()
    ):
        raise ValueError("portable identity has invalid train/validation speaker counts")
    class_count = portable_identity.get("train_class_count")
    if (
        not isinstance(class_count, int)
        or class_count < 1
        or label_range_raw != [0, class_count - 1]
    ):
        raise ValueError("portable identity class count and label range disagree")

    mapping_path = project_root / APPROVED_INPUTS["train_label_mapping"][0]
    mapping_raw = read_json(mapping_path, "train label mapping")
    try:
        label_mapping = {str(key): int(value) for key, value in mapping_raw.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid train label mapping: {error}") from error
    if (
        len(label_mapping) != class_count
        or set(label_mapping.values()) != set(range(class_count))
    ):
        raise ValueError("train label mapping is not contiguous and complete")

    label_range = (label_range_raw[0], label_range_raw[1])
    rows = {
        "train": read_portable_manifest(
            project_root / APPROVED_INPUTS["train_manifest"][0],
            "train",
            expected_rows=row_counts["train"],
            label_range=label_range,
            label_mapping=label_mapping,
        ),
        "validation": read_portable_manifest(
            project_root / APPROVED_INPUTS["validation_manifest"][0],
            "validation",
            expected_rows=row_counts["validation"],
            label_range=label_range,
            label_mapping=label_mapping,
        ),
    }
    train_speakers = {row.speaker_id for row in rows["train"]}
    validation_speakers = {row.speaker_id for row in rows["validation"]}
    if len(train_speakers) != speaker_counts["train"]:
        raise ValueError("train speaker count disagrees with approved identity")
    if len(validation_speakers) != speaker_counts["validation"]:
        raise ValueError("validation speaker count disagrees with approved identity")
    if train_speakers & validation_speakers:
        raise ValueError("train and validation speakers are not disjoint")
    train_labels = {row.label for row in rows["train"]}
    if train_labels != set(range(class_count)):
        raise ValueError("train labels do not cover the authoritative range")
    if any(row.label != -1 for row in rows["validation"]):
        raise ValueError("validation labels are not all -1")
    all_paths = [row.audio_path for split in SPLITS for row in rows[split]]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("audio paths overlap across train and validation")

    authority = {
        "bindings": bindings,
        "row_counts": row_counts,
        "speaker_counts": speaker_counts,
        "train_class_count": class_count,
        "train_label_range": list(label_range),
        "allowed_filename_groups": sorted(
            {row.filename_group for split in SPLITS for row in rows[split]}
        ),
    }
    return authority, rows


def source_path(dataset_root: Path, row: SourceRow) -> Path:
    return dataset_root / Path(*PurePosixPath(row.audio_path).parts)


def source_state(
    dataset_root: Path, rows: dict[str, list[SourceRow]]
) -> dict[str, tuple[int, int]]:
    state: dict[str, tuple[int, int]] = {}
    for split in SPLITS:
        for row in rows[split]:
            path = source_path(dataset_root, row)
            try:
                stat = path.stat()
            except OSError as error:
                raise FileNotFoundError(f"unavailable approved source WAV {row.audio_path}") from error
            if not path.is_file():
                raise ValueError(f"approved source path is not a file: {row.audio_path}")
            state[row.audio_path] = (stat.st_size, stat.st_mtime_ns)
    if len(state) != sum(len(rows[split]) for split in SPLITS):
        raise ValueError("source-state rows do not reconcile with approved allowlist")
    return state


def load_waveform(
    dataset_root: Path, row: SourceRow, expected_samples: int | None
) -> torch.Tensor:
    path = source_path(dataset_root, row)
    try:
        # torchaudio's normalize=True is PCM-to-float decoding, not per-file
        # loudness or peak normalization. No gain, resampling, or rewriting occurs.
        waveform, sample_rate = torchaudio.load(str(path), normalize=True)
    except Exception as error:
        raise RuntimeError(f"failed to load approved WAV {row.audio_path}: {error}") from error
    if sample_rate != SAMPLE_RATE_HZ:
        raise ValueError(
            f"{row.audio_path}: expected {SAMPLE_RATE_HZ} Hz, got {sample_rate}"
        )
    if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[1] < 1:
        raise ValueError(
            f"{row.audio_path}: expected non-empty mono waveform, got {tuple(waveform.shape)}"
        )
    if expected_samples is not None and waveform.shape[1] != expected_samples:
        raise ValueError(
            f"{row.audio_path}: expected {expected_samples} supplied samples, "
            f"got {waveform.shape[1]}"
        )
    waveform = waveform.squeeze(0).to(dtype=torch.float32, device="cpu").contiguous()
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError(f"{row.audio_path}: waveform contains NaN or Inf")
    return waveform


def validate_feature_batch(
    features: torch.Tensor, batch_count: int, feature_shape: tuple[int, int]
) -> None:
    if (
        not isinstance(features, torch.Tensor)
        or tuple(features.shape) != (batch_count, *feature_shape)
        or features.dtype != torch.float32
        or not bool(torch.isfinite(features).all().item())
    ):
        raise ValueError(
            f"invalid raw Fbank batch: expected [{batch_count}, "
            f"{feature_shape[0]}, {feature_shape[1]}] float32 finite, "
            f"got {getattr(features, 'shape', None)} {getattr(features, 'dtype', None)}"
        )


def measure_feature_contract(
    frontend: SpeechBrainECAPAFrontend,
    dataset_root: Path,
    train_rows: list[SourceRow],
) -> tuple[int, int, dict[str, Any]]:
    positions = (0, len(train_rows) // 3, (2 * len(train_rows)) // 3, len(train_rows) - 1)
    first = load_waveform(dataset_root, train_rows[positions[0]], expected_samples=None)
    waveform_samples = int(first.shape[0])
    waveforms = [first]
    waveforms.extend(
        load_waveform(dataset_root, train_rows[position], waveform_samples)
        for position in positions[1:]
    )
    batch = torch.stack(waveforms)
    with torch.inference_mode():
        features = frontend.compute_features(batch)
    if (
        features.ndim != 3
        or features.shape[0] != len(positions)
        or features.shape[1] < 1
        or features.shape[2] != FEATURE_BINS
    ):
        raise ValueError(f"unexpected measured SpeechBrain Fbank shape: {tuple(features.shape)}")
    measured_frames = int(features.shape[1])
    validate_feature_batch(features, len(positions), (measured_frames, FEATURE_BINS))
    return measured_frames, waveform_samples, {
        "split": "train",
        "manifest_row_indices": list(positions),
        "waveform_shape": [len(positions), waveform_samples],
        "feature_shape": list(features.shape),
        "feature_dtype": "float32",
        "finite": True,
        "raw_pre_normalization": True,
        "transposed": False,
    }


def run_batch_size_preflight(
    frontend: SpeechBrainECAPAFrontend,
    dataset_root: Path,
    train_rows: list[SourceRow],
    *,
    requested_batch_size: int,
    waveform_samples: int,
    feature_shape: tuple[int, int],
) -> tuple[int, bool, dict[str, Any]]:
    def run(batch_size: int) -> dict[str, Any]:
        waveforms = torch.stack(
            [
                load_waveform(dataset_root, row, waveform_samples)
                for row in train_rows[:batch_size]
            ]
        )
        with torch.inference_mode():
            features = frontend.compute_features(waveforms)
        validate_feature_batch(features, batch_size, feature_shape)
        return {
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": batch_size,
            "waveform_shape": list(waveforms.shape),
            "feature_shape": list(features.shape),
            "finite": True,
            "device": str(frontend.device),
        }

    try:
        return requested_batch_size, False, run(requested_batch_size)
    except torch.cuda.OutOfMemoryError:
        if frontend.device.type != "cuda" or requested_batch_size != 64:
            raise
        torch.cuda.empty_cache()
        fallback = 32
        result = run(fallback)
        result["fallback_reason"] = "cuda_out_of_memory_at_batch_size_64"
        return fallback, True, result


def shard_relative_path(split: str, shard_number: int) -> str:
    reject_forbidden_split(split)
    if shard_number < 0:
        raise ValueError("shard number must be non-negative")
    return f"{split}/shard_{shard_number:05d}.pt"


def plan_index_rows(
    rows: list[SourceRow],
    split: str,
    shard_size: int,
    feature_shape: tuple[int, int],
) -> list[dict[str, str | int]]:
    reject_forbidden_split(split)
    if shard_size < 1:
        raise ValueError("shard size must be positive")
    planned: list[dict[str, str | int]] = []
    for dataset_index, row in enumerate(rows):
        if row.final_split != split or row.manifest_row_index != dataset_index:
            raise ValueError("manifest row order or split identity is inconsistent")
        planned.append(
            {
                "audio_path": row.audio_path,
                "speaker_id": row.speaker_id,
                "label": row.label,
                "final_split": split,
                "shard_path": shard_relative_path(split, dataset_index // shard_size),
                "within_shard_index": dataset_index % shard_size,
                "feature_frames": feature_shape[0],
                "feature_bins": feature_shape[1],
                "feature_dtype": "float32",
                "manifest_row_index": row.manifest_row_index,
                "filename_group": row.filename_group,
                "duplicate_group": row.duplicate_group,
                "manifest_version": row.manifest_version,
            }
        )
    return planned


def csv_text(rows: Iterable[dict[str, str | int]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build_config(
    authority: dict[str, Any],
    *,
    feature_shape: tuple[int, int],
    waveform_samples: int,
    requested_batch_size: int,
    extraction_batch_size: int,
    batch_size_fallback_used: bool,
    shard_size: int,
    device_used: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": "v2",
        "model_source": MODEL_SOURCE,
        "frontend": MODEL_SOURCE,
        "speechbrain_version": speechbrain.__version__,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "input_bindings": authority["bindings"],
        "included_splits": list(SPLITS),
        "feature_stage": "raw_compute_features_before_mean_var_norm",
        "feature_shape": list(feature_shape),
        "feature_dtype": "float32",
        "raw_pre_normalization": True,
        "transposed": False,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "waveform_samples": waveform_samples,
        "shard_schema_version": SHARD_SCHEMA_VERSION,
        "shard_size": shard_size,
        "requested_batch_size": requested_batch_size,
        "extraction_batch_size": extraction_batch_size,
        "batch_size_fallback_used": batch_size_fallback_used,
        "device_used": device_used,
        "expected_rows": authority["row_counts"],
        "speaker_counts": authority["speaker_counts"],
        "train_class_count": authority["train_class_count"],
        "train_label_range": authority["train_label_range"],
        "validation_label": -1,
        "allowed_filename_groups": authority["allowed_filename_groups"],
        "index_filenames": INDEX_FILENAMES,
        "index_fields": list(INDEX_FIELDS),
        "index_order": "portable_manifest_row_order",
        "shard_assignment": "manifest_row_index divmod shard_size",
        "shard_path_pattern": "<split>/shard_<zero_padded_5_digit_number>.pt",
        "path_base_semantics": {
            "audio_path": "relative_to_runtime_supplied_dataset_root",
            "shard_path": "relative_to_cache_directory",
            "separator": "/",
        },
        "timestamps_in_identity_relevant_content": False,
    }


def prepare_cache_metadata(
    cache_dir: Path,
    config_text: str,
    index_texts: dict[str, str],
    *,
    resume: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = cache_dir / CONFIG_FILENAME
    index_paths = {split: cache_dir / INDEX_FILENAMES[split] for split in SPLITS}
    if resume:
        if not config_path.is_file() or any(
            not index_paths[split].is_file() for split in SPLITS
        ):
            raise FileNotFoundError(
                "resume requires the complete v2 config and both deterministic indexes"
            )
        if config_path.read_text(encoding="utf-8") != config_text:
            raise ValueError("resume rejected incompatible cache configuration")
        for split in SPLITS:
            if index_paths[split].read_text(encoding="utf-8") != index_texts[split]:
                raise ValueError(f"resume rejected incompatible {split} index plan")
        return

    existing = {path.name for path in cache_dir.iterdir()}
    unexpected = existing - LOG_FILENAMES
    if unexpected:
        raise FileExistsError(
            f"cache directory is not pristine; use --resume only for a compatible build: "
            f"{sorted(unexpected)}"
        )
    atomic_text(config_path, config_text)
    for split in SPLITS:
        atomic_text(index_paths[split], index_texts[split])


def validate_shard(
    shard: Any,
    expected_rows: list[SourceRow],
    split: str,
    feature_shape: tuple[int, int],
    shard_size: int,
) -> None:
    reject_forbidden_split(split)
    if not isinstance(shard, dict) or set(shard) != V2_SHARD_KEYS:
        raise ValueError(f"malformed v2 shard keys: {getattr(shard, 'keys', lambda: [])()}")
    if shard["schema_version"] != SHARD_SCHEMA_VERSION:
        raise ValueError("shard schema version mismatch")
    count = len(expected_rows)
    if count < 1 or count > shard_size:
        raise ValueError(f"shard count {count} is outside 1..{shard_size}")
    features = shard["features"]
    labels = shard["speaker_labels"]
    if (
        not isinstance(features, torch.Tensor)
        or tuple(features.shape) != (count, *feature_shape)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not features.is_contiguous()
        or not bool(torch.isfinite(features).all().item())
    ):
        raise ValueError("invalid float32 finite non-transposed feature tensor")
    if (
        not isinstance(labels, torch.Tensor)
        or tuple(labels.shape) != (count,)
        or labels.dtype != torch.long
        or labels.device.type != "cpu"
    ):
        raise ValueError("invalid shard labels")
    if labels.tolist() != [row.label for row in expected_rows]:
        raise ValueError("shard labels disagree with manifest")
    if shard["speaker_ids"] != [row.speaker_id for row in expected_rows]:
        raise ValueError("shard speaker IDs disagree with manifest")
    if shard["relative_audio_paths"] != [row.audio_path for row in expected_rows]:
        raise ValueError("shard audio paths disagree with manifest")
    if shard["manifest_row_indices"] != [
        row.manifest_row_index for row in expected_rows
    ]:
        raise ValueError("shard manifest row identities disagree with manifest")
    if shard["final_split"] != split:
        raise ValueError("shard split disagrees with manifest")


def extract_split(
    frontend: SpeechBrainECAPAFrontend,
    dataset_root: Path,
    cache_dir: Path,
    rows: list[SourceRow],
    split: str,
    *,
    feature_shape: tuple[int, int],
    waveform_samples: int,
    batch_size: int,
    shard_size: int,
    resume: bool,
) -> dict[str, int]:
    reject_forbidden_split(split)
    written = 0
    skipped = 0
    shard_count = math.ceil(len(rows) / shard_size)
    for shard_number, start in enumerate(range(0, len(rows), shard_size)):
        selected = rows[start : start + shard_size]
        target = cache_dir / Path(
            *PurePosixPath(shard_relative_path(split, shard_number)).parts
        )
        if target.exists():
            if not resume:
                raise FileExistsError(f"refusing to overwrite existing shard: {target}")
            try:
                payload = torch.load(target, map_location="cpu", weights_only=False)
                validate_shard(payload, selected, split, feature_shape, shard_size)
            except Exception as error:
                raise RuntimeError(f"resume rejected invalid shard {target}: {error}") from error
            skipped += 1
            print(
                f"{split}: shard {shard_number + 1}/{shard_count} validated and skipped",
                flush=True,
            )
            continue

        feature_batches: list[torch.Tensor] = []
        for batch_start in range(0, len(selected), batch_size):
            batch_rows = selected[batch_start : batch_start + batch_size]
            waveforms = torch.stack(
                [
                    load_waveform(dataset_root, row, waveform_samples)
                    for row in batch_rows
                ]
            )
            try:
                with torch.inference_mode():
                    features = frontend.compute_features(waveforms)
            except Exception as error:
                nearby = ", ".join(row.audio_path for row in batch_rows[:3])
                raise RuntimeError(
                    f"feature extraction failed near approved paths {nearby}: {error}"
                ) from error
            features = (
                features.detach().to(device="cpu", dtype=torch.float32).contiguous()
            )
            validate_feature_batch(features, len(batch_rows), feature_shape)
            feature_batches.append(features)
        payload = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "features": torch.cat(feature_batches, dim=0).contiguous(),
            "speaker_labels": torch.tensor(
                [row.label for row in selected], dtype=torch.long
            ),
            "speaker_ids": [row.speaker_id for row in selected],
            "relative_audio_paths": [row.audio_path for row in selected],
            "manifest_row_indices": [
                row.manifest_row_index for row in selected
            ],
            "final_split": split,
        }
        validate_shard(payload, selected, split, feature_shape, shard_size)
        atomic_torch_save(payload, target)
        validate_shard(
            torch.load(target, map_location="cpu", weights_only=False),
            selected,
            split,
            feature_shape,
            shard_size,
        )
        written += 1
        print(f"{split}: shard {shard_number + 1}/{shard_count} complete", flush=True)
    return {"written": written, "skipped": skipped, "total": shard_count}


def load_cached_feature(
    cache_dir: Path,
    row: SourceRow,
    split: str,
    shard_size: int,
) -> torch.Tensor:
    shard_number, position = divmod(row.manifest_row_index, shard_size)
    shard = torch.load(
        cache_dir
        / Path(*PurePosixPath(shard_relative_path(split, shard_number)).parts),
        map_location="cpu",
        weights_only=False,
    )
    return shard["features"][position]


def validate_complete_cache(
    cache_dir: Path,
    rows: dict[str, list[SourceRow]],
    *,
    config_text: str,
    index_texts: dict[str, str],
    feature_shape: tuple[int, int],
    shard_size: int,
    identity_may_exist: bool,
) -> dict[str, Any]:
    allowed_root_files = {
        CONFIG_FILENAME,
        *INDEX_FILENAMES.values(),
        *LOG_FILENAMES,
    }
    if identity_may_exist:
        allowed_root_files.update({IDENTITY_FILENAME, RUNTIME_RESULT_FILENAME})
    root_directories = {path.name for path in cache_dir.iterdir() if path.is_dir()}
    if root_directories != set(SPLITS):
        raise ValueError(
            f"cache split directories must be exactly {list(SPLITS)}, got "
            f"{sorted(root_directories)}"
        )
    unexpected_files = {
        path.name
        for path in cache_dir.iterdir()
        if path.is_file() and path.name not in allowed_root_files
    }
    if unexpected_files:
        raise ValueError(f"unexpected cache-root files: {sorted(unexpected_files)}")
    if (cache_dir / CONFIG_FILENAME).read_text(encoding="utf-8") != config_text:
        raise ValueError("finalized config differs from deterministic plan")

    seen_paths: set[str] = set()
    shard_counts: dict[str, int] = {}
    shard_bytes: dict[str, int] = {}
    for split in SPLITS:
        if (cache_dir / INDEX_FILENAMES[split]).read_text(
            encoding="utf-8"
        ) != index_texts[split]:
            raise ValueError(f"finalized {split} index differs from deterministic plan")
        expected_shards = math.ceil(len(rows[split]) / shard_size)
        shard_counts[split] = expected_shards
        expected_names = {
            f"shard_{number:05d}.pt" for number in range(expected_shards)
        }
        split_dir = cache_dir / split
        actual_names = {path.name for path in split_dir.iterdir() if path.is_file()}
        if actual_names != expected_names:
            raise ValueError(
                f"{split} shard files differ from plan: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        if any(path.is_dir() for path in split_dir.iterdir()):
            raise ValueError(f"unexpected nested directory in {split} cache")
        total_split_bytes = 0
        for shard_number in range(expected_shards):
            start = shard_number * shard_size
            selected = rows[split][start : start + shard_size]
            shard_path = split_dir / f"shard_{shard_number:05d}.pt"
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            validate_shard(payload, selected, split, feature_shape, shard_size)
            total_split_bytes += shard_path.stat().st_size
            for row in selected:
                if row.audio_path in seen_paths:
                    raise ValueError(f"duplicate cached audio path: {row.audio_path}")
                seen_paths.add(row.audio_path)
        shard_bytes[split] = total_split_bytes
    authoritative_total = sum(len(rows[split]) for split in SPLITS)
    if len(seen_paths) != authoritative_total:
        raise ValueError("global cached paths do not reconcile")
    return {
        "row_counts": {split: len(rows[split]) for split in SPLITS},
        "shard_counts": shard_counts,
        "shard_bytes": shard_bytes,
        "total_cached_utterances": authoritative_total,
        "total_shard_bytes": sum(shard_bytes.values()),
        "global_cached_paths_unique": True,
        "included_splits": list(SPLITS),
        "unexpected_shards_or_split_directories": 0,
    }


def validate_reextraction(
    frontend: SpeechBrainECAPAFrontend,
    dataset_root: Path,
    cache_dir: Path,
    rows: dict[str, list[SourceRow]],
    *,
    waveform_samples: int,
    feature_shape: tuple[int, int],
    extraction_batch_size: int,
    shard_size: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    global_maximum = 0.0
    for split in SPLITS:
        positions = (0, len(rows[split]) // 2)
        maximum = 0.0
        for position in positions:
            row = rows[split][position]
            shard_start = (position // shard_size) * shard_size
            within_shard = position - shard_start
            batch_start = (
                shard_start
                + (within_shard // extraction_batch_size) * extraction_batch_size
            )
            batch_end = min(
                batch_start + extraction_batch_size,
                shard_start + shard_size,
                len(rows[split]),
            )
            batch_rows = rows[split][batch_start:batch_end]
            waveforms = torch.stack(
                [
                    load_waveform(dataset_root, batch_row, waveform_samples)
                    for batch_row in batch_rows
                ]
            )
            with torch.inference_mode():
                reextracted_batch = frontend.compute_features(waveforms)
            reextracted_batch = (
                reextracted_batch.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
            validate_feature_batch(
                reextracted_batch, len(batch_rows), feature_shape
            )
            reextracted = reextracted_batch[position - batch_start]
            cached = load_cached_feature(cache_dir, row, split, shard_size)
            difference = float((cached - reextracted).abs().max().item())
            maximum = max(maximum, difference)
        if maximum != 0.0:
            raise ValueError(
                f"{split} fixed-sample re-extraction was not exactly reproducible: {maximum}"
            )
        result[split] = {
            "manifest_row_indices": list(positions),
            "sample_count": len(positions),
            "maximum_absolute_difference": maximum,
            "exact_tensor_equality": True,
        }
        global_maximum = max(global_maximum, maximum)
    result["global_maximum_absolute_difference"] = global_maximum
    result["exact_tensor_equality"] = True
    return result


def build_identity(
    authority: dict[str, Any],
    config: dict[str, Any],
    cache_validation: dict[str, Any],
    reproducibility: dict[str, Any],
    *,
    cache_dir: Path,
    config_planning_sha256: str,
    index_planning_sha256: dict[str, str],
    source_preservation_match: bool,
) -> dict[str, Any]:
    index_hashes = {
        split: file_sha256(cache_dir / INDEX_FILENAMES[split]) for split in SPLITS
    }
    if index_hashes != index_planning_sha256:
        raise ValueError("final index hashes differ from deterministic planning hashes")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity_kind": "fbank_cache_v2",
        "cache_version": "v2",
        "model_source": MODEL_SOURCE,
        "speechbrain_version": config["speechbrain_version"],
        "torch_version": config["torch_version"],
        "torchaudio_version": config["torchaudio_version"],
        "input_bindings": authority["bindings"],
        "config_path": f"outputs/fbank_cache_v2/{CONFIG_FILENAME}",
        "config_sha256": file_sha256(cache_dir / CONFIG_FILENAME),
        "included_splits": list(SPLITS),
        "feature_shape": config["feature_shape"],
        "feature_dtype": "float32",
        "raw_pre_normalization": True,
        "transposed": False,
        "shard_size": config["shard_size"],
        "extraction_batch_size": config["extraction_batch_size"],
        "requested_batch_size": config["requested_batch_size"],
        "batch_size_fallback_used": config["batch_size_fallback_used"],
        "device_used": config["device_used"],
        "row_counts": cache_validation["row_counts"],
        "speaker_counts": authority["speaker_counts"],
        "train_class_count": authority["train_class_count"],
        "train_label_range": authority["train_label_range"],
        "validation_label": -1,
        "index_paths": {
            split: f"outputs/fbank_cache_v2/{INDEX_FILENAMES[split]}"
            for split in SPLITS
        },
        "index_sha256": index_hashes,
        "shard_counts": cache_validation["shard_counts"],
        "total_cached_utterances": cache_validation["total_cached_utterances"],
        "total_shard_bytes": cache_validation["total_shard_bytes"],
        "deterministic_row_order": "portable_manifest_row_order",
        "deterministic_shard_order": "manifest_row_index divmod shard_size",
        "path_base_semantics": config["path_base_semantics"],
        "planning_reproducibility": {
            "config_generated_twice_identically": True,
            "config_planning_sha256": config_planning_sha256,
            "indexes_generated_twice_identically": True,
            "index_planning_sha256": index_planning_sha256,
        },
        "feature_reextraction_reproducibility": reproducibility,
        "source_allowlist_state_unchanged": source_preservation_match,
        "source_allowlist_checked_rows": cache_validation["total_cached_utterances"],
        "final_test_cache_absent": True,
        "timestamps_in_identity": False,
    }


def assert_no_absolute_dataset_root(value: Any, dataset_root: Path) -> None:
    serialized = canonical_json_text(value)
    candidates = {
        str(dataset_root),
        str(dataset_root).replace("\\", "/"),
        str(dataset_root).replace("\\", "\\\\"),
    }
    if any(candidate and candidate in serialized for candidate in candidates):
        raise ValueError("absolute dataset root would be persisted in cache metadata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/fbank_cache_v2")
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch size and shard size must be positive")
    if args.device != "cuda:0":
        raise ValueError("approved production extraction device is exactly cuda:0")
    if args.shard_size != 256:
        raise ValueError("approved production shard size is exactly 256")
    if args.batch_size != 64:
        raise ValueError("initial approved production batch size is exactly 64")

    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    cache_dir = args.cache_dir.expanduser().resolve()
    authority, rows = validate_approved_inputs(PROJECT_ROOT)
    before_source_state = source_state(dataset_root, rows)

    frontend = SpeechBrainECAPAFrontend(device=args.device)
    frontend.eval()
    if not frontend.pretrained_parameters_frozen:
        raise ValueError("pretrained SpeechBrain parameters are not frozen")
    measured_frames, waveform_samples, feature_preflight = measure_feature_contract(
        frontend, dataset_root, rows["train"]
    )
    feature_shape = (measured_frames, FEATURE_BINS)
    extraction_batch_size, fallback_used, batch_preflight = run_batch_size_preflight(
        frontend,
        dataset_root,
        rows["train"],
        requested_batch_size=args.batch_size,
        waveform_samples=waveform_samples,
        feature_shape=feature_shape,
    )
    print(
        "TRAIN-ONLY FEATURE PREFLIGHT PASS: "
        + json.dumps(feature_preflight, sort_keys=True),
        flush=True,
    )
    print(
        "BATCH-SIZE PREFLIGHT PASS: "
        + json.dumps(batch_preflight, sort_keys=True),
        flush=True,
    )

    estimated_bytes = (
        sum(len(rows[split]) for split in SPLITS)
        * feature_shape[0]
        * feature_shape[1]
        * 4
    )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_dir.parent).free
    required_bytes = math.ceil(estimated_bytes * 1.10)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient disk: require at least {required_bytes} bytes, "
            f"found {free_bytes}"
        )
    print(
        f"Estimated tensor bytes: {estimated_bytes}; free bytes: {free_bytes}",
        flush=True,
    )

    config = build_config(
        authority,
        feature_shape=feature_shape,
        waveform_samples=waveform_samples,
        requested_batch_size=args.batch_size,
        extraction_batch_size=extraction_batch_size,
        batch_size_fallback_used=fallback_used,
        shard_size=args.shard_size,
        device_used=str(frontend.device),
    )
    config_second = build_config(
        authority,
        feature_shape=feature_shape,
        waveform_samples=waveform_samples,
        requested_batch_size=args.batch_size,
        extraction_batch_size=extraction_batch_size,
        batch_size_fallback_used=fallback_used,
        shard_size=args.shard_size,
        device_used=str(frontend.device),
    )
    config_text = canonical_json_text(config)
    if config_text != canonical_json_text(config_second):
        raise ValueError("config planning is not deterministic")
    assert_no_absolute_dataset_root(config, dataset_root)

    planned_indexes = {
        split: csv_text(
            plan_index_rows(rows[split], split, args.shard_size, feature_shape)
        )
        for split in SPLITS
    }
    second_indexes = {
        split: csv_text(
            plan_index_rows(rows[split], split, args.shard_size, feature_shape)
        )
        for split in SPLITS
    }
    if planned_indexes != second_indexes:
        raise ValueError("index planning is not deterministic")
    dataset_root_texts = {
        str(dataset_root),
        str(dataset_root).replace("\\", "/"),
        str(dataset_root).replace("\\", "\\\\"),
    }
    if any(
        root_text and root_text in index_text
        for root_text in dataset_root_texts
        for index_text in planned_indexes.values()
    ):
        raise ValueError("absolute dataset root would be persisted in an index")

    prepare_cache_metadata(
        cache_dir, config_text, planned_indexes, resume=args.resume
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(frontend.device)
    started = time.perf_counter()
    extraction: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        extraction[split] = extract_split(
            frontend,
            dataset_root,
            cache_dir,
            rows[split],
            split,
            feature_shape=feature_shape,
            waveform_samples=waveform_samples,
            batch_size=extraction_batch_size,
            shard_size=args.shard_size,
            resume=args.resume,
        )
    extraction_seconds = time.perf_counter() - started

    cache_validation = validate_complete_cache(
        cache_dir,
        rows,
        config_text=config_text,
        index_texts=planned_indexes,
        feature_shape=feature_shape,
        shard_size=args.shard_size,
        identity_may_exist=args.resume,
    )
    reproducibility = validate_reextraction(
        frontend,
        dataset_root,
        cache_dir,
        rows,
        waveform_samples=waveform_samples,
        feature_shape=feature_shape,
        extraction_batch_size=extraction_batch_size,
        shard_size=args.shard_size,
    )
    after_source_state = source_state(dataset_root, rows)
    source_preservation_match = before_source_state == after_source_state
    if not source_preservation_match:
        raise ValueError("approved train/validation source WAV state changed")
    final_bindings = validate_approved_input_hashes(PROJECT_ROOT)
    if final_bindings != authority["bindings"]:
        raise ValueError("approved input identities changed during extraction")

    config_planning_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    index_planning_hashes = {
        split: hashlib.sha256(planned_indexes[split].encode("utf-8")).hexdigest()
        for split in SPLITS
    }
    identity = build_identity(
        authority,
        config,
        cache_validation,
        reproducibility,
        cache_dir=cache_dir,
        config_planning_sha256=config_planning_hash,
        index_planning_sha256=index_planning_hashes,
        source_preservation_match=source_preservation_match,
    )
    assert_no_absolute_dataset_root(identity, dataset_root)
    identity_path = cache_dir / IDENTITY_FILENAME
    identity_text = canonical_json_text(identity)
    if identity_path.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite cache identity: {identity_path}")
        if identity_path.read_text(encoding="utf-8") != identity_text:
            raise ValueError("resume identity differs from the finalized cache identity")
    else:
        atomic_text(identity_path, identity_text)
    if read_json(identity_path, "v2 cache identity") != identity:
        raise ValueError("finalized cache identity failed read-back")

    final_validation = validate_complete_cache(
        cache_dir,
        rows,
        config_text=config_text,
        index_texts=planned_indexes,
        feature_shape=feature_shape,
        shard_size=args.shard_size,
        identity_may_exist=True,
    )
    runtime_result = {
        "result": "PASS",
        "feature_preflight": feature_preflight,
        "batch_size_preflight": batch_preflight,
        "estimated_tensor_bytes": estimated_bytes,
        "free_bytes_before_publication": free_bytes,
        "extraction_seconds": extraction_seconds,
        "extraction": extraction,
        "cache_validation": final_validation,
        "reproducibility": reproducibility,
        "source_allowlist_state_unchanged": source_preservation_match,
        "input_identities_unchanged": True,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(frontend.device)
            if frontend.device.type == "cuda"
            else 0
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(frontend.device)
            if frontend.device.type == "cuda"
            else 0
        ),
        "identity_sha256": file_sha256(identity_path),
    }
    assert_no_absolute_dataset_root(runtime_result, dataset_root)
    runtime_path = cache_dir / RUNTIME_RESULT_FILENAME
    runtime_text = canonical_json_text(runtime_result)
    if runtime_path.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite runtime result: {runtime_path}")
        if runtime_path.read_text(encoding="utf-8") != runtime_text:
            raise ValueError("resume runtime result differs from completed result")
    else:
        atomic_text(runtime_path, runtime_text)
    if read_json(runtime_path, "v2 cache runtime result") != runtime_result:
        raise ValueError("runtime result failed finalized read-back")
    print(json.dumps(runtime_result, indent=2, sort_keys=True), flush=True)
    print("VIESPEAKER2.0 TRAIN/VALIDATION FBANK CACHE: PASS", flush=True)


if __name__ == "__main__":
    main()
