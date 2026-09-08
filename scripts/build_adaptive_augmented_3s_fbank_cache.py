"""Build or verify the adaptive_augmented_3s_v1 train/validation Fbank cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import speechbrain
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cached_fbank_dataset import CachedFbankDataset, create_cached_fbank_dataloader
from src.speechbrain_frontend import SpeechBrainECAPAFrontend


SPLITS = ("train", "validation")
PORTABLE_FIELDS = ("relative_audio_path", "speaker_id", "speaker_label", "final_split")
INDEX_FIELDS = (
    "relative_audio_path", "speaker_id", "speaker_label", "final_split",
    "shard_path", "within_shard_index",
)
CACHE_VERSION = "adaptive_augmented_3s_v1"
CONFIG_FILENAME = "fbank_cache_config_adaptive_augmented_3s_v1.json"
IDENTITY_FILENAME = "fbank_cache_identity_adaptive_augmented_3s_v1.json"
RUNTIME_FILENAME = "fbank_cache_runtime_adaptive_augmented_3s_v1.json"
INDEX_FILENAMES = {
    "train": "train_feature_index_adaptive_augmented_3s_v1.csv",
    "validation": "validation_feature_index_adaptive_augmented_3s_v1.csv",
}
EXPECTED_ROWS = {"train": 48_640, "validation": 6_076}
EXPECTED_SPEAKERS = {"train": 488, "validation": 61}
INPUTS = {
    "dataset_identity": (
        "manifests/adaptive_augmented_3s_v1_dataset_identity.json",
        "8fd9fcc0b802d56d4a96080e53180e73e317e57165c25f8d81b77ce4f18d421d",
    ),
    "split_identity": (
        "splits/adaptive_augmented_3s_v1_split_identity.json",
        "6aa9f029cec5cfd76e00bcf4eebad92d4f61aeb39467055009ce0e06ab22885c",
    ),
    "train_manifest": (
        "manifests/adaptive_augmented_3s_v1_train_manifest.csv",
        "16be3c155430c018976d240853a34e28e272668db7c5a3190fd6e674a39f8d3f",
    ),
    "validation_manifest": (
        "manifests/adaptive_augmented_3s_v1_validation_manifest.csv",
        "3ed60cf7d2041d564476de52808d375c3bb9cb84e5f01689bdf5a18ced98ca69",
    ),
    "speaker_to_label": (
        "manifests/adaptive_augmented_3s_v1_speaker_to_label.json",
        "d99df32a65bc95d7b2bc6f73155a335ef736167b925a2e5d630d4dccf4ff159d",
    ),
}


@dataclass(frozen=True)
class SourceRow:
    relative_audio_path: str
    speaker_id: str
    speaker_label: int
    final_split: str
    manifest_row_index: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"remove incomplete temporary file before resuming: {temporary}")
    temporary.write_text(content, encoding="utf-8", newline="")
    _atomic_replace(temporary, path)


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        # A prior interrupted publication never names the completed shard and is
        # therefore safe to discard before regenerating that deterministic shard.
        temporary.unlink()
    torch.save(value, temporary)
    _atomic_replace(temporary, path)


def _atomic_replace(temporary: Path, target: Path) -> None:
    """Retry the Windows rename when a transient scanner holds the new file."""
    for attempt in range(30):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.2)


def safe_relative_path(value: str, description: str) -> str:
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts or "\\" in value or Path(value).is_absolute():
        raise ValueError(f"unsafe {description}: {value!r}")
    return value


def read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def validate_inputs() -> tuple[dict[str, dict[str, str]], dict[str, list[SourceRow]], dict[str, int]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, (relative, expected_hash) in INPUTS.items():
        path = PROJECT_ROOT / Path(*PurePosixPath(relative).parts)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"approved {name} hash mismatch: {actual_hash}")
        bindings[name] = {"path": relative, "sha256": actual_hash}

    dataset_identity = read_json(PROJECT_ROOT / INPUTS["dataset_identity"][0], "dataset identity")
    split_identity = read_json(PROJECT_ROOT / INPUTS["split_identity"][0], "split identity")
    if (
        dataset_identity.get("package_version") != CACHE_VERSION
        or dataset_identity.get("wav_count") != 60_803
        or dataset_identity.get("speaker_count") != 610
        or split_identity.get("dataset_identity_sha256") != INPUTS["dataset_identity"][1]
        or split_identity.get("speaker_split_sha256") != "eaa3a695079bfb00d077415df440e4a744a24cd93e5eb7d7ec98c84f67d0eb61"
        or split_identity.get("row_counts") != {**EXPECTED_ROWS, "final_test": 6087}
        or split_identity.get("speaker_counts") != {**EXPECTED_SPEAKERS, "final_test": 61}
        or split_identity.get("final_test_locked_after_package_acceptance") is not True
    ):
        raise ValueError("Task 1 identity metadata does not match the approved package")

    mapping = read_json(PROJECT_ROOT / INPUTS["speaker_to_label"][0], "speaker label mapping")
    try:
        label_mapping = {str(key): int(value) for key, value in mapping.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("malformed speaker label mapping") from error
    if len(label_mapping) != 488 or set(label_mapping.values()) != set(range(488)):
        raise ValueError("train label mapping is not the approved contiguous range")

    all_rows: dict[str, list[SourceRow]] = {}
    for split in SPLITS:
        manifest_name = f"{split}_manifest"
        path = PROJECT_ROOT / INPUTS[manifest_name][0]
        rows: list[SourceRow] = []
        seen: set[str] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != PORTABLE_FIELDS:
                raise ValueError(f"{path} does not have the approved portable schema")
            for index, raw in enumerate(reader):
                line = index + 2
                try:
                    relative = safe_relative_path(raw["relative_audio_path"].strip(), "audio path")
                    label = int(raw["speaker_label"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line}: invalid manifest row") from error
                speaker_id = raw["speaker_id"].strip()
                if (
                    not speaker_id or raw["final_split"].strip() != split
                    or PurePosixPath(relative).parent.name != speaker_id
                    or relative in seen
                ):
                    raise ValueError(f"{path}:{line}: invalid manifest row identity")
                if split == "train":
                    if label_mapping.get(speaker_id) != label:
                        raise ValueError(f"{path}:{line}: label does not match speaker mapping")
                elif label != -1:
                    raise ValueError(f"{path}:{line}: validation label must be -1")
                seen.add(relative)
                rows.append(SourceRow(relative, speaker_id, label, split, index))
        if len(rows) != EXPECTED_ROWS[split] or len({row.speaker_id for row in rows}) != EXPECTED_SPEAKERS[split]:
            raise ValueError(f"{split} manifest count or speaker count is not approved")
        all_rows[split] = rows
    if {row.relative_audio_path for row in all_rows["train"]} & {row.relative_audio_path for row in all_rows["validation"]}:
        raise ValueError("train and validation manifests overlap")
    return bindings, all_rows, label_mapping


def source_state(dataset_root: Path, rows: dict[str, list[SourceRow]]) -> dict[str, tuple[int, int]]:
    state: dict[str, tuple[int, int]] = {}
    for split in SPLITS:
        for row in rows[split]:
            path = dataset_root / Path(*PurePosixPath(row.relative_audio_path).parts)
            metadata = path.stat()
            if not path.is_file():
                raise FileNotFoundError(f"missing approved source WAV: {path}")
            state[row.relative_audio_path] = (metadata.st_size, metadata.st_mtime_ns)
    return state


def index_text(rows: list[SourceRow], split: str, shard_size: int) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        shard_id, offset = divmod(row.manifest_row_index, shard_size)
        writer.writerow({
            "relative_audio_path": row.relative_audio_path,
            "speaker_id": row.speaker_id,
            "speaker_label": row.speaker_label,
            "final_split": split,
            "shard_path": f"{split}/shard_{shard_id:05d}.pt",
            "within_shard_index": offset,
        })
    return stream.getvalue()


def cache_config(bindings: dict[str, dict[str, str]], shard_size: int, batch_size: int) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "cache_version": CACHE_VERSION,
        "identity_kind": "adaptive_augmented_3s_fbank_cache_config",
        "model_source": SpeechBrainECAPAFrontend.SOURCE,
        "speechbrain_version": speechbrain.__version__,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "input_bindings": bindings,
        "task1_identity_sha256": {
            **{name: value["sha256"] for name, value in bindings.items()},
            "speaker_split": "eaa3a695079bfb00d077415df440e4a744a24cd93e5eb7d7ec98c84f67d0eb61",
        },
        "included_splits": list(SPLITS),
        "feature_stage": "raw_compute_features_before_mean_var_norm",
        "raw_pre_normalization": True,
        "transposed": False,
        "feature_shape": [301, 80],
        "feature_dtype": "float32",
        "fbank": {
            "sample_rate": 16000, "n_fft": 400, "win_length": 400,
            "hop_length": 160, "center": True, "n_mels": 80,
            "f_min": 0, "f_max": 8000,
        },
        "shard_schema_version": 3,
        "shard_size": shard_size,
        "extraction_batch_size": batch_size,
        "expected_rows": EXPECTED_ROWS,
        "train_class_count": 488,
        "train_label_range": [0, 487],
        "validation_label": -1,
        "index_filenames": INDEX_FILENAMES,
        "index_fields": list(INDEX_FIELDS),
        "index_order": "portable_manifest_row_order",
        "shard_assignment": "manifest_row_index divmod shard_size",
        "path_base_semantics": {
            "audio_path": "relative_to_runtime_supplied_dataset_root",
            "shard_path": "relative_to_cache_root",
            "separator": "/",
        },
        "timestamps_in_identity_relevant_content": False,
    }


def check_no_absolute_dataset_root(value: Any, dataset_root: Path) -> None:
    serialized = canonical_json(value)
    roots = {str(dataset_root), str(dataset_root).replace("\\", "/"), str(dataset_root).replace("\\", "\\\\")}
    if any(root and root in serialized for root in roots):
        raise ValueError("absolute dataset root would be stored in portable cache metadata")


def validate_feature_batch(features: torch.Tensor, expected_count: int) -> None:
    if (
        tuple(features.shape) != (expected_count, 301, 80)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not features.is_contiguous()
        or not bool(torch.isfinite(features).all().item())
    ):
        raise ValueError("raw feature batch violates the [B, 301, 80] float32 finite contract")


def load_waveform(dataset_root: Path, row: SourceRow) -> torch.Tensor:
    path = dataset_root / Path(*PurePosixPath(row.relative_audio_path).parts)
    waveform, sample_rate = torchaudio.load(path, normalize=True)
    if sample_rate != 16000 or tuple(waveform.shape) != (1, 48000) or not bool(torch.isfinite(waveform).all().item()):
        raise ValueError(f"invalid approved source waveform: {path}")
    return waveform.squeeze(0)


def expected_payload(selected: list[SourceRow], split: str, features: torch.Tensor) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "features": features,
        "speaker_labels": torch.tensor([row.speaker_label for row in selected], dtype=torch.long),
        "speaker_ids": [row.speaker_id for row in selected],
        "relative_audio_paths": [row.relative_audio_path for row in selected],
        "final_split": split,
    }


def validate_shard(payload: Any, selected: list[SourceRow], split: str, shard_size: int) -> None:
    expected_keys = {"schema_version", "features", "speaker_labels", "speaker_ids", "relative_audio_paths", "final_split"}
    if not isinstance(payload, dict) or set(payload) != expected_keys or payload["schema_version"] != 3:
        raise ValueError("invalid cache shard schema")
    count = len(selected)
    features = payload["features"]
    labels = payload["speaker_labels"]
    if count < 1 or count > shard_size:
        raise ValueError("invalid shard row count")
    validate_feature_batch(features, count)
    if (
        not isinstance(labels, torch.Tensor) or labels.dtype != torch.long
        or labels.device.type != "cpu" or tuple(labels.shape) != (count,)
        or labels.tolist() != [row.speaker_label for row in selected]
        or payload["speaker_ids"] != [row.speaker_id for row in selected]
        or payload["relative_audio_paths"] != [row.relative_audio_path for row in selected]
        or payload["final_split"] != split
    ):
        raise ValueError("cache shard metadata disagrees with its manifest rows")


def prepare_plan(cache_root: Path, config_text: str, indexes: dict[str, str]) -> bool:
    config_path = cache_root / CONFIG_FILENAME
    identity_path = cache_root / IDENTITY_FILENAME
    index_paths = {split: cache_root / INDEX_FILENAMES[split] for split in SPLITS}
    if identity_path.exists():
        if not config_path.is_file() or any(not path.is_file() for path in index_paths.values()):
            raise ValueError("complete cache identity exists without its plan files")
        if config_path.read_text(encoding="utf-8") != config_text or any(index_paths[split].read_text(encoding="utf-8") != indexes[split] for split in SPLITS):
            raise ValueError("existing cache conflicts with requested approved identity")
        return True
    cache_root.mkdir(parents=True, exist_ok=True)
    existing_files = {path.name for path in cache_root.iterdir() if path.is_file()}
    permitted = {CONFIG_FILENAME, *INDEX_FILENAMES.values(), RUNTIME_FILENAME}
    if existing_files - permitted:
        raise FileExistsError(f"cache root has unexpected files: {sorted(existing_files - permitted)}")
    if config_path.exists():
        if config_path.read_text(encoding="utf-8") != config_text:
            raise ValueError("resume rejected incompatible cache configuration")
        for split, path in index_paths.items():
            if not path.is_file() or path.read_text(encoding="utf-8") != indexes[split]:
                raise ValueError(f"resume rejected incompatible {split} index")
    else:
        if any(path.name not in {RUNTIME_FILENAME} for path in cache_root.iterdir()):
            raise FileExistsError("new cache root is not empty")
        atomic_text(config_path, config_text)
        for split, path in index_paths.items():
            atomic_text(path, indexes[split])
    return False


def extract_cache(
    frontend: SpeechBrainECAPAFrontend, dataset_root: Path, cache_root: Path,
    rows: dict[str, list[SourceRow]], shard_size: int, batch_size: int,
) -> dict[str, dict[str, int]]:
    outcome: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        written = skipped = 0
        shard_count = math.ceil(len(rows[split]) / shard_size)
        for shard_id, start in enumerate(range(0, len(rows[split]), shard_size)):
            selected = rows[split][start:start + shard_size]
            target = cache_root / split / f"shard_{shard_id:05d}.pt"
            if target.exists():
                payload = torch.load(target, map_location="cpu", weights_only=False)
                validate_shard(payload, selected, split, shard_size)
                skipped += 1
                continue
            batches: list[torch.Tensor] = []
            for batch_start in range(0, len(selected), batch_size):
                batch_rows = selected[batch_start:batch_start + batch_size]
                waveforms = torch.stack([load_waveform(dataset_root, row) for row in batch_rows])
                with torch.inference_mode():
                    extracted = frontend.compute_features(waveforms)
                extracted = extracted.detach().to(device="cpu", dtype=torch.float32).contiguous()
                validate_feature_batch(extracted, len(batch_rows))
                batches.append(extracted)
            payload = expected_payload(selected, split, torch.cat(batches).contiguous())
            validate_shard(payload, selected, split, shard_size)
            atomic_torch_save(payload, target)
            validate_shard(torch.load(target, map_location="cpu", weights_only=False), selected, split, shard_size)
            written += 1
            print(f"{split} shard {shard_id + 1}/{shard_count} complete", flush=True)
        outcome[split] = {"written": written, "reused": skipped, "total": shard_count}
    return outcome


def validate_complete_cache(cache_root: Path, rows: dict[str, list[SourceRow]], indexes: dict[str, str], shard_size: int) -> dict[str, Any]:
    root_dirs = {path.name for path in cache_root.iterdir() if path.is_dir()}
    if root_dirs != set(SPLITS):
        raise ValueError(f"cache directories must be exactly {list(SPLITS)}")
    seen: set[str] = set()
    shard_counts: dict[str, int] = {}
    shard_bytes: dict[str, int] = {}
    for split in SPLITS:
        if (cache_root / INDEX_FILENAMES[split]).read_text(encoding="utf-8") != indexes[split]:
            raise ValueError(f"{split} finalized index differs from deterministic plan")
        count = math.ceil(len(rows[split]) / shard_size)
        expected_names = {f"shard_{number:05d}.pt" for number in range(count)}
        split_dir = cache_root / split
        actual_names = {path.name for path in split_dir.iterdir() if path.is_file()}
        if actual_names != expected_names or any(path.is_dir() for path in split_dir.iterdir()):
            raise ValueError(f"{split} shard layout differs from deterministic plan")
        shard_counts[split] = count
        shard_bytes[split] = 0
        for shard_id in range(count):
            selected = rows[split][shard_id * shard_size:(shard_id + 1) * shard_size]
            path = split_dir / f"shard_{shard_id:05d}.pt"
            validate_shard(torch.load(path, map_location="cpu", weights_only=False), selected, split, shard_size)
            shard_bytes[split] += path.stat().st_size
            for row in selected:
                if row.relative_audio_path in seen:
                    raise ValueError("duplicate path in cache")
                seen.add(row.relative_audio_path)
    if len(seen) != sum(EXPECTED_ROWS.values()):
        raise ValueError("cache rows do not reconcile with approved manifests")
    return {
        "row_counts": EXPECTED_ROWS,
        "shard_counts": shard_counts,
        "shard_bytes": shard_bytes,
        "total_cached_utterances": len(seen),
        "total_shard_bytes": sum(shard_bytes.values()),
        "features_validated": "all finite float32 [301, 80]",
        "final_test_cached_entries": 0,
    }


def build_identity(bindings: dict[str, dict[str, str]], config_path: Path, cache_root: Path, validation: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "schema_version": 3,
        "identity_kind": "adaptive_augmented_3s_fbank_cache",
        "cache_version": CACHE_VERSION,
        "config_path": CONFIG_FILENAME,
        "config_sha256": sha256_file(config_path),
        "input_bindings": bindings,
        "task1_identity_sha256": {
            **{name: value["sha256"] for name, value in bindings.items()},
            "speaker_split": "eaa3a695079bfb00d077415df440e4a744a24cd93e5eb7d7ec98c84f67d0eb61",
        },
        "model_source": SpeechBrainECAPAFrontend.SOURCE,
        "included_splits": list(SPLITS),
        "feature_shape": [301, 80], "feature_dtype": "float32",
        "raw_pre_normalization": True, "transposed": False,
        "shard_size": json.loads(config_path.read_text(encoding="utf-8"))["shard_size"],
        "row_counts": validation["row_counts"], "shard_counts": validation["shard_counts"],
        "total_cached_utterances": validation["total_cached_utterances"],
        "index_sha256": {split: sha256_file(cache_root / INDEX_FILENAMES[split]) for split in SPLITS},
        "deterministic_row_order": "portable_manifest_row_order",
        "deterministic_shard_order": "manifest_row_index divmod shard_size",
        "final_test_cache_absent": True,
        "timestamps_in_identity": False,
    }
    identity["identity_sha256"] = canonical_digest(identity)
    return identity


def fresh_vs_cached(frontend: SpeechBrainECAPAFrontend, dataset_root: Path, cache_root: Path, rows: dict[str, list[SourceRow]], shard_size: int) -> dict[str, Any]:
    maximum = 0.0
    samples = 0
    for split in SPLITS:
        for index in (0, len(rows[split]) // 2):
            row = rows[split][index]
            with torch.inference_mode():
                fresh = frontend.compute_features(load_waveform(dataset_root, row).unsqueeze(0))
            fresh = fresh.squeeze(0).to(device="cpu", dtype=torch.float32)
            shard_id, offset = divmod(index, shard_size)
            cached = torch.load(cache_root / split / f"shard_{shard_id:05d}.pt", map_location="cpu", weights_only=False)["features"][offset]
            difference = float((fresh - cached).abs().max().item())
            maximum = max(maximum, difference)
            if not torch.allclose(fresh, cached, rtol=1e-5, atol=1e-6):
                raise ValueError(f"fresh feature differs from cached feature for {split} row {index}")
            samples += 1
    return {
        "sample_count": samples,
        "maximum_absolute_difference": maximum,
        "allclose": True,
        "rtol": 1e-5,
        "atol": 1e-6,
    }


def cached_dataset_check(cache_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        dataset = CachedFbankDataset(cache_root, split, max_cached_shards=2, validate_finite=True)
        batch = next(iter(create_cached_fbank_dataloader(dataset, batch_size=4, num_workers=0)))
        if (
            tuple(batch["fbank"].shape) != (4, 301, 80)
            or batch["fbank"].dtype != torch.float32
            or not bool(torch.isfinite(batch["fbank"]).all().item())
            or any(value != split for value in batch["final_split"])
        ):
            raise ValueError(f"CachedFbankDataset {split} mini-batch failed")
        result[split] = {"batch_shape": list(batch["fbank"].shape), "labels": batch["speaker_label"].tolist(), "speaker_ids": batch["speaker_id"]}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--shard-size", default=512, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch size and shard size must be positive")
    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    cache_root = args.cache_root.expanduser().resolve()
    if PROJECT_ROOT in cache_root.parents or cache_root == PROJECT_ROOT:
        if not str(cache_root).startswith(str(PROJECT_ROOT / "outputs")):
            raise ValueError("cache root under the repository must be within ignored outputs/")
    bindings, rows, _ = validate_inputs()
    before_source_state = source_state(dataset_root, rows)
    config = cache_config(bindings, args.shard_size, args.batch_size)
    indexes = {split: index_text(rows[split], split, args.shard_size) for split in SPLITS}
    check_no_absolute_dataset_root({"config": config, "indexes": indexes}, dataset_root)
    complete_before_run = prepare_plan(cache_root, canonical_json(config), indexes)

    frontend = SpeechBrainECAPAFrontend(device=args.device)
    frontend.eval()
    if not frontend.pretrained_parameters_frozen:
        raise RuntimeError("pretrained SpeechBrain parameters are not frozen")
    started = time.perf_counter()
    extraction = extract_cache(frontend, dataset_root, cache_root, rows, args.shard_size, args.batch_size)
    validation = validate_complete_cache(cache_root, rows, indexes, args.shard_size)
    after_source_state = source_state(dataset_root, rows)
    if after_source_state != before_source_state:
        raise RuntimeError("approved train/validation source audio changed during cache build")
    identity = build_identity(bindings, cache_root / CONFIG_FILENAME, cache_root, validation)
    check_no_absolute_dataset_root(identity, dataset_root)
    identity_path = cache_root / IDENTITY_FILENAME
    if identity_path.exists():
        if identity_path.read_text(encoding="utf-8") != canonical_json(identity):
            raise ValueError("existing cache identity conflicts with validated cache")
    else:
        atomic_text(identity_path, canonical_json(identity))
    spot_check = fresh_vs_cached(frontend, dataset_root, cache_root, rows, args.shard_size)
    dataset_check = cached_dataset_check(cache_root)
    runtime = {
        "cache_version": CACHE_VERSION,
        "status": "reused_complete_cache" if complete_before_run else "built_or_resumed_cache",
        "device": str(frontend.device), "batch_size": args.batch_size, "shard_size": args.shard_size,
        "duration_seconds": time.perf_counter() - started,
        "extraction": extraction, "validation": validation,
        "fresh_vs_cached": spot_check, "cached_dataset_check": dataset_check,
        "source_state_unchanged": True, "final_test_cached_entries": 0,
    }
    atomic_text(cache_root / RUNTIME_FILENAME, canonical_json(runtime))
    print(canonical_json(runtime), end="")


if __name__ == "__main__":
    main()
