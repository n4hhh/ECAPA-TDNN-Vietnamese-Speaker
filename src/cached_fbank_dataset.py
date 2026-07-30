"""Lazy Dataset and collation for versioned SpeechBrain Fbank caches."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

V1_SPLITS = ("train", "validation", "test")
V2_SPLITS = ("train", "validation")
V1_INDEX_FIELDS = {
    "relative_audio_path",
    "feature_shard_path",
    "feature_index",
    "speaker_id",
    "speaker_label",
    "final_split",
    "filename_group",
    "feature_frames",
    "feature_dim",
    "feature_dtype",
}
V2_INDEX_FIELDS = {
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
}
V1_SHARD_KEYS = {
    "features",
    "speaker_labels",
    "speaker_ids",
    "relative_audio_paths",
    "final_split",
}
V2_SHARD_KEYS = V1_SHARD_KEYS | {"schema_version", "manifest_row_indices"}


@dataclass(frozen=True)
class CachedFbankRow:
    relative_audio_path: str
    feature_shard_path: str
    feature_index: int
    speaker_id: str
    speaker_label: int
    final_split: str
    filename_group: str
    manifest_row_index: int = -1
    duplicate_group: str = ""
    manifest_version: str = "v1"


class CachedFbankDataset(Dataset[dict[str, Any]]):
    """Read one split without loading all versioned feature shards into RAM."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        max_cached_shards: int = 2,
        validate_finite: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if split not in V1_SPLITS:
            raise ValueError(f"split must be one of {V1_SPLITS}, got {split!r}")
        if max_cached_shards < 1:
            raise ValueError("max_cached_shards must be at least 1")
        if not isinstance(validate_finite, bool):
            raise TypeError("validate_finite must be a bool")
        self.split = split
        self.max_cached_shards = max_cached_shards
        # False is only appropriate for an already fully validated immutable cache.
        self.validate_finite = validate_finite
        self._shard_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._shard_load_count = 0
        self.config = self._read_config()
        self.feature_shape = tuple(self.config["feature_shape"])
        if self.cache_version == 2:
            if split not in V2_SPLITS:
                raise ValueError(
                    "v2 cache permits only train and validation; "
                    "final-test and excluded cache access is forbidden"
                )
            self.identity: dict[str, Any] | None = self._read_v2_identity()
        else:
            self.identity = None
        self.rows = self._read_index()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _read_config(self) -> dict[str, Any]:
        candidates = (
            (1, self.cache_dir / "fbank_cache_config_v1.json"),
            (2, self.cache_dir / "fbank_cache_config_v2.json"),
        )
        existing = [(version, path) for version, path in candidates if path.is_file()]
        if not existing:
            raise FileNotFoundError(
                f"missing versioned Fbank cache config under {self.cache_dir}"
            )
        if len(existing) != 1:
            raise ValueError("cache directory contains ambiguous v1 and v2 configs")
        self.cache_version, self.config_path = existing[0]
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"malformed Fbank cache config {self.config_path}: {error}"
            ) from error
        if not isinstance(config, dict):
            raise ValueError(f"Fbank cache config must be an object: {self.config_path}")
        if self.cache_version == 1:
            self._validate_v1_config(config)
        else:
            self._validate_v2_config(config)
        return config

    @staticmethod
    def _validate_v1_config(config: dict[str, Any]) -> None:
        required = {
            "version",
            "frontend",
            "feature_stage",
            "feature_shape",
            "feature_dtype",
            "shard_size",
            "expected_rows",
        }
        missing = required - set(config)
        if missing:
            raise ValueError(f"cache config is missing keys: {sorted(missing)}")
        if config["version"] != 1:
            raise ValueError(f"unsupported cache version: {config['version']!r}")
        if config["frontend"] != "speechbrain/spkrec-ecapa-voxceleb":
            raise ValueError(f"unexpected cache frontend: {config['frontend']!r}")
        if config["feature_stage"] != "raw_compute_features_before_mean_var_norm":
            raise ValueError(f"unexpected feature stage: {config['feature_stage']!r}")
        if config["feature_shape"] != [301, 80] or config["feature_dtype"] != "float32":
            raise ValueError("cache config must declare [301, 80] float32 features")
        if not isinstance(config["shard_size"], int) or config["shard_size"] < 1:
            raise ValueError("cache config has invalid shard_size")
        expected = config["expected_rows"]
        if not isinstance(expected, dict) or set(expected) != set(V1_SPLITS):
            raise ValueError(
                "cache config expected_rows must define train, validation, and test"
            )

    @staticmethod
    def _validate_v2_config(config: dict[str, Any]) -> None:
        required = {
            "schema_version",
            "cache_version",
            "model_source",
            "frontend",
            "speechbrain_version",
            "torch_version",
            "torchaudio_version",
            "input_bindings",
            "included_splits",
            "feature_stage",
            "feature_shape",
            "feature_dtype",
            "raw_pre_normalization",
            "transposed",
            "shard_schema_version",
            "shard_size",
            "expected_rows",
            "train_class_count",
            "train_label_range",
            "validation_label",
            "allowed_filename_groups",
            "index_filenames",
            "index_fields",
            "index_order",
            "shard_assignment",
            "path_base_semantics",
        }
        missing = required - set(config)
        if missing:
            raise ValueError(f"v2 cache config is missing keys: {sorted(missing)}")
        if config["schema_version"] != 2 or config["cache_version"] != "v2":
            raise ValueError("unsupported v2 cache schema")
        if (
            config["model_source"] != "speechbrain/spkrec-ecapa-voxceleb"
            or config["frontend"] != "speechbrain/spkrec-ecapa-voxceleb"
        ):
            raise ValueError("unexpected v2 cache model source")
        if config["feature_stage"] != "raw_compute_features_before_mean_var_norm":
            raise ValueError("unexpected v2 feature stage")
        shape = config["feature_shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not isinstance(shape[0], int)
            or shape[0] < 1
            or shape[1] != 80
            or config["feature_dtype"] != "float32"
        ):
            raise ValueError("v2 cache must declare dynamic [T, 80] float32 features")
        if config["raw_pre_normalization"] is not True or config["transposed"] is not False:
            raise ValueError("v2 cache must contain raw, non-transposed features")
        if config["included_splits"] != list(V2_SPLITS):
            raise ValueError("v2 cache included_splits must be exactly train/validation")
        if config["shard_schema_version"] != 2:
            raise ValueError("unsupported v2 shard schema")
        if not isinstance(config["shard_size"], int) or config["shard_size"] < 1:
            raise ValueError("v2 cache has invalid shard_size")
        expected = config["expected_rows"]
        if (
            not isinstance(expected, dict)
            or set(expected) != set(V2_SPLITS)
            or any(
                not isinstance(expected[split], int) or expected[split] < 1
                for split in V2_SPLITS
            )
        ):
            raise ValueError(
                "v2 expected_rows must define positive train/validation counts"
            )
        class_count = config["train_class_count"]
        if (
            not isinstance(class_count, int)
            or class_count < 1
            or config["train_label_range"] != [0, class_count - 1]
            or config["validation_label"] != -1
        ):
            raise ValueError("v2 label configuration is inconsistent")
        groups = config["allowed_filename_groups"]
        if (
            not isinstance(groups, list)
            or not groups
            or any(not isinstance(value, str) or not value for value in groups)
            or len(groups) != len(set(groups))
        ):
            raise ValueError("v2 allowed_filename_groups is invalid")
        indexes = config["index_filenames"]
        if indexes != {
            "train": "train_feature_index_v2.csv",
            "validation": "validation_feature_index_v2.csv",
        }:
            raise ValueError("v2 index filenames are invalid")
        if set(config["index_fields"]) != V2_INDEX_FIELDS:
            raise ValueError("v2 index field schema is invalid")
        if (
            config["index_order"] != "portable_manifest_row_order"
            or config["shard_assignment"] != "manifest_row_index divmod shard_size"
        ):
            raise ValueError("v2 deterministic ordering contract is invalid")

    def _read_v2_identity(self) -> dict[str, Any]:
        path = self.cache_dir / "fbank_cache_identity_v2.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing finalized v2 cache identity: {path}")
        try:
            identity = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"malformed v2 cache identity {path}: {error}") from error
        required = {
            "schema_version",
            "identity_kind",
            "cache_version",
            "model_source",
            "config_sha256",
            "included_splits",
            "feature_shape",
            "feature_dtype",
            "raw_pre_normalization",
            "transposed",
            "shard_size",
            "row_counts",
            "train_class_count",
            "train_label_range",
            "validation_label",
            "index_sha256",
            "shard_counts",
            "total_cached_utterances",
            "final_test_cache_absent",
        }
        if not isinstance(identity, dict) or required - set(identity):
            raise ValueError("v2 cache identity is missing required fields")
        if (
            identity["schema_version"] != 2
            or identity["identity_kind"] != "fbank_cache_v2"
            or identity["cache_version"] != "v2"
            or identity["model_source"] != self.config["model_source"]
            or identity["included_splits"] != list(V2_SPLITS)
            or identity["feature_shape"] != self.config["feature_shape"]
            or identity["feature_dtype"] != "float32"
            or identity["raw_pre_normalization"] is not True
            or identity["transposed"] is not False
            or identity["shard_size"] != self.config["shard_size"]
            or identity["row_counts"] != self.config["expected_rows"]
            or identity["train_class_count"] != self.config["train_class_count"]
            or identity["train_label_range"] != self.config["train_label_range"]
            or identity["validation_label"] != -1
            or identity["final_test_cache_absent"] is not True
        ):
            raise ValueError("v2 cache identity disagrees with its config")
        if identity["config_sha256"] != self._file_sha256(self.config_path):
            raise ValueError("v2 cache config hash disagrees with identity")
        index_hashes = identity["index_sha256"]
        if not isinstance(index_hashes, dict) or set(index_hashes) != set(V2_SPLITS):
            raise ValueError("v2 cache identity has invalid index hashes")
        return identity

    @staticmethod
    def _safe_relative_path(value: str, description: str) -> str:
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

    def _read_index(self) -> tuple[CachedFbankRow, ...]:
        if self.cache_version == 1:
            path = self.cache_dir / f"{self.split}_feature_index_v1.csv"
            required_fields = V1_INDEX_FIELDS
        else:
            path = self.cache_dir / self.config["index_filenames"][self.split]
            required_fields = V2_INDEX_FIELDS
        if not path.is_file():
            raise FileNotFoundError(f"missing cache index: {path}")
        if (
            self.cache_version == 2
            and self.identity is not None
            and self.identity["index_sha256"][self.split] != self._file_sha256(path)
        ):
            raise ValueError(f"{self.split} index hash disagrees with v2 identity")

        rows: list[CachedFbankRow] = []
        seen_audio_paths: set[str] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required_fields - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for line, raw in enumerate(reader, start=2):
                try:
                    if self.cache_version == 1:
                        relative = self._safe_relative_path(
                            raw["relative_audio_path"].strip(), "audio path"
                        )
                        shard_path = self._safe_relative_path(
                            raw["feature_shard_path"].strip(), "shard path"
                        )
                        position = int(raw["feature_index"])
                        label = int(raw["speaker_label"])
                        dimension = int(raw["feature_dim"])
                        manifest_row_index = len(rows)
                        duplicate_group = ""
                        manifest_version = "v1"
                    else:
                        relative = self._safe_relative_path(
                            raw["audio_path"].strip(), "audio path"
                        )
                        shard_path = self._safe_relative_path(
                            raw["shard_path"].strip(), "shard path"
                        )
                        position = int(raw["within_shard_index"])
                        label = int(raw["label"])
                        dimension = int(raw["feature_bins"])
                        manifest_row_index = int(raw["manifest_row_index"])
                        duplicate_group = raw["duplicate_group"].strip()
                        manifest_version = raw["manifest_version"].strip()
                    speaker_id = raw["speaker_id"].strip()
                    final_split = raw["final_split"].strip()
                    group = raw["filename_group"].strip()
                    frames = int(raw["feature_frames"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{path}:{line}: malformed index metadata: {error}"
                    ) from error

                allowed_groups = (
                    {"train", "train_small"}
                    if self.cache_version == 1
                    else set(self.config["allowed_filename_groups"])
                )
                if not speaker_id or group not in allowed_groups:
                    raise ValueError(f"{path}:{line}: invalid speaker or filename group")
                if final_split != self.split:
                    raise ValueError(
                        f"{path}:{line}: expected split {self.split}, got {final_split!r}"
                    )
                if position < 0 or position >= int(self.config["shard_size"]):
                    raise ValueError(f"{path}:{line}: invalid feature index {position}")
                if (
                    (frames, dimension) != self.feature_shape
                    or raw["feature_dtype"].strip() != "float32"
                ):
                    raise ValueError(
                        f"{path}:{line}: feature metadata disagrees with config"
                    )
                label_range = (
                    [0, 487]
                    if self.cache_version == 1
                    else self.config["train_label_range"]
                )
                if self.split == "train" and not label_range[0] <= label <= label_range[1]:
                    raise ValueError(
                        f"{path}:{line}: train label must be in "
                        f"{label_range[0]}..{label_range[1]}"
                    )
                if self.split != "train" and label != -1:
                    raise ValueError(f"{path}:{line}: evaluation label must be -1")

                if self.cache_version == 2:
                    dataset_index = len(rows)
                    expected_shard = (
                        f"{self.split}/shard_"
                        f"{dataset_index // int(self.config['shard_size']):05d}.pt"
                    )
                    if (
                        manifest_row_index != dataset_index
                        or position != dataset_index % int(self.config["shard_size"])
                        or shard_path != expected_shard
                        or manifest_version != "v2"
                        or PurePosixPath(relative).parent.name != speaker_id
                    ):
                        raise ValueError(
                            f"{path}:{line}: v2 index row/shard/speaker alignment failed"
                        )
                    if relative in seen_audio_paths:
                        raise ValueError(f"{path}:{line}: duplicate audio path")
                    seen_audio_paths.add(relative)

                rows.append(
                    CachedFbankRow(
                        relative_audio_path=relative,
                        feature_shard_path=shard_path,
                        feature_index=position,
                        speaker_id=speaker_id,
                        speaker_label=label,
                        final_split=final_split,
                        filename_group=group,
                        manifest_row_index=manifest_row_index,
                        duplicate_group=duplicate_group,
                        manifest_version=manifest_version,
                    )
                )
        expected = self.config["expected_rows"].get(self.split)
        if not isinstance(expected, int) or len(rows) != expected:
            raise ValueError(
                f"{self.split} index has {len(rows)} rows, expected {expected}"
            )
        if self.cache_version == 2 and self.split == "train":
            expected_labels = set(range(self.config["train_class_count"]))
            if {row.speaker_label for row in rows} != expected_labels:
                raise ValueError("v2 train index does not cover the complete label range")
        return tuple(rows)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def cached_shard_count(self) -> int:
        return len(self._shard_cache)

    @property
    def shard_load_count(self) -> int:
        return self._shard_load_count

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_shard_cache"] = OrderedDict()
        state["_shard_load_count"] = 0
        return state

    def _load_shard(self, relative_path: str) -> dict[str, Any]:
        if relative_path in self._shard_cache:
            shard = self._shard_cache.pop(relative_path)
            self._shard_cache[relative_path] = shard
            return shard
        if self.cache_version == 2:
            pure = PurePosixPath(relative_path)
            if len(pure.parts) != 2 or pure.parts[0] != self.split:
                raise ValueError(
                    f"v2 shard path escapes expected split: {relative_path!r}"
                )
        path = self.cache_dir / Path(*PurePosixPath(relative_path).parts)
        if not path.is_file():
            raise FileNotFoundError(f"missing feature shard: {path}")
        try:
            shard = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RuntimeError(f"failed to load feature shard {path}: {error}") from error
        self._validate_shard(shard, path)
        self._shard_cache[relative_path] = shard
        self._shard_load_count += 1
        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return shard

    def _validate_shard(self, shard: Any, path: Path) -> None:
        shard_keys = V1_SHARD_KEYS if self.cache_version == 1 else V2_SHARD_KEYS
        if not isinstance(shard, dict) or set(shard) != shard_keys:
            raise ValueError(
                f"malformed shard {path}: expected keys {sorted(shard_keys)}"
            )
        if self.cache_version == 2 and shard["schema_version"] != 2:
            raise ValueError(f"malformed shard {path}: unsupported schema version")
        features = shard["features"]
        labels = shard["speaker_labels"]
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError(
                f"malformed shard {path}: features must be a rank-3 tensor"
            )
        count = features.shape[0]
        if count < 1 or count > int(self.config["shard_size"]):
            raise ValueError(f"malformed shard {path}: invalid row count {count}")
        if tuple(features.shape[1:]) != self.feature_shape:
            raise ValueError(
                f"wrong feature shape in {path}: {tuple(features.shape[1:])}"
            )
        if features.dtype != torch.float32:
            raise TypeError(f"wrong feature dtype in {path}: {features.dtype}")
        if features.device.type != "cpu":
            raise ValueError(f"features in {path} are not CPU tensors")
        if (
            not isinstance(labels, torch.Tensor)
            or tuple(labels.shape) != (count,)
            or labels.dtype != torch.long
        ):
            raise ValueError(f"malformed speaker labels in {path}")
        if labels.device.type != "cpu":
            raise ValueError(f"speaker labels in {path} are not CPU tensors")
        for key in ("speaker_ids", "relative_audio_paths"):
            if not isinstance(shard[key], list) or len(shard[key]) != count:
                raise ValueError(f"malformed {key} in {path}")
        if self.cache_version == 2 and (
            not isinstance(shard["manifest_row_indices"], list)
            or len(shard["manifest_row_indices"]) != count
            or any(not isinstance(value, int) for value in shard["manifest_row_indices"])
        ):
            raise ValueError(f"malformed manifest_row_indices in {path}")
        if shard["final_split"] != self.split:
            raise ValueError(
                f"shard split mismatch in {path}: {shard['final_split']!r}"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        shard = self._load_shard(row.feature_shard_path)
        count = shard["features"].shape[0]
        if row.feature_index >= count:
            raise IndexError(
                f"invalid feature position {row.feature_index} for "
                f"{row.feature_shard_path} with {count} rows"
            )
        position = row.feature_index
        actual_label = int(shard["speaker_labels"][position].item())
        actual_id = shard["speaker_ids"][position]
        actual_path = shard["relative_audio_paths"][position]
        if (actual_label, actual_id, actual_path) != (
            row.speaker_label,
            row.speaker_id,
            row.relative_audio_path,
        ):
            raise ValueError(
                f"index metadata disagrees with shard at "
                f"{row.feature_shard_path}[{position}]"
            )
        if (
            self.cache_version == 2
            and shard["manifest_row_indices"][position] != row.manifest_row_index
        ):
            raise ValueError(
                f"manifest row identity disagrees with shard at "
                f"{row.feature_shard_path}[{position}]"
            )
        feature = shard["features"][position]
        if (
            tuple(feature.shape) != self.feature_shape
            or feature.dtype != torch.float32
            or feature.device.type != "cpu"
        ):
            raise ValueError(
                f"invalid feature at {row.feature_shard_path}[{position}]"
            )
        if self.validate_finite and not bool(torch.isfinite(feature).all().item()):
            raise ValueError(
                f"non-finite feature at {row.feature_shard_path}[{position}]"
            )
        sample = {
            "fbank": feature,
            "dataset_index": index,
            "speaker_label": row.speaker_label,
            "speaker_id": row.speaker_id,
            "relative_audio_path": row.relative_audio_path,
            "final_split": row.final_split,
            "filename_group": row.filename_group,
        }
        if self.cache_version == 2:
            sample.update(
                {
                    "manifest_row_index": row.manifest_row_index,
                    "duplicate_group": row.duplicate_group,
                    "manifest_version": row.manifest_version,
                }
            )
        return sample


def collate_cached_fbank(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch = {
        "fbank": torch.stack([sample["fbank"] for sample in samples]),
        "dataset_index": torch.tensor(
            [sample["dataset_index"] for sample in samples], dtype=torch.long
        ),
        "speaker_label": torch.tensor(
            [sample["speaker_label"] for sample in samples], dtype=torch.long
        ),
        "speaker_id": [sample["speaker_id"] for sample in samples],
        "relative_audio_path": [
            sample["relative_audio_path"] for sample in samples
        ],
        "final_split": [sample["final_split"] for sample in samples],
        "filename_group": [sample["filename_group"] for sample in samples],
    }
    if "manifest_row_index" in samples[0]:
        if not all("manifest_row_index" in sample for sample in samples):
            raise ValueError("cannot collate mixed v1/v2 cache samples")
        batch.update(
            {
                "manifest_row_index": torch.tensor(
                    [sample["manifest_row_index"] for sample in samples],
                    dtype=torch.long,
                ),
                "duplicate_group": [
                    sample["duplicate_group"] for sample in samples
                ],
                "manifest_version": [
                    sample["manifest_version"] for sample in samples
                ],
            }
        )
    return batch


def create_cached_fbank_dataloader(
    dataset: CachedFbankDataset,
    batch_size: int,
    *,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_cached_fbank,
        drop_last=False,
    )


def create_cached_fbank_training_dataloader(
    dataset: CachedFbankDataset,
    batch_sampler: Sampler[list[int]],
    *,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create a train loader whose batch structure is supplied by a batch sampler."""
    if dataset.split != "train":
        raise ValueError("training DataLoader requires the train split")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_cached_fbank,
        generator=generator,
    )
