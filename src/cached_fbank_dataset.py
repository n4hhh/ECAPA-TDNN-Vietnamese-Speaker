"""Lazy Dataset and collation for the SpeechBrain Fbank cache v1."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "validation", "test")
INDEX_FIELDS = {
    "relative_audio_path", "feature_shard_path", "feature_index", "speaker_id",
    "speaker_label", "final_split", "filename_group", "feature_frames",
    "feature_dim", "feature_dtype",
}
SHARD_KEYS = {
    "features", "speaker_labels", "speaker_ids", "relative_audio_paths", "final_split"
}


@dataclass(frozen=True)
class CachedFbankRow:
    relative_audio_path: str
    feature_shard_path: str
    feature_index: int
    speaker_id: str
    speaker_label: int
    final_split: str
    filename_group: str


class CachedFbankDataset(Dataset[dict[str, Any]]):
    """Read one split from cache v1 without loading all feature shards into RAM."""

    def __init__(
        self, cache_dir: str | Path, split: str, max_cached_shards: int = 2,
        validate_finite: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if max_cached_shards < 1:
            raise ValueError("max_cached_shards must be at least 1")
        self.split = split
        self.max_cached_shards = max_cached_shards
        if not isinstance(validate_finite, bool):
            raise TypeError("validate_finite must be a bool")
        # False is only appropriate for an already validated, immutable cache.
        self.validate_finite = validate_finite
        self._shard_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._shard_load_count = 0
        self.config = self._read_config()
        self.feature_shape = tuple(self.config["feature_shape"])
        self.rows = self._read_index()

    def _read_config(self) -> dict[str, Any]:
        path = self.cache_dir / "fbank_cache_config_v1.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing Fbank cache config: {path}")
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"malformed Fbank cache config {path}: {error}") from error
        required = {
            "version", "frontend", "feature_stage", "feature_shape", "feature_dtype",
            "shard_size", "expected_rows",
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
        if not isinstance(expected, dict) or set(expected) != set(SPLITS):
            raise ValueError("cache config expected_rows must define train, validation, and test")
        return config

    @staticmethod
    def _safe_relative_path(value: str, description: str) -> str:
        pure = PurePosixPath(value)
        if not value or pure.is_absolute() or ".." in pure.parts or "\\" in value or Path(value).is_absolute():
            raise ValueError(f"unsafe {description}: {value!r}")
        return value

    def _read_index(self) -> tuple[CachedFbankRow, ...]:
        path = self.cache_dir / f"{self.split}_feature_index_v1.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing cache index: {path}")
        rows: list[CachedFbankRow] = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = INDEX_FIELDS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            for line, raw in enumerate(reader, start=2):
                try:
                    relative = self._safe_relative_path(raw["relative_audio_path"].strip(), "audio path")
                    shard_path = self._safe_relative_path(raw["feature_shard_path"].strip(), "shard path")
                    position = int(raw["feature_index"])
                    label = int(raw["speaker_label"])
                    frames = int(raw["feature_frames"])
                    dimension = int(raw["feature_dim"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line}: malformed index metadata: {error}") from error
                speaker_id = raw["speaker_id"].strip()
                final_split = raw["final_split"].strip()
                group = raw["filename_group"].strip()
                if not speaker_id or group not in {"train", "train_small"}:
                    raise ValueError(f"{path}:{line}: invalid speaker or filename group")
                if final_split != self.split:
                    raise ValueError(f"{path}:{line}: expected split {self.split}, got {final_split!r}")
                if position < 0 or position >= int(self.config["shard_size"]):
                    raise ValueError(f"{path}:{line}: invalid feature_index {position}")
                if (frames, dimension) != self.feature_shape or raw["feature_dtype"].strip() != "float32":
                    raise ValueError(f"{path}:{line}: feature metadata disagrees with config")
                if self.split == "train" and not 0 <= label <= 487:
                    raise ValueError(f"{path}:{line}: train label must be in 0..487")
                if self.split != "train" and label != -1:
                    raise ValueError(f"{path}:{line}: evaluation label must be -1")
                rows.append(CachedFbankRow(relative, shard_path, position, speaker_id, label, final_split, group))
        expected = self.config["expected_rows"].get(self.split)
        if not isinstance(expected, int) or len(rows) != expected:
            raise ValueError(f"{self.split} index has {len(rows)} rows, expected {expected}")
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
        if not isinstance(shard, dict) or set(shard) != SHARD_KEYS:
            raise ValueError(f"malformed shard {path}: expected keys {sorted(SHARD_KEYS)}")
        features = shard["features"]
        labels = shard["speaker_labels"]
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError(f"malformed shard {path}: features must be a rank-3 tensor")
        count = features.shape[0]
        if tuple(features.shape[1:]) != self.feature_shape:
            raise ValueError(f"wrong feature shape in {path}: {tuple(features.shape[1:])}")
        if features.dtype != torch.float32:
            raise TypeError(f"wrong feature dtype in {path}: {features.dtype}")
        if features.device.type != "cpu":
            raise ValueError(f"features in {path} are not CPU tensors")
        if not isinstance(labels, torch.Tensor) or tuple(labels.shape) != (count,) or labels.dtype != torch.long:
            raise ValueError(f"malformed speaker labels in {path}")
        if labels.device.type != "cpu":
            raise ValueError(f"speaker labels in {path} are not CPU tensors")
        for key in ("speaker_ids", "relative_audio_paths"):
            if not isinstance(shard[key], list) or len(shard[key]) != count:
                raise ValueError(f"malformed {key} in {path}")
        if shard["final_split"] != self.split:
            raise ValueError(f"shard split mismatch in {path}: {shard['final_split']!r}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        shard = self._load_shard(row.feature_shard_path)
        count = shard["features"].shape[0]
        if row.feature_index >= count:
            raise IndexError(
                f"invalid feature position {row.feature_index} for {row.feature_shard_path} with {count} rows"
            )
        position = row.feature_index
        actual_label = int(shard["speaker_labels"][position].item())
        actual_id = shard["speaker_ids"][position]
        actual_path = shard["relative_audio_paths"][position]
        if (actual_label, actual_id, actual_path) != (row.speaker_label, row.speaker_id, row.relative_audio_path):
            raise ValueError(f"index metadata disagrees with shard at {row.feature_shard_path}[{position}]")
        feature = shard["features"][position]
        if tuple(feature.shape) != self.feature_shape or feature.dtype != torch.float32 or feature.device.type != "cpu":
            raise ValueError(f"invalid feature at {row.feature_shard_path}[{position}]")
        if self.validate_finite and not bool(torch.isfinite(feature).all().item()):
            raise ValueError(f"non-finite feature at {row.feature_shard_path}[{position}]")
        return {
            "fbank": feature,
            "speaker_label": row.speaker_label,
            "speaker_id": row.speaker_id,
            "relative_audio_path": row.relative_audio_path,
            "final_split": row.final_split,
            "filename_group": row.filename_group,
        }


def collate_cached_fbank(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return {
        "fbank": torch.stack([sample["fbank"] for sample in samples]),
        "speaker_label": torch.tensor([sample["speaker_label"] for sample in samples], dtype=torch.long),
        "speaker_id": [sample["speaker_id"] for sample in samples],
        "relative_audio_path": [sample["relative_audio_path"] for sample in samples],
        "final_split": [sample["final_split"] for sample in samples],
        "filename_group": [sample["filename_group"] for sample in samples],
    }


def create_cached_fbank_dataloader(
    dataset: CachedFbankDataset, batch_size: int, *, shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_cached_fbank, drop_last=False,
    )
