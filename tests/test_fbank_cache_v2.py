"""Focused synthetic tests for the VieSpeaker2.0 Fbank cache contract."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.precompute_speechbrain_fbank_v2 import (
    CONFIG_FILENAME,
    INDEX_FIELDS,
    INDEX_FILENAMES,
    SHARD_SCHEMA_VERSION,
    SourceRow,
    assert_no_absolute_dataset_root,
    atomic_torch_save,
    canonical_json_text,
    csv_text,
    plan_index_rows,
    prepare_cache_metadata,
    read_portable_manifest,
    reject_forbidden_split,
    require_sha256,
    validate_shard,
)
from src.cached_fbank_dataset import (
    CachedFbankDataset,
    create_cached_fbank_dataloader,
)


class FbankCacheV2Tests(unittest.TestCase):
    feature_shape = (7, 80)
    shard_size = 2

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def rows(split: str) -> list[SourceRow]:
        if split == "train":
            values = (("10/a.wav", "10", 0, "train"), ("11/b.wav", "11", 1, "part"))
        else:
            values = (("12/c.wav", "12", -1, "train_small"),)
        return [
            SourceRow(
                audio_path=audio_path,
                speaker_id=speaker_id,
                label=label,
                final_split=split,
                filename_group=group,
                duplicate_group="",
                manifest_version="v2",
                manifest_row_index=index,
            )
            for index, (audio_path, speaker_id, label, group) in enumerate(values)
        ]

    def v2_config(self) -> dict:
        return {
            "schema_version": 2,
            "cache_version": "v2",
            "model_source": "speechbrain/spkrec-ecapa-voxceleb",
            "frontend": "speechbrain/spkrec-ecapa-voxceleb",
            "speechbrain_version": "synthetic",
            "torch_version": torch.__version__,
            "torchaudio_version": "synthetic",
            "input_bindings": {},
            "included_splits": ["train", "validation"],
            "feature_stage": "raw_compute_features_before_mean_var_norm",
            "feature_shape": list(self.feature_shape),
            "feature_dtype": "float32",
            "raw_pre_normalization": True,
            "transposed": False,
            "sample_rate_hz": 16000,
            "waveform_samples": 10,
            "shard_schema_version": 2,
            "shard_size": self.shard_size,
            "requested_batch_size": 64,
            "extraction_batch_size": 64,
            "batch_size_fallback_used": False,
            "device_used": "cuda:0",
            "expected_rows": {"train": 2, "validation": 1},
            "speaker_counts": {"train": 2, "validation": 1},
            "train_class_count": 2,
            "train_label_range": [0, 1],
            "validation_label": -1,
            "allowed_filename_groups": ["part", "train", "train_small"],
            "index_filenames": dict(INDEX_FILENAMES),
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

    def shard(self, split: str) -> dict:
        rows = self.rows(split)
        return {
            "schema_version": SHARD_SCHEMA_VERSION,
            "features": torch.stack(
                [
                    torch.full(self.feature_shape, float(index), dtype=torch.float32)
                    for index in range(len(rows))
                ]
            ),
            "speaker_labels": torch.tensor(
                [row.label for row in rows], dtype=torch.long
            ),
            "speaker_ids": [row.speaker_id for row in rows],
            "relative_audio_paths": [row.audio_path for row in rows],
            "manifest_row_indices": [row.manifest_row_index for row in rows],
            "final_split": split,
        }

    def write_v2_cache(self) -> None:
        config = self.v2_config()
        config_path = self.root / CONFIG_FILENAME
        config_path.write_text(canonical_json_text(config), encoding="utf-8")
        index_hashes = {}
        for split in ("train", "validation"):
            rows = self.rows(split)
            index_path = self.root / INDEX_FILENAMES[split]
            index_path.write_text(
                csv_text(
                    plan_index_rows(
                        rows, split, self.shard_size, self.feature_shape
                    )
                ),
                encoding="utf-8",
            )
            index_hashes[split] = hashlib.sha256(index_path.read_bytes()).hexdigest()
            shard_path = self.root / split / "shard_00000.pt"
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.shard(split), shard_path)
        identity = {
            "schema_version": 2,
            "identity_kind": "fbank_cache_v2",
            "cache_version": "v2",
            "model_source": config["model_source"],
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "included_splits": ["train", "validation"],
            "feature_shape": list(self.feature_shape),
            "feature_dtype": "float32",
            "raw_pre_normalization": True,
            "transposed": False,
            "shard_size": self.shard_size,
            "row_counts": config["expected_rows"],
            "train_class_count": 2,
            "train_label_range": [0, 1],
            "validation_label": -1,
            "index_sha256": index_hashes,
            "shard_counts": {"train": 1, "validation": 1},
            "total_cached_utterances": 3,
            "final_test_cache_absent": True,
        }
        (self.root / "fbank_cache_identity_v2.json").write_text(
            canonical_json_text(identity), encoding="utf-8"
        )

    def write_manifest(self, split: str, label: int) -> Path:
        path = self.root / f"{split}.csv"
        fields = (
            "relative_audio_path",
            "speaker_id",
            "speaker_label",
            "final_split",
            "filename_group",
            "duplicate_group",
            "manifest_version",
        )
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerow(
                {
                    "relative_audio_path": "10/a.wav",
                    "speaker_id": "10",
                    "speaker_label": label,
                    "final_split": split,
                    "filename_group": "part",
                    "duplicate_group": "",
                    "manifest_version": "v2",
                }
            )
        return path

    def write_v1_cache(self) -> None:
        config = {
            "version": 1,
            "frontend": "speechbrain/spkrec-ecapa-voxceleb",
            "feature_stage": "raw_compute_features_before_mean_var_norm",
            "feature_shape": [301, 80],
            "feature_dtype": "float32",
            "shard_size": 2,
            "expected_rows": {"train": 1, "validation": 1, "test": 1},
        }
        (self.root / "fbank_cache_config_v1.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        fields = (
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
        )
        for split, label in (("train", 7), ("validation", -1), ("test", -1)):
            index_path = self.root / f"{split}_feature_index_v1.csv"
            with index_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerow(
                    {
                        "relative_audio_path": f"audio/{split}.wav",
                        "feature_shard_path": f"{split}/shard_00000.pt",
                        "feature_index": 0,
                        "speaker_id": "speaker",
                        "speaker_label": label,
                        "final_split": split,
                        "filename_group": "train",
                        "feature_frames": 301,
                        "feature_dim": 80,
                        "feature_dtype": "float32",
                    }
                )
            shard_path = self.root / split / "shard_00000.pt"
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "features": torch.zeros(1, 301, 80),
                    "speaker_labels": torch.tensor([label]),
                    "speaker_ids": ["speaker"],
                    "relative_audio_paths": [f"audio/{split}.wav"],
                    "final_split": split,
                },
                shard_path,
            )

    def test_identity_mismatch_rejected(self) -> None:
        path = self.root / "identity.bin"
        path.write_bytes(b"approved")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            require_sha256(path, "0" * 64, "synthetic input")

    def test_train_validation_allowlist_and_test_rejection(self) -> None:
        train_path = self.write_manifest("train", 0)
        rows = read_portable_manifest(
            train_path,
            "train",
            expected_rows=1,
            label_range=(0, 0),
            label_mapping={"10": 0},
        )
        self.assertEqual(rows[0].audio_path, "10/a.wav")
        with self.assertRaisesRegex(ValueError, "final-test"):
            reject_forbidden_split("test")
        with self.assertRaisesRegex(ValueError, "final-test"):
            read_portable_manifest(
                train_path,
                "test",
                expected_rows=1,
                label_range=(0, 0),
                label_mapping={"10": 0},
            )

    def test_dynamic_feature_t_float32_finite_and_no_transpose(self) -> None:
        rows = self.rows("train")
        validate_shard(
            self.shard("train"),
            rows,
            "train",
            self.feature_shape,
            self.shard_size,
        )
        transposed = self.shard("train")
        transposed["features"] = torch.zeros(2, 80, 7)
        with self.assertRaisesRegex(ValueError, "non-transposed"):
            validate_shard(
                transposed, rows, "train", self.feature_shape, self.shard_size
            )
        nonfinite = self.shard("train")
        nonfinite["features"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_shard(
                nonfinite, rows, "train", self.feature_shape, self.shard_size
            )
        wrong_dtype = self.shard("train")
        wrong_dtype["features"] = wrong_dtype["features"].double()
        with self.assertRaisesRegex(ValueError, "float32"):
            validate_shard(
                wrong_dtype, rows, "train", self.feature_shape, self.shard_size
            )

    def test_deterministic_shard_planning_and_portable_paths(self) -> None:
        rows = self.rows("train")
        first = plan_index_rows(rows, "train", 1, self.feature_shape)
        second = plan_index_rows(rows, "train", 1, self.feature_shape)
        self.assertEqual(first, second)
        self.assertEqual(first[1]["shard_path"], "train/shard_00001.pt")
        self.assertEqual(first[1]["within_shard_index"], 0)
        text = csv_text(first)
        self.assertNotIn("E:\\", text)
        self.assertNotIn("\\", first[0]["audio_path"])

    def test_label_validation(self) -> None:
        train_path = self.write_manifest("train", 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            read_portable_manifest(
                train_path,
                "train",
                expected_rows=1,
                label_range=(0, 0),
                label_mapping={"10": 0},
            )
        validation_path = self.write_manifest("validation", 0)
        with self.assertRaisesRegex(ValueError, "must be -1"):
            read_portable_manifest(
                validation_path,
                "validation",
                expected_rows=1,
                label_range=(0, 0),
                label_mapping={"10": 0},
            )

    def test_atomic_publication_and_stale_partial_rejection(self) -> None:
        path = self.root / "train" / "shard_00000.pt"
        atomic_torch_save(self.shard("train"), path)
        self.assertTrue(path.is_file())
        self.assertFalse(path.with_name(path.name + ".tmp").exists())
        second = self.root / "train" / "shard_00001.pt"
        second.with_name(second.name + ".tmp").write_bytes(b"partial")
        with self.assertRaisesRegex(FileExistsError, "stale temporary"):
            atomic_torch_save(self.shard("train"), second)
        self.assertFalse(second.exists())

    def test_compatible_and_incompatible_resume(self) -> None:
        config_text = canonical_json_text(self.v2_config())
        indexes = {
            split: csv_text(
                plan_index_rows(
                    self.rows(split),
                    split,
                    self.shard_size,
                    self.feature_shape,
                )
            )
            for split in ("train", "validation")
        }
        prepare_cache_metadata(self.root, config_text, indexes, resume=False)
        prepare_cache_metadata(self.root, config_text, indexes, resume=True)
        changed = dict(indexes)
        changed["train"] += "\n"
        with self.assertRaisesRegex(ValueError, "incompatible train index"):
            prepare_cache_metadata(self.root, config_text, changed, resume=True)
        incompatible_config = config_text.replace('"shard_size": 2', '"shard_size": 3')
        with self.assertRaisesRegex(ValueError, "incompatible cache configuration"):
            prepare_cache_metadata(
                self.root, incompatible_config, indexes, resume=True
            )

    def test_v2_dataset_index_alignment_dynamic_t_and_lru(self) -> None:
        self.write_v2_cache()
        dataset = CachedFbankDataset(self.root, "train", max_cached_shards=1)
        self.assertEqual(dataset.feature_shape, self.feature_shape)
        sample = dataset[1]
        self.assertEqual(sample["dataset_index"], 1)
        self.assertEqual(sample["manifest_row_index"], 1)
        self.assertEqual(sample["speaker_id"], "11")
        self.assertEqual(tuple(sample["fbank"].shape), self.feature_shape)
        batch = next(
            iter(
                create_cached_fbank_dataloader(
                    dataset, 2, shuffle=False, num_workers=0
                )
            )
        )
        self.assertEqual(tuple(batch["fbank"].shape), (2, *self.feature_shape))
        self.assertEqual(batch["manifest_row_index"].tolist(), [0, 1])
        self.assertLessEqual(dataset.cached_shard_count, 1)

    def test_missing_and_corrupt_shard_rejected(self) -> None:
        self.write_v2_cache()
        shard_path = self.root / "train" / "shard_00000.pt"
        shard_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing feature shard"):
            CachedFbankDataset(self.root, "train")[0]
        torch.save({"corrupt": True}, shard_path)
        with self.assertRaisesRegex(ValueError, "malformed shard"):
            CachedFbankDataset(self.root, "train")[0]

    def test_v2_identity_and_index_hash_mismatch_rejected(self) -> None:
        self.write_v2_cache()
        config_path = self.root / CONFIG_FILENAME
        config_path.write_text(
            config_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "config hash"):
            CachedFbankDataset(self.root, "train")

    def test_v2_test_dataset_request_rejected(self) -> None:
        self.write_v2_cache()
        with self.assertRaisesRegex(ValueError, "final-test"):
            CachedFbankDataset(self.root, "test")

    def test_v1_backward_compatibility(self) -> None:
        self.write_v1_cache()
        train = CachedFbankDataset(self.root, "train")
        validation = CachedFbankDataset(self.root, "validation")
        test = CachedFbankDataset(self.root, "test")
        self.assertEqual(tuple(train[0]["fbank"].shape), (301, 80))
        self.assertEqual(train[0]["speaker_label"], 7)
        self.assertEqual(validation[0]["speaker_label"], -1)
        self.assertEqual(test[0]["speaker_label"], -1)

    def test_no_persisted_absolute_dataset_root(self) -> None:
        config = self.v2_config()
        assert_no_absolute_dataset_root(config, Path("E:/VieSpeaker2.0/augmented_dataset"))
        config["bad"] = "E:/VieSpeaker2.0/augmented_dataset"
        with self.assertRaisesRegex(ValueError, "absolute dataset root"):
            assert_no_absolute_dataset_root(
                config, Path("E:/VieSpeaker2.0/augmented_dataset")
            )


if __name__ == "__main__":
    unittest.main()
