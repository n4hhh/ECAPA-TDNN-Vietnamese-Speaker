"""Unit tests for the lazy cached Fbank Dataset and collate path."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.cached_fbank_dataset import CachedFbankDataset, create_cached_fbank_dataloader


class CachedFbankDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self._write_config({"train": 3, "validation": 2, "test": 2})
        for split, count, label in (("train", 3, 7), ("validation", 2, -1), ("test", 2, -1)):
            self._write_split(split, count, label)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_config(self, counts: dict[str, int]) -> None:
        config = {
            "version": 1, "frontend": "speechbrain/spkrec-ecapa-voxceleb",
            "feature_stage": "raw_compute_features_before_mean_var_norm",
            "feature_shape": [301, 80], "feature_dtype": "float32",
            "shard_size": 2, "expected_rows": counts,
        }
        (self.root / "fbank_cache_config_v1.json").write_text(json.dumps(config), encoding="utf-8")

    def _write_split(self, split: str, count: int, label: int) -> None:
        fields = (
            "relative_audio_path", "feature_shard_path", "feature_index", "speaker_id",
            "speaker_label", "final_split", "filename_group", "feature_frames",
            "feature_dim", "feature_dtype",
        )
        rows = []
        for index in range(count):
            shard_number, position = divmod(index, 2)
            rows.append({
                "relative_audio_path": f"audio/{split}_{index}.wav",
                "feature_shard_path": f"{split}/shard_{shard_number:05d}.pt",
                "feature_index": position, "speaker_id": f"spk_{split}_{index}",
                "speaker_label": label if split != "train" else label + index,
                "final_split": split, "filename_group": "train_small" if index % 2 else "train",
                "feature_frames": 301, "feature_dim": 80, "feature_dtype": "float32",
            })
        path = self.root / f"{split}_feature_index_v1.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        for shard_number, start in enumerate(range(0, count, 2)):
            selected = rows[start:start + 2]
            features = torch.stack([torch.full((301, 80), float(start + offset)) for offset in range(len(selected))])
            shard = {
                "features": features, "speaker_labels": torch.tensor([int(row["speaker_label"]) for row in selected]),
                "speaker_ids": [row["speaker_id"] for row in selected],
                "relative_audio_paths": [row["relative_audio_path"] for row in selected],
                "final_split": split,
            }
            target = self.root / split / f"shard_{shard_number:05d}.pt"
            target.parent.mkdir(parents=True, exist_ok=True); torch.save(shard, target)

    def test_length_mapping_metadata_shape_dtype_and_repeat(self) -> None:
        dataset = CachedFbankDataset(self.root, "train")
        self.assertEqual(len(dataset), 3)
        sample = dataset[1]
        self.assertTrue(torch.equal(sample["fbank"], torch.ones(301, 80)))
        self.assertEqual(tuple(sample["fbank"].shape), (301, 80))
        self.assertEqual(sample["fbank"].dtype, torch.float32)
        self.assertEqual(sample["fbank"].device.type, "cpu")
        self.assertEqual(sample["speaker_label"], 8)
        self.assertEqual(sample["speaker_id"], "spk_train_1")
        self.assertEqual(sample["relative_audio_path"], "audio/train_1.wav")
        self.assertEqual(sample["final_split"], "train")
        self.assertEqual(sample["filename_group"], "train_small")
        self.assertTrue(torch.equal(sample["fbank"], dataset[1]["fbank"]))
        self.assertEqual(dataset.shard_load_count, 1)

    def test_batch_shapes_order_and_labels(self) -> None:
        dataset = CachedFbankDataset(self.root, "train")
        batch = next(iter(create_cached_fbank_dataloader(dataset, 3, shuffle=False, num_workers=0)))
        self.assertEqual(tuple(batch["fbank"].shape), (3, 301, 80))
        self.assertEqual(tuple(batch["speaker_label"].shape), (3,))
        self.assertEqual(batch["speaker_label"].dtype, torch.long)
        self.assertEqual(batch["speaker_label"].tolist(), [7, 8, 9])
        self.assertEqual(batch["speaker_id"], ["spk_train_0", "spk_train_1", "spk_train_2"])
        self.assertEqual(batch["relative_audio_path"], [f"audio/train_{i}.wav" for i in range(3)])

    def test_train_and_evaluation_label_rules(self) -> None:
        train = CachedFbankDataset(self.root, "train")
        validation = CachedFbankDataset(self.root, "validation")
        test = CachedFbankDataset(self.root, "test")
        self.assertTrue(all(0 <= row.speaker_label <= 487 for row in train.rows))
        self.assertTrue(all(row.speaker_label == -1 for row in validation.rows))
        self.assertTrue(all(row.speaker_label == -1 for row in test.rows))

    def test_missing_shard_error(self) -> None:
        (self.root / "train" / "shard_00000.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing feature shard"):
            CachedFbankDataset(self.root, "train")[0]

    def test_invalid_tensor_position_error(self) -> None:
        path = self.root / "train_feature_index_v1.csv"
        text = path.read_text(encoding="utf-8").replace(
            "shard_00001.pt,0,", "shard_00001.pt,1,", 1
        )
        path.write_text(text, encoding="utf-8")
        dataset = CachedFbankDataset(self.root, "train")
        with self.assertRaisesRegex(IndexError, "invalid feature position"):
            dataset[2]

    def test_wrong_shape_error(self) -> None:
        path = self.root / "train" / "shard_00000.pt"
        shard = torch.load(path, weights_only=False); shard["features"] = torch.zeros(2, 80, 301); torch.save(shard, path)
        with self.assertRaisesRegex(ValueError, "wrong feature shape"):
            CachedFbankDataset(self.root, "train")[0]

    def test_wrong_dtype_error(self) -> None:
        path = self.root / "train" / "shard_00000.pt"
        shard = torch.load(path, weights_only=False); shard["features"] = shard["features"].double(); torch.save(shard, path)
        with self.assertRaisesRegex(TypeError, "wrong feature dtype"):
            CachedFbankDataset(self.root, "train")[0]

    def _set_nonfinite(self, value: float) -> None:
        path = self.root / "train" / "shard_00000.pt"
        shard = torch.load(path, weights_only=False)
        shard["features"][0, 0, 0] = value
        torch.save(shard, path)

    def test_validate_finite_default_true_rejects_nan(self) -> None:
        self._set_nonfinite(float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite feature"):
            CachedFbankDataset(self.root, "train")[0]

    def test_validate_finite_explicit_true_rejects_inf(self) -> None:
        self._set_nonfinite(float("inf"))
        with self.assertRaisesRegex(ValueError, "non-finite feature"):
            CachedFbankDataset(self.root, "train", validate_finite=True)[0]

    def test_validate_finite_false_skips_only_finite_scan(self) -> None:
        self._set_nonfinite(float("nan"))
        sample = CachedFbankDataset(self.root, "train", validate_finite=False)[0]
        self.assertTrue(torch.isnan(sample["fbank"][0, 0]))
        self.assertEqual(sample["speaker_id"], "spk_train_0")
        self.assertEqual(tuple(sample["fbank"].shape), (301, 80))

    def test_validate_finite_false_preserves_other_validation(self) -> None:
        path = self.root / "train" / "shard_00000.pt"
        shard = torch.load(path, weights_only=False)
        shard["speaker_ids"][0] = "wrong"
        torch.save(shard, path)
        with self.assertRaisesRegex(ValueError, "index metadata disagrees"):
            CachedFbankDataset(self.root, "train", validate_finite=False)[0]

    def test_bounded_lru_shard_cache(self) -> None:
        dataset = CachedFbankDataset(self.root, "train", max_cached_shards=1)
        dataset[0]; dataset[2]
        self.assertEqual(dataset.cached_shard_count, 1)
        self.assertEqual(dataset.shard_load_count, 2)
        dataset[0]
        self.assertEqual(dataset.shard_load_count, 3)


if __name__ == "__main__":
    unittest.main()
