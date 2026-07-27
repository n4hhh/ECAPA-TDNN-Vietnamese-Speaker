"""Focused tests for deterministic SpeechBrain Fbank cache mechanics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.precompute_speechbrain_fbank import (
    DIM, FRAMES, SourceRow, atomic_torch_save, index_rows,
    shard_relative_path, validate_shard,
)


class FbankCacheTests(unittest.TestCase):
    def rows(self) -> list[SourceRow]:
        return [
            SourceRow(f"audio/{index}.wav", str(index), index, "train", "train")
            for index in range(3)
        ]

    def shard(self, rows: list[SourceRow]) -> dict:
        return {
            "features": torch.zeros(len(rows), FRAMES, DIM, dtype=torch.float32),
            "speaker_labels": torch.tensor([row.speaker_label for row in rows]),
            "speaker_ids": [row.speaker_id for row in rows],
            "relative_audio_paths": [row.relative_audio_path for row in rows],
            "final_split": "train",
        }

    def test_valid_shard_and_atomic_save(self) -> None:
        rows = self.rows()
        shard = self.shard(rows)
        validate_shard(shard, rows, "train")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train" / "shard_00000.pt"
            atomic_torch_save(shard, path)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            validate_shard(torch.load(path, weights_only=False), rows, "train")

    def test_invalid_shape_and_nonfinite_rejected(self) -> None:
        rows = self.rows()
        shard = self.shard(rows)
        shard["features"] = torch.zeros(len(rows), DIM, FRAMES)
        with self.assertRaisesRegex(ValueError, "invalid feature"):
            validate_shard(shard, rows, "train")
        shard = self.shard(rows)
        shard["features"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid feature"):
            validate_shard(shard, rows, "train")

    def test_deterministic_index_and_shard_paths(self) -> None:
        rows = self.rows()
        first = index_rows(rows, "train", 2)
        second = index_rows(rows, "train", 2)
        self.assertEqual(first, second)
        self.assertEqual(first[2]["feature_shard_path"], "train/shard_00001.pt")
        self.assertEqual(first[2]["feature_index"], 0)
        self.assertEqual(shard_relative_path("validation", 12), "validation/shard_00012.pt")

    def test_metadata_mismatch_rejected(self) -> None:
        rows = self.rows()
        shard = self.shard(rows)
        shard["relative_audio_paths"][0] = "wrong.wav"
        with self.assertRaisesRegex(ValueError, "WAV paths"):
            validate_shard(shard, rows, "train")


if __name__ == "__main__":
    unittest.main()
