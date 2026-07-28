"""Tests for deterministic cached-Fbank training batch samplers."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from src.cached_fbank_dataset import (
    CachedFbankDataset,
    create_cached_fbank_training_dataloader,
)
from src.cached_fbank_samplers import (
    DeterministicRandomBatchSampler,
    GlobalSpeakerBalancedBatchSampler,
    HybridShardAwareSpeakerBatchSampler,
    validate_train_sampler_metadata,
)


class SamplerFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        counts = {"train": 32, "validation": 2, "test": 2}
        config = {
            "version": 1, "frontend": "speechbrain/spkrec-ecapa-voxceleb",
            "feature_stage": "raw_compute_features_before_mean_var_norm",
            "feature_shape": [301, 80], "feature_dtype": "float32",
            "shard_size": 8, "expected_rows": counts,
        }
        (self.root / "fbank_cache_config_v1.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        self._write_train()
        self._write_eval("validation")
        self._write_eval("test")
        self.dataset = CachedFbankDataset(
            self.root, "train", max_cached_shards=2, validate_finite=False
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _fields() -> tuple[str, ...]:
        return (
            "relative_audio_path", "feature_shard_path", "feature_index", "speaker_id",
            "speaker_label", "final_split", "filename_group", "feature_frames",
            "feature_dim", "feature_dtype",
        )

    def _save(self, split: str, shard_number: int, rows: list[dict[str, object]]) -> None:
        target = self.root / split / f"shard_{shard_number:05d}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "features": torch.stack([
                torch.full((301, 80), float(row["value"])) for row in rows
            ]),
            "speaker_labels": torch.tensor([int(row["speaker_label"]) for row in rows]),
            "speaker_ids": [str(row["speaker_id"]) for row in rows],
            "relative_audio_paths": [str(row["relative_audio_path"]) for row in rows],
            "final_split": split,
        }, target)

    def _write_train(self) -> None:
        rows: list[dict[str, object]] = []
        # Four shards, each containing two speakers with four utterances apiece.
        for shard in range(4):
            shard_rows = []
            for label in (shard * 2, shard * 2 + 1):
                for local in range(4):
                    value = len(rows)
                    row = {
                        "relative_audio_path": f"audio/train_{value}.wav",
                        "feature_shard_path": f"train/shard_{shard:05d}.pt",
                        "feature_index": len(shard_rows), "speaker_id": f"s{label}",
                        "speaker_label": label, "final_split": "train",
                        "filename_group": "train", "feature_frames": 301,
                        "feature_dim": 80, "feature_dtype": "float32", "value": value,
                    }
                    rows.append(row); shard_rows.append(row)
            self._save("train", shard, shard_rows)
        with (self.root / "train_feature_index_v1.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=self._fields(), extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)

    def _write_eval(self, split: str) -> None:
        rows = []
        for index in range(2):
            rows.append({
                "relative_audio_path": f"audio/{split}_{index}.wav",
                "feature_shard_path": f"{split}/shard_00000.pt",
                "feature_index": index, "speaker_id": f"{split}_{index}",
                "speaker_label": -1, "final_split": split, "filename_group": "train",
                "feature_frames": 301, "feature_dim": 80,
                "feature_dtype": "float32", "value": index,
            })
        with (self.root / f"{split}_feature_index_v1.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=self._fields(), extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        self._save(split, 0, rows)

    def test_bijection_and_invalid_mappings(self) -> None:
        result = validate_train_sampler_metadata(self.dataset)
        self.assertEqual((result["rows"], result["speakers"], result["shards"]), (32, 8, 4))
        original = self.dataset.rows
        self.dataset.rows = (replace(original[0], speaker_label=1),) + original[1:]
        with self.assertRaisesRegex(ValueError, "maps to"):
            validate_train_sampler_metadata(self.dataset)
        self.dataset.rows = original

    def test_rejects_evaluation_datasets(self) -> None:
        for split in ("validation", "test"):
            dataset = CachedFbankDataset(self.root, split)
            with self.assertRaisesRegex(ValueError, "train split"):
                DeterministicRandomBatchSampler(dataset, 4)

    def test_random_complete_drop_last_and_epochs(self) -> None:
        sampler = DeterministicRandomBatchSampler(
            self.dataset, 6, seed=17, drop_last=False
        )
        first = list(sampler); second = list(sampler)
        self.assertEqual(first, second)
        self.assertEqual(len(sampler), 6)
        self.assertEqual(sorted(index for batch in first for index in batch), list(range(32)))
        self.assertTrue(all(len(batch) == len(set(batch)) for batch in first))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        dropped = DeterministicRandomBatchSampler(self.dataset, 6, drop_last=True)
        self.assertEqual(len(dropped), 5)
        self.assertEqual(sum(map(len, dropped)), 30)

    def _assert_pk(self, sampler: object, batches: list[list[int]], p: int, k: int) -> None:
        self.assertEqual(len(batches), len(sampler))  # type: ignore[arg-type]
        for batch in batches:
            self.assertEqual(len(batch), p * k)
            self.assertEqual(len(batch), len(set(batch)))
            counts: dict[str, int] = {}
            for index in batch:
                self.assertIn(index, range(len(self.dataset)))
                speaker = self.dataset.rows[index].speaker_id
                counts[speaker] = counts.get(speaker, 0) + 1
            self.assertEqual(len(counts), p)
            self.assertEqual(set(counts.values()), {k})

    def test_global_pk_balance_cycles_and_determinism(self) -> None:
        sampler = GlobalSpeakerBalancedBatchSampler(
            self.dataset, speakers_per_batch=4, samples_per_speaker=2,
            num_batches=10, seed=23,
        )
        first = list(sampler); self._assert_pk(sampler, first, 4, 2)
        self.assertEqual(first, list(sampler))
        exposure = {speaker: 0 for speaker in sampler.speakers}
        for batch in first:
            for speaker in {self.dataset.rows[index].speaker_id for index in batch}:
                exposure[speaker] += 1
        self.assertLessEqual(max(exposure.values()) - min(exposure.values()), 1)
        self.assertGreater(sampler.last_epoch_stats["queue_cycles"], 0)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))

    def test_hybrid_exact_local_and_balanced(self) -> None:
        hybrid = HybridShardAwareSpeakerBatchSampler(
            self.dataset, speakers_per_batch=2, samples_per_speaker=2,
            num_batches=12, active_shard_window=1, seed=29,
        )
        batches = list(hybrid); self._assert_pk(hybrid, batches, 2, 2)
        self.assertTrue(all(value == 1 for value in hybrid.last_epoch_stats["distinct_shards_per_batch"]))
        exposure = hybrid.last_epoch_stats["speaker_batch_exposure"]
        self.assertLessEqual(max(exposure.values()) - min(exposure.values()), 1)
        self.assertEqual(batches, list(hybrid))
        hybrid.set_epoch(1)
        self.assertNotEqual(batches, list(hybrid))

    def test_hybrid_determinism_across_processes_and_hash_seeds(self) -> None:
        helper = Path(__file__).with_name("run_hybrid_sampler_fixture.py")

        def run(hash_seed: str, epoch: int) -> list[list[int]]:
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            completed = subprocess.run(
                [sys.executable, str(helper), str(self.root), str(epoch)],
                cwd=Path(__file__).resolve().parents[1],
                env=environment, check=True, capture_output=True, text=True,
            )
            return json.loads(completed.stdout)

        epoch_zero_a = run("1", 0)
        epoch_zero_b = run("999", 0)
        epoch_one = run("1", 1)
        self.assertEqual(epoch_zero_a, epoch_zero_b)
        self.assertNotEqual(epoch_zero_a, epoch_one)
        self._assert_pk(
            HybridShardAwareSpeakerBatchSampler(
                self.dataset, speakers_per_batch=2, samples_per_speaker=2,
                num_batches=12, active_shard_window=1, seed=20260727,
            ),
            epoch_zero_a, 2, 2,
        )

    def test_hybrid_locality_beats_global_fixture(self) -> None:
        global_sampler = GlobalSpeakerBalancedBatchSampler(
            self.dataset, speakers_per_batch=2, samples_per_speaker=2,
            num_batches=20, seed=31,
        )
        hybrid = HybridShardAwareSpeakerBatchSampler(
            self.dataset, speakers_per_batch=2, samples_per_speaker=2,
            num_batches=20, active_shard_window=1, seed=31,
        )
        list(global_sampler); list(hybrid)
        self.assertLess(
            sum(hybrid.last_epoch_stats["distinct_shards_per_batch"]),
            sum(global_sampler.last_epoch_stats["distinct_shards_per_batch"]),
        )

    def test_invalid_pk_window_and_impossible_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "P exceeds"):
            GlobalSpeakerBalancedBatchSampler(self.dataset, speakers_per_batch=9)
        with self.assertRaisesRegex(ValueError, "K=5"):
            GlobalSpeakerBalancedBatchSampler(
                self.dataset, speakers_per_batch=2, samples_per_speaker=5
            )
        with self.assertRaisesRegex(ValueError, "active_shard_window"):
            HybridShardAwareSpeakerBatchSampler(
                self.dataset, speakers_per_batch=2, active_shard_window=0
            )
        # Every one-shard window has only two speakers, so P=3 forces deterministic expansion.
        expanded = HybridShardAwareSpeakerBatchSampler(
            self.dataset, speakers_per_batch=3, samples_per_speaker=2,
            num_batches=2, active_shard_window=1,
        )
        self._assert_pk(expanded, list(expanded), 3, 2)

    def test_training_dataloader_alignment(self) -> None:
        sampler = GlobalSpeakerBalancedBatchSampler(
            self.dataset, speakers_per_batch=2, samples_per_speaker=2,
            num_batches=1, seed=37,
        )
        expected = list(sampler)[0]
        batch = next(iter(create_cached_fbank_training_dataloader(
            self.dataset, sampler, num_workers=0
        )))
        self.assertEqual(tuple(batch["fbank"].shape), (4, 301, 80))
        self.assertEqual(batch["speaker_label"].dtype, torch.long)
        self.assertEqual(
            batch["speaker_id"],
            [self.dataset.rows[index].speaker_id for index in expected],
        )
        self.assertEqual(
            batch["fbank"][:, 0, 0].tolist(), [float(index) for index in expected]
        )


if __name__ == "__main__":
    unittest.main()
