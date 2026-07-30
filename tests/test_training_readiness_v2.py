from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.cached_fbank_dataset import CachedFbankDataset, CachedFbankRow
from src.cached_fbank_samplers import HybridShardAwareSpeakerBatchSampler
from src.training_readiness_v2 import (
    SAMPLER_PARAMETERS,
    compact_json_sha256,
    plan_sha256,
)


def synthetic_dataset() -> CachedFbankDataset:
    dataset = CachedFbankDataset.__new__(CachedFbankDataset)
    dataset.split = "train"
    dataset.cache_dir = Path(".")
    rows = []
    for label, speaker in enumerate(("1", "2", "3", "4")):
        for position in range(4):
            rows.append(
                CachedFbankRow(
                    relative_audio_path=f"{speaker}/{position}.wav",
                    feature_shard_path=f"train/shard_{position % 2:05d}.pt",
                    feature_index=position,
                    speaker_id=speaker,
                    speaker_label=label,
                    final_split="train",
                    filename_group="train",
                    manifest_row_index=len(rows),
                    duplicate_group=(
                        "duplicate-pair"
                        if speaker == "1" and position in (0, 1)
                        else ""
                    ),
                    manifest_version="v2",
                )
            )
    dataset.rows = tuple(rows)
    return dataset


class TrainingReadinessV2Tests(unittest.TestCase):
    def test_approved_sampler_parameters_are_fixed(self) -> None:
        self.assertEqual(
            SAMPLER_PARAMETERS,
            {
                "sampler_class": "HybridShardAwareSpeakerBatchSampler",
                "speakers_per_batch": 16,
                "samples_per_speaker": 2,
                "batch_size": 32,
                "active_shard_window": 8,
                "seed": 20260729,
                "batches_per_epoch": 2969,
                "logical_selections_per_epoch": 95008,
                "num_workers": 0,
                "dataset_max_cached_shards": 8,
                "dataset_validate_finite": False,
            },
        )

    def test_hybrid_sampler_is_exact_and_duplicate_group_safe(self) -> None:
        dataset = synthetic_dataset()
        before = tuple(dataset.rows)
        sampler = HybridShardAwareSpeakerBatchSampler(
            dataset,
            speakers_per_batch=2,
            samples_per_speaker=2,
            num_batches=20,
            seed=20260729,
            active_shard_window=2,
            validate_shard_existence=False,
        )
        batches = list(sampler)
        self.assertEqual(dataset.rows, before)
        self.assertEqual(len(batches), 20)
        for batch in batches:
            self.assertEqual(len(batch), 4)
            speakers = {}
            for index in batch:
                row = dataset.rows[index]
                speakers.setdefault(row.speaker_id, []).append(row)
            self.assertEqual(sorted(map(len, speakers.values())), [2, 2])
            for rows in speakers.values():
                groups = [row.duplicate_group for row in rows if row.duplicate_group]
                self.assertEqual(len(groups), len(set(groups)))
        self.assertEqual(sampler.last_epoch_stats["duplicate_group_rejections"] >= 0, True)

    def test_duplicate_safe_capacity_fails_closed(self) -> None:
        dataset = synthetic_dataset()
        rows = list(dataset.rows)
        rows[2] = CachedFbankRow(
            **{
                **rows[2].__dict__,
                "duplicate_group": "duplicate-pair",
            }
        )
        rows[3] = CachedFbankRow(
            **{
                **rows[3].__dict__,
                "duplicate_group": "duplicate-pair",
            }
        )
        dataset.rows = tuple(rows)
        with self.assertRaisesRegex(ValueError, "duplicate-safe"):
            HybridShardAwareSpeakerBatchSampler(
                dataset,
                speakers_per_batch=2,
                samples_per_speaker=2,
                num_batches=1,
                active_shard_window=2,
                validate_shard_existence=False,
            )

    def test_plan_hash_changes_by_epoch(self) -> None:
        dataset = synthetic_dataset()
        plans = []
        for epoch in (0, 1):
            sampler = HybridShardAwareSpeakerBatchSampler(
                dataset,
                speakers_per_batch=2,
                samples_per_speaker=2,
                num_batches=10,
                seed=20260729,
                active_shard_window=2,
                validate_shard_existence=False,
            )
            sampler.set_epoch(epoch)
            batches = list(sampler)
            plans.append((batches, plan_sha256(epoch, batches)))
        self.assertNotEqual(plans[0][0], plans[1][0])
        self.assertNotEqual(plans[0][1], plans[1][1])

    def test_compact_json_hash_is_pythonhashseed_independent(self) -> None:
        code = (
            "from src.training_readiness_v2 import compact_json_sha256;"
            "print(compact_json_sha256({'b':[3,2,1],'a':{'y':2,'x':1}}))"
        )
        outputs = []
        for hash_seed in ("1", "987654"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", code],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    text=True,
                ).strip()
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(
            outputs[0],
            compact_json_sha256({"a": {"x": 1, "y": 2}, "b": [3, 2, 1]}),
        )


if __name__ == "__main__":
    unittest.main()
