"""Emit stable hybrid-sampler batches for the subprocess determinism test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cached_fbank_dataset import CachedFbankDataset
from src.cached_fbank_samplers import HybridShardAwareSpeakerBatchSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("epoch", type=int)
    args = parser.parse_args()
    dataset = CachedFbankDataset(args.cache_dir, "train", validate_finite=False)
    sampler = HybridShardAwareSpeakerBatchSampler(
        dataset, speakers_per_batch=2, samples_per_speaker=2,
        num_batches=12, active_shard_window=1, seed=20260727,
    )
    sampler.set_epoch(args.epoch)
    print(json.dumps(list(sampler), separators=(",", ":")))


if __name__ == "__main__":
    main()
