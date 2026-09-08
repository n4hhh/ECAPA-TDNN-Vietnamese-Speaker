"""Production cached-Fbank input constructors for adaptive_augmented_3s_v1."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.cached_fbank_dataset import (
    CachedFbankDataset,
    create_cached_fbank_dataloader,
    create_cached_fbank_training_dataloader,
)
from src.cached_fbank_samplers import HybridShardAwareSpeakerBatchSampler


PRODUCTION_CACHE_IDENTITY = "019e2734ad6e1b62b230c3938befa2d3620a70af895778952622f9ab2bf7446b"


def _dataset(
    cache_root: str | Path, split: str, *, max_cached_shards: int,
) -> CachedFbankDataset:
    dataset = CachedFbankDataset(
        cache_root,
        split,
        max_cached_shards=max_cached_shards,
        # The completed Task 3 cache already validated every persisted tensor.
        validate_finite=False,
    )
    if (
        dataset.identity is None
        or dataset.identity.get("identity_sha256") != PRODUCTION_CACHE_IDENTITY
    ):
        raise ValueError("cache identity is not the approved adaptive_augmented_3s_v1 cache")
    return dataset


def create_train_input(
    cache_root: str | Path,
    *,
    speakers_per_batch: int = 16,
    samples_per_speaker: int = 2,
    active_shard_window: int = 8, seed: int = 20260729,
    num_workers: int = 0, max_cached_shards: int = 8,
) -> tuple[CachedFbankDataset, HybridShardAwareSpeakerBatchSampler, DataLoader]:
    """Create the train-only P x K loader; callers set the sampler epoch."""
    dataset = _dataset(cache_root, "train", max_cached_shards=max_cached_shards)
    sampler = HybridShardAwareSpeakerBatchSampler(
        dataset,
        speakers_per_batch=speakers_per_batch,
        samples_per_speaker=samples_per_speaker,
        active_shard_window=active_shard_window, seed=seed,
    )
    return dataset, sampler, create_cached_fbank_training_dataloader(
        dataset, sampler, num_workers=num_workers
    )


def create_validation_input(
    cache_root: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0, max_cached_shards: int = 8,
) -> tuple[CachedFbankDataset, DataLoader]:
    """Create the fixed-order validation loader without a train sampler."""
    dataset = _dataset(cache_root, "validation", max_cached_shards=max_cached_shards)
    return dataset, create_cached_fbank_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
