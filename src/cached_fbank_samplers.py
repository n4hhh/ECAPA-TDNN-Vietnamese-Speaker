"""Deterministic speaker-balanced samplers for the production cached Fbanks."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterator

from torch.utils.data import Sampler

from src.cached_fbank_dataset import CachedFbankDataset


def validate_train_sampler_metadata(
    dataset: CachedFbankDataset, *, validate_shard_existence: bool = True,
) -> dict[str, object]:
    """Validate the train-only metadata consumed by the production sampler."""
    if not isinstance(dataset, CachedFbankDataset):
        raise TypeError("dataset must be a CachedFbankDataset")
    if dataset.cache_version != 3 or dataset.split != "train":
        raise ValueError("adaptive_augmented_3s_v1 train cache required")
    if not dataset.rows:
        raise ValueError("train dataset is empty")

    speaker_to_label: dict[str, int] = {}
    label_to_speaker: dict[int, str] = {}
    shards: set[str] = set()
    for index, row in enumerate(dataset.rows):
        if row.final_split != "train" or not row.speaker_id or row.speaker_label < 0:
            raise ValueError(f"row {index} is not valid train metadata")
        if speaker_to_label.setdefault(row.speaker_id, row.speaker_label) != row.speaker_label:
            raise ValueError(f"speaker {row.speaker_id!r} has inconsistent labels")
        if label_to_speaker.setdefault(row.speaker_label, row.speaker_id) != row.speaker_id:
            raise ValueError(f"label {row.speaker_label} has inconsistent speakers")
        shards.add(row.feature_shard_path)

    labels = set(label_to_speaker)
    if labels != set(range(len(labels))):
        raise ValueError("train labels must be contiguous from zero")
    if validate_shard_existence:
        missing = [
            path for path in sorted(shards)
            if not (dataset.cache_dir / Path(*PurePosixPath(path).parts)).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing referenced train shard: {missing[0]}")
    return {
        "rows": len(dataset), "speakers": len(speaker_to_label),
        "labels": tuple(sorted(labels)), "shards": len(shards),
        "speaker_to_label": speaker_to_label,
    }


class HybridShardAwareSpeakerBatchSampler(Sampler[list[int]]):
    """Generate full deterministic P×K batches with a moving shard window.

    Speakers are selected by least exposure, with a local seeded tie-break. Each
    selected speaker takes K distinct rows, preferring the active shard window.
    A speaker queue is reshuffled only after all its rows have been used, making
    replacement explicit in ``last_epoch_stats`` rather than a hidden policy.
    """

    def __init__(
        self,
        dataset: CachedFbankDataset,
        *,
        speakers_per_batch: int = 16,
        samples_per_speaker: int = 2,
        active_shard_window: int = 8,
        num_batches: int | None = None,
        seed: int = 0,
        validate_shard_existence: bool = True,
    ) -> None:
        metadata = validate_train_sampler_metadata(
            dataset, validate_shard_existence=validate_shard_existence
        )
        if speakers_per_batch < 1 or samples_per_speaker < 1:
            raise ValueError("speakers_per_batch and samples_per_speaker must be positive")
        if speakers_per_batch > int(metadata["speakers"]):
            raise ValueError("speakers_per_batch exceeds available train speakers")
        self.dataset = dataset
        self.speakers_per_batch = speakers_per_batch
        self.samples_per_speaker = samples_per_speaker
        self.batch_size = speakers_per_batch * samples_per_speaker
        self.num_batches = (
            math.ceil(len(dataset) / self.batch_size)
            if num_batches is None else num_batches
        )
        if self.num_batches < 1:
            raise ValueError("num_batches must be positive")
        self.seed = seed
        self.epoch = 0

        grouped: dict[str, list[int]] = defaultdict(list)
        shards: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            grouped[row.speaker_id].append(index)
            shards[row.feature_shard_path].append(index)
        if any(len(indexes) < samples_per_speaker for indexes in grouped.values()):
            raise ValueError("a train speaker has fewer rows than samples_per_speaker")
        if not 1 <= active_shard_window <= len(shards):
            raise ValueError("active_shard_window must be within the train shard count")
        self.speaker_indexes = {speaker: tuple(indexes) for speaker, indexes in grouped.items()}
        self.speakers = tuple(sorted(grouped, key=lambda speaker: dataset.rows[grouped[speaker][0]].speaker_label))
        self.shard_indexes = {shard: tuple(indexes) for shard, indexes in shards.items()}
        self.shards = tuple(sorted(shards))
        self.active_shard_window = active_shard_window
        self.last_epoch_stats: dict[str, object] = {}

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def _take_rows(
        self, speaker: str, queue: list[int], preferred: set[int], rng: random.Random,
        queue_cycles: Counter[str],
    ) -> list[int]:
        chosen: list[int] = []
        while len(chosen) < self.samples_per_speaker:
            candidates = [index for index in queue if index not in chosen and index in preferred]
            if not candidates:
                candidates = [index for index in queue if index not in chosen]
            if not candidates:
                queue[:] = list(self.speaker_indexes[speaker])
                rng.shuffle(queue)
                queue_cycles[speaker] += 1
                continue
            selected = candidates[0]
            queue.remove(selected)
            chosen.append(selected)
        return chosen

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(f"{self.seed}:{self.epoch}")
        shard_order = list(self.shards)
        rng.shuffle(shard_order)
        queues = {speaker: list(indexes) for speaker, indexes in self.speaker_indexes.items()}
        for queue in queues.values():
            rng.shuffle(queue)
        exposure: Counter[str] = Counter()
        queue_cycles: Counter[str] = Counter()
        batches: list[list[int]] = []
        cursor = 0

        for _ in range(self.num_batches):
            width = self.active_shard_window
            while True:
                active_shards = [shard_order[(cursor + offset) % len(shard_order)] for offset in range(width)]
                preferred: dict[str, set[int]] = defaultdict(set)
                for shard in active_shards:
                    for index in self.shard_indexes[shard]:
                        preferred[self.dataset.rows[index].speaker_id].add(index)
                if len(preferred) >= self.speakers_per_batch or width == len(shard_order):
                    break
                width = min(len(shard_order), width + self.active_shard_window)
            tie_break = {speaker: rng.random() for speaker in preferred}
            selected_speakers = sorted(
                preferred, key=lambda speaker: (exposure[speaker], tie_break[speaker])
            )[:self.speakers_per_batch]
            if len(selected_speakers) != self.speakers_per_batch:
                raise RuntimeError("unable to form a complete speaker-balanced batch")
            batch: list[int] = []
            for speaker in selected_speakers:
                batch.extend(self._take_rows(speaker, queues[speaker], preferred[speaker], rng, queue_cycles))
                exposure[speaker] += 1
            if len(set(batch)) != len(batch):
                raise RuntimeError("sampler selected a duplicate row within one batch")
            rng.shuffle(batch)
            batches.append(batch)
            cursor = (cursor + self.active_shard_window) % len(shard_order)

        selections = Counter(index for batch in batches for index in batch)
        self.last_epoch_stats = {
            "batches": len(batches),
            "logical_selections": sum(map(len, batches)),
            "unique_selected_indexes": len(selections),
            "repeated_selections": sum(count - 1 for count in selections.values()),
            "queue_cycles": sum(queue_cycles.values()),
            "queue_cycles_by_speaker": dict(queue_cycles),
            "speaker_batch_exposure": dict(exposure),
            "distinct_shards_per_batch": [
                len({self.dataset.rows[index].feature_shard_path for index in batch})
                for batch in batches
            ],
        }
        yield from batches
