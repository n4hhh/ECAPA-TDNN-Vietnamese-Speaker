"""Deterministic metadata-only batch samplers for cached train Fbanks."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterator, Sequence

from torch.utils.data import Sampler

from src.cached_fbank_dataset import CachedFbankDataset


def validate_train_sampler_metadata(
    dataset: CachedFbankDataset,
    *,
    validate_shard_existence: bool = True,
) -> dict[str, object]:
    """Validate the train index invariants required by all sampler implementations."""
    if not isinstance(dataset, CachedFbankDataset):
        raise TypeError("dataset must be a CachedFbankDataset")
    if not isinstance(validate_shard_existence, bool):
        raise TypeError("validate_shard_existence must be a bool")
    if dataset.split != "train":
        raise ValueError("training batch samplers require the train split")
    if not dataset.rows:
        raise ValueError("train dataset is empty")

    speaker_to_label: dict[str, int] = {}
    label_to_speaker: dict[int, str] = {}
    shards: set[str] = set()
    assigned = 0
    for index, row in enumerate(dataset.rows):
        if row.final_split != "train":
            raise ValueError(f"row {index} is not in the train split")
        if row.speaker_label < 0:
            raise ValueError(f"row {index} has invalid train label {row.speaker_label}")
        previous_label = speaker_to_label.setdefault(row.speaker_id, row.speaker_label)
        if previous_label != row.speaker_label:
            raise ValueError(
                f"speaker_id {row.speaker_id!r} maps to labels "
                f"{previous_label} and {row.speaker_label}"
            )
        previous_speaker = label_to_speaker.setdefault(row.speaker_label, row.speaker_id)
        if previous_speaker != row.speaker_id:
            raise ValueError(
                f"speaker_label {row.speaker_label} maps to speakers "
                f"{previous_speaker!r} and {row.speaker_id!r}"
            )
        shards.add(row.feature_shard_path)
        assigned += 1

    expected_labels = set(range(len(speaker_to_label)))
    actual_labels = set(label_to_speaker)
    if actual_labels != expected_labels:
        raise ValueError(
            "train speaker labels must be contiguous from zero; "
            f"expected 0..{len(speaker_to_label) - 1}, got "
            f"{min(actual_labels)}..{max(actual_labels)} with gaps"
        )
    if validate_shard_existence:
        missing = []
        for relative in sorted(shards):
            path = dataset.cache_dir / Path(*PurePosixPath(relative).parts)
            if not path.is_file():
                missing.append(relative)
        if missing:
            raise FileNotFoundError(f"missing referenced train shards: {missing[:3]}")
    if assigned != len(dataset):
        raise ValueError(
            f"sampler grouping assigned {assigned} rows, expected {len(dataset)}"
        )
    return {
        "rows": len(dataset),
        "speakers": len(speaker_to_label),
        "labels": tuple(sorted(actual_labels)),
        "shards": len(shards),
        "speaker_to_label": speaker_to_label,
        "label_to_speaker": label_to_speaker,
    }


class DeterministicRandomBatchSampler(Sampler[list[int]]):
    """Full sample-level shuffle with deterministic epoch control."""

    def __init__(
        self, dataset: CachedFbankDataset, batch_size: int, *,
        seed: int = 0, drop_last: bool = False,
    ) -> None:
        validate_train_sampler_metadata(dataset)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        operation = math.floor if self.drop_last else math.ceil
        return operation(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        order = list(range(len(self.dataset)))
        random.Random(self.seed + self.epoch).shuffle(order)
        stop = len(order) if not self.drop_last else len(order) // self.batch_size * self.batch_size
        for start in range(0, stop, self.batch_size):
            yield order[start:start + self.batch_size]


class _PKBase(Sampler[list[int]]):
    def __init__(
        self, dataset: CachedFbankDataset, *, speakers_per_batch: int = 16,
        samples_per_speaker: int = 2, num_batches: int | None = None,
        seed: int = 0, validate_shard_existence: bool = True,
    ) -> None:
        validation = validate_train_sampler_metadata(
            dataset,
            validate_shard_existence=validate_shard_existence,
        )
        if speakers_per_batch < 1:
            raise ValueError("speakers_per_batch (P) must be at least 1")
        if samples_per_speaker < 1:
            raise ValueError("samples_per_speaker (K) must be at least 1")
        if speakers_per_batch > int(validation["speakers"]):
            raise ValueError("P exceeds the number of train speakers")
        if num_batches is None:
            num_batches = math.ceil(len(dataset) / (speakers_per_batch * samples_per_speaker))
        if num_batches < 1:
            raise ValueError("num_batches must be at least 1")
        self.dataset = dataset
        self.speakers_per_batch = speakers_per_batch
        self.samples_per_speaker = samples_per_speaker
        self.batch_size = speakers_per_batch * samples_per_speaker
        self.num_batches = num_batches
        self.seed = seed
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            grouped[row.speaker_id].append(index)
        too_small = sorted(
            speaker
            for speaker, indexes in grouped.items()
            if self._distinct_selection_capacity(dataset, indexes) < samples_per_speaker
        )
        if too_small:
            raise ValueError(
                f"K={samples_per_speaker} exceeds duplicate-safe distinct samples for "
                f"{len(too_small)} speakers; first={too_small[0]!r}"
            )
        self.speaker_indexes = {speaker: tuple(indexes) for speaker, indexes in grouped.items()}
        self.speakers = tuple(sorted(grouped, key=lambda value: dataset.rows[grouped[value][0]].speaker_label))
        self.last_epoch_stats: dict[str, object] = {}
        self._duplicate_group_rejections = 0

    @staticmethod
    def _distinct_selection_capacity(
        dataset: CachedFbankDataset, indexes: Sequence[int],
    ) -> int:
        empty_groups = 0
        nonempty_groups: set[str] = set()
        for index in indexes:
            duplicate_group = dataset.rows[index].duplicate_group
            if duplicate_group:
                nonempty_groups.add(duplicate_group)
            else:
                empty_groups += 1
        return empty_groups + len(nonempty_groups)

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def _queues(self, rng: random.Random) -> tuple[dict[str, list[int]], Counter[str]]:
        queues = {speaker: list(indexes) for speaker, indexes in self.speaker_indexes.items()}
        for queue in queues.values():
            rng.shuffle(queue)
        return queues, Counter()

    def _take(
        self, speaker: str, queue: list[int], count: int, rng: random.Random,
        cycles: Counter[str], preferred: set[int] | None = None,
    ) -> list[int]:
        chosen: list[int] = []
        chosen_duplicate_groups: set[str] = set()
        while len(chosen) < count:
            duplicate_safe = []
            for index in queue:
                duplicate_group = self.dataset.rows[index].duplicate_group
                if duplicate_group and duplicate_group in chosen_duplicate_groups:
                    self._duplicate_group_rejections += 1
                    continue
                duplicate_safe.append(index)
            candidates = [
                index for index in duplicate_safe
                if index not in chosen and (preferred is None or index in preferred)
            ]
            if not candidates and preferred is not None:
                candidates = [index for index in duplicate_safe if index not in chosen]
            if not candidates:
                queue[:] = list(self.speaker_indexes[speaker])
                rng.shuffle(queue)
                cycles[speaker] += 1
                continue
            index = candidates[0]
            queue.remove(index)
            chosen.append(index)
            duplicate_group = self.dataset.rows[index].duplicate_group
            if duplicate_group:
                chosen_duplicate_groups.add(duplicate_group)
        return chosen

    def _finish_stats(
        self, batches: Sequence[Sequence[int]], cycles: Counter[str],
    ) -> None:
        counts = Counter(index for batch in batches for index in batch)
        self.last_epoch_stats = {
            "queue_cycles": sum(cycles.values()),
            "queue_cycles_by_speaker": dict(cycles),
            "repeated_selections": sum(value - 1 for value in counts.values()),
            "unique_selected_indexes": len(counts),
            "duplicate_group_rejections": self._duplicate_group_rejections,
            "distinct_shards_per_batch": [
                len({self.dataset.rows[index].feature_shard_path for index in batch})
                for batch in batches
            ],
        }


class GlobalSpeakerBalancedBatchSampler(_PKBase):
    """Globally balanced P x K sampler using shuffled speaker and sample queues."""

    def __iter__(self) -> Iterator[list[int]]:
        self._duplicate_group_rejections = 0
        rng = random.Random(self.seed + self.epoch)
        queues, cycles = self._queues(rng)
        speaker_queue: list[str] = []
        batches: list[list[int]] = []
        for _ in range(self.num_batches):
            chosen_speakers: list[str] = []
            while len(chosen_speakers) < self.speakers_per_batch:
                if not speaker_queue:
                    speaker_queue = list(self.speakers)
                    rng.shuffle(speaker_queue)
                speaker = speaker_queue.pop()
                if speaker not in chosen_speakers:
                    chosen_speakers.append(speaker)
            batch: list[int] = []
            for speaker in chosen_speakers:
                batch.extend(self._take(
                    speaker, queues[speaker], self.samples_per_speaker, rng, cycles
                ))
            rng.shuffle(batch)
            batches.append(batch)
        self._finish_stats(batches, cycles)
        yield from batches


class HybridShardAwareSpeakerBatchSampler(_PKBase):
    """Speaker-balanced P x K sampler biased to a deterministic active shard window."""

    def __init__(self, dataset: CachedFbankDataset, *, active_shard_window: int = 16, **kwargs: object) -> None:
        super().__init__(dataset, **kwargs)
        shards: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            shards[row.feature_shard_path].append(index)
        if active_shard_window < 1 or active_shard_window > len(shards):
            raise ValueError("active_shard_window must be between 1 and the shard count")
        self.active_shard_window = active_shard_window
        self.shard_indexes = {path: tuple(indexes) for path, indexes in shards.items()}
        self.shards = tuple(sorted(shards))

    def __iter__(self) -> Iterator[list[int]]:
        self._duplicate_group_rejections = 0
        rng = random.Random(self.seed + self.epoch)
        shard_order = list(self.shards)
        rng.shuffle(shard_order)
        queues, cycles = self._queues(rng)
        exposure: Counter[str] = Counter()
        batches: list[list[int]] = []
        cursor = 0
        for _ in range(self.num_batches):
            width = self.active_shard_window
            eligible: dict[str, set[int]] = {}
            while True:
                active = [
                    shard_order[(cursor + offset) % len(shard_order)]
                    for offset in range(width)
                ]
                eligible = defaultdict(set)
                for shard in active:
                    for index in self.shard_indexes[shard]:
                        eligible[self.dataset.rows[index].speaker_id].add(index)
                usable = sorted(
                    speaker for speaker, indexes in eligible.items()
                    if self._distinct_selection_capacity(
                        self.dataset, tuple(indexes)
                    ) >= self.samples_per_speaker
                )
                if len(usable) >= self.speakers_per_batch:
                    break
                if width == len(shard_order):
                    raise ValueError(
                        "cannot form an exact P x K batch from train metadata"
                    )
                width = min(len(shard_order), width + self.active_shard_window)
            tie_break = {speaker: rng.random() for speaker in usable}
            chosen = sorted(usable, key=lambda speaker: (exposure[speaker], tie_break[speaker]))[
                :self.speakers_per_batch
            ]
            batch: list[int] = []
            for speaker in chosen:
                batch.extend(self._take(
                    speaker, queues[speaker], self.samples_per_speaker, rng, cycles,
                    preferred=eligible[speaker],
                ))
                exposure[speaker] += 1
            rng.shuffle(batch)
            batches.append(batch)
            cursor = (cursor + self.active_shard_window) % len(shard_order)
        self._finish_stats(batches, cycles)
        self.last_epoch_stats["speaker_batch_exposure"] = dict(exposure)
        yield from batches
