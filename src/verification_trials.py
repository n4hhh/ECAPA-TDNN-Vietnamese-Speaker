"""Deterministic metadata-only speaker-verification trial generation."""

from __future__ import annotations

import csv
import hashlib
import io
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Mapping
from typing import Iterable, Sequence

TRIAL_FIELDS = (
    "trial_id", "target", "left_relative_audio_path", "right_relative_audio_path",
    "left_speaker_id", "right_speaker_id",
)


@dataclass(frozen=True)
class Utterance:
    relative_audio_path: str
    speaker_id: str


@dataclass(frozen=True)
class Trial:
    trial_id: int
    target: int
    left_relative_audio_path: str
    right_relative_audio_path: str
    left_speaker_id: str
    right_speaker_id: str


def validate_relative_path(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value or "\\" in value or ":" in value or pure.is_absolute() or ".." in pure.parts
        or Path(value).is_absolute()
    ):
        raise ValueError(f"unsafe relative audio path: {value!r}")
    return value


def _group(rows: Iterable[Utterance]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        path = validate_relative_path(row.relative_audio_path)
        if not row.speaker_id:
            raise ValueError("speaker_id must be non-empty")
        if path in seen:
            raise ValueError(f"duplicate audio path: {path}")
        seen.add(path)
        grouped[row.speaker_id].append(path)
    if not grouped:
        raise ValueError("no utterance metadata supplied")
    for speaker, paths in grouped.items():
        paths.sort()
        if len(paths) < 2:
            raise ValueError(f"speaker {speaker!r} has fewer than two utterances")
    return dict(sorted(grouped.items()))


def generate_trials(
    rows: Sequence[Utterance], seed: int, positive_cap_per_speaker: int = 100
) -> list[Trial]:
    if positive_cap_per_speaker < 1:
        raise ValueError("positive cap must be positive")
    grouped = _group(rows)
    rng = random.Random(seed)
    selected: list[tuple[int, str, str, str, str]] = []
    for speaker, paths in grouped.items():
        pairs = [(paths[i], paths[j]) for i in range(len(paths)) for j in range(i + 1, len(paths))]
        rng.shuffle(pairs)
        for left, right in pairs[:positive_cap_per_speaker]:
            selected.append((1, left, right, speaker, speaker))

    required = len(selected)
    speakers = list(grouped)
    participation: Counter[str] = Counter()
    negative_pairs: set[tuple[str, str]] = set()
    negatives: list[tuple[int, str, str, str, str]] = []
    attempts = 0
    while len(negatives) < required:
        attempts += 1
        if attempts > required * 1000:
            raise RuntimeError("unable to create enough unique balanced negative pairs")
        minimum = min(participation[speaker] for speaker in speakers)
        first_pool = [s for s in speakers if participation[s] == minimum]
        first = first_pool[rng.randrange(len(first_pool))]
        other_min = min(participation[s] for s in speakers if s != first)
        second_pool = [s for s in speakers if s != first and participation[s] == other_min]
        second = second_pool[rng.randrange(len(second_pool))]
        left = grouped[first][rng.randrange(len(grouped[first]))]
        right = grouped[second][rng.randrange(len(grouped[second]))]
        pair = tuple(sorted((left, right)))
        if pair in negative_pairs:
            continue
        negative_pairs.add(pair)
        if pair[0] != left:
            left, right, first, second = right, left, second, first
        negatives.append((0, left, right, first, second))
        participation[first] += 1
        participation[second] += 1

    selected.extend(negatives)
    rng.shuffle(selected)
    trials = [Trial(index, *values) for index, values in enumerate(selected)]
    validate_trials(trials, set(grouped))
    return trials


def validate_trials(trials: Sequence[Trial], expected_speakers: set[str] | None = None) -> None:
    if not trials:
        raise ValueError("trial list is empty")
    pairs: set[tuple[str, str]] = set()
    targets = Counter()
    speakers: set[str] = set()
    for index, trial in enumerate(trials):
        if trial.trial_id != index:
            raise ValueError("trial IDs must be contiguous in file order")
        left = validate_relative_path(trial.left_relative_audio_path)
        right = validate_relative_path(trial.right_relative_audio_path)
        if left == right:
            raise ValueError("self-pair detected")
        pair = tuple(sorted((left, right)))
        if pair in pairs:
            raise ValueError("duplicate unordered pair detected")
        pairs.add(pair)
        if trial.target not in (0, 1):
            raise ValueError("target must be 0 or 1")
        if (trial.left_speaker_id == trial.right_speaker_id) != bool(trial.target):
            raise ValueError("target contradicts speaker IDs")
        targets[trial.target] += 1
        speakers.update((trial.left_speaker_id, trial.right_speaker_id))
    if targets[0] != targets[1]:
        raise ValueError("positive and negative trial counts differ")
    if expected_speakers is not None and speakers != expected_speakers:
        raise ValueError("trials do not represent exactly all expected speakers")


def validate_trials_against_metadata(
    trials: Sequence[Trial],
    path_to_speaker: Mapping[str, str] | Iterable[tuple[str, str]],
    expected_speakers: set[str] | None = None,
) -> None:
    """Validate every trial path and declared speaker against authoritative metadata."""
    items = list(path_to_speaker.items()) if isinstance(path_to_speaker, Mapping) else list(path_to_speaker)
    metadata: dict[str, str] = {}
    for path, speaker in items:
        path = validate_relative_path(path)
        if path in metadata:
            raise ValueError(f"duplicate metadata path: {path}")
        if not speaker:
            raise ValueError(f"empty metadata speaker for path: {path}")
        metadata[path] = speaker
    if not metadata:
        raise ValueError("validation metadata is empty")

    represented: set[str] = set()
    for trial in trials:
        owners: list[str] = []
        for side, path, declared in (
            ("left", trial.left_relative_audio_path, trial.left_speaker_id),
            ("right", trial.right_relative_audio_path, trial.right_speaker_id),
        ):
            if path not in metadata:
                raise ValueError(f"trial {trial.trial_id}: missing {side} path {path!r}")
            owner = metadata[path]
            if owner != declared:
                raise ValueError(
                    f"trial {trial.trial_id}: {side} path {path!r} belongs to "
                    f"{owner!r}, not declared speaker {declared!r}"
                )
            owners.append(owner)
            represented.add(owner)
        if (owners[0] == owners[1]) != bool(trial.target):
            raise ValueError(
                f"trial {trial.trial_id}: target {trial.target} contradicts real "
                f"path owners for {trial.left_relative_audio_path!r} and "
                f"{trial.right_relative_audio_path!r}"
            )
    if expected_speakers is not None and represented != expected_speakers:
        raise ValueError("trials do not represent exactly all expected metadata speakers")


def trials_csv_text(trials: Sequence[Trial]) -> str:
    validate_trials(trials)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS, lineterminator="\n")
    writer.writeheader()
    for trial in trials:
        writer.writerow(trial.__dict__)
    return stream.getvalue()


def read_trials(path: str | Path) -> list[Trial]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRIAL_FIELDS:
            raise ValueError("unexpected trial CSV schema")
        trials = [Trial(
            int(row["trial_id"]), int(row["target"]), row["left_relative_audio_path"],
            row["right_relative_audio_path"], row["left_speaker_id"], row["right_speaker_id"],
        ) for row in reader]
    validate_trials(trials)
    return trials


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
