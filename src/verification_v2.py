"""Deterministic VieSpeaker2.0 validation trials and baseline metrics."""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


TRIAL_FIELDS = (
    "trial_id",
    "left_audio_path",
    "right_audio_path",
    "left_speaker_id",
    "right_speaker_id",
    "target",
)


@dataclass(frozen=True)
class ValidationManifestRow:
    audio_path: str
    speaker_id: str
    duplicate_group: str


@dataclass(frozen=True)
class ValidationTrial:
    trial_id: str
    left_audio_path: str
    right_audio_path: str
    left_speaker_id: str
    right_speaker_id: str
    target: int

    @property
    def left_relative_audio_path(self) -> str:
        return self.left_audio_path

    @property
    def right_relative_audio_path(self) -> str:
        return self.right_audio_path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(*parts: object) -> str:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_validation_manifest(path: Path) -> tuple[ValidationManifestRow, ...]:
    rows: list[ValidationManifestRow] = []
    seen_paths: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "relative_audio_path",
            "speaker_id",
            "speaker_label",
            "final_split",
            "duplicate_group",
            "manifest_version",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"validation manifest is missing columns: {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            audio_path = raw["relative_audio_path"].strip()
            speaker_id = raw["speaker_id"].strip()
            pure = PurePosixPath(audio_path)
            if (
                not audio_path
                or pure.is_absolute()
                or ".." in pure.parts
                or "\\" in audio_path
                or pure.parent.name != speaker_id
            ):
                raise ValueError(f"{path}:{line_number}: invalid portable audio path")
            if (
                not speaker_id
                or raw["speaker_label"].strip() != "-1"
                or raw["final_split"].strip() != "validation"
                or raw["manifest_version"].strip() != "v2"
            ):
                raise ValueError(f"{path}:{line_number}: invalid validation metadata")
            if audio_path in seen_paths:
                raise ValueError(f"{path}:{line_number}: duplicate audio path")
            seen_paths.add(audio_path)
            rows.append(
                ValidationManifestRow(
                    audio_path=audio_path,
                    speaker_id=speaker_id,
                    duplicate_group=raw["duplicate_group"].strip(),
                )
            )
    if not rows:
        raise ValueError("validation manifest is empty")
    return tuple(rows)


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("a verification pair cannot contain the same path twice")
    return (left, right) if left < right else (right, left)


def _round_robin_pairs(speakers: Sequence[str]) -> tuple[tuple[str, str], ...]:
    if len(speakers) < 2 or len(speakers) % 2:
        raise ValueError("round-robin scheduling requires an even speaker count")
    fixed = speakers[0]
    rotating = list(speakers[1:])
    rounds: list[tuple[str, str]] = []
    for _ in range(len(speakers) - 1):
        order = [fixed, *rotating]
        for position in range(len(order) // 2):
            rounds.append((order[position], order[-1 - position]))
        rotating = [rotating[-1], *rotating[:-1]]
    return tuple(rounds)


def generate_validation_trials(
    rows: Sequence[ValidationManifestRow],
    *,
    seed: int = 20260729,
    positive_per_speaker: int = 100,
    negative_rounds: int = 200,
) -> tuple[tuple[ValidationTrial, ...], dict[str, object]]:
    """Generate fixed validation trials with hash-ranked positives and round-robin negatives."""
    if positive_per_speaker < 1 or negative_rounds < 1:
        raise ValueError("trial counts must be positive")
    by_speaker: dict[str, list[ValidationManifestRow]] = defaultdict(list)
    for row in rows:
        by_speaker[row.speaker_id].append(row)
    speakers = sorted(
        by_speaker,
        key=lambda speaker: (stable_digest(seed, "speaker", speaker), speaker),
    )
    if len(speakers) % 2:
        raise ValueError("validation speaker count must be even")

    selected: list[tuple[str, str, str, str, int]] = []
    seen_pairs: set[tuple[str, str]] = set()
    positive_counts: Counter[str] = Counter()
    positive_duplicate_rejections = 0
    for speaker in speakers:
        speaker_rows = sorted(by_speaker[speaker], key=lambda row: row.audio_path)

        def candidates() -> Iterable[tuple[str, str, str]]:
            nonlocal positive_duplicate_rejections
            for left_index, left in enumerate(speaker_rows):
                for right in speaker_rows[left_index + 1:]:
                    if (
                        left.duplicate_group
                        and left.duplicate_group == right.duplicate_group
                    ):
                        positive_duplicate_rejections += 1
                        continue
                    pair = _canonical_pair(left.audio_path, right.audio_path)
                    yield (
                        stable_digest(seed, "positive", speaker, *pair),
                        pair[0],
                        pair[1],
                    )

        ranked = heapq.nsmallest(positive_per_speaker, candidates())
        if len(ranked) != positive_per_speaker:
            raise ValueError(
                f"speaker {speaker!r} cannot supply {positive_per_speaker} "
                "duplicate-safe positive pairs"
            )
        for _, left, right in ranked:
            pair = _canonical_pair(left, right)
            if pair in seen_pairs:
                raise RuntimeError("duplicate positive path pair generated")
            seen_pairs.add(pair)
            selected.append((left, right, speaker, speaker, 1))
            positive_counts[speaker] += 1

    utterance_use: Counter[str] = Counter()
    negative_participation: Counter[str] = Counter()
    base_rounds = _round_robin_pairs(speakers)
    pairs_per_round = len(speakers) // 2
    for round_number in range(negative_rounds):
        cycle_round = round_number % (len(speakers) - 1)
        start = cycle_round * pairs_per_round
        scheduled = base_rounds[start:start + pairs_per_round]
        for speaker_a, speaker_b in scheduled:
            rows_a = sorted(
                by_speaker[speaker_a],
                key=lambda row: (
                    utterance_use[row.audio_path],
                    stable_digest(
                        seed, "negative", round_number, speaker_a, row.audio_path
                    ),
                    row.audio_path,
                ),
            )
            rows_b = sorted(
                by_speaker[speaker_b],
                key=lambda row: (
                    utterance_use[row.audio_path],
                    stable_digest(
                        seed, "negative", round_number, speaker_b, row.audio_path
                    ),
                    row.audio_path,
                ),
            )
            selected_paths: tuple[str, str] | None = None
            for left in rows_a:
                for right in rows_b:
                    pair = _canonical_pair(left.audio_path, right.audio_path)
                    if pair in seen_pairs:
                        continue
                    selected_paths = (left.audio_path, right.audio_path)
                    break
                if selected_paths is not None:
                    break
            if selected_paths is None:
                raise RuntimeError("negative schedule exhausted unique utterance pairs")
            left_path, right_path = selected_paths
            pair = _canonical_pair(left_path, right_path)
            seen_pairs.add(pair)
            selected.append(
                (left_path, right_path, speaker_a, speaker_b, 0)
            )
            utterance_use[left_path] += 1
            utterance_use[right_path] += 1
            negative_participation[speaker_a] += 1
            negative_participation[speaker_b] += 1

    trials = tuple(
        ValidationTrial(
            trial_id=f"validation-v2-{index:05d}",
            left_audio_path=left,
            right_audio_path=right,
            left_speaker_id=left_speaker,
            right_speaker_id=right_speaker,
            target=target,
        )
        for index, (left, right, left_speaker, right_speaker, target)
        in enumerate(selected)
    )
    expected_positive = len(speakers) * positive_per_speaker
    expected_negative = negative_rounds * pairs_per_round
    complete_cycles, partial_rounds = divmod(
        negative_rounds, len(speakers) - 1
    )
    validate_validation_trials(
        trials,
        expected_speakers=set(speakers),
        expected_positive=expected_positive,
        expected_negative=expected_negative,
        expected_positive_per_speaker=positive_per_speaker,
        expected_negative_participation=negative_rounds,
    )
    metadata: dict[str, object] = {
        "speaker_count": len(speakers),
        "positive_trials": expected_positive,
        "negative_trials": expected_negative,
        "positive_pairs_per_speaker": positive_per_speaker,
        "negative_rounds": negative_rounds,
        "negative_round_schedule": (
            f"{complete_cycles} complete {len(speakers) - 1}-round cycles plus first "
            f"{partial_rounds} rounds of the next cycle"
        ),
        "negative_participation_per_speaker": sorted(
            set(negative_participation.values())
        ),
        "positive_duplicate_group_pair_rejections": positive_duplicate_rejections,
        "negative_utterance_usage": {
            "minimum": min(utterance_use.values()),
            "maximum": max(utterance_use.values()),
        },
    }
    return trials, metadata


def validate_validation_trials(
    trials: Sequence[ValidationTrial],
    *,
    expected_speakers: set[str],
    expected_positive: int,
    expected_negative: int,
    expected_positive_per_speaker: int,
    expected_negative_participation: int,
) -> None:
    ids = [trial.trial_id for trial in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("trial IDs are not unique")
    pair_keys = [
        _canonical_pair(trial.left_audio_path, trial.right_audio_path)
        for trial in trials
    ]
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError("trial path pairs are not unique")
    positives = [trial for trial in trials if trial.target == 1]
    negatives = [trial for trial in trials if trial.target == 0]
    if len(positives) != expected_positive or len(negatives) != expected_negative:
        raise ValueError("trial class counts disagree with protocol")
    if any(
        trial.left_speaker_id != trial.right_speaker_id for trial in positives
    ):
        raise ValueError("positive trial contains different speakers")
    if any(
        trial.left_speaker_id == trial.right_speaker_id for trial in negatives
    ):
        raise ValueError("negative trial contains the same speaker")
    positive_counts = Counter(trial.left_speaker_id for trial in positives)
    if set(positive_counts) != expected_speakers or set(positive_counts.values()) != {
        expected_positive_per_speaker
    }:
        raise ValueError("positive speaker balance disagrees with protocol")
    participation = Counter()
    for trial in negatives:
        participation[trial.left_speaker_id] += 1
        participation[trial.right_speaker_id] += 1
    if set(participation) != expected_speakers or set(participation.values()) != {
        expected_negative_participation
    }:
        raise ValueError("negative speaker participation disagrees with protocol")


def trials_csv_bytes(trials: Sequence[ValidationTrial]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(TRIAL_FIELDS),
        lineterminator="\n",
    )
    writer.writeheader()
    for trial in trials:
        writer.writerow(asdict(trial))
    return stream.getvalue().encode("utf-8")


def read_trials_csv(path: Path) -> tuple[ValidationTrial, ...]:
    trials: list[ValidationTrial] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRIAL_FIELDS:
            raise ValueError("validation trial CSV has an unexpected schema")
        for line_number, raw in enumerate(reader, start=2):
            try:
                target = int(raw["target"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid target") from error
            if target not in (0, 1):
                raise ValueError(f"{path}:{line_number}: invalid target")
            trials.append(
                ValidationTrial(
                    trial_id=raw["trial_id"],
                    left_audio_path=raw["left_audio_path"],
                    right_audio_path=raw["right_audio_path"],
                    left_speaker_id=raw["left_speaker_id"],
                    right_speaker_id=raw["right_speaker_id"],
                    target=target,
                )
            )
    return tuple(trials)


def empirical_confusion(
    scores: Sequence[float], targets: Sequence[int], threshold: float,
) -> dict[str, float | int]:
    if len(scores) != len(targets) or not scores:
        raise ValueError("scores and targets must have equal non-zero length")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    tp = tn = fp = fn = 0
    for score, target in zip(scores, targets):
        if not math.isfinite(float(score)) or target not in (0, 1):
            raise ValueError("invalid verification score or target")
        accepted = score >= threshold
        if target == 1 and accepted:
            tp += 1
        elif target == 1:
            fn += 1
        elif accepted:
            fp += 1
        else:
            tn += 1
    positives, negatives = tp + fn, tn + fp
    total = positives + negatives
    return {
        "threshold": float(threshold),
        "threshold_semantics": "accept same speaker when score >= threshold",
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positives,
        "tpr": tp / positives,
        "specificity": tn / negatives,
        "tnr": tn / negatives,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "far": fp / negatives,
        "frr": fn / positives,
    }
