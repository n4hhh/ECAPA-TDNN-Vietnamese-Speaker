"""Fixed validation-trial generation for adaptive_augmented_3s_v1."""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


TRIAL_FIELDS = (
    "trial_id",
    "left_audio_path",
    "right_audio_path",
    "left_speaker_id",
    "right_speaker_id",
    "target",
)


@dataclass(frozen=True)
class ValidationRow:
    audio_path: str
    speaker_id: str


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("self-pairs are forbidden")
    return (left, right) if left < right else (right, left)


def read_validation_manifest(path: Path) -> tuple[ValidationRow, ...]:
    rows: list[ValidationRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != (
            "relative_audio_path", "speaker_id", "speaker_label", "final_split",
        ):
            raise ValueError("validation manifest does not have the approved portable schema")
        for line, raw in enumerate(reader, start=2):
            audio_path = raw["relative_audio_path"].strip()
            speaker_id = raw["speaker_id"].strip()
            pure = PurePosixPath(audio_path)
            if (
                not audio_path or pure.is_absolute() or ".." in pure.parts
                or "\\" in audio_path or pure.parent.name != speaker_id
                or raw["speaker_label"].strip() != "-1"
                or raw["final_split"].strip() != "validation" or audio_path in seen
            ):
                raise ValueError(f"{path}:{line}: invalid validation manifest row")
            seen.add(audio_path)
            rows.append(ValidationRow(audio_path, speaker_id))
    if not rows:
        raise ValueError("validation manifest is empty")
    return tuple(rows)


def _balanced_quotas(speakers: Sequence[str], total: int, seed: int, purpose: str) -> dict[str, int]:
    base, remainder = divmod(total, len(speakers))
    ranked = sorted(speakers, key=lambda speaker: (stable_digest(seed, purpose, speaker), speaker))
    return {speaker: base + int(index < remainder) for index, speaker in enumerate(ranked)}


def _negative_speaker_pairs(speakers: Sequence[str], total: int, seed: int) -> list[tuple[str, str]]:
    remaining = _balanced_quotas(speakers, total * 2, seed, "negative-participation")
    pairs: list[tuple[str, str]] = []
    for position in range(total):
        ranked = sorted(
            (speaker for speaker, count in remaining.items() if count),
            key=lambda speaker: (-remaining[speaker], stable_digest(seed, "negative-speaker", position, speaker), speaker),
        )
        if len(ranked) < 2:
            raise RuntimeError("unable to balance impostor speaker participation")
        left, right = ranked[:2]
        remaining[left] -= 1
        remaining[right] -= 1
        pairs.append((left, right))
    if any(remaining.values()):
        raise RuntimeError("impostor speaker participation did not reconcile")
    return pairs


def generate_validation_trials(
    rows: Sequence[ValidationRow], *, seed: int = 2026,
    genuine_count: int = 10_000, impostor_count: int = 10_000,
) -> tuple[ValidationTrial, ...]:
    """Generate balanced, hash-ranked genuine and endpoint-balanced impostor pairs."""
    if genuine_count < 1 or impostor_count < 1:
        raise ValueError("trial counts must be positive")
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_speaker[row.speaker_id].append(row.audio_path)
    speakers = tuple(sorted(by_speaker))
    if len(speakers) < 2:
        raise ValueError("at least two validation speakers are required")
    for values in by_speaker.values():
        values.sort()

    selected: list[tuple[str, str, str, str, int]] = []
    seen_pairs: set[tuple[str, str]] = set()
    genuine_quotas = _balanced_quotas(speakers, genuine_count, seed, "genuine-quota")
    for speaker in speakers:
        candidates = (
            (stable_digest(seed, "genuine", speaker, *canonical_pair(left, right)), *canonical_pair(left, right))
            for offset, left in enumerate(by_speaker[speaker])
            for right in by_speaker[speaker][offset + 1:]
        )
        ranked = heapq.nsmallest(genuine_quotas[speaker], candidates)
        if len(ranked) != genuine_quotas[speaker]:
            raise ValueError(f"speaker {speaker!r} cannot supply its genuine quota")
        for _, left, right in ranked:
            pair = canonical_pair(left, right)
            if pair in seen_pairs:
                raise RuntimeError("duplicate genuine pair")
            seen_pairs.add(pair)
            selected.append((left, right, speaker, speaker, 1))

    utterance_use: Counter[str] = Counter()
    for position, (speaker_a, speaker_b) in enumerate(_negative_speaker_pairs(speakers, impostor_count, seed)):
        paths_a = sorted(
            by_speaker[speaker_a],
            key=lambda path: (utterance_use[path], stable_digest(seed, "impostor", position, speaker_a, path), path),
        )
        paths_b = sorted(
            by_speaker[speaker_b],
            key=lambda path: (utterance_use[path], stable_digest(seed, "impostor", position, speaker_b, path), path),
        )
        chosen: tuple[str, str] | None = None
        for left in paths_a:
            for right in paths_b:
                pair = canonical_pair(left, right)
                if pair not in seen_pairs:
                    chosen = pair
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError("impostor utterance-pair capacity exhausted")
        left, right = chosen
        seen_pairs.add(chosen)
        left_speaker, right_speaker = (
            (speaker_a, speaker_b)
            if left in by_speaker[speaker_a]
            else (speaker_b, speaker_a)
        )
        selected.append((left, right, left_speaker, right_speaker, 0))
        utterance_use[left] += 1
        utterance_use[right] += 1

    trials = tuple(
        ValidationTrial(
            f"adaptive-augmented-3s-v1-validation-{index:05d}",
            left, right, left_speaker, right_speaker, target,
        )
        for index, (left, right, left_speaker, right_speaker, target) in enumerate(selected)
    )
    validate_validation_trials(trials, rows, genuine_count=genuine_count, impostor_count=impostor_count)
    return trials


def validate_validation_trials(
    trials: Sequence[ValidationTrial], rows: Sequence[ValidationRow], *,
    genuine_count: int = 10_000, impostor_count: int = 10_000,
) -> None:
    ownership = {row.audio_path: row.speaker_id for row in rows}
    if len(ownership) != len(rows) or len(trials) != genuine_count + impostor_count:
        raise ValueError("validation rows or trial count is invalid")
    trial_ids = [trial.trial_id for trial in trials]
    pair_keys = [canonical_pair(trial.left_audio_path, trial.right_audio_path) for trial in trials]
    if len(trial_ids) != len(set(trial_ids)) or len(pair_keys) != len(set(pair_keys)):
        raise ValueError("trial IDs or unordered path pairs are not unique")
    genuine = [trial for trial in trials if trial.target == 1]
    impostor = [trial for trial in trials if trial.target == 0]
    if len(genuine) != genuine_count or len(impostor) != impostor_count:
        raise ValueError("trial target counts are invalid")
    participation: Counter[str] = Counter()
    for trial in trials:
        if (
            trial.target not in (0, 1)
            or ownership.get(trial.left_audio_path) != trial.left_speaker_id
            or ownership.get(trial.right_audio_path) != trial.right_speaker_id
            or (trial.target == 1) != (trial.left_speaker_id == trial.right_speaker_id)
        ):
            raise ValueError("trial ownership or target is invalid")
        participation[trial.left_speaker_id] += 1
        participation[trial.right_speaker_id] += 1
    if set(participation) != set(row.speaker_id for row in rows):
        raise ValueError("not all validation speakers are represented")


def trials_csv_bytes(trials: Sequence[ValidationTrial]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS, lineterminator="\n")
    writer.writeheader()
    for trial in trials:
        writer.writerow(asdict(trial))
    return stream.getvalue().encode("utf-8")
