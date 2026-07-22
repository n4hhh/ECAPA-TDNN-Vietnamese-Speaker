#!/usr/bin/env python3
"""Analyze filename-group relationships without assigning dataset splits.

The manifest is streamed, WAV files are opened read-only, and waveform content
is loaded only for a small deterministic sample. Existing WAVs and the manifest
are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import math
import os
import re
import struct
import statistics
import sys
import uuid
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "full_manifest.csv"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
EXPECTED_MANIFEST_COLUMNS = (
    "audio_path",
    "speaker_id",
    "filename",
    "filename_group",
    "sample_rate",
    "channels",
    "num_frames",
    "duration_seconds",
    "file_size_bytes",
)
GROUPS = ("train", "train_small", "test", "part", "unknown")
KNOWN_GROUPS = GROUPS[:-1]
FILENAME_PATTERN = re.compile(
    r"^aug_(?P<group>train_small|train|test|part)"
    r"-(?P<shard>[0-9]{5})-of-(?P<shard_total>[0-9]{5})"
    r"_(?P<row>[0-9]+)\.wav$",
    flags=re.IGNORECASE,
)
LOW_UTTERANCE_COLUMNS = (
    "speaker_id",
    "total_utterances",
    "groups",
    "train_utterances",
    "train_small_utterances",
    "test_utterances",
    "part_utterances",
    "unknown_utterances",
    "exactly_1_utterance",
    "fewer_than_2",
    "fewer_than_3",
    "fewer_than_5",
    "fewer_than_10",
)
MAX_FILENAME_SAMPLES_PER_GROUP = 5
CORRELATION_EXAMPLE_THRESHOLD = 0.80
CORRELATION_EXAMPLE_LIMIT = 20
MAX_WAV_HEADER_BYTES = 1024 * 1024
MAX_PCM_SAMPLE_SPEAKERS = 256
MAX_PCM_FILES_PER_GROUP = 16


@dataclass
class SpeakerSummary:
    total: int = 0
    group_counts: Counter[str] = field(default_factory=Counter)
    coordinate_keys: dict[str, set[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    suffix_keys: dict[str, set[tuple[int, int, int]]] = field(
        default_factory=lambda: defaultdict(set)
    )


@dataclass
class GroupPatternSummary:
    files: int = 0
    strict_pattern_matches: int = 0
    speakers: set[str] = field(default_factory=set)
    shard_total_counts: Counter[int] = field(default_factory=Counter)
    shard_indices: set[int] = field(default_factory=set)
    shard_indices_by_total: dict[int, set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    shard_total_speakers: dict[int, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    coordinate_keys: set[tuple[int, int]] = field(default_factory=set)
    suffix_keys: set[tuple[int, int, int]] = field(default_factory=set)
    row_min: int | None = None
    row_max: int | None = None
    filename_samples: list[tuple[str, str]] = field(default_factory=list)

    def add_row_index(self, row_index: int) -> None:
        self.row_min = row_index if self.row_min is None else min(self.row_min, row_index)
        self.row_max = row_index if self.row_max is None else max(self.row_max, row_index)


@dataclass
class ManifestAnalysis:
    manifest_rows: int = 0
    speakers: dict[str, SpeakerSummary] = field(default_factory=dict)
    group_patterns: dict[str, GroupPatternSummary] = field(
        default_factory=lambda: {group: GroupPatternSummary() for group in GROUPS}
    )
    filenames: dict[str, set[str]] = field(
        default_factory=lambda: {group: set() for group in GROUPS}
    )
    invalid_speaker_rows: int = 0
    wav_layout_counts: Counter[tuple[str, int, int, int]] = field(
        default_factory=Counter
    )
    wav_layout_errors: int = 0
    prior_hashed_files: int | None = None
    prior_duplicate_content_groups: int | None = None
    expected_directory_layout_rows: int = 0
    unexpected_directory_layout_rows: int = 0


@dataclass
class PCMComparison:
    sampled_speakers: list[str] = field(default_factory=list)
    train_files: int = 0
    train_small_files: int = 0
    cross_group_pairs: int = 0
    exact_pcm_matches: int = 0
    maximum_zero_lag_correlation: float = 0.0
    maximum_100ms_lag_correlation: float = 0.0
    pairs_at_least_080: int = 0
    pairs_at_least_090: int = 0
    pairs_at_least_095: int = 0
    pairs_at_least_099: int = 0
    unscored_zero_energy_pairs: int = 0
    high_correlation_examples: list[tuple[float, str, str, str, float]] = field(
        default_factory=list
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the generated WAV manifest and write relationship/split "
            "recommendation reports without assigning splits."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="existing full manifest CSV",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="directory for the three generated reports",
    )
    parser.add_argument(
        "--pcm-sample-speakers",
        type=positive_int,
        default=64,
        help="shared speakers sampled for bounded PCM comparison (default: 64)",
    )
    parser.add_argument(
        "--pcm-files-per-group",
        type=positive_int,
        default=4,
        help="files retained per sampled speaker and group (default: 4)",
    )
    return parser.parse_args(argv)


def numeric_speaker_sort_key(speaker_id: str) -> tuple[int, str]:
    try:
        return int(speaker_id), speaker_id
    except ValueError:
        return sys.maxsize, speaker_id


def display_path(audio_path: str) -> str:
    path = Path(audio_path)
    return str(Path(*path.parts[-3:])) if len(path.parts) >= 3 else str(path)


def read_manifest_header(stream: Any) -> csv.DictReader:
    reader = csv.DictReader(stream)
    if tuple(reader.fieldnames or ()) != EXPECTED_MANIFEST_COLUMNS:
        raise ValueError(
            "unexpected manifest columns; expected "
            f"{list(EXPECTED_MANIFEST_COLUMNS)}, got {reader.fieldnames}"
        )
    return reader


def inspect_wav_layout(path: Path, file_size: int) -> tuple[str, int, int, int]:
    """Hash only the RIFF header through the data-chunk descriptor."""

    with path.open("rb") as stream:
        riff_header = stream.read(12)
        if (
            len(riff_header) != 12
            or riff_header[:4] != b"RIFF"
            or riff_header[8:12] != b"WAVE"
        ):
            raise ValueError("not a standard RIFF/WAVE header")
        header = bytearray(riff_header)
        while len(header) <= MAX_WAV_HEADER_BYTES:
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise ValueError("truncated before WAV data chunk")
            header.extend(chunk_header)
            chunk_id = chunk_header[:4]
            chunk_size = struct.unpack("<I", chunk_header[4:])[0]
            if chunk_id == b"data":
                return (
                    hashlib.sha256(header).hexdigest(),
                    stream.tell(),
                    chunk_size,
                    file_size,
                )
            padded_size = chunk_size + (chunk_size % 2)
            if len(header) + padded_size > MAX_WAV_HEADER_BYTES:
                raise ValueError("WAV metadata chunks exceed bounded inspection limit")
            payload = stream.read(padded_size)
            if len(payload) != padded_size:
                raise ValueError("truncated WAV metadata chunk")
            header.extend(payload)
        raise ValueError("WAV header exceeds bounded inspection limit")


def load_prior_duplicate_evidence(
    analysis: ManifestAnalysis,
    manifest_path: Path,
) -> None:
    report_path = manifest_path.parent.parent / "reports" / "dataset_split_report.txt"
    if not report_path.is_file():
        return
    report_text = report_path.read_text(encoding="utf-8")
    hashed_match = re.search(
        r"Successfully hashed files:\s*([0-9,]+)", report_text
    )
    duplicate_match = re.search(
        r"Same-size \+ same-SHA256 groups:\s*([0-9,]+)", report_text
    )
    if hashed_match:
        analysis.prior_hashed_files = int(hashed_match.group(1).replace(",", ""))
    if duplicate_match:
        analysis.prior_duplicate_content_groups = int(
            duplicate_match.group(1).replace(",", "")
        )


def analyze_manifest(manifest_path: Path) -> ManifestAnalysis:
    result = ManifestAnalysis()
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        for row in read_manifest_header(stream):
            result.manifest_rows += 1
            speaker_id = row["speaker_id"].strip()
            group = row["filename_group"].strip().casefold()
            if group not in GROUPS:
                group = "unknown"

            if speaker_id:
                speaker = result.speakers.setdefault(speaker_id, SpeakerSummary())
                speaker.total += 1
                speaker.group_counts[group] += 1
            else:
                result.invalid_speaker_rows += 1

            group_summary = result.group_patterns[group]
            group_summary.files += 1
            if speaker_id:
                group_summary.speakers.add(speaker_id)
            filename = row["filename"]
            result.filenames[group].add(filename.casefold())
            audio_path = Path(row["audio_path"])
            if (
                len(audio_path.parts) >= 3
                and audio_path.parts[-3].casefold() == "augmented_dataset"
                and audio_path.parts[-2] == speaker_id
                and audio_path.name == filename
            ):
                result.expected_directory_layout_rows += 1
            else:
                result.unexpected_directory_layout_rows += 1

            match = FILENAME_PATTERN.fullmatch(filename)
            if match and match.group("group").casefold() == group:
                group_summary.strict_pattern_matches += 1
                shard_total = int(match.group("shard_total"))
                shard_index = int(match.group("shard"))
                row_index = int(match.group("row"))
                group_summary.shard_total_counts[shard_total] += 1
                group_summary.shard_indices.add(shard_index)
                group_summary.shard_indices_by_total[shard_total].add(shard_index)
                if speaker_id:
                    group_summary.shard_total_speakers[shard_total].add(speaker_id)
                    speaker.coordinate_keys[group].add((shard_index, row_index))
                    speaker.suffix_keys[group].add(
                        (shard_index, shard_total, row_index)
                    )
                group_summary.coordinate_keys.add((shard_index, row_index))
                group_summary.suffix_keys.add(
                    (shard_index, shard_total, row_index)
                )
                group_summary.add_row_index(row_index)

            if len(group_summary.filename_samples) < MAX_FILENAME_SAMPLES_PER_GROUP:
                group_summary.filename_samples.append(
                    (speaker_id or "<invalid>", display_path(row["audio_path"]))
                )
            try:
                layout_key = inspect_wav_layout(
                    Path(row["audio_path"]), int(row["file_size_bytes"])
                )
                result.wav_layout_counts[layout_key] += 1
            except (OSError, ValueError):
                result.wav_layout_errors += 1
    load_prior_duplicate_evidence(result, manifest_path)
    return result


def deterministic_hash_sample(values: list[str], requested: int) -> list[str]:
    if len(values) <= requested:
        return values
    ranked = sorted(
        values,
        key=lambda value: hashlib.sha256(
            b"pcm-audit-speaker-v1\0" + value.encode("utf-8")
        ).digest(),
    )
    return sorted(ranked[:requested], key=numeric_speaker_sort_key)


def retain_smallest_keys(
    destination: list[tuple[str, Path]],
    candidate: tuple[str, Path],
    limit: int,
) -> None:
    destination.append(candidate)
    destination.sort(key=lambda item: item[0].casefold())
    del destination[limit:]


def collect_pcm_sample_paths(
    manifest_path: Path,
    sampled_speakers: list[str],
    files_per_group: int,
) -> dict[str, dict[str, list[Path]]]:
    selected = set(sampled_speakers)
    retained: dict[str, dict[str, list[tuple[str, Path]]]] = {
        speaker: {"train": [], "train_small": []}
        for speaker in sampled_speakers
    }
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        for row in read_manifest_header(stream):
            speaker_id = row["speaker_id"]
            group = row["filename_group"]
            if speaker_id not in selected or group not in {"train", "train_small"}:
                continue
            retain_smallest_keys(
                retained[speaker_id][group],
                (
                    hashlib.sha256(
                        b"pcm-audit-file-v1\0"
                        + row["filename"].encode("utf-8")
                    ).hexdigest(),
                    Path(row["audio_path"]),
                ),
                files_per_group,
            )
    return {
        speaker: {
            group: [path for _, path in retained[speaker][group]]
            for group in ("train", "train_small")
        }
        for speaker in sampled_speakers
    }


def raw_pcm_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as wav_file:
        while True:
            frames = wav_file.readframes(4_096)
            if not frames:
                break
            digest.update(frames)
    return digest.hexdigest()


def analyze_pcm_sample(
    sample_paths: dict[str, dict[str, list[Path]]],
) -> PCMComparison:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "numpy and soundfile are required for the bounded PCM comparison"
        ) from exc

    result = PCMComparison(sampled_speakers=list(sample_paths))
    for speaker_id, groups in sample_paths.items():
        train_paths = groups["train"]
        small_paths = groups["train_small"]
        result.train_files += len(train_paths)
        result.train_small_files += len(small_paths)
        all_paths = train_paths + small_paths
        if not all_paths:
            continue

        waveforms: dict[Path, Any] = {}
        sample_rates: dict[Path, int] = {}
        energies: dict[Path, float] = {}
        pcm_hashes: dict[Path, str] = {}
        maximum_length = 0
        for path in all_paths:
            waveform, sample_rate = sf.read(
                str(path), dtype="float32", always_2d=False
            )
            waveform = np.asarray(waveform, dtype=np.float64)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            waveform = waveform - waveform.mean() if waveform.size else waveform
            waveforms[path] = waveform
            sample_rates[path] = int(sample_rate)
            energies[path] = float(np.linalg.norm(waveform))
            pcm_hashes[path] = raw_pcm_sha256(path)
            maximum_length = max(maximum_length, int(waveform.size))

        fft_length = 1
        while fft_length < max(1, maximum_length * 2 - 1):
            fft_length <<= 1
        frequency_domain = {
            path: np.fft.rfft(waveform, n=fft_length)
            for path, waveform in waveforms.items()
        }

        for train_path in train_paths:
            for small_path in small_paths:
                result.cross_group_pairs += 1
                exact_pcm_match = pcm_hashes[train_path] == pcm_hashes[small_path]
                if exact_pcm_match:
                    result.exact_pcm_matches += 1

                train_waveform = waveforms[train_path]
                small_waveform = waveforms[small_path]
                denominator = energies[train_path] * energies[small_path]
                if denominator == 0.0:
                    if exact_pcm_match:
                        zero_lag = 1.0
                        lagged = 1.0
                    else:
                        result.unscored_zero_energy_pairs += 1
                        continue
                else:
                    common_length = min(train_waveform.size, small_waveform.size)
                    zero_lag = abs(
                        float(
                            np.dot(
                                train_waveform[:common_length],
                                small_waveform[:common_length],
                            )
                            / denominator
                        )
                    )
                    correlation = np.fft.irfft(
                        frequency_domain[train_path]
                        * np.conj(frequency_domain[small_path]),
                        n=fft_length,
                    )
                    lag_limit = min(
                        int(
                            min(
                                sample_rates[train_path],
                                sample_rates[small_path],
                            )
                            * 0.100
                        ),
                        max(0, common_length - 1),
                    )
                    lag_window = np.concatenate(
                        (
                            correlation[: lag_limit + 1],
                            correlation[-lag_limit:] if lag_limit else correlation[:0],
                        )
                    )
                    lagged = (
                        float(np.max(np.abs(lag_window)) / denominator)
                        if lag_window.size
                        else zero_lag
                    )
                    zero_lag = min(1.0, max(0.0, zero_lag))
                    lagged = min(1.0, max(0.0, lagged))

                result.maximum_zero_lag_correlation = max(
                    result.maximum_zero_lag_correlation, zero_lag
                )
                result.maximum_100ms_lag_correlation = max(
                    result.maximum_100ms_lag_correlation, lagged
                )
                result.pairs_at_least_080 += lagged >= 0.80
                result.pairs_at_least_090 += lagged >= 0.90
                result.pairs_at_least_095 += lagged >= 0.95
                result.pairs_at_least_099 += lagged >= 0.99
                if lagged >= CORRELATION_EXAMPLE_THRESHOLD:
                    example = (
                        lagged,
                        speaker_id,
                        train_path.name,
                        small_path.name,
                        zero_lag,
                    )
                    if len(result.high_correlation_examples) < CORRELATION_EXAMPLE_LIMIT:
                        heapq.heappush(result.high_correlation_examples, example)
                    elif example > result.high_correlation_examples[0]:
                        heapq.heapreplace(result.high_correlation_examples, example)

        # Bound waveform memory to one sampled speaker at a time.
        del waveforms, frequency_domain, pcm_hashes, energies

    result.high_correlation_examples.sort(reverse=True)
    return result


def threshold_counts(analysis: ManifestAnalysis) -> dict[str, int]:
    totals = [speaker.total for speaker in analysis.speakers.values()]
    return {
        "exactly_1": sum(total == 1 for total in totals),
        "fewer_than_2": sum(total < 2 for total in totals),
        "fewer_than_3": sum(total < 3 for total in totals),
        "fewer_than_5": sum(total < 5 for total in totals),
        "fewer_than_10": sum(total < 10 for total in totals),
    }


def speaker_groups(summary: SpeakerSummary) -> list[str]:
    return [group for group in GROUPS if summary.group_counts[group] > 0]


def speaker_membership_combinations(
    analysis: ManifestAnalysis,
) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(speaker_groups(summary))
        for summary in analysis.speakers.values()
    )


def pairwise_shared_speakers(
    analysis: ManifestAnalysis,
    group_a: str,
    group_b: str,
) -> list[str]:
    return sorted(
        (
            speaker_id
            for speaker_id, summary in analysis.speakers.items()
            if summary.group_counts[group_a] and summary.group_counts[group_b]
        ),
        key=numeric_speaker_sort_key,
    )


def pearson_correlation(left: list[int], right: list[int]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_energy = sum((value - left_mean) ** 2 for value in left)
    right_energy = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator else 0.0


def corpus_exact_pcm_unique(analysis: ManifestAnalysis) -> bool:
    sole_layout_covers_file = False
    if len(analysis.wav_layout_counts) == 1:
        (_, data_offset, data_size, file_size), _ = next(
            iter(analysis.wav_layout_counts.items())
        )
        sole_layout_covers_file = data_offset + data_size == file_size
    return (
        len(analysis.wav_layout_counts) == 1
        and sole_layout_covers_file
        and analysis.wav_layout_errors == 0
        and analysis.prior_hashed_files == analysis.manifest_rows
        and analysis.prior_duplicate_content_groups == 0
    )


def temporary_path(final_path: Path) -> Path:
    return final_path.with_name(
        f"{final_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )


def write_low_utterance_report(
    analysis: ManifestAnalysis,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(LOW_UTTERANCE_COLUMNS)
        rows = sorted(
            (
                (speaker_id, summary)
                for speaker_id, summary in analysis.speakers.items()
                if summary.total < 10
            ),
            key=lambda item: (item[1].total, numeric_speaker_sort_key(item[0])),
        )
        for speaker_id, summary in rows:
            groups = speaker_groups(summary)
            writer.writerow(
                (
                    speaker_id,
                    summary.total,
                    "|".join(groups),
                    *(summary.group_counts[group] for group in GROUPS),
                    int(summary.total == 1),
                    int(summary.total < 2),
                    int(summary.total < 3),
                    int(summary.total < 5),
                    int(summary.total < 10),
                )
            )


def format_counter(counter: Counter[int]) -> str:
    return ", ".join(
        f"{key}: {value:,} files" for key, value in sorted(counter.items())
    ) or "none"


def write_group_relationship_report(
    analysis: ManifestAnalysis,
    pcm: PCMComparison,
    output_path: Path,
) -> None:
    train_speakers = analysis.group_patterns["train"].speakers
    small_speakers = analysis.group_patterns["train_small"].speakers
    shared = pairwise_shared_speakers(analysis, "train", "train_small")
    normalized_small_filenames = {
        filename.replace("aug_train_small-", "aug_train-", 1)
        for filename in analysis.filenames["train_small"]
    }
    exact_filename_overlap = len(
        analysis.filenames["train"] & analysis.filenames["train_small"]
    )
    token_normalized_overlap = len(
        analysis.filenames["train"] & normalized_small_filenames
    )
    membership_combinations = speaker_membership_combinations(analysis)
    shared_train_counts = [
        analysis.speakers[speaker_id].group_counts["train"]
        for speaker_id in shared
    ]
    shared_small_counts = [
        analysis.speakers[speaker_id].group_counts["train_small"]
        for speaker_id in shared
    ]
    count_differences = [
        train_count - small_count
        for train_count, small_count in zip(
            shared_train_counts, shared_small_counts
        )
    ]
    lower_count = sum(
        train_count < small_count
        for train_count, small_count in zip(
            shared_train_counts, shared_small_counts
        )
    )
    equal_count = sum(
        train_count == small_count
        for train_count, small_count in zip(
            shared_train_counts, shared_small_counts
        )
    )
    higher_count = len(shared) - lower_count - equal_count
    small_overlap_fraction = (
        100.0 * len(shared) / len(small_speakers)
        if small_speakers
        else None
    )
    train_pattern = analysis.group_patterns["train"]
    small_pattern = analysis.group_patterns["train_small"]
    shared_coordinate_matches = sum(
        len(
            analysis.speakers[speaker_id].coordinate_keys["train"]
            & analysis.speakers[speaker_id].coordinate_keys["train_small"]
        )
        for speaker_id in shared
    )
    shared_suffix_matches = sum(
        len(
            analysis.speakers[speaker_id].suffix_keys["train"]
            & analysis.speakers[speaker_id].suffix_keys["train_small"]
        )
        for speaker_id in shared
    )

    with output_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("Filename-group relationship analysis\n")
        stream.write("=" * 36 + "\n")
        stream.write(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n")
        stream.write(f"Manifest rows analyzed: {analysis.manifest_rows:,}\n")
        stream.write("Final train/validation/test splits assigned: no\n")
        stream.write("WAV or manifest files modified: no\n")

        stream.write("\nDirectory relationships\n")
        stream.write(
            "Manifest paths matching augmented_dataset/<speaker_id>/<filename>: "
            f"{analysis.expected_directory_layout_rows:,}/{analysis.manifest_rows:,}; "
            f"nonmatching rows: {analysis.unexpected_directory_layout_rows:,}.\n"
        )
        if analysis.unexpected_directory_layout_rows == 0:
            stream.write(
                "Groups are mixed inside speaker directories rather than separated "
                "into group directories.\n"
            )
        for combination, count in sorted(
            membership_combinations.items(),
            key=lambda item: (len(item[0]), item[0]),
        ):
            stream.write(f"  {' + '.join(combination)} only: {count:,} speakers\n")

        stream.write("\nFilename patterns\n")
        stream.write(
            "Observed strict grammar: aug_<group>-<5-digit shard>-of-"
            "<5-digit shard total>_<row index>.wav\n"
        )
        for group in KNOWN_GROUPS:
            summary = analysis.group_patterns[group]
            shard_range = (
                f"{min(summary.shard_indices)}..{max(summary.shard_indices)}"
                if summary.shard_indices
                else "none"
            )
            row_range = (
                f"{summary.row_min}..{summary.row_max}"
                if summary.row_min is not None
                else "none"
            )
            stream.write(f"\n  {group}\n")
            stream.write(
                f"    Files/speakers: {summary.files:,}/{len(summary.speakers):,}\n"
            )
            stream.write(
                "    Strict-pattern matches: "
                f"{summary.strict_pattern_matches:,}/{summary.files:,}\n"
            )
            stream.write(
                f"    Shard-total values: {format_counter(summary.shard_total_counts)}\n"
            )
            for shard_total in sorted(summary.shard_total_counts):
                total_shards = summary.shard_indices_by_total[shard_total]
                total_speakers = summary.shard_total_speakers[shard_total]
                stream.write(
                    f"      of-{shard_total:05d}: "
                    f"{len(total_speakers):,} speakers, "
                    f"{len(total_shards):,} observed shard indices, "
                    f"range {min(total_shards)}..{max(total_shards)}\n"
                )
            stream.write(
                f"    Observed shard indices: {len(summary.shard_indices):,} "
                f"unique, range {shard_range}\n"
            )
            stream.write(f"    Row-index range: {row_range}\n")
            stream.write("    Bounded filename/path sample:\n")
            for speaker_id, path in summary.filename_samples:
                stream.write(f"      - speaker {speaker_id}: {path}\n")

        unknown = analysis.group_patterns["unknown"]
        stream.write(
            f"\nUnknown filename group: {unknown.files:,} files; "
            f"strict-pattern failures among known groups: "
            f"{sum(s.files - s.strict_pattern_matches for s in analysis.group_patterns.values()):,}\n"
        )

        stream.write("\nWhat the names do and do not encode\n")
        stream.write(
            "- The common aug_ prefix suggests augmentation or an augmented export, "
            "but a filename prefix alone does not prove either or name a method.\n"
        )
        stream.write(
            "- train, train_small, test, and part are source-side group labels. The "
            "names alone do not establish an authoritative ML split protocol.\n"
        )
        stream.write(
            "- shard-of-total and row values look like exporter/shard coordinates. "
            "They do not identify the source corpus, session, original utterance, "
            "noise source, speed factor, pitch factor, or other augmentation recipe.\n"
        )
        stream.write(
            "- part is semantically ambiguous and is not treated as training data.\n"
        )
        train_regime_totals = sorted(train_pattern.shard_total_speakers)
        if len(train_regime_totals) > 1:
            stream.write(
                "- train itself contains multiple shard-total regimes. Their labels "
                "must not be assumed to represent one homogeneous source.\n"
            )

        stream.write("\ntrain versus train_small\n")
        stream.write(
            f"  train: {analysis.group_patterns['train'].files:,} files / "
            f"{len(train_speakers):,} speakers\n"
        )
        stream.write(
            f"  train_small: {analysis.group_patterns['train_small'].files:,} files / "
            f"{len(small_speakers):,} speakers\n"
        )
        stream.write(f"  Shared speakers: {len(shared):,}\n")
        stream.write(
            f"  train-only speakers: {len(train_speakers - small_speakers):,}\n"
        )
        stream.write(
            f"  train_small-only speakers: {len(small_speakers - train_speakers):,}\n"
        )
        if small_overlap_fraction is None:
            stream.write("  Fraction of train_small speakers also in train: n/a\n")
        else:
            stream.write(
                "  Fraction of train_small speakers also in train: "
                f"{small_overlap_fraction:.3f}%\n"
            )
        stream.write(
            f"  Shared-speaker files: train={sum(shared_train_counts):,}, "
            f"train_small={sum(shared_small_counts):,}\n"
        )
        if count_differences:
            stream.write(
                "  Shared-speaker utterance-count relation: "
                f"train lower/equal/higher={lower_count}/{equal_count}/{higher_count}; "
                f"Pearson r={pearson_correlation(shared_train_counts, shared_small_counts):.6f}; "
                f"median(train-small)={statistics.median(count_differences):.3f}; "
                f"median absolute difference={statistics.median(abs(value) for value in count_differences):.3f}\n"
            )
        else:
            stream.write("  Shared-speaker utterance-count relation: n/a\n")
        if train_regime_totals:
            stream.write("  train shard-regime relationship to train_small:\n")
            for shard_total in train_regime_totals:
                regime_speakers = train_pattern.shard_total_speakers[shard_total]
                stream.write(
                    f"    - of-{shard_total:05d}: {len(regime_speakers):,} "
                    f"speakers; {len(regime_speakers & small_speakers):,} shared "
                    "with train_small\n"
                )
            for left_total, right_total in zip(
                train_regime_totals, train_regime_totals[1:]
            ):
                overlap = len(
                    train_pattern.shard_total_speakers[left_total]
                    & train_pattern.shard_total_speakers[right_total]
                )
                stream.write(
                    f"    - train of-{left_total:05d} / of-{right_total:05d} "
                    f"speaker overlap: {overlap:,}\n"
                )
        stream.write(f"  Exact cross-group filename matches: {exact_filename_overlap:,}\n")
        stream.write(
            "  Matches after replacing aug_train_small- with aug_train-: "
            f"{token_normalized_overlap:,}\n"
        )
        stream.write(
            "  Global exact suffix (shard,total,row) matches: "
            f"{len(train_pattern.suffix_keys & small_pattern.suffix_keys):,}\n"
        )
        stream.write(
            "  Global (shard,row) collisions when ignoring shard-total: "
            f"{len(train_pattern.coordinate_keys & small_pattern.coordinate_keys):,}\n"
        )
        stream.write(
            "  Exact suffix matches within the same shared speaker: "
            f"{shared_suffix_matches:,}\n"
        )
        stream.write(
            "  (shard,row) matches within the same shared speaker: "
            f"{shared_coordinate_matches:,}\n"
        )
        stream.write(
            "  Global coordinate collisions are non-identifying because shard "
            "namespaces differ; same-speaker correspondence is the relevant check.\n"
        )

        stream.write("\nAll-file exact-content evidence\n")
        stream.write(
            f"  WAV header/layout variants: {len(analysis.wav_layout_counts):,}\n"
        )
        stream.write(
            f"  WAV header/layout read errors: {analysis.wav_layout_errors:,}\n"
        )
        for (
            header_hash,
            data_offset,
            data_size,
            file_size,
        ), count in analysis.wav_layout_counts.most_common(10):
            stream.write(
                f"  - {count:,} files: header_sha256={header_hash}, "
                f"PCM_offset={data_offset}, PCM_bytes={data_size}, "
                f"file_bytes={file_size}\n"
            )
        if analysis.prior_hashed_files is not None:
            stream.write(
                "  Prior full-file SHA-256 pass: "
                f"{analysis.prior_hashed_files:,} files hashed\n"
            )
        if analysis.prior_duplicate_content_groups is not None:
            stream.write(
                "  Prior same-size + same-SHA256 groups: "
                f"{analysis.prior_duplicate_content_groups:,}\n"
            )
        if corpus_exact_pcm_unique(analysis):
            stream.write(
                "  Conclusion: all files have the same header/layout and every full "
                "file hash is unique, so every raw PCM payload is also unique under "
                "the standard SHA-256 collision assumption.\n"
            )
        else:
            stream.write(
                "  Conclusion: available evidence is insufficient for a corpus-wide "
                "exact-PCM uniqueness claim.\n"
            )

        stream.write("\nBounded PCM-content comparison\n")
        stream.write(
            "Method: select shared speakers and per-group files by stable SHA-256 "
            "ranking; compare raw PCM SHA-256 and normalized "
            "waveform correlation at zero lag and within +/-100 ms. Only one "
            "sampled speaker's retained waveforms are resident at once. This is "
            "diagnostic, not a full "
            "near-duplicate proof.\n"
        )
        stream.write(
            f"  Sampled shared speakers: {len(pcm.sampled_speakers):,}\n"
        )
        stream.write(f"  Sampled train files: {pcm.train_files:,}\n")
        stream.write(
            f"  Sampled train_small files: {pcm.train_small_files:,}\n"
        )
        stream.write(f"  Cross-group pairs compared: {pcm.cross_group_pairs:,}\n")
        stream.write(f"  Exact raw-PCM matches: {pcm.exact_pcm_matches:,}\n")
        stream.write(
            "  Unscored zero-energy/non-identical pairs: "
            f"{pcm.unscored_zero_energy_pairs:,}\n"
        )
        stream.write(
            "  Maximum absolute zero-lag correlation: "
            f"{pcm.maximum_zero_lag_correlation:.6f}\n"
        )
        stream.write(
            "  Maximum absolute correlation within +/-100 ms: "
            f"{pcm.maximum_100ms_lag_correlation:.6f}\n"
        )
        stream.write(f"  Pairs >= 0.80: {pcm.pairs_at_least_080:,}\n")
        stream.write(f"  Pairs >= 0.90: {pcm.pairs_at_least_090:,}\n")
        stream.write(f"  Pairs >= 0.95: {pcm.pairs_at_least_095:,}\n")
        stream.write(f"  Pairs >= 0.99: {pcm.pairs_at_least_099:,}\n")
        stream.write("  High-correlation examples (bounded):\n")
        if pcm.high_correlation_examples:
            for lagged, speaker, train_name, small_name, zero_lag in pcm.high_correlation_examples:
                stream.write(
                    f"    - speaker {speaker}: {train_name} <-> {small_name}; "
                    f"zero_lag={zero_lag:.6f}, max_100ms={lagged:.6f}\n"
                )
        else:
            stream.write("    (none)\n")

        stream.write("\nRelationship conclusion\n")
        if corpus_exact_pcm_unique(analysis):
            exact_pcm_statement = (
                "Corpus-wide header/hash evidence establishes that there are no "
                "exact PCM duplicates"
            )
        elif pcm.cross_group_pairs == 0:
            exact_pcm_statement = (
                "No bounded cross-group PCM pairs were available, and a corpus-wide "
                "claim remains unavailable"
            )
        elif pcm.exact_pcm_matches == 0:
            exact_pcm_statement = (
                "The bounded PCM sample found no exact PCM duplicates, while a "
                "corpus-wide claim remains unavailable"
            )
        else:
            exact_pcm_statement = (
                f"The bounded PCM sample found {pcm.exact_pcm_matches} exact PCM "
                "match(es), while corpus-wide coverage remains unavailable"
            )
        small_only_count = len(small_speakers - train_speakers)
        if not small_speakers:
            stream.write("No train_small files were available for a subset claim.\n")
        elif small_only_count:
            stream.write(
                "train_small is not a strict speaker subset of train because it has "
                f"{small_only_count:,} speaker IDs absent from train. "
                f"It is not a filename subset. {exact_pcm_statement}.\n"
            )
        else:
            stream.write(
                "train_small is a speaker-set subset of train, but it is not a "
                f"filename subset. {exact_pcm_statement}.\n"
            )
        if pcm.high_correlation_examples:
            correlation_statement = (
                f"{pcm.pairs_at_least_080} bounded pair(s) "
                f"reached correlation >= {CORRELATION_EXAMPLE_THRESHOLD:.2f}"
            )
        else:
            correlation_statement = (
                "no bounded example pair reached the configured high-correlation "
                "threshold"
            )
        stream.write(
            "The shared-speaker rate, common aug_ prefix, different shard namespaces, "
            f"and the fact that {correlation_statement} are consistent with partial "
            "source overlap or alternate processing/export of some recordings. They "
            "do not prove that all train_small files derive from train. Recovering "
            "source metadata or original-utterance IDs is required before treating "
            "the groups as augmentation families.\n"
        )

        stream.write("\nShared train/train_small speakers and utterance counts\n")
        stream.write("speaker_id | train | train_small | combined\n")
        for speaker_id in shared:
            counts = analysis.speakers[speaker_id].group_counts
            stream.write(
                f"{speaker_id} | {counts['train']} | {counts['train_small']} | "
                f"{counts['train'] + counts['train_small']}\n"
            )


def write_recommended_split_report(
    analysis: ManifestAnalysis,
    pcm: PCMComparison,
    output_path: Path,
) -> None:
    thresholds = threshold_counts(analysis)
    shared = pairwise_shared_speakers(analysis, "train", "train_small")
    total_speakers = len(analysis.speakers)

    with output_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("Safe split strategy recommendation (no split assigned)\n")
        stream.write("=" * 54 + "\n")
        stream.write(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n")
        stream.write(f"Manifest speakers/files: {total_speakers:,}/{analysis.manifest_rows:,}\n")
        stream.write("This report is advisory. It does not assign any file or speaker.\n")

        stream.write("\nLow-utterance speaker counts (overlapping thresholds)\n")
        stream.write(f"  Exactly 1 utterance: {thresholds['exactly_1']:,}\n")
        stream.write(f"  Fewer than 2: {thresholds['fewer_than_2']:,}\n")
        stream.write(f"  Fewer than 3: {thresholds['fewer_than_3']:,}\n")
        stream.write(f"  Fewer than 5: {thresholds['fewer_than_5']:,}\n")
        stream.write(f"  Fewer than 10: {thresholds['fewer_than_10']:,}\n")
        stream.write(
            "These speaker thresholds are based on per-speaker file counts; paths/names "
            "suggest augmentation, but files are not verified independent originals or "
            "sessions. Suspected "
            "derivatives must not be used to inflate evaluation eligibility.\n"
        )

        stream.write("\nEvidence-driven provisional handling by group\n")
        stream.write(
            "  train: candidate model-development pool only after provenance review.\n"
        )
        stream.write(
            "  train_small: its name is training-related but unverified; use only as "
            "a smoke-testing source, not an independent validation set, because "
            f"{len(shared):,} speakers overlap train.\n"
        )
        stream.write(
            "  test: quarantine as a possible official evaluation group until its "
            "protocol and augmentation status are documented. Do not train on it.\n"
        )
        stream.write(
            "  part: quarantine. Its name is ambiguous; do not assume it is training "
            "data or merge it with train.\n"
        )

        stream.write("\nRecommended safe strategy for later ECAPA-TDNN fine-tuning\n")
        stream.write(
            "1. Recover provenance first: source corpus, original utterance ID, "
            "recording session, augmentation recipe, and any official trial lists.\n"
        )
        stream.write(
            "2. Create a canonical speaker identity across all groups. Keep every file "
            "for one physical speaker in exactly one partition. The "
            f"{len(shared):,} train/train_small overlaps must never cross partitions.\n"
        )
        stream.write(
            "3. Once original_utterance_family is recovered, keep all confirmed or "
            "suspected derivative siblings together. Byte-level uniqueness does not "
            "prevent augmentation leakage; the bounded comparison found "
            f"{pcm.pairs_at_least_090} cross-group pair(s) with correlation >=0.90.\n"
        )
        stream.write(
            "4. Split by speaker, never by file. If no official protocol exists, a "
            "deterministic speaker-level development split such as 80/10/10 can be a "
            "starting point only after provenance resolution, stratified by utterance-"
            "count bins and source group. This is not an assignment made here.\n"
        )
        stream.write(
            "5. For dev/test, require at least two independent originals to form a "
            "genuine pair; prefer at least five, ideally ten and multiple sessions. "
            "Quarantine singletons and very sparse speakers from final verification "
            "metrics.\n"
        )
        stream.write(
            "6. Use dev only for early stopping, calibration, thresholds, and score-"
            "normalization decisions. Open final test once. Enrollment and probes for "
            "held-out speakers should use different original recordings/sessions.\n"
        )
        stream.write(
            "7. Prefer a clean, unaugmented, speaker-disjoint Vietnamese evaluation "
            "set. If provenance cannot be recovered, label results as augmented-corpus "
            "internal evaluation rather than clean unseen-speaker generalization.\n"
        )

        stream.write("\nSmoke-test recommendation\n")
        stream.write(
            "Use a tiny, speaker-balanced subset of train_small for data-loader and "
            "fine-tuning-loop smoke tests because its name is training-related and it "
            "is smaller than train, while acknowledging that the label is not "
            "authoritative. Select several speakers with multiple files, cap files per "
            "speaker, and discard all smoke weights, thresholds, and metrics. Keep "
            "those speakers training-only later if untouched holdouts matter. Do not "
            "use test or ambiguous part for plumbing checks.\n"
        )

        stream.write("\nDecisions still required before any final split\n")
        stream.write("- Is test an official source-dataset evaluation protocol?\n")
        stream.write("- What does part represent?\n")
        stream.write("- Are IDs globally consistent physical-speaker identities?\n")
        stream.write("- Which files share one unaugmented source recording?\n")
        stream.write("- How many independent sessions exist per speaker?\n")


def write_reports_atomically(
    analysis: ManifestAnalysis,
    pcm: PCMComparison,
    reports_dir: Path,
) -> tuple[Path, Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    final_paths = (
        reports_dir / "low_utterance_speakers.csv",
        reports_dir / "group_relationship_report.txt",
        reports_dir / "recommended_split_report.txt",
    )
    temporary_paths = tuple(temporary_path(path) for path in final_paths)
    try:
        write_low_utterance_report(analysis, temporary_paths[0])
        write_group_relationship_report(analysis, pcm, temporary_paths[1])
        write_recommended_split_report(analysis, pcm, temporary_paths[2])
        for temporary, final in zip(temporary_paths, final_paths):
            os.replace(temporary, final)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return final_paths


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve(strict=False)
    reports_dir = args.reports_dir.expanduser().resolve(strict=False)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if reports_dir == manifest_path.parent:
        # The standard layout uses sibling manifests/ and reports/. Refuse the one
        # configuration most likely to overwrite or clutter the manifest directory.
        print("ERROR: reports directory must differ from manifest directory", file=sys.stderr)
        return 2
    if args.pcm_sample_speakers > MAX_PCM_SAMPLE_SPEAKERS:
        print(
            f"ERROR: --pcm-sample-speakers cannot exceed {MAX_PCM_SAMPLE_SPEAKERS}",
            file=sys.stderr,
        )
        return 2
    if args.pcm_files_per_group > MAX_PCM_FILES_PER_GROUP:
        print(
            f"ERROR: --pcm-files-per-group cannot exceed {MAX_PCM_FILES_PER_GROUP}",
            file=sys.stderr,
        )
        return 2

    print(f"[1/4] Streaming manifest metadata: {manifest_path}", flush=True)
    try:
        analysis = analyze_manifest(manifest_path)
        shared = pairwise_shared_speakers(analysis, "train", "train_small")
        sampled_speakers = deterministic_hash_sample(
            shared, args.pcm_sample_speakers
        )
        print(
            f"[2/4] Selecting bounded PCM sample: {len(sampled_speakers)} "
            "shared speakers",
            flush=True,
        )
        sample_paths = collect_pcm_sample_paths(
            manifest_path,
            sampled_speakers,
            args.pcm_files_per_group,
        )
        print("[3/4] Comparing sampled PCM content read-only", flush=True)
        pcm = analyze_pcm_sample(sample_paths)
        print("[4/4] Writing reports", flush=True)
        paths = write_reports_atomically(analysis, pcm, reports_dir)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    thresholds = threshold_counts(analysis)
    print("Analysis complete; no split was assigned and source files were unchanged.")
    for path in paths:
        print(f"  {path}")
    print(
        "Low-utterance speakers: "
        f"exactly1={thresholds['exactly_1']}, "
        f"<3={thresholds['fewer_than_3']}, "
        f"<5={thresholds['fewer_than_5']}, "
        f"<10={thresholds['fewer_than_10']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
