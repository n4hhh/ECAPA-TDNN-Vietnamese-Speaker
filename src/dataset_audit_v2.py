"""Deterministic, read-only VieSpeaker2.0 dataset-audit helpers.

This module audits filesystem placement, WAV container metadata, filename
provenance, candidate source identities, and exact byte duplicates.  It does
not decode waveforms, preprocess audio, assign labels, or construct splits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import struct
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


AUDIT_VERSION = 2
INVENTORY_SCHEMA = "dataset_inventory_v2_non_production"
SUMMARY_SCHEMA = "dataset_understanding_v2"
PARTIAL_HASH_BYTES = 64 * 1024

INVENTORY_FIELDS = (
    "dataset_relative_path",
    "filename",
    "extension",
    "file_size_bytes",
    "direct_parent_folder",
    "candidate_speaker_id",
    "relative_depth",
    "placement_status",
    "speaker_folder_status",
    "provenance_class",
    "provenance_parse_status",
    "candidate_source_group",
    "source_group_parse_status",
    "wav_read_status",
    "wav_error_type",
    "wav_error_message",
    "wav_encoding",
    "sample_rate_hz",
    "channel_count",
    "sample_width_bytes",
    "bits_per_sample",
    "frame_count",
    "duration_seconds",
    "zero_byte_file",
    "zero_frame_audio",
    "duration_outlier_status",
    "exact_duplicate_group",
    "audit_exclusion_recommendation",
    "audit_notes",
)

SPEAKER_FIELDS = (
    "speaker_id",
    "utterance_count",
    "valid_readable_wav_count",
    "invalid_wav_count",
    "zero_frame_count",
    "duration_seconds",
    "provenance_classes",
    "exact_duplicate_file_count",
    "candidate_source_group_count",
    "positive_pair_support_status",
)

DUPLICATE_FIELDS = (
    "exact_duplicate_group",
    "sha256",
    "file_size_bytes",
    "file_count",
    "speaker_count",
    "scope",
    "potential_repeated_bytes",
    "dataset_relative_paths",
    "candidate_speaker_ids",
)

DISTRIBUTION_FIELDS = ("metric", "value", "count")

PROVENANCE_CLASSES = ("train", "train_small", "part", "test", "other", "unparseable")

_PROVENANCE_RE = re.compile(
    r"^(?:aug_)?(?P<group>train_small|train|part|test)(?:[-_].+)?\.wav$"
)
_SOURCE_EXACT_RE = re.compile(
    r"^aug_(?P<group>train_small|train|part|test)-"
    r"(?P<shard>\d+)-of-(?P<total>\d+)_(?P<row>\d+)\.wav$"
)
_SOURCE_SUFFIX_RE = re.compile(
    r"^aug_(?P<group>train_small|train|part|test)-"
    r"(?P<shard>\d+)-of-(?P<total>\d+)_(?P<row>\d+)(?P<suffix>[_-].+)\.wav$"
)
_SOURCE_PARTIAL_RE = re.compile(
    r"^aug_(?P<group>train_small|train|part|test)-(?P<body>.+)\.wav$"
)
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True)
class WavHeader:
    wav_read_status: str
    wav_error_type: str
    wav_error_message: str
    wav_encoding: str
    sample_rate_hz: int | None
    channel_count: int | None
    sample_width_bytes: int | None
    bits_per_sample: int | None
    frame_count: int | None
    duration_seconds: float | None
    zero_frame_audio: bool


@dataclass(frozen=True)
class Snapshot:
    file_count: int
    total_bytes: int
    directory_count: int
    identity_sha256: str
    errors: tuple[str, ...]


def portable_relative(path: Path, root: Path) -> str:
    """Return a portable dataset-root-relative path or fail closed."""
    relative = path.relative_to(root)
    value = PurePosixPath(*relative.parts).as_posix()
    if not value or value == "." or value.startswith("/") or "\\" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    if PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return value


def normalized_relative_key(value: str) -> str:
    """Normalize Unicode and separators without applying case folding."""
    if "\\" in value or PurePosixPath(value).is_absolute():
        raise ValueError(f"non-portable relative path: {value!r}")
    return unicodedata.normalize("NFC", value)


def windows_collision_key(value: str) -> str:
    return normalized_relative_key(value).casefold()


def unsafe_path_reasons(value: str) -> list[str]:
    reasons: list[str] = []
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        reasons.append("non_relative_or_non_posix")
    for part in path.parts:
        if not part or part in {".", ".."}:
            reasons.append("empty_or_navigation_component")
        if any(ord(character) < 32 for character in part):
            reasons.append("control_character")
        if ":" in part:
            reasons.append("colon")
        if part.endswith((" ", ".")):
            reasons.append("windows_trailing_space_or_dot")
        if part.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            reasons.append("windows_reserved_name")
        if unicodedata.normalize("NFC", part) != part:
            reasons.append("non_nfc_unicode")
    return sorted(set(reasons))


def parse_provenance(filename: str) -> tuple[str, str]:
    """Classify source provenance without treating it as a final split."""
    if not filename or not filename.lower().endswith(".wav"):
        return "unparseable", "not_a_wav_filename"
    exact = _PROVENANCE_RE.fullmatch(filename)
    if exact:
        return exact.group("group"), "exact_case_supported"
    folded = _PROVENANCE_RE.fullmatch(filename.lower())
    if folded:
        return folded.group("group"), "case_variant_supported"
    stem = filename[:-4]
    lower = stem.lower()
    recognized_prefixes = (
        "aug_train",
        "aug_train_small",
        "aug_part",
        "aug_test",
        "train",
        "train_small",
        "part",
        "test",
    )
    if any(lower == prefix or lower.startswith(prefix) for prefix in recognized_prefixes):
        return "unparseable", "recognized_prefix_malformed"
    return "other", "unrecognized_pattern"


def parse_candidate_source_group(filename: str) -> tuple[str, str]:
    """Extract only filename-supported candidate source identity.

    The exact pattern retains provenance, shard number, declared shard total,
    and row number.  A suffix after the row makes the interpretation ambiguous
    because the filename alone does not prove that it is an augmentation index.
    """
    match = _SOURCE_EXACT_RE.fullmatch(filename)
    status = "exact_pattern_supported"
    if match is None:
        match = _SOURCE_EXACT_RE.fullmatch(filename.lower())
        status = "exact_pattern_case_variant"
    if match is not None:
        group = match.group("group")
        shard = match.group("shard")
        total = match.group("total")
        row = match.group("row")
        return f"{group}/{shard}-of-{total}/{row}", status

    match = _SOURCE_SUFFIX_RE.fullmatch(filename)
    if match is None:
        match = _SOURCE_SUFFIX_RE.fullmatch(filename.lower())
    if match is not None:
        return (
            f"{match.group('group')}/{match.group('shard')}-of-"
            f"{match.group('total')}/{match.group('row')}",
            "ambiguous_suffix",
        )

    partial = _SOURCE_PARTIAL_RE.fullmatch(filename)
    if partial is None:
        partial = _SOURCE_PARTIAL_RE.fullmatch(filename.lower())
    if partial is not None:
        return "", "partial_pattern_unsupported"
    provenance, provenance_status = parse_provenance(filename)
    if provenance == "unparseable":
        return "", "unparseable"
    if provenance in {"train", "train_small", "part", "test"}:
        return "", "unavailable"
    if provenance_status == "unrecognized_pattern":
        return "", "unavailable"
    return "", "unparseable"


def filename_pattern(filename: str) -> str:
    """Return a compact deterministic structural filename pattern."""
    pattern = re.sub(r"\d+", "{n}", filename)
    return pattern if len(pattern) <= 240 else pattern[:237] + "..."


def _encoding_name(audio_format: int, fmt_payload: bytes) -> str:
    names = {
        1: "pcm",
        3: "ieee_float",
        6: "alaw",
        7: "mulaw",
        17: "ima_adpcm",
        80: "mpeg",
    }
    if audio_format != 0xFFFE:
        return names.get(audio_format, f"format_tag_{audio_format}")
    if len(fmt_payload) >= 40:
        subtype = struct.unpack_from("<H", fmt_payload, 24)[0]
        return f"extensible_{names.get(subtype, f'subtype_{subtype}')}"
    return "extensible_unknown_subtype"


def inspect_wav_header(path: Path, file_size: int) -> WavHeader:
    """Inspect RIFF/WAVE metadata without decoding audio samples."""
    empty = WavHeader(
        "unreadable",
        "zero_byte_file",
        "file is empty",
        "",
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    if file_size == 0:
        return empty
    try:
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) < 12:
                raise ValueError("truncated_riff_header|fewer than 12 header bytes")
            container, declared_size, wave_id = struct.unpack("<4sI4s", header)
            if container != b"RIFF":
                if container in {b"RF64", b"RIFX"}:
                    raise ValueError(
                        "unsupported_container|RF64/RIFX requires a parser not used by this audit"
                    )
                raise ValueError("invalid_container|missing RIFF signature")
            if wave_id != b"WAVE":
                raise ValueError("invalid_container|missing WAVE signature")
            if declared_size + 8 > file_size:
                raise ValueError(
                    "truncated_container|RIFF declared size exceeds the physical file size"
                )

            fmt_payload: bytes | None = None
            data_bytes = 0
            while stream.tell() + 8 <= file_size:
                chunk_header = stream.read(8)
                if len(chunk_header) != 8:
                    raise ValueError("truncated_chunk_header|incomplete chunk header")
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
                chunk_start = stream.tell()
                chunk_end = chunk_start + chunk_size
                if chunk_end > file_size:
                    raise ValueError(
                        "truncated_chunk_data|declared chunk exceeds physical file size"
                    )
                if chunk_id == b"fmt " and fmt_payload is None:
                    fmt_payload = stream.read(chunk_size)
                elif chunk_id == b"data":
                    data_bytes += chunk_size
                    stream.seek(chunk_size, os.SEEK_CUR)
                else:
                    stream.seek(chunk_size, os.SEEK_CUR)
                if chunk_size % 2 and stream.tell() < file_size:
                    stream.seek(1, os.SEEK_CUR)

            if fmt_payload is None:
                raise ValueError("missing_fmt_chunk|no fmt chunk was found")
            if len(fmt_payload) < 16:
                raise ValueError("invalid_fmt_chunk|fmt chunk is shorter than 16 bytes")
            audio_format, channels, sample_rate, _, block_align, bits = struct.unpack_from(
                "<HHIIHH", fmt_payload, 0
            )
            if channels <= 0:
                raise ValueError("invalid_channel_count|channel count is not positive")
            if sample_rate <= 0:
                raise ValueError("invalid_sample_rate|sample rate is not positive")
            if block_align <= 0:
                raise ValueError("invalid_block_align|block alignment is not positive")
            if data_bytes % block_align:
                raise ValueError(
                    "partial_audio_frame|data bytes are not divisible by block alignment"
                )
            frames = data_bytes // block_align
            duration = frames / sample_rate
            width = math.ceil(bits / 8) if bits > 0 else None
            return WavHeader(
                "readable",
                "",
                "",
                _encoding_name(audio_format, fmt_payload),
                sample_rate,
                channels,
                width,
                bits,
                frames,
                duration,
                frames == 0,
            )
    except (OSError, PermissionError) as error:
        return WavHeader(
            "unreadable",
            type(error).__name__,
            str(error)[:500],
            "",
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        )
    except ValueError as error:
        kind, _, message = str(error).partition("|")
        return WavHeader(
            "unreadable",
            kind,
            message[:500],
            "",
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        )


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def descriptive_stats(values: Iterable[float | int]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {
            key: (0 if key == "count" else None)
            for key in (
                "count",
                "minimum",
                "mean",
                "standard_deviation_population",
                "p01",
                "p05",
                "p25",
                "median",
                "p75",
                "p95",
                "p99",
                "maximum",
            )
        }
    return {
        "count": len(materialized),
        "minimum": min(materialized),
        "mean": statistics.fmean(materialized),
        "standard_deviation_population": statistics.pstdev(materialized),
        "p01": percentile(materialized, 0.01),
        "p05": percentile(materialized, 0.05),
        "p25": percentile(materialized, 0.25),
        "median": percentile(materialized, 0.50),
        "p75": percentile(materialized, 0.75),
        "p95": percentile(materialized, 0.95),
        "p99": percentile(materialized, 0.99),
        "maximum": max(materialized),
    }


def histogram(values: Iterable[float], boundaries: Sequence[float]) -> list[dict[str, Any]]:
    ordered = sorted(float(value) for value in boundaries)
    counts = [0] * (len(ordered) + 1)
    for value in values:
        index = 0
        while index < len(ordered) and value >= ordered[index]:
            index += 1
        counts[index] += 1
    rows: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        if index == 0:
            label = f"<{ordered[0]:g}"
        elif index == len(ordered):
            label = f">={ordered[-1]:g}"
        else:
            label = f"[{ordered[index - 1]:g},{ordered[index]:g})"
        rows.append({"bucket": label, "count": count})
    return rows


def _snapshot_sort_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


def snapshot_dataset(root: Path) -> Snapshot:
    """Fingerprint path, size, and mtime without opening file contents."""
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    directory_count = 0
    errors: list[str] = []
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        directories.sort(key=_snapshot_sort_key)
        filenames.sort(key=_snapshot_sort_key)
        current_path = Path(current)
        directory_count += 1
        for filename in filenames:
            path = current_path / filename
            try:
                stat = path.stat()
                relative = portable_relative(path, root)
                digest.update(relative.encode("utf-8", errors="surrogatepass"))
                digest.update(b"\0")
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(b"\0")
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
                digest.update(b"\n")
                file_count += 1
                total_bytes += stat.st_size
            except (OSError, ValueError) as error:
                errors.append(f"{path.name}: {type(error).__name__}: {error}")
    return Snapshot(file_count, total_bytes, directory_count, digest.hexdigest(), tuple(errors))


def _partial_digest(path: Path, file_size: int) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", file_size))
    with path.open("rb") as stream:
        first = stream.read(PARTIAL_HASH_BYTES)
        digest.update(first)
        if file_size > PARTIAL_HASH_BYTES:
            stream.seek(max(PARTIAL_HASH_BYTES, file_size - PARTIAL_HASH_BYTES))
            digest.update(stream.read(PARTIAL_HASH_BYTES))
    return digest.hexdigest()


def _full_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_duplicate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["file_size_bytes"],
        row["wav_read_status"],
        row["wav_error_type"],
        row["wav_encoding"],
        row["sample_rate_hz"],
        row["channel_count"],
        row["sample_width_bytes"],
        row["frame_count"],
    )


def find_exact_duplicates(
    rows: list[dict[str, Any]], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hash only metadata/partial-hash candidates, then verify with SHA-256."""
    metadata_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        metadata_groups[_metadata_duplicate_key(row)].append(index)
    metadata_candidates = [
        indexes for indexes in metadata_groups.values() if len(indexes) > 1
    ]

    partial_groups: dict[tuple[tuple[Any, ...], str], list[int]] = defaultdict(list)
    hash_errors: list[dict[str, str]] = []
    partial_files = 0
    partial_bytes = 0
    for indexes in metadata_candidates:
        key = _metadata_duplicate_key(rows[indexes[0]])
        for index in indexes:
            row = rows[index]
            try:
                digest = _partial_digest(
                    root / PurePosixPath(row["dataset_relative_path"]),
                    int(row["file_size_bytes"]),
                )
                partial_files += 1
                partial_bytes += min(
                    int(row["file_size_bytes"]), 2 * PARTIAL_HASH_BYTES
                )
                partial_groups[(key, digest)].append(index)
            except OSError as error:
                hash_errors.append(
                    {
                        "dataset_relative_path": row["dataset_relative_path"],
                        "stage": "partial_hash",
                        "error": f"{type(error).__name__}: {error}"[:500],
                    }
                )

    full_groups: dict[tuple[tuple[Any, ...], str], list[int]] = defaultdict(list)
    full_files = 0
    full_bytes = 0
    for (key, _), indexes in partial_groups.items():
        if len(indexes) < 2:
            continue
        for index in indexes:
            row = rows[index]
            try:
                digest = _full_digest(
                    root / PurePosixPath(row["dataset_relative_path"])
                )
                full_files += 1
                full_bytes += int(row["file_size_bytes"])
                full_groups[(key, digest)].append(index)
            except OSError as error:
                hash_errors.append(
                    {
                        "dataset_relative_path": row["dataset_relative_path"],
                        "stage": "full_hash",
                        "error": f"{type(error).__name__}: {error}"[:500],
                    }
                )

    exact = [
        (digest, indexes)
        for (_, digest), indexes in full_groups.items()
        if len(indexes) > 1
    ]
    exact.sort(
        key=lambda item: tuple(
            rows[index]["dataset_relative_path"] for index in sorted(item[1])
        )
    )
    duplicate_rows: list[dict[str, Any]] = []
    for group_index, (digest, indexes) in enumerate(exact, start=1):
        group_id = f"dup_{group_index:06d}"
        indexes = sorted(indexes, key=lambda index: rows[index]["dataset_relative_path"])
        paths = [rows[index]["dataset_relative_path"] for index in indexes]
        speakers = sorted(
            {
                str(rows[index]["candidate_speaker_id"])
                for index in indexes
                if rows[index]["candidate_speaker_id"]
            },
            key=_snapshot_sort_key,
        )
        for index in indexes:
            rows[index]["exact_duplicate_group"] = group_id
        size = int(rows[indexes[0]]["file_size_bytes"])
        duplicate_rows.append(
            {
                "exact_duplicate_group": group_id,
                "sha256": digest,
                "file_size_bytes": size,
                "file_count": len(indexes),
                "speaker_count": len(speakers),
                "scope": "cross_speaker" if len(speakers) > 1 else "within_speaker",
                "potential_repeated_bytes": size * (len(indexes) - 1),
                "dataset_relative_paths": "|".join(paths),
                "candidate_speaker_ids": "|".join(speakers),
            }
        )
    return duplicate_rows, {
        "metadata_candidate_group_count": len(metadata_candidates),
        "metadata_candidate_file_count": sum(len(value) for value in metadata_candidates),
        "partial_hash_file_count": partial_files,
        "partial_hash_approximate_bytes_read": partial_bytes,
        "full_hash_file_count": full_files,
        "full_hash_bytes_read": full_bytes,
        "hash_errors": hash_errors,
    }


def _duration_outlier_limits(durations: Sequence[float]) -> dict[str, float | None]:
    if not durations:
        return {"lower_outer_fence": None, "upper_outer_fence": None}
    q1 = percentile(durations, 0.25)
    q3 = percentile(durations, 0.75)
    assert q1 is not None and q3 is not None
    iqr = q3 - q1
    return {
        "lower_outer_fence": max(0.0, q1 - 3.0 * iqr),
        "upper_outer_fence": q3 + 3.0 * iqr,
    }


def _classify_duration(
    duration: float | None, limits: Mapping[str, float | None]
) -> str:
    if duration is None:
        return "unavailable"
    if duration == 0:
        return "zero_frame"
    if duration < 0.1:
        return "extremely_short_under_0.1s"
    if duration > 30:
        return "extremely_long_over_30s"
    lower = limits["lower_outer_fence"]
    upper = limits["upper_outer_fence"]
    if lower is not None and duration < lower:
        return "statistical_low_outer_fence"
    if upper is not None and duration > upper:
        return "statistical_high_outer_fence"
    return "not_outlier"


def _placement(parts: Sequence[str]) -> tuple[str, str, str]:
    if len(parts) == 1:
        return "root_level_wav", "", "root_level_no_speaker"
    direct_parent = parts[-2]
    if len(parts) == 2 and direct_parent.isdecimal():
        return "expected_depth", direct_parent, "valid_numeric_top_level"
    if len(parts) == 2:
        return "expected_depth_invalid_speaker", direct_parent, "non_numeric_top_level"
    if direct_parent.isdecimal():
        return "nested_below_expected_depth", direct_parent, "nested_numeric_ambiguous"
    return "nested_below_expected_depth", direct_parent, "nested_non_numeric_ambiguous"


def _append_note(row: dict[str, Any], note: str) -> None:
    notes = [value for value in str(row.get("audit_notes", "")).split(";") if value]
    if note not in notes:
        notes.append(note)
    row["audit_notes"] = ";".join(sorted(notes))


def _speaker_sort_key(value: str) -> tuple[int, int | str, str]:
    if value.isdecimal():
        return 0, int(value), value
    return 1, value.casefold(), value


def scan_dataset(root: Path) -> dict[str, Any]:
    """Perform the complete audit and return an in-memory validated result."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    pre_snapshot = snapshot_dataset(root)
    traversal_errors: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    non_wav_files: list[dict[str, Any]] = []
    special_entries: list[dict[str, str]] = []
    directory_records: list[dict[str, Any]] = []
    all_file_relatives: list[str] = []

    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        directories.sort(key=_snapshot_sort_key)
        filenames.sort(key=_snapshot_sort_key)
        current_path = Path(current)
        relative_dir = (
            "." if current_path == root else portable_relative(current_path, root)
        )
        directory_records.append(
            {
                "relative_path": relative_dir,
                "depth": 0 if relative_dir == "." else len(PurePosixPath(relative_dir).parts),
                "direct_subdirectory_count": len(directories),
                "direct_file_count": len(filenames),
            }
        )
        for filename in filenames:
            path = current_path / filename
            try:
                relative = portable_relative(path, root)
                stat = path.stat()
            except (OSError, ValueError) as error:
                traversal_errors.append(
                    {
                        "entry": filename,
                        "error": f"{type(error).__name__}: {error}"[:500],
                    }
                )
                continue
            all_file_relatives.append(relative)
            parts = PurePosixPath(relative).parts
            extension = Path(filename).suffix
            size = stat.st_size
            if path.is_symlink():
                special_entries.append(
                    {"dataset_relative_path": relative, "type": "symbolic_link"}
                )
            if extension.lower() != ".wav":
                non_wav_files.append(
                    {
                        "dataset_relative_path": relative,
                        "filename": filename,
                        "extension": extension,
                        "file_size_bytes": size,
                        "unsafe_path_reasons": "|".join(unsafe_path_reasons(relative)),
                    }
                )
                continue

            placement, candidate, speaker_status = _placement(parts)
            provenance, provenance_status = parse_provenance(filename)
            source_group, source_status = parse_candidate_source_group(filename)
            header = inspect_wav_header(path, size)
            notes = unsafe_path_reasons(relative)
            if path.is_symlink():
                notes.append("symbolic_link")
            recommendation = "none"
            if header.wav_read_status != "readable":
                recommendation = "review_unreadable_before_future_manifest_approval"
            elif placement != "expected_depth" or speaker_status != "valid_numeric_top_level":
                recommendation = "review_structural_placement_before_future_manifest_approval"
            elif header.zero_frame_audio:
                recommendation = "review_zero_frame_before_future_manifest_approval"
            row: dict[str, Any] = {
                "dataset_relative_path": relative,
                "filename": filename,
                "extension": extension,
                "file_size_bytes": size,
                "direct_parent_folder": parts[-2] if len(parts) > 1 else "",
                "candidate_speaker_id": candidate,
                "relative_depth": len(parts),
                "placement_status": placement,
                "speaker_folder_status": speaker_status,
                "provenance_class": provenance,
                "provenance_parse_status": provenance_status,
                "candidate_source_group": source_group,
                "source_group_parse_status": source_status,
                **asdict(header),
                "zero_byte_file": size == 0,
                "duration_outlier_status": "pending",
                "exact_duplicate_group": "",
                "audit_exclusion_recommendation": recommendation,
                "audit_notes": ";".join(sorted(set(notes))),
                "_physical_path": path,
            }
            rows.append(row)

    rows.sort(key=lambda row: _snapshot_sort_key(row["dataset_relative_path"]))
    non_wav_files.sort(
        key=lambda row: _snapshot_sort_key(row["dataset_relative_path"])
    )
    directory_records.sort(key=lambda row: _snapshot_sort_key(row["relative_path"]))

    durations = [
        float(row["duration_seconds"])
        for row in rows
        if row["wav_read_status"] == "readable"
        and row["duration_seconds"] is not None
    ]
    outlier_limits = _duration_outlier_limits(durations)
    for row in rows:
        row["duration_outlier_status"] = _classify_duration(
            row["duration_seconds"], outlier_limits
        )

    normalized_groups: dict[str, list[int]] = defaultdict(list)
    windows_groups: dict[str, list[int]] = defaultdict(list)
    filename_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        normalized_groups[normalized_relative_key(row["dataset_relative_path"])].append(index)
        windows_groups[windows_collision_key(row["dataset_relative_path"])].append(index)
        filename_groups[row["filename"]].append(index)

    duplicate_normalized_paths = [
        [rows[index]["dataset_relative_path"] for index in indexes]
        for indexes in normalized_groups.values()
        if len(indexes) > 1
    ]
    case_collision_groups = [
        [rows[index]["dataset_relative_path"] for index in indexes]
        for indexes in windows_groups.values()
        if len({rows[index]["dataset_relative_path"] for index in indexes}) > 1
    ]
    for paths in case_collision_groups:
        path_set = set(paths)
        for row in rows:
            if row["dataset_relative_path"] in path_set:
                _append_note(row, "windows_case_collision_risk")

    duplicate_filename_groups: list[dict[str, Any]] = []
    for filename, indexes in filename_groups.items():
        if len(indexes) < 2:
            continue
        speakers = {
            rows[index]["candidate_speaker_id"]
            for index in indexes
            if rows[index]["candidate_speaker_id"]
        }
        duplicate_filename_groups.append(
            {
                "filename": filename,
                "file_count": len(indexes),
                "speaker_count": len(speakers),
                "scope": "cross_speaker" if len(speakers) > 1 else "within_speaker",
                "paths": [rows[index]["dataset_relative_path"] for index in indexes],
            }
        )
        note = (
            "duplicate_filename_across_speakers"
            if len(speakers) > 1
            else "duplicate_filename_within_speaker"
        )
        for index in indexes:
            _append_note(rows[index], note)
    duplicate_filename_groups.sort(
        key=lambda item: _snapshot_sort_key(item["filename"])
    )

    duplicate_rows, duplicate_method = find_exact_duplicates(rows, root)
    duplicate_members = {
        path
        for group in duplicate_rows
        for path in group["dataset_relative_paths"].split("|")
    }
    cross_duplicate_members = {
        path
        for group in duplicate_rows
        if group["scope"] == "cross_speaker"
        for path in group["dataset_relative_paths"].split("|")
    }
    for row in rows:
        if row["dataset_relative_path"] in duplicate_members:
            _append_note(row, "exact_byte_duplicate")
        if row["dataset_relative_path"] in cross_duplicate_members:
            _append_note(row, "high_priority_cross_speaker_exact_duplicate")
            if row["audit_exclusion_recommendation"] == "none":
                row["audit_exclusion_recommendation"] = (
                    "review_cross_speaker_exact_duplicate_before_future_manifest_approval"
                )

    top_level_dirs = [
        record
        for record in directory_records
        if record["depth"] == 1
    ]
    top_level_names = [record["relative_path"] for record in top_level_dirs]
    numeric_top_level = {name for name in top_level_names if name.isdecimal()}
    nonnumeric_top_level = sorted(
        (name for name in top_level_names if not name.isdecimal()),
        key=_snapshot_sort_key,
    )
    empty_directories = [
        record["relative_path"]
        for record in directory_records
        if record["relative_path"] != "."
        and record["direct_subdirectory_count"] == 0
        and record["direct_file_count"] == 0
    ]
    files_by_top_level = Counter(
        PurePosixPath(value).parts[0]
        for value in all_file_relatives
        if len(PurePosixPath(value).parts) > 1
    )
    empty_speaker_directories = sorted(
        (name for name in top_level_names if files_by_top_level[name] == 0),
        key=_snapshot_sort_key,
    )
    nested_directories = [
        record["relative_path"] for record in directory_records if record["depth"] >= 2
    ]

    speaker_rows: list[dict[str, Any]] = []
    speaker_detail: dict[str, dict[str, Any]] = {}
    for speaker in sorted(numeric_top_level, key=_speaker_sort_key):
        speaker_wavs = [
            row
            for row in rows
            if row["candidate_speaker_id"] == speaker
            and row["placement_status"] == "expected_depth"
            and row["speaker_folder_status"] == "valid_numeric_top_level"
        ]
        if not speaker_wavs:
            continue
        readable = [
            row for row in speaker_wavs if row["wav_read_status"] == "readable"
        ]
        invalid = [
            row for row in speaker_wavs if row["wav_read_status"] != "readable"
        ]
        valid_for_pairs = [row for row in readable if not row["zero_frame_audio"]]
        source_groups = {
            row["candidate_source_group"]
            for row in speaker_wavs
            if row["candidate_source_group"]
        }
        provenance = sorted(
            {row["provenance_class"] for row in speaker_wavs},
            key=lambda value: PROVENANCE_CLASSES.index(value),
        )
        detail = {
            "speaker_id": speaker,
            "utterance_count": len(speaker_wavs),
            "valid_readable_wav_count": len(readable),
            "invalid_wav_count": len(invalid),
            "zero_frame_count": sum(bool(row["zero_frame_audio"]) for row in speaker_wavs),
            "duration_seconds": sum(
                float(row["duration_seconds"] or 0.0) for row in readable
            ),
            "provenance_classes": "|".join(provenance),
            "exact_duplicate_file_count": sum(
                bool(row["exact_duplicate_group"]) for row in speaker_wavs
            ),
            "candidate_source_group_count": len(source_groups),
            "positive_pair_support_status": (
                "at_least_two_nonzero_readable"
                if len(valid_for_pairs) >= 2
                else "fewer_than_two_nonzero_readable"
            ),
        }
        speaker_rows.append(detail)
        speaker_detail[speaker] = detail

    provenance_summary: list[dict[str, Any]] = []
    for provenance in PROVENANCE_CLASSES:
        class_rows = [row for row in rows if row["provenance_class"] == provenance]
        provenance_summary.append(
            {
                "provenance_class": provenance,
                "file_count": len(class_rows),
                "speaker_count": len(
                    {
                        row["candidate_speaker_id"]
                        for row in class_rows
                        if row["candidate_speaker_id"]
                    }
                ),
                "readable_wav_count": sum(
                    row["wav_read_status"] == "readable" for row in class_rows
                ),
                "duration_seconds": sum(
                    float(row["duration_seconds"] or 0.0) for row in class_rows
                ),
            }
        )

    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["candidate_source_group"]:
            source_groups[row["candidate_source_group"]].append(row)
    source_spanning_speakers = [
        {
            "candidate_source_group": group,
            "speakers": sorted(
                {row["candidate_speaker_id"] for row in members if row["candidate_speaker_id"]},
                key=_speaker_sort_key,
            ),
            "paths": [row["dataset_relative_path"] for row in members],
        }
        for group, members in source_groups.items()
        if len({row["candidate_speaker_id"] for row in members if row["candidate_speaker_id"]}) > 1
    ]
    source_spanning_provenance = [
        {
            "candidate_source_group": group,
            "provenance_classes": sorted(
                {row["provenance_class"] for row in members}
            ),
            "paths": [row["dataset_relative_path"] for row in members],
        }
        for group, members in source_groups.items()
        if len({row["provenance_class"] for row in members}) > 1
    ]

    exact_distributions: dict[str, Counter[Any]] = {
        "sample_rate_hz": Counter(
            row["sample_rate_hz"] for row in rows if row["sample_rate_hz"] is not None
        ),
        "channel_count": Counter(
            row["channel_count"] for row in rows if row["channel_count"] is not None
        ),
        "sample_width_bytes": Counter(
            row["sample_width_bytes"]
            for row in rows
            if row["sample_width_bytes"] is not None
        ),
        "wav_encoding": Counter(
            row["wav_encoding"] for row in rows if row["wav_encoding"]
        ),
        "frame_count": Counter(
            row["frame_count"] for row in rows if row["frame_count"] is not None
        ),
        "duration_seconds": Counter(
            format(float(row["duration_seconds"]), ".12g")
            for row in rows
            if row["duration_seconds"] is not None
        ),
        "file_size_bytes": Counter(row["file_size_bytes"] for row in rows),
    }
    distribution_rows = [
        {"metric": metric, "value": str(value), "count": count}
        for metric, counter in exact_distributions.items()
        for value, count in sorted(counter.items(), key=lambda item: str(item[0]))
    ]

    expected_rows = [
        row
        for row in rows
        if row["placement_status"] == "expected_depth"
        and row["speaker_folder_status"] == "valid_numeric_top_level"
    ]
    utterance_counts = [row["utterance_count"] for row in speaker_rows]
    readable_counts = [row["valid_readable_wav_count"] for row in speaker_rows]
    duration_by_speaker = [row["duration_seconds"] for row in speaker_rows]
    pattern_counts = Counter(filename_pattern(row["filename"]) for row in rows)
    parse_status_counts = Counter(row["source_group_parse_status"] for row in rows)
    invalid_error_counts = Counter(
        row["wav_error_type"]
        for row in rows
        if row["wav_read_status"] != "readable"
    )
    invalid_speakers = sorted(
        {
            row["candidate_speaker_id"]
            for row in rows
            if row["wav_read_status"] != "readable"
            and row["candidate_speaker_id"]
        },
        key=_speaker_sort_key,
    )
    multi_provenance_speakers = [
        row["speaker_id"]
        for row in speaker_rows
        if "|" in row["provenance_classes"]
    ]
    dominant_format = None
    readable_format_counts = Counter(
        (
            row["sample_rate_hz"],
            row["channel_count"],
            row["sample_width_bytes"],
            row["wav_encoding"],
        )
        for row in rows
        if row["wav_read_status"] == "readable"
    )
    if readable_format_counts:
        dominant_format, dominant_count = readable_format_counts.most_common(1)[0]
        dominant_format = {
            "sample_rate_hz": dominant_format[0],
            "channel_count": dominant_format[1],
            "sample_width_bytes": dominant_format[2],
            "wav_encoding": dominant_format[3],
            "file_count": dominant_count,
        }
    non_dominant_format_files = []
    if dominant_format:
        dominant_tuple = (
            dominant_format["sample_rate_hz"],
            dominant_format["channel_count"],
            dominant_format["sample_width_bytes"],
            dominant_format["wav_encoding"],
        )
        non_dominant_format_files = [
            row["dataset_relative_path"]
            for row in rows
            if row["wav_read_status"] == "readable"
            and (
                row["sample_rate_hz"],
                row["channel_count"],
                row["sample_width_bytes"],
                row["wav_encoding"],
            )
            != dominant_tuple
        ]

    post_snapshot = snapshot_dataset(root)
    preservation_match = pre_snapshot == post_snapshot
    total_file_count = len(all_file_relatives)
    total_bytes = sum(
        row["file_size_bytes"] for row in rows
    ) + sum(row["file_size_bytes"] for row in non_wav_files)
    reconciliation = {
        "wav_inventory_rows_equal_wav_files": len(rows)
        == sum(Path(value).suffix.lower() == ".wav" for value in all_file_relatives),
        "wav_plus_non_wav_equal_files": len(rows) + len(non_wav_files)
        == total_file_count,
        "detail_bytes_equal_traversal_bytes": total_bytes == pre_snapshot.total_bytes,
        "speaker_summary_rows_equal_candidate_speakers": len(speaker_rows)
        == len({row["candidate_speaker_id"] for row in expected_rows}),
    }
    scan_complete = (
        not traversal_errors
        and not pre_snapshot.errors
        and not post_snapshot.errors
        and not duplicate_method["hash_errors"]
        and all(reconciliation.values())
        and preservation_match
    )

    for row in rows:
        row.pop("_physical_path", None)

    result: dict[str, Any] = {
        "schema_name": SUMMARY_SCHEMA,
        "schema_version": AUDIT_VERSION,
        "inventory_schema": INVENTORY_SCHEMA,
        "dataset_root_reference": "configured VieSpeaker2.0 augmented_dataset root",
        "result": "PASS" if scan_complete else "INCOMPLETE",
        "scan_complete": scan_complete,
        "counts": {
            "total_file_count": total_file_count,
            "wav_file_count": len(rows),
            "non_wav_file_count": len(non_wav_files),
            "total_byte_size": pre_snapshot.total_bytes,
            "speaker_folder_count": len(top_level_dirs),
            "numeric_top_level_folder_count": len(numeric_top_level),
            "valid_candidate_speaker_count": len(speaker_rows),
            "valid_readable_wav_count": sum(
                row["wav_read_status"] == "readable" for row in rows
            ),
            "invalid_unreadable_wav_count": sum(
                row["wav_read_status"] != "readable" for row in rows
            ),
            "zero_byte_wav_count": sum(bool(row["zero_byte_file"]) for row in rows),
            "zero_frame_wav_count": sum(bool(row["zero_frame_audio"]) for row in rows),
            "total_duration_seconds": sum(
                float(row["duration_seconds"] or 0.0)
                for row in rows
                if row["wav_read_status"] == "readable"
            ),
        },
        "layout": {
            "relative_depth_distribution": dict(
                sorted(Counter(len(PurePosixPath(value).parts) for value in all_file_relatives).items())
            ),
            "root_level_files": [
                value for value in all_file_relatives if len(PurePosixPath(value).parts) == 1
            ],
            "wav_below_expected_depth": [
                row["dataset_relative_path"]
                for row in rows
                if row["relative_depth"] > 2
            ],
            "nested_directories_inside_speaker_directories": nested_directories,
            "empty_directories": empty_directories,
            "empty_speaker_directories": empty_speaker_directories,
            "non_numeric_speaker_folder_names": nonnumeric_top_level,
            "non_wav_files": non_wav_files,
            "unsupported_filename_extensions": dict(
                sorted(Counter(row["extension"] or "<none>" for row in non_wav_files).items())
            ),
            "duplicate_normalized_relative_paths": duplicate_normalized_paths,
            "windows_case_collision_groups": case_collision_groups,
            "unsafe_or_nonportable_wav_paths": [
                {
                    "dataset_relative_path": row["dataset_relative_path"],
                    "audit_notes": row["audit_notes"],
                }
                for row in rows
                if any(
                    token in row["audit_notes"]
                    for token in (
                        "control_character",
                        "colon",
                        "windows_trailing",
                        "windows_reserved",
                        "non_nfc_unicode",
                        "windows_case_collision",
                        "symbolic_link",
                    )
                )
            ],
            "special_entries": special_entries,
        },
        "wav": {
            "read_status_distribution": dict(
                sorted(Counter(row["wav_read_status"] for row in rows).items())
            ),
            "invalid_error_type_distribution": dict(sorted(invalid_error_counts.items())),
            "speakers_containing_invalid_files": invalid_speakers,
            "dominant_format": dominant_format,
            "non_dominant_format_file_count": len(non_dominant_format_files),
            "non_dominant_format_files": non_dominant_format_files,
            "duration_outlier_limits": outlier_limits,
            "duration_outlier_distribution": dict(
                sorted(Counter(row["duration_outlier_status"] for row in rows).items())
            ),
            "duration_statistics_seconds": descriptive_stats(durations),
            "frame_count_statistics": descriptive_stats(
                row["frame_count"] for row in rows if row["frame_count"] is not None
            ),
            "file_size_statistics_bytes": descriptive_stats(
                row["file_size_bytes"] for row in rows
            ),
            "duration_histogram_seconds": histogram(
                durations, (0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 10, 30)
            ),
        },
        "speakers": {
            "utterance_count_statistics": descriptive_stats(utterance_counts),
            "valid_readable_utterance_count_statistics": descriptive_stats(readable_counts),
            "duration_statistics_seconds": descriptive_stats(duration_by_speaker),
            "utterance_histogram": histogram(
                utterance_counts, (2, 3, 5, 10, 20, 50, 100, 500)
            ),
            "threshold_counts": {
                "fewer_than_2": sum(value < 2 for value in utterance_counts),
                "fewer_than_3": sum(value < 3 for value in utterance_counts),
                "fewer_than_5": sum(value < 5 for value in utterance_counts),
                "fewer_than_10": sum(value < 10 for value in utterance_counts),
                "fewer_than_20": sum(value < 20 for value in utterance_counts),
                "fewer_than_50": sum(value < 50 for value in utterance_counts),
                "more_than_100": sum(value > 100 for value in utterance_counts),
                "more_than_500": sum(value > 500 for value in utterance_counts),
            },
            "speakers_without_positive_pair_support": [
                row["speaker_id"]
                for row in speaker_rows
                if row["positive_pair_support_status"]
                == "fewer_than_two_nonzero_readable"
            ],
        },
        "provenance": {
            "summary": provenance_summary,
            "parse_status_distribution": dict(
                sorted(Counter(row["provenance_parse_status"] for row in rows).items())
            ),
            "filename_pattern_variants": [
                {"pattern": pattern, "count": count}
                for pattern, count in sorted(
                    pattern_counts.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "speakers_across_multiple_classes_count": len(multi_provenance_speakers),
            "speakers_across_multiple_classes": multi_provenance_speakers,
        },
        "source_groups": {
            "parse_status_distribution": dict(sorted(parse_status_counts.items())),
            "files_with_candidate_group": sum(
                bool(row["candidate_source_group"]) for row in rows
            ),
            "percentage_with_candidate_group": (
                100.0
                * sum(bool(row["candidate_source_group"]) for row in rows)
                / len(rows)
                if rows
                else 0.0
            ),
            "candidate_group_count": len(source_groups),
            "group_size_statistics": descriptive_stats(
                len(members) for members in source_groups.values()
            ),
            "group_size_histogram": histogram(
                (len(members) for members in source_groups.values()), (2, 3, 5, 10)
            ),
            "groups_spanning_speakers": source_spanning_speakers,
            "groups_spanning_provenance": source_spanning_provenance,
        },
        "duplicates": {
            **duplicate_method,
            "duplicate_filename_group_count": len(duplicate_filename_groups),
            "duplicate_filename_groups": duplicate_filename_groups,
            "exact_duplicate_group_count": len(duplicate_rows),
            "files_in_exact_duplicate_groups": sum(
                int(row["file_count"]) for row in duplicate_rows
            ),
            "within_speaker_exact_duplicate_group_count": sum(
                row["scope"] == "within_speaker" for row in duplicate_rows
            ),
            "cross_speaker_exact_duplicate_group_count": sum(
                row["scope"] == "cross_speaker" for row in duplicate_rows
            ),
            "potential_repeated_bytes": sum(
                int(row["potential_repeated_bytes"]) for row in duplicate_rows
            ),
        },
        "preservation": {
            "pre_snapshot": asdict(pre_snapshot),
            "post_snapshot": asdict(post_snapshot),
            "snapshots_match": preservation_match,
        },
        "reconciliation": reconciliation,
        "traversal_errors": traversal_errors,
        "inventory_rows": rows,
        "speaker_rows": speaker_rows,
        "duplicate_rows": duplicate_rows,
        "distribution_rows": distribution_rows,
    }
    validate_result(result)
    return result


def validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_name") != SUMMARY_SCHEMA or result.get("schema_version") != 2:
        raise ValueError("audit summary schema is invalid")
    rows = result.get("inventory_rows")
    if not isinstance(rows, list):
        raise ValueError("inventory rows are missing")
    if len(rows) != result["counts"]["wav_file_count"]:
        raise ValueError("inventory/WAV row counts do not reconcile")
    previous = None
    for row in rows:
        if set(row) != set(INVENTORY_FIELDS):
            raise ValueError("inventory row schema is invalid")
        relative = row["dataset_relative_path"]
        if (
            not relative
            or "\\" in relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or re.match(r"^[A-Za-z]:", relative)
        ):
            raise ValueError(f"inventory contains a non-portable path: {relative!r}")
        key = _snapshot_sort_key(relative)
        if previous is not None and key < previous:
            raise ValueError("inventory ordering is non-deterministic")
        previous = key
    if len(result["speaker_rows"]) != result["counts"]["valid_candidate_speaker_count"]:
        raise ValueError("speaker summary count does not reconcile")
    if not all(result["reconciliation"].values()):
        raise ValueError("audit detail and summary counts do not reconcile")
    expected_result = "PASS" if result["scan_complete"] else "INCOMPLETE"
    if result["result"] != expected_result:
        raise ValueError("audit result does not match completeness")


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(result: Mapping[str, Any], runtime_dir: Path, report_dir: Path) -> dict[str, str]:
    """Validate and atomically write large runtime and compact report artifacts."""
    validate_result(result)
    inventory_path = runtime_dir / "dataset_audit_inventory_nonproduction_v2.csv"
    distribution_path = runtime_dir / "dataset_metadata_distributions_v2.csv"
    duplicate_path = runtime_dir / "dataset_duplicate_groups_v2.csv"
    speaker_path = report_dir / "dataset_speaker_summary_v2.csv"
    provenance_path = report_dir / "dataset_provenance_summary_v2.csv"
    summary_path = report_dir / "dataset_understanding_v2.json"

    _atomic_csv(inventory_path, INVENTORY_FIELDS, result["inventory_rows"])
    _atomic_csv(
        distribution_path, DISTRIBUTION_FIELDS, result["distribution_rows"]
    )
    _atomic_csv(duplicate_path, DUPLICATE_FIELDS, result["duplicate_rows"])
    _atomic_csv(speaker_path, SPEAKER_FIELDS, result["speaker_rows"])
    provenance_fields = (
        "provenance_class",
        "file_count",
        "speaker_count",
        "readable_wav_count",
        "duration_seconds",
    )
    _atomic_csv(provenance_path, provenance_fields, result["provenance"]["summary"])

    compact = {
        key: value
        for key, value in result.items()
        if key not in {"inventory_rows", "speaker_rows", "duplicate_rows", "distribution_rows"}
    }
    compact["artifact_paths"] = {
        "inventory": f"outputs/dataset_understanding_v2/{inventory_path.name}",
        "metadata_distributions": (
            f"outputs/dataset_understanding_v2/{distribution_path.name}"
        ),
        "duplicate_groups": f"outputs/dataset_understanding_v2/{duplicate_path.name}",
        "speaker_summary": f"reports/{speaker_path.name}",
        "provenance_summary": f"reports/{provenance_path.name}",
    }
    _atomic_json(summary_path, compact)

    # Fail closed by reading schemas/counts back before returning success.
    with inventory_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
            raise ValueError("persisted inventory schema differs")
        inventory_count = sum(1 for _ in reader)
    if inventory_count != result["counts"]["wav_file_count"]:
        raise ValueError("persisted inventory row count differs")
    with summary_path.open("r", encoding="utf-8") as stream:
        persisted = json.load(stream)
    if persisted["result"] != result["result"] or persisted["counts"] != result["counts"]:
        raise ValueError("persisted summary differs")
    return {
        "inventory": inventory_path.as_posix(),
        "metadata_distributions": distribution_path.as_posix(),
        "duplicate_groups": duplicate_path.as_posix(),
        "speaker_summary": speaker_path.as_posix(),
        "provenance_summary": provenance_path.as_posix(),
        "summary": summary_path.as_posix(),
    }
