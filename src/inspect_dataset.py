#!/usr/bin/env python3
"""Read-only structural inspection for the Vietnamese speaker dataset.

The script deliberately does not decode audio samples or load a complete Parquet
dataset. It streams the directory walk, reads one byte per file to check basic
readability, uses audio headers for metadata, and reads Parquet in bounded
batches.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import stat
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DEFAULT_DATASET_ROOT = Path(r"E:\VieSpeaker")
PARQUET_EXTENSIONS = {".parquet", ".pq"}
AUDIO_EXTENSIONS = {
    ".aif",
    ".aiff",
    ".au",
    ".flac",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".wave",
}
ARCHIVE_EXTENSIONS = {
    ".7z",
    ".bz2",
    ".gz",
    ".rar",
    ".tar",
    ".tbz2",
    ".tgz",
    ".txz",
    ".xz",
    ".zip",
}

SPEAKER_COLUMN_CANDIDATES = (
    "speaker_id",
    "speaker",
    "speakerid",
    "spk_id",
    "spkid",
    "client_id",
    "clientid",
    "user_id",
    "userid",
)
SPLIT_COLUMN_CANDIDATES = (
    "split",
    "subset",
    "partition",
    "dataset_split",
    "set",
)
AUDIO_COLUMN_CANDIDATES = (
    "audio",
    "speech",
    "waveform",
    "audio_data",
    "audio_bytes",
)

SPLIT_PATTERN = re.compile(
    r"(?<![a-z0-9])"
    r"(train_small|training|train|validation|valid|val|dev|testing|test|evaluation|eval)"
    r"(?![a-z0-9])",
    flags=re.IGNORECASE,
)
SPLIT_NORMALIZATION = {
    "train_small": "train_small",
    "training": "train",
    "train": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "dev": "dev",
    "testing": "test",
    "test": "test",
    "evaluation": "eval",
    "eval": "eval",
}
FILENAME_GROUP_PATTERN = re.compile(
    r"^(?:aug_)?(?P<label>.+?)-\d{5}-of-\d{5}(?:_|$)",
    flags=re.IGNORECASE,
)

MAX_TOP_LEVEL_EXAMPLES = 30
MAX_ISSUE_EXAMPLES = 30
MAX_UNIQUE_PARQUET_VALUES = 1_000_000


@dataclass
class IssueCollector:
    """Count all issues while retaining only a bounded number of examples."""

    counts: Counter[str] = field(default_factory=Counter)
    examples: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, category: str, path: Path, detail: str) -> None:
        self.counts[category] += 1
        if len(self.examples) < MAX_ISSUE_EXAMPLES:
            self.examples.append((category, str(path), detail))

    def count(self, category: str) -> int:
        return self.counts[category]


@dataclass
class ScanSummary:
    root: Path
    file_count: int = 0
    directory_count: int = 0
    total_size_bytes: int = 0
    extensions: Counter[str] = field(default_factory=Counter)
    top_level_count: int = 0
    top_level_examples: list[str] = field(default_factory=list)
    symlink_count: int = 0
    other_entry_count: int = 0
    zero_byte_count: int = 0
    audio_count: int = 0
    parquet_count: int = 0
    archive_count: int = 0
    audio_parent_counts: Counter[str] = field(default_factory=Counter)
    inferred_split_counts: Counter[str] = field(default_factory=Counter)
    filename_group_counts: Counter[str] = field(default_factory=Counter)
    audio_candidates: list[Path] = field(default_factory=list)
    _audio_seen: int = 0
    _candidate_capacity: int = 50
    _random: random.Random = field(default_factory=lambda: random.Random(20260722))

    def consider_audio_candidate(self, path: Path) -> None:
        """Keep a deterministic reservoir instead of every audio path."""

        self._audio_seen += 1
        if len(self.audio_candidates) < self._candidate_capacity:
            self.audio_candidates.append(path)
            return

        replacement_index = self._random.randrange(self._audio_seen)
        if replacement_index < self._candidate_capacity:
            self.audio_candidates[replacement_index] = path


@dataclass
class AudioMetadata:
    path: Path
    sample_rate: int
    channels: int
    duration_seconds: float
    format_name: str
    subtype: str
    frames: int


@dataclass
class ParquetSummary:
    files_opened: int = 0
    files_failed: int = 0
    union_columns: set[str] = field(default_factory=set)
    schema_variants: Counter[str] = field(default_factory=Counter)
    representative_path: Path | None = None
    representative_schema: str | None = None
    representative_rows: list[dict[str, Any]] = field(default_factory=list)
    audio_column: str | None = None
    audio_source_path: Path | None = None
    audio_arrow_type: str | None = None
    audio_value_structure: str | None = None
    speaker_columns: set[str] = field(default_factory=set)
    split_columns: set[str] = field(default_factory=set)
    speaker_values: set[str] = field(default_factory=set)
    split_values: set[str] = field(default_factory=set)
    speaker_null_count: int = 0
    split_null_count: int = 0
    speaker_values_truncated: bool = False
    split_values_truncated: bool = False


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect dataset structure and metadata without modifying data."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=r"dataset root (default: E:\VieSpeaker)",
    )
    parser.add_argument(
        "--audio-samples",
        type=positive_int,
        default=10,
        help="number of normal audio files to inspect (default: 10)",
    )
    parser.add_argument(
        "--parquet-rows",
        type=positive_int,
        default=5,
        help="total rows to inspect across readable Parquet files (default: 5)",
    )
    return parser.parse_args(argv)


def human_size(number_of_bytes: int) -> str:
    value = float(number_of_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{number_of_bytes} B"


def extension_label(path: Path) -> str:
    return path.suffix.lower() if path.suffix else "[no extension]"


def relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def split_tokens_for_path(path: Path, root: Path) -> set[str]:
    relative_text = relative_display(path, root)
    return {
        SPLIT_NORMALIZATION[match.lower()]
        for match in SPLIT_PATTERN.findall(relative_text)
    }


def filename_group(path: Path) -> str | None:
    """Extract shard labels such as train_small from common export filenames."""

    match = FILENAME_GROUP_PATTERN.match(path.stem)
    return match.group("label").casefold() if match else None


def check_file_readability(path: Path, issues: IssueCollector) -> None:
    try:
        with path.open("rb") as stream:
            stream.read(1)
    except FileNotFoundError as exc:
        issues.add("missing_file", path, str(exc))
    except OSError as exc:
        issues.add("unreadable_file", path, str(exc))


def entry_is_link_or_reparse_point(entry: os.DirEntry[str]) -> bool:
    """Identify links/junctions so traversal cannot escape the requested root."""

    if entry.is_symlink():
        return True
    if os.name != "nt":
        return False
    attributes = getattr(
        entry.stat(follow_symlinks=False), "st_file_attributes", 0
    )
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def scan_dataset(
    root: Path,
    issues: IssueCollector,
    audio_candidate_capacity: int,
) -> ScanSummary:
    """Walk the tree without materializing its file listing."""

    summary = ScanSummary(root=root, _candidate_capacity=audio_candidate_capacity)
    pending_directories = [root]

    while pending_directories:
        directory = pending_directories.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    is_top_level = directory == root
                    if is_top_level:
                        summary.top_level_count += 1
                        if len(summary.top_level_examples) < MAX_TOP_LEVEL_EXAMPLES:
                            summary.top_level_examples.append(entry.name)

                    try:
                        if entry_is_link_or_reparse_point(entry):
                            summary.symlink_count += 1
                            if not path.exists():
                                issues.add(
                                    "missing_file",
                                    path,
                                    "broken link/reparse point or unavailable target",
                                )
                            # Never follow a link or Windows reparse point. It could form a
                            # cycle or point outside the requested dataset root.
                            continue

                        if entry.is_dir(follow_symlinks=False):
                            summary.directory_count += 1
                            pending_directories.append(path)
                            continue
                        is_file = entry.is_file(follow_symlinks=False)

                        if not is_file:
                            summary.other_entry_count += 1
                            continue

                        try:
                            stat_result = entry.stat(follow_symlinks=False)
                        except FileNotFoundError as exc:
                            issues.add("missing_file", path, str(exc))
                            continue
                        except OSError as exc:
                            issues.add("unreadable_file", path, f"cannot stat: {exc}")
                            continue

                        size = stat_result.st_size
                        suffix = extension_label(path)
                        summary.file_count += 1
                        summary.total_size_bytes += size
                        summary.extensions[suffix] += 1
                        if size == 0:
                            summary.zero_byte_count += 1

                        check_file_readability(path, issues)

                        if suffix in PARQUET_EXTENSIONS:
                            summary.parquet_count += 1
                        if suffix in ARCHIVE_EXTENSIONS:
                            summary.archive_count += 1
                        if suffix in AUDIO_EXTENSIONS:
                            summary.audio_count += 1
                            relative_parent = relative_display(path.parent, root)
                            summary.audio_parent_counts[relative_parent] += 1
                            for split_name in split_tokens_for_path(path, root):
                                summary.inferred_split_counts[split_name] += 1
                            group_name = filename_group(path)
                            if group_name:
                                summary.filename_group_counts[group_name] += 1
                            summary.consider_audio_candidate(path)
                    except OSError as exc:
                        issues.add("unreadable_file", path, str(exc))
        except FileNotFoundError as exc:
            issues.add("missing_file", directory, str(exc))
        except OSError as exc:
            issues.add("traversal_error", directory, str(exc))

    return summary


def iter_files_with_extensions(root: Path, extensions: set[str]) -> Iterator[Path]:
    """Perform a second, streaming walk for a small class of files."""

    pending_directories = [root]
    while pending_directories:
        directory = pending_directories.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    try:
                        if entry_is_link_or_reparse_point(entry):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending_directories.append(path)
                        elif (
                            entry.is_file(follow_symlinks=False)
                            and extension_label(path) in extensions
                        ):
                            yield path
                    except OSError:
                        continue
        except OSError:
            continue


def detect_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    by_normalized_name = {column.casefold(): column for column in columns}
    for candidate in candidates:
        if candidate.casefold() in by_normalized_name:
            return by_normalized_name[candidate.casefold()]
    return None


def safe_scalar_key(value: Any) -> str:
    sanitized = sanitize_value(value)
    if isinstance(sanitized, str):
        return sanitized
    return json.dumps(sanitized, ensure_ascii=False, sort_keys=True)


def sanitize_value(value: Any, depth: int = 0) -> Any:
    """Make values printable while guaranteeing that binary payloads stay hidden."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary: {len(value)} bytes>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        if len(value) <= 200:
            return value
        return value[:197] + "..."
    if depth >= 4:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 25:
                sanitized["..."] = f"{len(value) - 25} more fields"
                break
            safe_key = sanitize_value(key, depth + 1)
            key_text = (
                safe_key
                if isinstance(safe_key, str)
                else json.dumps(safe_key, ensure_ascii=False, sort_keys=True)
            )
            sanitized[key_text] = sanitize_value(item, depth + 1)
        return sanitized
    if isinstance(value, (list, tuple)):
        items = [sanitize_value(item, depth + 1) for item in value[:10]]
        if len(value) > 10:
            items.append(f"<{len(value) - 10} more items>")
        return items
    return str(value)


def describe_value_structure(value: Any, depth: int = 0) -> str:
    if value is None:
        return "null"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"binary({len(value)} bytes; content hidden)"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        if depth >= 3:
            return "struct{...}"
        rendered_members = []
        for key, item in list(value.items())[:20]:
            safe_key = sanitize_value(key, depth + 1)
            key_text = (
                safe_key
                if isinstance(safe_key, str)
                else json.dumps(safe_key, ensure_ascii=False, sort_keys=True)
            )
            rendered_members.append(
                f"{key_text}: {describe_value_structure(item, depth + 1)}"
            )
        members = ", ".join(rendered_members)
        if len(value) > 20:
            members += ", ..."
        return f"struct{{{members}}}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "list[empty]"
        return f"list[{describe_value_structure(value[0], depth + 1)}] (length {len(value)})"
    return type(value).__name__


def audio_value_has_embedded_bytes(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value) > 0
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in {"bytes", "audio_bytes", "data"} and isinstance(
                item, (bytes, bytearray, memoryview)
            ):
                return len(item) > 0
    return False


def audio_path_references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, dict):
        return []
    references: list[str] = []
    for key, item in value.items():
        if key.casefold() in {"path", "file", "filepath", "file_path", "filename"}:
            if isinstance(item, str) and item.strip():
                references.append(item)
    return references


def path_is_within_root(candidate: Path, root: Path) -> bool:
    """Check lexical and resolved boundaries before probing a referenced path."""

    root_absolute = Path(os.path.abspath(root))
    candidate_absolute = Path(os.path.abspath(candidate))
    try:
        lexical_common = os.path.commonpath((root_absolute, candidate_absolute))
    except ValueError:
        return False
    if os.path.normcase(lexical_common) != os.path.normcase(str(root_absolute)):
        return False

    try:
        resolved_root = root_absolute.resolve(strict=False)
        resolved_candidate = candidate_absolute.resolve(strict=False)
        resolved_common = os.path.commonpath((resolved_root, resolved_candidate))
    except (OSError, ValueError):
        return False
    return os.path.normcase(resolved_common) == os.path.normcase(str(resolved_root))


def reference_exists(reference: str, parquet_path: Path, root: Path) -> bool:
    raw_candidate = Path(reference)
    candidates = (
        [raw_candidate]
        if raw_candidate.is_absolute()
        else [parquet_path.parent / raw_candidate, root / raw_candidate]
    )
    for candidate in candidates:
        if path_is_within_root(candidate, root) and candidate.exists():
            return True
    return False


def add_unique_value(
    destination: set[str],
    value: Any,
    already_truncated: bool,
) -> bool:
    if value is None or already_truncated:
        return already_truncated
    if len(destination) >= MAX_UNIQUE_PARQUET_VALUES:
        return True
    destination.add(safe_scalar_key(value))
    return False


def read_representative_rows(parquet_file: Any, row_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in parquet_file.iter_batches(batch_size=row_count):
        rows.extend(batch.to_pylist())
        if len(rows) >= row_count:
            break
    return rows[:row_count]


def inspect_parquet(
    root: Path,
    expected_file_count: int,
    row_count: int,
    issues: IssueCollector,
) -> ParquetSummary | None:
    if expected_file_count == 0:
        return None

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        issues.add("dependency_error", root, f"pyarrow is required: {exc}")
        return None

    result = ParquetSummary()
    first_readable_path: Path | None = None
    first_readable_schema: str | None = None
    for path in iter_files_with_extensions(root, PARQUET_EXTENSIONS):
        try:
            parquet_file = pq.ParquetFile(path)
            schema = parquet_file.schema_arrow.remove_metadata()
            columns = list(schema.names)
            if first_readable_path is None:
                first_readable_path = path
                first_readable_schema = str(schema)
            result.files_opened += 1
            result.union_columns.update(columns)
            result.schema_variants[str(schema)] += 1

            speaker_column = detect_column(columns, SPEAKER_COLUMN_CANDIDATES)
            split_column = detect_column(columns, SPLIT_COLUMN_CANDIDATES)
            audio_column = detect_column(columns, AUDIO_COLUMN_CANDIDATES)
            if speaker_column:
                result.speaker_columns.add(speaker_column)
            if split_column:
                result.split_columns.add(split_column)

            preview_rows_for_file: list[dict[str, Any]] = []
            rows_still_needed = row_count - len(result.representative_rows)
            if rows_still_needed > 0:
                preview_rows_for_file = read_representative_rows(
                    parquet_file, rows_still_needed
                )
                if preview_rows_for_file:
                    if not result.representative_rows:
                        result.representative_path = path
                        result.representative_schema = str(schema)
                    result.representative_rows.extend(preview_rows_for_file)

            needs_audio_example = (
                audio_column is not None
                and (
                    result.audio_column is None
                    or result.audio_value_structure in {None, "null"}
                )
            )
            if needs_audio_example and audio_column is not None:
                audio_preview_rows = preview_rows_for_file
                if not audio_preview_rows or audio_column not in audio_preview_rows[0]:
                    audio_preview_rows = read_representative_rows(parquet_file, row_count)

                first_non_null = next(
                    (
                        row.get(audio_column)
                        for row in audio_preview_rows
                        if row.get(audio_column) is not None
                    ),
                    None,
                )
                if result.audio_column is None or first_non_null is not None:
                    result.audio_column = audio_column
                    result.audio_source_path = path
                    result.audio_arrow_type = str(schema.field(audio_column).type)
                    result.audio_value_structure = describe_value_structure(first_non_null)

                    for row_index, row in enumerate(audio_preview_rows):
                        audio_value = row.get(audio_column)
                        if audio_value_has_embedded_bytes(audio_value):
                            continue
                        for reference in audio_path_references(audio_value):
                            if not reference_exists(reference, path, root):
                                issues.add(
                                    "missing_audio_reference",
                                    path,
                                    f"row {row_index}: {reference}",
                                )

            selected_columns = [
                column for column in (speaker_column, split_column) if column is not None
            ]
            if selected_columns:
                for batch in parquet_file.iter_batches(
                    batch_size=65_536, columns=selected_columns
                ):
                    for column in selected_columns:
                        values = batch.column(batch.schema.get_field_index(column)).to_pylist()
                        if column == speaker_column:
                            for value in values:
                                if value is None:
                                    result.speaker_null_count += 1
                                else:
                                    result.speaker_values_truncated = add_unique_value(
                                        result.speaker_values,
                                        value,
                                        result.speaker_values_truncated,
                                    )
                        if column == split_column:
                            for value in values:
                                if value is None:
                                    result.split_null_count += 1
                                else:
                                    result.split_values_truncated = add_unique_value(
                                        result.split_values,
                                        value,
                                        result.split_values_truncated,
                                    )
        except Exception as exc:  # PyArrow raises several format-specific exceptions.
            result.files_failed += 1
            issues.add("parquet_error", path, f"{type(exc).__name__}: {exc}")

    if result.representative_path is None:
        result.representative_path = first_readable_path
        result.representative_schema = first_readable_schema
    return result


def inspect_audio(
    summary: ScanSummary,
    requested_count: int,
    issues: IssueCollector,
) -> list[AudioMetadata]:
    if summary.audio_count == 0:
        return []

    try:
        import soundfile as sf
    except ImportError as exc:
        issues.add("dependency_error", summary.root, f"soundfile is required: {exc}")
        return []

    inspected: list[AudioMetadata] = []

    def try_path(path: Path) -> None:
        try:
            information = sf.info(str(path))
            duration = (
                float(information.frames) / information.samplerate
                if information.samplerate
                else 0.0
            )
            inspected.append(
                AudioMetadata(
                    path=path,
                    sample_rate=int(information.samplerate),
                    channels=int(information.channels),
                    duration_seconds=duration,
                    format_name=str(information.format),
                    subtype=str(information.subtype),
                    frames=int(information.frames),
                )
            )
        except Exception as exc:  # libsndfile errors vary by platform and codec.
            issues.add("audio_metadata_error", path, f"{type(exc).__name__}: {exc}")

    initial_candidates = sorted(
        summary.audio_candidates, key=lambda item: str(item).casefold()
    )
    attempted_candidates = set(initial_candidates)
    for path in initial_candidates:
        if len(inspected) >= requested_count:
            break
        try_path(path)

    if len(inspected) < requested_count:
        for path in iter_files_with_extensions(summary.root, AUDIO_EXTENSIONS):
            if path in attempted_candidates:
                continue
            try_path(path)
            if len(inspected) >= requested_count:
                break
    return inspected


def print_scan_summary(summary: ScanSummary) -> None:
    print("\n=== Dataset inventory ===")
    print(f"Root: {summary.root}")
    print(f"Files: {summary.file_count:,}")
    print(f"Directories (excluding root): {summary.directory_count:,}")
    print(
        f"Total file size: {summary.total_size_bytes:,} bytes "
        f"({human_size(summary.total_size_bytes)})"
    )
    print(f"Top-level entries: {summary.top_level_count:,}")
    if summary.file_count == 0:
        print("No files found; the dataset may not have been extracted yet.")
    if summary.top_level_examples:
        print("Top-level examples:")
        for name in sorted(summary.top_level_examples, key=str.casefold):
            print(f"  - {name}")
        omitted = summary.top_level_count - len(summary.top_level_examples)
        if omitted > 0:
            print(f"  ... {omitted:,} more")

    print("\nFile extensions:")
    if not summary.extensions:
        print("  (none)")
    else:
        for extension, count in sorted(
            summary.extensions.items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"  {extension}: {count:,}")

    print("\n=== Detected storage structure ===")
    print(f"Folder hierarchy: {'yes' if summary.directory_count else 'no'}")
    print(f"Parquet files: {summary.parquet_count:,}")
    print(f"WAV files: {summary.extensions['.wav']:,}")
    print(f"FLAC files: {summary.extensions['.flac']:,}")
    print(f"MP3 files: {summary.extensions['.mp3']:,}")
    other_audio_count = summary.audio_count - sum(
        summary.extensions[extension] for extension in (".wav", ".flac", ".mp3")
    )
    print(f"Other recognized audio files: {other_audio_count:,}")
    print(f"Archive files inside dataset root: {summary.archive_count:,}")

    recognized = AUDIO_EXTENSIONS | PARQUET_EXTENSIONS | ARCHIVE_EXTENSIONS
    other_extensions = {
        extension: count
        for extension, count in summary.extensions.items()
        if extension not in recognized
    }
    if other_extensions:
        rendered = ", ".join(
            f"{extension} ({count:,})"
            for extension, count in sorted(other_extensions.items())
        )
        print(f"Other file types: {rendered}")
    else:
        print("Other file types: none")

    if summary.audio_parent_counts:
        parent_names = {
            Path(parent).name
            for parent in summary.audio_parent_counts
            if parent not in {"", "."}
        }
        print(
            "Audio parent directories: "
            f"{len(summary.audio_parent_counts):,} immediate parent path(s)"
        )
        if len(summary.audio_parent_counts) > 1 and parent_names:
            print(
                "Folder-based speaker candidates (heuristic): "
                f"{len(parent_names):,} unique immediate parent name(s)"
            )
        else:
            print("Folder-based speaker candidates: not detectable reliably")
    else:
        print("Folder-based speaker candidates: not detectable")

    if summary.inferred_split_counts:
        rendered_splits = ", ".join(
            f"{name} ({count:,} file paths)"
            for name, count in sorted(summary.inferred_split_counts.items())
        )
        print(f"Splits inferred from audio paths (heuristic): {rendered_splits}")
    else:
        print("Splits inferred from audio paths: none")

    if summary.filename_group_counts:
        rendered_groups = ", ".join(
            f"{name} ({count:,} files)"
            for name, count in sorted(summary.filename_group_counts.items())
        )
        print(f"Filename-derived shard groups (heuristic): {rendered_groups}")
    else:
        print("Filename-derived shard groups: none")


def print_audio_summary(
    metadata: list[AudioMetadata],
    requested_count: int,
    root: Path,
) -> None:
    print("\n=== Normal audio metadata (header-only inspection) ===")
    if not metadata:
        print("No audio metadata was available.")
        return

    print(f"Successfully inspected: {len(metadata):,} of {requested_count:,} requested")
    for index, item in enumerate(metadata, start=1):
        print(f"[{index}] {relative_display(item.path, root)}")
        print(
            f"    sample_rate={item.sample_rate} Hz, channels={item.channels}, "
            f"duration={item.duration_seconds:.3f} s, format={item.format_name}, "
            f"subtype={item.subtype}, frames={item.frames}"
        )

    sample_rates = Counter(item.sample_rate for item in metadata)
    channels = Counter(item.channels for item in metadata)
    formats = Counter(item.format_name for item in metadata)
    print("Observed among inspected files:")
    print(
        "  Sample rates: "
        + ", ".join(f"{key} Hz ({value})" for key, value in sorted(sample_rates.items()))
    )
    print(
        "  Channel counts: "
        + ", ".join(f"{key} ({value})" for key, value in sorted(channels.items()))
    )
    print(
        "  Formats: "
        + ", ".join(f"{key} ({value})" for key, value in sorted(formats.items()))
    )


def print_parquet_summary(result: ParquetSummary | None, root: Path) -> None:
    print("\n=== Parquet inspection ===")
    if result is None:
        print("No readable Parquet inspection result (or no Parquet files present).")
        return

    print(f"Parquet files with readable metadata: {result.files_opened:,}")
    print(f"Parquet files with metadata/data read errors: {result.files_failed:,}")
    print(f"Distinct schema variants: {len(result.schema_variants):,}")
    print(f"Available columns (union): {sorted(result.union_columns, key=str.casefold)}")

    if result.representative_path is None:
        print("No readable Parquet file was found.")
        return

    print(f"Preview schema source: {relative_display(result.representative_path, root)}")
    print("Schema for preview's first source (metadata omitted):")
    print(result.representative_schema)
    print(
        "Column names (union across readable files): "
        f"{sorted(result.union_columns, key=str.casefold)}"
    )

    print(f"Audio field: {result.audio_column or 'not detected'}")
    if result.audio_column:
        if result.audio_source_path is not None:
            print(
                "Audio field source: "
                f"{relative_display(result.audio_source_path, root)}"
            )
        print(f"Audio Arrow type: {result.audio_arrow_type}")
        print(f"Audio value structure: {result.audio_value_structure}")

    print(f"Speaker field(s): {sorted(result.speaker_columns) or 'not detected'}")
    if result.speaker_columns:
        qualifier = ">=" if result.speaker_values_truncated else ""
        print(
            f"Unique speaker IDs: {qualifier}{len(result.speaker_values):,}; "
            f"null speaker values: {result.speaker_null_count:,}"
        )
        speaker_examples = sorted(result.speaker_values, key=str.casefold)[:20]
        print(f"Speaker ID examples: {speaker_examples}")

    print(f"Split field(s): {sorted(result.split_columns) or 'not detected'}")
    if result.split_columns:
        qualifier = ">=" if result.split_values_truncated else ""
        print(
            f"Unique split values: {qualifier}{len(result.split_values):,}; "
            f"null split values: {result.split_null_count:,}"
        )
        split_examples = sorted(result.split_values, key=str.casefold)[:50]
        omitted = len(result.split_values) - len(split_examples)
        print(f"Split value examples: {split_examples}")
        if omitted > 0:
            print(f"  ... {omitted:,} more distinct split values not printed")

    print(f"Rows inspected: {len(result.representative_rows):,}")
    for index, row in enumerate(result.representative_rows, start=1):
        # sanitize_value replaces every bytes-like object, including binary fields
        # outside the detected audio column.
        printable = sanitize_value(row)
        print(f"Row {index}: {json.dumps(printable, ensure_ascii=False, sort_keys=True)}")


def print_issue_summary(summary: ScanSummary, issues: IssueCollector) -> None:
    print("\n=== Missing, unreadable, and invalid data checks ===")
    print(f"Missing/disappeared files or broken links: {issues.count('missing_file'):,}")
    print(f"Unreadable regular files: {issues.count('unreadable_file'):,}")
    print(f"Directory traversal errors: {issues.count('traversal_error'):,}")
    print(f"Zero-byte files: {summary.zero_byte_count:,}")
    print(f"Audio metadata errors in sampled candidates: {issues.count('audio_metadata_error'):,}")
    print(f"Parquet read errors: {issues.count('parquet_error'):,}")
    print(f"Missing external audio references in sampled Parquet rows: {issues.count('missing_audio_reference'):,}")
    print(f"Dependency errors: {issues.count('dependency_error'):,}")
    print(f"Links or Windows reparse points encountered: {summary.symlink_count:,}")
    print(f"Other filesystem entries: {summary.other_entry_count:,}")

    if issues.examples:
        print("Issue examples (bounded):")
        for category, path, detail in issues.examples:
            print(f"  - [{category}] {path}: {detail}")
    else:
        print("Issue examples: none")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()

    print("Vietnamese speaker dataset inspection (read-only)")
    print("No audio samples are decoded and no dataset files are modified.")

    if not root.exists():
        print(f"ERROR: dataset root does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"ERROR: dataset root is not a directory: {root}", file=sys.stderr)
        return 2

    issues = IssueCollector()
    candidate_capacity = max(args.audio_samples * 5, args.audio_samples)
    summary = scan_dataset(root, issues, candidate_capacity)
    print_scan_summary(summary)

    audio_metadata = inspect_audio(summary, args.audio_samples, issues)
    print_audio_summary(audio_metadata, args.audio_samples, root)

    parquet_summary = inspect_parquet(
        root,
        summary.parquet_count,
        args.parquet_rows,
        issues,
    )
    print_parquet_summary(parquet_summary, root)
    print_issue_summary(summary, issues)

    print("\nInspection complete. The dataset was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
