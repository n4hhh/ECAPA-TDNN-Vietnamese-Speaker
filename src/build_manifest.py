#!/usr/bin/env python3
"""Build read-only WAV metadata manifests and dataset-analysis reports.

Audio waveforms are never decoded. The directory walk and CSV exports stream,
and a temporary SQLite database is used as an on-disk working index so memory
usage does not grow with the number of utterances. SHA-256 is calculated only
for files whose byte size occurs more than once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sqlite3
import stat
import sys
import tempfile
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TextIO


DEFAULT_DATASET_ROOT = Path(r"E:\VieSpeaker")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_COLUMNS = (
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
KNOWN_GROUPS = ("train", "train_small", "test", "part")
ALL_GROUPS = (*KNOWN_GROUPS, "unknown")
GROUP_PATTERN = re.compile(
    r"^(?:aug_)?(?P<group>train_small|train|test|part)"
    r"-[0-9]{5}-of-[0-9]{5}_[0-9]+\.wav$",
    flags=re.IGNORECASE,
)
NUMERIC_SPEAKER_PATTERN = re.compile(r"^[0-9]+$")

EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 1
EXPECTED_NUM_FRAMES = 48_000
EXPECTED_DURATION_SECONDS = 3.0
DURATION_TOLERANCE_SECONDS = 1e-6

HASH_CHUNK_BYTES = 4 * 1024 * 1024
DATABASE_BATCH_SIZE = 1_000
MAX_ISSUE_EXAMPLES = 100
DEFAULT_REPORT_EXAMPLES = 100
MAX_PATHS_PER_DUPLICATE_GROUP = 20


@dataclass
class IssueLog:
    counts: Counter[str] = field(default_factory=Counter)
    examples: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, category: str, path: Path, detail: str) -> None:
        self.counts[category] += 1
        if len(self.examples) < MAX_ISSUE_EXAMPLES:
            one_line_detail = " ".join(str(detail).splitlines())[:500]
            self.examples.append((category, str(path), one_line_detail))

    def count(self, category: str) -> int:
        return self.counts[category]


@dataclass
class ScanSummary:
    wav_files: int = 0
    directories: int = 0
    links_or_reparse_points_skipped: int = 0


@dataclass
class HashSummary:
    candidate_size_groups: int = 0
    candidate_files: int = 0
    hashed_files: int = 0
    read_errors: int = 0
    changed_files: int = 0
    duplicate_content_groups: int = 0
    duplicate_content_files: int = 0
    redundant_copies: int = 0


@dataclass(frozen=True)
class OutputPaths:
    manifest: Path
    split_report: Path
    speaker_distribution: Path
    speaker_group_overlap: Path

    def values(self) -> tuple[Path, ...]:
        return (
            self.manifest,
            self.split_report,
            self.speaker_distribution,
            self.speaker_group_overlap,
        )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a WAV metadata manifest and filename-group analysis reports "
            "without modifying dataset files."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=r"dataset root to scan (default: E:\VieSpeaker)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT,
        help="project root where manifests/ and reports/ are written",
    )
    parser.add_argument(
        "--progress-every",
        type=positive_int,
        default=1_000,
        help="print progress after this many files (default: 1000)",
    )
    parser.add_argument(
        "--report-examples",
        type=positive_int,
        default=DEFAULT_REPORT_EXAMPLES,
        help="maximum duplicate/error groups expanded in the text report",
    )
    return parser.parse_args(argv)


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_roots(dataset_root: Path, output_root: Path) -> None:
    if not dataset_root.exists():
        raise ValueError(f"dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    if is_within(output_root, dataset_root):
        raise ValueError(
            "output root must not be the dataset root or one of its descendants: "
            f"{output_root}"
        )
    system_temp_root = Path(tempfile.gettempdir()).resolve(strict=False)
    if is_within(system_temp_root, dataset_root):
        raise ValueError(
            "the process temporary directory resolves inside the dataset root; "
            f"change TEMP/TMP before running: {system_temp_root}"
        )


def output_paths(output_root: Path) -> OutputPaths:
    return OutputPaths(
        manifest=output_root / "manifests" / "full_manifest.csv",
        split_report=output_root / "reports" / "dataset_split_report.txt",
        speaker_distribution=output_root
        / "reports"
        / "speaker_distribution.csv",
        speaker_group_overlap=output_root
        / "reports"
        / "speaker_group_overlap.csv",
    )


def temporary_output_paths(final_paths: OutputPaths) -> OutputPaths:
    token = f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
    return OutputPaths(
        **{
            field_name: final_path.with_name(final_path.name + token)
            for field_name, final_path in final_paths.__dict__.items()
        }
    )


def entry_is_link_or_reparse_point(entry: os.DirEntry[str]) -> bool:
    if entry.is_symlink():
        return True
    if os.name != "nt":
        return False
    attributes = getattr(
        entry.stat(follow_symlinks=False), "st_file_attributes", 0
    )
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def walk_wav_files(
    root: Path,
    issues: IssueLog,
    summary: ScanSummary,
) -> Iterator[Path]:
    """Yield WAV files without following filesystem links or junctions."""

    pending_directories = [root]
    while pending_directories:
        directory = pending_directories.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    try:
                        if entry_is_link_or_reparse_point(entry):
                            summary.links_or_reparse_points_skipped += 1
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            summary.directories += 1
                            pending_directories.append(path)
                        elif (
                            entry.is_file(follow_symlinks=False)
                            and path.suffix.casefold() == ".wav"
                        ):
                            yield path
                    except OSError as exc:
                        issues.add("entry_error", path, str(exc))
        except OSError as exc:
            issues.add("traversal_error", directory, str(exc))


def derive_speaker_id(path: Path) -> tuple[str, str]:
    parent_name = path.parent.name
    if NUMERIC_SPEAKER_PATTERN.fullmatch(parent_name):
        return parent_name, parent_name
    return "", parent_name


def derive_filename_group(filename: str) -> str:
    match = GROUP_PATTERN.fullmatch(filename)
    if match is None:
        return "unknown"
    return match.group("group").casefold()


def normalized_filename_key(filename: str) -> str:
    return unicodedata.normalize("NFC", filename).casefold()


def create_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;

        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            audio_path TEXT NOT NULL UNIQUE,
            speaker_id TEXT NOT NULL,
            speaker_folder TEXT NOT NULL,
            filename TEXT NOT NULL,
            filename_key TEXT NOT NULL,
            filename_group TEXT NOT NULL,
            sample_rate INTEGER,
            channels INTEGER,
            num_frames INTEGER,
            duration_seconds REAL,
            file_size_bytes INTEGER,
            mtime_ns INTEGER,
            header_status TEXT NOT NULL,
            header_error TEXT NOT NULL,
            sha256 TEXT,
            hash_status TEXT NOT NULL DEFAULT 'not_candidate',
            hash_error TEXT NOT NULL DEFAULT ''
        );
        """
    )


def create_database_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX idx_files_filename_key ON files(filename_key);
        CREATE INDEX idx_files_size ON files(file_size_bytes);
        CREATE INDEX idx_files_speaker_group
            ON files(speaker_id, filename_group);
        """
    )
    connection.commit()


def progress_message(
    stage: str,
    current: int,
    started_at: float,
    total: int | None = None,
) -> None:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    rate = current / elapsed
    if total is None:
        progress = f"{current:,} files"
    else:
        percent = (100.0 * current / total) if total else 100.0
        progress = f"{current:,}/{total:,} files ({percent:.1f}%)"
    print(f"[{stage}] {progress} | {rate:,.1f} files/s", flush=True)


def scan_to_database(
    connection: sqlite3.Connection,
    dataset_root: Path,
    progress_every: int,
    issues: IssueLog,
) -> ScanSummary:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is required; install requirements.txt before running"
        ) from exc

    insert_sql = """
        INSERT INTO files (
            audio_path, speaker_id, speaker_folder, filename, filename_key,
            filename_group, sample_rate, channels, num_frames,
            duration_seconds, file_size_bytes, mtime_ns, header_status,
            header_error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    summary = ScanSummary()
    started_at = time.perf_counter()

    for path in walk_wav_files(dataset_root, issues, summary):
        summary.wav_files += 1
        speaker_id, speaker_folder = derive_speaker_id(path)
        group = derive_filename_group(path.name)

        file_size: int | None = None
        mtime_ns: int | None = None
        try:
            stat_result = path.stat()
            file_size = stat_result.st_size
            mtime_ns = stat_result.st_mtime_ns
        except OSError as exc:
            issues.add("stat_error", path, str(exc))

        sample_rate: int | None = None
        channels: int | None = None
        num_frames: int | None = None
        duration_seconds: float | None = None
        header_status = "ok"
        header_error = ""
        try:
            audio_info = sf.info(str(path))
            sample_rate = int(audio_info.samplerate)
            channels = int(audio_info.channels)
            num_frames = int(audio_info.frames)
            duration_seconds = (
                num_frames / sample_rate if sample_rate and num_frames >= 0 else None
            )
        except Exception as exc:  # libsndfile exceptions vary by build/platform.
            header_status = "error"
            header_error = f"{type(exc).__name__}: {' '.join(str(exc).splitlines())[:400]}"
            issues.add("header_error", path, header_error)

        connection.execute(
            insert_sql,
            (
                str(path.resolve(strict=False)),
                speaker_id,
                speaker_folder,
                path.name,
                normalized_filename_key(path.name),
                group,
                sample_rate,
                channels,
                num_frames,
                duration_seconds,
                file_size,
                mtime_ns,
                header_status,
                header_error,
            ),
        )

        if summary.wav_files % DATABASE_BATCH_SIZE == 0:
            connection.commit()
        if summary.wav_files % progress_every == 0:
            progress_message("scan", summary.wav_files, started_at)

    connection.commit()
    create_database_indexes(connection)
    progress_message("scan", summary.wav_files, started_at)
    return summary


def stable_sha256(
    path: Path,
    expected_size: int,
    expected_mtime_ns: int,
) -> tuple[str, str, str]:
    """Hash a file only if its size/mtime remain stable during the read."""

    try:
        before = path.stat()
        if (
            before.st_size != expected_size
            or before.st_mtime_ns != expected_mtime_ns
        ):
            return "changed", "", "size or modification time changed before hashing"

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                block = stream.read(HASH_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)

        after = path.stat()
        if (
            after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            return "changed", "", "size or modification time changed during hashing"
        return "hashed", digest.hexdigest(), ""
    except OSError as exc:
        return "read_error", "", f"{type(exc).__name__}: {exc}"


def scalar(
    connection: sqlite3.Connection,
    query: str,
    parameters: Sequence[Any] = (),
) -> Any:
    row = connection.execute(query, parameters).fetchone()
    return None if row is None else row[0]


def hash_candidate_duplicates(
    connection: sqlite3.Connection,
    progress_every: int,
    issues: IssueLog,
) -> HashSummary:
    summary = HashSummary()
    summary.candidate_size_groups = int(
        scalar(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT file_size_bytes
                FROM files
                WHERE file_size_bytes IS NOT NULL
                GROUP BY file_size_bytes
                HAVING COUNT(*) > 1
            )
            """,
        )
        or 0
    )

    connection.executescript(
        """
        CREATE TEMP TABLE hash_candidates AS
        SELECT id, audio_path, file_size_bytes, mtime_ns
        FROM files
        WHERE file_size_bytes IS NOT NULL
          AND file_size_bytes IN (
              SELECT file_size_bytes
              FROM files
              WHERE file_size_bytes IS NOT NULL
              GROUP BY file_size_bytes
              HAVING COUNT(*) > 1
          );
        CREATE INDEX idx_hash_candidates_id ON hash_candidates(id);
        """
    )
    summary.candidate_files = int(
        scalar(connection, "SELECT COUNT(*) FROM hash_candidates") or 0
    )
    print(
        "[hash] "
        f"{summary.candidate_size_groups:,} repeated-size group(s), "
        f"{summary.candidate_files:,} candidate file(s)",
        flush=True,
    )
    if summary.candidate_files == 0:
        return summary

    started_at = time.perf_counter()
    processed = 0
    read_cursor = connection.execute(
        """
        SELECT id, audio_path, file_size_bytes, mtime_ns
        FROM hash_candidates
        ORDER BY id
        """
    )
    while True:
        rows = read_cursor.fetchmany(DATABASE_BATCH_SIZE)
        if not rows:
            break
        updates: list[tuple[str, str, str, int]] = []
        for file_id, audio_path, file_size, mtime_ns in rows:
            status, digest, error = stable_sha256(
                Path(audio_path), int(file_size), int(mtime_ns)
            )
            updates.append((digest or None, status, error, file_id))
            processed += 1
            if status == "hashed":
                summary.hashed_files += 1
            elif status == "changed":
                summary.changed_files += 1
                issues.add("hash_changed", Path(audio_path), error)
            else:
                summary.read_errors += 1
                issues.add("hash_error", Path(audio_path), error)
            if processed % progress_every == 0:
                progress_message(
                    "hash", processed, started_at, summary.candidate_files
                )

        connection.executemany(
            """
            UPDATE files
            SET sha256 = ?, hash_status = ?, hash_error = ?
            WHERE id = ?
            """,
            updates,
        )
        connection.commit()

    progress_message("hash", processed, started_at, summary.candidate_files)
    connection.execute(
        "CREATE INDEX idx_files_size_sha256 ON files(file_size_bytes, sha256)"
    )
    connection.commit()

    duplicate_stats = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(member_count), 0),
               COALESCE(SUM(member_count - 1), 0)
        FROM (
            SELECT COUNT(*) AS member_count
            FROM files
            WHERE hash_status = 'hashed' AND sha256 IS NOT NULL
            GROUP BY file_size_bytes, sha256
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()
    summary.duplicate_content_groups = int(duplicate_stats[0])
    summary.duplicate_content_files = int(duplicate_stats[1])
    summary.redundant_copies = int(duplicate_stats[2])
    return summary


def open_csv(path: Path) -> TextIO:
    return path.open("w", encoding="utf-8", newline="")


def write_manifest(connection: sqlite3.Connection, path: Path) -> None:
    with open_csv(path) as stream:
        writer = csv.writer(stream)
        writer.writerow(MANIFEST_COLUMNS)
        cursor = connection.execute(
            """
            SELECT audio_path, speaker_id, filename, filename_group,
                   sample_rate, channels, num_frames, duration_seconds,
                   file_size_bytes
            FROM files
            ORDER BY audio_path COLLATE NOCASE
            """
        )
        while True:
            rows = cursor.fetchmany(DATABASE_BATCH_SIZE)
            if not rows:
                break
            writer.writerows(rows)


def speaker_distribution_query() -> str:
    return """
        SELECT speaker_id,
               COUNT(*) AS total_utterances,
               SUM(CASE WHEN filename_group = 'train' THEN 1 ELSE 0 END),
               SUM(CASE WHEN filename_group = 'train_small' THEN 1 ELSE 0 END),
               SUM(CASE WHEN filename_group = 'test' THEN 1 ELSE 0 END),
               SUM(CASE WHEN filename_group = 'part' THEN 1 ELSE 0 END),
               SUM(CASE WHEN filename_group = 'unknown' THEN 1 ELSE 0 END)
        FROM files
        WHERE speaker_id <> ''
        GROUP BY speaker_id
        ORDER BY CAST(speaker_id AS INTEGER), speaker_id
    """


def write_speaker_distribution(
    connection: sqlite3.Connection,
    path: Path,
) -> None:
    columns = (
        "speaker_id",
        "total_utterances",
        "train_utterances",
        "train_small_utterances",
        "test_utterances",
        "part_utterances",
        "unknown_utterances",
        "group_count",
        "groups",
    )
    with open_csv(path) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in connection.execute(speaker_distribution_query()):
            speaker_id, total, *group_counts = row
            present_groups = [
                group
                for group, count in zip(ALL_GROUPS, group_counts)
                if count > 0
            ]
            writer.writerow(
                (
                    speaker_id,
                    total,
                    *group_counts,
                    len(present_groups),
                    "|".join(present_groups),
                )
            )


def shared_speaker_count(
    connection: sqlite3.Connection,
    group_a: str,
    group_b: str,
) -> int:
    return int(
        scalar(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT speaker_id
                FROM files
                WHERE speaker_id <> ''
                  AND filename_group IN (?, ?)
                GROUP BY speaker_id
                HAVING COUNT(DISTINCT filename_group) = 2
            )
            """,
            (group_a, group_b),
        )
        or 0
    )


def write_speaker_group_overlap(
    connection: sqlite3.Connection,
    path: Path,
) -> None:
    with open_csv(path) as stream:
        writer = csv.writer(stream)
        writer.writerow(("group_a", "group_b", "shared_speaker_count"))
        for group_a, group_b in combinations(ALL_GROUPS, 2):
            writer.writerow(
                (
                    group_a,
                    group_b,
                    shared_speaker_count(connection, group_a, group_b),
                )
            )


def write_issue_examples(
    stream: TextIO,
    title: str,
    rows: Iterable[Sequence[Any]],
) -> None:
    stream.write(f"\n{title}\n")
    wrote_any = False
    for row in rows:
        wrote_any = True
        stream.write("  - " + " | ".join("" if item is None else str(item) for item in row) + "\n")
    if not wrote_any:
        stream.write("  (none)\n")


def write_duplicate_filename_examples(
    connection: sqlite3.Connection,
    stream: TextIO,
    report_examples: int,
) -> None:
    stream.write(
        f"\nDuplicate filename groups (up to {report_examples}; "
        f"up to {MAX_PATHS_PER_DUPLICATE_GROUP} paths each)\n"
    )
    groups = connection.execute(
        """
        SELECT filename_key, MIN(filename), COUNT(*)
        FROM files
        GROUP BY filename_key
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC, filename_key
        LIMIT ?
        """,
        (report_examples,),
    )
    wrote_any = False
    for filename_key, display_name, member_count in groups:
        wrote_any = True
        stream.write(f"  {display_name} | members={member_count}\n")
        paths = connection.execute(
            """
            SELECT audio_path
            FROM files
            WHERE filename_key = ?
            ORDER BY audio_path COLLATE NOCASE
            LIMIT ?
            """,
            (filename_key, MAX_PATHS_PER_DUPLICATE_GROUP),
        )
        for (audio_path,) in paths:
            stream.write(f"    - {audio_path}\n")
        omitted = member_count - MAX_PATHS_PER_DUPLICATE_GROUP
        if omitted > 0:
            stream.write(f"    ... {omitted:,} more path(s) not printed\n")
    if not wrote_any:
        stream.write("  (none)\n")


def write_duplicate_content_examples(
    connection: sqlite3.Connection,
    stream: TextIO,
    report_examples: int,
) -> None:
    stream.write(
        f"\nPossible duplicate-content groups (up to {report_examples}; "
        f"up to {MAX_PATHS_PER_DUPLICATE_GROUP} paths each)\n"
    )
    groups = connection.execute(
        """
        SELECT file_size_bytes, sha256, COUNT(*)
        FROM files
        WHERE hash_status = 'hashed' AND sha256 IS NOT NULL
        GROUP BY file_size_bytes, sha256
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC, file_size_bytes, sha256
        LIMIT ?
        """,
        (report_examples,),
    )
    wrote_any = False
    for file_size, digest, member_count in groups:
        wrote_any = True
        stream.write(
            f"  size={file_size} | sha256={digest} | members={member_count}\n"
        )
        paths = connection.execute(
            """
            SELECT audio_path
            FROM files
            WHERE file_size_bytes = ? AND sha256 = ? AND hash_status = 'hashed'
            ORDER BY audio_path COLLATE NOCASE
            LIMIT ?
            """,
            (file_size, digest, MAX_PATHS_PER_DUPLICATE_GROUP),
        )
        for (audio_path,) in paths:
            stream.write(f"    - {audio_path}\n")
        omitted = member_count - MAX_PATHS_PER_DUPLICATE_GROUP
        if omitted > 0:
            stream.write(f"    ... {omitted:,} more path(s) not printed\n")
    if not wrote_any:
        stream.write("  (none)\n")


def filename_duplicate_stats(
    connection: sqlite3.Connection,
) -> tuple[int, int, int]:
    row = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(member_count), 0),
               COALESCE(SUM(member_count - 1), 0)
        FROM (
            SELECT COUNT(*) AS member_count
            FROM files
            GROUP BY filename_key
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()
    return int(row[0]), int(row[1]), int(row[2])


def query_count(connection: sqlite3.Connection, condition: str) -> int:
    return int(scalar(connection, f"SELECT COUNT(*) FROM files WHERE {condition}") or 0)


def write_multi_group_speakers(
    connection: sqlite3.Connection,
    stream: TextIO,
) -> None:
    stream.write("\nSpeakers appearing in more than one filename group\n")
    cursor = connection.execute(
        """
        SELECT speaker_id, filename_group, COUNT(*)
        FROM files
        WHERE speaker_id IN (
            SELECT speaker_id
            FROM files
            WHERE speaker_id <> ''
            GROUP BY speaker_id
            HAVING COUNT(DISTINCT filename_group) > 1
        )
        GROUP BY speaker_id, filename_group
        ORDER BY CAST(speaker_id AS INTEGER), speaker_id,
                 CASE filename_group
                     WHEN 'train' THEN 1
                     WHEN 'train_small' THEN 2
                     WHEN 'test' THEN 3
                     WHEN 'part' THEN 4
                     ELSE 5
                 END
        """
    )
    current_speaker: str | None = None
    memberships: list[str] = []
    for speaker_id, group, file_count in cursor:
        if current_speaker is not None and speaker_id != current_speaker:
            stream.write(f"  {current_speaker}: {', '.join(memberships)}\n")
            memberships = []
        current_speaker = speaker_id
        memberships.append(f"{group} ({file_count})")
    if current_speaker is None:
        stream.write("  (none)\n")
    else:
        stream.write(f"  {current_speaker}: {', '.join(memberships)}\n")


def write_dataset_report(
    connection: sqlite3.Connection,
    path: Path,
    dataset_root: Path,
    scan_summary: ScanSummary,
    hash_summary: HashSummary,
    issues: IssueLog,
    report_examples: int,
) -> None:
    speaker_stats = connection.execute(
        """
        SELECT COUNT(*), COALESCE(MIN(utterance_count), 0),
               COALESCE(MAX(utterance_count), 0),
               COALESCE(AVG(utterance_count), 0.0)
        FROM (
            SELECT COUNT(*) AS utterance_count
            FROM files
            WHERE speaker_id <> ''
            GROUP BY speaker_id
        )
        """
    ).fetchone()
    total_speakers = int(speaker_stats[0])
    min_utterances = int(speaker_stats[1])
    max_utterances = int(speaker_stats[2])
    mean_utterances = float(speaker_stats[3])
    if total_speakers:
        median_offset = (total_speakers - 1) // 2
        median_row_count = 2 if total_speakers % 2 == 0 else 1
        median_values = [
            int(row[0])
            for row in connection.execute(
                """
                SELECT utterance_count
                FROM (
                    SELECT COUNT(*) AS utterance_count
                    FROM files
                    WHERE speaker_id <> ''
                    GROUP BY speaker_id
                )
                ORDER BY utterance_count
                LIMIT ? OFFSET ?
                """,
                (median_row_count, median_offset),
            )
        ]
        median_utterances = sum(median_values) / len(median_values)
    else:
        median_utterances = 0.0

    invalid_speakers = query_count(connection, "speaker_id = ''")
    header_errors = query_count(connection, "header_status <> 'ok'")
    unexpected_rate = query_count(
        connection,
        f"header_status = 'ok' AND (sample_rate IS NULL OR sample_rate <> {EXPECTED_SAMPLE_RATE})",
    )
    unexpected_channels = query_count(
        connection,
        f"header_status = 'ok' AND (channels IS NULL OR channels <> {EXPECTED_CHANNELS})",
    )
    unexpected_frames = query_count(
        connection,
        f"header_status = 'ok' AND (num_frames IS NULL OR num_frames <> {EXPECTED_NUM_FRAMES})",
    )
    unexpected_duration = query_count(
        connection,
        "header_status = 'ok' AND (duration_seconds IS NULL OR "
        f"ABS(duration_seconds - {EXPECTED_DURATION_SECONDS}) > {DURATION_TOLERANCE_SECONDS})",
    )
    multi_group_speakers = int(
        scalar(
            connection,
            """
            SELECT COUNT(*) FROM (
                SELECT speaker_id
                FROM files
                WHERE speaker_id <> ''
                GROUP BY speaker_id
                HAVING COUNT(DISTINCT filename_group) > 1
            )
            """,
        )
        or 0
    )
    duplicate_name_groups, duplicate_name_files, duplicate_name_excess = (
        filename_duplicate_stats(connection)
    )

    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("Vietnamese speaker dataset filename-group analysis\n")
        stream.write("=" * 52 + "\n")
        stream.write(f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}\n")
        stream.write(f"Dataset root: {dataset_root}\n")
        stream.write(f"WAV files in manifest: {scan_summary.wav_files:,}\n")
        stream.write(f"Directories traversed: {scan_summary.directories:,}\n")
        stream.write(
            "Links/reparse points skipped: "
            f"{scan_summary.links_or_reparse_points_skipped:,}\n"
        )
        stream.write("Dataset files modified: no\n")
        stream.write("Final train/validation/test split assigned: no\n")
        stream.write(
            "Speaker/group statistics include every discovered WAV with a valid "
            "numeric immediate parent folder.\n"
        )

        stream.write("\nSpeaker summary\n")
        stream.write(f"  Total speakers: {total_speakers:,}\n")
        stream.write(f"  Utterances per speaker - min: {min_utterances:,}\n")
        stream.write(f"  Utterances per speaker - max: {max_utterances:,}\n")
        stream.write(f"  Utterances per speaker - mean: {mean_utterances:.6f}\n")
        stream.write(f"  Utterances per speaker - median: {median_utterances:.6f}\n")

        stream.write("\nFilename groups (heuristic, not final splits)\n")
        for group in ALL_GROUPS:
            file_count = int(
                scalar(
                    connection,
                    "SELECT COUNT(*) FROM files WHERE filename_group = ?",
                    (group,),
                )
                or 0
            )
            speaker_count = int(
                scalar(
                    connection,
                    """
                    SELECT COUNT(DISTINCT speaker_id)
                    FROM files
                    WHERE filename_group = ? AND speaker_id <> ''
                    """,
                    (group,),
                )
                or 0
            )
            stream.write(
                f"  {group}: {file_count:,} files; {speaker_count:,} speakers\n"
            )

        stream.write(f"\nSpeakers in multiple groups: {multi_group_speakers:,}\n")
        stream.write("Pairwise speaker overlaps\n")
        for group_a, group_b in combinations(ALL_GROUPS, 2):
            overlap = shared_speaker_count(connection, group_a, group_b)
            stream.write(f"  {group_a} + {group_b}: {overlap:,}\n")
        write_multi_group_speakers(connection, stream)

        stream.write("\nIdentity and audio metadata validation\n")
        stream.write(
            "  Files with invalid numeric speaker folder names: "
            f"{invalid_speakers:,}\n"
        )
        stream.write(f"  Unreadable/invalid WAV headers: {header_errors:,}\n")
        stream.write(f"  Unexpected sample rate (expected 16000 Hz): {unexpected_rate:,}\n")
        stream.write(f"  Unexpected channel count (expected 1): {unexpected_channels:,}\n")
        stream.write(f"  Unexpected frame count (expected 48000): {unexpected_frames:,}\n")
        stream.write(
            "  Unexpected duration (expected 3.0 s): "
            f"{unexpected_duration:,}\n"
        )

        invalid_rows = connection.execute(
            """
            SELECT audio_path, speaker_folder
            FROM files
            WHERE speaker_id = ''
            ORDER BY audio_path COLLATE NOCASE
            LIMIT ?
            """,
            (report_examples,),
        )
        write_issue_examples(
            stream,
            f"Files with invalid speaker folders (up to {report_examples})",
            invalid_rows,
        )

        unexpected_condition = f"""
            header_status <> 'ok'
            OR sample_rate IS NULL OR sample_rate <> {EXPECTED_SAMPLE_RATE}
            OR channels IS NULL OR channels <> {EXPECTED_CHANNELS}
            OR num_frames IS NULL OR num_frames <> {EXPECTED_NUM_FRAMES}
            OR duration_seconds IS NULL
            OR ABS(duration_seconds - {EXPECTED_DURATION_SECONDS}) > {DURATION_TOLERANCE_SECONDS}
        """
        unexpected_rows = connection.execute(
            f"""
            SELECT audio_path, header_status, sample_rate, channels,
                   num_frames, duration_seconds, header_error
            FROM files
            WHERE {unexpected_condition}
            ORDER BY audio_path COLLATE NOCASE
            LIMIT ?
            """,
            (report_examples,),
        )
        write_issue_examples(
            stream,
            f"Unexpected audio-metadata examples (up to {report_examples})",
            unexpected_rows,
        )

        stream.write("\nDuplicate filenames\n")
        stream.write(
            "  Definition: Unicode-NFC and case-insensitive basename equality; "
            "content may differ.\n"
        )
        stream.write(f"  Duplicate filename groups: {duplicate_name_groups:,}\n")
        stream.write(f"  Files in duplicate filename groups: {duplicate_name_files:,}\n")
        stream.write(f"  Excess filename occurrences: {duplicate_name_excess:,}\n")
        write_duplicate_filename_examples(
            connection, stream, report_examples
        )

        stream.write("\nPossible duplicate file content\n")
        stream.write(
            "  Method: group by file size, then SHA-256 only for repeated-size "
            "candidates. Waveforms are not decoded.\n"
        )
        stream.write(
            f"  Repeated-size groups: {hash_summary.candidate_size_groups:,}\n"
        )
        stream.write(f"  Size-candidate files: {hash_summary.candidate_files:,}\n")
        stream.write(f"  Successfully hashed files: {hash_summary.hashed_files:,}\n")
        stream.write(f"  Hash read errors: {hash_summary.read_errors:,}\n")
        stream.write(f"  Files changed during hashing: {hash_summary.changed_files:,}\n")
        stream.write(
            "  Same-size + same-SHA256 groups: "
            f"{hash_summary.duplicate_content_groups:,}\n"
        )
        stream.write(
            "  Files in same-size + same-SHA256 groups: "
            f"{hash_summary.duplicate_content_files:,}\n"
        )
        stream.write(
            f"  Redundant copies beyond one per group: {hash_summary.redundant_copies:,}\n"
        )
        write_duplicate_content_examples(
            connection, stream, report_examples
        )

        stream.write("\nScan/hash operational issues\n")
        if issues.counts:
            for category, count in sorted(issues.counts.items()):
                stream.write(f"  {category}: {count:,}\n")
        else:
            stream.write("  (none)\n")
        if issues.examples:
            stream.write(f"Issue examples (first {len(issues.examples)}):\n")
            for category, issue_path, detail in issues.examples:
                stream.write(f"  [{category}] {issue_path} | {detail}\n")


def write_all_outputs(
    connection: sqlite3.Connection,
    final_paths: OutputPaths,
    dataset_root: Path,
    scan_summary: ScanSummary,
    hash_summary: HashSummary,
    issues: IssueLog,
    report_examples: int,
) -> None:
    for path in final_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = path.parent.resolve(strict=False)
        if is_within(resolved_parent, dataset_root):
            raise ValueError(
                "refusing to write through an output directory that resolves "
                f"inside the dataset root: {resolved_parent}"
            )
    temporary_paths = temporary_output_paths(final_paths)
    try:
        print("[write] full manifest", flush=True)
        write_manifest(connection, temporary_paths.manifest)
        print("[write] speaker distribution", flush=True)
        write_speaker_distribution(connection, temporary_paths.speaker_distribution)
        print("[write] speaker group overlap", flush=True)
        write_speaker_group_overlap(connection, temporary_paths.speaker_group_overlap)
        print("[write] dataset report", flush=True)
        write_dataset_report(
            connection,
            temporary_paths.split_report,
            dataset_root,
            scan_summary,
            hash_summary,
            issues,
            report_examples,
        )
        for temporary_path, final_path in zip(
            temporary_paths.values(), final_paths.values()
        ):
            os.replace(temporary_path, final_path)
    finally:
        for temporary_path in temporary_paths.values():
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve(strict=False)
    output_root = args.output_root.expanduser().resolve(strict=False)

    try:
        validate_roots(dataset_root, output_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    paths = output_paths(output_root)
    issues = IssueLog()
    print("Vietnamese speaker manifest builder (read-only dataset access)")
    print(f"Dataset root: {dataset_root}")
    print(f"Output root: {output_root}")
    print("No final train/validation/test split will be assigned.", flush=True)

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="speaker_manifest_index_", dir=output_root
        ) as temp_dir:
            database_path = Path(temp_dir) / "manifest_index.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                create_database(connection)
                scan_summary = scan_to_database(
                    connection,
                    dataset_root,
                    args.progress_every,
                    issues,
                )
                hash_summary = hash_candidate_duplicates(
                    connection,
                    args.progress_every,
                    issues,
                )
                write_all_outputs(
                    connection,
                    paths,
                    dataset_root,
                    scan_summary,
                    hash_summary,
                    issues,
                    args.report_examples,
                )
            finally:
                connection.close()
    except KeyboardInterrupt:
        print(
            "\nInterrupted; unpublished temporary files were cleaned up. "
            "Any output file already published was not rolled back.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("\nManifest/report generation complete.")
    for path in paths.values():
        print(f"  {path}")
    print(f"WAV files: {scan_summary.wav_files:,}")
    print(f"Possible duplicate-content groups: {hash_summary.duplicate_content_groups:,}")
    print("Dataset files modified: no")

    operational_errors = sum(
        issues.count(category)
        for category in (
            "traversal_error",
            "entry_error",
            "stat_error",
            "hash_error",
            "hash_changed",
        )
    )
    if scan_summary.wav_files == 0 or operational_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
