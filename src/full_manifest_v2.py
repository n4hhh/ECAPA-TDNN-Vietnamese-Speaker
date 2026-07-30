"""Deterministic production full-manifest v2 construction.

The builder consumes the approved non-production audit inventory, reconciles
every row with the read-only dataset and approved audit summary, and publishes
the manifest and identity JSON together.  It never creates splits or labels.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from src.dataset_audit_v2 import INVENTORY_FIELDS, Snapshot, snapshot_dataset


MANIFEST_VERSION = "v2"
MANIFEST_SCHEMA_VERSION = 1
IDENTITY_SCHEMA_VERSION = 1
CREATION_TOOL = "scripts/create_full_manifest_v2.py"
CREATION_TOOL_VERSION = "full_manifest_v2_schema_1"
DATASET_ID = "VieSpeaker2.0_augmented_dataset"
MANIFEST_FILENAME = "full_manifest_v2.csv"
IDENTITY_FILENAME = "full_manifest_v2_identity.json"
CANONICAL_MANIFEST_RELATIVE_PATH = "manifests/v2/full_manifest_v2.csv"

MANIFEST_FIELDS = (
    "audio_path",
    "speaker_id",
    "filename",
    "provenance",
    "provenance_parse_status",
    "sample_rate_hz",
    "channel_count",
    "sample_width_bytes",
    "bits_per_sample",
    "wav_encoding",
    "frame_count",
    "duration_seconds",
    "file_size_bytes",
    "wav_status",
    "duplicate_group",
    "candidate_source_group",
    "source_group_parse_status",
    "manifest_version",
)

FORBIDDEN_MANIFEST_FIELDS = frozenset(
    {
        "final_split",
        "label",
        "speaker_label",
        "training_eligible",
        "verification_eligible",
        "exclusion_reason",
    }
)

PROVENANCE_ORDER = ("train", "train_small", "part", "test")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DRIVE_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
DURATION_QUANTUM = Decimal("0.000001")


@dataclass(frozen=True)
class ManifestExpectations:
    row_count: int
    speaker_count: int
    total_bytes: int
    total_duration_seconds: Decimal
    readable_wav_count: int
    duplicate_group_count: int
    duplicate_file_count: int
    cross_speaker_duplicate_group_count: int
    singleton_speaker_count: int
    provenance_counts: Mapping[str, int]


APPROVED_EXPECTATIONS = ManifestExpectations(
    row_count=125_847,
    speaker_count=1_675,
    total_bytes=12_091_128_066,
    total_duration_seconds=Decimal("377541"),
    readable_wav_count=125_847,
    duplicate_group_count=4,
    duplicate_file_count=8,
    cross_speaker_duplicate_group_count=0,
    singleton_speaker_count=128,
    provenance_counts={
        "train": 86_197,
        "train_small": 16_128,
        "part": 17_858,
        "test": 5_664,
    },
)


@dataclass(frozen=True)
class ManifestRow:
    audio_path: str
    speaker_id: str
    filename: str
    provenance: str
    provenance_parse_status: str
    sample_rate_hz: int
    channel_count: int
    sample_width_bytes: int
    bits_per_sample: int
    wav_encoding: str
    frame_count: int
    duration_seconds: str
    file_size_bytes: int
    wav_status: str
    duplicate_group: str
    candidate_source_group: str
    source_group_parse_status: str
    manifest_version: str = MANIFEST_VERSION

    def csv_row(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True)
class ApprovedAudit:
    inventory_relative_path: str
    inventory_sha256: str
    report_relative_path: str
    report_sha256: str
    json_relative_path: str
    json_sha256: str
    dataset_snapshot: Snapshot
    summary: Mapping[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def repository_relative(path: Path, project_root: Path) -> str:
    root = project_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"input must be inside the repository: {path}") from error
    value = PurePosixPath(*relative.parts).as_posix()
    validate_portable_path(value)
    return value


def validate_portable_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("portable path must be a non-empty string")
    pure = PurePosixPath(value)
    if (
        "\\" in value
        or ":" in value
        or pure.is_absolute()
        or ".." in pure.parts
        or any(part in {"", "."} for part in pure.parts)
        or Path(value).is_absolute()
        or DRIVE_PATH_PATTERN.match(value)
    ):
        raise ValueError(f"non-portable path: {value!r}")
    return value


def _positive_int(value: str, field: str, row_number: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"audit row {row_number}: invalid {field}") from error
    if result <= 0:
        raise ValueError(f"audit row {row_number}: {field} must be positive")
    return result


def canonical_duration(value: str, row_number: int) -> str:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise ValueError(
            f"audit row {row_number}: invalid duration_seconds"
        ) from error
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(
            f"audit row {row_number}: duration_seconds must be finite and positive"
        )
    return format(parsed.quantize(DURATION_QUANTUM), "f")


def speaker_sort_key(row: ManifestRow) -> tuple[int, str]:
    return int(row.speaker_id), row.audio_path


def _audit_row_to_manifest(
    source: Mapping[str, str], row_number: int
) -> ManifestRow:
    path = validate_portable_path(source["dataset_relative_path"].strip())
    pure = PurePosixPath(path)
    if len(pure.parts) != 2:
        raise ValueError(f"audit row {row_number}: WAV must have depth 2")
    speaker = source["candidate_speaker_id"].strip()
    if not speaker or not speaker.isdecimal():
        raise ValueError(f"audit row {row_number}: speaker ID must be numeric")
    if source["direct_parent_folder"].strip() != speaker or pure.parent.name != speaker:
        raise ValueError(
            f"audit row {row_number}: speaker ID differs from direct parent"
        )
    filename = source["filename"].strip()
    if pure.name != filename:
        raise ValueError(f"audit row {row_number}: filename differs from path basename")
    if (
        source["extension"].strip().lower() != ".wav"
        or source["placement_status"].strip() != "expected_depth"
        or source["speaker_folder_status"].strip() != "valid_numeric_top_level"
        or source["relative_depth"].strip() != "2"
    ):
        raise ValueError(f"audit row {row_number}: placement metadata is invalid")
    provenance = source["provenance_class"].strip()
    if provenance not in PROVENANCE_ORDER:
        raise ValueError(f"audit row {row_number}: unsupported provenance")
    provenance_status = source["provenance_parse_status"].strip()
    if provenance_status != "exact_case_supported":
        raise ValueError(f"audit row {row_number}: provenance is not exactly supported")
    source_status = source["source_group_parse_status"].strip()
    candidate_source = source["candidate_source_group"].strip()
    if source_status != "exact_pattern_supported" or not candidate_source:
        raise ValueError(f"audit row {row_number}: source coordinate is unsupported")
    wav_status = source["wav_read_status"].strip()
    if wav_status != "readable":
        raise ValueError(f"audit row {row_number}: WAV is not readable")
    if (
        source["wav_error_type"].strip()
        or source["wav_error_message"].strip()
        or source["zero_byte_file"].strip() != "False"
        or source["zero_frame_audio"].strip() != "False"
    ):
        raise ValueError(f"audit row {row_number}: invalid WAV audit state")

    duplicate_group = source["exact_duplicate_group"].strip()
    if duplicate_group and not re.fullmatch(r"dup_\d{6}", duplicate_group):
        raise ValueError(f"audit row {row_number}: invalid duplicate group")

    return ManifestRow(
        audio_path=path,
        speaker_id=speaker,
        filename=filename,
        provenance=provenance,
        provenance_parse_status=provenance_status,
        sample_rate_hz=_positive_int(
            source["sample_rate_hz"], "sample_rate_hz", row_number
        ),
        channel_count=_positive_int(
            source["channel_count"], "channel_count", row_number
        ),
        sample_width_bytes=_positive_int(
            source["sample_width_bytes"], "sample_width_bytes", row_number
        ),
        bits_per_sample=_positive_int(
            source["bits_per_sample"], "bits_per_sample", row_number
        ),
        wav_encoding=source["wav_encoding"].strip(),
        frame_count=_positive_int(source["frame_count"], "frame_count", row_number),
        duration_seconds=canonical_duration(
            source["duration_seconds"], row_number
        ),
        file_size_bytes=_positive_int(
            source["file_size_bytes"], "file_size_bytes", row_number
        ),
        wav_status=wav_status,
        duplicate_group=duplicate_group,
        candidate_source_group=candidate_source,
        source_group_parse_status=source_status,
    )


def load_audit_inventory(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
            raise ValueError("audit inventory schema is invalid")
        for row_number, source in enumerate(reader, start=2):
            rows.append(_audit_row_to_manifest(source, row_number))
    if not rows:
        raise ValueError("audit inventory contains no rows")
    rows.sort(key=speaker_sort_key)
    return rows


def _decimal_equal(value: Any, expected: Decimal) -> bool:
    try:
        return Decimal(str(value)) == expected
    except InvalidOperation:
        return False


def _provenance_from_summary(summary: Mapping[str, Any]) -> dict[str, int]:
    entries = summary.get("provenance", {}).get("summary", [])
    if not isinstance(entries, list):
        raise ValueError("audit provenance summary is invalid")
    result: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("audit provenance entry is invalid")
        name = entry.get("provenance_class")
        count = entry.get("file_count")
        if name in PROVENANCE_ORDER:
            result[str(name)] = int(count)
    return result


def validate_audit_summary(
    summary: Mapping[str, Any],
    expectations: ManifestExpectations,
    inventory_relative_path: str,
) -> Snapshot:
    if (
        summary.get("schema_name") != "dataset_understanding_v2"
        or summary.get("schema_version") != 2
        or summary.get("result") != "PASS"
        or summary.get("scan_complete") is not True
    ):
        raise ValueError("source audit did not complete with the approved schema")
    counts = summary.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("source audit counts are missing")
    expected_counts = {
        "total_file_count": expectations.row_count,
        "wav_file_count": expectations.row_count,
        "non_wav_file_count": 0,
        "total_byte_size": expectations.total_bytes,
        "valid_candidate_speaker_count": expectations.speaker_count,
        "valid_readable_wav_count": expectations.readable_wav_count,
        "invalid_unreadable_wav_count": 0,
        "zero_byte_wav_count": 0,
        "zero_frame_wav_count": 0,
    }
    for key, value in expected_counts.items():
        if counts.get(key) != value:
            raise ValueError(f"source audit count mismatch: {key}")
    if not _decimal_equal(
        counts.get("total_duration_seconds"), expectations.total_duration_seconds
    ):
        raise ValueError("source audit duration mismatch")
    if _provenance_from_summary(summary) != dict(expectations.provenance_counts):
        raise ValueError("source audit provenance counts mismatch")

    duplicates = summary.get("duplicates")
    if not isinstance(duplicates, Mapping):
        raise ValueError("source audit duplicate summary is missing")
    duplicate_expected = {
        "exact_duplicate_group_count": expectations.duplicate_group_count,
        "files_in_exact_duplicate_groups": expectations.duplicate_file_count,
        "cross_speaker_exact_duplicate_group_count": (
            expectations.cross_speaker_duplicate_group_count
        ),
    }
    for key, value in duplicate_expected.items():
        if duplicates.get(key) != value:
            raise ValueError(f"source audit duplicate mismatch: {key}")
    if (
        summary.get("speakers", {})
        .get("threshold_counts", {})
        .get("fewer_than_2")
        != expectations.singleton_speaker_count
    ):
        raise ValueError("source audit singleton count mismatch")
    if not all(summary.get("reconciliation", {}).values()):
        raise ValueError("source audit reconciliation failed")
    preservation = summary.get("preservation")
    if (
        not isinstance(preservation, Mapping)
        or preservation.get("snapshots_match") is not True
    ):
        raise ValueError("source audit preservation did not pass")
    snapshot = preservation.get("post_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("source audit snapshot is missing")
    try:
        approved_snapshot = Snapshot(
            file_count=int(snapshot["file_count"]),
            total_bytes=int(snapshot["total_bytes"]),
            directory_count=int(snapshot["directory_count"]),
            identity_sha256=str(snapshot["identity_sha256"]),
            errors=tuple(snapshot["errors"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source audit snapshot schema is invalid") from error
    if (
        approved_snapshot.file_count != expectations.row_count
        or approved_snapshot.total_bytes != expectations.total_bytes
        or approved_snapshot.errors
        or not SHA256_PATTERN.fullmatch(approved_snapshot.identity_sha256)
    ):
        raise ValueError("source audit snapshot values are invalid")
    artifact_inventory = (
        summary.get("artifact_paths", {}).get("inventory")
        if isinstance(summary.get("artifact_paths"), Mapping)
        else None
    )
    if artifact_inventory != inventory_relative_path:
        raise ValueError("supplied audit inventory differs from audit JSON identity")
    return approved_snapshot


def load_approved_audit(
    *,
    project_root: Path,
    inventory_path: Path,
    audit_report_path: Path,
    audit_json_path: Path,
    expectations: ManifestExpectations,
) -> ApprovedAudit:
    inventory_relative = repository_relative(inventory_path, project_root)
    report_relative = repository_relative(audit_report_path, project_root)
    json_relative = repository_relative(audit_json_path, project_root)
    with audit_json_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, Mapping):
        raise ValueError("source audit JSON must contain an object")
    snapshot = validate_audit_summary(summary, expectations, inventory_relative)
    report_text = audit_report_path.read_text(encoding="utf-8")
    if (
        "Result: PASS for the audit task" not in report_text
        or "125,847" not in report_text
        or "1,675" not in report_text
    ):
        raise ValueError("source audit Markdown does not identify the approved PASS")
    return ApprovedAudit(
        inventory_relative_path=inventory_relative,
        inventory_sha256=file_sha256(inventory_path),
        report_relative_path=report_relative,
        report_sha256=file_sha256(audit_report_path),
        json_relative_path=json_relative,
        json_sha256=file_sha256(audit_json_path),
        dataset_snapshot=snapshot,
        summary=summary,
    )


def validate_manifest_rows(
    rows: Sequence[ManifestRow], expectations: ManifestExpectations
) -> dict[str, Any]:
    if len(rows) != expectations.row_count:
        raise ValueError(
            f"manifest row count is {len(rows)}, expected {expectations.row_count}"
        )
    paths: set[str] = set()
    speaker_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    duplicate_members: dict[str, list[ManifestRow]] = defaultdict(list)
    total_bytes = 0
    total_duration = Decimal(0)
    numeric_spellings: dict[int, str] = {}
    previous_key: tuple[int, str] | None = None
    for index, row in enumerate(rows, start=2):
        if row.manifest_version != MANIFEST_VERSION:
            raise ValueError(f"manifest row {index}: wrong manifest version")
        validate_portable_path(row.audio_path)
        pure = PurePosixPath(row.audio_path)
        if (
            len(pure.parts) != 2
            or pure.parent.name != row.speaker_id
            or pure.name != row.filename
        ):
            raise ValueError(f"manifest row {index}: path identity mismatch")
        if not row.speaker_id.isdecimal():
            raise ValueError(f"manifest row {index}: non-numeric speaker")
        numeric = int(row.speaker_id)
        prior_spelling = numeric_spellings.setdefault(numeric, row.speaker_id)
        if prior_spelling != row.speaker_id:
            raise ValueError("numeric speaker IDs have ambiguous spellings")
        if row.audio_path in paths:
            raise ValueError(f"duplicate manifest path: {row.audio_path}")
        paths.add(row.audio_path)
        key = speaker_sort_key(row)
        if previous_key is not None and key < previous_key:
            raise ValueError("manifest rows are not in deterministic numeric-speaker order")
        previous_key = key
        if row.provenance not in PROVENANCE_ORDER:
            raise ValueError(f"manifest row {index}: invalid provenance")
        if row.provenance_parse_status != "exact_case_supported":
            raise ValueError(f"manifest row {index}: invalid provenance status")
        if (
            row.wav_status != "readable"
            or row.sample_rate_hz <= 0
            or row.channel_count <= 0
            or row.sample_width_bytes <= 0
            or row.bits_per_sample <= 0
            or row.frame_count <= 0
            or row.file_size_bytes <= 0
            or not row.wav_encoding
        ):
            raise ValueError(f"manifest row {index}: invalid WAV metadata")
        duration = Decimal(row.duration_seconds)
        if not duration.is_finite() or duration <= 0:
            raise ValueError(f"manifest row {index}: invalid duration")
        if not row.candidate_source_group or (
            row.source_group_parse_status != "exact_pattern_supported"
        ):
            raise ValueError(f"manifest row {index}: invalid source coordinate")
        speaker_counts[row.speaker_id] += 1
        provenance_counts[row.provenance] += 1
        total_bytes += row.file_size_bytes
        total_duration += duration
        if row.duplicate_group:
            duplicate_members[row.duplicate_group].append(row)

    if len(paths) != expectations.row_count:
        raise ValueError("manifest unique-path count mismatch")
    if len(speaker_counts) != expectations.speaker_count:
        raise ValueError("manifest speaker count mismatch")
    if total_bytes != expectations.total_bytes:
        raise ValueError("manifest byte total mismatch")
    if total_duration != expectations.total_duration_seconds:
        raise ValueError("manifest duration total mismatch")
    if dict(provenance_counts) != dict(expectations.provenance_counts):
        raise ValueError("manifest provenance counts mismatch")
    if len(duplicate_members) != expectations.duplicate_group_count:
        raise ValueError("manifest duplicate-group count mismatch")
    if sum(len(members) for members in duplicate_members.values()) != (
        expectations.duplicate_file_count
    ):
        raise ValueError("manifest duplicate-file count mismatch")
    cross_speaker = sum(
        len({row.speaker_id for row in members}) > 1
        for members in duplicate_members.values()
    )
    if cross_speaker != expectations.cross_speaker_duplicate_group_count:
        raise ValueError("manifest cross-speaker duplicate count mismatch")
    if sum(count == 1 for count in speaker_counts.values()) != (
        expectations.singleton_speaker_count
    ):
        raise ValueError("manifest singleton-speaker count mismatch")
    expected_group_ids = {
        f"dup_{index:06d}"
        for index in range(1, expectations.duplicate_group_count + 1)
    }
    if set(duplicate_members) != expected_group_ids:
        raise ValueError("manifest duplicate identities differ from audit")
    return {
        "row_count": len(rows),
        "speaker_count": len(speaker_counts),
        "unique_audio_path_count": len(paths),
        "total_bytes": total_bytes,
        "total_duration_seconds": int(total_duration),
        "readable_wav_count": sum(row.wav_status == "readable" for row in rows),
        "duplicate_group_count": len(duplicate_members),
        "duplicate_file_count": sum(
            len(members) for members in duplicate_members.values()
        ),
        "cross_speaker_duplicate_group_count": cross_speaker,
        "singleton_speaker_count": sum(
            count == 1 for count in speaker_counts.values()
        ),
        "provenance_counts": {
            name: provenance_counts[name] for name in PROVENANCE_ORDER
        },
    }


def reconcile_dataset(
    rows: Sequence[ManifestRow],
    dataset_root: Path,
    approved_snapshot: Snapshot,
) -> Snapshot:
    root = dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    current_snapshot = snapshot_dataset(root)
    if current_snapshot != approved_snapshot:
        raise ValueError("dataset snapshot differs from the approved audit")
    seen_resolved: set[Path] = set()
    for row in rows:
        pure = PurePosixPath(row.audio_path)
        path = root.joinpath(*pure.parts)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"manifest path does not resolve inside dataset root: {row.audio_path}"
            ) from error
        if not resolved.is_file():
            raise ValueError(f"manifest WAV is missing: {row.audio_path}")
        if resolved in seen_resolved:
            raise ValueError(f"multiple manifest paths resolve to one file: {row.audio_path}")
        seen_resolved.add(resolved)
        if resolved.stat().st_size != row.file_size_bytes:
            raise ValueError(f"manifest file size differs: {row.audio_path}")
    if len(seen_resolved) != approved_snapshot.file_count:
        raise ValueError("manifest does not represent every audited dataset file")
    return current_snapshot


def render_manifest_csv(rows: Sequence[ManifestRow]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=MANIFEST_FIELDS,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(row.csv_row() for row in rows)
    return stream.getvalue().encode("utf-8")


def read_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("persisted manifest schema is invalid")
        if FORBIDDEN_MANIFEST_FIELDS.intersection(reader.fieldnames or ()):
            raise ValueError("persisted manifest contains a forbidden field")
        for number, raw in enumerate(reader, start=2):
            try:
                rows.append(
                    ManifestRow(
                        audio_path=raw["audio_path"],
                        speaker_id=raw["speaker_id"],
                        filename=raw["filename"],
                        provenance=raw["provenance"],
                        provenance_parse_status=raw["provenance_parse_status"],
                        sample_rate_hz=int(raw["sample_rate_hz"]),
                        channel_count=int(raw["channel_count"]),
                        sample_width_bytes=int(raw["sample_width_bytes"]),
                        bits_per_sample=int(raw["bits_per_sample"]),
                        wav_encoding=raw["wav_encoding"],
                        frame_count=int(raw["frame_count"]),
                        duration_seconds=raw["duration_seconds"],
                        file_size_bytes=int(raw["file_size_bytes"]),
                        wav_status=raw["wav_status"],
                        duplicate_group=raw["duplicate_group"],
                        candidate_source_group=raw["candidate_source_group"],
                        source_group_parse_status=raw[
                            "source_group_parse_status"
                        ],
                        manifest_version=raw["manifest_version"],
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"persisted manifest row {number} is invalid") from error
    return rows


def render_identity_json(identity: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            identity,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _assert_no_absolute_strings(value: Any, context: str = "identity") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_absolute_strings(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_absolute_strings(child, f"{context}[{index}]")
    elif isinstance(value, str):
        if DRIVE_PATH_PATTERN.match(value) or value.startswith("\\") or (
            len(value) > 1 and value.startswith("/")
        ):
            raise ValueError(f"{context} contains an absolute local path")


def build_identity(
    *,
    manifest_sha256: str,
    canonical_manifest_relative_path: str,
    counts: Mapping[str, Any],
    audit: ApprovedAudit,
) -> dict[str, Any]:
    validate_portable_path(canonical_manifest_relative_path)
    identity = {
        "dataset_id": DATASET_ID,
        "manifest_version": MANIFEST_VERSION,
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_relative_path": canonical_manifest_relative_path,
        "path_base_semantics": "relative_to_supplied_dataset_root",
        "path_separator": "/",
        "sort_order": [
            "numeric speaker_id ascending",
            "audio_path stable ordinal ascending",
        ],
        **dict(counts),
        "manifest_sha256": manifest_sha256,
        "source_audit_report_relative_path": audit.report_relative_path,
        "source_audit_report_sha256": audit.report_sha256,
        "source_audit_json_relative_path": audit.json_relative_path,
        "source_audit_json_sha256": audit.json_sha256,
        "source_audit_inventory_relative_path": audit.inventory_relative_path,
        "source_audit_inventory_sha256": audit.inventory_sha256,
        "source_dataset_snapshot_identity_sha256": (
            audit.dataset_snapshot.identity_sha256
        ),
        "creation_tool": CREATION_TOOL,
        "creation_tool_version": CREATION_TOOL_VERSION,
    }
    _assert_no_absolute_strings(identity)
    return identity


def validate_identity(
    identity: Mapping[str, Any],
    manifest_path: Path,
    expectations: ManifestExpectations,
    expected_audit: ApprovedAudit,
    canonical_manifest_relative_path: str,
) -> None:
    required = {
        "dataset_id",
        "manifest_version",
        "schema_version",
        "manifest_schema_version",
        "manifest_relative_path",
        "path_base_semantics",
        "path_separator",
        "sort_order",
        "row_count",
        "speaker_count",
        "unique_audio_path_count",
        "total_bytes",
        "total_duration_seconds",
        "readable_wav_count",
        "duplicate_group_count",
        "duplicate_file_count",
        "cross_speaker_duplicate_group_count",
        "singleton_speaker_count",
        "provenance_counts",
        "manifest_sha256",
        "source_audit_report_relative_path",
        "source_audit_report_sha256",
        "source_audit_json_relative_path",
        "source_audit_json_sha256",
        "source_audit_inventory_relative_path",
        "source_audit_inventory_sha256",
        "source_dataset_snapshot_identity_sha256",
        "creation_tool",
        "creation_tool_version",
    }
    if set(identity) != required:
        raise ValueError("identity JSON schema is invalid")
    if (
        identity["dataset_id"] != DATASET_ID
        or identity["manifest_version"] != MANIFEST_VERSION
        or identity["schema_version"] != IDENTITY_SCHEMA_VERSION
        or identity["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION
        or identity["manifest_relative_path"] != canonical_manifest_relative_path
        or identity["path_base_semantics"] != "relative_to_supplied_dataset_root"
        or identity["path_separator"] != "/"
        or identity["creation_tool"] != CREATION_TOOL
        or identity["creation_tool_version"] != CREATION_TOOL_VERSION
    ):
        raise ValueError("identity JSON fixed metadata is invalid")
    expected_values = {
        "row_count": expectations.row_count,
        "speaker_count": expectations.speaker_count,
        "unique_audio_path_count": expectations.row_count,
        "total_bytes": expectations.total_bytes,
        "total_duration_seconds": int(expectations.total_duration_seconds),
        "readable_wav_count": expectations.readable_wav_count,
        "duplicate_group_count": expectations.duplicate_group_count,
        "duplicate_file_count": expectations.duplicate_file_count,
        "cross_speaker_duplicate_group_count": (
            expectations.cross_speaker_duplicate_group_count
        ),
        "singleton_speaker_count": expectations.singleton_speaker_count,
        "provenance_counts": dict(expectations.provenance_counts),
    }
    for key, value in expected_values.items():
        if identity[key] != value:
            raise ValueError(f"identity JSON value mismatch: {key}")
    if (
        not SHA256_PATTERN.fullmatch(str(identity["manifest_sha256"]))
        or identity["manifest_sha256"] != file_sha256(manifest_path)
    ):
        raise ValueError("identity manifest hash is invalid")
    source_expected = {
        "source_audit_report_relative_path": expected_audit.report_relative_path,
        "source_audit_report_sha256": expected_audit.report_sha256,
        "source_audit_json_relative_path": expected_audit.json_relative_path,
        "source_audit_json_sha256": expected_audit.json_sha256,
        "source_audit_inventory_relative_path": (
            expected_audit.inventory_relative_path
        ),
        "source_audit_inventory_sha256": expected_audit.inventory_sha256,
        "source_dataset_snapshot_identity_sha256": (
            expected_audit.dataset_snapshot.identity_sha256
        ),
    }
    for key, value in source_expected.items():
        if identity[key] != value:
            raise ValueError(f"identity source-audit mismatch: {key}")
    for key in (
        "source_audit_report_sha256",
        "source_audit_json_sha256",
        "source_audit_inventory_sha256",
        "source_dataset_snapshot_identity_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(identity[key])):
            raise ValueError(f"identity hash is malformed: {key}")
    _assert_no_absolute_strings(identity)


def _safe_remove_staging(path: Path, parent: Path) -> None:
    try:
        resolved_parent = parent.resolve(strict=True)
        resolved = path.resolve(strict=False)
        resolved.relative_to(resolved_parent)
    except (OSError, ValueError):
        return
    if path.is_dir():
        shutil.rmtree(path)


def publish_artifact_pair(
    *,
    output_dir: Path,
    manifest_bytes: bytes,
    identity_bytes: bytes,
    validate_staged: Any,
) -> str:
    """Atomically publish a new directory or accept an identical existing pair."""
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        manifest = output_dir / MANIFEST_FILENAME
        identity = output_dir / IDENTITY_FILENAME
        if (
            output_dir.is_dir()
            and manifest.is_file()
            and identity.is_file()
            and manifest.read_bytes() == manifest_bytes
            and identity.read_bytes() == identity_bytes
        ):
            validate_staged(manifest, identity)
            return "already_present_byte_identical"
        raise FileExistsError(
            f"output directory exists with non-identical contents: {output_dir}"
        )

    staging = parent / (
        f".{output_dir.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    staging.mkdir()
    try:
        manifest = staging / MANIFEST_FILENAME
        identity = staging / IDENTITY_FILENAME
        manifest.write_bytes(manifest_bytes)
        identity.write_bytes(identity_bytes)
        validate_staged(manifest, identity)
        os.replace(staging, output_dir)
        return "published_atomically"
    finally:
        if staging.exists():
            _safe_remove_staging(staging, parent)


def create_full_manifest(
    *,
    project_root: Path,
    dataset_root: Path,
    inventory_path: Path,
    audit_report_path: Path,
    audit_json_path: Path,
    output_dir: Path,
    canonical_manifest_relative_path: str = CANONICAL_MANIFEST_RELATIVE_PATH,
    expectations: ManifestExpectations = APPROVED_EXPECTATIONS,
    reference_dir: Path | None = None,
) -> dict[str, Any]:
    project = project_root.resolve(strict=True)
    dataset = dataset_root.resolve(strict=True)
    output = output_dir.resolve(strict=False)
    try:
        output.relative_to(project)
    except ValueError as error:
        raise ValueError("output directory must be inside the repository") from error
    try:
        output.relative_to(dataset)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside the dataset root")
    validate_portable_path(canonical_manifest_relative_path)

    audit = load_approved_audit(
        project_root=project,
        inventory_path=inventory_path.resolve(strict=True),
        audit_report_path=audit_report_path.resolve(strict=True),
        audit_json_path=audit_json_path.resolve(strict=True),
        expectations=expectations,
    )
    source_hashes_before = (
        audit.inventory_sha256,
        audit.report_sha256,
        audit.json_sha256,
    )
    rows = load_audit_inventory(inventory_path)
    counts = validate_manifest_rows(rows, expectations)
    dataset_snapshot_before = reconcile_dataset(
        rows, dataset, audit.dataset_snapshot
    )

    manifest_bytes = render_manifest_csv(rows)
    manifest_hash = bytes_sha256(manifest_bytes)
    identity = build_identity(
        manifest_sha256=manifest_hash,
        canonical_manifest_relative_path=canonical_manifest_relative_path,
        counts=counts,
        audit=audit,
    )
    identity_bytes = render_identity_json(identity)

    def validate_pair(manifest_path: Path, identity_path: Path) -> None:
        persisted_rows = read_manifest(manifest_path)
        validate_manifest_rows(persisted_rows, expectations)
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("persisted manifest bytes differ from deterministic render")
        with identity_path.open("r", encoding="utf-8") as stream:
            persisted_identity = json.load(stream)
        if not isinstance(persisted_identity, Mapping):
            raise ValueError("persisted identity JSON must contain an object")
        validate_identity(
            persisted_identity,
            manifest_path,
            expectations,
            audit,
            canonical_manifest_relative_path,
        )
        if identity_path.read_bytes() != identity_bytes:
            raise ValueError("persisted identity bytes differ from deterministic render")

    publication = publish_artifact_pair(
        output_dir=output,
        manifest_bytes=manifest_bytes,
        identity_bytes=identity_bytes,
        validate_staged=validate_pair,
    )
    final_manifest = output / MANIFEST_FILENAME
    final_identity = output / IDENTITY_FILENAME
    validate_pair(final_manifest, final_identity)

    source_hashes_after = (
        file_sha256(inventory_path),
        file_sha256(audit_report_path),
        file_sha256(audit_json_path),
    )
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("approved audit inputs changed during manifest creation")
    dataset_snapshot_after = snapshot_dataset(dataset)
    if (
        dataset_snapshot_after != dataset_snapshot_before
        or dataset_snapshot_after != audit.dataset_snapshot
    ):
        raise RuntimeError("dataset changed during manifest creation")

    reproducibility = {
        "reference_compared": reference_dir is not None,
        "manifest_byte_identical": None,
        "identity_byte_identical": None,
    }
    if reference_dir is not None:
        reference_manifest = reference_dir / MANIFEST_FILENAME
        reference_identity = reference_dir / IDENTITY_FILENAME
        if not reference_manifest.is_file() or not reference_identity.is_file():
            raise FileNotFoundError("reference manifest artifact pair is incomplete")
        reproducibility["manifest_byte_identical"] = (
            reference_manifest.read_bytes() == final_manifest.read_bytes()
        )
        reproducibility["identity_byte_identical"] = (
            reference_identity.read_bytes() == final_identity.read_bytes()
        )
        if not all(
            (
                reproducibility["manifest_byte_identical"],
                reproducibility["identity_byte_identical"],
            )
        ):
            raise ValueError("reproduction artifacts differ from approved reference")

    return {
        "result": "PASS",
        "publication": publication,
        "manifest_relative_path": canonical_manifest_relative_path,
        "manifest_sha256": file_sha256(final_manifest),
        "identity_sha256": file_sha256(final_identity),
        "counts": counts,
        "source_audit_hashes": {
            "report": audit.report_sha256,
            "json": audit.json_sha256,
            "inventory": audit.inventory_sha256,
        },
        "dataset_snapshot_identity_sha256": audit.dataset_snapshot.identity_sha256,
        "dataset_preservation_match": True,
        "reproducibility": reproducibility,
        "formatting": {
            "csv_encoding": "UTF-8 without BOM",
            "csv_newline": "LF",
            "csv_column_order": list(MANIFEST_FIELDS),
            "duration_serialization": "fixed six decimal places",
            "json_encoding": "UTF-8 without BOM",
            "json_newline": "LF",
            "json_key_order": "lexicographic sort_keys",
            "json_indent": 2,
            "timestamps_in_identity": False,
        },
    }
