"""Build the production dataset-definition package for adaptive_augmented_3s."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


PACKAGE_VERSION = "adaptive_augmented_3s_v1"
SCHEMA_VERSION = 1
DEFAULT_SEED = 2026
EXPECTED_WAVS = 60_803
EXPECTED_SPEAKERS = 610
EXPECTED_SPLIT_SPEAKERS = {"train": 488, "validation": 61, "final_test": 61}
EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLES = 48_000
EXPECTED_DURATION_SEC = "3.0"
EXPECTED_SAMPLE_WIDTH = 2

FULL_MANIFEST = "manifests/adaptive_augmented_3s_v1_full_manifest.csv"
DATASET_IDENTITY = "manifests/adaptive_augmented_3s_v1_dataset_identity.json"
TRAIN_MANIFEST = "manifests/adaptive_augmented_3s_v1_train_manifest.csv"
VALIDATION_MANIFEST = "manifests/adaptive_augmented_3s_v1_validation_manifest.csv"
FINAL_TEST_MANIFEST = "manifests/adaptive_augmented_3s_v1_final_test_manifest.csv"
LABEL_MAPPING = "manifests/adaptive_augmented_3s_v1_speaker_to_label.json"
SPEAKER_SPLIT = "splits/adaptive_augmented_3s_v1_speaker_split.csv"
SPLIT_IDENTITY = "splits/adaptive_augmented_3s_v1_split_identity.json"

FULL_FIELDS = (
    "audio_path",
    "speaker_id",
    "sample_rate",
    "num_channels",
    "num_samples",
    "duration_sec",
)
SPLIT_FIELDS = ("speaker_id", "split", "ranking_sha256")
PORTABLE_FIELDS = (
    "relative_audio_path",
    "speaker_id",
    "speaker_label",
    "final_split",
)
@dataclass(frozen=True)
class ManifestRow:
    audio_path: str
    speaker_id: str
    sample_rate: int
    num_channels: int
    num_samples: int
    duration_sec: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def render_json(value: Any) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def render_csv(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def portable_path(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or ":" in value
        or pure.is_absolute()
        or ".." in pure.parts
        or len(pure.parts) != 2
    ):
        raise ValueError(f"invalid portable dataset path: {value!r}")
    return value


def speaker_sort_key(speaker_id: str) -> tuple[int, int | str, str]:
    """Match the repository's established numeric-aware speaker ordering."""
    try:
        return (0, int(speaker_id), speaker_id)
    except ValueError:
        return (1, speaker_id.casefold(), speaker_id)


def snapshot_dataset(dataset_root: Path) -> str:
    """Hash path, size, and mtime without hashing WAV contents."""
    digest = hashlib.sha256()
    for path in sorted(
        (item for item in dataset_root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(dataset_root).as_posix(),
    ):
        stat = path.stat()
        relative = path.relative_to(dataset_root).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def discover_wavs(dataset_root: Path) -> list[Path]:
    wavs = [
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".wav"
    ]
    wavs.sort(key=lambda path: path.relative_to(dataset_root).as_posix())
    if len(wavs) != EXPECTED_WAVS:
        raise ValueError(f"found {len(wavs)} WAV files; expected {EXPECTED_WAVS}")
    return wavs


def inspect_wav(path: Path, dataset_root: Path) -> ManifestRow:
    relative = portable_path(path.relative_to(dataset_root).as_posix())
    speaker_id = path.parent.name
    if PurePosixPath(relative).parent.name != speaker_id:
        raise ValueError(f"direct-parent speaker mismatch: {relative}")
    try:
        with wave.open(str(path), "rb") as stream:
            sample_rate = stream.getframerate()
            channels = stream.getnchannels()
            samples = stream.getnframes()
            sample_width = stream.getsampwidth()
            compression = stream.getcomptype()
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"cannot read WAV header {relative}: {error}") from error

    duration = samples / sample_rate if sample_rate else 0.0
    violations: list[str] = []
    if sample_rate != EXPECTED_SAMPLE_RATE:
        violations.append(f"sample_rate={sample_rate}")
    if channels != EXPECTED_CHANNELS:
        violations.append(f"channels={channels}")
    if samples != EXPECTED_SAMPLES:
        violations.append(f"samples={samples}")
    if duration != float(EXPECTED_DURATION_SEC):
        violations.append(f"duration_sec={duration}")
    if sample_width != EXPECTED_SAMPLE_WIDTH:
        violations.append(f"sample_width={sample_width}")
    if compression != "NONE":
        violations.append(f"compression={compression}")
    if violations:
        raise ValueError(f"audio contract violation for {relative}: {', '.join(violations)}")
    return ManifestRow(
        audio_path=relative,
        speaker_id=speaker_id,
        sample_rate=sample_rate,
        num_channels=channels,
        num_samples=samples,
        duration_sec=EXPECTED_DURATION_SEC,
    )


def build_inventory(dataset_root: Path) -> list[ManifestRow]:
    rows = [inspect_wav(path, dataset_root) for path in discover_wavs(dataset_root)]
    rows.sort(key=lambda row: (speaker_sort_key(row.speaker_id), row.audio_path))
    paths = [row.audio_path for row in rows]
    if len(paths) != len(set(paths)) or len(paths) != len({path.casefold() for path in paths}):
        raise ValueError("duplicate or case-colliding relative WAV paths found")
    speakers = {row.speaker_id for row in rows}
    if len(speakers) != EXPECTED_SPEAKERS:
        raise ValueError(f"found {len(speakers)} speakers; expected {EXPECTED_SPEAKERS}")
    return rows


def read_provenance_paths(path: Path) -> set[str]:
    paths: set[str] = set()
    folded_paths: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"relative_path", "speaker_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"normalization provenance missing columns: {sorted(missing)}")
        for line, raw in enumerate(reader, start=2):
            relative = portable_path(raw["relative_path"].strip())
            speaker_id = raw["speaker_id"].strip()
            if PurePosixPath(relative).parent.name != speaker_id:
                raise ValueError(f"provenance speaker mismatch at line {line}: {relative}")
            folded = relative.casefold()
            if relative in paths or folded in folded_paths:
                raise ValueError(f"duplicate provenance path at line {line}: {relative}")
            paths.add(relative)
            folded_paths.add(folded)
    return paths


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def validate_lineage(
    dataset_root: Path, rows: Sequence[ManifestRow]
) -> dict[str, str | int]:
    provenance_path = dataset_root / "normalization_provenance.csv"
    summary_path = dataset_root / "normalization_summary.json"
    validation_path = dataset_root.parent / "adaptive_augmented_validation" / "validation_summary.json"
    for path in (provenance_path, summary_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing required preprocessing evidence: {path}")

    manifest_paths = {row.audio_path for row in rows}
    provenance_paths = read_provenance_paths(provenance_path)
    missing = manifest_paths - provenance_paths
    extra = provenance_paths - manifest_paths
    if missing or extra:
        raise ValueError(
            "provenance path set differs from WAV inventory: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    summary = read_json_object(summary_path)
    if (
        summary.get("total_files") != EXPECTED_WAVS
        or summary.get("processed_files") != EXPECTED_WAVS
        or summary.get("failure_count") != 0
    ):
        raise ValueError("normalization summary does not record a complete zero-failure run")
    config = summary.get("configuration", {})
    if (
        config.get("target_sample_rate") != EXPECTED_SAMPLE_RATE
        or config.get("target_channels") != EXPECTED_CHANNELS
        or config.get("target_samples") != EXPECTED_SAMPLES
        or config.get("target_duration_seconds") != float(EXPECTED_DURATION_SEC)
    ):
        raise ValueError("normalization summary target contract mismatch")

    validation = read_json_object(validation_path)
    expected = validation.get("expected", {})
    if (
        validation.get("passed") is not True
        or validation.get("failure_count") != 0
        or validation.get("total_wav_files") != EXPECTED_WAVS
        or validation.get("readable_wav_files") != EXPECTED_WAVS
        or expected.get("sample_rate") != EXPECTED_SAMPLE_RATE
        or expected.get("channels") != EXPECTED_CHANNELS
        or expected.get("samples") != EXPECTED_SAMPLES
        or expected.get("duration_seconds") != float(EXPECTED_DURATION_SEC)
    ):
        raise ValueError("external validation summary does not confirm the dataset contract")
    return {
        "normalization_provenance_row_count": len(provenance_paths),
        "normalization_provenance_sha256": file_sha256(provenance_path),
        "normalization_summary_sha256": file_sha256(summary_path),
        "validation_summary_sha256": file_sha256(validation_path),
    }


def ranking_digest(seed: int, speaker_id: str) -> str:
    return hashlib.sha256(f"{seed}|{speaker_id}".encode("utf-8")).hexdigest()


def build_split(rows: Sequence[ManifestRow], seed: int) -> list[dict[str, str]]:
    speakers = sorted({row.speaker_id for row in rows})
    ranked = sorted(speakers, key=lambda speaker: (ranking_digest(seed, speaker), speaker))
    assignments: dict[str, str] = {}
    for index, speaker in enumerate(ranked):
        if index < EXPECTED_SPLIT_SPEAKERS["train"]:
            split = "train"
        elif index < EXPECTED_SPLIT_SPEAKERS["train"] + EXPECTED_SPLIT_SPEAKERS["validation"]:
            split = "validation"
        else:
            split = "final_test"
        assignments[speaker] = split
    return [
        {
            "speaker_id": speaker,
            "split": assignments[speaker],
            "ranking_sha256": ranking_digest(seed, speaker),
        }
        for speaker in sorted(speakers, key=speaker_sort_key)
    ]


def build_labels(split_rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    train_speakers = sorted(
        (row["speaker_id"] for row in split_rows if row["split"] == "train"),
        key=speaker_sort_key,
    )
    return {speaker: label for label, speaker in enumerate(train_speakers)}


def build_portable_manifests(
    rows: Sequence[ManifestRow],
    split_rows: Sequence[Mapping[str, str]],
    labels: Mapping[str, int],
) -> dict[str, list[dict[str, str | int]]]:
    assignments = {row["speaker_id"]: row["split"] for row in split_rows}
    result: dict[str, list[dict[str, str | int]]] = {
        split: [] for split in EXPECTED_SPLIT_SPEAKERS
    }
    for row in rows:
        split = assignments[row.speaker_id]
        result[split].append(
            {
                "relative_audio_path": row.audio_path,
                "speaker_id": row.speaker_id,
                "speaker_label": labels[row.speaker_id] if split == "train" else -1,
                "final_split": split,
            }
        )
    for split, manifest_rows in result.items():
        manifest_rows.sort(
            key=lambda row: (
                speaker_sort_key(str(row["speaker_id"])),
                str(row["relative_audio_path"]),
            )
        )
    return result


def canonical_manifest_identity(rows: Sequence[ManifestRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            asdict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update((payload + "\n").encode("utf-8"))
    return digest.hexdigest()


def validate_package(
    rows: Sequence[ManifestRow],
    split_rows: Sequence[Mapping[str, str]],
    manifests: Mapping[str, Sequence[Mapping[str, str | int]]],
    labels: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    all_speakers = {row.speaker_id for row in rows}
    assignments = {row["speaker_id"]: row["split"] for row in split_rows}
    if len(assignments) != EXPECTED_SPEAKERS or set(assignments) != all_speakers:
        raise ValueError("speaker split does not assign every speaker exactly once")
    speaker_sets = {
        split: {speaker for speaker, assigned in assignments.items() if assigned == split}
        for split in EXPECTED_SPLIT_SPEAKERS
    }
    speaker_counts = {split: len(values) for split, values in speaker_sets.items()}
    if speaker_counts != EXPECTED_SPLIT_SPEAKERS:
        raise ValueError(f"speaker split counts mismatch: {speaker_counts}")
    names = tuple(EXPECTED_SPLIT_SPEAKERS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if speaker_sets[left] & speaker_sets[right]:
                raise ValueError(f"speaker overlap between {left} and {right}")
    if set().union(*speaker_sets.values()) != all_speakers:
        raise ValueError("split speaker union does not equal the dataset speaker set")

    all_manifest_rows = [row for split in names for row in manifests[split]]
    paths = [str(row["relative_audio_path"]) for row in all_manifest_rows]
    if len(paths) != EXPECTED_WAVS or len(paths) != len(set(paths)):
        raise ValueError("split manifests contain dropped or duplicated audio paths")
    if set(paths) != {row.audio_path for row in rows}:
        raise ValueError("split manifest path union differs from the full manifest")
    for split in names:
        for row in manifests[split]:
            speaker = str(row["speaker_id"])
            if assignments.get(speaker) != split or row["final_split"] != split:
                raise ValueError("a split manifest row disagrees with speaker assignment")
            portable_path(str(row["relative_audio_path"]))
            label = int(row["speaker_label"])
            if split == "train" and labels.get(speaker) != label:
                raise ValueError("a train row disagrees with the label mapping")
            if split != "train" and label != -1:
                raise ValueError("an evaluation row has a classifier label")

    train_speakers = speaker_sets["train"]
    if set(labels) != train_speakers or set(labels.values()) != set(range(488)):
        raise ValueError("train labels are not a complete contiguous 0..487 mapping")
    if set(labels) & (speaker_sets["validation"] | speaker_sets["final_test"]):
        raise ValueError("an evaluation speaker appears in train labels")
    row_counts = {split: len(manifests[split]) for split in names}
    return {"speaker_counts": speaker_counts, "row_counts": row_counts}


def build_payloads(
    rows: Sequence[ManifestRow],
    lineage_hashes: Mapping[str, str | int],
    seed: int,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    split_rows = build_split(rows, seed)
    labels = build_labels(split_rows)
    manifests = build_portable_manifests(rows, split_rows, labels)
    counts = validate_package(rows, split_rows, manifests, labels)

    payloads: dict[str, bytes] = {
        FULL_MANIFEST: render_csv(FULL_FIELDS, (asdict(row) for row in rows)),
        SPEAKER_SPLIT: render_csv(SPLIT_FIELDS, split_rows),
        TRAIN_MANIFEST: render_csv(PORTABLE_FIELDS, manifests["train"]),
        VALIDATION_MANIFEST: render_csv(PORTABLE_FIELDS, manifests["validation"]),
        FINAL_TEST_MANIFEST: render_csv(PORTABLE_FIELDS, manifests["final_test"]),
        LABEL_MAPPING: render_json(dict(labels)),
    }
    dataset_identity = {
        "schema_version": SCHEMA_VERSION,
        "identity_kind": "adaptive_augmented_3s_dataset",
        "package_version": PACKAGE_VERSION,
        "creation_tool": "scripts/create_adaptive_augmented_3s_package.py",
        "creation_tool_version": "adaptive_augmented_3s_package_schema_1",
        "path_base": "dataset_root_cli_argument",
        "path_separator": "/",
        "wav_content_hashing": False,
        "canonical_manifest_identity_method": "sorted compact JSON rows with LF",
        "canonical_manifest_identity_sha256": canonical_manifest_identity(rows),
        "full_manifest_path": FULL_MANIFEST,
        "full_manifest_sha256": bytes_sha256(payloads[FULL_MANIFEST]),
        "normalization_provenance_sha256": lineage_hashes[
            "normalization_provenance_sha256"
        ],
        "normalization_provenance_row_count": lineage_hashes[
            "normalization_provenance_row_count"
        ],
        "normalization_summary_sha256": lineage_hashes["normalization_summary_sha256"],
        "validation_summary_sha256": lineage_hashes["validation_summary_sha256"],
        "wav_count": len(rows),
        "speaker_count": len({row.speaker_id for row in rows}),
        "total_duration_sec": f"{len(rows) * float(EXPECTED_DURATION_SEC):.1f}",
        "audio_contract": {
            "sample_rate": EXPECTED_SAMPLE_RATE,
            "num_channels": EXPECTED_CHANNELS,
            "num_samples": EXPECTED_SAMPLES,
            "duration_sec": EXPECTED_DURATION_SEC,
            "sample_width_bytes": EXPECTED_SAMPLE_WIDTH,
            "compression_type": "NONE",
        },
        "provenance_path_set_matches_inventory": True,
        "source_audio_modified": False,
    }
    payloads[DATASET_IDENTITY] = render_json(dataset_identity)
    artifact_hashes = {path: bytes_sha256(payload) for path, payload in payloads.items()}
    split_identity = {
        "schema_version": SCHEMA_VERSION,
        "identity_kind": "adaptive_augmented_3s_speaker_split",
        "package_version": PACKAGE_VERSION,
        "dataset_identity_path": DATASET_IDENTITY,
        "dataset_identity_sha256": artifact_hashes[DATASET_IDENTITY],
        "full_manifest_path": FULL_MANIFEST,
        "full_manifest_sha256": artifact_hashes[FULL_MANIFEST],
        "speaker_split_path": SPEAKER_SPLIT,
        "speaker_split_sha256": artifact_hashes[SPEAKER_SPLIT],
        "split_algorithm": "sort by SHA256(UTF8('<seed>|<exact speaker_id>')), then exact speaker_id",
        "split_seed": seed,
        "split_ranges": {
            "train": [0, 487],
            "validation": [488, 548],
            "final_test": [549, 609],
        },
        "speaker_counts": counts["speaker_counts"],
        "row_counts": counts["row_counts"],
        "speaker_sets_pairwise_disjoint": True,
        "all_dataset_speakers_assigned": True,
        "all_dataset_rows_assigned_once": True,
        "manifest_paths": {
            "train": TRAIN_MANIFEST,
            "validation": VALIDATION_MANIFEST,
            "final_test": FINAL_TEST_MANIFEST,
        },
        "manifest_sha256": {
            "train": artifact_hashes[TRAIN_MANIFEST],
            "validation": artifact_hashes[VALIDATION_MANIFEST],
            "final_test": artifact_hashes[FINAL_TEST_MANIFEST],
        },
        "speaker_to_label_path": LABEL_MAPPING,
        "speaker_to_label_sha256": artifact_hashes[LABEL_MAPPING],
        "train_label_order": "numeric-aware exact speaker_id ascending",
        "train_label_range": [0, 487],
        "evaluation_label_sentinel": -1,
        "final_test_locked_after_package_acceptance": True,
        "timestamps_in_identity": False,
    }
    payloads[SPLIT_IDENTITY] = render_json(split_identity)
    return payloads, {"dataset": dataset_identity, "split": split_identity}


def publish_payloads(project_root: Path, payloads: Mapping[str, bytes]) -> str:
    existing = [path for path in payloads if (project_root / path).exists()]
    if existing and len(existing) != len(payloads):
        missing = sorted(set(payloads) - set(existing))
        raise ValueError(f"partial package exists; refusing publication; missing={missing}")
    if existing:
        conflicts = [
            path for path, payload in payloads.items() if (project_root / path).read_bytes() != payload
        ]
        if conflicts:
            raise ValueError(f"existing normalized-dataset package conflicts: {conflicts}")
        return "verified_existing"

    written: list[Path] = []
    try:
        for relative, payload in payloads.items():
            target = project_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, target)
            written.append(target)
    except Exception:
        for target in written:
            target.unlink(missing_ok=True)
        raise
    return "created"


def validate_published(project_root: Path, payloads: Mapping[str, bytes]) -> None:
    for relative, expected in payloads.items():
        path = project_root / relative
        if path.read_bytes() != expected:
            raise ValueError(f"published artifact differs from deterministic payload: {relative}")
    dataset_identity = read_json_object(project_root / DATASET_IDENTITY)
    split_identity = read_json_object(project_root / SPLIT_IDENTITY)
    if dataset_identity["full_manifest_sha256"] != file_sha256(project_root / FULL_MANIFEST):
        raise ValueError("published dataset identity does not bind the full manifest")
    if split_identity["dataset_identity_sha256"] != file_sha256(project_root / DATASET_IDENTITY):
        raise ValueError("published split identity does not bind the dataset identity")
    if split_identity["speaker_split_sha256"] != file_sha256(project_root / SPEAKER_SPLIT):
        raise ValueError("published split identity does not bind the speaker split")


def create_package(dataset_root: Path, project_root: Path, seed: int) -> dict[str, Any]:
    if seed != DEFAULT_SEED:
        raise ValueError(f"production split seed must be {DEFAULT_SEED}, got {seed}")
    dataset_root = dataset_root.resolve(strict=True)
    project_root = project_root.resolve(strict=True)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    before_snapshot = snapshot_dataset(dataset_root)
    rows = build_inventory(dataset_root)
    lineage_hashes = validate_lineage(dataset_root, rows)
    payloads, identities = build_payloads(rows, lineage_hashes, seed)
    repeated_payloads, _ = build_payloads(rows, lineage_hashes, seed)
    if payloads != repeated_payloads:
        raise ValueError("repeated in-process generation was not byte-identical")
    publication = publish_payloads(project_root, payloads)
    validate_published(project_root, payloads)
    after_snapshot = snapshot_dataset(dataset_root)
    if before_snapshot != after_snapshot:
        raise ValueError("dataset path/size/mtime snapshot changed during package creation")
    return {
        "result": "PASS",
        "publication": publication,
        "dataset_snapshot_preserved": True,
        "repeated_generation_byte_identical": True,
        "output_sha256": {
            path: bytes_sha256(payload) for path, payload in payloads.items()
        },
        "output_sizes": {path: len(payload) for path, payload in payloads.items()},
        "dataset_identity": identities["dataset"],
        "split_identity": identities["split"],
    }
