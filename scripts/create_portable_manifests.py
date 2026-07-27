"""Create deterministic, dataset-root-relative manifests from approved split v1."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TRUSTED_GROUPS = frozenset({"train", "train_small"})
QUARANTINE_GROUPS = frozenset({"test", "part"})
OUTPUT_SPLITS = ("train", "validation", "test")
FIELDS = (
    "relative_audio_path",
    "speaker_id",
    "speaker_label",
    "final_split",
    "filename_group",
)
EXPECTED_ROWS = {"train": 31_998, "validation": 8_504, "test": 9_198}
EXPECTED_SPEAKERS = {"train": 488, "validation": 100, "test": 100}


@dataclass(frozen=True)
class BuildResult:
    rows: dict[str, list[dict[str, str | int]]]
    labels: dict[str, int]
    input_rows: int
    input_speakers: int
    columns: tuple[str, str, str]
    excluded_selected: int
    quarantine_selected: int
    unresolved_paths: int


def detect_column(fieldnames: list[str] | None, candidates: tuple[str, ...], role: str) -> str:
    fields = fieldnames or []
    folded = {field.strip().casefold(): field for field in fields}
    matches = [folded[name] for name in candidates if name in folded]
    if len(matches) != 1:
        raise ValueError(
            f"could not uniquely detect {role} column; candidates={candidates}, fields={fields}"
        )
    return matches[0]


def speaker_sort_key(speaker_id: str) -> tuple[int, int | str, str]:
    try:
        return (0, int(speaker_id), speaker_id)
    except ValueError:
        return (1, speaker_id.casefold(), speaker_id)


def relative_wav_path(audio_path: str, dataset_root: Path) -> str:
    raw = audio_path.strip()
    if not raw:
        raise ValueError("audio path is empty")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"input audio path is not absolute: {raw}")
    root = dataset_root.resolve(strict=True)
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"audio path is outside dataset root: {raw}") from error
    portable = relative.as_posix()
    pure = PurePosixPath(portable)
    if pure.is_absolute() or ".." in pure.parts or not portable:
        raise ValueError(f"unsafe relative audio path: {portable}")
    return portable


def read_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        speaker_col = detect_column(
            reader.fieldnames, ("speaker_id", "speaker", "spk_id", "speakerid"), "speaker ID"
        )
        split_col = detect_column(reader.fieldnames, ("final_split", "split"), "final split")
        for number, row in enumerate(reader, start=2):
            speaker_id = row[speaker_col].strip()
            split = row[split_col].strip().casefold()
            if not speaker_id or split not in {*OUTPUT_SPLITS, "excluded"}:
                raise ValueError(f"invalid speaker assignment at row {number}")
            if speaker_id in assignments:
                raise ValueError(f"duplicate speaker assignment at row {number}: {speaker_id}")
            assignments[speaker_id] = split
    if not assignments:
        raise ValueError("speaker split contains no assignments")
    return assignments


def build_manifests(
    full_manifest: Path,
    speaker_split: Path,
    dataset_root: Path,
    expected_rows: dict[str, int] | None = None,
    expected_speakers: dict[str, int] | None = None,
) -> BuildResult:
    assignments = read_assignments(speaker_split)
    split_speakers = {
        split: {sid for sid, assigned in assignments.items() if assigned == split}
        for split in OUTPUT_SPLITS
    }
    labels = {
        speaker_id: label
        for label, speaker_id in enumerate(sorted(split_speakers["train"], key=speaker_sort_key))
    }
    rows: dict[str, list[dict[str, str | int]]] = {split: [] for split in OUTPUT_SPLITS}
    seen_paths: dict[str, tuple[str, int]] = {}
    all_speakers: set[str] = set()
    trusted_selected: Counter[str] = Counter()
    excluded_selected = 0
    quarantine_selected = 0
    unresolved: list[str] = []

    with full_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        path_col = detect_column(
            reader.fieldnames, ("audio_path", "wav_path", "path", "file"), "WAV path"
        )
        speaker_col = detect_column(
            reader.fieldnames, ("speaker_id", "speaker", "spk_id", "speakerid"), "speaker ID"
        )
        group_col = detect_column(
            reader.fieldnames,
            ("filename_group", "provenance_group", "group", "source_group"),
            "provenance group",
        )
        input_rows = 0
        for number, source in enumerate(reader, start=2):
            input_rows += 1
            speaker_id = source[speaker_col].strip()
            group = source[group_col].strip().casefold()
            all_speakers.add(speaker_id)
            assigned = assignments.get(speaker_id)
            if group in QUARANTINE_GROUPS:
                if assigned in OUTPUT_SPLITS:
                    quarantine_selected += 1
                continue
            if group not in TRUSTED_GROUPS:
                raise ValueError(f"unsupported provenance group at row {number}: {group!r}")
            if assigned is None:
                raise ValueError(f"trusted speaker has no approved assignment: {speaker_id}")
            if assigned == "excluded":
                excluded_selected += 1
                continue
            relative = relative_wav_path(source[path_col], dataset_root)
            path_key = os.path.normcase(relative)
            if path_key in seen_paths:
                prior_split, prior_row = seen_paths[path_key]
                raise ValueError(
                    f"duplicate audio path at rows {prior_row} ({prior_split}) and "
                    f"{number} ({assigned}): {relative}"
                )
            seen_paths[path_key] = (assigned, number)
            resolved = dataset_root / Path(*PurePosixPath(relative).parts)
            if not resolved.is_file():
                unresolved.append(relative)
            rows[assigned].append(
                {
                    "relative_audio_path": relative,
                    "speaker_id": speaker_id,
                    "speaker_label": labels[speaker_id] if assigned == "train" else -1,
                    "final_split": assigned,
                    "filename_group": group,
                }
            )
            trusted_selected[assigned] += 1

    for split in OUTPUT_SPLITS:
        rows[split].sort(
            key=lambda row: (
                int(row["speaker_label"]),
                speaker_sort_key(str(row["speaker_id"])),
                str(row["relative_audio_path"]),
            )
        )

    validate(
        rows,
        labels,
        assignments,
        unresolved,
        expected_rows,
        expected_speakers,
    )
    return BuildResult(
        rows=rows,
        labels=labels,
        input_rows=input_rows,
        input_speakers=len(all_speakers),
        columns=(path_col, speaker_col, group_col),
        excluded_selected=0,
        quarantine_selected=0,
        unresolved_paths=len(unresolved),
    )


def validate(
    rows: dict[str, list[dict[str, str | int]]],
    labels: dict[str, int],
    assignments: dict[str, str],
    unresolved: list[str],
    expected_rows: dict[str, int] | None,
    expected_speakers: dict[str, int] | None,
) -> None:
    speakers = {
        split: {str(row["speaker_id"]) for row in rows[split]} for split in OUTPUT_SPLITS
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = speakers[left] & speakers[right]
        if overlap:
            raise ValueError(f"speaker leakage between {left} and {right}: {sorted(overlap)[:5]}")
    if set(labels) != speakers["train"]:
        raise ValueError("train label mapping does not exactly match train manifest speakers")
    label_values = set(labels.values())
    if label_values != set(range(len(labels))) or len(label_values) != len(labels):
        raise ValueError("train labels are not unique and contiguous from zero")
    if any(int(row["speaker_label"]) != labels[str(row["speaker_id"])] for row in rows["train"]):
        raise ValueError("a train row has an invalid speaker label")
    if any(int(row["speaker_label"]) != -1 for split in ("validation", "test") for row in rows[split]):
        raise ValueError("validation and test labels must all be -1")
    all_rows = [row for split in OUTPUT_SPLITS for row in rows[split]]
    if any(str(row["filename_group"]) not in TRUSTED_GROUPS for row in all_rows):
        raise ValueError("quarantined provenance appeared in an output manifest")
    if any(assignments[str(row["speaker_id"])] != row["final_split"] for row in all_rows):
        raise ValueError("manifest row disagrees with approved speaker assignment")
    paths = [str(row["relative_audio_path"]) for row in all_rows]
    if len(paths) != len({os.path.normcase(path) for path in paths}):
        raise ValueError("duplicate paths appear across generated manifests")
    for path in paths:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or Path(path).is_absolute():
            raise ValueError(f"generated path is not portable: {path}")
        if "\\" in path:
            raise ValueError(f"generated path does not use '/' separators: {path}")
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} selected WAV paths do not exist under dataset root; "
            f"first: {unresolved[0]}"
        )
    if expected_rows:
        actual = {split: len(rows[split]) for split in OUTPUT_SPLITS}
        if actual != expected_rows:
            raise ValueError(f"row counts do not match approved split: {actual} != {expected_rows}")
    if expected_speakers:
        actual = {split: len(speakers[split]) for split in OUTPUT_SPLITS}
        if actual != expected_speakers:
            raise ValueError(
                f"speaker counts do not match approved split: {actual} != {expected_speakers}"
            )


def config_document() -> dict:
    return {
        "manifest_version": 1,
        "path_base": "dataset_root_cli_argument",
        "path_separator": "/",
        "trusted_groups": sorted(TRUSTED_GROUPS),
        "quarantine_groups": sorted(QUARANTINE_GROUPS),
        "output_splits": list(OUTPUT_SPLITS),
        "train_label_order": "numeric_aware_speaker_id",
        "validation_test_label": -1,
        "expected_rows": EXPECTED_ROWS,
        "expected_speakers": EXPECTED_SPEAKERS,
    }


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_report(result: BuildResult) -> str:
    speaker_counts = {
        split: len({str(row["speaker_id"]) for row in result.rows[split]})
        for split in OUTPUT_SPLITS
    }
    return "\n".join(
        [
            "# Portable manifest v1 summary",
            "",
            "## Input",
            "",
            f"- Detected columns: `{result.columns[0]}`, `{result.columns[1]}`, `{result.columns[2]}`",
            "- Dataset root: supplied at runtime (not persisted for portability)",
            f"- Rows: {result.input_rows}",
            f"- Speakers: {result.input_speakers}",
            "",
            "## Output manifests",
            "",
            f"- Train: {len(result.rows['train'])} rows, {speaker_counts['train']} speakers",
            f"- Validation: {len(result.rows['validation'])} rows, {speaker_counts['validation']} speakers",
            f"- Test: {len(result.rows['test'])} rows, {speaker_counts['test']} speakers",
            "- Path schema: dataset-root-relative WAV path",
            "- Path separator: `/`",
            "- Path existence: PASS",
            "",
            "## Label mapping",
            "",
            f"- Mapped speakers: {len(result.labels)}",
            f"- Minimum label: {min(result.labels.values())}",
            f"- Maximum label: {max(result.labels.values())}",
            "- Unique and contiguous: PASS",
            "",
            "## Validation",
            "",
            "- Train/validation intersection: 0",
            "- Train/test intersection: 0",
            "- Validation/test intersection: 0",
            "- Excluded rows included: 0",
            "- Quarantine rows included: 0",
            "- Duplicate paths: 0",
            "- Absolute paths: 0",
            "- Unresolved paths: 0",
            "- Approved totals reconciled: PASS",
            "",
            "## Overall result",
            "",
            "PASS",
            "",
        ]
    )


def write_outputs(output_dir: Path, report: Path, result: BuildResult) -> None:
    for split in OUTPUT_SPLITS:
        write_csv(output_dir / f"{split}_manifest_v1.csv", result.rows[split])
    (output_dir / "speaker_to_label_v1.json").write_text(
        json.dumps(result.labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "portable_manifest_config_v1.json").write_text(
        json.dumps(config_document(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(result), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--speaker-split", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("manifests/portable"))
    parser.add_argument("--report", type=Path, default=Path("reports/portable_manifest_v1_summary.md"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_manifests(
        args.full_manifest.resolve(strict=True),
        args.speaker_split.resolve(strict=True),
        args.dataset_root.resolve(strict=True),
        EXPECTED_ROWS,
        EXPECTED_SPEAKERS,
    )
    print(render_report(result))
    if args.dry_run:
        print("DRY RUN: no output files written")
        return
    write_outputs(args.output_dir, args.report, result)
    print(f"Wrote portable manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
