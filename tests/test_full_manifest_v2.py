"""Focused synthetic tests for production full manifest v2 construction."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path
from unittest import mock

from src.dataset_audit_v2 import INVENTORY_FIELDS, snapshot_dataset
from src.full_manifest_v2 import (
    FORBIDDEN_MANIFEST_FIELDS,
    IDENTITY_FILENAME,
    MANIFEST_FIELDS,
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    ManifestExpectations,
    build_identity,
    create_full_manifest,
    file_sha256,
    load_audit_inventory,
    publish_artifact_pair,
    read_manifest,
    render_identity_json,
    render_manifest_csv,
    validate_identity,
    validate_manifest_rows,
    validate_portable_path,
)


class FullManifestV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        self.dataset = self.project / "dataset"
        self.dataset.mkdir()
        (self.project / "reports").mkdir()
        (self.project / "outputs" / "dataset_understanding_v2").mkdir(
            parents=True
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def add_file(
        self,
        speaker: str,
        filename: str,
        provenance: str,
        *,
        duplicate_group: str = "",
        content: bytes = b"wav-data",
    ) -> dict[str, str]:
        path = self.dataset / speaker / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        candidate = filename[:-4].replace("aug_", "", 1)
        source_group = candidate.replace("-", "/", 1).replace("_", "/", 1)
        row = {field: "" for field in INVENTORY_FIELDS}
        row.update(
            {
                "dataset_relative_path": f"{speaker}/{filename}",
                "filename": filename,
                "extension": ".wav",
                "file_size_bytes": str(len(content)),
                "direct_parent_folder": speaker,
                "candidate_speaker_id": speaker,
                "relative_depth": "2",
                "placement_status": "expected_depth",
                "speaker_folder_status": "valid_numeric_top_level",
                "provenance_class": provenance,
                "provenance_parse_status": "exact_case_supported",
                "candidate_source_group": source_group,
                "source_group_parse_status": "exact_pattern_supported",
                "wav_read_status": "readable",
                "wav_encoding": "pcm",
                "sample_rate_hz": "16000",
                "channel_count": "1",
                "sample_width_bytes": "2",
                "bits_per_sample": "16",
                "frame_count": "48000",
                "duration_seconds": "3.0",
                "zero_byte_file": "False",
                "zero_frame_audio": "False",
                "duration_outlier_status": "not_outlier",
                "exact_duplicate_group": duplicate_group,
                "audit_exclusion_recommendation": "none",
            }
        )
        return row

    def write_fixture(self) -> tuple[Path, Path, Path, ManifestExpectations]:
        rows = [
            self.add_file(
                "10",
                "aug_train-00001-of-00002_7.wav",
                "train",
            ),
            self.add_file(
                "2",
                "aug_train_small-00001-of-00002_9.wav",
                "train_small",
                duplicate_group="dup_000001",
                content=b"duplicate",
            ),
            self.add_file(
                "2",
                "aug_part-00002-of-00002_1.wav",
                "part",
                duplicate_group="dup_000001",
                content=b"duplicate",
            ),
            self.add_file(
                "30",
                "aug_test-00001-of-00001_30.wav",
                "test",
            ),
        ]
        inventory = (
            self.project
            / "outputs"
            / "dataset_understanding_v2"
            / "dataset_audit_inventory_nonproduction_v2.csv"
        )
        with inventory.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=INVENTORY_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        snapshot = snapshot_dataset(self.dataset)
        expectations = ManifestExpectations(
            row_count=4,
            speaker_count=3,
            total_bytes=sum(
                path.stat().st_size for path in self.dataset.rglob("*.wav")
            ),
            total_duration_seconds=Decimal("12"),
            readable_wav_count=4,
            duplicate_group_count=1,
            duplicate_file_count=2,
            cross_speaker_duplicate_group_count=0,
            singleton_speaker_count=2,
            provenance_counts={
                "train": 1,
                "train_small": 1,
                "part": 1,
                "test": 1,
            },
        )
        audit_json = self.project / "reports" / "dataset_understanding_v2.json"
        audit_payload = {
            "schema_name": "dataset_understanding_v2",
            "schema_version": 2,
            "result": "PASS",
            "scan_complete": True,
            "counts": {
                "total_file_count": 4,
                "wav_file_count": 4,
                "non_wav_file_count": 0,
                "total_byte_size": expectations.total_bytes,
                "valid_candidate_speaker_count": 3,
                "valid_readable_wav_count": 4,
                "invalid_unreadable_wav_count": 0,
                "zero_byte_wav_count": 0,
                "zero_frame_wav_count": 0,
                "total_duration_seconds": 12.0,
            },
            "duplicates": {
                "exact_duplicate_group_count": 1,
                "files_in_exact_duplicate_groups": 2,
                "cross_speaker_exact_duplicate_group_count": 0,
            },
            "speakers": {"threshold_counts": {"fewer_than_2": 2}},
            "provenance": {
                "summary": [
                    {"provenance_class": name, "file_count": count}
                    for name, count in expectations.provenance_counts.items()
                ]
            },
            "reconciliation": {
                "rows": True,
                "bytes": True,
                "speakers": True,
            },
            "preservation": {
                "snapshots_match": True,
                "post_snapshot": {
                    **snapshot.__dict__,
                    "errors": list(snapshot.errors),
                },
            },
            "artifact_paths": {
                "inventory": (
                    "outputs/dataset_understanding_v2/"
                    "dataset_audit_inventory_nonproduction_v2.csv"
                )
            },
        }
        audit_json.write_text(
            json.dumps(audit_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = self.project / "reports" / "dataset_understanding_v2.md"
        report.write_text(
            "# Audit\n\nResult: PASS for the audit task\n\n125,847\n1,675\n",
            encoding="utf-8",
        )
        return inventory, report, audit_json, expectations

    def create(self, output_name: str = "v2"):
        inventory, report, audit_json, expectations = self.write_fixture()
        output = self.project / "manifests" / output_name
        result = create_full_manifest(
            project_root=self.project,
            dataset_root=self.dataset,
            inventory_path=inventory,
            audit_report_path=report,
            audit_json_path=audit_json,
            output_dir=output,
            expectations=expectations,
        )
        return output, result, expectations, inventory, report, audit_json

    def test_direct_parent_identity_numeric_order_and_filename_independence(self) -> None:
        output, _, expectations, *_ = self.create()
        rows = read_manifest(output / MANIFEST_FILENAME)
        validate_manifest_rows(rows, expectations)
        self.assertEqual([row.speaker_id for row in rows], ["2", "2", "10", "30"])
        self.assertEqual(rows[-1].speaker_id, "30")
        self.assertIn("_30.wav", rows[-1].filename)
        self.assertEqual(rows[-1].audio_path.split("/")[0], "30")
        self.assertTrue(all("\\" not in row.audio_path for row in rows))

    def test_schema_has_no_split_label_or_eligibility(self) -> None:
        output, *_ = self.create()
        with (output / MANIFEST_FILENAME).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            fields = tuple(csv.DictReader(stream).fieldnames or ())
        self.assertEqual(fields, MANIFEST_FIELDS)
        self.assertFalse(FORBIDDEN_MANIFEST_FIELDS.intersection(fields))

    def test_duplicate_and_singleton_rows_are_preserved(self) -> None:
        output, _, _, *_ = self.create()
        rows = read_manifest(output / MANIFEST_FILENAME)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [row.duplicate_group for row in rows].count("dup_000001"), 2
        )
        counts = Counter(row.speaker_id for row in rows)
        self.assertEqual(sum(count == 1 for count in counts.values()), 2)
        self.assertEqual(set(row.provenance for row in rows), {
            "train", "train_small", "part", "test"
        })

    def test_provenance_does_not_create_a_split(self) -> None:
        output, *_ = self.create()
        text = (output / MANIFEST_FILENAME).read_text(encoding="utf-8")
        self.assertNotIn("final_split", text.splitlines()[0])
        self.assertEqual(len(text.splitlines()), 5)

    def test_deterministic_csv_json_and_hash_validation(self) -> None:
        output, result, expectations, inventory, report, audit_json = self.create()
        second = self.project / "outputs" / "reproduction"
        reproduced = create_full_manifest(
            project_root=self.project,
            dataset_root=self.dataset,
            inventory_path=inventory,
            audit_report_path=report,
            audit_json_path=audit_json,
            output_dir=second,
            expectations=expectations,
            reference_dir=output,
        )
        self.assertTrue(reproduced["reproducibility"]["manifest_byte_identical"])
        self.assertTrue(reproduced["reproducibility"]["identity_byte_identical"])
        self.assertEqual(
            (output / MANIFEST_FILENAME).read_bytes(),
            (second / MANIFEST_FILENAME).read_bytes(),
        )
        self.assertEqual(
            (output / IDENTITY_FILENAME).read_bytes(),
            (second / IDENTITY_FILENAME).read_bytes(),
        )
        self.assertEqual(
            result["manifest_sha256"], file_sha256(output / MANIFEST_FILENAME)
        )

    def test_identity_contains_no_absolute_dataset_root(self) -> None:
        output, _, expectations, inventory, report, audit_json = self.create()
        text = (output / IDENTITY_FILENAME).read_text(encoding="utf-8")
        self.assertNotIn(str(self.dataset), text)
        identity = json.loads(text)
        rows = read_manifest(output / MANIFEST_FILENAME)
        self.assertEqual(identity["row_count"], len(rows))
        self.assertEqual(identity["manifest_version"], MANIFEST_VERSION)

    def test_portable_path_rejections(self) -> None:
        for invalid in (
            "C:/absolute.wav",
            r"speaker\file.wav",
            "../speaker/file.wav",
            "/speaker/file.wav",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "non-portable"):
                    validate_portable_path(invalid)

    def test_inconsistent_audit_fails_closed(self) -> None:
        inventory, report, audit_json, expectations = self.write_fixture()
        payload = json.loads(audit_json.read_text(encoding="utf-8"))
        payload["counts"]["wav_file_count"] = 3
        audit_json.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "wav_file_count"):
            create_full_manifest(
                project_root=self.project,
                dataset_root=self.dataset,
                inventory_path=inventory,
                audit_report_path=report,
                audit_json_path=audit_json,
                output_dir=self.project / "manifests" / "v2",
                expectations=expectations,
            )
        self.assertFalse((self.project / "manifests" / "v2").exists())

    def test_missing_wav_fails_closed(self) -> None:
        inventory, report, audit_json, expectations = self.write_fixture()
        missing = next(self.dataset.rglob("*.wav"))
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "snapshot differs"):
            create_full_manifest(
                project_root=self.project,
                dataset_root=self.dataset,
                inventory_path=inventory,
                audit_report_path=report,
                audit_json_path=audit_json,
                output_dir=self.project / "manifests" / "v2",
                expectations=expectations,
            )

    def test_atomic_publish_failure_leaves_no_partial_directory(self) -> None:
        output = self.project / "manifests" / "v2"
        with mock.patch("src.full_manifest_v2.os.replace", side_effect=OSError("fail")):
            with self.assertRaisesRegex(OSError, "fail"):
                publish_artifact_pair(
                    output_dir=output,
                    manifest_bytes=b"manifest",
                    identity_bytes=b"identity",
                    validate_staged=lambda *_: None,
                )
        self.assertFalse(output.exists())
        leftovers = list(output.parent.glob(".v2.tmp.*"))
        self.assertEqual(leftovers, [])

    def test_existing_nonidentical_output_is_never_overwritten(self) -> None:
        output = self.project / "manifests" / "v2"
        output.mkdir(parents=True)
        (output / MANIFEST_FILENAME).write_bytes(b"old")
        (output / IDENTITY_FILENAME).write_bytes(b"old")
        with self.assertRaisesRegex(FileExistsError, "non-identical"):
            publish_artifact_pair(
                output_dir=output,
                manifest_bytes=b"new",
                identity_bytes=b"new",
                validate_staged=lambda *_: None,
            )
        self.assertEqual((output / MANIFEST_FILENAME).read_bytes(), b"old")

    def test_tampered_manifest_hash_is_rejected(self) -> None:
        output, _, expectations, inventory, report, audit_json = self.create()
        identity = json.loads(
            (output / IDENTITY_FILENAME).read_text(encoding="utf-8")
        )
        with (output / MANIFEST_FILENAME).open("ab") as stream:
            stream.write(b"tamper")
        from src.full_manifest_v2 import load_approved_audit

        audit = load_approved_audit(
            project_root=self.project,
            inventory_path=inventory,
            audit_report_path=report,
            audit_json_path=audit_json,
            expectations=expectations,
        )
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            validate_identity(
                identity,
                output / MANIFEST_FILENAME,
                expectations,
                audit,
                "manifests/v2/full_manifest_v2.csv",
            )


if __name__ == "__main__":
    unittest.main()
