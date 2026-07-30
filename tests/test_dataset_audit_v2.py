"""Focused synthetic tests for the VieSpeaker2.0 read-only audit."""

from __future__ import annotations

import csv
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from src.dataset_audit_v2 import (
    INVENTORY_FIELDS,
    parse_candidate_source_group,
    parse_provenance,
    scan_dataset,
    write_outputs,
)


class DatasetAuditV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "dataset"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def wav(
        self,
        relative: str,
        frames: int = 160,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(sample_width)
            stream.setframerate(sample_rate)
            stream.writeframes(b"\0" * frames * channels * sample_width)
        return path

    def test_provenance_classes_and_case_awareness(self) -> None:
        cases = {
            "aug_train-00001-of-00002_3.wav": ("train", "exact_case_supported"),
            "aug_train_small-00001-of-00002_3.wav": (
                "train_small",
                "exact_case_supported",
            ),
            "aug_part-00001-of-00002_3.wav": ("part", "exact_case_supported"),
            "aug_test-00001-of-00002_3.wav": ("test", "exact_case_supported"),
            "recording.wav": ("other", "unrecognized_pattern"),
            "AUG_TRAIN-00001-OF-00002_3.WAV": (
                "train",
                "case_variant_supported",
            ),
            "aug_train-.wav": ("unparseable", "recognized_prefix_malformed"),
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(parse_provenance(filename), expected)

    def test_source_group_exact_ambiguous_and_unavailable(self) -> None:
        self.assertEqual(
            parse_candidate_source_group("aug_train-00006-of-00204_477.wav"),
            ("train/00006-of-00204/477", "exact_pattern_supported"),
        )
        self.assertEqual(
            parse_candidate_source_group("aug_test-00001-of-00002_7_variant.wav"),
            ("test/00001-of-00002/7", "ambiguous_suffix"),
        )
        self.assertEqual(
            parse_candidate_source_group("recording.wav"), ("", "unavailable")
        )

    def test_layout_headers_duplicates_ordering_and_reconciliation(self) -> None:
        original = self.wav("10/aug_train-00001-of-00002_1.wav")
        duplicate = self.root / "20" / "aug_test-00001-of-00002_1.wav"
        duplicate.parent.mkdir()
        duplicate.write_bytes(original.read_bytes())
        self.wav("not_numeric/other.wav", frames=80)
        self.wav("root.wav", frames=40)
        self.wav("30/nested/aug_part-00001-of-00002_4.wav", frames=20)
        (self.root / "40").mkdir()
        (self.root / "10" / "notes.txt").write_text("note", encoding="utf-8")
        (self.root / "50").mkdir()
        (self.root / "50" / "zero.wav").touch()
        corrupt = self.root / "60" / "corrupt.wav"
        corrupt.parent.mkdir()
        corrupt.write_bytes(b"not a wav")
        self.wav("70/zero_frame.wav", frames=0)

        before = {path.relative_to(self.root).as_posix(): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        result = scan_dataset(self.root)
        after = {path.relative_to(self.root).as_posix(): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

        self.assertEqual(before, after)
        self.assertEqual(result["result"], "PASS")
        self.assertEqual(result["counts"]["wav_file_count"], 8)
        self.assertEqual(result["counts"]["non_wav_file_count"], 1)
        self.assertEqual(result["counts"]["zero_byte_wav_count"], 1)
        self.assertEqual(result["counts"]["zero_frame_wav_count"], 1)
        self.assertEqual(result["counts"]["invalid_unreadable_wav_count"], 2)
        self.assertIn("root.wav", result["layout"]["root_level_files"])
        self.assertIn(
            "30/nested/aug_part-00001-of-00002_4.wav",
            result["layout"]["wav_below_expected_depth"],
        )
        self.assertIn("40", result["layout"]["empty_speaker_directories"])
        self.assertIn("not_numeric", result["layout"]["non_numeric_speaker_folder_names"])
        self.assertEqual(result["duplicates"]["exact_duplicate_group_count"], 1)
        self.assertEqual(
            result["duplicates"]["cross_speaker_exact_duplicate_group_count"], 1
        )
        paths = [row["dataset_relative_path"] for row in result["inventory_rows"]]
        self.assertEqual(paths, sorted(paths, key=lambda value: (value.casefold(), value)))
        self.assertTrue(all("\\" not in value and not Path(value).is_absolute() for value in paths))
        self.assertTrue(all(result["reconciliation"].values()))
        self.assertTrue(result["preservation"]["snapshots_match"])

    def test_same_filename_across_speakers(self) -> None:
        self.wav("1/same.wav", frames=10)
        self.wav("2/same.wav", frames=20)
        self.wav("branch_a/3/within.wav", frames=30)
        self.wav("branch_b/3/within.wav", frames=40)
        result = scan_dataset(self.root)
        groups = result["duplicates"]["duplicate_filename_groups"]
        self.assertEqual(len(groups), 2)
        by_name = {group["filename"]: group for group in groups}
        self.assertEqual(by_name["same.wav"]["scope"], "cross_speaker")
        self.assertEqual(by_name["within.wav"]["scope"], "within_speaker")
        self.assertEqual(result["duplicates"]["exact_duplicate_group_count"], 0)

    def test_fail_closed_output_validation_and_no_absolute_paths(self) -> None:
        self.wav("1/aug_train-00001-of-00001_0.wav")
        result = scan_dataset(self.root)
        runtime = Path(self.temporary.name) / "outputs"
        reports = Path(self.temporary.name) / "reports"
        artifacts = write_outputs(result, runtime, reports)
        with Path(artifacts["inventory"]).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(tuple(reader.fieldnames or ()), INVENTORY_FIELDS)
            persisted = list(reader)
        self.assertEqual(len(persisted), 1)
        self.assertFalse(Path(persisted[0]["dataset_relative_path"]).is_absolute())
        self.assertNotIn(str(self.root), Path(artifacts["summary"]).read_text(encoding="utf-8"))
        summary = json.loads(Path(artifacts["summary"]).read_text(encoding="utf-8"))
        self.assertEqual(summary["counts"]["wav_file_count"], 1)

        broken = dict(result)
        broken["inventory_rows"] = [dict(result["inventory_rows"][0])]
        broken["inventory_rows"][0]["dataset_relative_path"] = "C:/absolute.wav"
        with self.assertRaisesRegex(ValueError, "non-portable"):
            write_outputs(broken, runtime, reports)

    def test_duplicate_normalized_path_key_is_deterministic(self) -> None:
        # The real filesystem cannot contain two identical directory entries;
        # deterministic normalization/collision behavior is exercised through
        # a composed/decomposed Unicode pair where supported.
        first = "1/caf\u00e9.wav"
        second = "1/cafe\u0301.wav"
        self.wav(first, frames=10)
        try:
            self.wav(second, frames=20)
        except FileExistsError:
            self.skipTest("filesystem normalizes Unicode names")
        result = scan_dataset(self.root)
        self.assertEqual(len(result["layout"]["duplicate_normalized_relative_paths"]), 1)
        self.assertEqual(result["counts"]["wav_file_count"], 2)


if __name__ == "__main__":
    unittest.main()
