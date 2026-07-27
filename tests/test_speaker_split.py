"""Focused synthetic tests for candidate speaker split v1."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.create_speaker_split import (
    SPLIT_FIELDS, create_assignments, read_manifest, write_outputs,
)


class SpeakerSplitTests(unittest.TestCase):
    def make_manifest(self, root: Path) -> Path:
        path = root / "manifest.csv"
        rows = []

        def add(speaker: str, group: str, count: int) -> None:
            for index in range(count):
                rows.append({"audio_path": f"C:/audio/{speaker}/{group}_{index}.wav", "speaker_id": speaker, "filename_group": group})

        add("excluded_three", "train", 3)
        add("train_seven", "train_small", 7)
        add("mixed", "train", 6)
        add("mixed", "train_small", 6)
        add("overlap", "train", 10)
        add("overlap", "test", 2)
        add("part_only", "part", 4)
        for index in range(205):
            add(f"eligible_{index:03d}", "train" if index % 2 else "train_small", 10 + index % 45)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("audio_path", "speaker_id", "filename_group"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_rules_determinism_and_zero_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = read_manifest(self.make_manifest(root))
            first = create_assignments(data, 2026, 5, 10, 100, 100)
            second = create_assignments(data, 2026, 5, 10, 100, 100)
            self.assertEqual(first, second)
            by_id = {row["speaker_id"]: row for row in first}
            self.assertEqual(by_id["excluded_three"]["final_split"], "excluded")
            self.assertEqual(by_id["train_seven"]["final_split"], "train")
            self.assertIn(by_id["overlap"]["final_split"], {"train", "validation", "test"})
            self.assertEqual(by_id["mixed"]["provenance_signature"], "train_and_train_small")
            self.assertNotIn("part_only", by_id)
            self.assertIn("overlap", data.quarantine_counts)
            splits = {name: {row["speaker_id"] for row in first if row["final_split"] == name} for name in ("train", "validation", "test")}
            self.assertFalse(splits["train"] & splits["validation"])
            self.assertFalse(splits["train"] & splits["test"])
            self.assertFalse(splits["validation"] & splits["test"])
            self.assertEqual(len(splits["validation"]), 100)
            self.assertEqual(len(splits["test"]), 100)

            config = {"manifest": "manifest.csv", "seed": 2026, "trusted_groups": ["train", "train_small"], "quarantine_groups": ["test", "part"], "minimum_train_utterances": 5, "minimum_evaluation_utterances": 10, "validation_target": 100, "test_target": 100, "utterance_buckets": {"10-19": [10, 19], "20-49": [20, 49], "50+": [50, None]}}
            out_a, out_b = root / "a", root / "b"
            write_outputs(out_a, data, first, config)
            write_outputs(out_b, data, second, config)
            for relative in ("splits/speaker_split_v1.csv", "splits/excluded_speakers_v1.csv", "splits/split_config_v1.json", "reports/quarantine_audit_v1.csv", "reports/split_v1_summary.md"):
                self.assertEqual((out_a / relative).read_bytes(), (out_b / relative).read_bytes())

    def test_duplicate_audio_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.csv"
            path.write_text("audio_path,speaker_id,filename_group\nC:/same.wav,a,train\nC:/SAME.wav,b,part\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate audio path"):
                read_manifest(path)


if __name__ == "__main__":
    unittest.main()
