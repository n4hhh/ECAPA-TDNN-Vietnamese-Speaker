"""Focused synthetic tests for portable manifest v1."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.create_portable_manifests import build_manifests, relative_wav_path, write_outputs


class PortableManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def wav(self, speaker: str, name: str) -> Path:
        path = self.dataset / "audio" / speaker / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def write_inputs(
        self, rows: list[tuple[Path, str, str]], assignments: list[tuple[str, str]]
    ) -> tuple[Path, Path]:
        manifest = self.root / "full.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("audio_path", "speaker_id", "filename_group"))
            writer.writerows((str(path), speaker, group) for path, speaker, group in rows)
        split = self.root / "split.csv"
        with split.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("speaker_id", "final_split"))
            writer.writerows(assignments)
        return manifest, split

    def build_sample(self):
        rows = [
            (self.wav("10", "b.wav"), "10", "train"),
            (self.wav("2", "a.wav"), "2", "train_small"),
            (self.wav("20", "v.wav"), "20", "train"),
            (self.wav("30", "t.wav"), "30", "train_small"),
            (self.wav("40", "x.wav"), "40", "train"),
            (self.wav("50", "q.wav"), "50", "test"),
        ]
        inputs = self.write_inputs(
            rows,
            [("10", "train"), ("2", "train"), ("20", "validation"),
             ("30", "test"), ("40", "excluded")],
        )
        return build_manifests(*inputs, self.dataset)

    def test_absolute_path_becomes_dataset_relative(self) -> None:
        wav = self.wav("2", "a.wav")
        self.assertEqual(relative_wav_path(str(wav), self.dataset), "audio/2/a.wav")

    def test_path_outside_dataset_root_is_rejected(self) -> None:
        outside = self.root / "outside.wav"
        outside.touch()
        with self.assertRaisesRegex(ValueError, "outside dataset root"):
            relative_wav_path(str(outside), self.dataset)

    def test_labels_filtering_and_no_leakage(self) -> None:
        result = self.build_sample()
        self.assertEqual(result.labels, {"2": 0, "10": 1})
        self.assertEqual({row["speaker_label"] for row in result.rows["validation"]}, {-1})
        self.assertEqual({row["speaker_label"] for row in result.rows["test"]}, {-1})
        all_ids = {row["speaker_id"] for rows in result.rows.values() for row in rows}
        self.assertNotIn("40", all_ids)
        self.assertNotIn("50", all_ids)
        split_ids = [{row["speaker_id"] for row in result.rows[name]} for name in ("train", "validation", "test")]
        self.assertFalse(split_ids[0] & split_ids[1])
        self.assertFalse(split_ids[0] & split_ids[2])
        self.assertFalse(split_ids[1] & split_ids[2])

    def test_duplicate_path_is_rejected(self) -> None:
        wav = self.wav("2", "same.wav")
        inputs = self.write_inputs(
            [(wav, "2", "train"), (wav, "3", "train")],
            [("2", "train"), ("3", "validation")],
        )
        with self.assertRaisesRegex(ValueError, "duplicate audio path"):
            build_manifests(*inputs, self.dataset)

    def test_deterministic_rerun(self) -> None:
        first = self.build_sample()
        second = self.build_sample()
        out_a, out_b = self.root / "a", self.root / "b"
        write_outputs(out_a, self.root / "report-a.md", first)
        write_outputs(out_b, self.root / "report-b.md", second)
        for name in (
            "train_manifest_v1.csv",
            "validation_manifest_v1.csv",
            "test_manifest_v1.csv",
            "speaker_to_label_v1.json",
            "portable_manifest_config_v1.json",
        ):
            self.assertEqual((out_a / name).read_bytes(), (out_b / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
