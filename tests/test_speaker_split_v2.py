"""Focused synthetic tests for the final speaker-disjoint split package v2."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from unittest import mock

from src.full_manifest_v2 import ManifestRow, render_manifest_csv
from src.speaker_split_v2 import (
    FULL_MANIFEST_IDENTITY_RELATIVE_PATH,
    FULL_MANIFEST_RELATIVE_PATH,
    LABEL_MAPPING_RELATIVE_PATH,
    PORTABLE_FIELDS,
    PORTABLE_IDENTITY_RELATIVE_PATH,
    REPRODUCIBILITY_PATHS,
    SPLIT_CSV_RELATIVE_PATH,
    SPLIT_FIELDS,
    SPLIT_IDENTITY_RELATIVE_PATH,
    SPLIT_POLICY_RELATIVE_PATH,
    TEST_MANIFEST_RELATIVE_PATH,
    TRAIN_MANIFEST_RELATIVE_PATH,
    VALIDATION_MANIFEST_RELATIVE_PATH,
    PackageExpectations,
    SpeakerStats,
    SplitPolicy,
    build_portable_rows,
    bytes_sha256,
    create_assignments,
    create_split_package,
    largest_remainder_allocation,
    numeric_speaker_key,
    render_json,
    selection_hash,
)


def speaker(
    speaker_id: int,
    count: int,
    *,
    seed: int = 20260729,
    duplicates: int = 0,
    provenance: str = "train",
) -> SpeakerStats:
    composition = {"train": 0, "train_small": 0, "part": 0, "test": 0}
    composition[provenance] = count
    return SpeakerStats(
        speaker_id=str(speaker_id),
        utterance_count=count,
        duration_seconds=Decimal(count * 3),
        provenance_counts=composition,
        duplicate_file_count=duplicates,
        selection_hash=selection_hash(seed, str(speaker_id)),
    )


class SpeakerSplitV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        (self.project / "manifests" / "v2").mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def policy() -> SplitPolicy:
        return SplitPolicy(validation_speakers=2, test_speakers=2)

    def fixture_rows(self, *, provenance_variant: bool = False) -> list[ManifestRow]:
        counts = {
            "2": 20,
            "3": 25,
            "10": 55,
            "11": 60,
            "20": 105,
            "21": 210,
            "30": 2,
            "31": 3,
            "40": 1,
            "41": 1,
        }
        rows: list[ManifestRow] = []
        provenance_names = ("train", "train_small", "part", "test")
        for speaker_id, count in counts.items():
            for index in range(count):
                filename = f"sample_{index:04d}.wav"
                provenance = (
                    provenance_names[(index + int(speaker_id)) % 4]
                    if provenance_variant
                    else provenance_names[index % 4]
                )
                duplicate_group = (
                    "dup_000001"
                    if speaker_id == "20" and index in {0, 1}
                    else ""
                )
                rows.append(
                    ManifestRow(
                        audio_path=f"{speaker_id}/{filename}",
                        speaker_id=speaker_id,
                        filename=filename,
                        provenance=provenance,
                        provenance_parse_status="exact_case_supported",
                        sample_rate_hz=16000,
                        channel_count=1,
                        sample_width_bytes=2,
                        bits_per_sample=16,
                        wav_encoding="pcm",
                        frame_count=48000,
                        duration_seconds="3.000000",
                        file_size_bytes=96078,
                        wav_status="readable",
                        duplicate_group=duplicate_group,
                        candidate_source_group=f"{provenance}/fixture/{index}",
                        source_group_parse_status="exact_pattern_supported",
                        manifest_version="v2",
                    )
                )
        return sorted(
            rows,
            key=lambda row: (
                numeric_speaker_key(row.speaker_id),
                row.audio_path,
            ),
        )

    def write_authoritative(
        self,
        *,
        rows: list[ManifestRow] | None = None,
        identity_override: dict | None = None,
    ) -> tuple[Path, Path, str, str, PackageExpectations]:
        rows = rows or self.fixture_rows()
        manifest_path = self.project / FULL_MANIFEST_RELATIVE_PATH
        manifest_bytes = render_manifest_csv(rows)
        manifest_path.write_bytes(manifest_bytes)
        counts = Counter(row.provenance for row in rows)
        speaker_counts = Counter(row.speaker_id for row in rows)
        duplicate_groups: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row.duplicate_group:
                duplicate_groups[row.duplicate_group].add(row.speaker_id)
        identity = {
            "manifest_relative_path": FULL_MANIFEST_RELATIVE_PATH,
            "manifest_sha256": bytes_sha256(manifest_bytes),
            "path_separator": "/",
            "manifest_version": "v2",
            "row_count": len(rows),
            "unique_audio_path_count": len({row.audio_path for row in rows}),
            "speaker_count": len(speaker_counts),
            "total_bytes": sum(row.file_size_bytes for row in rows),
            "total_duration_seconds": int(
                sum(
                    (Decimal(row.duration_seconds) for row in rows),
                    Decimal(0),
                )
            ),
            "readable_wav_count": len(rows),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_file_count": sum(
                bool(row.duplicate_group) for row in rows
            ),
            "cross_speaker_duplicate_group_count": sum(
                len(owners) > 1 for owners in duplicate_groups.values()
            ),
            "singleton_speaker_count": sum(
                count == 1 for count in speaker_counts.values()
            ),
            "provenance_counts": {
                name: counts[name]
                for name in ("train", "train_small", "part", "test")
            },
        }
        if identity_override:
            identity.update(identity_override)
        identity_path = self.project / FULL_MANIFEST_IDENTITY_RELATIVE_PATH
        identity_bytes = render_json(identity)
        identity_path.write_bytes(identity_bytes)
        expectations = PackageExpectations(
            total_rows=len(rows),
            total_speakers=10,
            train_speakers=4,
            validation_speakers=2,
            test_speakers=2,
            excluded_speakers=2,
            singleton_speakers=2,
            duplicate_groups=1,
            duplicate_files=2,
            evaluation_eligible_speakers=6,
        )
        return (
            manifest_path,
            identity_path,
            bytes_sha256(manifest_bytes),
            bytes_sha256(identity_bytes),
            expectations,
        )

    def create(
        self,
        name: str,
        *,
        rows: list[ManifestRow] | None = None,
        reference: Path | None = None,
    ):
        manifest, identity, manifest_hash, identity_hash, expectations = (
            self.write_authoritative(rows=rows)
        )
        output = self.project / name
        output.mkdir()
        result = create_split_package(
            project_root=self.project,
            manifest_path=manifest,
            identity_path=identity,
            output_root=output,
            expected_manifest_sha256=manifest_hash,
            expected_identity_sha256=identity_hash,
            policy=self.policy(),
            expectations=expectations,
            reference_root=reference,
        )
        return output, result

    def test_exact_100_100_disjoint_eligibility_and_singletons(self) -> None:
        counts = (20, 50, 100, 200, 500)
        speakers = [
            speaker(index + 1, counts[index % len(counts)])
            for index in range(205)
        ]
        speakers.extend([speaker(1000, 2), speaker(1001, 7)])
        speakers.extend([speaker(2000, 1), speaker(2001, 1)])
        policy = SplitPolicy()
        assignments, capacities, quotas, balance = create_assignments(
            speakers, policy
        )
        by_split = {
            split: {
                assignment.speaker.speaker_id
                for assignment in assignments
                if assignment.final_split == split
            }
            for split in ("train", "validation", "test", "excluded")
        }
        self.assertEqual(len(by_split["validation"]), 100)
        self.assertEqual(len(by_split["test"]), 100)
        self.assertEqual(len(by_split["excluded"]), 2)
        self.assertFalse(by_split["train"] & by_split["validation"])
        self.assertFalse(by_split["train"] & by_split["test"])
        self.assertFalse(by_split["validation"] & by_split["test"])
        self.assertTrue(
            all(
                assignment.speaker.utterance_count >= 20
                for assignment in assignments
                if assignment.final_split in {"validation", "test"}
            )
        )
        self.assertTrue(
            all(
                assignment.eligibility_reason == "singleton_speaker"
                for assignment in assignments
                if assignment.final_split == "excluded"
            )
        )
        self.assertEqual(sum(quotas.values()), 200)
        self.assertEqual(
            balance["validation_utterances"]
            + balance["test_utterances"],
            sum(
                assignment.speaker.utterance_count
                for assignment in assignments
                if assignment.final_split in {"validation", "test"}
            ),
        )
        self.assertEqual(sum(capacities.values()), 205)

    def test_proportional_largest_remainder_allocation(self) -> None:
        capacities = {
            "20-49": 250,
            "50-99": 199,
            "100-199": 202,
            "200-499": 120,
            "500+": 29,
        }
        names = tuple(capacities)
        self.assertEqual(
            largest_remainder_allocation(capacities, 200, names),
            {
                "20-49": 63,
                "50-99": 50,
                "100-199": 50,
                "200-499": 30,
                "500+": 7,
            },
        )

    def test_stable_sha256_order_and_hash_seed_determinism(self) -> None:
        code = """
import json
from decimal import Decimal
from src.speaker_split_v2 import SpeakerStats, SplitPolicy, create_assignments, selection_hash
counts=(20,50,100,200,500)
items=[]
for i in range(205):
    sid=str(i+1); count=counts[i%5]
    items.append(SpeakerStats(sid,count,Decimal(count*3),{"train":count,"train_small":0,"part":0,"test":0},0,selection_hash(20260729,sid)))
result=create_assignments(items,SplitPolicy())[0]
print(json.dumps([(a.speaker.speaker_id,a.final_split) for a in result],separators=(",",":")))
"""
        outputs = []
        for hash_seed in ("1", "999"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", code],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    text=True,
                )
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(
            selection_hash(20260729, "10"),
            selection_hash(20260729, "10"),
        )
        self.assertNotEqual(
            selection_hash(20260729, "10"),
            selection_hash(20260729, "11"),
        )

    def test_numeric_labels_contiguous_and_evaluation_minus_one(self) -> None:
        output, _ = self.create("package")
        labels = json.loads(
            (output / LABEL_MAPPING_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        ordered_ids = sorted(labels, key=numeric_speaker_key)
        self.assertEqual(
            [labels[speaker_id] for speaker_id in ordered_ids],
            list(range(len(labels))),
        )
        for relative in (
            VALIDATION_MANIFEST_RELATIVE_PATH,
            TEST_MANIFEST_RELATIVE_PATH,
        ):
            with (output / relative).open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["speaker_label"] for row in rows}, {"-1"})

    def test_provenance_neutrality(self) -> None:
        baseline = self.fixture_rows()
        changed = self.fixture_rows(provenance_variant=True)
        baseline_speakers = self._aggregate_for_assignment(baseline)
        changed_speakers = self._aggregate_for_assignment(changed)
        first = create_assignments(baseline_speakers, self.policy())[0]
        second = create_assignments(changed_speakers, self.policy())[0]
        self.assertEqual(
            [(item.speaker.speaker_id, item.final_split) for item in first],
            [(item.speaker.speaker_id, item.final_split) for item in second],
        )

    @staticmethod
    def _aggregate_for_assignment(rows: list[ManifestRow]) -> list[SpeakerStats]:
        grouped: dict[str, list[ManifestRow]] = defaultdict(list)
        for row in rows:
            grouped[row.speaker_id].append(row)
        result = []
        for speaker_id, speaker_rows in grouped.items():
            counts = Counter(row.provenance for row in speaker_rows)
            result.append(
                speaker(
                    int(speaker_id),
                    len(speaker_rows),
                    duplicates=sum(bool(row.duplicate_group) for row in speaker_rows),
                    provenance=max(counts, key=counts.get),
                )
            )
        return result

    def test_duplicate_preservation_no_cross_split_and_row_reconciliation(self) -> None:
        output, result = self.create("package")
        split_rows = {}
        duplicate_paths = []
        all_paths = []
        for split, relative in (
            ("train", TRAIN_MANIFEST_RELATIVE_PATH),
            ("validation", VALIDATION_MANIFEST_RELATIVE_PATH),
            ("test", TEST_MANIFEST_RELATIVE_PATH),
        ):
            with (output / relative).open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            split_rows[split] = rows
            all_paths.extend(row["relative_audio_path"] for row in rows)
            duplicate_paths.extend(
                (split, row["speaker_id"], row["relative_audio_path"])
                for row in rows
                if row["duplicate_group"]
            )
        self.assertEqual(len(duplicate_paths), 2)
        self.assertEqual(len({item[0] for item in duplicate_paths}), 1)
        self.assertEqual(len({item[1] for item in duplicate_paths}), 1)
        self.assertEqual(len(all_paths), len(set(all_paths)))
        self.assertEqual(
            sum(result["row_counts"].values()),
            len(self.fixture_rows()),
        )

    def test_portable_path_safety_and_exact_schemas(self) -> None:
        output, _ = self.create("package")
        with (output / SPLIT_CSV_RELATIVE_PATH).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            self.assertEqual(tuple(csv.DictReader(stream).fieldnames or ()), SPLIT_FIELDS)
        for relative in (
            TRAIN_MANIFEST_RELATIVE_PATH,
            VALIDATION_MANIFEST_RELATIVE_PATH,
            TEST_MANIFEST_RELATIVE_PATH,
        ):
            with (output / relative).open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames or ()), PORTABLE_FIELDS)
                for row in reader:
                    path = row["relative_audio_path"]
                    self.assertNotIn("\\", path)
                    self.assertNotIn("..", path.split("/"))
                    self.assertFalse(Path(path).is_absolute())
                    self.assertNotIn(":", path)

    def test_hash_mismatch_fails_before_writing(self) -> None:
        manifest, identity, _, identity_hash, expectations = (
            self.write_authoritative()
        )
        output = self.project / "output"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
            create_split_package(
                project_root=self.project,
                manifest_path=manifest,
                identity_path=identity,
                output_root=output,
                expected_manifest_sha256="0" * 64,
                expected_identity_sha256=identity_hash,
                policy=self.policy(),
                expectations=expectations,
            )
        self.assertFalse((output / "splits").exists())

    def test_inconsistent_manifest_identity_fails_closed(self) -> None:
        manifest, identity, manifest_hash, _, expectations = (
            self.write_authoritative(identity_override={"row_count": 1})
        )
        identity_hash = bytes_sha256(identity.read_bytes())
        output = self.project / "output"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "row_count"):
            create_split_package(
                project_root=self.project,
                manifest_path=manifest,
                identity_path=identity,
                output_root=output,
                expected_manifest_sha256=manifest_hash,
                expected_identity_sha256=identity_hash,
                policy=self.policy(),
                expectations=expectations,
            )
        self.assertFalse((output / "splits").exists())

    def test_missing_or_incomplete_full_manifest_fails(self) -> None:
        rows = self.fixture_rows()
        manifest, identity, _, identity_hash, expectations = (
            self.write_authoritative(rows=rows)
        )
        manifest.write_bytes(render_manifest_csv(rows[:-1]))
        tampered_hash = bytes_sha256(manifest.read_bytes())
        output = self.project / "output"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "identity"):
            create_split_package(
                project_root=self.project,
                manifest_path=manifest,
                identity_path=identity,
                output_root=output,
                expected_manifest_sha256=tampered_hash,
                expected_identity_sha256=identity_hash,
                policy=self.policy(),
                expectations=expectations,
            )

    def test_atomic_publication_failure_rolls_back_every_artifact(self) -> None:
        manifest, identity, manifest_hash, identity_hash, expectations = (
            self.write_authoritative()
        )
        output = self.project / "output"
        output.mkdir()
        original_replace = os.replace
        calls = 0

        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected publication failure")
            return original_replace(source, destination)

        with mock.patch("src.speaker_split_v2.os.replace", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "injected"):
                create_split_package(
                    project_root=self.project,
                    manifest_path=manifest,
                    identity_path=identity,
                    output_root=output,
                    expected_manifest_sha256=manifest_hash,
                    expected_identity_sha256=identity_hash,
                    policy=self.policy(),
                    expectations=expectations,
                )
        for relative in REPRODUCIBILITY_PATHS:
            self.assertFalse((output / relative).exists())
        self.assertEqual(list(output.glob(".speaker_split_v2.tmp.*")), [])

    def test_byte_identical_reproduction_and_no_absolute_path_persistence(self) -> None:
        first, _ = self.create("first")
        manifest = self.project / FULL_MANIFEST_RELATIVE_PATH
        identity = self.project / FULL_MANIFEST_IDENTITY_RELATIVE_PATH
        manifest_hash = bytes_sha256(manifest.read_bytes())
        identity_hash = bytes_sha256(identity.read_bytes())
        expectations = PackageExpectations(
            total_rows=len(self.fixture_rows()),
            total_speakers=10,
            train_speakers=4,
            validation_speakers=2,
            test_speakers=2,
            excluded_speakers=2,
            singleton_speakers=2,
            duplicate_groups=1,
            duplicate_files=2,
            evaluation_eligible_speakers=6,
        )
        second = self.project / "second"
        second.mkdir()
        result = create_split_package(
            project_root=self.project,
            manifest_path=manifest,
            identity_path=identity,
            output_root=second,
            expected_manifest_sha256=manifest_hash,
            expected_identity_sha256=identity_hash,
            policy=self.policy(),
            expectations=expectations,
            reference_root=first,
        )
        self.assertTrue(
            result["reproducibility"][
                "all_required_artifacts_byte_identical"
            ]
        )
        for relative in REPRODUCIBILITY_PATHS:
            self.assertEqual(
                (first / relative).read_bytes(),
                (second / relative).read_bytes(),
            )
        for root in (first, second):
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in {".csv", ".json"}:
                    text = path.read_text(encoding="utf-8")
                    self.assertNotIn(str(self.project), text)
                    self.assertNotRegex(text, r"[A-Za-z]:[\\/]")

    def test_identity_hash_bindings_and_final_test_quarantine(self) -> None:
        output, _ = self.create("package")
        for relative, expected_kind in (
            (SPLIT_IDENTITY_RELATIVE_PATH, "speaker_split_v2"),
            (PORTABLE_IDENTITY_RELATIVE_PATH, "portable_manifests_v2"),
        ):
            identity = json.loads((output / relative).read_text(encoding="utf-8"))
            self.assertEqual(identity["identity_kind"], expected_kind)
            self.assertTrue(identity["final_test_quarantine"])
            self.assertEqual(
                identity["split_csv_sha256"],
                bytes_sha256((output / SPLIT_CSV_RELATIVE_PATH).read_bytes()),
            )
            self.assertEqual(
                identity["split_policy_sha256"],
                bytes_sha256((output / SPLIT_POLICY_RELATIVE_PATH).read_bytes()),
            )
            self.assertEqual(
                identity["test_manifest_sha256"],
                bytes_sha256((output / TEST_MANIFEST_RELATIVE_PATH).read_bytes()),
            )


if __name__ == "__main__":
    unittest.main()
