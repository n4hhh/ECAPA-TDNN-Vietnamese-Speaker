from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.evaluate_final_test_v2 import score_embeddings
from scripts.precompute_final_test_fbank_v2 import (
    index_bytes,
    validate_shard,
)
from src.final_evaluation_v2 import (
    ALLOWED_TRANSITIONS,
    APPROVED_INPUTS,
    BATCH_SIZE,
    CHECKPOINT_SHA256,
    EMBEDDING_DIMENSION,
    EXPECTED_NEGATIVE,
    EXPECTED_POSITIVE,
    EXPECTED_ROWS,
    EXPECTED_SPEAKERS,
    FEATURE_SHAPE,
    LOCKED_THRESHOLD,
    NEGATIVE_ROUNDS,
    ONE_TIME_POLICY_VERSION,
    POSITIVE_PER_SPEAKER,
    SHARD_SIZE,
    THRESHOLD_SOURCE,
    FinalManifestRow,
    assert_no_absolute_paths,
    atomic_json,
    build_cache_config,
    build_lock_config,
    build_lock_identity,
    calculate_final_metrics,
    canonical_json,
    final_trials_csv_bytes,
    generate_final_trials,
    initialize_state,
    read_final_manifest,
    read_json,
    read_state,
    safe_portable_path,
    score_alignment_sha256,
    sha256_bytes,
    sha256_file,
    transition_state,
    validate_final_trials,
    validate_split_disjointness,
    verify_approved_inputs,
)
from src.verification_v2 import ValidationTrial


def synthetic_rows() -> tuple[FinalManifestRow, ...]:
    rows = []
    for speaker in range(100):
        speaker_id = str(1000 + speaker)
        for utterance in range(16):
            rows.append(
                FinalManifestRow(
                    relative_audio_path=f"{speaker_id}/{utterance:03d}.wav",
                    speaker_id=speaker_id,
                    speaker_label=-1,
                    final_split="test",
                    filename_group="test",
                    duplicate_group="",
                    manifest_version="v2",
                    manifest_row_index=len(rows),
                )
            )
    return tuple(rows)


def synthetic_manifest_text(rows: tuple[FinalManifestRow, ...]) -> str:
    lines = [
        "relative_audio_path,speaker_id,speaker_label,final_split,"
        "filename_group,duplicate_group,manifest_version"
    ]
    for row in rows:
        lines.append(
            f"{row.relative_audio_path},{row.speaker_id},-1,test,"
            f"{row.filename_group},{row.duplicate_group},v2"
        )
    return "\n".join(lines) + "\n"


def fake_verified() -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": binding["path"],
            "sha256": binding["sha256"] or "a" * 64,
        }
        for name, binding in APPROVED_INPUTS.items()
    }


def validation_reference() -> dict:
    return {
        "eer": 0.25,
        "far": 0.25,
        "frr": 0.25,
        "accuracy": 0.75,
        "score_distributions": {
            "positive": {
                "mean": 0.6,
                "standard_deviation": 0.1,
                "median": 0.6,
            },
            "negative": {
                "mean": 0.1,
                "standard_deviation": 0.1,
                "median": 0.1,
            },
            "full_range": {"minimum": -0.2, "maximum": 0.9},
        },
    }


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = synthetic_rows()
        cls.trials, cls.metadata = generate_final_trials(cls.rows)

    def test_locked_constants(self) -> None:
        self.assertEqual(EXPECTED_ROWS, 15355)
        self.assertEqual(EXPECTED_SPEAKERS, 100)
        self.assertEqual(FEATURE_SHAPE, (301, 80))
        self.assertEqual((BATCH_SIZE, SHARD_SIZE), (64, 256))
        self.assertEqual(EMBEDDING_DIMENSION, 192)
        self.assertEqual(LOCKED_THRESHOLD, 0.16545939445495605)
        self.assertEqual(THRESHOLD_SOURCE, "selected epoch-3 validation threshold")

    def test_exact_trial_balance_and_participation(self) -> None:
        self.assertEqual(len(self.trials), 20000)
        positives = [trial for trial in self.trials if trial.target == 1]
        negatives = [trial for trial in self.trials if trial.target == 0]
        self.assertEqual(len(positives), EXPECTED_POSITIVE)
        self.assertEqual(len(negatives), EXPECTED_NEGATIVE)
        self.assertEqual(POSITIVE_PER_SPEAKER, 100)
        self.assertEqual(NEGATIVE_ROUNDS, 200)
        self.assertEqual(
            self.metadata["negative_participation_per_speaker"], [200]
        )

    def test_trials_are_deterministic(self) -> None:
        second, metadata = generate_final_trials(self.rows)
        self.assertEqual(final_trials_csv_bytes(self.trials), final_trials_csv_bytes(second))
        self.assertEqual(self.metadata, metadata)

    def test_trials_are_pythonhashseed_independent(self) -> None:
        code = (
            "from tests.test_final_evaluation_v2 import synthetic_rows;"
            "from src.final_evaluation_v2 import generate_final_trials,"
            "final_trials_csv_bytes,sha256_bytes;"
            "print(sha256_bytes(final_trials_csv_bytes(generate_final_trials("
            "synthetic_rows())[0])))"
        )
        hashes = []
        for seed in ("1", "999"):
            env = dict(__import__("os").environ)
            env["PYTHONHASHSEED"] = seed
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            hashes.append(result.stdout.strip())
        self.assertEqual(hashes[0], hashes[1])

    def test_duplicate_canonical_pair_rejected(self) -> None:
        duplicate = list(self.trials)
        first = duplicate[0]
        duplicate[1] = ValidationTrial(
            trial_id=duplicate[1].trial_id,
            left_audio_path=first.right_audio_path,
            right_audio_path=first.left_audio_path,
            left_speaker_id=first.right_speaker_id,
            right_speaker_id=first.left_speaker_id,
            target=first.target,
        )
        with self.assertRaises(ValueError):
            validate_final_trials(tuple(duplicate), self.rows)

    def test_positive_duplicate_group_rejected(self) -> None:
        rows = list(self.rows)
        trial = self.trials[0]
        for position, row in enumerate(rows):
            if row.relative_audio_path in {
                trial.left_audio_path,
                trial.right_audio_path,
            }:
                rows[position] = FinalManifestRow(
                    **{**row.__dict__, "duplicate_group": "dup"}
                )
        with self.assertRaises(ValueError):
            validate_final_trials(self.trials, rows)

    def test_safe_path_validation(self) -> None:
        self.assertEqual(safe_portable_path("123/a.wav"), "123/a.wav")
        for invalid in (
            r"E:\data\a.wav",
            "/root/a.wav",
            "../a.wav",
            r"123\a.wav",
            "123:bad/a.wav",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                safe_portable_path(invalid)

    def test_manifest_direct_parent_ownership_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            text = synthetic_manifest_text(self.rows).replace(
                "1000/000.wav,1000,", "wrong/000.wav,1000,", 1
            )
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_final_manifest(path)

    def test_split_disjointness_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            validation = root / "validation.csv"
            split = root / "split.csv"
            train.write_text("speaker_id\n1000\n", encoding="utf-8")
            validation.write_text("speaker_id\n9999\n", encoding="utf-8")
            lines = ["speaker_id,final_split"]
            lines.extend(
                f"{1000 + number},test" for number in range(100)
            )
            split.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_split_disjointness(self.rows, train, validation, split)

    def test_required_hash_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings = {}
            names = (
                "checkpoint",
                "epoch_003_checkpoint",
                "final_training_report",
            )
            for name in names:
                path = root / f"{name}.bin"
                if name == "final_training_report":
                    path.write_text(
                        canonical_json(
                            {
                                "result": "PASS",
                                "best": {
                                    "epoch": 3,
                                    "checkpoint_sha256": CHECKPOINT_SHA256,
                                    "empirical_threshold": LOCKED_THRESHOLD,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    path.write_bytes(b"checkpoint")
                bindings[name] = {
                    "path": path.name,
                    "sha256": sha256_file(path),
                }
            bindings["checkpoint"]["sha256"] = "0" * 64
            with mock.patch.dict(APPROVED_INPUTS, bindings, clear=True):
                with self.assertRaises(ValueError):
                    verify_approved_inputs(root)

    def test_checkpoint_byte_identity_rejection(self) -> None:
        self.assertEqual(len(CHECKPOINT_SHA256), 64)
        verified = fake_verified()
        verified["checkpoint"]["sha256"] = "0" * 64
        lock = build_lock_config(
            verified,
            trial_csv_hash="1" * 64,
            trial_config_hash="2" * 64,
            trial_identity_hash="3" * 64,
            fbank_config_hash="4" * 64,
        )
        self.assertNotEqual(lock["checkpoint"]["sha256"], verified["checkpoint"]["sha256"])

    def test_threshold_is_not_parameterized_by_test_metrics(self) -> None:
        lock = build_lock_config(
            fake_verified(),
            trial_csv_hash="1" * 64,
            trial_config_hash="2" * 64,
            trial_identity_hash="3" * 64,
            fbank_config_hash="4" * 64,
        )
        self.assertEqual(lock["operational_threshold"], LOCKED_THRESHOLD)
        self.assertEqual(lock["threshold_source"], THRESHOLD_SOURCE)

    def test_no_absolute_path_persistence(self) -> None:
        cache = build_cache_config(fake_verified())
        assert_no_absolute_paths(cache)
        with self.assertRaises(ValueError):
            assert_no_absolute_paths({"dataset_root": r"E:\VieSpeaker2.0\augmented_dataset"})


class StateMachineTests(unittest.TestCase):
    def test_exact_phase_chain(self) -> None:
        self.assertEqual(
            set(ALLOWED_TRANSITIONS["protocol_locked"]),
            {"cache_complete", "failed"},
        )
        self.assertEqual(ALLOWED_TRANSITIONS["finalized"], set())
        self.assertEqual(ONE_TIME_POLICY_VERSION.endswith("_v2"), True)

    def test_interrupted_resume_and_artifact_hash_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.json"
            atomic_json(artifact, {"ok": True})
            initialize_state(root, "a" * 64, "evaluation")
            transition_state(
                root,
                "protocol_locked",
                "cache_complete",
                {"artifact.json": sha256_file(artifact)},
            )
            self.assertEqual(read_state(root, "a" * 64)["phase"], "cache_complete")
            artifact.write_text("corrupt", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_state(root, "a" * 64)

    def test_lock_identity_is_single_deterministic_id(self) -> None:
        lock = build_lock_config(
            fake_verified(),
            trial_csv_hash="1" * 64,
            trial_config_hash="2" * 64,
            trial_identity_hash="3" * 64,
            fbank_config_hash="4" * 64,
        )
        first = build_lock_identity("5" * 64, lock)
        second = build_lock_identity("5" * 64, copy.deepcopy(lock))
        self.assertEqual(first, second)
        self.assertEqual(first["evaluation_id"], second["evaluation_id"])

    def test_lock_immutability_changes_identity(self) -> None:
        lock = build_lock_config(
            fake_verified(),
            trial_csv_hash="1" * 64,
            trial_config_hash="2" * 64,
            trial_identity_hash="3" * 64,
            fbank_config_hash="4" * 64,
        )
        first = sha256_bytes(canonical_json(lock).encode())
        lock["operational_threshold"] += 0.01
        second = sha256_bytes(canonical_json(lock).encode())
        self.assertNotEqual(first, second)

    def test_no_force_bypass(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts/evaluate_final_test_v2.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--force", source)
        self.assertIn('state["phase"] == "finalized"', source)


class CacheAndModelTests(unittest.TestCase):
    def test_cache_is_separate_and_test_only(self) -> None:
        config = build_cache_config(fake_verified())
        self.assertEqual(config["included_splits"], ["test"])
        self.assertEqual(config["expected_rows"], 15355)
        self.assertEqual(config["expected_labels"], [-1])
        self.assertNotIn("outputs/fbank_cache_v2", config["index_path"])

    def test_index_manifest_alignment(self) -> None:
        rows = synthetic_rows()[:20]
        text = index_bytes(rows).decode("utf-8")
        self.assertEqual(text.count("\n"), 21)
        self.assertIn("1000/000.wav,1000,-1,test,test/shard_00000.pt,0,301,80,float32,0", text)

    def test_shard_contract(self) -> None:
        rows = synthetic_rows()[:3]
        payload = {
            "schema_version": 2,
            "features": torch.zeros((3, 301, 80), dtype=torch.float32),
            "speaker_labels": torch.full((3,), -1, dtype=torch.long),
            "speaker_ids": [row.speaker_id for row in rows],
            "relative_audio_paths": [row.relative_audio_path for row in rows],
            "manifest_row_indices": [row.manifest_row_index for row in rows],
            "shard_number": 0,
            "final_split": "test",
        }
        validate_shard(payload, rows, 0)
        payload["features"] = payload["features"].transpose(1, 2)
        with self.assertRaises(ValueError):
            validate_shard(payload, rows, 0)

    def test_finalized_shard_hash_mismatch_is_rejected_in_source(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts/precompute_final_test_fbank_v2.py"
        ).read_text(encoding="utf-8")
        self.assertIn("untracked or mismatching finalized shard", source)
        self.assertIn("validated and skipped", source)

    def test_no_cache_preprocessing_or_transpose(self) -> None:
        config = build_cache_config(fake_verified())
        self.assertFalse(any(config["preprocessing"].values()))
        self.assertFalse(config["transposed"])
        self.assertTrue(config["raw_pre_normalization"])

    def test_cosine_score_correctness_and_alignment(self) -> None:
        embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
        )
        trials = (
            ValidationTrial("a", "x", "y", "1", "1", 1),
            ValidationTrial("b", "x", "z", "1", "2", 0),
        )
        scores = score_embeddings(embeddings, ["x", "y", "z"], trials)
        self.assertTrue(torch.equal(scores, torch.tensor([1.0, 0.0])))
        self.assertNotEqual(
            score_alignment_sha256(["a", "b"], [1, 0]),
            score_alignment_sha256(["b", "a"], [1, 0]),
        )

    def test_model_script_avoids_training_object_construction(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts/evaluate_final_test_v2.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("create_training_objects(", source)
        self.assertNotIn("AAMSoftmax(", source)
        self.assertNotIn("GradScaler(", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertIn("load_trainable_modules", source)


class MetricsTests(unittest.TestCase):
    def test_locked_confusion_and_extended_metrics(self) -> None:
        scores = [0.9, 0.1, 0.8, 0.0]
        targets = [1, 1, 0, 0]
        metrics = calculate_final_metrics(scores, targets, validation_reference())
        primary = metrics["primary_locked_validation_threshold"]
        self.assertEqual(
            [primary[key] for key in ("tp", "tn", "fp", "fn")],
            [1, 1, 1, 1],
        )
        self.assertEqual(primary["far"], 0.5)
        self.assertEqual(primary["frr"], 0.5)
        self.assertEqual(primary["accuracy"], 0.5)
        self.assertEqual(primary["precision"], 0.5)
        self.assertEqual(primary["recall"], 0.5)
        self.assertEqual(primary["specificity"], 0.5)
        self.assertEqual(primary["f1"], 0.5)

    def test_threshold_rule_is_score_greater_equal(self) -> None:
        scores = [LOCKED_THRESHOLD, LOCKED_THRESHOLD - 1e-9, 0.0, 0.9]
        targets = [1, 1, 0, 0]
        primary = calculate_final_metrics(
            scores, targets, validation_reference()
        )["primary_locked_validation_threshold"]
        self.assertEqual(primary["tp"], 1)
        self.assertEqual(primary["fn"], 1)
        self.assertEqual(primary["fp"], 1)

    def test_descriptive_threshold_cannot_replace_operational(self) -> None:
        metrics = calculate_final_metrics(
            [0.9, 0.8, 0.2, 0.1],
            [1, 1, 0, 0],
            validation_reference(),
        )
        descriptive = metrics["descriptive_test_diagnostic_non_operational"]
        self.assertFalse(descriptive["operational"])
        self.assertTrue(descriptive["must_not_replace_locked_threshold"])
        self.assertEqual(
            metrics["primary_locked_validation_threshold"]["threshold"],
            LOCKED_THRESHOLD,
        )

    def test_generalization_and_distribution_math(self) -> None:
        metrics = calculate_final_metrics(
            [0.9, 0.7, 0.2, 0.0],
            [1, 1, 0, 0],
            validation_reference(),
        )
        self.assertIn("relative_eer_change", metrics["generalization_analysis"])
        self.assertAlmostEqual(
            metrics["score_distributions"]["positive"]["mean"], 0.8
        )
        self.assertEqual(
            metrics["score_distributions"]["full_range"],
            {"minimum": 0.0, "maximum": 0.9},
        )

    def test_balanced_protocol_caveat_present(self) -> None:
        metrics = calculate_final_metrics(
            [0.9, 0.8, 0.2, 0.1],
            [1, 1, 0, 0],
            validation_reference(),
        )
        caveat = metrics["primary_locked_validation_threshold"][
            "balanced_protocol_caveat"
        ]
        self.assertIn("balanced 50/50", caveat)
        self.assertIn("real-world", caveat)

    def test_metrics_only_source_has_no_model_or_cache_import(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts/recompute_final_test_metrics_v2.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import speechbrain", source)
        self.assertNotIn("torch.cuda", source)
        self.assertNotIn("torchaudio", source)


if __name__ == "__main__":
    unittest.main()
