"""Synthetic CPU tests for the fixed one-epoch ECAPA/AAM pilot helpers."""

from __future__ import annotations

import copy
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.aam_training import (
    AAMSoftmax,
    apply_batchnorm_policy,
    assert_batchnorm_running_state_exact,
    batchnorm_running_state,
    build_adamw_optimizer,
    capture_rng_state,
    restore_rng_state,
    to_cpu_tree,
)
from src.cached_fbank_dataset import CachedFbankDataset, CachedFbankRow
from src.cached_fbank_samplers import HybridShardAwareSpeakerBatchSampler
from src.ecapa_one_epoch_pilot import (
    AAM_CONFIGURATION,
    ACCUMULATION_STEPS,
    AMP_POLICY,
    BASELINE_INTERPOLATED_EER,
    BATCHNORM_POLICY,
    CHECKPOINT_KEYS,
    EXPECTED_TRIAL_SHA256,
    LOGGING_INTERVAL,
    NUM_LOGICAL_BATCHES,
    OPTIMIZER_CONFIGURATION,
    PHYSICAL_MICROBATCH_SIZE,
    PILOT_RUNTIME_SCHEMA_NAME,
    PILOT_RUNTIME_SCHEMA_VERSION,
    PILOT_SCHEMA_NAME,
    PILOT_SCHEMA_VERSION,
    ROLLING_CHECKPOINT_STEPS,
    SAMPLER_CONFIGURATION,
    LoggingWindow,
    OneEpochController,
    append_jsonl,
    atomic_save_pilot_checkpoint,
    baseline_comparison,
    checkpoint_state_roundtrip_exact,
    classify_model_quality,
    hash_mapping_digest,
    load_pilot_checkpoint,
    require_trial_hash,
    resume_cursor,
    rolling_checkpoint_due,
    select_best_checkpoint,
    summarize_finite_values,
    validate_pilot_checkpoint,
    validate_relative_artifact_path,
    validate_runtime_payload,
    validate_scalar_tree,
    validate_training_log_record,
    validate_validation_metrics_payload,
    validation_forward_batch,
)


class OneEpochLimitTests(unittest.TestCase):
    def test_exact_one_epoch_and_no_epoch_one_transition(self) -> None:
        controller = OneEpochController()
        controller.start(0)
        for position in range(NUM_LOGICAL_BATCHES):
            controller.record_update(position)
        result = controller.finish()
        self.assertEqual(result["optimizer_updates"], 1000)
        self.assertEqual(result["logical_sample_selections"], 32000)
        self.assertEqual(result["epochs_started"], [0])
        self.assertEqual(result["epochs_completed"], [0])
        self.assertFalse(result["epoch_1_started"])
        self.assertEqual((result["next_epoch"], result["next_logical_batch_position"]), (1, 0))
        with self.assertRaises(RuntimeError):
            controller.start(1)
        with self.assertRaises(RuntimeError):
            controller.record_update(1000)

    def test_incomplete_epoch_and_nonzero_start_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            OneEpochController().start(1)
        controller = OneEpochController()
        controller.start(0)
        controller.record_update(0)
        with self.assertRaises(RuntimeError):
            controller.finish()
        with self.assertRaises(ValueError):
            controller.record_update(2)

    def test_resume_cursors_and_exact_rolling_triggers(self) -> None:
        self.assertEqual(resume_cursor(250), (0, 250))
        self.assertEqual(resume_cursor(999), (0, 999))
        self.assertEqual(resume_cursor(1000), (1, 0))
        actual = tuple(step for step in range(1, 1001) if rolling_checkpoint_due(step))
        self.assertEqual(actual, ROLLING_CHECKPOINT_STEPS)
        for invalid in (-1, 1001):
            with self.assertRaises(ValueError):
                resume_cursor(invalid)
        logging_steps = tuple(
            step
            for step in range(1, NUM_LOGICAL_BATCHES + 1)
            if step % LOGGING_INTERVAL == 0 or step == NUM_LOGICAL_BATCHES
        )
        self.assertEqual(logging_steps, tuple(range(50, 1001, 50)))
        self.assertEqual(len(logging_steps), 20)


class LoggingTests(unittest.TestCase):
    def test_window_aggregation_and_jsonl_scalars(self) -> None:
        window = LoggingWindow()
        for value in range(1, LOGGING_INTERVAL + 1):
            window.add(float(value), float(value) / 10.0)
        result = window.emit(LOGGING_INTERVAL)
        self.assertEqual(result["window_start_step"], 1)
        self.assertEqual(result["window_end_step"], 50)
        self.assertEqual(result["mean_logical_loss"], 25.5)
        self.assertAlmostEqual(result["mean_step_duration_seconds"], 2.55)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            append_jsonl(path, {"step": 50, "loss": 1.25, "checkpoint": None})
            decoded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(decoded["step"], 50)
        self.assertIsInstance(decoded["loss"], float)

    def test_training_log_schema_and_checkpoint_cadence(self) -> None:
        base = {
            "schema_name": "ecapa_aam_pilot_training_log",
            "schema_version": 1,
            "epoch": 0,
            "global_optimizer_step": 50,
            "logical_batch_position": 49,
            "current_logical_loss": 1.0,
            "mean_logical_loss_latest_window": 1.1,
            "ecapa_learning_rate": 1e-5,
            "aam_learning_rate": 1e-3,
            "grad_scaler_scale": 128.0,
            "ecapa_gradient_norm": 2.0,
            "aam_gradient_norm": 3.0,
            "step_duration_seconds": 1.0,
            "rolling_mean_step_duration_seconds": 1.1,
            "logical_samples_per_second": 32.0,
            "cuda_memory_allocated_bytes": 1,
            "cuda_memory_reserved_bytes": 2,
            "cuda_max_memory_allocated_bytes": 3,
            "cuda_max_memory_reserved_bytes": 4,
            "rolling_checkpoint_path": None,
        }
        validate_training_log_record(base)
        boundary = dict(
            base,
            global_optimizer_step=250,
            logical_batch_position=249,
            rolling_checkpoint_path=(
                "outputs/ecapa_aam_one_epoch_pilot_v1/last.pt"
            ),
        )
        validate_training_log_record(boundary)
        for malformed in (
            dict(base, global_optimizer_step=51),
            dict(base, rolling_checkpoint_path="outputs/premature/last.pt"),
            dict(boundary, rolling_checkpoint_path=None),
            dict(base, ecapa_gradient_norm=math.nan),
        ):
            with self.assertRaises(ValueError):
                validate_training_log_record(malformed)

    def test_logging_rejects_tensors_arrays_and_nonfinite_values(self) -> None:
        invalid = (
            {"tensor": torch.tensor(1.0)},
            {"nan": math.nan},
            {"inf": math.inf},
            {1: "bad key"},
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises((TypeError, ValueError)):
                validate_scalar_tree(payload)
        window = LoggingWindow()
        with self.assertRaises(ValueError):
            window.add(math.nan, 1.0)
        with self.assertRaises(ValueError):
            window.add(1.0, 0.0)

    def test_loss_summary_is_detached_python_metadata(self) -> None:
        summary = summarize_finite_values([3.0, 2.0, 4.0], "losses")
        self.assertEqual(
            summary,
            {
                "count": 3,
                "first": 3.0,
                "final": 4.0,
                "mean": 3.0,
                "minimum": 2.0,
                "maximum": 4.0,
            },
        )
        validate_scalar_tree(summary)
        with self.assertRaises(ValueError):
            summarize_finite_values([], "losses")


class SelectionAndComparisonTests(unittest.TestCase):
    def test_best_checkpoint_metric_and_tie_breaks(self) -> None:
        candidates = [
            {
                "name": "later",
                "interpolated_eer": 0.2,
                "empirical_average_error": 0.19,
                "order": 1,
            },
            {
                "name": "lower_eer",
                "interpolated_eer": 0.1,
                "empirical_average_error": 0.2,
                "order": 2,
            },
        ]
        self.assertEqual(select_best_checkpoint(candidates)["name"], "lower_eer")
        tied_eer = [
            {
                "name": "worse_empirical",
                "interpolated_eer": 0.1,
                "empirical_average_error": 0.11,
                "order": 0,
            },
            {
                "name": "better_empirical",
                "interpolated_eer": 0.1,
                "empirical_average_error": 0.10,
                "order": 1,
            },
        ]
        self.assertEqual(
            select_best_checkpoint(tied_eer)["name"], "better_empirical"
        )
        exact_tie = [
            {
                "name": "early",
                "interpolated_eer": 0.1,
                "empirical_average_error": 0.1,
                "order": 0,
            },
            {
                "name": "late",
                "interpolated_eer": 0.1,
                "empirical_average_error": 0.1,
                "order": 1,
            },
        ]
        self.assertEqual(select_best_checkpoint(exact_tie)["name"], "early")

    def test_model_quality_classification_and_baseline_math(self) -> None:
        self.assertEqual(
            classify_model_quality(BASELINE_INTERPOLATED_EER - 1e-4), "IMPROVED"
        )
        self.assertEqual(
            classify_model_quality(BASELINE_INTERPOLATED_EER + 1e-4), "DEGRADED"
        )
        self.assertEqual(
            classify_model_quality(BASELINE_INTERPOLATED_EER + 5e-13),
            "UNCHANGED_WITHIN_NUMERICAL_TOLERANCE",
        )
        comparison = baseline_comparison(
            {"interpolated_eer": 0.12}, {"interpolated_eer": 0.10}
        )
        self.assertAlmostEqual(comparison["signed_eer_difference"], 0.02)
        self.assertAlmostEqual(comparison["percentage_point_difference"], 2.0)
        self.assertAlmostEqual(comparison["relative_eer_change"], 0.2)
        self.assertEqual(comparison["model_quality_result"], "DEGRADED")


class PathAndTrialTests(unittest.TestCase):
    def test_final_test_and_unsafe_paths_are_rejected(self) -> None:
        for value in (
            "../outputs/train/x.pt",
            "C:/cache/train.pt",
            "outputs/fbank_cache_v1/test/shard_00000.pt",
            "manifests/portable/test_manifest_v1.csv",
            "outputs/fbank_cache_v1/part/shard_00000.pt",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_relative_artifact_path(value, "fixture")
        self.assertEqual(
            validate_relative_artifact_path(
                "outputs/fbank_cache_v1/validation/shard_00000.pt", "fixture"
            ),
            "outputs/fbank_cache_v1/validation/shard_00000.pt",
        )

    def test_immutable_trial_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trials.csv"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                require_trial_hash(path)
        self.assertEqual(len(EXPECTED_TRIAL_SHA256), 64)


def valid_metric_payload() -> dict[str, object]:
    summary = {
        "count": 9764,
        "minimum": -0.2,
        "mean": 0.4,
        "standard_deviation": 0.1,
        "median": 0.4,
        "maximum": 0.9,
    }
    return {
        "interpolated_eer": 0.1,
        "interpolated_eer_percentage": 10.0,
        "interpolated_threshold": 0.3,
        "interpolated_far": 0.1,
        "interpolated_frr": 0.1,
        "eer_kind": "linearly_interpolated_roc_crossing",
        "eer_threshold_kind": "interpolated_non_empirical",
        "empirical_threshold": 0.3,
        "empirical_far": 0.1,
        "empirical_frr": 0.1,
        "empirical_far_frr_gap": 0.0,
        "empirical_average_error": 0.1,
        "same_speaker_scores": summary,
        "different_speaker_scores": dict(summary),
        "score_range": [-0.2, 0.9],
        "score_count": 19528,
        "positive_trials": 9764,
        "negative_trials": 9764,
        "threshold_semantics": "accept same speaker when score >= threshold",
    }


class CheckpointTests(unittest.TestCase):
    class SyntheticScaler:
        def __init__(self) -> None:
            self.state = {
                "scale": 128.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 7,
            }

        def state_dict(self):
            return dict(self.state)

        def load_state_dict(self, state):
            self.state = dict(state)

    @staticmethod
    def modules():
        encoder = nn.Sequential(
            nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 192)
        )
        normalizer = nn.Identity()
        classifier = AAMSoftmax()
        optimizer = build_adamw_optimizer(encoder, classifier)
        scaler = CheckpointTests.SyntheticScaler()
        return encoder, normalizer, classifier, optimizer, scaler

    def checkpoint(self, *, step: int = 1000, reason: str = "epoch_complete"):
        encoder, normalizer, classifier, optimizer, scaler = self.modules()
        features = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        loss = F.cross_entropy(classifier(encoder(features), labels), labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(optimizer.state)
        self.assertTrue(scaler.state_dict())
        for state in optimizer.state.values():
            state["step"].fill_(step)
        hashes = {
            "outputs/fbank_cache_v1/fbank_cache_config_v1.json": "a" * 64,
            "outputs/fbank_cache_v1/train_feature_index_v1.csv": "b" * 64,
            "outputs/fbank_cache_v1/train/shard_00000.pt": "c" * 64,
        }
        next_epoch, next_position = resume_cursor(step)
        losses = [float(index + 1) for index in range(step)]
        checkpoint = {
            "schema_name": PILOT_SCHEMA_NAME,
            "schema_version": PILOT_SCHEMA_VERSION,
            "pretrained_model_identifier": "speechbrain/spkrec-ecapa-voxceleb",
            "embedding_model_state_dict": to_cpu_tree(encoder.state_dict()),
            "mean_var_norm_state_dict": to_cpu_tree(normalizer.state_dict()),
            "aam_classifier_state_dict": to_cpu_tree(classifier.state_dict()),
            "optimizer_state_dict": to_cpu_tree(optimizer.state_dict()),
            "grad_scaler_state_dict": to_cpu_tree(scaler.state_dict()),
            "epoch": 0,
            "next_epoch": next_epoch,
            "next_logical_batch_position": next_position,
            "global_optimizer_step": step,
            "sampler_configuration": dict(SAMPLER_CONFIGURATION),
            "physical_microbatch_size": PHYSICAL_MICROBATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "aam_configuration": dict(AAM_CONFIGURATION),
            "optimizer_configuration": copy.deepcopy(OPTIMIZER_CONFIGURATION),
            "batchnorm_policy": dict(BATCHNORM_POLICY),
            "amp_policy": dict(AMP_POLICY),
            "train_cache_identity": {
                "files": [
                    {"path": path, "sha256": digest}
                    for path, digest in sorted(hashes.items())
                ],
                "set_sha256": hash_mapping_digest(hashes),
            },
            "rng_state": capture_rng_state(include_cuda=False),
            "completed_training_loss_summary": summarize_finite_values(losses, "losses"),
            "checkpoint_creation_reason": reason,
            "validation_selection": (
                {
                    "selected_checkpoint": "epoch_000.pt",
                    "selection_scope": "best among trained pilot checkpoints",
                    "baseline_is_comparison_reference_only": True,
                    "model_quality_result": "DEGRADED",
                    "pilot_metrics": valid_metric_payload(),
                }
                if reason == "best_validation"
                else None
            ),
        }
        self.assertEqual(set(checkpoint), CHECKPOINT_KEYS)
        return checkpoint, (encoder, normalizer, classifier, optimizer, scaler)

    def test_atomic_epoch_checkpoint_and_full_synthetic_roundtrip(self) -> None:
        checkpoint, _ = self.checkpoint()
        validate_pilot_checkpoint(checkpoint)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "epoch_000.pt"
            atomic_save_pilot_checkpoint(checkpoint, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            loaded = load_pilot_checkpoint(path)
        fresh = self.modules()
        encoder, normalizer, classifier, optimizer, scaler = fresh
        encoder.load_state_dict(loaded["embedding_model_state_dict"])
        normalizer.load_state_dict(loaded["mean_var_norm_state_dict"])
        classifier.load_state_dict(loaded["aam_classifier_state_dict"])
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        scaler.load_state_dict(loaded["grad_scaler_state_dict"])
        apply_batchnorm_policy(encoder)
        checks = checkpoint_state_roundtrip_exact(
            loaded,
            embedding_model=encoder,
            mean_var_norm=normalizer,
            aam_classifier=classifier,
            optimizer=optimizer,
            grad_scaler=scaler,
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(loaded["next_epoch"], 1)
        self.assertEqual(loaded["next_logical_batch_position"], 0)
        self.assertEqual(loaded["global_optimizer_step"], 1000)

    def test_rolling_and_best_checkpoint_counter_rules(self) -> None:
        rolling, _ = self.checkpoint(step=250, reason="rolling")
        validate_pilot_checkpoint(rolling)
        self.assertEqual(rolling["next_logical_batch_position"], 250)
        best, _ = self.checkpoint(reason="best_validation")
        validate_pilot_checkpoint(best)
        self.assertEqual(best["checkpoint_creation_reason"], "best_validation")
        self.assertEqual(best["validation_selection"]["selected_checkpoint"], "epoch_000.pt")

    def test_malformed_checkpoints_fail_closed(self) -> None:
        checkpoint, _ = self.checkpoint()
        mutations = []
        missing = copy.deepcopy(checkpoint)
        del missing["rng_state"]
        mutations.append(missing)
        wrong_step = copy.deepcopy(checkpoint)
        wrong_step["global_optimizer_step"] = 999
        mutations.append(wrong_step)
        wrong_cursor = copy.deepcopy(checkpoint)
        wrong_cursor["next_epoch"] = 0
        mutations.append(wrong_cursor)
        absolute = copy.deepcopy(checkpoint)
        absolute["train_cache_identity"]["files"][0]["path"] = "C:/secret.pt"
        mutations.append(absolute)
        wrong_weight = copy.deepcopy(checkpoint)
        wrong_weight["aam_classifier_state_dict"]["weight"] = torch.zeros(2, 2)
        mutations.append(wrong_weight)
        for malformed in mutations:
            with self.subTest(keys=set(malformed)), self.assertRaises(ValueError):
                validate_pilot_checkpoint(malformed)


class BatchNormAndValidationTests(unittest.TestCase):
    def test_pilot_sampler_planning_does_not_stat_unselected_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = object.__new__(CachedFbankDataset)
            dataset.cache_dir = Path(temporary)
            dataset.split = "train"
            dataset.rows = tuple(
                CachedFbankRow(
                    relative_audio_path=f"audio/speaker_{speaker}_{sample}.wav",
                    feature_shard_path=f"train/shard_{speaker // 2:05d}.pt",
                    feature_index=2 * (speaker % 2) + sample,
                    speaker_id=f"speaker_{speaker}",
                    speaker_label=speaker,
                    final_split="train",
                    filename_group="train",
                )
                for speaker in range(4)
                for sample in range(2)
            )
            with self.assertRaises(FileNotFoundError):
                HybridShardAwareSpeakerBatchSampler(
                    dataset,
                    speakers_per_batch=2,
                    samples_per_speaker=2,
                    active_shard_window=1,
                    num_batches=1,
                )
            sampler = HybridShardAwareSpeakerBatchSampler(
                dataset,
                speakers_per_batch=2,
                samples_per_speaker=2,
                active_shard_window=1,
                num_batches=1,
                validate_shard_existence=False,
            )
            batch = next(iter(sampler))
            self.assertEqual(len(batch), 4)
            self.assertEqual(len(batch), len(set(batch)))

    def test_batchnorm_policy_reapplied_after_train_eval_transitions(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout())
        apply_batchnorm_policy(model)
        baseline = batchnorm_running_state(model)
        model.eval()
        apply_batchnorm_policy(model)
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        self.assertTrue(model[1].weight.requires_grad)
        assert_batchnorm_running_state_exact(model, baseline)

    def test_validation_forward_has_no_aam_argument_or_dependency(self) -> None:
        signature = inspect.signature(validation_forward_batch)
        self.assertNotIn("aam", signature.parameters)

        class Normalizer(nn.Module):
            def forward(self, features, lengths):
                return features

        class Embedding(nn.Module):
            def forward(self, features, lengths):
                pooled = features.mean(dim=(1, 2))
                return pooled[:, None, None].repeat(1, 1, 192)

        features = torch.ones(3, 301, 80, dtype=torch.float32)
        normalizer = Normalizer().eval()
        embedding_model = Embedding().eval()
        embeddings = validation_forward_batch(
            normalizer, embedding_model, features, device=torch.device("cpu")
        )
        self.assertEqual(tuple(embeddings.shape), (3, 192))
        self.assertEqual(embeddings.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(embeddings).all()))

    def test_validation_and_runtime_payloads_fail_closed(self) -> None:
        metric = valid_metric_payload()
        validate_validation_metrics_payload(metric)
        malformed = copy.deepcopy(metric)
        malformed["interpolated_eer"] = math.nan
        with self.assertRaises(ValueError):
            validate_validation_metrics_payload(malformed)
        runtime = {
            "schema_name": PILOT_RUNTIME_SCHEMA_NAME,
            "schema_version": PILOT_RUNTIME_SCHEMA_VERSION,
            "technical_pass": True,
            "training_pass": True,
            "checkpoint_pass": True,
            "validation_pass": True,
            "scope_compliance_pass": True,
            "overall_pass": True,
            "model_quality_result": "DEGRADED",
            "training_summary": {
                "optimizer_updates": 1000,
                "logical_sample_selections": 32000,
                "epoch_1_started": False,
                "loss": {"count": 1000},
                "logging_record_count": 20,
            },
            "checkpoint_summary": {
                "rolling_checkpoints": [
                    {"global_optimizer_step": step}
                    for step in ROLLING_CHECKPOINT_STEPS
                ],
                "epoch_checkpoint": {"global_optimizer_step": 1000},
                "epoch_roundtrip": {"optimizer_steps_taken_after_load": 0},
                "best_checkpoint": {"state_matches_epoch_000": True},
            },
            "validation_summary": {
                "runtime": {
                    "validation_rows": 8504,
                    "validation_speakers": 100,
                    "score_count": 19528,
                    "aam_classifier_calls": 0,
                }
            },
            "protected_hashes": {
                "unchanged": True,
                "mismatches": {},
                "files_compared": 6,
            },
            "validation_trial_hash": EXPECTED_TRIAL_SHA256,
            "final_test_accessed": False,
            "recursive_listing_used": False,
        }
        validate_runtime_payload(runtime)
        inconsistent = dict(runtime, overall_pass=False)
        with self.assertRaises(ValueError):
            validate_runtime_payload(inconsistent)
        partial = dict(runtime, validation_pass=False)
        with self.assertRaises(ValueError):
            validate_runtime_payload(partial)

    def test_rng_roundtrip_is_resumable(self) -> None:
        torch.manual_seed(9)
        state = capture_rng_state(include_cuda=False)
        expected = torch.rand(4)
        restore_rng_state(state, restore_cuda=False)
        self.assertTrue(torch.equal(expected, torch.rand(4)))

    def test_explicit_runner_has_no_waveform_frontend_or_epoch_one_path(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_ecapa_aam_one_epoch_pilot.py"
        ).read_text(encoding="utf-8")
        forbidden = (
            "mods.compute_features",
            "mods.classifier(",
            "encode_batch(",
            "src.fbank",
            "torchaudio",
            "set_epoch(1)",
            "for epoch in",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, script)
        self.assertIn("for position in range(NUM_LOGICAL_BATCHES)", script)
        self.assertIn('"microbatch_2_fallback_attempted": False', script)
        self.assertIn("validate_shard_existence=False", script)


if __name__ == "__main__":
    unittest.main()
