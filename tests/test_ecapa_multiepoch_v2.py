from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.run_ecapa_aam_multiepoch_v2 import (
    atomic_torch_save,
    truncate_training_log,
)
from src.aam_training import (
    values_exactly_equal,
)
from src.ecapa_multiepoch_v2 import (
    AMP_OVERFLOW_POLICY,
    BASE_LRS,
    BATCHNORM_POLICY,
    EPOCH_PLAN_HASHES,
    FINAL_UPDATE_INDEX,
    IMPROVEMENT_TOLERANCE,
    INITIAL_BEST_EER,
    INITIAL_EMPIRICAL_AVERAGE_ERROR,
    INITIAL_EMPIRICAL_THRESHOLD,
    MULTIEPOCH_TRAINING_CONFIGURATION,
    NUM_CLASSES,
    OPTIMIZER_CONFIGURATION,
    RECOVERY_SCHEMA_VERSION,
    SAMPLER_BINDING,
    SCHEDULER_CONFIGURATION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    START_CHECKPOINT_PATH,
    START_CHECKPOINT_SHA256,
    START_GLOBAL_STEP,
    START_REPORT_PATH,
    START_REPORT_SHA256,
    STEPS_PER_EPOCH,
    TOTAL_RESUMED_UPDATES,
    UPSTREAM_BINDINGS,
    EarlyStoppingState,
    ExactResumedCosineScheduler,
    audit_epoch_plan,
    classify_recoverable_overflow,
    gradient_diagnostics,
    initial_overflow_state,
    logical_batch_identity,
    migrate_checkpoint_for_amp_recovery,
    multiplier_for_update,
    recovery_action,
    reject_final_test_path,
    retry_scale,
    stop_reason_after_validation,
    validate_epoch0_migration,
    validate_failed_attempt_invariants,
    validate_multiepoch_checkpoint,
)
from src.ecapa_one_epoch_v2 import (
    AAM_CONFIGURATION,
    FEATURE_SHAPE,
    MIXED_PRECISION_CONFIGURATION,
    finite_gradient_norm,
    require_successful_scaler_update,
)
from src.verification_metrics import calculate_eer
from src.verification_v2 import empirical_confusion
from tests.test_ecapa_one_epoch_v2 import checkpoint_fixture


def two_group_optimizer() -> torch.optim.AdamW:
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW(
        [
            {"params": [first], "lr": BASE_LRS[0], "weight_decay": 1e-4},
            {"params": [second], "lr": BASE_LRS[1], "weight_decay": 1e-4},
        ]
    )
    for parameter in (first, second):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    return optimizer


def set_optimizer_steps(
    state: dict[str, object], step: int
) -> dict[str, object]:
    result = copy.deepcopy(state)
    for value in result["state"].values():  # type: ignore[union-attr]
        value["step"] = torch.tensor(float(step))
    return result


def validation_record(epoch: int) -> dict[str, object]:
    return {
        "epoch": epoch,
        "training": {
            "optimizer_steps": STEPS_PER_EPOCH,
            "global_optimizer_step": START_GLOBAL_STEP + epoch * STEPS_PER_EPOCH,
        },
        "metrics": {
            "interpolated_eer": INITIAL_BEST_EER,
            "empirical_average_error": INITIAL_EMPIRICAL_AVERAGE_ERROR,
            "empirical_threshold": INITIAL_EMPIRICAL_THRESHOLD,
        },
        "runtime": {},
        "artifacts": {},
        "best": epoch == 0,
        "bad_epoch_count": max(0, epoch),
    }


def multiepoch_fixture(
    *,
    phase: str = "epoch_complete",
    epoch: int = 0,
    position: int | None = None,
    bad_epoch_count: int = 0,
    stop_reason: str | None = None,
) -> dict[str, object]:
    one, _, _, _, optimizer, _, _ = checkpoint_fixture()
    if epoch == 0:
        resumed = 0
        position = STEPS_PER_EPOCH
        completed = [0]
    elif phase == "training":
        if position is None:
            position = 7
        resumed = (epoch - 1) * STEPS_PER_EPOCH + position
        completed = list(range(epoch))
    elif phase == "validation_pending":
        position = STEPS_PER_EPOCH
        resumed = epoch * STEPS_PER_EPOCH
        completed = list(range(epoch))
    else:
        position = STEPS_PER_EPOCH
        resumed = epoch * STEPS_PER_EPOCH
        completed = list(range(epoch + 1))
    global_step = START_GLOBAL_STEP + resumed
    optimizer_state = set_optimizer_steps(optimizer.state_dict(), global_step)
    optimizer.load_state_dict(optimizer_state)
    scheduler = ExactResumedCosineScheduler(
        optimizer, completed_updates=resumed
    )
    early = EarlyStoppingState(bad_epoch_count=bad_epoch_count)
    if phase == "training":
        cursor = {"next_epoch": epoch, "next_batch_position": position}
    else:
        cursor = {"next_epoch": epoch + 1, "next_batch_position": 0}
    history = [validation_record(value) for value in completed]
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "original_epoch_zero_checkpoint": {
            "path": START_CHECKPOINT_PATH,
            "sha256": START_CHECKPOINT_SHA256,
            "completed_epoch": 0,
            "global_optimizer_step": START_GLOBAL_STEP,
        },
        "epoch_zero_report": {
            "path": START_REPORT_PATH,
            "sha256": START_REPORT_SHA256,
        },
        "upstream_bindings": copy.deepcopy(UPSTREAM_BINDINGS),
        "sampler_binding": copy.deepcopy(SAMPLER_BINDING),
        "epoch_plan_hashes": {
            str(key): value for key, value in EPOCH_PLAN_HASHES.items()
        },
        "model_source": "speechbrain/spkrec-ecapa-voxceleb",
        "feature_shape": list(FEATURE_SHAPE),
        "class_count": NUM_CLASSES,
        "label_range": [0, NUM_CLASSES - 1],
        "aam_configuration": copy.deepcopy(AAM_CONFIGURATION),
        "optimizer_configuration": copy.deepcopy(OPTIMIZER_CONFIGURATION),
        "training_configuration": copy.deepcopy(
            MULTIEPOCH_TRAINING_CONFIGURATION
        ),
        "mixed_precision_configuration": copy.deepcopy(
            MIXED_PRECISION_CONFIGURATION
        ),
        "scheduler_configuration": copy.deepcopy(SCHEDULER_CONFIGURATION),
        "embedding_model_state": copy.deepcopy(one["embedding_model_state"]),
        "mean_var_norm_state": copy.deepcopy(one["mean_var_norm_state"]),
        "aam_state": copy.deepcopy(one["aam_state"]),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "grad_scaler_state": copy.deepcopy(one["grad_scaler_state"]),
        "scheduler_state": scheduler.state_dict(),
        "batchnorm_policy": copy.deepcopy(BATCHNORM_POLICY),
        "batchnorm_reference_state": copy.deepcopy(
            one["batchnorm_reference_state"]
        ),
        "rng_state": copy.deepcopy(one["rng_state"]),
        "dataloader_generator_state": one[
            "dataloader_generator_state"
        ].clone(),
        "current_epoch": epoch,
        "completed_batch_position": position,
        "global_optimizer_step": global_step,
        "resumed_optimizer_steps": resumed,
        "next_cursor": cursor,
        "phase": phase,
        "completed_epochs": completed,
        "validation_history": history,
        "best_state": {
            "epoch": 0,
            "interpolated_eer": INITIAL_BEST_EER,
            "empirical_average_error": INITIAL_EMPIRICAL_AVERAGE_ERROR,
            "empirical_threshold": INITIAL_EMPIRICAL_THRESHOLD,
            "checkpoint_path": START_CHECKPOINT_PATH,
            "validation_source_checkpoint_path": START_CHECKPOINT_PATH,
            "validation_source_checkpoint_sha256": START_CHECKPOINT_SHA256,
        },
        "early_stopping_state": early.state_dict(),
        "current_epoch_statistics": {},
        "checkpoint_creation_reason": "unit_test",
        "stop_reason": stop_reason,
        "epoch_zero_retrained": False,
        "invalid_or_skipped_updates": 0,
    }


@dataclass
class FakeRow:
    speaker_id: str
    speaker_label: int
    final_split: str = "train"
    duplicate_group: str = ""


class FakeDataset:
    def __init__(self) -> None:
        self.rows = [
            FakeRow(str(speaker), speaker)
            for speaker in range(NUM_CLASSES)
            for _ in range(2)
        ]

    def __len__(self) -> int:
        return len(self.rows)


def synthetic_plan() -> list[list[int]]:
    batches: list[list[int]] = []
    for batch_index in range(STEPS_PER_EPOCH):
        speakers = [
            (batch_index * 16 + offset) % NUM_CLASSES for offset in range(16)
        ]
        batches.append(
            [
                index
                for speaker in speakers
                for index in (2 * speaker, 2 * speaker + 1)
            ]
        )
    return batches


class MigrationAndSchedulerTests(unittest.TestCase):
    def test_epoch_zero_checkpoint_migration_and_no_retraining(self) -> None:
        checkpoint, *_ = checkpoint_fixture()
        checkpoint["creation_reason"] = "best_validation"
        checkpoint["validation_selection"] = {
            "selected_checkpoint": "epoch_000.pt",
            "classification": "IMPROVED",
            "epoch_0_interpolated_eer": INITIAL_BEST_EER,
            "epoch_0_empirical_threshold": INITIAL_EMPIRICAL_THRESHOLD,
        }
        report = {
            "result": "PASS",
            "training": {"epoch_1_started": False},
            "validation": {
                "metrics": {"interpolated_eer": INITIAL_BEST_EER}
            },
        }
        validate_epoch0_migration(checkpoint, report)
        migrated = multiepoch_fixture()
        validate_multiepoch_checkpoint(migrated)
        self.assertFalse(migrated["epoch_zero_retrained"])
        self.assertEqual(migrated["next_cursor"], {"next_epoch": 1, "next_batch_position": 0})

    def test_scheduler_first_and_final_multiplier(self) -> None:
        optimizer = two_group_optimizer()
        scheduler = ExactResumedCosineScheduler(optimizer)
        index, multiplier, lrs = scheduler.apply_for_next_update()
        self.assertEqual((index, multiplier, lrs), (0, 1.0, BASE_LRS))
        scheduler.mark_successful_update(index)
        final = ExactResumedCosineScheduler(
            two_group_optimizer(), completed_updates=FINAL_UPDATE_INDEX
        )
        index, multiplier, lrs = final.apply_for_next_update()
        self.assertEqual(index, FINAL_UPDATE_INDEX)
        self.assertEqual(multiplier, 0.1)
        self.assertAlmostEqual(lrs[0], 1e-6, places=18)
        self.assertAlmostEqual(lrs[1], 1e-4, places=18)
        final.mark_successful_update(index)
        self.assertEqual(final.scheduler_completed_updates, TOTAL_RESUMED_UPDATES)

    def test_exact_mid_epoch_scheduler_resume_without_repeat_or_skip(self) -> None:
        first = ExactResumedCosineScheduler(two_group_optimizer())
        for expected in range(4173):
            index, _, _ = first.apply_for_next_update()
            self.assertEqual(index, expected)
            first.mark_successful_update(index)
        state = copy.deepcopy(first.state_dict())
        second = ExactResumedCosineScheduler(
            two_group_optimizer(), completed_updates=4173
        )
        second.load_state_dict(state)
        left = first.apply_for_next_update()
        right = second.apply_for_next_update()
        self.assertEqual(left, right)
        self.assertEqual(left[0], 4173)
        first.mark_successful_update(left[0])
        second.mark_successful_update(right[0])
        self.assertEqual(first.state_dict(), second.state_dict())

    def test_scheduler_rejects_wrong_completion_position(self) -> None:
        scheduler = ExactResumedCosineScheduler(two_group_optimizer())
        scheduler.apply_for_next_update()
        with self.assertRaisesRegex(RuntimeError, "out of position"):
            scheduler.mark_successful_update(1)

    def test_exact_global_step_targets(self) -> None:
        self.assertEqual(
            [START_GLOBAL_STEP + epoch * STEPS_PER_EPOCH for epoch in range(5)],
            [2969, 5938, 8907, 11876, 14845],
        )


class PlanAndRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = FakeDataset()
        cls.plan = synthetic_plan()

    def test_epoch_plan_identities_are_locked(self) -> None:
        self.assertEqual(
            EPOCH_PLAN_HASHES,
            {
                1: "c5ff9c7fca4c88541c47738706b1852293e6727da9bb420350e75379179bba0f",
                2: "a6ecc945b67ec425efd6939045587f3f0313f1f1b934c89ebcba3a6a16d6705d",
                3: "1d3319abf5965382664993880b2d3e7e75c9441aa3be65cbf71359279dbc3988",
                4: "583e0a1bfb4800e5a44028d05aa6278ab1e9dc12dac095ba85492643e524c33a",
            },
        )

    def test_exact_p16k2_all_speakers_and_duplicate_safety(self) -> None:
        audit = audit_epoch_plan(self.dataset, self.plan, epoch=1)
        self.assertEqual(audit["logical_batches"], STEPS_PER_EPOCH)
        self.assertEqual(audit["represented_speakers"], NUM_CLASSES)
        self.assertEqual(audit["duplicate_group_conflicts"], 0)

    def test_duplicate_group_conflict_rejected(self) -> None:
        dataset = copy.deepcopy(self.dataset)
        first = self.plan[0]
        dataset.rows[first[0]].duplicate_group = "dup"
        dataset.rows[first[1]].duplicate_group = "dup"
        with self.assertRaisesRegex(ValueError, "duplicate-group"):
            audit_epoch_plan(dataset, self.plan, epoch=1)

    def test_validation_pending_recovery_never_retrains(self) -> None:
        checkpoint = multiepoch_fixture(
            phase="validation_pending", epoch=1
        )
        self.assertEqual(recovery_action(checkpoint), "resume_validation_only")
        self.assertEqual(checkpoint["global_optimizer_step"], 5938)

    def test_corrupt_or_incompatible_resume_rejected(self) -> None:
        checkpoint = multiepoch_fixture()
        checkpoint["original_epoch_zero_checkpoint"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "epoch-zero"):
            validate_multiepoch_checkpoint(checkpoint)

    def test_no_absolute_path_persistence(self) -> None:
        checkpoint = multiepoch_fixture()
        checkpoint["best_state"]["checkpoint_path"] = "E:\\unsafe\\best.pt"
        with self.assertRaisesRegex(ValueError, "absolute"):
            validate_multiepoch_checkpoint(checkpoint)

    def test_final_test_paths_are_rejected_before_access(self) -> None:
        for path in (
            "manifests/portable_v2/test_manifest_v2.csv",
            "outputs/final-test/scores.pt",
            "outputs/fbank_cache_v2/test/shard.pt",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "final-test"):
                    reject_final_test_path(path)


class CheckpointAndNumericsTests(unittest.TestCase):
    def test_atomic_checkpoint_publication_and_full_state_roundtrip(self) -> None:
        checkpoint = multiepoch_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_torch_save(checkpoint, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            validate_multiepoch_checkpoint(loaded)
            self.assertTrue(values_exactly_equal(checkpoint, loaded))
            self.assertFalse((Path(directory) / "last.pt.tmp").exists())
            for key in (
                "optimizer_state",
                "grad_scaler_state",
                "rng_state",
                "dataloader_generator_state",
            ):
                self.assertTrue(values_exactly_equal(checkpoint[key], loaded[key]))

    def test_batchnorm_buffers_preserved_in_checkpoint(self) -> None:
        checkpoint = multiepoch_fixture()
        clone = copy.deepcopy(checkpoint["batchnorm_reference_state"])
        self.assertTrue(
            values_exactly_equal(checkpoint["batchnorm_reference_state"], clone)
        )

    def test_skipped_grad_scaler_update_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "skipped"):
            require_successful_scaler_update(
                hook_before=0,
                hook_after=0,
                scale_before=128.0,
                scale_after=64.0,
            )

    def test_nonfinite_gradient_rejected(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("nan")])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            finite_gradient_norm([("weight", parameter)], "test")

    def test_finite_zero_gradient_is_allowed_by_multiepoch_contract(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.zeros_like(parameter)
        self.assertEqual(
            finite_gradient_norm(
                [("weight", parameter)],
                "test",
                require_positive=False,
            ),
            0.0,
        )

    def test_large_finite_gradient_norm_uses_nonoverflowing_diagnostic(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([1.0e20])
        norm = finite_gradient_norm(
            [("weight", parameter)],
            "test",
            require_positive=False,
        )
        self.assertTrue(torch.isfinite(torch.tensor(norm, dtype=torch.float64)))

    def test_metrics_recomputation(self) -> None:
        scores = [0.9, 0.8, 0.2, 0.1]
        targets = [1, 1, 0, 0]
        result = calculate_eer(scores, targets)
        confusion = empirical_confusion(
            scores, targets, result.empirical_threshold
        )
        self.assertEqual(result.interpolated_eer, 0.0)
        self.assertEqual((confusion["tp"], confusion["tn"]), (2, 2))


class AmpOverflowRecoveryTests(unittest.TestCase):
    @staticmethod
    def diagnostic(*, finite: bool) -> dict[str, object]:
        return {
            "all_finite": finite,
            "nonfinite_elements": 0 if finite else 1,
        }

    def test_recoverable_overflow_classification(self) -> None:
        values = {
            "cached_features_finite": True,
            "normalized_features_finite": True,
            "embeddings_finite": True,
            "logits_finite": True,
            "unscaled_loss_finite": True,
            "model_parameters_finite": True,
            "optimizer_state_finite": True,
            "post_unscale_diagnostics": self.diagnostic(finite=False),
            "optimizer_step_called": False,
            "counters_advanced": False,
        }
        self.assertTrue(classify_recoverable_overflow(**values))
        for key in ("cached_features_finite", "unscaled_loss_finite"):
            changed = dict(values)
            changed[key] = False
            with self.subTest(key=key):
                self.assertFalse(classify_recoverable_overflow(**changed))

    def test_no_retry_after_step_or_counter_advance(self) -> None:
        common = {
            "cached_features_finite": True,
            "normalized_features_finite": True,
            "embeddings_finite": True,
            "logits_finite": True,
            "unscaled_loss_finite": True,
            "model_parameters_finite": True,
            "optimizer_state_finite": True,
            "post_unscale_diagnostics": self.diagnostic(finite=False),
        }
        self.assertFalse(
            classify_recoverable_overflow(
                **common, optimizer_step_called=True, counters_advanced=False
            )
        )
        self.assertFalse(
            classify_recoverable_overflow(
                **common, optimizer_step_called=False, counters_advanced=True
            )
        )

    def test_pre_and_post_unscale_elementwise_diagnostics(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(4))
        parameter.grad = torch.tensor(
            [1.0, float("nan"), float("inf"), float("-inf")]
        )
        result = gradient_diagnostics([("blocks.weight", parameter)])
        self.assertEqual(result["finite_elements"], 1)
        self.assertEqual(result["nan_elements"], 1)
        self.assertEqual(result["positive_inf_elements"], 1)
        self.assertEqual(result["negative_inf_elements"], 1)
        self.assertEqual(result["affected_parameter_names"], ["blocks.weight"])

    def test_exact_half_scale_and_floor(self) -> None:
        self.assertEqual(retry_scale(2048.0), 1024.0)
        self.assertEqual(
            AMP_OVERFLOW_POLICY["maximum_retries_per_logical_batch"], 1
        )
        with self.assertRaisesRegex(RuntimeError, "floor"):
            retry_scale(1.0)

    def test_stable_identity_accepts_round_robin_order_only(self) -> None:
        batch = {
            "dataset_index": torch.tensor([3, 1, 4, 2]),
            "speaker_label": torch.tensor([1, 0, 1, 0]),
            "speaker_id": ["1", "0", "1", "0"],
            "relative_audio_path": ["d", "b", "e", "c"],
            "duplicate_group": ["", "", "", ""],
        }
        identity = logical_batch_identity(batch, [1, 2, 3, 4])
        self.assertEqual(identity, logical_batch_identity(batch, [1, 2, 3, 4]))
        self.assertEqual(len(identity), 64)
        changed = copy.deepcopy(batch)
        changed["dataset_index"][0] = 99
        with self.assertRaisesRegex(ValueError, "sampler"):
            logical_batch_identity(changed, [1, 2, 3, 4])

    def test_failed_attempt_transaction_does_not_advance(self) -> None:
        snapshot = {
            "optimizer_steps": (7969, 7969),
            "optimizer_hook_count": 0,
            "parameter_versions": [10, 11],
            "scheduler_completed_updates": 5000,
            "global_optimizer_step": 7969,
            "batch_position": 2032,
            "logical_loss_count": 2031,
            "batch_identity": "a" * 64,
            "learning_rates": [6.6e-6, 6.6e-4],
        }
        validate_failed_attempt_invariants(snapshot, copy.deepcopy(snapshot))
        for key in (
            "optimizer_steps",
            "optimizer_hook_count",
            "parameter_versions",
            "scheduler_completed_updates",
            "global_optimizer_step",
            "batch_position",
            "logical_loss_count",
            "batch_identity",
            "learning_rates",
        ):
            changed = copy.deepcopy(snapshot)
            changed[key] = "changed"
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, "advanced"):
                    validate_failed_attempt_invariants(snapshot, changed)

    def test_exact_recovery_cursor_migration_and_roundtrip(self) -> None:
        checkpoint = multiepoch_fixture(
            phase="training", epoch=2, position=2031
        )
        migrated = migrate_checkpoint_for_amp_recovery(checkpoint)
        self.assertEqual(migrated["schema_version"], RECOVERY_SCHEMA_VERSION)
        self.assertEqual(migrated["next_cursor"], {
            "next_epoch": 2,
            "next_batch_position": 2031,
        })
        self.assertEqual(migrated["global_optimizer_step"], 7969)
        self.assertEqual(migrated["overflow_state"], initial_overflow_state())
        validate_multiepoch_checkpoint(migrated)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_torch_save(migrated, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            validate_multiepoch_checkpoint(loaded)
            self.assertTrue(values_exactly_equal(migrated, loaded))

    def test_atomic_log_reconciliation_removes_only_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_scalars.jsonl"
            records = [
                {"resumed_optimizer_steps": step, "value": step}
                for step in (4900, 5000, 5100, 5200)
            ]
            path.write_text(
                "".join(
                    f"{__import__('json').dumps(record)}\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            result = truncate_training_log(5000, path)
            self.assertEqual(result["removed_resumed_steps"], [5100, 5200])
            retained = [
                __import__("json").loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["resumed_optimizer_steps"] for row in retained],
                [4900, 5000],
            )
            second = truncate_training_log(5000, path)
            self.assertEqual(second["removed_count"], 0)


class SelectionAndStopTests(unittest.TestCase):
    def test_best_checkpoint_selection_and_eer_tie_break(self) -> None:
        state = EarlyStoppingState()
        improved = state.observe(
            1,
            interpolated_eer=INITIAL_BEST_EER,
            empirical_average_error=INITIAL_EMPIRICAL_AVERAGE_ERROR - 0.001,
            empirical_threshold=0.2,
        )
        self.assertTrue(improved)
        self.assertEqual(state.best_epoch, 1)
        retained = state.observe(
            2,
            interpolated_eer=INITIAL_BEST_EER + IMPROVEMENT_TOLERANCE,
            empirical_average_error=state.best_empirical_average_error,
            empirical_threshold=0.3,
        )
        self.assertFalse(retained)
        self.assertEqual(state.best_epoch, 1)

    def test_early_stopping_patience(self) -> None:
        state = EarlyStoppingState()
        for epoch in (1, 2):
            self.assertFalse(
                state.observe(
                    epoch,
                    interpolated_eer=INITIAL_BEST_EER + 0.01,
                    empirical_average_error=0.1,
                    empirical_threshold=0.2,
                )
            )
        self.assertEqual(stop_reason_after_validation(2, state), "early_stopping")

    def test_max_epoch_stop(self) -> None:
        state = EarlyStoppingState()
        self.assertEqual(stop_reason_after_validation(4, state), "max_epoch")

    def test_stopped_checkpoint_state(self) -> None:
        checkpoint = multiepoch_fixture(
            phase="stopped",
            epoch=2,
            bad_epoch_count=2,
            stop_reason="early_stopping",
        )
        validate_multiepoch_checkpoint(checkpoint)
        self.assertEqual(recovery_action(checkpoint), "stop")


if __name__ == "__main__":
    unittest.main()
