from __future__ import annotations

import copy
import math
import unittest

import torch

from src.ecapa_multiepoch import (
    BASE_LRS,
    INITIAL_BEST_EER,
    INITIAL_EMPIRICAL_THRESHOLD,
    MAX_RESUMED_STEPS,
    EarlyStoppingState,
    ResumedCosineScheduler,
    cosine_factor,
    reject_final_test_path,
    start_epoch,
    validate_epoch0_migration,
)


def optimizer() -> torch.optim.Optimizer:
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    return torch.optim.SGD(
        [
            {"params": [first], "lr": BASE_LRS[0]},
            {"params": [second], "lr": BASE_LRS[1]},
        ]
    )


class FakeSampler:
    def __init__(self) -> None:
        self.epoch = -1
        self.calls: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.calls.append(epoch)
        self.epoch = epoch


class EpochPolicyTests(unittest.TestCase):
    def test_epoch_zero_is_never_retrained(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "never be retrained"):
            start_epoch(0, EarlyStoppingState(), FakeSampler())

    def test_epoch_transition_calls_sampler_set_epoch(self) -> None:
        sampler = FakeSampler()
        state = EarlyStoppingState()
        start_epoch(1, state, sampler)
        start_epoch(2, state, sampler)
        self.assertEqual(sampler.calls, [1, 2])
        self.assertEqual(sampler.epoch, 2)

    def test_no_epoch_starts_after_early_stopping(self) -> None:
        state = EarlyStoppingState()
        state.observe(1, INITIAL_BEST_EER)
        state.observe(2, INITIAL_BEST_EER)
        self.assertTrue(state.should_stop)
        sampler = FakeSampler()
        with self.assertRaisesRegex(RuntimeError, "after early stopping"):
            start_epoch(3, state, sampler)
        self.assertEqual(sampler.calls, [])


class SchedulerTests(unittest.TestCase):
    def test_cosine_start_and_end_values(self) -> None:
        self.assertEqual(cosine_factor(0), 1.0)
        self.assertTrue(
            math.isclose(
                cosine_factor(MAX_RESUMED_STEPS), 0.1, rel_tol=0, abs_tol=1e-15
            )
        )
        opt = optimizer()
        scheduler = ResumedCosineScheduler(opt)
        self.assertEqual(scheduler.lrs, BASE_LRS)
        for _ in range(MAX_RESUMED_STEPS):
            scheduler.step()
        self.assertTrue(
            math.isclose(scheduler.lrs[0], 1e-6, rel_tol=0, abs_tol=1e-18)
        )
        self.assertTrue(
            math.isclose(scheduler.lrs[1], 1e-4, rel_tol=0, abs_tol=1e-18)
        )
        with self.assertRaisesRegex(RuntimeError, "beyond"):
            scheduler.step()

    def test_scheduler_checkpoint_resume(self) -> None:
        first = ResumedCosineScheduler(optimizer())
        for _ in range(1379):
            first.step()
        state = copy.deepcopy(first.state_dict())
        second = ResumedCosineScheduler(optimizer())
        second.load_state_dict(state)
        self.assertEqual(second.completed_resumed_steps, 1379)
        self.assertEqual(second.state_dict(), state)
        first.step()
        second.step()
        self.assertEqual(second.state_dict(), first.state_dict())

    def test_scheduler_rejects_inconsistent_checkpoint(self) -> None:
        scheduler = ResumedCosineScheduler(optimizer())
        state = scheduler.state_dict()
        state["last_factor"] = 0.9
        with self.assertRaisesRegex(ValueError, "factor"):
            scheduler.load_state_dict(state)


class EarlyStoppingTests(unittest.TestCase):
    def test_patience_and_min_delta(self) -> None:
        state = EarlyStoppingState()
        # Exactly min_delta lower is not an improvement because the rule is strict.
        self.assertFalse(state.observe(1, INITIAL_BEST_EER - 0.0001))
        self.assertEqual(state.patience_counter, 1)
        self.assertFalse(state.observe(2, INITIAL_BEST_EER - 0.000099))
        self.assertTrue(state.should_stop)

    def test_improvement_resets_patience_and_selects_best(self) -> None:
        state = EarlyStoppingState()
        self.assertFalse(state.observe(1, INITIAL_BEST_EER))
        candidate = INITIAL_BEST_EER - 0.0001001
        self.assertTrue(state.observe(2, candidate))
        self.assertEqual(state.best_epoch, 2)
        self.assertEqual(state.best_eer, candidate)
        self.assertEqual(state.patience_counter, 0)
        self.assertFalse(state.should_stop)

    def test_best_checkpoint_not_replaced_without_required_delta(self) -> None:
        state = EarlyStoppingState(best_epoch=1, best_eer=0.05)
        self.assertFalse(state.observe(2, 0.04995))
        self.assertEqual(state.best_epoch, 1)
        self.assertEqual(state.best_eer, 0.05)


class MigrationAndQuarantineTests(unittest.TestCase):
    def test_epoch_zero_schema_migration(self) -> None:
        checkpoint = {
            "epoch": 0,
            "next_epoch": 1,
            "next_logical_batch_position": 0,
            "global_optimizer_step": 1000,
        }
        metrics = {
            "interpolated_eer": INITIAL_BEST_EER,
            "empirical_threshold": INITIAL_EMPIRICAL_THRESHOLD,
            "score_count": 19528,
        }
        validate_epoch0_migration(checkpoint, metrics)

    def test_epoch_zero_schema_migration_fails_closed(self) -> None:
        checkpoint = {
            "epoch": 0,
            "next_epoch": 0,
            "next_logical_batch_position": 0,
            "global_optimizer_step": 1000,
        }
        metrics = {
            "interpolated_eer": INITIAL_BEST_EER,
            "empirical_threshold": INITIAL_EMPIRICAL_THRESHOLD,
            "score_count": 19528,
        }
        with self.assertRaisesRegex(ValueError, "counters"):
            validate_epoch0_migration(checkpoint, metrics)
        checkpoint["next_epoch"] = 1
        checkpoint["scheduler_state_dict"] = {}
        with self.assertRaisesRegex(ValueError, "scheduler-less"):
            validate_epoch0_migration(checkpoint, metrics)

    def test_final_test_paths_are_rejected(self) -> None:
        forbidden = (
            "outputs/fbank_cache_v1/test/shard_00000.pt",
            "manifests/portable/test_manifest_v1.csv",
            "outputs/final-test/scores.pt",
            "reports/final_test_metrics.json",
        )
        for path in forbidden:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "final-test"):
                    reject_final_test_path(path)
        self.assertEqual(
            reject_final_test_path("manifests/verification/validation_trials_v1.csv"),
            "manifests/verification/validation_trials_v1.csv",
        )


if __name__ == "__main__":
    unittest.main()
