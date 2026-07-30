from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from src.aam_training import (
    AAMSoftmax,
    build_adamw_optimizer,
    capture_rng_state,
    restore_rng_state,
    scaled_cross_entropy_sum,
    validate_optimizer_coverage,
    values_exactly_equal,
)
from src.ecapa_one_epoch_v2 import (
    AAM_CONFIGURATION,
    BATCHNORM_POLICY,
    CHECKPOINT_KEYS,
    FEATURE_SHAPE,
    LOGICAL_BATCH_SIZE,
    MIXED_PRECISION_CONFIGURATION,
    NUM_CLASSES,
    NUM_LOGICAL_BATCHES,
    OPTIMIZER_CONFIGURATION,
    PRETRAINED_MODEL_ID,
    SAMPLER_BINDING,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TRAINING_CONFIGURATION,
    UPSTREAM_BINDINGS,
    OneEpochV2Controller,
    assert_batchnorm_reference_exact,
    atomic_save_checkpoint,
    batchnorm_reference_state,
    checkpoint_roundtrip_exact,
    classify_baseline_comparison,
    finite_gradient_norm,
    load_checkpoint,
    require_successful_scaler_update,
    resume_cursor,
    summarize_values,
    validate_checkpoint_v2,
    validate_logical_batch,
    validate_relative_path,
    validate_sampler_binding,
)
from src.verification_v2 import empirical_confusion


class FakeScaler:
    def __init__(self) -> None:
        self.state = {"scale": 128.0, "growth_tracker": 0}

    def state_dict(self) -> dict[str, float | int]:
        return dict(self.state)

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.state = dict(state)


def checkpoint_fixture() -> tuple[
    dict[str, object], nn.Module, nn.Module, AAMSoftmax, torch.optim.AdamW,
    FakeScaler, torch.Generator,
]:
    embedding = nn.Linear(3, 2)
    mean_var_norm = nn.BatchNorm1d(3)
    aam = AAMSoftmax(
        embedding_dim=192,
        num_classes=NUM_CLASSES,
        seed=20260729,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding.parameters(), "lr": 1e-5, "weight_decay": 1e-4},
            {"params": aam.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
        ]
    )
    for parameter in list(embedding.parameters()) + list(aam.parameters()):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for state in optimizer.state.values():
        state["step"] = torch.tensor(float(NUM_LOGICAL_BATCHES))
    scaler = FakeScaler()
    generator = torch.Generator().manual_seed(20261729)
    checkpoint: dict[str, object] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "model_source": PRETRAINED_MODEL_ID,
        "upstream_bindings": copy.deepcopy(UPSTREAM_BINDINGS),
        "sampler_binding": dict(SAMPLER_BINDING),
        "feature_shape": list(FEATURE_SHAPE),
        "embedding_dimension": 192,
        "class_count": NUM_CLASSES,
        "label_range": [0, NUM_CLASSES - 1],
        "training_configuration": dict(TRAINING_CONFIGURATION),
        "aam_configuration": dict(AAM_CONFIGURATION),
        "optimizer_configuration": copy.deepcopy(OPTIMIZER_CONFIGURATION),
        "mixed_precision_configuration": dict(MIXED_PRECISION_CONFIGURATION),
        "batchnorm_policy": dict(BATCHNORM_POLICY),
        "epoch": 0,
        "completed_batch_position": NUM_LOGICAL_BATCHES - 1,
        "global_optimizer_step": NUM_LOGICAL_BATCHES,
        "next_cursor": resume_cursor(NUM_LOGICAL_BATCHES),
        "embedding_model_state": copy.deepcopy(embedding.state_dict()),
        "mean_var_norm_state": copy.deepcopy(mean_var_norm.state_dict()),
        "aam_state": copy.deepcopy(aam.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "grad_scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(include_cuda=False),
        "dataloader_generator_state": generator.get_state().clone(),
        "batchnorm_reference_state": {
            "bn.running_mean": torch.zeros(3),
            "bn.running_var": torch.ones(3),
            "bn.num_batches_tracked": torch.tensor(0),
        },
        "loss_summary": summarize_values([1.0] * NUM_LOGICAL_BATCHES),
        "creation_reason": "epoch_complete",
        "validation_selection": None,
        "stop_reason": "epoch_0_complete",
        "epoch_1_started": False,
    }
    return checkpoint, embedding, mean_var_norm, aam, optimizer, scaler, generator


class AAMAndLossTests(unittest.TestCase):
    def test_dynamic_1347_class_shape_and_v1_default_compatibility(self) -> None:
        v2 = AAMSoftmax(num_classes=NUM_CLASSES, seed=20260729)
        self.assertEqual(tuple(v2.weight.shape), (1347, 192))
        embeddings = torch.randn(4, 192)
        labels = torch.tensor([0, 1, 1345, 1346])
        logits = v2(embeddings, labels)
        self.assertEqual(tuple(logits.shape), (4, 1347))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(tuple(AAMSoftmax().weight.shape), (488, 192))

    def test_target_only_margin_and_exact_accumulated_loss(self) -> None:
        aam = AAMSoftmax(num_classes=NUM_CLASSES, seed=20260729)
        embeddings = torch.randn(8, 192)
        labels = torch.arange(8)
        margin_logits = aam(embeddings, labels)
        plain_logits = aam.normalized_softmax_logits(embeddings)
        mask = torch.nn.functional.one_hot(labels, NUM_CLASSES).bool()
        self.assertTrue(torch.equal(margin_logits[~mask], plain_logits[~mask]))
        direct_logits = margin_logits.detach().clone().requires_grad_(True)
        direct = torch.nn.functional.cross_entropy(
            direct_logits, labels, reduction="sum"
        ) / 8
        direct.backward()
        partitioned_logits = margin_logits.detach().clone().requires_grad_(True)
        accumulated = torch.tensor(0.0)
        for start in range(0, 8, 2):
            scaled = scaled_cross_entropy_sum(
                partitioned_logits[start:start + 2],
                labels[start:start + 2],
                8,
            )
            accumulated += scaled.detach()
            scaled.backward()
        self.assertTrue(
            torch.allclose(direct.detach(), accumulated, atol=2e-6, rtol=0)
        )
        self.assertTrue(
            torch.allclose(
                direct_logits.grad, partitioned_logits.grad, atol=1e-7, rtol=0
            )
        )


class PolicyAndBatchTests(unittest.TestCase):
    def test_optimizer_groups_cover_dynamic_ecapa_and_aam_once(self) -> None:
        embedding = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 3))
        aam = AAMSoftmax(embedding_dim=3, num_classes=NUM_CLASSES)
        optimizer = build_adamw_optimizer(embedding, aam)
        validate_optimizer_coverage(optimizer, embedding, aam)
        ids = [
            {id(parameter) for parameter in group["params"]}
            for group in optimizer.param_groups
        ]
        self.assertTrue(ids[0].isdisjoint(ids[1]))
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups], [1e-5, 1e-3]
        )

    def test_epoch_zero_only_and_exact_completion(self) -> None:
        controller = OneEpochV2Controller()
        controller.start(0)
        for position in range(NUM_LOGICAL_BATCHES):
            controller.record_update(position)
        result = controller.finish()
        self.assertEqual(result["optimizer_updates"], 2969)
        self.assertEqual(result["logical_selections"], 95008)
        self.assertFalse(result["epoch_1_started"])
        with self.assertRaises(RuntimeError):
            controller.start(1)

    def _batch(self) -> dict[str, object]:
        speakers = [str(index) for index in range(16) for _ in range(2)]
        return {
            "fbank": torch.zeros(32, 301, 80),
            "dataset_index": torch.arange(32),
            "speaker_label": torch.tensor(
                [index for index in range(16) for _ in range(2)]
            ),
            "speaker_id": speakers,
            "relative_audio_path": [f"{speaker}/{i}.wav" for i, speaker in enumerate(speakers)],
            "final_split": ["train"] * 32,
            "duplicate_group": [""] * 32,
        }

    def test_sampler_binding_and_duplicate_group_rejection(self) -> None:
        validate_sampler_binding(SAMPLER_BINDING)
        self.assertEqual(
            SAMPLER_BINDING["epoch_0_plan_sha256"],
            "010e80f042ae91f1c890045d12833005359833466ba6d5a4b5fc70e5c8f04a68",
        )
        batch = self._batch()
        validate_logical_batch(batch, list(range(32)))
        batch["duplicate_group"][0] = "dup"
        batch["duplicate_group"][1] = "dup"
        with self.assertRaisesRegex(ValueError, "duplicate-group"):
            validate_logical_batch(batch, list(range(32)))

    def test_batchnorm_buffers_preserved_and_mutation_detected(self) -> None:
        model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))
        model[0].eval()
        reference = batchnorm_reference_state(model)
        model(torch.randn(4, 3))
        assert_batchnorm_reference_exact(model, reference)
        model[0].running_mean.add_(1)
        with self.assertRaisesRegex(RuntimeError, "BatchNorm"):
            assert_batchnorm_reference_exact(model, reference)

    def test_scaler_skip_and_nonfinite_gradient_detection(self) -> None:
        require_successful_scaler_update(
            hook_before=3, hook_after=4, scale_before=128, scale_after=128
        )
        with self.assertRaisesRegex(RuntimeError, "GradScaler"):
            require_successful_scaler_update(
                hook_before=3, hook_after=3, scale_before=128, scale_after=64
            )
        parameter = nn.Parameter(torch.ones(2))
        parameter.grad = torch.tensor([1.0, float("nan")])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            finite_gradient_norm([("weight", parameter)], "synthetic")


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_roundtrip_rng_cursor_and_states(self) -> None:
        checkpoint, embedding, norm, aam, optimizer, scaler, generator = (
            checkpoint_fixture()
        )
        validate_checkpoint_v2(checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch_000.pt"
            atomic_save_checkpoint(checkpoint, path)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded["next_cursor"], {
                "next_epoch": 1, "next_batch_position": 0,
            })
            fresh_embedding = nn.Linear(3, 2)
            fresh_norm = nn.BatchNorm1d(3)
            fresh_aam = AAMSoftmax(num_classes=NUM_CLASSES, seed=20260729)
            fresh_optimizer = torch.optim.AdamW(
                [
                    {"params": fresh_embedding.parameters(), "lr": 1e-5, "weight_decay": 1e-4},
                    {"params": fresh_aam.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
                ]
            )
            fresh_scaler = FakeScaler()
            fresh_generator = torch.Generator()
            fresh_embedding.load_state_dict(loaded["embedding_model_state"])
            fresh_norm.load_state_dict(loaded["mean_var_norm_state"])
            fresh_aam.load_state_dict(loaded["aam_state"])
            fresh_optimizer.load_state_dict(loaded["optimizer_state"])
            fresh_scaler.load_state_dict(loaded["grad_scaler_state"])
            fresh_generator.set_state(loaded["dataloader_generator_state"])
            checks = checkpoint_roundtrip_exact(
                loaded,
                embedding_model=fresh_embedding,
                mean_var_norm=fresh_norm,
                aam=fresh_aam,
                optimizer=fresh_optimizer,
                scaler=fresh_scaler,
                dataloader_generator=fresh_generator,
            )
            self.assertTrue(all(checks.values()))
            restore_rng_state(loaded["rng_state"], restore_cuda=False)
            self.assertTrue(
                values_exactly_equal(
                    capture_rng_state(include_cuda=False), loaded["rng_state"]
                )
            )

    def test_identity_mismatch_and_atomic_failure_rejected(self) -> None:
        checkpoint, *_ = checkpoint_fixture()
        tampered = copy.deepcopy(checkpoint)
        tampered["upstream_bindings"]["train_manifest"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "upstream"):
            validate_checkpoint_v2(tampered)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            with mock.patch("torch.save", side_effect=OSError("synthetic")):
                with self.assertRaises(OSError):
                    atomic_save_checkpoint(checkpoint, path)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_checkpoint_schema_has_no_absolute_path(self) -> None:
        checkpoint, *_ = checkpoint_fixture()
        serialized_strings = []

        def collect(value: object) -> None:
            if isinstance(value, str):
                serialized_strings.append(value)
            elif isinstance(value, dict):
                for child in value.values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        collect(checkpoint)
        self.assertFalse(any(":\\" in value for value in serialized_strings))
        self.assertEqual(set(checkpoint), CHECKPOINT_KEYS)


class MetricsAndQuarantineTests(unittest.TestCase):
    def test_confusion_and_baseline_classifications(self) -> None:
        confusion = empirical_confusion([0.9, 0.6, 0.4, 0.1], [1, 0, 1, 0], 0.5)
        self.assertEqual(
            [confusion[key] for key in ("tp", "tn", "fp", "fn")],
            [1, 1, 1, 1],
        )
        self.assertEqual(
            classify_baseline_comparison(0.1)["classification"], "IMPROVED"
        )
        self.assertEqual(
            classify_baseline_comparison(0.1257)["classification"], "UNCHANGED"
        )
        self.assertEqual(
            classify_baseline_comparison(0.2)["classification"], "REGRESSED"
        )

    def test_final_test_paths_rejected_and_trial_binding_fixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "quarantined"):
            validate_relative_path(
                "manifests/portable_v2/test_manifest_v2.csv", "path"
            )
        self.assertEqual(
            UPSTREAM_BINDINGS["validation_trial_csv"]["sha256"],
            "11bec5ff0a0a4ca4930e2664bdc391388a9a677795afaefe5de3fee2d0d39e3d",
        )


if __name__ == "__main__":
    unittest.main()
