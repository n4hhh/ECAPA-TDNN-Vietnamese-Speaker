"""Deterministic CPU tests for AAM training and checkpoint helpers."""

from __future__ import annotations

import copy
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.aam_training import (
    AAMSoftmax,
    CHECKPOINT_SCHEMA_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    PRETRAINED_MODEL_ID,
    SEED,
    apply_batchnorm_policy,
    assert_batchnorm_running_state_exact,
    atomic_save_checkpoint,
    batch_at_position,
    batchnorm_running_state,
    build_adamw_optimizer,
    capture_rng_state,
    cpu_clone_state_dict,
    create_train_only_smoke_dataset,
    iter_microbatches,
    restore_rng_state,
    round_robin_reorder,
    scaled_cross_entropy_sum,
    to_cpu_tree,
    validate_checkpoint_v1,
    validate_optimizer_coverage,
    validate_train_only_smoke_request,
    values_exactly_equal,
)
from src.cached_fbank_dataset import create_cached_fbank_training_dataloader


class AAMSoftmaxTests(unittest.TestCase):
    def test_shape_finite_forward_backward_and_gradients(self) -> None:
        classifier = AAMSoftmax(embedding_dim=5, num_classes=4, seed=SEED)
        embeddings = torch.randn(6, 5, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
        logits = classifier(embeddings, labels)
        self.assertEqual(tuple(logits.shape), (6, 4))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        F.cross_entropy(logits, labels).backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertTrue(bool(torch.isfinite(embeddings.grad).all()))
        self.assertGreater(float(embeddings.grad.norm()), 0.0)
        self.assertIsNotNone(classifier.weight.grad)
        self.assertTrue(bool(torch.isfinite(classifier.weight.grad).all()))
        self.assertGreater(float(classifier.weight.grad.norm()), 0.0)

    def test_only_target_logits_receive_margin(self) -> None:
        classifier = AAMSoftmax(embedding_dim=3, num_classes=4, margin=0.2, scale=30.0)
        embeddings = torch.tensor(
            [[0.2, 0.5, 0.7], [0.8, -0.2, 0.1]], dtype=torch.float64
        )
        labels = torch.tensor([1, 3], dtype=torch.long)
        base = classifier.normalized_softmax_logits(embeddings)
        margin = classifier(embeddings, labels)
        mask = F.one_hot(labels, num_classes=4).bool()
        self.assertTrue(torch.equal(margin[~mask], base[~mask]))
        self.assertTrue(bool((margin[mask] != base[mask]).all()))

    def test_zero_margin_is_scaled_normalized_softmax(self) -> None:
        classifier = AAMSoftmax(
            embedding_dim=4, num_classes=3, margin=0.0, scale=11.0
        )
        embeddings = torch.randn(5, 4)
        labels = torch.tensor([0, 1, 2, 0, 1])
        self.assertTrue(torch.equal(
            classifier(embeddings, labels),
            classifier.normalized_softmax_logits(embeddings),
        ))

    def test_cosine_boundary_has_finite_backward(self) -> None:
        classifier = AAMSoftmax(embedding_dim=3, num_classes=3)
        with torch.no_grad():
            classifier.weight[0] = torch.tensor([1.0, 0.0, 0.0])
        embedding = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        classifier(embedding, torch.tensor([0])).sum().backward()
        self.assertTrue(bool(torch.isfinite(embedding.grad).all()))
        self.assertTrue(bool(torch.isfinite(classifier.weight.grad).all()))

    def test_deterministic_initialization_without_global_rng_change(self) -> None:
        torch.manual_seed(19)
        before = torch.get_rng_state().clone()
        first = AAMSoftmax(embedding_dim=4, num_classes=3, seed=SEED)
        after = torch.get_rng_state()
        second = AAMSoftmax(embedding_dim=4, num_classes=3, seed=SEED)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(first.weight, second.weight))

    def test_rejects_invalid_inputs_and_labels(self) -> None:
        classifier = AAMSoftmax(embedding_dim=3, num_classes=4)
        valid = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, "shape"):
            classifier(torch.randn(2, 3, 1), torch.tensor([0, 1]))
        with self.assertRaisesRegex(ValueError, "shape"):
            classifier(valid, torch.tensor([[0], [1]]))
        with self.assertRaisesRegex(TypeError, "torch.long"):
            classifier(valid, torch.tensor([0.0, 1.0]))
        for labels in (torch.tensor([-1, 1]), torch.tensor([0, 4])):
            with self.assertRaisesRegex(ValueError, "0..3"):
                classifier(valid, labels)
        invalid = valid.clone()
        invalid[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            classifier(invalid, torch.tensor([0, 1]))
        with torch.no_grad():
            classifier.weight[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            classifier(valid, torch.tensor([0, 1]))


class BatchAndOptimizerTests(unittest.TestCase):
    @staticmethod
    def logical_batch() -> dict[str, object]:
        labels = torch.tensor([3, 1, 0, 2, 0, 3, 2, 1], dtype=torch.long)
        indexes = torch.tensor([30, 10, 0, 20, 1, 31, 21, 11], dtype=torch.long)
        speakers = [f"s{label}" for label in labels.tolist()]
        return {
            "fbank": indexes.float().view(8, 1, 1),
            "dataset_index": indexes,
            "speaker_label": labels,
            "speaker_id": speakers,
            "relative_audio_path": [f"audio/{index}.wav" for index in indexes.tolist()],
            "final_split": ["train"] * 8,
            "filename_group": ["train"] * 8,
        }

    def test_round_robin_reorder_preserves_alignment_and_diversity(self) -> None:
        original = self.logical_batch()
        original_indexes = original["dataset_index"].clone()  # type: ignore[union-attr]
        reordered = round_robin_reorder(
            original, speakers_per_batch=4, samples_per_speaker=2
        )
        self.assertTrue(torch.equal(original["dataset_index"], original_indexes))
        self.assertEqual(
            reordered["speaker_label"].tolist(),  # type: ignore[union-attr]
            [0, 1, 2, 3, 0, 1, 2, 3],
        )
        self.assertEqual(
            reordered["fbank"].flatten().tolist(),  # type: ignore[union-attr]
            [float(value) for value in reordered["dataset_index"].tolist()],  # type: ignore[union-attr]
        )
        self.assertEqual(
            sorted(reordered["dataset_index"].tolist()),  # type: ignore[union-attr]
            sorted(original_indexes.tolist()),
        )
        for size in (2, 4):
            microbatches = list(iter_microbatches(reordered, size))
            self.assertTrue(
                all(len(set(microbatch["speaker_id"])) == size for microbatch in microbatches)
            )

    def test_round_robin_rejects_duplicates_and_bad_pk(self) -> None:
        batch = self.logical_batch()
        batch["dataset_index"][1] = batch["dataset_index"][0]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            round_robin_reorder(batch, speakers_per_batch=4, samples_per_speaker=2)
        batch = self.logical_batch()
        batch["speaker_label"][0] = 0  # type: ignore[index]
        batch["speaker_id"][0] = "s0"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "exactly"):
            round_robin_reorder(batch, speakers_per_batch=4, samples_per_speaker=2)

    def test_exact_loss_scaling_across_partitions(self) -> None:
        torch.manual_seed(11)
        base_logits = torch.randn(5, 4)
        labels = torch.tensor([0, 1, 2, 3, 1])
        direct_logits = base_logits.clone().requires_grad_(True)
        direct = F.cross_entropy(direct_logits, labels, reduction="mean")
        direct.backward()
        for partitions in ((2, 3), (1, 1, 1, 1, 1)):
            partitioned_logits = base_logits.clone().requires_grad_(True)
            start = 0
            total = torch.tensor(0.0)
            for width in partitions:
                stop = start + width
                scaled = scaled_cross_entropy_sum(
                    partitioned_logits[start:stop], labels[start:stop], 5
                )
                total = total + scaled.detach()
                scaled.backward()
                start = stop
            # Partitioned float32 additions may differ by one ULP while
            # representing the same sum-of-samples divided by five.
            self.assertTrue(torch.allclose(total, direct.detach(), atol=2e-7, rtol=0))
            self.assertTrue(torch.allclose(
                partitioned_logits.grad, direct_logits.grad, atol=1e-7, rtol=0
            ))

    def test_optimizer_groups_are_exact_and_disjoint(self) -> None:
        encoder = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 3))
        classifier = AAMSoftmax(embedding_dim=3, num_classes=4)
        optimizer = build_adamw_optimizer(encoder, classifier)
        validate_optimizer_coverage(optimizer, encoder, classifier)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-5, 1e-3])
        self.assertEqual(
            [group["weight_decay"] for group in optimizer.param_groups],
            [1e-4, 1e-4],
        )
        ids = [
            [id(parameter) for parameter in group["params"]]
            for group in optimizer.param_groups
        ]
        self.assertTrue(set(ids[0]).isdisjoint(ids[1]))
        optimizer.param_groups[1]["params"].append(optimizer.param_groups[0]["params"][0])
        with self.assertRaisesRegex(ValueError, "coverage|more than one"):
            validate_optimizer_coverage(optimizer, encoder, classifier)

    def test_batchnorm_policy_freezes_buffers_not_affine_or_other_modules(self) -> None:
        model = nn.Sequential(
            nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Dropout(0.2), nn.Linear(4, 2)
        )
        names = apply_batchnorm_policy(model)
        self.assertEqual(names, ("1",))
        self.assertTrue(model.training)
        self.assertTrue(model[0].training)
        self.assertFalse(model[1].training)
        self.assertTrue(model[3].training)
        self.assertTrue(model[1].weight.requires_grad)
        self.assertTrue(model[1].bias.requires_grad)
        before = batchnorm_running_state(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = model(torch.randn(8, 4)).square().mean()
        loss.backward()
        optimizer.step()
        assert_batchnorm_running_state_exact(model, before)

    def test_deterministic_resume_batch_positioning(self) -> None:
        batches = [[0, 1], [2, 3], [4, 5]]
        self.assertEqual(batch_at_position(batches, 1), [2, 3])
        self.assertEqual(batch_at_position(iter(batches), 1), [2, 3])
        with self.assertRaises(IndexError):
            batch_at_position(batches, 3)


class TrainOnlyGuardTests(unittest.TestCase):
    FIELDS = (
        "relative_audio_path", "feature_shard_path", "feature_index", "speaker_id",
        "speaker_label", "final_split", "filename_group", "feature_frames",
        "feature_dim", "feature_dtype",
    )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = self.root / "outputs" / "fbank_cache_v1"
        (self.cache / "train").mkdir(parents=True)
        config = {
            "version": 1,
            "frontend": "speechbrain/spkrec-ecapa-voxceleb",
            "feature_stage": "raw_compute_features_before_mean_var_norm",
            "feature_shape": [301, 80],
            "feature_dtype": "float32",
            "shard_size": 2,
            "expected_rows": {"train": 2, "validation": 0, "test": 0},
        }
        (self.cache / "fbank_cache_config_v1.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        self._write_index([0, 1])
        torch.save(
            {
                "features": torch.stack([
                    torch.zeros(301, 80), torch.ones(301, 80)
                ]),
                "speaker_labels": torch.tensor([0, 1], dtype=torch.long),
                "speaker_ids": ["s0", "s1"],
                "relative_audio_paths": ["audio/0.wav", "audio/1.wav"],
                "final_split": "train",
            },
            self.cache / "train" / "shard_00000.pt",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_index(
        self,
        labels: list[int],
        *,
        final_splits: list[str] | None = None,
        shard_paths: list[str] | None = None,
    ) -> None:
        final_splits = final_splits or ["train", "train"]
        shard_paths = shard_paths or ["train/shard_00000.pt", "train/shard_00000.pt"]
        rows = [
            {
                "relative_audio_path": f"audio/{index}.wav",
                "feature_shard_path": shard_paths[index],
                "feature_index": index,
                "speaker_id": f"s{index}",
                "speaker_label": labels[index],
                "final_split": final_splits[index],
                "filename_group": "train",
                "feature_frames": 301,
                "feature_dim": 80,
                "feature_dtype": "float32",
            }
            for index in range(2)
        ]
        with (self.cache / "train_feature_index_v1.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def create_dataset(self):
        return create_train_only_smoke_dataset(
            project_root=self.root,
            cache_dir=Path("outputs/fbank_cache_v1"),
            split="train",
            index_filename="train_feature_index_v1.csv",
            max_cached_shards=2,
            validate_finite=False,
        )

    def test_exact_train_request_and_all_train_metadata_are_accepted(self) -> None:
        dataset, guard = self.create_dataset()
        self.assertEqual(dataset.split, "train")
        self.assertEqual(len(dataset), 2)
        self.assertEqual(guard["request"]["index_path"], (
            "outputs/fbank_cache_v1/train_feature_index_v1.csv"
        ))
        self.assertTrue(guard["metadata"]["all_final_split_train"])
        self.assertTrue(guard["metadata"]["labels_in_0_487"])

    def test_non_train_splits_and_index_filenames_are_rejected_before_open(self) -> None:
        for split in ("validation", "test", "TRAIN", "train ", ""):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "split"):
                validate_train_only_smoke_request(
                    project_root=self.root,
                    cache_dir=Path("outputs/fbank_cache_v1"),
                    split=split,
                    index_filename="train_feature_index_v1.csv",
                )
        for filename in (
            "validation_feature_index_v1.csv",
            "test_feature_index_v1.csv",
            "./train_feature_index_v1.csv",
            "train_feature_index_v1.csv ",
        ):
            with self.subTest(filename=filename), self.assertRaisesRegex(
                ValueError, "index filename"
            ):
                validate_train_only_smoke_request(
                    project_root=self.root,
                    cache_dir=Path("outputs/fbank_cache_v1"),
                    split="train",
                    index_filename=filename,
                )

    def test_outside_and_ambiguous_cache_paths_are_rejected(self) -> None:
        for cache_dir in (
            Path("outputs"),
            Path("outputs/fbank_cache_v1/train"),
            self.root / "other_cache",
        ):
            with self.subTest(cache_dir=str(cache_dir)), self.assertRaisesRegex(
                ValueError, "cache directory"
            ):
                validate_train_only_smoke_request(
                    project_root=self.root,
                    cache_dir=cache_dir,
                    split="train",
                    index_filename="train_feature_index_v1.csv",
                )

    def test_negative_and_out_of_range_labels_are_rejected(self) -> None:
        for invalid in (-1, 488):
            self._write_index([invalid, 1])
            with self.subTest(label=invalid), self.assertRaisesRegex(
                ValueError, "0..487"
            ):
                self.create_dataset()

    def test_non_train_rows_and_non_train_shard_paths_are_rejected(self) -> None:
        self._write_index([0, 1], final_splits=["validation", "train"])
        with self.assertRaisesRegex(ValueError, "expected split train"):
            self.create_dataset()
        self._write_index(
            [0, 1],
            shard_paths=["other/shard_00000.pt", "train/shard_00000.pt"],
        )
        with self.assertRaisesRegex(ValueError, "train allowlist"):
            self.create_dataset()

    def test_dataset_schema_index_alignment_and_generator_compatibility(self) -> None:
        dataset, _ = self.create_dataset()
        rows_before = dataset.rows
        sample = dataset[1]
        legacy_keys = {
            "fbank", "speaker_label", "speaker_id", "relative_audio_path",
            "final_split", "filename_group",
        }
        self.assertTrue(legacy_keys.issubset(sample))
        self.assertEqual(set(sample), legacy_keys | {"dataset_index"})
        self.assertEqual(sample["dataset_index"], 1)
        generator = torch.Generator().manual_seed(7)
        batch = next(iter(create_cached_fbank_training_dataloader(
            dataset, [[1, 0]], num_workers=0, generator=generator  # type: ignore[arg-type]
        )))
        self.assertEqual(batch["dataset_index"].tolist(), [1, 0])
        self.assertEqual(batch["speaker_id"], ["s1", "s0"])
        self.assertIs(dataset.rows, rows_before)


class CheckpointTests(unittest.TestCase):
    def make_checkpoint(self) -> dict[str, object]:
        encoder = nn.Linear(2, 2)
        mean_var_norm = nn.Identity()
        classifier = AAMSoftmax(embedding_dim=2, num_classes=3)
        optimizer = build_adamw_optimizer(encoder, classifier)
        labels = torch.tensor([0, 1])
        loss = F.cross_entropy(classifier(encoder(torch.randn(2, 2)), labels), labels)
        loss.backward()
        optimizer.step()
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        return {
            "schema_name": CHECKPOINT_SCHEMA_NAME,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "pretrained_model_identifier": PRETRAINED_MODEL_ID,
            "embedding_model_state_dict": cpu_clone_state_dict(encoder),
            "mean_var_norm_state_dict": cpu_clone_state_dict(mean_var_norm),
            "aam_classifier_state_dict": cpu_clone_state_dict(classifier),
            "optimizer_state_dict": to_cpu_tree(optimizer.state_dict()),
            "grad_scaler_state_dict": scaler.state_dict(),
            "epoch": 0,
            "next_logical_batch_position": 1,
            "global_optimizer_step": 1,
            "sampler": {
                "name": "HybridShardAwareSpeakerBatchSampler",
                "seed": SEED,
                "epoch": 0,
                "speakers_per_batch": 16,
                "samples_per_speaker": 2,
                "active_shard_window": 8,
                "logical_batch_size": 32,
            },
            "physical_microbatch_size": 2,
            "accumulation_steps": 16,
            "aam": {
                "embedding_dim": 192,
                "num_classes": 488,
                "margin_radians": 0.2,
                "scale": 30.0,
                "initialization_seed": SEED,
            },
            "optimizer_groups": [
                {"name": "embedding_model", "learning_rate": 1e-5, "weight_decay": 1e-4},
                {"name": "aam_classifier", "learning_rate": 1e-3, "weight_decay": 1e-4},
            ],
            "batchnorm_policy": {
                "embedding_model_training": True,
                "batchnorm_modules_eval": True,
                "batchnorm_affine_trainable": True,
                "running_buffers_exactly_frozen": True,
            },
            "amp_policy": {
                "enabled": True,
                "ecapa_autocast_dtype": "float16",
                "aam_math_dtype": "float32",
                "loss_dtype": "float32",
                "grad_scaler_initial_scale": 128.0,
            },
            "train_data_identity": {
                "files": [{"path": "cache/train.csv", "sha256": "a" * 64}]
            },
            "next_logical_batch_identity": {
                "dataset_indexes": list(range(32)),
                "speaker_ids": [f"s{index // 2}" for index in range(32)],
                "relative_audio_paths": [f"audio/{index}.wav" for index in range(32)],
            },
            "rng_state": capture_rng_state(include_cuda=False),
        }

    def test_schema_and_atomic_roundtrip_with_synthetic_modules(self) -> None:
        checkpoint = self.make_checkpoint()
        validate_checkpoint_v1(checkpoint)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            atomic_save_checkpoint(checkpoint, path)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        self.assertTrue(values_exactly_equal(checkpoint, loaded))

    def test_rng_serialization_and_restoration(self) -> None:
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        state = capture_rng_state(include_cuda=False)
        expected = (random.random(), float(np.random.rand()), torch.rand(3))
        restore_rng_state(state, restore_cuda=False)
        actual = (random.random(), float(np.random.rand()), torch.rand(3))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_malformed_checkpoint_and_config_fail_closed(self) -> None:
        checkpoint = self.make_checkpoint()
        for mutation in ("missing_key", "wrong_microbatch", "unsafe_path", "bad_label_count"):
            malformed = copy.deepcopy(checkpoint)
            if mutation == "missing_key":
                del malformed["rng_state"]
            elif mutation == "wrong_microbatch":
                malformed["physical_microbatch_size"] = 1
            elif mutation == "unsafe_path":
                malformed["train_data_identity"]["files"][0]["path"] = "../train.csv"
            else:
                malformed["next_logical_batch_identity"]["dataset_indexes"] = [0] * 32
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_checkpoint_v1(malformed)


if __name__ == "__main__":
    unittest.main()
