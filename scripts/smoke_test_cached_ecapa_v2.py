"""Validate v2 cached DataLoaders and run cached train/validation ECAPA preflights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.cached_fbank_dataset import (
    CachedFbankDataset,
    create_cached_fbank_dataloader,
)
from src.speechbrain_frontend import SpeechBrainECAPAFrontend


SPLITS = ("train", "validation")
RESULT_FILENAME = "cached_ecapa_preflight_v2.json"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing preflight result: {path}")
    if temporary.exists():
        raise FileExistsError(f"refusing to replace stale temporary file: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    os.replace(temporary, path)


def validate_batch(
    batch: dict[str, Any],
    dataset: CachedFbankDataset,
    *,
    expected_indices: list[int],
) -> dict[str, Any]:
    count = len(expected_indices)
    expected_shape = (count, *dataset.feature_shape)
    features = batch["fbank"]
    labels = batch["speaker_label"]
    dataset_indices = batch["dataset_index"]
    manifest_indices = batch["manifest_row_index"]
    if (
        tuple(features.shape) != expected_shape
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not bool(torch.isfinite(features).all().item())
    ):
        raise AssertionError(f"invalid cached Fbank batch: {tuple(features.shape)}")
    if (
        labels.dtype != torch.long
        or labels.tolist()
        != [dataset.rows[index].speaker_label for index in expected_indices]
    ):
        raise AssertionError("cached batch labels are not aligned")
    if dataset_indices.tolist() != expected_indices:
        raise AssertionError("cached batch dataset_index values are not aligned")
    if manifest_indices.tolist() != expected_indices:
        raise AssertionError("cached batch manifest_row_index values are not aligned")
    for offset, index in enumerate(expected_indices):
        row = dataset.rows[index]
        if (
            batch["speaker_id"][offset] != row.speaker_id
            or batch["relative_audio_path"][offset] != row.relative_audio_path
            or batch["final_split"][offset] != row.final_split
            or batch["filename_group"][offset] != row.filename_group
            or batch["duplicate_group"][offset] != row.duplicate_group
            or batch["manifest_version"][offset] != "v2"
        ):
            raise AssertionError("cached batch metadata is not aligned")
    if dataset.split == "train":
        low, high = dataset.config["train_label_range"]
        if not all(low <= label <= high for label in labels.tolist()):
            raise AssertionError("train labels are outside the approved range")
    elif labels.tolist() != [-1] * count:
        raise AssertionError("validation labels must all be -1")
    return {
        "batch_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "label_dtype": str(labels.dtype),
        "dataset_indices": expected_indices,
        "finite": True,
        "metadata_aligned": True,
    }


def run_dataloader_checks(
    datasets: dict[str, CachedFbankDataset], batch_size: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        dataset = datasets[split]
        if len(dataset) != dataset.config["expected_rows"][split]:
            raise AssertionError(f"{split} Dataset length disagrees with authoritative index")
        loader = create_cached_fbank_dataloader(
            dataset, batch_size, shuffle=False, num_workers=0
        )
        iterator = iter(loader)
        first = next(iterator)
        second = next(iterator)
        zero_worker = [
            validate_batch(
                first, dataset, expected_indices=list(range(0, batch_size))
            ),
            validate_batch(
                second,
                dataset,
                expected_indices=list(range(batch_size, 2 * batch_size)),
            ),
        ]
        worker_batch_size = 2
        worker_loader = create_cached_fbank_dataloader(
            dataset, worker_batch_size, shuffle=False, num_workers=2
        )
        worker_batch = next(iter(worker_loader))
        two_workers = validate_batch(
            worker_batch,
            dataset,
            expected_indices=list(range(worker_batch_size)),
        )
        result[split] = {
            "dataset_length": len(dataset),
            "num_workers_0_sequential_batches": zero_worker,
            "num_workers_2_batch": two_workers,
            "bounded_lru_limit": dataset.max_cached_shards,
            "cache_version": dataset.cache_version,
        }
    return result


def run_cached_ecapa_preflight(
    datasets: dict[str, CachedFbankDataset],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    batches = {
        split: next(
            iter(
                create_cached_fbank_dataloader(
                    datasets[split],
                    batch_size,
                    shuffle=False,
                    num_workers=0,
                )
            )
        )
        for split in SPLITS
    }
    frontend = SpeechBrainECAPAFrontend(device=device)
    frontend.eval()
    frontend.classifier.eval()
    if not frontend.pretrained_parameters_frozen:
        raise AssertionError("pretrained parameters are not frozen")
    if frontend.training or frontend.classifier.training:
        raise AssertionError("cached preflight model is not in eval mode")

    module_call_counts = {"compute_features": 0, "classifier": 0}

    def count_compute_features(_module: Any, _inputs: Any, _output: Any) -> None:
        module_call_counts["compute_features"] += 1

    def count_classifier(_module: Any, _inputs: Any, _output: Any) -> None:
        module_call_counts["classifier"] += 1

    hooks = [
        frontend.classifier.mods.compute_features.register_forward_hook(
            count_compute_features
        )
    ]
    if "classifier" in frontend.classifier.mods:
        hooks.append(
            frontend.classifier.mods.classifier.register_forward_hook(count_classifier)
        )
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in frontend.named_parameters()
    }
    result: dict[str, Any] = {}
    try:
        for split in SPLITS:
            features_cpu = batches[split]["fbank"]
            labels = batches[split]["speaker_label"]
            if tuple(features_cpu.shape) != (
                batch_size,
                *datasets[split].feature_shape,
            ):
                raise AssertionError(f"{split} cached batch shape is invalid")
            if split == "train":
                low, high = datasets[split].config["train_label_range"]
                if not all(low <= label <= high for label in labels.tolist()):
                    raise AssertionError("train labels are invalid")
            elif labels.tolist() != [-1] * batch_size:
                raise AssertionError("validation labels are not all -1")
            features = features_cpu.to(device=device, dtype=torch.float32)
            lengths = torch.ones(batch_size, device=device, dtype=torch.float32)
            with torch.inference_mode():
                normalized = frontend.mean_var_norm(features, lengths)
                embeddings_3d = frontend.embedding_model(normalized, lengths)
                embeddings = embeddings_3d.squeeze(1)
            if (
                tuple(normalized.shape) != tuple(features.shape)
                or tuple(embeddings_3d.shape) != (batch_size, 1, 192)
                or tuple(embeddings.shape) != (batch_size, 192)
            ):
                raise AssertionError(f"{split} cached ECAPA tensor shape is invalid")
            if (
                features.dtype != torch.float32
                or normalized.dtype != torch.float32
                or embeddings.dtype != torch.float32
                or features.device != device
                or normalized.device != device
                or embeddings.device != device
            ):
                raise AssertionError(f"{split} cached ECAPA dtype/device is invalid")
            if not all(
                bool(torch.isfinite(tensor).all().item())
                for tensor in (features, normalized, embeddings)
            ):
                raise AssertionError(f"{split} cached ECAPA tensor is non-finite")
            result[split] = {
                "input_shape": list(features.shape),
                "normalized_shape": list(normalized.shape),
                "embedding_model_shape": list(embeddings_3d.shape),
                "embedding_shape_after_squeeze": list(embeddings.shape),
                "input_dtype": str(features.dtype),
                "normalized_dtype": str(normalized.dtype),
                "embedding_dtype": str(embeddings.dtype),
                "device": str(device),
                "labels": labels.tolist(),
                "input_finite": True,
                "normalized_finite": True,
                "embeddings_finite": True,
            }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        for hook in hooks:
            hook.remove()

    if module_call_counts != {"compute_features": 0, "classifier": 0}:
        raise AssertionError(f"forbidden pretrained module call: {module_call_counts}")
    parameters_unchanged = all(
        torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in frontend.named_parameters()
    )
    gradients_absent = all(
        parameter.grad is None for parameter in frontend.parameters()
    )
    if not parameters_unchanged or not gradients_absent:
        raise AssertionError("model parameter or gradient integrity check failed")
    result["invariants"] = {
        "eval_mode": True,
        "inference_mode_used": True,
        "parameters_frozen": True,
        "parameters_unchanged": parameters_unchanged,
        "gradients_absent": gradients_absent,
        "optimizer_created": False,
        "compute_features_call_count": module_call_counts["compute_features"],
        "pretrained_classifier_call_count": module_call_counts["classifier"],
        "feature_transpose_used": False,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/fbank_cache_v2")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-cached-shards", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda:0":
        raise ValueError("approved cached ECAPA preflight device is exactly cuda:0")
    if args.batch_size != 4:
        raise ValueError("approved cached ECAPA preflight batch size is exactly 4")
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved cached ECAPA preflight")
    cache_dir = args.cache_dir.expanduser().resolve(strict=True)
    datasets = {
        split: CachedFbankDataset(
            cache_dir,
            split,
            max_cached_shards=args.max_cached_shards,
            validate_finite=True,
        )
        for split in SPLITS
    }
    dataloader_checks = run_dataloader_checks(datasets, args.batch_size)
    ecapa_checks = run_cached_ecapa_preflight(
        datasets, device=device, batch_size=args.batch_size
    )
    result = {
        "result": "PASS",
        "included_splits": list(SPLITS),
        "dataloader_checks": dataloader_checks,
        "cached_ecapa_preflight": ecapa_checks,
        "final_test_cache_accessed": False,
        "waveform_accessed": False,
        "training_performed": False,
    }
    atomic_json(cache_dir / RESULT_FILENAME, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("V2 CACHED DATALOADER AND ECAPA CUDA PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
