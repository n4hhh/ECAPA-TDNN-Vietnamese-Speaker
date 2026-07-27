"""Inference-only CUDA smoke test from cached Fbank to ECAPA embeddings."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import speechbrain
import torch

from src.cached_fbank_dataset import CachedFbankDataset, create_cached_fbank_dataloader
from src.speechbrain_frontend import SpeechBrainECAPAFrontend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/fbank_cache_v1"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cached-shards", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    datasets = {
        split: CachedFbankDataset(args.cache_dir, split, args.max_cached_shards)
        for split in ("train", "validation", "test")
    }
    if any(row.speaker_label != -1 for split in ("validation", "test") for row in datasets[split].rows):
        raise AssertionError("validation/test labels must all remain -1")
    if any(not 0 <= row.speaker_label <= 487 for row in datasets["train"].rows):
        raise AssertionError("train labels must all be in 0..487")
    loader = create_cached_fbank_dataloader(
        datasets["train"], args.batch_size, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    features_cpu = batch["fbank"]
    labels = batch["speaker_label"]
    if tuple(features_cpu.shape) != (args.batch_size, 301, 80):
        raise AssertionError(f"unexpected cached batch shape: {tuple(features_cpu.shape)}")
    if features_cpu.dtype != torch.float32 or labels.dtype != torch.long:
        raise AssertionError("unexpected batch dtype")
    if not bool(torch.isfinite(features_cpu).all().item()):
        raise AssertionError("cached batch contains NaN or Inf")

    frontend = SpeechBrainECAPAFrontend(device=device)
    frontend.eval()
    encoder = frontend.classifier
    encoder.eval()
    if not frontend.pretrained_parameters_frozen:
        raise AssertionError("pretrained parameters are not frozen")
    before = {name: parameter.detach().cpu().clone() for name, parameter in encoder.named_parameters()}
    features = features_cpu.to(device)
    lengths = torch.ones(args.batch_size, device=device, dtype=torch.float32)
    with torch.inference_mode():
        normalized = encoder.mods.mean_var_norm(features, lengths)
        embeddings = encoder.mods.embedding_model(normalized, lengths)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    unchanged = all(torch.equal(before[name], parameter.detach().cpu()) for name, parameter in encoder.named_parameters())
    checks = {
        "input_shape": list(features.shape),
        "normalized_shape": list(normalized.shape),
        "label_shape": list(labels.shape),
        "embedding_shape": list(embeddings.shape),
        "input_dtype": str(features.dtype),
        "label_dtype": str(labels.dtype),
        "embedding_dtype": str(embeddings.dtype),
        "device": str(features.device),
        "input_finite": bool(torch.isfinite(features).all().item()),
        "normalized_finite": bool(torch.isfinite(normalized).all().item()),
        "embedding_finite": bool(torch.isfinite(embeddings).all().item()),
        "parameters_frozen": frontend.pretrained_parameters_frozen,
        "parameters_unchanged": unchanged,
        "dataset_lengths": {split: len(dataset) for split, dataset in datasets.items()},
        "validation_labels_all_minus_one": True,
        "test_labels_all_minus_one": True,
        "train_label_min": min(row.speaker_label for row in datasets["train"].rows),
        "train_label_max": max(row.speaker_label for row in datasets["train"].rows),
        "num_workers": 0,
    }
    if checks["normalized_shape"] != [args.batch_size, 301, 80]:
        raise AssertionError("mean_var_norm changed cached feature shape")
    if checks["embedding_shape"] != [args.batch_size, 1, 192]:
        raise AssertionError("unexpected embedding shape")
    if not all((checks["input_finite"], checks["normalized_finite"], checks["embedding_finite"], unchanged)):
        raise AssertionError("finite-value or parameter-integrity check failed")
    print(f"Python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"SpeechBrain: {speechbrain.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(json.dumps(checks, indent=2, sort_keys=True))
    print("CACHED ECAPA INFERENCE SMOKE TEST: PASS")
    print("No waveform loading, compute_features, classifier inference, training, backward pass, optimizer, or parameter update was used.")


if __name__ == "__main__":
    main()
