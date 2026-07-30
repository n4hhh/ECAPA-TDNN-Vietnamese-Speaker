#!/usr/bin/env python
"""Validate and approve the fixed VieSpeaker2.0 training sampler."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cached_fbank_dataset import (  # noqa: E402
    CachedFbankDataset,
    create_cached_fbank_training_dataloader,
)
from src.training_readiness_v2 import (  # noqa: E402
    SAMPLER_PARAMETERS,
    atomic_write_bytes,
    build_sampler_readiness,
    canonical_json_bytes,
    sha256_file,
)


def write_exact_or_fail(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"refusing to replace incompatible artifact: {path}")
    atomic_write_bytes(path, payload)


def smoke_first_batches(
    plans: list[list[int]], *, batch_count: int = 16,
) -> dict[str, object]:
    dataset = CachedFbankDataset(
        REPO_ROOT / "outputs/fbank_cache_v2",
        "train",
        max_cached_shards=SAMPLER_PARAMETERS["dataset_max_cached_shards"],
        validate_finite=SAMPLER_PARAMETERS["dataset_validate_finite"],
    )
    rows_before = tuple(dataset.rows)
    loader = create_cached_fbank_training_dataloader(
        dataset,
        plans[:batch_count],
        num_workers=SAMPLER_PARAMETERS["num_workers"],
    )
    started = time.perf_counter()
    observed = 0
    for batch_number, batch in enumerate(loader):
        if tuple(batch["fbank"].shape) != (32, 301, 80):
            raise RuntimeError(f"smoke batch {batch_number} has wrong feature shape")
        if batch["fbank"].dtype != torch.float32:
            raise RuntimeError("smoke batch has wrong feature dtype")
        if set(batch["final_split"]) != {"train"}:
            raise RuntimeError("smoke batch contains a non-train row")
        counts = Counter(batch["speaker_id"])
        if len(counts) != 16 or set(counts.values()) != {2}:
            raise RuntimeError("smoke batch violates exact P=16, K=2")
        for speaker_id, duplicate_group in zip(
            batch["speaker_id"], batch["duplicate_group"]
        ):
            if duplicate_group:
                matching = [
                    value
                    for current_speaker, value in zip(
                        batch["speaker_id"], batch["duplicate_group"]
                    )
                    if current_speaker == speaker_id and value
                ]
                if len(matching) != len(set(matching)):
                    raise RuntimeError("smoke batch paired a duplicate group")
        observed += 1
    duration = time.perf_counter() - started
    if observed != batch_count:
        raise RuntimeError("smoke loader returned an unexpected batch count")
    if tuple(dataset.rows) != rows_before:
        raise RuntimeError("smoke loading mutated dataset metadata")
    return {
        "status": "PASS",
        "scope": "first_16_real_cache_batches_only",
        "batches": observed,
        "samples": observed * 32,
        "seconds": duration,
        "samples_per_second": observed * 32 / duration,
        "feature_shape": [32, 301, 80],
        "feature_dtype": "float32",
        "num_workers": 0,
        "dataset_max_cached_shards": 8,
        "dataset_validate_finite": False,
        "shard_load_count": dataset.shard_load_count,
        "maximum_resident_shards": dataset.max_cached_shards,
        "metadata_unchanged": True,
        "exact_p_k": True,
        "duplicate_group_safe": True,
    }


def markdown_report(result: dict[str, object]) -> bytes:
    params = result["sampler_config"]["parameters"]
    epochs = result["sampler_config"]["epoch_validation"]
    smoke = result["real_cache_smoke"]
    lines = [
        "# VieSpeaker2.0 training sampler readiness",
        "",
        f"Result: **{result['result']}**",
        "",
        "The approved sampler is "
        f"`{params['sampler_class']}` with P={params['speakers_per_batch']}, "
        f"K={params['samples_per_speaker']}, batch size {params['batch_size']}, "
        f"active shard window {params['active_shard_window']}, seed "
        f"{params['seed']}, and {params['batches_per_epoch']} batches per epoch.",
        "",
        "Metadata-only plans for epochs 0 and 1 passed exact P x K, train-only "
        "selection, speaker coverage, duplicate-group separation, LRU simulation, "
        "same-epoch reproduction, and different-epoch change checks.",
        "",
        f"- Sampler plan identity: `{result['sampler_config']['sampler_plan']['sampler_plan_identity_sha256']}`",
        f"- Epoch 0 plan: `{epochs['0']['plan_sha256']}`",
        f"- Epoch 1 plan: `{epochs['1']['plan_sha256']}`",
        f"- Epoch 0 LRU(8) hit rate: {epochs['0']['lru_simulation']['hit_rate']:.6f}",
        f"- Epoch 1 LRU(8) hit rate: {epochs['1']['lru_simulation']['hit_rate']:.6f}",
        f"- Real-cache smoke: {smoke['batches']} batches / {smoke['samples']} samples in {smoke['seconds']:.3f} seconds",
        "",
        "No comparative sampler benchmark was run. This task approves only the "
        "fixed requested configuration; it does not start training.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-hash-only",
        action="store_true",
        help="emit the deterministic combined plan hash without writing artifacts",
    )
    args = parser.parse_args()

    config, plans = build_sampler_readiness(REPO_ROOT)
    if args.plan_hash_only:
        print(config["sampler_plan"]["sampler_plan_identity_sha256"])
        return
    smoke = smoke_first_batches(plans[0])
    config_payload = canonical_json_bytes(config)
    config_path = REPO_ROOT / "configs/v2/training_sampler_v2.json"
    write_exact_or_fail(config_path, config_payload)
    result = {
        "schema_version": 2,
        "report_kind": "training_sampler_v2",
        "result": "PASS",
        "sampler_config_path": "configs/v2/training_sampler_v2.json",
        "sampler_config_sha256": sha256_file(config_path),
        "sampler_config": config,
        "real_cache_smoke": smoke,
        "final_test_quarantine": {
            "manifest_opened": False,
            "audio_opened": False,
            "features_extracted": False,
            "trials_generated": False,
            "evaluation_run": False,
        },
        "training_started": False,
    }
    json_path = REPO_ROOT / "reports/training_sampler_v2.json"
    write_exact_or_fail(json_path, canonical_json_bytes(result))
    write_exact_or_fail(
        REPO_ROOT / "reports/training_sampler_v2.md",
        markdown_report(result),
    )
    print(json.dumps({
        "result": "PASS",
        "sampler_config_sha256": result["sampler_config_sha256"],
        "sampler_plan_identity_sha256": config["sampler_plan"][
            "sampler_plan_identity_sha256"
        ],
        "smoke_batches": smoke["batches"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
