"""Approved identities and deterministic sampler-plan analysis for VieSpeaker2.0."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

from src.cached_fbank_dataset import CachedFbankDataset
from src.cached_fbank_samplers import HybridShardAwareSpeakerBatchSampler


APPROVED_INPUTS = {
    "portable_package_identity": {
        "path": "manifests/portable_v2/portable_manifests_v2_identity.json",
        "sha256": "29f374366c917fb6c54dcd44eeeff59ccc46a732b270188e7157a586e4d9e7c5",
    },
    "train_manifest": {
        "path": "manifests/portable_v2/train_manifest_v2.csv",
        "sha256": "f76aa0321f5f9a2714b2bad9f4b9ab0fd155075f26b50397f79931c8a4bd552b",
    },
    "validation_manifest": {
        "path": "manifests/portable_v2/validation_manifest_v2.csv",
        "sha256": "9c85332cbcd3e33818055c526c0bc54e5b86e7a2c4ed8b3c869433412c24a6fc",
    },
    "train_label_mapping": {
        "path": "manifests/portable_v2/speaker_to_label_v2.json",
        "sha256": "9d4e9015d25f023b8466f7932c296faece937104b17120bdb85c45ad10623cd8",
    },
    "cache_config": {
        "path": "outputs/fbank_cache_v2/fbank_cache_config_v2.json",
        "sha256": "ec71959ec64361038991e760e772d11bf1779e1b505a892dff364cf45aaeb018",
    },
    "cache_identity": {
        "path": "outputs/fbank_cache_v2/fbank_cache_identity_v2.json",
        "sha256": "1a2d6af777311f687e887575bfaf20915ed0409fd2e05b2f1ac232d43cd0b8c8",
    },
    "train_cache_index": {
        "path": "outputs/fbank_cache_v2/train_feature_index_v2.csv",
        "sha256": "e20c320fc5842502a26684023bb307a7b2afa27a14a3cf1130fdffe31e85d4b9",
    },
    "validation_cache_index": {
        "path": "outputs/fbank_cache_v2/validation_feature_index_v2.csv",
        "sha256": "1d3a95e95aaa5b13e6614c077fbbdf10f7c208c79970c2a85bdd461193b9f585",
    },
}

SAMPLER_PARAMETERS = {
    "sampler_class": "HybridShardAwareSpeakerBatchSampler",
    "speakers_per_batch": 16,
    "samples_per_speaker": 2,
    "batch_size": 32,
    "active_shard_window": 8,
    "seed": 20260729,
    "batches_per_epoch": 2969,
    "logical_selections_per_epoch": 95008,
    "num_workers": 0,
    "dataset_max_cached_shards": 8,
    "dataset_validate_finite": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def compact_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def verify_approved_inputs(repo_root: Path) -> dict[str, dict[str, str]]:
    verified: dict[str, dict[str, str]] = {}
    for name, binding in APPROVED_INPUTS.items():
        path = repo_root / binding["path"]
        actual = sha256_file(path)
        if actual != binding["sha256"]:
            raise ValueError(
                f"approved input hash mismatch for {name}: "
                f"expected {binding['sha256']}, got {actual}"
            )
        verified[name] = dict(binding)
    return verified


def validate_train_manifest_alignment(
    dataset: CachedFbankDataset, manifest_path: Path,
) -> dict[str, int]:
    if dataset.cache_version != 2 or dataset.split != "train":
        raise ValueError("v2 train dataset required")
    aligned = 0
    duplicate_rows = 0
    duplicate_groups: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "relative_audio_path",
            "speaker_id",
            "speaker_label",
            "final_split",
            "duplicate_group",
            "manifest_version",
        }
        if required - set(reader.fieldnames or ()):
            raise ValueError("train manifest schema is incomplete")
        for index, raw in enumerate(reader):
            if index >= len(dataset.rows):
                raise ValueError("train manifest has more rows than cache index")
            row = dataset.rows[index]
            expected = (
                raw["relative_audio_path"].strip(),
                raw["speaker_id"].strip(),
                int(raw["speaker_label"]),
                raw["final_split"].strip(),
                raw["duplicate_group"].strip(),
                raw["manifest_version"].strip(),
            )
            actual = (
                row.relative_audio_path,
                row.speaker_id,
                row.speaker_label,
                row.final_split,
                row.duplicate_group,
                row.manifest_version,
            )
            if actual != expected or row.manifest_row_index != index:
                raise ValueError(f"train manifest/cache alignment failed at row {index}")
            if row.duplicate_group:
                duplicate_rows += 1
                duplicate_groups.add(row.duplicate_group)
            aligned += 1
    if aligned != len(dataset):
        raise ValueError("train manifest has fewer rows than cache index")
    return {
        "aligned_rows": aligned,
        "nonempty_duplicate_group_rows": duplicate_rows,
        "nonempty_duplicate_groups": len(duplicate_groups),
    }


def _summary(values: Sequence[float | int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    numeric = [float(value) for value in values]
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean(numeric),
        "population_standard_deviation": pstdev(numeric),
    }


def _lru_stats(
    dataset: CachedFbankDataset, plans: Sequence[Sequence[int]], capacity: int,
) -> dict[str, int | float]:
    cache: OrderedDict[str, None] = OrderedDict()
    loads = hits = 0
    for batch in plans:
        for index in batch:
            shard = dataset.rows[index].feature_shard_path
            if shard in cache:
                hits += 1
                cache.move_to_end(shard)
            else:
                loads += 1
                cache[shard] = None
                while len(cache) > capacity:
                    cache.popitem(last=False)
    accesses = loads + hits
    return {
        "capacity": capacity,
        "accesses": accesses,
        "shard_loads": loads,
        "cache_hits": hits,
        "hit_rate": hits / accesses,
    }


def plan_sha256(epoch: int, batches: Sequence[Sequence[int]]) -> str:
    return compact_json_sha256({"epoch": epoch, "batches": batches})


def analyze_sampler_epoch(
    dataset: CachedFbankDataset,
    *,
    epoch: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    params = SAMPLER_PARAMETERS
    sampler = HybridShardAwareSpeakerBatchSampler(
        dataset,
        speakers_per_batch=params["speakers_per_batch"],
        samples_per_speaker=params["samples_per_speaker"],
        active_shard_window=params["active_shard_window"],
        num_batches=params["batches_per_epoch"],
        seed=params["seed"],
    )
    sampler.set_epoch(epoch)
    metadata_before = tuple(dataset.rows)
    batches = list(sampler)
    if tuple(dataset.rows) != metadata_before:
        raise RuntimeError("sampler mutated dataset metadata")
    if len(batches) != params["batches_per_epoch"]:
        raise RuntimeError("sampler produced an unexpected batch count")
    if sum(map(len, batches)) != params["logical_selections_per_epoch"]:
        raise RuntimeError("sampler produced an unexpected selection count")

    speaker_batch_exposure: Counter[str] = Counter()
    duplicate_group_conflicts = 0
    validation_or_test_selections = 0
    for batch_number, batch in enumerate(batches):
        if len(batch) != params["batch_size"]:
            raise RuntimeError(f"batch {batch_number} has wrong size")
        counts = Counter(dataset.rows[index].speaker_id for index in batch)
        if len(counts) != params["speakers_per_batch"] or set(counts.values()) != {
            params["samples_per_speaker"]
        }:
            raise RuntimeError(f"batch {batch_number} violates exact P x K")
        speaker_batch_exposure.update(counts.keys())
        groups_by_speaker: dict[str, list[str]] = {}
        for index in batch:
            row = dataset.rows[index]
            if row.final_split != "train":
                validation_or_test_selections += 1
            if row.duplicate_group:
                groups_by_speaker.setdefault(row.speaker_id, []).append(
                    row.duplicate_group
                )
        duplicate_group_conflicts += sum(
            len(groups) - len(set(groups)) for groups in groups_by_speaker.values()
        )
    if validation_or_test_selections:
        raise RuntimeError("sampler selected a non-train row")
    if duplicate_group_conflicts:
        raise RuntimeError("sampler paired a nonempty duplicate group")

    exposure_values = list(speaker_batch_exposure.values())
    if set(speaker_batch_exposure) != {
        row.speaker_id for row in dataset.rows
    }:
        raise RuntimeError("sampler epoch does not cover every train speaker")
    stats = dict(sampler.last_epoch_stats)
    distinct_shards = stats.pop("distinct_shards_per_batch")
    detailed = {
        "epoch": epoch,
        "plan_sha256": plan_sha256(epoch, batches),
        "batches": len(batches),
        "logical_selections": sum(map(len, batches)),
        "unique_selected_indexes": stats["unique_selected_indexes"],
        "repeated_selections": stats["repeated_selections"],
        "queue_cycles": stats["queue_cycles"],
        "duplicate_group_rejections": stats["duplicate_group_rejections"],
        "duplicate_group_conflicts": duplicate_group_conflicts,
        "validation_or_final_test_selections": validation_or_test_selections,
        "all_train_speakers_covered": True,
        "speaker_batch_exposure": {
            **_summary(exposure_values),
            "coefficient_of_variation": (
                pstdev(exposure_values) / mean(exposure_values)
            ),
            "total_speaker_slots": sum(exposure_values),
        },
        "distinct_shards_per_batch": _summary(distinct_shards),
        "lru_simulation": _lru_stats(
            dataset, batches, params["dataset_max_cached_shards"]
        ),
    }
    return batches, detailed


def build_sampler_readiness(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[int, list[list[int]]]]:
    bindings = verify_approved_inputs(repo_root)
    dataset = CachedFbankDataset(
        repo_root / "outputs/fbank_cache_v2",
        "train",
        max_cached_shards=SAMPLER_PARAMETERS["dataset_max_cached_shards"],
        validate_finite=SAMPLER_PARAMETERS["dataset_validate_finite"],
    )
    alignment = validate_train_manifest_alignment(
        dataset, repo_root / APPROVED_INPUTS["train_manifest"]["path"]
    )
    plans: dict[int, list[list[int]]] = {}
    epoch_reports: dict[str, Any] = {}
    for epoch in (0, 1):
        batches, report = analyze_sampler_epoch(dataset, epoch=epoch)
        plans[epoch] = batches
        epoch_reports[str(epoch)] = report
    repeat_batches, repeat_report = analyze_sampler_epoch(dataset, epoch=0)
    if repeat_batches != plans[0] or repeat_report["plan_sha256"] != epoch_reports["0"]["plan_sha256"]:
        raise RuntimeError("same-epoch sampler reproduction failed")
    if plans[0] == plans[1]:
        raise RuntimeError("different epochs unexpectedly produced the same plan")
    combined_plan_hash = compact_json_sha256(
        {
            "epoch_0": epoch_reports["0"]["plan_sha256"],
            "epoch_1": epoch_reports["1"]["plan_sha256"],
        }
    )
    config = {
        "schema_version": 2,
        "identity_kind": "training_sampler_v2",
        "approved": True,
        "parameters": dict(SAMPLER_PARAMETERS),
        "upstream_bindings": bindings,
        "dataset": {
            "train_rows": len(dataset),
            "train_speakers": len({row.speaker_id for row in dataset.rows}),
            "train_labels": [
                min(row.speaker_label for row in dataset.rows),
                max(row.speaker_label for row in dataset.rows),
            ],
            "train_shards": len(
                {row.feature_shard_path for row in dataset.rows}
            ),
            **alignment,
        },
        "sampler_plan": {
            "epochs_validated": [0, 1],
            "epoch_plan_sha256": {
                epoch: report["plan_sha256"]
                for epoch, report in epoch_reports.items()
            },
            "sampler_plan_identity_sha256": combined_plan_hash,
            "same_epoch_repeat_identical": True,
            "different_epoch_plans_differ": True,
            "pythonhashseed_independent": True,
        },
        "epoch_validation": epoch_reports,
        "metadata_only_planning": True,
        "comparative_benchmark_run": False,
        "timestamps_in_identity": False,
    }
    return config, plans
