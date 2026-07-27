"""Analyze train-cache metadata and benchmark short deterministic access runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.cached_fbank_dataset import CachedFbankDataset, collate_cached_fbank

SEED = 20260727
PATTERNS = ("sequential", "sample-random", "shard-local")


def percentile(values: Sequence[float | int], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values: Sequence[float | int]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "mean": float(statistics.fmean(values)),
        "q1": percentile(values, 0.25),
        "median": float(statistics.median(values)),
        "q3": percentile(values, 0.75),
        "max": float(max(values)),
    }


def diagnostic_order(rows: Sequence[Any], pattern: str, seed: int) -> list[int]:
    if pattern == "sequential":
        return list(range(len(rows)))
    rng = random.Random(seed)
    if pattern == "sample-random":
        order = list(range(len(rows)))
        rng.shuffle(order)
        return order
    if pattern == "shard-local":
        by_shard: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            by_shard[row.feature_shard_path].append(index)
        shards = sorted(by_shard)
        rng.shuffle(shards)
        order = []
        for shard in shards:
            local = by_shard[shard][:]
            rng.shuffle(local)
            order.extend(local)
        return order
    raise ValueError(f"unknown access pattern: {pattern}")


def analyze_metadata(dataset: CachedFbankDataset, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = dataset.rows
    speakers = {row.speaker_id for row in rows}
    labels = {row.speaker_label for row in rows}
    shards: dict[str, list[Any]] = defaultdict(list)
    speaker_rows: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        shards[row.feature_shard_path].append(row)
        speaker_rows[row.speaker_id].append(row)
    if len(rows) != 31_998 or len(speakers) != 488 or labels != set(range(488)) or len(shards) != 125:
        raise AssertionError(
            f"unexpected train totals: rows={len(rows)}, speakers={len(speakers)}, "
            f"labels={len(labels)}, shards={len(shards)}"
        )

    shard_stats = []
    for path in sorted(shards):
        selected = shards[path]
        counts = Counter(row.speaker_id for row in selected)
        positions = [row.feature_index for row in selected]
        valid_unique = (
            len(positions) == len(set(positions))
            and min(positions) >= 0
            and max(positions) < int(dataset.config["shard_size"])
            and set(positions) == set(range(len(selected)))
        )
        dominant = max(counts.values())
        shard_stats.append({
            "feature_shard_path": path,
            "utterance_count": len(selected),
            "unique_speakers": len(counts),
            "min_utterances_per_represented_speaker": min(counts.values()),
            "max_utterances_per_represented_speaker": max(counts.values()),
            "dominant_speaker_utterance_count": dominant,
            "dominant_speaker_share": dominant / len(selected),
            "feature_indexes_valid_and_unique": valid_unique,
        })

    speaker_stats = []
    for speaker_id in sorted(speaker_rows, key=lambda value: speaker_rows[value][0].speaker_label):
        selected = speaker_rows[speaker_id]
        speaker_stats.append({
            "speaker_id": speaker_id,
            "speaker_label": selected[0].speaker_label,
            "utterance_count": len(selected),
            "shard_count": len({row.feature_shard_path for row in selected}),
        })

    run_lengths: list[int] = []
    current = rows[0].speaker_id
    run = 0
    for row in rows:
        if row.speaker_id != current:
            run_lengths.append(run)
            current, run = row.speaker_id, 0
        run += 1
    run_lengths.append(run)
    transitions = sum(left.speaker_id != right.speaker_id for left, right in zip(rows, rows[1:]))

    diversity: dict[str, Any] = {}
    for pattern in PATTERNS:
        order = diagnostic_order(rows, pattern, seed)
        diversity[pattern] = {}
        for batch_size in (16, 32):
            complete = len(order) // batch_size
            unique_counts, maxima = [], []
            duplicate_batches = quarter_batches = half_batches = 0
            represented_counts: Counter[int] = Counter()
            for start in range(0, complete * batch_size, batch_size):
                counts = Counter(rows[index].speaker_id for index in order[start:start + batch_size])
                unique_counts.append(len(counts))
                maximum = max(counts.values())
                maxima.append(maximum)
                represented_counts.update(counts.values())
                duplicate_batches += maximum > 1
                quarter_batches += maximum >= math.ceil(batch_size * 0.25)
                half_batches += maximum >= math.ceil(batch_size * 0.50)
            diversity[pattern][str(batch_size)] = {
                "complete_batches": complete,
                "unique_speakers_per_batch": summary(unique_counts),
                "max_samples_from_one_speaker": {
                    "min": min(maxima), "mean": statistics.fmean(maxima),
                    "median": statistics.median(maxima), "max": max(maxima),
                },
                "batches_with_duplicate_speakers_pct": duplicate_batches * 100 / complete,
                "batches_one_speaker_at_least_25pct_pct": quarter_batches * 100 / complete,
                "batches_one_speaker_at_least_50pct_pct": half_batches * 100 / complete,
                "represented_speaker_sample_count_distribution": {
                    str(key): represented_counts[key] for key in sorted(represented_counts)
                },
            }

    unique_values = [row["unique_speakers"] for row in shard_stats]
    dominant_values = [row["dominant_speaker_share"] for row in shard_stats]
    utterance_values = [row["utterance_count"] for row in speaker_stats]
    analysis = {
        "seed": seed,
        "train_confirmation": {
            "rows": len(rows), "speakers": len(speakers), "label_min": min(labels),
            "label_max": max(labels), "labels_contiguous_0_487": labels == set(range(488)),
            "shards": len(shards),
        },
        "shard_distribution": {
            "unique_speakers": summary(unique_values),
            "dominant_speaker_share": summary(dominant_values),
            "shards_at_least_25pct_dominant": sum(value >= 0.25 for value in dominant_values),
            "shards_at_least_50pct_dominant": sum(value >= 0.50 for value in dominant_values),
            "shards_at_least_75pct_dominant": sum(value >= 0.75 for value in dominant_values),
            "single_speaker_shards": sum(value == 1 for value in unique_values),
            "all_feature_indexes_valid_and_unique": all(
                row["feature_indexes_valid_and_unique"] for row in shard_stats
            ),
        },
        "speaker_distribution": {
            "utterances_per_speaker": summary(utterance_values),
            "speakers_fewer_than_10": sum(value < 10 for value in utterance_values),
            "speakers_fewer_than_20": sum(value < 20 for value in utterance_values),
            "speakers_fewer_than_50": sum(value < 50 for value in utterance_values),
            "speakers_more_than_100": sum(value > 100 for value in utterance_values),
            "speakers_more_than_500": sum(value > 500 for value in utterance_values),
        },
        "index_order": {
            "consecutive_same_speaker_run_length": summary(run_lengths),
            "runs": len(run_lengths),
            "speaker_transitions": transitions,
            "neighbor_pairs": len(rows) - 1,
            "neighbor_same_speaker_pct": (len(rows) - 1 - transitions) * 100 / (len(rows) - 1),
        },
        "batch_diversity": diversity,
    }
    return analysis, shard_stats, speaker_stats


def validate_batch(batch: dict[str, Any], expected_count: int) -> None:
    if tuple(batch["fbank"].shape) != (expected_count, 301, 80):
        raise AssertionError(f"invalid batch feature shape: {tuple(batch['fbank'].shape)}")
    if batch["fbank"].dtype != torch.float32 or batch["fbank"].device.type != "cpu":
        raise AssertionError("invalid feature dtype or device")
    if tuple(batch["speaker_label"].shape) != (expected_count,) or batch["speaker_label"].dtype != torch.long:
        raise AssertionError("invalid label shape or dtype")
    for key in ("speaker_id", "relative_audio_path", "final_split", "filename_group"):
        if len(batch[key]) != expected_count:
            raise AssertionError(f"invalid {key} metadata length")
    if any(split != "train" for split in batch["final_split"]):
        raise AssertionError("non-train metadata returned")


def benchmark_one(
    cache_dir: Path, *, pattern: str, batch_size: int, measured_samples: int,
    warmup_batches: int, seed: int, max_cached_shards: int, num_workers: int,
    validate_finite: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "access_pattern": pattern, "batch_size": batch_size,
        "requested_measured_samples": measured_samples, "actual_measured_samples": 0,
        "warmup_batches": warmup_batches, "max_cached_shards": max_cached_shards,
        "num_workers": num_workers, "validate_finite": validate_finite,
        "first_batch_latency_seconds": None, "measured_elapsed_seconds": None,
        "batches_per_second": None, "samples_per_second": None,
        "all_shapes_and_metadata_valid": False, "completed_successfully": False,
        "error": None, "dataset_shard_load_count": None,
        "measured_shard_load_count": None,
        "shard_loads_per_measured_sample": None, "final_cached_shard_count": None,
        "memory_metric": None,
    }
    try:
        dataset = CachedFbankDataset(
            cache_dir, "train", max_cached_shards=max_cached_shards,
            validate_finite=validate_finite,
        )
        measured_batches = math.ceil(measured_samples / batch_size)
        actual_samples = measured_batches * batch_size
        needed = (1 + warmup_batches) * batch_size + actual_samples
        full_order = diagnostic_order(dataset.rows, pattern, seed)
        if needed > len(full_order):
            raise ValueError(f"requested run needs {needed} samples but only {len(full_order)} exist")
        sampler = full_order[:needed]
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
            collate_fn=collate_cached_fbank, drop_last=False,
        )
        start = time.perf_counter()
        iterator = iter(loader)
        first = next(iterator)
        result["first_batch_latency_seconds"] = time.perf_counter() - start
        validate_batch(first, batch_size)
        for _ in range(warmup_batches):
            validate_batch(next(iterator), batch_size)
        start_loads = dataset.shard_load_count
        start = time.perf_counter()
        seen = 0
        for _ in range(measured_batches):
            batch = next(iterator)
            validate_batch(batch, batch_size)
            seen += batch_size
        elapsed = time.perf_counter() - start
        result.update({
            "actual_measured_samples": seen,
            "measured_elapsed_seconds": elapsed,
            "batches_per_second": measured_batches / elapsed,
            "samples_per_second": seen / elapsed,
            "all_shapes_and_metadata_valid": True,
            "completed_successfully": True,
        })
        if num_workers == 0:
            measured_loads = dataset.shard_load_count - start_loads
            result.update({
                "dataset_shard_load_count": dataset.shard_load_count,
                "measured_shard_load_count": measured_loads,
                "shard_loads_per_measured_sample": measured_loads / seen,
                "final_cached_shard_count": dataset.cached_shard_count,
            })
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/fbank_cache_v1"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--measured-samples", type=int, default=512)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-cached-shards", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validate-finite", type=parse_bool, default=True)
    parser.add_argument("--access-pattern", choices=PATTERNS, default="sequential")
    parser.add_argument("--single", action="store_true", help="Run only the selected benchmark configuration.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.measured_samples < 1 or args.warmup_batches < 0:
        raise ValueError("batch size and measured samples must be positive; warm-up must be non-negative")
    cache_dir = args.cache_dir.resolve()
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_dataset = CachedFbankDataset(cache_dir, "train", validate_finite=False)
    analysis, shard_stats, speaker_stats = analyze_metadata(metadata_dataset, args.seed)
    write_csv(report_dir / "train_cache_shard_stats_v1.csv", shard_stats)
    write_csv(report_dir / "train_cache_speaker_stats_v1.csv", speaker_stats)

    if args.single:
        configurations = [{
            "pattern": args.access_pattern, "max_cached_shards": args.max_cached_shards,
            "num_workers": args.num_workers, "validate_finite": args.validate_finite,
        }]
    else:
        configurations = [
            {"pattern": "sequential", "max_cached_shards": 2, "num_workers": 0, "validate_finite": True},
            {"pattern": "sample-random", "max_cached_shards": 2, "num_workers": 0, "validate_finite": True},
            {"pattern": "sample-random", "max_cached_shards": 2, "num_workers": 0, "validate_finite": False},
            {"pattern": "sample-random", "max_cached_shards": 8, "num_workers": 0, "validate_finite": False},
            {"pattern": "sample-random", "max_cached_shards": 8, "num_workers": 1, "validate_finite": False},
            {"pattern": "sample-random", "max_cached_shards": 8, "num_workers": 2, "validate_finite": False},
            {"pattern": "shard-local", "max_cached_shards": 2, "num_workers": 0, "validate_finite": False},
        ]
    benchmark = [
        benchmark_one(
            cache_dir, pattern=config["pattern"], batch_size=args.batch_size,
            measured_samples=args.measured_samples, warmup_batches=args.warmup_batches,
            seed=args.seed, max_cached_shards=config["max_cached_shards"],
            num_workers=config["num_workers"], validate_finite=config["validate_finite"],
        )
        for config in configurations
    ]
    payload = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "operating_system": platform.system(),
        },
        "method": {
            "cache_dir": "outputs/fbank_cache_v1", "seed": args.seed,
            "short_diagnostic_not_full_epoch": True,
            "first_batch_precedes_warmup": True,
            "operating_system_file_cache_affects_results": True,
            "worker_shard_load_counts_available": False,
            "memory_reporting": "unavailable; no new dependency installed",
        },
        "analysis": analysis,
        "benchmarks": benchmark,
    }
    output = report_dir / "train_cache_benchmark_v1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all(row["completed_successfully"] for row in benchmark):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
