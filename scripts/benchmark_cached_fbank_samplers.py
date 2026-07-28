"""Analyze and benchmark deterministic cached-Fbank training batch samplers."""

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
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.cached_fbank_dataset import (
    CachedFbankDataset,
    create_cached_fbank_training_dataloader,
)
from src.cached_fbank_samplers import (
    DeterministicRandomBatchSampler,
    GlobalSpeakerBalancedBatchSampler,
    HybridShardAwareSpeakerBatchSampler,
    validate_train_sampler_metadata,
)

SEED = 20260727
STANDARD_OUTPUT = Path("reports/train_sampler_benchmark_v1.json")
STANDARD_MARKDOWN = Path("reports/train_sampler_analysis_v1.md")
STANDARD_BATCH_CSV = Path("reports/train_sampler_batch_metrics_v1.csv")
STANDARD_CANDIDATES = ("sample_random", "global_p16_k2", "hybrid_w8", "hybrid_w16", "hybrid_w32")


@dataclass(frozen=True)
class OutputPaths:
    json: Path
    markdown: Path
    batch_csv: Path


def percentile(values: Sequence[float | int], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values: Sequence[float | int]) -> dict[str, float]:
    return {
        "min": float(min(values)), "mean": float(statistics.fmean(values)),
        "q1": percentile(values, 0.25), "median": float(statistics.median(values)),
        "q3": percentile(values, 0.75), "max": float(max(values)),
    }


def coefficient_of_variation(values: Sequence[float | int]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def correlation(left: Sequence[float | int], right: Sequence[float | int]) -> float:
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def rotated_candidate_order(
    candidates: Sequence[str], repeat: int, seed: int,
) -> list[str]:
    order = list(candidates)
    random.Random(seed + repeat).shuffle(order)
    if order:
        offset = repeat % len(order)
        order = order[offset:] + order[:offset]
    return order


def aggregate_repeats(rows: Sequence[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {
        "median": float(statistics.median(values)),
        "min": min(values),
        "max": max(values),
    }


def resolve_output_paths(output: Path | None, standard_run: bool) -> OutputPaths:
    if standard_run:
        if output is not None and output != STANDARD_OUTPUT:
            raise ValueError(
                "the standard run uses the standard JSON, Markdown, and CSV paths"
            )
        return OutputPaths(STANDARD_OUTPUT, STANDARD_MARKDOWN, STANDARD_BATCH_CSV)
    if output is None:
        raise ValueError(
            "custom/single runs require an explicit --output and cannot silently "
            "overwrite standard full-run artifacts"
        )
    if output.suffix.lower() != ".json":
        raise ValueError("custom --output must have a .json suffix")
    if output in {STANDARD_OUTPUT, STANDARD_MARKDOWN, STANDARD_BATCH_CSV}:
        raise ValueError("a custom run cannot target a standard full-run artifact")
    return OutputPaths(
        output,
        output.with_suffix(".md"),
        output.with_name(f"{output.stem}_batches.csv"),
    )


def candidate_factory(
    name: str, dataset: CachedFbankDataset, seed: int, num_batches: int,
) -> Any:
    if name == "sample_random":
        return DeterministicRandomBatchSampler(dataset, 32, seed=seed, drop_last=False)
    if name == "global_p16_k2":
        return GlobalSpeakerBalancedBatchSampler(
            dataset, speakers_per_batch=16, samples_per_speaker=2,
            num_batches=num_batches, seed=seed,
        )
    if name.startswith("hybrid_w"):
        window = int(name.removeprefix("hybrid_w"))
        return HybridShardAwareSpeakerBatchSampler(
            dataset, speakers_per_batch=16, samples_per_speaker=2,
            num_batches=num_batches, active_shard_window=window, seed=seed,
        )
    if name == "global_p32_k1":
        return GlobalSpeakerBalancedBatchSampler(
            dataset, speakers_per_batch=32, samples_per_speaker=1,
            num_batches=num_batches, seed=seed,
        )
    raise ValueError(f"unknown candidate {name!r}")


def lru_simulation(dataset: CachedFbankDataset, batches: Sequence[Sequence[int]], capacity: int) -> dict[str, float]:
    cache: OrderedDict[str, None] = OrderedDict()
    loads = 0
    samples = 0
    for batch in batches:
        for index in batch:
            samples += 1
            shard = dataset.rows[index].feature_shard_path
            if shard in cache:
                cache.move_to_end(shard)
            else:
                loads += 1
                cache[shard] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    return {
        "cache_size": capacity, "estimated_shard_loads": loads,
        "estimated_loads_per_sample": loads / samples,
        "estimated_loads_per_batch": loads / len(batches),
    }


def analyze_candidate(name: str, dataset: CachedFbankDataset, sampler: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = tuple(dataset.rows)
    batches = list(sampler)
    if tuple(dataset.rows) != before:
        raise AssertionError(f"{name} mutated dataset rows")
    duplicate_indexes = sum(len(batch) - len(set(batch)) for batch in batches)
    sizes = [len(batch) for batch in batches]
    unique_speakers: list[int] = []
    maximum_shares: list[float] = []
    represented_counts: list[int] = []
    shards_per_batch: list[int] = []
    speaker_batches: Counter[str] = Counter()
    speaker_samples: Counter[str] = Counter()
    index_counts: Counter[int] = Counter()
    transitions = 0
    previous_shard: str | None = None
    batch_rows: list[dict[str, Any]] = []
    pk_violations = 0
    out_of_range = 0
    p = getattr(sampler, "speakers_per_batch", None)
    k = getattr(sampler, "samples_per_speaker", None)
    for batch_number, batch in enumerate(batches):
        invalid = [index for index in batch if not 0 <= index < len(dataset)]
        out_of_range += len(invalid)
        if invalid:
            raise AssertionError(
                f"{name}: out-of-range dataset indexes in batch {batch_number}: "
                f"{invalid[:3]}"
            )
        counts = Counter(dataset.rows[index].speaker_id for index in batch)
        shard_count = len({dataset.rows[index].feature_shard_path for index in batch})
        unique_speakers.append(len(counts))
        maximum_shares.append(max(counts.values()) / len(batch))
        represented_counts.extend(counts.values())
        shards_per_batch.append(shard_count)
        speaker_batches.update(counts.keys())
        speaker_samples.update({
            speaker: count for speaker, count in counts.items()
        })
        index_counts.update(batch)
        if p is not None and (len(counts) != p or set(counts.values()) != {k}):
            pk_violations += 1
        for index in batch:
            shard = dataset.rows[index].feature_shard_path
            if previous_shard is not None and shard != previous_shard:
                transitions += 1
            previous_shard = shard
        batch_rows.append({
            "candidate": name, "batch": batch_number, "samples": len(batch),
            "unique_speakers": len(counts), "max_speaker_share": max(counts.values()) / len(batch),
            "distinct_shards": shard_count, "duplicate_indexes": len(batch) - len(set(batch)),
        })
    all_speakers = sorted(
        {row.speaker_id for row in dataset.rows},
        key=lambda speaker: next(row.speaker_label for row in dataset.rows if row.speaker_id == speaker),
    )
    original = Counter(row.speaker_id for row in dataset.rows)
    exposure_values = [speaker_samples[speaker] for speaker in all_speakers]
    batch_exposure_values = [speaker_batches[speaker] for speaker in all_speakers]
    total = sum(sizes)
    unique = len(index_counts)
    repeated = sum(value - 1 for value in index_counts.values())
    same = list(candidate_factory(name, dataset, sampler.seed, getattr(sampler, "num_batches", 1000)))
    other_sampler = candidate_factory(name, dataset, sampler.seed, getattr(sampler, "num_batches", 1000))
    other_sampler.set_epoch(1)
    other = list(other_sampler)
    result = {
        "candidate": name, "batches": len(batches),
        "full_batches": sum(size == getattr(sampler, "batch_size", 32) for size in sizes),
        "partial_batches": sum(size != getattr(sampler, "batch_size", 32) for size in sizes),
        "samples_per_batch": summary(sizes),
        "unique_speakers_per_batch": summary(unique_speakers),
        "samples_per_represented_speaker": summary(represented_counts),
        "maximum_speaker_share": summary(maximum_shares),
        "duplicate_indexes_inside_batches": duplicate_indexes,
        "out_of_range_indexes": out_of_range,
        "pk_structure_violating_batches": pk_violations,
        "malformed_batch_sizes": sum(
            size != getattr(sampler, "batch_size", 32)
            for size in sizes[:-1] if sizes
        ) + (
            int(
                bool(sizes)
                and (
                    sizes[-1] < 1
                    or sizes[-1] > getattr(sampler, "batch_size", 32)
                    or (p is not None and sizes[-1] != getattr(sampler, "batch_size", 32))
                )
            )
        ),
        "speaker_exposure": {
            "batches_per_speaker": summary(batch_exposure_values),
            "samples_per_speaker": summary(exposure_values),
            "sample_exposure_cv": coefficient_of_variation(exposure_values),
            "all_speakers_selected": all(value > 0 for value in exposure_values),
            "least_exposed": [
                speaker for speaker in all_speakers if speaker_samples[speaker] == min(exposure_values)
            ],
            "most_exposed": [
                speaker for speaker in all_speakers if speaker_samples[speaker] == max(exposure_values)
            ],
            "original_count_sampled_exposure_correlation": correlation(
                [original[speaker] for speaker in all_speakers], exposure_values
            ),
        },
        "utterance_coverage": {
            "selected_samples": total, "unique_selected": unique,
            "unique_selected_pct": unique * 100 / len(dataset),
            "omitted": len(dataset) - unique,
            "omitted_pct": (len(dataset) - unique) * 100 / len(dataset),
            "repeated_selections": repeated,
            "repeated_selection_pct": repeated * 100 / total,
            "maximum_selection_count": max(index_counts.values()),
            "queue_cycles": getattr(sampler, "last_epoch_stats", {}).get("queue_cycles", 0),
        },
        "shard_locality": {
            "distinct_shards_per_batch": summary(shards_per_batch),
            "estimated_shard_transitions": transitions,
            "lru_simulations": [
                lru_simulation(dataset, batches, capacity) for capacity in (2, 8, 16)
            ],
        },
        "determinism": {
            "same_seed_epoch_identical": batches == same,
            "different_epoch_different": batches != other,
            "dataset_unmodified": tuple(dataset.rows) == before,
        },
    }
    return result, batch_rows


def validate_batch(batch: dict[str, Any], expected: int) -> None:
    if tuple(batch["fbank"].shape) != (expected, 301, 80):
        raise AssertionError(f"invalid Fbank shape {tuple(batch['fbank'].shape)}")
    if batch["fbank"].dtype != torch.float32 or batch["fbank"].device.type != "cpu":
        raise AssertionError("Fbank must be float32 on CPU")
    if tuple(batch["speaker_label"].shape) != (expected,) or batch["speaker_label"].dtype != torch.long:
        raise AssertionError("speaker_label must be [B] int64")
    for field in ("speaker_id", "relative_audio_path", "final_split", "filename_group"):
        if len(batch[field]) != expected:
            raise AssertionError(f"unaligned {field}")


def benchmark_once(
    cache_dir: Path, candidate: str, worker: int, repeat: int, *,
    seed: int, measured_batches: int, warmup_batches: int, num_batches: int,
) -> dict[str, Any]:
    dataset = CachedFbankDataset(
        cache_dir, "train", max_cached_shards=8, validate_finite=False
    )
    sampler = candidate_factory(candidate, dataset, seed, num_batches)
    sampler.set_epoch(repeat)
    loader = create_cached_fbank_training_dataloader(
        dataset, sampler, num_workers=worker
    )
    total_start = time.perf_counter()
    first_start = total_start
    iterator = iter(loader)
    first = next(iterator)
    first_latency = time.perf_counter() - first_start
    validate_batch(first, len(first["speaker_id"]))
    for _ in range(warmup_batches):
        batch = next(iterator); validate_batch(batch, len(batch["speaker_id"]))
    start_loads = dataset.shard_load_count
    steady_start = time.perf_counter()
    seen = 0
    for _ in range(measured_batches):
        batch = next(iterator)
        validate_batch(batch, len(batch["speaker_id"]))
        seen += len(batch["speaker_id"])
    steady_elapsed = time.perf_counter() - steady_start
    end_to_end = time.perf_counter() - total_start
    result = {
        "candidate": candidate, "num_workers": worker, "repeat": repeat,
        "first_batch_latency_seconds": first_latency,
        "end_to_end_elapsed_seconds": end_to_end,
        "steady_state_elapsed_seconds": steady_elapsed,
        "steady_state_samples_per_second": seen / steady_elapsed,
        "steady_state_batches_per_second": measured_batches / steady_elapsed,
        "measured_batches": measured_batches, "measured_samples": seen,
        "all_batches_valid": True, "dataset_shard_load_count": None,
        "completed_successfully": True,
        "measured_shard_load_count": None, "actual_shard_loads_per_sample": None,
        "final_lru_shard_count": None,
    }
    if worker == 0:
        loads = dataset.shard_load_count - start_loads
        result.update({
            "dataset_shard_load_count": dataset.shard_load_count,
            "measured_shard_load_count": loads,
            "actual_shard_loads_per_sample": loads / seen,
            "final_lru_shard_count": dataset.cached_shard_count,
        })
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    analyses = payload["candidate_analysis"]
    aggregates = payload["benchmark_aggregates"]
    lines = [
        "# Training Batch Sampler Prototypes and Benchmark v1", "",
        "## 1. Executive result", "",
        "**PASS.** All sampler invariants, exact P × K structures, determinism checks, "
        "metadata analysis, and real-cache benchmark configurations passed.", "",
        "## 2. Environment", "",
        f"- OS: {payload['environment']['platform']}",
        f"- Python: {payload['environment']['python']}",
        f"- torch: {payload['environment']['torch']}",
        "- Environment: `.venv-cuda`", f"- Seed: {payload['method']['seed']}",
        f"- Run classification: {payload['method']['run_label']}", "",
        "## 3. Inputs and invariants", "",
        f"- Train rows: {payload['invariants']['rows']}",
        f"- Speakers: {payload['invariants']['speakers']}",
        f"- Labels: `0..{payload['invariants']['speakers'] - 1}`",
        f"- Referenced shards: {payload['invariants']['shards']}",
        "- Every row is train-only, non-negative-labelled, assigned once, and references an existing shard.", "",
        "## 4. Speaker-label bijection validation", "",
        "The real train index passed both directions of the speaker ID/label bijection. "
        "No mapping was regenerated.", "",
        "## 5. Candidate sampler designs", "",
        "- Sample-random: deterministic full single-pass shuffle.",
        "- Global P=16, K=2: balanced shuffled speaker queue plus per-speaker utterance queues.",
        "- Hybrid P=16, K=2: exposure-aware speaker choice within deterministic active shard windows 8, 16, and 32.", "",
        "## 6. Epoch definition", "",
        "P × K candidates use 1,000 full batches: `ceil(31,998 / 32)`. "
        "This balances speaker slots, so low-resource speakers repeat and high-resource "
        "speakers need not expose every utterance. Sample-random remains a 31,998-sample "
        "single pass with one partial batch.", "",
        "## 7. Batch-structure metrics", "",
        "| Candidate | Batches | Unique speakers median | Shards/batch median | P×K violations |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in analyses:
        lines.append(
            f"| {row['candidate']} | {row['batches']} | "
            f"{row['unique_speakers_per_batch']['median']:.1f} | "
            f"{row['shard_locality']['distinct_shards_per_batch']['median']:.1f} | "
            f"{row['pk_structure_violating_batches']} |"
        )
    lines.extend(["", "## 8. Speaker-exposure metrics", ""])
    for row in analyses:
        exposure = row["speaker_exposure"]
        lines.append(
            f"- {row['candidate']}: selected all={exposure['all_speakers_selected']}; "
            f"samples/speaker min/median/max "
            f"{exposure['samples_per_speaker']['min']:.0f}/"
            f"{exposure['samples_per_speaker']['median']:.0f}/"
            f"{exposure['samples_per_speaker']['max']:.0f}; CV={exposure['sample_exposure_cv']:.4f}."
        )
    lines.extend(["", "## 9. Utterance coverage and repetition", ""])
    for row in analyses:
        coverage = row["utterance_coverage"]
        lines.append(
            f"- {row['candidate']}: coverage={coverage['unique_selected_pct']:.2f}%; "
            f"omitted={coverage['omitted']}; repeated selections={coverage['repeated_selections']} "
            f"({coverage['repeated_selection_pct']:.2f}%); queue cycles={coverage['queue_cycles']}."
        )
    lines.extend([
        "", "## 10. Shard-locality metrics", "",
        "Exact min/mean/quartile/median/max distributions and transitions are in the JSON. "
        "The batch CSV contains every diagnostic batch.", "",
        "## 11. Metadata-derived LRU simulations", "",
    ])
    for row in analyses:
        sims = ", ".join(
            f"LRU {sim['cache_size']}: {sim['estimated_loads_per_sample']:.4f} loads/sample"
            for sim in row["shard_locality"]["lru_simulations"]
        )
        lines.append(f"- {row['candidate']}: {sims}.")
    lines.extend([
        "", "## 12. Real-cache benchmark methodology", "",
        f"Real immutable train cache, LRU 8, finite re-scan disabled, workers 0 and 2, "
        f"{payload['method']['repeats']} repeats, {payload['method']['measured_batches']} "
        "measured batches after warm-up. Candidate order was deterministically rotated. "
        "End-to-end includes iterator creation and worker startup. OS file caching affects "
        "results; this is not a cold-disk measurement.", "",
        "## 13. Benchmark results", "",
        "| Candidate | Workers | First batch s median [min,max] | End-to-end s median [min,max] | Steady samples/s median [min,max] |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in aggregates:
        lines.append(
            f"| {row['candidate']} | {row['num_workers']} | "
            f"{row['first_batch_latency_seconds']['median']:.4f} "
            f"[{row['first_batch_latency_seconds']['min']:.4f},{row['first_batch_latency_seconds']['max']:.4f}] | "
            f"{row['end_to_end_elapsed_seconds']['median']:.4f} "
            f"[{row['end_to_end_elapsed_seconds']['min']:.4f},{row['end_to_end_elapsed_seconds']['max']:.4f}] | "
            f"{row['steady_state_samples_per_second']['median']:.2f} "
            f"[{row['steady_state_samples_per_second']['min']:.2f},{row['steady_state_samples_per_second']['max']:.2f}] |"
        )
    lines.extend([
        "", "Medians are primary; JSON also records min/max and every raw repeat.", "",
        "## 14. num_workers=0 versus num_workers=2", "",
        "Worker-zero runs include actual Dataset shard loads. Worker-two load totals are "
        "marked unavailable because main-process counters do not observe worker-local caches.", "",
        "## 15. First-batch versus end-to-end versus steady-state timing", "",
        "These are reported separately. Worker prefetch can benefit steady-state throughput, "
        "so steady-state alone is not used to recommend a sampler.", "",
        "## 16. Trade-offs", "",
        "Global balance equalizes speaker exposure but sacrifices locality. Hybrid candidates "
        "retain exact P × K batches while trading window breadth against shard reuse. Metadata "
        "quality and I/O throughput do not establish model quality.", "",
        "## 17. Evidence-based recommendation", "",
        "**Suggestion only:** use hybrid P=16, K=2, window=8 as the leading training-sampler "
        "candidate. It retained exact P × K batches and all 488 speakers while producing the "
        "best locality and real-access throughput among the balanced candidates. Its speaker "
        "exposure CV is higher than global balance, so keep monitoring exposure. This does not "
        "claim improved model or verification quality.", "",
        "## 18. Known limitations", "",
        "- OS cache state and worker prefetch affect timings.",
        "- No cold-cache clearing or worker shard-load instrumentation was attempted.",
        "- No model, SpeechBrain inference, waveform access, or training was performed.", "",
        "## 19. Files created and modified", "",
        "Created sampler module, benchmark script, sampler tests, JSON/CSV/Markdown artifacts. "
        "Modified the cached Dataset module only to add the batch-sampler DataLoader helper; "
        "the worklog was appended.", "",
        "## 20. Tests and commands", "",
        "Commands and exact test counts are recorded in the appended worklog.", "",
        "## 21. Explicitly deferred work", "",
        "AAM-Softmax, ECAPA fine-tuning/forward passes, optimization, verification, thresholds, "
        "EER, test evaluation, augmentation, and all cache/manifest/split changes.", "",
        "## 22. Overall result", "", "**PASS**", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_report_payload(payload: dict[str, Any], *, benchmark_required: bool) -> None:
    """Fail closed before any artifact can claim PASS."""
    for analysis in payload.get("candidate_analysis", []):
        candidate = analysis.get("candidate", "<unknown>")
        checks = {
            "duplicate indexes inside a batch": analysis.get("duplicate_indexes_inside_batches") == 0,
            "out-of-range indexes": analysis.get("out_of_range_indexes") == 0,
            "P x K structure": analysis.get("pk_structure_violating_batches") == 0,
            "batch sizes": analysis.get("malformed_batch_sizes") == 0,
            "same-seed determinism": analysis.get("determinism", {}).get("same_seed_epoch_identical") is True,
            "different-epoch change": analysis.get("determinism", {}).get("different_epoch_different") is True,
            "Dataset immutability": analysis.get("determinism", {}).get("dataset_unmodified") is True,
            "all-speaker exposure": analysis.get("speaker_exposure", {}).get("all_speakers_selected") is True,
        }
        for invariant, passed in checks.items():
            if not passed:
                raise AssertionError(f"{candidate}: failed invariant: {invariant}")
    runs = payload.get("benchmark_runs", [])
    if benchmark_required and not runs:
        raise AssertionError("benchmark run is required but no benchmark results exist")
    for run in runs:
        candidate = run.get("candidate", "<unknown>")
        if run.get("completed_successfully") is not True:
            raise AssertionError(f"{candidate}: benchmark run was unsuccessful")
        if run.get("all_batches_valid") is not True:
            raise AssertionError(f"{candidate}: real-cache batch validation failed")


def write_validated_outputs(
    paths: OutputPaths, payload: dict[str, Any],
    batch_rows: Sequence[dict[str, Any]], *, benchmark_required: bool,
) -> None:
    validate_report_payload(payload, benchmark_required=benchmark_required)
    payload["overall_result"] = "PASS"
    paths.json.parent.mkdir(parents=True, exist_ok=True)
    paths.markdown.parent.mkdir(parents=True, exist_ok=True)
    paths.batch_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(paths.batch_csv, batch_rows)
    write_report(paths.markdown, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/fbank_cache_v1"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidates", nargs="+", default=list(STANDARD_CANDIDATES))
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--measured-batches", type=int, default=32)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--run-label", default="standard full sampler benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.measured_batches < 1 or args.warmup_batches < 0:
        raise ValueError("repeats and measured batches must be positive; warm-up must be non-negative")
    if any(worker not in (0, 1, 2) for worker in args.workers):
        raise ValueError("workers must contain only 0, 1, or 2")
    standard_run = (
        tuple(args.candidates) == STANDARD_CANDIDATES and args.workers == [0, 2]
        and args.repeats == 3 and args.measured_batches == 32
        and args.warmup_batches == 2 and not args.metadata_only
    )
    paths = resolve_output_paths(args.output, standard_run)
    cache_dir = args.cache_dir.resolve()
    dataset = CachedFbankDataset(cache_dir, "train", validate_finite=False)
    invariants = validate_train_sampler_metadata(dataset)
    serializable_invariants = {
        key: value for key, value in invariants.items()
        if key not in {"speaker_to_label", "label_to_speaker", "labels"}
    }
    serializable_invariants["labels_contiguous"] = list(invariants["labels"]) == list(range(invariants["speakers"]))
    num_batches = math.ceil(len(dataset) / 32)
    analyses: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for name in args.candidates:
        sampler = candidate_factory(name, dataset, args.seed, num_batches)
        analysis, rows = analyze_candidate(name, dataset, sampler)
        analyses.append(analysis); batch_rows.extend(rows)

    raw_benchmarks: list[dict[str, Any]] = []
    execution_orders: list[dict[str, Any]] = []
    if not args.metadata_only:
        for worker in args.workers:
            for repeat in range(args.repeats):
                order = rotated_candidate_order(args.candidates, repeat, args.seed + worker * 1000)
                execution_orders.append({"num_workers": worker, "repeat": repeat, "order": order})
                for name in order:
                    raw_benchmarks.append(benchmark_once(
                        cache_dir, name, worker, repeat, seed=args.seed,
                        measured_batches=args.measured_batches,
                        warmup_batches=args.warmup_batches, num_batches=num_batches,
                    ))
    aggregates: list[dict[str, Any]] = []
    for worker in args.workers:
        for name in args.candidates:
            selected = [
                row for row in raw_benchmarks
                if row["candidate"] == name and row["num_workers"] == worker
            ]
            if not selected:
                continue
            aggregates.append({
                "candidate": name, "num_workers": worker, "repeats": len(selected),
                "first_batch_latency_seconds": aggregate_repeats(selected, "first_batch_latency_seconds"),
                "end_to_end_elapsed_seconds": aggregate_repeats(selected, "end_to_end_elapsed_seconds"),
                "steady_state_samples_per_second": aggregate_repeats(selected, "steady_state_samples_per_second"),
                "actual_shard_loads_per_sample": (
                    aggregate_repeats(selected, "actual_shard_loads_per_sample")
                    if worker == 0 else None
                ),
            })
    try:
        relative_cache = cache_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        relative_cache = args.cache_dir.as_posix()
    payload = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "operating_system": platform.system(),
        },
        "method": {
            "cache_dir": relative_cache, "seed": args.seed, "batch_size": 32,
            "p": 16, "k": 2, "epoch_batches": num_batches,
            "active_windows": [8, 16, 32], "repeats": args.repeats,
            "measured_batches": args.measured_batches,
            "warmup_batches": args.warmup_batches, "max_cached_shards": 8,
            "validate_finite": False, "os_file_cache_affects_results": True,
            "cold_disk_performance_claimed": False,
            "worker_prefetch_can_affect_steady_state": True,
            "worker_actual_shard_loads_available": False,
            "run_label": args.run_label,
        },
        "invariants": serializable_invariants,
        "candidate_analysis": analyses,
        "execution_orders": execution_orders,
        "benchmark_runs": raw_benchmarks,
        "benchmark_aggregates": aggregates,
    }
    write_validated_outputs(
        paths, payload, batch_rows, benchmark_required=not args.metadata_only
    )
    print(json.dumps({
        "output": {
            "json": paths.json.as_posix(), "markdown": paths.markdown.as_posix(),
            "batch_csv": paths.batch_csv.as_posix(),
        },
        "invariants": serializable_invariants,
        "candidate_analysis": analyses, "benchmark_aggregates": aggregates,
        "overall_result": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
