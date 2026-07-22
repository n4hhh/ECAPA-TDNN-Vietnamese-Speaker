"""Bounded end-to-end Fbank benchmark over ``train_small`` only."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

try:  # Support direct and module execution.
    from .fbank import FbankExtractor
    from .fbank_dataset import DEFAULT_MANIFEST_PATH, FbankSmokeTestDataset
except ImportError:  # pragma: no cover - exercised during direct execution
    from fbank import FbankExtractor
    from fbank_dataset import DEFAULT_MANIFEST_PATH, FbankSmokeTestDataset


MAX_BENCHMARK_SAMPLES = 1_000


def count_manifest_rows(manifest_path: Path) -> int:
    """Stream the manifest to count rows without retaining them in memory."""

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("manifest has no header")
        return sum(1 for _ in reader)


def format_duration(seconds: float) -> str:
    """Format seconds with a useful equivalent for longer estimates."""

    if seconds < 60.0:
        return f"{seconds:.2f} seconds"
    if seconds < 3_600.0:
        return f"{seconds:.2f} seconds ({seconds / 60.0:.2f} minutes)"
    return f"{seconds:.2f} seconds ({seconds / 3_600.0:.2f} hours)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Existing manifest CSV (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_BENCHMARK_SAMPLES,
        help=f"Files to process, capped at {MAX_BENCHMARK_SAMPLES}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="DataLoader batch size (default: 4)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count; zero is the Windows-safe default",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many processed files",
    )
    args = parser.parse_args()

    if not 1 <= args.max_samples <= MAX_BENCHMARK_SAMPLES:
        parser.error(f"--max-samples must be between 1 and {MAX_BENCHMARK_SAMPLES}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def main() -> None:
    """Benchmark lazy decode plus extraction and discard every feature batch."""

    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    total_manifest_files = count_manifest_rows(manifest_path)

    extractor = FbankExtractor()
    dataset = FbankSmokeTestDataset(
        manifest_path=manifest_path,
        max_samples=args.max_samples,
        extractor=extractor,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    processed = 0
    next_progress = args.progress_every
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"]
            if not isinstance(features, Tensor) or features.ndim != 3:
                raise RuntimeError("benchmark received an invalid feature batch")
            processed += int(features.shape[0])
            if processed >= next_progress or processed == len(dataset):
                elapsed = time.perf_counter() - start_time
                print(
                    f"Progress: {processed}/{len(dataset)} files "
                    f"({processed / elapsed:.2f} files/s)"
                )
                while next_progress <= processed:
                    next_progress += args.progress_every
            del features, batch
    elapsed_seconds = time.perf_counter() - start_time

    if processed == 0 or elapsed_seconds <= 0.0:
        raise RuntimeError("benchmark did not process any files")
    files_per_second = processed / elapsed_seconds

    durations: list[float] = []
    for record in dataset.records[:processed]:
        if record.duration_seconds is None:
            raise ValueError(
                "duration_seconds is required in selected manifest rows for the benchmark"
            )
        durations.append(record.duration_seconds)
    audio_seconds = math.fsum(durations)
    audio_realtime_factor = audio_seconds / elapsed_seconds
    estimated_all_seconds = total_manifest_files / files_per_second

    print("Fbank benchmark complete (WAV decode + Fbank + DataLoader collation).")
    print("Processed group: train_small")
    print(f"Files processed: {processed}")
    print(f"Total processing time: {elapsed_seconds:.3f} seconds")
    print(f"Files per second: {files_per_second:.3f}")
    print(f"Audio seconds processed: {audio_seconds:.3f}")
    print(f"Audio seconds per second: {audio_realtime_factor:.3f}")
    print(f"Full manifest files used for estimate: {total_manifest_files}")
    print(f"Estimated time for all files: {format_duration(estimated_all_seconds)}")
    print(
        "Estimate is a linear projection from this bounded train_small sample; "
        "manifest scanning and dataset setup are outside the timed region."
    )
    print("No Fbank features were saved.")


if __name__ == "__main__":
    main()
