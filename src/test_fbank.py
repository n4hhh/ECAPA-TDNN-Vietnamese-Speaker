"""Standalone synthetic and real-data smoke tests for Fbank extraction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader

try:  # Support direct and module execution.
    from .fbank import FbankExtractor
    from .fbank_dataset import DEFAULT_MANIFEST_PATH, FbankSmokeTestDataset
except ImportError:  # pragma: no cover - exercised during direct execution
    from fbank import FbankExtractor
    from fbank_dataset import DEFAULT_MANIFEST_PATH, FbankSmokeTestDataset


SAMPLE_RATE = 16_000
NUM_SAMPLES = 48_000
NUM_MEL_BINS = 80


def _assert_raises(exception_type: type[Exception], operation: Callable[[], object]) -> None:
    try:
        operation()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__} to be raised")


def _assert_valid_features(features: Tensor, expected_time: int | None = None) -> None:
    assert features.ndim == 2
    assert features.shape[0] == NUM_MEL_BINS
    if expected_time is not None:
        assert features.shape[1] == expected_time
    assert features.dtype == torch.float32
    assert features.is_contiguous()
    assert bool(torch.isfinite(features).all().item())


def synthetic_waveform_test(extractor: FbankExtractor) -> int:
    """Check extraction and required input validation on synthetic PCM."""

    time_axis = torch.arange(NUM_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
    sine = 0.25 * torch.sin(2.0 * math.pi * 440.0 * time_axis)
    pcm16 = (sine * 32_767.0).round().to(torch.int16)
    features = extractor(pcm16, sample_rate=SAMPLE_RATE)
    _assert_valid_features(features)

    batched = extractor(
        torch.stack((pcm16, pcm16), dim=0),
        sample_rate=SAMPLE_RATE,
        input_is_batched=True,
    )
    assert tuple(batched.shape) == (2, NUM_MEL_BINS, features.shape[-1])
    assert batched.dtype == torch.float32
    assert batched.is_contiguous()
    assert bool(torch.isfinite(batched).all().item())

    explicit_channel_batch = extractor(
        torch.stack((pcm16, pcm16), dim=0).unsqueeze(1),
        sample_rate=SAMPLE_RATE,
    )
    assert tuple(explicit_channel_batch.shape) == (
        2,
        NUM_MEL_BINS,
        features.shape[-1],
    )
    assert explicit_channel_batch.dtype == torch.float32
    assert explicit_channel_batch.is_contiguous()
    assert bool(torch.isfinite(explicit_channel_batch).all().item())

    _assert_raises(ValueError, lambda: extractor(pcm16, sample_rate=8_000))
    _assert_raises(
        ValueError,
        lambda: extractor(torch.empty(0), sample_rate=SAMPLE_RATE),
    )
    invalid = torch.zeros(NUM_SAMPLES)
    invalid[0] = float("nan")
    _assert_raises(
        ValueError,
        lambda: extractor(invalid, sample_rate=SAMPLE_RATE),
    )
    _assert_raises(
        ValueError,
        lambda: extractor(
            torch.zeros(1, 2, NUM_SAMPLES),
            sample_rate=SAMPLE_RATE,
        ),
    )
    print(f"[PASS] synthetic waveform test: {tuple(features.shape)}")
    return int(features.shape[-1])


def one_real_wav_test(
    dataset: FbankSmokeTestDataset,
    extractor: FbankExtractor,
    expected_time: int,
) -> None:
    """Decode one real WAV directly and verify its expected source shape."""

    first_record = dataset.records[0]
    waveform, sample_rate = torchaudio.load(first_record.audio_path, normalize=True)
    assert tuple(waveform.shape) == (1, NUM_SAMPLES)
    assert sample_rate == SAMPLE_RATE
    assert waveform.dtype == torch.float32
    assert waveform.is_contiguous()
    assert bool(torch.isfinite(waveform).all().item())

    with torch.inference_mode():
        features = extractor(
            waveform,
            sample_rate=sample_rate,
            input_is_batched=False,
        )
    _assert_valid_features(features, expected_time)
    print(f"Waveform shape: {list(waveform.shape)}")
    print(f"One Fbank shape: {list(features.shape)}")
    print("[PASS] one real WAV test")


def ten_real_wav_tests(dataset: FbankSmokeTestDataset, expected_time: int) -> None:
    """Check ten lazy samples, common frame counts, and metadata alignment."""

    assert len(dataset) >= 10
    observed_times: set[int] = set()
    for index in range(10):
        item = dataset[index]
        features = item["features"]
        assert isinstance(features, Tensor)
        _assert_valid_features(features, expected_time)
        observed_times.add(int(features.shape[-1]))

        record = dataset.records[index]
        assert item["speaker_id"] == record.speaker_id
        assert item["audio_path"] == record.audio_path
        assert item["feature_length"] == features.shape[-1]

    assert observed_times == {expected_time}
    print(f"[PASS] ten real WAV tests: common T={expected_time}")


def dataloader_batch_test(dataset: FbankSmokeTestDataset, expected_time: int) -> None:
    """Check default collation and identifier ordering for a four-item batch."""

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    features = batch["features"]
    assert isinstance(features, Tensor)
    assert tuple(features.shape) == (4, NUM_MEL_BINS, expected_time)
    assert features.dtype == torch.float32
    assert features.is_contiguous()
    assert bool(torch.isfinite(features).all().item())

    feature_lengths = batch["feature_length"]
    assert isinstance(feature_lengths, Tensor)
    assert feature_lengths.tolist() == [expected_time] * 4

    expected_speakers = [record.speaker_id for record in dataset.records[:4]]
    expected_paths = [record.audio_path for record in dataset.records[:4]]
    assert list(batch["speaker_id"]) == expected_speakers
    assert list(batch["audio_path"]) == expected_paths

    print(f"Batch Fbank shape: {list(features.shape)}")
    print("[PASS] DataLoader batch test (batch_size=4)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Existing manifest CSV (default: {DEFAULT_MANIFEST_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """Run all requested bounded Fbank tests."""

    args = parse_args()
    extractor = FbankExtractor()
    expected_time = synthetic_waveform_test(extractor)
    dataset = FbankSmokeTestDataset(
        manifest_path=args.manifest,
        max_samples=10,
        extractor=extractor,
    )
    one_real_wav_test(dataset, extractor, expected_time)
    ten_real_wav_tests(dataset, expected_time)
    dataloader_batch_test(dataset, expected_time)
    print(f"All Fbank tests passed. Actual time dimension T={expected_time}.")


if __name__ == "__main__":
    main()
