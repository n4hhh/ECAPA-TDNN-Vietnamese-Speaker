"""Four-file smoke test for the frozen SpeechBrain ECAPA checkpoint."""

from __future__ import annotations

import argparse
import csv
import platform
from dataclasses import dataclass
from pathlib import Path

import speechbrain
import torch
import torchaudio
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

try:  # Support direct and module execution.
    from .fbank import FbankExtractor
    from .fbank_dataset import DEFAULT_MANIFEST_PATH
    from .speechbrain_frontend import (
        DEFAULT_CACHE_DIR,
        SpeechBrainECAPAFrontend,
    )
except ImportError:  # pragma: no cover - exercised during direct execution
    from fbank import FbankExtractor
    from fbank_dataset import DEFAULT_MANIFEST_PATH
    from speechbrain_frontend import DEFAULT_CACHE_DIR, SpeechBrainECAPAFrontend


SAMPLE_RATE = 16_000
EXPECTED_NUM_SAMPLES = 48_000
SMOKE_TEST_GROUP = "train_small"
NUM_TEST_FILES = 4


@dataclass(frozen=True)
class AudioRecord:
    """Minimal manifest metadata for one smoke-test waveform."""

    audio_path: str
    speaker_id: str


@dataclass(frozen=True)
class PipelineOutputs:
    """Outputs retained from one direct/encode pipeline comparison."""

    raw_features: Tensor
    normalized_features: Tensor
    direct_embedding: Tensor
    encoded_embedding: Tensor


def select_train_small_records(manifest_path: Path) -> tuple[AudioRecord, ...]:
    """Stream exactly four ``train_small`` rows from the existing manifest."""

    records: list[AudioRecord] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {"audio_path", "speaker_id", "filename_group"}
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"manifest is missing required columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            if row["filename_group"].strip() != SMOKE_TEST_GROUP:
                continue
            audio_path = row["audio_path"].strip()
            speaker_id = row["speaker_id"].strip()
            if not audio_path or not speaker_id:
                raise ValueError(
                    f"manifest row {row_number} has an empty audio_path or speaker_id"
                )
            records.append(AudioRecord(audio_path=audio_path, speaker_id=speaker_id))
            if len(records) == NUM_TEST_FILES:
                break

    if len(records) != NUM_TEST_FILES:
        raise ValueError(
            f"needed {NUM_TEST_FILES} {SMOKE_TEST_GROUP!r} rows, found {len(records)}"
        )
    return tuple(records)


def load_waveform_batch(records: tuple[AudioRecord, ...]) -> tuple[Tensor, Tensor]:
    """Decode four mono 16 kHz WAVs and return padded audio plus relative lengths."""

    waveforms: list[Tensor] = []
    sample_counts: list[int] = []
    for record in records:
        audio_path = Path(record.audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file does not exist: {audio_path}")
        waveform, sample_rate = torchaudio.load(str(audio_path), normalize=True)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"expected {SAMPLE_RATE} Hz for {audio_path}, got {sample_rate}"
            )
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError(
                f"expected mono [1, N] waveform for {audio_path}, "
                f"got {tuple(waveform.shape)}"
            )
        if waveform.shape[1] == 0:
            raise ValueError(f"decoded empty waveform: {audio_path}")
        if waveform.dtype != torch.float32:
            raise TypeError(f"expected float32 waveform for {audio_path}")
        if not bool(torch.isfinite(waveform).all().item()):
            raise ValueError(f"waveform contains NaN or Inf: {audio_path}")
        waveforms.append(waveform.squeeze(0))
        sample_counts.append(int(waveform.shape[1]))

    padded = pad_sequence(waveforms, batch_first=True)
    maximum_samples = int(padded.shape[1])
    relative_lengths = torch.tensor(sample_counts, dtype=torch.float32) / maximum_samples
    return padded.contiguous(), relative_lengths


def print_tensor_status(label: str, tensor: Tensor) -> None:
    """Print shape, dtype, device, and explicit NaN/Inf status."""

    has_nan = bool(torch.isnan(tensor).any().item())
    has_inf = bool(torch.isinf(tensor).any().item())
    print(
        f"{label}: shape={list(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, has_nan={has_nan}, has_inf={has_inf}"
    )


def assert_finite(tensor: Tensor, label: str) -> None:
    """Assert that a floating tensor has neither NaN nor Inf values."""

    if not bool(torch.isfinite(tensor).all().item()):
        raise AssertionError(f"{label} contains NaN or Inf")


def run_pipeline_test(
    label: str,
    frontend: SpeechBrainECAPAFrontend,
    waveforms: Tensor,
    relative_lengths: Tensor,
) -> PipelineOutputs:
    """Run and compare the direct checkpoint pipeline and ``encode_batch``."""

    with torch.inference_mode():
        raw_features = frontend.compute_features(waveforms)
        normalized_features = frontend.mean_var_norm(
            raw_features,
            relative_lengths,
        )
        direct_embedding = frontend.embedding_model(
            normalized_features,
            relative_lengths,
        )
        encoded_embedding = frontend.encode_batch(
            waveforms,
            relative_lengths,
            normalize=False,
        )

    batch_size = waveforms.shape[0]
    assert raw_features.ndim == 3
    assert raw_features.shape[0] == batch_size
    assert raw_features.shape[-1] == frontend.NUM_MEL_BINS
    assert normalized_features.shape == raw_features.shape
    assert direct_embedding.shape == (
        batch_size,
        1,
        frontend.EMBEDDING_DIM,
    )
    assert encoded_embedding.shape == direct_embedding.shape

    tensors = {
        "waveform": waveforms,
        "raw SpeechBrain features": raw_features,
        "normalized SpeechBrain features": normalized_features,
        "direct pretrained embedding": direct_embedding,
        "encode_batch embedding": encoded_embedding,
    }
    print(f"{label} relative waveform lengths: {relative_lengths.tolist()}")
    for tensor_name, tensor in tensors.items():
        assert tensor.dtype == torch.float32
        assert_finite(tensor, tensor_name)
        print_tensor_status(f"{label} {tensor_name}", tensor)

    difference = (direct_embedding - encoded_embedding).abs()
    maximum_difference = float(difference.max().item())
    mean_difference = float(difference.mean().item())
    numerically_close = torch.allclose(
        direct_embedding,
        encoded_embedding,
        rtol=1e-5,
        atol=1e-6,
    )
    print(
        f"{label} direct-vs-encode_batch: shape_compatible=True, "
        f"allclose={numerically_close}, max_abs_diff={maximum_difference:.9g}, "
        f"mean_abs_diff={mean_difference:.9g}"
    )

    return PipelineOutputs(
        raw_features=raw_features,
        normalized_features=normalized_features,
        direct_embedding=direct_embedding,
        encoded_embedding=encoded_embedding,
    )


def tensor_summary(tensor: Tensor) -> dict[str, float]:
    """Return bounded scalar summary statistics for a feature tensor."""

    values = tensor.detach().to(device="cpu", dtype=torch.float32)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def compare_custom_frontend(waveform: Tensor, speechbrain_features: Tensor) -> None:
    """Compare only shapes, dtype, finiteness, and summaries for one WAV."""

    custom_extractor = FbankExtractor()
    with torch.inference_mode():
        custom_features = custom_extractor(
            waveform,
            sample_rate=SAMPLE_RATE,
            input_is_batched=False,
        )

    assert custom_features.shape[0] == speechbrain_features.shape[-1] == 80
    assert_finite(custom_features, "custom Fbank")
    assert_finite(speechbrain_features, "SpeechBrain raw features")

    print("Custom-versus-SpeechBrain frontend comparison (no equality required):")
    print(
        "  custom: "
        f"mel_bins={custom_features.shape[0]}, time_frames={custom_features.shape[1]}, "
        f"dtype={custom_features.dtype}, finite=True, "
        f"summary={tensor_summary(custom_features)}"
    )
    print(
        "  SpeechBrain: "
        f"mel_bins={speechbrain_features.shape[-1]}, "
        f"time_frames={speechbrain_features.shape[1]}, "
        f"dtype={speechbrain_features.dtype}, finite=True, "
        f"summary={tensor_summary(speechbrain_features)}"
    )


def verify_checkpoint_files(frontend: SpeechBrainECAPAFrontend) -> None:
    """Verify and print every required downloaded checkpoint file."""

    print(f"Checkpoint cache: {frontend.cache_dir}")
    for path in frontend.checkpoint_files():
        status = "OK" if path.is_file() and path.stat().st_size > 0 else "MISSING"
        size = path.stat().st_size if path.is_file() else 0
        print(f"  {path.name}: {status}, {size} bytes")
    complete = frontend.checkpoint_files_complete()
    print(f"All required pretrained files downloaded successfully: {complete}")
    if not complete:
        raise AssertionError("one or more required pretrained files are missing or empty")


def print_package_versions() -> None:
    """Print the exact runtime versions relevant to this integration."""

    print(f"Python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"speechbrain: {speechbrain.__version__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Existing manifest CSV (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Pretrained checkpoint cache (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu or cuda:0 (default: cpu)",
    )
    parser.add_argument(
        "--skip-custom-comparison",
        action="store_true",
        help="Skip the optional existing-custom-Fbank comparison",
    )
    return parser.parse_args()


def main() -> None:
    """Run one-file and four-file frozen pretrained ECAPA smoke tests."""

    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    print_package_versions()

    records = select_train_small_records(manifest_path)
    waveforms, relative_lengths = load_waveform_batch(records)
    if tuple(waveforms.shape) != (NUM_TEST_FILES, EXPECTED_NUM_SAMPLES):
        raise AssertionError(
            f"expected waveform batch [{NUM_TEST_FILES}, {EXPECTED_NUM_SAMPLES}], "
            f"got {list(waveforms.shape)}"
        )
    print(f"Selected filename_group: {SMOKE_TEST_GROUP}")
    print(f"Selected speaker IDs: {[record.speaker_id for record in records]}")
    print(f"Waveform batch shape: {list(waveforms.shape)}")
    print(f"Relative waveform lengths: {relative_lengths.tolist()}")

    frontend = SpeechBrainECAPAFrontend(
        cache_dir=args.cache_dir,
        device=args.device,
    )
    print(f"Configured device: {frontend.device}")
    print(f"Pretrained source: {frontend.SOURCE}")
    print(f"All pretrained parameters frozen: {frontend.pretrained_parameters_frozen}")
    assert frontend.pretrained_parameters_frozen
    verify_checkpoint_files(frontend)

    one_outputs = run_pipeline_test(
        "One-file",
        frontend,
        waveforms[:1],
        relative_lengths[:1],
    )
    run_pipeline_test(
        "Batch-of-four",
        frontend,
        waveforms,
        relative_lengths,
    )

    if not args.skip_custom_comparison:
        compare_custom_frontend(
            waveforms[:1],
            one_outputs.raw_features,
        )

    print("All frozen pretrained ECAPA integration smoke tests passed.")
    print("No optimizer, backward pass, weight update, or classifier inference was used.")


if __name__ == "__main__":
    main()
