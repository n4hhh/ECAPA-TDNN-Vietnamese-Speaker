"""CUDA smoke test for the frozen SpeechBrain ECAPA-TDNN checkpoint."""

from __future__ import annotations

import csv
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import huggingface_hub
import speechbrain
import torch
import torchaudio

from src.speechbrain_frontend import DEFAULT_CACHE_DIR, SpeechBrainECAPAFrontend

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "full_manifest.csv"
DEVICE = torch.device("cuda:0")
EXPECTED_SAMPLES = 48_000
EXPECTED_FRAMES = 301
NUM_FILES = 4


def mib(value: int) -> float:
    return value / (1024**2)


def select_waveforms() -> tuple[list[Path], torch.Tensor]:
    paths: list[Path] = []
    waveforms: list[torch.Tensor] = []
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            path = Path(row["audio_path"].strip())
            if not path.is_file():
                continue
            try:
                waveform, sample_rate = torchaudio.load(str(path), normalize=True)
            except (RuntimeError, OSError):
                continue
            if (
                sample_rate != 16_000
                or waveform.shape != (1, EXPECTED_SAMPLES)
                or waveform.dtype != torch.float32
                or not bool(torch.isfinite(waveform).all())
            ):
                continue
            paths.append(path)
            waveforms.append(waveform.squeeze(0))
            if len(paths) == NUM_FILES:
                break
    if len(paths) != NUM_FILES:
        raise RuntimeError(f"needed four valid 16 kHz mono 3-second WAVs, found {len(paths)}")
    return paths, torch.stack(waveforms)


def check_tensor(label: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise AssertionError(f"{label}: expected shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != DEVICE:
        raise AssertionError(f"{label}: expected {DEVICE}, got {tensor.device}")
    if tensor.dtype != torch.float32:
        raise AssertionError(f"{label}: expected float32, got {tensor.dtype}")
    finite = bool(torch.isfinite(tensor).all().item())
    if not finite:
        raise AssertionError(f"{label}: contains NaN or Inf")
    print(f"{label}: shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, finite={finite}")


def run_case(label: str, model: SpeechBrainECAPAFrontend, waveforms: torch.Tensor) -> None:
    batch_size = waveforms.shape[0]
    lengths = torch.ones(batch_size, device=DEVICE, dtype=torch.float32)
    with torch.inference_mode():
        features = model.compute_features(waveforms)
        normalized = model.mean_var_norm(features, lengths)
        direct = model.embedding_model(normalized, lengths)
        encoded = model.encode_batch(waveforms, lengths, normalize=False)

    check_tensor(f"{label} waveform", waveforms, (batch_size, EXPECTED_SAMPLES))
    check_tensor(f"{label} features", features, (batch_size, EXPECTED_FRAMES, 80))
    check_tensor(f"{label} normalized features", normalized, (batch_size, EXPECTED_FRAMES, 80))
    check_tensor(f"{label} direct embedding", direct, (batch_size, 1, 192))
    check_tensor(f"{label} encode_batch embedding", encoded, (batch_size, 1, 192))
    max_difference = float((direct - encoded).abs().max().item())
    close = bool(torch.allclose(direct, encoded, atol=1e-5, rtol=1e-4))
    print(f"{label} direct-vs-encode_batch: max_abs_diff={max_difference:.9g}, allclose={close}, atol=1e-5, rtol=1e-4")


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"torchaudio: {torchaudio.__version__}")
    print(f"speechbrain: {speechbrain.__version__}")
    print(f"huggingface_hub: {huggingface_hub.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    left = torch.arange(8, device=DEVICE, dtype=torch.float32)
    result = left.square().add_(1.0)
    check_tensor("Basic CUDA result", result, (8,))
    print("Basic CUDA tensor test: PASS")

    paths, cpu_waveforms = select_waveforms()
    print("Selected WAV files:")
    for path in paths:
        print(f"  {path}")

    torch.cuda.reset_peak_memory_stats(DEVICE)
    model = SpeechBrainECAPAFrontend(cache_dir=DEFAULT_CACHE_DIR, device=DEVICE)
    model.eval()
    waveforms = cpu_waveforms.to(DEVICE)
    run_case("Single", model, waveforms[:1])
    run_case("Batch", model, waveforms)
    torch.cuda.synchronize(DEVICE)

    free_bytes, total_bytes = torch.cuda.mem_get_info(DEVICE)
    allocated = torch.cuda.max_memory_allocated(DEVICE)
    reserved = torch.cuda.max_memory_reserved(DEVICE)
    print(f"VRAM free: {mib(free_bytes):.2f} MiB")
    print(f"VRAM total: {mib(total_bytes):.2f} MiB")
    print(f"VRAM max allocated: {mib(allocated):.2f} MiB")
    print(f"VRAM max reserved: {mib(reserved):.2f} MiB")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
