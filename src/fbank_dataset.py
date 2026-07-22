"""Lazy ``train_small`` dataset for Fbank smoke tests only."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

try:  # Support both ``python src/file.py`` and ``python -m src.file``.
    from .fbank import FbankExtractor
except ImportError:  # pragma: no cover - exercised by direct script execution
    from fbank import FbankExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "manifests" / "full_manifest.csv"
SMOKE_TEST_GROUP = "train_small"


@dataclass(frozen=True)
class FbankManifestRecord:
    """Metadata retained for one lazily decoded smoke-test waveform."""

    audio_path: str
    speaker_id: str
    duration_seconds: float | None


FbankSample = dict[str, Tensor | str | int]


class FbankSmokeTestDataset(Dataset[FbankSample]):
    """Lazily decode only ``train_small`` rows and compute Fbank in memory.

    Manifest metadata is retained in a bounded list, while waveforms are loaded
    one at a time by :meth:`__getitem__`. The filename group is intentionally
    fixed and cannot be changed to ``test`` or ``part``.

    Args:
        manifest_path: Existing CSV manifest path.
        max_samples: Maximum number of ``train_small`` rows to retain. ``None``
            retains all matching metadata, never the waveforms. The conservative
            default is 100.
        extractor: Optional reusable extractor. Its configured sample rate must
            match ``expected_sample_rate``.
        expected_sample_rate: Required decoded WAV sample rate.
    """

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        max_samples: int | None = 100,
        extractor: FbankExtractor | None = None,
        expected_sample_rate: int = 16_000,
    ) -> None:
        if max_samples is not None:
            if isinstance(max_samples, bool) or not isinstance(max_samples, int):
                raise TypeError("max_samples must be an integer or None")
            if max_samples <= 0:
                raise ValueError("max_samples must be positive when provided")
        if isinstance(expected_sample_rate, bool) or not isinstance(
            expected_sample_rate, int
        ):
            raise TypeError("expected_sample_rate must be an integer")
        if expected_sample_rate <= 0:
            raise ValueError("expected_sample_rate must be positive")

        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
        self.expected_sample_rate = expected_sample_rate
        self.extractor = extractor or FbankExtractor(sample_rate=expected_sample_rate)
        if self.extractor.sample_rate != expected_sample_rate:
            raise ValueError(
                "extractor sample rate does not match dataset expectation: "
                f"{self.extractor.sample_rate} != {expected_sample_rate}"
            )

        records: list[FbankManifestRecord] = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required_columns = {"audio_path", "speaker_id", "filename_group"}
            columns = set(reader.fieldnames or ())
            missing_columns = required_columns - columns
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"manifest is missing required columns: {missing}")

            for row_number, row in enumerate(reader, start=2):
                if row["filename_group"].strip() != SMOKE_TEST_GROUP:
                    continue

                raw_path = row["audio_path"].strip()
                speaker_id = row["speaker_id"].strip()
                if not raw_path:
                    raise ValueError(f"manifest row {row_number} has an empty audio_path")
                if not speaker_id:
                    raise ValueError(f"manifest row {row_number} has an empty speaker_id")

                audio_path = Path(raw_path).expanduser()
                if not audio_path.is_absolute():
                    audio_path = (PROJECT_ROOT / audio_path).resolve()

                duration_seconds: float | None = None
                raw_duration = (row.get("duration_seconds") or "").strip()
                if raw_duration:
                    try:
                        parsed_duration = float(raw_duration)
                    except ValueError as error:
                        raise ValueError(
                            f"manifest row {row_number} has invalid duration_seconds: "
                            f"{raw_duration!r}"
                        ) from error
                    if not math.isfinite(parsed_duration) or parsed_duration <= 0.0:
                        raise ValueError(
                            f"manifest row {row_number} has non-positive or non-finite "
                            "duration_seconds"
                        )
                    duration_seconds = parsed_duration

                records.append(
                    FbankManifestRecord(
                        audio_path=str(audio_path),
                        speaker_id=speaker_id,
                        duration_seconds=duration_seconds,
                    )
                )
                if max_samples is not None and len(records) >= max_samples:
                    break

        if not records:
            raise ValueError(
                f"manifest contains no rows with filename_group={SMOKE_TEST_GROUP!r}"
            )
        self._records = tuple(records)

    @property
    def records(self) -> tuple[FbankManifestRecord, ...]:
        """Return immutable selected metadata for alignment checks."""

        return self._records

    def __len__(self) -> int:
        """Return the bounded number of selected ``train_small`` rows."""

        return len(self._records)

    def __getitem__(self, index: int) -> FbankSample:
        """Decode one mono WAV and return its in-memory Fbank and identifiers."""

        record = self._records[index]
        audio_path = Path(record.audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file does not exist: {audio_path}")

        waveform, sample_rate = torchaudio.load(str(audio_path), normalize=True)
        if waveform.ndim != 2:
            raise ValueError(
                f"expected channel-first [C, N] audio for {audio_path}, "
                f"got shape {tuple(waveform.shape)}"
            )
        if waveform.shape[0] != 1:
            raise ValueError(
                f"expected mono audio for {audio_path}, got {waveform.shape[0]} channels"
            )
        if waveform.shape[1] == 0:
            raise ValueError(f"decoded empty waveform: {audio_path}")
        if sample_rate != self.expected_sample_rate:
            raise ValueError(
                f"sample rate mismatch for {audio_path}: expected "
                f"{self.expected_sample_rate}, got {sample_rate}"
            )

        with torch.inference_mode():
            features = self.extractor(
                waveform,
                sample_rate=sample_rate,
                input_is_batched=False,
            )
        if features.ndim != 2:
            raise RuntimeError(
                f"expected one [M, T] feature tensor, got {tuple(features.shape)}"
            )

        return {
            "features": features.contiguous(),
            "speaker_id": record.speaker_id,
            "feature_length": int(features.shape[-1]),
            "audio_path": record.audio_path,
        }
