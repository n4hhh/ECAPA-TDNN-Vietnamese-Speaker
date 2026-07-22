"""Reusable, in-memory log-Mel filterbank extraction."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torchaudio.compliance import kaldi


class FbankExtractor(nn.Module):
    """Extract Kaldi-compatible log-Mel filterbanks with torchaudio.

    Shape conventions are:

    * ``[N]`` or ``[1, N]``: one mono waveform -> ``[M, T]``
    * ``[B, N]`` (where ``B > 1``): a channel-free mono batch -> ``[B, M, T]``
    * ``[B, 1, N]``: an explicit mono batch -> ``[B, M, T]``

    A rank-two ``[B, N]`` batch is inherently indistinguishable from a
    channel-first ``[C, N]`` waveform. Set ``input_is_batched=False`` when a
    rank-two tensor represents one channel-first waveform; stereo input will
    then be rejected. Use ``[1, 1, N]`` or ``input_is_batched=True`` to retain
    a batch dimension for a one-item batch.

    Floating-point inputs are assumed to be normalized audio and are cast to
    ``float32``. Integer PCM tensors are scaled to the conventional ``[-1, 1)``
    range before extraction. Features are returned in memory only.

    Args:
        sample_rate: Expected waveform sample rate in Hz.
        num_mel_bins: Number of Mel filterbank bins, ``M``.
        frame_length_ms: Analysis window length in milliseconds.
        frame_shift_ms: Frame hop in milliseconds.
        dither: Kaldi dither magnitude. The deterministic default is zero.
        low_freq: Lower filterbank edge in Hz.
        high_freq: Upper edge in Hz; zero means Nyquist in Kaldi semantics.
        snip_edges: Use only frames whose windows fit within the waveform.
        window_type: Kaldi analysis-window name.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        num_mel_bins: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        dither: float = 0.0,
        low_freq: float = 20.0,
        high_freq: float = 0.0,
        snip_edges: bool = True,
        window_type: str = "povey",
    ) -> None:
        super().__init__()

        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if isinstance(num_mel_bins, bool) or not isinstance(num_mel_bins, int):
            raise TypeError("num_mel_bins must be an integer")
        if num_mel_bins <= 0:
            raise ValueError("num_mel_bins must be positive")

        self._validate_positive_finite("frame_length_ms", frame_length_ms)
        self._validate_positive_finite("frame_shift_ms", frame_shift_ms)
        self._validate_nonnegative_finite("dither", dither)
        self._validate_nonnegative_finite("low_freq", low_freq)
        if not isinstance(high_freq, (int, float)) or isinstance(high_freq, bool):
            raise TypeError("high_freq must be a real number")
        if not math.isfinite(float(high_freq)):
            raise ValueError("high_freq must be finite")
        if not isinstance(snip_edges, bool):
            raise TypeError("snip_edges must be a bool")
        if not isinstance(window_type, str) or not window_type:
            raise ValueError("window_type must be a non-empty string")

        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length_ms = float(frame_length_ms)
        self.frame_shift_ms = float(frame_shift_ms)
        self.dither = float(dither)
        self.low_freq = float(low_freq)
        self.high_freq = float(high_freq)
        self.snip_edges = snip_edges
        self.window_type = window_type

    @staticmethod
    def _validate_positive_finite(name: str, value: float) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")

    @staticmethod
    def _validate_nonnegative_finite(name: str, value: float) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative and finite")

    @property
    def minimum_num_samples(self) -> int:
        """Minimum sample count needed for one complete analysis window."""

        return max(1, int(self.sample_rate * self.frame_length_ms / 1_000.0))

    @staticmethod
    def _pcm_to_float32(waveform: Tensor) -> Tensor:
        """Convert floating or integer PCM audio to a finite float32 tensor."""

        if waveform.layout != torch.strided:
            raise TypeError("waveform must be a dense strided tensor")
        if waveform.dtype == torch.bool or waveform.is_complex():
            raise TypeError(f"unsupported waveform dtype: {waveform.dtype}")

        if waveform.is_floating_point():
            converted = waveform.to(dtype=torch.float32)
        else:
            try:
                dtype_info = torch.iinfo(waveform.dtype)
            except TypeError as error:
                raise TypeError(f"unsupported waveform dtype: {waveform.dtype}") from error

            converted = waveform.to(dtype=torch.float32)
            if dtype_info.min < 0:
                scale = float(max(-dtype_info.min, dtype_info.max))
                converted = converted / scale
            else:
                midpoint = float(dtype_info.max + 1) / 2.0
                converted = (converted - midpoint) / midpoint

        if not bool(torch.isfinite(converted).all().item()):
            raise ValueError("waveform contains NaN or Inf")
        return converted.contiguous()

    def _normalize_shape(
        self,
        waveform: Tensor,
        input_is_batched: bool | None,
    ) -> tuple[Tensor, bool]:
        """Return channel-free ``[B, N]`` audio and whether input was single."""

        if input_is_batched is not None and not isinstance(input_is_batched, bool):
            raise TypeError("input_is_batched must be bool or None")

        if waveform.ndim == 1:
            if input_is_batched is True:
                raise ValueError("rank-one [N] input represents one waveform, not a batch")
            batch = waveform.unsqueeze(0)
            is_single = True
        elif waveform.ndim == 2:
            if input_is_batched is False:
                if waveform.shape[0] != 1:
                    raise ValueError(
                        "expected one mono channel in [1, N] input; "
                        f"received {waveform.shape[0]} channels"
                    )
                batch = waveform
                is_single = True
            elif input_is_batched is True:
                batch = waveform
                is_single = False
            else:
                batch = waveform
                is_single = waveform.shape[0] == 1
        elif waveform.ndim == 3:
            if input_is_batched is False:
                raise ValueError("rank-three input is always interpreted as a batch")
            if waveform.shape[1] != 1:
                raise ValueError(
                    "expected mono [B, 1, N] input; "
                    f"received {waveform.shape[1]} channels"
                )
            batch = waveform[:, 0, :]
            is_single = False
        else:
            raise ValueError(
                "waveform shape must be [N], [1, N], [B, N], or [B, 1, N]; "
                f"received rank {waveform.ndim}"
            )

        if batch.shape[0] == 0:
            raise ValueError("waveform batch must contain at least one sample")
        if batch.shape[1] == 0:
            raise ValueError("waveform must be non-empty")
        if batch.shape[1] < self.minimum_num_samples:
            raise ValueError(
                f"waveform has {batch.shape[1]} samples, but at least "
                f"{self.minimum_num_samples} are needed for one frame"
            )
        return batch, is_single

    def forward(
        self,
        waveform: Tensor,
        sample_rate: int | None = None,
        *,
        input_is_batched: bool | None = None,
    ) -> Tensor:
        """Compute log-Mel filterbanks.

        Args:
            waveform: A mono tensor with one of the documented input shapes.
            sample_rate: Runtime sample rate. When supplied, it must equal the
                configured rate. If omitted, the configured rate is assumed.
            input_is_batched: Optional rank-two shape disambiguation. See the
                class docstring.

        Returns:
            A contiguous float32 tensor shaped ``[M, T]`` for one waveform or
            ``[B, M, T]`` for a batch.

        Raises:
            TypeError: If the tensor or its dtype is unsupported.
            ValueError: If shape, rate, length, channel count, or values are
                invalid, or if extraction produces a non-finite result.
        """

        if not isinstance(waveform, Tensor):
            raise TypeError("waveform must be a torch.Tensor")
        if sample_rate is not None:
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
                raise TypeError("sample_rate must be an integer")
            if sample_rate != self.sample_rate:
                raise ValueError(
                    f"sample rate mismatch: expected {self.sample_rate}, got {sample_rate}"
                )

        batch, is_single = self._normalize_shape(waveform, input_is_batched)
        batch = self._pcm_to_float32(batch)

        outputs: list[Tensor] = []
        for sample in batch:
            features_time_major = kaldi.fbank(
                sample.unsqueeze(0),
                channel=0,
                dither=self.dither,
                frame_length=self.frame_length_ms,
                frame_shift=self.frame_shift_ms,
                high_freq=self.high_freq,
                low_freq=self.low_freq,
                num_mel_bins=self.num_mel_bins,
                sample_frequency=float(self.sample_rate),
                snip_edges=self.snip_edges,
                subtract_mean=False,
                use_energy=False,
                use_log_fbank=True,
                use_power=True,
                window_type=self.window_type,
            )
            features = features_time_major.transpose(0, 1).to(torch.float32).contiguous()
            if features.ndim != 2 or features.shape[0] != self.num_mel_bins:
                raise RuntimeError(f"unexpected Fbank output shape: {tuple(features.shape)}")
            if features.shape[1] == 0:
                raise ValueError("Fbank extraction produced no frames")
            if not bool(torch.isfinite(features).all().item()):
                raise ValueError("Fbank output contains NaN or Inf")
            outputs.append(features)

        if is_single:
            return outputs[0].contiguous()
        return torch.stack(outputs, dim=0).to(torch.float32).contiguous()

    def extra_repr(self) -> str:
        """Return the primary extractor configuration for module displays."""

        return (
            f"sample_rate={self.sample_rate}, num_mel_bins={self.num_mel_bins}, "
            f"frame_length_ms={self.frame_length_ms}, "
            f"frame_shift_ms={self.frame_shift_ms}"
        )
