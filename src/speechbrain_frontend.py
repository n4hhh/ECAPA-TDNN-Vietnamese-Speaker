"""Frozen SpeechBrain ECAPA frontend for pretrained integration checks."""

from __future__ import annotations

from pathlib import Path

import torch
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"
)


class SpeechBrainECAPAFrontend(nn.Module):
    """Load and expose the frozen pretrained SpeechBrain ECAPA pipeline.

    This wrapper intentionally exposes the checkpoint's feature extraction,
    sentence normalization, embedding model, and convenience ``encode_batch``
    operation separately. It never calls the pretrained VoxCeleb classifier.

    SpeechBrain feature tensors retain their native ``[B, T, 80]`` layout.
    Waveforms must be normalized floating-point tensors shaped ``[B, N]`` and
    relative lengths must be shaped ``[B]`` with values in ``(0, 1]``.

    Args:
        cache_dir: Directory in which checkpoint files are copied. A private
            Hugging Face cache is also kept below this directory.
        device: PyTorch CPU or CUDA device string/object.
    """

    SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
    SAMPLE_RATE = 16_000
    NUM_MEL_BINS = 80
    EMBEDDING_DIM = 192
    REQUIRED_CHECKPOINT_FILES = (
        "hyperparams.yaml",
        "embedding_model.ckpt",
        "mean_var_norm_emb.ckpt",
        "classifier.ckpt",
        # SpeechBrain copies the remote label_encoder.txt under its loadable key.
        "label_encoder.ckpt",
    )

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._device = self._resolve_device(device)

        self.classifier = EncoderClassifier.from_hparams(
            source=self.SOURCE,
            # This checkpoint has no custom.py. Reusing the already-fetched
            # YAML path avoids SpeechBrain 1.0.x's optional remote 404 probe;
            # from_hparams only adds its parent directory to sys.path.
            pymodule_file="hyperparams.yaml",
            savedir=str(self.cache_dir),
            run_opts={"device": str(self._device)},
            freeze_params=True,
            local_strategy=LocalStrategy.COPY,
            huggingface_cache_dir=str(self.cache_dir / ".hf_cache"),
        )
        self.classifier.eval()

        required_modules = {
            "compute_features",
            "mean_var_norm",
            "embedding_model",
        }
        missing_modules = required_modules - set(self.classifier.mods.keys())
        if missing_modules:
            missing = ", ".join(sorted(missing_modules))
            raise RuntimeError(f"checkpoint is missing required modules: {missing}")
        if not self.pretrained_parameters_frozen:
            raise RuntimeError("one or more pretrained parameters are not frozen")

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        try:
            resolved = torch.device(device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"invalid PyTorch device: {device!r}") from error
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError("device must be a CPU or CUDA device")
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            if resolved.index is not None and resolved.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device index {resolved.index} is unavailable; "
                    f"found {torch.cuda.device_count()} device(s)"
                )
        return resolved

    @property
    def device(self) -> torch.device:
        """Return the device selected when loading the checkpoint."""

        return self._device

    @property
    def pretrained_parameters_frozen(self) -> bool:
        """Return whether every loaded checkpoint parameter is frozen."""

        return all(not parameter.requires_grad for parameter in self.classifier.parameters())

    def checkpoint_files(self) -> tuple[Path, ...]:
        """Return the expected local checkpoint paths in deterministic order."""

        return tuple(
            self.cache_dir / filename for filename in self.REQUIRED_CHECKPOINT_FILES
        )

    def checkpoint_files_complete(self) -> bool:
        """Return whether all expected local checkpoint files are non-empty."""

        return all(path.is_file() and path.stat().st_size > 0 for path in self.checkpoint_files())

    def _prepare_waveforms(self, waveforms: Tensor) -> Tensor:
        if not isinstance(waveforms, Tensor):
            raise TypeError("waveforms must be a torch.Tensor")
        if waveforms.ndim != 2:
            raise ValueError(
                "waveforms must have shape [B, N]; decoded channel dimensions "
                "must be validated and removed by the caller"
            )
        if waveforms.shape[0] == 0 or waveforms.shape[1] == 0:
            raise ValueError("waveforms must contain a non-empty batch and time axis")
        if not waveforms.is_floating_point():
            raise TypeError("waveforms must be normalized floating-point audio")
        prepared = waveforms.to(device=self.device, dtype=torch.float32)
        self._require_finite(prepared, "waveforms")
        return prepared

    def _prepare_features(self, features: Tensor, name: str) -> Tensor:
        if not isinstance(features, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if features.ndim != 3 or features.shape[-1] != self.NUM_MEL_BINS:
            raise ValueError(
                f"{name} must have shape [B, T, {self.NUM_MEL_BINS}], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError(f"{name} must have non-empty batch and time axes")
        if not features.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        prepared = features.to(device=self.device, dtype=torch.float32)
        self._require_finite(prepared, name)
        return prepared

    def _prepare_lengths(self, relative_lengths: Tensor, batch_size: int) -> Tensor:
        if not isinstance(relative_lengths, Tensor):
            raise TypeError("relative_lengths must be a torch.Tensor")
        if relative_lengths.ndim != 1 or relative_lengths.shape[0] != batch_size:
            raise ValueError(
                f"relative_lengths must have shape [{batch_size}], "
                f"got {tuple(relative_lengths.shape)}"
            )
        lengths = relative_lengths.to(device=self.device, dtype=torch.float32)
        self._require_finite(lengths, "relative_lengths")
        if bool(((lengths <= 0.0) | (lengths > 1.0)).any().item()):
            raise ValueError("relative_lengths values must be in (0, 1]")
        return lengths

    @staticmethod
    def _require_finite(tensor: Tensor, name: str) -> None:
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains NaN or Inf")

    def compute_features(self, waveforms: Tensor) -> Tensor:
        """Run the checkpoint-supplied frontend and return ``[B, T, 80]``."""

        prepared = self._prepare_waveforms(waveforms)
        features = self.classifier.mods.compute_features(prepared)
        features = self._prepare_features(features.to(torch.float32), "raw features")
        return features

    def mean_var_norm(self, features: Tensor, relative_lengths: Tensor) -> Tensor:
        """Apply checkpoint-supplied sentence normalization in ``[B, T, 80]``."""

        prepared = self._prepare_features(features, "features")
        lengths = self._prepare_lengths(relative_lengths, prepared.shape[0])
        normalized = self.classifier.mods.mean_var_norm(prepared, lengths)
        normalized = self._prepare_features(
            normalized.to(torch.float32),
            "normalized features",
        )
        if normalized.shape != prepared.shape:
            raise RuntimeError(
                "mean_var_norm changed the feature shape: "
                f"{tuple(prepared.shape)} -> {tuple(normalized.shape)}"
            )
        return normalized

    def embedding_model(
        self,
        normalized_features: Tensor,
        relative_lengths: Tensor,
    ) -> Tensor:
        """Run the pretrained ECAPA encoder and return ``[B, 1, 192]``."""

        prepared = self._prepare_features(normalized_features, "normalized features")
        lengths = self._prepare_lengths(relative_lengths, prepared.shape[0])
        embeddings = self.classifier.mods.embedding_model(prepared, lengths)
        embeddings = embeddings.to(torch.float32)
        if embeddings.ndim != 3 or embeddings.shape != (
            prepared.shape[0],
            1,
            self.EMBEDDING_DIM,
        ):
            raise RuntimeError(
                "unexpected embedding shape: "
                f"expected [{prepared.shape[0]}, 1, {self.EMBEDDING_DIM}], "
                f"got {tuple(embeddings.shape)}"
            )
        self._require_finite(embeddings, "embeddings")
        return embeddings

    def encode_batch(
        self,
        waveforms: Tensor,
        relative_lengths: Tensor,
        *,
        normalize: bool = False,
    ) -> Tensor:
        """Run SpeechBrain ``encode_batch`` without using its classifier head.

        ``normalize=False`` matches the requested direct frontend-to-encoder
        pipeline. Setting it to ``True`` additionally applies the checkpoint's
        embedding-level normalization.
        """

        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool")
        prepared = self._prepare_waveforms(waveforms)
        lengths = self._prepare_lengths(relative_lengths, prepared.shape[0])
        embeddings = self.classifier.encode_batch(
            prepared,
            wav_lens=lengths,
            normalize=normalize,
        ).to(torch.float32)
        if embeddings.ndim != 3 or embeddings.shape != (
            prepared.shape[0],
            1,
            self.EMBEDDING_DIM,
        ):
            raise RuntimeError(f"unexpected encode_batch shape: {tuple(embeddings.shape)}")
        self._require_finite(embeddings, "encode_batch embeddings")
        return embeddings
