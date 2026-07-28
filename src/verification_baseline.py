"""Dependency-light helpers for validation embedding scoring and summaries."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as functional

from src.verification_trials import Trial


def score_trials(
    embeddings: torch.Tensor, paths: Sequence[str], trials: Sequence[Trial]
) -> torch.Tensor:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(paths):
        raise ValueError("embedding/path dimensions disagree")
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate embedding path")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings contain NaN or Inf")
    lookup = {path: index for index, path in enumerate(paths)}
    scores = []
    for trial in trials:
        try:
            left, right = lookup[trial.left_relative_audio_path], lookup[trial.right_relative_audio_path]
        except KeyError as error:
            raise ValueError(f"missing embedding path: {error.args[0]}") from error
        scores.append(functional.cosine_similarity(
            embeddings[left].unsqueeze(0), embeddings[right].unsqueeze(0), dim=1
        )[0])
    result = torch.stack(scores).to(torch.float32).cpu()
    if not bool(torch.isfinite(result).all()) or result.min() < -1.000001 or result.max() > 1.000001:
        raise ValueError("invalid cosine scores")
    return result


def numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("summary requires finite values")
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values), "minimum": float(tensor.min()), "mean": float(tensor.mean()),
        "standard_deviation": float(tensor.std(unbiased=False)), "q1": float(torch.quantile(tensor, .25)),
        "median": float(torch.quantile(tensor, .5)), "q3": float(torch.quantile(tensor, .75)),
        "maximum": float(tensor.max()),
    }


def validate_saved_artifacts(
    embedding_artifact: dict, score_artifact: dict, trials: Sequence[Trial],
    expected_paths: Sequence[str], expected_speakers: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fail-closed validation for metrics-only reuse of saved baseline artifacts."""
    if len(expected_paths) != len(set(expected_paths)):
        raise ValueError("duplicate validation metadata path")
    paths = embedding_artifact.get("relative_audio_paths")
    speakers = embedding_artifact.get("speaker_ids")
    embeddings = embedding_artifact.get("embeddings")
    if paths != list(expected_paths) or speakers != list(expected_speakers):
        raise ValueError("saved embedding metadata does not match validation order")
    if (
        not isinstance(embeddings, torch.Tensor)
        or tuple(embeddings.shape) != (len(expected_paths), 192)
        or embeddings.dtype != torch.float32
        or embeddings.device.type != "cpu"
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("invalid saved embedding tensor")
    scores = score_artifact.get("scores")
    targets = score_artifact.get("targets")
    trial_ids = score_artifact.get("trial_ids")
    expected_targets = torch.tensor([trial.target for trial in trials], dtype=torch.long)
    if trial_ids != [trial.trial_id for trial in trials]:
        raise ValueError("saved trial IDs do not match trial order")
    if not isinstance(targets, torch.Tensor) or not torch.equal(targets.cpu(), expected_targets):
        raise ValueError("saved targets do not match trial order")
    if (
        not isinstance(scores, torch.Tensor) or tuple(scores.shape) != (len(trials),)
        or not scores.is_floating_point() or scores.device.type != "cpu"
        or not bool(torch.isfinite(scores).all())
        or float(scores.min()) < -1.000001 or float(scores.max()) > 1.000001
    ):
        raise ValueError("invalid saved score tensor")
    return embeddings, scores
