"""O(N log N) grouped verification operating points and EER metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    far: float
    frr: float


@dataclass(frozen=True)
class EERResult:
    interpolated_eer: float
    interpolated_eer_percentage: float
    interpolated_threshold: float
    interpolated_far: float
    interpolated_frr: float
    empirical_threshold: float
    empirical_far: float
    empirical_frr: float
    empirical_far_frr_gap: float
    empirical_average_error: float
    empirical_threshold_semantics: str
    eer_kind: str
    eer_threshold_kind: str
    eer: float
    eer_percentage: float
    eer_threshold: float
    far: float
    frr: float

    @property
    def threshold(self) -> float:
        """Backward-compatible alias for the non-empirical interpolated threshold."""
        return self.interpolated_threshold


def verification_operating_points(
    scores: Sequence[float], targets: Sequence[int]
) -> tuple[OperatingPoint, ...]:
    if len(scores) != len(targets) or not scores:
        raise ValueError("scores and targets must have equal non-zero length")
    numeric_scores = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in numeric_scores):
        raise ValueError("scores contain NaN or Inf")
    if any(target not in (0, 1) for target in targets):
        raise ValueError("targets must be 0 or 1")
    positives, negatives = sum(targets), len(targets) - sum(targets)
    if positives == 0 or negatives == 0:
        raise ValueError("both positive and negative trials are required")
    if min(numeric_scores) < -1.000001 or max(numeric_scores) > 1.000001:
        raise ValueError("scores exceed cosine range")

    ordered = sorted(zip(numeric_scores, targets), key=lambda item: item[0], reverse=True)
    points = [OperatingPoint(math.nextafter(ordered[0][0], math.inf), 0.0, 1.0)]
    accepted_positive = accepted_negative = 0
    position = 0
    while position < len(ordered):
        threshold = ordered[position][0]
        while position < len(ordered) and ordered[position][0] == threshold:
            if ordered[position][1] == 1:
                accepted_positive += 1
            else:
                accepted_negative += 1
            position += 1
        points.append(OperatingPoint(
            threshold, accepted_negative / negatives, (positives - accepted_positive) / positives
        ))
    return tuple(points)


def _empirical_errors(
    scores: Sequence[float], targets: Sequence[int], threshold: float
) -> tuple[float, float]:
    positives, negatives = sum(targets), len(targets) - sum(targets)
    far = sum(target == 0 and score >= threshold for score, target in zip(scores, targets)) / negatives
    frr = sum(target == 1 and score < threshold for score, target in zip(scores, targets)) / positives
    return far, frr


def calculate_eer(scores: Sequence[float], targets: Sequence[int]) -> EERResult:
    points = verification_operating_points(scores, targets)
    crossing: tuple[float, float, float] | None = None
    for point in points:
        if point.far == point.frr:
            crossing = (point.threshold, point.far, point.frr)
            break
    if crossing is None:
        for first, second in zip(points, points[1:]):
            d1, d2 = first.far - first.frr, second.far - second.frr
            if d1 < 0 < d2:
                weight = -d1 / (d2 - d1)
                crossing = (
                    first.threshold + weight * (second.threshold - first.threshold),
                    first.far + weight * (second.far - first.far),
                    first.frr + weight * (second.frr - first.frr),
                )
                break
    if crossing is None:
        raise RuntimeError("FAR/FRR crossing not found")
    interpolated_threshold, interpolated_far, interpolated_frr = crossing
    interpolated_eer = (interpolated_far + interpolated_frr) / 2.0

    # Boundary point is retained for ROC completeness but an executable threshold
    # is selected from actual score thresholds, then ties prefer the higher value.
    empirical = min(
        points[1:],
        key=lambda point: (
            abs(point.far - point.frr),
            (point.far + point.frr) / 2.0,
            -point.threshold,
        ),
    )
    recomputed_far, recomputed_frr = _empirical_errors(scores, targets, empirical.threshold)
    if not (
        math.isclose(recomputed_far, empirical.far, abs_tol=1e-15)
        and math.isclose(recomputed_frr, empirical.frr, abs_tol=1e-15)
    ):
        raise RuntimeError("empirical FAR/FRR recomputation mismatch")
    semantics = "accept same speaker when score >= threshold"
    return EERResult(
        interpolated_eer=interpolated_eer,
        interpolated_eer_percentage=interpolated_eer * 100.0,
        interpolated_threshold=interpolated_threshold,
        interpolated_far=interpolated_far,
        interpolated_frr=interpolated_frr,
        empirical_threshold=empirical.threshold,
        empirical_far=recomputed_far,
        empirical_frr=recomputed_frr,
        empirical_far_frr_gap=abs(recomputed_far - recomputed_frr),
        empirical_average_error=(recomputed_far + recomputed_frr) / 2.0,
        empirical_threshold_semantics=semantics,
        eer_kind="linearly_interpolated_roc_crossing",
        eer_threshold_kind="interpolated_non_empirical",
        eer=interpolated_eer,
        eer_percentage=interpolated_eer * 100.0,
        eer_threshold=interpolated_threshold,
        far=interpolated_far,
        frr=interpolated_frr,
    )
