"""Small categorical-distribution utilities with no third-party dependency."""

from __future__ import annotations

from math import isfinite
from random import Random
from typing import Iterable, Sequence


Distribution = list[float]


class DistributionError(ValueError):
    """Raised when a probability vector is invalid."""


def normalize(values: Iterable[float], *, eps: float = 1e-15) -> Distribution:
    probs = [float(v) for v in values]
    if not probs:
        raise DistributionError("distribution must contain at least one value")
    if any((not isfinite(p)) or p < -eps for p in probs):
        raise DistributionError(f"distribution contains invalid values: {probs!r}")

    clipped = [0.0 if p < 0.0 else p for p in probs]
    total = sum(clipped)
    if total <= eps:
        raise DistributionError("distribution has zero total mass")
    return [p / total for p in clipped]


def categorical_sample(probs: Sequence[float], rng: Random) -> int:
    """Sample an index from a normalized categorical distribution."""

    if not probs:
        raise DistributionError("cannot sample from an empty distribution")
    threshold = rng.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if threshold < cumulative:
            return idx
    return len(probs) - 1


def acceptance_probability(target_prob: float, draft_prob: float) -> float:
    """Return min(1, p(x) / q(x)), with deterministic handling of q(x)=0."""

    if draft_prob <= 0.0:
        return 1.0 if target_prob > 0.0 else 0.0
    return min(1.0, target_prob / draft_prob)


def positive_difference_distribution(
    target: Sequence[float], draft: Sequence[float], *, eps: float = 1e-15
) -> Distribution:
    """Normalize max(target - draft, 0), the paper's rejection correction."""

    if len(target) != len(draft):
        raise DistributionError("target and draft distributions must share a vocabulary")

    residual = [max(float(p) - float(q), 0.0) for p, q in zip(target, draft)]
    if sum(residual) <= eps:
        # This can happen from floating-point roundoff when p <= q coordinate-wise.
        # Falling back to the target distribution preserves validity while keeping
        # the branch reachable for numerically awkward toy cases.
        return normalize(target)
    return normalize(residual)


def l1_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise DistributionError("distributions must share a vocabulary")
    return sum(abs(a - b) for a, b in zip(left, right))
