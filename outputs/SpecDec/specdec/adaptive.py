"""Adaptive speculation-depth controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveSpeculationController:
    """Tune draft depth from recent acceptance behavior.

    The controller intentionally uses a conservative additive rule. Large
    jumps make benchmark curves harder to interpret and can thrash when prompt
    difficulty changes.
    """

    initial_depth: int = 4
    min_depth: int = 1
    max_depth: int = 12
    target_acceptance: float = 0.72
    tolerance: float = 0.05
    smoothing: float = 0.25

    def __post_init__(self) -> None:
        if self.min_depth < 1:
            raise ValueError("min_depth must be >= 1")
        if self.max_depth < self.min_depth:
            raise ValueError("max_depth must be >= min_depth")
        if not (0.0 < self.target_acceptance < 1.0):
            raise ValueError("target_acceptance must be in (0, 1)")
        if not (0.0 < self.smoothing <= 1.0):
            raise ValueError("smoothing must be in (0, 1]")
        self.current_depth = max(self.min_depth, min(self.max_depth, self.initial_depth))
        self.acceptance_ewma: float | None = None

    def observe(self, accepted: int, proposed: int) -> int:
        if proposed <= 0:
            return self.current_depth
        rate = accepted / proposed
        if self.acceptance_ewma is None:
            self.acceptance_ewma = rate
        else:
            alpha = self.smoothing
            self.acceptance_ewma = alpha * rate + (1.0 - alpha) * self.acceptance_ewma

        if self.acceptance_ewma > self.target_acceptance + self.tolerance:
            self.current_depth = min(self.max_depth, self.current_depth + 1)
        elif self.acceptance_ewma < self.target_acceptance - self.tolerance:
            self.current_depth = max(self.min_depth, self.current_depth - 1)
        return self.current_depth
