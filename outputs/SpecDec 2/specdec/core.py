"""Exact speculative decoding implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from time import perf_counter
from typing import Sequence

from .adaptive import AdaptiveSpeculationController
from .distributions import (
    acceptance_probability,
    categorical_sample,
    normalize,
    positive_difference_distribution,
)
from .models import LanguageModel


@dataclass(frozen=True)
class SpeculativeStep:
    proposed: list[int]
    accepted: list[int]
    emitted: list[int]
    rejected_at: int | None
    depth: int

    @property
    def all_accepted(self) -> bool:
        return self.rejected_at is None


@dataclass
class GenerationStats:
    target_calls: int = 0
    draft_calls: int = 0
    target_token_positions: int = 0
    draft_token_positions: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    emitted_tokens: int = 0
    elapsed_seconds: float = 0.0
    depth_history: list[int] = field(default_factory=list)
    accepted_per_step: list[int] = field(default_factory=list)

    @property
    def mean_acceptance_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return self.emitted_tokens / self.elapsed_seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "target_calls": self.target_calls,
            "draft_calls": self.draft_calls,
            "target_token_positions": self.target_token_positions,
            "draft_token_positions": self.draft_token_positions,
            "proposed_tokens": self.proposed_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "emitted_tokens": self.emitted_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens_per_second": self.tokens_per_second,
            "mean_acceptance_rate": self.mean_acceptance_rate,
            "depth_history": list(self.depth_history),
            "accepted_per_step": list(self.accepted_per_step),
        }


class SpeculativeDecoder:
    """Run exact speculative decoding with a draft and target model."""

    def __init__(
        self,
        *,
        target: LanguageModel,
        draft: LanguageModel,
        rng: Random | None = None,
        controller: AdaptiveSpeculationController | None = None,
    ) -> None:
        if target.vocab_size != draft.vocab_size:
            raise ValueError("target and draft models must share the same vocabulary")
        self.target = target
        self.draft = draft
        self.rng = rng or Random()
        self.controller = controller

    def draft_tokens(
        self, prefix: Sequence[int], depth: int, stats: GenerationStats | None = None
    ) -> tuple[list[int], list[list[float]]]:
        tokens: list[int] = []
        draft_probs: list[list[float]] = []
        context = list(prefix)
        for _ in range(depth):
            probs = normalize(self.draft.next_token_distribution(context))
            token = categorical_sample(probs, self.rng)
            tokens.append(token)
            draft_probs.append(probs)
            context.append(token)
            if stats is not None:
                stats.draft_calls += 1
                stats.draft_token_positions += 1
        return tokens, draft_probs

    def speculative_step(
        self, prefix: Sequence[int], depth: int, stats: GenerationStats | None = None
    ) -> SpeculativeStep:
        if depth < 1:
            raise ValueError("depth must be >= 1")

        proposed, draft_probs = self.draft_tokens(prefix, depth, stats)
        target_probs = [
            normalize(row)
            for row in self.target.batch_next_token_distributions(prefix, proposed)
        ]

        if stats is not None:
            stats.target_calls += 1
            stats.target_token_positions += len(proposed) + 1
            stats.proposed_tokens += len(proposed)
            stats.depth_history.append(depth)

        accepted: list[int] = []
        emitted: list[int] = []
        rejected_at: int | None = None

        for idx, token in enumerate(proposed):
            p = target_probs[idx][token]
            q = draft_probs[idx][token]
            alpha = acceptance_probability(p, q)
            if self.rng.random() <= alpha:
                accepted.append(token)
                emitted.append(token)
                continue

            replacement_probs = positive_difference_distribution(
                target_probs[idx], draft_probs[idx]
            )
            emitted.append(categorical_sample(replacement_probs, self.rng))
            rejected_at = idx
            break

        if rejected_at is None:
            bonus = categorical_sample(target_probs[-1], self.rng)
            emitted.append(bonus)

        if stats is not None:
            stats.accepted_tokens += len(accepted)
            stats.accepted_per_step.append(len(accepted))
            stats.rejected_tokens += 0 if rejected_at is None else 1
            stats.emitted_tokens += len(emitted)
            if self.controller is not None:
                self.controller.observe(len(accepted), len(proposed))

        return SpeculativeStep(
            proposed=proposed,
            accepted=accepted,
            emitted=emitted,
            rejected_at=rejected_at,
            depth=depth,
        )

    def generate(
        self,
        prefix: Sequence[int],
        *,
        max_new_tokens: int,
        depth: int = 4,
    ) -> tuple[list[int], GenerationStats]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        output = list(prefix)
        stats = GenerationStats()
        start = perf_counter()

        while len(output) - len(prefix) < max_new_tokens:
            remaining = max_new_tokens - (len(output) - len(prefix))
            if remaining == 1:
                probs = normalize(self.target.next_token_distribution(output))
                output.append(categorical_sample(probs, self.rng))
                stats.target_calls += 1
                stats.target_token_positions += 1
                stats.emitted_tokens += 1
                break

            step_depth = self.controller.current_depth if self.controller else depth
            step_depth = max(1, min(step_depth, remaining - 1))
            step = self.speculative_step(output, step_depth, stats)
            output.extend(step.emitted[:remaining])

        stats.elapsed_seconds = perf_counter() - start
        return output, stats


def autoregressive_generate(
    model: LanguageModel,
    prefix: Sequence[int],
    *,
    max_new_tokens: int,
    rng: Random | None = None,
) -> tuple[list[int], GenerationStats]:
    sampler = rng or Random()
    output = list(prefix)
    stats = GenerationStats()
    start = perf_counter()
    for _ in range(max_new_tokens):
        probs = normalize(model.next_token_distribution(output))
        output.append(categorical_sample(probs, sampler))
        stats.target_calls += 1
        stats.target_token_positions += 1
        stats.emitted_tokens += 1
    stats.elapsed_seconds = perf_counter() - start
    return output, stats
