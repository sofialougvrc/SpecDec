"""Model interfaces and deterministic toy models used by tests and demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .distributions import Distribution, normalize


TokenIds = list[int]


class LanguageModel(Protocol):
    """Minimal autoregressive interface required by speculative decoding."""

    @property
    def vocab_size(self) -> int:
        ...

    def next_token_distribution(self, prefix: Sequence[int]) -> Distribution:
        """Return P(next_token | prefix)."""

    def batch_next_token_distributions(
        self, prefix: Sequence[int], continuation: Sequence[int]
    ) -> list[Distribution]:
        """Return distributions for prefix, then each progressive continuation.

        For continuation [c1, c2], this returns:
        P(. | prefix), P(. | prefix c1), P(. | prefix c1 c2).
        """


@dataclass(frozen=True)
class BigramModel:
    """A tiny Markov language model for exactness tests.

    The next-token distribution is selected by the previous token id. Prefixes
    must contain at least one token so the model has a state.
    """

    transition_rows: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.transition_rows:
            raise ValueError("transition table cannot be empty")
        width = len(self.transition_rows[0])
        if width == 0:
            raise ValueError("transition rows cannot be empty")
        if any(len(row) != width for row in self.transition_rows):
            raise ValueError("all transition rows must have the same width")
        if len(self.transition_rows) != width:
            raise ValueError("bigram transition table must be square")
        for row in self.transition_rows:
            normalize(row)

    @property
    def vocab_size(self) -> int:
        return len(self.transition_rows)

    def next_token_distribution(self, prefix: Sequence[int]) -> Distribution:
        if not prefix:
            raise ValueError("BigramModel requires a non-empty prefix")
        state = int(prefix[-1])
        if state < 0 or state >= self.vocab_size:
            raise ValueError(f"token {state} is outside vocabulary")
        return normalize(self.transition_rows[state])

    def batch_next_token_distributions(
        self, prefix: Sequence[int], continuation: Sequence[int]
    ) -> list[Distribution]:
        seq = list(prefix)
        out: list[Distribution] = []
        out.append(self.next_token_distribution(seq))
        for token in continuation:
            seq.append(int(token))
            out.append(self.next_token_distribution(seq))
        return out


def default_toy_pair() -> tuple[BigramModel, BigramModel]:
    """Return target and draft models with high but imperfect agreement."""

    target = BigramModel(
        (
            (0.05, 0.65, 0.20, 0.10),
            (0.10, 0.15, 0.65, 0.10),
            (0.15, 0.05, 0.15, 0.65),
            (0.70, 0.10, 0.10, 0.10),
        )
    )
    draft = BigramModel(
        (
            (0.08, 0.58, 0.24, 0.10),
            (0.12, 0.18, 0.58, 0.12),
            (0.16, 0.08, 0.18, 0.58),
            (0.62, 0.13, 0.13, 0.12),
        )
    )
    return target, draft
