"""Benchmark harness for autoregressive vs speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Sequence

from .adaptive import AdaptiveSpeculationController
from .core import SpeculativeDecoder, autoregressive_generate
from .models import LanguageModel


@dataclass(frozen=True)
class BenchmarkRun:
    name: str
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    mean_acceptance_rate: float | None
    speedup_vs_baseline: float | None
    stats: dict[str, object]


def run_depth_sweep(
    *,
    target: LanguageModel,
    draft: LanguageModel,
    prefix: Sequence[int],
    max_new_tokens: int,
    depths: Sequence[int],
    seed: int = 0,
    repeats: int = 1,
) -> list[BenchmarkRun]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    baseline_tps: list[float] = []
    baseline_stats: list[dict[str, object]] = []
    for idx in range(repeats):
        _, stats = autoregressive_generate(
            target, prefix, max_new_tokens=max_new_tokens, rng=Random(seed + idx)
        )
        baseline_tps.append(stats.tokens_per_second)
        baseline_stats.append(stats.as_dict())

    baseline = BenchmarkRun(
        name="autoregressive",
        generated_tokens=max_new_tokens,
        elapsed_seconds=mean(float(s["elapsed_seconds"]) for s in baseline_stats),
        tokens_per_second=mean(baseline_tps),
        mean_acceptance_rate=None,
        speedup_vs_baseline=None,
        stats={
            "repeats": repeats,
            "runs": baseline_stats,
        },
    )

    runs = [baseline]
    baseline_rate = baseline.tokens_per_second
    for depth in depths:
        tps: list[float] = []
        acceptance: list[float] = []
        stats_dicts: list[dict[str, object]] = []
        for idx in range(repeats):
            decoder = SpeculativeDecoder(
                target=target, draft=draft, rng=Random(seed + 10_000 + idx)
            )
            _, stats = decoder.generate(prefix, max_new_tokens=max_new_tokens, depth=depth)
            tps.append(stats.tokens_per_second)
            acceptance.append(stats.mean_acceptance_rate)
            stats_dicts.append(stats.as_dict())
        mean_tps = mean(tps)
        runs.append(
            BenchmarkRun(
                name=f"speculative_depth_{depth}",
                generated_tokens=max_new_tokens,
                elapsed_seconds=mean(float(s["elapsed_seconds"]) for s in stats_dicts),
                tokens_per_second=mean_tps,
                mean_acceptance_rate=mean(acceptance),
                speedup_vs_baseline=mean_tps / baseline_rate if baseline_rate > 0 else None,
                stats={"repeats": repeats, "runs": stats_dicts},
            )
        )
    return runs


def run_adaptive(
    *,
    target: LanguageModel,
    draft: LanguageModel,
    prefix: Sequence[int],
    max_new_tokens: int,
    seed: int = 0,
    initial_depth: int = 4,
    max_depth: int = 12,
) -> BenchmarkRun:
    controller = AdaptiveSpeculationController(
        initial_depth=initial_depth, max_depth=max_depth
    )
    decoder = SpeculativeDecoder(target=target, draft=draft, rng=Random(seed), controller=controller)
    _, stats = decoder.generate(prefix, max_new_tokens=max_new_tokens, depth=initial_depth)
    return BenchmarkRun(
        name="speculative_adaptive",
        generated_tokens=max_new_tokens,
        elapsed_seconds=stats.elapsed_seconds,
        tokens_per_second=stats.tokens_per_second,
        mean_acceptance_rate=stats.mean_acceptance_rate,
        speedup_vs_baseline=None,
        stats=stats.as_dict(),
    )
