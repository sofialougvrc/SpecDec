"""Command line interface for demos and benchmarks."""

from __future__ import annotations

import argparse
import json
from random import Random
from typing import Any

from .adaptive import AdaptiveSpeculationController
from .benchmark import run_adaptive, run_depth_sweep
from .core import SpeculativeDecoder, autoregressive_generate
from .hf import HuggingFaceCausalLM
from .models import default_toy_pair


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _histogram(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _compact_stats(stats: dict[str, object]) -> dict[str, object]:
    depth_history = list(stats.get("depth_history", []))
    accepted_per_step = list(stats.get("accepted_per_step", []))
    return {
        key: value
        for key, value in stats.items()
        if key not in {"depth_history", "accepted_per_step"}
    } | {
        "depth_distribution": _histogram([int(v) for v in depth_history]),
        "accepted_per_step_distribution": _histogram([int(v) for v in accepted_per_step]),
    }


def _compact_run(run: Any) -> dict[str, object]:
    stats = run.stats
    if isinstance(stats, dict) and "runs" in stats:
        compact_runs = [_compact_stats(dict(item)) for item in stats["runs"]]
        stats = {"repeats": stats.get("repeats", len(compact_runs)), "runs": compact_runs}
    elif isinstance(stats, dict):
        stats = _compact_stats(stats)
    return {
        "name": run.name,
        "generated_tokens": run.generated_tokens,
        "elapsed_seconds": run.elapsed_seconds,
        "tokens_per_second": run.tokens_per_second,
        "mean_acceptance_rate": run.mean_acceptance_rate,
        "speedup_vs_baseline": run.speedup_vs_baseline,
        "stats": stats,
    }


def toy_demo(args: argparse.Namespace) -> None:
    target, draft = default_toy_pair()
    prefix = [args.start_token]
    if args.adaptive:
        controller = AdaptiveSpeculationController(
            initial_depth=args.depth,
            max_depth=args.max_depth,
            target_acceptance=args.target_acceptance,
        )
    else:
        controller = None
    decoder = SpeculativeDecoder(target=target, draft=draft, rng=Random(args.seed), controller=controller)
    output, stats = decoder.generate(prefix, max_new_tokens=args.max_new_tokens, depth=args.depth)
    _print_json({"tokens": output, "stats": stats.as_dict()})


def toy_benchmark(args: argparse.Namespace) -> None:
    target, draft = default_toy_pair()
    runs = run_depth_sweep(
        target=target,
        draft=draft,
        prefix=[args.start_token],
        max_new_tokens=args.max_new_tokens,
        depths=args.depths,
        seed=args.seed,
        repeats=args.repeats,
    )
    _print_json([_compact_run(run) for run in runs])


def hf_benchmark(args: argparse.Namespace) -> None:
    target = HuggingFaceCausalLM(args.target_model, device=args.device, dtype=args.dtype)
    draft = HuggingFaceCausalLM(args.draft_model, device=args.device, dtype=args.dtype)
    prefix = target.encode(args.prompt)
    if target.tokenizer.get_vocab() != draft.tokenizer.get_vocab():
        raise ValueError("target and draft tokenizers must have identical vocabularies")

    if args.adaptive:
        baseline, *_ = run_depth_sweep(
            target=target,
            draft=draft,
            prefix=prefix,
            max_new_tokens=args.max_new_tokens,
            depths=[],
            seed=args.seed,
            repeats=args.repeats,
        )
        adaptive = run_adaptive(
            target=target,
            draft=draft,
            prefix=prefix,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            initial_depth=args.initial_depth,
            max_depth=args.max_depth,
        )
        payload = [_compact_run(baseline), _compact_run(adaptive)]
        payload[1]["speedup_vs_baseline"] = (
            adaptive.tokens_per_second / baseline.tokens_per_second
            if baseline.tokens_per_second > 0
            else None
        )
        _print_json(payload)
    else:
        runs = run_depth_sweep(
            target=target,
            draft=draft,
            prefix=prefix,
            max_new_tokens=args.max_new_tokens,
            depths=args.depths,
            seed=args.seed,
            repeats=args.repeats,
        )
        _print_json([_compact_run(run) for run in runs])


def hf_generate(args: argparse.Namespace) -> None:
    target = HuggingFaceCausalLM(args.target_model, device=args.device, dtype=args.dtype)
    draft = HuggingFaceCausalLM(args.draft_model, device=args.device, dtype=args.dtype)
    prefix = target.encode(args.prompt)
    decoder = SpeculativeDecoder(target=target, draft=draft, rng=Random(args.seed))
    output, stats = decoder.generate(prefix, max_new_tokens=args.max_new_tokens, depth=args.depth)
    _print_json(
        {
            "text": target.decode(output),
            "tokens": output,
            "stats": stats.as_dict(),
        }
    )


def baseline_generate(args: argparse.Namespace) -> None:
    target = HuggingFaceCausalLM(args.target_model, device=args.device, dtype=args.dtype)
    output, stats = autoregressive_generate(
        target,
        target.encode(args.prompt),
        max_new_tokens=args.max_new_tokens,
        rng=Random(args.seed),
    )
    _print_json({"text": target.decode(output), "tokens": output, "stats": stats.as_dict()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="specdec")
    sub = parser.add_subparsers(dest="command", required=True)

    toy = sub.add_parser("toy", help="run the dependency-free toy demo")
    toy.add_argument("--max-new-tokens", type=int, default=32)
    toy.add_argument("--depth", type=int, default=4)
    toy.add_argument("--max-depth", type=int, default=12)
    toy.add_argument("--target-acceptance", type=float, default=0.72)
    toy.add_argument("--start-token", type=int, default=0)
    toy.add_argument("--seed", type=int, default=0)
    toy.add_argument("--adaptive", action="store_true")
    toy.set_defaults(func=toy_demo)

    toy_bench = sub.add_parser("toy-benchmark", help="run a local correctness-speed smoke benchmark")
    toy_bench.add_argument("--max-new-tokens", type=int, default=20_000)
    toy_bench.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    toy_bench.add_argument("--repeats", type=int, default=3)
    toy_bench.add_argument("--start-token", type=int, default=0)
    toy_bench.add_argument("--seed", type=int, default=0)
    toy_bench.set_defaults(func=toy_benchmark)

    hf_common = argparse.ArgumentParser(add_help=False)
    hf_common.add_argument("--target-model", default="gpt2-medium")
    hf_common.add_argument("--draft-model", default="gpt2")
    hf_common.add_argument("--prompt", default="Speculative decoding is")
    hf_common.add_argument("--max-new-tokens", type=int, default=64)
    hf_common.add_argument("--device", default="cpu")
    hf_common.add_argument("--dtype", default=None)
    hf_common.add_argument("--seed", type=int, default=0)

    hf_bench = sub.add_parser("hf-benchmark", parents=[hf_common], help="benchmark real HF causal LMs")
    hf_bench.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    hf_bench.add_argument("--repeats", type=int, default=1)
    hf_bench.add_argument("--adaptive", action="store_true")
    hf_bench.add_argument("--initial-depth", type=int, default=4)
    hf_bench.add_argument("--max-depth", type=int, default=12)
    hf_bench.set_defaults(func=hf_benchmark)

    hf_gen = sub.add_parser("hf-generate", parents=[hf_common], help="generate text using speculative decoding")
    hf_gen.add_argument("--depth", type=int, default=4)
    hf_gen.set_defaults(func=hf_generate)

    baseline = sub.add_parser("hf-baseline", parents=[hf_common], help="generate text autoregressively")
    baseline.set_defaults(func=baseline_generate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
