#!/usr/bin/env python3
"""Benchmark CUDA acceptance-rejection kernel vs a pure PyTorch loop.

Run on Colab / RunPod:

    python scripts/benchmark_acceptance_kernel.py --depths 1 2 --iters 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from specdec.cuda_extension import AcceptanceRejectionCuda


def make_inputs(torch, *, depth: int, vocab_size: int, device: str, force_reject: bool):
    target = torch.rand((depth, vocab_size), device=device, dtype=torch.float32)
    draft = torch.rand((depth, vocab_size), device=device, dtype=torch.float32)
    target = target / target.sum(dim=-1, keepdim=True)
    draft = draft / draft.sum(dim=-1, keepdim=True)

    draft_tokens = torch.multinomial(draft, num_samples=1).squeeze(1).to(torch.int32)
    if force_reject:
        rows = torch.arange(depth, device=device)
        target[rows, draft_tokens.long()] = 0.0
        target = target / target.sum(dim=-1, keepdim=True)

    return target.contiguous(), draft.contiguous(), draft_tokens.contiguous()


def pytorch_acceptance_rejection(torch, target, draft, draft_tokens, *, seed: int):
    depth, vocab_size = target.shape
    rows = torch.arange(depth, device=target.device)
    accept_probs = torch.minimum(
        torch.ones(depth, device=target.device, dtype=torch.float32),
        target[rows, draft_tokens.long()] / (draft[rows, draft_tokens.long()] + 1e-8),
    )
    generator = torch.Generator(device=target.device)
    generator.manual_seed(seed)
    accepted = (torch.rand(depth, device=target.device, generator=generator) < accept_probs).to(
        torch.int32
    )
    corrected = torch.zeros((depth, vocab_size), device=target.device, dtype=torch.float32)
    for i in range(depth):
        if int(accepted[i].item()) == 0:
            residual = torch.clamp(target[i] - draft[i], min=0.0)
            corrected[i] = residual / (residual.sum() + 1e-8)
    return accept_probs, accepted, corrected


def time_cuda_events(torch, fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--arch", default=None, help="Example: sm_75 for Colab T4")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--force-reject",
        action="store_true",
        help="Force rejection so corrected_probs work is included every iteration.",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    runner = AcceptanceRejectionCuda.build(arch=args.arch, force=args.force_rebuild)
    device = "cuda"
    results = []

    for depth in args.depths:
        target, draft, draft_tokens = make_inputs(
            torch,
            depth=depth,
            vocab_size=args.vocab_size,
            device=device,
            force_reject=args.force_reject,
        )
        accept_probs = torch.empty((depth,), device=device, dtype=torch.float32)
        accepted = torch.empty((depth,), device=device, dtype=torch.int32)
        corrected = torch.empty((depth, args.vocab_size), device=device, dtype=torch.float32)

        runner(
            target,
            draft,
            draft_tokens,
            accept_probs,
            accepted,
            corrected,
            seed=123,
            threads_per_block=args.threads,
            sync=True,
        )

        kernel_us = time_cuda_events(
            torch,
            lambda: runner(
                target,
                draft,
                draft_tokens,
                accept_probs,
                accepted,
                corrected,
                seed=123,
                threads_per_block=args.threads,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        torch_us = time_cuda_events(
            torch,
            lambda: pytorch_acceptance_rejection(torch, target, draft, draft_tokens, seed=123),
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append(
            {
                "gpu": torch.cuda.get_device_name(0),
                "depth": depth,
                "vocab_size": args.vocab_size,
                "threads_per_block": args.threads,
                "force_reject": args.force_reject,
                "kernel_latency_us": kernel_us,
                "pytorch_latency_us": torch_us,
                "speedup_vs_pytorch": torch_us / kernel_us if kernel_us > 0 else None,
            }
        )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
