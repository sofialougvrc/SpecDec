# SpecDec

SpecDec is an ML systems project implementing **exact speculative decoding** for
transformer inference, with a custom CUDA microkernel for one of the algorithm's
hot paths: the acceptance-rejection verification step.

Based on Leviathan, Kalman, and Matias,
["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192),
ICML 2023. Speculative decoding accelerates autoregressive generation by using
a smaller draft model to propose multiple future tokens, then verifying those
tokens with the target model in parallel. The difficult part is not simply
generating draft tokens; it is applying the acceptance-rejection correction that
keeps the output distribution exactly equal to the target model's distribution.

SpecDec implements that full correction path, exposes it through a benchmarkable
Python package, and includes a standalone CUDA C++ kernel for the
acceptance-rejection verification step.

## Summary

| Area | What this project demonstrates |
| --- | --- |
| ML systems | Implemented a real transformer inference algorithm from a published ICML paper. |
| Probabilistic correctness | Preserved exact target-model sampling with acceptance-rejection correction. |
| GPU programming | Wrote and launched a CUDA C++ microkernel with NVCC, cuRAND, and `ctypes`. |
| Benchmarking | Measured latency against a PyTorch baseline on Google Colab Tesla T4. |
| Engineering quality | Packaged the project with tests, CLI entrypoints, reproducible scripts, and documented limitations. |

## Highlights

- **From-scratch speculative decoding engine.** Implements draft-token
  generation, target-model verification, token-level acceptance sampling,
  rejection correction, bonus-token sampling, and adaptive speculation depth.
- **Exact distribution preservation.** Uses the paper's correction
  `normalize(max(p - q, 0))` on rejection, rather than approximating with a
  target-model resample. This is the key statistical detail that prevents bias.
- **Hugging Face/PyTorch inference path.** Supports GPT-2-style draft/target
  experiments through a causal language model adapter and CLI benchmark harness.
- **CUDA microkernel path.** Adds a standalone CUDA C++ acceptance-rejection
  kernel compiled with NVCC and called from Python through `ctypes`, avoiding
  Colab `load_inline` shared-library import failures.
- **Measured GPU result.** Benchmarks the CUDA kernel on a Google Colab Tesla T4
  against a pure PyTorch acceptance-rejection loop at GPT-2 vocabulary size.
- **Correctness-first tests.** Includes dependency-free toy language models and
  unit tests so the sampling logic can be validated without downloading model
  weights.
- **Honest systems analysis.** Reports both the successful depth-1 kernel result
  and the depth-2 bottleneck, with the next optimization target clearly
  identified.

## What I Built

| Component | Purpose |
| --- | --- |
| Speculative decoder | Generates draft continuations, verifies them with the target model, and emits exact corrected samples. |
| Adaptive depth controller | Adjusts speculation depth based on observed acceptance rate. |
| Benchmark harness | Compares autoregressive decoding, fixed-depth speculation, adaptive speculation, and CUDA microkernel latency. |
| CUDA launcher | Builds a `.so` with NVCC and launches `<<<blocks, threads>>>` from Python without PyTorch extension import machinery. |
| Hugging Face adapter | Lets the same algorithm run against causal LMs such as GPT-2 small and GPT-2 medium. |
| Toy model testbed | Provides deterministic, dependency-free tests for algorithmic correctness. |

## Measured Result

Acceptance-rejection microbenchmark on **Google Colab Tesla T4**, GPT-2
vocabulary size `50,257`, `128` CUDA threads per block:

| Depth | CUDA kernel latency | PyTorch loop latency | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 12.41 us | 230.64 us | 18.58x |
| 2 | 7629.21 us | 253.15 us | 0.03x |

The depth-1 result demonstrates a strong targeted kernel win: the custom CUDA
launcher is **18.6x faster** than the equivalent PyTorch loop for the
acceptance-rejection correction. The depth-2 result is intentionally reported
instead of hidden: it exposes the next systems bottleneck. When rejection occurs,
the current kernel computes the corrected distribution with a serial
full-vocabulary scan inside one CUDA thread. A production version should
parallelize residual computation and normalization across vocabulary lanes.

Raw result: [`results/acceptance_kernel_t4.json`](results/acceptance_kernel_t4.json)

Experimental setup:

- GPU: Google Colab Tesla T4
- Kernel target: `sm_75`
- Vocabulary size: `50,257`
- Benchmark iterations: `2,000`
- Warmup iterations: `200`
- Timing method: CUDA events
- Baseline: pure PyTorch acceptance-rejection loop

## Why This Matters

Autoregressive transformer inference is latency-bound because each generated
token usually requires a target-model forward pass. Speculative decoding uses a
smaller draft model to propose multiple tokens, then verifies them with the
target model in parallel. Correct implementations can reduce target-model calls
without changing the sampled distribution.

For inference teams, this project touches the same concerns that show up in
production LLM serving: target-model latency, draft-model quality, acceptance
rate, GPU launch overhead, sampling correctness, and the boundary between
framework-level tensor code and custom kernels.

SpecDec is not a model server. It is a focused implementation of the algorithmic
and systems pieces needed to understand, measure, and optimize speculative
decoding.

## System Design

```text
Prompt tokens
    |
    v
Draft model proposes n tokens
    |
    v
Target model verifies all proposed positions in one parallel pass
    |
    v
Acceptance-rejection correction
    |
    +-- accept token with min(1, p(x) / q(x))
    |
    +-- on rejection, resample from normalize(max(p - q, 0))
    |
    v
Emit accepted tokens or corrected replacement
```

Main components:

- `specdec/core.py`: exact speculative decoding loop.
- `specdec/distributions.py`: categorical sampling and rejection correction.
- `specdec/adaptive.py`: adaptive speculation-depth controller.
- `specdec/hf.py`: Hugging Face causal-LM adapter.
- `specdec/cuda/acceptance_rejection_kernel.cu`: CUDA kernel and C launcher.
- `specdec/cuda_extension.py`: NVCC build helper and `ctypes` Python binding.
- `scripts/benchmark_acceptance_kernel.py`: CUDA kernel vs PyTorch benchmark.

## Engineering Decisions

- **Kept the mathematical core dependency-free.** The exact sampler can be
  tested with toy models using only the Python standard library. PyTorch and
  Transformers are optional for real model experiments.
- **Separated algorithm correctness from GPU optimization.** The speculative
  decoding loop is implemented in Python first, then the acceptance-rejection
  path is isolated for CUDA benchmarking.
- **Used a standalone NVCC build instead of `load_inline`.** Colab can compile
  inline CUDA extensions and still fail when importing the generated shared
  object. The project avoids that by compiling a normal `.so` and loading it
  through `ctypes`.
- **Reported negative results.** Depth 2 currently exposes a serial
  full-vocabulary correction bottleneck. That result is included because it
  points directly to the next CUDA optimization.
- **Kept benchmarks reproducible.** Scripts include fixed depth settings,
  warmup iterations, CUDA-event timing, and saved JSON outputs.

## Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- CUDA C++
- NVCC
- cuRAND
- `ctypes`
- CUDA events
- Python `unittest`
- Git/GitHub

## Quick Start

The core algorithm and tests do not require PyTorch:

```bash
python3 -m unittest discover -s tests -v
python3 -m specdec toy --max-new-tokens 32 --depth 4 --seed 7
python3 -m specdec toy-benchmark --max-new-tokens 5000 --depths 1 2 4 8 --repeats 2
```

For Hugging Face model experiments:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,dev]'
```

CPU benchmark:

```bash
python3 -m specdec hf-benchmark \
  --draft-model gpt2 \
  --target-model gpt2-medium \
  --prompt "Speculative decoding is" \
  --max-new-tokens 64 \
  --depths 1 2 4 6 8 \
  --device cpu
```

CUDA benchmark:

```bash
python3 -m specdec hf-benchmark \
  --draft-model gpt2 \
  --target-model gpt2-medium \
  --prompt "Speculative decoding is" \
  --max-new-tokens 128 \
  --depths 2 4 6 8 \
  --device cuda \
  --dtype float16
```

## CUDA Kernel Benchmark

The CUDA microbenchmark uses a standalone `.cu` file compiled with NVCC and
loaded with `ctypes`. This avoids the Colab failure mode where
`torch.utils.cpp_extension.load_inline` compiles successfully but fails during
the generated `.so` import step.

On Google Colab with a T4 runtime:

```bash
!nvidia-smi
!git clone https://github.com/sofialougvrc/SpecDec.git
%cd SpecDec
!pip install -e '.[hf,dev]'
!bash scripts/colab_cuda_kernel_setup.sh
```

Direct benchmark command:

```bash
!python scripts/benchmark_acceptance_kernel.py \
  --arch sm_75 \
  --depths 1 2 \
  --iters 2000 \
  --warmup 200 \
  > outputs/acceptance_kernel_t4.json
```

Python callable:

```python
from specdec.cuda_extension import AcceptanceRejectionCuda

runner = AcceptanceRejectionCuda.build(arch="sm_75", force=True)
runner(
    target_probs,       # CUDA float32 [depth, vocab_size]
    draft_probs,        # CUDA float32 [depth, vocab_size]
    draft_tokens,       # CUDA int32 [depth]
    accept_probs,       # CUDA float32 [depth]
    accepted,           # CUDA int32 [depth]
    corrected_probs,    # CUDA float32 [depth, vocab_size]
    seed=123,
    threads_per_block=128,
)
```

The launcher computes:

```text
blocks = ceil(speculation_depth / threads_per_block)
threads = threads_per_block
```

For Colab T4, use `--arch sm_75`.

## Algorithm Correctness

Let `q_i` be the draft distribution and `p_i` the target distribution at
position `i`. The draft model samples proposed tokens `x_1 ... x_n`. The target
model scores all proposed positions in one verification pass.

For each proposed token:

```text
accept with probability min(1, p_i(x_i) / q_i(x_i))
```

If the token is rejected, the sampler emits a replacement token from:

```text
normalize(max(p_i - q_i, 0))
```

If all proposed tokens are accepted, the sampler emits one additional token from
the final target distribution. This correction is the key step that preserves
the target model's distribution exactly.

## Repository Layout

```text
specdec/
  core.py                         speculative decoding loop
  distributions.py                probability utilities
  adaptive.py                     speculation-depth controller
  models.py                       model protocol and toy models
  hf.py                           Hugging Face causal-LM adapter
  cuda_extension.py               ctypes CUDA loader
  cuda/acceptance_rejection_kernel.cu
scripts/
  benchmark_acceptance_kernel.py  CUDA vs PyTorch microbenchmark
  colab_cuda_kernel_setup.sh      Colab T4 helper
  benchmark_gpt2_cuda.sh          GPT-2 CUDA benchmark helper
tests/
  test_*                          correctness tests
results/
  acceptance_kernel_t4.json       measured Colab T4 result
```

## Current Limitations

SpecDec is an ML systems prototype, not a full inference server. It does not yet
include paged KV-cache management, multi-user batching, distributed serving, or
parallelized residual normalization for depth greater than 1. The most important
next optimization is rewriting the rejection branch so the full-vocabulary
`max(p - q, 0)` correction is parallelized across CUDA threads.

## References

- Leviathan, Kalman, Matias. ["Fast Inference from Transformers via Speculative
  Decoding"](https://arxiv.org/abs/2211.17192), 2023.
- Chen et al. ["Accelerating Large Language Model Decoding with Speculative
  Sampling"](https://arxiv.org/abs/2302.01318), 2023.
