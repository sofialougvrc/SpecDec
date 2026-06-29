# Speculative Decoding Engine

From-scratch implementation of exact speculative decoding after Yaniv Leviathan,
Matan Kalman, and Yossi Matias, ["Fast Inference from Transformers via
Speculative Decoding"](https://arxiv.org/abs/2211.17192), ICML 2023.

The implementation includes:

- Draft-model speculation for `n` future tokens.
- One parallel target-model verification pass over the speculated continuation.
- The exact acceptance-rejection correction:
  - accept proposed token `x` with probability `min(1, p(x) / q(x))`;
  - on rejection, sample from `normalize(max(p - q, 0))`;
  - if all draft tokens are accepted, sample one bonus token from the target model.
- Adaptive speculation depth based on observed acceptance rate.
- Benchmark harness for autoregressive baseline vs. fixed-depth and adaptive
  speculative decoding.
- Dependency-free toy models and tests for the statistical core.
- Optional Hugging Face/PyTorch adapter for real causal LMs such as GPT-2.

## Quick Start

The exact algorithm and tests run without third-party packages:

```bash
python3 -m unittest discover -s tests -v
python3 -m specdec toy --max-new-tokens 32 --depth 4 --seed 7
python3 -m specdec toy-benchmark --max-new-tokens 5000 --depths 1 2 4 8 --repeats 2
```

For real GPT-2 experiments:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,dev]'

python3 -m specdec hf-benchmark \
  --draft-model gpt2 \
  --target-model gpt2-medium \
  --prompt "Speculative decoding is" \
  --max-new-tokens 64 \
  --depths 1 2 4 6 8 \
  --device cpu
```

On a CUDA machine:

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

## Colab T4 CUDA Kernel Benchmark

For the acceptance-rejection CUDA microbenchmark, use the standalone NVCC plus
ctypes path instead of `torch.utils.cpp_extension.load_inline`. This avoids the
Colab failure mode where compilation succeeds but the inline extension import
fails at the generated `.so` step.

On Google Colab with a T4 runtime:

```bash
!nvidia-smi
!git clone <your-repo-url> SpecDec
%cd SpecDec
!pip install -e '.[hf,dev]'
!bash scripts/colab_cuda_kernel_setup.sh
```

Or run the benchmark directly:

```bash
!python scripts/benchmark_acceptance_kernel.py \
  --arch sm_75 \
  --depths 1 2 \
  --iters 2000 \
  --warmup 200 \
  > outputs/acceptance_kernel_t4.json
```

The Python callable is:

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

For a T4, `--arch sm_75` is the right target.

## Repository Layout

```text
specdec/
  core.py           exact speculative decoding loop
  distributions.py  categorical sampling and residual distribution math
  adaptive.py       speculation-depth controller
  models.py         language-model protocol and toy bigram model
  hf.py             optional Hugging Face causal-LM adapter
  benchmark.py      autoregressive vs speculative benchmark harness
  cli.py            command-line entrypoint
tests/
  test_*            dependency-free correctness tests
scripts/
  *.sh              repeatable benchmark/test helpers
```

## Algorithm Notes

Let `q_i` be the draft distribution and `p_i` the target distribution at
position `i`, conditioned on the prompt plus previously accepted draft tokens.
The draft samples candidates `x_1 ... x_gamma`. The target then scores every
candidate position plus the final bonus position in one forward pass.

For each candidate `x_i`, the sampler accepts it with probability:

```text
alpha_i = min(1, p_i(x_i) / q_i(x_i))
```

If the candidate is rejected, the sampler emits one replacement token from:

```text
normalize(max(p_i - q_i, 0))
```

and discards the rest of the draft. If all candidates are accepted, the sampler
emits one additional token from the final target distribution `p_{gamma+1}`.
This is the nontrivial part that preserves the target model's distribution.

## Measurement

The benchmark output reports:

- wall-clock elapsed seconds;
- tokens per second;
- speedup vs. autoregressive decoding;
- mean acceptance rate;
- target and draft call counts;
- proposed, accepted, rejected, and emitted token counts;
- depth and accepted-per-step distributions.

### T4 Acceptance-Rejection Kernel Result

The Colab T4 CUDA microbenchmark result is stored in
[`results/acceptance_kernel_t4.json`](results/acceptance_kernel_t4.json).

| GPU | Depth | Kernel latency | PyTorch loop latency | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Tesla T4 | 1 | 12.41 us | 230.64 us | 18.58x |
| Tesla T4 | 2 | 7629.21 us | 253.15 us | 0.03x |

The depth-1 result is strong and CV-worthy as a targeted CUDA microkernel
optimization: the custom launcher beats the pure PyTorch acceptance-rejection
loop by about 18.6x for GPT-2 vocabulary size. The depth-2 result exposes the
next optimization target. When a rejection occurs, the current kernel computes
the full corrected distribution in a serial per-position loop, so one CUDA
thread can end up scanning the entire 50,257-token vocabulary. A production
version should parallelize the residual `max(p - q, 0)` computation and
normalization across vocabulary lanes.

The dependency-free toy benchmark validates the harness but should not be read
as a real speed result. Python toy models have no transformer parallelism, so
speculation is often slower. The real speed experiment is the Hugging
Face/PyTorch path where the target model verifies `n + 1` positions in one model
call.

## Production Caveats

This project is a production-quality algorithmic prototype, not a model server.
It does not implement paged KV-cache management, batching across users, custom
CUDA kernels, or distributed serving. Those are the right next layers if this is
turned into an inference system. The current adapter already supports `--device
cuda` through PyTorch, so CUDA benchmarking is available when torch and model
weights are installed on a GPU machine.

For CPU-only M-series runs with GPT-2 small as draft and GPT-2 medium as target,
expect modest speedups at high acceptance rates and possible slowdowns at low
depths or low acceptance. That is consistent with the paper's hardware argument:
the algorithm benefits most when verifying several positions costs close to one
target-model step.

## References

- Leviathan, Kalman, Matias. ["Fast Inference from Transformers via Speculative
  Decoding"](https://arxiv.org/abs/2211.17192), 2023.
- Chen et al. ["Accelerating Large Language Model Decoding with Speculative
  Sampling"](https://arxiv.org/abs/2302.01318), 2023.
