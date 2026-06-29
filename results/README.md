# Benchmark Results

## Colab T4 Acceptance-Rejection Kernel

Source: `acceptance_kernel_t4.json`

| GPU | Depth | Kernel latency | PyTorch loop latency | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Tesla T4 | 1 | 12.41 us | 230.64 us | 18.58x |
| Tesla T4 | 2 | 7629.21 us | 253.15 us | 0.03x |

Interpretation: depth 1 is a strong microkernel result. Depth 2 reveals that
the current rejection correction path should be parallelized across the
vocabulary before claiming a general speedup for multiple speculative tokens.
