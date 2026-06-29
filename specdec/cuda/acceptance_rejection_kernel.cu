#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" __global__ void acceptance_rejection_kernel(
    const float* __restrict__ target_probs,
    const float* __restrict__ draft_probs,
    const int*   __restrict__ draft_tokens,
    float*       __restrict__ accept_probs,
    int*         __restrict__ accepted,
    float*       __restrict__ corrected_probs,
    const int vocab_size,
    const int speculation_depth,
    unsigned long long seed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= speculation_depth) return;

    int token = draft_tokens[i];
    float p_target = target_probs[i * vocab_size + token];
    float p_draft  = draft_probs[i * vocab_size + token];
    float accept_p = fminf(1.0f, p_target / (p_draft + 1e-8f));
    accept_probs[i] = accept_p;

    curandState state;
    curand_init(seed + i, 0, 0, &state);
    float u = curand_uniform(&state);
    accepted[i] = (u < accept_p) ? 1 : 0;

    if (!accepted[i]) {
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float c = fmaxf(0.0f, target_probs[i * vocab_size + v]
                                - draft_probs[i * vocab_size + v]);
            corrected_probs[i * vocab_size + v] = c;
            sum += c;
        }
        for (int v = 0; v < vocab_size; v++) {
            corrected_probs[i * vocab_size + v] /= (sum + 1e-8f);
        }
    }
}

extern "C" int run_acceptance_rejection(
    const float* target_probs,
    const float* draft_probs,
    const int* draft_tokens,
    float* accept_probs,
    int* accepted,
    float* corrected_probs,
    int vocab_size,
    int speculation_depth,
    unsigned long long seed,
    int threads_per_block
) {
    if (speculation_depth <= 0 || vocab_size <= 0) {
        return 1;
    }
    if (threads_per_block <= 0) {
        threads_per_block = 128;
    }

    const int blocks = (speculation_depth + threads_per_block - 1) / threads_per_block;
    acceptance_rejection_kernel<<<blocks, threads_per_block>>>(
        target_probs,
        draft_probs,
        draft_tokens,
        accept_probs,
        accepted,
        corrected_probs,
        vocab_size,
        speculation_depth,
        seed
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C" int run_acceptance_rejection_sync(
    const float* target_probs,
    const float* draft_probs,
    const int* draft_tokens,
    float* accept_probs,
    int* accepted,
    float* corrected_probs,
    int vocab_size,
    int speculation_depth,
    unsigned long long seed,
    int threads_per_block
) {
    int launch_status = run_acceptance_rejection(
        target_probs,
        draft_probs,
        draft_tokens,
        accept_probs,
        accepted,
        corrected_probs,
        vocab_size,
        speculation_depth,
        seed,
        threads_per_block
    );
    if (launch_status != 0) {
        return launch_status;
    }
    return static_cast<int>(cudaDeviceSynchronize());
}
