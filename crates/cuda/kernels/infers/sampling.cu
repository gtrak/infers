// @lat: [[arch#Kernel Extraction and Build System#Kernel Source Files]]
/// Argmax sampling kernel — greedy token selection from FP32 logits.
///
/// For each row of logits, finds the index of the maximum value.

#include "common.cuh"

/// Argmax kernel — each block finds the max index for one logit row.
///
/// # Launch configuration
/// * grid: `batch_size`
/// * block: `min(vocab_size, INFERS_BLOCK_SIZE)`
/// * shared: `block_size * 2 * sizeof(float)`
extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_argmax_f32(
    const float* __restrict__ logits,
    int* __restrict__ output,
    int batch_size,
    int vocab_size
) {
    extern __shared__ char shared_raw[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // Shared arrays: values and indices
    float* s_vals = (float*)shared_raw;
    float* s_idxs = s_vals + total_threads;

    // --- Phase 1: Each thread finds local max in its chunk ---
    float my_max = -1e38f;
    int my_idx = 0;
    const float* row_logits = logits + row * vocab_size;

    for (int i = tid; i < vocab_size; i += total_threads) {
        float val = row_logits[i];
        if (val > my_max) {
            my_max = val;
            my_idx = i;
        }
    }

    s_vals[tid] = my_max;
    s_idxs[tid] = (float)my_idx;
    __syncthreads();

    // --- Phase 2: Block-wide reduction ---
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
    }
    __syncthreads();

    // --- Phase 3: Thread 0 writes result ---
    if (tid == 0) {
        output[row] = (int)s_idxs[0];
    }
}

/// Argmax kernel for BF16 logits — greedy token selection without CPU round-trip.
///
/// Each block finds the max index for one logit row.
///
/// # Launch configuration
/// * grid: `batch_size`
/// * block: `min(vocab_size, INFERS_BLOCK_SIZE)`
/// * shared: `block_size * 2 * sizeof(float)`
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_argmax_bf16(
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ output,
    int batch_size,
    int vocab_size
) {
    extern __shared__ char shared_raw[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // Shared arrays: values and indices
    float* s_vals = (float*)shared_raw;
    float* s_idxs = s_vals + total_threads;

    // --- Phase 1: Each thread finds local max in its chunk ---
    float my_max = -1e38f;
    int my_idx = 0;
    const __nv_bfloat16* row_logits = logits + row * vocab_size;

    for (int i = tid; i < vocab_size; i += total_threads) {
        float val = __bfloat162float(row_logits[i]);
        if (val > my_max) {
            my_max = val;
            my_idx = i;
        }
    }

    s_vals[tid] = my_max;
    s_idxs[tid] = (float)my_idx;
    __syncthreads();

    // --- Phase 2: Block-wide reduction ---
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
    }
    __syncthreads();

    // --- Phase 3: Thread 0 writes result ---
    if (tid == 0) {
        output[row] = (int)s_idxs[0];
    }
}

} // extern "C"
