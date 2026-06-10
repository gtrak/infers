// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Argmax sampling kernel — greedy token selection from FP32 logits.
///
/// For each row of logits, finds the index of the maximum value.

#include "common.cuh"

/// Argmax kernel — each block finds the max index for one logit row.
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void argmax_kernel(
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

extern "C" {

/// Launch argmax sampling for FP32 logits.
///
/// # Arguments
/// * `logits` — Logit tensor [batch_size × vocab_size] in FP32
/// * `output` — Output token IDs [batch_size] as int32
/// * `batch_size` — Number of independent logit vectors
/// * `vocab_size` — Vocabulary size
void infers_argmax_f32(
    const float* logits,
    int* output,
    int batch_size,
    int vocab_size
) {
    int block_size = (vocab_size <= INFERS_BLOCK_SIZE) ? vocab_size : INFERS_BLOCK_SIZE;
    int shared_bytes = block_size * 2 * sizeof(float);

    argmax_kernel<<<batch_size, block_size, shared_bytes>>>(
        logits, output, batch_size, vocab_size
    );
}

} // extern "C"
