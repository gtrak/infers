// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Online softmax kernel for attention scores with optional causal masking.
///
/// Uses the numerically stable online softmax algorithm: find row-wise max
/// via parallel reduction, compute sum of exp(x - max), then normalize.
/// When causal masking is enabled, positions (i, j) with j > i are treated as -inf.

#include "common.cuh"

/// Online softmax kernel (BF16 scores, single row per block).
///
/// Each block processes one row of the seq_len × seq_len score matrix.
/// Three-phase parallel reduction using shared memory:
///   Phase 1 — find row maximum (for numerical stability)
///   Phase 2 — compute sum of exp(x - max)
///   Phase 3 — write normalized output: exp(x - max) / sum
///
/// # Launch configuration
/// * grid: `seq_len`
/// * block: next power of 2 up to `min(seq_len, INFERS_BLOCK_SIZE)`
/// * shared: `block_size * sizeof(float)`
extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_softmax_bf16(
    const __nv_bfloat16* __restrict__ scores,
    __nv_bfloat16* __restrict__ output,
    int seq_len,
    int use_causal
) {
    extern __shared__ float shared_mem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // --- Phase 1: Parallel reduction for row maximum ---
    float row_max = -1e38f; // very small initial value
    for (int col = tid; col < seq_len; col += total_threads) {
        float val = __bfloat162float(scores[row * seq_len + col]);
        if (use_causal && col > row) {
            // Causal mask: upper triangle is -inf
            continue;
        }
        if (val > row_max) row_max = val;
    }

    float* sdata = shared_mem;
    sdata[tid] = row_max;
    __syncthreads();

    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride && tid + stride < blockDim.x) {
            if (sdata[tid + stride] > sdata[tid]) {
                sdata[tid] = sdata[tid + stride];
            }
        }
    }
    __syncthreads();
    float max_val = sdata[0];

    // --- Phase 2: Parallel reduction for sum of exp(x - max) ---
    float row_sum = 0.0f;
    for (int col = tid; col < seq_len; col += total_threads) {
        float val = __bfloat162float(scores[row * seq_len + col]);
        if (use_causal && col > row) {
            // Causal mask: exp(-inf) = 0
            continue;
        }
        row_sum += expf(val - max_val);
    }

    sdata[tid] = row_sum;
    __syncthreads();

    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride && tid + stride < blockDim.x) {
            sdata[tid] += sdata[tid + stride];
        }
    }
    __syncthreads();

    // --- Phase 3: Write softmax output ---
    float sum = sdata[0];
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int col = tid; col < seq_len; col += total_threads) {
        if (use_causal && col > row) {
            // Causal mask: output is 0 for masked positions
            output[row * seq_len + col] = __float2bfloat16(0.0f);
        } else {
            float val = __bfloat162float(scores[row * seq_len + col]);
            output[row * seq_len + col] = __float2bfloat16(expf(val - max_val) * inv_sum);
        }
    }
}

} // extern "C"
