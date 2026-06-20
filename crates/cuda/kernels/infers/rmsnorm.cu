// @lat: [[arch#Kernel Extraction and Build System#Kernel Source Files]]
/// RMSNorm kernel — Root Mean Square Layer Normalization for BF16 tensors.
///
/// Qwen3_5 uses zero-initialized weights with additive offset: output = x * rsqrt(mean(x^2) + eps) * (1 + weight)
///
/// Grid: one block per row. Each block cooperatively computes the RMS
/// normalization for a single row using shared memory for reduction.

#include "common.cuh"

/// RMSNorm kernel — one block per row, shared-memory reduction.
///
/// Each block handles one row. Threads collaboratively compute sum of squares
/// via shared memory reduction, then broadcast the scaling factor.
///
/// # Launch configuration
/// * grid: `rows` blocks
/// * block: `min(hidden, INFERS_BLOCK_SIZE)` threads
/// * shared: `block_size * sizeof(float)` bytes
extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int hidden,
    float eps
) {
    extern __shared__ float shared_mem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // --- Phase 1: Compute sum of squares (accumulated in float for precision) ---
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += total_threads) {
        float val = __bfloat162float(x[row * hidden + i]);
        sum_sq += val * val;
    }

    // Reduce within block via shared memory — use float throughout for precision
    float* sdata = shared_mem;
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
    }
    __syncthreads();

    // --- Phase 2: Broadcast scaling factor and apply ---
    float rms = sdata[0];
    float scale = rsqrtf(rms / hidden + eps);

    for (int i = tid; i < hidden; i += total_threads) {
        float x_val = __bfloat162float(x[row * hidden + i]);
        float w_val = __bfloat162float(weight[i]);
        output[row * hidden + i] = __float2bfloat16(x_val * scale * (1.0f + w_val));
    }
}

} // extern "C"
