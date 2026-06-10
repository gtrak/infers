// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// RMSNorm kernel — Root Mean Square Layer Normalization for BF16 tensors.
///
/// Computes: output = x * rsqrt(mean(x^2) + eps) * weight
///
/// Grid: one block per row. Each block cooperatively computes the RMS
/// normalization for a single row using shared memory for reduction.

#include "common.cuh"

/// RMSNorm kernel.
///
/// Each block handles one row. Threads collaboratively compute sum of squares
/// via shared memory reduction, then broadcast the scaling factor.
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void rmsnorm_kernel(
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
        output[row * hidden + i] = __float2bfloat16(x_val * scale * w_val);
    }
}

extern "C" {

/// Launch RMSNorm for BF16 tensors.
///
/// # Arguments
/// * `x` — Input [rows × hidden] in BF16
/// * `weight` — Weights [hidden] in BF16
/// * `output` — Output [rows × hidden] in BF16
/// * `hidden` — Hidden dimension
/// * `rows` — Number of rows (batch × seq)
/// * `eps` — Epsilon (typically 1e-5)
void infers_rmsnorm_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int hidden,
    int rows,
    float eps
) {
    int block_size;
    if (hidden <= INFERS_BLOCK_SIZE) {
        block_size = hidden;
    } else {
        block_size = INFERS_BLOCK_SIZE;
    }

    int shared_bytes = block_size * sizeof(float);
    rmsnorm_kernel<<<rows, block_size, shared_bytes>>>(
        x, weight, output, hidden, eps
    );
}

} // extern "C"
