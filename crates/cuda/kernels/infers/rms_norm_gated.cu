// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// RMSNorm with SiLU-gated output (Qwen3_5RMSNormGated).
///
/// Computes:
///   var = mean(x^2) + eps
///   x_norm = x / sqrt(var)
///   x_norm = weight * x_norm
///   result = x_norm * SiLU(gate)
///
/// where x is the last dimension of the input (head_v_dim per head) and
/// gate has the same shape.
///
/// All computation in float32 for numerical stability.

#include "common.cuh"

extern "C" {

/// RMSNorm with SiLU gating (BF16 I/O, float32 internal).
///
/// # Arguments
/// * input — [N, D] BF16 input
/// * gate  — [N, D] BF16 gate tensor
/// * weight — [D] BF16 learned weight
/// * output — [N, D] BF16 output
/// * N — number of rows
/// * D — hidden/feature dimension
/// * eps — epsilon for numerical stability
///
/// # Launch configuration
/// * grid: `(N, 1, 1)`
/// * block: `min(D, INFERS_BLOCK_SIZE)`
/// * shared: `blockDim.x * sizeof(float)` for variance reduction
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_rms_norm_gated_bf16(
    const __nv_bfloat16* __restrict__ input,   // [N, D]
    const __nv_bfloat16* __restrict__ gate,    // [N, D]
    const __nv_bfloat16* __restrict__ weight,  // [D]
    __nv_bfloat16* __restrict__ output,        // [N, D]
    int N,
    int D,
    float eps
) {
    extern __shared__ float shared_reduce[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    if (row >= N) return;

    // Phase 1: Compute sum(x^2)
    float sum_sq = 0.0f;
    for (int j = tid; j < D; j += total_threads) {
        float x = __bfloat162float(input[row * D + j]);
        sum_sq += x * x;
    }

    shared_reduce[tid] = sum_sq;
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            shared_reduce[tid] += shared_reduce[tid + stride];
        }
    }
    __syncthreads();

    float var = shared_reduce[0] / (float)D + eps;
    float rsqrt_var = rsqrtf(var);

    // Phase 2: Normalize, apply weight and gate
    for (int j = tid; j < D; j += total_threads) {
        float x = __bfloat162float(input[row * D + j]);
        float g = __bfloat162float(gate[row * D + j]);
        float w = __bfloat162float(weight[j]);
        float x_norm = x * rsqrt_var;
        // Qwen3_5RMSNormGated: weight is one-initialized, used directly (no additive offset)
        float w_scale = w;
        float gated = w_scale * x_norm * (g / (1.0f + expf(-g)));  // SiLU(gate) = gate * sigmoid(gate)
        output[row * D + j] = __float2bfloat16(gated);
    }
}

} // extern "C"
