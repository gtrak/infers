// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// SiLU and SwiGLU activation kernels for BF16 tensors.
///
/// SiLU: output = x * sigmoid(x)
/// SwiGLU: output = x * sigmoid(gate) — used for MLP gating

#include "common.cuh"

/// SiLU activation kernel: output[i] = x[i] * sigmoid(x[i])
///
/// # Launch configuration
/// * grid: `(total_elements + block_size - 1) / block_size`
/// * block: `INFERS_BLOCK_SIZE`
extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_silu_bf16(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ output,
    int total_elements
) {
    int idx = INFERS_THREAD_IDX;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        float val = __bfloat162float(x[i]);
        float sig = 1.0f / (1.0f + expf(-val));
        output[i] = __float2bfloat16(val * sig);
    }
}

/// SwiGLU kernel: output[i] = x[i] * sigmoid(gate[i])
///
/// # Launch configuration
/// * grid: `(total_elements + block_size - 1) / block_size`
/// * block: `INFERS_BLOCK_SIZE`
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_silu_glu_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output,
    int total_elements
) {
    int idx = INFERS_THREAD_IDX;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        float x_val = __bfloat162float(x[i]);
        float g_val = __bfloat162float(gate[i]);
        float sig = 1.0f / (1.0f + expf(-g_val));
        output[i] = __float2bfloat16(x_val * sig);
    }
}

} // extern "C"
