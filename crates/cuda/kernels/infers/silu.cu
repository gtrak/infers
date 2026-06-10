// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// SiLU and SwiGLU activation kernels for BF16 tensors.
///
/// SiLU: output = x * sigmoid(x)
/// SwiGLU: output = x * sigmoid(gate) — used for MLP gating

#include "common.cuh"

/// SiLU activation kernel: output[i] = x[i] * sigmoid(x[i])
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void silu_kernel(
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
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void silu_glu_kernel(
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

extern "C" {

/// Launch SiLU activation for BF16 tensor.
///
/// # Arguments
/// * `x` — Input tensor
/// * `output` — Output tensor (can alias x for in-place)
/// * `total_elements` — Total number of elements
void infers_silu_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* output,
    int total_elements
) {
    int block_size = INFERS_BLOCK_SIZE;
    int grid_size = (total_elements + block_size - 1) / block_size;
    silu_kernel<<<grid_size, block_size>>>(x, output, total_elements);
}

/// Launch SwiGLU activation for BF16 tensors.
///
/// # Arguments
/// * `x` — Input tensor (the x branch of SwiGLU)
/// * `gate` — Gate tensor (the gate branch)
/// * `output` — Output tensor (same size as x)
/// * `total_elements` — Total number of elements in x and gate
void infers_silu_glu_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gate,
    __nv_bfloat16* output,
    int total_elements
) {
    int block_size = INFERS_BLOCK_SIZE;
    int grid_size = (total_elements + block_size - 1) / block_size;
    silu_glu_kernel<<<grid_size, block_size>>>(x, gate, output, total_elements);
}

} // extern "C"
