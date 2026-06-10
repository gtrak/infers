// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Element-wise operation kernels for BF16 tensors.
///
/// Simple element-wise operations used for residual connections and
/// other pointwise computations in the transformer forward pass.

#include "common.cuh"

/// Element-wise addition kernel: output[i] = a[i] + b[i]
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void add_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ output,
    int total_elements
) {
    int idx = INFERS_THREAD_IDX;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        float a_val = __bfloat162float(a[i]);
        float b_val = __bfloat162float(b[i]);
        output[i] = __float2bfloat16(a_val + b_val);
    }
}

extern "C" {

/// Launch element-wise addition for BF16 tensors.
///
/// # Arguments
/// * `a` — First input tensor
/// * `b` — Second input tensor
/// * `output` — Output tensor (same shape as inputs)
/// * `total_elements` — Total number of elements
void infers_add_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* output,
    int total_elements
) {
    int block_size = INFERS_BLOCK_SIZE;
    int grid_size = (total_elements + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size>>>(a, b, output, total_elements);
}

} // extern "C"
