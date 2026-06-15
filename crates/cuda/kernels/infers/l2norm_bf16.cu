// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// L2 normalization kernel — normalizes each row of a BF16 tensor to unit length.
///
/// For each row: output[i] = input[i] / sqrt(sum(input^2) + eps)
/// The norm is computed in fp32 for precision, then the result is rounded to bf16.
///
/// This matches PyTorch's torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)
/// when applied in-place on bf16 tensors.

#include "common.cuh"

extern "C" {

/// L2-normalize each row of a BF16 tensor.
///
/// # Launch configuration
/// * grid: `rows` blocks (one block per row)
/// * block: `min(dim, INFERS_BLOCK_SIZE)` threads
/// * shared: `block_size * sizeof(float)` bytes (for sum-of-squares reduction)
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_l2norm_bf16(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int dim,
    float eps
) {
    extern __shared__ float shared_mem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // --- Phase 1: Compute sum of squares (accumulated in fp32) ---
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += total_threads) {
        float val = __bfloat162float(input[row * dim + i]);
        sum_sq += val * val;
    }

    // Reduce within block via shared memory
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

    // --- Phase 2: Normalize each element ---
    float norm = sqrtf(sdata[0] + eps);
    float rcp_norm = 1.0f / norm;

    for (int i = tid; i < dim; i += total_threads) {
        float val = __bfloat162float(input[row * dim + i]);
        output[row * dim + i] = __float2bfloat16(val * rcp_norm);
    }
}

} // extern "C"
