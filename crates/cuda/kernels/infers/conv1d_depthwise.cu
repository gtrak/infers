// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Depthwise 1D convolution with SiLU activation for GDN linear attention.
///
/// Operates on [B, D, T] channels-first layout (PyTorch conv1d convention).
/// Each thread handles one (b, d, t) output element, summing over kernel_size
/// positions with padding = kernel_size - 1 on both sides.
///
/// The conv output is truncated to the original seq_len (the extra padding
/// positions are discarded per the reference implementation).

#include "common.cuh"

extern "C" {

/// Depthwise 1D convolution with SiLU activation (BF16).
///
/// # Arguments
/// * input  — [B, T, D] input tensor (batch-first row-major, same as our data layout)
/// * weight — [D, K] depthwise convolution weights (one K-element filter per channel)
/// * output — [B, T, D] output tensor
///
/// # Launch configuration
/// * grid: `(B * T * D + INFERS_BLOCK_SIZE - 1) / INFERS_BLOCK_SIZE`
/// * block: `INFERS_BLOCK_SIZE`
/// * shared: none
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_conv1d_depthwise_silu_bf16(
    const __nv_bfloat16* __restrict__ input,   // [B, T, D]
    const __nv_bfloat16* __restrict__ weight,  // [D, K]
    __nv_bfloat16* __restrict__ output,        // [B, T, D]
    int batch_size,
    int conv_dim,
    int seq_len,
    int kernel_size
) {
    int total = batch_size * seq_len * conv_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose idx → (b, t, d) with d as innermost
    int d = idx % conv_dim;
    int t = (idx / conv_dim) % seq_len;
    int b = idx / (seq_len * conv_dim);

    int padding = kernel_size - 1;

    // Row-major [B, T, D] → position (b, t, d) = b * T * D + t * D + d
    int base = b * seq_len * conv_dim + t * conv_dim;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int pos = t + k - padding;
        float inp_val = 0.0f;
        if (pos >= 0 && pos < seq_len) {
            inp_val = __bfloat162float(input[b * seq_len * conv_dim + pos * conv_dim + d]);
        }
        float w_val = __bfloat162float(weight[d * kernel_size + k]);
        sum += inp_val * w_val;
    }

    // SiLU activation: x * sigmoid(x)
    float silu = isfinite(sum) ? sum / (1.0f + expf(-sum)) : 0.0f;

    output[base + d] = __float2bfloat16(silu);
}

} // extern "C"
