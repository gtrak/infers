// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Rotary Position Embedding (RoPE) kernel for BF16 tensors.
///
/// Applies standard RoPE to query and key tensors. Operates on
/// tensors of shape [seq_len, num_heads, head_dim] (or batched).
///
/// RoPE rotates pairs of dimensions by position-dependent angles
/// using precomputed sin/cos values.

#include "common.cuh"

/// RoPE kernel applying rotary position embeddings.
///
/// For each position `pos`, head `h`, and dimension pair `(2k, 2k+1)`,
/// rotates the pair by angle `pos * freq[k]` using precomputed sin/cos.
/// Operates on tensors of shape [total_tokens × num_heads × head_dim].
///
/// # Launch configuration
/// * grid: `(total_tokens * num_heads * head_dim/2 + INFERS_BLOCK_SIZE - 1) / INFERS_BLOCK_SIZE`
/// * block: `INFERS_BLOCK_SIZE`
extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_rope_bf16(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k_tensor,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    const int* __restrict__ positions,
    int total_tokens,
    int num_heads,
    int head_dim
) {
    int idx = INFERS_THREAD_IDX;
    int stride = blockDim.x * gridDim.x;
    int half_dim = head_dim / 2;
    int pairs_per_token = num_heads * half_dim;

    for (int t = idx; t < total_tokens * pairs_per_token; t += stride) {
        int token_idx = t / pairs_per_token;
        int remainder = t % pairs_per_token;
        int head_idx = remainder / half_dim;
        int dim_pair = remainder % half_dim;
        int pos = positions[token_idx];

        int cos_idx = pos * half_dim + dim_pair;
        float cos_val = cos[cos_idx];
        float sin_val = sin[cos_idx];

        // Index into [total_tokens × num_heads × head_dim]
        int i0 = token_idx * num_heads * head_dim + head_idx * head_dim + dim_pair * 2;
        int i1 = i0 + 1;

        float q0 = __bfloat162float(q[i0]);
        float q1 = __bfloat162float(q[i1]);
        float k0 = __bfloat162float(k_tensor[i0]);
        float k1 = __bfloat162float(k_tensor[i1]);

        // Apply rotation: [x0*cos - x1*sin, x0*sin + x1*cos]
        q[i0] = __float2bfloat16(q0 * cos_val - q1 * sin_val);
        q[i1] = __float2bfloat16(q0 * sin_val + q1 * cos_val);
        k_tensor[i0] = __float2bfloat16(k0 * cos_val - k1 * sin_val);
        k_tensor[i1] = __float2bfloat16(k0 * sin_val + k1 * cos_val);
    }
}

} // extern "C"
