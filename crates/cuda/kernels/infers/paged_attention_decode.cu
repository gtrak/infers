// @lat: [[lat#Phase 4.6 Deliverables]]
/// Paged attention decode kernel for single-token inference.
///
/// Computes attention for one query token against all cached K/V tokens
/// stored in paged KV cache. One block per KV head, head_dim threads per
/// block.
///
/// Two-pass approach:
/// Phase 1: each thread processes a strided subset of tokens, computing
///          dot products and tracking per-thread online softmax stats.
///          Block reduction yields global normalization values (max, sum).
/// Phase 2: each thread (tid < head_dim) independently loops over ALL
///          tokens, recomputing dot products and accumulating weighted V
///          for its single assigned output dimension.
///
/// Shared memory layout (in `sdata`):
///   [0 .. bdim): Q values for the current head (loaded once per block)
///   [bdim .. 2*bdim): scratch for max reduction
///   [2*bdim .. 3*bdim): scratch for sum reduction
///
/// Launch configuration:
///   grid: `num_kv_heads` (one block per KV head)
///   block: `min(head_dim, INFERS_BLOCK_SIZE)` threads
///   shared: `3 * bdim * sizeof(float)` bytes
///
/// For large context lengths (e.g., 262K tokens), this kernel should be
/// optimized with multi-block per head or FlashAttention-style tiling
/// in a future phase.

#include "common.cuh"

extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_paged_attention_decode_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ page_pool,
    const int* __restrict__ block_table,
    int num_pages,
    int num_cached_tokens,
    int head_dim,
    int num_kv_heads,
    int page_size,
    int kv_dim,
    __nv_bfloat16* __restrict__ output
) {
    int head_idx = blockIdx.x;
    if (head_idx >= num_kv_heads) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int page_stride = 2 * page_size * kv_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    extern __shared__ float sdata[];

    // ================================================================
    // Load Q_h into shared memory (one copy per block, not per thread)
    // ================================================================
    if (tid < head_dim) {
        sdata[tid] = bf16_to_float(q[head_idx * head_dim + tid]);
    }
    __syncthreads();

    // ================================================================
    // Phase 1: Compute attention scores with per-thread online softmax
    // ================================================================
    // Each thread processes tokens [tid, tid+bdim, tid+2*bdim, ...]
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (int token_pos = tid; token_pos < num_cached_tokens; token_pos += bdim) {
        int logical_page = token_pos / page_size;
        int token_in_page = token_pos % page_size;
        int physical_page = block_table[logical_page];

        // Compute Q_h . K_h[token_pos] using shared Q and global K
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_v = sdata[d];
            int k_off = physical_page * page_stride
                       + token_in_page * kv_dim
                       + head_idx * head_dim + d;
            float k_v = bf16_to_float(page_pool[k_off]);
            dot += q_v * k_v;
        }
        dot *= scale;

        // Online softmax: update per-thread max and sum
        float new_max = fmaxf(local_max, dot);
        float correction = expf(local_max - new_max);
        local_sum = local_sum * correction + expf(dot - new_max);
        local_max = new_max;
    }

    // --- Block reduction: global max ---
    sdata[bdim + tid] = local_max;
    __syncthreads();

    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[bdim + tid] = fmaxf(sdata[bdim + tid], sdata[bdim + tid + s]);
        }
        __syncthreads();
    }
    float global_max = sdata[bdim];

    // --- Adjust per-thread sums to global max, then reduce ---
    float adjusted_sum = local_sum * expf(local_max - global_max);
    sdata[2 * bdim + tid] = adjusted_sum;
    __syncthreads();

    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[2 * bdim + tid] += sdata[2 * bdim + tid + s];
        }
        __syncthreads();
    }
    float global_sum = sdata[2 * bdim];

    // ================================================================
    // Phase 2: Compute weighted V accumulation
    // ================================================================
    // Each thread (tid < head_dim) independently loops over ALL tokens,
    // computing the full dot product and accumulating for its assigned
    // dimension. Dot-product computation is redundant across threads in
    // the block — a future optimization may use cooperative computation.
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    if (tid < head_dim) {
        float out_val = 0.0f;
        for (int token_pos = 0; token_pos < num_cached_tokens; token_pos++) {
            int logical_page = token_pos / page_size;
            int token_in_page = token_pos % page_size;
            int physical_page = block_table[logical_page];

            // Compute Q_h . K_h[token_pos] for this dimension's dot product
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float q_v = sdata[d];
                int k_off = physical_page * page_stride
                           + token_in_page * kv_dim
                           + head_idx * head_dim + d;
                float k_v = bf16_to_float(page_pool[k_off]);
                dot += q_v * k_v;
            }
            dot *= scale;

            // Softmax weight * V[tid]
            float weight = expf(dot - global_max) * inv_sum;
            float v_val = bf16_to_float(
                page_pool[physical_page * page_stride
                         + page_size * kv_dim
                         + token_in_page * kv_dim
                         + head_idx * head_dim + tid]
            );
            out_val += weight * v_val;
        }
        output[head_idx * head_dim + tid] = float_to_bf16(out_val);
    }
}

} // extern "C"
