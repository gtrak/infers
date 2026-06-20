// @lat: [[arch#Kernel Extraction and Build System#Kernel Source Files]]
/// KV cache write kernel for writing K and V tensors into a paged KV cache.
///
/// Each thread writes one element of K and V at its assigned position in the
/// cache. Positions are looked up from the positions array, allowing scattered
/// writes into the cache layout.

#include "common.cuh"

/// KV cache write kernel (BF16).
///
/// Each thread processes one (token, dimension) pair using a strided loop.
/// Writes K to the first half of the cache and V to the second half.
///
/// # Launch configuration
/// * grid: `(seq_len * head_dim + INFERS_BLOCK_SIZE - 1) / INFERS_BLOCK_SIZE`
/// * block: `INFERS_BLOCK_SIZE`
extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_kv_cache_write_bf16(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ kv_cache,
    const int* __restrict__ positions,
    int seq_len,
    int head_dim,
    int max_seq_len
) {
    int idx = INFERS_THREAD_IDX;
    int stride = blockDim.x * gridDim.x;

    // total_elements = seq_len * head_dim
    int total_elements = seq_len * head_dim;

    for (int i = idx; i < total_elements; i += stride) {
        int token = i / head_dim;
        int dim = i % head_dim;
        int pos = positions[token];

        // Write K: kv_cache[pos * head_dim + dim] = k[token * head_dim + dim]
        kv_cache[pos * head_dim + dim] = k[token * head_dim + dim];

        // Write V: kv_cache[max_seq_len * head_dim + pos * head_dim + dim] = v[token * head_dim + dim]
        kv_cache[max_seq_len * head_dim + pos * head_dim + dim] = v[token * head_dim + dim];
    }
}

} // extern "C"
