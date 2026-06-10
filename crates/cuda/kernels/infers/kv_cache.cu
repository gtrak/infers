// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
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
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void kv_cache_write_kernel(
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

extern "C" {

/// Launch KV cache write for BF16 tensors.
///
/// # Arguments
/// * `k` — Key tensor [seq_len × head_dim] row-major
/// * `v` — Value tensor [seq_len × head_dim] row-major
/// * `kv_cache` — KV cache [2 × max_seq_len × head_dim] row-major
///                    Layout: K layer [max_seq_len × head_dim] then V layer
/// * `positions` — Position IDs for each token [seq_len]
/// * `seq_len` — Number of tokens to write
/// * `head_dim` — Dimension per head
/// * `max_seq_len` — Maximum sequence length (for cache bounds)
void infers_kv_cache_write_bf16(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* kv_cache,
    const int* positions,
    int seq_len,
    int head_dim,
    int max_seq_len
) {
    int total_elements = seq_len * head_dim;
    int block_size = INFERS_BLOCK_SIZE;
    int grid_size = (total_elements + block_size - 1) / block_size;
    kv_cache_write_kernel<<<grid_size, block_size>>>(
        k, v, kv_cache, positions, seq_len, head_dim, max_seq_len
    );
}

} // extern "C"
