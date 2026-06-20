// @lat: [[lat#Paged KV Types]]
/// Paged KV cache read kernel for gathering K and V data from paged storage
/// into contiguous output buffers.
///
/// Each thread processes one (token, dimension) pair using a strided loop.
/// For each cached token, looks up its physical page via block_table, then
/// copies the K and V rows into contiguous output buffers.
///
/// Per-page layout: [K tokens | V tokens]
/// - K offset: `physical_page * page_stride + token_in_page * kv_dim + dim`
/// - V offset: `physical_page * page_stride + page_size * kv_dim + token_in_page * kv_dim + dim`
///
/// # Launch configuration
/// * grid: `(num_cached_tokens * kv_dim + INFERS_BLOCK_SIZE - 1) / INFERS_BLOCK_SIZE`
/// * block: `INFERS_BLOCK_SIZE`
/// * shared_mem: 0
///
/// # Arguments
/// * page_pool — Flat GPU buffer containing all paged KV data (K+V interleaved per page)
/// * block_table — Maps logical page index → physical page ID [num_pages]
/// * num_pages — Number of pages in block table
/// * num_cached_tokens — Number of cached tokens to gather
/// * head_dim — Dimension per head
/// * page_size — Number of tokens per page
/// * kv_dim — Total dimension (num_kv_heads × head_dim)
/// * k_out — Output buffer for K data [num_cached_tokens × kv_dim]
/// * v_out — Output buffer for V data [num_cached_tokens × kv_dim]

#include "common.cuh"

extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_paged_kv_read_bf16(
    const __nv_bfloat16* __restrict__ page_pool,
    const int* __restrict__ block_table,
    int num_pages,
    int num_cached_tokens,
    int head_dim,
    int page_size,
    int kv_dim,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out
) {
    int idx = INFERS_THREAD_IDX;
    int stride = blockDim.x * gridDim.x;
    int total_elements = num_cached_tokens * kv_dim;
    int page_stride = 2 * page_size * kv_dim;

    for (int i = idx; i < total_elements; i += stride) {
        int token_pos = i / kv_dim;
        int dim = i % kv_dim;

        int logical_page = token_pos / page_size;
        int token_in_page = token_pos % page_size;
        int physical_page = block_table[logical_page];

        // Read K (first half of page)
        int k_offset = physical_page * page_stride + token_in_page * kv_dim + dim;
        k_out[token_pos * kv_dim + dim] = page_pool[k_offset];

        // Read V (second half of page, offset by page_size * kv_dim)
        int v_offset = physical_page * page_stride + page_size * kv_dim + token_in_page * kv_dim + dim;
        v_out[token_pos * kv_dim + dim] = page_pool[v_offset];
    }
}

} // extern "C"
