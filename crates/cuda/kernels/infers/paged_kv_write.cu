// @lat: [[lat#Phase 4.6 Deliverables]]
/// Paged KV cache write kernel for writing K and V tensors into the paged KV cache.
///
/// Each thread processes one (token, dimension) pair using a strided loop.
/// Token positions are looked up from the block_table and positions arrays.
/// K and V are stored within each page: K first, then V.
///
/// Per-page layout: [K tokens | V tokens], where each side holds `page_size * kv_dim` elements.
/// page_stride = 2 * page_size * kv_dim (accounts for both K and V per page).

#include "common.cuh"

/// Paged KV cache write kernel (BF16).
///
/// Writes K and V data into a paged KV cache using a block table for
/// address translation. For each token, looks up which physical page it
/// belongs to, then writes the K and V data at the correct offset within
/// that page. This eliminates CPU round-trips during prefill.
///
/// Memory layout per page: [K tokens | V tokens]
/// - K offset: `physical_page * page_stride + token_in_page * kv_dim + dim`
/// - V offset: `physical_page * page_stride + page_size * kv_dim + token_in_page * kv_dim + dim`
///
/// # Launch configuration
/// * grid: `(seq_len * kv_dim + INFERS_BLOCK_SIZE - 1) / INFERS_BLOCK_SIZE`
/// * block: `INFERS_BLOCK_SIZE`
/// * shared_mem: 0
///
/// # Arguments
/// * k — K tensor [seq_len × kv_dim] in row-major BF16
/// * v — V tensor [seq_len × kv_dim] in row-major BF16
/// * page_pool — Flat GPU buffer containing all paged KV data (K+V interleaved per page)
/// * block_table — Maps logical page index → physical page ID [num_pages]
/// * positions — Token position IDs [seq_len], used for token-in-page calculation
/// * seq_len — Number of tokens to write
/// * head_dim — Dimension per head
/// * page_size — Number of tokens per page
/// * kv_dim — = num_kv_heads * head_dim
extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_paged_kv_write_bf16(
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ page_pool,
    const int* __restrict__ block_table,
    const int* __restrict__ positions,
    int seq_len,
    int head_dim,
    int page_size,
    int kv_dim
) {
    int idx = INFERS_THREAD_IDX;
    int stride = blockDim.x * gridDim.x;

    int total_elements = seq_len * kv_dim;
    int page_stride = 2 * page_size * kv_dim;  // K + V per page

    for (int i = idx; i < total_elements; i += stride) {
        int token = i / kv_dim;
        int dim = i % kv_dim;
        int pos = positions[token];

        // Determine which logical page and position within the page
        int logical_page = pos / page_size;
        int token_in_page = pos % page_size;

        // Look up physical page from block table
        int physical_page = block_table[logical_page];

        // Write K (first half of page)
        int k_offset = physical_page * page_stride + token_in_page * kv_dim + dim;
        page_pool[k_offset] = k[token * kv_dim + dim];

        // Write V (second half of page, offset by page_size * kv_dim)
        int v_offset = physical_page * page_stride + page_size * kv_dim + token_in_page * kv_dim + dim;
        page_pool[v_offset] = v[token * kv_dim + dim];
    }
}

} // extern "C"
