// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Token embedding gather kernel for BF16 tensors.
///
/// Gathers rows from an embedding weight matrix based on token IDs.
/// Input: token_ids[seq_len], weight[vocab_size, hidden_size]
/// Output: hidden[seq_len, hidden_size]

#include "common.cuh"

/// Embedding gather kernel — each thread handles one output element.
///
/// # Launch configuration
/// * grid: `(seq_len * hidden_size + INFERS_BLOCK_SIZE - 1) / INFERS_BLOCK_SIZE`
/// * block: `INFERS_BLOCK_SIZE`
extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_embedding_gather_bf16(
    const __nv_bfloat16* __restrict__ weight,
    const int* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ output,
    int seq_len,
    int hidden_size
) {
    int idx = INFERS_THREAD_IDX;
    int stride = blockDim.x * gridDim.x;
    int total_elements = seq_len * hidden_size;

    for (int i = idx; i < total_elements; i += stride) {
        int pos = i / hidden_size;
        int dim = i % hidden_size;
        int token_id = token_ids[pos];
        output[i] = weight[token_id * hidden_size + dim];
    }
}

} // extern "C"
