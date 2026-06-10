// @lat: [[lat#Kernel Extraction and Build System#Kernel Source Files]]
/// Token embedding gather kernel for BF16 tensors.
///
/// Gathers rows from an embedding weight matrix based on token IDs.
/// Input: token_ids[seq_len], weight[vocab_size, hidden_size]
/// Output: hidden[seq_len, hidden_size]

#include "common.cuh"

/// Embedding gather kernel — each thread handles one output element.
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void embedding_gather_kernel(
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

extern "C" {

/// Launch embedding gather for BF16 weights.
///
/// # Arguments
/// * `weight` — Embedding matrix [vocab_size × hidden_size] in BF16
/// * `token_ids` — Token IDs [seq_len] as int32
/// * `output` — Output tensor [seq_len × hidden_size] in BF16
/// * `seq_len` — Sequence length
/// * `hidden_size` — Hidden dimension
void infers_embedding_gather_bf16(
    const __nv_bfloat16* weight,
    const int* token_ids,
    __nv_bfloat16* output,
    int seq_len,
    int hidden_size
) {
    int total_elements = seq_len * hidden_size;
    int block_size = INFERS_BLOCK_SIZE;
    int grid_size = (total_elements + block_size - 1) / block_size;

    embedding_gather_kernel<<<grid_size, block_size>>>(
        weight, token_ids, output, seq_len, hidden_size
    );
}

} // extern "C"
