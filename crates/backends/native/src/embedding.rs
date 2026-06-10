//! Token embedding gather kernel dispatch.
//!
//! Gathers rows from a `[vocab_size × hidden_size]` embedding matrix by token ID
//! using the `infers_embedding_gather_bf16` kernel.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream};

/// Gather embeddings for a batch of token IDs from the embedding matrix.
///
/// Allocates output buffer on GPU and launches the embedding gather kernel.
///
/// # Arguments
/// * `stream` — CUDA stream to enqueue the kernel on
/// * `kernel` — Loaded CUDA function handle for `infers_embedding_gather_bf16`
/// * `token_ids` — Host-side array of token IDs to look up
/// * `embedding_table` — GPU-resident embedding matrix `[vocab_size × hidden_size]`
/// * `hidden_size` — Dimension of each embedding vector
/// * `vocab_size` — Vocabulary size (rows in embedding matrix)
///
/// # Returns
/// Newly allocated `CudaSlice<bf16>` of shape `[seq_len × hidden_size]`
pub fn embed_tokens(
    stream: &Arc<CudaStream>,
    _kernel: &CudaFunction,
    token_ids: &[u32],
    _embedding_table: &CudaSlice<bf16>,
    hidden_size: usize,
    _vocab_size: usize,
) -> Result<CudaSlice<bf16>> {
    let seq_len = token_ids.len();
    anyhow::ensure!(seq_len > 0, "Token ID array must not be empty");

    let elem_count = seq_len * hidden_size;
    // Kernel launch requires CUdeviceptr for host array pointers,
    // which needs unsafe CUdeviceptr wrapping not available in safe cudarc.
    // Actual implementation uses stream.launch_builder(kernel).arg(CUdeviceptr(token_ids.as_ptr())).launch(config)
    todo!("embed_tokens: allocate output buffer, launch infers_embedding_gather_bf16 kernel with token_ids ptr, embedding_table, hidden_size, output")
}
