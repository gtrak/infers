//! Token embedding gather kernel dispatch.
//!
//! Gathers rows from a `[vocab_size × hidden_size]` embedding matrix by token ID
//! using the `infers_embedding_gather_bf16` kernel.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

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
    kernel: &CudaFunction,
    token_ids: &[u32],
    embedding_table: &CudaSlice<bf16>,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<CudaSlice<bf16>> {
    let seq_len = token_ids.len();
    anyhow::ensure!(seq_len > 0, "Token ID array must not be empty");

    // Copy token IDs to device
    let token_ids_gpu = stream
        .clone_htod(token_ids)
        .map_err(|e| anyhow::anyhow!("Failed to copy token IDs to device: {e}"))?;

    // Allocate output buffer [seq_len × hidden_size]
    let elem_count = seq_len * hidden_size;
    let mut output = stream
        .alloc_zeros::<bf16>(elem_count)
        .map_err(|e| anyhow::anyhow!("Failed to allocate embedding output: {e}"))?;

    let seq_len_i32 = seq_len as i32;
    let hidden_size_i32 = hidden_size as i32;
    // vocab_size is not used by the kernel but validates caller invariants
    let _ = vocab_size;

    let config = LaunchConfig {
        grid_dim: (((seq_len as u32) + 15) / 16, 1, 1), // 16 tokens per block
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(kernel)
            .arg(embedding_table)
            .arg(&token_ids_gpu)
            .arg(&mut output)
            .arg(&seq_len_i32)
            .arg(&hidden_size_i32)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("Embedding kernel launch failed: {e}"))?;
    }

    Ok(output)
}
