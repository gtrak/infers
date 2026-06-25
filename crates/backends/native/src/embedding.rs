//! Token embedding gather kernel dispatch.
//!
//! Gathers rows from a `[vocab_size × hidden_size]` embedding matrix by token ID
//! using the `infers_embedding_gather_bf16` kernel.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{OxideKernels, CudaSlice, CudaStream};

/// Gather embeddings for a batch of token IDs from the embedding matrix.
///
/// Allocates output buffer on GPU and launches the embedding gather kernel.
///
/// # Arguments
/// * `stream` — CUDA stream to enqueue the kernel on
/// * `oxide` — Loaded OxideKernels bridge handle for `infers_embedding_gather_bf16`
/// * `token_ids` — Host-side array of token IDs to look up
/// * `embedding_table` — GPU-resident embedding matrix `[vocab_size × hidden_size]`
/// * `hidden_size` — Dimension of each embedding vector
/// * `vocab_size` — Vocabulary size (rows in embedding matrix)
///
/// # Returns
/// Newly allocated `CudaSlice<bf16>` of shape `[seq_len × hidden_size]`
pub fn embed_tokens(
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    token_ids: &[u32],
    embedding_table: &CudaSlice<bf16>,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<CudaSlice<bf16>> {
    let seq_len = token_ids.len();
    anyhow::ensure!(seq_len > 0, "Token ID array must not be empty");

    // The kernel expects i32 for token_ids, but we receive u32.
    // Convert to i32 before copying to device.
    let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();

    // Copy token IDs to device
    let token_ids_gpu = stream.clone_htod(&token_ids_i32)
        .map_err(|e| anyhow::anyhow!("Failed to copy token IDs to device: {e}"))?;

    // Allocate output buffer [seq_len × hidden_size]
    let elem_count = seq_len * hidden_size;
    let mut output = stream.alloc_zeros::<bf16>(elem_count)
        .map_err(|e| anyhow::anyhow!("Failed to allocate embedding output: {e}"))?;

    // vocab_size is not used by the kernel but validates caller invariants
    let _ = vocab_size;

    oxide.launch_embedding_gather_bf16(stream, embedding_table, &token_ids_gpu, &mut output, seq_len as u32, hidden_size as u32)?;

    Ok(output)
}

/// Gather embeddings into a pre-allocated output buffer (zero-alloc variant).
///
/// Writes token IDs into the provided staging buffer via `memcpy_htod`, then
/// launches the embedding gather kernel into the pre-allocated output buffer.
/// Replaces per-step `clone_htod` and `alloc_zeros` calls in the decode path,
/// making it CUDA-graph compatible.
///
/// # Arguments
/// * `stream` — CUDA stream to enqueue the kernel on
/// * `oxide` — Loaded OxideKernels bridge handle for `infers_embedding_gather_bf16`
/// * `embedding_table` — GPU-resident embedding matrix `[vocab_size × hidden_size]`
/// * `token_ids_i32` — Host-side token IDs as i32 (single element for decode)
/// * `token_ids_gpu` — Pre-allocated staging buffer on device (must be at least token_ids_i32.len())
/// * `output` — Pre-allocated output buffer `[seq_len × hidden_size]`
/// * `seq_len` — Number of tokens to embed (1 for decode)
/// * `hidden_size` — Dimension of each embedding vector
pub fn embed_tokens_into(
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    embedding_table: &CudaSlice<bf16>,
    token_ids_i32: &[i32],
    token_ids_gpu: &mut CudaSlice<i32>,
    output: &mut CudaSlice<bf16>,
    seq_len: usize,
    hidden_size: usize,
) -> Result<()> {
    anyhow::ensure!(seq_len > 0, "seq_len must be > 0");
    anyhow::ensure!(!token_ids_i32.is_empty(), "Token IDs must not be empty");

    // Write token IDs into pre-allocated staging buffer (no allocation)
    stream.memcpy_htod(token_ids_i32, token_ids_gpu)
        .map_err(|e| anyhow::anyhow!("Failed to copy token IDs to staging buffer: {e}"))?;

    oxide.launch_embedding_gather_bf16(stream, embedding_table, token_ids_gpu, output, seq_len as u32, hidden_size as u32)?;

    Ok(())
}
