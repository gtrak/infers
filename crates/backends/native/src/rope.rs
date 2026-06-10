//! Rotary Position Embedding (RoPE) kernel dispatch.
//!
//! Applies rotary embeddings to query and key tensors in-place using the
//! `infers_rope_bf16` CUDA kernel.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream};

/// Apply RoPE to query and key tensors in-place.
///
/// Rotates the q and k tensors by the given position indices using the
/// rotary embedding defined by the model's `rope_theta` and `partial_rotary_factor`.
///
/// # Arguments
/// * `stream` — CUDA stream to enqueue the kernel on
/// * `kernel` — Loaded CUDA function handle for `infers_rope_bf16`
/// * `q` — Query tensor `[batch_size × seq_len × num_heads × head_dim]` (mutated in-place)
/// * `k` — Key tensor `[batch_size × seq_len × num_kv_heads × head_dim]` (mutated in-place)
/// * `positions` — Per-token position indices for rotary embedding
/// * `head_dim` — Per-head dimension (e.g. 256)
/// * `num_heads` — Number of attention heads
pub fn apply_rope(
    _stream: &Arc<CudaStream>,
    _kernel: &CudaFunction,
    _q: &mut CudaSlice<bf16>,
    _k: &mut CudaSlice<bf16>,
    _positions: &[u32],
    _head_dim: usize,
    _num_heads: usize,
) -> Result<()> {
    // Kernel launch: stream.launch_builder(kernel).arg(q).arg(k).arg(CUdeviceptr(positions.as_ptr())).arg(&seq_len_i32).arg(&head_dim_i32).arg(&num_heads_i32).launch(config)
    todo!("apply_rope: launch infers_rope_bf16 kernel with q, k, positions ptr, head_dim, num_heads")
}
