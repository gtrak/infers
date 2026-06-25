//! Rotary Position Embedding (RoPE) kernel dispatch.
//!
//! Applies rotary embeddings to query and key tensors in-place using the
//! `infers_rope_bf16` CUDA kernel. Requires precomputed sin/cos tables.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{OxideKernels, CudaSlice, CudaStream};

/// Precompute RoPE sin/cos embedding tables.
///
/// Generates the rotation matrices for all positions up to `max_position` using the
/// standard RoPE formulation: `freq[k] = 1.0 / (theta^(2k/d))`.
///
/// The table is indexed by position value: `table[pos * half_dim + k]`, matching
/// the CUDA kernel's indexing scheme.
///
/// # Arguments
/// * `max_position` — Maximum position value (table covers 0..=max_position)
/// * `head_dim` — Per-head dimension (must be even)
/// * `rope_theta` — Base frequency (typically 10000000.0 for Qwen)
/// * `partial_rotary_factor` — Fraction of head dimensions to apply RoPE to (typically 0.25)
///
/// # Returns
/// `(cos_table, sin_table)` as Vec<f32>, each of length `(max_position + 1) * half_dim`
pub(crate) fn precompute_rope_tables(
    max_position: u32,
    head_dim: usize,
    rope_theta: f64,
    partial_rotary_factor: f32,
) -> (Vec<f32>, Vec<f32>) {
    let rotary_dim = (head_dim as f32 * partial_rotary_factor) as usize;
    let half_dim = rotary_dim / 2;
    let num_positions = (max_position + 1) as usize;
    let table_len = num_positions * half_dim;

    let mut cos_table = vec![0.0f32; table_len];
    let mut sin_table = vec![0.0f32; table_len];

    for pos in 0..=max_position {
        for k in 0..half_dim {
            let freq = 1.0 / (rope_theta.powf(2.0 * k as f64 / rotary_dim as f64));
            let angle = pos as f64 * freq;
            let idx = (pos as usize) * half_dim + k;
            cos_table[idx] = angle.cos() as f32;
            sin_table[idx] = angle.sin() as f32;
        }
    }

    (cos_table, sin_table)
}

/// Apply RoPE to query and key tensors in-place.
///
/// Precomputes sin/cos tables, copies them and position IDs to device,
/// then launches the `infers_rope_bf16` kernel.
///
/// # Arguments
/// * `stream` — CUDA stream
/// * `oxide` — Loaded OxideKernels bridge handle for `infers_rope_bf16`
/// * `q` — Query tensor `[total_tokens × num_heads × head_dim]`, modified in-place
/// * `k` — Key tensor `[total_tokens × num_heads × head_dim]`, modified in-place
/// * `positions` — Per-token position indices (host slice)
/// * `num_heads` — Number of attention heads
/// * `head_dim` — Per-head dimension (e.g., 256)
/// * `rope_theta` — RoPE base frequency (typically 10000000.0)
/// * `partial_rotary_factor` — Fraction of head_dim to apply RoPE to (typically 0.25)
pub fn apply_rope(
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    q: &mut CudaSlice<bf16>,
    k: &mut CudaSlice<bf16>,
    positions: &[u32],
    num_heads: i32,
    head_dim: usize,
    rope_theta: f64,
    partial_rotary_factor: f32,
    cached_cos: Option<&CudaSlice<f32>>,  // pre-computed cos table (GPU-resident)
    cached_sin: Option<&CudaSlice<f32>>,  // pre-computed sin table (GPU-resident)
) -> Result<()> {
    anyhow::ensure!(!positions.is_empty(), "Positions must not be empty");
    anyhow::ensure!(head_dim.is_multiple_of(2), "head_dim must be even, got {}", head_dim);

    let rotary_dim = (head_dim as f32 * partial_rotary_factor) as usize;

    // The kernel expects i32 for positions, but we receive u32.
    // Convert to i32 before copying to device.
    let positions_i32: Vec<i32> = positions.iter().map(|&x| x as i32).collect();

    let (cos_gpu, sin_gpu, positions_gpu) = if let (Some(cos), Some(sin)) = (cached_cos, cached_sin) {
        // Use cached GPU tables — still need positions on GPU per-step.
        // clone_htod is stream-ordered; subsequent kernels on the same stream will see the write.
        let positions_gpu = stream.clone_htod(&positions_i32)
            .map_err(|e| anyhow::anyhow!("Failed to copy positions to device: {e}"))?;
        (cos.clone(), sin.clone(), positions_gpu)
    } else {
        // Fallback: compute and upload (old path, no synchronize needed — stream-ordered).
        let max_position = *positions.iter().max().unwrap();
        let (cos_table, sin_table) = precompute_rope_tables(max_position, head_dim, rope_theta, partial_rotary_factor);

        let positions_gpu = stream.clone_htod(&positions_i32)
            .map_err(|e| anyhow::anyhow!("Failed to copy positions to device: {e}"))?;
        let cos_gpu = stream.clone_htod(&cos_table)
            .map_err(|e| anyhow::anyhow!("Failed to copy cos table to device: {e}"))?;
        let sin_gpu = stream.clone_htod(&sin_table)
            .map_err(|e| anyhow::anyhow!("Failed to copy sin table to device: {e}"))?;

        (cos_gpu, sin_gpu, positions_gpu)
    };

    oxide.launch_rope_bf16(
        stream, q, k, &cos_gpu, &sin_gpu, &positions_gpu,
        positions.len() as u32, num_heads as u32, head_dim as u32, rotary_dim as u32,
    )?;

    Ok(())
}

/// Apply RoPE using a pre-allocated position staging buffer (zero-alloc variant).
///
/// Writes the position into the provided staging buffer via `memcpy_htod` instead
/// of allocating a new device buffer each step. Requires cached cos/sin tables.
/// Used in the decode path for CUDA-graph compatibility.
///
/// # Arguments
/// * `stream` — CUDA stream
/// * `oxide` — Loaded OxideKernels bridge handle for `infers_rope_bf16`
/// * `q` — Query tensor, modified in-place
/// * `k` — Key tensor, modified in-place
/// * `position_i32` — Host-side position as i32 (single element for decode)
/// * `positions_gpu` — Pre-allocated staging buffer on device (at least 1 element)
/// * `num_heads` — Number of attention heads
/// * `head_dim` — Per-head dimension
/// * `cos_gpu` — Pre-computed cos table on GPU
/// * `sin_gpu` — Pre-computed sin table on GPU
/// * `partial_rotary_factor` — Fraction of head_dim to apply RoPE to
pub fn apply_rope_with_staging(
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    q: &mut CudaSlice<bf16>,
    k: &mut CudaSlice<bf16>,
    position_i32: &[i32],
    positions_gpu: &mut CudaSlice<i32>,
    num_heads: i32,
    head_dim: usize,
    cos_gpu: &CudaSlice<f32>,
    sin_gpu: &CudaSlice<f32>,
    partial_rotary_factor: f32,
) -> Result<()> {
    anyhow::ensure!(!position_i32.is_empty(), "Position must not be empty");

    let rotary_dim = (head_dim as f32 * partial_rotary_factor) as usize;

    // Write position into pre-allocated staging buffer (no allocation)
    stream.memcpy_htod(position_i32, positions_gpu)
        .map_err(|e| anyhow::anyhow!("Failed to copy position to staging buffer: {e}"))?;

    oxide.launch_rope_bf16(
        stream, q, k, cos_gpu, sin_gpu, positions_gpu,
        position_i32.len() as u32, num_heads as u32, head_dim as u32, rotary_dim as u32,
    )?;

    Ok(())
}
