//! Rotary Position Embedding (RoPE) kernel dispatch.
//!
//! Applies rotary embeddings to query and key tensors in-place using the
//! `infers_rope_bf16` CUDA kernel. Requires precomputed sin/cos tables.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

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
fn precompute_rope_tables(
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
/// * `kernel` — Loaded function handle for `infers_rope_bf16`
/// * `q` — Query tensor `[total_tokens × num_heads × head_dim]`, modified in-place
/// * `k` — Key tensor `[total_tokens × num_heads × head_dim]`, modified in-place
/// * `positions` — Per-token position indices (host slice)
/// * `num_heads` — Number of attention heads
/// * `head_dim` — Per-head dimension (e.g., 256)
/// * `rope_theta` — RoPE base frequency (typically 10000000.0)
/// * `partial_rotary_factor` — Fraction of head_dim to apply RoPE to (typically 0.25)
pub fn apply_rope(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    q: &mut CudaSlice<bf16>,
    k: &mut CudaSlice<bf16>,
    positions: &[u32],
    num_heads: i32,
    head_dim: usize,
    rope_theta: f64,
    partial_rotary_factor: f32,
) -> Result<()> {
    anyhow::ensure!(!positions.is_empty(), "Positions must not be empty");
    anyhow::ensure!(head_dim % 2 == 0, "head_dim must be even, got {}", head_dim);

    let max_position = *positions.iter().max().unwrap();

    // TODO: Cache RoPE tables in ForwardEngine at init time instead of recomputing per call
    // Precompute sin/cos tables on host — indexed by position value, not sequential index
    let (cos_table, sin_table) = precompute_rope_tables(max_position, head_dim, rope_theta, partial_rotary_factor);

    // Copy data to device
    let positions_gpu = stream
        .clone_htod(positions)
        .map_err(|e| anyhow::anyhow!("Failed to copy positions to device: {e}"))?;
    let cos_gpu = stream
        .clone_htod(&cos_table)
        .map_err(|e| anyhow::anyhow!("Failed to copy cos table to device: {e}"))?;
    let sin_gpu = stream
        .clone_htod(&sin_table)
        .map_err(|e| anyhow::anyhow!("Failed to copy sin table to device: {e}"))?;

    let total_tokens = positions.len() as i32;
    let head_dim_i32 = head_dim as i32;

    // Calculate grid size based on total_tokens * num_heads * half_dim work items
    let rotary_dim = (head_dim as f32 * partial_rotary_factor) as usize;
    let half_dim = rotary_dim / 2;
    let total_pairs = positions.len() * num_heads as usize * half_dim;
    let config = LaunchConfig {
        grid_dim: (((total_pairs as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(kernel)
            .arg(q)                // query (mutable, in-place)
            .arg(k)                // key (mutable, in-place)
            .arg(&cos_gpu)        // cos table
            .arg(&sin_gpu)        // sin table
            .arg(&positions_gpu)   // position IDs
            .arg(&total_tokens)    // total_tokens
            .arg(&num_heads)       // num_heads
            .arg(&head_dim_i32)   // head_dim
            .launch(config)
            .map_err(|e| anyhow::anyhow!("RoPE kernel launch failed: {e}"))?;
    }

    Ok(())
}
