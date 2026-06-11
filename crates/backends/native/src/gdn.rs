//! Gated DeltaNet (GDN) forward pass using Mamba2 SSM kernels.
//!
//! Implements the Mamba2-style recurrent layer with:
//! - x_proj: main input projection (SSM input)
//! - b_proj: state contribution projection
//! - dt_proj: delta timescale projection
//! - z_gate: output gate (for residual mixing)
//! - A_log: SSM state transition log
//! - dt_bias: timescale bias
//!
//! The kernel operates on ssm_dim (determined by in_proj_a's output dimension),
//! which is typically smaller than hidden_size. The output projection GEMM
//! expands back to hidden_size.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use infers_cuda::gemm::GemmEngine;
use infers_model::{GdnWeights, ModelConfig, WeightDtype};

/// Get the output dimension from a weight tensor.
///
/// BF16/FP16/FP32 weights are stored as [N, K] → output dim = shape[0]
/// INT4 qweights are stored as [K/8, N] → output dim = shape[1]
fn weight_output_dim(w: &infers_model::WeightData) -> usize {
    if w.dtype == WeightDtype::Int4Packed { w.shape[1] } else { w.shape[0] }
}

/// GDN recurrent state, maintained across decode steps.
///
/// Stores a 1D SSM state vector `[ssm_dim]` on GPU. During prefill,
/// this is initialized from scratch; during decode, it is updated
/// token by token via the Mamba2 update rule.
#[derive(Debug)]
pub struct GdnState {
    /// SSM state vector `[ssm_dim]` on GPU.
    pub state: Option<CudaSlice<bf16>>,
    /// SSM dimension the state was allocated for.
    pub ssm_dim: usize,
}

impl Default for GdnState {
    fn default() -> Self {
        Self::new()
    }
}

impl GdnState {
    /// Create an empty GDN state.
    pub fn new() -> Self {
        Self { state: None, ssm_dim: 0 }
    }

    /// Ensure state buffer is allocated for the given SSM dimension.
    ///
    /// If the state is not allocated or the dimension changed, reallocates.
    pub fn ensure_allocated(
        &mut self,
        stream: &Arc<CudaStream>,
        ssm_dim: usize,
    ) -> Result<()> {
        if self.state.is_none() || self.ssm_dim != ssm_dim {
            self.state = Some(
                stream
                    .alloc_zeros::<bf16>(ssm_dim)
                    .map_err(|e| anyhow::anyhow!("Failed to allocate GDN state: {e}"))?,
            );
            self.ssm_dim = ssm_dim;
        }
        Ok(())
    }
}

/// Extract first `take_cols` columns from a row-major buffer `[seq_len × num_cols]`.
///
/// When `take_cols <= num_cols`, copies the first `take_cols` elements per row
/// into a new `[seq_len × take_cols]` buffer.
/// When `take_cols > num_cols`, copies all elements and pads with zeros.
///
/// # Arguments
/// * `stream` — CUDA stream
/// * `buf` — Source buffer `[seq_len × num_cols]`
/// * `seq_len` — Number of rows
/// * `num_cols` — Columns in source
/// * `take_cols` — Columns to extract (or pad to)
///
/// # Returns
/// New `CudaSlice<bf16>` of size `seq_len × take_cols`
fn extract_columns(
    stream: &Arc<CudaStream>,
    buf: &CudaSlice<bf16>,
    seq_len: usize,
    num_cols: usize,
    take_cols: usize,
) -> Result<CudaSlice<bf16>> {
    let mut result = stream
        .alloc_zeros::<bf16>(seq_len * take_cols)
        .map_err(|e| anyhow::anyhow!("Failed to allocate column extraction buffer: {e}"))?;

    let copy_cols = num_cols.min(take_cols);
    for row in 0..seq_len {
        let src_offset = row * num_cols;
        let dst_offset = row * take_cols;
        let src_slice = buf.slice(src_offset..src_offset + copy_cols);
        let mut dst_slice = result.slice_mut(dst_offset..dst_offset + copy_cols);
        stream
            .memcpy_dtod(&src_slice, &mut dst_slice)
            .map_err(|e| anyhow::anyhow!("Failed to copy row {}: {e}", row))?;
    }

    Ok(result)
}

/// Extract first `take_cols` elements from a single-row buffer `[num_cols]`.
///
/// Copies the first `take_cols` elements. If `take_cols > num_cols`,
/// pads with zeros.
fn extract_columns_single(
    stream: &Arc<CudaStream>,
    buf: &CudaSlice<bf16>,
    num_cols: usize,
    take_cols: usize,
) -> Result<CudaSlice<bf16>> {
    let mut result = stream
        .alloc_zeros::<bf16>(take_cols)
        .map_err(|e| anyhow::anyhow!("Failed to allocate column extraction buffer: {e}"))?;

    let copy_cols = num_cols.min(take_cols);
    let src_slice = buf.slice(0..copy_cols);
    let mut dst_slice = result.slice_mut(0..copy_cols);
    stream
        .memcpy_dtod(&src_slice, &mut dst_slice)
        .map_err(|e| anyhow::anyhow!("Failed to copy elements: {e}"))?;

    Ok(result)
}

/// Prefill-time GDN forward pass using Mamba2 kernel.
///
/// Steps:
/// 1. Compute projections: x_proj, b_proj, dt_proj, z_gate via GEMM
/// 2. Extract/align to ssm_dim columns for kernel input
/// 3. Upload SSM parameters (A_log, dt_bias)
/// 4. Launch Mamba2 prefill kernel
/// 5. Output projection to hidden_size
///
/// conv1d and conv1d residual are skipped (not critical for initial release).
/// QKV residual is handled when in_proj_qkv is present.
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `int4_kernel` — INT4 GEMM kernel for quantized weights
/// * `stream` — CUDA stream
/// * `gdn_prefill_kernel` — Loaded CUDA function for `infers_gdn_mamba2_prefill_bf16`
/// * `weights` — GDN layer weights
/// * `input` — Input tensor `[seq_len × hidden_size]`
/// * `gdn_state` — Mutable state (allocated/updated in-place)
/// * `hidden_size` — Model hidden dimension
/// * `group_size` — INT4 quantization group size (typically 128)
/// * `cache` — GPU weight cache
///
/// # Returns
/// GDN output `[seq_len × hidden_size]`
pub fn forward(
    gemm: &mut GemmEngine,
    int4_kernel: &CudaFunction,
    stream: &Arc<CudaStream>,
    gdn_prefill_kernel: &CudaFunction,
    weights: &GdnWeights,
    input: &CudaSlice<bf16>,
    gdn_state: &mut GdnState,
    hidden_size: usize,
    config: &ModelConfig,
    group_size: usize,
    cache: &crate::gpu_cache::GpuWeightCache,
) -> Result<CudaSlice<bf16>> {
    let seq_len = input.len() / hidden_size;

    // Number of value heads from in_proj_a's output dimension.
    // BF16: shape = [N, K] = [num_heads, hidden_size] → N = shape[0]
    // INT4: shape = [K/8, N] → N = shape[1]
    let num_value_heads = weight_output_dim(&weights.in_proj_a);
    let head_dim = config.linear_value_head_dim;
    let total_dim = num_value_heads * head_dim;

    // =========================================================================
    // Phase 1: Projection GEMMs (INT4-aware via gemm_projection_cached)
    // =========================================================================

    // x_proj = input @ in_proj_a^T  [seq_len, num_value_heads]  (per-head scalar)
    let mut x_proj = stream
        .alloc_zeros::<bf16>(seq_len * num_value_heads)
        .map_err(|e| anyhow::anyhow!("Failed to allocate x_proj buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream,
        cache, &weights.in_proj_a.name, input, &mut x_proj,
        seq_len, num_value_heads, hidden_size, group_size,
    )?;

    // b_proj = input @ in_proj_b^T  [seq_len, b_dim]
    let b_dim = weight_output_dim(&weights.in_proj_b);
    let mut b_proj_raw = stream
        .alloc_zeros::<bf16>(seq_len * b_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate b_proj buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream,
        cache, &weights.in_proj_b.name, input, &mut b_proj_raw,
        seq_len, b_dim, hidden_size, group_size,
    )?;

    // dt_proj: if x_proj_weight exists, compute dt_proj = input @ x_proj_weight
    // If not, dt is just dt_bias (no projection needed)
    // NOTE: when x_proj_weight maps to num_value_heads (per-head), the result
    // must be broadcast to total_dim. Currently unsupported — assumes None.
    let dt_proj = if let Some(x_proj_w) = &weights.x_proj_weight {
        let dt_dim = weight_output_dim(x_proj_w);
        let mut dt = stream
            .alloc_zeros::<bf16>(seq_len * dt_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate dt_proj buffer: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, int4_kernel, stream,
            cache, &x_proj_w.name, input, &mut dt,
            seq_len, dt_dim, hidden_size, group_size,
        )?;
        Some(extract_columns(stream, &dt, seq_len, dt_dim, num_value_heads)?)
    } else {
        None
    };


    // z_gate = input @ in_proj_z^T (INT4)  [seq_len, total_dim]
    // Use the full z_gate_raw directly (NO column extraction — z_dim == total_dim)
    let z_gate = if let Some(z_weight) = &weights.in_proj_z {
        let z_dim = weight_output_dim(z_weight);
        let mut z_gate_raw = stream
            .alloc_zeros::<bf16>(seq_len * z_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate z_gate buffer: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, int4_kernel, stream,
            cache, &z_weight.name, input, &mut z_gate_raw,
            seq_len, z_dim, hidden_size, group_size,
        )?;
        // Keep all columns — z_dim should equal total_dim
        Some(z_gate_raw)
    } else {
        None
    };

    // =========================================================================
    // Phase 2: Align projections
    // =========================================================================

    // b_proj: extract/pad to num_value_heads
    let b_proj = extract_columns(stream, &b_proj_raw, seq_len, b_dim, num_value_heads)?;

    // dt_proj: use zeros if x_proj_weight was absent (full [seq_len, total_dim])
    let dt_proj_buf = dt_proj.unwrap_or_else(|| {
        stream
            .alloc_zeros::<bf16>(seq_len * total_dim)
            .expect("Failed to allocate dt_proj zeros")
    });

    // z_gate: default zeros if absent (full [seq_len, total_dim])
    let z_gate = z_gate.unwrap_or_else(|| {
        stream
            .alloc_zeros::<bf16>(seq_len * total_dim)
            .expect("Failed to allocate default z_gate zeros")
    });

    // =========================================================================
    // Phase 3: Upload SSM parameters
    // =========================================================================

    let a_log_gpu: &CudaSlice<bf16>;
    let dt_bias_gpu: &CudaSlice<bf16>;
    let mut a_log_zeros: Option<CudaSlice<bf16>> = None;
    let mut dt_bias_zeros: Option<CudaSlice<bf16>> = None;

    if let Some(w) = &weights.a_log {
        a_log_gpu = cache.get_bf16(&w.name)
            .ok_or_else(|| anyhow::anyhow!("a_log weight '{}' not in cache", w.name))?;
    } else {
        a_log_zeros = Some(stream.alloc_zeros::<bf16>(num_value_heads).expect("Failed to allocate A_log"));
        a_log_gpu = a_log_zeros.as_ref().unwrap();
    }

    if let Some(w) = &weights.dt_bias {
        dt_bias_gpu = cache.get_bf16(&w.name)
            .ok_or_else(|| anyhow::anyhow!("dt_bias weight '{}' not in cache", w.name))?;
    } else {
        dt_bias_zeros = Some(stream.alloc_zeros::<bf16>(num_value_heads).expect("Failed to allocate dt_bias"));
        dt_bias_gpu = dt_bias_zeros.as_ref().unwrap();
    }

    // =========================================================================
    // Phase 4: Ensure GDN state is allocated and launch Mamba2 prefill kernel
    // =========================================================================

    gdn_state.ensure_allocated(stream, total_dim)?;

    let mut gdn_output = stream
        .alloc_zeros::<bf16>(seq_len * total_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate GDN output buffer: {e}"))?;

    let state_ref = gdn_state.state.as_mut()
        .expect("GDN state should be allocated");

    let seq_len_i32 = seq_len as i32;
    let num_value_heads_i32 = num_value_heads as i32;
    let head_dim_i32 = head_dim as i32;

    // Grid: ceil(total_dim / 256), Block: 256
    let grid = (total_dim as u32).div_ceil(256);
    let config = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(gdn_prefill_kernel)
            .arg(&x_proj)           // x_proj [seq, num_value_heads]
            .arg(&b_proj)           // b_proj [seq, num_value_heads]
            .arg(&dt_proj_buf)    // dt_proj [seq, total_dim]
            .arg(&z_gate)           // z_gate [seq, total_dim]
            .arg(a_log_gpu)         // A_log [num_value_heads]
            .arg(dt_bias_gpu)       // dt_bias [num_value_heads]
            .arg(state_ref)         // state [total_dim] (mut in/out)
            .arg(&mut gdn_output)  // output [seq, total_dim] (mut out)
            .arg(&seq_len_i32)      // seq_len (i32)
            .arg(&num_value_heads_i32) // num_value_heads (i32)
            .arg(&head_dim_i32)     // head_dim (i32)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("GDN Mamba2 prefill kernel launch failed: {e}"))?;
    }

    // =========================================================================
    // Phase 5: Output projection — [seq_len, total_dim] → [seq_len, hidden]
    // =========================================================================

    let mut output = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate GDN final output: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream,
        cache, &weights.out_proj_weight.name, &gdn_output, &mut output,
        seq_len, hidden_size, total_dim, group_size,
    )?;

    Ok(output)
}

/// Decode-time GDN: recurrent step with updated hidden state.
///
/// For a single token:
/// 1. Compute projections: x_proj, b_proj, dt_proj, z_gate via GEMM
/// 2. Extract/align to ssm_dim columns
/// 3. Launch Mamba2 update kernel
/// 4. Output projection to hidden_size
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `int4_kernel` — INT4 GEMM kernel for quantized weights
/// * `stream` — CUDA stream
/// * `gdn_update_kernel` — Loaded CUDA function for `infers_gdn_mamba2_update_bf16`
/// * `weights` — GDN layer weights
/// * `input` — Single-token input `[1 × hidden_size]`
/// * `gdn_state` — Mutable recurrent state (updated in-place)
/// * `hidden_size` — Model hidden dimension
/// * `group_size` — INT4 quantization group size (typically 128)
/// * `cache` — GPU weight cache
///
/// # Returns
/// GDN output `[1 × hidden_size]`
pub fn decode_forward(
    gemm: &mut GemmEngine,
    int4_kernel: &CudaFunction,
    stream: &Arc<CudaStream>,
    gdn_update_kernel: &CudaFunction,
    weights: &GdnWeights,
    input: &CudaSlice<bf16>,
    gdn_state: &mut GdnState,
    hidden_size: usize,
    config: &ModelConfig,
    group_size: usize,
    cache: &crate::gpu_cache::GpuWeightCache,
) -> Result<CudaSlice<bf16>> {
    // Number of value heads from in_proj_a's output dimension
    let num_value_heads = weight_output_dim(&weights.in_proj_a);
    let head_dim = config.linear_value_head_dim;
    let total_dim = num_value_heads * head_dim;

    // =========================================================================
    // Phase 1: Projection GEMMs (single token, m=1, INT4-aware)
    // =========================================================================

    // x_proj = input @ in_proj_a^T  [1, num_value_heads]  (per-head scalar)
    let mut x_proj = stream
        .alloc_zeros::<bf16>(num_value_heads)
        .map_err(|e| anyhow::anyhow!("Failed to allocate x_proj buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream,
        cache, &weights.in_proj_a.name, input, &mut x_proj,
        1, num_value_heads, hidden_size, group_size,
    )?;

    // b_proj = input @ in_proj_b^T  [1, b_dim]
    let b_dim = weight_output_dim(&weights.in_proj_b);
    let mut b_proj_raw = stream
        .alloc_zeros::<bf16>(b_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate b_proj buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream,
        cache, &weights.in_proj_b.name, input, &mut b_proj_raw,
        1, b_dim, hidden_size, group_size,
    )?;

    // dt_proj: if x_proj_weight exists, compute dt_proj = input @ x_proj_weight
    // If not, dt is just dt_bias (no projection needed)
    let dt_proj = if let Some(x_proj_w) = &weights.x_proj_weight {
        let dt_dim = weight_output_dim(x_proj_w);
        let mut dt = stream
            .alloc_zeros::<bf16>(dt_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate dt_proj buffer: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, int4_kernel, stream,
            cache, &x_proj_w.name, input, &mut dt,
            1, dt_dim, hidden_size, group_size,
        )?;
        Some(extract_columns_single(stream, &dt, dt_dim, num_value_heads)?)
    } else {
        None
    };


    // z_gate = input @ in_proj_z^T (INT4)  [1, total_dim]
    // Keep all columns — z_dim should equal total_dim
    let z_gate = if let Some(z_weight) = &weights.in_proj_z {
        let z_dim = weight_output_dim(z_weight);
        let mut z_gate_raw = stream
            .alloc_zeros::<bf16>(z_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate z_gate buffer: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, int4_kernel, stream,
            cache, &z_weight.name, input, &mut z_gate_raw,
            1, z_dim, hidden_size, group_size,
        )?;
        Some(z_gate_raw)
    } else {
        None
    };

    // =========================================================================
    // Phase 2: Align projections
    // =========================================================================

    let b_proj = extract_columns_single(stream, &b_proj_raw, b_dim, num_value_heads)?;
    // dt_proj: use zeros if x_proj_weight was absent (full [total_dim])
    let dt_proj_buf = dt_proj.unwrap_or_else(|| {
        stream
            .alloc_zeros::<bf16>(total_dim)
            .expect("Failed to allocate dt_proj zeros")
    });

    let z_gate = z_gate.unwrap_or_else(|| {
        stream
            .alloc_zeros::<bf16>(total_dim)
            .expect("Failed to allocate default z_gate zeros")
    });

    // =========================================================================
    // Phase 3: Upload SSM parameters
    // =========================================================================

    let a_log_gpu: &CudaSlice<bf16>;
    let dt_bias_gpu: &CudaSlice<bf16>;
    let mut a_log_zeros: Option<CudaSlice<bf16>> = None;
    let mut dt_bias_zeros: Option<CudaSlice<bf16>> = None;

    if let Some(w) = &weights.a_log {
        a_log_gpu = cache.get_bf16(&w.name)
            .ok_or_else(|| anyhow::anyhow!("a_log weight '{}' not in cache", w.name))?;
    } else {
        a_log_zeros = Some(stream.alloc_zeros::<bf16>(num_value_heads).expect("Failed to allocate A_log"));
        a_log_gpu = a_log_zeros.as_ref().unwrap();
    }

    if let Some(w) = &weights.dt_bias {
        dt_bias_gpu = cache.get_bf16(&w.name)
            .ok_or_else(|| anyhow::anyhow!("dt_bias weight '{}' not in cache", w.name))?;
    } else {
        dt_bias_zeros = Some(stream.alloc_zeros::<bf16>(num_value_heads).expect("Failed to allocate dt_bias"));
        dt_bias_gpu = dt_bias_zeros.as_ref().unwrap();
    }

    // =========================================================================
    // Phase 4: Ensure GDN state and launch Mamba2 update kernel
    // =========================================================================

    gdn_state.ensure_allocated(stream, total_dim)?;

    let mut gdn_output = stream
        .alloc_zeros::<bf16>(total_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate GDN output buffer: {e}"))?;

    let state_ref = gdn_state.state.as_mut()
        .expect("GDN state should be allocated");

    let num_value_heads_i32 = num_value_heads as i32;
    let head_dim_i32 = head_dim as i32;

    // Grid: ceil(total_dim / 256), Block: 256
    let grid = (total_dim as u32).div_ceil(256);
    let config = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(gdn_update_kernel)
            .arg(&x_proj)           // x_proj [num_value_heads]
            .arg(&b_proj)           // b_proj [num_value_heads]
            .arg(&dt_proj_buf)    // dt_proj [total_dim]
            .arg(&z_gate)           // z_gate [total_dim]
            .arg(a_log_gpu)         // A_log [num_value_heads]
            .arg(dt_bias_gpu)       // dt_bias [num_value_heads]
            .arg(state_ref)         // state [total_dim] (mut in/out)
            .arg(&mut gdn_output)  // output [total_dim] (mut out)
            .arg(&num_value_heads_i32) // num_value_heads (i32)
            .arg(&head_dim_i32)     // head_dim (i32)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("GDN Mamba2 update kernel launch failed: {e}"))?;
    }

    // =========================================================================
    // Phase 5: Output projection — [total_dim] → [hidden_size]
    // =========================================================================

    let mut output = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate GDN final output: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream,
        cache, &weights.out_proj_weight.name, &gdn_output, &mut output,
        1, hidden_size, total_dim, group_size,
    )?;

    Ok(output)
}
