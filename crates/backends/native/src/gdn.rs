//! Gated DeltaNet (GDN) forward pass.
//!
//! Implements the recurrent linear-attention layer used in the hybrid
//! attention pattern. Uses custom CUDA GDN kernels for state update.
//! Uses INT4-aware GEMM dispatch via `gemm_projection`.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use infers_cuda::gemm::GemmEngine;
use infers_model::{GdnWeights, Int4Companions};

/// GDN recurrent state, maintained across decode steps.
///
/// Stores the H×H state matrix on the GPU. During prefill, this is
/// initialized from scratch; during decode, it is updated token by token.
#[derive(Debug)]
pub struct GdnState {
    /// SSM state matrix `[hidden_size × hidden_size]` on GPU.
    pub state: Option<CudaSlice<bf16>>,
    /// Hidden size the state was allocated for.
    pub hidden_size: usize,
}

impl Default for GdnState {
    fn default() -> Self {
        Self::new()
    }
}

impl GdnState {
    /// Create an empty GDN state.
    pub fn new() -> Self {
        Self { state: None, hidden_size: 0 }
    }

    /// Ensure state buffer is allocated.
    pub fn ensure_allocated(
        &mut self,
        stream: &Arc<CudaStream>,
        hidden_size: usize,
    ) -> Result<()> {
        if self.state.is_none() || self.hidden_size != hidden_size {
            let total = hidden_size * hidden_size;
            self.state = Some(
                stream
                    .alloc_zeros::<bf16>(total)
                    .map_err(|e| anyhow::anyhow!("Failed to allocate GDN state: {e}"))?,
            );
            self.hidden_size = hidden_size;
        }
        Ok(())
    }
}

/// Prefill-time GDN forward pass.
///
/// Steps:
/// 1. RMSNorm (handled by caller)
/// 2. GDN projections: in_proj_a, in_proj_b, x_proj, dt_proj (INT4-aware)
/// 3. GDN prefill kernel (gated delta rule state update)
/// 4. Output projection (INT4-aware)
///
/// conv1d is skipped (not critical for Phase 4.5).
/// TP all-reduce is handled by the caller in prefill.rs.
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `int4_kernel` — INT4 GEMM kernel for quantized weights
/// * `stream` — CUDA stream
/// * `gdn_prefill_kernel` — Loaded CUDA function for `infers_gdn_prefill_bf16`
/// * `weights` — GDN layer weights
/// * `input` — Input tensor `[seq_len × hidden_size]`
/// * `gdn_state` — Mutable state (allocated/updated in-place)
/// * `hidden_size` — Model hidden dimension
/// * `group_size` — INT4 quantization group size (typically 128)
/// * `int4_companions` — Companion tensors for INT4 weights
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
    group_size: usize,
    int4_companions: &HashMap<String, Int4Companions>,
) -> Result<CudaSlice<bf16>> {
    let seq_len = input.len() / hidden_size;

    // =========================================================================
    // Phase 1: Projection GEMMs (INT4-aware via gemm_projection)
    // =========================================================================

    // a = GEMM(input, in_proj_a^T)  [seq_len × hidden_size]
    let mut a = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate a buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.in_proj_a, input, &mut a,
        seq_len, hidden_size, hidden_size, group_size, int4_companions,
    )?;

    // b = GEMM(input, in_proj_b^T)  [seq_len × hidden_size]
    let mut b = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate b buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.in_proj_b, input, &mut b,
        seq_len, hidden_size, hidden_size, group_size, int4_companions,
    )?;

    // x = GEMM(input, x_proj^T)  [seq_len × hidden_size]
    let mut x = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate x buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.x_proj_weight, input, &mut x,
        seq_len, hidden_size, hidden_size, group_size, int4_companions,
    )?;

    // dt = GEMM(input, dt_proj^T)  [seq_len × hidden_size]
    let mut dt = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dt buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.dt_proj_weight, input, &mut dt,
        seq_len, hidden_size, hidden_size, group_size, int4_companions,
    )?;

    // =========================================================================
    // Phase 2: GDN prefill kernel (gated delta rule)
    // =========================================================================

    // Ensure GDN state is allocated
    gdn_state.ensure_allocated(stream, hidden_size)?;

    // Allocate GDN kernel output buffer
    let mut gdn_output = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate GDN output buffer: {e}"))?;

    // Grid: hidden_size blocks (one per state row), block_size power of 2 up to 256
    let mut block_size: usize = 1;
    while block_size < hidden_size && block_size < 256 {
        block_size *= 2;
    }
    let shared_mem = block_size * std::mem::size_of::<f32>();

    let hidden_size_i32 = hidden_size as i32;
    let seq_len_i32 = seq_len as i32;

    let config = LaunchConfig {
        grid_dim: (hidden_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    // Kernel requires mutable state pointer (IN/OUT).
    let state_ref = gdn_state.state.as_mut()
        .expect("GDN state should be allocated");

    unsafe {
        stream
            .launch_builder(gdn_prefill_kernel)
            .arg(state_ref)       // state (IN/OUT)
            .arg(&mut gdn_output) // output (OUT)
            .arg(&a)              // a projections
            .arg(&b)              // b projections
            .arg(&dt)             // dt projections
            .arg(&x)              // x projections
            .arg(&hidden_size_i32)
            .arg(&seq_len_i32)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("GDN prefill kernel launch failed: {e}"))?;
    }

    // =========================================================================
    // Phase 3: Output projection (INT4-aware)
    // =========================================================================

    // output = GEMM(gdn_output, out_proj^T)  [seq_len × hidden_size]
    let mut output = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate GDN final output: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.out_proj_weight, &gdn_output, &mut output,
        seq_len, hidden_size, hidden_size, group_size, int4_companions,
    )?;

    Ok(output)
}

/// Decode-time GDN: recurrent step with updated hidden state.
/// @lat: [[lat.md/lat#Phase 4 Deliverables#Module Structure#GDN Decode Forward Pass]]
///
/// For a single token:
/// 1. a = GEMM(input, in_proj_a^T) → [1 × hidden_size]
/// 2. b = GEMM(input, in_proj_b^T) → [1 × hidden_size]
/// 3. x = GEMM(input, x_proj^T) → [1 × hidden_size]
/// 4. dt = GEMM(input, dt_proj^T) → [1 × hidden_size]
/// 5. Ensure GDN state is allocated
/// 6. Allocate output buffer [hidden_size]
/// 7. Launch infers_gdn_update_bf16 kernel
/// 8. output_proj = GEMM(gdn_output, out_proj^T) → [1 × hidden_size]
///
/// conv1d is skipped (same as forward).
/// TP all-reduce is handled by the caller.
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `int4_kernel` — INT4 GEMM kernel for quantized weights
/// * `stream` — CUDA stream
/// * `gdn_update_kernel` — Loaded CUDA function for `infers_gdn_update_bf16`
/// * `weights` — GDN layer weights
/// * `input` — Single-token input `[1 × hidden_size]`
/// * `gdn_state` — Mutable recurrent state (updated in-place)
/// * `hidden_size` — Model hidden dimension
/// * `group_size` — INT4 quantization group size (typically 128)
/// * `int4_companions` — Companion tensors for INT4 weights
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
    group_size: usize,
    int4_companions: &HashMap<String, Int4Companions>,
) -> Result<CudaSlice<bf16>> {
    // =========================================================================
    // Phase 1: Projection GEMMs (single token, m=1, INT4-aware)
    // =========================================================================

    // a = GEMM(input, in_proj_a^T)  [1 × hidden_size]
    let mut a = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate a buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.in_proj_a, input, &mut a,
        1, hidden_size, hidden_size, group_size, int4_companions,
    )?;

    // b = GEMM(input, in_proj_b^T)  [1 × hidden_size]
    let mut b = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate b buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.in_proj_b, input, &mut b,
        1, hidden_size, hidden_size, group_size, int4_companions,
    )?;

    // x = GEMM(input, x_proj^T)  [1 × hidden_size]
    let mut x = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate x buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.x_proj_weight, input, &mut x,
        1, hidden_size, hidden_size, group_size, int4_companions,
    )?;

    // dt = GEMM(input, dt_proj^T)  [1 × hidden_size]
    let mut dt = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dt buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.dt_proj_weight, input, &mut dt,
        1, hidden_size, hidden_size, group_size, int4_companions,
    )?;

    // =========================================================================
    // Phase 2: GDN update kernel (recurrent state update)
    // =========================================================================

    // Ensure GDN state is allocated
    gdn_state.ensure_allocated(stream, hidden_size)?;

    // Allocate GDN kernel output buffer [hidden_size]
    let mut gdn_output = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate GDN output buffer: {e}"))?;

    // Grid: hidden_size blocks (one per state row), block_size power of 2 up to 256
    let mut block_size: usize = 1;
    while block_size < hidden_size && block_size < 256 {
        block_size *= 2;
    }
    let shared_mem = block_size * std::mem::size_of::<f32>();

    let hidden_size_i32 = hidden_size as i32;

    let config = LaunchConfig {
        grid_dim: (hidden_size as u32, 1, 1),
        block_dim: (block_size as u32, 1, 1),
        shared_mem_bytes: shared_mem as u32,
    };

    // Kernel requires mutable state pointer (IN/OUT).
    let state_ref = gdn_state.state.as_mut()
        .expect("GDN state should be allocated");

    unsafe {
        stream
            .launch_builder(gdn_update_kernel)
            .arg(state_ref)       // state (IN/OUT) H×H
            .arg(&mut gdn_output) // output (OUT) [hidden_size]
            .arg(&a)              // a [hidden_size]
            .arg(&b)              // b [hidden_size]
            .arg(&dt)             // dt [hidden_size]
            .arg(&x)              // x [hidden_size]
            .arg(&hidden_size_i32)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("GDN update kernel launch failed: {e}"))?;
    }

    // =========================================================================
    // Phase 3: Output projection (INT4-aware)
    // =========================================================================

    // output = GEMM(gdn_output, out_proj^T)  [1 × hidden_size]
    let mut output = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate GDN final output: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.out_proj_weight, &gdn_output, &mut output,
        1, hidden_size, hidden_size, group_size, int4_companions,
    )?;

    Ok(output)
}
