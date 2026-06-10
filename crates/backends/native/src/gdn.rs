//! Gated DeltaNet (GDN) forward pass.
//!
//! Implements the recurrent linear-attention layer used in the hybrid
//! attention pattern. Uses FlashInfer-style GDN kernels for state update.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_model::{GdnWeights, WeightData};

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

/// Convert [WeightData] bytes to [Vec<bf16>] and upload to GPU.
fn upload_weight(
    stream: &Arc<CudaStream>,
    weight: &WeightData,
) -> Result<CudaSlice<bf16>> {
    let bf16_vec: Vec<bf16> = weight
        .data
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();
    stream
        .clone_htod(&bf16_vec)
        .map_err(|e| anyhow::anyhow!("Failed to upload weight '{}': {}", weight.name, e))
}

/// Prefill-time GDN forward pass.
///
/// Steps:
/// 1. RMSNorm (handled by caller)
/// 2. GDN projections: in_proj_a, in_proj_b, x_proj, dt_proj
/// 3. GDN prefill kernel (gated delta rule state update)
/// 4. Output projection
///
/// conv1d is skipped (not critical for Phase 4.5).
/// TP all-reduce is handled by the caller in prefill.rs.
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `stream` — CUDA stream
/// * `gdn_prefill_kernel` — Loaded CUDA function for `infers_gdn_prefill_bf16`
/// * `weights` — GDN layer weights
/// * `input` — Input tensor `[seq_len × hidden_size]`
/// * `gdn_state` — Mutable state (allocated/updated in-place)
/// * `hidden_size` — Model hidden dimension
///
/// # Returns
/// GDN output `[seq_len × hidden_size]`
pub fn forward(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    gdn_prefill_kernel: &CudaFunction,
    weights: &GdnWeights,
    input: &CudaSlice<bf16>,
    gdn_state: &mut GdnState,
    hidden_size: usize,
) -> Result<CudaSlice<bf16>> {
    let seq_len = input.len() / hidden_size;

    // =========================================================================
    // Phase 1: Projection GEMMs
    // =========================================================================

    // Upload projection weights
    let in_proj_a = upload_weight(stream, &weights.in_proj_a)?;
    let in_proj_b = upload_weight(stream, &weights.in_proj_b)?;
    let x_proj = upload_weight(stream, &weights.x_proj_weight)?;
    let dt_proj = upload_weight(stream, &weights.dt_proj_weight)?;
    let out_proj = upload_weight(stream, &weights.out_proj_weight)?;

    // TODO: conv1d_weight is skipped — not critical for Phase 4.5

    // a = GEMM(input, in_proj_a^T)  [seq_len × hidden_size]
    let mut a = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate a buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
            n: hidden_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        input,
        &in_proj_a,
        &mut a,
    )?;

    // b = GEMM(input, in_proj_b^T)  [seq_len × hidden_size]
    let mut b = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate b buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
            n: hidden_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        input,
        &in_proj_b,
        &mut b,
    )?;

    // x = GEMM(input, x_proj^T)  [seq_len × hidden_size]
    let mut x = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate x buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
            n: hidden_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        input,
        &x_proj,
        &mut x,
    )?;

    // dt = GEMM(input, dt_proj^T)  [seq_len × hidden_size]
    let mut dt = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dt buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
            n: hidden_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        input,
        &dt_proj,
        &mut dt,
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
    // We extract the &mut CudaSlice and pass it directly.
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
    // Phase 3: Output projection
    // =========================================================================

    // output = GEMM(gdn_output, out_proj^T)  [seq_len × hidden_size]
    let mut output = stream
        .alloc_zeros::<bf16>(seq_len * hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate GDN final output: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
            n: hidden_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        &gdn_output,
        &out_proj,
        &mut output,
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
/// * `stream` — CUDA stream
/// * `gdn_update_kernel` — Loaded CUDA function for `infers_gdn_update_bf16`
/// * `weights` — GDN layer weights
/// * `input` — Single-token input `[1 × hidden_size]`
/// * `gdn_state` — Mutable recurrent state (updated in-place)
/// * `hidden_size` — Model hidden dimension
///
/// # Returns
/// GDN output `[1 × hidden_size]`
pub fn decode_forward(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    gdn_update_kernel: &CudaFunction,
    weights: &GdnWeights,
    input: &CudaSlice<bf16>,
    gdn_state: &mut GdnState,
    hidden_size: usize,
) -> Result<CudaSlice<bf16>> {
    // =========================================================================
    // Phase 1: Projection GEMMs (single token, m=1)
    // =========================================================================

    // Upload projection weights
    let in_proj_a = upload_weight(stream, &weights.in_proj_a)?;
    let in_proj_b = upload_weight(stream, &weights.in_proj_b)?;
    let x_proj = upload_weight(stream, &weights.x_proj_weight)?;
    let dt_proj = upload_weight(stream, &weights.dt_proj_weight)?;
    let out_proj = upload_weight(stream, &weights.out_proj_weight)?;

    // a = GEMM(input, in_proj_a^T)  [1 × hidden_size]
    let mut a = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate a buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: 1,
            n: hidden_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        input,
        &in_proj_a,
        &mut a,
    )?;

    // b = GEMM(input, in_proj_b^T)  [1 × hidden_size]
    let mut b = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate b buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: 1,
            n: hidden_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        input,
        &in_proj_b,
        &mut b,
    )?;

    // x = GEMM(input, x_proj^T)  [1 × hidden_size]
    let mut x = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate x buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: 1,
            n: hidden_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        input,
        &x_proj,
        &mut x,
    )?;

    // dt = GEMM(input, dt_proj^T)  [1 × hidden_size]
    let mut dt = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dt buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: 1,
            n: hidden_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        input,
        &dt_proj,
        &mut dt,
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
    // Phase 3: Output projection
    // =========================================================================

    // output = GEMM(gdn_output, out_proj^T)  [1 × hidden_size]
    // gdn_output is [hidden_size], treated as [1 × hidden_size]
    let mut output = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate GDN final output: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: 1,
            n: hidden_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        &gdn_output,
        &out_proj,
        &mut output,
    )?;

    Ok(output)
}
