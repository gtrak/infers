//! Gated DeltaNet (GDN) forward pass.
//!
//! Implements the recurrent linear-attention layer used in the hybrid
//! attention pattern. Uses FlashInfer-style GDN kernels for state update.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaSlice, CudaStream};
use infers_cuda::gemm::GemmEngine;
use infers_model::GdnWeights;

/// GDN recurrent state, maintained across decode steps.
///
/// Stores the hidden state for each GDN layer. During prefill, this is
/// initialized from scratch; during decode, it is updated token by token.
#[derive(Debug)]
pub struct GdnState {
    /// Recurrent state `[batch_size × hidden_state_size]`
    _state: Option<CudaSlice<bf16>>,
}

impl GdnState {
    /// Create an empty GDN state.
    pub fn new() -> Self {
        Self { _state: None }
    }
}

/// Prefill-time GDN forward pass.
///
/// Steps:
/// 1. RMSNorm (handled by caller)
/// 2. GDN projections: in_proj_a, in_proj_b, conv1d, x_proj, dt_proj
/// 3. Chunked gated delta rule state update
/// 4. Output projection
/// 5. Tensor-parallel all-reduce
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `stream` — CUDA stream
/// * `weights` — GDN layer weights
/// * `input` — Input tensor `[seq_len × hidden_size]`
/// * `hidden_size` — Model hidden dimension
///
/// # Returns
/// GDN output `[seq_len × hidden_size]`
pub fn forward(
    _gemm: &mut GemmEngine,
    _stream: &Arc<CudaStream>,
    _weights: &GdnWeights,
    _input: &CudaSlice<bf16>,
    _gdn_state: &mut GdnState,
    _hidden_size: usize,
) -> Result<CudaSlice<bf16>> {
    // Phase 1: Projection matrices
    // a = GEMM(input, in_proj_a)
    // b = GEMM(input, in_proj_b)
    // x_proj = GEMM(input, x_proj_weight)
    // dt_proj = GEMM(input, dt_proj_weight)
    // conv1d_output = conv1d(input, conv1d_weight)

    // Phase 2: Gated delta rule state update
    // Uses FlashInfer GDN kernel for efficient chunked recurrence

    // Phase 3: Output projection
    // output = GEMM(gdn_out, out_proj_weight)

    // Phase 4: TP all-reduce for output
    todo!("GDN forward: projections → conv1d → gated delta rule → output projection → TP all-reduce")
}

/// Decode-time GDN: recurrent step with updated hidden state.
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `stream` — CUDA stream
/// * `weights` — GDN layer weights
/// * `input` — Single-token input `[1 × hidden_size]`
/// * `gdn_state` — Mutable recurrent state (updated in-place)
/// * `hidden_size` — Model hidden dimension
///
/// # Returns
/// GDN output `[1 × hidden_size]`
pub fn decode_forward(
    _gemm: &mut GemmEngine,
    _stream: &Arc<CudaStream>,
    _weights: &GdnWeights,
    _input: &CudaSlice<bf16>,
    _gdn_state: &mut GdnState,
    _hidden_size: usize,
) -> Result<CudaSlice<bf16>> {
    // Phase 1: Project single token
    // a = GEMM(input, in_proj_a)
    // b = GEMM(input, in_proj_b)

    // Phase 2: Recurrent state update
    // state = gated_delta_rule(state, a, b, dt, x)

    // Phase 3: Output projection
    // output = GEMM(state, out_proj_weight)
    todo!("GDN decode: single-token x_proj → dt_proj → in_proj → recurrent state update → out_proj → TP all-reduce")
}
