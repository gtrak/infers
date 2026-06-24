//! RMSNorm kernel dispatch.
//!
//! Applies Root Mean Square Layer Normalization: `output = x * rsqrt(mean(x²) + eps) * (1 + weight)`
//! using the `infers_rmsnorm_bf16` CUDA kernel. Qwen3_5RMSNorm uses zero-initialized weights
//! with the additive offset formula `(1 + weight)`, same as Gemma-style RMSNorm.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{OxideKernels, CudaSlice, CudaStream};

/// Apply RMSNorm to a hidden-state tensor.
///
/// Allocates an output buffer on the GPU and launches the `infers_rmsnorm_bf16` kernel.
/// The operation is element-wise: each hidden-dimension slice is normalized independently.
///
/// # Arguments
/// * `stream` — CUDA stream to enqueue the kernel on
/// * `oxide` — Loaded OxideKernels bridge handle for `infers_rmsnorm_bf16`
/// * `hidden` — Input hidden-state tensor `[batch_size × hidden_size]`
/// * `weight` — RMSNorm scale weights `[hidden_size]`
/// * `eps` — Numerical stability epsilon (typically 1e-6)
/// * `hidden_size` — Dimension of the hidden state
///
/// # Returns
/// Newly allocated `CudaSlice<bf16>` containing the normalized output
pub fn rms_norm(
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    hidden: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    eps: f32,
    hidden_size: usize,
) -> Result<CudaSlice<bf16>> {
    let elem_count = hidden.len();
    anyhow::ensure!(elem_count > 0, "RMSNorm input must not be empty");
    anyhow::ensure!(weight.len() >= hidden_size, "Weight vector too short for hidden_size");

    // Allocate output buffer
    let mut output = stream.alloc_zeros::<bf16>(elem_count)
        .map_err(|e| anyhow::anyhow!("Failed to allocate RMSNorm output: {e}"))?;

    oxide.launch_rmsnorm_bf16(stream, hidden, weight, &mut output, hidden_size as u32, eps)?;

    Ok(output)
}


/// Apply RMSNorm to a per-head Q or K vector.
///
/// This is a convenience wrapper around `rms_norm()` with `hidden_size=head_dim`.
/// Each head's Q/K buffer `[seq_len × head_dim]` is normalized independently
/// using the shared norm weight `[head_dim]`.
pub fn rms_norm_per_head(
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    head_tensor: &CudaSlice<bf16>,  // [seq_len × head_dim]
    norm_weight: &CudaSlice<bf16>,  // [head_dim]
    eps: f32,
    head_dim: usize,
) -> Result<CudaSlice<bf16>> {
    rms_norm(stream, oxide, head_tensor, norm_weight, eps, head_dim)
}

/// Apply RMSNorm, writing into a pre-allocated output buffer (zero-allocation variant).
///
/// Same computation as `rms_norm()` but writes into the caller-provided `output` buffer
/// instead of allocating a new one. Used in the decode hot path to avoid per-token allocations.
///
/// # Arguments
/// * `output` — Pre-allocated output buffer, must be same size as `hidden`
pub fn rms_norm_into(
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    output: &mut CudaSlice<bf16>,
    hidden: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    eps: f32,
    hidden_size: usize,
) -> Result<()> {
    let elem_count = hidden.len();
    anyhow::ensure!(elem_count > 0, "RMSNorm input must not be empty");
    anyhow::ensure!(weight.len() >= hidden_size, "Weight vector too short for hidden_size");
    anyhow::ensure!(output.len() >= elem_count, "Output buffer too small: {} < {}", output.len(), elem_count);

    oxide.launch_rmsnorm_bf16(stream, hidden, weight, output, hidden_size as u32, eps)?;

    Ok(())
}
