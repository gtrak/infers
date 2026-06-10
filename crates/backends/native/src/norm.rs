//! RMSNorm kernel dispatch.
//!
//! Applies Root Mean Square Layer Normalization: `output = x * rsqrt(mean(x²) + eps) * weight`
//! using the `infers_rmsnorm_bf16` CUDA kernel.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream};

/// Apply RMSNorm to a hidden-state tensor.
///
/// Allocates an output buffer on the GPU and launches the `infers_rmsnorm_bf16` kernel.
/// The operation is element-wise: each hidden-dimension slice is normalized independently.
///
/// # Arguments
/// * `stream` — CUDA stream to enqueue the kernel on
/// * `kernel` — Loaded CUDA function handle for `infers_rmsnorm_bf16`
/// * `hidden` — Input hidden-state tensor `[batch_size × hidden_size]`
/// * `weight` — RMSNorm scale weights `[hidden_size]`
/// * `eps` — Numerical stability epsilon (typically 1e-6)
/// * `hidden_size` — Dimension of the hidden state
///
/// # Returns
/// Newly allocated `CudaSlice<bf16>` containing the normalized output
pub fn rms_norm(
    stream: &Arc<CudaStream>,
    _kernel: &CudaFunction,
    hidden: &CudaSlice<bf16>,
    _weight: &CudaSlice<bf16>,
    _eps: f32,
    _hidden_size: usize,
) -> Result<CudaSlice<bf16>> {
    let elem_count = hidden.len();
    anyhow::ensure!(elem_count > 0, "RMSNorm input must not be empty");

    // Kernel launch: stream.launch_builder(kernel).arg(hidden).arg(weight).arg(&mut output).arg(&hidden_size_i32).arg(&eps).launch(config)
    todo!("rms_norm: allocate output buffer, launch infers_rmsnorm_bf16 kernel with hidden, weight, eps, hidden_size, output")
}
