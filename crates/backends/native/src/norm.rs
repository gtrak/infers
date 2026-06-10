//! RMSNorm kernel dispatch.
//!
//! Applies Root Mean Square Layer Normalization: `output = x * rsqrt(mean(x²) + eps) * weight`
//! using the `infers_rmsnorm_bf16` CUDA kernel.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

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
    kernel: &CudaFunction,
    hidden: &CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    eps: f32,
    hidden_size: usize,
) -> Result<CudaSlice<bf16>> {
    let elem_count = hidden.len();
    anyhow::ensure!(elem_count > 0, "RMSNorm input must not be empty");
    anyhow::ensure!(weight.len() >= hidden_size, "Weight vector too short for hidden_size");

    // Allocate output buffer
    let mut output = stream
        .alloc_zeros::<bf16>(elem_count)
        .map_err(|e| anyhow::anyhow!("Failed to allocate RMSNorm output: {e}"))?;

    let hidden_size_i32 = hidden_size as i32;
    let num_rows = (elem_count / hidden_size) as i32;

    // Launch config: one block per row, 256 threads per block
    let config = LaunchConfig {
        grid_dim: (num_rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (256 * std::mem::size_of::<f32>()) as u32,
    };

    unsafe {
        stream
            .launch_builder(kernel)
            .arg(hidden)
            .arg(weight)
            .arg(&mut output)
            .arg(&hidden_size_i32)
            .arg(&num_rows)
            .arg(&eps)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("RMSNorm kernel launch failed: {e}"))?;
    }

    Ok(output)
}
