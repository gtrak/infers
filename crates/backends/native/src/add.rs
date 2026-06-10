//! Element-wise addition for residual connections.
//!
//! Launches the `infers_add_bf16` kernel: `output[i] = a[i] + b[i]`

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

/// Element-wise addition of two BF16 tensors (residual connection).
///
/// Allocates an output buffer and launches the `infers_add_bf16` kernel.
///
/// # Arguments
/// * `stream` — CUDA stream
/// * `kernel` — Loaded function handle for `infers_add_bf16`
/// * `a` — First input tensor
/// * `b` — Second input tensor (must be same size as `a`)
///
/// # Returns
/// Newly allocated `CudaSlice<bf16>` containing `a + b`
pub fn add(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    a: &CudaSlice<bf16>,
    b: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    let elem_count = a.len();
    anyhow::ensure!(elem_count > 0, "Add input must not be empty");
    anyhow::ensure!(a.len() == b.len(), "Add inputs must have same size ({} != {})", a.len(), b.len());

    let mut output = stream
        .alloc_zeros::<bf16>(elem_count)
        .map_err(|e| anyhow::anyhow!("Failed to allocate add output: {e}"))?;

    let elem_count_i32 = elem_count as i32;
    let config = LaunchConfig {
        grid_dim: (((elem_count as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(kernel)
            .arg(a)                  // input a
            .arg(b)                  // input b
            .arg(&mut output)        // output
            .arg(&elem_count_i32)   // total_elements
            .launch(config)
            .map_err(|e| anyhow::anyhow!("Add kernel launch failed: {e}"))?;
    }

    Ok(output)
}
