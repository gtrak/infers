//! Element-wise addition for residual connections.
//!
//! Launches the `infers_add_bf16` kernel: `output[i] = a[i] + b[i]`

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{OxideKernels, CudaSlice, CudaStream};

/// Element-wise addition of two BF16 tensors (residual connection).
///
/// Allocates an output buffer and launches the `infers_add_bf16` kernel.
///
/// # Arguments
/// * `stream` — CUDA stream
/// * `oxide` — Loaded OxideKernels bridge handle for `infers_add_bf16`
/// * `a` — First input tensor
/// * `b` — Second input tensor (must be same size as `a`)
///
/// # Returns
/// Newly allocated `CudaSlice<bf16>` containing `a + b`
pub fn add(
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    a: &CudaSlice<bf16>,
    b: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    let elem_count = a.len();
    anyhow::ensure!(elem_count > 0, "Add input must not be empty");
    anyhow::ensure!(a.len() == b.len(), "Add inputs must have same size ({} != {})", a.len(), b.len());

    let mut output = stream.alloc_zeros::<bf16>(elem_count)
        .map_err(|e| anyhow::anyhow!("Failed to allocate add output: {e}"))?;

    oxide.launch_add_bf16(stream, a, b, &mut output)?;

    Ok(output)
}
