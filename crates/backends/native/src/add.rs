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

    oxide.launch_add_bf16(stream, &oxide.cc_stream(), a, b, &mut output)?;

    Ok(output)
}

/// Element-wise addition, writing into a pre-allocated output buffer (zero-allocation variant).
///
/// Same computation as `add()` but writes into the caller-provided `output` buffer
/// instead of allocating a new one. Used for residual connections in the decode hot path.
///
/// # Arguments
/// * `output` — Pre-allocated output buffer, must be same size as `a` and `b`
pub fn add_into(
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    output: &mut CudaSlice<bf16>,
    a: &CudaSlice<bf16>,
    b: &CudaSlice<bf16>,
) -> Result<()> {
    let elem_count = a.len();
    anyhow::ensure!(elem_count > 0, "Add input must not be empty");
    anyhow::ensure!(a.len() == b.len(), "Add inputs must have same size ({} != {})", a.len(), b.len());
    anyhow::ensure!(output.len() >= elem_count, "Output buffer too small: {} < {}", output.len(), elem_count);

    oxide.launch_add_bf16(stream, &oxide.cc_stream(), a, b, output)?;

    Ok(())
}
