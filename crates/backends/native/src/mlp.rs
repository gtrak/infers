//! SwiGLU MLP forward pass.
//!
//! Implements the gated linear unit: `output = SiLU(gate) ⊗ up → down_proj`
//! using cuBLASLt GEMM for projections and custom SiLU kernel for gating.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use infers_cuda::gemm::{GemmConfig, GemmEngine};

/// Execute SwiGLU MLP forward pass.
///
/// Steps:
/// 1. `gate = GEMM(input, gate_proj)`
/// 2. `up = GEMM(input, up_proj)`
/// 3. `silu_out = gate * sigmoid(gate) ⊗ up` (element-wise via SiLU+GLU kernel)
/// 4. `output = GEMM(silu_out, down_proj)`
///
/// # Arguments
/// * `gemm` — cuBLASLt engine for matrix multiplications
/// * `stream` — CUDA stream for kernel launches
/// * `silu_kernel` — Loaded CUDA function handle for `infers_silu_glu_bf16`
/// * `gate_proj` — Gate projection weights `[intermediate_size × hidden_size]`
/// * `up_proj` — Up projection weights `[intermediate_size × hidden_size]`
/// * `down_proj` — Down projection weights `[hidden_size × intermediate_size]`
/// * `input` — Input tensor `[seq_len × hidden_size]`
/// * `hidden_size` — Model hidden dimension
/// * `intermediate_size` — MLP intermediate dimension (typically 4× hidden_size)
///
/// # Returns
/// Output tensor `[seq_len × hidden_size]`
pub fn mlp_forward(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    silu_kernel: &CudaFunction,
    gate_proj: &CudaSlice<bf16>,
    up_proj: &CudaSlice<bf16>,
    down_proj: &CudaSlice<bf16>,
    input: &CudaSlice<bf16>,
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<CudaSlice<bf16>> {
    let seq_len = input.len() / hidden_size;

    // Step 1: gate = GEMM(input, gate_proj)
    let gate_buf_size = seq_len * intermediate_size;
    let mut gate = stream.alloc_zeros::<bf16>(gate_buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate gate buffer: {e}"))?;

    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
            n: intermediate_size,
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
        gate_proj,
        &mut gate,
    )?;

    // Step 2: up = GEMM(input, up_proj)
    let mut up = stream.alloc_zeros::<bf16>(gate_buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate up buffer: {e}"))?;

    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
            n: intermediate_size,
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
        up_proj,
        &mut up,
    )?;

    // Step 3: silu_out = SiLU(gate) ⊗ up
    let mut silu_out = stream
        .alloc_zeros::<bf16>(gate_buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate silu_out buffer: {e}"))?;

    let elem_count_i32 = gate_buf_size as i32;
    let config = LaunchConfig {
        grid_dim: (((gate_buf_size as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(silu_kernel)
            .arg(&gate)
            .arg(&up)
            .arg(&mut silu_out)
            .arg(&elem_count_i32)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("SiLU+GLU kernel launch failed: {e}"))?;
    }

    // Step 4: output = GEMM(silu_out, down_proj^T)
    let output_buf_size = seq_len * hidden_size;
    let mut output = stream
        .alloc_zeros::<bf16>(output_buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate MLP output: {e}"))?;

    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
            n: hidden_size,
            k: intermediate_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        &silu_out,
        down_proj,
        &mut output,
    )?;

    Ok(output)
}
