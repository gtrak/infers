//! cuBLASLt GEMM engine for matrix multiplication.
//!
//! Provides a wrapper around NVIDIA's cuBLASLt library for efficient
//! batched GEMM operations supporting FP16, BF16, and FP32 formats.

use cudarc::cublaslt::safe::{Activation, CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Configuration for a GEMM operation.
#[derive(Debug, Clone)]
pub struct GemmConfig {
    /// M dimension (rows of output C = A @ B).
    pub m: usize,
    /// N dimension (columns of output C = A @ B).
    pub n: usize,
    /// K dimension (inner dimension).
    pub k: usize,
    /// Whether A is transposed.
    pub transa: bool,
    /// Whether B is transposed.
    pub transb: bool,
    /// Alpha scalar (multiply A@B before adding to C).
    pub alpha: f32,
    /// Beta scalar (multiply C before adding).
    pub beta: f32,
    /// Leading dimension of A (row stride in memory). Default: inferred.
    pub lda: Option<i64>,
    /// Leading dimension of B. Default: inferred.
    pub ldb: Option<i64>,
    /// Leading dimension of C. Default: inferred.
    pub ldc: Option<i64>,
    /// Optional activation to fuse after matmul.
    pub activation: Option<Activation>,
}

/// cuBLASLt GEMM engine.
///
/// Wraps NVIDIA's cuBLASLt library for high-performance matrix multiplication.
/// The handle is created eagerly upon construction.
pub struct GemmEngine {
    handle: CudaBlasLT,
}

impl GemmEngine {
    /// Create a new GEMM engine with a cuBLASLt handle.
    ///
    /// Requires an active CUDA stream — the handle is tied to the stream's device.
    pub fn new(stream: Arc<CudaStream>) -> anyhow::Result<Self> {
        let handle = CudaBlasLT::new(stream)
            .map_err(|e| anyhow::anyhow!("Failed to create CudaBlasLT handle: {:?}", e))?;
        Ok(Self { handle })
    }

    /// Execute a BF16 GEMM: C = alpha * op(A) * op(B) + beta * C.
    pub fn matmul_bf16(
        &self,
        config: &GemmConfig,
        a: &CudaSlice<half::bf16>,
        b: &CudaSlice<half::bf16>,
        c: &mut CudaSlice<half::bf16>,
    ) -> anyhow::Result<()> {
        gemm_impl(&self.handle, config, a, b, c)
    }

 }

/// Internal helper that builds a `MatmulConfig` from `GemmConfig` and
/// calls the unsafe cuBLASLt `matmul`. Validates dimensions first.
fn gemm_impl<T>(
    handle: &CudaBlasLT,
    config: &GemmConfig,
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    c: &mut CudaSlice<T>,
) -> anyhow::Result<()>
where
    CudaBlasLT: Matmul<T>,
{
    anyhow::ensure!(
        config.m > 0 && config.n > 0 && config.k > 0,
        "GEMM dimensions must all be positive"
    );

    let lda = config.lda.unwrap_or({
        if config.transa { config.k as i64 } else { config.m as i64 }
    });
    let ldb = config.ldb.unwrap_or({
        if config.transb { config.n as i64 } else { config.k as i64 }
    });
    let ldc = config.ldc.unwrap_or(config.m as i64);

    let matmul_config = MatmulConfig {
        transa: config.transa,
        transb: config.transb,
        transc: false,
        m: config.m as u64,
        n: config.n as u64,
        k: config.k as u64,
        alpha: config.alpha,
        beta: config.beta,
        lda,
        ldb,
        ldc,
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
    };

    unsafe {
        handle
            .matmul(matmul_config, a, b, c, None::<&CudaSlice<T>>, config.activation.as_ref())
            .map_err(|e| anyhow::anyhow!("cuBLASLt matmul failed: {:?}", e))?;
    }

    Ok(())
}

/// Configuration for INT4 GEMM with per-group dequantization.
///
/// Weights are stored in INT4-packed format with per-group scale
/// and zero-point. Dequantization happens on-the-fly in registers.
///
/// Note: K must be divisible by group_size for the INT4 path.
/// For the BF16 (cuBLASLt) path, K can be any value.
#[derive(Debug, Clone)]
pub struct Int4GemmConfig {
    /// M dimension (rows of output from input).
    pub m: usize,
    /// N dimension (columns of output, rows of weight).
    pub n: usize,
    /// K dimension (inner dimension, columns of input).
    pub k: usize,
    /// Quantization group size (typically 128).
    pub group_size: usize,
    /// Whether weight is in transposed [K/8, N] layout (true) or standard [N, K/8] (false).
    pub transposed: bool,
}

/// Execute INT4 GEMM with on-the-fly dequantization.
///
/// Computes: output[M][N] = dequant(weight[N][K]) @ input[M][K]
///
/// Weights stay in INT4-packed format — no dequantized copy exists.
/// Dequantization happens in registers during the inner loop.
///
/// # Arguments
/// * `stream` — CUDA stream to launch on
/// * `kernel` — The `int4_gemm_kernel` CudaFunction handle
/// * `config` — M, N, K, group_size, transposed dimensions
/// * `output` — [M, N] BF16 output buffer
/// * `weight` — [N, K/8] packed INT4 weights
/// * `scales` — [N, K/group_size] FP16 group scales
/// * `zeros`  — [N, K/group_size/8] packed INT4 zero points
/// * `input`  — [M, K] BF16 input activations
pub fn matmul_int4(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    config: &Int4GemmConfig,
    output: &mut CudaSlice<half::bf16>,
    weight: &CudaSlice<u32>,
    scales: &CudaSlice<half::f16>,
    zeros: &CudaSlice<u32>,
    input: &CudaSlice<half::bf16>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.m > 0 && config.n > 0 && config.k > 0,
        "INT4 GEMM dimensions must all be positive"
    );
    anyhow::ensure!(
        config.group_size > 0 && config.k % config.group_size == 0,
        "K must be divisible by group_size"
    );

    // Launch config: for m=1 (single-token decode) use a 1D grid with 256
    // threads per block so every thread does useful work. For m>1 (prefill)
    // use 16x16 tiling where each thread computes one output element.
    // This avoids 93% wasted threads for the common m=1 decode case.
    let (tx, ty) = if config.m == 1 {
        (256u32, 1u32)
    } else {
        (16u32, 16u32)
    };
    let launch_config = LaunchConfig {
        grid_dim: (
            (config.n as u32 + tx - 1) / tx,
            (config.m as u32 + ty - 1) / ty,
            1,
        ),
        block_dim: (tx, ty, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let _ = stream
            .launch_builder(kernel)
            .arg(output)
            .arg(weight)
            .arg(scales)
            .arg(zeros)
            .arg(input)
            .arg(&(config.m as i32))
            .arg(&(config.n as i32))
            .arg(&(config.k as i32))
            .arg(&(config.group_size as i32))
            .arg(&(if config.transposed { 1i32 } else { 0i32 }))
            .launch(launch_config)
            .map_err(|e| anyhow::anyhow!("int4_gemm_kernel launch failed: {:?}", e))?;
    }

    Ok(())
}
