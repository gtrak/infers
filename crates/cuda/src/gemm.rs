//! cuBLASLt GEMM engine for matrix multiplication.
//!
//! Provides a wrapper around NVIDIA's cuBLASLt library for efficient
//! batched GEMM operations supporting FP16, BF16, and FP32 formats.

use cudarc::cublaslt::safe::{Activation, CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{CudaSlice, CudaStream};
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

    /// Execute an FP32 GEMM: C = alpha * op(A) * op(B) + beta * C.
    pub fn matmul_f32(
        &self,
        config: &GemmConfig,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        gemm_impl(&self.handle, config, a, b, c)
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

    /// Execute an FP16 GEMM: C = alpha * op(A) * op(B) + beta * C.
    pub fn matmul_fp16(
        &self,
        config: &GemmConfig,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        c: &mut CudaSlice<half::f16>,
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
