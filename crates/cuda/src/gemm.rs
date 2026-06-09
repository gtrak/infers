//! cuBLASLt GEMM engine for matrix multiplication.
//!
//! Provides a wrapper around NVIDIA's cuBLASLt library for efficient
//! batched GEMM operations supporting FP16, BF16, and NVFP4 formats.

/// Configuration for a GEMM operation.
#[derive(Debug, Clone)]
pub struct GemmConfig {
    /// M dimension (rows of A).
    pub m: usize,
    /// N dimension (columns of B / rows of A).
    pub n: usize,
    /// K dimension (columns of A / columns of B).
    pub k: usize,
    /// Whether A is transposed.
    pub transa: bool,
    /// Whether B is transposed.
    pub transb: bool,
    /// Data type for the operation.
    pub dtype: GemmDtype,
}

/// Supported GEMM data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmDtype {
    /// FP16 (half precision).
    Fp16,
    /// BF16 (bfloat16).
    Bf16,
    /// FP32 (single precision).
    Fp32,
    /// NVFP4 (Blackwell 4-bit floating point).
    Nvfp4,
}

#[cfg(feature = "cuda")]
mod cuda_impl {

    /// cuBLASLt GEMM engine.
    ///
    /// Wraps NVIDIA's cuBLASLt library for high-performance matrix multiplication.
    pub struct GemmEngine {
        _handle: cudarc::cublaslt::safe::CudaBlasLT,
    }

    impl GemmEngine {
        /// Create a new GEMM engine. Requires a CUDA stream.
        pub fn new(stream: std::sync::Arc<cudarc::driver::CudaStream>) -> anyhow::Result<Self> {
            let handle = cudarc::cublaslt::safe::CudaBlasLT::new(stream)?;
            Ok(Self { _handle: handle })
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::GemmEngine;

#[cfg(not(feature = "cuda"))]
/// Stub: GemmEngine requires the `cuda` feature.
pub struct GemmEngine;

#[cfg(not(feature = "cuda"))]
impl GemmEngine {
    pub fn new() -> anyhow::Result<Self> {
        anyhow::bail!("GemmEngine requires the 'cuda' feature")
    }
}
