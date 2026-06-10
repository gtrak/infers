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

/// cuBLASLt GEMM engine.
///
/// Wraps NVIDIA's cuBLASLt library for high-performance matrix multiplication.
pub struct GemmEngine {
    /// Lazily created cuBLASLt handle.
    handle: Option<cudarc::cublaslt::safe::CudaBlasLT>,
}

impl GemmEngine {
    /// Create a new GEMM engine.
    ///
    /// The underlying cuBLASLt handle is created lazily on the first
    /// `matmul` call.
    pub fn new() -> Self {
        Self { handle: None }
    }

    /// Execute a GEMM operation with the given configuration.
    ///
    /// The stream is passed to cuBLASLt so the kernel is enqueued on the
    /// correct async execution context.  The first call lazily creates
    /// the `CudaBlasLT` handle; subsequent calls reuse the cached handle.
    pub fn matmul(
        &mut self,
        _config: &GemmConfig,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> anyhow::Result<()> {
        if self.handle.is_none() {
            self.handle = Some(
                cudarc::cublaslt::safe::CudaBlasLT::new(std::sync::Arc::clone(stream))
                    .map_err(|e| anyhow::anyhow!("Failed to create CudaBlasLT: {:?}", e))?,
            );
        }
        let handle = self.handle.as_ref().unwrap();
        // TODO: actual matmul call using `handle`
        let _ = handle;
        anyhow::bail!("GemmEngine::matmul not yet implemented")
    }
}

impl Default for GemmEngine {
    fn default() -> Self {
        Self::new()
    }
}
