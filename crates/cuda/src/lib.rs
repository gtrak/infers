//! CUDA runtime for inference: context management, streams, memory allocation,
//! kernel loading, cuBLASLt GEMM, and NCCL communication.
//!
//! Requires the `cuda` feature to be enabled for actual GPU operations.
//! Without it, provides stub types that panic at runtime.

pub mod context;
pub mod stream;
pub mod memory;
pub mod kernels;
pub mod gemm;
pub mod nccl;
