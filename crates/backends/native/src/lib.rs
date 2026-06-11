//! Native inference backend using custom CUDA kernels + cuBLASLt.
//!
//! Implements the Qwen3.6-27B forward pass with:
//! - Custom BF16 kernels for RMSNorm, RoPE, SiLU, embedding
//! - cuBLASLt GEMM for linear projections
//! - INT4 GEMM dispatch for quantized weights (AutoRound/GPTQ)
//! - NCCL all-reduce for tensor parallelism
//! - Hybrid GDN/FullAttention layer dispatch

pub mod add;
pub mod engine;
pub mod prefill;
pub mod decode;
pub mod gdn;
pub mod attention;
pub mod mlp;
pub mod mtp;
pub mod norm;
pub mod rope;
pub mod sample;
pub mod embedding;
pub mod sync;
pub mod upload;
pub mod eviction;
pub mod quant;
pub mod gemm_dispatch;

pub use engine::ForwardEngine;
pub use eviction::BackendEvictionStore;
