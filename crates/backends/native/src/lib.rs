//! Native inference backend using FlashInfer + cuBLASLt.
//!
//! This backend implements the Qwen3.6-27B forward pass using:
//! - FlashInfer GDN kernels for the 48 Gated DeltaNet layers
//! - FlashInfer PagedAttention for the 16 full-attention layers
//! - cuBLASLt GEMM for linear projections
