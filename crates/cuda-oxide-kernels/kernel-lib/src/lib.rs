//! Kernel library for infers — cuda-oxide PTX kernels.
//!
//! Kernels are compiled to PTX by rustc-codegen-cuda. Each module group is
//! a separate `#[cuda_module]` block. Shared device helpers live in `shared.rs`.

pub mod shared;

pub mod common_kernels;
pub mod norm_kernels;
pub mod activation_kernels;
pub mod attention_kernels;
pub mod gdn_kernels;
pub mod int4_kernels;
pub mod nvfp4_kernels;
pub mod fp8_kernels;
pub mod bf16_kernels;

