//! Multi-format model loader for Qwen3.6-27B.
//!
//! Supports BF16, PrismaSCOUT (NVFP4), AutoRound (INT4), and GGUF
//! quantization formats with auto-detection.

pub mod config;
pub mod formats;

pub use config::*;
pub use formats::*;
