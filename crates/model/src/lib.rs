//! Multi-format model loader for Qwen3.6-27B.
//!
//! Supports BF16, PrismaSCOUT (NVFP4), AutoRound (INT4), and GGUF
//! quantization formats with auto-detection.

pub mod config;
pub mod formats;
pub mod weights;
pub mod loader;
pub mod sharding;
pub mod budget;

pub use config::*;
pub use formats::*;
pub use weights::*;
pub use loader::{build_mtp_weights, load_model, load_safetensors, LoadedModel, ShardIndex};
pub use sharding::*;
pub use budget::*;
