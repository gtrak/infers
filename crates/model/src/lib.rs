//! Multi-format model loader for Qwen3.6-27B.
//!
//! Supports BF16, PrismaSCOUT (NVFP4), AutoRound (INT4), and GGUF
//! quantization formats with auto-detection.

pub mod config;
pub mod formats;
pub mod weights;
pub mod loader;
pub mod mmap;
pub mod sharding;
pub mod budget;

pub use config::*;
pub use formats::*;
pub use weights::*;
pub use loader::{build_main_layers, build_mtp_weights, strip_language_model_prefix};
pub use sharding::*;
pub use budget::*;
pub use mmap::{MmapTensor, MmapWeightRegistry, MmapCompanions, MmapWeightShard, load_safetensors_mmap, strip_language_model_prefix_mmap, shard_weights_tp_mmap, build_metadata_registry};
