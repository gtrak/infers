//! Tensor parallelism (TP=2) and pipeline parallelism (PP=2) implementations.
//!
//! TP=2: NCCL all-reduce after attention and MLP layers.
//! PP=2: P2P send/recv between pipeline stages with microbatching.

pub mod comm;
pub mod microbatch;
pub mod pp;
pub mod stage;
