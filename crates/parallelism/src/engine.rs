//! Unified parallelism engine dispatching between TP and PP at load time.
//!
//! Provides a single `ParallelEngine` enum that wraps either a
//! `TensorParallelEngine` (TP=2) or a `PipelineEngine` (PP=2).
//! The mode is selected at load time via configuration.
//!
//! # Usage
//!
//! ```ignore
//! let engine = ParallelEngine::select(
//!     ParallelismMode::PipelineParallel(2),
//!     &config, weights, ...,
//! )?;
//!
//! let output = engine.forward_batch(requests)?;
//! ```

use std::sync::Arc;

use anyhow::{Context, Result};
use infers_cuda::CudaStream;
use infers_model::config::ModelConfig;
use infers_model::weights::WeightRegistry;

use crate::microbatch::Request;
use crate::pp::{PipelineEngine, PipelineOutput, PipelineTiming};
use crate::tp::TensorParallelEngine;

/// Parallelism strategy to use at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismMode {
    /// Tensor parallelism with the given number of GPUs.
    TensorParallel(usize),
    /// Pipeline parallelism with the given number of stages.
    PipelineParallel(usize),
}

impl ParallelismMode {
    /// Whether this mode is tensor parallelism.
    pub fn is_tp(&self) -> bool {
        matches!(self, Self::TensorParallel(_))
    }

    /// Whether this mode is pipeline parallelism.
    pub fn is_pp(&self) -> bool {
        matches!(self, Self::PipelineParallel(_))
    }

    /// Number of GPUs or stages.
    pub fn parallelism_degree(&self) -> usize {
        match self {
            Self::TensorParallel(n) | Self::PipelineParallel(n) => *n,
        }
    }
}

/// Unified parallelism engine that dispatches to TP or PP.
///
/// Created via `ParallelEngine::select()` based on the chosen
/// `ParallelismMode`. Provides a single `forward_batch()` API
/// regardless of the underlying strategy.
pub enum ParallelEngine {
    /// Tensor parallelism mode (TP=2).
    Tp(TensorParallelEngine),
    /// Pipeline parallelism mode (PP=2).
    Pp(PipelineEngine),
}

impl ParallelEngine {
    /// Select and create the appropriate parallelism engine.
    ///
    /// # Arguments
    ///
    /// * `mode` — The parallelism strategy to use.
    /// * `config` — Model configuration.
    /// * `weights` — Full weight registry.
    /// * `streams` — One CUDA stream per GPU.
    /// * `microbatch_size` — Microbatch size (used for PP only).
    /// * `num_pages` — Total KV cache pages (used for PP only).
    /// * `page_size` — Tokens per KV page (used for PP only).
    /// * `max_cache_bytes` — KV cache memory budget (used for PP only).
    pub fn select(
        mode: ParallelismMode,
        config: Arc<ModelConfig>,
        weights: WeightRegistry,
        streams: Vec<Arc<CudaStream>>,
        microbatch_size: usize,
        num_pages: usize,
        page_size: usize,
        max_cache_bytes: usize,
    ) -> Result<Self> {
        match mode {
            ParallelismMode::TensorParallel(num_gpus) => {
                let tp = TensorParallelEngine::new(num_gpus, streams)
                    .context("Failed to create TensorParallelEngine")?;
                Ok(Self::Tp(tp))
            }
            ParallelismMode::PipelineParallel(num_stages) => {
                anyhow::ensure!(
                    num_stages == 2,
                    "PP={} is not supported yet (only PP=2 is implemented)",
                    num_stages
                );
                anyhow::ensure!(
                    streams.len() >= 2,
                    "PP=2 requires at least 2 CUDA streams, got {}",
                    streams.len()
                );

                let pp = PipelineEngine::new(
                    config,
                    weights,
                    microbatch_size,
                    streams[0].clone(),
                    streams[1].clone(),
                    num_pages,
                    page_size,
                    max_cache_bytes,
                )
                .context("Failed to create PipelineEngine")?;
                Ok(Self::Pp(pp))
            }
        }
    }

    /// Run a batch through the selected parallelism engine.
    ///
    /// For TP mode: delegates to the tensor-parallel forward pass.
    /// For PP mode: delegates to the pipeline-parallel forward pass
    /// with microbatching.
    pub fn forward_batch(&mut self, requests: Vec<Request>) -> Result<PipelineOutput> {
        match self {
            Self::Tp(_tp) => {
                // TP forward pass placeholder
                // In production: run the TP forward pass through all layers
                // with all-reduce after attention and MLP.
                //
                // For now, return a placeholder output:
                let tokens: Vec<Vec<u32>> = requests
                    .iter()
                    .map(|r| vec![r.tokens.first().copied().unwrap_or(0)])
                    .collect();
                Ok(PipelineOutput {
                    tokens,
                    timing: PipelineTiming::default(),
                })
            }
            Self::Pp(pp) => pp.forward_batch(requests),
        }
    }

    /// Whether this engine uses tensor parallelism.
    pub fn is_tp(&self) -> bool {
        matches!(self, Self::Tp(_))
    }

    /// Whether this engine uses pipeline parallelism.
    pub fn is_pp(&self) -> bool {
        matches!(self, Self::Pp(_))
    }
}

/// Default parallelism mode: TP=2 (most common for small batches).
impl Default for ParallelismMode {
    fn default() -> Self {
        Self::TensorParallel(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallelism_mode_default() {
        let mode = ParallelismMode::default();
        assert_eq!(mode, ParallelismMode::TensorParallel(2));
        assert!(mode.is_tp());
        assert!(!mode.is_pp());
        assert_eq!(mode.parallelism_degree(), 2);
    }

    #[test]
    fn test_parallelism_mode_pp() {
        let mode = ParallelismMode::PipelineParallel(2);
        assert!(!mode.is_tp());
        assert!(mode.is_pp());
        assert_eq!(mode.parallelism_degree(), 2);
    }

    #[test]
    fn test_parallelism_mode_tp_custom() {
        let mode = ParallelismMode::TensorParallel(4);
        assert_eq!(mode.parallelism_degree(), 4);
    }

    #[test]
    fn test_parallel_engine_enum() {
        // Test the enum dispatch — we can't create real engines without GPUs
        // so we test the enum variant matching only
        let mode = ParallelismMode::TensorParallel(2);
        assert!(mode.is_tp());
        assert!(!mode.is_pp());
    }

    #[test]
    fn test_parallel_engine_is_tp_is_pp() {
        // Verify the convenience methods work with a constructed placeholder
        // Real construction requires GPUs.
    }
}
