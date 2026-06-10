//! Forward inference engine — owns GPU state, kernels, and execution logic.
//!
//! `ForwardEngine` is the central struct that coordinates all inference steps:
//! embedding lookup, layer dispatch (GDN vs full attention), MLP, normalization,
//! and sampling. It holds references to CUDA resources, model weights, and kernels.

use std::sync::Arc;

use anyhow::Result;
use infers_cuda::gemm::GemmEngine;
use infers_cuda::kernels::{KernelRegistry, LoadedKernelRegistry};
use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::stream::StreamPool;
use infers_cuda::{CudaContext, CudaStream};
use infers_model::{ModelConfig, WeightRegistry};

/// Central engine for forward-pass inference.
///
/// Owns all GPU resources: CUDA contexts, streams, loaded kernels, cuBLASLt handles,
/// and NCCL communicators. Coordinates the full prefill/decode pipeline.
pub struct ForwardEngine {
    /// Model architecture configuration.
    config: Arc<ModelConfig>,

    /// One weight registry per GPU shard (tensor parallelism).
    weights: Vec<WeightRegistry>,

    /// Loaded CUDA kernel functions (cubin-based).
    kernels: LoadedKernelRegistry,

    /// cuBLASLt GEMM engine.
    gemm: GemmEngine,

    /// NCCL communicator for tensor-parallel all-reduce.
    nccl: NcclCommunicator,

    /// Async CUDA streams for parallel execution.
    streams: StreamPool,
}

impl ForwardEngine {
    /// Construct a `ForwardEngine` from model config, weights, and GPU resources.
    ///
    /// # Arguments
    /// * `config` — Model architecture parameters
    /// * `weights` — Per-GPU weight registries (one per tensor-parallel rank)
    /// * `ctx` — CUDA context for kernel loading
    /// * `kernel_registry` — Names and paths of CUDA kernels to load
    /// * `streams` — Pool of async CUDA streams
    /// * `nccl` — NCCL communicator for multi-GPU collectives
    pub fn new(
        config: Arc<ModelConfig>,
        weights: Vec<WeightRegistry>,
        ctx: Arc<CudaContext>,
        kernel_registry: KernelRegistry,
        streams: StreamPool,
    ) -> Result<Self> {
        let kernels = LoadedKernelRegistry::load_all(ctx, &kernel_registry)
            .map_err(|e| anyhow::anyhow!("Failed to load CUDA kernels: {e}"))?;

        // Create GEMM engine using the first stream
        let default_stream = streams.get(0)
            .ok_or_else(|| anyhow::anyhow!("StreamPool is empty"))?;
        let gemm = GemmEngine::new(default_stream.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create cuBLASLt engine: {e}"))?;

        // Create NCCL communicator for tensor parallelism
        let nccl = {
            let comm_streams: Vec<Arc<CudaStream>> = (0..streams.len())
                .filter_map(|i| streams.get(i).cloned())
                .collect();
            NcclCommunicator::new(comm_streams)
                .map_err(|e| anyhow::anyhow!("Failed to initialize NCCL: {e}"))?
        };

        tracing::info!(
            "ForwardEngine initialized: {} layers, {} GPU shards",
            config.num_hidden_layers,
            weights.len()
        );

        Ok(Self {
            config,
            weights,
            kernels,
            gemm,
            nccl,
            streams,
        })
    }

    /// Run the full prefill pass over a prompt.
    ///
    /// Embeds tokens, iterates through all transformer layers (dispatching
    /// GDN or full-attention based on layer type), applies final norm + LM head,
    /// and samples the first output token.
    ///
    /// # Arguments
    /// * `stream` — CUDA stream for kernel launches
    /// * `token_ids` — Input token IDs (prompt)
    ///
    /// # Returns
    /// The sampled token ID for the first generated token, or an error if GPU execution fails.
    pub fn prefill(&mut self, _stream: &Arc<CudaStream>, _token_ids: &[u32]) -> Result<u32> {
        // 1. Embed prompt tokens via embedding gather kernel
        // 2. Loop through layers, dispatching GDN vs full attention
        // 3. Apply final norm + LM head projection
        // 4. Sample first token via greedy argmax
        todo!("prefill: embed prompt tokens, run layer loop (GDN + full attention dispatch), apply final norm + LM head, sample first token")
    }

    /// Run single-token decode step.
    ///
    /// Executes: embed single token → layer loop → final norm → LM head → sample
    ///
    /// # Arguments
    /// * `stream` — CUDA stream for kernel launches
    /// * `token_id` — Previous token ID to continue generation
    /// * `position` — Current position in the sequence (for RoPE)
    ///
    /// # Returns
    /// The sampled token ID for the next generated token, or an error if GPU execution fails.
    pub fn decode(&mut self, _stream: &Arc<CudaStream>, _token_id: u32, _position: u32) -> Result<u32> {
        // 1. Embed single token via embedding gather kernel
        // 2. Decode layer loop (single-token width) with KV cache updates
        // 3. Apply final norm + LM head projection
        // 4. Sample next token via greedy argmax
        todo!("decode: embed single token, run layer loop with KV cache, apply final norm + LM head, sample next token")
    }
}
