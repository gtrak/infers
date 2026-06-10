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
use infers_cuda::{CudaContext, CudaFunction, CudaStream};
use infers_model::{ModelConfig, WeightRegistry};

use crate::attention::KvCache;
use crate::gdn::GdnState;

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
    /// Retained for future multi-kernel lookup
    #[allow(dead_code)]
    kernels: LoadedKernelRegistry,

    /// Cached kernel function handles (resolved from LoadedKernelRegistry at init time).
    rmsnorm_kernel: CudaFunction,
    silu_glu_kernel: CudaFunction,
    rope_kernel: CudaFunction,
    embedding_kernel: CudaFunction,
    add_kernel: CudaFunction,
    argmax_kernel: CudaFunction,
    softmax_kernel: CudaFunction,
    kv_cache_write_kernel: CudaFunction,
    gdn_prefill_kernel: CudaFunction,
    gdn_update_kernel: CudaFunction,

    /// cuBLASLt GEMM engine.
    gemm: GemmEngine,

    /// NCCL communicator for tensor-parallel all-reduce.
    nccl: NcclCommunicator,

    /// Async CUDA streams for parallel execution.
    /// Retained for future multi-stream execution
    #[allow(dead_code)]
    streams: StreamPool,

    /// Per-layer KV caches for full-attention layers.
    kv_caches: Vec<KvCache>,

    /// Per-layer GDN recurrent states.
    gdn_states: Vec<GdnState>,
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

        // Resolve kernel function handles from loaded modules
        let rmsnorm_kernel = kernels.get_function("infers_rmsnorm_bf16")?;
        let silu_glu_kernel = kernels.get_function("infers_silu_glu_bf16")?;
        let rope_kernel = kernels.get_function("infers_rope_bf16")?;
        let embedding_kernel = kernels.get_function("infers_embedding_gather_bf16")?;
        let add_kernel = kernels.get_function("infers_add_bf16")?;
        let argmax_kernel = kernels.get_function("infers_argmax_f32")?;
        let softmax_kernel = kernels.get_function("infers_softmax_bf16")?;
        let kv_cache_write_kernel = kernels.get_function("infers_kv_cache_write_bf16")?;
        let gdn_prefill_kernel = kernels.get_function("infers_gdn_prefill_bf16")?;
        let gdn_update_kernel = kernels.get_function("infers_gdn_update_bf16")?;

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

        // Initialize per-layer caches and states
        let kv_caches: Vec<KvCache> = (0..config.num_hidden_layers).map(|_| KvCache::new()).collect();
        let gdn_states: Vec<GdnState> = (0..config.num_hidden_layers).map(|_| GdnState::new()).collect();

        tracing::info!(
            "ForwardEngine initialized: {} layers, {} GPU shards",
            config.num_hidden_layers,
            weights.len()
        );

        Ok(Self {
            config,
            weights,
            kernels,
            rmsnorm_kernel,
            silu_glu_kernel,
            rope_kernel,
            embedding_kernel,
            add_kernel,
            argmax_kernel,
            softmax_kernel,
            kv_cache_write_kernel,
            gdn_prefill_kernel,
            gdn_update_kernel,
            gemm,
            nccl,
            streams,
            kv_caches,
            gdn_states,
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
    pub fn prefill(&mut self, stream: &Arc<CudaStream>, token_ids: &[u32]) -> Result<u32> {
        let gpu_id = 0; // Single GPU for now
        let weights = &self.weights[gpu_id];

        let kernels = crate::prefill::PrefillKernels {
            rmsnorm: self.rmsnorm_kernel.clone(),
            silu_glu: self.silu_glu_kernel.clone(),
            rope: self.rope_kernel.clone(),
            embedding: self.embedding_kernel.clone(),
            add: self.add_kernel.clone(),
            argmax: self.argmax_kernel.clone(),
            softmax: self.softmax_kernel.clone(),
            kv_cache_write: self.kv_cache_write_kernel.clone(),
            gdn_prefill: self.gdn_prefill_kernel.clone(),
        };

        crate::prefill::prefill(
            &mut self.gemm, stream, &kernels, &self.nccl,
            &self.config, weights, token_ids,
            &mut self.kv_caches, &mut self.gdn_states,
        )
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
    pub fn decode(&mut self, stream: &Arc<CudaStream>, token_id: u32, position: u32) -> Result<u32> {
        let gpu_id = 0;
        let weights = &self.weights[gpu_id];

        let kernels = crate::decode::DecodeKernels {
            rmsnorm: self.rmsnorm_kernel.clone(),
            silu_glu: self.silu_glu_kernel.clone(),
            rope: self.rope_kernel.clone(),
            embedding: self.embedding_kernel.clone(),
            add: self.add_kernel.clone(),
            argmax: self.argmax_kernel.clone(),
            softmax: self.softmax_kernel.clone(),
            kv_cache_write: self.kv_cache_write_kernel.clone(),
            gdn_update: self.gdn_update_kernel.clone(),
        };

        crate::decode::decode(
            &mut self.gemm, stream, &kernels, &self.nccl,
            &self.config, weights, token_id, position,
            &mut self.kv_caches, &mut self.gdn_states,
        )
    }
}
