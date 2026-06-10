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

use crate::attention::{KvCache, PagedKvCache};
use crate::gdn::GdnState;

use infers_kv::PagedKvManager;

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


    /// Paged KV cache write kernel. Used by paged pipeline (wiring in progress).
    #[allow(dead_code)] // TODO: remove once paged pipeline is fully wired
    paged_kv_write_kernel: CudaFunction,
    /// Paged KV cache read kernel. Used by paged pipeline (wiring in progress).
    #[allow(dead_code)] // TODO: remove once paged pipeline is fully wired
    paged_kv_read_kernel: CudaFunction,
    /// Paged attention decode kernel. Used by paged pipeline (wiring in progress).
    #[allow(dead_code)] // TODO: remove once paged pipeline is fully wired
    paged_attention_decode_kernel: CudaFunction,

    /// Paged KV cache manager (pool + prefix cache + COW).
    paged_kv_manager: Option<PagedKvManager>,

    /// cuBLASLt GEMM engine.
    gemm: GemmEngine,

    /// NCCL communicator for tensor-parallel all-reduce.
    nccl: NcclCommunicator,

    /// Async CUDA streams for parallel execution.
    /// Retained for future multi-stream execution
    #[allow(dead_code)]
    streams: StreamPool,

    /// Per-layer KV caches for full-attention layers (flat cache, legacy).
    kv_caches: Vec<KvCache>,
    /// Per-layer paged KV caches (new paged system).
    paged_kv_caches: Vec<PagedKvCache>,

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

        // Resolve paged attention kernel handles
        let paged_kv_write_kernel = kernels.get_function("infers_paged_kv_write_bf16")?;
        let paged_kv_read_kernel = kernels.get_function("infers_paged_kv_read_bf16")?;
        let paged_attention_decode_kernel = kernels.get_function("infers_paged_attention_decode_bf16")?;
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
        let paged_kv_caches: Vec<PagedKvCache> = Vec::new();

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
            paged_kv_write_kernel,
            paged_kv_read_kernel,
            paged_attention_decode_kernel,
            paged_kv_manager: None,
            gemm,
            nccl,
            streams,
            kv_caches,
            paged_kv_caches,
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

    /// Initialize the paged KV cache system.
    ///
    /// Creates a `PagedKvManager` with the given configuration and
    /// per-layer `PagedKvCache` instances for all full-attention layers.
    ///
    /// # Arguments
    /// * `total_pages` — Total number of physical pages in the pool.
    /// * `page_size` — Number of tokens per page.
    /// * `max_cache_bytes` — Memory budget for the prefix cache.
    pub fn init_paged(
        &mut self,
        total_pages: usize,
        page_size: usize,
        max_cache_bytes: usize,
    ) -> Result<()> {
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Create the paged KV manager
        let eviction_max_bytes = total_pages * page_size * num_kv_heads * head_dim * 2;
        let manager = PagedKvManager::new(
            total_pages,
            page_size,
            num_kv_heads,
            head_dim,
            max_cache_bytes,
            eviction_max_bytes,
        );

        // Create per-layer paged KV caches for all layers
        let caches: Vec<PagedKvCache> = (0..self.config.num_hidden_layers)
            .map(|_| PagedKvCache::new(total_pages, page_size, kv_dim))
            .collect();

        self.paged_kv_manager = Some(manager);
        self.paged_kv_caches = caches;

        tracing::info!(
            "Paged KV system initialized: {} pages, page_size={}, {} layers",
            total_pages,
            page_size,
            self.config.num_hidden_layers
        );

        Ok(())
    }

    /// Run paged prefill — writes K/V to paged cache for all layers.
    ///
    /// Allocates pages for the sequence, uploads the block table to GPU,
    /// and writes K/V data into the paged cache using the paged KV write kernel.
    ///
    /// # Arguments
    /// * `stream` — CUDA stream for kernel launches
    /// * `token_ids` — Input token IDs (prompt)
    /// * `seq_id` — Sequence ID from PagedKvManager
    ///
    /// # Returns
    /// The number of pages allocated for the sequence.
    #[allow(unused_variables)]
    pub fn prefill_paged(
        &mut self,
        stream: &Arc<CudaStream>,
        token_ids: &[u32],
        seq_id: infers_kv::SequenceId,
    ) -> Result<usize> {
        let manager = self.paged_kv_manager.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?;

        let page_size = manager.page_size();
        let _num_kv_heads = self.config.num_key_value_heads;
        let _head_dim = self.config.head_dim;
        let _kv_dim = _num_kv_heads * _head_dim;

        // Allocate pages for the sequence based on token count
        let num_pages_needed = (token_ids.len().saturating_sub(1) / page_size) + 1;
        for _ in 0..num_pages_needed {
            manager.append_page(seq_id)?;
        }

        // Get block table and upload to GPU
        let block_table = manager.block_table(seq_id)?;
        let block_table_i32: Vec<i32> = block_table.iter().map(|p| *p as i32).collect();
        let block_table_gpu = stream
            .clone_htod(&block_table_i32)
            .map_err(|e| anyhow::anyhow!("Failed to upload block table: {e}"))?;

        // Upload positions to GPU
        let positions: Vec<u32> = (0..token_ids.len() as u32).collect();
        let positions_gpu = stream
            .clone_htod(&positions)
            .map_err(|e| anyhow::anyhow!("Failed to upload positions: {e}"))?;

        // Ensure page pools are allocated for each layer
        for cache in &mut self.paged_kv_caches {
            cache.ensure_allocated(stream)?;
        }

        // TODO: Full prefill pipeline with embedding, layer loop, projections, RoPE,
        //       and paged KV write for each layer. The paged_kv_write kernel is ready;
        //       the full integration requires the layer-by-layer dispatch from prefill.rs.
        //       For now, allocate pages and upload block table.

        Ok(num_pages_needed)
    }

    /// Run paged single-token decode — zero CPU round-trips.
    ///
    /// Reads K/V from the paged cache, computes attention, and returns
    /// the sampled token. Unlike the legacy `decode`, this path uses
    /// paged attention kernels that operate entirely on GPU.
    ///
    /// # Arguments
    /// * `stream` — CUDA stream for kernel launches
    /// * `token_id` — Previous token ID to continue generation
    /// * `position` — Current position in the sequence (for RoPE)
    /// * `seq_id` — Sequence ID from PagedKvManager
    ///
    /// # Returns
    /// The sampled token ID for the next generated token.
    #[allow(unused_variables)]
    pub fn decode_paged(
        &mut self,
        stream: &Arc<CudaStream>,
        token_id: u32,
        position: u32,
        seq_id: infers_kv::SequenceId,
    ) -> Result<u32> {
        let manager = self.paged_kv_manager.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?;

        // Get block table and upload to GPU
        let block_table = manager.block_table(seq_id)?;
        let block_table_i32: Vec<i32> = block_table.iter().map(|p| *p as i32).collect();
        let block_table_gpu = stream
            .clone_htod(&block_table_i32)
            .map_err(|e| anyhow::anyhow!("Failed to upload block table: {e}"))?;

        let num_pages = manager.num_pages(seq_id)?;
        let num_cached_tokens = manager.num_tokens(seq_id)?;
        let page_size = manager.page_size();

        // Ensure page pools are allocated for each layer
        for cache in &mut self.paged_kv_caches {
            cache.ensure_allocated(stream)?;
        }

        // TODO: Full decode pipeline with embedding, layer loop, Q projection,
        //       paged attention read/decode kernels, MLP, norm, and sampling.
        //       The paged_kv_read and paged_attention_decode kernels are ready;
        //       the full integration requires the layer-by-layer dispatch from decode.rs.

        // Placeholder: return the input token_id until full pipeline is wired.
        Ok(token_id)
    }
}
