//! Forward inference engine — owns GPU state, kernels, and execution logic.
//!
//! `ForwardEngine` is the central struct that coordinates all inference steps:
//! embedding lookup, layer dispatch (GDN vs full attention), MLP, normalization,
//! and sampling. It holds references to CUDA resources, model weights, and kernels.

use std::sync::Arc;

use anyhow::Result;
use infers_cuda::gemm::GemmEngine;
use infers_cuda::gemm::Int4GemmConfig;
use infers_cuda::kernels::{KernelRegistry, LoadedKernelRegistry};
use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::stream::StreamPool;
use infers_cuda::{CudaContext, CudaFunction, CudaStream, PushKernelArg};
use infers_model::{LayerType, ModelConfig, WeightRegistry};

use crate::attention::{KvCache, PagedKvCache};
use crate::eviction::BackendEvictionStore;
use crate::gdn::GdnState;

use infers_kv::PagedKvManager;

use half::bf16;
use infers_kv::KvCacheDtype;
use infers_cuda::CudaSlice;
use infers_model::MtpWeights;

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

    /// FP8 quantize kernel: BF16→FP8 on GPU.
    #[allow(dead_code)]
    fp8_quantize_kernel: CudaFunction,
    /// FP8 dequantize kernel: FP8→BF16 on GPU.
    #[allow(dead_code)]
    fp8_dequantize_kernel: CudaFunction,

    /// INT4 GEMM kernel: matmul with on-the-fly per-group dequantization.
    #[allow(dead_code)]
    int4_gemm_kernel: CudaFunction,

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

    /// INT4 quantization group size for on-the-fly dequantization.
    group_size: usize,
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
        group_size: usize,
    ) -> Result<Self> {
        let kernels = LoadedKernelRegistry::load_all(ctx, &kernel_registry)
            .map_err(|e| anyhow::anyhow!("Failed to load CUDA kernels: {e}"))?;

        // Resolve kernel function handles from loaded modules
        let rmsnorm_kernel = kernels.get_function("infers_rmsnorm_bf16")?;
        let silu_glu_kernel = kernels.get_function("infers_silu_glu_bf16")?;
        let rope_kernel = kernels.get_function("infers_rope_bf16")?;
        let embedding_kernel = kernels.get_function("infers_embedding_gather_bf16")?;
        let add_kernel = kernels.get_function("infers_add_bf16")?;
        let argmax_kernel = kernels.get_function("infers_argmax_bf16")?;
        let softmax_kernel = kernels.get_function("infers_softmax_bf16")?;
        let kv_cache_write_kernel = kernels.get_function("infers_kv_cache_write_bf16")?;
        let gdn_prefill_kernel = kernels.get_function("infers_gdn_mamba2_prefill_bf16")?;
        let gdn_update_kernel = kernels.get_function("infers_gdn_mamba2_update_bf16")?;

        // Resolve paged attention kernel handles
        let paged_kv_write_kernel = kernels.get_function("infers_paged_kv_write_bf16")?;
        let paged_kv_read_kernel = kernels.get_function("infers_paged_kv_read_bf16")?;
        let paged_attention_decode_kernel = kernels.get_function("infers_paged_attention_decode_bf16")?;

        // Resolve quantization kernel handles
        let fp8_quantize_kernel = kernels.get_function("infers_fp8_quantize_bf16")?;
        let fp8_dequantize_kernel = kernels.get_function("infers_fp8_dequantize_bf16")?;
        let int4_gemm_kernel = kernels.get_function("int4_gemm_kernel")?;

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
            fp8_quantize_kernel,
            fp8_dequantize_kernel,
            int4_gemm_kernel,
            paged_kv_manager: None,
            gemm,
            nccl,
            streams,
            kv_caches,
            paged_kv_caches,
            gdn_states,
            group_size,
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
            int4_gemm: self.int4_gemm_kernel.clone(),
        };

        crate::prefill::prefill(
            &mut self.gemm, stream, &kernels, &self.nccl,
            &self.config, weights, token_ids,
            &mut self.kv_caches, &mut self.gdn_states,
            self.group_size,
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
            int4_gemm: self.int4_gemm_kernel.clone(),
        };

        crate::decode::decode(
            &mut self.gemm, stream, &kernels, &self.nccl,
            &self.config, weights, token_id, position,
            &mut self.kv_caches, &mut self.gdn_states,
            self.group_size,
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

    /// Write FP8-quantized K/V to a page pool using GPU kernels.
    ///
    /// GPU-only: quantizes BF16→FP8 on device, copies into page pool via D2D memcpy.
    /// See [`attention::fp8_quantize_and_write`] for details.
    pub fn fp8_quantize_and_write(
        &self,
        stream: &Arc<CudaStream>,
        page_pool: &mut CudaSlice<u8>,
        page_id: usize,
        page_offset: usize,
        page_size: usize,
        kv_dim: usize,
        dtype: KvCacheDtype,
        k: &CudaSlice<half::bf16>,
        v: &CudaSlice<half::bf16>,
    ) -> Result<()> {
        crate::attention::fp8_quantize_and_write(
            stream,
            &self.fp8_quantize_kernel,
            page_pool,
            page_id,
            page_offset,
            page_size,
            kv_dim,
            dtype,
            k,
            v,
        )
    }

    /// Read FP8-quantized K/V from a page pool using GPU kernels.
    ///
    /// GPU-only: copies from page pool via D2D memcpy, dequantizes FP8→BF16 on device.
    /// See [`attention::fp8_dequantize_and_read`] for details.
    pub fn fp8_dequantize_and_read(
        &self,
        stream: &Arc<CudaStream>,
        page_pool: &CudaSlice<u8>,
        page_id: usize,
        page_offset: usize,
        len: usize,
        page_size: usize,
        kv_dim: usize,
        dtype: KvCacheDtype,
    ) -> Result<(CudaSlice<half::bf16>, CudaSlice<half::bf16>)> {
        crate::attention::fp8_dequantize_and_read(
            stream,
            &self.fp8_dequantize_kernel,
            page_pool,
            page_id,
            page_offset,
            len,
            page_size,
            kv_dim,
            dtype,
        )
    }

    /// Execute INT4 GEMM with on-the-fly dequantization.
    ///
    /// Weights stay in INT4-packed format in GPU memory — no dequantized copy exists.
    /// Dequantization happens in registers during the inner loop.
    /// See [`infers_cuda::gemm::matmul_int4`] for details.
    pub fn matmul_int4(
        &self,
        stream: &Arc<CudaStream>,
        config: &Int4GemmConfig,
        output: &mut CudaSlice<half::bf16>,
        weight: &CudaSlice<u32>,
        scales: &CudaSlice<half::bf16>,
        zeros: &CudaSlice<u32>,
        input: &CudaSlice<half::bf16>,
    ) -> Result<()> {
        infers_cuda::gemm::matmul_int4(
            stream,
            &self.int4_gemm_kernel,
            config,
            output,
            weight,
            scales,
            zeros,
            input,
        )
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
    // @lat: [[lat.md/lat#Phase 4 Deliverables#Forward Engine#Paged Prefill Path]]
    pub fn prefill_paged(
        &mut self,
        stream: &Arc<CudaStream>,
        token_ids: &[u32],
        seq_id: infers_kv::SequenceId,
    ) -> Result<usize> {
        let manager = self.paged_kv_manager.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?;

        let weights = &self.weights[0];
        let config = &self.config;
        let page_size = manager.page_size();
        let head_dim = config.head_dim;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        // Allocate pages for the sequence based on token count
        let num_pages_needed = (token_ids.len().saturating_sub(1) / page_size) + 1;
        for _ in 0..num_pages_needed {
            manager.append_page(seq_id)?;
        }

        // Upload block table to GPU
        let block_table = manager.block_table(seq_id)?;
        let block_table_i32: Vec<i32> = block_table.iter().map(|p| *p as i32).collect();
        let block_table_gpu = stream.clone_htod(&block_table_i32)?;

        // Upload positions
        let positions: Vec<u32> = (0..token_ids.len() as u32).collect();
        let positions_i32: Vec<i32> = positions.iter().map(|p| *p as i32).collect();
        let positions_gpu = stream.clone_htod(&positions_i32)?;

        // Ensure page pools allocated
        for cache in &mut self.paged_kv_caches {
            cache.ensure_allocated(stream)?;
        }

        // Embed tokens
        let embed_weight = weights.embedding.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
        let embed_table = crate::upload::upload_weight(stream, embed_weight)?;
        let mut hidden = crate::embedding::embed_tokens(
            stream, &self.embedding_kernel, token_ids, &embed_table,
            config.hidden_size, config.vocab_size,
        )?;

        // Layer loop
        for layer_idx in 0..config.num_hidden_layers {
            let layer = &weights.layers[layer_idx];
            let layer_type = config.get_layer_type(layer_idx);

            // Norm1
            let norm1_weight = crate::upload::upload_weight(stream, &layer.norm1)?;
            let norm1_out = crate::norm::rms_norm(
                stream, &self.rmsnorm_kernel, &hidden, &norm1_weight,
                config.rms_norm_eps, config.hidden_size,
            )?;

            // Attention/GDN dispatch
            let attn_or_gdn_out = match layer_type {
                LayerType::GatedDeltaNet => {
                    let gdn_weights = layer.gdn.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GDN weights not found for layer {}", layer_idx))?;
                    crate::gdn::forward(
                        &mut self.gemm, &self.int4_gemm_kernel, stream,
                        &self.gdn_prefill_kernel, gdn_weights, &norm1_out,
                        &mut self.gdn_states[layer_idx],
                        config.hidden_size, self.group_size, &weights.int4_companions,
                    )?
                }
                LayerType::FullAttention => {
                    let attn_weights = layer.attn.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
                    crate::attention::forward_paged(
                        &mut self.gemm, &self.int4_gemm_kernel, stream,
                        &self.softmax_kernel, &self.paged_kv_write_kernel,
                        &self.rope_kernel, &self.rmsnorm_kernel, &self.add_kernel,
                        attn_weights, &norm1_out,
                        &mut self.paged_kv_caches[layer_idx],
                        &block_table_gpu, &positions_gpu, &positions,
                        head_dim, num_heads, num_kv_heads, page_size,
                        config.rope_theta, config.partial_rotary_factor,
                        config.rms_norm_eps, self.group_size, &weights.int4_companions,
                    )?
                }
            };

            // Residual add
            hidden = crate::add::add(stream, &self.add_kernel, &hidden, &attn_or_gdn_out)?;

            // Norm2
            let norm2_weight = crate::upload::upload_weight(stream, &layer.norm2)?;
            let norm2_out = crate::norm::rms_norm(
                stream, &self.rmsnorm_kernel, &hidden, &norm2_weight,
                config.rms_norm_eps, config.hidden_size,
            )?;

            // MLP (INT4-aware)
            let mlp_weights = &layer.mlp;
            let intermediate_size = config.intermediate_size;
            let seq_len = token_ids.len();

            // gate
            let mut gate = stream.alloc_zeros::<bf16>(seq_len * intermediate_size)?;
            crate::gemm_dispatch::gemm_projection(
                &mut self.gemm, &self.int4_gemm_kernel, stream,
                &mlp_weights.gate_proj, &norm2_out, &mut gate,
                seq_len, intermediate_size, config.hidden_size,
                self.group_size, &weights.int4_companions,
            )?;

            // up
            let mut up = stream.alloc_zeros::<bf16>(seq_len * intermediate_size)?;
            crate::gemm_dispatch::gemm_projection(
                &mut self.gemm, &self.int4_gemm_kernel, stream,
                &mlp_weights.up_proj, &norm2_out, &mut up,
                seq_len, intermediate_size, config.hidden_size,
                self.group_size, &weights.int4_companions,
            )?;

            // silu = SiLU(gate) * up
            let mut silu_out = stream.alloc_zeros::<bf16>(seq_len * intermediate_size)?;
            let elem_i32 = (seq_len * intermediate_size) as i32;
            unsafe {
                stream.launch_builder(&self.silu_glu_kernel)
                    .arg(&gate).arg(&up).arg(&mut silu_out).arg(&elem_i32)
                    .launch(infers_cuda::LaunchConfig {
                        grid_dim: (((seq_len * intermediate_size) as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }

            // down_proj
            let mut mlp_out = stream.alloc_zeros::<bf16>(seq_len * config.hidden_size)?;
            crate::gemm_dispatch::gemm_projection(
                &mut self.gemm, &self.int4_gemm_kernel, stream,
                &mlp_weights.down_proj, &silu_out, &mut mlp_out,
                seq_len, config.hidden_size, intermediate_size,
                self.group_size, &weights.int4_companions,
            )?;

            // Residual add
            hidden = crate::add::add(stream, &self.add_kernel, &hidden, &mlp_out)?;
        }

        // Final norm
        let final_norm_weight = weights.norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
        let final_norm_gpu = crate::upload::upload_weight(stream, final_norm_weight)?;
        hidden = crate::norm::rms_norm(
            stream, &self.rmsnorm_kernel, &hidden, &final_norm_gpu,
            config.rms_norm_eps, config.hidden_size,
        )?;

        // LM head
        let lm_head_weight = weights.lm_head.as_ref()
            .or_else(|| weights.embedding.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;
        let seq_len = token_ids.len();
        let mut logits = stream.alloc_zeros::<bf16>(seq_len * config.vocab_size)?;
        crate::gemm_dispatch::gemm_projection(
            &mut self.gemm, &self.int4_gemm_kernel, stream,
            lm_head_weight, &hidden, &mut logits,
            seq_len, config.vocab_size, config.hidden_size,
            self.group_size, &weights.int4_companions,
        )?;

        // Sample: last row argmax (BF16)
        let last_row_start = (seq_len - 1) * config.vocab_size;
        let last_row_logits = logits.slice(last_row_start..last_row_start + config.vocab_size);
        let sampled = crate::sample::greedy_sample_bf16(
            stream, &self.argmax_kernel, &last_row_logits,
        )?;

        tracing::info!("Paged prefill sampled token: {}", sampled);

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
    // @lat: [[lat.md/lat#Phase 4 Deliverables#Forward Engine#Paged Decode Path]]
    pub fn decode_paged(
        &mut self,
        stream: &Arc<CudaStream>,
        token_id: u32,
        position: u32,
        seq_id: infers_kv::SequenceId,
    ) -> Result<u32> {
        let manager = self.paged_kv_manager.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?;

        let weights = &self.weights[0];
        let config = &self.config;
        let page_size = manager.page_size();
        let head_dim = config.head_dim;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let num_cached_tokens = manager.num_tokens(seq_id)? as i32;

        // Upload block table to GPU
        let block_table = manager.block_table(seq_id)?;
        let block_table_i32: Vec<i32> = block_table.iter().map(|p| *p as i32).collect();
        let block_table_gpu = stream.clone_htod(&block_table_i32)?;

        // Upload position to GPU
        let positions_i32 = vec![position as i32];
        let positions_gpu = stream.clone_htod(&positions_i32)?;

        // Ensure page pools allocated
        for cache in &mut self.paged_kv_caches {
            cache.ensure_allocated(stream)?;
        }

        // Embed single token
        let embed_weight = weights.embedding.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
        let embed_table = crate::upload::upload_weight(stream, embed_weight)?;
        let mut hidden = crate::embedding::embed_tokens(
            stream, &self.embedding_kernel, &[token_id], &embed_table,
            config.hidden_size, config.vocab_size,
        )?;

        // Layer loop
        for layer_idx in 0..config.num_hidden_layers {
            let layer = &weights.layers[layer_idx];
            let layer_type = config.get_layer_type(layer_idx);

            // Norm1
            let norm1_weight = crate::upload::upload_weight(stream, &layer.norm1)?;
            let norm1_out = crate::norm::rms_norm(
                stream, &self.rmsnorm_kernel, &hidden, &norm1_weight,
                config.rms_norm_eps, config.hidden_size,
            )?;

            // Attention/GDN dispatch
            let attn_or_gdn_out = match layer_type {
                LayerType::GatedDeltaNet => {
                    let gdn_weights = layer.gdn.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("GDN weights not found for layer {}", layer_idx))?;
                    crate::gdn::decode_forward(
                        &mut self.gemm, &self.int4_gemm_kernel, stream,
                        &self.gdn_update_kernel, gdn_weights, &norm1_out,
                        &mut self.gdn_states[layer_idx],
                        config.hidden_size, self.group_size, &weights.int4_companions,
                    )?
                }
                LayerType::FullAttention => {
                    let attn_weights = layer.attn.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
                    crate::attention::decode_forward_paged(
                        &mut self.gemm, &self.int4_gemm_kernel, stream,
                        &self.paged_kv_write_kernel, &self.paged_attention_decode_kernel,
                        &self.rope_kernel, &self.rmsnorm_kernel, &self.add_kernel,
                        attn_weights, &norm1_out,
                        &mut self.paged_kv_caches[layer_idx],
                        &block_table_gpu, &positions_gpu,
                        position, num_cached_tokens,
                        head_dim, num_heads, num_kv_heads, page_size,
                        config.rope_theta, config.partial_rotary_factor,
                        config.rms_norm_eps, self.group_size, &weights.int4_companions,
                    )?
                }
            };

            // Residual add
            hidden = crate::add::add(stream, &self.add_kernel, &hidden, &attn_or_gdn_out)?;

            // Norm2
            let norm2_weight = crate::upload::upload_weight(stream, &layer.norm2)?;
            let norm2_out = crate::norm::rms_norm(
                stream, &self.rmsnorm_kernel, &hidden, &norm2_weight,
                config.rms_norm_eps, config.hidden_size,
            )?;

            // MLP (INT4-aware)
            let mlp_weights = &layer.mlp;
            let intermediate_size = config.intermediate_size;

            // gate
            let mut gate = stream.alloc_zeros::<bf16>(intermediate_size)?;
            crate::gemm_dispatch::gemm_projection(
                &mut self.gemm, &self.int4_gemm_kernel, stream,
                &mlp_weights.gate_proj, &norm2_out, &mut gate,
                1, intermediate_size, config.hidden_size,
                self.group_size, &weights.int4_companions,
            )?;

            // up
            let mut up = stream.alloc_zeros::<bf16>(intermediate_size)?;
            crate::gemm_dispatch::gemm_projection(
                &mut self.gemm, &self.int4_gemm_kernel, stream,
                &mlp_weights.up_proj, &norm2_out, &mut up,
                1, intermediate_size, config.hidden_size,
                self.group_size, &weights.int4_companions,
            )?;

            // silu = SiLU(gate) * up
            let mut silu_out = stream.alloc_zeros::<bf16>(intermediate_size)?;
            let elem_i32 = intermediate_size as i32;
            unsafe {
                stream.launch_builder(&self.silu_glu_kernel)
                    .arg(&gate).arg(&up).arg(&mut silu_out).arg(&elem_i32)
                    .launch(infers_cuda::LaunchConfig {
                        grid_dim: ((intermediate_size as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }

            // down_proj
            let mut mlp_out = stream.alloc_zeros::<bf16>(config.hidden_size)?;
            crate::gemm_dispatch::gemm_projection(
                &mut self.gemm, &self.int4_gemm_kernel, stream,
                &mlp_weights.down_proj, &silu_out, &mut mlp_out,
                1, config.hidden_size, intermediate_size,
                self.group_size, &weights.int4_companions,
            )?;

            // Residual add
            hidden = crate::add::add(stream, &self.add_kernel, &hidden, &mlp_out)?;
        }

        // Final norm
        let final_norm_weight = weights.norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
        let final_norm_gpu = crate::upload::upload_weight(stream, final_norm_weight)?;
        hidden = crate::norm::rms_norm(
            stream, &self.rmsnorm_kernel, &hidden, &final_norm_gpu,
            config.rms_norm_eps, config.hidden_size,
        )?;

        // LM head
        let lm_head_weight = weights.lm_head.as_ref()
            .or_else(|| weights.embedding.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;
        let mut logits = stream.alloc_zeros::<bf16>(config.vocab_size)?;
        crate::gemm_dispatch::gemm_projection(
            &mut self.gemm, &self.int4_gemm_kernel, stream,
            lm_head_weight, &hidden, &mut logits,
            1, config.vocab_size, config.hidden_size,
            self.group_size, &weights.int4_companions,
        )?;

        // Sample (BF16 argmax)
        let sampled = crate::sample::greedy_sample_bf16(
            stream, &self.argmax_kernel, &logits.as_view(),
        )?;

        Ok(sampled)
    }

    /// Evict a sequence's pages from GPU to the backend eviction store.
    ///
    /// Copies page data from all layers' `PagedKvCache` GPU buffers to CPU,
    /// stores it in the backend eviction store, then marks the sequence as
    /// evicted in the KV manager.
    ///
    /// # Arguments
    /// * `seq_id` — The sequence to evict.
    /// * `stream` — CUDA stream for GPU→CPU data copies.
    /// * `store` — Backend eviction store to receive the page data.
    /// * `kv_manager` — Paged KV manager for metadata tracking.
    ///
    /// # Returns
    /// The `EvictedSequence` snapshot for later restoration.
    pub fn evict_session(
        &mut self,
        seq_id: infers_kv::SequenceId,
        stream: &Arc<CudaStream>,
        store: &mut BackendEvictionStore,
        kv_manager: &mut infers_kv::PagedKvManager,
    ) -> Result<infers_kv::EvictedSequence> {
        // Get block table (page IDs for this sequence)
        let block_table = kv_manager
            .block_table(seq_id)
            .map_err(|e| anyhow::anyhow!("Failed to get block table: {e:?}"))?
            .to_vec();

        if block_table.is_empty() {
            anyhow::bail!("Sequence {seq_id} has no pages to evict");
        }

        // For each page, copy data from all layers' GPU buffers
        for &page_id in &block_table {
            for (layer_idx, cache) in self.paged_kv_caches.iter().enumerate() {
                let page_pool = cache.page_pool()
                    .ok_or_else(|| anyhow::anyhow!("PagedKvCache not allocated for layer {layer_idx}"))?;

                let page_elements = 2 * cache.page_size() * cache.kv_dim();
                let offset = (page_id as usize) * page_elements;

                // Extract sub-slice of GPU buffer for this page
                let sub_slice = page_pool.slice(offset..offset + page_elements);

                // Copy from GPU to CPU
                let page_data_bf16: Vec<half::bf16> = stream
                    .clone_dtoh(&sub_slice)
                    .map_err(|e| anyhow::anyhow!("Failed to copy page from GPU: {e}"))?;

                // Convert bf16 to bytes and store
                let page_bytes = BackendEvictionStore::bf16_slice_to_bytes(&page_data_bf16);
                store.store(layer_idx, page_id, page_bytes);
            }
        }

        // Mark the sequence as evicted in the KV manager (metadata only)
        let evicted = kv_manager.mark_evicted(seq_id)
            .map_err(|e| anyhow::anyhow!("Failed to mark sequence evicted: {e:?}"))?;

        tracing::info!(
            "Evicted sequence {}: {} pages, {} tokens",
            seq_id,
            block_table.len(),
            evicted.num_tokens,
        );

        Ok(evicted)
    }

    /// Restore a previously evicted sequence back to GPU.
    ///
    /// Allocates new pages via the KV manager, retrieves page data from the
    /// backend eviction store, and copies it back to all layers' `PagedKvCache`
    /// GPU buffers.
    ///
    /// # Arguments
    /// * `evicted` — The `EvictedSequence` from a prior `evict_session()` call.
    /// * `stream` — CUDA stream for CPU→GPU data copies.
    /// * `store` — Backend eviction store containing the page data.
    /// * `kv_manager` — Paged KV manager for metadata tracking.
    ///
    /// # Returns
    /// The new `SequenceId` assigned to the restored sequence.
    pub fn restore_session(
        &mut self,
        evicted: infers_kv::EvictedSequence,
        stream: &Arc<CudaStream>,
        store: &mut BackendEvictionStore,
        kv_manager: &mut infers_kv::PagedKvManager,
    ) -> Result<infers_kv::SequenceId> {
        // Allocate new pages
        let new_seq_id = kv_manager.allocate_for_restore(&evicted)
            .map_err(|e| anyhow::anyhow!("Failed to allocate for restore: {e:?}"))?;

        // Get new block table (maps logical pages → new physical page IDs)
        let new_block_table = kv_manager
            .block_table(new_seq_id)
            .map_err(|e| anyhow::anyhow!("Failed to get new block table: {e:?}"))?
            .to_vec();

        // For each page, copy data from store back to all layers' GPU buffers
        for (i, &old_page_id) in evicted.page_ids.iter().enumerate() {
            let new_page_id = new_block_table[i];

            for layer_idx in 0..self.paged_kv_caches.len() {
                // Retrieve data from store
                let page_bytes = store
                    .retrieve(layer_idx, old_page_id)
                    .ok_or_else(|| anyhow::anyhow!(
                        "No evicted data for sequence {} page {} layer {}",
                        evicted.seq_id, old_page_id, layer_idx,
                    ))?;

                // Convert bytes back to bf16
                let page_data_bf16 = BackendEvictionStore::bytes_to_bf16_slice(&page_bytes);

                // Get mutable slice of GPU buffer for this page
                let cache = &mut self.paged_kv_caches[layer_idx];
                let page_elements = 2 * cache.page_size() * cache.kv_dim();
                let offset = (new_page_id as usize) * page_elements;
                let page_pool = cache.page_pool_mut()
                    .ok_or_else(|| anyhow::anyhow!("PagedKvCache not allocated for layer {layer_idx}"))?;

                let mut sub_slice = page_pool.slice_mut(offset..offset + page_elements);

                // Copy from CPU to GPU
                stream
                    .memcpy_htod(&page_data_bf16, &mut sub_slice)
                    .map_err(|e| anyhow::anyhow!("Failed to copy page to GPU: {e}"))?;
            }
        }

        tracing::info!(
            "Restored sequence {} (was {}): {} pages, {} tokens",
            new_seq_id,
            evicted.seq_id,
            new_block_table.len(),
            evicted.num_tokens,
        );

        Ok(new_seq_id)
    }

    /// Initialize the MTP speculative decoding engine.
    ///
    /// Creates an `MtpEngine` from MTP weights, uploading weight data to GPU.
    ///
    /// # Arguments
    /// * `mtp_weights` — MTP weights from model loading
    /// * `num_draft_tokens` — Number of draft tokens per speculative step (1-4, 2 recommended)
    /// * `stream` — CUDA stream for weight uploads
    ///
    /// # Returns
    /// A new `MtpEngine` ready for draft generation and verification.
    pub fn init_mtp(
        &self,
        mtp_weights: &MtpWeights,
        num_draft_tokens: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<infers_mtp::MtpEngine> {
        infers_mtp::MtpEngine::new(mtp_weights, &self.config, num_draft_tokens, stream)
    }

    /// Decode a single token and return both the sampled token and the
    /// final hidden state (pre-LM-head) for MTP speculative decoding.
    ///
    /// # Arguments
    /// * `stream` — CUDA stream for kernel launches
    /// * `token_id` — Previous token ID to continue generation
    /// * `position` — Current position in the sequence (for RoPE)
    ///
    /// # Returns
    /// `(sampled_token, hidden_state)` where `hidden_state` is the output
    /// of the final RMSNorm, preserved before LM head projection.
    pub fn decode_with_hidden(
        &mut self,
        stream: &Arc<CudaStream>,
        token_id: u32,
        position: u32,
    ) -> Result<(u32, CudaSlice<bf16>)> {
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
            int4_gemm: self.int4_gemm_kernel.clone(),
        };

        crate::decode::decode_with_hidden(
            &mut self.gemm, stream, &kernels, &self.nccl,
            &self.config, weights, token_id, position,
            &mut self.kv_caches, &mut self.gdn_states,
            self.group_size,
        )
    }

    /// Run the MTP speculative decoding loop.
    ///
    /// For each step:
    /// 1. Get the main model's hidden state via `decode_with_hidden`
    /// 2. Generate draft tokens from the MTP head
    /// 3. Verify drafts against the main model
    /// 4. Accept/reject drafts and extend output
    ///
    /// # Arguments
    /// * `stream` — CUDA stream for kernel launches
    /// * `token_id` — Initial token ID to start generation
    /// * `position` — Starting position in the sequence
    /// * `mtp` — MTP engine (created via `init_mtp`)
    /// * `max_tokens` — Maximum number of tokens to generate
    /// * `mtp_metrics` — Optional metrics tracker (can be `&mut MtpMetrics::new()`)
    ///
    /// # Returns
    /// All generated tokens including the initial decode step
    pub fn decode_with_mtp(
        &mut self,
        stream: &Arc<CudaStream>,
        token_id: u32,
        position: u32,
        mtp: &mut infers_mtp::MtpEngine,
        max_tokens: usize,
        mtp_metrics: &mut infers_mtp::MtpMetrics,
    ) -> Result<Vec<u32>> {
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
            int4_gemm: self.int4_gemm_kernel.clone(),
        };

        use std::cell::Cell;

        let mut output_tokens = Vec::new();
        let mut current_token = token_id;
        let current_pos = Cell::new(position);

        // SAFETY: raw pointer workaround for Fn closures that need &mut access.
        // The pointers are valid because:
        //   1. self is borrowed mutably for the entire duration of decode_with_mtp
        //   2. Closures are called sequentially (never concurrently)
        //   3. References remain valid as long as self is alive
        // keep the kv_caches and gdn_states pointers for mutable access in full_forward_fn
        let kv_caches_ptr: Arc<*mut Vec<KvCache>> =
            Arc::new(&mut self.kv_caches as *mut Vec<KvCache>);
        let gdn_states_ptr: Arc<*mut Vec<GdnState>> =
            Arc::new(&mut self.gdn_states as *mut Vec<GdnState>);

        // Read-only references captured via Arc<*const _> for use in Fn closures
        // alongside the mutable-access raw pointers above
        let weights_ref: &infers_model::WeightRegistry = &self.weights[0];
        let weights_ptr: Arc<*const infers_model::WeightRegistry> =
            Arc::new(weights_ref as *const infers_model::WeightRegistry);
        let nccl_ref: &NcclCommunicator = &self.nccl;
        let nccl_ptr: Arc<*const NcclCommunicator> =
            Arc::new(nccl_ref as *const NcclCommunicator);

        // Clone config Arc for use in closures (no raw pointer needed)
        let config = self.config.clone();

        // --- Draft position counter for verification ---
        // Tracks which draft index we're on so each draft gets the correct position
        let verify_pos = Cell::new(current_pos.get());

        // Build embed callback for MTP operations
        let embed_fn = |token: u32, s: &Arc<CudaStream>| -> Result<CudaSlice<bf16>> {
            let weights: &infers_model::WeightRegistry = unsafe { &**weights_ptr };
            let embed_weight = weights.embedding.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
            let embed_table = crate::upload::upload_weight(s, embed_weight)?;
            crate::embedding::embed_tokens(
                s,
                &kernels.embedding,
                &[token],
                &embed_table,
                config.hidden_size,
                config.vocab_size,
            )
        };

        // Build RMS norm callback
        let rms_norm_fn = |s: &Arc<CudaStream>,
                           input: &CudaSlice<bf16>,
                           weight: &CudaSlice<bf16>,
                           eps: f32,
                           hidden_size: usize|
         -> Result<CudaSlice<bf16>> {
            crate::norm::rms_norm(s, &kernels.rmsnorm, input, weight, eps, hidden_size)
        };

        // Build forward_layer callback (uses EPHEMERAL local state,
        // NOT the main model's kv_caches/gdn_states — MTP layers are speculative
        // and must not corrupt the main model's state)
        let forward_layer_fn =
            |layer: &infers_model::LayerWeights,
             input: &CudaSlice<bf16>,
             s: &Arc<CudaStream>,
             g: &mut GemmEngine| -> Result<CudaSlice<bf16>> {
                // Local ephemeral state for the MTP head's decoder layer
                let mut mtp_kv = vec![crate::attention::KvCache::new()];
                let mut mtp_gdn = vec![crate::gdn::GdnState::new()];
                let config: &ModelConfig = config.as_ref();
                crate::mtp::forward_layer_pass(
                    layer, input, g, s, &kernels, config,
                    &mut mtp_kv, &mut mtp_gdn,
                    current_pos.get(), 0,
                )
            };

        // Build LM head callback
        let lm_head_fn = |hidden: &CudaSlice<bf16>,
                          s: &Arc<CudaStream>,
                          g: &mut GemmEngine| -> Result<CudaSlice<bf16>> {
            let weights: &infers_model::WeightRegistry = unsafe { &**weights_ptr };
            let lm_head_weight = weights.lm_head.as_ref()
                .or_else(|| weights.embedding.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;
            let lm_head_gpu = crate::upload::upload_weight(s, lm_head_weight)?;
            let mut logits = s
                .alloc_zeros::<bf16>(config.vocab_size)
                .map_err(|e| anyhow::anyhow!("Failed to allocate logits buffer: {e}"))?;
            g.matmul_bf16(
                &infers_cuda::gemm::GemmConfig {
                    m: 1,
                    n: config.vocab_size,
                    k: config.hidden_size,
                    transa: true,
                    transb: false,
                    alpha: 1.0,
                    beta: 0.0,
                    lda: None,
                    ldb: None,
                    ldc: None,
                    activation: None,
                },
                hidden,
                &lm_head_gpu,
                &mut logits,
            )?;
            Ok(logits)
        };

        // Build sample callback — BF16 argmax directly on GPU (no CPU round-trip)
        let sample_fn = |logits: &CudaSlice<bf16>,
                          s: &Arc<CudaStream>| -> Result<u32> {
            crate::sample::greedy_sample_bf16(s, &kernels.argmax, &logits.as_view())
        };

        // Build full_forward callback (for draft verification)
        // NOTE: position advances with each call via verify_pos counter
        let full_forward_fn =
            |token: u32,
             s: &Arc<CudaStream>,
             g: &mut GemmEngine| -> Result<CudaSlice<bf16>> {
                let kv_caches: &mut [KvCache] = unsafe { &mut **kv_caches_ptr };
                let gdn_states: &mut [GdnState] = unsafe { &mut **gdn_states_ptr };
                let weights: &infers_model::WeightRegistry = unsafe { &**weights_ptr };
                let nccl: &NcclCommunicator = unsafe { &**nccl_ptr };
                let pos = verify_pos.get();
                verify_pos.set(pos + 1);
                crate::mtp::full_forward_logits(
                    token, g, s, &kernels, nccl,
                    config.as_ref(), weights,
                    kv_caches, gdn_states,
                    pos, // each draft gets incrementing position
                )
            };

        // Create MtpOperations from callbacks
        let ops = infers_mtp::MtpOperations::new(
            &embed_fn,
            &rms_norm_fn,
            &forward_layer_fn,
            &lm_head_fn,
            &sample_fn,
            &full_forward_fn,
        );

        while output_tokens.len() < max_tokens {
            // Step 1: Decode with hidden state
            let (sampled_token, hidden_state) =
                self.decode_with_hidden(stream, current_token, current_pos.get())?;
            output_tokens.push(sampled_token);
            current_pos.set(current_pos.get() + 1);

            // Step 2: Check for EOS
            // (EOS check placeholder — tokenizer not available at this level)

            // Step 3: Generate draft tokens from MTP head
            let num_drafts = mtp.adaptive_num_drafts();
            let drafts = mtp.generate_drafts(
                &hidden_state,
                sampled_token,
                num_drafts,
                stream,
                &mut self.gemm,
                &ops,
            )?;

            // Step 4: Verify drafts
            // Reset position counter before each verification batch
            verify_pos.set(current_pos.get());
            let verification =
                mtp.verify_drafts(&drafts, stream, &mut self.gemm, &ops)?;

            // Step 5: Record metrics
            mtp_metrics.record_step(verification.num_accepted(), verification.num_drafts());

            // Step 6: Accept longest valid prefix
            let accepted = mtp.accept_prefix(&verification);
            let accepted_len = accepted.len();

            output_tokens.extend(accepted);
            current_pos.set(current_pos.get() + accepted_len as u32);

            // Update current_token for next iteration
            if let Some(&last_accepted) = output_tokens.last() {
                current_token = last_accepted;
            }

            // Stop if we've hit max_tokens
            if output_tokens.len() >= max_tokens {
                break;
            }
        }

        Ok(output_tokens)
    }
}
