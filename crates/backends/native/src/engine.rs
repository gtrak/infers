//! Forward inference engine — owns GPU state, kernels, and execution logic.
//!
//! `ForwardEngine` is the central struct that coordinates all inference steps:
//! embedding lookup, layer dispatch (GDN vs full attention), MLP, normalization,
//! and sampling. It holds references to CUDA resources, model weights, and kernels.

use crate::probe;
use std::sync::Arc;

use anyhow::Result;
use infers_cuda::gemm::GemmEngine;
use infers_cuda::gemm::Int4GemmConfig;
use infers_cuda::kernels::{KernelRegistry, LoadedKernelRegistry};
use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::stream::StreamPool;
use infers_cuda::PinnedHostBuffer;
use infers_cuda::{CudaContext, CudaFunction, CudaStream, PushKernelArg};
use infers_model::{LayerType, MmapWeightRegistry, ModelConfig, WeightRegistry};

use crate::attention::{KvCache, PagedKvCache};
use crate::gpu_cache::GpuWeightCache;
use crate::eviction::BackendEvictionStore;
use crate::gdn::GdnState;

use infers_kv::PagedKvManager;

use half::{bf16, f16};
use infers_kv::KvCacheDtype;
use infers_cuda::CudaSlice;

use infers_cuda::{group_end, group_start};

use crate::sync;
use crate::sample::{Xoshiro256PlusPlus, sample_with_config};


/// Per-GPU cached kernel function handles.
/// Each GPU context needs its own set since CudaFunction handles are context-bound.
struct PerGpuKernels {
    rmsnorm: CudaFunction,
    silu_glu: CudaFunction,
    rope: CudaFunction,
    embedding: CudaFunction,
    add: CudaFunction,
    argmax: CudaFunction,
    softmax: CudaFunction,
    kv_cache_write: CudaFunction,
    gdn_prefill: CudaFunction,
    gdn_update: CudaFunction,
    paged_kv_write: CudaFunction,
    #[allow(dead_code)]
    paged_kv_read: CudaFunction,
    paged_attention_decode: CudaFunction,
    fp8_quantize: CudaFunction,
    fp8_dequantize: CudaFunction,
    int4_gemm: CudaFunction,
    // New GDN kernels (gated delta rule)
    gdn_gated_delta_prefill: CudaFunction,
    gdn_gated_delta_update: CudaFunction,
    gdn_recurrent_step: CudaFunction,
    gdn_chunked_prefill: CudaFunction,
    conv1d_depthwise: CudaFunction,
    rms_norm_gated: CudaFunction,
    attn_output_gate: CudaFunction,
}

/// Central engine for forward-pass inference.
///
/// Owns all GPU resources: CUDA contexts, streams, loaded kernels, cuBLASLt handles,
/// and NCCL communicators. Coordinates the full prefill/decode pipeline.
pub struct ForwardEngine {
    /// Model architecture configuration.
    config: Arc<ModelConfig>,

    /// One weight registry per GPU shard (tensor parallelism).
    weights: Vec<WeightRegistry>,

    /// Mmap weight registries (retained to keep mmap handles alive).
    _mmap_registries: Vec<MmapWeightRegistry>,

    /// Per-GPU weight caches with GPU-resident buffers.
    weight_caches: Vec<GpuWeightCache>,

    /// Per-GPU cached kernel function handles.
    per_gpu_kernels: Vec<PerGpuKernels>,

    /// Paged KV cache manager (pool + prefix cache + COW).
    paged_kv_manager: Option<PagedKvManager>,

    /// cuBLASLt GEMM engines (one per GPU for tensor parallelism).
    gemm_engines: Vec<GemmEngine>,

    /// NCCL communicator for tensor-parallel all-reduce.
    nccl: NcclCommunicator,

    /// Async CUDA streams for parallel execution.
    /// Retained for future multi-stream execution
    #[allow(dead_code)]
    streams: StreamPool,

    /// Per-GPU, per-layer KV caches for full-attention layers (flat cache, legacy).
    kv_caches: Vec<Vec<KvCache>>,          // [gpu_idx][layer_idx]
    /// Per-GPU, per-layer paged KV caches (new paged system).
    paged_kv_caches: Vec<Vec<PagedKvCache>>,  // [gpu_idx][layer_idx]

    /// Per-GPU, per-layer GDN recurrent states.
    gdn_states: Vec<Vec<GdnState>>,      // [gpu_idx][layer_idx]

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
        contexts: Vec<Arc<CudaContext>>,
        kernel_registry: KernelRegistry,
        streams: StreamPool,
        group_size: usize,
    ) -> Result<Self> {
        // Load CUDA kernel modules on each GPU context
        let num_gpus = streams.len();
        let mut per_gpu_kernels = Vec::with_capacity(num_gpus);
        for gpu_idx in 0..num_gpus {
            let ctx = contexts.get(gpu_idx)
                .ok_or_else(|| anyhow::anyhow!("Missing context for GPU {gpu_idx}"))?;
            let kernels = LoadedKernelRegistry::load_all(ctx.clone(), &kernel_registry)
                .map_err(|e| anyhow::anyhow!("Failed to load CUDA kernels on GPU {gpu_idx}: {e}"))?;

            let pk = PerGpuKernels {
                rmsnorm: kernels.get_function("infers_rmsnorm_bf16")?,
                silu_glu: kernels.get_function("infers_silu_glu_bf16")?,
                rope: kernels.get_function("infers_rope_bf16")?,
                embedding: kernels.get_function("infers_embedding_gather_bf16")?,
                add: kernels.get_function("infers_add_bf16")?,
                argmax: kernels.get_function("infers_argmax_bf16")?,
                softmax: kernels.get_function("infers_softmax_bf16")?,
                kv_cache_write: kernels.get_function("infers_kv_cache_write_bf16")?,
                gdn_prefill: kernels.get_function("infers_gdn_mamba2_prefill_bf16")?,
                gdn_update: kernels.get_function("infers_gdn_mamba2_update_bf16")?,
                paged_kv_write: kernels.get_function("infers_paged_kv_write_bf16")?,
                paged_kv_read: kernels.get_function("infers_paged_kv_read_bf16")?,
                paged_attention_decode: kernels.get_function("infers_paged_attention_decode_bf16")?,
                fp8_quantize: kernels.get_function("infers_fp8_quantize_bf16")?,
                fp8_dequantize: kernels.get_function("infers_fp8_dequantize_bf16")?,
                int4_gemm: kernels.get_function("int4_gemm_kernel")?,
                gdn_gated_delta_prefill: kernels.get_function("infers_gdn_gated_delta_prefill_bf16")?,
                gdn_gated_delta_update: kernels.get_function("infers_gdn_gated_delta_update_bf16")?,
                gdn_recurrent_step: kernels.get_function("infers_gdn_recurrent_step_bf16")?,
                gdn_chunked_prefill: {
                    let f = kernels.get_function("infers_gdn_chunked_gated_delta_prefill_bf16")?;
                    // Allow up to 100KB dynamic shared memory for the chunked GDN kernel
                    // (default is 48KB; the kernel uses ~81KB for C=64, K=128)
                    f.set_attribute(
                        infers_cuda::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        100000,
                    ).ok(); // ok() — ignore error if attribute not supported
                    f
                },
                conv1d_depthwise: kernels.get_function("infers_conv1d_depthwise_silu_bf16")?,
                rms_norm_gated: kernels.get_function("infers_rms_norm_gated_bf16")?,
                attn_output_gate: kernels.get_function("infers_attn_output_gate_bf16")?,
            };
            per_gpu_kernels.push(pk);
        }

        // Create one GEMM engine per GPU (one per stream in the pool)
        let mut gemm_engines = Vec::with_capacity(num_gpus);
        for i in 0..num_gpus {
            let s = streams.get(i).ok_or_else(|| anyhow::anyhow!("Missing stream {i}"))?;
            gemm_engines.push(
                GemmEngine::new(s.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to create cuBLASLt engine for GPU {i}: {e}"))?
            );
        }
        // Create NCCL communicator for tensor parallelism
        let nccl = {
            let comm_streams: Vec<Arc<CudaStream>> = (0..streams.len())
                .filter_map(|i| streams.get(i).cloned())
                .collect();
            NcclCommunicator::new(comm_streams)
                .map_err(|e| anyhow::anyhow!("Failed to initialize NCCL: {e}"))?
        };

        // Build GPU-resident weight caches for each GPU in parallel
        let mut handles = Vec::with_capacity(num_gpus);
        for gpu_idx in 0..num_gpus {
            let gpu_stream = streams.get(gpu_idx).unwrap().clone();
            let registry = weights[gpu_idx].clone(); // cheap clone (Bytes is Arc-based)
            handles.push(std::thread::spawn(move || {
                let result = GpuWeightCache::new(&gpu_stream, &registry);
                (gpu_idx, result)
            }));
        }

        let mut weight_caches: Vec<Option<GpuWeightCache>> = (0..num_gpus).map(|_| None).collect();
        for handle in handles {
            let (gpu_idx, result) = handle.join().expect("Weight cache thread panicked");
            let cache = result?;
            tracing::info!("GPU {}: cached {} weights", gpu_idx, cache.len());
            weight_caches[gpu_idx] = Some(cache);
        }
        // Unwrap now that all are filled
        let weight_caches: Vec<GpuWeightCache> = weight_caches.into_iter()
            .map(|c| c.expect("All weight caches should be filled"))
            .collect();

        // Initialize per-GPU, per-layer caches and states
        let num_layers = config.num_hidden_layers;
        let kv_caches: Vec<Vec<KvCache>> = (0..num_gpus).map(|_| (0..num_layers).map(|_| KvCache::new()).collect()).collect();
        let gdn_states: Vec<Vec<GdnState>> = (0..num_gpus).map(|_| (0..num_layers).map(|_| GdnState::new()).collect()).collect();
        let paged_kv_caches: Vec<Vec<PagedKvCache>> = (0..num_gpus).map(|_| Vec::new()).collect();

        tracing::info!(
            "ForwardEngine initialized: {} layers, {} GPU shards",
            config.num_hidden_layers,
            weights.len()
        );

        Ok(Self {
            config,
            weights,
            _mmap_registries: Vec::new(),
            weight_caches,
            per_gpu_kernels,
            paged_kv_manager: None,
            gemm_engines,
            nccl,
            streams,
            kv_caches,
            paged_kv_caches,
            gdn_states,
            group_size,
        })
    }

    /// Construct a `ForwardEngine` from mmap-backed weight registries.
    ///
    /// Similar to [`new`] but loads weights via zero-copy memory-mapped access.
    /// The `_mmap_registries` field retains the Arc-based mmap handles for
    /// the lifetime of the engine, preventing premature unmapping.
    ///
    /// # Arguments
    /// * `config` — Model architecture parameters
    /// * `mmap_registries` — Per-GPU mmap weight registries (one per tensor-parallel rank)
    /// * `contexts` — CUDA contexts for kernel loading
    /// * `kernel_registry` — Names and paths of CUDA kernels to load
    /// * `streams` — Pool of async CUDA streams
    /// * `pinned` — Pinned host buffer for dtype conversion during upload
    /// * `group_size` — INT4 quantization group size for on-the-fly dequantization
    pub fn new_from_mmap(
        config: Arc<ModelConfig>,
        mmap_registries: Vec<MmapWeightRegistry>,
        metadata_registries: Vec<WeightRegistry>,
        contexts: Vec<Arc<CudaContext>>,
        kernel_registry: KernelRegistry,
        streams: StreamPool,
        pinned: &mut PinnedHostBuffer,
        group_size: usize,
    ) -> Result<Self> {
        // Load CUDA kernel modules on each GPU context
        let num_gpus = streams.len();
        let mut per_gpu_kernels = Vec::with_capacity(num_gpus);
        for gpu_idx in 0..num_gpus {
            let ctx = contexts.get(gpu_idx)
                .ok_or_else(|| anyhow::anyhow!("Missing context for GPU {gpu_idx}"))?;
            let kernels = LoadedKernelRegistry::load_all(ctx.clone(), &kernel_registry)
                .map_err(|e| anyhow::anyhow!("Failed to load CUDA kernels on GPU {gpu_idx}: {e}"))?;

            let pk = PerGpuKernels {
                rmsnorm: kernels.get_function("infers_rmsnorm_bf16")?,
                silu_glu: kernels.get_function("infers_silu_glu_bf16")?,
                rope: kernels.get_function("infers_rope_bf16")?,
                embedding: kernels.get_function("infers_embedding_gather_bf16")?,
                add: kernels.get_function("infers_add_bf16")?,
                argmax: kernels.get_function("infers_argmax_bf16")?,
                softmax: kernels.get_function("infers_softmax_bf16")?,
                kv_cache_write: kernels.get_function("infers_kv_cache_write_bf16")?,
                gdn_prefill: kernels.get_function("infers_gdn_mamba2_prefill_bf16")?,
                gdn_update: kernels.get_function("infers_gdn_mamba2_update_bf16")?,
                paged_kv_write: kernels.get_function("infers_paged_kv_write_bf16")?,
                paged_kv_read: kernels.get_function("infers_paged_kv_read_bf16")?,
                paged_attention_decode: kernels.get_function("infers_paged_attention_decode_bf16")?,
                fp8_quantize: kernels.get_function("infers_fp8_quantize_bf16")?,
                fp8_dequantize: kernels.get_function("infers_fp8_dequantize_bf16")?,
                int4_gemm: kernels.get_function("int4_gemm_kernel")?,
                gdn_gated_delta_prefill: kernels.get_function("infers_gdn_gated_delta_prefill_bf16")?,
                gdn_gated_delta_update: kernels.get_function("infers_gdn_gated_delta_update_bf16")?,
                gdn_recurrent_step: kernels.get_function("infers_gdn_recurrent_step_bf16")?,
                gdn_chunked_prefill: {
                    let f = kernels.get_function("infers_gdn_chunked_gated_delta_prefill_bf16")?;
                    // Allow up to 100KB dynamic shared memory for the chunked GDN kernel
                    // (default is 48KB; the kernel uses ~81KB for C=64, K=128)
                    f.set_attribute(
                        infers_cuda::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        100000,
                    ).ok(); // ok() — ignore error if attribute not supported
                    f
                },
                conv1d_depthwise: kernels.get_function("infers_conv1d_depthwise_silu_bf16")?,
                rms_norm_gated: kernels.get_function("infers_rms_norm_gated_bf16")?,
                attn_output_gate: kernels.get_function("infers_attn_output_gate_bf16")?,
            };
            per_gpu_kernels.push(pk);
        }

        // Create one GEMM engine per GPU (one per stream in the pool)
        let mut gemm_engines = Vec::with_capacity(num_gpus);
        for i in 0..num_gpus {
            let s = streams.get(i).ok_or_else(|| anyhow::anyhow!("Missing stream {i}"))?;
            gemm_engines.push(
                GemmEngine::new(s.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to create cuBLASLt engine for GPU {i}: {e}"))?
            );
        }
        // Create NCCL communicator for tensor parallelism
        let nccl = {
            let comm_streams: Vec<Arc<CudaStream>> = (0..streams.len())
                .filter_map(|i| streams.get(i).cloned())
                .collect();
            NcclCommunicator::new(comm_streams)
                .map_err(|e| anyhow::anyhow!("Failed to initialize NCCL: {e}"))?
        };

        // Build GPU-resident weight caches for each GPU (mmap path).
        // Sequential — pinned buffer is per-thread and mmap upload requires CUDA context on current thread.
        let mut weight_caches = Vec::with_capacity(num_gpus);
        for gpu_idx in 0..num_gpus {
            let gpu_stream = streams.get(gpu_idx).unwrap().clone();
            let registry = &mmap_registries[gpu_idx];
            let cache = GpuWeightCache::new_from_mmap(&gpu_stream, registry, pinned)?;
            tracing::info!("GPU {}: cached {} weights (mmap)", gpu_idx, cache.len());
            weight_caches.push(cache);
        }

        // Initialize per-GPU, per-layer caches and states
        let num_layers = config.num_hidden_layers;
        let kv_caches: Vec<Vec<KvCache>> = (0..num_gpus).map(|_| (0..num_layers).map(|_| KvCache::new()).collect()).collect();
        let gdn_states: Vec<Vec<GdnState>> = (0..num_gpus).map(|_| (0..num_layers).map(|_| GdnState::new()).collect()).collect();
        let paged_kv_caches: Vec<Vec<PagedKvCache>> = (0..num_gpus).map(|_| Vec::new()).collect();

        tracing::info!(
            "ForwardEngine initialized (mmap): {} layers, {} GPU shards",
            config.num_hidden_layers,
            mmap_registries.len()
        );

        Ok(Self {
            config,
            weights: metadata_registries, // metadata registries for name lookups during inference
            _mmap_registries: mmap_registries,
            weight_caches,
            per_gpu_kernels,
            paged_kv_manager: None,
            gemm_engines,
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
        let weights = &self.weights[0];

        let kernels = crate::prefill::PrefillKernels {
            rmsnorm: self.per_gpu_kernels[0].rmsnorm.clone(),
            silu_glu: self.per_gpu_kernels[0].silu_glu.clone(),
            rope: self.per_gpu_kernels[0].rope.clone(),
            embedding: self.per_gpu_kernels[0].embedding.clone(),
            add: self.per_gpu_kernels[0].add.clone(),
            argmax: self.per_gpu_kernels[0].argmax.clone(),
            softmax: self.per_gpu_kernels[0].softmax.clone(),
            kv_cache_write: self.per_gpu_kernels[0].kv_cache_write.clone(),
            gdn_prefill: self.per_gpu_kernels[0].gdn_prefill.clone(),
            gdn_gated_delta_prefill: self.per_gpu_kernels[0].gdn_gated_delta_prefill.clone(),
            gdn_recurrent_step: self.per_gpu_kernels[0].gdn_recurrent_step.clone(),
            gdn_chunked_prefill: self.per_gpu_kernels[0].gdn_chunked_prefill.clone(),
            conv1d_depthwise: self.per_gpu_kernels[0].conv1d_depthwise.clone(),
            rms_norm_gated: self.per_gpu_kernels[0].rms_norm_gated.clone(),
            int4_gemm: self.per_gpu_kernels[0].int4_gemm.clone(),
            attn_output_gate: self.per_gpu_kernels[0].attn_output_gate.clone(),
        };

        crate::prefill::prefill(
            &mut self.gemm_engines[0], stream, &kernels, &self.nccl,
            &self.config, weights, &self.weight_caches[0], token_ids,
            &mut self.kv_caches[0], &mut self.gdn_states[0],
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

        // kv_dim is per GPU: num_kv_heads/num_gpus * head_dim
        let num_gpus = self.weights.len();
        let kv_dim_per_gpu = (num_kv_heads / num_gpus) * head_dim;

        let caches: Vec<Vec<PagedKvCache>> = (0..num_gpus)
            .map(|_| {
                (0..self.config.num_hidden_layers)
                    .map(|_| PagedKvCache::new(total_pages, page_size, kv_dim_per_gpu))
                    .collect()
            })
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

    /// Create a new sequence in the paged KV manager, returning its ID.
    pub fn create_sequence(&mut self) -> infers_kv::SequenceId {
        self.paged_kv_manager
            .as_mut()
            .map(|m| m.create_sequence())
            .unwrap_or(0)
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
            &self.per_gpu_kernels[0].fp8_quantize,
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
            &self.per_gpu_kernels[0].fp8_dequantize,
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
        scales: &CudaSlice<f16>,
        zeros: &CudaSlice<u32>,
        input: &CudaSlice<half::bf16>,
    ) -> Result<()> {
        infers_cuda::gemm::matmul_int4(
            stream,
            &self.per_gpu_kernels[0].int4_gemm,
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
    /// Tuple of (number of pages allocated, first sampled token ID).
    // @lat: [[lat.md/lat#Phase 4 Deliverables#Forward Engine#Paged Prefill Path]]
    pub fn prefill_paged(
        &mut self,
        _stream: &Arc<CudaStream>,
        token_ids: &[u32],
        seq_id: infers_kv::SequenceId,
        sampling_config: &infers_scheduler::SamplingConfig,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<(usize, u32)> {
        let span = tracing::info_span!("prefill", num_tokens = token_ids.len());
        let _enter = span.enter();

        // GPU timing: create events on each GPU's context
        let num_gpus = self.weights.len();
        let gpu_start_events: Vec<_> = (0..num_gpus)
            .map(|gpu_idx| {
                let ctx = self.streams.get(gpu_idx).unwrap().context();
                ctx.new_event(None).map_err(|e| anyhow::anyhow!("Failed to create GPU start event for GPU {gpu_idx}: {:?}", e))
            })
            .collect::<Result<Vec<_>>>()?;
        let gpu_end_events: Vec<_> = (0..num_gpus)
            .map(|gpu_idx| {
                let ctx = self.streams.get(gpu_idx).unwrap().context();
                ctx.new_event(None).map_err(|e| anyhow::anyhow!("Failed to create GPU end event for GPU {gpu_idx}: {:?}", e))
            })
            .collect::<Result<Vec<_>>>()?;

        // Record start events
        for gpu_idx in 0..num_gpus {
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
            gpu_start_events[gpu_idx].record(&gpu_stream)
                .map_err(|e| anyhow::anyhow!("Failed to record start event on GPU {gpu_idx}: {:?}", e))?;
        }

        let manager = self.paged_kv_manager.as_mut()
            .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?;

        let num_gpus = self.weights.len();
        let config = &self.config;
        let page_size = manager.page_size();
        let head_dim = config.head_dim;
        let seq_len = token_ids.len();

        // Probe instrumentation
        let probe = probe::ProbeConfig::from_env();
        probe::dump_config(&self.config, num_gpus, self.group_size);

        // Allocate pages for the sequence
        let num_pages_needed = (token_ids.len().saturating_sub(1) / page_size) + 1;
        for _ in 0..num_pages_needed {
            manager.append_page(seq_id)?;
        }

        // Upload block table and positions to ALL GPUs
        let block_table = manager.block_table(seq_id)?;
        let block_table_i32: Vec<i32> = block_table.iter().map(|p| *p as i32).collect();
        let positions: Vec<u32> = (0..token_ids.len() as u32).collect();
        let positions_i32: Vec<i32> = positions.iter().map(|p| *p as i32).collect();

        // Per-GPU block tables and positions
        let mut block_tables_gpu: Vec<CudaSlice<i32>> = Vec::new();
        let mut positions_gpu_vec: Vec<CudaSlice<i32>> = Vec::new();
        for gpu_idx in 0..num_gpus {
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
            block_tables_gpu.push(gpu_stream.clone_htod(&block_table_i32)?);
            positions_gpu_vec.push(gpu_stream.clone_htod(&positions_i32)?);
        }

        // Ensure page pools allocated on each GPU
        for gpu_idx in 0..num_gpus {
            for cache in &mut self.paged_kv_caches[gpu_idx] {
                cache.ensure_allocated(self.streams.get(gpu_idx).unwrap())?;
            }
        }

        // Embed tokens on each GPU
        let mut hidden_states: Vec<CudaSlice<bf16>> = Vec::new();
        for gpu_idx in 0..num_gpus {
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
            let w = &self.weights[gpu_idx];
            let embed_weight = w.embedding.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
            let embed_table = self.weight_caches[gpu_idx].get_bf16(&embed_weight.name)
                .ok_or_else(|| anyhow::anyhow!("Embedding weight '{}' not in cache", embed_weight.name))?;
            let h = crate::embedding::embed_tokens(
                &gpu_stream, &self.per_gpu_kernels[gpu_idx].embedding, token_ids, &embed_table,
                config.hidden_size, config.vocab_size,
            )?;
            probe::dump(&gpu_stream, &probe, usize::MAX, gpu_idx, "embed.output", &h, &[seq_len, config.hidden_size], "prefill");
            hidden_states.push(h);
        }

        // Per-GPU sharded head counts
        let num_kv_heads_per_gpu = config.num_key_value_heads / num_gpus;
        let num_heads_per_gpu = config.num_attention_heads / num_gpus;
        let sharded_intermediate = config.intermediate_size / num_gpus;

        // Layer loop
        for layer_idx in 0..config.num_hidden_layers {
            tracing::info!("Layer {}/{} (phase A)", layer_idx + 1, config.num_hidden_layers);

            let layer_type = config.get_layer_type(layer_idx);
            let stage_prefix = match layer_type {
                LayerType::FullAttention => "attn",
                LayerType::GatedDeltaNet => "gdn",
            };
            let layer_span = tracing::info_span!(
                "layer",
                layer_idx,
                layer_type = match layer_type {
                    LayerType::FullAttention => "full_attn",
                    LayerType::GatedDeltaNet => "gdn",
                }
            );
            let _layer_enter = layer_span.enter();

            // Dump hidden input at start of layer for reference comparison
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.norm1_input", stage_prefix), &hidden_states[gpu_idx], &[seq_len, config.hidden_size], "prefill");
            }

            // Phase A: Attention/GDN on each GPU
            let mut attn_outputs: Vec<CudaSlice<bf16>> = Vec::new();

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                let gemm = &mut self.gemm_engines[gpu_idx];
                let w = &self.weights[gpu_idx];
                let layer = &w.layers[layer_idx];

                // Norm1
                let norm1_weight = self.weight_caches[gpu_idx].get_bf16(&layer.norm1.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm1 weight '{}' not in cache", layer.norm1.name))?;
                let norm1_out = crate::norm::rms_norm(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].rmsnorm, &hidden_states[gpu_idx], &norm1_weight,
                    config.rms_norm_eps, config.hidden_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.norm1", stage_prefix), &norm1_out, &[seq_len, config.hidden_size], "prefill");

                // Attention or GDN with sharded weights
                let attn_out = match config.get_layer_type(layer_idx) {
                    LayerType::GatedDeltaNet => {
                        let gdn_weights = layer.gdn.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("GDN weights not found for layer {}", layer_idx))?;
                        crate::gdn::forward(
                            gemm, &self.per_gpu_kernels[gpu_idx].int4_gemm, &gpu_stream,
                            &self.per_gpu_kernels[gpu_idx].gdn_recurrent_step,
                            &self.per_gpu_kernels[gpu_idx].gdn_chunked_prefill,
                            &self.per_gpu_kernels[gpu_idx].conv1d_depthwise,
                            &self.per_gpu_kernels[gpu_idx].rms_norm_gated,
                            gdn_weights, &norm1_out,
                            &mut self.gdn_states[gpu_idx][layer_idx],
                            config.hidden_size, config.as_ref(), self.group_size,
                            &self.weight_caches[gpu_idx],
                            layer_idx,
                            gpu_idx,
                            &probe,
                        )?
                    }
                    LayerType::FullAttention => {
                        let attn_weights = layer.attn.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
                        crate::attention::forward_paged(
                            gemm, &self.per_gpu_kernels[gpu_idx].int4_gemm, &gpu_stream,
                            &self.per_gpu_kernels[gpu_idx].softmax, &self.per_gpu_kernels[gpu_idx].paged_kv_write,
                            &self.per_gpu_kernels[gpu_idx].rope, &self.per_gpu_kernels[gpu_idx].rmsnorm, &self.per_gpu_kernels[gpu_idx].add,
                            &self.per_gpu_kernels[gpu_idx].attn_output_gate,
                            attn_weights, &norm1_out,
                            &mut self.paged_kv_caches[gpu_idx][layer_idx],
                            &block_tables_gpu[gpu_idx], &positions_gpu_vec[gpu_idx], &positions,
                            head_dim, num_heads_per_gpu, num_kv_heads_per_gpu, page_size,
                            config.rope_theta, config.partial_rotary_factor,
                            config.rms_norm_eps, self.group_size, &self.weight_caches[gpu_idx],
                            config.hidden_size,
                            config.attn_output_gate,
                            layer_idx,
                            gpu_idx,
                            &probe,
                        )?
                    }
                };

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.o_proj", stage_prefix), &attn_out, &[seq_len, config.hidden_size], "prefill");

                attn_outputs.push(attn_out);
            }

            // All-reduce attention outputs across GPUs (grouped)
            group_start().map_err(|e| anyhow::anyhow!("NCCL group_start failed: {:?}", e))?;
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_attention(
                    &self.nccl, &gpu_stream, &mut attn_outputs[gpu_idx],
                )?;
            }
            group_end().map_err(|e| anyhow::anyhow!("NCCL group_end failed: {:?}", e))?;

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.after_ar", stage_prefix), &attn_outputs[gpu_idx], &[seq_len, config.hidden_size], "prefill");
            }
            // Phase B: Residual add on each GPU
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                hidden_states[gpu_idx] = crate::add::add(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].add,
                    &hidden_states[gpu_idx], &attn_outputs[gpu_idx],
                )?;
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "residual.attn", &hidden_states[gpu_idx], &[seq_len, config.hidden_size], "prefill");
            }

            // ================================================================
            // Phase C: MLP on each GPU (column-parallel gate/up, row-parallel down)
            // ================================================================
            let mut mlp_outputs: Vec<CudaSlice<bf16>> = Vec::new();

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                let gemm = &mut self.gemm_engines[gpu_idx];
                let w = &self.weights[gpu_idx];
                let mlp_weights = &w.layers[layer_idx].mlp;

                // Norm2
                let norm2_weight = self.weight_caches[gpu_idx].get_bf16(&w.layers[layer_idx].norm2.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm2 weight '{}' not in cache", w.layers[layer_idx].norm2.name))?;
                let norm2_out = crate::norm::rms_norm(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].rmsnorm, &hidden_states[gpu_idx], &norm2_weight,
                    config.rms_norm_eps, config.hidden_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.norm2", &norm2_out, &[seq_len, config.hidden_size], "prefill");

                // Gate projection (column-parallel: sharded_intermediate output dim)
                let mut gate = gpu_stream.alloc_zeros::<bf16>(seq_len * sharded_intermediate)?;
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &self.per_gpu_kernels[gpu_idx].int4_gemm, &gpu_stream,
                    &self.weight_caches[gpu_idx],
                    &mlp_weights.gate_proj.name,
                    &norm2_out, &mut gate,
                    seq_len, sharded_intermediate, config.hidden_size,
                    self.group_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.gate_proj", &gate, &[seq_len, config.intermediate_size / num_gpus], "prefill");

                // Up projection (column-parallel: sharded_intermediate output dim)
                let mut up = gpu_stream.alloc_zeros::<bf16>(seq_len * sharded_intermediate)?;
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &self.per_gpu_kernels[gpu_idx].int4_gemm, &gpu_stream,
                    &self.weight_caches[gpu_idx],
                    &mlp_weights.up_proj.name,
                    &norm2_out, &mut up,
                    seq_len, sharded_intermediate, config.hidden_size,
                    self.group_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.up_proj", &up, &[seq_len, config.intermediate_size / num_gpus], "prefill");

                // SiLU(gate) * up (elementwise on sharded_intermediate)
                let mut silu_out = gpu_stream.alloc_zeros::<bf16>(seq_len * sharded_intermediate)?;
                let elem_i32 = (seq_len * sharded_intermediate) as i32;
                unsafe {
                    gpu_stream.launch_builder(&self.per_gpu_kernels[gpu_idx].silu_glu)
                        .arg(&up).arg(&gate).arg(&mut silu_out).arg(&elem_i32)
                        .launch(infers_cuda::LaunchConfig {
                            grid_dim: (((seq_len * sharded_intermediate) as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.silu", &silu_out, &[seq_len, config.intermediate_size / num_gpus], "prefill");

                // Down projection (row-parallel: full hidden_size output, sharded_intermediate inner dim)
                let mut mlp_out = gpu_stream.alloc_zeros::<bf16>(seq_len * config.hidden_size)?;
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &self.per_gpu_kernels[gpu_idx].int4_gemm, &gpu_stream,
                    &self.weight_caches[gpu_idx],
                    &mlp_weights.down_proj.name,
                    &silu_out, &mut mlp_out,
                    seq_len, config.hidden_size, sharded_intermediate,
                    self.group_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.down_raw", &mlp_out, &[seq_len, config.hidden_size], "prefill");

                mlp_outputs.push(mlp_out);
            }

            // All-reduce MLP outputs across GPUs (grouped)
            group_start().map_err(|e| anyhow::anyhow!("NCCL group_start failed: {:?}", e))?;
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_mlp(
                    &self.nccl, &gpu_stream, &mut mlp_outputs[gpu_idx],
                )?;
            }
group_end().map_err(|e| anyhow::anyhow!("NCCL group_end failed: {:?}", e))?;

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.down_ar", &mlp_outputs[gpu_idx], &[seq_len, config.hidden_size], "prefill");
            }

            // Phase D: Residual add on each GPU
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                hidden_states[gpu_idx] = crate::add::add(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].add,
                    &hidden_states[gpu_idx], &mlp_outputs[gpu_idx],
                )?;
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "residual.mlp", &hidden_states[gpu_idx], &[seq_len, config.hidden_size], "prefill");
            }
        }

        // Record prefill tokens in the KV manager so decode sees the correct count
        if let Some(mgr) = self.paged_kv_manager.as_mut() {
            mgr.add_tokens(seq_id, seq_len)
                .map_err(|e| anyhow::anyhow!("Failed to record prefill tokens: {:?}", e))?;
        }

        // ================================================================
        // Final norm + LM head on GPU 0 (same on all GPUs after all-reduce)
        // ================================================================
        let final_stream = self.streams.get(0).unwrap().clone();
        let final_weights = &self.weights[0];
        let mut final_hidden = hidden_states.into_iter().next().unwrap(); // GPU 0's hidden state

        // Final norm
        let final_norm_weight = final_weights.norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
        let final_norm_gpu = self.weight_caches[0].get_bf16(&final_norm_weight.name)
            .ok_or_else(|| anyhow::anyhow!("Final norm weight '{}' not in cache", final_norm_weight.name))?;
        final_hidden = crate::norm::rms_norm(
            &final_stream, &self.per_gpu_kernels[0].rmsnorm, &final_hidden, &final_norm_gpu,
            config.rms_norm_eps, config.hidden_size,
        )?;

        probe::dump(&final_stream, &probe, config.num_hidden_layers - 1, 0, "final.norm", &final_hidden, &[1, config.hidden_size], "prefill");

        // LM head
        let lm_head_weight = final_weights.lm_head.as_ref()
            .or_else(|| final_weights.embedding.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;
        let mut logits = final_stream.alloc_zeros::<bf16>(seq_len * config.vocab_size)?;
        crate::gemm_dispatch::gemm_projection_cached(
            &mut self.gemm_engines[0], &self.per_gpu_kernels[0].int4_gemm, &final_stream,
            &self.weight_caches[0],
            &lm_head_weight.name,
            &final_hidden, &mut logits,
            seq_len, config.vocab_size, config.hidden_size,
            self.group_size,
        )?;

        probe::dump(&final_stream, &probe, config.num_hidden_layers - 1, 0, "final.logits", &logits, &[seq_len, config.vocab_size], "prefill");

        // Sample: last row argmax
        let last_row_start = (seq_len - 1) * config.vocab_size;
        let last_row_logits = logits.slice(last_row_start..last_row_start + config.vocab_size);
        let sampled = crate::sample::sample_with_config(
            &final_stream, &last_row_logits, &self.per_gpu_kernels[0].argmax,
            sampling_config, token_ids, token_ids.len(), rng,
        )?;

        tracing::info!("Paged prefill sampled token: {}", sampled);

        // Record end events and report GPU timing
        let mut max_gpu_ms: f32 = 0.0;
        for gpu_idx in 0..num_gpus {
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
            gpu_end_events[gpu_idx].record(&gpu_stream)
                .map_err(|e| anyhow::anyhow!("Failed to record end event on GPU {gpu_idx}: {:?}", e))?;
            gpu_end_events[gpu_idx].synchronize()
                .map_err(|e| anyhow::anyhow!("Failed to synchronize end event on GPU {gpu_idx}: {:?}", e))?;
            let gpu_ms = gpu_start_events[gpu_idx].elapsed_ms(&gpu_end_events[gpu_idx])
                .unwrap_or(0.0);
            max_gpu_ms = max_gpu_ms.max(gpu_ms);
        }
        tracing::info!(gpu_time_ms = max_gpu_ms as f64, phase = "prefill", "GPU execution complete");

        Ok((num_pages_needed, sampled))
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
        _stream: &Arc<CudaStream>,
        token_id: u32,
        position: u32,
        seq_id: infers_kv::SequenceId,
        sampling_config: &infers_scheduler::SamplingConfig,
        token_history: &[u32],
        num_prompt_tokens: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<u32> {
        let span = tracing::info_span!("decode");
        let _enter = span.enter();
        let num_gpus = self.weights.len();

        // GPU timing: create events on each GPU's context
        let gpu_start_events: Vec<_> = (0..num_gpus)
            .map(|gpu_idx| {
                let ctx = self.streams.get(gpu_idx).unwrap().context();
                ctx.new_event(None).map_err(|e| anyhow::anyhow!("Failed to create GPU start event for GPU {gpu_idx}: {:?}", e))
            })
            .collect::<Result<Vec<_>>>()?;
        let gpu_end_events: Vec<_> = (0..num_gpus)
            .map(|gpu_idx| {
                let ctx = self.streams.get(gpu_idx).unwrap().context();
                ctx.new_event(None).map_err(|e| anyhow::anyhow!("Failed to create GPU end event for GPU {gpu_idx}: {:?}", e))
            })
            .collect::<Result<Vec<_>>>()?;

        // Record start events
        for gpu_idx in 0..num_gpus {
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
            gpu_start_events[gpu_idx].record(&gpu_stream)
                .map_err(|e| anyhow::anyhow!("Failed to record start event on GPU {gpu_idx}: {:?}", e))?;
        }

        let config = &self.config;
        let head_dim = config.head_dim;

        // Probe instrumentation
        let probe = probe::ProbeConfig::from_env();
        probe::dump_config(&self.config, num_gpus, self.group_size);

        // Dynamically allocate pages as needed for the target position,
        // then read the (possibly updated) block table and cached-token count.
        // Use a scope to drop the mutable borrow before using `self` again.
        let (page_size, num_cached_tokens, block_table_i32): (usize, i32, Vec<i32>) = {
            let mgr = self.paged_kv_manager.as_mut()
                .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?;
            let ps = mgr.page_size();

            // Allocate pages up to the page index that `position` falls in.
            let needed_pages = (position as usize / ps) + 1;
            let current_pages = mgr.block_table(seq_id)?.len();
            for _ in current_pages..needed_pages {
                mgr.append_page(seq_id)
                    .map_err(|e| anyhow::anyhow!("Failed to allocate KV page for decode: {:?}", e))?;
            }

            let cached = mgr.num_tokens(seq_id)? as i32 + 1;  // +1 for current decode token
            let bt: Vec<i32> = mgr.block_table(seq_id)?.iter().map(|p| *p as i32).collect();
            (ps, cached, bt)
        };
        let position_i32 = vec![position as i32];

        let mut block_tables_gpu: Vec<CudaSlice<i32>> = Vec::new();
        let mut positions_gpu_vec: Vec<CudaSlice<i32>> = Vec::new();
        for gpu_idx in 0..num_gpus {
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
            block_tables_gpu.push(gpu_stream.clone_htod(&block_table_i32)?);
            positions_gpu_vec.push(gpu_stream.clone_htod(&position_i32)?);
        }

        // Ensure page pools allocated on each GPU
        for gpu_idx in 0..num_gpus {
            for cache in &mut self.paged_kv_caches[gpu_idx] {
                cache.ensure_allocated(self.streams.get(gpu_idx).unwrap())?;
            }
        }

        // Embed single token on each GPU
        let mut hidden_states: Vec<CudaSlice<bf16>> = Vec::new();
        for gpu_idx in 0..num_gpus {
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
            let w = &self.weights[gpu_idx];
            let embed_weight = w.embedding.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
            let embed_table = self.weight_caches[gpu_idx].get_bf16(&embed_weight.name)
                .ok_or_else(|| anyhow::anyhow!("Embedding weight '{}' not in cache", embed_weight.name))?;
            let h = crate::embedding::embed_tokens(
                &gpu_stream, &self.per_gpu_kernels[gpu_idx].embedding, &[token_id], &embed_table,
                config.hidden_size, config.vocab_size,
            )?;
            probe::dump(&gpu_stream, &probe, usize::MAX, gpu_idx, "embed.output", &h, &[1, config.hidden_size], "decode");
            hidden_states.push(h);
        }

        // Per-GPU sharded head counts
        let num_kv_heads_per_gpu = config.num_key_value_heads / num_gpus;
        let num_heads_per_gpu = config.num_attention_heads / num_gpus;
        let sharded_intermediate = config.intermediate_size / num_gpus;

        // Layer loop
        for layer_idx in 0..config.num_hidden_layers {
            let layer_type = config.get_layer_type(layer_idx);
            let stage_prefix = match layer_type {
                LayerType::FullAttention => "attn",
                LayerType::GatedDeltaNet => "gdn",
            };
            let layer_span = tracing::info_span!(
                "layer",
                layer_idx,
                layer_type = match layer_type {
                    LayerType::FullAttention => "full_attn",
                    LayerType::GatedDeltaNet => "gdn",
                }
            );
            let _layer_enter = layer_span.enter();

            // Dump hidden input at start of layer
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.norm1_input", stage_prefix), &hidden_states[gpu_idx], &[1, config.hidden_size], "decode");
            }

            // Phase A: Attention/GDN on each GPU
            let mut attn_outputs: Vec<CudaSlice<bf16>> = Vec::new();

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                let gemm = &mut self.gemm_engines[gpu_idx];
                let w = &self.weights[gpu_idx];
                let layer = &w.layers[layer_idx];

                // Norm1
                let norm1_weight = self.weight_caches[gpu_idx].get_bf16(&layer.norm1.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm1 weight '{}' not in cache", layer.norm1.name))?;
                let norm1_out = crate::norm::rms_norm(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].rmsnorm, &hidden_states[gpu_idx], &norm1_weight,
                    config.rms_norm_eps, config.hidden_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.norm1", stage_prefix), &norm1_out, &[1, config.hidden_size], "decode");

                // Attention or GDN (decode versions)
                let attn_out = match config.get_layer_type(layer_idx) {
                    LayerType::GatedDeltaNet => {
                        let gdn_weights = layer.gdn.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("GDN weights not found for layer {}", layer_idx))?;
                        crate::gdn::decode_forward(
                            gemm, &self.per_gpu_kernels[gpu_idx].int4_gemm, &gpu_stream,
                            &self.per_gpu_kernels[gpu_idx].gdn_recurrent_step,
                            &self.per_gpu_kernels[gpu_idx].conv1d_depthwise,
                            &self.per_gpu_kernels[gpu_idx].rms_norm_gated,
                            gdn_weights, &norm1_out,
                            &mut self.gdn_states[gpu_idx][layer_idx],
                            config.hidden_size, config.as_ref(), self.group_size,
                            &self.weight_caches[gpu_idx],
                            layer_idx,
                            gpu_idx,
                            &probe,
                        )?
                    }
                    LayerType::FullAttention => {
                        let attn_weights = layer.attn.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
                        crate::attention::decode_forward_paged(
                            gemm, &self.per_gpu_kernels[gpu_idx].int4_gemm, &gpu_stream,
                            &self.per_gpu_kernels[gpu_idx].paged_kv_write, &self.per_gpu_kernels[gpu_idx].paged_attention_decode,
                            &self.per_gpu_kernels[gpu_idx].rope, &self.per_gpu_kernels[gpu_idx].rmsnorm, &self.per_gpu_kernels[gpu_idx].add,
                            &self.per_gpu_kernels[gpu_idx].attn_output_gate,
                            attn_weights, &norm1_out,
                            &mut self.paged_kv_caches[gpu_idx][layer_idx],
                            &block_tables_gpu[gpu_idx], &positions_gpu_vec[gpu_idx],
                            position, num_cached_tokens,
                            head_dim, num_heads_per_gpu, num_kv_heads_per_gpu, page_size,
                            config.rope_theta, config.partial_rotary_factor,
                            config.rms_norm_eps, self.group_size, &self.weight_caches[gpu_idx],
                            config.hidden_size,
                            config.attn_output_gate,
                            layer_idx,
                            gpu_idx,
                            &probe,
                        )?
                    }
                };

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.o_proj", stage_prefix), &attn_out, &[1, config.hidden_size], "decode");

                attn_outputs.push(attn_out);
            }

            // All-reduce attention outputs across GPUs (grouped)
            group_start().map_err(|e| anyhow::anyhow!("NCCL group_start failed: {:?}", e))?;
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_attention(
                    &self.nccl, &gpu_stream, &mut attn_outputs[gpu_idx],
                )?;
            }
            group_end().map_err(|e| anyhow::anyhow!("NCCL group_end failed: {:?}", e))?;

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.after_ar", stage_prefix), &attn_outputs[gpu_idx], &[1, config.hidden_size], "decode");
            }

            // Phase B: Residual add on each GPU
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                hidden_states[gpu_idx] = crate::add::add(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].add,
                    &hidden_states[gpu_idx], &attn_outputs[gpu_idx],
                )?;
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "residual.attn", &hidden_states[gpu_idx], &[1, config.hidden_size], "decode");
            }

            // Phase C: MLP on each GPU (column-parallel gate/up, row-parallel down)
            let mut mlp_outputs: Vec<CudaSlice<bf16>> = Vec::new();

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                let gemm = &mut self.gemm_engines[gpu_idx];
                let w = &self.weights[gpu_idx];
                let mlp_weights = &w.layers[layer_idx].mlp;

                // Norm2
                let norm2_weight = self.weight_caches[gpu_idx].get_bf16(&w.layers[layer_idx].norm2.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm2 weight '{}' not in cache", w.layers[layer_idx].norm2.name))?;
                let norm2_out = crate::norm::rms_norm(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].rmsnorm, &hidden_states[gpu_idx], &norm2_weight,
                    config.rms_norm_eps, config.hidden_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.norm2", &norm2_out, &[1, config.hidden_size], "decode");

                // Gate projection (column-parallel: sharded_intermediate output dim)
                let mut gate = gpu_stream.alloc_zeros::<bf16>(sharded_intermediate)?;
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &self.per_gpu_kernels[gpu_idx].int4_gemm, &gpu_stream,
                    &self.weight_caches[gpu_idx],
                    &mlp_weights.gate_proj.name,
                    &norm2_out, &mut gate,
                    1, sharded_intermediate, config.hidden_size,
                    self.group_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.gate_proj", &gate, &[1, config.intermediate_size / num_gpus], "decode");

                // Up projection (column-parallel: sharded_intermediate output dim)
                let mut up = gpu_stream.alloc_zeros::<bf16>(sharded_intermediate)?;
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &self.per_gpu_kernels[gpu_idx].int4_gemm, &gpu_stream,
                    &self.weight_caches[gpu_idx],
                    &mlp_weights.up_proj.name,
                    &norm2_out, &mut up,
                    1, sharded_intermediate, config.hidden_size,
                    self.group_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.up_proj", &up, &[1, config.intermediate_size / num_gpus], "decode");

                // up * SiLU(gate) = SwiGLU (elementwise on sharded_intermediate)
                let mut silu_out = gpu_stream.alloc_zeros::<bf16>(sharded_intermediate)?;
                let elem_i32 = sharded_intermediate as i32;
                unsafe {
                    gpu_stream.launch_builder(&self.per_gpu_kernels[gpu_idx].silu_glu)
                        .arg(&up).arg(&gate).arg(&mut silu_out).arg(&elem_i32)
                        .launch(infers_cuda::LaunchConfig {
                            grid_dim: ((sharded_intermediate as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.silu", &silu_out, &[1, config.intermediate_size / num_gpus], "decode");

                // Down projection (row-parallel: full hidden_size output)
                let mut mlp_out = gpu_stream.alloc_zeros::<bf16>(config.hidden_size)?;
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &self.per_gpu_kernels[gpu_idx].int4_gemm, &gpu_stream,
                    &self.weight_caches[gpu_idx],
                    &mlp_weights.down_proj.name,
                    &silu_out, &mut mlp_out,
                    1, config.hidden_size, sharded_intermediate,
                    self.group_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.down_raw", &mlp_out, &[1, config.hidden_size], "decode");

                mlp_outputs.push(mlp_out);
            }

            // All-reduce MLP outputs across GPUs (grouped)
            group_start().map_err(|e| anyhow::anyhow!("NCCL group_start failed: {:?}", e))?;
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_mlp(
                    &self.nccl, &gpu_stream, &mut mlp_outputs[gpu_idx],
                )?;
            }
            group_end().map_err(|e| anyhow::anyhow!("NCCL group_end failed: {:?}", e))?;

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.down_ar", &mlp_outputs[gpu_idx], &[1, config.hidden_size], "decode");
            }

            // Phase D: Residual add on each GPU
            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                hidden_states[gpu_idx] = crate::add::add(
                    &gpu_stream, &self.per_gpu_kernels[gpu_idx].add,
                    &hidden_states[gpu_idx], &mlp_outputs[gpu_idx],
                )?;
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "residual.mlp", &hidden_states[gpu_idx], &[1, config.hidden_size], "decode");
            }
        }

        // ================================================================
        // Final norm + LM head + sample on GPU 0
        // ================================================================
        let final_stream = self.streams.get(0).unwrap().clone();
        let final_weights = &self.weights[0];
        let mut final_hidden = hidden_states.into_iter().next().unwrap();

        // Final norm
        let final_norm_weight = final_weights.norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
        let final_norm_gpu = self.weight_caches[0].get_bf16(&final_norm_weight.name)
            .ok_or_else(|| anyhow::anyhow!("Final norm weight '{}' not in cache", final_norm_weight.name))?;
        final_hidden = crate::norm::rms_norm(
            &final_stream, &self.per_gpu_kernels[0].rmsnorm, &final_hidden, &final_norm_gpu,
            config.rms_norm_eps, config.hidden_size,
        )?;

        probe::dump(&final_stream, &probe, config.num_hidden_layers - 1, 0, "final.norm", &final_hidden, &[1, config.hidden_size], "decode");

        // LM head
        let lm_head_weight = final_weights.lm_head.as_ref()
            .or_else(|| final_weights.embedding.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;
        let mut logits = final_stream.alloc_zeros::<bf16>(config.vocab_size)?;
        crate::gemm_dispatch::gemm_projection_cached(
            &mut self.gemm_engines[0], &self.per_gpu_kernels[0].int4_gemm, &final_stream,
            &self.weight_caches[0],
            &lm_head_weight.name,
            &final_hidden, &mut logits,
            1, config.vocab_size, config.hidden_size,
            self.group_size,
        )?;

        probe::dump(&final_stream, &probe, config.num_hidden_layers - 1, 0, "final.logits", &logits, &[1, config.vocab_size], "decode");

        // Sample (BF16 argmax)
        let sampled = crate::sample::sample_with_config(
            &final_stream, &logits.as_view(), &self.per_gpu_kernels[0].argmax,
            sampling_config, token_history, num_prompt_tokens, rng,
        )?;

        // Record the new token in the KV manager so the next decode step
        // sees the correct block table and cached-token count.
        if let Some(mgr) = self.paged_kv_manager.as_mut() {
            mgr.add_token(seq_id)
                .map_err(|e| anyhow::anyhow!("Failed to record decode token: {:?}", e))?;
        }

        // Record end events and report GPU timing
        let mut max_gpu_ms: f32 = 0.0;
        for gpu_idx in 0..num_gpus {
            let gpu_stream = self.streams.get(gpu_idx).unwrap().clone();
            gpu_end_events[gpu_idx].record(&gpu_stream)
                .map_err(|e| anyhow::anyhow!("Failed to record end event on GPU {gpu_idx}: {:?}", e))?;
            gpu_end_events[gpu_idx].synchronize()
                .map_err(|e| anyhow::anyhow!("Failed to synchronize end event on GPU {gpu_idx}: {:?}", e))?;
            let gpu_ms = gpu_start_events[gpu_idx].elapsed_ms(&gpu_end_events[gpu_idx])
                .unwrap_or(0.0);
            max_gpu_ms = max_gpu_ms.max(gpu_ms);
        }
        tracing::info!(gpu_time_ms = max_gpu_ms as f64, phase = "decode", "GPU execution complete");

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
        // For eviction, iterate GPU 0 only (TP eviction can be updated later)
        if let Some(gpu_cache) = self.paged_kv_caches.first() {
            for &page_id in &block_table {
                for (layer_idx, cache) in gpu_cache.iter().enumerate() {
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
            // For restore, use GPU 0 only (TP restore can be updated later)
            let gpu_cache = &mut self.paged_kv_caches[0];
            for layer_idx in 0..gpu_cache.len() {
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
                let cache = &mut gpu_cache[layer_idx];
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
}
