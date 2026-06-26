//! Forward inference engine — owns GPU state, kernels, and execution logic.
//!
//! `ForwardEngine` is the central struct that coordinates all inference steps:
//! embedding lookup, layer dispatch (GDN vs full attention), MLP, normalization,
//! and sampling. It holds references to CUDA resources, model weights, and kernels.

use crate::probe;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use infers_cuda::gemm::GemmEngine;
use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::stream::StreamPool;
use infers_cuda::PinnedHostBuffer;
use infers_cuda::{CudaContext, CudaStream};
use infers_model::{LayerType, MmapWeightRegistry, ModelConfig, WeightRegistry};

use crate::attention::{KvCache, PagedKvCache};
use crate::gpu_cache::GpuWeightCache;
use crate::eviction::BackendEvictionStore;
use crate::gdn::GdnState;
use crate::workspace::DecodeWorkspace;
use crate::resources::{GpuResources, DecodeState};

use infers_kv::PagedKvManager;

use half::bf16;
use infers_cuda::{CudaGraph, CudaSlice};

use infers_cuda::{group_end, group_start};
use crate::sync;
use crate::sample::Xoshiro256PlusPlus;

/// Force the C allocator to return freed memory to the OS.
///
/// After dropping large allocations (weight data), glibc's malloc keeps
/// the virtual address space mapped in thread-local arenas. This calls
/// `malloc_trim(0)` which returns all possible memory to the OS,
/// significantly reducing VmData and VmRSS.
#[cfg(target_os = "linux")]
fn trim_memory() {
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(not(target_os = "linux"))]
fn trim_memory() {
    // No-op on non-Linux
}

/// Per-GPU cached kernel function handles.
pub(crate) struct PerGpuKernels {
    /// Oxide bridge for all kernel launches.
    pub(crate) oxide: Arc<infers_cuda::OxideKernels>,
}

/// Central engine for forward-pass inference.
///
/// Owns all GPU resources: CUDA contexts, streams, loaded kernels, cuBLASLt handles,
/// and NCCL communicators. Coordinates the full prefill/decode pipeline.
///
/// Immutable GPU resources are wrapped in `Arc<GpuResources>` for cheap sharing
/// across cuda-async closures. Per-sequence mutable state lives in `DecodeState`,
/// which is temporarily taken during async decode operations.
pub struct ForwardEngine {
    /// Immutable GPU resources shared via Arc — cheap to clone across closures.
    pub(crate) resources: Arc<GpuResources>,

    /// Paged KV cache manager (pool + prefix cache + COW).
    pub(crate) paged_kv_manager: Option<Arc<Mutex<PagedKvManager>>>,

    /// Per-GPU, per-layer KV caches for full-attention layers (flat cache, legacy).
    pub(crate) kv_caches: Vec<Vec<KvCache>>,          // [gpu_idx][layer_idx]
    /// Per-GPU, per-layer paged KV caches (new paged system).
    pub(crate) paged_kv_caches: Vec<Vec<PagedKvCache>>,  // [gpu_idx][layer_idx]

    /// Per-GPU, per-layer GDN recurrent states.
    pub(crate) gdn_states: Vec<Vec<GdnState>>,      // [gpu_idx][layer_idx]

    /// Per-GPU pre-allocated workspace buffers for the decode hot path.
    /// Eliminates per-token alloc_zeros calls.
    pub(crate) workspaces: Vec<DecodeWorkspace>,

    /// CUDA graph for decode loop replay (one per GPU). None until first capture.
    pub(crate) decode_graphs: Vec<Option<CudaGraph>>,

    /// Step counter for graph capture scheduling.
    pub(crate) decode_step_count: usize,
}

impl ForwardEngine {
    /// Load CUDA kernel modules on each GPU and return per-GPU kernel handles.
    fn load_per_gpu_kernels(
        contexts: &[Arc<CudaContext>],
        num_gpus: usize,
    ) -> Result<Vec<PerGpuKernels>> {
        let mut per_gpu_kernels = Vec::with_capacity(num_gpus);

        // Create one oxide bridge per GPU — each loads the cubin on its device's context
        let cubin_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../cuda/kernels/compiled/oxide_kernels.cubin");

        for gpu_idx in 0..num_gpus {
            let _ = contexts.get(gpu_idx)
                .ok_or_else(|| anyhow::anyhow!("Missing context for GPU {gpu_idx}"))?;
            let oxide: Arc<infers_cuda::OxideKernels> = Arc::new(
                infers_cuda::OxideKernels::new(gpu_idx, cubin_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load OxideKernels for GPU {gpu_idx} from {}: {e}", cubin_path))?
            );
            let pk = PerGpuKernels {
                oxide,
            };
            per_gpu_kernels.push(pk);
        }
        Ok(per_gpu_kernels)
    }

    /// Create one cuBLASLt GEMM engine per GPU.
    fn create_gemm_engines(streams: &StreamPool, num_gpus: usize) -> Result<Vec<GemmEngine>> {
        let mut gemm_engines = Vec::with_capacity(num_gpus);
        for i in 0..num_gpus {
            let s = streams.get(i).ok_or_else(|| anyhow::anyhow!("Missing stream {i}"))?;
            gemm_engines.push(
                GemmEngine::new(s.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to create cuBLASLt engine for GPU {i}: {e}"))?
            );
        }
        Ok(gemm_engines)
    }

    /// Create NCCL communicator for tensor parallelism.
    fn create_nccl(streams: &StreamPool) -> Result<NcclCommunicator> {
        let comm_streams: Vec<Arc<CudaStream>> = (0..streams.len())
            .filter_map(|i| streams.get(i).cloned())
            .collect();
        NcclCommunicator::new(comm_streams)
            .map_err(|e| anyhow::anyhow!("Failed to initialize NCCL: {e}"))
    }

    /// Initialize per-GPU, per-layer KV caches, GDN states, and paged KV caches.
    fn init_layer_states(
        num_gpus: usize,
        num_layers: usize,
    ) -> (Vec<Vec<KvCache>>, Vec<Vec<GdnState>>, Vec<Vec<PagedKvCache>>) {
        let kv_caches = (0..num_gpus).map(|_| (0..num_layers).map(|_| KvCache::new()).collect()).collect();
        let gdn_states = (0..num_gpus).map(|_| (0..num_layers).map(|_| GdnState::new()).collect()).collect();
        let paged_kv_caches = (0..num_gpus).map(|_| Vec::new()).collect();
        (kv_caches, gdn_states, paged_kv_caches)
    }

    /// Construct a `ForwardEngine` from model config, weights, and GPU resources.
    ///
    /// # Arguments
    /// * `config` — Model architecture parameters
    /// * `weights` — Per-GPU weight registries (one per tensor-parallel rank)
    /// * `contexts` — CUDA contexts for kernel loading
    /// * `streams` — Pool of async CUDA streams
    /// * `nccl` — NCCL communicator for multi-GPU collectives
    pub fn new(
        config: Arc<ModelConfig>,
        weights: Vec<WeightRegistry>,
        contexts: Vec<Arc<CudaContext>>,
        streams: StreamPool,
        group_size: usize,
    ) -> Result<Self> {
        let num_gpus = streams.len();

        // Initialize cuda-async thread-local device contexts for async pipeline
        // scheduling. Required for DeviceOperation::sync() / .await / async_on().
        // Must be called before any async operations. Uses the default device (0)
        // and prepares the context map for num_gpus devices.
        cuda_async::device_context::init_device_contexts(0, num_gpus)
            .map_err(|e| anyhow::anyhow!("Failed to init cuda-async device contexts: {e:?}"))?;

        let mut weights = weights; // mutable for clear_data() after GPU upload
        let per_gpu_kernels = Self::load_per_gpu_kernels(&contexts, num_gpus)?;
        let gemm_engines = Self::create_gemm_engines(&streams, num_gpus)?;
        let nccl = Self::create_nccl(&streams)?;

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

        // Free CPU-side weight data now that it's on the GPU.
        // This drops ~5 GB per GPU of persistent heap residency.
        for registry in &mut weights {
            registry.clear_data();
        }

        trim_memory();

        // Initialize per-GPU, per-layer caches and states
        let (kv_caches, gdn_states, paged_kv_caches) = Self::init_layer_states(num_gpus, config.num_hidden_layers);

        // Precompute RoPE tables and upload to each GPU at init time
        let (rope_cos_cpu, rope_sin_cpu) = crate::rope::precompute_rope_tables(
            config.max_position_embeddings as u32,
            config.head_dim,
            config.rope_theta,
            config.partial_rotary_factor,
        );

        let mut rope_cos: Vec<CudaSlice<f32>> = Vec::with_capacity(num_gpus);
        let mut rope_sin: Vec<CudaSlice<f32>> = Vec::with_capacity(num_gpus);
        for gpu_idx in 0..num_gpus {
            let gpu_stream = streams.get(gpu_idx).unwrap().clone();
            rope_cos.push(gpu_stream.clone_htod(&rope_cos_cpu)
                .map_err(|e| anyhow::anyhow!("Failed to upload RoPE cos table for GPU {}: {}", gpu_idx, e))?);
            rope_sin.push(gpu_stream.clone_htod(&rope_sin_cpu)
                .map_err(|e| anyhow::anyhow!("Failed to upload RoPE sin table for GPU {}: {}", gpu_idx, e))?);
        }

        // Pre-allocate decode workspace buffers for each GPU
        let sharded_intermediate = config.intermediate_size / num_gpus;
        let mut workspaces = Vec::with_capacity(num_gpus);
        for gpu_idx in 0..num_gpus {
            let gpu_stream = streams.get(gpu_idx).unwrap().clone();
            workspaces.push(DecodeWorkspace::new(
                &gpu_stream,
                config.as_ref(),
                config.hidden_size,
                sharded_intermediate,
                config.vocab_size,
                num_gpus,
            )?);
        }

        tracing::info!(
            "ForwardEngine initialized: {} layers, {} GPU shards",
            config.num_hidden_layers,
            weights.len()
        );

        Ok(Self {
            resources: Arc::new(GpuResources {
                config,
                metadata: weights,
                weight_caches,
                per_gpu_kernels,
                gemm_engines,
                nccl,
                streams,
                rope_cos: Some(rope_cos),
                rope_sin: Some(rope_sin),
                group_size,
                probe_config: probe::ProbeConfig::from_env(),
            }),
            paged_kv_manager: None,
            kv_caches,
            paged_kv_caches,
            gdn_states,
            workspaces,
            decode_graphs: (0..num_gpus).map(|_| None).collect(),
            decode_step_count: 0,
        })
    }

    /// Access the per-GPU weight caches (diagnostic access for testing).
    pub fn weight_caches(&self) -> &[GpuWeightCache] {
        &self.resources.weight_caches
    }

    /// Construct a `ForwardEngine` from mmap-backed weight registries.
    ///
    /// Similar to [`new`] but loads weights via zero-copy memory-mapped access.
    /// mmap handles are dropped after GPU upload to free page cache.
    ///
    /// # Arguments
    /// * `config` — Model architecture parameters
    /// * `mmap_registries` — Per-GPU mmap weight registries (one per tensor-parallel rank)
    /// * `contexts` — CUDA contexts for kernel loading
    /// * `streams` — Pool of async CUDA streams
    /// * `pinned` — Pinned host buffer for dtype conversion during upload
    /// * `group_size` — INT4 quantization group size for on-the-fly dequantization
    pub fn new_from_mmap(
        config: Arc<ModelConfig>,
        mmap_registries: Vec<MmapWeightRegistry>,
        metadata_registries: Vec<WeightRegistry>,
        contexts: Vec<Arc<CudaContext>>,
        streams: StreamPool,
        pinned: &mut PinnedHostBuffer,
        group_size: usize,
    ) -> Result<Self> {
       let num_gpus = streams.len();
        let mut mmap_registries = mmap_registries; // mutable for clear_owned_data after upload
        let per_gpu_kernels = Self::load_per_gpu_kernels(&contexts, num_gpus)?;
        let gemm_engines = Self::create_gemm_engines(&streams, num_gpus)?;
        let nccl = Self::create_nccl(&streams)?;

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

        // Free heap-owned sharded data now that it's on the GPU.
        // This drops ~2 GB of persistent heap residency from fused QKV per-segment shards.
        for registry in &mut mmap_registries {
            registry.clear_owned_data();
        }

        trim_memory();

        // Drop all mmap references to unmap the safetensor files from page cache.
        // The GPU weight cache holds the actual buffers; CPU-side data is no longer needed.
        // This frees ~17 GB of shared page cache for the Qwen3.6-27B model.
        let num_shards = mmap_registries.len();
        drop(mmap_registries);

        // Initialize per-GPU, per-layer caches and states
        let (kv_caches, gdn_states, paged_kv_caches) = Self::init_layer_states(num_gpus, config.num_hidden_layers);

        // Precompute RoPE tables and upload to each GPU at init time
        let (rope_cos_cpu, rope_sin_cpu) = crate::rope::precompute_rope_tables(
            config.max_position_embeddings as u32,
            config.head_dim,
            config.rope_theta,
            config.partial_rotary_factor,
        );

        let mut rope_cos: Vec<CudaSlice<f32>> = Vec::with_capacity(num_gpus);
        let mut rope_sin: Vec<CudaSlice<f32>> = Vec::with_capacity(num_gpus);
        for gpu_idx in 0..num_gpus {
            let gpu_stream = streams.get(gpu_idx).unwrap().clone();
            rope_cos.push(gpu_stream.clone_htod(&rope_cos_cpu)
                .map_err(|e| anyhow::anyhow!("Failed to upload RoPE cos table for GPU {}: {}", gpu_idx, e))?);
            rope_sin.push(gpu_stream.clone_htod(&rope_sin_cpu)
                .map_err(|e| anyhow::anyhow!("Failed to upload RoPE sin table for GPU {}: {}", gpu_idx, e))?);
        }

        // Pre-allocate decode workspace buffers for each GPU
        let sharded_intermediate = config.intermediate_size / num_gpus;
        let mut workspaces = Vec::with_capacity(num_gpus);
        for gpu_idx in 0..num_gpus {
            let gpu_stream = streams.get(gpu_idx).unwrap().clone();
            workspaces.push(DecodeWorkspace::new(
                &gpu_stream,
                config.as_ref(),
                config.hidden_size,
                sharded_intermediate,
                config.vocab_size,
                num_gpus,
            )?);
        }

        tracing::info!(
            "ForwardEngine initialized (mmap): {} layers, {} GPU shards",
            config.num_hidden_layers,
            num_shards
        );

        Ok(Self {
            resources: Arc::new(GpuResources {
                config,
                metadata: metadata_registries, // metadata registries for name lookups during inference
                weight_caches,
                per_gpu_kernels,
                gemm_engines,
                nccl,
                streams,
                rope_cos: Some(rope_cos),
                rope_sin: Some(rope_sin),
                group_size,
                probe_config: probe::ProbeConfig::from_env(),
            }),
            paged_kv_manager: None,
            kv_caches,
            paged_kv_caches,
            gdn_states,
            workspaces,
            decode_graphs: (0..num_gpus).map(|_| None).collect(),
            decode_step_count: 0,
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
        let res = &self.resources;
        let weights = &res.metadata[0];

        let kernels = crate::prefill::PrefillKernels {
            oxide: res.per_gpu_kernels[0].oxide.clone(),
        };

        crate::prefill::prefill(
            &res.gemm_engines[0], stream, &kernels, &res.nccl,
            &res.config, weights, &res.weight_caches[0], token_ids,
            &mut self.kv_caches[0], &mut self.gdn_states[0],
            res.group_size,
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
        let res = &self.resources;
        let config = &res.config;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

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
        let num_gpus = res.metadata.len();
        let kv_dim_per_gpu = (num_kv_heads / num_gpus) * head_dim;

        let caches: Vec<Vec<PagedKvCache>> = (0..num_gpus)
            .map(|_| {
                (0..config.num_hidden_layers)
                    .map(|_| PagedKvCache::new(total_pages, page_size, kv_dim_per_gpu))
                    .collect()
            })
            .collect();

        self.paged_kv_manager = Some(Arc::new(Mutex::new(manager)));
        self.paged_kv_caches = caches;
        tracing::info!(
            "Paged KV system initialized: {} pages, page_size={}, {} layers",
            total_pages,
            page_size,
            config.num_hidden_layers
        );

        Ok(())
    }

    /// Create a new sequence in the paged KV manager, returning its ID.
    pub fn create_sequence(&mut self) -> infers_kv::SequenceId {
        self.paged_kv_manager
            .as_ref()
            .map(|m| m.lock().unwrap().create_sequence())
            .unwrap_or(0)
    }

    /// Allocate a new DecodeState for a sequence.

    /// Creates separate per-GPU workspaces and GDN recurrent states that can be

    /// used with [[Self::decode_with_state]]. The PagedKvManager is NOT included

    /// in the returned state — it stays owned by the engine as shared CPU-side

    /// bookkeeping (page pool, block tables per sequence). Callers must provide

    /// it via `decode_with_state`'s engine borrow.

    // @lat: [[lat.md/lat#Forward Engine#GpuResources and DecodeState Architecture#Per-Sequence DecodeState Management]]
    pub fn create_decode_state(&self) -> Result<DecodeState> {
        let res = &self.resources;
        let num_gpus = res.metadata.len();
        let config = &res.config;

        // Allocate workspaces — same logic as ForwardEngine::new
        let sharded_intermediate = config.intermediate_size / num_gpus;
        let mut workspaces = Vec::with_capacity(num_gpus);
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            workspaces.push(DecodeWorkspace::new(
                &gpu_stream, config.as_ref(),
                config.hidden_size, sharded_intermediate,
                config.vocab_size, num_gpus,
            )?);
        }

        // GDN states — same logic as ForwardEngine::new / init_layer_states
        let gdn_states = (0..num_gpus)
            .map(|_| (0..config.num_hidden_layers).map(|_| GdnState::new()).collect())
            .collect();

        // PagedKV caches are NOT allocated here — they live on the engine and
        // are shared across sequences via the global page pool.
        let paged_kv_caches = Vec::new();

        Ok(DecodeState {
            workspaces,
            paged_kv_caches,
            gdn_states,
            paged_kv_manager: None, // managed by engine, not per-sequence
        })
    }

    /// Prepare a DecodeState for batched decode by cloning shared paged KV state.
    ///
    /// Clones the paged_kv_manager Arc and paged_kv_caches (shallow GPU handle clones)
    /// into the state. Each state gets its own handle to the shared page pool.
    pub fn prepare_batched_state(&mut self, state: &mut DecodeState) {
        state.paged_kv_manager = self.paged_kv_manager.clone();
        state.paged_kv_caches = self.paged_kv_caches.clone();
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
    // @lat: [[lat.md/lat#Forward Engine#Paged Prefill Path]]
    pub fn prefill_paged(
        &mut self,
        _stream: &Arc<CudaStream>,
        token_ids: &[u32],
        seq_id: infers_kv::SequenceId,
        sampling_config: &infers_scheduler::SamplingConfig,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<(usize, u32)> {
        let res = &self.resources;
        let span = tracing::info_span!("prefill", num_tokens = token_ids.len());
        let _enter = span.enter();

        // GPU timing: create events on each GPU's context
        let num_gpus = res.metadata.len();
        let gpu_start_events: Vec<_> = (0..num_gpus)
            .map(|gpu_idx| {
                let ctx = res.streams.get(gpu_idx).unwrap().context();
                ctx.new_event(None).map_err(|e| anyhow::anyhow!("Failed to create GPU start event for GPU {gpu_idx}: {:?}", e))
            })
            .collect::<Result<Vec<_>>>()?;
        let gpu_end_events: Vec<_> = (0..num_gpus)
            .map(|gpu_idx| {
                let ctx = res.streams.get(gpu_idx).unwrap().context();
                ctx.new_event(None).map_err(|e| anyhow::anyhow!("Failed to create GPU end event for GPU {gpu_idx}: {:?}", e))
            })
            .collect::<Result<Vec<_>>>()?;

        // Record start events
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            gpu_start_events[gpu_idx].record(&gpu_stream)
                .map_err(|e| anyhow::anyhow!("Failed to record start event on GPU {gpu_idx}: {:?}", e))?;
        }
       let (page_size, block_table_i32, positions_i32): (usize, Vec<i32>, Vec<i32>) = {
           let mut manager = self.paged_kv_manager.as_ref()
               .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?
               .lock()
               .unwrap();

           let page_size = manager.page_size();

           // Allocate pages for the sequence
           let num_pages_needed = (token_ids.len().saturating_sub(1) / page_size) + 1;
           for _ in 0..num_pages_needed {
               manager.append_page(seq_id)?;
           }

           let block_table = manager.block_table(seq_id)?;
           let block_table_i32: Vec<i32> = block_table.iter().map(|p| *p as i32).collect();
           let positions_i32: Vec<i32> = (0..token_ids.len() as i32).collect();
           (page_size, block_table_i32, positions_i32)
       };
       let num_gpus = res.metadata.len();
       let config = &res.config;
       let head_dim = config.head_dim;
       let seq_len = token_ids.len();
       let probe = probe::ProbeConfig::from_env();
       probe::dump_config(&res.config, num_gpus, res.group_size);
       let page_size = self.paged_kv_manager.as_ref()
           .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?
           .lock()
           .unwrap()
           .page_size();
       let num_pages_needed = (token_ids.len().saturating_sub(1) / page_size) + 1;
       let positions: Vec<u32> = (0..token_ids.len() as u32).collect();

        // Per-GPU block tables and positions
        let mut block_tables_gpu: Vec<CudaSlice<i32>> = Vec::new();
        let mut positions_gpu_vec: Vec<CudaSlice<i32>> = Vec::new();
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            block_tables_gpu.push(gpu_stream.clone_htod(&block_table_i32)?);
            positions_gpu_vec.push(gpu_stream.clone_htod(&positions_i32)?);
        }

        // Ensure page pools allocated on each GPU
      for gpu_idx in 0..num_gpus {
            for cache in &mut self.paged_kv_caches[gpu_idx] {
                cache.ensure_allocated(res.streams.get(gpu_idx).unwrap())?;
            }
        }
        // Embed tokens on each GPU

        let mut hidden_states: Vec<CudaSlice<bf16>> = Vec::new();
        for gpu_idx in 0..num_gpus {
            let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
            let w = &res.metadata[gpu_idx];
            let embed_weight = w.embedding.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
            let embed_table = res.weight_caches[gpu_idx].get_bf16(&embed_weight.name)
                .ok_or_else(|| anyhow::anyhow!("Embedding weight '{}' not in cache", embed_weight.name))?;
            let h = crate::embedding::embed_tokens(
                &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide, token_ids, &embed_table,
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
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.norm1_input", stage_prefix), &hidden_states[gpu_idx], &[seq_len, config.hidden_size], "prefill");
            }

            // Phase A: Attention/GDN on each GPU
            let mut attn_outputs: Vec<CudaSlice<bf16>> = Vec::new();

            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                let gemm = &res.gemm_engines[gpu_idx];
                let w = &res.metadata[gpu_idx];
                let layer = &w.layers[layer_idx];

                // Norm1
                let norm1_weight = res.weight_caches[gpu_idx].get_bf16(&layer.norm1.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm1 weight '{}' not in cache", layer.norm1.name))?;
                let norm1_out = crate::norm::rms_norm(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide, &hidden_states[gpu_idx], &norm1_weight,
                    config.rms_norm_eps, config.hidden_size,
                )?;
              probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.norm1", stage_prefix), &norm1_out, &[seq_len, config.hidden_size], "prefill");

                // Attention or GDN with sharded weights
                let attn_out = match config.get_layer_type(layer_idx) {
                    LayerType::GatedDeltaNet => {
                        let gdn_weights = layer.gdn.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("GDN weights not found for layer {}", layer_idx))?;
                crate::gdn::forward(
                              gemm, &gpu_stream,
                              &res.per_gpu_kernels[gpu_idx].oxide,
                             gdn_weights, &norm1_out,
                            &mut self.gdn_states[gpu_idx][layer_idx],
                            config.hidden_size, config.as_ref(), res.group_size,
                            &res.weight_caches[gpu_idx],
                            layer_idx,
                            gpu_idx,
                            &probe,
                        )?
                    }
                    LayerType::FullAttention => {
                        let attn_weights = layer.attn.as_ref()
                            .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
               crate::attention::forward_paged(
                             gemm, &gpu_stream,
                             &res.per_gpu_kernels[gpu_idx].oxide,
                            attn_weights, &norm1_out,
                            &mut self.paged_kv_caches[gpu_idx][layer_idx],
                            &block_tables_gpu[gpu_idx], &positions_gpu_vec[gpu_idx], &positions,
                            head_dim, num_heads_per_gpu, num_kv_heads_per_gpu, page_size,
                            config.rope_theta, config.partial_rotary_factor,
                            config.rms_norm_eps, res.group_size, &res.weight_caches[gpu_idx],
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
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_attention(
                    &res.nccl, &gpu_stream, &mut attn_outputs[gpu_idx],
                )?;
            }
            group_end().map_err(|e| anyhow::anyhow!("NCCL group_end failed: {:?}", e))?;


            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, &format!("{}.after_ar", stage_prefix), &attn_outputs[gpu_idx], &[seq_len, config.hidden_size], "prefill");
            }
            // Phase B: Residual add on each GPU
            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                hidden_states[gpu_idx] = crate::add::add(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide,
                    &hidden_states[gpu_idx], &attn_outputs[gpu_idx],
                )?;
            }


            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "residual.attn", &hidden_states[gpu_idx], &[seq_len, config.hidden_size], "prefill");
            }

            // ================================================================
            // Phase C: MLP on each GPU (column-parallel gate/up, row-parallel down)
            // ================================================================
            let mut mlp_outputs: Vec<CudaSlice<bf16>> = Vec::new();

            for gpu_idx in 0..num_gpus {
                let mut _ps = None; // pre-fill never uses partial_sums buffer
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                let gemm = &res.gemm_engines[gpu_idx];
                let w = &res.metadata[gpu_idx];
                let mlp_weights = &w.layers[layer_idx].mlp;

                // Norm2
                let norm2_weight = res.weight_caches[gpu_idx].get_bf16(&w.layers[layer_idx].norm2.name)
                    .ok_or_else(|| anyhow::anyhow!("Norm2 weight '{}' not in cache", w.layers[layer_idx].norm2.name))?;
                let norm2_out = crate::norm::rms_norm(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide, &hidden_states[gpu_idx], &norm2_weight,
                    config.rms_norm_eps, config.hidden_size,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.norm2", &norm2_out, &[seq_len, config.hidden_size], "prefill");

                // Gate projection (column-parallel: sharded_intermediate output dim)
                let mut gate = gpu_stream.alloc_zeros::<bf16>(seq_len * sharded_intermediate)?;
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &res.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                    &res.weight_caches[gpu_idx],
                    &mlp_weights.gate_proj.name,
                    &norm2_out, &mut gate,
                    seq_len, sharded_intermediate, config.hidden_size,
                    res.group_size,
                    &mut _ps,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.gate_proj", &gate, &[seq_len, config.intermediate_size / num_gpus], "prefill");

                // Up projection (column-parallel: sharded_intermediate output dim)
                let mut up = gpu_stream.alloc_zeros::<bf16>(seq_len * sharded_intermediate)?;
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &res.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                    &res.weight_caches[gpu_idx],
                    &mlp_weights.up_proj.name,
                    &norm2_out, &mut up,
                    seq_len, sharded_intermediate, config.hidden_size,
                    res.group_size,
                    &mut _ps,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.up_proj", &up, &[seq_len, config.intermediate_size / num_gpus], "prefill");

                // SiLU(gate) * up (elementwise on sharded_intermediate)
                let mut silu_out = gpu_stream.alloc_zeros::<bf16>(seq_len * sharded_intermediate)?;
                res.per_gpu_kernels[gpu_idx].oxide.launch_silu_glu_bf16(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide.cc_stream(), &up, &gate, &mut silu_out, (seq_len * sharded_intermediate) as u32,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.silu", &silu_out, &[seq_len, config.intermediate_size / num_gpus], "prefill");

                // Down projection (row-parallel: full hidden_size output, sharded_intermediate inner dim)
                let mut mlp_out = gpu_stream.alloc_zeros::<bf16>(seq_len * config.hidden_size)?;
                crate::gemm_dispatch::gemm_projection_cached(
                    gemm, &res.per_gpu_kernels[gpu_idx].oxide, &gpu_stream,
                    &res.weight_caches[gpu_idx],
                    &mlp_weights.down_proj.name,
                    &silu_out, &mut mlp_out,
                    seq_len, config.hidden_size, sharded_intermediate,
                    res.group_size,
                    &mut _ps,
                )?;

                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.down_raw", &mlp_out, &[seq_len, config.hidden_size], "prefill");

                mlp_outputs.push(mlp_out);
            }

            // All-reduce MLP outputs across GPUs (grouped)
            group_start().map_err(|e| anyhow::anyhow!("NCCL group_start failed: {:?}", e))?;
            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                sync::all_reduce_mlp(
                    &res.nccl, &gpu_stream, &mut mlp_outputs[gpu_idx],
                )?;
            }
group_end().map_err(|e| anyhow::anyhow!("NCCL group_end failed: {:?}", e))?;


            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "mlp.down_ar", &mlp_outputs[gpu_idx], &[seq_len, config.hidden_size], "prefill");
            }

        // Phase D: Residual add on each GPU
            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                hidden_states[gpu_idx] = crate::add::add(
                    &gpu_stream, &res.per_gpu_kernels[gpu_idx].oxide,
                    &hidden_states[gpu_idx], &mlp_outputs[gpu_idx],
                )?;
            }

            for gpu_idx in 0..num_gpus {
                let gpu_stream = res.streams.get(gpu_idx).unwrap().clone();
                probe::dump(&gpu_stream, &probe, layer_idx, gpu_idx, "residual.mlp", &hidden_states[gpu_idx], &[seq_len, config.hidden_size], "prefill");
            }


        }

        // Record prefill tokens in the KV manager so decode sees the correct count
        if let Some(m) = &self.paged_kv_manager {
            m.lock().unwrap().add_tokens(seq_id, seq_len)
                .map_err(|e| anyhow::anyhow!("Failed to record prefill tokens: {:?}", e))?;
        }

        // ================================================================
        // Final norm + LM head on GPU 0 (same on all GPUs after all-reduce)
        // ================================================================
        let final_stream = res.streams.get(0).unwrap().clone();
        let final_weights = &res.metadata[0];
        let mut final_hidden = hidden_states.into_iter().next().unwrap(); // GPU 0's hidden state

        // Final norm
        let final_norm_weight = final_weights.norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
        let final_norm_gpu = res.weight_caches[0].get_bf16(&final_norm_weight.name)
            .ok_or_else(|| anyhow::anyhow!("Final norm weight '{}' not in cache", final_norm_weight.name))?;
        final_hidden = crate::norm::rms_norm(
            &final_stream, &res.per_gpu_kernels[0].oxide, &final_hidden, &final_norm_gpu,
            config.rms_norm_eps, config.hidden_size,
        )?;

        probe::dump(&final_stream, &probe, config.num_hidden_layers - 1, 0, "final.norm", &final_hidden, &[1, config.hidden_size], "prefill");

        // LM head
        let lm_head_weight = final_weights.lm_head.as_ref()
            .or_else(|| final_weights.embedding.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;
        let mut logits = final_stream.alloc_zeros::<bf16>(seq_len * config.vocab_size)?;
        let mut _ps = None;
        crate::gemm_dispatch::gemm_projection_cached(
            &res.gemm_engines[0], &res.per_gpu_kernels[0].oxide, &final_stream,
            &res.weight_caches[0],
            &lm_head_weight.name,
            &final_hidden, &mut logits,
            seq_len, config.vocab_size, config.hidden_size,
            res.group_size,
            &mut _ps,
        )?;

        probe::dump(&final_stream, &probe, config.num_hidden_layers - 1, 0, "final.logits", &logits, &[seq_len, config.vocab_size], "prefill");

        // Sample: last row argmax
        let last_row_start = (seq_len - 1) * config.vocab_size;
        let last_row_logits = logits.slice(last_row_start..last_row_start + config.vocab_size);
        let sampled = crate::sample::sample_with_config(
            &final_stream, &last_row_logits, &res.per_gpu_kernels[0].oxide,
            sampling_config, token_ids, token_ids.len(), rng,
        )?;

        tracing::info!("Paged prefill sampled token: {}", sampled);

        // GPU timing: disabled to avoid implicit stream synchronization via cudaEventSynchronize
        // (prevents CUDA graph capture in subsequent decode steps).
        tracing::info!(phase = "prefill", "GPU execution complete");

        Ok((num_pages_needed, sampled))
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
