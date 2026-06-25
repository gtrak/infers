//! Shared GPU resource and decode state structs for async pipeline construction.
//!
//! `GpuResources` holds all immutable GPU resources behind `Arc`, enabling
//! cheap cloning across cuda-async closures. `DecodeState` holds per-sequence
//! mutable state wrapped in `Arc<Mutex<>>` for sharing across `and_then` closures.

use std::sync::Arc;

use infers_cuda::{CudaSlice, CudaStream};
use infers_cuda::gemm::GemmEngine;
use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::stream::StreamPool;
use infers_model::{ModelConfig, WeightRegistry};

use crate::gpu_cache::GpuWeightCache;
use crate::workspace::DecodeWorkspace;
use crate::gdn::GdnState;
use crate::attention::PagedKvCache;
use crate::engine::PerGpuKernels;
use crate::probe;
use infers_kv::PagedKvManager;

/// Immutable GPU resources shared across all sequences.
///
/// All fields are `Send + Sync`, making this safe to share via `Arc` across
/// cuda-async closure boundaries. Cloning the `Arc<GpuResources>` is cheap —
/// no GPU data is copied.
// @lat: [[lat.md/lat#Forward Engine#GpuResources and DecodeState Architecture]]
pub struct GpuResources {
    /// Model architecture configuration.
    pub config: Arc<ModelConfig>,

    /// Weight metadata for name lookups during inference.
    pub metadata: Vec<WeightRegistry>,

    /// Per-GPU weight caches with GPU-resident buffers.
    pub weight_caches: Vec<GpuWeightCache>,

    /// Per-GPU cached kernel function handles.
    pub per_gpu_kernels: Vec<PerGpuKernels>,

    /// cuBLASLt GEMM engines (one per GPU for tensor parallelism).
    pub gemm_engines: Vec<GemmEngine>,

    /// NCCL communicator for tensor-parallel all-reduce.
    pub nccl: NcclCommunicator,

    /// Async CUDA streams for parallel execution.
    pub streams: StreamPool,

    /// Precomputed RoPE cos/sin tables on GPU (uploaded once at init, one per GPU).
    pub rope_cos: Option<Vec<CudaSlice<f32>>>,

    /// Precomputed RoPE sin tables on GPU (uploaded once at init, one per GPU).
    pub rope_sin: Option<Vec<CudaSlice<f32>>>,

    /// INT4 quantization group size for on-the-fly dequantization.
    pub group_size: usize,

    /// Cached ProbeConfig — avoids per-step env::var lookups.
    pub probe_config: probe::ProbeConfig,
}

/// Per-sequence mutable state for decode operations.
///
/// Wrapped in `Arc<Mutex<>>` for sharing across cuda-async `and_then` closures.
/// The mutex is only held during sequential execution — no concurrent access occurs.
// @lat: [[lat.md/lat#Forward Engine#GpuResources and DecodeState Architecture]]
pub struct DecodeState {
    /// Per-GPU pre-allocated workspace buffers for the decode hot path.
    pub workspaces: Vec<DecodeWorkspace>,

    /// Per-GPU, per-layer paged KV caches (new paged system).
    pub paged_kv_caches: Vec<Vec<PagedKvCache>>,

    /// Per-GPU, per-layer GDN recurrent states.
    pub gdn_states: Vec<Vec<GdnState>>,

    /// Paged KV cache manager (pool + prefix cache + COW).
    pub paged_kv_manager: Option<PagedKvManager>,
}

impl std::fmt::Debug for DecodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeState")
            .field("workspaces", &format!("[{} workspaces]", self.workspaces.len()))
            .field("paged_kv_caches", &format!("[{}] GPUs", self.paged_kv_caches.len()))
            .field("gdn_states", &format!("[{}] GPUs", self.gdn_states.len()))
            .finish()
    }
}
