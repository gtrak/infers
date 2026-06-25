//! Standard softmax attention forward pass and paged KV cache attention.
//!
//! Implements both the original flat-cache attention and the new paged attention
//! subsystem for zero CPU round-trip decode. Flat-cache path uses GPU-resident
//! weight cache with full GEMM projections + per-head extraction via device-to-device
//! memcpy, eliminating CPU dequantization overhead. Paged path adds block-pool
//! KV management for prefix caching.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, OxideKernels};
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_model::AttentionWeights;

use infers_kv::KvCacheDtype;


use crate::gpu_cache::GpuWeightCache;
use crate::probe;
use crate::probe::ProbeConfig;
use crate::rope;

/// Block size used by paged attention kernels.
const BLOCK_SIZE: usize = 256;

/// KV cache for a single attention layer.
///
/// Stores a single contiguous BF16 buffer of shape `[2 × max_seq_len × head_dim]`
/// (K cache followed by V cache) on the GPU.
#[derive(Debug)]
pub struct KvCache {
    /// GPU buffer for KV cache data (`[2 × max_seq_len × head_dim]`).
    buffer: Option<CudaSlice<bf16>>,
    /// Maximum sequence length (cache dimension).
    max_seq_len: usize,
    /// KV dimension per cache entry (`num_kv_heads × head_dim`).
    kv_dim: usize,
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KvCache {
    /// Create an empty (unallocated) KV cache.
    pub fn new() -> Self {
        Self {
            buffer: None,
            max_seq_len: 0,
            kv_dim: 0,
        }
    }

    /// Ensure the GPU buffer is allocated with the given dimensions.
    ///
    /// Allocates lazily on first call; reuses the buffer on subsequent calls
    /// if the dimensions match.
    fn ensure_allocated(
        &mut self,
        stream: &Arc<CudaStream>,
        max_seq_len: usize,
        kv_dim: usize,
    ) -> Result<&CudaSlice<bf16>> {
        if self.buffer.is_none() || self.max_seq_len != max_seq_len || self.kv_dim != kv_dim {
            let total = 2 * max_seq_len * kv_dim;
            self.buffer = Some(
                stream
                    .alloc_zeros::<bf16>(total)
                    .map_err(|e| anyhow::anyhow!("Failed to allocate KV cache buffer: {e}"))?,
            );
            self.max_seq_len = max_seq_len;
            self.kv_dim = kv_dim;
        }
        Ok(self.buffer.as_ref().expect("buffer should be allocated"))
    }
}

/// Paged KV cache for a single attention layer.
///
/// Stores the GPU page pool buffer and per-sequence block table for
/// paged attention. Unlike the old flat KvCache, this enables:
/// - Zero CPU round-trips during decode
/// - Prefix sharing across sequences
/// - Copy-on-write page sharing
///
/// @lat: [[lat.md/lat#Paged Attention Implementation#PagedKvCache]]
#[derive(Debug)]
pub struct PagedKvCache {
    /// GPU buffer holding all paged KV data: [num_pages * 2 * page_size * kv_dim].
    /// Layout per page: [K tokens | V tokens], each side = page_size * kv_dim elements.
    page_pool: Option<CudaSlice<bf16>>,
    /// Total number of pages in the pool.
    num_pages: usize,
    /// Page size (tokens per page).
    page_size: usize,
    /// KV dimension (num_kv_heads * head_dim).
    kv_dim: usize,
}

impl PagedKvCache {
    /// Create an empty (unallocated) paged KV cache.
    pub fn new(num_pages: usize, page_size: usize, kv_dim: usize) -> Self {
        Self {
            page_pool: None,
            num_pages,
            page_size,
            kv_dim,
        }
    }

    /// Ensure the GPU page pool buffer is allocated.
    ///
    /// Allocates lazily on first call; reuses the buffer on subsequent calls
    /// if the dimensions match.
    ///
    /// Total elements: `num_pages * 2 * page_size * kv_dim` (K + V per page).
    pub fn ensure_allocated(
        &mut self,
        stream: &Arc<CudaStream>,
    ) -> Result<&CudaSlice<bf16>> {
        if self.page_pool.is_none() {
            let total = self.num_pages * 2 * self.page_size * self.kv_dim;
            self.page_pool = Some(
                stream
                    .alloc_zeros::<bf16>(total)
                    .map_err(|e| anyhow::anyhow!("Failed to allocate paged KV page pool: {e}"))?,
            );
        }
        Ok(self.page_pool.as_ref().expect("page pool should be allocated"))
    }

    /// Get a reference to the GPU page pool buffer.
    pub fn page_pool(&self) -> Option<&CudaSlice<bf16>> {
        self.page_pool.as_ref()
    }

    /// Get a mutable reference to the GPU page pool buffer.
    pub fn page_pool_mut(&mut self) -> Option<&mut CudaSlice<bf16>> {
        self.page_pool.as_mut()
    }

    /// Number of pages in the pool.
    pub fn num_pages(&self) -> usize {
        self.num_pages
    }

    /// Page size (tokens per page).
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// KV dimension (num_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

// ============================================================================
// Paged Kernel Dispatch Functions
// ============================================================================

/// Launch `infers_paged_kv_write_bf16` to write K and V into the paged KV cache.
///
/// # Arguments
/// * `stream` — CUDA stream
/// * `oxide` — Oxide bridge for kernel dispatch
/// * `k` — K tensor `[seq_len × kv_dim]`
/// * `v` — V tensor `[seq_len × kv_dim]`
/// * `page_pool` — Flat GPU buffer for paged KV data
/// * `block_table_gpu` — Block table (page IDs) on GPU `[num_pages]`
/// * `positions_gpu` — Token positions on GPU `[seq_len]`
/// * `seq_len` — Number of tokens to write
/// * `head_dim` — Per-head dimension
/// * `kv_dim` — num_kv_heads × head_dim
/// * `page_size` — Tokens per page
// @lat: [[lat.md/lat#Paged Attention Implementation#Paged Kernel Dispatch]]
pub fn paged_kv_write(
    stream: &Arc<CudaStream>,
    oxide: &infers_cuda::OxideKernels,
    k: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    page_pool: &mut CudaSlice<bf16>,
    block_table_gpu: &CudaSlice<i32>,
    positions_gpu: &CudaSlice<i32>,
    seq_len: usize,
    head_dim: usize,
    kv_dim: usize,
    page_size: usize,
) -> Result<()> {
    oxide.launch_paged_kv_write_bf16(
        stream, k, v, page_pool, block_table_gpu, positions_gpu,
        seq_len as u32, head_dim as u32, page_size as u32, kv_dim as u32,
    ).map_err(|e| anyhow::anyhow!("Paged KV write kernel launch failed: {e}"))?;

    Ok(())
}

/// Launch `infers_paged_attention_decode_bf16` for full decode attention
/// in a single kernel call.
///
/// Computes Q×K scores, softmax, and V accumulation across all cached tokens
/// and KV heads in one kernel launch. Returns `[num_kv_heads * head_dim]` output.
///
/// # Arguments
/// * `stream` — CUDA stream
/// * `oxide` — Oxide bridge for kernel dispatch
/// * `q` — Query tensor `[num_kv_heads × head_dim]` (single token)
/// * `page_pool` — Flat GPU buffer for paged KV data
/// * `block_table_gpu` — Block table on GPU `[num_pages]`
/// * `num_pages` — Number of pages in block table
/// * `num_cached_tokens` — Number of cached tokens
/// * `head_dim` — Per-head dimension
/// * `num_query_heads` — Number of query heads (for GQA)
/// * `num_kv_heads` — Number of KV heads
/// * `page_size` — Tokens per page
/// * `kv_dim` — num_kv_heads × head_dim
pub fn paged_attention_decode(
    stream: &Arc<CudaStream>,
    oxide: &infers_cuda::OxideKernels,
    q: &CudaSlice<bf16>,
    page_pool: &CudaSlice<bf16>,
    block_table_gpu: &CudaSlice<i32>,
    num_pages: usize,
    num_cached_tokens: usize,
    head_dim: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    kv_dim: usize,
) -> Result<CudaSlice<bf16>> {
    let output_size = num_query_heads * head_dim;
    let mut output = stream
        .alloc_zeros::<bf16>(output_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate attention output buffer: {e}"))?;

    oxide.launch_paged_attention_decode_bf16(
        stream, q, page_pool, block_table_gpu, &mut output,
        num_pages as u32, num_cached_tokens as u32, head_dim as u32,
        num_kv_heads as u32, num_query_heads as u32, page_size as u32, kv_dim as u32,
    ).map_err(|e| anyhow::anyhow!("Paged attention decode kernel launch failed: {e}"))?;

    Ok(output)
}

/// Paged attention decode, writing into a pre-allocated output buffer (zero-alloc variant).
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_decode_into(
    stream: &Arc<CudaStream>,
    oxide: &infers_cuda::OxideKernels,
    q: &CudaSlice<bf16>,
    page_pool: &CudaSlice<bf16>,
    block_table_gpu: &CudaSlice<i32>,
    output: &mut CudaSlice<bf16>,
    num_pages: usize,
    num_cached_tokens: usize,
    head_dim: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    kv_dim: usize,
) -> Result<()> {
    oxide.launch_paged_attention_decode_bf16(
        stream, q, page_pool, block_table_gpu, output,
        num_pages as u32, num_cached_tokens as u32, head_dim as u32,
        num_kv_heads as u32, num_query_heads as u32, page_size as u32, kv_dim as u32,
    ).map_err(|e| anyhow::anyhow!("Paged attention decode kernel failed: {e}"))?;
    Ok(())
}

// ============================================================================
// Original Flat-Cache Functions (preserved for backward compatibility)
// ============================================================================

/// Full-attention forward pass for a single transformer layer (prefill path).
///
/// Uses per-head weight slicing: each head's Q/K/V/O weights are extracted
/// on the CPU, uploaded to GPU, and processed with per-head GEMMs.
///
/// # Steps
/// 1. Compute full K and V (for KV cache write)
/// 2. Apply RoPE to full K (with dummy Q buffer)
/// 3. Write RoPE'd K and V to KV cache
/// 4. Per-head: Q_h, K_h, V_h projections → RoPE → scores → softmax → attn_out → partial O-proj
/// 5. Accumulate partial O-proj results into final output
///
/// # Arguments
/// * `gemm` — cuBLASLt engine for projections
/// * `stream` — CUDA stream for kernel launches
/// * `softmax_kernel` — CUDA kernel for softmax
/// * `kv_cache_write_kernel` — CUDA kernel for KV cache write
/// * `oxide` — Oxide bridge for norm and rope kernels
/// * `add_kernel` — CUDA kernel for element-wise addition
/// * `weights` — Attention weights for this layer
/// * `input` — Input tensor `[seq_len × hidden_size]`
/// * `kv_cache` — KV cache state for this layer
/// * `positions` — Position indices for RoPE embedding
/// * `head_dim` — Per-head dimension
/// * `num_heads` — Number of attention heads
/// * `num_kv_heads` — Number of KV heads (must equal num_heads for now)
/// * `max_seq_len` — Maximum sequence length for cache allocation
/// * `rope_theta` — RoPE base frequency
/// * `partial_rotary_factor` — Fraction of head_dim to apply RoPE to
///
/// # Returns
/// Attention output `[seq_len × hidden_size]`
pub fn forward(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    weights: &AttentionWeights,
    input: &CudaSlice<bf16>,
    kv_cache: &mut KvCache,
    positions: &[u32],
    hidden_size: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    rope_theta: f64,
    partial_rotary_factor: f32,
    rms_norm_eps: f32,
    group_size: usize,
    cache: &GpuWeightCache,
    attn_output_gate: bool,
) -> Result<CudaSlice<bf16>> {
    let kv_dim = num_kv_heads * head_dim;
    let seq_len = positions.len();

    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "num_heads {} must be divisible by num_kv_heads {} for GQA",
        num_heads, num_kv_heads
    );

    // =========================================================================
    // Phase 1: Full K, V computation + RoPE + KV cache write
    // =========================================================================

    let mut _ps = None; // prefill doesn't use pre-allocated partial_sums buffer

    // k_full = GEMM(input, k_proj^T)  [seq_len × kv_dim] (INT4-aware)
    let mut k_full = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate K buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.k_proj.name, input, &mut k_full,
        seq_len, kv_dim, hidden_size, group_size,
        &mut _ps,
    )?;

    // v_full = GEMM(input, v_proj^T)  [seq_len × kv_dim] (INT4-aware)
    let mut v_full = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate V buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.v_proj.name, input, &mut v_full,
        seq_len, kv_dim, hidden_size, group_size,
        &mut _ps,
    )?;

    // --- K-norm on full K before Phase 1 RoPE ---
    if let Some(k_norm_w) = weights.k_norm.as_ref() {
        let k_norm_gpu = cache.get_bf16(&k_norm_w.name)
            .ok_or_else(|| anyhow::anyhow!("K-norm weight '{}' not in cache", k_norm_w.name))?;
        k_full = crate::norm::rms_norm(
            stream, oxide, &k_full, &k_norm_gpu, rms_norm_eps, head_dim,
        )?;
    }

    // Apply RoPE to K_full. rope::apply_rope modifies both Q and K in-place;
    // we allocate a dummy Q buffer and discard it.
    let mut q_dummy = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dummy Q buffer for RoPE: {e}"))?;
   rope::apply_rope(
            stream,
            oxide,
            &mut q_dummy,
            &mut k_full,
            positions,
            num_kv_heads as i32,
            head_dim,
            rope_theta,
            partial_rotary_factor,
            None, None, // prefill path: no cached tables yet
        )?;


   // Write K and V to KV cache
    let _ = kv_cache.ensure_allocated(stream, max_seq_len, kv_dim)?;
    let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
    let positions_gpu = stream
        .clone_htod(&positions_i32)
        .map_err(|e| anyhow::anyhow!("Failed to copy positions to device: {e}"))?;

   oxide.launch_kv_cache_write_bf16(
        stream, &k_full, &v_full, kv_cache.buffer.as_mut().unwrap(), &positions_gpu,
        seq_len as u32, kv_dim as u32, max_seq_len as u32,
    ).map_err(|e| anyhow::anyhow!("KV cache write kernel launch failed: {e}"))?;

    // =========================================================================
    // Phase 2: Full Q projection + combined attention output buffer
    // =========================================================================

    let buf_size = seq_len * hidden_size;
    let per_gpu_head_dim = num_heads * head_dim;

    // When attn_output_gate is true, the Q projection produces doubled output:
    // [Q_head_0, G_head_0, Q_head_1, G_head_1, ...] per row (per-head interleaved).
    let q_out_dim = per_gpu_head_dim * if attn_output_gate { 2 } else { 1 };

    // q_full = GEMM(input, q_proj^T)  [seq_len × q_out_dim]
    let mut q_full = stream
        .alloc_zeros::<bf16>(seq_len * q_out_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate Q buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.q_proj.name, input, &mut q_full,
        seq_len, q_out_dim, hidden_size, group_size,
        &mut _ps,
    )?;

    // --- Q-norm on Q portion only (not gate) before RoPE ---
    if let Some(q_norm_w) = weights.q_norm.as_ref() {
        let q_norm_gpu = cache.get_bf16(&q_norm_w.name)
            .ok_or_else(|| anyhow::anyhow!("Q-norm weight '{}' not in cache", q_norm_w.name))?;
        // Extract Q per-head from interleaved layout for norm
        let mut q_only = stream.alloc_zeros::<bf16>(seq_len * per_gpu_head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate Q-only buffer for norm: {e}"))?;
        for s in 0..seq_len {
            for h in 0..num_heads {
                let src_offset = s * q_out_dim + h * (head_dim * 2);
                let dst_offset = s * per_gpu_head_dim + h * head_dim;
                let src_slice = q_full.slice(src_offset..src_offset + head_dim);
                let mut dst_slice = q_only.slice_mut(dst_offset..dst_offset + head_dim);
                stream.memcpy_dtod(&src_slice, &mut dst_slice)
                    .map_err(|e| anyhow::anyhow!("Copy Q portion for norm failed: {e}"))?;
            }
        }
        let q_normed = crate::norm::rms_norm(
            stream, oxide, &q_only, &q_norm_gpu, rms_norm_eps, head_dim,
        )?;
        // Write normalized Q back into interleaved positions
        for s in 0..seq_len {
            for h in 0..num_heads {
                let src_offset = s * per_gpu_head_dim + h * head_dim;
                let dst_offset = s * q_out_dim + h * (head_dim * 2);
                let src_slice = q_normed.slice(src_offset..src_offset + head_dim);
                let mut dst_slice = q_full.slice_mut(dst_offset..dst_offset + head_dim);
                stream.memcpy_dtod(&src_slice, &mut dst_slice)
                    .map_err(|e| anyhow::anyhow!("Write normalized Q back failed: {e}"))?;
            }
        }
    }

    // --- Gate extraction from interleaved layout ---
    let gate_heads = if attn_output_gate {
        let mut gate_buf = stream.alloc_zeros::<bf16>(seq_len * per_gpu_head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate gate buffer: {e}"))?;
        for s in 0..seq_len {
            for h in 0..num_heads {
                let src_offset = s * q_out_dim + h * (head_dim * 2) + head_dim;
                let dst_offset = s * per_gpu_head_dim + h * head_dim;
                let src_slice = q_full.slice(src_offset..src_offset + head_dim);
                let mut dst_slice = gate_buf.slice_mut(dst_offset..dst_offset + head_dim);
                stream.memcpy_dtod(&src_slice, &mut dst_slice)
                    .map_err(|e| anyhow::anyhow!("Copy gate data from q_full failed: {e}"))?;
            }
        }
        Some(gate_buf)
    } else {
        None
    };

    // --- Combined attention output buffer [seq_len x per_gpu_head_dim] ---
    let attn_combined_size = seq_len * per_gpu_head_dim;
    let mut attn_combined = stream
        .alloc_zeros::<bf16>(attn_combined_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate attn_combined buffer: {e}"))?;
    for head_idx in 0..num_heads {
        let kv_head_idx = head_idx / (num_heads / num_kv_heads);

        // --- Extract per-head Q from q_full via GPU copy ---
        // q_full has per-head interleaved layout: [Q_h0, G_h0, Q_h1, G_h1, ...] when gate enabled
        let head_stride = if attn_output_gate { head_dim * 2 } else { head_dim };
        let mut q_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate q_h buffer: {e}"))?;
        for s in 0..seq_len {
            let src_offset = s * q_out_dim + head_idx * head_stride;
            let dst_offset = s * head_dim;
            let src_slice = q_full.slice(src_offset..src_offset + head_dim);
            let mut dst_slice = q_h.slice_mut(dst_offset..dst_offset + head_dim);
            stream
                .memcpy_dtod(&src_slice, &mut dst_slice)
                .map_err(|e| anyhow::anyhow!("Copy per-head Q from q_full failed: {e}"))?;
        }

        // --- Extract per-head K from k_full (already has RoPE applied) ---
        let mut k_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate k_h buffer: {e}"))?;
        for s in 0..seq_len {
            let src_offset = s * kv_dim + kv_head_idx * head_dim;
            let dst_offset = s * head_dim;
            let src_slice = k_full.slice(src_offset..src_offset + head_dim);
            let mut dst_slice = k_h.slice_mut(dst_offset..dst_offset + head_dim);
            stream
                .memcpy_dtod(&src_slice, &mut dst_slice)
                .map_err(|e| anyhow::anyhow!("Copy per-head K from k_full failed: {e}"))?;
        }

        // --- Extract per-head V from v_full ---
        let mut v_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate v_h buffer: {e}"))?;
        for s in 0..seq_len {
            let src_offset = s * kv_dim + kv_head_idx * head_dim;
            let dst_offset = s * head_dim;
            let src_slice = v_full.slice(src_offset..src_offset + head_dim);
            let mut dst_slice = v_h.slice_mut(dst_offset..dst_offset + head_dim);
            stream
                .memcpy_dtod(&src_slice, &mut dst_slice)
                .map_err(|e| anyhow::anyhow!("Copy per-head V from v_full failed: {e}"))?;
        }

        // --- RoPE (per-head, num_heads=1) — apply only to q_h (k_h already has RoPE from Phase 1) ---
        let mut k_h_dummy = stream.alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate dummy K buffer for RoPE: {e}"))?;
        rope::apply_rope(
            stream,
            oxide,
            &mut q_h,
            &mut k_h_dummy,  // dummy — k_h already has RoPE from Phase 1
            positions,
            1,
            head_dim,
            rope_theta,
            partial_rotary_factor,
            None, None,
        )?;

        // --- Attention scores: Q_h @ K_h^T → [seq_len × seq_len] ---
        let scores_size = seq_len * seq_len;
        let mut scores_h = stream
            .alloc_zeros::<bf16>(scores_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate scores buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: seq_len,
                k: head_dim,
                transa: true,
                transb: false,
                alpha: 1.0 / (head_dim as f32).sqrt(),
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &q_h,
            &k_h,
            &mut scores_h,
        )?;

        // --- Softmax with causal masking ---
        let mut softmax_out_h = stream
            .alloc_zeros::<bf16>(scores_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate softmax output buffer: {e}"))?;

        oxide.launch_softmax_bf16(
            stream, &scores_h, &mut softmax_out_h, seq_len as u32, 1u32,
        ).map_err(|e| anyhow::anyhow!("Softmax kernel launch failed: {e}"))?;

        // --- Attention output: softmax_out_h @ V_h → [seq_len × head_dim] ---
        let mut attn_out_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate attn_out_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: head_dim,
                n: seq_len,
                k: seq_len,
                transa: false,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &v_h,
            &softmax_out_h,
            &mut attn_out_h,
        )?;

        // --- Copy attention output to combined buffer at correct head offset ---
        for s in 0..seq_len {
            let src_offset = s * head_dim;
            let dst_offset = s * per_gpu_head_dim + head_idx * head_dim;
            let src_slice = attn_out_h.slice(src_offset..src_offset + head_dim);
            let mut dst_slice = attn_combined.slice_mut(dst_offset..dst_offset + head_dim);
            stream
                .memcpy_dtod(&src_slice, &mut dst_slice)
                .map_err(|e| anyhow::anyhow!("Copy attn_out_h to combined buffer failed: {e}"))?;
        }
    }

    // =========================================================================
    // Gate application: attn_output = attn_output * sigmoid(gate)
    // =========================================================================
     // @lat: [[lat.md/lat#Paged Attention Implementation#Attention Output Gate]]
    let gated_attn = if let Some(ref gate_heads) = gate_heads {
        let mut gated = stream.alloc_zeros::<bf16>(attn_combined_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate gated output buffer: {e}"))?;
        oxide.launch_attn_output_gate_bf16(
            stream, &attn_combined, gate_heads, &mut gated, attn_combined_size as u32,
        ).map_err(|e| anyhow::anyhow!("Gate application kernel failed: {e}"))?;
        gated
    } else {
        attn_combined
    };

    // =========================================================================
    // O-projection using gated attention output
    // =========================================================================
    let mut _ps = None;
    let mut output = stream
        .alloc_zeros::<bf16>(buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate O-proj output buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.o_proj.name, &gated_attn, &mut output,
        seq_len, hidden_size, per_gpu_head_dim, group_size,
        &mut _ps,
    )?;

    Ok(output)
}

// ============================================================================
// Paged Attention Functions (new zero CPU round-trip implementation)
// ============================================================================

/// Paged prefill attention: writes K/V to paged cache, uses per-head GEMM.
///
/// Same as [[forward]] but writes K/V to paged cache instead of flat buffer.
/// The attention computation still uses per-head GEMMs (prefill benefits less
/// from paged decode kernel since all tokens are processed at once).
///
/// The key difference from [[forward]]:
/// - Phase 1: Same K/V computation + RoPE
/// - Phase 2: Writes to paged cache via `infers_paged_kv_write_bf16` instead of flat buffer
/// - Phase 3: Same per-head attention using the already-computed K/V buffers
pub fn forward_paged(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    weights: &AttentionWeights,
    input: &CudaSlice<bf16>,
    paged_cache: &mut PagedKvCache,
    block_table_gpu: &CudaSlice<i32>,
    positions_gpu: &CudaSlice<i32>,
    positions: &[u32],
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    rope_theta: f64,
    partial_rotary_factor: f32,
    rms_norm_eps: f32,
   group_size: usize,
    cache: &GpuWeightCache,
    hidden_size: usize,
    attn_output_gate: bool,
    layer_idx: usize,
    gpu_idx: usize,
    probe: &ProbeConfig,
) -> Result<CudaSlice<bf16>> {
    let per_gpu_head_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let seq_len = positions.len();

    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "num_heads {} must be divisible by num_kv_heads {} for GQA",
        num_heads, num_kv_heads
    );

    // =========================================================================
    // Phase 1: Full K, V computation + RoPE
    // =========================================================================

    // k_full = GEMM(input, k_proj^T)  [seq_len × kv_dim] (INT4-aware)
    let mut _ps_fp = None;
    let mut k_full = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate K buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.k_proj.name, input, &mut k_full,
        seq_len, kv_dim, hidden_size, group_size,
        &mut _ps_fp,
    )?;

    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_proj", &k_full, &[seq_len, kv_dim], "prefill");

    // v_full = GEMM(input, v_proj^T)  [seq_len × kv_dim] (INT4-aware)
    let mut v_full = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate V buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.v_proj.name, input, &mut v_full,
        seq_len, kv_dim, hidden_size, group_size,
        &mut _ps_fp,
    )?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.v_proj", &v_full, &[seq_len, kv_dim], "prefill");

    // --- K-norm on full K before Phase 1 RoPE ---
    if let Some(k_norm_w) = weights.k_norm.as_ref() {
        let k_norm_gpu = cache.get_bf16(&k_norm_w.name)
            .ok_or_else(|| anyhow::anyhow!("K-norm weight '{}' not in cache", k_norm_w.name))?;
        k_full = crate::norm::rms_norm(
            stream, oxide, &k_full, &k_norm_gpu, rms_norm_eps, head_dim,
        )?;
            probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_norm", &k_full, &[seq_len, kv_dim], "prefill");
    }

    // Apply RoPE to K_full
    let mut q_dummy = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dummy Q buffer for RoPE: {e}"))?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_before_rope", &k_full, &[seq_len, kv_dim], "prefill");
    rope::apply_rope(
            stream,
            oxide,
            &mut q_dummy,
            &mut k_full,
            positions,
            num_kv_heads as i32,
            head_dim,
            rope_theta,
            partial_rotary_factor,
            None, None,
        )?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_after_rope", &k_full, &[seq_len, kv_dim], "prefill");

    // =========================================================================
    // Phase 2: Paged KV write
    // =========================================================================

   let _ = paged_cache.ensure_allocated(stream)?;

    // Probe: K and V data right before writing to paged KV cache (after norm+RoPE)
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_cached", &k_full, &[seq_len, kv_dim], "prefill");
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.v_cached", &v_full, &[seq_len, kv_dim], "prefill");

   paged_kv_write(
        stream,
        oxide,
        &k_full,
        &v_full,
        paged_cache.page_pool.as_mut().unwrap(),
        block_table_gpu,
        positions_gpu,
        seq_len,
        head_dim,
        kv_dim,
        page_size,
    )?;

    // =========================================================================
    // Phase 2.5: Full Q projection + gate split (when attn_output_gate enabled)
    // =========================================================================

    // When attn_output_gate is true, the Q projection produces doubled output:
    // [Q_head_0, G_head_0, Q_head_1, G_head_1, ...] per row (per-head interleaved).
    // We compute it as a single GEMM and then extract Q/gate from interleaved positions.
    let q_out_dim = per_gpu_head_dim * if attn_output_gate { 2 } else { 1 };

    let mut _ps_fused = None;
    let mut q_full = stream
        .alloc_zeros::<bf16>(seq_len * q_out_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate Q buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.q_proj.name, input, &mut q_full,
        seq_len, q_out_dim, hidden_size, group_size,
        &mut _ps_fused,
    )?;

    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.q_proj_raw", &q_full, &[seq_len, q_out_dim], "prefill");
    // --- Q-norm on Q portion only (not gate) before split ---
    if let Some(q_norm_w) = weights.q_norm.as_ref() {
        let q_norm_gpu = cache.get_bf16(&q_norm_w.name)
            .ok_or_else(|| anyhow::anyhow!("Q-norm weight '{}' not in cache", q_norm_w.name))?;
        // Normalize only the Q portion [0 .. seq_len * per_gpu_head_dim], not the gate portion.
        let mut q_only = stream
            .alloc_zeros::<bf16>(seq_len * per_gpu_head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate Q-only buffer for norm: {e}"))?;
        // Copy Q portion from q_full (per-head interleaved layout: [Q_h0, G_h0, Q_h1, G_h1, ...])
        for s in 0..seq_len {
            for h in 0..num_heads {
                let src_offset = s * q_out_dim + h * (head_dim * 2);
                let dst_offset = s * per_gpu_head_dim + h * head_dim;
                let src_slice = q_full.slice(src_offset..src_offset + head_dim);
                let mut dst_slice = q_only.slice_mut(dst_offset..dst_offset + head_dim);
                stream
                    .memcpy_dtod(&src_slice, &mut dst_slice)
                    .map_err(|e| anyhow::anyhow!("Copy Q portion for norm failed: {e}"))?;
            }
        }
        let q_normed = crate::norm::rms_norm(
            stream, oxide, &q_only, &q_norm_gpu, rms_norm_eps, head_dim,
        )?;
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.q_norm", &q_normed, &[seq_len, per_gpu_head_dim], "prefill");
        // Write normalized Q back into q_full (per-head interleaved)
        for s in 0..seq_len {
            for h in 0..num_heads {
                let src_offset = s * per_gpu_head_dim + h * head_dim;
                let dst_offset = s * q_out_dim + h * (head_dim * 2);
                let src_slice = q_normed.slice(src_offset..src_offset + head_dim);
                let mut dst_slice = q_full.slice_mut(dst_offset..dst_offset + head_dim);
                stream
                    .memcpy_dtod(&src_slice, &mut dst_slice)
                    .map_err(|e| anyhow::anyhow!("Write normalized Q back failed: {e}"))?;
            }
        }
    }


    // When gate is enabled, split q_full into q_heads and gate_heads.
    // q_heads has shape [seq_len, per_gpu_head_dim] (first half of each row)
    // gate_heads has shape [seq_len, per_gpu_head_dim] (second half of each row)
    let gate_heads = if attn_output_gate {
        // Allocate and copy the gate portion (per-head interleaved layout)
        let mut gate_buf = stream
            .alloc_zeros::<bf16>(seq_len * per_gpu_head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate gate buffer: {e}"))?;

        // Copy per-head gate from interleaved layout: [Q_h0, G_h0, Q_h1, G_h1, ...]
        for s in 0..seq_len {
            for h in 0..num_heads {
                let src_offset = s * q_out_dim + h * (head_dim * 2) + head_dim;
                let dst_offset = s * per_gpu_head_dim + h * head_dim;
                let src_slice = q_full.slice(src_offset..src_offset + head_dim);
                let mut dst_slice = gate_buf.slice_mut(dst_offset..dst_offset + head_dim);
                stream
                    .memcpy_dtod(&src_slice, &mut dst_slice)
                    .map_err(|e| anyhow::anyhow!("Copy gate data from q_full failed: {e}"))?;
            }
        }
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.gate", &gate_buf, &[seq_len, per_gpu_head_dim], "prefill");
        Some(gate_buf)
    } else {
        None
    };

    // =========================================================================
    // Phase 3: Per-head attention — extract K/V from full buffers (GPU copies)
    // =========================================================================

    let buf_size = seq_len * hidden_size;  // full output buffer [seq_len x config.hidden_size]
    // --- Combined attention output buffer [seq_len x per_gpu_head_dim] ---
    let attn_combined_size = seq_len * per_gpu_head_dim;
    let mut attn_combined = stream
        .alloc_zeros::<bf16>(attn_combined_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate attn_combined buffer: {e}"))?;
    for head_idx in 0..num_heads {
        // --- Extract and upload per-head weight slices ---
        let kv_head_idx = head_idx / (num_heads / num_kv_heads);

        // --- Q projection: copy from precomputed q_full ---
        // q_full has per-head interleaved layout: [Q_h0, G_h0, Q_h1, G_h1, ...] when gate enabled
        let head_stride = if attn_output_gate { head_dim * 2 } else { head_dim };
        let mut q_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate q_h buffer: {e}"))?;
        for s in 0..seq_len {
            let src_offset = s * q_out_dim + head_idx * head_stride;
            let dst_offset = s * head_dim;
            let src_slice = q_full.slice(src_offset..src_offset + head_dim);
            let mut dst_slice = q_h.slice_mut(dst_offset..dst_offset + head_dim);
            stream
                .memcpy_dtod(&src_slice, &mut dst_slice)
                .map_err(|e| anyhow::anyhow!("Copy per-head Q from q_full failed: {e}"))?;
        }

        // --- Extract per-head K from k_full (already has RoPE applied from Phase 1) ---
        let mut k_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate k_h buffer: {e}"))?;
        for s in 0..seq_len {
            let src_offset = s * kv_dim + kv_head_idx * head_dim;
            let dst_offset = s * head_dim;
            let src_slice = k_full.slice(src_offset..src_offset + head_dim);
            let mut dst_slice = k_h.slice_mut(dst_offset..dst_offset + head_dim);
            stream
                .memcpy_dtod(&src_slice, &mut dst_slice)
                .map_err(|e| anyhow::anyhow!("Copy per-head K from k_full failed: {e}"))?;
        }
        if head_idx == 0 && probe.should_dump(layer_idx, "attn.heads") {
            probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_h0", &k_h, &[seq_len, head_dim], "prefill");
        }

        // --- Extract per-head V from v_full ---
        let mut v_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate v_h buffer: {e}"))?;
        for s in 0..seq_len {
            let src_offset = s * kv_dim + kv_head_idx * head_dim;
            let dst_offset = s * head_dim;
            let src_slice = v_full.slice(src_offset..src_offset + head_dim);
            let mut dst_slice = v_h.slice_mut(dst_offset..dst_offset + head_dim);
            stream
                .memcpy_dtod(&src_slice, &mut dst_slice)
                .map_err(|e| anyhow::anyhow!("Copy per-head V from v_full failed: {e}"))?;
        }
        if head_idx == 0 && probe.should_dump(layer_idx, "attn.heads") {
            probe::dump(stream, probe, layer_idx, gpu_idx, "attn.v_h0", &v_h, &[seq_len, head_dim], "prefill");
        }

        // --- RoPE (per-head, num_heads=1) — apply only to q_h (k_h already has RoPE from Phase 1) ---
        let mut k_h_dummy = stream.alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate dummy K buffer for RoPE: {e}"))?;
        rope::apply_rope(
            stream,
            oxide,
            &mut q_h,
            &mut k_h_dummy,  // dummy — k_h already has RoPE from Phase 1
            positions,
            1,
            head_dim,
            rope_theta,
            partial_rotary_factor,
            None, None,
        )?;
        if head_idx == 0 && probe.should_dump(layer_idx, "attn.heads") {
            probe::dump(stream, probe, layer_idx, gpu_idx, "attn.q_h0", &q_h, &[seq_len, head_dim], "prefill");
        }

        // --- Attention scores: Q_h @ K_h^T → [seq_len × seq_len] ---
        // Scale by 1/sqrt(head_dim) for stable softmax (standard attention scaling).
        let scores_size = seq_len * seq_len;
        let mut scores_h = stream
            .alloc_zeros::<bf16>(scores_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate scores buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: seq_len,
                k: head_dim,
                transa: true,
                transb: false,
                alpha: 1.0 / (head_dim as f32).sqrt(),
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &q_h,
            &k_h,
            &mut scores_h,
        )?;
        if head_idx == 0 && probe.should_dump(layer_idx, "attn.heads") {
            probe::dump(stream, probe, layer_idx, gpu_idx, "attn.scores_h0", &scores_h, &[seq_len, seq_len], "prefill");
        }

        // --- Softmax with causal masking ---
        let mut softmax_out_h = stream
            .alloc_zeros::<bf16>(scores_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate softmax output buffer: {e}"))?;

        oxide.launch_softmax_bf16(
            stream, &scores_h, &mut softmax_out_h, seq_len as u32, 1u32,
        ).map_err(|e| anyhow::anyhow!("Softmax kernel launch failed: {e}"))?;
        if head_idx == 0 && probe.should_dump(layer_idx, "attn.heads") {
            probe::dump(stream, probe, layer_idx, gpu_idx, "attn.softmax_h0", &softmax_out_h, &[seq_len, seq_len], "prefill");
        }

        // --- Attention output: softmax_out_h @ V_h → [seq_len × head_dim] ---
        let mut attn_out_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate attn_out_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: head_dim,
                n: seq_len,
                k: seq_len,
                transa: false,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &v_h,
            &softmax_out_h,
            &mut attn_out_h,
        )?;

        // --- Copy attention output to combined buffer at correct head offset ---
        for s in 0..seq_len {
            let src_offset = s * head_dim;
            let dst_offset = s * per_gpu_head_dim + head_idx * head_dim;
            let src_slice = attn_out_h.slice(src_offset..src_offset + head_dim);
            let mut dst_slice = attn_combined.slice_mut(dst_offset..dst_offset + head_dim);
            stream
                .memcpy_dtod(&src_slice, &mut dst_slice)
                .map_err(|e| anyhow::anyhow!("Copy attn_out_h to combined buffer failed: {e}"))?;
        }
    }

    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.combined", &attn_combined, &[seq_len, per_gpu_head_dim], "prefill");

    // =========================================================================
    // Gate application: attn_output = attn_output * sigmoid(gate)
    // =========================================================================
    // @lat: [[lat.md/lat#Paged Attention Implementation#Attention Output Gate]]
    let gated_attn = if let Some(ref gate_heads) = gate_heads {
        let mut gated = stream
            .alloc_zeros::<bf16>(attn_combined_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate gated output buffer: {e}"))?;
        oxide.launch_attn_output_gate_bf16(
            stream, &attn_combined, gate_heads, &mut gated, attn_combined_size as u32,
        ).map_err(|e| anyhow::anyhow!("Gate application kernel failed: {e}"))?;
        gated
    } else {
        attn_combined
    };
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.gated", &gated_attn, &[seq_len, per_gpu_head_dim], "prefill");

    // =========================================================================
    // O-projection using gated attention output
    // =========================================================================
    let mut _ps = None;
    let mut output = stream
        .alloc_zeros::<bf16>(buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate O-proj output buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.o_proj.name, &gated_attn, &mut output,
        seq_len, hidden_size, per_gpu_head_dim, group_size,
        &mut _ps,
    )?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.o_proj", &output, &[seq_len, hidden_size], "prefill");

    Ok(output)
}

/// Paged decode attention: single-token attention with zero CPU round-trips.
///
/// Uses the paged KV cache and GPU-side attention decode kernel to eliminate
/// the CPU download/re-upload bottleneck of the flat cache design.
///
/// # Steps
/// 1. Compute single-token K, V via GEMM
/// 2. Apply RoPE to K
/// 3. Write K, V to paged cache via `infers_paged_kv_write_bf16`
/// 4. Launch `infers_paged_attention_decode_bf16` for full decode attention
/// 5. Apply O-projection to attention output
pub fn decode_forward_paged(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    weights: &AttentionWeights,
    input: &CudaSlice<bf16>,
    paged_cache: &mut PagedKvCache,
    block_table_gpu: &CudaSlice<i32>,
    positions_gpu: &CudaSlice<i32>,
    position: u32,
    num_cached_tokens: i32,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    rope_theta: f64,
    partial_rotary_factor: f32,
    rms_norm_eps: f32,
    group_size: usize,
    cache: &GpuWeightCache,
    hidden_size: usize,
    attn_output_gate: bool,
    layer_idx: usize,
    gpu_idx: usize,
    probe: &ProbeConfig,
    cached_cos: Option<&CudaSlice<f32>>,  // pre-computed RoPE cos table
    cached_sin: Option<&CudaSlice<f32>>,  // pre-computed RoPE sin table,
    ws: &mut crate::workspace::AttnWorkspace,   // attention workspace buffers
    output: &mut CudaSlice<bf16>,                // writes into workspace.attn_out
    partial_sums_buf: &mut Option<&mut CudaSlice<f32>>, // pre-allocated partial sums for K-split (mutable ref to allow reuse across calls)
) -> Result<()> {
    let per_gpu_head_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "num_heads {} must be divisible by num_kv_heads {} for GQA",
        num_heads, num_kv_heads
    );

    // =========================================================================
    // Phase 1: Single-token K, V computation + RoPE
    // =========================================================================
  // k_single = GEMM(input, k_proj^T)  [1 × kv_dim] (INT4-aware)
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.k_proj.name, input, &mut ws.k_single,
        1, kv_dim, hidden_size, group_size,
        &mut *partial_sums_buf,
    )?;

    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_proj", &ws.k_single, &[1, kv_dim], "decode");

    // v_single = GEMM(input, v_proj^T)  [1 × kv_dim] (INT4-aware)
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.v_proj.name, input, &mut ws.v_single,
        1, kv_dim, hidden_size, group_size,
        &mut *partial_sums_buf,
    )?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.v_proj", &ws.v_single, &[1, kv_dim], "decode");

    // --- K-norm on full K before RoPE ---
    let k_norm_exists = if let Some(k_norm_w) = weights.k_norm.as_ref() {
        let k_norm_gpu = cache.get_bf16(&k_norm_w.name)
            .ok_or_else(|| anyhow::anyhow!("K-norm weight '{}' not in cache", k_norm_w.name))?;
        crate::norm::rms_norm_into(
            stream, oxide, &mut ws.k_norm_out, &ws.k_single, &k_norm_gpu, rms_norm_eps, head_dim,
        )?;
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_norm", &ws.k_norm_out, &[1, kv_dim], "decode");
        true
    } else {
        false
    };

    // Apply RoPE to K — must do inside each branch to get the right mutable reference
    if k_norm_exists {
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_before_rope", &ws.k_norm_out, &[1, kv_dim], "decode");
        rope::apply_rope(
            stream,
            oxide,
            &mut ws.q_dummy,
            &mut ws.k_norm_out,
            &[position],
            num_kv_heads as i32,
            head_dim,
            rope_theta,
            partial_rotary_factor,
            cached_cos,
            cached_sin,
        )?;
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_after_rope", &ws.k_norm_out, &[1, kv_dim], "decode");
    } else {
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_before_rope", &ws.k_single, &[1, kv_dim], "decode");
        rope::apply_rope(
            stream,
            oxide,
            &mut ws.q_dummy,
            &mut ws.k_single,
            &[position],
            num_kv_heads as i32,
            head_dim,
            rope_theta,
            partial_rotary_factor,
            cached_cos,
            cached_sin,
        )?;
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_after_rope", &ws.k_single, &[1, kv_dim], "decode");
    }

    // =========================================================================
    // Phase 2: Paged KV write — write new token to page pool
    // =========================================================================

   let _ = paged_cache.ensure_allocated(stream)?;

    // Get K reference for probes and paged_kv_write
    let k_ref = if k_norm_exists { &ws.k_norm_out } else { &ws.k_single };

    // Probe: K and V data right before writing to paged KV cache (after norm+RoPE)
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_cached", k_ref, &[1, kv_dim], "decode");
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.v_cached", &ws.v_single, &[1, kv_dim], "decode");

   paged_kv_write(
        stream,
        oxide,
        k_ref,
        &ws.v_single,
        paged_cache.page_pool.as_mut().unwrap(),
        block_table_gpu,
        positions_gpu,
        1, // seq_len = 1 for decode
        head_dim,
        kv_dim,
        page_size,
    )?;

    // =========================================================================
    // Phase 3: Compute Q for attention decode kernel (zero-alloc via workspace)
    // =========================================================================

    // Q projection: full Q via GEMM (doubled output when attn_output_gate enabled)
    let q_out_dim = per_gpu_head_dim * if attn_output_gate { 2 } else { 1 };
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream,
        cache, &weights.q_proj.name, input, &mut ws.q_full,
        1, q_out_dim, hidden_size, group_size,
        &mut *partial_sums_buf,
    )?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "attn.q_proj_raw", &ws.q_full, &[1, q_out_dim], "decode");

    // --- Q-norm on Q portion only (not gate) ---
    // Split first, then normalize only the Q part.  Must do Q-extraction,
    // Q-norm, and RoPE inside each branch to satisfy mutable borrow requirements.
    if attn_output_gate {
        // Extract Q and gate portions from q_full (per-head interleaved layout)
        oxide.launch_split_qgate_bf16(
            stream, &ws.q_full, &mut ws.q_buf, &mut ws.gate_buf,
            num_heads as u32, head_dim as u32,
        )?;

        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.gate", &ws.gate_buf, &[1, per_gpu_head_dim], "decode");

        // Apply Q-norm only to the Q portion (into ws.q_norm_out)
        let q_norm_exists = if let Some(q_norm_w) = weights.q_norm.as_ref() {
            let q_norm_gpu = cache.get_bf16(&q_norm_w.name)
                .ok_or_else(|| anyhow::anyhow!("Q-norm weight '{}' not in cache", q_norm_w.name))?;
            crate::norm::rms_norm_into(
                stream, oxide, &mut ws.q_norm_out, &ws.q_buf, &q_norm_gpu, rms_norm_eps, head_dim,
            )?;
            probe::dump(stream, probe, layer_idx, gpu_idx, "attn.q_norm", &ws.q_norm_out, &[1, per_gpu_head_dim], "decode");
            true
        } else {
            false
        };

        // Apply RoPE to Q — inside branch for mutable access
        if q_norm_exists {
            rope::apply_rope(
                stream,
                oxide,
                &mut ws.q_norm_out,
                &mut ws.k_rope_dummy,
                &[position],
                num_heads as i32,
                head_dim,
                rope_theta,
                partial_rotary_factor,
                None, None,
            )?;
        } else {
            rope::apply_rope(
                stream,
                oxide,
                &mut ws.q_buf,
                &mut ws.k_rope_dummy,
                &[position],
                num_heads as i32,
                head_dim,
                rope_theta,
                partial_rotary_factor,
                None, None,
            )?;
        }

        // Phase 4: Paged attention decode — scores, softmax, V accumulation in one kernel
        let num_pages = block_table_gpu.len();
        {
            let q_for_attn = if q_norm_exists { &ws.q_norm_out } else { &ws.q_buf };
            paged_attention_decode_into(
                stream,
                oxide,
                q_for_attn,
                paged_cache.page_pool.as_ref().unwrap(),
                block_table_gpu,
                &mut ws.attn_output,
                num_pages,
                num_cached_tokens as usize,
                head_dim,
                num_heads,
                num_kv_heads,
                page_size,
                kv_dim,
            )?;
        }
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.combined", &ws.attn_output, &[1, per_gpu_head_dim], "decode");

        // Gate application: attn_output = attn_output * sigmoid(gate)
        oxide.launch_attn_output_gate_bf16(
            stream, &ws.attn_output, &ws.gate_buf, &mut ws.gated, per_gpu_head_dim as u32,
        ).map_err(|e| anyhow::anyhow!("Gate application kernel failed: {e}"))?;
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.gated", &ws.gated, &[1, per_gpu_head_dim], "decode");

        // Phase 5: O-projection — single GEMM over all heads (INT4-aware)
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, oxide, stream,
            cache, &weights.o_proj.name, &ws.gated, output,
            1, hidden_size, per_gpu_head_dim, group_size,
            &mut *partial_sums_buf,
        )?;
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.o_proj", output, &[1, hidden_size], "decode");

    } else {
        // No gate — use ws.q_full as Q (first per_gpu_head_dim elements are valid)
        // Apply Q-norm to the Q portion (into ws.q_norm_out)
        let q_norm_exists = if let Some(q_norm_w) = weights.q_norm.as_ref() {
            let q_norm_gpu = cache.get_bf16(&q_norm_w.name)
                .ok_or_else(|| anyhow::anyhow!("Q-norm weight '{}' not in cache", q_norm_w.name))?;
            crate::norm::rms_norm_into(
                stream, oxide, &mut ws.q_norm_out, &ws.q_full, &q_norm_gpu, rms_norm_eps, head_dim,
            )?;
            probe::dump(stream, probe, layer_idx, gpu_idx, "attn.q_norm", &ws.q_norm_out, &[1, per_gpu_head_dim], "decode");
            true
        } else {
            false
        };

        // Apply RoPE to Q — inside branch for mutable access
        if q_norm_exists {
            rope::apply_rope(
                stream,
                oxide,
                &mut ws.q_norm_out,
                &mut ws.k_rope_dummy,
                &[position],
                num_heads as i32,
                head_dim,
                rope_theta,
                partial_rotary_factor,
                None, None,
            )?;
        } else {
            rope::apply_rope(
                stream,
                oxide,
                &mut ws.q_full,
                &mut ws.k_rope_dummy,
                &[position],
                num_heads as i32,
                head_dim,
                rope_theta,
                partial_rotary_factor,
                None, None,
            )?;
        }

        // Phase 4: Paged attention decode — scores, softmax, V accumulation in one kernel
        let num_pages = block_table_gpu.len();
        {
            let q_for_attn = if q_norm_exists { &ws.q_norm_out } else { &ws.q_full };
            paged_attention_decode_into(
                stream,
                oxide,
                q_for_attn,
                paged_cache.page_pool.as_ref().unwrap(),
                block_table_gpu,
                &mut ws.attn_output,
                num_pages,
                num_cached_tokens as usize,
                head_dim,
                num_heads,
                num_kv_heads,
                page_size,
                kv_dim,
            )?;
        }
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.combined", &ws.attn_output, &[1, per_gpu_head_dim], "decode");

        // No gate application needed — use attn_output directly for O-proj
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.gated", &ws.attn_output, &[1, per_gpu_head_dim], "decode");

        // Phase 5: O-projection — single GEMM over all heads (INT4-aware)
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, oxide, stream,
            cache, &weights.o_proj.name, &ws.attn_output, output,
            1, hidden_size, per_gpu_head_dim, group_size,
            &mut *partial_sums_buf,
        )?;
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.o_proj", output, &[1, hidden_size], "decode");
    }

    Ok(())
}

// ── GPU-native FP8 KV cache operations ──────────────────────────────

/// Quantize BF16 K/V to FP8 and write into a `QuantizedKvCache` page pool — GPU-only.
///
/// Launches `infers_fp8_quantize_bf16` on K and V, then copies the quantized
/// bytes into the interleaved page pool via device-to-device memcpy.
/// No CPU round-trip.
///
/// # Arguments
/// * `stream` — CUDA stream for kernel launches and memcpys
/// * `quant_kernel` — The `infers_fp8_quantize_bf16` kernel handle
/// * `page_pool` — Mutable reference to the `QuantizedKvCache` page pool
/// * `page_id` — Target physical page
/// * `page_offset` — Token offset within the page
/// * `page_size` — Page size (tokens per page)
/// * `kv_dim` — KV dimension
/// * `dtype` — Quantized data type (Fp8E4M3 or Fp8E5M2)
/// * `k` — GPU buffer of key values (BF16, length = kv_dim)
/// * `v` — GPU buffer of value values (BF16, length = kv_dim)
pub fn fp8_quantize_and_write(
    stream: &Arc<CudaStream>,
    quant_kernel: &CudaFunction,
    page_pool: &mut CudaSlice<u8>,
    page_id: usize,
    page_offset: usize,
    page_size: usize,
    kv_dim: usize,
    dtype: KvCacheDtype,
    k: &CudaSlice<half::bf16>,
    v: &CudaSlice<half::bf16>,
) -> Result<()> {
    let bpe = dtype.bytes_per_element();
    let elem_count = kv_dim; // one token's worth of K or V

    // Determine FP8 mode from dtype
    let mode: i32 = match dtype {
        KvCacheDtype::Fp8E4M3 => 0,
        KvCacheDtype::Fp8E5M2 => 1,
        _ => anyhow::bail!("fp8_quantize_and_write requires Fp8E4M3 or Fp8E5M2 dtype"),
    };

    // Allocate temp GPU buffers for quantized output (1 token's K and V)
    let mut k_q = stream
        .alloc_zeros::<u8>(elem_count * bpe)
        .map_err(|e| anyhow::anyhow!("Failed to allocate temp K quant buffer: {e}"))?;
    let mut v_q = stream
        .alloc_zeros::<u8>(elem_count * bpe)
        .map_err(|e| anyhow::anyhow!("Failed to allocate temp V quant buffer: {e}"))?;

    // Launch quantize kernel on K
    let grid = ((elem_count as u32 + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32, 1, 1);
    let block = (BLOCK_SIZE as u32, 1, 1);
    let launch_cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };

    let elem_count_i32 = elem_count as i32;

    unsafe {
        stream
            .launch_builder(quant_kernel)
            .arg(k)
            .arg(&mut k_q)
            .arg(&elem_count_i32)
            .arg(&mode)
            .launch(launch_cfg)
            .map_err(|e| anyhow::anyhow!("FP8 quantize kernel (K) launch failed: {e}"))?;

        // Launch quantize kernel on V
        stream
            .launch_builder(quant_kernel)
            .arg(v)
            .arg(&mut v_q)
            .arg(&elem_count_i32)
            .arg(&mode)
            .launch(launch_cfg)
            .map_err(|e| anyhow::anyhow!("FP8 quantize kernel (V) launch failed: {e}"))?;
    }

    // Calculate byte offsets in the interleaved page pool
    let per_side_bytes = page_size * kv_dim * bpe;
    let page_stride = 2 * per_side_bytes;
    let base = page_id * page_stride;
    let k_offset = base + page_offset * kv_dim * bpe;
    let v_offset = base + per_side_bytes + page_offset * kv_dim * bpe;
    let copy_bytes = elem_count * bpe;

    // Device-to-device copy quantized K into page pool
    {
        let mut dst = page_pool.slice_mut(k_offset..k_offset + copy_bytes);
        stream
            .memcpy_dtod(&k_q, &mut dst)
            .map_err(|e| anyhow::anyhow!("D2D copy K failed: {e}"))?;
    }

    // Device-to-device copy quantized V into page pool
    {
        let mut dst = page_pool.slice_mut(v_offset..v_offset + copy_bytes);
        stream
            .memcpy_dtod(&v_q, &mut dst)
            .map_err(|e| anyhow::anyhow!("D2D copy V failed: {e}"))?;
    }

    Ok(())
}

/// Dequantize FP8 K/V from a `QuantizedKvCache` page pool to BF16 — GPU-only.
///
/// Copies FP8 bytes from the interleaved page pool to temp buffers via
/// device-to-device memcpy, then launches `infers_fp8_dequantize_bf16`.
/// No CPU round-trip.
///
/// # Arguments
/// * `stream` — CUDA stream for memcpys and kernel launches
/// * `dequant_kernel` — The `infers_fp8_dequantize_bf16` kernel handle
/// * `page_pool` — Reference to the `QuantizedKvCache` page pool
/// * `page_id` — Source physical page
/// * `page_offset` — Token offset within the page
/// * `len` — Number of tokens to read
/// * `page_size` — Page size (tokens per page)
/// * `kv_dim` — KV dimension
/// * `dtype` — Quantized data type (Fp8E4M3 or Fp8E5M2)
///
/// # Returns
/// `(k_gpu, v_gpu)` — dequantized BF16 K and V buffers on GPU
pub fn fp8_dequantize_and_read(
    stream: &Arc<CudaStream>,
    dequant_kernel: &CudaFunction,
    page_pool: &CudaSlice<u8>,
    page_id: usize,
    page_offset: usize,
    len: usize,
    page_size: usize,
    kv_dim: usize,
    dtype: KvCacheDtype,
) -> Result<(CudaSlice<half::bf16>, CudaSlice<half::bf16>)> {
    let bpe = dtype.bytes_per_element();
    let total_elems = len * kv_dim;
    let total_bytes = total_elems * bpe;

    // Calculate byte offsets
    let per_side_bytes = page_size * kv_dim * bpe;
    let page_stride = 2 * per_side_bytes;
    let base = page_id * page_stride;
    let k_offset = base + page_offset * kv_dim * bpe;
    let v_offset = base + per_side_bytes + page_offset * kv_dim * bpe;

    // Allocate temp GPU buffers for quantized reads
    let mut k_q = stream
        .alloc_zeros::<u8>(total_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to allocate temp K dequant buffer: {e}"))?;
    let mut v_q = stream
        .alloc_zeros::<u8>(total_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to allocate temp V dequant buffer: {e}"))?;

    // Device-to-device copy FP8 bytes from page pool to temp buffers
    {
        let src = page_pool.slice(k_offset..k_offset + total_bytes);
        stream
            .memcpy_dtod(&src, &mut k_q)
            .map_err(|e| anyhow::anyhow!("D2D copy K read failed: {e}"))?;
    }
    {
        let src = page_pool.slice(v_offset..v_offset + total_bytes);
        stream
            .memcpy_dtod(&src, &mut v_q)
            .map_err(|e| anyhow::anyhow!("D2D copy V read failed: {e}"))?;
    }

    // Allocate output BF16 buffers
    let mut k_out = stream
        .alloc_zeros::<half::bf16>(total_elems)
        .map_err(|e| anyhow::anyhow!("Failed to allocate K output buffer: {e}"))?;
    let mut v_out = stream
        .alloc_zeros::<half::bf16>(total_elems)
        .map_err(|e| anyhow::anyhow!("Failed to allocate V output buffer: {e}"))?;

    // Determine FP8 mode
    let mode: i32 = match dtype {
        KvCacheDtype::Fp8E4M3 => 0,
        KvCacheDtype::Fp8E5M2 => 1,
        _ => anyhow::bail!("fp8_dequantize_and_read requires Fp8E4M3 or Fp8E5M2 dtype"),
    };

    // Launch dequantize kernel on K
    let grid = ((total_elems as u32 + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32, 1, 1);
    let block = (BLOCK_SIZE as u32, 1, 1);
    let launch_cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };

    let total_elems_i32 = total_elems as i32;

    unsafe {
        stream
            .launch_builder(dequant_kernel)
            .arg(&k_q)
            .arg(&mut k_out)
            .arg(&total_elems_i32)
            .arg(&mode)
            .launch(launch_cfg)
            .map_err(|e| anyhow::anyhow!("FP8 dequantize kernel (K) launch failed: {e}"))?;

        // Launch dequantize kernel on V
        stream
            .launch_builder(dequant_kernel)
            .arg(&v_q)
            .arg(&mut v_out)
            .arg(&total_elems_i32)
            .arg(&mode)
            .launch(launch_cfg)
            .map_err(|e| anyhow::anyhow!("FP8 dequantize kernel (V) launch failed: {e}"))?;
    }

    Ok((k_out, v_out))
}
