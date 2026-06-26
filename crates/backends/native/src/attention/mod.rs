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
#[derive(Debug, Clone)]
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
        stream, &oxide.cc_stream(), k, v, page_pool, block_table_gpu, positions_gpu,
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
/// * `cached_tokens_count` — Device buffer [1] u32 holding num cached tokens (CUDA graph compatible)
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
    cached_tokens_count: &CudaSlice<u32>,  // device buffer with 1 element
    num_pages: usize,
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
        stream, &oxide.cc_stream(), q, page_pool, block_table_gpu, cached_tokens_count, &mut output,
        num_pages as u32, head_dim as u32,
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
    cached_tokens_count: &CudaSlice<u32>,  // device buffer with 1 element
    output: &mut CudaSlice<bf16>,
    num_pages: usize,
    head_dim: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    page_size: usize,
    kv_dim: usize,
) -> Result<()> {
    oxide.launch_paged_attention_decode_bf16(
        stream, &oxide.cc_stream(), q, page_pool, block_table_gpu, cached_tokens_count, output,
        num_pages as u32, head_dim as u32,
        num_kv_heads as u32, num_query_heads as u32, page_size as u32, kv_dim as u32,
    ).map_err(|e| anyhow::anyhow!("Paged attention decode kernel failed: {e}"))?;
    Ok(())
}


// ── Submodules ──────────────────────────────────────────────

mod flat_forward;
mod paged_decode;
mod paged_forward;

pub use flat_forward::forward;
pub use paged_decode::decode_forward_paged;
pub use paged_forward::forward_paged;

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
