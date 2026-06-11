//! Standard softmax attention forward pass and paged KV cache attention.
//!
//! Implements both the original flat-cache attention and the new paged attention
//! subsystem for zero CPU round-trip decode. Uses per-head weight slicing to
//! avoid strided GPU sub-slices (cudarc limitation).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use half::{bf16, f16};
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_model::{AttentionWeights, Int4Companions, WeightData, WeightDtype};

use infers_kv::KvCacheDtype;

use crate::add;
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
/// @lat: [[lat.md/lat#Phase 4.6 Deliverables#Paged Attention Implementation#PagedKvCache]]
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

/// Upload a [&[bf16]] slice to GPU memory.
fn upload_bf16_slice(
    stream: &Arc<CudaStream>,
    data: &[bf16],
) -> Result<CudaSlice<bf16>> {
    stream
        .clone_htod(data)
        .map_err(|e| anyhow::anyhow!("Failed to upload BF16 slice: {e}"))
}

/// Extract rows [head_idx * head_dim .. (head_idx+1) * head_dim] from a
/// row-major BF16 weight matrix.
fn extract_head_weight_slice(
    full_weight: &WeightData,
    head_idx: usize,
    head_dim: usize,
) -> Result<Vec<bf16>> {
    let in_dim = full_weight.shape[1];
    let out_dim = full_weight.shape[0];
    let head_start_row = head_idx * head_dim;
    let head_end_row = head_start_row + head_dim;
    anyhow::ensure!(
        head_end_row <= out_dim,
        "Head index {} out of bounds (out_dim={})",
        head_end_row,
        out_dim
    );
    let mut result = Vec::with_capacity(head_dim * in_dim);
    for row in head_start_row..head_end_row {
        let row_start = row * in_dim * 2;
        for col_idx in 0..in_dim {
            let byte_offset = row_start + col_idx * 2;
            let lo = full_weight.data[byte_offset];
            let hi = full_weight.data[byte_offset + 1];
            result.push(bf16::from_bits(u16::from_le_bytes([lo, hi])));
        }
    }
    Ok(result)
}

/// Extract columns [head_idx * head_dim .. (head_idx+1) * head_dim] from a
/// row-major BF16 weight matrix (for output projection).
fn extract_o_proj_head_slice(
    o_proj: &WeightData,
    head_idx: usize,
    head_dim: usize,
) -> Result<Vec<bf16>> {
    let in_dim = o_proj.shape[1];
    let out_dim = o_proj.shape[0];
    let col_start = head_idx * head_dim;
    let col_end = col_start + head_dim;
    anyhow::ensure!(
        col_end <= in_dim,
        "O-proj head column {} out of bounds (in_dim={})",
        col_end,
        in_dim
    );
    let mut result = Vec::with_capacity(out_dim * head_dim);
    for row in 0..out_dim {
        for col in col_start..col_end {
            let byte_offset = (row * in_dim + col) * 2;
            let lo = o_proj.data[byte_offset];
            let hi = o_proj.data[byte_offset + 1];
            result.push(bf16::from_bits(u16::from_le_bytes([lo, hi])));
        }
    }
    Ok(result)
}

/// Convert an attention projection weight to a flat BF16 Vec.
///
/// If the weight is already BF16/FP16/FP32, converts directly.
/// If the weight is Int4Packed, dequantizes it using companion tensors.
fn attention_weight_to_bf16_vec(
    weight: &WeightData,
    int4_companions: &HashMap<String, Int4Companions>,
    group_size: usize,
) -> Result<Vec<bf16>> {
    match weight.dtype {
        WeightDtype::Bf16 => {
            Ok(weight.data.chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        WeightDtype::Fp16 => {
            Ok(weight.data.chunks_exact(2)
                .map(|c| {
                    let f16_val = f16::from_bits(u16::from_le_bytes([c[0], c[1]]));
                    bf16::from_f32(f16_val.to_f32())
                })
                .collect())
        }
        WeightDtype::Fp32 => {
            Ok(weight.data.chunks_exact(4)
                .map(|c| bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect())
        }
        WeightDtype::Int4Packed => {
            let companions = int4_companions.get(&weight.name)
                .ok_or_else(|| anyhow::anyhow!("No INT4 companions for weight '{}'", weight.name))?;
            let qzeros = &companions.qzeros;
            let scales = &companions.scales;
            let qweight_wd = WeightData {
                data: weight.data.clone(),
                shape: weight.shape.clone(),
                dtype: WeightDtype::Int4Packed,
                name: weight.name.clone(),
            };
            Ok(crate::upload::dequantize_int4_to_bf16(&qweight_wd, scales, qzeros, group_size))
        }
        _ => anyhow::bail!("Unsupported weight dtype {:?} for '{}'", weight.dtype, weight.name),
    }
}

/// Create a BF16 WeightData from any weight dtype (dequantizes INT4 if needed).
fn weight_to_bf16_wd(
    weight: &WeightData,
    int4_companions: &HashMap<String, Int4Companions>,
    group_size: usize,
) -> Result<WeightData> {
    let bf16_data = attention_weight_to_bf16_vec(weight, int4_companions, group_size)?;
    let bytes: Vec<u8> = bf16_data.iter().flat_map(|b| b.to_bits().to_le_bytes()).collect();
    let shape = {
        let mut s = weight.shape.clone();
        // For INT4, shape is [N, K/8] — convert K/8 back to K
        if weight.dtype == WeightDtype::Int4Packed && s.len() >= 2 {
            s[1] *= 8;
        }
        s
    };
    Ok(WeightData {
        data: bytes,
        shape,
        dtype: WeightDtype::Bf16,
        name: weight.name.clone(),
    })
}


// ============================================================================
// Paged Kernel Dispatch Functions
// ============================================================================

/// Launch `infers_paged_kv_write_bf16` to write K and V into the paged KV cache.
///
/// # Arguments
/// * `stream` — CUDA stream
/// * `kernel` — Loaded function handle for `infers_paged_kv_write_bf16`
/// * `k` — K tensor `[seq_len × kv_dim]`
/// * `v` — V tensor `[seq_len × kv_dim]`
/// * `page_pool` — Flat GPU buffer for paged KV data
/// * `block_table_gpu` — Block table (page IDs) on GPU `[num_pages]`
/// * `positions_gpu` — Token positions on GPU `[seq_len]`
/// * `seq_len` — Number of tokens to write
/// * `kv_dim` — num_kv_heads × head_dim
/// * `page_size` — Tokens per page
// @lat: [[lat.md/lat#Phase 4.6 Deliverables#Paged Attention Implementation#Paged Kernel Dispatch]]
pub fn paged_kv_write(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    k: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    page_pool: &CudaSlice<bf16>,
    block_table_gpu: &CudaSlice<i32>,
    positions_gpu: &CudaSlice<i32>,
    seq_len: usize,
    head_dim: usize,
    kv_dim: usize,
    page_size: usize,
) -> Result<()> {
    let total_elements = seq_len * kv_dim;
    let grid = (total_elements as u32).div_ceil(256);
    let config = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let seq_len_i32 = seq_len as i32;
    let head_dim_i32 = head_dim as i32;
    let kv_dim_i32 = kv_dim as i32;
    let page_size_i32 = page_size as i32;

    unsafe {
        stream
            .launch_builder(kernel)
            .arg(k)
            .arg(v)
            .arg(page_pool)
            .arg(block_table_gpu)
            .arg(positions_gpu)
            .arg(&seq_len_i32)
            .arg(&head_dim_i32)
            .arg(&page_size_i32)
            .arg(&kv_dim_i32)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("Paged KV write kernel launch failed: {e}"))?;
    }

    Ok(())
}

/// Launch `infers_paged_kv_read_bf16` to gather K/V from page pool into
/// contiguous buffers for GEMM consumption.
///
/// Returns `(CudaSlice<bf16>, CudaSlice<bf16>)` — contiguous K and V buffers.
///
/// # Arguments
/// * `stream` — CUDA stream
/// * `kernel` — Loaded function handle for `infers_paged_kv_read_bf16`
/// * `page_pool` — Flat GPU buffer for paged KV data
/// * `block_table_gpu` — Block table on GPU `[num_pages]`
/// * `num_pages` — Number of pages in block table
/// * `num_cached_tokens` — Number of cached tokens to gather
/// * `kv_dim` — num_kv_heads × head_dim
/// * `page_size` — Tokens per page
// @lat: [[lat.md/lat#Phase 4.6 Deliverables#Paged Attention Implementation#Paged Kernel Dispatch]]
pub fn paged_kv_read(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    page_pool: &CudaSlice<bf16>,
    block_table_gpu: &CudaSlice<i32>,
    num_pages: usize,
    num_cached_tokens: usize,
    head_dim: usize,
    kv_dim: usize,
    page_size: usize,
) -> Result<(CudaSlice<bf16>, CudaSlice<bf16>)> {
    let gather_size = num_cached_tokens * kv_dim;

    let mut k_out = stream
        .alloc_zeros::<bf16>(gather_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate K gather buffer: {e}"))?;
    let mut v_out = stream
        .alloc_zeros::<bf16>(gather_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate V gather buffer: {e}"))?;

    let total_elements = num_cached_tokens * kv_dim;
    let grid = (total_elements as u32).div_ceil(256);
    let config = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let num_pages_i32 = num_pages as i32;
    let num_cached_tokens_i32 = num_cached_tokens as i32;
    let head_dim_i32 = head_dim as i32;
    let kv_dim_i32 = kv_dim as i32;
    let page_size_i32 = page_size as i32;

    unsafe {
        stream
            .launch_builder(kernel)
            .arg(page_pool)
            .arg(block_table_gpu)
            .arg(&num_pages_i32)
            .arg(&num_cached_tokens_i32)
            .arg(&head_dim_i32)
            .arg(&page_size_i32)
            .arg(&kv_dim_i32)
            .arg(&mut k_out)
            .arg(&mut v_out)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("Paged KV read kernel launch failed: {e}"))?;
    }

    Ok((k_out, v_out))
}

/// Launch `infers_paged_attention_decode_bf16` for full decode attention
/// in a single kernel call.
///
/// Computes Q×K scores, softmax, and V accumulation across all cached tokens
/// and KV heads in one kernel launch. Returns `[num_kv_heads * head_dim]` output.
///
/// # Arguments
/// * `stream` — CUDA stream
/// * `kernel` — Loaded function handle for `infers_paged_attention_decode_bf16`
/// * `q` — Query tensor `[num_kv_heads × head_dim]` (single token)
/// * `page_pool` — Flat GPU buffer for paged KV data
/// * `block_table_gpu` — Block table on GPU `[num_pages]`
/// * `num_pages` — Number of pages in block table
/// * `num_cached_tokens` — Number of cached tokens
/// * `head_dim` — Per-head dimension
/// * `num_kv_heads` — Number of KV heads
/// * `page_size` — Tokens per page
/// * `kv_dim` — num_kv_heads × head_dim
pub fn paged_attention_decode(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    q: &CudaSlice<bf16>,
    page_pool: &CudaSlice<bf16>,
    block_table_gpu: &CudaSlice<i32>,
    num_pages: usize,
    num_cached_tokens: usize,
    head_dim: usize,
    num_kv_heads: usize,
    page_size: usize,
    kv_dim: usize,
) -> Result<CudaSlice<bf16>> {
    let output_size = num_kv_heads * head_dim;
    let mut output = stream
        .alloc_zeros::<bf16>(output_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate attention output buffer: {e}"))?;

    // One block per KV head
    let grid = (num_kv_heads as u32, 1, 1);
    let block = (BLOCK_SIZE as u32, 1, 1);
    // Shared memory: 3 × blockDim × sizeof(float)
    let shared_mem_bytes = (3 * BLOCK_SIZE * std::mem::size_of::<f32>()) as u32;

    let config = LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes,
    };

    let num_pages_i32 = num_pages as i32;
    let num_cached_tokens_i32 = num_cached_tokens as i32;
    let head_dim_i32 = head_dim as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;
    let page_size_i32 = page_size as i32;
    let kv_dim_i32 = kv_dim as i32;

    unsafe {
        stream
            .launch_builder(kernel)
            .arg(q)
            .arg(page_pool)
            .arg(block_table_gpu)
            .arg(&num_pages_i32)
            .arg(&num_cached_tokens_i32)
            .arg(&head_dim_i32)
            .arg(&num_kv_heads_i32)
            .arg(&page_size_i32)
            .arg(&kv_dim_i32)
            .arg(&mut output)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("Paged attention decode kernel launch failed: {e}"))?;
    }

    Ok(output)
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
/// * `rope_kernel` — CUDA kernel for RoPE
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
    int4_kernel: &CudaFunction,
    stream: &Arc<CudaStream>,
    softmax_kernel: &CudaFunction,
    kv_cache_write_kernel: &CudaFunction,
    rope_kernel: &CudaFunction,
    rmsnorm_kernel: &CudaFunction,
    add_kernel: &CudaFunction,
    weights: &AttentionWeights,
    input: &CudaSlice<bf16>,
    kv_cache: &mut KvCache,
    positions: &[u32],
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    rope_theta: f64,
    partial_rotary_factor: f32,
    rms_norm_eps: f32,
    group_size: usize,
    int4_companions: &HashMap<String, Int4Companions>,
) -> Result<CudaSlice<bf16>> {
    let hidden_size = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let seq_len = positions.len();

    anyhow::ensure!(
        num_heads == num_kv_heads,
        "GQA (num_heads != num_kv_heads) not yet supported"
    );

    // =========================================================================
    // Phase 1: Full K, V computation + RoPE + KV cache write
    // =========================================================================

    // k_full = GEMM(input, k_proj^T)  [seq_len × kv_dim] (INT4-aware)
    let mut k_full = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate K buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.k_proj, input, &mut k_full,
        seq_len, kv_dim, hidden_size, group_size, int4_companions,
    )?;

    // v_full = GEMM(input, v_proj^T)  [seq_len × kv_dim] (INT4-aware)
    let mut v_full = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate V buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.v_proj, input, &mut v_full,
        seq_len, kv_dim, hidden_size, group_size, int4_companions,
    )?;

    // --- K-norm on full K before Phase 1 RoPE ---
    if let Some(k_norm_w) = weights.k_norm.as_ref() {
        let k_norm_gpu = crate::upload::upload_weight(stream, k_norm_w)?;
        k_full = crate::norm::rms_norm(
            stream, rmsnorm_kernel, &k_full, &k_norm_gpu, rms_norm_eps, head_dim,
        )?;
    }

    // Apply RoPE to K_full. rope::apply_rope modifies both Q and K in-place;
    // we allocate a dummy Q buffer and discard it.
    let mut q_dummy = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dummy Q buffer for RoPE: {e}"))?;
    rope::apply_rope(
        stream,
        rope_kernel,
        &mut q_dummy,
        &mut k_full,
        positions,
        num_kv_heads as i32,
        head_dim,
        rope_theta,
        partial_rotary_factor,
    )?;

    // Write K and V to KV cache
    let kv_buf = kv_cache.ensure_allocated(stream, max_seq_len, kv_dim)?;
    let positions_gpu = stream
        .clone_htod(positions)
        .map_err(|e| anyhow::anyhow!("Failed to copy positions to device: {e}"))?;

    let kv_total = seq_len * kv_dim;
    let kv_grid = (kv_total as u32).div_ceil(256);
    let kv_config = LaunchConfig {
        grid_dim: (kv_grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let kv_seq_len_i32 = seq_len as i32;
    let kv_head_dim_i32 = kv_dim as i32;
    let kv_max_seq_len_i32 = max_seq_len as i32;

    unsafe {
        stream
            .launch_builder(kv_cache_write_kernel)
            .arg(&k_full)
            .arg(&v_full)
            .arg(kv_buf)
            .arg(&positions_gpu)
            .arg(&kv_seq_len_i32)
            .arg(&kv_head_dim_i32)
            .arg(&kv_max_seq_len_i32)
            .launch(kv_config)
            .map_err(|e| anyhow::anyhow!("KV cache write kernel launch failed: {e}"))?;
    }

    // =========================================================================
    // Phase 2: Per-head attention with alternating accumulation
    // =========================================================================

    let buf_size = seq_len * hidden_size;
    let mut accum_a = stream
        .alloc_zeros::<bf16>(buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate accum_a buffer: {e}"))?;
    let mut accum_b = stream
        .alloc_zeros::<bf16>(buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate accum_b buffer: {e}"))?;

    // --- Dequantize weights to BF16 once before the loop ---
    let q_proj_bf16 = weight_to_bf16_wd(&weights.q_proj, int4_companions, group_size)?;
    let k_proj_bf16 = weight_to_bf16_wd(&weights.k_proj, int4_companions, group_size)?;
    let v_proj_bf16 = weight_to_bf16_wd(&weights.v_proj, int4_companions, group_size)?;
    let o_proj_bf16 = weight_to_bf16_wd(&weights.o_proj, int4_companions, group_size)?;

    // --- Upload QK-norm weights to GPU once before the loop ---
    let q_norm_gpu = weights
        .q_norm
        .as_ref()
        .map(|w| crate::upload::upload_weight(stream, w))
        .transpose()?;
    let k_norm_gpu = weights
        .k_norm
        .as_ref()
        .map(|w| crate::upload::upload_weight(stream, w))
        .transpose()?;
    for head_idx in 0..num_heads {
        // --- Extract and upload per-head weight slices ---
        let q_proj_h = extract_head_weight_slice(&q_proj_bf16, head_idx, head_dim)?;
        let k_proj_h = extract_head_weight_slice(&k_proj_bf16, head_idx, head_dim)?;
        let v_proj_h = extract_head_weight_slice(&v_proj_bf16, head_idx, head_dim)?;
        let o_proj_h = extract_o_proj_head_slice(&o_proj_bf16, head_idx, head_dim)?;

        let q_proj_h_gpu = upload_bf16_slice(stream, &q_proj_h)?;
        let k_proj_h_gpu = upload_bf16_slice(stream, &k_proj_h)?;
        let v_proj_h_gpu = upload_bf16_slice(stream, &v_proj_h)?;
        let o_proj_h_gpu = upload_bf16_slice(stream, &o_proj_h)?;

        // --- Q, K, V projections (per-head) ---
        let mut q_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate q_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: head_dim,
                k: hidden_size,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            input,
            &q_proj_h_gpu,
            &mut q_h,
        )?;

        let mut k_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate k_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: head_dim,
                k: hidden_size,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            input,
            &k_proj_h_gpu,
            &mut k_h,
        )?;

        let mut v_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate v_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: head_dim,
                k: hidden_size,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            input,
            &v_proj_h_gpu,
            &mut v_h,
        )?;

        // --- QK-norm (before RoPE) ---
        if let Some(q_norm_w) = &q_norm_gpu {
            q_h = crate::norm::rms_norm_per_head(
                stream,
                rmsnorm_kernel,
                &q_h,
                q_norm_w,
                rms_norm_eps,
                head_dim,
            )?;
        }
        if let Some(k_norm_w) = &k_norm_gpu {
            k_h = crate::norm::rms_norm_per_head(
                stream,
                rmsnorm_kernel,
                &k_h,
                k_norm_w,
                rms_norm_eps,
                head_dim,
            )?;
        }

        // --- RoPE (per-head, num_heads=1) ---
        rope::apply_rope(
            stream,
            rope_kernel,
            &mut q_h,
            &mut k_h,
            positions,
            1,
            head_dim,
            rope_theta,
            partial_rotary_factor,
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
                alpha: 1.0,
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

        // Block size: next power of 2 up to seq_len, capped at 256
        let block_size = {
            let mut sz = 1usize;
            while sz < seq_len && sz < 256 {
                sz *= 2;
            }
            sz
        };
        let shared_mem_bytes = block_size * std::mem::size_of::<f32>();

        let softmax_config = LaunchConfig {
            grid_dim: (seq_len as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        let seq_len_i32 = seq_len as i32;
        let use_causal = 1i32;

        unsafe {
            stream
                .launch_builder(softmax_kernel)
                .arg(&scores_h)
                .arg(&mut softmax_out_h)
                .arg(&seq_len_i32)
                .arg(&use_causal)
                .launch(softmax_config)
                .map_err(|e| anyhow::anyhow!("Softmax kernel launch failed: {e}"))?;
        }

        // --- Attention output: softmax_out_h @ V_h → [seq_len × head_dim] ---
        let mut attn_out_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate attn_out_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: head_dim,
                k: seq_len,
                transa: true,
                transb: true,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &softmax_out_h,
            &v_h,
            &mut attn_out_h,
        )?;

        // --- Partial O-projection: attn_out_h @ o_proj_h^T → [seq_len × hidden_size] ---
        let mut partial_out = stream
            .alloc_zeros::<bf16>(buf_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate partial_out buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: hidden_size,
                k: head_dim,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &attn_out_h,
            &o_proj_h_gpu,
            &mut partial_out,
        )?;

        // --- Accumulate into alternating buffers ---
        if head_idx % 2 == 0 {
            accum_b = add::add(stream, add_kernel, &accum_a, &partial_out)?;
        } else {
            accum_a = add::add(stream, add_kernel, &accum_b, &partial_out)?;
        }
    }

    // Final result: accum_a if num_heads is even (last head odd → wrote accum_a), accum_b if odd (last head even → wrote accum_b)
    let output = if num_heads.is_multiple_of(2) { accum_a } else { accum_b };
    Ok(output)
}

/// Decode-time attention: single-token attention over cached KV.
///
/// Projects a single token into Q/K/V, applies RoPE, appends to KV cache,
/// and computes attention against all previously cached tokens.
///
/// Uses per-head weight slicing like [[forward]], with per-head KV cache
/// extraction from the flat buffer on the CPU.
///
/// # Steps
/// 1. Compute full K and V for single token, apply RoPE, write to KV cache
/// 2. Download full KV cache to CPU, extract per-head K and V buffers
/// 3. Per-head: Q projection → RoPE → attention scores → softmax → attn out → partial O-proj
/// 4. Accumulate partial results into final output
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `stream` — CUDA stream
/// * `softmax_kernel` — CUDA kernel for softmax
/// * `kv_cache_write_kernel` — CUDA kernel for KV cache write
/// * `rope_kernel` — CUDA kernel for RoPE
/// * `add_kernel` — CUDA kernel for element-wise addition
/// * `weights` — Attention weights for this layer
/// * `input` — Single-token input `[1 × hidden_size]`
/// * `kv_cache` — KV cache state (pre-populated from prefill)
/// * `position` — Current decode position for RoPE (0-based)
/// * `head_dim` — Per-head dimension
/// * `num_heads` — Number of attention heads
/// * `num_kv_heads` — Number of KV heads (must equal num_heads for now)
/// * `max_seq_len` — Maximum sequence length for cache allocation
/// * `rope_theta` — RoPE base frequency
/// * `partial_rotary_factor` — Fraction of head_dim to apply RoPE to
///
/// # Returns
/// Attention output `[1 × hidden_size]`
pub fn decode_forward(
    gemm: &mut GemmEngine,
    int4_kernel: &CudaFunction,
    stream: &Arc<CudaStream>,
    softmax_kernel: &CudaFunction,
    kv_cache_write_kernel: &CudaFunction,
    rope_kernel: &CudaFunction,
    rmsnorm_kernel: &CudaFunction,
    add_kernel: &CudaFunction,
    weights: &AttentionWeights,
    input: &CudaSlice<bf16>,
    kv_cache: &mut KvCache,
    position: u32,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    rope_theta: f64,
    partial_rotary_factor: f32,
    rms_norm_eps: f32,
    group_size: usize,
    int4_companions: &HashMap<String, Int4Companions>,
) -> Result<CudaSlice<bf16>> {
    let hidden_size = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let num_cached = (position + 1) as usize;

    anyhow::ensure!(
        num_heads == num_kv_heads,
        "GQA (num_heads != num_kv_heads) not yet supported"
    );

    // =========================================================================
    // Phase 1: Full K, V computation + RoPE + KV cache write
    // =========================================================================

    // k_single = GEMM(input, k_proj^T)  [1 × kv_dim] (INT4-aware)
    let mut k_single = stream
        .alloc_zeros::<bf16>(kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate K buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.k_proj, input, &mut k_single,
        1, kv_dim, hidden_size, group_size, int4_companions,
    )?;

    // v_single = GEMM(input, v_proj^T)  [1 × kv_dim] (INT4-aware)
    let mut v_single = stream
        .alloc_zeros::<bf16>(kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate V buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.v_proj, input, &mut v_single,
        1, kv_dim, hidden_size, group_size, int4_companions,
    )?;

    // --- K-norm on full K before Phase 1 RoPE ---
    if let Some(k_norm_w) = weights.k_norm.as_ref() {
        let k_norm_gpu = crate::upload::upload_weight(stream, k_norm_w)?;
        k_single = crate::norm::rms_norm(
            stream, rmsnorm_kernel, &k_single, &k_norm_gpu, rms_norm_eps, head_dim,
        )?;
    }

    // Apply RoPE to K_single
    let mut q_dummy = stream
        .alloc_zeros::<bf16>(kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dummy Q buffer for RoPE: {e}"))?;
    rope::apply_rope(
        stream,
        rope_kernel,
        &mut q_dummy,
        &mut k_single,
        &[position],
        num_kv_heads as i32,
        head_dim,
        rope_theta,
        partial_rotary_factor,
    )?;

    // Write K and V to KV cache at position
    let kv_buf = kv_cache.ensure_allocated(stream, max_seq_len, kv_dim)?;
    let positions_gpu = stream
        .clone_htod(&[position])
        .map_err(|e| anyhow::anyhow!("Failed to copy position to device: {e}"))?;

    let kv_grid = (kv_dim as u32).div_ceil(256);
    let kv_config = LaunchConfig {
        grid_dim: (kv_grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let kv_seq_len_i32: i32 = 1;
    let kv_head_dim_i32 = kv_dim as i32;
    let kv_max_seq_len_i32 = max_seq_len as i32;

    unsafe {
        stream
            .launch_builder(kv_cache_write_kernel)
            .arg(&k_single)
            .arg(&v_single)
            .arg(kv_buf)
            .arg(&positions_gpu)
            .arg(&kv_seq_len_i32)
            .arg(&kv_head_dim_i32)
            .arg(&kv_max_seq_len_i32)
            .launch(kv_config)
            .map_err(|e| anyhow::anyhow!("KV cache write kernel launch failed: {e}"))?;
    }

    // =========================================================================
    // Phase 2: Download full KV cache to CPU, extract per-head data
    // =========================================================================

    let kv_cache_data: Vec<bf16> = stream
        .clone_dtoh(kv_buf)
        .map_err(|e| anyhow::anyhow!("Failed to download KV cache to host: {e}"))?;

    let k_cache_host = &kv_cache_data[..max_seq_len * kv_dim];
    let v_cache_host = &kv_cache_data[max_seq_len * kv_dim..];

    // =========================================================================
    // Phase 3: Per-head attention with alternating accumulation
    // =========================================================================

    let buf_size = hidden_size;
    let mut accum_a = stream
        .alloc_zeros::<bf16>(buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate accum_a buffer: {e}"))?;
    let mut accum_b = stream
        .alloc_zeros::<bf16>(buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate accum_b buffer: {e}"))?;
    // --- Dequantize weights to BF16 once before the loop ---
    let q_proj_bf16 = weight_to_bf16_wd(&weights.q_proj, int4_companions, group_size)?;
    let o_proj_bf16 = weight_to_bf16_wd(&weights.o_proj, int4_companions, group_size)?;

    // --- Upload QK-norm weight to GPU once before the loop ---
    let q_norm_gpu = weights
        .q_norm
        .as_ref()
        .map(|w| crate::upload::upload_weight(stream, w))
        .transpose()?;
    for head_idx in 0..num_heads {
        // --- Extract and upload per-head weight slices ---
        let q_proj_h = extract_head_weight_slice(&q_proj_bf16, head_idx, head_dim)?;
        let o_proj_h = extract_o_proj_head_slice(&o_proj_bf16, head_idx, head_dim)?;

        let q_proj_h_gpu = upload_bf16_slice(stream, &q_proj_h)?;
        let o_proj_h_gpu = upload_bf16_slice(stream, &o_proj_h)?;

        // --- Extract per-head KV cache on CPU ---
        let mut k_h_cache = Vec::with_capacity(num_cached * head_dim);
        let mut v_h_cache = Vec::with_capacity(num_cached * head_dim);
        for pos in 0..num_cached {
            for d in 0..head_dim {
                k_h_cache.push(k_cache_host[pos * kv_dim + head_idx * head_dim + d]);
                v_h_cache.push(v_cache_host[pos * kv_dim + head_idx * head_dim + d]);
            }
        }

        let k_h_cache_gpu = upload_bf16_slice(stream, &k_h_cache)?;
        let v_h_cache_gpu = upload_bf16_slice(stream, &v_h_cache)?;

        // --- Q projection (per-head) ---
        let mut q_h = stream
            .alloc_zeros::<bf16>(head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate q_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: 1,
                n: head_dim,
                k: hidden_size,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            input,
            &q_proj_h_gpu,
            &mut q_h,
        )?;

        // --- Q-norm (before RoPE) ---
        if let Some(q_norm_w) = &q_norm_gpu {
            q_h = crate::norm::rms_norm_per_head(
                stream,
                rmsnorm_kernel,
                &q_h,
                q_norm_w,
                rms_norm_eps,
                head_dim,
            )?;
        }

        // --- RoPE (per-head, num_heads=1) ---
        let mut k_rope_dummy = stream
            .alloc_zeros::<bf16>(head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate dummy K buffer for RoPE: {e}"))?;
        rope::apply_rope(
            stream,
            rope_kernel,
            &mut q_h,
            &mut k_rope_dummy,
            &[position],
            1,
            head_dim,
            rope_theta,
            partial_rotary_factor,
        )?;

        // --- Attention scores: Q_h @ K_h_cache^T → [1 × num_cached] ---
        let mut scores_h = stream
            .alloc_zeros::<bf16>(num_cached)
            .map_err(|e| anyhow::anyhow!("Failed to allocate scores buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: 1,
                n: num_cached,
                k: head_dim,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &q_h,
            &k_h_cache_gpu,
            &mut scores_h,
        )?;

        // --- Softmax (no causal mask for decode) ---
        let mut softmax_out_h = stream
            .alloc_zeros::<bf16>(num_cached)
            .map_err(|e| anyhow::anyhow!("Failed to allocate softmax output buffer: {e}"))?;

        let block_size = {
            let mut sz = 1usize;
            while sz < num_cached && sz < 256 {
                sz *= 2;
            }
            sz
        };
        let shared_mem_bytes = block_size * std::mem::size_of::<f32>();

        let softmax_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        let num_cached_i32 = num_cached as i32;
        let use_causal = 0i32;

        unsafe {
            stream
                .launch_builder(softmax_kernel)
                .arg(&scores_h)
                .arg(&mut softmax_out_h)
                .arg(&num_cached_i32)
                .arg(&use_causal)
                .launch(softmax_config)
                .map_err(|e| anyhow::anyhow!("Softmax kernel launch failed: {e}"))?;
        }

        // --- Attention output: softmax_out_h @ V_h_cache → [1 × head_dim] ---
        let mut attn_out_h = stream
            .alloc_zeros::<bf16>(head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate attn_out_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: 1,
                n: head_dim,
                k: num_cached,
                transa: true,
                transb: true,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &softmax_out_h,
            &v_h_cache_gpu,
            &mut attn_out_h,
        )?;

        // --- Partial O-projection: attn_out_h @ o_proj_h^T → [1 × hidden_size] ---
        let mut partial_out = stream
            .alloc_zeros::<bf16>(buf_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate partial_out buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: 1,
                n: hidden_size,
                k: head_dim,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &attn_out_h,
            &o_proj_h_gpu,
            &mut partial_out,
        )?;

        // --- Accumulate into alternating buffers ---
        if head_idx % 2 == 0 {
            accum_b = add::add(stream, add_kernel, &accum_a, &partial_out)?;
        } else {
            accum_a = add::add(stream, add_kernel, &accum_b, &partial_out)?;
        }
    }

    let output = if num_heads.is_multiple_of(2) { accum_a } else { accum_b };
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
    int4_kernel: &CudaFunction,
    stream: &Arc<CudaStream>,
    softmax_kernel: &CudaFunction,
    paged_kv_write_kernel: &CudaFunction,
    rope_kernel: &CudaFunction,
    rmsnorm_kernel: &CudaFunction,
    add_kernel: &CudaFunction,
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
    int4_companions: &HashMap<String, Int4Companions>,
) -> Result<CudaSlice<bf16>> {
    let hidden_size = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let seq_len = positions.len();

    anyhow::ensure!(
        num_heads == num_kv_heads,
        "GQA (num_heads != num_kv_heads) not yet supported"
    );

    // =========================================================================
    // Phase 1: Full K, V computation + RoPE
    // =========================================================================

    // k_full = GEMM(input, k_proj^T)  [seq_len × kv_dim] (INT4-aware)
    let mut k_full = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate K buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.k_proj, input, &mut k_full,
        seq_len, kv_dim, hidden_size, group_size, int4_companions,
    )?;

    // v_full = GEMM(input, v_proj^T)  [seq_len × kv_dim] (INT4-aware)
    let mut v_full = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate V buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.v_proj, input, &mut v_full,
        seq_len, kv_dim, hidden_size, group_size, int4_companions,
    )?;

    // --- K-norm on full K before Phase 1 RoPE ---
    if let Some(k_norm_w) = weights.k_norm.as_ref() {
        let k_norm_gpu = crate::upload::upload_weight(stream, k_norm_w)?;
        k_full = crate::norm::rms_norm(
            stream, rmsnorm_kernel, &k_full, &k_norm_gpu, rms_norm_eps, head_dim,
        )?;
    }

    // Apply RoPE to K_full
    let mut q_dummy = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dummy Q buffer for RoPE: {e}"))?;
    rope::apply_rope(
        stream,
        rope_kernel,
        &mut q_dummy,
        &mut k_full,
        positions,
        num_kv_heads as i32,
        head_dim,
        rope_theta,
        partial_rotary_factor,
    )?;

    // =========================================================================
    // Phase 2: Paged KV write
    // =========================================================================

    let page_pool = paged_cache.ensure_allocated(stream)?;

    paged_kv_write(
        stream,
        paged_kv_write_kernel,
        &k_full,
        &v_full,
        page_pool,
        block_table_gpu,
        positions_gpu,
        seq_len,
        head_dim,
        kv_dim,
        page_size,
    )?;

    // =========================================================================
    // Phase 3: Per-head attention (same as [[forward]] but using full K/V buffers)
    // =========================================================================

    let buf_size = seq_len * hidden_size;
    let mut accum_a = stream
        .alloc_zeros::<bf16>(buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate accum_a buffer: {e}"))?;
    let mut accum_b = stream
        .alloc_zeros::<bf16>(buf_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate accum_b buffer: {e}"))?;
    // --- Dequantize weights to BF16 once before the loop ---
    let q_proj_bf16 = weight_to_bf16_wd(&weights.q_proj, int4_companions, group_size)?;
    let k_proj_bf16 = weight_to_bf16_wd(&weights.k_proj, int4_companions, group_size)?;
    let v_proj_bf16 = weight_to_bf16_wd(&weights.v_proj, int4_companions, group_size)?;
    let o_proj_bf16 = weight_to_bf16_wd(&weights.o_proj, int4_companions, group_size)?;

    // --- Upload QK-norm weights to GPU once before the loop ---
    let q_norm_gpu = weights
        .q_norm
        .as_ref()
        .map(|w| crate::upload::upload_weight(stream, w))
        .transpose()?;
    let k_norm_gpu = weights
        .k_norm
        .as_ref()
        .map(|w| crate::upload::upload_weight(stream, w))
        .transpose()?;
    for head_idx in 0..num_heads {
        // --- Extract and upload per-head weight slices ---
        let q_proj_h = extract_head_weight_slice(&q_proj_bf16, head_idx, head_dim)?;
        let k_proj_h = extract_head_weight_slice(&k_proj_bf16, head_idx, head_dim)?;
        let v_proj_h = extract_head_weight_slice(&v_proj_bf16, head_idx, head_dim)?;
        let o_proj_h = extract_o_proj_head_slice(&o_proj_bf16, head_idx, head_dim)?;

        let q_proj_h_gpu = upload_bf16_slice(stream, &q_proj_h)?;
        let k_proj_h_gpu = upload_bf16_slice(stream, &k_proj_h)?;
        let v_proj_h_gpu = upload_bf16_slice(stream, &v_proj_h)?;
        let o_proj_h_gpu = upload_bf16_slice(stream, &o_proj_h)?;

        // --- Q, K, V projections (per-head) ---
        let mut q_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate q_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: head_dim,
                k: hidden_size,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            input,
            &q_proj_h_gpu,
            &mut q_h,
        )?;

        let mut k_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate k_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: head_dim,
                k: hidden_size,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            input,
            &k_proj_h_gpu,
            &mut k_h,
        )?;

        let mut v_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate v_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: head_dim,
                k: hidden_size,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            input,
            &v_proj_h_gpu,
            &mut v_h,
        )?;

        // --- QK-norm (before RoPE) ---
        if let Some(q_norm_w) = &q_norm_gpu {
            q_h = crate::norm::rms_norm_per_head(
                stream,
                rmsnorm_kernel,
                &q_h,
                q_norm_w,
                rms_norm_eps,
                head_dim,
            )?;
        }
        if let Some(k_norm_w) = &k_norm_gpu {
            k_h = crate::norm::rms_norm_per_head(
                stream,
                rmsnorm_kernel,
                &k_h,
                k_norm_w,
                rms_norm_eps,
                head_dim,
            )?;
        }

        // --- RoPE (per-head, num_heads=1) ---
        rope::apply_rope(
            stream,
            rope_kernel,
            &mut q_h,
            &mut k_h,
            positions,
            1,
            head_dim,
            rope_theta,
            partial_rotary_factor,
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
                alpha: 1.0,
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

        let block_size = {
            let mut sz = 1usize;
            while sz < seq_len && sz < 256 {
                sz *= 2;
            }
            sz
        };
        let shared_mem_bytes = block_size * std::mem::size_of::<f32>();

        let softmax_config = LaunchConfig {
            grid_dim: (seq_len as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        let seq_len_i32 = seq_len as i32;
        let use_causal = 1i32;

        unsafe {
            stream
                .launch_builder(softmax_kernel)
                .arg(&scores_h)
                .arg(&mut softmax_out_h)
                .arg(&seq_len_i32)
                .arg(&use_causal)
                .launch(softmax_config)
                .map_err(|e| anyhow::anyhow!("Softmax kernel launch failed: {e}"))?;
        }

        // --- Attention output: softmax_out_h @ V_h → [seq_len × head_dim] ---
        let mut attn_out_h = stream
            .alloc_zeros::<bf16>(seq_len * head_dim)
            .map_err(|e| anyhow::anyhow!("Failed to allocate attn_out_h buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: head_dim,
                k: seq_len,
                transa: true,
                transb: true,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &softmax_out_h,
            &v_h,
            &mut attn_out_h,
        )?;

        // --- Partial O-projection ---
        let mut partial_out = stream
            .alloc_zeros::<bf16>(buf_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate partial_out buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: seq_len,
                n: hidden_size,
                k: head_dim,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &attn_out_h,
            &o_proj_h_gpu,
            &mut partial_out,
        )?;

        // --- Accumulate ---
        if head_idx % 2 == 0 {
            accum_b = add::add(stream, add_kernel, &accum_a, &partial_out)?;
        } else {
            accum_a = add::add(stream, add_kernel, &accum_b, &partial_out)?;
        }
    }

    let output = if num_heads.is_multiple_of(2) { accum_a } else { accum_b };
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
    int4_kernel: &CudaFunction,
    stream: &Arc<CudaStream>,
    paged_kv_write_kernel: &CudaFunction,
    paged_attention_decode_kernel: &CudaFunction,
    rope_kernel: &CudaFunction,
    rmsnorm_kernel: &CudaFunction,
    _add_kernel: &CudaFunction,
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
    int4_companions: &HashMap<String, Int4Companions>,
) -> Result<CudaSlice<bf16>> {
    let hidden_size = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    anyhow::ensure!(
        num_heads == num_kv_heads,
        "GQA (num_heads != num_kv_heads) not yet supported"
    );

    // =========================================================================
    // Phase 1: Single-token K, V computation + RoPE
    // =========================================================================

    // k_single = GEMM(input, k_proj^T)  [1 × kv_dim] (INT4-aware)
    let mut k_single = stream
        .alloc_zeros::<bf16>(kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate K buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.k_proj, input, &mut k_single,
        1, kv_dim, hidden_size, group_size, int4_companions,
    )?;

    // v_single = GEMM(input, v_proj^T)  [1 × kv_dim] (INT4-aware)
    let mut v_single = stream
        .alloc_zeros::<bf16>(kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate V buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.v_proj, input, &mut v_single,
        1, kv_dim, hidden_size, group_size, int4_companions,
    )?;

    // --- K-norm on full K before RoPE ---
    if let Some(k_norm_w) = weights.k_norm.as_ref() {
        let k_norm_gpu = crate::upload::upload_weight(stream, k_norm_w)?;
        k_single = crate::norm::rms_norm(
            stream, rmsnorm_kernel, &k_single, &k_norm_gpu, rms_norm_eps, head_dim,
        )?;
    }

    // Apply RoPE to K_single
    let mut q_dummy = stream
        .alloc_zeros::<bf16>(kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dummy Q buffer for RoPE: {e}"))?;
    rope::apply_rope(
        stream,
        rope_kernel,
        &mut q_dummy,
        &mut k_single,
        &[position],
        num_kv_heads as i32,
        head_dim,
        rope_theta,
        partial_rotary_factor,
    )?;

    // =========================================================================
    // Phase 2: Paged KV write — write new token to page pool
    // =========================================================================

    let page_pool = paged_cache.ensure_allocated(stream)?;

    paged_kv_write(
        stream,
        paged_kv_write_kernel,
        &k_single,
        &v_single,
        page_pool,
        block_table_gpu,
        positions_gpu,
        1, // seq_len = 1 for decode
        head_dim,
        kv_dim,
        page_size,
    )?;

    // =========================================================================
    // Phase 3: Compute Q for attention decode kernel
    // =========================================================================

    // Q projection: full Q via GEMM [1 × hidden_size] @ [hidden_size × kv_dim] → [1 × kv_dim] (INT4-aware)
    let mut q_single = stream
        .alloc_zeros::<bf16>(kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate Q buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.q_proj, input, &mut q_single,
        1, kv_dim, hidden_size, group_size, int4_companions,
    )?;

    // --- QK-norm on full Q before RoPE ---
    if let Some(q_norm_w) = weights.q_norm.as_ref() {
        let q_norm_gpu = crate::upload::upload_weight(stream, q_norm_w)?;
        q_single = crate::norm::rms_norm(
            stream, rmsnorm_kernel, &q_single, &q_norm_gpu, rms_norm_eps, head_dim,
        )?;
    }
    // Apply RoPE to Q_single
    let mut k_rope_dummy = stream
        .alloc_zeros::<bf16>(kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate dummy K buffer for RoPE: {e}"))?;
    rope::apply_rope(
        stream,
        rope_kernel,
        &mut q_single,
        &mut k_rope_dummy,
        &[position],
        num_kv_heads as i32,
        head_dim,
        rope_theta,
        partial_rotary_factor,
    )?;

    // =========================================================================
    // Phase 4: Paged attention decode — scores, softmax, V accumulation in one kernel
    // =========================================================================

    let num_pages = block_table_gpu.len();

    let attn_output = paged_attention_decode(
        stream,
        paged_attention_decode_kernel,
        &q_single,
        page_pool,
        block_table_gpu,
        num_pages,
        num_cached_tokens as usize,
        head_dim,
        num_kv_heads,
        page_size,
        kv_dim,
    )?;

    // =========================================================================
    // Phase 5: O-projection — single GEMM over all heads (INT4-aware)
    // =========================================================================

    // attention_output is [num_kv_heads * head_dim] = [kv_dim]
    // o_proj is [kv_dim × hidden_size]
    // output = attention_output @ o_proj^T → [1 × hidden_size]

    let mut output = stream
        .alloc_zeros::<bf16>(hidden_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate O-proj output buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection(
        gemm, int4_kernel, stream,
        &weights.o_proj, &attn_output, &mut output,
        1, hidden_size, kv_dim, group_size, int4_companions,
    )?;

    Ok(output)
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
