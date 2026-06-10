//! Standard softmax attention forward pass.
//!
//! Implements the full-attention layer used every 4th layer in the hybrid
//! attention pattern. Uses per-head weight slicing to avoid strided GPU
//! sub-slices (cudarc limitation).

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_model::{AttentionWeights, WeightData};

use crate::add;
use crate::rope;

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
    stream: &Arc<CudaStream>,
    softmax_kernel: &CudaFunction,
    kv_cache_write_kernel: &CudaFunction,
    rope_kernel: &CudaFunction,
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

    // Upload full K and V projection weights
    let k_proj_full = crate::upload::upload_weight(stream, &weights.k_proj)?;
    let v_proj_full = crate::upload::upload_weight(stream, &weights.v_proj)?;

    // k_full = GEMM(input, k_proj^T)  [seq_len × kv_dim]
    let mut k_full = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate K buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
            n: kv_dim,
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
        &k_proj_full,
        &mut k_full,
    )?;

    // v_full = GEMM(input, v_proj^T)  [seq_len × kv_dim]
    let mut v_full = stream
        .alloc_zeros::<bf16>(seq_len * kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate V buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
            n: kv_dim,
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
        &v_proj_full,
        &mut v_full,
    )?;

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
    let kv_grid = ((kv_total as u32) + 255) / 256;
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

    for head_idx in 0..num_heads {
        // --- Extract and upload per-head weight slices ---
        let q_proj_h = extract_head_weight_slice(&weights.q_proj, head_idx, head_dim)?;
        let k_proj_h = extract_head_weight_slice(&weights.k_proj, head_idx, head_dim)?;
        let v_proj_h = extract_head_weight_slice(&weights.v_proj, head_idx, head_dim)?;
        let o_proj_h = extract_o_proj_head_slice(&weights.o_proj, head_idx, head_dim)?;

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
    let output = if num_heads % 2 == 0 { accum_a } else { accum_b };
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
    stream: &Arc<CudaStream>,
    softmax_kernel: &CudaFunction,
    kv_cache_write_kernel: &CudaFunction,
    rope_kernel: &CudaFunction,
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

    let k_proj_full = crate::upload::upload_weight(stream, &weights.k_proj)?;
    let v_proj_full = crate::upload::upload_weight(stream, &weights.v_proj)?;

    // k_single = GEMM(input, k_proj^T)  [1 × kv_dim]
    let mut k_single = stream
        .alloc_zeros::<bf16>(kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate K buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: 1,
            n: kv_dim,
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
        &k_proj_full,
        &mut k_single,
    )?;

    // v_single = GEMM(input, v_proj^T)  [1 × kv_dim]
    let mut v_single = stream
        .alloc_zeros::<bf16>(kv_dim)
        .map_err(|e| anyhow::anyhow!("Failed to allocate V buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: 1,
            n: kv_dim,
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
        &v_proj_full,
        &mut v_single,
    )?;

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

    let kv_grid = ((kv_dim as u32) + 255) / 256;
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

    for head_idx in 0..num_heads {
        // --- Extract and upload per-head weight slices ---
        let q_proj_h = extract_head_weight_slice(&weights.q_proj, head_idx, head_dim)?;
        let o_proj_h = extract_o_proj_head_slice(&weights.o_proj, head_idx, head_dim)?;

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

    let output = if num_heads % 2 == 0 { accum_a } else { accum_b };
    Ok(output)
}
