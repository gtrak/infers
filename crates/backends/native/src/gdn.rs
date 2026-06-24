//! Gated DeltaNet (GDN) forward pass using the correct Gated Delta Rule.
//!
//! Implements the Gated Delta Rule recurrence from the HuggingFace
//! Qwen3_5GatedDeltaNet reference:
//!
//!   S[t] = S[t-1] * exp(g[t])                        // state decay
//!   S[t] += k[t] ⊗ (β[t] ⊙ (v[t] - S[t-1] @ k[t]))  // delta rule update
//!   y[t] = S[t] @ q[t]                                // output readout
//!
//! And the output gate:
//!   output = RMSNormGated(y, z) @ out_proj
//!
//! Depends on:
//! - `conv1d_depthwise_kernel` — depthwise 1D conv on in_proj_qkv output
//! - `gdn_gated_delta_prefill` — multi-token prefill kernel
//! - `gdn_gated_delta_update` — single-token decode kernel
//! - `rms_norm_gated_kernel` — RMSNorm with SiLU gating

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaSlice, CudaStream, OxideKernels};
use infers_cuda::gemm::GemmEngine;
use infers_model::{GdnWeights, ModelConfig, WeightDtype};
use crate::probe;
use crate::probe::ProbeConfig;

/// Get the output dimension from a weight tensor.
fn weight_output_dim(w: &infers_model::WeightData) -> usize {
    if w.dtype == WeightDtype::Int4Packed { w.shape[1] } else { w.shape[0] }
}

/// GDN recurrent state — 2D matrix [num_heads × head_k_dim × head_v_dim] float32.
/// Also stores the conv1d state buffer: last (kernel_size - 1) tokens' mixed_qkv values,
/// needed for causal conv1d during decode (matches HF `causal_conv1d_update`).
#[derive(Debug)]
pub struct GdnState {
    pub state: Option<CudaSlice<f32>>,
    /// Conv state buffer: last (kernel_size - 1) tokens' mixed_qkv, shape [conv_dim * (kernel_size - 1)] bf16.
    pub conv_state: Option<CudaSlice<bf16>>,
    pub num_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_state_len: usize,  // kernel_size - 1
    pub conv_dim: usize,
}

impl Default for GdnState {
    fn default() -> Self { Self::new() }
}

impl GdnState {
    pub fn new() -> Self {
        Self { state: None, conv_state: None, num_heads: 0, head_k_dim: 0, head_v_dim: 0, conv_state_len: 0, conv_dim: 0 }
    }

    pub fn ensure_allocated(
        &mut self,
        stream: &Arc<CudaStream>,
        num_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    ) -> Result<()> {
        let total = num_heads * head_k_dim * head_v_dim;
        if self.state.is_none() || self.num_heads != num_heads
            || self.head_k_dim != head_k_dim || self.head_v_dim != head_v_dim
        {
            self.state = Some(
                stream.alloc_zeros::<f32>(total)
                    .map_err(|e| anyhow::anyhow!("Failed to allocate GDN state: {e}"))?,
            );
            self.num_heads = num_heads;
            self.head_k_dim = head_k_dim;
            self.head_v_dim = head_v_dim;
        }
        Ok(())
    }

    pub fn ensure_conv_state_allocated(
        &mut self,
        stream: &Arc<CudaStream>,
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<()> {
        let state_len = kernel_size - 1;
        let total = conv_dim * state_len;
        if self.conv_state.is_none() || self.conv_dim != conv_dim || self.conv_state_len != state_len {
            self.conv_state = Some(
                stream.alloc_zeros::<bf16>(total)
                    .map_err(|e| anyhow::anyhow!("Failed to allocate GDN conv state: {e}"))?,
            );
            self.conv_dim = conv_dim;
            self.conv_state_len = state_len;
        }
        Ok(())
    }
}

/// Upload a small BF16 GPU buffer as a float32 GPU buffer.
///
/// Downloads `src` to CPU (assuming batch=1 layout), converts each element
/// to float32, and uploads to a new device buffer of `count` elements.
/// Copy a range from a CudaSlice into a new CudaSlice (avoids CudaView lifetime issues).
fn clone_view_to_slice(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<bf16>,
    range: std::ops::Range<usize>,
) -> Result<CudaSlice<bf16>> {
    let len = range.end - range.start;
    let mut dst = stream.alloc_zeros::<bf16>(len)
        .map_err(|e| anyhow::anyhow!("Failed to allocate cloned slice: {e}"))?;
    let view = src.slice(range);
    stream.memcpy_dtod(&view, &mut dst)
        .map_err(|e| anyhow::anyhow!("Failed to copy slice: {e}"))?;
    Ok(dst)
}



/// Extract columns `col_start..col_end` from a row-major `[seq_len, conv_dim]` tensor.
///
/// Each row's slice is copied independently — this is NOT a contiguous flat copy.
fn extract_columns(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<bf16>,
    seq_len: usize,
    conv_dim: usize,
    col_start: usize,
    col_end: usize,
) -> Result<CudaSlice<bf16>> {
    let out_width = col_end - col_start;
    let total = seq_len * out_width;
    let mut dst = stream.alloc_zeros::<bf16>(total)
        .map_err(|e| anyhow::anyhow!("Failed to allocate column extract buffer: {e}"))?;
    for t in 0..seq_len {
        let src_offset = t * conv_dim + col_start;
        let dst_offset = t * out_width;
        let copy_len = out_width;
        let src_slice = src.slice(src_offset..src_offset + copy_len);
        let mut dst_slice = dst.slice_mut(dst_offset..dst_offset + copy_len);
        stream.memcpy_dtod(&src_slice, &mut dst_slice)
            .map_err(|e| anyhow::anyhow!("Failed to copy row {t}: {e}"))?;
    }
    Ok(dst)
}

/// Upload a small BF16 GPU buffer as a float32 GPU buffer.

/// Used for A_log and dt_bias (small per-head constant arrays).
#[allow(dead_code)]
fn bf16_to_f32_gpu(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<bf16>,
    _count: usize,
) -> Result<CudaSlice<f32>> {
    let cpu_bf16: Vec<bf16> = stream.clone_dtoh(src)
        .map_err(|e| anyhow::anyhow!("Failed to download BF16 buffer: {e}"))?;
    let cpu_f32: Vec<f32> = cpu_bf16.iter().map(|v| v.to_f32()).collect();
    let dst = stream.clone_htod(&cpu_f32)
        .map_err(|e| anyhow::anyhow!("Failed to upload f32 buffer: {e}"))?;
    stream.synchronize()
        .map_err(|e| anyhow::anyhow!("Failed to sync stream after bf16_to_f32 upload: {e}"))?;
    Ok(dst)
}
// ──────────────────────────────────────────────
// Small helper kernel: repeat_interleave for q/k
// ──────────────────────────────────────────────

/// Repeat-interleave the head dimension: [T, H_src, D] → [T, H_dst, D].
/// Each source head is repeated `ratio = H_dst / H_src` times contiguously.
fn repeat_interleave_heads(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<bf16>,
    seq_len: usize,
    h_src: usize,
    h_dst: usize,
    head_dim: usize,
) -> Result<CudaSlice<bf16>> {
    let total = seq_len * h_dst * head_dim;
    let mut dst = stream.alloc_zeros::<bf16>(total)
        .map_err(|e| anyhow::anyhow!("Failed to allocate repeat buffer: {e}"))?;

    let ratio = h_dst / h_src;
    for t in 0..seq_len {
        for h in 0..h_dst {
            let src_h = h / ratio;
            let src_offset = (t * h_src + src_h) * head_dim;
            let dst_offset = (t * h_dst + h) * head_dim;
            let copy_len = head_dim;
            let src_slice = src.slice(src_offset..src_offset + copy_len);
            let mut dst_slice = dst.slice_mut(dst_offset..dst_offset + copy_len);
            stream.memcpy_dtod(&src_slice, &mut dst_slice)
                .map_err(|e| anyhow::anyhow!("Failed to copy head {h}: {e}"))?;
        }
    }
    Ok(dst)
}

// ──────────────────────────────────────────────
// Prefill forward pass
// ──────────────────────────────────────────────

/// GDN prefill forward with the correct Gated Delta Rule.
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `stream` — CUDA stream
/// * `oxide` — OxideKernels bridge for GDN kernel launches
/// * `weights` — GDN layer weights (includes in_proj_qkv, conv1d, etc.)
/// * `input` — `[seq_len × hidden_size]` BF16
/// * `gdn_state` — Mutable recurrent state
/// * `hidden_size` — Model hidden dimension
/// * `config` — Model config
/// * `group_size` — INT4 group size
/// * `cache` — GPU weight cache
///
/// # Returns
/// GDN output `[seq_len × hidden_size]`
#[allow(unused_assignments, clippy::too_many_arguments)]
  pub fn forward(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    weights: &GdnWeights,
    input: &CudaSlice<bf16>,
    gdn_state: &mut GdnState,
    hidden_size: usize,
    config: &ModelConfig,
    group_size: usize,
    cache: &crate::gpu_cache::GpuWeightCache,
    layer_idx: usize,
    gpu_idx: usize,
    probe: &ProbeConfig,
) -> Result<CudaSlice<bf16>> {
    let seq_len = input.len() / hidden_size;

   // Dump the layer input hidden state for reference comparison
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.hidden_input", input, &[seq_len, hidden_size], "prefill");

    // Compute sharded dimensions from actual weight shapes (TP-aware)
    let num_v_heads = weight_output_dim(&weights.in_proj_b);  // Per-GPU: e.g. 24 at TP=2, 48 at TP=1
    let kv_ratio = config.linear_num_value_heads / config.linear_num_key_heads;  // 3 (model-level constant)
    let num_k_heads = num_v_heads / kv_ratio;                  // Per-GPU: e.g. 8 at TP=2, 16 at TP=1
    let head_k_dim = config.linear_key_head_dim;               // 128
    let head_v_dim = config.linear_value_head_dim;             // 128
    let key_dim = num_k_heads * head_k_dim;                    // Per-GPU: e.g. 1024 at TP=2
    let value_dim = num_v_heads * head_v_dim;                  // Per-GPU: e.g. 3072 at TP=2
    let conv_dim = key_dim * 2 + value_dim;                    // Matches per-GPU dimensions

    // =========================================================================
    // Phase 1: in_proj_qkv projection (if available)
    //   mixed_qkv = input @ in_proj_qkv^T  [seq_len, conv_dim]
    // =========================================================================
    let mut mixed_qkv = stream.alloc_zeros::<bf16>(seq_len * conv_dim)?;
    if let Some(ref qkv_weight) = weights.in_proj_qkv {
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, oxide, stream, cache,
            &qkv_weight.name, input, &mut mixed_qkv,
            seq_len, conv_dim, hidden_size, group_size,
        )?;
    }
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.mixed_qkv", &mixed_qkv, &[seq_len, conv_dim], "prefill");
    // =========================================================================
    // Phase 2: Depthwise conv1d on mixed_qkv (SiLU activation)
    // =========================================================================
    let mut conv_out = stream.alloc_zeros::<bf16>(seq_len * conv_dim)?;
    // conv1d_weight is always present for Qwen3.6
    let conv1d_gpu = cache.get_bf16(&weights.conv1d_weight.name)
        .ok_or_else(|| anyhow::anyhow!("conv1d weight '{}' not in cache", weights.conv1d_weight.name))?;

    // Note: conv1d_weight is a weight, not an intermediate — skipped for probing
    oxide.launch_conv1d_depthwise_silu_bf16(
        stream, &mixed_qkv, conv1d_gpu, &mut conv_out,
        1, conv_dim as u32, seq_len as u32, config.linear_conv_kernel_dim as u32,
    ).map_err(|e| anyhow::anyhow!("conv1d kernel launch failed: {e}"))?;
 probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.conv_out", &conv_out, &[seq_len, conv_dim], "prefill");

    // Save conv state: last (kernel_size - 1) tokens' mixed_qkv for decode causal conv1d.
    // Matches HF: cache_params.update_conv_state(new_conv_state)
    let kernel_size_usize = config.linear_conv_kernel_dim as usize;
    let conv_state_len = kernel_size_usize - 1;
    gdn_state.ensure_conv_state_allocated(stream, conv_dim, kernel_size_usize)?;
    if let Some(ref mut cs) = gdn_state.conv_state {
        if seq_len >= conv_state_len {
            // Copy last conv_state_len rows of mixed_qkv to conv_state
            let src_offset = (seq_len - conv_state_len) * conv_dim;
            let src_view = mixed_qkv.slice(src_offset..seq_len * conv_dim);
            stream.memcpy_dtod(&src_view, cs)
                .map_err(|e| anyhow::anyhow!("Failed to copy conv state: {e}"))?;
        } else {
            // seq_len < conv_state_len: right-shift existing state and prepend zeros
            // This shouldn't happen in normal operation (prefill >= kernel_size)
            let zeros = stream.alloc_zeros::<bf16>(conv_state_len * conv_dim - seq_len * conv_dim)?;
            let mut tmp = stream.alloc_zeros::<bf16>(conv_state_len * conv_dim)?;
            stream.memcpy_dtod(&zeros, &mut tmp)?;
            let offset = (conv_state_len - seq_len) * conv_dim;
            let mut tail = tmp.slice_mut(offset..conv_state_len * conv_dim);
            stream.memcpy_dtod(&mixed_qkv, &mut tail)?;
            stream.memcpy_dtod(&tmp, cs)?;
        }
    }

    // =========================================================================
    // Phase 3: Split conv_out into query, key, value (extract as proper slices)
    //
    // conv_out layout: [seq_len, conv_dim]
    //   query = conv_out[..., :key_dim]        [seq_len, key_dim]
    //   key   = conv_out[..., key_dim:2*key_dim] [seq_len, key_dim]
    //   value = conv_out[..., 2*key_dim:]      [seq_len, value_dim]
    // =========================================================================
    // Extract column slices per-row: conv_out is [seq_len, conv_dim] row-major.
    // query = conv_out[:, :key_dim], key = conv_out[:, key_dim:2*key_dim], value = conv_out[:, 2*key_dim:]
    let query_flat = extract_columns(stream, &conv_out, seq_len, conv_dim, 0, key_dim)?;
    let key_flat = extract_columns(stream, &conv_out, seq_len, conv_dim, key_dim, 2 * key_dim)?;
    let value_flat = extract_columns(stream, &conv_out, seq_len, conv_dim, 2 * key_dim, 2 * key_dim + value_dim)?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.query", &query_flat, &[seq_len, key_dim], "prefill");
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.key", &key_flat, &[seq_len, key_dim], "prefill");
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.value", &value_flat, &[seq_len, value_dim], "prefill");

    // =========================================================================
    // Phase 4: Reshape and repeat_interleave query/key for num_v_heads
    //
    // query: [seq_len, num_k_heads, head_k_dim] → repeat_interleave × kv_ratio
    //      → [seq_len, num_v_heads, head_k_dim]
    // key: same
    // value: [seq_len, num_v_heads, head_v_dim] — already correct layout
    // =========================================================================
    let query_expanded = if kv_ratio > 1 {
        repeat_interleave_heads(stream, &query_flat, seq_len, num_k_heads, num_v_heads, head_k_dim)?
    } else {
        query_flat
    };
    let key_expanded = if kv_ratio > 1 {
        repeat_interleave_heads(stream, &key_flat, seq_len, num_k_heads, num_v_heads, head_k_dim)?
    } else {
        key_flat
    };
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.query_expanded", &query_expanded, &[seq_len, num_v_heads * head_k_dim], "prefill");
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.key_expanded", &key_expanded, &[seq_len, num_v_heads * head_k_dim], "prefill");

    // =========================================================================
    // Phase 5: Per-head scalar projections
    //
    // a_proj = input @ in_proj_a^T  [seq_len, num_v_heads]
    // b_proj = input @ in_proj_b^T  [seq_len, num_v_heads] (extract from b_dim)
    // =========================================================================
    // =========================================================================
    // Phase 5: Per-head scalar projections
    //
    // a_proj = input @ in_proj_a^T  [seq_len, num_v_heads]
    // b_proj = input @ in_proj_b^T  [seq_len, num_v_heads] (extract from b_dim)
    // =========================================================================
    let mut a_proj = stream.alloc_zeros::<bf16>(seq_len * num_v_heads)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream, cache,
        &weights.in_proj_a.name, input, &mut a_proj,
        seq_len, num_v_heads, hidden_size, group_size,
    )?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.a_proj", &a_proj, &[seq_len, num_v_heads], "prefill");

    let b_dim = weight_output_dim(&weights.in_proj_b);
    let mut b_proj_raw = stream.alloc_zeros::<bf16>(seq_len * b_dim)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream, cache,
        &weights.in_proj_b.name, input, &mut b_proj_raw,
        seq_len, b_dim, hidden_size, group_size,
    )?;
    // Use b_proj_raw directly (extraction not needed since b_dim == num_v_heads)
    let b_proj = b_proj_raw;
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.b_proj", &b_proj, &[seq_len, num_v_heads], "prefill");

    // =========================================================================
    // Phase 6: Upload A_log and dt_bias as float32
    // =========================================================================
    let a_log_f32 = if let Some(ref w) = weights.a_log {
        cache.get_f32(&w.name)
            .ok_or_else(|| anyhow::anyhow!("a_log not in f32 cache: {}", w.name))?
            .clone()
    } else {
        stream.alloc_zeros::<f32>(num_v_heads)?
    };

    let dt_bias_f32 = if let Some(ref w) = weights.dt_bias {
        cache.get_f32(&w.name)
            .ok_or_else(|| anyhow::anyhow!("dt_bias not in f32 cache: {}", w.name))?
            .clone()
    } else {
        stream.alloc_zeros::<f32>(num_v_heads)?
    };
    // =========================================================================
    // Phase 7: Chunked parallel GDN recurrence (fp32 state, bf16 I/O)
    // =========================================================================
    gdn_state.ensure_allocated(stream, num_v_heads, head_k_dim, head_v_dim)?;
    let mut gdn_output = stream.alloc_zeros::<bf16>(seq_len * num_v_heads * head_v_dim)?;
    let state_ref = gdn_state.state.as_mut()
        .ok_or_else(|| anyhow::anyhow!("GDN state not allocated"))?;

    let num_v_heads_i32 = num_v_heads as i32;
    let head_k_dim_i32 = head_k_dim as i32;
    let head_v_dim_i32 = head_v_dim as i32;
    // Launch sequential kernel via oxide bridge
    oxide.launch_gdn_gated_delta_prefill_bf16(
        stream, &query_expanded, &key_expanded, &value_flat,
        &a_proj, &b_proj, &a_log_f32, &dt_bias_f32,
        state_ref, &mut gdn_output,
        seq_len as u32, num_v_heads_i32 as u32, head_k_dim_i32 as u32, head_v_dim_i32 as u32,
    ).map_err(|e| anyhow::anyhow!("GDN prefill kernel launch failed: {e}"))?;

    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.core_attn_out", &gdn_output, &[seq_len, num_v_heads * head_v_dim], "prefill");

    // =========================================================================
    // Phase 8: RMSNormGated — norm(gdn_output, z_gate, weight)
    // =========================================================================
    let norm_output = if let Some(ref z_weight) = weights.in_proj_z {
        let z_dim = weight_output_dim(z_weight);
        let mut z_gate_raw = stream.alloc_zeros::<bf16>(seq_len * z_dim)?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, oxide, stream, cache,
            &z_weight.name, input, &mut z_gate_raw,
            seq_len, z_dim, hidden_size, group_size,
        )?;

        let n_rows = seq_len * num_v_heads;
        let norm_dim = head_v_dim;

        let norm_weight = weights.norm.as_ref()
            .and_then(|w| cache.get_bf16(&w.name))
            .ok_or_else(|| anyhow::anyhow!("GDN norm weight not in cache"))?;

        // Note: norm_weight is a weight, not an intermediate — skipped for probing


        let mut norm_out = stream.alloc_zeros::<bf16>(n_rows * norm_dim)?;
        probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.z_gate", &z_gate_raw, &[seq_len, z_dim], "prefill");

        unsafe {
            oxide.launch_rms_norm_gated_bf16(
                stream, &gdn_output, &z_gate_raw, norm_weight, &mut norm_out,
                n_rows as u32, norm_dim as u32, 1e-6f32,
            ).map_err(|e| anyhow::anyhow!("RMSNormGated kernel launch failed: {e}"))?;
        }
      norm_out
    } else {
        gdn_output.try_clone()
            .map_err(|e| anyhow::anyhow!("Failed to clone GDN output: {e}"))?
    };
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.norm_output", &norm_output, &[seq_len, num_v_heads * head_v_dim], "prefill");

    // =========================================================================
    // Phase 9: Output projection — [seq_len, value_dim] → [seq_len, hidden_size]
    // =========================================================================
    let mut output = stream.alloc_zeros::<bf16>(seq_len * hidden_size)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream, cache,
        &weights.out_proj_weight.name, &norm_output, &mut output,
        seq_len, hidden_size, value_dim, group_size,
    )?;
   probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.output", &output, &[seq_len, hidden_size], "prefill");

    Ok(output)
}

// ──────────────────────────────────────────────
// Decode forward pass (single token)
// ──────────────────────────────────────────────

/// GDN decode forward with the correct Gated Delta Rule (single token).
#[allow(unused_assignments, clippy::too_many_arguments)]
pub fn decode_forward(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    weights: &GdnWeights,
    input: &CudaSlice<bf16>,
    gdn_state: &mut GdnState,
    hidden_size: usize,
    config: &ModelConfig,
    group_size: usize,
    cache: &crate::gpu_cache::GpuWeightCache,
    layer_idx: usize,
    gpu_idx: usize,
    probe: &ProbeConfig,
) -> Result<CudaSlice<bf16>> {
    let seq_len = 1usize;
    // Decoding shares the same probe infrastructure as forward().
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.hidden_input", input, &[1, hidden_size], "decode");
    // Probe config controls whether intermediates are dumped for this path too.

    // Compute sharded dimensions from actual weight shapes (TP-aware)
    let num_v_heads = weight_output_dim(&weights.in_proj_b);  // Per-GPU: e.g. 24 at TP=2, 48 at TP=1
    let kv_ratio = config.linear_num_value_heads / config.linear_num_key_heads;  // 3 (model-level constant)
    let num_k_heads = num_v_heads / kv_ratio;                  // Per-GPU: e.g. 8 at TP=2, 16 at TP=1
    let head_k_dim = config.linear_key_head_dim;               // 128
    let head_v_dim = config.linear_value_head_dim;             // 128
    let key_dim = num_k_heads * head_k_dim;                    // Per-GPU: e.g. 1024 at TP=2
    let value_dim = num_v_heads * head_v_dim;                  // Per-GPU: e.g. 3072 at TP=2
    let conv_dim = key_dim * 2 + value_dim;                    // Matches per-GPU dimensions

    // =========================================================================
    // Phase 1: in_proj_qkv
    // =========================================================================
    let mut mixed_qkv = stream.alloc_zeros::<bf16>(conv_dim)?;
    if let Some(ref qkv_weight) = weights.in_proj_qkv {
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, oxide, stream, cache,
            &qkv_weight.name, input, &mut mixed_qkv,
            1, conv_dim, hidden_size, group_size,
        )?;
    }
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.mixed_qkv", &mixed_qkv, &[1, conv_dim], "decode");

    // =========================================================================
    // Phase 2: Conv1d with causal conv_state (matches HF causal_conv1d_update)
    //
    // HF decode path:
    //   hidden_states_new = cat([conv_state, hidden_states])  // prepend buffered context
    //   conv_state = hidden_states_new[:, :, -state_len:]     // update buffer
    //   out = conv1d(hidden_states_new, padding=0)            // no padding needed
    //   out = silu(out[:, :, -seq_len:])                      // take last position
    //
    // We replicate this by prepending conv_state to mixed_qkv, calling our
    // depthwise conv1d kernel with seq_len=kernel_size, and taking the last row.
    // =========================================================================
    let kernel_size = config.linear_conv_kernel_dim as i32;
    let kernel_size_usize = kernel_size as usize;
    let conv_state_len = kernel_size_usize - 1;  // 3 for kernel_size=4

    gdn_state.ensure_conv_state_allocated(stream, conv_dim, kernel_size_usize)?;

    // Build the conv input: [conv_state | mixed_qkv] → [kernel_size * conv_dim]
    let mut conv_input = stream.alloc_zeros::<bf16>(kernel_size_usize * conv_dim)?;
    if let Some(ref cs) = gdn_state.conv_state {
        // Copy conv_state (conv_state_len * conv_dim elements) to the start
        stream.memcpy_dtod(cs, &mut conv_input)
            .map_err(|e| anyhow::anyhow!("Failed to copy conv state to input: {e}"))?;
    }
    // Copy current token's mixed_qkv to the end (conv_state_len * conv_dim offset)
    let mixed_qkv_offset = conv_state_len * conv_dim;
    {
        let mut dst = conv_input.slice_mut(mixed_qkv_offset..kernel_size_usize * conv_dim);
        stream.memcpy_dtod(&mixed_qkv, &mut dst)
            .map_err(|e| anyhow::anyhow!("Failed to copy mixed_qkv to conv input: {e}"))?;
    }

    // Update conv_state: shift left by conv_dim, append current mixed_qkv
    // conv_state = cat([conv_state[:, :, -state_len+1:], hidden_states], dim=-1)
    // = shift left by conv_dim and put mixed_qkv at the end
    {
        let src_offset = conv_dim;  // skip first conv_dim elements (shift left)
        let src_view = conv_input.slice(src_offset..kernel_size_usize * conv_dim);
        stream.memcpy_dtod(&src_view, gdn_state.conv_state.as_mut().unwrap())
            .map_err(|e| anyhow::anyhow!("Failed to update conv state: {e}"))?;
    }

    // Launch conv1d with seq_len = kernel_size (instead of 1)
    let mut conv_out = stream.alloc_zeros::<bf16>(kernel_size_usize * conv_dim)?;
    let conv1d_gpu = cache.get_bf16(&weights.conv1d_weight.name)
        .ok_or_else(|| anyhow::anyhow!("conv1d weight not in cache"))?;
    oxide.launch_conv1d_depthwise_silu_bf16(
        stream, &conv_input, conv1d_gpu, &mut conv_out,
        1, conv_dim as u32, kernel_size_usize as u32, kernel_size as u32,
    ).map_err(|e| anyhow::anyhow!("Decode conv1d kernel launch failed: {e}"))?;

    // Extract the last position's output (the conv result for the current token)
    // conv_out is [kernel_size, conv_dim], take the last row
    let conv_out_last = {
        let last_row_offset = conv_state_len * conv_dim;
        let src_view = conv_out.slice(last_row_offset..kernel_size_usize * conv_dim);
        let mut dst = stream.alloc_zeros::<bf16>(conv_dim)?;
        stream.memcpy_dtod(&src_view, &mut dst)?;
        dst
    };
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.conv_out", &conv_out_last, &[1, conv_dim], "decode");

    // =========================================================================
    // Phase 3: Split
    // =========================================================================
    let query_flat = clone_view_to_slice(stream, &conv_out_last, 0..key_dim)?;
    let key_flat = clone_view_to_slice(stream, &conv_out_last, key_dim..2 * key_dim)?;
    let value_flat = clone_view_to_slice(stream, &conv_out_last, 2 * key_dim..2 * key_dim + value_dim)?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.query", &query_flat, &[1, key_dim], "decode");
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.key", &key_flat, &[1, key_dim], "decode");
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.value", &value_flat, &[1, value_dim], "decode");

    // =========================================================================
    // Phase 4: repeat_interleave q/k
    // =========================================================================
    let query_expanded = if kv_ratio > 1 {
        repeat_interleave_heads(stream, &query_flat, 1, num_k_heads, num_v_heads, head_k_dim)?
    } else {
        query_flat
    };
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.query_expanded", &query_expanded, &[1, num_v_heads * head_k_dim], "decode");
    let key_expanded = if kv_ratio > 1 {
        repeat_interleave_heads(stream, &key_flat, 1, num_k_heads, num_v_heads, head_k_dim)?
    } else {
        key_flat
    };
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.key_expanded", &key_expanded, &[1, num_v_heads * head_k_dim], "decode");

    // =========================================================================
    // Phase 5: in_proj_a, in_proj_b
    // =========================================================================
    let mut a_proj = stream.alloc_zeros::<bf16>(num_v_heads)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream, cache,
        &weights.in_proj_a.name, input, &mut a_proj,
        1, num_v_heads, hidden_size, group_size,
    )?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.a_proj", &a_proj, &[1, num_v_heads], "decode");

    let b_dim = weight_output_dim(&weights.in_proj_b);
    let mut b_proj_raw = stream.alloc_zeros::<bf16>(b_dim)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream, cache,
        &weights.in_proj_b.name, input, &mut b_proj_raw,
        1, b_dim, hidden_size, group_size,
    )?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.b_proj", &b_proj_raw, &[1, b_dim], "decode");
    // Use b_proj_raw directly (extraction not needed since b_dim == num_v_heads for BF16 weight)
    let b_proj = b_proj_raw;

    // =========================================================================
    // Phase 6: A_log, dt_bias as float32
    // =========================================================================
    let a_log_f32 = if let Some(ref w) = weights.a_log {
        cache.get_f32(&w.name)
            .ok_or_else(|| anyhow::anyhow!("a_log not in f32 cache: {}", w.name))?
            .clone()
    } else {
        stream.alloc_zeros::<f32>(num_v_heads)?
    };

    let dt_bias_f32 = if let Some(ref w) = weights.dt_bias {
        cache.get_f32(&w.name)
            .ok_or_else(|| anyhow::anyhow!("dt_bias not in f32 cache: {}", w.name))?
            .clone()
    } else {
        stream.alloc_zeros::<f32>(num_v_heads)?
    };

    // =========================================================================
    // Phase 7: Gated delta update kernel (bf16 inputs, fp32 state)
    // =========================================================================
    gdn_state.ensure_allocated(stream, num_v_heads, head_k_dim, head_v_dim)?;
    let mut gdn_output = stream.alloc_zeros::<bf16>(num_v_heads * head_v_dim)?;
    let state_ref = gdn_state.state.as_mut()
        .ok_or_else(|| anyhow::anyhow!("GDN state not allocated"))?;

    let num_v_heads_i32 = num_v_heads as i32;
    let head_k_dim_i32 = head_k_dim as i32;
    let head_v_dim_i32 = head_v_dim as i32;
  oxide.launch_gdn_recurrent_step_bf16(
        stream, &query_expanded, &key_expanded, &value_flat,
        &a_proj, &b_proj, &a_log_f32, &dt_bias_f32,
        state_ref, &mut gdn_output,
        num_v_heads_i32 as u32, head_k_dim_i32 as u32, head_v_dim_i32 as u32,
    ).map_err(|e| anyhow::anyhow!("GDN recurrent step kernel launch failed: {e}"))?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.core_attn_out", &gdn_output, &[1, num_v_heads * head_v_dim], "decode");

    // =========================================================================
    // Phase 8: RMSNormGated
    // =========================================================================
    let norm_output = if let Some(ref z_weight) = weights.in_proj_z {
        let z_dim = weight_output_dim(z_weight);
        let mut z_gate_raw = stream.alloc_zeros::<bf16>(z_dim)?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, oxide, stream, cache,
            &z_weight.name, input, &mut z_gate_raw,
            1, z_dim, hidden_size, group_size,
        )?;
        probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.z_gate", &z_gate_raw, &[1, z_dim], "decode");

        let n_rows = num_v_heads;
        let norm_dim = head_v_dim;

        let norm_weight = weights.norm.as_ref()
            .and_then(|w| cache.get_bf16(&w.name))
            .ok_or_else(|| anyhow::anyhow!("GDN norm weight not in cache"))?;

        let mut norm_out = stream.alloc_zeros::<bf16>(n_rows * norm_dim)?;
        unsafe {
            oxide.launch_rms_norm_gated_bf16(
                stream, &gdn_output, &z_gate_raw, norm_weight, &mut norm_out,
                n_rows as u32, norm_dim as u32, 1e-6f32,
            ).map_err(|e| anyhow::anyhow!("Decode RMSNormGated kernel launch failed: {e}"))?;
        }
        norm_out
    } else {
        gdn_output.try_clone()
            .map_err(|e| anyhow::anyhow!("Failed to clone decode GDN output: {e}"))?
    };
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.norm_output", &norm_output, &[1, num_v_heads * head_v_dim], "decode");

    // =========================================================================
    // Phase 9: Output projection → hidden_size
    // =========================================================================
    let mut output = stream.alloc_zeros::<bf16>(hidden_size)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, oxide, stream, cache,
        &weights.out_proj_weight.name, &norm_output, &mut output,
        1, hidden_size, value_dim, group_size,
    )?;
    probe::dump(stream, probe, layer_idx, gpu_idx, "gdn.output", &output, &[1, hidden_size], "decode");

    Ok(output)
}
