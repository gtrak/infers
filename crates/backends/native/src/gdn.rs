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
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use infers_cuda::gemm::GemmEngine;
use infers_model::{GdnWeights, ModelConfig, WeightDtype};

// DEBUG: download a tensor and print statistics (min, max, mean_abs, stddev, first 5).
fn debug_tensor_stats_bf16(
    stream: &Arc<CudaStream>,
    buf: &CudaSlice<bf16>,
    n: usize,
    label: &str,
) {
    let cpu: Vec<bf16> = stream.clone_dtoh(buf).expect("download failed");
    let mut f_sum_abs = 0.0f64;
    let mut f_min = f64::MAX;
    let mut f_max = f64::MIN;
    let mut f_nan = 0usize;
    let mut f_inf = 0usize;
    for i in 0..n.min(cpu.len()) {
        let f = cpu[i].to_f32() as f64;
        if f.is_nan() { f_nan += 1; continue; }
        if f.is_infinite() { f_inf += 1; continue; }
        f_sum_abs += f.abs();
        if f < f_min { f_min = f; }
        if f > f_max { f_max = f; }
    }
    let valid = n.min(cpu.len()) - f_nan - f_inf;
    let mean_abs = if valid > 0 { f_sum_abs / valid as f64 } else { 0.0 };
    // Compute stddev of abs values (rough)
    let mut f_sum_sq = 0.0f64;
    for i in 0..n.min(cpu.len()) {
        let f = cpu[i].to_f32() as f64;
        if f.is_nan() || f.is_infinite() { continue; }
        let d = f.abs() - mean_abs;
        f_sum_sq += d * d;
    }
    let stddev = if valid > 1 { (f_sum_sq / (valid - 1) as f64).sqrt() } else { 0.0 };
    let samples: Vec<String> = cpu[..n.min(5)].iter()
        .map(|v| format!("{:.6}", v.to_f32())).collect();
    eprintln!("DEBUG {label}[{n}]: min={f_min:.6} max={f_max:.6} mean_abs={mean_abs:.6} std={stddev:.6} nan={f_nan} inf={f_inf} first5=[{}]", samples.join(", "));
    // Don't panic on NaN/Inf for debugging
    // if f_nan > 0 { panic!("DEBUG NAN FOUND: {label} has {f_nan} NaN values"); }
    // if f_inf > 0 { panic!("DEBUG INF FOUND: {label} has {f_inf} Inf values"); }
}

#[allow(dead_code)]
fn debug_tensor_stats_f32(
    stream: &Arc<CudaStream>,
    buf: &CudaSlice<f32>,
    n: usize,
    label: &str,
) {
    let cpu: Vec<f32> = stream.clone_dtoh(buf).expect("download failed");
    let mut f_sum_abs = 0.0f64;
    let mut f_min = f64::MAX;
    let mut f_max = f64::MIN;
    let mut f_nan = 0usize;
    let mut f_inf = 0usize;
    for i in 0..n.min(cpu.len()) {
        let f = cpu[i] as f64;
        if f.is_nan() { f_nan += 1; continue; }
        if f.is_infinite() { f_inf += 1; continue; }
        f_sum_abs += f.abs();
        if f < f_min { f_min = f; }
        if f > f_max { f_max = f; }
    }
    let valid = n.min(cpu.len()) - f_nan - f_inf;
    let mean_abs = if valid > 0 { f_sum_abs / valid as f64 } else { 0.0 };
    let samples: Vec<String> = cpu[..n.min(5)].iter()
        .map(|v| format!("{:.6}", v)).collect();
    eprintln!("DEBUG {label}[{n}]: min={f_min:.6} max={f_max:.6} mean_abs={mean_abs:.6} nan={f_nan} inf={f_inf} first5=[{}]", samples.join(", "));
    // Don't panic on NaN/Inf for debugging
    // if f_nan > 0 { panic!("DEBUG NAN FOUND: {label} has {f_nan} NaN values"); }
    // if f_inf > 0 { panic!("DEBUG INF FOUND: {label} has {f_inf} Inf values"); }
}

/// Get the output dimension from a weight tensor.
fn weight_output_dim(w: &infers_model::WeightData) -> usize {
    if w.dtype == WeightDtype::Int4Packed { w.shape[1] } else { w.shape[0] }
}

/// GDN recurrent state — 2D matrix [num_heads × head_k_dim × head_v_dim] float32.
#[derive(Debug)]
pub struct GdnState {
    pub state: Option<CudaSlice<f32>>,
    pub num_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
}

impl Default for GdnState {
    fn default() -> Self { Self::new() }
}

impl GdnState {
    pub fn new() -> Self {
        Self { state: None, num_heads: 0, head_k_dim: 0, head_v_dim: 0 }
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

fn bf16_to_f32_gpu(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<bf16>,
    count: usize,
) -> Result<CudaSlice<f32>> {
    let cpu_bf16: Vec<bf16> = stream.clone_dtoh(src)
        .map_err(|e| anyhow::anyhow!("Failed to download BF16 buffer: {e}"))?;
    let cpu_f32: Vec<f32> = cpu_bf16.iter().map(|v| v.to_f32()).collect();
    let dst = stream.clone_htod(&cpu_f32)
        .map_err(|e| anyhow::anyhow!("Failed to upload f32 buffer: {e}"))?;
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
/// * `int4_kernel` — INT4 GEMM kernel
/// * `stream` — CUDA stream
/// * `gdn_prefill_kernel` — `infers_gdn_gated_delta_prefill_bf16`
/// * `conv1d_kernel` — `infers_conv1d_depthwise_silu_bf16`
/// * `rms_norm_gated_kernel` — `infers_rms_norm_gated_bf16`
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
    int4_kernel: &CudaFunction,
    stream: &Arc<CudaStream>,
    gdn_prefill_kernel: &CudaFunction,
    conv1d_kernel: &CudaFunction,
    rms_norm_gated_kernel: &CudaFunction,
    weights: &GdnWeights,
    input: &CudaSlice<bf16>,
    gdn_state: &mut GdnState,
    hidden_size: usize,
    config: &ModelConfig,
    group_size: usize,
    cache: &crate::gpu_cache::GpuWeightCache,
) -> Result<CudaSlice<bf16>> {
    let seq_len = input.len() / hidden_size;
    debug_tensor_stats_bf16(stream, input, seq_len * hidden_size, "gdn_input");

    let num_k_heads = config.linear_num_key_heads;       // 16
    let num_v_heads = config.linear_num_value_heads;     // 48
    let head_k_dim = config.linear_key_head_dim;         // 128
    let head_v_dim = config.linear_value_head_dim;       // 128
    let key_dim = num_k_heads * head_k_dim;               // 2048
    let value_dim = num_v_heads * head_v_dim;             // 6144
    let conv_dim = key_dim * 2 + value_dim;               // 10240
    let kv_ratio = num_v_heads / num_k_heads;              // 3

    // =========================================================================
    // Phase 1: in_proj_qkv projection (if available)
    //   mixed_qkv = input @ in_proj_qkv^T  [seq_len, conv_dim]
    // =========================================================================
    let mut mixed_qkv = stream.alloc_zeros::<bf16>(seq_len * conv_dim)?;
    if let Some(ref qkv_weight) = weights.in_proj_qkv {
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, int4_kernel, stream, cache,
            &qkv_weight.name, input, &mut mixed_qkv,
            seq_len, conv_dim, hidden_size, group_size,
        )?;
    }
    debug_tensor_stats_bf16(stream, &mixed_qkv, seq_len * conv_dim, "mixed_qkv");

    // =========================================================================
    // Phase 2: Depthwise conv1d on mixed_qkv (SiLU activation)
    // =========================================================================
    let mut conv_out = stream.alloc_zeros::<bf16>(seq_len * conv_dim)?;
    // conv1d_weight is always present for Qwen3.6
    let conv1d_gpu = cache.get_bf16(&weights.conv1d_weight.name)
        .ok_or_else(|| anyhow::anyhow!("conv1d weight '{}' not in cache", weights.conv1d_weight.name))?;
    let batch_i32 = 1i32;
    let conv_dim_i32 = conv_dim as i32;
    let seq_len_i32 = seq_len as i32;
    let kernel_size = config.linear_conv_kernel_dim as i32;
    let total = seq_len * conv_dim;
    let grid = (total as u32).div_ceil(256);

    unsafe {
        stream.launch_builder(conv1d_kernel)
            .arg(&mixed_qkv)
            .arg(conv1d_gpu)
            .arg(&mut conv_out)
            .arg(&batch_i32)
            .arg(&conv_dim_i32)
            .arg(&seq_len_i32)
            .arg(&kernel_size)
            .launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| anyhow::anyhow!("conv1d kernel launch failed: {e}"))?;
    }
    debug_tensor_stats_bf16(stream, &conv_out, seq_len * conv_dim, "conv_out");

    // =========================================================================
    // Phase 3: Split conv_out into query, key, value (extract as proper slices)
    //
    // conv_out layout: [seq_len, conv_dim]
    //   query = conv_out[..., :key_dim]        [seq_len, key_dim]
    //   key   = conv_out[..., key_dim:2*key_dim] [seq_len, key_dim]
    //   value = conv_out[..., 2*key_dim:]      [seq_len, value_dim]
    // =========================================================================
    // Extract slices by copying from conv_out views into new buffers
    let query_flat = clone_view_to_slice(stream, &conv_out, 0..seq_len * key_dim)?;
    let key_flat = clone_view_to_slice(stream, &conv_out, seq_len * key_dim..seq_len * 2 * key_dim)?;
    let value_flat = clone_view_to_slice(stream, &conv_out, seq_len * 2 * key_dim..seq_len * (2 * key_dim + value_dim))?;
    debug_tensor_stats_bf16(stream, &query_flat, seq_len * key_dim, "query_flat");
    debug_tensor_stats_bf16(stream, &key_flat, seq_len * key_dim, "key_flat");
    debug_tensor_stats_bf16(stream, &value_flat, seq_len * value_dim, "value_flat");

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
        gemm, int4_kernel, stream, cache,
        &weights.in_proj_a.name, input, &mut a_proj,
        seq_len, num_v_heads, hidden_size, group_size,
    )?;
    debug_tensor_stats_bf16(stream, &a_proj, seq_len * num_v_heads, "a_proj");

    let b_dim = weight_output_dim(&weights.in_proj_b);
    let mut b_proj_raw = stream.alloc_zeros::<bf16>(seq_len * b_dim)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream, cache,
        &weights.in_proj_b.name, input, &mut b_proj_raw,
        seq_len, b_dim, hidden_size, group_size,
    )?;
    debug_tensor_stats_bf16(stream, &b_proj_raw, seq_len * b_dim, "b_proj_raw");
    // Use b_proj_raw directly (extraction not needed since b_dim == num_v_heads)
    let b_proj = b_proj_raw;

    // =========================================================================
    // Phase 6: Upload A_log and dt_bias as float32
    // =========================================================================
    let a_log_f32 = if let Some(ref w) = weights.a_log {
        let gpu_bf16 = cache.get_bf16(&w.name)
            .ok_or_else(|| anyhow::anyhow!("A_log not in cache"))?;
        bf16_to_f32_gpu(stream, gpu_bf16, num_v_heads)?
    } else {
        stream.alloc_zeros::<f32>(num_v_heads)?
    };

    let dt_bias_f32 = if let Some(ref w) = weights.dt_bias {
        let gpu_bf16 = cache.get_bf16(&w.name)
            .ok_or_else(|| anyhow::anyhow!("dt_bias not in cache"))?;
        bf16_to_f32_gpu(stream, gpu_bf16, num_v_heads)?
    } else {
        stream.alloc_zeros::<f32>(num_v_heads)?
    };
    debug_tensor_stats_f32(stream, &a_log_f32, num_v_heads, "a_log_f32");
    debug_tensor_stats_f32(stream, &dt_bias_f32, num_v_heads, "dt_bias_f32");

    // =========================================================================
    // Phase 7: Allocate state and launch gated delta prefill kernel
    // =========================================================================
    gdn_state.ensure_allocated(stream, num_v_heads, head_k_dim, head_v_dim)?;
    let mut gdn_output = stream.alloc_zeros::<bf16>(seq_len * num_v_heads * head_v_dim)?;
    let state_ref = gdn_state.state.as_mut()
        .ok_or_else(|| anyhow::anyhow!("GDN state not allocated"))?;

    let seq_len_i32 = seq_len as i32;
    let num_v_heads_i32 = num_v_heads as i32;
    let head_k_dim_i32 = head_k_dim as i32;
    let head_v_dim_i32 = head_v_dim as i32;

    let total_threads = num_v_heads * head_v_dim;

    unsafe {
        stream.launch_builder(gdn_prefill_kernel)
            .arg(&query_expanded)   // [S, H, K]
            .arg(&key_expanded)     // [S, H, K]
            .arg(&value_flat)       // [S, H, V]
            .arg(&a_proj)           // [S, H]
            .arg(&b_proj)           // [S, H]
            .arg(&a_log_f32)        // [H] float32
            .arg(&dt_bias_f32)      // [H] float32
            .arg(state_ref)         // [H, K, V] float32 mutable
            .arg(&mut gdn_output)   // [S, H, V]
            .arg(&seq_len_i32)
            .arg(&num_v_heads_i32)
            .arg(&head_k_dim_i32)
            .arg(&head_v_dim_i32)
            .launch(LaunchConfig {
                grid_dim: ((total_threads as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| anyhow::anyhow!("GDN gated delta prefill kernel launch failed: {e}"))?;
    }
    debug_tensor_stats_bf16(stream, &gdn_output, seq_len * num_v_heads * head_v_dim, "gdn_output");

    // =========================================================================
    // Phase 8: RMSNormGated — norm(gdn_output, z_gate, weight)
    // =========================================================================
    let norm_output = if let Some(ref z_weight) = weights.in_proj_z {
        let z_dim = weight_output_dim(z_weight);
        let mut z_gate_raw = stream.alloc_zeros::<bf16>(seq_len * z_dim)?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, int4_kernel, stream, cache,
            &z_weight.name, input, &mut z_gate_raw,
            seq_len, z_dim, hidden_size, group_size,
        )?;

        let n_rows = seq_len * num_v_heads;
        let norm_dim = head_v_dim;

        let norm_weight = weights.norm.as_ref()
            .and_then(|w| cache.get_bf16(&w.name))
            .ok_or_else(|| anyhow::anyhow!("GDN norm weight not in cache"))?;

        let mut norm_out = stream.alloc_zeros::<bf16>(n_rows * norm_dim)?;
        debug_tensor_stats_bf16(stream, &z_gate_raw, seq_len * z_dim, "z_gate_raw");

        unsafe {
            stream.launch_builder(rms_norm_gated_kernel)
                .arg(&gdn_output)
                .arg(&z_gate_raw)
                .arg(norm_weight)
                .arg(&mut norm_out)
                .arg(&(n_rows as i32))
                .arg(&(norm_dim as i32))
                .arg(&1e-6f32)
                .launch(LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (norm_dim.min(256) as u32, 1, 1),
                    shared_mem_bytes: (norm_dim.min(256) * 4) as u32,
                })
                .map_err(|e| anyhow::anyhow!("RMSNormGated kernel launch failed: {e}"))?;
        }
        norm_out
    } else {
        gdn_output.try_clone()
            .map_err(|e| anyhow::anyhow!("Failed to clone GDN output: {e}"))?
    };
    debug_tensor_stats_bf16(stream, &norm_output, seq_len * num_v_heads * head_v_dim, "norm_output");

    // =========================================================================
    // Phase 9: Output projection — [seq_len, value_dim] → [seq_len, hidden_size]
    // =========================================================================
    // DEBUG: Make a fresh copy of norm_output to avoid any aliasing issues
    let gemm_input = norm_output.try_clone()
        .map_err(|e| anyhow::anyhow!("Failed to clone norm_output for GEMM: {e}"))?;
    // Synchronize by downloading a sync marker
    let _sync: Vec<bf16> = stream.clone_dtoh(&gemm_input)
        .map_err(|e| anyhow::anyhow!("Sync download failed: {e}"))?;
    let mut output = stream.alloc_zeros::<bf16>(seq_len * hidden_size)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream, cache,
        &weights.out_proj_weight.name, &gemm_input, &mut output,
        seq_len, hidden_size, value_dim, group_size,
    )?;
    debug_tensor_stats_bf16(stream, &output, seq_len * hidden_size, "output");

    Ok(output)
}

// ──────────────────────────────────────────────
// Decode forward pass (single token)
// ──────────────────────────────────────────────

/// GDN decode forward with the correct Gated Delta Rule (single token).
#[allow(unused_assignments, clippy::too_many_arguments)]
pub fn decode_forward(
    gemm: &mut GemmEngine,
    int4_kernel: &CudaFunction,
    stream: &Arc<CudaStream>,
    gdn_update_kernel: &CudaFunction,
    conv1d_kernel: &CudaFunction,
    rms_norm_gated_kernel: &CudaFunction,
    weights: &GdnWeights,
    input: &CudaSlice<bf16>,
    gdn_state: &mut GdnState,
    hidden_size: usize,
    config: &ModelConfig,
    group_size: usize,
    cache: &crate::gpu_cache::GpuWeightCache,
) -> Result<CudaSlice<bf16>> {
    let seq_len = 1usize;

    let num_k_heads = config.linear_num_key_heads;
    let num_v_heads = config.linear_num_value_heads;
    let head_k_dim = config.linear_key_head_dim;
    let head_v_dim = config.linear_value_head_dim;
    let key_dim = num_k_heads * head_k_dim;
    let value_dim = num_v_heads * head_v_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let kv_ratio = num_v_heads / num_k_heads;

    // =========================================================================
    // Phase 1: in_proj_qkv
    // =========================================================================
    let mut mixed_qkv = stream.alloc_zeros::<bf16>(conv_dim)?;
    if let Some(ref qkv_weight) = weights.in_proj_qkv {
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, int4_kernel, stream, cache,
            &qkv_weight.name, input, &mut mixed_qkv,
            1, conv_dim, hidden_size, group_size,
        )?;
    }

    // =========================================================================
    // Phase 2: Conv1d
    // =========================================================================
    let mut conv_out = stream.alloc_zeros::<bf16>(conv_dim)?;
    let conv1d_gpu = cache.get_bf16(&weights.conv1d_weight.name)
        .ok_or_else(|| anyhow::anyhow!("conv1d weight not in cache"))?;
    let batch_i32 = 1i32;
    let conv_dim_i32 = conv_dim as i32;
    let seq_len_i32 = 1i32;
    let kernel_size = config.linear_conv_kernel_dim as i32;
    let grid = (conv_dim as u32).div_ceil(256);

    unsafe {
        stream.launch_builder(conv1d_kernel)
            .arg(&mixed_qkv)
            .arg(conv1d_gpu)
            .arg(&mut conv_out)
            .arg(&batch_i32)
            .arg(&conv_dim_i32)
            .arg(&seq_len_i32)
            .arg(&kernel_size)
            .launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| anyhow::anyhow!("Decode conv1d kernel launch failed: {e}"))?;
    }

    // =========================================================================
    // Phase 3: Split
    // =========================================================================
    let query_flat = clone_view_to_slice(stream, &conv_out, 0..key_dim)?;
    let key_flat = clone_view_to_slice(stream, &conv_out, key_dim..2 * key_dim)?;
    let value_flat = clone_view_to_slice(stream, &conv_out, 2 * key_dim..2 * key_dim + value_dim)?;

    // =========================================================================
    // Phase 4: repeat_interleave q/k
    // =========================================================================
    let query_expanded = if kv_ratio > 1 {
        repeat_interleave_heads(stream, &query_flat, 1, num_k_heads, num_v_heads, head_k_dim)?
    } else {
        query_flat
    };
    let key_expanded = if kv_ratio > 1 {
        repeat_interleave_heads(stream, &key_flat, 1, num_k_heads, num_v_heads, head_k_dim)?
    } else {
        key_flat
    };

    // =========================================================================
    // Phase 5: in_proj_a, in_proj_b
    // =========================================================================
    let mut a_proj = stream.alloc_zeros::<bf16>(num_v_heads)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream, cache,
        &weights.in_proj_a.name, input, &mut a_proj,
        1, num_v_heads, hidden_size, group_size,
    )?;

    let b_dim = weight_output_dim(&weights.in_proj_b);
    let mut b_proj_raw = stream.alloc_zeros::<bf16>(b_dim)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream, cache,
        &weights.in_proj_b.name, input, &mut b_proj_raw,
        1, b_dim, hidden_size, group_size,
    )?;
    // Use b_proj_raw directly (extraction not needed since b_dim == num_v_heads for BF16 weight)
    let b_proj = b_proj_raw;

    // =========================================================================
    // Phase 6: A_log, dt_bias as float32
    // =========================================================================
    let a_log_f32 = if let Some(ref w) = weights.a_log {
        let gpu_bf16 = cache.get_bf16(&w.name)
            .ok_or_else(|| anyhow::anyhow!("A_log not in cache"))?;
        bf16_to_f32_gpu(stream, gpu_bf16, num_v_heads)?
    } else {
        stream.alloc_zeros::<f32>(num_v_heads)?
    };

    let dt_bias_f32 = if let Some(ref w) = weights.dt_bias {
        let gpu_bf16 = cache.get_bf16(&w.name)
            .ok_or_else(|| anyhow::anyhow!("dt_bias not in cache"))?;
        bf16_to_f32_gpu(stream, gpu_bf16, num_v_heads)?
    } else {
        stream.alloc_zeros::<f32>(num_v_heads)?
    };

    // =========================================================================
    // Phase 7: Gated delta update kernel
    // =========================================================================
    gdn_state.ensure_allocated(stream, num_v_heads, head_k_dim, head_v_dim)?;
    let mut gdn_output = stream.alloc_zeros::<bf16>(num_v_heads * head_v_dim)?;
    let state_ref = gdn_state.state.as_mut()
        .ok_or_else(|| anyhow::anyhow!("GDN state not allocated"))?;

    let num_v_heads_i32 = num_v_heads as i32;
    let head_k_dim_i32 = head_k_dim as i32;
    let head_v_dim_i32 = head_v_dim as i32;
    // No shared memory needed — state lives in global memory
    let total_threads = num_v_heads * head_v_dim;

    unsafe {
        stream.launch_builder(gdn_update_kernel)
            .arg(&query_expanded)
            .arg(&key_expanded)
            .arg(&value_flat)
            .arg(&a_proj)
            .arg(&b_proj)
            .arg(&a_log_f32)
            .arg(&dt_bias_f32)
            .arg(state_ref)
            .arg(&mut gdn_output)
            .arg(&num_v_heads_i32)
            .arg(&head_k_dim_i32)
            .arg(&head_v_dim_i32)
            .launch(LaunchConfig {
                grid_dim: ((total_threads as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| anyhow::anyhow!("GDN gated delta update kernel launch failed: {e}"))?;
    }

    // =========================================================================
    // Phase 8: RMSNormGated
    // =========================================================================
    let norm_output = if let Some(ref z_weight) = weights.in_proj_z {
        let z_dim = weight_output_dim(z_weight);
        let mut z_gate_raw = stream.alloc_zeros::<bf16>(z_dim)?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, int4_kernel, stream, cache,
            &z_weight.name, input, &mut z_gate_raw,
            1, z_dim, hidden_size, group_size,
        )?;

        let n_rows = num_v_heads;
        let norm_dim = head_v_dim;

        let norm_weight = weights.norm.as_ref()
            .and_then(|w| cache.get_bf16(&w.name))
            .ok_or_else(|| anyhow::anyhow!("GDN norm weight not in cache"))?;

        let mut norm_out = stream.alloc_zeros::<bf16>(n_rows * norm_dim)?;
        unsafe {
            stream.launch_builder(rms_norm_gated_kernel)
                .arg(&gdn_output)
                .arg(&z_gate_raw)
                .arg(norm_weight)
                .arg(&mut norm_out)
                .arg(&(n_rows as i32))
                .arg(&(norm_dim as i32))
                .arg(&1e-6f32)
                .launch(LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (norm_dim.min(256) as u32, 1, 1),
                    shared_mem_bytes: (norm_dim.min(256) * 4) as u32,
                })
                .map_err(|e| anyhow::anyhow!("Decode RMSNormGated kernel launch failed: {e}"))?;
        }
        norm_out
    } else {
        gdn_output.try_clone()
            .map_err(|e| anyhow::anyhow!("Failed to clone decode GDN output: {e}"))?
    };

    // =========================================================================
    // Phase 9: Output projection → hidden_size
    // =========================================================================
    let mut output = stream.alloc_zeros::<bf16>(hidden_size)?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, int4_kernel, stream, cache,
        &weights.out_proj_weight.name, &norm_output, &mut output,
        1, hidden_size, value_dim, group_size,
    )?;

    Ok(output)
}
