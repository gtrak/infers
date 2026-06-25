use super::*;

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
    gemm: &GemmEngine,
    stream: &Arc<CudaStream>,
    oxide: &OxideKernels,
    weights: &AttentionWeights,
    input: &CudaSlice<bf16>,
    paged_cache: &mut PagedKvCache,
    block_table_gpu: &CudaSlice<i32>,
    positions_gpu: &CudaSlice<i32>,
    position: u32,
    // num_cached_tokens removed — now read from cached_tokens_count device buffer
    cached_tokens_count: &CudaSlice<u32>,  // device buffer with 1 element
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
    rope_position_staging: &mut CudaSlice<i32>, // pre-allocated staging buffer for RoPE positions (zero-alloc)
    position_i32: &[i32],                      // host-side position as i32
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
        // Use staging buffer for RoPE positions when cached tables available (zero-alloc)
        match (cached_cos, cached_sin) {
            (Some(cos), Some(sin)) => {
                rope::apply_rope_with_staging(
                    stream, oxide, &mut ws.q_dummy, &mut ws.k_norm_out,
                    position_i32, rope_position_staging,
                    num_kv_heads as i32, head_dim, cos, sin, partial_rotary_factor,
                )?;
            }
            _ => {
                rope::apply_rope(
                    stream, oxide, &mut ws.q_dummy, &mut ws.k_norm_out,
                    &[position], num_kv_heads as i32, head_dim, rope_theta, partial_rotary_factor,
                    cached_cos, cached_sin,
                )?;
            }
        }
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_after_rope", &ws.k_norm_out, &[1, kv_dim], "decode");
    } else {
        probe::dump(stream, probe, layer_idx, gpu_idx, "attn.k_before_rope", &ws.k_single, &[1, kv_dim], "decode");
        match (cached_cos, cached_sin) {
            (Some(cos), Some(sin)) => {
                rope::apply_rope_with_staging(
                    stream, oxide, &mut ws.q_dummy, &mut ws.k_single,
                    position_i32, rope_position_staging,
                    num_kv_heads as i32, head_dim, cos, sin, partial_rotary_factor,
                )?;
            }
            _ => {
                rope::apply_rope(
                    stream, oxide, &mut ws.q_dummy, &mut ws.k_single,
                    &[position], num_kv_heads as i32, head_dim, rope_theta, partial_rotary_factor,
                    cached_cos, cached_sin,
                )?;
            }
        }
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
            stream, &oxide.cc_stream(), &ws.q_full, &mut ws.q_buf, &mut ws.gate_buf,
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

        // Apply RoPE to Q — inside branch for mutable access (zero-alloc via staging)
        match (cached_cos, cached_sin) {
            (Some(cos), Some(sin)) => {
                if q_norm_exists {
                    rope::apply_rope_with_staging(
                        stream, oxide, &mut ws.q_norm_out, &mut ws.k_rope_dummy,
                        position_i32, rope_position_staging,
                        num_heads as i32, head_dim, cos, sin, partial_rotary_factor,
                    )?;
                } else {
                    rope::apply_rope_with_staging(
                        stream, oxide, &mut ws.q_buf, &mut ws.k_rope_dummy,
                        position_i32, rope_position_staging,
                        num_heads as i32, head_dim, cos, sin, partial_rotary_factor,
                    )?;
                }
            }
            _ => {
                if q_norm_exists {
                    rope::apply_rope(
                        stream, oxide, &mut ws.q_norm_out, &mut ws.k_rope_dummy,
                        &[position], num_heads as i32, head_dim, rope_theta, partial_rotary_factor,
                        cached_cos, cached_sin,
                    )?;
                } else {
                    rope::apply_rope(
                        stream, oxide, &mut ws.q_buf, &mut ws.k_rope_dummy,
                        &[position], num_heads as i32, head_dim, rope_theta, partial_rotary_factor,
                        cached_cos, cached_sin,
                    )?;
                }
            }
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
                cached_tokens_count,
                &mut ws.attn_output,
                num_pages,
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
            stream, &oxide.cc_stream(), &ws.attn_output, &ws.gate_buf, &mut ws.gated, per_gpu_head_dim as u32,
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

        // Apply RoPE to Q — inside branch for mutable access (zero-alloc via staging)
        match (cached_cos, cached_sin) {
            (Some(cos), Some(sin)) => {
                if q_norm_exists {
                    rope::apply_rope_with_staging(
                        stream, oxide, &mut ws.q_norm_out, &mut ws.k_rope_dummy,
                        position_i32, rope_position_staging,
                        num_heads as i32, head_dim, cos, sin, partial_rotary_factor,
                    )?;
                } else {
                    rope::apply_rope_with_staging(
                        stream, oxide, &mut ws.q_full, &mut ws.k_rope_dummy,
                        position_i32, rope_position_staging,
                        num_heads as i32, head_dim, cos, sin, partial_rotary_factor,
                    )?;
                }
            }
            _ => {
                if q_norm_exists {
                    rope::apply_rope(
                        stream, oxide, &mut ws.q_norm_out, &mut ws.k_rope_dummy,
                        &[position], num_heads as i32, head_dim, rope_theta, partial_rotary_factor,
                        cached_cos, cached_sin,
                    )?;
                } else {
                    rope::apply_rope(
                        stream, oxide, &mut ws.q_full, &mut ws.k_rope_dummy,
                        &[position], num_heads as i32, head_dim, rope_theta, partial_rotary_factor,
                        cached_cos, cached_sin,
                    )?;
                }
            }
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
                cached_tokens_count,
                &mut ws.attn_output,
                num_pages,
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
