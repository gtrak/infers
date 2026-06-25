use super::*;

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
            stream, &oxide.cc_stream(), &scores_h, &mut softmax_out_h, seq_len as u32, 1u32,
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
            stream, &oxide.cc_stream(), &attn_combined, gate_heads, &mut gated, attn_combined_size as u32,
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
