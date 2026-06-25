use super::*;

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
