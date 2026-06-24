//! GDN chunked prefill kernel verification against sequential CPU reference.
//!
//! Generates deterministic BF16 inputs, runs `launch_gdn_chunked_gated_delta_prefill_bf16` on GPU,
//! and compares output with an f32 CPU sequential reference implementation of the same algorithm.
//! Also saves all intermediate data to `/tmp/gdn_prefill_test_inputs/` for external Python comparison.

use std::sync::Arc;
use std::fs;

use half::bf16;
use infers_cuda::OxideKernels;
use cudarc::driver::{CudaContext, CudaStream};

// ── Model weights (layer 0, first 4 heads) ────────────────────────────────
const A_LOG: [f32; 4] = [
    -3.203125, -2.640625, -2.718750, -4.656250,
];

const DT_BIAS: [f32; 4] = [
    -3.468750, 13.125000, -2.578125, -1.078125,
];

// ── Deterministic BF16 generator (LCG) ───────────────────────────────────
fn lcg_next(state: &mut u64) -> u64 {
    *state = (*state).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

/// Generate bf16 values from a fixed seed using LCG.
fn generate_bf16(seed: u64, count: usize) -> Vec<bf16> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            let raw = lcg_next(&mut state) as u32;
            let frac = (raw >> 8) as f32 / (1u32 << 24) as f32;
            let val = frac * 2.0 - 1.0;
            bf16::from_f32(val)
        })
        .collect()
}

// ── Helpers ──────────────────────────────────────────────────────────────
fn create_context() -> anyhow::Result<(Arc<CudaContext>, Arc<CudaStream>, OxideKernels)> {
    let ctx: Arc<CudaContext> = CudaContext::new(0)?;
    let stream: Arc<CudaStream> = ctx.new_stream()?;
    let kernels = OxideKernels::new(
        0,
        "/home/gary/dev/infers/crates/cuda/kernels/compiled/oxide_kernels.cubin",
    )?;
    Ok((ctx, stream, kernels))
}

/// Convert bf16 bits to f32 the same way as the kernel: f32::from_bits(bits << 16)
fn bf16_to_f32_kernel(val: u16) -> f32 {
    f32::from_bits((val as u32) << 16)
}

/// Write raw binary data to a file.
fn save_raw(filepath: &str, bytes: &[u8]) {
    fs::write(filepath, bytes).unwrap_or_else(|e| panic!("Failed to write {}: {}", filepath, e));
}

// ── CPU sequential reference implementation (f32) ────────────────────────
/// Memory layout matches the GPU kernel:
///   query: [seq_len, H, K] — query[t * H * K + h * K + d]
///   key:   [seq_len, H, K] — key[t * H * K + h * K + d]
///   value: [seq_len, H, V] — value[t * H * V + h * V + v]
///   a_proj: [seq_len, H]   — a_proj[t * H + h]
///   b_proj: [seq_len, H]   — b_proj[t * H + h]
///   state:  [H, K, V] f32  — state[h * K * V + k * V + v]
fn gdn_chunked_prefill_ref_f32(
    query: &[bf16],
    key: &[bf16],
    value: &[bf16],
    a_proj: &[bf16],
    b_proj: &[bf16],
    a_log: &[f32],
    dt_bias: &[f32],
    state: &mut Vec<f32>,
    seq_len: usize,
    H: usize,
    K: usize,
    V: usize,
) -> Vec<f32> {
    let eps = 1e-6f32;
    let output_size = seq_len * H * V;
    let mut output_f32 = vec![0.0f32; output_size];

    for t in 0..seq_len {
        for h in 0..H {
            let q_start = t * H * K + h * K;
            let k_start = t * H * K + h * K;
            let v_start = t * H * V + h * V;
            let state_h = h * K * V;

            // Per-head scalars
            let a_proj_val = bf16_to_f32_kernel(a_proj[t * H + h].to_bits());
            let b_proj_val = bf16_to_f32_kernel(b_proj[t * H + h].to_bits());

            let decay_rate = a_log[h].exp();
            let sp_val = a_proj_val + dt_bias[h];

            // Softplus with clamping (matches kernel)
            let softplus_val: f32;
            if sp_val > 20.0 {
                softplus_val = sp_val;
            } else if sp_val < -20.0 {
                softplus_val = 0.0;
            } else {
                softplus_val = (1.0f32 + sp_val.exp()).ln();
            }

            let g_val = -decay_rate * softplus_val;
            let decay = g_val.exp();
            let beta = 1.0f32 / (1.0f32 + (-b_proj_val).exp());

            // L2-normalize key and query
            let mut k_l2_sq = 0.0f32;
            let mut q_l2_sq = 0.0f32;
            for d in 0..K {
                let kv = bf16_to_f32_kernel(key[k_start + d].to_bits());
                let qv = bf16_to_f32_kernel(query[q_start + d].to_bits());
                k_l2_sq += kv * kv;
                q_l2_sq += qv * qv;
            }

            let k_rcp = 1.0f32 / (k_l2_sq + eps).sqrt();
            let rcp_sqrt_k = 1.0f32 / (K as f32).sqrt();
            let q_rcp = 1.0f32 / (q_l2_sq + eps).sqrt() * rcp_sqrt_k;

            // Step 1: State decay
            for k in 0..K {
                for v in 0..V {
                    state[state_h + k * V + v] *= decay;
                }
            }

            // Step 2: kv_mem[v] = sum_k(state[h,k,v] * k_normed[k])
            let mut kv_mem = vec![0.0f32; V];
            for k in 0..K {
                let k_val = bf16_to_f32_kernel(key[k_start + k].to_bits()) * k_rcp;
                for v in 0..V {
                    kv_mem[v] += state[state_h + k * V + v] * k_val;
                }
            }

            // Step 3: delta[v] = beta * (value[h,v] - kv_mem[v])
            let mut delta = vec![0.0f32; V];
            for v in 0..V {
                let v_val = bf16_to_f32_kernel(value[v_start + v].to_bits());
                delta[v] = beta * (v_val - kv_mem[v]);
            }

            // Step 4: State update
            for k in 0..K {
                let k_val = bf16_to_f32_kernel(key[k_start + k].to_bits()) * k_rcp;
                for v in 0..V {
                    state[state_h + k * V + v] += k_val * delta[v];
                }
            }

            // Step 5: Output
            let out_start = t * H * V + h * V;
            for v in 0..V {
                let mut y_val = 0.0f32;
                for k in 0..K {
                    let s_val = state[state_h + k * V + v];
                    let q_val = bf16_to_f32_kernel(query[q_start + k].to_bits()) * q_rcp;
                    y_val += s_val * q_val;
                }
                output_f32[out_start + v] = y_val;
            }
        }
    }

    output_f32
}

// ── Main test ───────────────────────────────────────────────────────────
// @lat: [[tests/gdn_chunked_prefill_test#GDN Chunked Prefill Kernel Test]]
#[test]
fn test_gdn_chunked_prefill() -> anyhow::Result<()> {
    let seq_len: usize = 8;
    let H: usize = 4;     // num_heads (per-GPU, small for testing)
    let K: usize = 32;    // head_k_dim
    let V: usize = 32;    // head_v_dim
    let chunk_size: u32 = 4;
    let seed: u64 = 42;

    println!("\\n=== GDN Chunked Prefill Test ===");
    println!("  seq_len={}, H={}, K={}, V={}, chunk_size={}", seq_len, H, K, V, chunk_size);

    // Generate deterministic BF16 inputs
    let query = generate_bf16(seed + 0, seq_len * H * K);
    let key = generate_bf16(seed + 1, seq_len * H * K);
    let value = generate_bf16(seed + 2, seq_len * H * V);
    let a_proj = generate_bf16(seed + 3, seq_len * H);
    let b_proj = generate_bf16(seed + 4, seq_len * H);

    // State: [H, K, V] f32 initialized to zeros
    let state_size = H * K * V;
    let mut state_cpu = vec![0.0f32; state_size];

    // Save inputs to /tmp/gdn_prefill_test_inputs/
    let dir = "/tmp/gdn_prefill_test_inputs";
    fs::create_dir_all(dir)?;

    fn save_bf16(filepath: &str, data: &[bf16]) {
        let bytes: Vec<u8> = data.iter()
            .map(|v| v.to_bits())
            .flat_map(|b| b.to_le_bytes())
            .collect();
        save_raw(filepath, &bytes);
    }

    fn save_f32(filepath: &str, data: &[f32]) {
        let bytes: Vec<u8> = data.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        save_raw(filepath, &bytes);
    }

    save_bf16(&format!("{}/query.bf16", dir), &query);
    save_bf16(&format!("{}/key.bf16", dir), &key);
    save_bf16(&format!("{}/value.bf16", dir), &value);
    save_bf16(&format!("{}/a_proj.bf16", dir), &a_proj);
    save_bf16(&format!("{}/b_proj.bf16", dir), &b_proj);
    save_f32(&format!("{}/a_log.f32", dir), &A_LOG.to_vec());
    save_f32(&format!("{}/dt_bias.f32", dir), &DT_BIAS.to_vec());
    save_f32(&format!("{}/state.f32", dir), &state_cpu);

    // JSON manifest
    let manifest = serde_json::json!({
        "seq_len": seq_len,
        "num_heads": H,
        "head_k_dim": K,
        "head_v_dim": V,
        "chunk_size": chunk_size,
        "seed": seed,
        "files": {
            "query": { "path": "query.bf16", "shape": [seq_len, H, K], "dtype": "bf16" },
            "key": { "path": "key.bf16", "shape": [seq_len, H, K], "dtype": "bf16" },
            "value": { "path": "value.bf16", "shape": [seq_len, H, V], "dtype": "bf16" },
            "a_proj": { "path": "a_proj.bf16", "shape": [seq_len, H], "dtype": "bf16" },
            "b_proj": { "path": "b_proj.bf16", "shape": [seq_len, H], "dtype": "bf16" },
            "a_log": { "path": "a_log.f32", "shape": [H], "dtype": "f32" },
            "dt_bias": { "path": "dt_bias.f32", "shape": [H], "dtype": "f32" },
            "state": { "path": "state.f32", "shape": [H, K, V], "dtype": "f32" },
        }
    });
    fs::write(format!("{}/manifest.json", dir), serde_json::to_string_pretty(&manifest)?)?;

    println!("  Inputs saved to {}", dir);

    // CPU sequential reference (compute with mutable state)
    let cpu_output_f32 = gdn_chunked_prefill_ref_f32(
        &query, &key, &value, &a_proj, &b_proj,
        &A_LOG.to_vec(), &DT_BIAS.to_vec(),
        &mut state_cpu, seq_len, H, K, V,
    );

    // Save CPU output and final state
    save_f32(&format!("{}/cpu_output.f32", dir), &cpu_output_f32);
    save_f32(&format!("{}/cpu_state.f32", dir), &state_cpu);
    println!("  CPU reference computed");

    // GPU kernel
    let (_ctx, stream, kernels) = create_context()?;

    let query_gpu = stream.clone_htod(&query)?;
    let key_gpu = stream.clone_htod(&key)?;
    let value_gpu = stream.clone_htod(&value)?;
    let a_proj_gpu = stream.clone_htod(&a_proj)?;
    let b_proj_gpu = stream.clone_htod(&b_proj)?;
    let a_log_gpu = stream.clone_htod(&A_LOG.to_vec())?;
    let dt_bias_gpu = stream.clone_htod(&DT_BIAS.to_vec())?;

    // State: zeros initially (f32)
    let mut state_gpu = stream.alloc_zeros::<f32>(state_size)?;
    let mut output_gpu = stream.alloc_zeros::<bf16>(seq_len * H * V)?;

    kernels.launch_gdn_chunked_gated_delta_prefill_bf16(
        &stream,
        &query_gpu, &key_gpu, &value_gpu,
        &a_proj_gpu, &b_proj_gpu,
        &a_log_gpu, &dt_bias_gpu,
        &mut state_gpu,
        &mut output_gpu,
        seq_len as u32, H as u32, K as u32, V as u32, chunk_size,
    )?;

    stream.synchronize()?;

    // Download GPU output and final state
    let gpu_output: Vec<bf16> = stream.clone_dtoh(&output_gpu)?;
    let gpu_state: Vec<f32> = stream.clone_dtoh(&state_gpu)?;

    save_bf16(&format!("{}/gpu_output.bf16", dir), &gpu_output);
    let gpu_output_f32: Vec<f32> = gpu_output.iter().map(|v| bf16_to_f32_kernel(v.to_bits())).collect();
    save_f32(&format!("{}/gpu_output.f32", dir), &gpu_output_f32);
    save_f32(&format!("{}/gpu_state.f32", dir), &gpu_state);

    println!("  GPU kernel completed");

    // ── Compare outputs (GPU bf16 vs CPU f32) ───────────────────────
    let mut max_abs_err = 0.0f32;
    let mut global_dot = 0.0f32;
    let mut global_norm_cpu_sq = 0.0f32;
    let mut global_norm_gpu_sq = 0.0f32;

    for i in 0..(seq_len * H * V) {
        let cpu_f = cpu_output_f32[i];
        let gpu_f = bf16_to_f32_kernel(gpu_output[i].to_bits());
        let err = (cpu_f - gpu_f).abs();

        max_abs_err = max_abs_err.max(err);
        global_dot += cpu_f * gpu_f;
        global_norm_cpu_sq += cpu_f * cpu_f;
        global_norm_gpu_sq += gpu_f * gpu_f;
    }

    let output_cosine = global_dot / (global_norm_cpu_sq * global_norm_gpu_sq).sqrt();

    println!("\\n  ─── Output Comparison (GPU vs CPU Ref) ───");
    println!("  Output cosine similarity: {:.8}", output_cosine);
    println!("  Max absolute error:       {:.6}", max_abs_err);

    // ── Compare states (GPU vs CPU) ────────────────────────────────
    let mut state_max_abs_err = 0.0f32;
    let mut state_dot = 0.0f32;
    let mut state_norm_cpu_sq = 0.0f32;
    let mut state_norm_gpu_sq = 0.0f32;

    for i in 0..state_size {
        let cpu_f = state_cpu[i];
        let gpu_f = gpu_state[i];
        let err = (cpu_f - gpu_f).abs();

        state_max_abs_err = state_max_abs_err.max(err);
        state_dot += cpu_f * gpu_f;
        state_norm_cpu_sq += cpu_f * cpu_f;
        state_norm_gpu_sq += gpu_f * gpu_f;
    }

    let state_cosine = if (state_norm_cpu_sq * state_norm_gpu_sq) > 0.0 {
        state_dot / (state_norm_cpu_sq * state_norm_gpu_sq).sqrt()
    } else {
        1.0 // both zero means perfect agreement
    };

    println!("\\n  ─── State Comparison (GPU vs CPU Ref) ───");
    println!("  State cosine similarity: {:.8}", state_cosine);
    println!("  Max state error:         {:.6}", state_max_abs_err);

    // Save comparison results for Python verification
    let results = serde_json::json!({
        "output_cosine_similarity": output_cosine,
        "output_max_absolute_error": max_abs_err,
        "state_cosine_similarity": state_cosine,
        "state_max_absolute_error": state_max_abs_err,
    });
    fs::write(format!("{}/comparison.json", dir), serde_json::to_string_pretty(&results)?)?;
    // Per-token diagnostics
    println!("\\n  ─── Per-Token Output Comparison ───");
    for t in 0..seq_len {
        let mut td = 0.0f32;
        let mut tn_c = 0.0f32;
        let mut tn_g = 0.0f32;
        let mut tmax = 0.0f32;
        for i in t * H * V..(t + 1) * H * V {
            let c = cpu_output_f32[i];
            let g = bf16_to_f32_kernel(gpu_output[i].to_bits());
            td += c * g;
            tn_c += c * c;
            tn_g += g * g;
            tmax = tmax.max((c - g).abs());
        }
        let cos = if (tn_c * tn_g) > 0.0 { td / (tn_c * tn_g).sqrt() } else { 1.0 };
        println!("  Token {:2}: cosine={:.8}, max_err={:.6}", t, cos, tmax);
    }

    // Per-head diagnostics for output
    println!("\\n  ─── Per-Head Output Comparison ───");
    for h in 0..H {
        let mut hd = 0.0f32;
        let mut hn_c = 0.0f32;
        let mut hn_g = 0.0f32;
        let mut hmax = 0.0f32;
        for t in 0..seq_len {
            for v in 0..V {
                let idx = t * H * V + h * V + v;
                let c = cpu_output_f32[idx];
                let g = bf16_to_f32_kernel(gpu_output[idx].to_bits());
                hd += c * g;
                hn_c += c * c;
                hn_g += g * g;
                hmax = hmax.max((c - g).abs());
            }
        }
        let cos = if (hn_c * hn_g) > 0.0 { hd / (hn_c * hn_g).sqrt() } else { 1.0 };
        println!("  Head {:2}: cosine={:.8}, max_err={:.6}", h, cos, hmax);
    }

    // Assert: cosine similarity > 0.95 (accounts for bf16 intermediate round-trips in kernel)
    assert!(output_cosine > 0.95,
        "Output cosine similarity {:.8} is below threshold 0.95", output_cosine);
    assert!(state_cosine > 0.95,
        "State cosine similarity {:.8} is below threshold 0.95", state_cosine);

    println!("\\n  ✓ GDN chunked prefill test PASSED");
    Ok(())
}
