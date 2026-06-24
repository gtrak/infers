//! GDN recurrent step kernel verification against CPU reference.
//!
//! Generates deterministic BF16 inputs, runs `launch_gdn_recurrent_step_bf16` on GPU,
//! and compares output with an f32 CPU reference implementation of the same algorithm.
//! Also saves all intermediate data to `/tmp/gdn_test_inputs/` for external Python comparison.

use std::sync::Arc;
use std::fs;

use half::bf16;
use infers_cuda::OxideKernels;
use cudarc::driver::{CudaContext, CudaStream};

// ── Model weights (layer 0, num_v_heads=24) ──────────────────────────────
/// a_log values from Qwen3.6-27B-AutoRound-INT4 layer 0 (f32, first 24 heads)
const A_LOG: [f32; 24] = [
    -3.203125, -2.640625, -2.718750, -4.656250, -4.906250, -5.406250,
    -2.812500, -4.218750, -2.375000, -3.921875, -3.468750, -2.843750,
    -4.906250, -4.937500, -4.312500, -4.500000, -4.656250, -4.218750,
    -2.640625, -2.531250, -2.125000, -3.296875, -4.593750, -4.531250,
];

/// dt_bias values from Qwen3.6-27B-AutoRound-INT4 layer 0 (f32, first 24 heads)
const DT_BIAS: [f32; 24] = [
    -3.468750, 13.125000, -2.578125, -1.078125, -2.921875, -1.703125,
     18.375000, -5.468750, 19.250000,  5.062500, -1.765625,  6.312500,
    -2.140625, -1.648438, -2.578125, -3.125000, -2.609375, -2.593750,
     15.375000, 14.812500, 16.375000, -0.781250, -1.460938, -1.429688,
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
            // Generate f32 in [-1, 1] from LCG output
            let raw = lcg_next(&mut state) as u32;
            let frac = (raw >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
            let val = frac * 2.0 - 1.0; // [-1, 1]
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

// ── CPU reference implementation (f32) ───────────────────────────────────
/// Compute the GDN recurrent step for all heads sequentially.
fn gdn_recurrent_step_ref_f32(
    query: &[bf16],           // [H, K] bf16
    key: &[bf16],             // [H, K] bf16
    value: &[bf16],           // [H, V] bf16
    a_proj: &[bf16],          // [H] bf16
    b_proj: &[bf16],          // [H] bf16
    a_log: &[f32],            // [H] f32
    dt_bias: &[f32],          // [H] f32
    state: &mut Vec<f32>,     // [H, K, V] f32 (row-major: state[h*K*V + k*V + v])
    H: usize,
    K: usize,
    V: usize,
) -> Vec<f32> {
    let eps = 1e-6f32;
    let mut output = vec![0.0f32; H * V];

    for h in 0..H {
        let q_start = h * K;
        let k_start = h * K;
        let v_start = h * V;
        let state_h = h * K * V;

        // ── Per-head scalars ────────────────────────────────────────
        let a_proj_val = bf16_to_f32_kernel(a_proj[h].to_bits());
        let b_proj_val = bf16_to_f32_kernel(b_proj[h].to_bits());

        let decay_rate = a_log[h].exp();
        let sp_val = a_proj_val + dt_bias[h];

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

        // ── L2-normalize key and query ──────────────────────────────
        let mut k_l2_sq = 0.0f32;
        let mut q_l2_sq = 0.0f32;
        for k in 0..K {
            let kv = bf16_to_f32_kernel(key[k_start + k].to_bits());
            let qv = bf16_to_f32_kernel(query[q_start + k].to_bits());
            k_l2_sq += kv * kv;
            q_l2_sq += qv * qv;
        }

        let k_rcp = 1.0f32 / (k_l2_sq + eps).sqrt();
        let rcp_sqrt_k = 1.0f32 / (K as f32).sqrt();
        let q_rcp = 1.0f32 / (q_l2_sq + eps).sqrt() * rcp_sqrt_k;

        // ── Step 1: State decay ─────────────────────────────────────
        for k in 0..K {
            for v in 0..V {
                state[state_h + k * V + v] *= decay;
            }
        }

        // ── Step 2: kv_mem[v] = sum_k(state[h,k,v] * key_normed[k]) ─
        let mut kv_mem = vec![0.0f32; V];
        for k in 0..K {
            let k_val = bf16_to_f32_kernel(key[k_start + k].to_bits()) * k_rcp;
            for v in 0..V {
                kv_mem[v] += state[state_h + k * V + v] * k_val;
            }
        }

        // ── Step 3: delta[v] = beta * (value[h,v] - kv_mem[v]) ───────
        let mut delta = vec![0.0f32; V];
        for v in 0..V {
            let v_val = bf16_to_f32_kernel(value[v_start + v].to_bits());
            delta[v] = beta * (v_val - kv_mem[v]);
        }

        // ── Step 4: State update ─────────────────────────────────────
        for k in 0..K {
            let k_val = bf16_to_f32_kernel(key[k_start + k].to_bits()) * k_rcp;
            for v in 0..V {
                state[state_h + k * V + v] += k_val * delta[v];
            }
        }

        // ── Step 5: Output ───────────────────────────────────────────
        for v in 0..V {
            let mut y_val = 0.0f32;
            for k in 0..K {
                let s_val = state[state_h + k * V + v];
                let q_val = bf16_to_f32_kernel(query[q_start + k].to_bits()) * q_rcp;
                y_val += s_val * q_val;
            }
            output[h * V + v] = y_val;
        }
    }

    output
}

// ── Main test ────────────────────────────────────────────────────────────
// @lat: [[tests/gdn_recurrent_step_test#GDN Recurrent Step Kernel Test]]
#[test]
fn test_gdn_recurrent_step() -> anyhow::Result<()> {
    let H: usize = 24;   // num_v_heads
    let K: usize = 128;  // head_k_dim
    let V: usize = 128;  // head_v_dim
    let seed: u64 = 42;

    println!("\n=== GDN Recurrent Step Test ===");
    println!("  H={}, K={}, V={}, seed={}", H, K, V, seed);

    // ── Generate deterministic BF16 inputs ─────────────────────────────
    let query = generate_bf16(seed + 0, H * K);
    let key = generate_bf16(seed + 1, H * K);
    let value = generate_bf16(seed + 2, H * V);
    let a_proj = generate_bf16(seed + 3, H);
    let b_proj = generate_bf16(seed + 4, H);

    // State: [H, K, V] f32 initialized to zeros
    let state_size = H * K * V;
    let mut state_cpu = vec![0.0f32; state_size];

    // ── Save inputs to /tmp/gdn_test_inputs/ ───────────────────────────
    let dir = "/tmp/gdn_test_inputs";
    fs::create_dir_all(dir)?;

    // Helper: save bf16 as raw binary
    fn save_bf16(filepath: &str, data: &[bf16]) {
        let bytes: Vec<u8> = data.iter()
            .map(|v| v.to_bits())
            .flat_map(|b| b.to_le_bytes())
            .collect();
        save_raw(filepath, &bytes);
    }

    // Helper: save f32 as raw binary
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
        "num_v_heads": H,
        "head_k_dim": K,
        "head_v_dim": V,
        "seed": seed,
        "files": {
            "query": { "path": "query.bf16", "shape": [H, K], "dtype": "bf16" },
            "key": { "path": "key.bf16", "shape": [H, K], "dtype": "bf16" },
            "value": { "path": "value.bf16", "shape": [H, V], "dtype": "bf16" },
            "a_proj": { "path": "a_proj.bf16", "shape": [H], "dtype": "bf16" },
            "b_proj": { "path": "b_proj.bf16", "shape": [H], "dtype": "bf16" },
            "a_log": { "path": "a_log.f32", "shape": [H], "dtype": "f32" },
            "dt_bias": { "path": "dt_bias.f32", "shape": [H], "dtype": "f32" },
            "state": { "path": "state.f32", "shape": [H, K, V], "dtype": "f32" },
        }
    });
    fs::write(format!("{}/manifest.json", dir), serde_json::to_string_pretty(&manifest)?)?;

    println!("  Inputs saved to {}", dir);

    // ── CPU reference (compute with mutable copy of state) ─────────────
    let cpu_output_f32 = gdn_recurrent_step_ref_f32(
        &query, &key, &value, &a_proj, &b_proj,
        &A_LOG.to_vec(), &DT_BIAS.to_vec(),
        &mut state_cpu, H, K, V,
    );

    // ── GPU kernel ─────────────────────────────────────────────────────
    let (_ctx, stream, kernels) = create_context()?;

    let query_gpu = stream.clone_htod(&query)?;
    let key_gpu = stream.clone_htod(&key)?;
    let value_gpu = stream.clone_htod(&value)?;
    let a_proj_gpu = stream.clone_htod(&a_proj)?;
    let b_proj_gpu = stream.clone_htod(&b_proj)?;
    let a_log_gpu = stream.clone_htod(&A_LOG.to_vec())?;
    let dt_bias_gpu = stream.clone_htod(&DT_BIAS.to_vec())?;

    // State: zeros initially (f32)
    let mut state_gpu = stream.alloc_zeros::<f32>(H * K * V)?;
    let mut output_gpu = stream.alloc_zeros::<bf16>(H * V)?;

    kernels.launch_gdn_recurrent_step_bf16(
        &stream,
        &query_gpu, &key_gpu, &value_gpu,
        &a_proj_gpu, &b_proj_gpu,
        &a_log_gpu, &dt_bias_gpu,
        &mut state_gpu,
        &mut output_gpu,
        H as u32, K as u32, V as u32,
    )?;

    stream.synchronize()?;

    // Download GPU output
    let gpu_output: Vec<bf16> = stream.clone_dtoh(&output_gpu)?;

    // Save GPU output
    save_bf16(&format!("{}/gpu_output.bf16", dir), &gpu_output);
    println!("  GPU output saved to {}/gpu_output.bf16", dir);

    // ── Compare GPU vs CPU reference ───────────────────────────────────
    let mut max_abs_err = 0.0f32;
    let mut global_dot = 0.0f32;
    let mut global_norm_cpu_sq = 0.0f32;
    let mut global_norm_gpu_sq = 0.0f32;

    let mut head_cosine_vals = vec![0.0f32; H];
    let mut head_max_abs_errs = vec![0.0f32; H];
    let mut head_dot = vec![0.0f32; H];
    let mut head_norm_cpu_sq = vec![0.0f32; H];
    let mut head_norm_gpu_sq = vec![0.0f32; H];

    for h in 0..H {
        for v in 0..V {
            let cpu_f = cpu_output_f32[h * V + v];
            let gpu_f = gpu_output[h * V + v].to_f32();
            let err = (cpu_f - gpu_f).abs();

            max_abs_err = max_abs_err.max(err);
            head_max_abs_errs[h] = head_max_abs_errs[h].max(err);

            global_dot += cpu_f * gpu_f;
            global_norm_cpu_sq += cpu_f * cpu_f;
            global_norm_gpu_sq += gpu_f * gpu_f;

            head_dot[h] += cpu_f * gpu_f;
            head_norm_cpu_sq[h] += cpu_f * cpu_f;
            head_norm_gpu_sq[h] += gpu_f * gpu_f;
        }

        let denom = (head_norm_cpu_sq[h] * head_norm_gpu_sq[h]).sqrt();
        head_cosine_vals[h] = if denom > 0.0 { head_dot[h] / denom } else { 0.0 };
    }

    let global_cosine = global_dot / (global_norm_cpu_sq * global_norm_gpu_sq).sqrt();

    println!("\n  ─── Per-Head Comparison (GPU vs CPU Ref) ───");
    for h in 0..H {
        println!(
            "  Head {:2}: cosine={:.8}, max_abs_err={:.6}",
            h, head_cosine_vals[h], head_max_abs_errs[h]
        );
    }
    println!("\n  ─── Global Summary ───");
    println!("  Global cosine similarity: {:.8}", global_cosine);
    println!("  Max absolute error:       {:.6}", max_abs_err);

    // Save comparison results for Python verification
    let results = serde_json::json!({
        "global_cosine_similarity": global_cosine,
        "max_absolute_error": max_abs_err,
        "per_head_cosine": head_cosine_vals,
        "per_head_max_abs_err": head_max_abs_errs,
    });
    fs::write(format!("{}/comparison.json", dir), serde_json::to_string_pretty(&results)?)?;

    // Save CPU reference output as bf16 for Python comparison
    let cpu_output_bf16: Vec<bf16> = cpu_output_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    save_bf16(&format!("{}/cpu_reference.bf16", dir), &cpu_output_bf16);

    // ── Assert ─────────────────────────────────────────────────────────
    assert!(
        global_cosine > 0.999,
        "Global cosine similarity {:.8} is below threshold 0.999",
        global_cosine
    );

    println!("\n  ✓ GDN recurrent step test PASSED (cosine={:.8})", global_cosine);
    Ok(())
}
