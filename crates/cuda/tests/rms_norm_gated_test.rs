//! RMSNorm + SiLU gate kernel verification against CPU reference.
//!
//! Generates deterministic BF16 inputs, runs `infers_rms_norm_gated_bf16` on GPU,
//! and compares output with an f32 CPU reference implementation of the same algorithm.
//! Also saves all intermediate data to `/tmp/rms_norm_test_inputs/` for external Python comparison.

use std::sync::Arc;
use std::fs;

use half::bf16;
use infers_cuda::OxideKernels;
use cudarc::driver::{CudaContext, CudaStream};

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
/// Compute RMSNorm + SiLU gate for all rows.
fn rms_norm_gated_ref_f32(
    input: &[bf16],  // [n_rows, d] bf16
    gate: &[bf16],   // [n_rows, d] bf16
    weight: &[bf16], // [d] bf16
    n_rows: usize,
    d: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_rows * d];

    for r in 0..n_rows {
        // Step 1: Compute sum_sq across d elements
        let mut sum_sq = 0.0f32;
        for i in 0..d {
            let val = bf16_to_f32_kernel(input[r * d + i].to_bits());
            sum_sq += val * val;
        }

        // Step 2: inv_rms
        let inv_rms = 1.0f32 / (sum_sq / d as f32 + eps).sqrt();

        // Step 3: For each element i
        for i in 0..d {
            let x_val = bf16_to_f32_kernel(input[r * d + i].to_bits());
            let g_val = bf16_to_f32_kernel(gate[r * d + i].to_bits());
            let w_val = bf16_to_f32_kernel(weight[i].to_bits());

            let x_norm = x_val * inv_rms;
            // SiLU(x) = x / (1 + exp(-x))
            let silu_gate = g_val / (1.0f32 + (-g_val).exp());
            output[r * d + i] = w_val * x_norm * silu_gate;
        }
    }

    output
}

// ── Main test ────────────────────────────────────────────────────────────
// @lat: [[tests/rms_norm_gated_test#RMSNorm Gated Kernel Test]]
#[test]
fn test_rms_norm_gated() -> anyhow::Result<()> {
    let n_rows: usize = 24; // GDN head configuration
    let d: usize = 128;    // dimension per row
    let eps = 1e-6f32;
    let seed: u64 = 42;

    println!("\n=== RMSNorm + SiLU Gate Test ===");
    println!("  n_rows={}, d={}, eps={:.0e}, seed={}", n_rows, d, eps, seed);

    // ── Generate deterministic BF16 inputs ─────────────────────────────
    let input_data = generate_bf16(seed + 0, n_rows * d);
    let gate_data = generate_bf16(seed + 1, n_rows * d);
    let weight_data = generate_bf16(seed + 2, d);

    // ── Save inputs to /tmp/rms_norm_test_inputs/ ───────────────────────
    let dir = "/tmp/rms_norm_test_inputs";
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

    save_bf16(&format!("{}/input.bf16", dir), &input_data);
    save_bf16(&format!("{}/gate.bf16", dir), &gate_data);
    save_bf16(&format!("{}/weight.bf16", dir), &weight_data);

    // JSON manifest
    let manifest = serde_json::json!({
        "n_rows": n_rows,
        "d": d,
        "eps": eps,
        "seed": seed,
        "files": {
            "input": { "path": "input.bf16", "shape": [n_rows, d], "dtype": "bf16" },
            "gate": { "path": "gate.bf16", "shape": [n_rows, d], "dtype": "bf16" },
            "weight": { "path": "weight.bf16", "shape": [d], "dtype": "bf16" },
        }
    });
    fs::write(format!("{}/manifest.json", dir), serde_json::to_string_pretty(&manifest)?)?;

    println!("  Inputs saved to {}", dir);

    // ── CPU reference ──────────────────────────────────────────────────
    let cpu_output_f32 = rms_norm_gated_ref_f32(
        &input_data, &gate_data, &weight_data, n_rows, d, eps,
    );

    // ── GPU kernel ─────────────────────────────────────────────────────
    let (_ctx, stream, kernels) = create_context()?;

    let input_gpu = stream.clone_htod(&input_data)?;
    let gate_gpu = stream.clone_htod(&gate_data)?;
    let weight_gpu = stream.clone_htod(&weight_data)?;
    let mut output_gpu = stream.alloc_zeros::<bf16>(n_rows * d)?;

    kernels.launch_rms_norm_gated_bf16(
        &stream,
        &input_gpu,
        &gate_gpu,
        &weight_gpu,
        &mut output_gpu,
        n_rows as u32,
        d as u32,
        eps,
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

    let mut row_cosine_vals = vec![0.0f32; n_rows];
    let mut row_max_abs_errs = vec![0.0f32; n_rows];
    let mut row_dot = vec![0.0f32; n_rows];
    let mut row_norm_cpu_sq = vec![0.0f32; n_rows];
    let mut row_norm_gpu_sq = vec![0.0f32; n_rows];

    for r in 0..n_rows {
        for i in 0..d {
            let idx = r * d + i;
            let cpu_f = cpu_output_f32[idx];
            let gpu_f = gpu_output[idx].to_f32();
            let err = (cpu_f - gpu_f).abs();

            max_abs_err = max_abs_err.max(err);
            row_max_abs_errs[r] = row_max_abs_errs[r].max(err);

            global_dot += cpu_f * gpu_f;
            global_norm_cpu_sq += cpu_f * cpu_f;
            global_norm_gpu_sq += gpu_f * gpu_f;

            row_dot[r] += cpu_f * gpu_f;
            row_norm_cpu_sq[r] += cpu_f * cpu_f;
            row_norm_gpu_sq[r] += gpu_f * gpu_f;
        }

        let denom = (row_norm_cpu_sq[r] * row_norm_gpu_sq[r]).sqrt();
        row_cosine_vals[r] = if denom > 0.0 { row_dot[r] / denom } else { 0.0 };
    }

    let global_cosine = global_dot / (global_norm_cpu_sq * global_norm_gpu_sq).sqrt();

    println!("\n  ─── Per-Row Comparison (GPU vs CPU Ref) ───");
    for r in 0..n_rows {
        println!(
            "  Row {:2}: cosine={:.8}, max_abs_err={:.6}",
            r, row_cosine_vals[r], row_max_abs_errs[r]
        );
    }
    println!("\n  ─── Global Summary ───");
    println!("  Global cosine similarity: {:.8}", global_cosine);
    println!("  Max absolute error:       {:.6}", max_abs_err);

    // Save comparison results for Python verification
    let results = serde_json::json!({
        "global_cosine_similarity": global_cosine,
        "max_absolute_error": max_abs_err,
        "per_row_cosine": row_cosine_vals,
        "per_row_max_abs_err": row_max_abs_errs,
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

    println!("\n  ✓ RMSNorm + SiLU gate test PASSED (cosine={:.8})", global_cosine);
    Ok(())
}
