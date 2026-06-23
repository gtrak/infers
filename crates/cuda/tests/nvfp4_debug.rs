//! Debug NVFP4 GEMM pipeline: isolate which kernel fails for large N dimensions.
//!
//! The oracle shows in_proj_qkv (N=5120) fails with cos=-0.001 while a_proj/b_proj
//! (N=24) succeed with cos≈1.0. Both use the same dequant→bf16_gemm_tiled pipeline.
//! These tests isolate the failing component.

use half::bf16;
use std::sync::Arc;

use infers_cuda::OxideKernels;
use cudarc::driver::{CudaContext, CudaStream};

// ── fp4_e2m1_to_f32 LUT (matches kernel) ──────────────────────────────
fn fp4_e2m1_to_f32_cpu(nibble: u8) -> f32 {
    let sign = (nibble >> 3) & 1;
    let magnitude = match nibble & 0x7 {
        0 => 0.0f32,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        7 => 6.0,
        _ => unreachable!(),
    };
    if sign != 0 { -magnitude } else { magnitude }
}

// ── Fp8E4M3 dequantize (matches kernel) ───────────────────────────────
fn fp8_e4m3_dequantize_cpu(val: u8) -> f32 {
    let sign = (val >> 7) & 1;
    let exp = (val >> 3) & 0xF;
    let mant = val & 0x7;

    // NaN
    if exp == 0xF {
        return f32::from_bits(0x7FC00000);
    }
    // Zero
    if exp == 0 && mant == 0 {
        return if sign != 0 { -0.0f32 } else { 0.0f32 };
    }

    let fp32_exp = if exp == 0 { 0 } else { (exp as u32) + 120 }; // 127 - 7 = 120
    let fp32_mant = (mant as u32) << 20;
    f32::from_bits(((sign as u32) << 31) | (fp32_exp << 23) | fp32_mant)
}

// ── CPU reference: NVFP4 dequant to bf16 ──────────────────────────────
fn nvfp4_dequant_cpu(
    weight_packed: &[u8],
    weight_scale: &[u8],
    weight_global_scale: f32,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<bf16> {
    let num_groups = k / group_size;
    let mut output = vec![bf16::from_f32(0.0); n * k];

    for row in 0..n {
        for g in 0..num_groups {
            let scale_fp8 = weight_scale[row * num_groups + g];
            let scale = fp8_e4m3_dequantize_cpu(scale_fp8);

            for i in 0..(group_size / 2) {
                let byte_idx = row * (k / 2) + g * group_size / 2 + i;
                let packed_byte = weight_packed[byte_idx];

                let hi_nibble = (packed_byte >> 4) & 0xF;
                let hi_val = fp4_e2m1_to_f32_cpu(hi_nibble);
                let hi_fp32 = hi_val * scale / weight_global_scale;

                let lo_nibble = packed_byte & 0xF;
                let lo_val = fp4_e2m1_to_f32_cpu(lo_nibble);
                let lo_fp32 = lo_val * scale / weight_global_scale;

                let out_base = row * k + g * group_size + i * 2;
                output[out_base] = bf16::from_f32(hi_fp32);
                output[out_base + 1] = bf16::from_f32(lo_fp32);
            }
        }
    }
    output
}

// ── CPU reference: BF16 GEMM (C = A @ B^T) in fp32 accumulation ───────
fn bf16_gemm_cpu(
    input: &[bf16],   // [M, K]
    weight: &[bf16],  // [N, K]
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += input[i * k + ki].to_f32() * weight[j * k + ki].to_f32();
            }
            output[i * n + j] = sum;
        }
    }
    output
}

// ── Helpers to create synthetic NVFP4 data ────────────────────────────
fn make_nvfp4_data(
    n: usize,
    k: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<u8>, f32) {
    let num_groups = k / group_size;

    // Weight packed: [N, K/2] — each byte holds 2 FP4 E2M1 values
    // Use deterministic pattern with small magnitudes that FP4 can represent well
    let mut weight_packed = Vec::with_capacity(n * k / 2);
    for i in 0..(n * k / 2) {
        // Deterministic: cycle through nibbles, alternate sign every other byte
        let hi_mag = (i % 7) as u8;
        let lo_mag = ((i + 3) % 7) as u8;
        let hi_sign = (((i / 2) % 2) << 3) as u8; // alternate sign
        let lo_sign = ((((i + 1) / 2) % 2) << 3) as u8;
        weight_packed.push((hi_mag | hi_sign) << 4 | lo_mag | lo_sign);
    }

    // Weight scale: [N, K/group_size] — fp8 e4m3 values
    let mut weight_scale = Vec::with_capacity(n * num_groups);
    for _ in 0..(n * num_groups) {
        // Generate a valid fp8 e4m3 value (small positive scale like 1.0-2.0)
        let val = 0x51u8; // sign=0, exp=5 → bias 7 → 2^(-2), mant=1 → ≈0.25 * 1.125 ≈ 0.281
        weight_scale.push(val);
    }

    // Weight global scale: small positive f32
    let weight_global_scale = 0.1f32;

    (weight_packed, weight_scale, weight_global_scale)
}

// ── Test helpers ───────────────────────────────────────────────────────
fn create_context() -> anyhow::Result<(Arc<CudaContext>, Arc<CudaStream>, OxideKernels)> {
    let ctx: Arc<CudaContext> = CudaContext::new(0)?;
    let stream: Arc<CudaStream> = ctx.new_stream()?;
    let kernels = OxideKernels::new(
        0,
        "/home/gary/dev/infers/crates/cuda/kernels/compiled/oxide_kernels.cubin",
    )?;
    Ok((ctx, stream, kernels))
}

/// Test 1: Dequant for small N (N=24) — should match CPU reference
// @lat: [[tests/gemm_compare#Gemm Compare Test#Multi-Tile GEMM Bug]]
#[test]
fn test_dequant_small_n() -> anyhow::Result<()> {
    let n = 24usize;
    let k = 5120usize;
    let group_size = 16usize;

    println!("\n=== Test 1: Dequant small N={} ===", n);

    let (weight_packed, weight_scale, weight_global_scale) = make_nvfp4_data(n, k, group_size);

    // CPU reference
    let cpu_ref = nvfp4_dequant_cpu(&weight_packed, &weight_scale, weight_global_scale, n, k, group_size);

    // GPU dequant
    let (_ctx, stream, kernels) = create_context()?;
    let mut dequant_gpu = stream.alloc_zeros::<bf16>(n * k)?;
    let packed_gpu = stream.clone_htod(&weight_packed)?;
    let scale_gpu = stream.clone_htod(&weight_scale)?;

    kernels.launch_nvfp4_dequant_to_bf16(
        &stream,
        &mut dequant_gpu,
        &packed_gpu,
        &scale_gpu,
        weight_global_scale,
        n as u32,
        k as u32,
        group_size as u32,
    )?;

    stream.synchronize()?;
    let gpu_result: Vec<bf16> = stream.clone_dtoh(&dequant_gpu)?;

    // Compare
    let mut max_diff = 0.0f32;
    let mut mismatches = 0usize;
    for i in 0..(n * k) {
        let diff = (cpu_ref[i].to_f32() - gpu_result[i].to_f32()).abs();
        max_diff = max_diff.max(diff);
        if diff > 1e-5 {
            mismatches += 1;
        }
    }

    println!("  Max diff: {:.8}", max_diff);
    println!("  Mismatches (diff>1e-5): {} / {}", mismatches, n * k);

    assert!(max_diff < 1e-3, "Dequant small N: max_diff={:.8} exceeds tolerance", max_diff);
    Ok(())
}

/// Test 2: Dequant for large N (N=5120) — should match CPU reference
#[test]
fn test_dequant_large_n() -> anyhow::Result<()> {
    let n = 5120usize;
    let k = 5120usize;
    let group_size = 16usize;

    println!("\n=== Test 2: Dequant large N={} ===", n);

    let (weight_packed, weight_scale, weight_global_scale) = make_nvfp4_data(n, k, group_size);

    // CPU reference
    let cpu_ref = nvfp4_dequant_cpu(&weight_packed, &weight_scale, weight_global_scale, n, k, group_size);

    // GPU dequant
    let (_ctx, stream, kernels) = create_context()?;
    let mut dequant_gpu = stream.alloc_zeros::<bf16>(n * k)?;
    let packed_gpu = stream.clone_htod(&weight_packed)?;
    let scale_gpu = stream.clone_htod(&weight_scale)?;

    kernels.launch_nvfp4_dequant_to_bf16(
        &stream,
        &mut dequant_gpu,
        &packed_gpu,
        &scale_gpu,
        weight_global_scale,
        n as u32,
        k as u32,
        group_size as u32,
    )?;

    stream.synchronize()?;
    let gpu_result: Vec<bf16> = stream.clone_dtoh(&dequant_gpu)?;

    // Compare
    let mut max_diff = 0.0f32;
    let mut mismatches = 0usize;
    for i in 0..(n * k) {
        let diff = (cpu_ref[i].to_f32() - gpu_result[i].to_f32()).abs();
        max_diff = max_diff.max(diff);
        if diff > 1e-5 {
            mismatches += 1;
        }
    }

    println!("  Max diff: {:.8}", max_diff);
    println!("  Mismatches (diff>1e-5): {} / {}", mismatches, n * k);

    assert!(max_diff < 1e-3, "Dequant large N: max_diff={:.8} exceeds tolerance", max_diff);
    Ok(())
}

/// Test 3a: bf16_gemm_tiled for moderate N with K=5120 — isolate if K is the issue
#[test]
fn test_gemm_moderate_n() -> anyhow::Result<()> {
    let n = 256usize;
    let m = 19usize;
    let k = 5120usize;

    println!("\n=== Test 3a: GEMM moderate N={}, K={} ===", n, k);

    // Simple bf16 input: small positive values
    let input_host: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(0.01 * (i as f32 % 100.0 + 1.0)))
        .collect();

    // Simple bf16 weight: small positive values
    let weight_host: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(0.01 * (i as f32 % 50.0 + 0.5)))
        .collect();

    // CPU reference (fp32 accumulation)
    let cpu_ref = bf16_gemm_cpu(&input_host, &weight_host, m, n, k);

    // GPU GEMM
    let (_ctx, stream, kernels) = create_context()?;
    let mut gemm_gpu = stream.alloc_zeros::<bf16>(m * n)?;
    let input_gpu = stream.clone_htod(&input_host)?;
    let weight_gpu = stream.clone_htod(&weight_host)?;

    kernels.launch_bf16_gemm_tiled(
        &stream,
        &mut gemm_gpu,
        &input_gpu,
        &weight_gpu,
        m as u32,
        n as u32,
        k as u32,
    )?;

    stream.synchronize()?;
    let gpu_result: Vec<bf16> = stream.clone_dtoh(&gemm_gpu)?;

    // Compare
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut dot_cc = 0.0f32;
    let mut norm_cpu_sq = 0.0f32;
    let mut norm_gpu_sq = 0.0f32;

    for i in 0..(m * n) {
        let gpu_f32 = gpu_result[i].to_f32();
        let diff = (cpu_ref[i] - gpu_f32).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;

        dot_cc += cpu_ref[i] * gpu_f32;
        norm_cpu_sq += cpu_ref[i].powi(2);
        norm_gpu_sq += gpu_f32.powi(2);
    }

    let cosine_sim = dot_cc / (norm_cpu_sq * norm_gpu_sq).sqrt();
    let mean_diff = sum_diff / (m * n) as f32;

    println!("  Max diff: {:.6}", max_diff);
    println!("  Mean diff: {:.6}", mean_diff);
    println!("  Cosine similarity: {:.6}", cosine_sim);

    assert!(cosine_sim > 0.99, "GEMM moderate N: cosine_sim={:.6} too low", cosine_sim);
    Ok(())
}

/// Test 3b: bf16_gemm_tiled for large N (N=5120) — should match fp32 CPU reference
#[test]
fn test_gemm_large_n() -> anyhow::Result<()> {
    let m = 19usize;
    let n = 5120usize;
    let k = 5120usize;

    println!("\n=== Test 3b: GEMM large N={}, K={} ===", n, k);

    // Simple bf16 input: small positive values
    let input_host: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(0.01 * (i as f32 % 100.0 + 1.0)))
        .collect();

    // Simple bf16 weight: small positive values
    let weight_host: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(0.01 * (i as f32 % 50.0 + 0.5)))
        .collect();

    // CPU reference (fp32 accumulation)
    let cpu_ref = bf16_gemm_cpu(&input_host, &weight_host, m, n, k);

    // GPU GEMM
    let (_ctx, stream, kernels) = create_context()?;
    let mut gemm_gpu = stream.alloc_zeros::<bf16>(m * n)?;
    let input_gpu = stream.clone_htod(&input_host)?;
    let weight_gpu = stream.clone_htod(&weight_host)?;

    kernels.launch_bf16_gemm_tiled(
        &stream,
        &mut gemm_gpu,
        &input_gpu,
        &weight_gpu,
        m as u32,
        n as u32,
        k as u32,
    )?;

    stream.synchronize()?;
    let gpu_result: Vec<bf16> = stream.clone_dtoh(&gemm_gpu)?;

    // Compare
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut dot_cc = 0.0f32;
    let mut norm_cpu_sq = 0.0f32;
    let mut norm_gpu_sq = 0.0f32;

    for i in 0..(m * n) {
        let gpu_f32 = gpu_result[i].to_f32();
        let diff = (cpu_ref[i] - gpu_f32).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;

        dot_cc += cpu_ref[i] * gpu_f32;
        norm_cpu_sq += cpu_ref[i].powi(2);
        norm_gpu_sq += gpu_f32.powi(2);
    }

    let cosine_sim = dot_cc / (norm_cpu_sq * norm_gpu_sq).sqrt();
    let mean_diff = sum_diff / (m * n) as f32;

    println!("  Max diff: {:.6}", max_diff);
    println!("  Mean diff: {:.6}", mean_diff);
    println!("  Cosine similarity: {:.6}", cosine_sim);

    // BF16 has limited precision; check relative error
    assert!(cosine_sim > 0.99, "GEMM large N: cosine_sim={:.6} too low", cosine_sim);
    Ok(())
}

/// Test 3c: bf16_gemm_tiled for large N with small K — isolate if it's K causing the issue
#[test]
fn test_gemm_large_n_small_k() -> anyhow::Result<()> {
    let m = 19usize;
    let n = 5120usize;
    let k = 16usize;

    println!("\n=== Test 3c: GEMM large N={}, small K={} ===", n, k);

    // Simple bf16 input
    let input_host: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(0.1 * (i as f32 + 1.0)))
        .collect();

    // Simple bf16 weight
    let weight_host: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(0.5 + 0.1 * i as f32))
        .collect();

    // CPU reference (fp32 accumulation)
    let cpu_ref = bf16_gemm_cpu(&input_host, &weight_host, m, n, k);

    // GPU GEMM
    let (_ctx, stream, kernels) = create_context()?;
    let mut gemm_gpu = stream.alloc_zeros::<bf16>(m * n)?;
    let input_gpu = stream.clone_htod(&input_host)?;
    let weight_gpu = stream.clone_htod(&weight_host)?;

    kernels.launch_bf16_gemm_tiled(
        &stream,
        &mut gemm_gpu,
        &input_gpu,
        &weight_gpu,
        m as u32,
        n as u32,
        k as u32,
    )?;

    stream.synchronize()?;
    let gpu_result: Vec<bf16> = stream.clone_dtoh(&gemm_gpu)?;

    // Compare
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut dot_cc = 0.0f32;
    let mut norm_cpu_sq = 0.0f32;
    let mut norm_gpu_sq = 0.0f32;

    for i in 0..(m * n) {
        let gpu_f32 = gpu_result[i].to_f32();
        let diff = (cpu_ref[i] - gpu_f32).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;

        dot_cc += cpu_ref[i] * gpu_f32;
        norm_cpu_sq += cpu_ref[i].powi(2);
        norm_gpu_sq += gpu_f32.powi(2);
    }

    let cosine_sim = dot_cc / (norm_cpu_sq * norm_gpu_sq).sqrt();
    let mean_diff = sum_diff / (m * n) as f32;

    println!("  Max diff: {:.6}", max_diff);
    println!("  Mean diff: {:.6}", mean_diff);
    println!("  Cosine similarity: {:.6}", cosine_sim);

    assert!(cosine_sim > 0.99, "GEMM large N small K: cosine_sim={:.6} too low", cosine_sim);
    Ok(())
}

/// Test 4: Progressive N to find the exact failure boundary in bf16_gemm_tiled
#[test]
fn test_gemm_progressive_n() -> anyhow::Result<()> {
    let m = 19usize;
    let k = 5120usize;

    println!("\n=== Test 4: Progressive N (K={}) ===", k);

    // Use the same input across all tests for comparability
    let input_host: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(0.01 * (i as f32 % 100.0 + 1.0)))
        .collect();

    for &n in &[8usize, 64, 128, 256, 512, 1024, 2048, 4096] {
        let weight_host: Vec<bf16> = (0..n * k)
            .map(|i| bf16::from_f32(0.01 * (i as f32 % 50.0 + 0.5)))
            .collect();

        let cpu_ref = bf16_gemm_cpu(&input_host, &weight_host, m, n, k);

        let (_ctx, stream, kernels) = create_context()?;
        let mut gemm_gpu = stream.alloc_zeros::<bf16>(m * n)?;
        let input_gpu = stream.clone_htod(&input_host)?;
        let weight_gpu = stream.clone_htod(&weight_host)?;

        kernels.launch_bf16_gemm_tiled(
            &stream,
            &mut gemm_gpu,
            &input_gpu,
            &weight_gpu,
            m as u32,
            n as u32,
            k as u32,
        )?;

        stream.synchronize()?;
        let gpu_result: Vec<bf16> = stream.clone_dtoh(&gemm_gpu)?;

        let mut max_diff = 0.0f32;
        let mut dot_cc = 0.0f32;
        let mut norm_cpu_sq = 0.0f32;
        let mut norm_gpu_sq = 0.0f32;

        for i in 0..(m * n) {
            let gpu_f32 = gpu_result[i].to_f32();
            max_diff = max_diff.max((cpu_ref[i] - gpu_f32).abs());
            dot_cc += cpu_ref[i] * gpu_f32;
            norm_cpu_sq += cpu_ref[i].powi(2);
            norm_gpu_sq += gpu_f32.powi(2);
        }

        let cosine_sim = dot_cc / (norm_cpu_sq * norm_gpu_sq).sqrt();

        println!(
            "  N={:4}: max_diff={:.2}, cosine_sim={:.6}",
            n, max_diff, cosine_sim
        );
    }

    // Don't assert - just report the boundary
    Ok(())
}

/// Test 5: Check whether only first tile (cols 0-63) is correct with large K
// @lat: [[tests/gemm_compare#Gemm Compare Test#Multi-Tile GEMM Bug]]
#[test]
fn test_tile_boundary() -> anyhow::Result<()> {
    let m = 19usize;
    let n = 128usize; // exactly 2 tiles
    let k = 5120usize;

    println!("\n=== Test 5: Tile boundary check (N={}, K={}) ===", n, k);

    let input_host: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(0.01 * (i as f32 % 100.0 + 1.0)))
        .collect();

    let weight_host: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(0.01 * (i as f32 % 50.0 + 0.5)))
        .collect();

    let cpu_ref = bf16_gemm_cpu(&input_host, &weight_host, m, n, k);

    let (_ctx, stream, kernels) = create_context()?;
    let mut gemm_gpu = stream.alloc_zeros::<bf16>(m * n)?;
    let input_gpu = stream.clone_htod(&input_host)?;
    let weight_gpu = stream.clone_htod(&weight_host)?;

    kernels.launch_bf16_gemm_tiled(
        &stream,
        &mut gemm_gpu,
        &input_gpu,
        &weight_gpu,
        m as u32,
        n as u32,
        k as u32,
    )?;

    stream.synchronize()?;
    let gpu_result: Vec<bf16> = stream.clone_dtoh(&gemm_gpu)?;

    // Check first tile (cols 0-63) vs second tile (cols 64-127)
    let mut tile1_max_diff = 0.0f32;
    let mut tile2_max_diff = 0.0f32;
    for i in 0..m {
        // Tile 1: cols 0-63
        for j in 0..64 {
            let idx = i * n + j;
            let diff = (cpu_ref[idx] - gpu_result[idx].to_f32()).abs();
            tile1_max_diff = tile1_max_diff.max(diff);
        }
        // Tile 2: cols 64-127
        for j in 64..128 {
            let idx = i * n + j;
            let diff = (cpu_ref[idx] - gpu_result[idx].to_f32()).abs();
            tile2_max_diff = tile2_max_diff.max(diff);
        }
    }

    println!("  Tile 1 (cols 0-63) max diff: {:.4}", tile1_max_diff);
    println!("  Tile 2 (cols 64-127) max diff: {:.4}", tile2_max_diff);

    // Sample a few specific elements from each tile
    for i in [0, 5, 18] {
        for j in [0, 31, 63, 64, 95, 127] {
            let idx = i * n + j;
            println!(
                "  C[{}][{}] = cpu:{:.4} gpu:{:.4} diff:{:.4}",
                i, j, cpu_ref[idx], gpu_result[idx].to_f32(), (cpu_ref[idx] - gpu_result[idx].to_f32()).abs()
            );
        }
    }

    Ok(())
}

/// Test 6: Check if the bug is specific to large K — test with small K but multi-tile N
#[test]
fn test_tile_boundary_small_k() -> anyhow::Result<()> {
    let m = 4usize;
    let n = 128usize; // exactly 2 tiles
    let k = 16usize;

    println!("\n=== Test 6: Tile boundary with small K (N={}, K={}) ===", n, k);

    let input_host: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(0.1 * (i as f32 + 1.0)))
        .collect();

    let weight_host: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(0.5 + 0.1 * i as f32))
        .collect();

    let cpu_ref = bf16_gemm_cpu(&input_host, &weight_host, m, n, k);

    let (_ctx, stream, kernels) = create_context()?;
    let mut gemm_gpu = stream.alloc_zeros::<bf16>(m * n)?;
    let input_gpu = stream.clone_htod(&input_host)?;
    let weight_gpu = stream.clone_htod(&weight_host)?;

    kernels.launch_bf16_gemm_tiled(
        &stream,
        &mut gemm_gpu,
        &input_gpu,
        &weight_gpu,
        m as u32,
        n as u32,
        k as u32,
    )?;

    stream.synchronize()?;
    let gpu_result: Vec<bf16> = stream.clone_dtoh(&gemm_gpu)?;

    // Check first tile vs second tile
    let mut tile1_max_diff = 0.0f32;
    let mut tile2_max_diff = 0.0f32;
    for i in 0..m {
        for j in 0..64 {
            let idx = i * n + j;
            let diff = (cpu_ref[idx] - gpu_result[idx].to_f32()).abs();
            tile1_max_diff = tile1_max_diff.max(diff);
        }
        for j in 64..128 {
            let idx = i * n + j;
            let diff = (cpu_ref[idx] - gpu_result[idx].to_f32()).abs();
            tile2_max_diff = tile2_max_diff.max(diff);
        }
    }

    println!("  Tile 1 (cols 0-63) max diff: {:.4}", tile1_max_diff);
    println!("  Tile 2 (cols 64-127) max diff: {:.4}", tile2_max_diff);

    // Check specific elements
    for i in [0, m-1] {
        for j in [0, 31, 63, 64, 95, 127] {
            let idx = i * n + j;
            println!(
                "  C[{}][{}] = cpu:{:.4} gpu:{:.4} diff:{:.4}",
                i, j, cpu_ref[idx], gpu_result[idx].to_f32(), (cpu_ref[idx] - gpu_result[idx].to_f32()).abs()
            );
        }
    }

    Ok(())
}

/// Test 7: Compare GPU dequant + GEMM pipeline against Python reference for layer 0 in_proj_qkv.
/// Uses actual model weights from the safetensors file (saved as raw binary by Python).
/// This is the end-to-end validation: dequant(NVFP4) → bf16_gemm_tiled → compare output.
// @lat: [[tests/nvfp4_ref_compare#Python Reference Compare]]
#[test]
fn test_python_ref_compare() -> anyhow::Result<()> {
    use std::fs;

    let n: usize = 10240;
    let k: usize = 5120;
    let group_size: usize = 16;
    let m: usize = 1;

    // ── Load raw binary reference data from /tmp ──────────────
    let weight_packed_bytes = fs::read("/tmp/ref_weight_packed.bin")
        .map_err(|e| anyhow::anyhow!("Failed to read ref_weight_packed.bin: {}", e))?;
    let weight_scale_bytes = fs::read("/tmp/ref_weight_scale.bin")
        .map_err(|e| anyhow::anyhow!("Failed to read ref_weight_scale.bin: {}", e))?;
    let global_scale_bytes = fs::read("/tmp/ref_weight_global_scale.f32")
        .map_err(|e| anyhow::anyhow!("Failed to read ref_weight_global_scale.f32: {}", e))?;
    let input_bytes = fs::read("/tmp/ref_input.bf16")
        .map_err(|e| anyhow::anyhow!("Failed to read ref_input.bf16: {}", e))?;
    let ref_output_bytes = fs::read("/tmp/ref_output.bf16")
        .map_err(|e| anyhow::anyhow!("Failed to read ref_output.bf16: {}", e))?;
    let ref_dequant_bytes = fs::read("/tmp/ref_dequant_weight.bf16")
        .map_err(|e| anyhow::anyhow!("Failed to read ref_dequant_weight.bf16: {}", e))?;

    // Parse global scale
    let weight_global_scale = f32::from_le_bytes(
        global_scale_bytes[..4].try_into().unwrap()
    );

    // Assert expected sizes
    assert_eq!(weight_packed_bytes.len(), n * k / 2, "weight_packed size mismatch");
    assert_eq!(weight_scale_bytes.len(), n * (k / group_size), "weight_scale size mismatch");
    assert_eq!(input_bytes.len(), m * k * 2, "input bf16 size mismatch");
    assert_eq!(ref_output_bytes.len(), m * n * 2, "ref output bf16 size mismatch");
    assert_eq!(ref_dequant_bytes.len(), n * k * 2, "ref dequant weight bf16 size mismatch");

    // Convert input bf16 bytes to Vec<bf16>
    let input_host: Vec<bf16> = input_bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();
    assert_eq!(input_host.len(), m * k);

    // Convert ref output bf16 bytes to Vec<bf16>
    let ref_output: Vec<bf16> = ref_output_bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();
    assert_eq!(ref_output.len(), m * n);

    // Convert ref dequant weight bf16 bytes to Vec<bf16>
    let ref_dequant: Vec<bf16> = ref_dequant_bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();
    assert_eq!(ref_dequant.len(), n * k);

    println!("\n=== Test 7: Python reference compare (layer 0 in_proj_qkv) ===");
    println!("  N={}, K={}, M={}, group_size={}", n, k, m, group_size);
    println!("  weight_global_scale={}", weight_global_scale);
    println!("  ref_input[0,:5]:  {:?}", input_host[..5].iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    println!("  ref_output[0,:5]: {:?}", ref_output[..5].iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    println!("  ref_dequant[0,:5]: {:?}", ref_dequant[..5].iter().map(|v| v.to_f32()).collect::<Vec<_>>());

    // ── GPU pipeline ──────────────────────────────────────────
    let (_ctx, stream, kernels) = create_context()?;

    // Step 1: Dequant NVFP4 → bf16
    let packed_gpu = stream.clone_htod(&weight_packed_bytes)?;
    let scale_gpu = stream.clone_htod(&weight_scale_bytes)?;
    let mut dequant_gpu = stream.alloc_zeros::<bf16>(n * k)?;

    kernels.launch_nvfp4_dequant_to_bf16(
        &stream,
        &mut dequant_gpu,
        &packed_gpu,
        &scale_gpu,
        weight_global_scale,
        n as u32,
        k as u32,
        group_size as u32,
    )?;
    stream.synchronize()?;

    // Download GPU dequant for comparison
    let gpu_dequant: Vec<bf16> = stream.clone_dtoh(&dequant_gpu)?;

    // Compare dequant weights
    {
        let mut max_diff = 0.0f64;
        let mut mean_diff = 0.0f64;
        let mut mismatches = 0usize;
        let mut nan_count = 0usize;
        for i in 0..(n * k) {
            let ref_f = ref_dequant[i].to_f32();
            let gpu_f = gpu_dequant[i].to_f32();
            if ref_f.is_nan() || gpu_f.is_nan() { nan_count += 1; continue; }
            let diff = (ref_f as f64 - gpu_f as f64).abs();
            max_diff = max_diff.max(diff);
            mean_diff += diff;
            if diff > 1e-5 { mismatches += 1; }
        }
        mean_diff /= (n * k) as f64;
        println!("\n  [Dequant] max_diff={:.8}, mean_diff={:.8}, mismatches={}/{}, nan_count={}",
                 max_diff, mean_diff, mismatches, n * k, nan_count);
        println!("  [Dequant] ref[0,:5]: {:?}", ref_dequant[..5].iter().map(|v| v.to_f32()).collect::<Vec<_>>());
        println!("  [Dequant] gpu[0,:5]: {:?}", gpu_dequant[..5].iter().map(|v| v.to_f32()).collect::<Vec<_>>());
    }

    // Step 2: GEMM — output = input @ dequant_weight^T
    let input_gpu = stream.clone_htod(&input_host)?;
    let mut output_gpu = stream.alloc_zeros::<bf16>(m * n)?;

    kernels.launch_bf16_gemm_tiled(
        &stream,
        &mut output_gpu,
        &input_gpu,
        &dequant_gpu,
        m as u32,
        n as u32,
        k as u32,
    )?;
    stream.synchronize()?;

    // Download GPU output
    let gpu_output: Vec<bf16> = stream.clone_dtoh(&output_gpu)?;

    // ── Compare pipeline output ───────────────────────────────
    {
        let mut max_abs_err = 0.0f64;
        let mut mean_abs_err = 0.0f64;
        let mut dot = 0.0f64;
        let mut norm_ref_sq = 0.0f64;
        let mut norm_gpu_sq = 0.0f64;
        let mut nan_count = 0usize;

        for i in 0..(m * n) {
            let ref_f = ref_output[i].to_f32();
            let gpu_f = gpu_output[i].to_f32();
            if ref_f.is_nan() || gpu_f.is_nan() { nan_count += 1; continue; }
            let abs_err = (ref_f as f64 - gpu_f as f64).abs();
            max_abs_err = max_abs_err.max(abs_err);
            mean_abs_err += abs_err;
            dot += ref_f as f64 * gpu_f as f64;
            norm_ref_sq += (ref_f as f64) * (ref_f as f64);
            norm_gpu_sq += (gpu_f as f64) * (gpu_f as f64);
        }
        mean_abs_err /= (m * n) as f64;
        let cos_sim = if norm_ref_sq > 0.0 && norm_gpu_sq > 0.0 {
            dot / (norm_ref_sq * norm_gpu_sq).sqrt()
        } else { 0.0 };

        println!("\n  [Pipeline Output]");
        println!("    Cosine similarity: {:.8}", cos_sim);
        println!("    Max absolute error: {:.8}", max_abs_err);
        println!("    Mean absolute error: {:.8}", mean_abs_err);
        println!("    NaN count: {}", nan_count);
        println!("    ref[0,:10]:  {:?}", ref_output[..10].iter().map(|v| v.to_f32()).collect::<Vec<_>>());
        println!("    gpu[0,:10]:  {:?}", gpu_output[..10].iter().map(|v| v.to_f32()).collect::<Vec<_>>());

    }
    Ok(())
}
