//! Compare bf16_gemm_tiled vs cuBLAS on known small data.
//!
//! Verifies that both kernels produce numerically equivalent results
//! for C = A @ B^T where all buffers are row-major BF16.

use half::bf16;
use std::sync::Arc;

use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_cuda::OxideKernels;
use cudarc::driver::CudaContext;

#[test]
// @lat: [[tests/gemm_compare#Gemm Compare Test]]
fn bf16_gemm_tiled_vs_cublas() -> anyhow::Result<()> {
    // ── 1. Create CUDA context and stream ────────────────────────
    let ctx = Arc::new(CudaContext::new(0)?);
    let stream: Arc<_> = ctx.new_stream()?;

    // ── 2. Load kernels ──────────────────────────────────────────
    let kernels = OxideKernels::new(
        0,
        "/home/gary/dev/infers/crates/cuda/kernels/compiled/oxide_kernels.cubin",
    )?;

    // ── 3. Create GemmEngine ─────────────────────────────────────
    let gemm = GemmEngine::new(stream.clone())?;

    // ── 4. Define dimensions and host data ────────────────────────
    // A (input) [M=4, K=16], B (weight) [N=8, K=16]
    let m = 4usize;
    let n = 8usize;
    let k = 16usize;

    let a_host: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(0.1 * (i as f32 + 1.0)))
        .collect(); // 0.1, 0.2, 0.3, ..., 1.6

    let b_host: Vec<bf16> = (0..n * k)
        .map(|i| bf16::from_f32(0.5 + 0.1 * i as f32))
        .collect(); // 0.5, 0.6, 0.7, ..., 1.29

    let c_size = m * n;

    // ── 5. Allocate GPU memory for outputs ───────────────────────
    let mut c_cublas = stream.alloc_zeros::<bf16>(c_size)?;
    let mut c_tiled = stream.alloc_zeros::<bf16>(c_size)?;

    // ── 6. Copy host data to GPU ─────────────────────────────────
    let a_gpu = stream.clone_htod(&a_host)?;
    let b_gpu = stream.clone_htod(&b_host)?;

    // ── 7. Run cuBLAS GEMM: C = A @ B^T ─────────────────────────
    // With transa=true and transb=false, passing weight as A and input as B:
    //   C[m=n=8, n=m=4] = op(B) * op(A)
    //   where B is [N=8, K=16], so B^T = [K=16, N=8] → but we need [M,K]=[8,16]
    // Actually the config matches: m=8 (rows of C), n=4 (cols of C), k=16
    // A=b_gpu[N=8, K=16], transa=true → op(A)=A^T → [K=16, N=8]
    //   Wait, this is confusing. Let me just use the config from the task spec.
    gemm.matmul_bf16(
        &GemmConfig {
            m: n,      // 8
            n: m,      // 4
            k,         // 16
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        &b_gpu,      // A argument (weight)
        &a_gpu,      // B argument (input)
        &mut c_cublas,
    )?;

    // ── 8. Run bf16_gemm_tiled: C = A @ B^T ─────────────────────
    kernels.launch_bf16_gemm_tiled(&stream, &mut c_tiled, &a_gpu, &b_gpu, m as u32, n as u32, k as u32)?;

    // ── 9. Synchronize ───────────────────────────────────────────
    stream.synchronize()?;

    // ── 10. Copy results back to CPU ─────────────────────────────
    let c_cublas_host: Vec<bf16> = stream.clone_dtoh(&c_cublas)?;
    let c_tiled_host: Vec<bf16> = stream.clone_dtoh(&c_tiled)?;

    // ── Compute ground truth in fp32 ─────────────────────────────
    let mut c_gold: Vec<f32> = vec![0.0; c_size];
    for i in 0..m {
        for j in 0..n {
            for k_idx in 0..k {
                c_gold[i * n + j] += a_host[i * k + k_idx].to_f32()
                    * b_host[j * k + k_idx].to_f32();
            }
        }
    }

    // ── 11. Print all values for visual inspection ────────────────
    println!("=== cuBLAS result [{}] ===", c_size);
    for i in 0..c_size {
        let row = i / n;
        let col = i % n;
        println!(
            "  C_cublas[{}][{}] = {:.4}",
            row,
            col,
            c_cublas_host[i].to_f32()
        );
    }
    println!("=== tiled GEMM result [{}] ===", c_size);
    for i in 0..c_size {
        let row = i / n;
        let col = i % n;
        println!(
            "  C_tiled[{}][{}] = {:.4}",
            row,
            col,
            c_tiled_host[i].to_f32()
        );
    }
    println!("=== Difference (cublas - tiled) ===");
    for i in 0..c_size {
        let row = i / n;
        let col = i % n;
        let diff = c_cublas_host[i].to_f32() - c_tiled_host[i].to_f32();
        println!(
            "  diff[{}][{}] = {:.4}",
            row,
            col,
            diff
        );
    }
    println!("=== Ground truth (fp32) [{}] ===", c_size);
    for i in 0..c_size {
        let row = i / n;
        let col = i % n;
        println!(
            "  C_gold[{}][{}] = {:.4}",
            row,
            col,
            c_gold[i]
        );
    }

    // ── 12. Compute error metrics ────────────────────────────────
    let mut max_abs_err: f32 = 0.0;
    let mut sum_abs_err: f32 = 0.0;
    let mut dot_cc: f32 = 0.0;
    let mut norm_cublas_sq: f32 = 0.0;
    let mut norm_tiled_sq: f32 = 0.0;

    for i in 0..c_size {
        let diff = (c_cublas_host[i].to_f32() - c_tiled_host[i].to_f32()).abs();
        max_abs_err = max_abs_err.max(diff);
        sum_abs_err += diff;

        dot_cc += c_cublas_host[i].to_f32() * c_tiled_host[i].to_f32();
        norm_cublas_sq += c_cublas_host[i].to_f32().powi(2);
        norm_tiled_sq += c_tiled_host[i].to_f32().powi(2);
    }

    let mean_abs_err = sum_abs_err / c_size as f32;
    let cosine_sim = dot_cc / (norm_cublas_sq * norm_tiled_sq).sqrt();

    // Compare each against ground truth too
    let mut cublas_max_gold: f32 = 0.0;
    let mut tiled_max_gold: f32 = 0.0;
    for i in 0..c_size {
        cublas_max_gold = cublas_max_gold.max((c_cublas_host[i].to_f32() - c_gold[i]).abs());
        tiled_max_gold = tiled_max_gold.max((c_tiled_host[i].to_f32() - c_gold[i]).abs());
    }

    println!("=== Error metrics ===");
    println!("  Max absolute error (cublas vs tiled): {:.6}", max_abs_err);
    println!("  Mean absolute error: {:.6}", mean_abs_err);
    println!("  Cosine similarity: {:.6}", cosine_sim);
    println!("  Max cublas vs gold: {:.6}", cublas_max_gold);
    println!("  Max tiled vs gold: {:.6}", tiled_max_gold);

    // ── 13. Assert bf16-level precision ───────────────────────────
    assert!(
        max_abs_err < 0.01,
        "Max absolute error {:.6} exceeds bf16 tolerance of 0.01",
        max_abs_err
    );

    Ok(())
}
