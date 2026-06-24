//! Cached GEMM dispatch for INT4 and BF16 weights.
//!
//! Provides `gemm_projection_cached` which dispatches GEMM using GPU-resident
//! weight buffers from `GpuWeightCache`, avoiding per-call CPU→GPU uploads.
//! Handles both BF16 contiguous weights and INT4 quantized triplets in a
//! single call. For INT4, uses the fused `int4_gemm_auto_round` kernel
//! (no dequantize buffer needed). NVFP4 still dequantizes to bf16 on GPU then
//! dispatches via `bf16_gemm_tiled` (bypassing cuBLAS).

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_cuda::{CudaSlice, CudaStream, OxideKernels};

/// Dispatch a single projection GEMM using cached GPU-resident weights.
/// Looks up the weight by name from the `GpuWeightCache`. For BF16 weights,
/// calls `gemm.matmul_bf16` directly. For INT4, uses the fused
/// `int4_gemm_auto_round` kernel. NVFP4 dequantizes to bf16 on GPU then
/// dispatches via `bf16_gemm_tiled` (bypassing cuBLAS).
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `oxide` — Oxide bridge for INT4 GEMM dispatch
/// * `stream` — CUDA stream
/// * `cache` — GPU weight cache
/// * `weight_name` — Name of the weight to look up (same as WeightData.name)
/// * `input` — Input activations `[M × K]`
/// * `output` — Output buffer `[M × N]` (pre-allocated)
/// * `m` — Batch/sequence dimension
/// * `n` — Output feature dimension
/// * `k` — Inner dimension
/// * `group_size` — INT4 group size
///
/// # Returns
/// The output buffer (same `output` that was passed in).
pub fn gemm_projection_cached(
    gemm: &mut GemmEngine,
    oxide: &OxideKernels,
    stream: &Arc<CudaStream>,
    cache: &crate::gpu_cache::GpuWeightCache,
    weight_name: &str,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<CudaSlice<bf16>> {
    // Gate eprintln behind INFERS_DEBUG env var — only prints once at first call.
    static DEBUG_GEMM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let debug = *DEBUG_GEMM.get_or_init(|| std::env::var("INFERS_DEBUG").is_ok());

    match cache.get(weight_name) {
        Some(crate::gpu_cache::CachedWeight::Bf16(weight_gpu)) => {
            if debug { eprintln!("[GEMM-DISPATCH] Bf16 weight '{}': len={}", weight_name, weight_gpu.len()); }
            gemm.matmul_bf16(
                &GemmConfig {
                    m: n,
                    n: m,
                    k,
                    transa: true,
                    transb: false,
                    alpha: 1.0,
                    beta: 0.0,
                    lda: None,
                    ldb: None,
                    ldc: None,
                    activation: None,
                },
                weight_gpu,
                input,
                output,
            )?;
        }
        Some(crate::gpu_cache::CachedWeight::Int4(int4_bufs)) => {
            if debug { eprintln!("[GEMM-DISPATCH] Int4 weight '{}': n={}, k={}", weight_name, n, k); }

            // Compute actual batch/sequence dimension from input
            let m = input.len() / k;

            // Determine transposition from weight shape: [K/8, N] is transposed layout.
            let transposed: u32 = if int4_bufs.shape.len() >= 2 {
                // If dim0 != n (N), the weight is in transposed [K/8, N] layout
                if int4_bufs.shape[1] == n { 1 } else { 0 }
            } else { 1 }; // default to transposed (AutoRound convention)

            oxide.launch_int4_gemm_auto_round(
                stream,
                output,
                &int4_bufs.qweight,
                &int4_bufs.scales,
                &int4_bufs.qzeros,
                input,
                m as u32,
                n as u32,
                k as u32,
                group_size as u32,
                transposed,
            )?;
        }
        Some(crate::gpu_cache::CachedWeight::Nvfp4(nvfp4_bufs)) => {
            // NVFP4 group_size is always 16 (fixed property of the format)
            const NVFP4_GROUP_SIZE: u32 = 16;

            // 1. Allocate bf16 buffer for dequantized weights: [N, K]
            let mut dequant_buf = stream.alloc_zeros::<bf16>(n * k)?;

            // 2. Launch NVFP4 dequant kernel
            oxide.launch_nvfp4_dequant_to_bf16(
                stream,
                &mut dequant_buf,
                &nvfp4_bufs.weight_packed,
                &nvfp4_bufs.weight_scale,
                nvfp4_bufs.weight_global_scale,
                n as u32,
                k as u32,
                NVFP4_GROUP_SIZE,
            )?;

            // 3. Sanitize NaN values in dequant buffer (stale GPU memory may contain NaN)
            oxide.launch_sanitize_nan_bf16(stream, &mut dequant_buf)?;

            // 4. Compute actual batch/sequence dimension from input
            let m = input.len() / k;

            // 5. bf16 tiled GEMM: output = input @ dequant_buf^T
            //    Bypasses cuBLAS to avoid workspace corruption with NVFP4 dequant buffers.
            oxide.launch_bf16_gemm_tiled(
                stream,
                output,
                input,
                &dequant_buf,
                m as u32,
                n as u32,
                k as u32,
            )?;
        }
        None => {
            if debug { eprintln!("[GEMM-DISPATCH] Weight '{}' NOT FOUND in cache", weight_name); }
            anyhow::bail!("Weight '{}' not found in GpuWeightCache", weight_name);
        }
    }

    Ok(output.clone())
}
