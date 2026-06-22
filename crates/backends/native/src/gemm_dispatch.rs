//! Cached GEMM dispatch for INT4 and BF16 weights.
//!
//! Provides `gemm_projection_cached` which dispatches GEMM using GPU-resident
//! weight buffers from `GpuWeightCache`, avoiding per-call CPU→GPU uploads.
//! Handles both BF16 contiguous weights and INT4 quantized triplets in a
//! single call. For INT4 weights, the INT4 GEMM kernel performs on-the-fly
//! per-group dequantization in registers.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_cuda::{CudaSlice, CudaStream, OxideKernels};

/// Dispatch a single projection GEMM using cached GPU-resident weights.
///
/// Looks up the weight by name from the `GpuWeightCache`. For BF16 weights,
/// calls `gemm.matmul_bf16` directly. For INT4 weights, dispatches through
/// `OxideKernels::launch_int4_gemm_auto_round` using the cached qweight/scales/qzeros buffers.
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
    match cache.get(weight_name) {
        Some(crate::gpu_cache::CachedWeight::Bf16(weight_gpu)) => {
            // cuBLASLt uses column-major storage convention.
            // Our input/weight are row-major [M,K] and [N,K].
            //
            // The naive approach would be:
            //   gemm.matmul_bf16(config(m,n,k, transa=true,transb=false), input, weight, output)
            // which computes: C = input^T @ weight = (input[M,K] as [K,M])^T @ (weight[N,K] as [K,N])
            // = [M,K] @ [K,N] = [M,N]. This is correct MATHEMATICALLY.
            //
            // BUT cuBLASLt writes C in COLUMN-major: C(m,n) at offset m + n*M.
            // Our code reads the flat output buffer as ROW-major: C[m][n] at offset m*N + n.
            // These only agree at (0,0) — all other elements are WRONG.
            //
            // The fix: swap arguments AND swap output dimensions.
            //   gemm.matmul_bf16(config(m=N,n=M,k=K, transa=true,transb=false), weight, input, output)
            // This computes: C' = weight^T @ input = [N,K]^T @ [M,K]^T = (weight[N,K] as [K,N])^T @ (input[M,K] as [K,M])
            // = [K,N]^T @ [K,M] = [N,K] @ [K,M] = [N,M]
            // C'(n,m) = sum_k weight[n][k] * input[m][k] = sum_k input[m][k] * weight[n][k] = C[m][n]
            //
            // cuBLASLt writes C' in column-major [N,M] with ldc=C'(row=n at offset C'(n,m) at offset n + m*N.
            // Reading row-major [M,N]: buffer[m*N + n] = n + m*N = C'(n,m) = C[m][n]. ✓
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
                weight_gpu,  // FIRST arg — transposed view: [K, N] → op(A) = [N, K]
                input,       // SECOND arg — transposed view: [K, M] → op(B) = [K, M]
                output,
            )?;
        }
        Some(crate::gpu_cache::CachedWeight::Int4(int4_bufs)) => {
            // Determine transposition from shape and K dimension — same logic as gemm_projection
            let is_transposed = int4_bufs.shape.len() >= 2 && int4_bufs.shape[0] * 8 == k;

            // Scales are stored as f16 (IEEE half) but the kernel reads raw u16 bits.
            // bf16 and f16 have identical memory layout (both are repr(transparent) over u16),
            // so we can safely transmute the reference for the bridge call.
            let scales_bf16: &CudaSlice<bf16> = unsafe {
                &*(std::ptr::addr_of!(int4_bufs.scales) as *const CudaSlice<bf16>)
            };

            oxide.launch_int4_gemm_auto_round(
                stream, output, &int4_bufs.qweight, scales_bf16, &int4_bufs.qzeros, input,
                m as u32, n as u32, k as u32, group_size as u32, is_transposed as u32,
            )?;
        }
        None => {
            anyhow::bail!("Weight '{}' not found in GpuWeightCache", weight_name);
        }
    }

    Ok(output.clone())
}
