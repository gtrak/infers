//! Cached GEMM dispatch for INT4 and BF16 weights.
//!
//! Provides `gemm_projection_cached` which dispatches GEMM using GPU-resident
//! weight buffers from `GpuWeightCache`, avoiding per-call CPU→GPU uploads.
//! Handles both BF16 contiguous weights and INT4 quantized triplets in a
//! single call. For INT4, uses the fused `int4_gemm_auto_round` kernel
//! (no dequantize buffer needed). NVFP4 uses the fused `nvfp4_gemm_fused`
//! kernel — dequantizes FP4 in registers, no intermediate bf16 buffer.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_cuda::{CudaSlice, CudaStream, OxideKernels};

/// Dispatch a single projection GEMM using cached GPU-resident weights.
/// Looks up the weight by name from the `GpuWeightCache`. For BF16 weights,
/// calls `gemm.matmul_bf16` directly. For INT4, uses the fused
/// `int4_gemm_auto_round` kernel. NVFP4 uses the fused `nvfp4_gemm_fused`
/// kernel — dequantizes FP4 in registers, no intermediate bf16 buffer.
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
/// * `partial_sums_buf` — Optional pre-allocated f32 buffer for K-split partial sums (M=1 path). 
///   If provided and large enough, it's reused instead of allocating per-GEMM. The ksplit kernels 
///   write every position unconditionally, so no memset is needed even on fallback alloc.
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
    partial_sums_buf: &mut Option<&mut CudaSlice<f32>>,
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

            if m == 1 {
                // K-split GEMV: split K across blocks for better occupancy.
                // v3 kernel ceil-groups full quantization groups across splits, so any
                // K_SPLIT is correct regardless of divisibility (40, 136, 9, 34 groups
                // all handled). K_SPLIT=20 gives ~1.25 waves on 40 SMs for the main
                // hidden=5120 layers (2 groups/split, 1600 blocks).
                const K_SPLIT: u32 = 20;
                let required_len = K_SPLIT as usize * n;
                // Use the provided buffer if large enough; else fall back to unsafe alloc (no memset needed — kernel writes every position).
                let mut local_ps_owner: Option<CudaSlice<f32>> = None;
                let partial_sums: &mut CudaSlice<f32> = if let Some(buf) = partial_sums_buf.as_mut() {
                    if buf.len() >= required_len {
                        buf
                    } else {
                        local_ps_owner = Some(unsafe { stream.alloc::<f32>(required_len)? });
                        local_ps_owner.as_mut().unwrap()
                    }
                } else {
                    local_ps_owner = Some(unsafe { stream.alloc::<f32>(required_len)? });
                    local_ps_owner.as_mut().unwrap()
                };
                oxide.launch_int4_gemm_v3_ksplit_sm(
                    stream, &oxide.cc_stream(), partial_sums,
                    &int4_bufs.qweight, &int4_bufs.scales, &int4_bufs.qzeros,
                    input, n as u32, k as u32, group_size as u32, transposed, K_SPLIT,
                )?;
                oxide.launch_reduce_partial_sums_bf16(
                    stream, &oxide.cc_stream(), output, partial_sums, n as u32, K_SPLIT,
                )?;
            } else {
                oxide.launch_int4_gemm_auto_round(
                    stream, &oxide.cc_stream(),
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
        }
        Some(crate::gpu_cache::CachedWeight::Nvfp4(nvfp4_bufs)) => {
            // NVFP4 group_size is always 16 (fixed property of the format)
            const NVFP4_GROUP_SIZE: u32 = 16;

            if debug { eprintln!("[GEMM-DISPATCH] Nvfp4 weight '{}': n={}, k={}", weight_name, n, k); }

            // Compute actual batch/sequence dimension from input
            let m = input.len() / k;

            if m == 1 {
                // K-split for M=1: v3 with 4 accumulators + ceil-grouped K-split + 2-u32 stride
                const K_SPLIT: u32 = 20;
                let required_len = K_SPLIT as usize * n;
                // Use the provided buffer if large enough; else fall back to unsafe alloc (no memset needed — kernel writes every position).
                let mut local_ps_owner: Option<CudaSlice<f32>> = None;
                let partial_sums: &mut CudaSlice<f32> = if let Some(buf) = partial_sums_buf.as_mut() {
                    if buf.len() >= required_len {
                        buf
                    } else {
                        local_ps_owner = Some(unsafe { stream.alloc::<f32>(required_len)? });
                        local_ps_owner.as_mut().unwrap()
                    }
                } else {
                    local_ps_owner = Some(unsafe { stream.alloc::<f32>(required_len)? });
                    local_ps_owner.as_mut().unwrap()
                };
                oxide.launch_nvfp4_gemm_v3_ksplit(
                    stream, &oxide.cc_stream(), partial_sums,
                    &nvfp4_bufs.weight_packed, &nvfp4_bufs.weight_scale,
                    input, nvfp4_bufs.weight_global_scale,
                    n as u32, k as u32, NVFP4_GROUP_SIZE, K_SPLIT,
                )?;
                oxide.launch_reduce_partial_sums_bf16(
                    stream, &oxide.cc_stream(), output, partial_sums, n as u32, K_SPLIT,
                )?;
            } else {
                // Fused NVFP4 GEMM: dequant FP4 in registers and multiply — no intermediate buffer
                oxide.launch_nvfp4_gemm_fused(
                    stream, &oxide.cc_stream(),
                    output,
                    &nvfp4_bufs.weight_packed,
                    &nvfp4_bufs.weight_scale,
                    input,
                    nvfp4_bufs.weight_global_scale,
                    m as u32,
                    n as u32,
                    k as u32,
                    NVFP4_GROUP_SIZE,
                )?;
            }
        }
        None => {
            if debug { eprintln!("[GEMM-DISPATCH] Weight '{}' NOT FOUND in cache", weight_name); }
            anyhow::bail!("Weight '{}' not found in GpuWeightCache", weight_name);
        }
    }

    Ok(output.clone())
}
