//! GEMM dispatch for INT4 and BF16 weights.
//!
//! Provides `gemm_projection` which handles both BF16 contiguous weights
//! and INT4 quantized triplets (qweight + scales + qzeros) in a single
//! function call. For INT4 weights, the INT4 GEMM kernel performs
//! on-the-fly per-group dequantization in registers.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::gemm::{GemmConfig, GemmEngine, Int4GemmConfig};
use infers_cuda::{CudaFunction, CudaSlice, CudaStream};
use infers_model::{Int4Companions, WeightData, WeightDtype};

/// Dispatch a single projection GEMM, handling both BF16 and INT4 weights.
///
/// For **BF16/FP16/FP32** weights: uploads the weight as BF16 and calls `gemm.matmul_bf16`.
/// For **INT4** weights: looks up companion tensors (scales + qzeros) from
/// `int4_companions`, uploads all three buffers, and calls `matmul_int4`
/// with the INT4 GEMM kernel.
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `int4_kernel` — INT4 GEMM kernel handle (only used for INT4 weights)
/// * `stream` — CUDA stream
/// * `weight` — Weight data (BF16 or INT4 packed)
/// * `input` — Input activations `[M × K]`
/// * `output` — Output buffer `[M × N]`
/// * `m` — Batch/sequence dimension (rows of output)
/// * `n` — Output feature dimension (columns of output)
/// * `k` — Inner dimension
/// * `group_size` — INT4 group size (default 128)
/// * `int4_companions` — Companion tensor map from WeightRegistry
///
/// # Returns
/// The output buffer (same `output` that was passed in).
pub fn gemm_projection(
    gemm: &mut GemmEngine,
    int4_kernel: &CudaFunction,
    stream: &Arc<CudaStream>,
    weight: &WeightData,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    int4_companions: &HashMap<String, Int4Companions>,
) -> Result<CudaSlice<bf16>> {
    match weight.dtype {
        WeightDtype::Bf16 | WeightDtype::Fp16 | WeightDtype::Fp32 => {
            // Upload BF16 weight and perform GEMM
            let weight_gpu = crate::upload::upload_weight(stream, weight)?;
            gemm.matmul_bf16(
                &GemmConfig {
                    m,
                    n,
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
                input,
                &weight_gpu,
                output,
            )?;
        }
        WeightDtype::Int4Packed => {
            // Look up INT4 companions and perform INT4 GEMM
            let companions = int4_companions
                .get(&weight.name)
                .ok_or_else(|| anyhow::anyhow!("INT4 companions not found for weight '{}'", weight.name))?;

            let (qweight_gpu, scales_gpu, qzeros_gpu) = crate::upload::upload_int4_weight(
                stream, &weight, &companions.scales, &companions.qzeros,
            )?;

            let is_transposed = weight.shape.len() >= 2 && weight.shape[0] * 8 == k;

            infers_cuda::gemm::matmul_int4(
                stream,
                int4_kernel,
                &Int4GemmConfig { m, n, k, group_size, transposed: is_transposed },
                output,
                &qweight_gpu,
                &scales_gpu,
                &qzeros_gpu,
                input,
            )?;
        }
        other => {
            anyhow::bail!(
                "gemm_projection does not support dtype {:?} for weight '{}'. \
                 Only BF16, FP16, FP32, and Int4Packed are supported.",
                other, weight.name
            );
        }
    }

    Ok(output.clone())
}
