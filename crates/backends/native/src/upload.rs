//! Weight upload utilities.
//!
//! Converts `WeightData` raw bytes to GPU-resident buffers in various formats:
//! - BF16 / FP16 / FP32 → contiguous `CudaSlice<bf16>` (converted to BF16 on GPU)
//! - INT4 packed triplet (qweight + scales + qzeros) → separate GPU buffers

use std::sync::Arc;

use anyhow::Result;
use half::{bf16, f16};
use infers_cuda::{CudaSlice, CudaStream};
use infers_model::{WeightData, WeightDtype};

// ---------------------------------------------------------------------------
// Contiguous-weight upload (BF16, FP16, FP32)
// ---------------------------------------------------------------------------

/// Convert `WeightData` bytes to a GPU-resident BF16 buffer.
///
/// Handles multiple source dtypes:
/// - **Bf16**: 2 bytes → `bf16::from_bits` (direct, no conversion)
/// - **Fp16**: 2 bytes → `f16::from_bits` → `bf16` cast
/// - **Fp32**: 4 bytes → `f32::from_le_bytes` → `bf16` cast
/// - Other formats return an error (use `upload_int4_weight` for INT4).
///
/// The weight data stays as bytes on CPU until upload time to avoid
/// requiring GPU hardware at model load time.
pub fn upload_weight(
    stream: &Arc<CudaStream>,
    weight: &WeightData,
) -> Result<CudaSlice<bf16>> {
    let span = tracing::debug_span!("weight_upload", tensor = %weight.name, bytes = weight.data.len());
    let _enter = span.enter();
    let bf16_vec = bytes_to_bf16(&weight.data, weight.dtype)?;


    let gpu_slice = stream
        .clone_htod(&bf16_vec)
        .map_err(|e| anyhow::anyhow!("Failed to upload weight '{}': {}", weight.name, e))?;

    // CRITICAL FIX: Synchronize the stream after each upload to prevent cudaMallocAsync
    // from returning the same GPU address for consecutive allocations on this stream.
    // The async memory pool can return overlapping addresses if prior allocations are
    // still pending (copy not yet completed) when the next allocation is made.
    {
        let sync_span = tracing::debug_span!("cuda_sync", reason = "weight_upload");
        let _enter = sync_span.enter();
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("Failed to sync stream after uploading '{}': {}", weight.name, e))?;
    }

    Ok(gpu_slice)
}

/// Convert raw bytes to a `Vec<bf16>` based on the declared dtype.
///
/// This is the CPU-side conversion logic extracted from `upload_weight`
/// so it can be unit-tested without CUDA.
pub fn bytes_to_bf16(data: &[u8], dtype: WeightDtype) -> Result<Vec<bf16>> {
    match dtype {
        WeightDtype::Bf16 => {
            // 2 bytes per bf16 value, little-endian
            let count = data.len() / 2;
            let mut result = Vec::with_capacity(count);
            for chunk in data.chunks_exact(2) {
                result.push(bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
            Ok(result)
        }
        WeightDtype::Fp16 => {
            // 2 bytes per f16 value → convert to bf16
            let count = data.len() / 2;
            let mut result = Vec::with_capacity(count);
            for chunk in data.chunks_exact(2) {
                let f16_val = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
                result.push(bf16::from_f32(f16_val.to_f32()));
            }
            Ok(result)
        }
        WeightDtype::Fp32 => {
            // 4 bytes per f32 value → convert to bf16
            let count = data.len() / 4;
            let mut result = Vec::with_capacity(count);
            for chunk in data.chunks_exact(4) {
                let f32_val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                result.push(bf16::from_f32(f32_val));
            }
            Ok(result)
        }
        _ => {
            anyhow::bail!(
                "upload_weight does not support dtype {:?}. \
                 Use upload_int4_weight() for INT4-packed weights.",
                dtype
            )
        }
    }
}

/// Convert raw bytes to a `Vec<f16>` based on the declared dtype.
///
/// Preserves FP16 data as-is without converting to BF16, maintaining full
/// 10-bit mantissa precision for INT4 quantization scales.
// @lat: [[lat.md/lat#Forward Engine#INT4 Triplet Upload]]
pub fn bytes_to_fp16(data: &[u8], dtype: WeightDtype) -> Result<Vec<f16>> {
    match dtype {
        WeightDtype::Fp16 => {
            let count = data.len() / 2;
            let mut result = Vec::with_capacity(count);
            for chunk in data.chunks_exact(2) {
                result.push(f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
            Ok(result)
        }
        _ => {
            anyhow::bail!("bytes_to_fp16 only supports Fp16 dtype, got {:?}", dtype)
        }
    }
}

// ---------------------------------------------------------------------------
// INT4 triplet upload (qweight + scales + qzeros)
// ---------------------------------------------------------------------------

/// Upload an INT4 quantized weight triplet to GPU.
///
/// AutoRound / GPTQ-style INT4 stores weights as three separate buffers:
/// - **qweight**: Packed INT4 weights as `u32` (8 weights per u32).
/// - **scales**: FP16 group-wise scales (preserves full 10-bit mantissa precision).
/// - **qzeros**: Packed INT4 zero-points as `u32` (8 per u32). Stored values are offset by -1 (actual_zero = stored + 1).
///
/// The `int4_gemm_kernel` handles both standard [N, K/8] and transposed [K/8, N]
/// layouts natively, so weights are uploaded as-is without CPU transposition.
///
/// The kernel performs on-the-fly dequantization in registers,
/// so no dequantized copy is needed on GPU.
///
/// # Returns
/// `(qweight_gpu, scales_gpu, qzeros_gpu)` triple of GPU buffers.
// @lat: [[lat.md/lat#Forward Engine#INT4 Triplet Upload]]
pub fn upload_int4_weight(
    stream: &Arc<CudaStream>,
    qweight: &WeightData,
    scales: &WeightData,
    qzeros: &WeightData,
) -> Result<(CudaSlice<u32>, CudaSlice<f16>, CudaSlice<u32>)> {
    // Parse packed u32 data
    let qweight_u32: Vec<u32> = qweight
        .data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let scales_vec = bytes_to_fp16(&scales.data, scales.dtype)
        .map_err(|e| anyhow::anyhow!("Invalid scales data for '{}': {}", scales.name, e))?;
    let qzeros_u32: Vec<u32> = qzeros
        .data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let qweight_gpu = stream
        .clone_htod(&qweight_u32)
        .map_err(|e| anyhow::anyhow!("Failed to upload qweight '{}': {}", qweight.name, e))?;
    // CRITICAL: Sync after each upload to prevent cudaMallocAsync
    // from returning overlapping addresses on the same stream.
    {
        let sync_span = tracing::debug_span!("cuda_sync", reason = "int4_qweight");
        let _enter = sync_span.enter();
        stream.synchronize()
            .map_err(|e| anyhow::anyhow!("Failed to sync stream after qweight '{}': {}", qweight.name, e))?;
    }

    let scales_gpu = stream
        .clone_htod(&scales_vec)
        .map_err(|e| anyhow::anyhow!("Failed to upload scales '{}': {}", scales.name, e))?;
    {
        let sync_span = tracing::debug_span!("cuda_sync", reason = "int4_scales");
        let _enter = sync_span.enter();
        stream.synchronize()
            .map_err(|e| anyhow::anyhow!("Failed to sync stream after scales '{}': {}", scales.name, e))?;
    }

    let qzeros_gpu = stream
        .clone_htod(&qzeros_u32)
        .map_err(|e| anyhow::anyhow!("Failed to upload qzeros '{}': {}", qzeros.name, e))?;
    // Final sync not strictly needed since nothing follows in this function,
    // but good practice for callers that immediately launch kernels.
    {
        let sync_span = tracing::debug_span!("cuda_sync", reason = "int4_qzeros");
        let _enter = sync_span.enter();
        stream.synchronize()
            .map_err(|e| anyhow::anyhow!("Failed to sync stream after qzeros '{}': {}", qzeros.name, e))?;
    }

    Ok((qweight_gpu, scales_gpu, qzeros_gpu))
}


#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};

    // =====================================================================
    // CPU conversion tests (no GPU required)
    // =====================================================================

    #[test]
    fn bytes_to_bf16_direct_roundtrip() {
        // Create a known bf16 value, serialize to bytes, deserialize back
        let original = bf16::from_f32(3.14);
        let bits = original.to_bits();
        let bytes = bits.to_le_bytes();

        let result = bytes_to_bf16(&bytes.to_vec(), WeightDtype::Bf16).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].to_bits(), bits);
    }

    #[test]
    fn bytes_to_bf16_multiple_values() {
        let values = [1.0f32, -2.5, 0.0, 100.0, -0.5];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| {
                let bf = bf16::from_f32(*v);
                let bits = bf.to_bits();
                bits.to_le_bytes()
            })
            .collect();

        let result = bytes_to_bf16(&bytes, WeightDtype::Bf16).unwrap();
        assert_eq!(result.len(), 5);
        for (i, &v) in values.iter().enumerate() {
            let expected = bf16::from_f32(v);
            // Compare bits for exact match
            assert_eq!(result[i].to_bits(), expected.to_bits(), "mismatch at index {}", i);
        }
    }

    #[test]
    fn upload_weight_bf16_roundtrip() {
        // Test BF16 byte interpretation without GPU — just the conversion logic
        let bf_values = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(-0.5),
            bf16::from_f32(42.0),
        ];
        let bytes: Vec<u8> = bf_values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let result = bytes_to_bf16(&bytes, WeightDtype::Bf16).unwrap();
        assert_eq!(result.len(), 3);
        for (a, b) in bf_values.iter().zip(result.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn upload_weight_fp16_roundtrip() {
        // Test FP16 → BF16 conversion via bytes_to_bf16
        let f16_values = vec![
            f16::from_f32(1.0),
            f16::from_f32(-0.5),
            f16::from_f32(42.0),
        ];
        let bytes: Vec<u8> = f16_values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let result = bytes_to_bf16(&bytes, WeightDtype::Fp16).unwrap();
        assert_eq!(result.len(), 3);

        // Verify approximate equality (FP16→BF16 conversion has small precision loss)
        for (i, &v) in f16_values.iter().enumerate() {
            let expected = bf16::from_f32(v.to_f32());
            let diff = (result[i].to_f32() - expected.to_f32()).abs();
            assert!(
                diff < 0.05,
                "FP16→BF16 diff too large at {}: {} vs {} (diff {})",
                i,
                result[i].to_f32(),
                expected.to_f32(),
                diff
            );
        }
    }

    #[test]
    fn upload_weight_fp32_roundtrip() {
        // Test FP32 → BF16 conversion via bytes_to_bf16
        let f32_values = vec![1.0f32, -0.5, 42.0, 3.14159];
        let bytes: Vec<u8> = f32_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let result = bytes_to_bf16(&bytes, WeightDtype::Fp32).unwrap();
        assert_eq!(result.len(), 4);

        // Verify approximate equality (FP32→BF16 conversion has precision loss)
        for (i, &v) in f32_values.iter().enumerate() {
            let expected = bf16::from_f32(v);
            let diff = (result[i].to_f32() - expected.to_f32()).abs();
            assert!(
                diff < 0.1,
                "FP32→BF16 diff too large at {}: {} vs {} (diff {})",
                i,
                result[i].to_f32(),
                expected.to_f32(),
                diff
            );
        }
    }

    #[test]
    fn bytes_to_bf16_unsupported_dtype() {
        let data = vec![0u8; 4];
        let result = bytes_to_bf16(&data, WeightDtype::Int4Packed);
        assert!(result.is_err());
    }

    #[test]
    fn bytes_to_fp16_roundtrip() {
        let values = [1.0f32, -2.5, 0.0, 100.0, -0.5];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| {
                let f = f16::from_f32(*v);
                f.to_bits().to_le_bytes()
            })
            .collect();

        let result = bytes_to_fp16(&bytes, WeightDtype::Fp16).unwrap();
        assert_eq!(result.len(), 5);
        for (i, &v) in values.iter().enumerate() {
            let expected = f16::from_f32(v);
            // Exact bit comparison — no conversion loss since we stay in FP16
            assert_eq!(result[i].to_bits(), expected.to_bits(), "mismatch at index {}", i);
        }
    }

    #[test]
    fn bytes_to_fp16_unsupported_dtype() {
        let data = vec![0u8; 4];
        let result = bytes_to_fp16(&data, WeightDtype::Bf16);
        assert!(result.is_err());
    }

}
