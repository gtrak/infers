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
    let bf16_vec = bytes_to_bf16(&weight.data, weight.dtype)?;

    // Debug: dump conv1d weight bytes just before GPU upload
    if weight.name.contains("conv1d.weight") {
        let debug_dir = "/tmp/weight_debug";
        let _ = std::fs::create_dir_all(debug_dir);
        // Extract layer number from name like "layers.0.linear_attn.conv1d.weight"
        let layer_tag = weight.name
            .strip_prefix("layers.")
            .and_then(|s| s.split('.').next())
            .unwrap_or("unknown");
        let path = format!("{}/conv1d_before_upload_layer{}.raw", debug_dir, layer_tag);
        std::fs::write(&path, &weight.data).expect("Failed to write pre-upload debug");
        eprintln!("  Dumped conv1d weight before upload: {} ({} bytes) name={}", path, weight.data.len(), weight.name);
    }

    // Also dump after conversion to bf16 vector
    if weight.name.contains("conv1d.weight") {
        let debug_dir = "/tmp/weight_debug";
        let layer_tag = weight.name
            .strip_prefix("layers.")
            .and_then(|s| s.split('.').next())
            .unwrap_or("unknown");
        let path = format!("{}/conv1d_converted_layer{}.raw", debug_dir, layer_tag);
        let bf16_bytes: Vec<u8> = bf16_vec.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(&path, &bf16_bytes).expect("Failed to write post-convert debug");
        eprintln!("  Dumped converted conv1d layer{}: {} ({} bytes)", layer_tag, path, bf16_bytes.len());
    }

    let gpu_slice = stream
        .clone_htod(&bf16_vec)
        .map_err(|e| anyhow::anyhow!("Failed to upload weight '{}': {}", weight.name, e))?;

    // CRITICAL FIX: Synchronize the stream after each upload to prevent cudaMallocAsync
    // from returning the same GPU address for consecutive allocations on this stream.
    // The async memory pool can return overlapping addresses if prior allocations are
    // still pending (copy not yet completed) when the next allocation is made.
    stream
        .synchronize()
        .map_err(|e| anyhow::anyhow!("Failed to sync stream after uploading '{}': {}", weight.name, e))?;

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
// @lat: [[lat.md/lat#Phase 4 Deliverables#Forward Engine#INT4 Triplet Upload]]
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
// @lat: [[lat.md/lat#Phase 4 Deliverables#Forward Engine#INT4 Triplet Upload]]
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
    stream.synchronize()
        .map_err(|e| anyhow::anyhow!("Failed to sync stream after qweight '{}': {}", qweight.name, e))?;

    let scales_gpu = stream
        .clone_htod(&scales_vec)
        .map_err(|e| anyhow::anyhow!("Failed to upload scales '{}': {}", scales.name, e))?;
    stream.synchronize()
        .map_err(|e| anyhow::anyhow!("Failed to sync stream after scales '{}': {}", scales.name, e))?;

    let qzeros_gpu = stream
        .clone_htod(&qzeros_u32)
        .map_err(|e| anyhow::anyhow!("Failed to upload qzeros '{}': {}", qzeros.name, e))?;
    // Final sync not strictly needed since nothing follows in this function,
    // but good practice for callers that immediately launch kernels.
    stream.synchronize()
        .map_err(|e| anyhow::anyhow!("Failed to sync stream after qzeros '{}': {}", qzeros.name, e))?;

    Ok((qweight_gpu, scales_gpu, qzeros_gpu))
}

// ---------------------------------------------------------------------------
// CPU dequantization helper
// ---------------------------------------------------------------------------

/// Extract a signed INT4 value from a 4-bit nibble in a u32.
///
/// The CUDA kernel uses `(int8_t)((packed >> shift) & 0xF)` which treats
/// the raw 4-bit value as unsigned (0..15) when cast to i8.
/// This matches the kernel behavior — no sign extension.
fn extract_int4(packed: u32, shift: u32) -> i8 {
    ((packed >> shift) & 0xF) as i8
}

/// Dequantize an INT4-packed weight triplet to BF16 on CPU.
///
/// Useful for the dequantize-at-upload fallback path or for validation.
///
/// # Formula
/// `bf16_val = (int4_val - zero_point) * scale`
///
/// Each `u32` in `qweight` contains 8 INT4 values (lower 4 bits each).
/// Zero-points are packed similarly (8 per `u32`).
///
/// # Returns
/// Flat `Vec<bf16>` of length `N * K`.
pub fn dequantize_int4_to_bf16(
    qweight: &WeightData,
    scales: &WeightData,
    qzeros: &WeightData,
    group_size: usize,
) -> Vec<bf16> {
    // Parse packed u32 data into vecs
    let qweight_u32: Vec<u32> = qweight
        .data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let scales_f32: Vec<f32> = scales
        .data
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect();

    let qzeros_u32: Vec<u32> = qzeros
        .data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Derive dimensions from qweight shape [N, K/8]
    let n = qweight.shape[0];
    let k_packed = qweight.shape[1];
    let k = k_packed * 8; // each u32 holds 8 weights

    let mut output = Vec::with_capacity(n * k);

    for n_idx in 0..n {
        let weight_offset = n_idx * k_packed;
        let scales_offset = n_idx * (k / group_size);

        for k_idx in 0..k {
            let group_idx = k_idx / group_size;

            // Scale for this group
            let scale = scales_f32[scales_offset + group_idx];

            // Zero point for this group (packed, 8 per u32)
            let zero_packed_idx = (scales_offset + group_idx) / 8;
            let zero_shift = ((scales_offset + group_idx) % 8) * 4;
            let zero_packed = qzeros_u32[zero_packed_idx];
            let zero_point = extract_int4(zero_packed, zero_shift as u32) as f32;

            // INT4 weight value (packed, 8 per u32)
            let weight_packed_idx = weight_offset + (k_idx / 8);
            let weight_shift = (k_idx % 8) * 4;
            let weight_packed = qweight_u32[weight_packed_idx];
            let int4_val = extract_int4(weight_packed, weight_shift as u32) as f32;

            // Dequantize: (int4_val - zero_point) * scale
            let dequantized = (int4_val - zero_point) * scale;
            output.push(bf16::from_f32(dequantized));
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
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


    // =====================================================================
    // INT4 dequantization tests (CPU-side, no GPU)
    // =====================================================================

    /// Helper to create a synthetic INT4 triplet for testing.
    ///
    /// Creates weights with known values: `values[i] * scale`.
    /// All values share the same scale and zero-point for simplicity.
    fn create_int4_triplet(
        values: &[i32],
        scale: f32,
        zero_point: i32,
        group_size: usize,
    ) -> (WeightData, WeightData, WeightData) {
        let num_values = values.len();
        assert_eq!(num_values % group_size, 0, "values must be divisible by group_size");

        let n = 1; // single row for simplicity
        let k_packed = num_values / 8;
        let k = num_values;

        // Pack INT4 weights into u32s
        let mut qweight_bytes = Vec::with_capacity(k_packed * 4);
        for i in 0..k_packed {
            let mut packed: u32 = 0;
            for w in 0..8 {
                let idx = i * 8 + w;
                let val = values[idx] as i8;
                packed |= (val as u32) << (w * 4);
            }
            qweight_bytes.extend_from_slice(&packed.to_le_bytes());
        }

        // Scale: one scale per group, stored as BF16
        let num_groups = k / group_size;
        let scale_bits = bf16::from_f32(scale).to_bits();
        let mut scales_bytes = Vec::with_capacity(num_groups * 2);
        for _ in 0..num_groups {
            scales_bytes.extend_from_slice(&scale_bits.to_le_bytes());
        }

        // Zero points: packed as INT4 in u32s (8 per u32)
        let num_zeros_packed = (num_groups + 7) / 8;
        let mut qzeros_bytes = Vec::with_capacity(num_zeros_packed * 4);
        for i in 0..num_zeros_packed {
            let mut packed: u32 = 0;
            for w in 0..8 {
                let idx = i * 8 + w;
                if idx < num_groups {
                    let zp = zero_point as i8;
                    packed |= (zp as u32) << (w * 4);
                }
            }
            qzeros_bytes.extend_from_slice(&packed.to_le_bytes());
        }

        let qweight = WeightData {
            data: Bytes::from(qweight_bytes),
            shape: vec![n, k_packed],
            dtype: WeightDtype::Int4Packed,
            name: "test_qweight".to_string(),
        };
        let scales = WeightData {
            data: Bytes::from(scales_bytes),
            shape: vec![n, num_groups],
            dtype: WeightDtype::Bf16,
            name: "test_scales".to_string(),
        };
        let qzeros = WeightData {
            data: Bytes::from(qzeros_bytes),
            shape: vec![n, num_groups / 8],
            dtype: WeightDtype::Int4Packed,
            name: "test_qzeros".to_string(),
        };

        (qweight, scales, qzeros)
    }

    #[test]
    fn dequantize_int4_to_bf16_basic() {
        // INT4 values must fit in 4 bits (0..=15).  scale=1.0, zero=0.
        // Expected output: (val - 0) * 1.0 = val as bf16
        let values: Vec<i32> = (0..16).collect();
        let group_size = 16;

        let (qweight, scales, qzeros) = create_int4_triplet(&values, 1.0, 0, group_size);

        let result = dequantize_int4_to_bf16(&qweight, &scales, &qzeros, group_size);

        assert_eq!(result.len(), 16);
        for (i, &v) in values.iter().enumerate() {
            let expected = bf16::from_f32(v as f32);
            assert_eq!(
                result[i].to_bits(),
                expected.to_bits(),
                "mismatch at index {}: got {}, expected {}",
                i,
                result[i].to_f32(),
                expected.to_f32()
            );
        }
    }

    #[test]
    fn dequantize_int4_to_bf16_with_scale() {
        // Scale=2.5, zero=1 → expected = (val - 1) * 2.5
        let values: Vec<i32> = (0..=15).collect();
        let scale = 2.5;
        let zero_point = 1;
        let group_size = 16;

        let (qweight, scales, qzeros) = create_int4_triplet(&values, scale, zero_point, group_size);

        let result = dequantize_int4_to_bf16(&qweight, &scales, &qzeros, group_size);

        assert_eq!(result.len(), 16);
        for (i, &v) in values.iter().enumerate() {
            let expected_f32 = (v as f32 - zero_point as f32) * scale;
            let expected = bf16::from_f32(expected_f32);
            let diff = (result[i].to_f32() - expected.to_f32()).abs();
            assert!(
                diff < 0.1,
                "mismatch at index {}: got {} (expected {}), diff {}",
                i,
                result[i].to_f32(),
                expected.to_f32(),
                diff
            );
        }
    }

    #[test]
    fn dequantize_int4_to_bf16_zero_point_nonzero() {
        // zero_point=3, scale=1.0 → expected = (val - 3) * 1.0
        let values: Vec<i32> = (0..=15).collect();
        let scale = 1.0;
        let zero_point = 3;
        let group_size = 16;

        let (qweight, scales, qzeros) = create_int4_triplet(&values, scale, zero_point, group_size);

        let result = dequantize_int4_to_bf16(&qweight, &scales, &qzeros, group_size);

        assert_eq!(result.len(), 16);
        for (i, &v) in values.iter().enumerate() {
            let expected_f32 = (v as f32 - zero_point as f32) * scale;
            let expected = bf16::from_f32(expected_f32);
            assert_eq!(
                result[i].to_bits(),
                expected.to_bits(),
                "mismatch at index {}: got {}, expected {}",
                i,
                result[i].to_f32(),
                expected.to_f32()
            );
        }
    }

    #[test]
    fn dequantize_int4_to_bf16_multiple_groups() {
        // Two groups of 8 each (group_size=8), different scales per group.
        // Group 0: scale=2.0, Group 1: scale=0.5, zero_point=0 for both.
        // INT4 is unsigned 4-bit: range [0..15].
        let values: Vec<i32> = (0..16).collect(); // 0 to 15
        let group_size = 8;

        let n = 1;
        let k = 16;
        let k_packed = k / 8; // 2 u32s
        let num_groups = k / group_size; // 2 groups

        // qweight: pack all 16 values into 2 u32s
        let mut qweight_bytes = Vec::new();
        for i in 0..k_packed {
            let mut packed: u32 = 0;
            for w in 0..8 {
                let idx = i * 8 + w;
                let val = values[idx];
                packed |= (val as u32) << (w * 4);
            }
            qweight_bytes.extend_from_slice(&packed.to_le_bytes());
        }

        // scales: [2.0, 0.5] as BF16
        let scale0_bits = bf16::from_f32(2.0).to_bits();
        let scale1_bits = bf16::from_f32(0.5).to_bits();
        let mut scales_bytes = Vec::new();
        scales_bytes.extend_from_slice(&scale0_bits.to_le_bytes());
        scales_bytes.extend_from_slice(&scale1_bits.to_le_bytes());

        // qzeros: all zeros packed into one u32
        let qzeros_bytes = vec![0u8; 4];

        let qweight = WeightData {
            data: Bytes::from(qweight_bytes),
            shape: vec![n, k_packed],
            dtype: WeightDtype::Int4Packed,
            name: "multi_group_qweight".to_string(),
        };
        let scales = WeightData {
            data: Bytes::from(scales_bytes),
            shape: vec![n, num_groups],
            dtype: WeightDtype::Bf16,
            name: "multi_group_scales".to_string(),
        };
        let qzeros = WeightData {
            data: Bytes::from(qzeros_bytes),
            shape: vec![n, 1],
            dtype: WeightDtype::Int4Packed,
            name: "multi_group_qzeros".to_string(),
        };

        let result = dequantize_int4_to_bf16(&qweight, &scales, &qzeros, group_size);

        assert_eq!(result.len(), 16);

        // Group 0 (indices 0-7, values 0..7): scale=2.0, zero=0
        for i in 0..8 {
            let v = values[i] as f32;
            let expected_f32 = v * 2.0;
            let diff = (result[i].to_f32() - expected_f32).abs();
            assert!(
                diff < 0.1,
                "Group 0 index {} mismatch: got {} expected {} diff {}",
                i,
                result[i].to_f32(),
                expected_f32,
                diff
            );
        }

        // Group 1 (indices 8-15, values 8..15): scale=0.5, zero=0
        for i in 8..16 {
            let v = values[i] as f32;
            let expected_f32 = v * 0.5;
            let diff = (result[i].to_f32() - expected_f32).abs();
            assert!(
                diff < 0.1,
                "Group 1 index {} mismatch: got {} expected {} diff {}",
                i,
                result[i].to_f32(),
                expected_f32,
                diff
            );
        }
    }

    #[test]
    fn dequantize_int4_empty() {
        // Edge case: empty weights
        let qweight = WeightData {
            data: Bytes::from(vec![]),
            shape: vec![0, 0],
            dtype: WeightDtype::Int4Packed,
            name: "empty".to_string(),
        };
        let scales = WeightData {
            data: Bytes::from(vec![]),
            shape: vec![0, 0],
            dtype: WeightDtype::Bf16,
            name: "empty_scales".to_string(),
        };
        let qzeros = WeightData {
            data: Bytes::from(vec![]),
            shape: vec![0, 0],
            dtype: WeightDtype::Int4Packed,
            name: "empty_qzeros".to_string(),
        };

        let result = dequantize_int4_to_bf16(&qweight, &scales, &qzeros, 128);
        assert!(result.is_empty());
    }
}
