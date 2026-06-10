/// KV cache quantization data types.
///
/// Defines the supported precision formats for storing key/value
/// activation tensors in the paged attention cache. Each variant
/// carries a different trade-off between memory footprint and
/// numerical fidelity.

use cudarc::driver::{CudaSlice, CudaStream};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheDtype {
    /// Brain float 16 — 2 bytes per element.
    Bf16,
    /// FP8 with 4 exponent bits, 3 mantissa bits — 1 byte per element.
    Fp8E4M3,
    /// FP8 with 5 exponent bits, 2 mantissa bits — 1 byte per element.
    Fp8E5M2,
    /// NVIDIA 4-bit float — packed with block scales, 1 byte per element.
    Nvfp4,
}

impl KvCacheDtype {
    /// Returns the number of bytes used for a single element.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8E4M3 => 1,
            Self::Fp8E5M2 => 1,
            Self::Nvfp4 => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_bytes_per_element() {
        assert_eq!(KvCacheDtype::Bf16.bytes_per_element(), 2);
    }

    #[test]
    fn fp8_e4m3_bytes_per_element() {
        assert_eq!(KvCacheDtype::Fp8E4M3.bytes_per_element(), 1);
    }

    #[test]
    fn fp8_e5m2_bytes_per_element() {
        assert_eq!(KvCacheDtype::Fp8E5M2.bytes_per_element(), 1);
    }

    #[test]
    fn nvfp4_bytes_per_element() {
        assert_eq!(KvCacheDtype::Nvfp4.bytes_per_element(), 1);
    }

    #[test]
    fn all_variants_are_distinct() {
        assert_ne!(KvCacheDtype::Bf16, KvCacheDtype::Fp8E4M3);
        assert_ne!(KvCacheDtype::Fp8E4M3, KvCacheDtype::Fp8E5M2);
        assert_ne!(KvCacheDtype::Fp8E5M2, KvCacheDtype::Nvfp4);
    }
}

// ── FP8 CPU reference helpers ────────────────────────────────────────────
// These public functions provide CPU reference implementations for FP8
// quantization and dequantization. Used by tests and re-exported by
// `infers-backend-native` for CPU fallback paths.
// Production GPU paths use CUDA kernels (infers_fp8_quantize_bf16, etc.).

/// Quantize BF16 slice to FP8 E4M3 bytes (CPU reference).
pub fn quantize_fp8_e4m3(data: &[half::bf16]) -> Vec<u8> {
    data.iter().map(|&v| f32_to_fp8_e4m3(v.to_f32())).collect()
}

/// Dequantize FP8 E4M3 bytes to BF16 slice (CPU reference).
pub fn dequantize_fp8_e4m3(data: &[u8]) -> Vec<half::bf16> {
    data.iter().map(|&v| half::bf16::from_f32(fp8_e4m3_to_f32(v))).collect()
}

/// Quantize BF16 slice to FP8 E5M2 bytes (CPU reference).
pub fn quantize_fp8_e5m2(data: &[half::bf16]) -> Vec<u8> {
    data.iter().map(|&v| f32_to_fp8_e5m2(v.to_f32())).collect()
}

/// Dequantize FP8 E5M2 bytes to BF16 slice (CPU reference).
pub fn dequantize_fp8_e5m2(data: &[u8]) -> Vec<half::bf16> {
    data.iter().map(|&v| half::bf16::from_f32(fp8_e5m2_to_f32(v))).collect()
}

/// Convert f32 to FP8 E4M3 encoded byte.
fn f32_to_fp8_e4m3(value: f32) -> u8 {
    let bits = value.to_bits();
    let sign = (bits >> 31) as u8;
    let exp = ((bits >> 23) & 0xFF) as i16;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0xFF {
        if mantissa != 0 { return 0x7F; } // qNaN
        return if sign == 0 { 0x77 } else { 0xF7 }; // ±Inf → clamp to max finite
    }
    if exp == 0 && mantissa == 0 {
        return if sign == 0 { 0x00 } else { 0x80 };
    }

    let fp8_exp = exp - 127 + 7;
    if fp8_exp >= 0xF {
        return if sign == 0 { 0x77 } else { 0xF7 }; // Clamp to max finite
    }
    if fp8_exp < 0 {
        return if sign == 0 { 0x00 } else { 0x80 };
    }

    let fp8_mant = (mantissa >> 20) as u8;
    (sign << 7) | ((fp8_exp as u8) << 3) | (fp8_mant & 0x7)
}

/// Convert FP8 E4M3 byte to f32.
fn fp8_e4m3_to_f32(value: u8) -> f32 {
    let sign = (value >> 7) as u32;
    let exp = ((value >> 3) & 0xF) as u32;
    let mant = (value & 0x7) as u32;

    if exp == 0xF {
        if mant != 0 { return f32::NAN; }
        return f32::NAN; // E4M3 has no Inf
    }
    if exp == 0 && mant == 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }

    let fp32_exp = if exp == 0 { 0 } else { exp + 120 };
    let fp32_mant = mant << 20;
    f32::from_bits((sign << 31) | (fp32_exp << 23) | fp32_mant)
}

/// Convert f32 to FP8 E5M2 encoded byte.
fn f32_to_fp8_e5m2(value: f32) -> u8 {
    let bits = value.to_bits();
    let sign = (bits >> 31) as u8;
    let exp = ((bits >> 23) & 0xFF) as i16;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0xFF {
        if mantissa != 0 { return if sign == 0 { 0x7F } else { 0xFF }; }
        return if sign == 0 { 0x7C } else { 0xFC };
    }
    if exp == 0 && mantissa == 0 {
        return if sign == 0 { 0x00 } else { 0x80 };
    }

    let fp8_exp = exp - 127 + 15;
    if fp8_exp >= 0x1F {
        return if sign == 0 { 0x7B } else { 0xFB }; // Clamp to max finite
    }
    if fp8_exp < 0 {
        return if sign == 0 { 0x00 } else { 0x80 };
    }

    let fp8_mant = (mantissa >> 21) as u8;
    (sign << 7) | ((fp8_exp as u8) << 2) | (fp8_mant & 0x3)
}

/// Convert FP8 E5M2 byte to f32.
fn fp8_e5m2_to_f32(value: u8) -> f32 {
    let sign = (value >> 7) as u32;
    let exp = ((value >> 2) & 0x1F) as u32;
    let mant = (value & 0x3) as u32;

    if exp == 0x1F {
        if mant != 0 { return f32::NAN; }
        return f32::NAN;
    }
    if exp == 0 && mant == 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }

    let fp32_exp = if exp == 0 { 0 } else { exp + 112 };
    let fp32_mant = mant << 21;
    f32::from_bits((sign << 31) | (fp32_exp << 23) | fp32_mant)
}

/// Quantized paged KV cache using interleaved K+V per page layout.
///
/// Matches the PagedKvCache layout but stores quantized values:
/// page_pool[page_id * page_stride + side * page_size * kv_dim + ...]
/// where side=0 for K, side=1 for V.
#[derive(Debug)]
pub struct QuantizedKvCache {
    /// Quantized data type for cache entries.
    pub dtype: KvCacheDtype,
    /// Interleaved page pool (K then V per page), quantized to dtype.
    pub page_pool: CudaSlice<u8>,
    /// Number of pages in the pool.
    pub num_pages: usize,
    /// Page size (tokens per page).
    pub page_size: usize,
    /// KV dimension (num_kv_heads * head_dim).
    pub kv_dim: usize,
    /// Block scales for NVFP4 (one per 128-element block).
    /// Two scales per block: one for K, one for V.
    pub scales: Option<CudaSlice<half::bf16>>,
}

impl QuantizedKvCache {
    /// Allocate a new quantized KV cache on the GPU.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream for memory allocation
    /// * `num_pages` - Total number of physical pages
    /// * `page_size` - Number of tokens per page
    /// * `kv_dim` - KV dimension (num_kv_heads * head_dim)
    /// * `dtype` - Quantized data type for cache entries
    ///
    /// # Returns
    /// A new `QuantizedKvCache` with GPU memory allocated and zeroed.
    pub fn allocate(
        stream: &std::sync::Arc<CudaStream>,
        num_pages: usize,
        page_size: usize,
        kv_dim: usize,
        dtype: KvCacheDtype,
    ) -> anyhow::Result<Self> {
        let bytes_per_elem = dtype.bytes_per_element();
        let page_stride = 2 * page_size * kv_dim; // elements per page (K + V)
        let page_bytes = page_stride * bytes_per_elem; // bytes per page
        let total_bytes = num_pages * page_bytes;

        let page_pool = stream
            .alloc_zeros::<u8>(total_bytes)?;

        let scales = if matches!(dtype, KvCacheDtype::Nvfp4) {
            // NVFP4 uses block scales: one scale per 128 elements
            // Each page has 2 * page_size * kv_dim elements, so:
            let num_blocks = (num_pages * 2 * page_size * kv_dim).div_ceil(128);
            // Two scale values per block (one for K-side, one for V-side)
            Some(stream.alloc_zeros::<half::bf16>(num_blocks * 2)?)
        } else {
            None
        };

        Ok(Self {
            dtype,
            page_pool,
            num_pages,
            page_size,
            kv_dim,
            scales,
        })
    }
}

#[cfg(test)]
mod fp8_tests {
    use super::*;

    #[test]
    fn quantize_dequantize_e4m3_roundtrip_zero() {
        let input = vec![half::bf16::ZERO; 10];
        let quantized = quantize_fp8_e4m3(&input);
        let dequantized = dequantize_fp8_e4m3(&quantized);
        for (a, b) in input.iter().zip(dequantized.iter()) {
            assert_eq!(a.to_f32(), b.to_f32());
        }
    }

    #[test]
    fn quantize_dequantize_e4m3_roundtrip_values() {
        let values = [1.0f32, 0.5, 2.0, 0.25, 3.5];
        let input: Vec<half::bf16> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let quantized = quantize_fp8_e4m3(&input);
        let dequantized = dequantize_fp8_e4m3(&quantized);
        for (&orig, &deq) in values.iter().zip(dequantized.iter()) {
            let diff = (orig - deq.to_f32()).abs();
            assert!(diff <= 0.2, "diff too large: {} vs {}", orig, deq.to_f32());
        }
    }

    #[test]
    fn quantize_dequantize_e5m2_roundtrip_zero() {
        let input = vec![half::bf16::ZERO; 10];
        let quantized = quantize_fp8_e5m2(&input);
        let dequantized = dequantize_fp8_e5m2(&quantized);
        for (a, b) in input.iter().zip(dequantized.iter()) {
            assert_eq!(a.to_f32(), b.to_f32());
        }
    }

    #[test]
    fn quantize_dequantize_e5m2_roundtrip_values() {
        let values = [1.0f32, 0.5, 2.0, 0.25, 4.0];
        let input: Vec<half::bf16> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let quantized = quantize_fp8_e5m2(&input);
        let dequantized = dequantize_fp8_e5m2(&quantized);
        for (&orig, &deq) in values.iter().zip(dequantized.iter()) {
            let diff = (orig - deq.to_f32()).abs();
            assert!(diff <= 0.5, "diff too large: {} vs {}", orig, deq.to_f32());
        }
    }

    #[test]
    fn quantize_e4m3_handles_nan() {
        let input = vec![half::bf16::from_f32(f32::NAN)];
        let quantized = quantize_fp8_e4m3(&input);
        let dequantized = dequantize_fp8_e4m3(&quantized);
        assert!(dequantized[0].to_f32().is_nan());
    }

    #[test]
    fn e4m3_clamps_large_values() {
        let input = vec![half::bf16::from_f32(1000.0f32)];
        let quantized = quantize_fp8_e4m3(&input);
        let dequantized = dequantize_fp8_e4m3(&quantized);
        assert!(dequantized[0].to_f32().is_finite());
    }
}
