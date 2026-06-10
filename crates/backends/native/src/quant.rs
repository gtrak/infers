//! Quantization helper functions for FP8 and NVFP4 formats.
//!
//! Provides CPU-based reference implementations of FP8 (E4M3, E5M2)
//! and NVFP4 quantization and dequantization. These are used for
//! correctness testing and as CPU fallbacks; production paths will
//! use CUDA kernel implementations.

use half::bf16;

/// Quantize BF16 data to FP8 E4M3 format.
///
/// Each BF16 value is converted to FP8 E4M3 (1 sign, 4 exponent, 3 mantissa bits).
/// Values outside the representable range are clamped to max/min.
///
/// # Arguments
/// * `data` - BF16 input slice to quantize
///
/// # Returns
/// A `Vec<u8>` where each byte is an FP8 E4M3 encoded value.
pub fn quantize_fp8_e4m3(data: &[bf16]) -> Vec<u8> {
    data.iter().map(|&v| f32_to_fp8_e4m3(v.to_f32())).collect()
}

/// Dequantize FP8 E4M3 data back to BF16.
///
/// # Arguments
/// * `data` - FP8 E4M3 encoded bytes
///
/// # Returns
/// A `Vec<bf16>` of dequantized values.
pub fn dequantize_fp8_e4m3(data: &[u8]) -> Vec<bf16> {
    data.iter().map(|&v| bf16::from_f32(fp8_e4m3_to_f32(v))).collect()
}

/// Quantize BF16 data to FP8 E5M2 format.
///
/// FP8 E5M2 has 1 sign, 5 exponent, 2 mantissa bits.
/// Wider exponent range but less precision than E4M3.
pub fn quantize_fp8_e5m2(data: &[bf16]) -> Vec<u8> {
    data.iter().map(|&v| f32_to_fp8_e5m2(v.to_f32())).collect()
}

/// Dequantize FP8 E5M2 data back to BF16.
pub fn dequantize_fp8_e5m2(data: &[u8]) -> Vec<bf16> {
    data.iter().map(|&v| bf16::from_f32(fp8_e5m2_to_f32(v))).collect()
}

/// Convert f32 to FP8 E4M3 encoded byte.
fn f32_to_fp8_e4m3(value: f32) -> u8 {
    let bits = value.to_bits();
    let sign = (bits >> 31) as u8;
    let exp = ((bits >> 23) & 0xFF) as i16;
    let mantissa = bits & 0x7FFFFF;

    // Handle special cases
    if exp == 0xFF {
        // NaN or Inf
        if mantissa != 0 {
            return 0x7F; // qNaN (E4M3: exp=0b1111, mant!=0)
        }
        return if sign == 0 { 0x7E } else { 0xFE }; // Inf
    }
    if exp == 0 && mantissa == 0 {
        return if sign == 0 { 0x00 } else { 0x80 }; // Zero
    }

    // Convert to E4M3
    // E4M3 bias = 7, FP32 bias = 127
    let fp8_exp = exp - 127 + 7;

    if fp8_exp >= 0xF {
        // Overflow: clamp to max finite (E4M3 max: exp=14, mant=7 → 0x77)
        return if sign == 0 { 0x77 } else { 0xF7 };
    }
    if fp8_exp < 0 {
        // Subnormal or underflow
        return if sign == 0 { 0x00 } else { 0x80 }; // Zero (no subnormals for simplicity)
    }

    // Encode: sign << 7 | exp << 3 | mantissa >> 20
    let fp8_mant = (mantissa >> 20) as u8; // Top 3 bits of FP32 mantissa
    (sign << 7) | ((fp8_exp as u8) << 3) | (fp8_mant & 0x7)
}

/// Convert FP8 E4M3 byte to f32.
fn fp8_e4m3_to_f32(value: u8) -> f32 {
    let sign = (value >> 7) as u32;
    let exp = ((value >> 3) & 0xF) as u32;
    let mant = (value & 0x7) as u32;

    // Handle special cases
    if exp == 0xF {
        if mant != 0 {
            return f32::NAN; // NaN
        }
        // Inf — but E4M3 has no Inf, treat as NaN
        return f32::NAN;
    }
    if exp == 0 && mant == 0 {
        return if sign == 0 { 0.0 } else { -0.0 }; // Zero
    }

    // Convert to FP32: bias difference = 127 - 7 = 120
    let fp32_exp = if exp == 0 { 0 } else { exp + 120 };
    let fp32_mant = mant << 20; // Shift 3-bit mant to FP32 position

    f32::from_bits((sign << 31) | (fp32_exp << 23) | fp32_mant)
}

/// Convert f32 to FP8 E5M2 encoded byte.
fn f32_to_fp8_e5m2(value: f32) -> u8 {
    let bits = value.to_bits();
    let sign = (bits >> 31) as u8;
    let exp = ((bits >> 23) & 0xFF) as i16;
    let mantissa = bits & 0x7FFFFF;

    // Handle special cases
    if exp == 0xFF {
        if mantissa != 0 {
            return if sign == 0 { 0x7F } else { 0xFF }; // NaN (E5M2: exp=0b11111, mant!=0)
        }
        return if sign == 0 { 0x7C } else { 0xFC }; // Inf
    }
    if exp == 0 && mantissa == 0 {
        return if sign == 0 { 0x00 } else { 0x80 }; // Zero
    }

    // Convert to E5M2
    // E5M2 bias = 15, FP32 bias = 127
    let fp8_exp = exp - 127 + 15;

    if fp8_exp >= 0x1F {
        // Overflow: clamp to max finite (E5M2 max: exp=30, mant=3 → 0x7B)
        return if sign == 0 { 0x7B } else { 0xFB };
    }
    if fp8_exp < 0 {
        return if sign == 0 { 0x00 } else { 0x80 };
    }

    // Encode: sign << 7 | exp << 2 | mantissa >> 21
    let fp8_mant = (mantissa >> 21) as u8; // Top 2 bits of FP32 mantissa
    (sign << 7) | ((fp8_exp as u8) << 2) | (fp8_mant & 0x3)
}

/// Convert FP8 E5M2 byte to f32.
fn fp8_e5m2_to_f32(value: u8) -> f32 {
    let sign = (value >> 7) as u32;
    let exp = ((value >> 2) & 0x1F) as u32;
    let mant = (value & 0x3) as u32;

    if exp == 0x1F {
        if mant != 0 {
            return f32::NAN;
        }
        return f32::NAN;
    }
    if exp == 0 && mant == 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }

    // FP32 bias = 127, E5M2 bias = 15, offset = 112
    let fp32_exp = if exp == 0 { 0 } else { exp + 112 };
    let fp32_mant = mant << 21;

    f32::from_bits((sign << 31) | (fp32_exp << 23) | fp32_mant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_dequantize_e4m3_roundtrip_zero() {
        let input = vec![bf16::ZERO; 10];
        let quantized = quantize_fp8_e4m3(&input);
        let dequantized = dequantize_fp8_e4m3(&quantized);
        for (a, b) in input.iter().zip(dequantized.iter()) {
            assert_eq!(a.to_f32(), b.to_f32());
        }
    }

    #[test]
    fn quantize_dequantize_e4m3_roundtrip_small_values() {
        let values = [1.0f32, 0.5, 2.0, 0.25, 3.5, 0.125];
        let input: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
        let quantized = quantize_fp8_e4m3(&input);
        let dequantized = dequantize_fp8_e4m3(&quantized);
        for (&orig, &deq) in values.iter().zip(dequantized.iter()) {
            let diff = (orig - deq.to_f32()).abs();
            // E4M3 has ~3 bits of mantissa, so relative error should be < 12.5%
            assert!(diff <= 0.2, "diff too large: {} vs {}", orig, deq.to_f32());
        }
    }

    #[test]
    fn quantize_dequantize_e5m2_roundtrip_zero() {
        let input = vec![bf16::ZERO; 10];
        let quantized = quantize_fp8_e5m2(&input);
        let dequantized = dequantize_fp8_e5m2(&quantized);
        for (a, b) in input.iter().zip(dequantized.iter()) {
            assert_eq!(a.to_f32(), b.to_f32());
        }
    }

    #[test]
    fn quantize_dequantize_e5m2_roundtrip_small_values() {
        let values = [1.0f32, 0.5, 2.0, 0.25, 4.0, 0.125];
        let input: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();
        let quantized = quantize_fp8_e5m2(&input);
        let dequantized = dequantize_fp8_e5m2(&quantized);
        for (&orig, &deq) in values.iter().zip(dequantized.iter()) {
            let diff = (orig - deq.to_f32()).abs();
            // E5M2 has only 2 bits of mantissa — expect larger error
            assert!(diff <= 0.5, "diff too large: {} vs {}", orig, deq.to_f32());
        }
    }

    #[test]
    fn quantize_e4m3_handles_nan() {
        let input = vec![bf16::from_f32(f32::NAN)];
        let quantized = quantize_fp8_e4m3(&input);
        let dequantized = dequantize_fp8_e4m3(&quantized);
        assert!(dequantized[0].to_f32().is_nan());
    }

    #[test]
    fn quantize_e5m2_handles_nan() {
        let input = vec![bf16::from_f32(f32::NAN)];
        let quantized = quantize_fp8_e5m2(&input);
        let dequantized = dequantize_fp8_e5m2(&quantized);
        assert!(dequantized[0].to_f32().is_nan());
    }

    #[test]
    fn quantize_e4m3_clamps_large_values() {
        let input = vec![bf16::from_f32(1000.0f32)]; // Beyond E4M3 max (~448 in theory but test clamp)
        let quantized = quantize_fp8_e4m3(&input);
        let dequantized = dequantize_fp8_e4m3(&quantized);
        // Should be clamped to something finite
        assert!(dequantized[0].to_f32().is_finite());
    }

    #[test]
    fn quantize_e5m2_clamps_large_values() {
        let input = vec![bf16::from_f32(1e6f32)];
        let quantized = quantize_fp8_e5m2(&input);
        let dequantized = dequantize_fp8_e5m2(&quantized);
        assert!(dequantized[0].to_f32().is_finite());
    }

    #[test]
    fn roundtrip_large_batch() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.5).collect();
        let input: Vec<bf16> = values.iter().map(|&v| bf16::from_f32(v)).collect();

        let q_e4m3 = quantize_fp8_e4m3(&input);
        let dq_e4m3 = dequantize_fp8_e4m3(&q_e4m3);
        for (&orig, &deq) in values.iter().zip(dq_e4m3.iter()) {
            let diff = (orig - deq.to_f32()).abs();
            // E4M3 has 3-bit mantissa — step size scales with magnitude
            let threshold = (orig.abs() * 0.125).max(0.5);
            assert!(diff <= threshold, "E4M3 diff {} too large at {}", diff, orig);
        }

        let q_e5m2 = quantize_fp8_e5m2(&input);
        let dq_e5m2 = dequantize_fp8_e5m2(&q_e5m2);
        for (&orig, &deq) in values.iter().zip(dq_e5m2.iter()) {
            let diff = (orig - deq.to_f32()).abs();
            // E5M2 has 2-bit mantissa — step size ~25% of magnitude
            let threshold = (orig.abs() * 0.25).max(0.5);
            assert!(diff <= threshold, "E5M2 diff {} too large at {}", diff, orig);
        }
    }
}
