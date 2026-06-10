//! Quantization helper functions for FP8 and NVFP4 formats.
//!
//! Re-exports from `infers-kv` crate's quant module. Kept as a standalone
//! module for backend-specific additions (e.g., INT4 GEMM dispatch,
//! NVFP4 kernel wrappers) in future phases.

pub use infers_kv::quant::{
    dequantize_fp8_e4m3,
    dequantize_fp8_e5m2,
    quantize_fp8_e4m3,
    quantize_fp8_e5m2,
};

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

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
        let input = vec![bf16::from_f32(1000.0f32)];
        let quantized = quantize_fp8_e4m3(&input);
        let dequantized = dequantize_fp8_e4m3(&quantized);
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
            let threshold = (orig.abs() * 0.125).max(0.5);
            assert!(diff <= threshold, "E4M3 diff {} too large at {}", diff, orig);
        }

        let q_e5m2 = quantize_fp8_e5m2(&input);
        let dq_e5m2 = dequantize_fp8_e5m2(&q_e5m2);
        for (&orig, &deq) in values.iter().zip(dq_e5m2.iter()) {
            let diff = (orig - deq.to_f32()).abs();
            let threshold = (orig.abs() * 0.25).max(0.5);
            assert!(diff <= threshold, "E5M2 diff {} too large at {}", diff, orig);
        }
    }
}
