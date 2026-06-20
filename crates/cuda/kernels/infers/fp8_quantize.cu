// @lat: [[arch#Kernel Extraction and Build System#Kernel Source Files]]
/// FP8 quantize and dequantize kernels for KV cache quantization.
///
/// BF16→FP8 quantize: converts BF16 values to FP8 E4M3 or E5M2 bytes.
/// FP8→BF16 dequantize: converts FP8 bytes back to BF16.
///
/// Grid: (N + block_size - 1) / block_size blocks, block_size = 256.
/// Each thread processes one element.

#include "common.cuh"
#include <stdint.h>

// ── FP8 E4M3 device helpers ──────────────────────────────────────────

/// FP8 E4M3 format: 1 sign, 4 exponent (bias 7), 3 mantissa bits.
/// Max finite: sign=0, exp=14, mant=7 → 0x77 (positive) / 0xF7 (negative).

/// Convert float to FP8 E4M3 byte.
__device__ inline uint8_t float_to_fp8_e4m3(float value) {
    uint32_t bits = __float_as_uint(value);
    uint8_t sign = (uint8_t)(bits >> 31);
    int16_t exp = (int16_t)((bits >> 23) & 0xFF);
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exp == 0xFF) {
        if (mantissa != 0) return 0x7F;  // NaN
        return sign == 0 ? 0x77 : 0xF7;  // Inf → clamp to max finite
    }
    if (exp == 0 && mantissa == 0) {
        return sign == 0 ? 0x00 : 0x80;
    }

    int16_t fp8_exp = exp - 127 + 7;
    if (fp8_exp >= 0xF) {
        return sign == 0 ? 0x77 : 0xF7;  // Clamp to max finite
    }
    if (fp8_exp < 0) {
        return sign == 0 ? 0x00 : 0x80;
    }

    uint8_t fp8_mant = (uint8_t)(mantissa >> 20);
    return (sign << 7) | ((uint8_t)fp8_exp << 3) | (fp8_mant & 0x7);
}

/// Convert FP8 E4M3 byte to float.
__device__ inline float fp8_e4m3_to_float(uint8_t value) {
    uint8_t sign = value >> 7;
    uint8_t exp = (value >> 3) & 0xF;
    uint8_t mant = value & 0x7;

    if (exp == 0xF) {
        if (mant != 0) return __int_as_float(0x7fc00000u);  // NaN
        return __int_as_float(0x7fc00000u);  // E4M3 has no Inf
    }
    if (exp == 0 && mant == 0) {
        return sign == 0 ? 0.0f : -0.0f;
    }

    uint32_t fp32_exp = exp == 0 ? 0 : (uint32_t)exp + 120;
    uint32_t fp32_mant = (uint32_t)mant << 20;
    return __uint_as_float((sign << 31) | (fp32_exp << 23) | fp32_mant);
}

// ── FP8 E5M2 device helpers ──────────────────────────────────────────

/// FP8 E5M2 format: 1 sign, 5 exponent (bias 15), 2 mantissa bits.
/// Max finite: sign=0, exp=30, mant=3 → 0x7B (positive) / 0xFB (negative).

/// Convert float to FP8 E5M2 byte.
__device__ inline uint8_t float_to_fp8_e5m2(float value) {
    uint32_t bits = __float_as_uint(value);
    uint8_t sign = (uint8_t)(bits >> 31);
    int16_t exp = (int16_t)((bits >> 23) & 0xFF);
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exp == 0xFF) {
        if (mantissa != 0) return sign == 0 ? 0x7F : 0xFF;  // NaN
        return sign == 0 ? 0x7C : 0xFC;  // Inf
    }
    if (exp == 0 && mantissa == 0) {
        return sign == 0 ? 0x00 : 0x80;
    }

    int16_t fp8_exp = exp - 127 + 15;
    if (fp8_exp >= 0x1F) {
        return sign == 0 ? 0x7B : 0xFB;  // Clamp to max finite
    }
    if (fp8_exp < 0) {
        return sign == 0 ? 0x00 : 0x80;
    }

    uint8_t fp8_mant = (uint8_t)(mantissa >> 21);
    return (sign << 7) | ((uint8_t)fp8_exp << 2) | (fp8_mant & 0x3);
}

/// Convert FP8 E5M2 byte to float.
__device__ inline float fp8_e5m2_to_float(uint8_t value) {
    uint8_t sign = value >> 7;
    uint8_t exp = (value >> 2) & 0x1F;
    uint8_t mant = value & 0x3;

    if (exp == 0x1F) {
        if (mant != 0) return __int_as_float(0x7fc00000u);  // NaN
        return __int_as_float(0x7fc00000u);  // NaN
    }
    if (exp == 0 && mant == 0) {
        return sign == 0 ? 0.0f : -0.0f;
    }

    uint32_t fp32_exp = exp == 0 ? 0 : (uint32_t)exp + 112;
    uint32_t fp32_mant = (uint32_t)mant << 21;
    return __uint_as_float((sign << 31) | (fp32_exp << 23) | fp32_mant);
}

// ── Quantize kernel: BF16 → FP8 ──────────────────────────────────────

/// Quantize BF16 values to FP8 format.
///
/// Grid: 1D, (N + INFERS_BLOCK_SIZE - 1) / INFERS_BLOCK_SIZE blocks.
/// Each thread converts one BF16 element to FP8.
///
/// mode: 0 = E4M3, 1 = E5M2
extern "C" {
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_fp8_quantize_bf16(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    int N,
    int mode
) {
    int idx = INFERS_THREAD_IDX;
    if (idx >= N) return;

    float val = __bfloat162float(input[idx]);

    if (mode == 0) {
        output[idx] = float_to_fp8_e4m3(val);
    } else {
        output[idx] = float_to_fp8_e5m2(val);
    }
}

// ── Dequantize kernel: FP8 → BF16 ────────────────────────────────────

/// Dequantize FP8 bytes to BF16 values.
///
/// Grid: 1D, (N + INFERS_BLOCK_SIZE - 1) / INFERS_BLOCK_SIZE blocks.
/// Each thread converts one FP8 byte to BF16.
///
/// mode: 0 = E4M3, 1 = E5M2
__launch_bounds__(INFERS_BLOCK_SIZE)
__global__ void infers_fp8_dequantize_bf16(
    const uint8_t* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int N,
    int mode
) {
    int idx = INFERS_THREAD_IDX;
    if (idx >= N) return;

    float val;
    if (mode == 0) {
        val = fp8_e4m3_to_float(input[idx]);
    } else {
        val = fp8_e5m2_to_float(input[idx]);
    }

    output[idx] = __float2bfloat16(val);
}

} // extern "C"
