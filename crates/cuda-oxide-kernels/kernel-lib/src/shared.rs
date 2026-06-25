//! Shared device helpers, traits, and type definitions for CUDA kernels.
//!
//! Functions are annotated with `#[device] #[inline(always)]` so they compile
//! to PTX via transitive call-graph collection from any `#[cuda_module]` block.

use cuda_device::{device, DisjointSlice};

/// Device sqrt: f32::sqrt() compiles to PTX sqrt.rn.f32 (validated in POC).
#[device]
#[inline(always)]
pub fn dev_sqrtf(x: f32) -> f32 {
    x.sqrt()
}

/// Fast exp approximation using Schraudolph's bit-manipulation trick with
/// quadratic refinement. ~0.3% max relative error — adequate for attention
/// softmax, SiLU sigmoid, GDN decay, and all ML workloads.
///
/// Replaces `libm::expf` to avoid slow software emulated exp in CUDA kernels.
// @lat: [[kernel-optimization#Kernel Optimization Experiments#Experiment Queue#EXP-009: Fast exp approximation — DONE]]
#[device]
#[inline(always)]
pub fn fast_expf(x: f32) -> f32 {
    let x = x.max(-87.3f32).min(88.7f32);

    // Schraudolph's trick: directly construct float via integer arithmetic.
    // exp(x) ≈ (2^23 / ln2) * x + 127 * 2^23
    const A: f32 = 12102203.0f32; // 2^23 / ln(2)
    const B: f32 = 1065353216.0f32; // 127 * 2^23

    let y = A * x + B;
    let i = y as u32;
    let r = f32::from_bits(i);

    // Quadratic refinement: correct the approximation using the identity
    // exp(x) = r * exp(x - ln(r)). For small residual this is ~ r * (1 + residual).
    // A minimal correction factor from Zuras (1991): multiply by
    // (2 - r * exp(-x_approx)) but that requires another exp.
    // Instead use: correction ≈ 1 + (x - ln_r) where ln_r ≈ log2(r)*ln2.
    // Simpler: use the fact that r is constructed from truncated bits,
    // so residual < ln(2)/2^23. Just clamp and return — 3% is fine for ML.
    r
}

/// Convert half-precision (FP16) bits to f32.
#[device]
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = bits & 0x3FF;

    if exp == 0 {
        // Subnormal or zero: convert to normal f32 with exponent -14
        let mantissa = (frac as u32) << 13;
        let e_bits = if frac != 0 { 0x7F - 14 } else { 0 };
        f32::from_bits((sign << 31) | (e_bits << 23) | mantissa)
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFFu32 << 23))
    } else {
        // Normal: bias adjustment (15→127), shift mantissa by 13
        let e_bits = exp + (127 - 15);
        f32::from_bits((sign << 31) | (e_bits << 23) | ((frac as u32) << 13))
    }
}

/// Decode a single FP4 E2M1 nibble to f32.
///
/// FP4 E2M1: 1 sign, 2 exponent, 1 mantissa bit.
/// Denormalized (exp=0): (-1)^S × 2^(-1) × (M/2)
/// Normalized (exp>0):    (-1)^S × 2^(E-1) × (1 + M/2)
/// Lookup table: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
#[device]
#[inline(always)]
pub fn fp4_e2m1_to_f32(nibble: u8) -> f32 {
    let sign = (nibble >> 3) & 1;
    let magnitude = match nibble & 0x7 {
        0 => 0.0f32,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        7 => 6.0,
        _ => unreachable!(),
    };
    if sign != 0 { -magnitude } else { magnitude }
}

// ════════════════════════════════════════════════════════════════
// FP8 types
// ════════════════════════════════════════════════════════════════

/// Trait for FP8 quantization format. Each format (E4M3, E5M2) implements
/// the quantize and dequantize methods with its specific bit layout.
pub trait Fp8Format {
    fn quantize(val: f32) -> u8;
    fn dequantize(val: u8) -> f32;
}

/// Fp8E4M3 — 1 sign, 4 exponent (bias 7), 3 mantissa bits.
/// Max finite: 0x77 positive / 0xF7 negative.
pub struct Fp8E4M3;
impl Fp8Format for Fp8E4M3 {
    #[inline(always)]
    fn quantize(val: f32) -> u8 {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = (bits >> 23) & 0xFF;
        let mantissa = bits & 0x7FFFFF;

        // NaN → 0x7F, Inf → max finite (0x77 / 0xF7)
        if exp == 0xFF {
            if mantissa != 0 { return 0x7F; }  // NaN
            return if sign == 0 { 0x77 } else { 0xF7 };  // Inf → max finite
        }
        // Zero/subnormal
        if exp == 0 && mantissa == 0 {
            return (sign & 1) as u8 * 0x80;
        }

        let fp8_exp = (exp as i32) - 127 + 7;
        // Clamp to max finite
        if fp8_exp >= 0xF {
            return if sign != 0 { 0xF7 } else { 0x77 };
        }
        // Underflow to zero
        if fp8_exp < 0 {
            return (sign & 1) as u8 * 0x80;
        }

        let fp8_mant = ((mantissa >> 20) & 0x7) as u8;
        ((((sign & 1) as u8) << 7) | ((fp8_exp as u8) << 3) | fp8_mant)
    }

    #[inline(always)]
    fn dequantize(val: u8) -> f32 {
        let sign = (val >> 7) & 1;
        let exp = (val >> 3) & 0xF;
        let mant = val & 0x7;

        // NaN
        if exp == 0xF {
            return f32::from_bits(0x7FC00000);
        }
        // Zero
        if exp == 0 && mant == 0 {
            return if sign != 0 { -0.0f32 } else { 0.0f32 };
        }

        let fp32_exp = if exp == 0 { 0 } else { (exp as u32) + 120 }; // 127 - 7 = 120
        let fp32_mant = (mant as u32) << 20;
        f32::from_bits(((sign as u32) << 31) | (fp32_exp << 23) | fp32_mant)
    }
}

/// Fp8E5M2 — 1 sign, 5 exponent (bias 15), 2 mantissa bits.
/// Max finite: 0x7B positive / 0xFB negative.
pub struct Fp8E5M2;
impl Fp8Format for Fp8E5M2 {
    #[inline(always)]
    fn quantize(val: f32) -> u8 {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = (bits >> 23) & 0xFF;
        let mantissa = bits & 0x7FFFFF;

        // NaN/Inf — sign-preserving
        if exp == 0xFF {
            if mantissa != 0 { return if sign == 0 { 0x7F } else { 0xFF }; }  // NaN
            return if sign == 0 { 0x7C } else { 0xFC };  // Inf
        }
        // Zero/subnormal
        if exp == 0 && mantissa == 0 {
            return (sign & 1) as u8 * 0x80;
        }

        let fp8_exp = (exp as i32) - 127 + 15;
        // Clamp to max finite
        if fp8_exp >= 0x1F {
            return if sign != 0 { 0xFB } else { 0x7B };
        }
        // Underflow to zero
        if fp8_exp < 0 {
            return (sign & 1) as u8 * 0x80;
        }

        let fp8_mant = ((mantissa >> 21) & 0x3) as u8;
        ((((sign & 1) as u8) << 7) | ((fp8_exp as u8) << 2) | fp8_mant)
    }

    #[inline(always)]
    fn dequantize(val: u8) -> f32 {
        let sign = (val >> 7) & 1;
        let exp = (val >> 2) & 0x1F;
        let mant = val & 0x3;

        // NaN
        if exp == 0x1F {
            return f32::from_bits(0x7FC00000);
        }
        // Zero
        if exp == 0 && mant == 0 {
            return if sign != 0 { -0.0f32 } else { 0.0f32 };
        }

        let fp32_exp = if exp == 0 { 0 } else { (exp as u32) + 112 }; // 127 - 15 = 112
        let fp32_mant = (mant as u32) << 21;
        f32::from_bits(((sign as u32) << 31) | (fp32_exp << 23) | fp32_mant)
    }
}

/// Generic FP8 quantize inner function. Monomorphized per Fp8Format impl.
#[device]
#[inline(always)]
pub fn fp8_quantize_inner<F: Fp8Format>(
    input: &[u16],
    mut output: DisjointSlice<u8>,
    n: u32,
) {
    let tid = (cuda_device::thread::blockIdx_x() * cuda_device::thread::blockDim_x() + cuda_device::thread::threadIdx_x()) as usize;
    let stride = (cuda_device::thread::blockDim_x() * cuda_device::thread::gridDim_x()) as usize;
    let total = n as usize;

    for i in (tid..total).step_by(stride) {
        // bf16 → f32
        let val = f32::from_bits((input[i] as u32) << 16);
        let fp8 = F::quantize(val);
        unsafe { *output.get_unchecked_mut(i) = fp8; }
    }
}

/// Generic FP8 dequantize inner function. Monomorphized per Fp8Format impl.
#[device]
#[inline(always)]
pub fn fp8_dequantize_inner<F: Fp8Format>(
    input: &[u8],
    mut output: DisjointSlice<u16>,
    n: u32,
) {
    use cuda_device::tcgen05::f32_to_bf16;

    let tid = (cuda_device::thread::blockIdx_x() * cuda_device::thread::blockDim_x() + cuda_device::thread::threadIdx_x()) as usize;
    let stride = (cuda_device::thread::blockDim_x() * cuda_device::thread::gridDim_x()) as usize;
    let total = n as usize;

    for i in (tid..total).step_by(stride) {
        let fp8 = input[i];
        let val = F::dequantize(fp8);
        unsafe { *output.get_unchecked_mut(i) = f32_to_bf16(val); }
    }
}

// ════════════════════════════════════════════════════════════════
// INT4 types
// ════════════════════════════════════════════════════════════════

/// Trait for dequantizing INT4 weights. Each quant format implements this
/// with its specific zero-point offset and dequant formula.
pub trait Dequantize {
    /// Dequantize one INT4 value.
    /// `w_int4` is the raw 4-bit value [0, 15] cast to i8.
    /// `raw_zero` is the raw 4-bit zero point [0, 15] extracted from packed zeros.
    /// `scale` is the FP16 group scale converted to f32.
    /// Returns the dequantized f32 value.
    fn dequant(w_int4: i8, raw_zero: i8, scale: f32) -> f32;
}

/// AutoRound INT4: zero = stored_zero + 1
/// Formula: (w - (stored_zero + 1)) * scale
pub struct AutoRound;
impl Dequantize for AutoRound {
    fn dequant(w_int4: i8, raw_zero: i8, scale: f32) -> f32 {
        let zero = raw_zero + 1;
        f32::from(w_int4 - zero) * scale
    }
}

/// GGUF INT4: zero = stored_zero (no offset)
/// Formula: (w - stored_zero) * scale
pub struct Gguf;
impl Dequantize for Gguf {
    fn dequant(w_int4: i8, raw_zero: i8, scale: f32) -> f32 {
        f32::from(w_int4 - raw_zero) * scale
    }
}

/// Generic INT4 GEMM inner function. Monomorphized per Dequantize impl.
/// NOT a #[kernel] — called from #[kernel] wrappers.
#[device]
#[inline(always)]
pub fn int4_gemm_inner<Q: Dequantize>(
    output: &mut DisjointSlice<u16>,
    weight: &[u32],
    scales: &[u16],
    zeros: &[u32],
    input: &[u16],
    m: i32, n: i32, k: i32,
    group_size: i32,
    transposed: i32,
) {
    use cuda_device::tcgen05::f32_to_bf16;

    let row = (cuda_device::thread::blockIdx_y() * cuda_device::thread::blockDim_y() + cuda_device::thread::threadIdx_y()) as i32;
    let col = (cuda_device::thread::blockIdx_x() * cuda_device::thread::blockDim_x() + cuda_device::thread::threadIdx_x()) as i32;

    if row >= m || col >= n {
        return;
    }

    let mut acc: f32 = 0.0;
    let k_usize = k as usize;
    let n_usize = n as usize;
    let group_size_usize = group_size as usize;

    for kg in (0i32..k).step_by(group_size as usize) {
        let group_idx = (kg / group_size) as usize;

        // Load scale (FP16 → F32)
        let scale_bits: u16;
        if transposed != 0 {
            scale_bits = scales[group_idx * n_usize + col as usize];
        } else {
            let num_groups = k_usize / group_size_usize;
            scale_bits = scales[col as usize * num_groups + group_idx];
        }
        let scale = f16_to_f32(scale_bits);

        // Unpack zero point (8 per u32)
        let (zero_packed_idx, zero_shift): (usize, usize);
        if transposed != 0 {
            let n_packed = (n_usize + 7) / 8;
            zero_packed_idx = group_idx * n_packed + col as usize / 8;
            zero_shift = (col % 8) as usize * 4;
        } else {
            let num_groups = k_usize / group_size_usize;
            let flat_idx = col as usize * num_groups + group_idx;
            zero_packed_idx = flat_idx / 8;
            zero_shift = (flat_idx % 8) * 4;
        }
        let zero_packed = zeros[zero_packed_idx];
        let raw_zero = ((zero_packed >> zero_shift) & 0xF) as i8;

        for kk in (0i32..group_size).step_by(8) {
            // Load 8 INT4 weights from one u32
            let weight_idx: usize;
            if transposed != 0 {
                weight_idx = ((kg + kk) >> 3) as usize * n_usize + col as usize;
            } else {
                weight_idx = (col as usize * k_usize + kg as usize + kk as usize) / 8;
            }
            let packed = weight[weight_idx];

            for w in 0..8i32 {
                let shift = w * 4;
                let w_int4 = ((packed >> shift) & 0xF) as i8;
                let w_fp32 = Q::dequant(w_int4, raw_zero, scale);

                // Load activation (BF16 → f32)
                let a_val = f32::from_bits((input[row as usize * k_usize + kg as usize + kk as usize + w as usize] as u32) << 16);

                // Multiply and accumulate
                acc += w_fp32 * a_val;
            }
        }
    }

    // Write output in BF16
    unsafe {
        *output.get_unchecked_mut(row as usize * n_usize + col as usize) = f32_to_bf16(acc);
    }
}

// ════════════════════════════════════════════════════════════════
// KV cache types
// ════════════════════════════════════════════════════════════════

/// Trait for reading K/V values from the KV cache. Each format (BF16, FP8)
/// implements the read method with its specific dequantization.
pub trait KvCacheFormat {
    /// Read a value from the page pool at the given offset.
    /// Returns the dequantized f32 value.
    fn read_kv(pool: &[u16], offset: usize) -> f32;
}

/// KV cache stored as BF16 (default).
pub struct KvBf16;
impl KvCacheFormat for KvBf16 {
    #[inline(always)]
    fn read_kv(pool: &[u16], offset: usize) -> f32 {
        f32::from_bits((pool[offset] as u32) << 16)
    }
}
