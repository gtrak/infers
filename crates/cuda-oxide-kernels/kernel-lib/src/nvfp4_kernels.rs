//! NVFP4 kernels — dequantize, fused GEMM, K-split.

use cuda_device::{cuda_module, kernel, launch_bounds, thread, DisjointSlice};
use super::shared::*;

#[cuda_module]
pub mod nvfp4 {
    use super::*;

    /// Dequantize NVFP4 weights to BF16.
    ///
    /// NVFP4 packing: each byte holds 2 FP4 values (E2M1 format).
    /// Scales: fp8_e4m3 per-group, weight global scale (f32 scalar), and input global scale (f32 scalar).
    ///
    /// weight_packed: [N, K/2] u8 (each byte = 2 FP4 values)
    /// weight_scale: [N, K/group_size] fp8_e4m3 per-group scale
    /// weight_global_scale: f32 global scale scalar
    /// output: [N, K] bf16
    #[kernel]
    pub fn nvfp4_dequant_to_bf16(
        mut output: DisjointSlice<u16>,   // [N, K] bf16
        weight_packed: &[u8],              // [N, K/2] packed FP4
        weight_scale: &[u8],               // [N, K/group_size] fp8_e4m3
        weight_global_scale: f32,          // scalar global scale
        n: u32,
        k: u32,
        group_size: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let row = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if row >= n as usize { return; }

        let num_groups = (k / group_size) as usize;
        let k_usize = k as usize;
        let group_size_usize = group_size as usize;

        for g in 0..num_groups {
            // Load per-group scale (fp8_e4m3 → f32)
            let scale_fp8 = weight_scale[row * num_groups + g];
            let scale = Fp8E4M3::dequantize(scale_fp8);

            // Decode FP4 values within this group
            for i in 0..(group_size_usize / 2) {
                let byte_idx = row * (k_usize / 2) + g * group_size_usize / 2 + i;
                let packed_byte = weight_packed[byte_idx];

                // High nibble (bits 7:4)
                let hi_nibble = (packed_byte >> 4) & 0xF;
                let hi_val = fp4_e2m1_to_f32(hi_nibble);
                let hi_fp32 = hi_val * scale / weight_global_scale;

                // Low nibble (bits 3:0)
                let lo_nibble = packed_byte & 0xF;
                let lo_val = fp4_e2m1_to_f32(lo_nibble);
                let lo_fp32 = lo_val * scale / weight_global_scale;

                let out_base = row * k_usize + g * group_size_usize + i * 2;
                // NOTE: compressed-tensors packs two FP4 values per byte as [HIGH | LOW].
                // The unpack order is LOW first, then HIGH:
                //   combined = stack([low, high])  => [low, high]
                // So out_base gets LOW nibble, out_base+1 gets HIGH nibble.
                // Source: https://github.com/vllm-project/vllm/pull/16362
                unsafe {
                    *output.get_unchecked_mut(out_base) = f32_to_bf16(lo_fp32);
                    *output.get_unchecked_mut(out_base + 1) = f32_to_bf16(hi_fp32);
                }
            }
        }
    }

    /// Fused NVFP4 GEMM: dequantize FP4 weights in registers and multiply
    /// with BF16 activations, accumulating into BF16 output.
    ///
    /// Reads compressed NVFP4 weights directly (no intermediate bf16 buffer).
    /// Weight layout: [N, K/2] packed FP4 + [N, K/group_size] fp8_e4m3 scales.
    /// Thread mapping: one thread per (row, col) of the output matrix [M, N].
    #[kernel]
    pub fn nvfp4_gemm_fused(
        mut output: DisjointSlice<u16>,   // [M, N] bf16
        weight_packed: &[u8],              // [N, K/2] packed FP4
        weight_scale: &[u8],               // [N, K/group_size] fp8_e4m3
        input: &[u16],                      // [M, K] bf16
        weight_global_scale: f32,           // scalar global scale
        m: u32, n: u32, k: u32,
        group_size: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let row = (thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y()) as i32;
        let col = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;

        if row >= m as i32 || col >= n as i32 {
            return;
        }

        let mut acc: f32 = 0.0;
        let k_usize = k as usize;
        let n_usize = n as usize;
        let gs = group_size as usize;
        let num_groups = (k as usize) / gs;

      // Iterate over groups
        for g in 0..num_groups {
            // Load scale once per group and precompute effective_scale (hoist division)
            let scale_fp8 = weight_scale[col as usize * num_groups + g];
            let scale = Fp8E4M3::dequantize(scale_fp8);
            let effective_scale = scale / weight_global_scale;

            // Read 4 bytes at a time as u32, processing 8 FP4 values per iteration
            for i in 0..(gs / 8) {
                let byte_offset = col as usize * (k_usize / 2) + g * gs / 2 + i * 4;
                let packed_u32: u32 = unsafe {
                    let b0 = *weight_packed.get_unchecked(byte_offset) as u32;
                    let b1 = *weight_packed.get_unchecked(byte_offset + 1) as u32;
                    let b2 = *weight_packed.get_unchecked(byte_offset + 2) as u32;
                    let b3 = *weight_packed.get_unchecked(byte_offset + 3) as u32;
                    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                };

                for w in 0..8usize {
                    let nibble = ((packed_u32 >> (w * 4)) & 0xF) as u8;
                    let w_f32 = fp4_e2m1_to_f32(nibble) * effective_scale;
                    let w_bf16 = f32_to_bf16(w_f32);
                    let mut w_val = f32::from_bits((w_bf16 as u32) << 16);
                    if !w_val.is_finite() { w_val = 0.0; }

                    let ki = g * gs + i * 8 + w;
                    let input_val = f32::from_bits((input[row as usize * k_usize + ki] as u32) << 16);
                    acc += w_val * input_val;
                }
            }
        }

        unsafe {
            *output.get_unchecked_mut(row as usize * n_usize + col as usize) = f32_to_bf16(acc);
        }
    }

    /// Fused NVFP4 GEMM with K-splitting for M=1 decode.
    #[kernel]
    #[launch_bounds(64)]
    pub fn nvfp4_gemm_fused_ksplit(
        partial_sums: &mut [f32],         // [K_SPLIT, N] f32 output
        weight_packed: &[u8],             // [N, K/2] packed FP4
        weight_scale: &[u8],              // [N, K/group_size] fp8_e4m3
        input: &[u16],                    // [K] bf16 (M=1)
        weight_global_scale: f32,         // scalar global scale
        n: u32, k: u32,
        group_size: u32,
        k_split: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let col = (thread::blockIdx_x() * 64u32 + thread::threadIdx_x()) as usize;
        let split_idx = thread::blockIdx_y() as usize;
        let n_usize = n as usize;
        let k_usize = k as usize;
        let gs = group_size as usize;

        if col >= n_usize {
            return;
        }

        let k_per_split = (k_usize + k_split as usize - 1) / k_split as usize;
        let k_start = split_idx * k_per_split;
        let k_end = (k_start + k_per_split).min(k_usize);

        // Align to group boundaries
        let k_start_aligned = (k_start / gs) * gs;
        let k_end_aligned = ((k_end + gs - 1) / gs) * gs;
        let k_end_aligned = k_end_aligned.min(k_usize);

        let num_groups = k_usize / gs;
        let mut acc: f32 = 0.0;

        for g in (k_start_aligned / gs..k_end_aligned / gs) {
            let kg = g * gs;
            if kg >= k_end {
                break;
            }

            let scale_fp8 = weight_scale[col * num_groups + g];
            let scale = Fp8E4M3::dequantize(scale_fp8);
            let effective_scale = scale / weight_global_scale;

            for i in 0..(gs / 8) {
                let byte_offset = col * (k_usize / 2) + g * gs / 2 + i * 4;
                if byte_offset + 3 >= weight_packed.len() {
                    break;
                }
                let packed_u32: u32 = {
                    let b0 = weight_packed[byte_offset] as u32;
                    let b1 = weight_packed[byte_offset + 1] as u32;
                    let b2 = weight_packed[byte_offset + 2] as u32;
                    let b3 = weight_packed[byte_offset + 3] as u32;
                    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                };

                for w in 0..8usize {
                    let k_full = g * gs + i * 8 + w;
                    if k_full < k_start || k_full >= k_end {
                        continue;
                    }
                    let nibble = ((packed_u32 >> (w * 4)) & 0xF) as u8;
                    let w_f32 = fp4_e2m1_to_f32(nibble) * effective_scale;
                    let w_bf16 = f32_to_bf16(w_f32);
                    let mut w_val = f32::from_bits((w_bf16 as u32) << 16);
                    if !w_val.is_finite() { w_val = 0.0; }
                    let input_val = f32::from_bits((input[k_full] as u32) << 16);
                    acc += w_val * input_val;
                }
            }
        }

        partial_sums[split_idx * n_usize + col] = acc;
    }

    /// Fused NVFP4 GEMM v3 with K-splitting for M=1 decode.
    ///
    /// Bandwidth-focused rewrite of [`nvfp4_gemm_fused_ksplit`]:
    /// 1. Four independent f32 accumulators to expose FMA pipeline depth (ILP).
    /// 2. Group-aligned K-split via ceil-grouping: full quantization groups are
    ///    distributed across splits so each is covered exactly once regardless of
    ///    divisibility — the per-element inner loop is fully branchless (no
    ///    k_start/k_end fixups). Empty splits write a zero partial sum.
    /// 3. Two-u32 (16 FP4) stride per outer step: for group_size=16, each group
    ///    has exactly 2 u32s — packed0 → acc0/acc1, packed1 → acc2/acc3 so the
    ///    second global load overlaps the first chunk's compute.
    /// 4. Per-group `effective_scale = dequant(scale) / global_scale` hoist.
    /// 5. BF16 rounding roundtrip and is_finite guard preserved for calibration match.
    ///
    /// Requires `k % group_size == 0`.
    /// - Grid: (ceil(N/64), k_split, 1)  ·  Block: (64, 1, 1)
    /// - blockIdx.x: output column tile  ·  blockIdx.y: K-split index
    #[kernel]
    #[launch_bounds(64)]
    pub fn nvfp4_gemm_v3_ksplit(
        partial_sums: &mut [f32],         // [K_SPLIT, N] f32 output
        weight_packed: &[u8],             // [N, K/2] packed FP4
        weight_scale: &[u8],              // [N, K/group_size] fp8_e4m3
        input: &[u16],                    // [K] bf16 (M=1)
        weight_global_scale: f32,         // scalar global scale
        n: u32, k: u32,
        group_size: u32,
        k_split: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let col = (thread::blockIdx_x() * 64u32 + thread::threadIdx_x()) as usize;
        let split_idx = thread::blockIdx_y() as usize;
        let n_usize = n as usize;
        let k_usize = k as usize;
        let gs = group_size as usize;

        if col >= n_usize {
            return;
        }

        // v3: distribute full groups across splits via ceil-grouping so every
        // group is covered exactly once regardless of divisibility. Last split(s)
        // may be shorter or empty. Per-element inner loop stays branchless.
        let num_groups = k_usize / gs;
        let groups_per_split = (num_groups + k_split as usize - 1) / k_split as usize;
        let group_start = split_idx * groups_per_split;
        if group_start >= num_groups {
            partial_sums[split_idx * n_usize + col] = 0.0;
            return;
        }
        let group_end = if group_start + groups_per_split > num_groups {
            num_groups
        } else {
            group_start + groups_per_split
        };

        let mut acc0: f32 = 0.0;
        let mut acc1: f32 = 0.0;
        let mut acc2: f32 = 0.0;
        let mut acc3: f32 = 0.0;

        for group_idx in group_start..group_end {
            let kg = group_idx * gs;

            // Per-group scale (fp8_e4m3 → f32) and hoist effective_scale division.
            let scale_fp8 = unsafe { *weight_scale.get_unchecked(col * num_groups + group_idx) };
            let scale = Fp8E4M3::dequantize(scale_fp8);
            let effective_scale = scale / weight_global_scale;

            // Two-u32 (16 FP4) stride: packed0 → acc0/acc1, packed1 → acc2/acc3.
            unsafe {
                // packed0: bytes [kg/2 .. kg/2+4] → 8 FP4 nibbles → acc0 (even), acc1 (odd).
                let byte_offset0 = col * (k_usize / 2) + group_idx * gs / 2;
                let b0 = *weight_packed.get_unchecked(byte_offset0) as u32;
                let b1 = *weight_packed.get_unchecked(byte_offset0 + 1) as u32;
                let b2 = *weight_packed.get_unchecked(byte_offset0 + 2) as u32;
                let b3 = *weight_packed.get_unchecked(byte_offset0 + 3) as u32;
                let packed0: u32 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);

                // Dequant nibbles from packed0 into acc0/acc1 with bf16 rounding.
                let n0 = (packed0 & 0xF) as u8;
                let n1 = ((packed0 >> 4) & 0xF) as u8;
                let n2 = ((packed0 >> 8) & 0xF) as u8;
                let n3 = ((packed0 >> 12) & 0xF) as u8;
                let n4 = ((packed0 >> 16) & 0xF) as u8;
                let n5 = ((packed0 >> 20) & 0xF) as u8;
                let n6 = ((packed0 >> 24) & 0xF) as u8;
                let n7 = ((packed0 >> 28) & 0xF) as u8;

                let a0 = f32::from_bits((*input.get_unchecked(kg + 0) as u32) << 16);
                let a1 = f32::from_bits((*input.get_unchecked(kg + 1) as u32) << 16);
                let a2 = f32::from_bits((*input.get_unchecked(kg + 2) as u32) << 16);
                let a3 = f32::from_bits((*input.get_unchecked(kg + 3) as u32) << 16);
                let a4 = f32::from_bits((*input.get_unchecked(kg + 4) as u32) << 16);
                let a5 = f32::from_bits((*input.get_unchecked(kg + 5) as u32) << 16);
                let a6 = f32::from_bits((*input.get_unchecked(kg + 6) as u32) << 16);
                let a7 = f32::from_bits((*input.get_unchecked(kg + 7) as u32) << 16);

                let w0_bf16 = f32_to_bf16(fp4_e2m1_to_f32(n0) * effective_scale);
                let mut w0 = f32::from_bits((w0_bf16 as u32) << 16);
                if !w0.is_finite() { w0 = 0.0; }

                let w1_bf16 = f32_to_bf16(fp4_e2m1_to_f32(n1) * effective_scale);
                let mut w1 = f32::from_bits((w1_bf16 as u32) << 16);
                if !w1.is_finite() { w1 = 0.0; }

                let w2_bf16 = f32_to_bf16(fp4_e2m1_to_f32(n2) * effective_scale);
                let mut w2 = f32::from_bits((w2_bf16 as u32) << 16);
                if !w2.is_finite() { w2 = 0.0; }

                let w3_bf16 = f32_to_bf16(fp4_e2m1_to_f32(n3) * effective_scale);
                let mut w3 = f32::from_bits((w3_bf16 as u32) << 16);
                if !w3.is_finite() { w3 = 0.0; }

                let w4_bf16 = f32_to_bf16(fp4_e2m1_to_f32(n4) * effective_scale);
                let mut w4 = f32::from_bits((w4_bf16 as u32) << 16);
                if !w4.is_finite() { w4 = 0.0; }

                let w5_bf16 = f32_to_bf16(fp4_e2m1_to_f32(n5) * effective_scale);
                let mut w5 = f32::from_bits((w5_bf16 as u32) << 16);
                if !w5.is_finite() { w5 = 0.0; }

                let w6_bf16 = f32_to_bf16(fp4_e2m1_to_f32(n6) * effective_scale);
                let mut w6 = f32::from_bits((w6_bf16 as u32) << 16);
                if !w6.is_finite() { w6 = 0.0; }

                let w7_bf16 = f32_to_bf16(fp4_e2m1_to_f32(n7) * effective_scale);
                let mut w7 = f32::from_bits((w7_bf16 as u32) << 16);
                if !w7.is_finite() { w7 = 0.0; }

                acc0 += w0 * a0;
                acc1 += w1 * a1;
                acc0 += w2 * a2;
                acc1 += w3 * a3;
                acc0 += w4 * a4;
                acc1 += w5 * a5;
                acc0 += w6 * a6;
                acc1 += w7 * a7;

                // packed1: bytes [kg/2+4 .. kg/2+8] → 8 FP4 nibbles → acc2 (even), acc3 (odd).
                let byte_offset1 = byte_offset0 + 4;
                let c0 = *weight_packed.get_unchecked(byte_offset1) as u32;
                let c1 = *weight_packed.get_unchecked(byte_offset1 + 1) as u32;
                let c2 = *weight_packed.get_unchecked(byte_offset1 + 2) as u32;
                let c3 = *weight_packed.get_unchecked(byte_offset1 + 3) as u32;
                let packed1: u32 = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24);

                // Dequant nibbles from packed1 into acc2/acc3 with bf16 rounding.
                let p0 = (packed1 & 0xF) as u8;
                let p1 = ((packed1 >> 4) & 0xF) as u8;
                let p2 = ((packed1 >> 8) & 0xF) as u8;
                let p3 = ((packed1 >> 12) & 0xF) as u8;
                let p4 = ((packed1 >> 16) & 0xF) as u8;
                let p5 = ((packed1 >> 20) & 0xF) as u8;
                let p6 = ((packed1 >> 24) & 0xF) as u8;
                let p7 = ((packed1 >> 28) & 0xF) as u8;

                let a8 = f32::from_bits((*input.get_unchecked(kg + 8) as u32) << 16);
                let a9 = f32::from_bits((*input.get_unchecked(kg + 9) as u32) << 16);
                let a10 = f32::from_bits((*input.get_unchecked(kg + 10) as u32) << 16);
                let a11 = f32::from_bits((*input.get_unchecked(kg + 11) as u32) << 16);
                let a12 = f32::from_bits((*input.get_unchecked(kg + 12) as u32) << 16);
                let a13 = f32::from_bits((*input.get_unchecked(kg + 13) as u32) << 16);
                let a14 = f32::from_bits((*input.get_unchecked(kg + 14) as u32) << 16);
                let a15 = f32::from_bits((*input.get_unchecked(kg + 15) as u32) << 16);

                let v0_bf16 = f32_to_bf16(fp4_e2m1_to_f32(p0) * effective_scale);
                let mut v0 = f32::from_bits((v0_bf16 as u32) << 16);
                if !v0.is_finite() { v0 = 0.0; }

                let v1_bf16 = f32_to_bf16(fp4_e2m1_to_f32(p1) * effective_scale);
                let mut v1 = f32::from_bits((v1_bf16 as u32) << 16);
                if !v1.is_finite() { v1 = 0.0; }

                let v2_bf16 = f32_to_bf16(fp4_e2m1_to_f32(p2) * effective_scale);
                let mut v2 = f32::from_bits((v2_bf16 as u32) << 16);
                if !v2.is_finite() { v2 = 0.0; }

                let v3_bf16 = f32_to_bf16(fp4_e2m1_to_f32(p3) * effective_scale);
                let mut v3 = f32::from_bits((v3_bf16 as u32) << 16);
                if !v3.is_finite() { v3 = 0.0; }

                let v4_bf16 = f32_to_bf16(fp4_e2m1_to_f32(p4) * effective_scale);
                let mut v4 = f32::from_bits((v4_bf16 as u32) << 16);
                if !v4.is_finite() { v4 = 0.0; }

                let v5_bf16 = f32_to_bf16(fp4_e2m1_to_f32(p5) * effective_scale);
                let mut v5 = f32::from_bits((v5_bf16 as u32) << 16);
                if !v5.is_finite() { v5 = 0.0; }

                let v6_bf16 = f32_to_bf16(fp4_e2m1_to_f32(p6) * effective_scale);
                let mut v6 = f32::from_bits((v6_bf16 as u32) << 16);
                if !v6.is_finite() { v6 = 0.0; }

                let v7_bf16 = f32_to_bf16(fp4_e2m1_to_f32(p7) * effective_scale);
                let mut v7 = f32::from_bits((v7_bf16 as u32) << 16);
                if !v7.is_finite() { v7 = 0.0; }

                acc2 += v0 * a8;
                acc3 += v1 * a9;
                acc2 += v2 * a10;
                acc3 += v3 * a11;
                acc2 += v4 * a12;
                acc3 += v5 * a13;
                acc2 += v6 * a14;
                acc3 += v7 * a15;
            }
        }

        let acc = acc0 + acc1 + acc2 + acc3;
        partial_sums[split_idx * n_usize + col] = acc;
    }
}
