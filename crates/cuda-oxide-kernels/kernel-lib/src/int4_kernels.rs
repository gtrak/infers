//! INT4 quantized GEMM kernels — AutoRound, GGUF, warp-based, dequantize.

use cuda_device::{cuda_module, kernel, launch_bounds, thread, DisjointSlice, DynamicSharedArray};
use super::shared::*;

#[cuda_module]
pub mod int4 {
    use super::*;

    /// INT4 GEMM kernel for AutoRound format (zero offset +1).
    #[kernel]
    pub fn int4_gemm_auto_round(
        mut output: DisjointSlice<u16>,
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        m: u32, n: u32, k: u32,
        group_size: u32, transposed: u32,
    ) {
        int4_gemm_inner::<AutoRound>(
            &mut output, weight, scales, zeros, input,
            m as i32, n as i32, k as i32,
            group_size as i32, transposed as i32,
        );
    }

    /// INT4 GEMM kernel for AutoRound format with shared memory input tiling.
    ///
    /// Optimized for M=1 decode: loads input vector chunks into shared memory
    /// so all threads in a block share one copy instead of each reading from
    /// global memory independently.
    ///
    /// - Block: (64, 1) — 64 threads compute 64 output columns
    /// - Grid: (ceil(N/64), M, 1)
    /// - Smem: K_TILE * sizeof(u16) = 128 * 2 = 256 bytes (input staging)
    /// - K_TILE = group_size (128): aligns smem chunks with quant groups
    #[kernel]
    #[launch_bounds(64)]
    pub fn int4_gemm_auto_round_tiled(
        mut output: DisjointSlice<u16>,
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        m: u32, n: u32, k: u32,
        group_size: u32, transposed: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let tid = thread::threadIdx_x() as usize;
        let col = (thread::blockIdx_x() * 64u32 + thread::threadIdx_x()) as usize;
        let row = thread::blockIdx_y() as usize;

        let n_usize = n as usize;
        let k_usize = k as usize;
        let group_size_usize = group_size as usize;

        // Only check row boundary before cooperative load — all threads in a block
        // share the same row and must participate together. Column boundary is checked
        // later to allow valid columns to compute while invalid ones still cooperate
        // on shared memory loads.
        if row >= m as usize {
            return;
        }

        // Shared memory: K_TILE = group_size input values (BF16 as u16)
        let smem = DynamicSharedArray::<u16>::get();

        let mut acc: f32 = 0.0;

        // Iterate K in chunks of K_TILE (= group_size)
        for kg in (0..k_usize).step_by(group_size_usize) {
            let k_tile_end = (kg + group_size_usize).min(k_usize);
            let k_tile_len = k_tile_end - kg;

            // Cooperative load: ALL 64 threads participate using tid (not col),
            // so even when N < GROUP_SIZE, the full K tile is loaded into smem.
            for i in (tid..k_tile_len).step_by(64) {
                unsafe {
                    *smem.add(i) = input[row * k_usize + kg + i];
                }
            }
            cuda_device::sync_threads();

            // Only threads with valid columns do the GEMM computation
            if col < n_usize {
                // Load scale for this group
                let group_idx = kg / group_size_usize;
                let scale_bits: u16;
                if transposed != 0 {
                    scale_bits = scales[group_idx * n_usize + col];
                } else {
                    let num_groups = k_usize / group_size_usize;
                    scale_bits = scales[col * num_groups + group_idx];
                }
                let scale = f16_to_f32(scale_bits);

                // Unpack zero point
                let (zero_packed_idx, zero_shift): (usize, usize);
                if transposed != 0 {
                    let n_packed = (n_usize + 7) / 8;
                    zero_packed_idx = group_idx * n_packed + col / 8;
                    zero_shift = (col % 8) * 4;
                } else {
                    let num_groups = k_usize / group_size_usize;
                    let flat_idx = col * num_groups + group_idx;
                    zero_packed_idx = flat_idx / 8;
                    zero_shift = (flat_idx % 8) * 4;
                }
                let zero_packed = zeros[zero_packed_idx];
                let raw_zero = ((zero_packed >> zero_shift) & 0xF) as i8;

                // Process 8 INT4 weights at a time (one u32)
                for kk in (0..k_tile_len).step_by(8) {
                    let weight_idx: usize;
                    if transposed != 0 {
                        weight_idx = ((kg + kk) >> 3) * n_usize + col;
                    } else {
                        weight_idx = (col * k_usize + kg + kk) / 8;
                    }
                    let packed = weight[weight_idx];

                    for w in 0..8usize {
                        let shift = w * 4;
                        let w_int4 = ((packed >> shift) & 0xF) as i8;
                        // AutoRound dequant: (w_int4 - (raw_zero + 1)) * scale
                        let w_fp32 = (w_int4 as f32 - (raw_zero as f32 + 1.0)) * scale;

                        // Read input from shared memory
                        let a_val = f32::from_bits((unsafe { *smem.add(kk + w) } as u32) << 16);

                        acc += w_fp32 * a_val;
                    }
                }
            }

            cuda_device::sync_threads();
        }

        // Write output in BF16 only for valid columns
        if col < n_usize {
            unsafe {
                *output.get_unchecked_mut(row * n_usize + col) = f32_to_bf16(acc);
            }
        }
    }

    /// INT4 GEMM kernel with K-splitting for M=1 decode (AutoRound format).
    ///
    /// Splits the K dimension across multiple thread blocks. Each block computes
    /// partial sums for 64 output columns over a portion of K. A subsequent
    /// reduction kernel (`reduce_partial_sums_bf16`) combines the partial sums.
    ///
    /// - Grid: (ceil(N/64), K_SPLIT, 1)
    /// - Block: (64, 1, 1)
    /// - blockIdx.x: output column tile
    /// - blockIdx.y: K-split index (0..K_SPLIT)
    #[kernel]
    #[launch_bounds(64)]
    pub fn int4_gemm_auto_round_ksplit(
        partial_sums: &mut [f32],          // [K_SPLIT, N] f32 output
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        n: u32, k: u32,
        group_size: u32,
        transposed: u32,
        k_split: u32,
    ) {
        let col = (thread::blockIdx_x() * 64u32 + thread::threadIdx_x()) as usize;
        let split_idx = thread::blockIdx_y() as usize;
        let n_usize = n as usize;
        let k_usize = k as usize;
        let group_size_usize = group_size as usize;

        if col >= n_usize {
            return;
        }

        // K range for this split
        let k_per_split = (k_usize + k_split as usize - 1) / k_split as usize;
        let k_start = split_idx * k_per_split;
        let k_end = (k_start + k_per_split).min(k_usize);

        // Align k_start/k_end to group boundaries so we don't split a quantization group
        let k_start_aligned = (k_start / group_size_usize) * group_size_usize;
        let k_end_aligned = ((k_end + group_size_usize - 1) / group_size_usize) * group_size_usize;
        let k_end_aligned = k_end_aligned.min(k_usize);

        let mut acc: f32 = 0.0;

        for kg in (k_start_aligned..k_end_aligned).step_by(group_size_usize) {
            let group_idx = kg / group_size_usize;

            // Load scale
            let scale_bits: u16;
            if transposed != 0 {
                scale_bits = scales[group_idx * n_usize + col];
            } else {
                let num_groups = k_usize / group_size_usize;
                scale_bits = scales[col * num_groups + group_idx];
            }
            let scale = f16_to_f32(scale_bits);

            // Unpack zero point
            let (zero_packed_idx, zero_shift): (usize, usize);
            if transposed != 0 {
                let n_packed = (n_usize + 7) / 8;
                zero_packed_idx = group_idx * n_packed + col / 8;
                zero_shift = (col % 8) * 4;
            } else {
                let num_groups = k_usize / group_size_usize;
                let flat_idx = col * num_groups + group_idx;
                zero_packed_idx = flat_idx / 8;
                zero_shift = (flat_idx % 8) * 4;
            }
            let zero_packed = zeros[zero_packed_idx];
            let raw_zero = ((zero_packed >> zero_shift) & 0xF) as i8;

            for kk in (0..group_size_usize).step_by(8) {
                let k_pos = kg + kk;
                if k_pos >= k_end {
                    break;
                }
                // Skip u32 that starts before this split's range
                if k_pos + 7 < k_start {
                    continue;
                }
                let weight_idx: usize;
                if transposed != 0 {
                    weight_idx = (k_pos >> 3) * n_usize + col;
                } else {
                    weight_idx = (col * k_usize + k_pos) / 8;
                }
                let packed = weight[weight_idx];

                for w in 0..8usize {
                    let k_full = k_pos + w;
                    if k_full >= k_end {
                        break;
                    }
                    // Skip weights before this split's range
                    if k_full < k_start {
                        continue;
                    }
                    let shift = w * 4;
                    let w_int4 = ((packed >> shift) & 0xF) as i8;
                    // AutoRound: zero = stored_zero + 1
                    let w_fp32 = (w_int4 as f32 - (raw_zero as f32 + 1.0)) * scale;
                    let a_val = f32::from_bits((input[k_full] as u32) << 16);
                    acc += w_fp32 * a_val;
                }
            }
        }

        // Write partial sum to partial_sums[split_idx][col]
        partial_sums[split_idx * n_usize + col] = acc;
    }

    /// INT4 GEMM v3 with K-splitting for M=1 decode (AutoRound format).
    ///
    /// Bandwidth-focused rewrite of [`int4_gemm_auto_round_ksplit`]:
    /// 1. Four independent f32 accumulators to expose FMA pipeline depth (ILP).
    /// 2. Group-aligned K-split via ceil-grouping: full quantization groups are
    ///    distributed across splits so each is covered exactly once regardless of
    ///    divisibility — the per-element inner loop is fully branchless (no
    ///    k_start/k_end fixups). Empty splits write a zero partial sum.
    /// 3. Two-u32 (16 INT4) stride per outer step so the second global load
    ///    overlaps the first chunk's compute, hiding DRAM latency.
    /// 4. Per-group `scaled_zero = (raw_zero + 1) * scale` hoist.
    ///
    /// Requires `k % group_size == 0` (AutoRound pads K to group_size).
    /// - Grid: (ceil(N/64), k_split, 1)  ·  Block: (64, 1, 1)
    /// - blockIdx.x: output column tile  ·  blockIdx.y: K-split index
    /// INT4 GEMM v3 with K-splitting and shared memory input tiling.
    ///
    /// Same as [`int4_gemm_v3_ksplit`] but tiles the input vector into shared
    /// memory per group, eliminating 64x redundant DRAM reads of the same data.
    /// All 64 threads cooperatively load one copy of each group's input tile,
    /// then read from shared memory for GEMM computation.
    ///
    /// - Block: (64, 1, 1) · Grid: (ceil(N/64), k_split, 1)
    /// - Smem: group_size * sizeof(u16) = 128 * 2 = 256 bytes per block
    #[kernel]
    #[launch_bounds(64)]
    pub fn int4_gemm_v3_ksplit_sm(
        partial_sums: &mut [f32],
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        n: u32, k: u32,
        group_size: u32,
        transposed: u32,
        k_split: u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let col = (thread::blockIdx_x() * 64u32 + thread::threadIdx_x()) as usize;
        let split_idx = thread::blockIdx_y() as usize;
        let n_usize = n as usize;
        let k_usize = k as usize;
        let gs = group_size as usize;
        let ks = k_split as usize;
        let num_groups = k_usize / gs;

        // v3: distribute full groups across splits via ceil-grouping so every
        // group is covered exactly once regardless of divisibility. Last split(s)
        // may be shorter or empty. Per-element inner loop stays branchless.
        let groups_per_split = (num_groups + ks - 1) / ks;
        let group_start = split_idx * groups_per_split;

        if group_start >= num_groups {
            // Empty split — write zero only for valid columns.
            if col < n_usize {
                partial_sums[split_idx * n_usize + col] = 0.0;
            }
            return;
        }
        let group_end = if group_start + groups_per_split > num_groups {
            num_groups
        } else {
            group_start + groups_per_split
        };

        // Shared memory: one tile per group (group_size bf16 values)
        let smem = DynamicSharedArray::<u16>::get();

        let mut acc0: f32 = 0.0;
        let mut acc1: f32 = 0.0;
        let mut acc2: f32 = 0.0;
        let mut acc3: f32 = 0.0;

        let n_packed = (n_usize + 7) / 8;
        let u32s_per_group = gs / 8;

        for group_idx in group_start..group_end {
            let kg = group_idx * gs;

            // Cooperative load: strided pattern so each thread loads multiple elements
            // when group_size > block_dim.x (e.g. gs=128, block=64 → 2 iterations).
            let mut i = tid;
            while i < gs {
                unsafe {
                    *smem.add(i) = *input.get_unchecked(kg + i);
                }
                i += 64; // block_dim x
            }
            cuda_device::sync_threads();

            // Per-group scale (fp16 → f32).
            let scale_bits: u16 = if transposed != 0 {
                unsafe { *scales.get_unchecked(group_idx * n_usize + col) }
            } else {
                unsafe { *scales.get_unchecked(col * num_groups + group_idx) }
            };
            let scale = f16_to_f32(scale_bits);

            // Per-group packed zero point.
            let (zero_packed_idx, zero_shift): (usize, usize) = if transposed != 0 {
                (group_idx * n_packed + col / 8, (col % 8) * 4)
            } else {
                let flat_idx = col * num_groups + group_idx;
                (flat_idx / 8, (flat_idx % 8) * 4)
            };
            let raw_zero = ((unsafe { *zeros.get_unchecked(zero_packed_idx) } >> zero_shift) & 0xF) as i8;
            let scaled_zero = (raw_zero as f32 + 1.0) * scale;

            if transposed == 0 {
                // Non-transposed: 128-bit LDG.128 loads, 4 u32s at a time = 32 INT4 values.
                // In non-transposed layout weight[(col*K+k)/8], consecutive u32s are adjacent
                // in memory (stride 1 in u32 index), so 4 contiguous u32s = 128-bit load.
                for u32_chunk in (0..u32s_per_group).step_by(4) {
                    let k_base = kg + u32_chunk * 8;
                    let w_idx = (col * k_usize + k_base) / 8;

                    #[cfg(debug_assertions)]
                    assert!(w_idx % 4 == 0, "128-bit load requires 16-byte alignment, w_idx={}", w_idx);

                    let packed4: [u32; 4] = unsafe { *(weight.as_ptr().add(w_idx) as *const [u32; 4]) };

                    for chunk_lane in 0..4usize {
                        let packed = packed4[chunk_lane];
                        let smem_off = (u32_chunk + chunk_lane) * 8;

                        unsafe {
                            let a0 = f32::from_bits((*smem.add(smem_off + 0) as u32) << 16);
                            let a1 = f32::from_bits((*smem.add(smem_off + 1) as u32) << 16);
                            let a2 = f32::from_bits((*smem.add(smem_off + 2) as u32) << 16);
                            let a3 = f32::from_bits((*smem.add(smem_off + 3) as u32) << 16);
                            let a4 = f32::from_bits((*smem.add(smem_off + 4) as u32) << 16);
                            let a5 = f32::from_bits((*smem.add(smem_off + 5) as u32) << 16);
                            let a6 = f32::from_bits((*smem.add(smem_off + 6) as u32) << 16);
                            let a7 = f32::from_bits((*smem.add(smem_off + 7) as u32) << 16);

                            let w0 = (packed & 0xF) as i8;
                            let w1 = ((packed >> 4) & 0xF) as i8;
                            let w2 = ((packed >> 8) & 0xF) as i8;
                            let w3 = ((packed >> 12) & 0xF) as i8;
                            let w4 = ((packed >> 16) & 0xF) as i8;
                            let w5 = ((packed >> 20) & 0xF) as i8;
                            let w6 = ((packed >> 24) & 0xF) as i8;
                            let w7 = ((packed >> 28) & 0xF) as i8;

                            if chunk_lane % 2 == 0 {
                                acc0 += (w0 as f32 * scale - scaled_zero) * a0;
                                acc1 += (w1 as f32 * scale - scaled_zero) * a1;
                                acc0 += (w2 as f32 * scale - scaled_zero) * a2;
                                acc1 += (w3 as f32 * scale - scaled_zero) * a3;
                                acc0 += (w4 as f32 * scale - scaled_zero) * a4;
                                acc1 += (w5 as f32 * scale - scaled_zero) * a5;
                                acc0 += (w6 as f32 * scale - scaled_zero) * a6;
                                acc1 += (w7 as f32 * scale - scaled_zero) * a7;
                            } else {
                                acc2 += (w0 as f32 * scale - scaled_zero) * a0;
                                acc3 += (w1 as f32 * scale - scaled_zero) * a1;
                                acc2 += (w2 as f32 * scale - scaled_zero) * a2;
                                acc3 += (w3 as f32 * scale - scaled_zero) * a3;
                                acc2 += (w4 as f32 * scale - scaled_zero) * a4;
                                acc3 += (w5 as f32 * scale - scaled_zero) * a5;
                                acc2 += (w6 as f32 * scale - scaled_zero) * a6;
                                acc3 += (w7 as f32 * scale - scaled_zero) * a7;
                            }
                        }
                    }
                }
            } else {
                // Transposed: scalar loads (weights are stride-N between columns, not contiguous).
                for u32_idx in (0..u32s_per_group).step_by(2) {
                    let k0 = kg + u32_idx * 8;
                    let w_idx0: usize = (k0 >> 3) * n_usize + col;
                    let packed0: u32 = unsafe { *weight.get_unchecked(w_idx0) };

                    unsafe {
                        let a0 = f32::from_bits((*smem.add(u32_idx * 8 + 0) as u32) << 16);
                        let a1 = f32::from_bits((*smem.add(u32_idx * 8 + 1) as u32) << 16);
                        let a2 = f32::from_bits((*smem.add(u32_idx * 8 + 2) as u32) << 16);
                        let a3 = f32::from_bits((*smem.add(u32_idx * 8 + 3) as u32) << 16);
                        let a4 = f32::from_bits((*smem.add(u32_idx * 8 + 4) as u32) << 16);
                        let a5 = f32::from_bits((*smem.add(u32_idx * 8 + 5) as u32) << 16);
                        let a6 = f32::from_bits((*smem.add(u32_idx * 8 + 6) as u32) << 16);
                        let a7 = f32::from_bits((*smem.add(u32_idx * 8 + 7) as u32) << 16);

                        let w0 = (packed0 & 0xF) as i8;
                        let w1 = ((packed0 >> 4) & 0xF) as i8;
                        let w2 = ((packed0 >> 8) & 0xF) as i8;
                        let w3 = ((packed0 >> 12) & 0xF) as i8;
                        let w4 = ((packed0 >> 16) & 0xF) as i8;
                        let w5 = ((packed0 >> 20) & 0xF) as i8;
                        let w6 = ((packed0 >> 24) & 0xF) as i8;
                        let w7 = ((packed0 >> 28) & 0xF) as i8;

                        acc0 += (w0 as f32 * scale - scaled_zero) * a0;
                        acc1 += (w1 as f32 * scale - scaled_zero) * a1;
                        acc0 += (w2 as f32 * scale - scaled_zero) * a2;
                        acc1 += (w3 as f32 * scale - scaled_zero) * a3;
                        acc0 += (w4 as f32 * scale - scaled_zero) * a4;
                        acc1 += (w5 as f32 * scale - scaled_zero) * a5;
                        acc0 += (w6 as f32 * scale - scaled_zero) * a6;
                        acc1 += (w7 as f32 * scale - scaled_zero) * a7;
                    }

                    if u32_idx + 1 < u32s_per_group {
                        let k1 = k0 + 8;
                        let w_idx1: usize = (k1 >> 3) * n_usize + col;
                        let packed1: u32 = unsafe { *weight.get_unchecked(w_idx1) };

                        unsafe {
                            let b0 = f32::from_bits((*smem.add(u32_idx * 8 + 8) as u32) << 16);
                            let b1 = f32::from_bits((*smem.add(u32_idx * 8 + 9) as u32) << 16);
                            let b2 = f32::from_bits((*smem.add(u32_idx * 8 + 10) as u32) << 16);
                            let b3 = f32::from_bits((*smem.add(u32_idx * 8 + 11) as u32) << 16);
                            let b4 = f32::from_bits((*smem.add(u32_idx * 8 + 12) as u32) << 16);
                            let b5 = f32::from_bits((*smem.add(u32_idx * 8 + 13) as u32) << 16);
                            let b6 = f32::from_bits((*smem.add(u32_idx * 8 + 14) as u32) << 16);
                            let b7 = f32::from_bits((*smem.add(u32_idx * 8 + 15) as u32) << 16);

                            let v0 = (packed1 & 0xF) as i8;
                            let v1 = ((packed1 >> 4) & 0xF) as i8;
                            let v2 = ((packed1 >> 8) & 0xF) as i8;
                            let v3 = ((packed1 >> 12) & 0xF) as i8;
                            let v4 = ((packed1 >> 16) & 0xF) as i8;
                            let v5 = ((packed1 >> 20) & 0xF) as i8;
                            let v6 = ((packed1 >> 24) & 0xF) as i8;
                            let v7 = ((packed1 >> 28) & 0xF) as i8;

                            acc2 += (v0 as f32 * scale - scaled_zero) * b0;
                            acc3 += (v1 as f32 * scale - scaled_zero) * b1;
                            acc2 += (v2 as f32 * scale - scaled_zero) * b2;
                            acc3 += (v3 as f32 * scale - scaled_zero) * b3;
                            acc2 += (v4 as f32 * scale - scaled_zero) * b4;
                            acc3 += (v5 as f32 * scale - scaled_zero) * b5;
                            acc2 += (v6 as f32 * scale - scaled_zero) * b6;
                            acc3 += (v7 as f32 * scale - scaled_zero) * b7;
                        }
                    }
                }
            }

            // Sync before next group overwrites shared memory.
            cuda_device::sync_threads();
        }

        let acc = acc0 + acc1 + acc2 + acc3;
        if col < n_usize {
            partial_sums[split_idx * n_usize + col] = acc;
        }
    }
    ///
    /// Each thread (16/block) handles 4 output columns. Weight loads are 128-bit
    /// ([u32;4]) and input loads are 128-bit ([u16;8]). Hardcodes transposed=1 layout.
    ///
    /// - Block: (16, 1, 1) · Grid: (ceil(N/64), k_split, 1)
    /// - Thread tid handles columns [base_col + tid*4 .. base_col + tid*4+3]
    /// - 8 accumulators: col_c → acc[2c]/acc[2c+1] (even/odd lanes)

    #[kernel]
    #[launch_bounds(64)]
    pub fn int4_gemm_v4_ksplit(
        partial_sums: &mut [f32],
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        n: u32, k: u32,
        group_size: u32,
        transposed: u32,
        k_split: u32,
    ) {
        let base_col = (thread::blockIdx_x() * 64u32) as usize;
        let tid = thread::threadIdx_x() as usize; // 0..15
        let split_idx = thread::blockIdx_y() as usize;
        let n_usize = n as usize;
        let k_usize = k as usize;
        let gs = group_size as usize;
        let ks = k_split as usize;
        let num_groups = k_usize / gs;

        if base_col + tid * 4 >= n_usize {
            return;
        }

        // v3-style ceil-grouping across splits
        let groups_per_split = (num_groups + ks - 1) / ks;
        let group_start = split_idx * groups_per_split;
        if group_start >= num_groups {
            for c in 0..4usize {
                let col = base_col + tid * 4 + c;
                if col < n_usize {
                    partial_sums[split_idx * n_usize + col] = 0.0;
                }
            }
            return;
        }
        let group_end = if group_start + groups_per_split > num_groups {
            num_groups
        } else {
            group_start + groups_per_split
        };

        // 8 accumulators: col_c uses acc[2c] (even) / acc[2c+1] (odd)
        let mut acc0: f32 = 0.0;
        let mut acc1: f32 = 0.0;
        let mut acc2: f32 = 0.0;
        let mut acc3: f32 = 0.0;
        let mut acc4: f32 = 0.0;
        let mut acc5: f32 = 0.0;
        let mut acc6: f32 = 0.0;
        let mut acc7: f32 = 0.0;

        // Pre-compute column base for weight indexing
        let col_base = base_col + tid * 4;

        for group_idx in group_start..group_end {
            let kg = group_idx * gs;

            // Load scale for each of 4 columns (transposed layout)
            let s0 = f16_to_f32(unsafe { *scales.get_unchecked(group_idx * n_usize + col_base) });
            let s1 = f16_to_f32(unsafe { *scales.get_unchecked(group_idx * n_usize + col_base + 1) });
            let s2 = f16_to_f32(unsafe { *scales.get_unchecked(group_idx * n_usize + col_base + 2) });
            let s3 = f16_to_f32(unsafe { *scales.get_unchecked(group_idx * n_usize + col_base + 3) });

            // Load zero for each column (transposed layout)
            let n_packed = (n_usize + 7) / 8;
            let z0_packed = unsafe { *zeros.get_unchecked(group_idx * n_packed + col_base / 8) };
            let rz0 = ((z0_packed >> ((col_base % 8) * 4)) & 0xF) as i8;

            let z1_packed = unsafe { *zeros.get_unchecked(group_idx * n_packed + (col_base + 1) / 8) };
            let rz1 = ((z1_packed >> (((col_base + 1) % 8) * 4)) & 0xF) as i8;

            let z2_packed = unsafe { *zeros.get_unchecked(group_idx * n_packed + (col_base + 2) / 8) };
            let rz2 = ((z2_packed >> (((col_base + 2) % 8) * 4)) & 0xF) as i8;

            let z3_packed = unsafe { *zeros.get_unchecked(group_idx * n_packed + (col_base + 3) / 8) };
            let rz3 = ((z3_packed >> (((col_base + 3) % 8) * 4)) & 0xF) as i8;

            // scaled_zero = (raw_zero + 1) * scale, hoisted per group
            let sz0 = (rz0 as f32 + 1.0) * s0;
            let sz1 = (rz1 as f32 + 1.0) * s1;
            let sz2 = (rz2 as f32 + 1.0) * s2;
            let sz3 = (rz3 as f32 + 1.0) * s3;

            // Two-u32 stride per column: second load overlaps first compute
            for u32_idx in (0..gs / 8).step_by(2) {
                let k0 = kg + u32_idx * 8;

                // 128-bit weight load: 4 consecutive u32s = [col_base, col_base+1, col_base+2, col_base+3]
                let w_ptr0 = unsafe { weight.as_ptr().add((k0 >> 3) * n_usize + col_base) };
                let packed4_0: [u32; 4] = unsafe { *(w_ptr0 as *const [u32; 4]) };

                // 128-bit input load: 8 bf16 values at k0..k0+7
                let a_vals: [u16; 8] = unsafe { *(input.as_ptr().add(k0) as *const [u16; 8]) };

                // Convert input to f32 once, shared across all 4 columns at this k position
                unsafe {
                    let a0 = f32::from_bits((a_vals[0] as u32) << 16);
                    let a1 = f32::from_bits((a_vals[1] as u32) << 16);
                    let a2 = f32::from_bits((a_vals[2] as u32) << 16);
                    let a3 = f32::from_bits((a_vals[3] as u32) << 16);
                    let a4 = f32::from_bits((a_vals[4] as u32) << 16);
                    let a5 = f32::from_bits((a_vals[5] as u32) << 16);
                    let a6 = f32::from_bits((a_vals[6] as u32) << 16);
                    let a7 = f32::from_bits((a_vals[7] as u32) << 16);

                    // Column 0 → acc0/acc1 with scale s0
                    {
                        let p = packed4_0[0];
                        let w0 = (p & 0xF) as i8;
                        let w1 = ((p >> 4) & 0xF) as i8;
                        let w2 = ((p >> 8) & 0xF) as i8;
                        let w3 = ((p >> 12) & 0xF) as i8;
                        acc0 += (w0 as f32 * s0 - sz0) * a0;
                        acc1 += (w1 as f32 * s0 - sz0) * a1;
                        acc0 += (w2 as f32 * s0 - sz0) * a2;
                        acc1 += (w3 as f32 * s0 - sz0) * a3;

                        let w4 = ((p >> 16) & 0xF) as i8;
                        let w5 = ((p >> 20) & 0xF) as i8;
                        let w6 = ((p >> 24) & 0xF) as i8;
                        let w7 = ((p >> 28) & 0xF) as i8;

                        acc0 += (w4 as f32 * s0 - sz0) * a4;
                        acc1 += (w5 as f32 * s0 - sz0) * a5;
                        acc0 += (w6 as f32 * s0 - sz0) * a6;
                        acc1 += (w7 as f32 * s0 - sz0) * a7;
                    }

                    // Column 1 → acc2/acc3 with scale s1
                    {
                        let p = packed4_0[1];
                        let w0 = (p & 0xF) as i8;
                        let w1 = ((p >> 4) & 0xF) as i8;
                        let w2 = ((p >> 8) & 0xF) as i8;
                        let w3 = ((p >> 12) & 0xF) as i8;
                        acc2 += (w0 as f32 * s1 - sz1) * a0;
                        acc3 += (w1 as f32 * s1 - sz1) * a1;
                        acc2 += (w2 as f32 * s1 - sz1) * a2;
                        acc3 += (w3 as f32 * s1 - sz1) * a3;

                        let w4 = ((p >> 16) & 0xF) as i8;
                        let w5 = ((p >> 20) & 0xF) as i8;
                        let w6 = ((p >> 24) & 0xF) as i8;
                        let w7 = ((p >> 28) & 0xF) as i8;

                        acc2 += (w4 as f32 * s1 - sz1) * a4;
                        acc3 += (w5 as f32 * s1 - sz1) * a5;
                        acc2 += (w6 as f32 * s1 - sz1) * a6;
                        acc3 += (w7 as f32 * s1 - sz1) * a7;
                    }

                    // Column 2 → acc4/acc5 with scale s2
                    {
                        let p = packed4_0[2];
                        let w0 = (p & 0xF) as i8;
                        let w1 = ((p >> 4) & 0xF) as i8;
                        let w2 = ((p >> 8) & 0xF) as i8;
                        let w3 = ((p >> 12) & 0xF) as i8;
                        acc4 += (w0 as f32 * s2 - sz2) * a0;
                        acc5 += (w1 as f32 * s2 - sz2) * a1;
                        acc4 += (w2 as f32 * s2 - sz2) * a2;
                        acc5 += (w3 as f32 * s2 - sz2) * a3;

                        let w4 = ((p >> 16) & 0xF) as i8;
                        let w5 = ((p >> 20) & 0xF) as i8;
                        let w6 = ((p >> 24) & 0xF) as i8;
                        let w7 = ((p >> 28) & 0xF) as i8;

                        acc4 += (w4 as f32 * s2 - sz2) * a4;
                        acc5 += (w5 as f32 * s2 - sz2) * a5;
                        acc4 += (w6 as f32 * s2 - sz2) * a6;
                        acc5 += (w7 as f32 * s2 - sz2) * a7;
                    }

                    // Column 3 → acc6/acc7 with scale s3
                    {
                        let p = packed4_0[3];
                        let w0 = (p & 0xF) as i8;
                        let w1 = ((p >> 4) & 0xF) as i8;
                        let w2 = ((p >> 8) & 0xF) as i8;
                        let w3 = ((p >> 12) & 0xF) as i8;
                        acc6 += (w0 as f32 * s3 - sz3) * a0;
                        acc7 += (w1 as f32 * s3 - sz3) * a1;
                        acc6 += (w2 as f32 * s3 - sz3) * a2;
                        acc7 += (w3 as f32 * s3 - sz3) * a3;

                        let w4 = ((p >> 16) & 0xF) as i8;
                        let w5 = ((p >> 20) & 0xF) as i8;
                        let w6 = ((p >> 24) & 0xF) as i8;
                        let w7 = ((p >> 28) & 0xF) as i8;

                        acc6 += (w4 as f32 * s3 - sz3) * a4;
                        acc7 += (w5 as f32 * s3 - sz3) * a5;
                        acc6 += (w6 as f32 * s3 - sz3) * a6;
                        acc7 += (w7 as f32 * s3 - sz3) * a7;
                    }
                }

                // Second half of two-u32 stride
                if u32_idx + 1 < gs / 8 {
                    let k1 = k0 + 8;

                    let w_ptr1 = unsafe { weight.as_ptr().add((k1 >> 3) * n_usize + col_base) };
                    let packed4_1: [u32; 4] = unsafe { *(w_ptr1 as *const [u32; 4]) };

                    let a_vals1: [u16; 8] = unsafe { *(input.as_ptr().add(k1) as *const [u16; 8]) };

                    // Convert input to f32 once, shared across all 4 columns at this k position
                    unsafe {
                        let a0 = f32::from_bits((a_vals1[0] as u32) << 16);
                        let a1 = f32::from_bits((a_vals1[1] as u32) << 16);
                        let a2 = f32::from_bits((a_vals1[2] as u32) << 16);
                        let a3 = f32::from_bits((a_vals1[3] as u32) << 16);
                        let a4 = f32::from_bits((a_vals1[4] as u32) << 16);
                        let a5 = f32::from_bits((a_vals1[5] as u32) << 16);
                        let a6 = f32::from_bits((a_vals1[6] as u32) << 16);
                        let a7 = f32::from_bits((a_vals1[7] as u32) << 16);

                        // Column 0 → acc0/acc1
                        {
                            let p = packed4_1[0];
                            let w0 = (p & 0xF) as i8;
                            let w1 = ((p >> 4) & 0xF) as i8;
                            let w2 = ((p >> 8) & 0xF) as i8;
                            let w3 = ((p >> 12) & 0xF) as i8;

                            acc0 += (w0 as f32 * s0 - sz0) * a0;
                            acc1 += (w1 as f32 * s0 - sz0) * a1;
                            acc0 += (w2 as f32 * s0 - sz0) * a2;
                            acc1 += (w3 as f32 * s0 - sz0) * a3;

                            let w4 = ((p >> 16) & 0xF) as i8;
                            let w5 = ((p >> 20) & 0xF) as i8;
                            let w6 = ((p >> 24) & 0xF) as i8;
                            let w7 = ((p >> 28) & 0xF) as i8;

                            acc0 += (w4 as f32 * s0 - sz0) * a4;
                            acc1 += (w5 as f32 * s0 - sz0) * a5;
                            acc0 += (w6 as f32 * s0 - sz0) * a6;
                            acc1 += (w7 as f32 * s0 - sz0) * a7;
                        }

                        // Column 1 → acc2/acc3
                        {
                            let p = packed4_1[1];
                            let w0 = (p & 0xF) as i8;
                            let w1 = ((p >> 4) & 0xF) as i8;
                            let w2 = ((p >> 8) & 0xF) as i8;
                            let w3 = ((p >> 12) & 0xF) as i8;

                            acc2 += (w0 as f32 * s1 - sz1) * a0;
                            acc3 += (w1 as f32 * s1 - sz1) * a1;
                            acc2 += (w2 as f32 * s1 - sz1) * a2;
                            acc3 += (w3 as f32 * s1 - sz1) * a3;

                            let w4 = ((p >> 16) & 0xF) as i8;
                            let w5 = ((p >> 20) & 0xF) as i8;
                            let w6 = ((p >> 24) & 0xF) as i8;
                            let w7 = ((p >> 28) & 0xF) as i8;

                            acc2 += (w4 as f32 * s1 - sz1) * a4;
                            acc3 += (w5 as f32 * s1 - sz1) * a5;
                            acc2 += (w6 as f32 * s1 - sz1) * a6;
                            acc3 += (w7 as f32 * s1 - sz1) * a7;
                        }

                        // Column 2 → acc4/acc5
                        {
                            let p = packed4_1[2];
                            let w0 = (p & 0xF) as i8;
                            let w1 = ((p >> 4) & 0xF) as i8;
                            let w2 = ((p >> 8) & 0xF) as i8;
                            let w3 = ((p >> 12) & 0xF) as i8;

                            acc4 += (w0 as f32 * s2 - sz2) * a0;
                            acc5 += (w1 as f32 * s2 - sz2) * a1;
                            acc4 += (w2 as f32 * s2 - sz2) * a2;
                            acc5 += (w3 as f32 * s2 - sz2) * a3;

                            let w4 = ((p >> 16) & 0xF) as i8;
                            let w5 = ((p >> 20) & 0xF) as i8;
                            let w6 = ((p >> 24) & 0xF) as i8;
                            let w7 = ((p >> 28) & 0xF) as i8;

                            acc4 += (w4 as f32 * s2 - sz2) * a4;
                            acc5 += (w5 as f32 * s2 - sz2) * a5;
                            acc4 += (w6 as f32 * s2 - sz2) * a6;
                            acc5 += (w7 as f32 * s2 - sz2) * a7;
                        }

                        // Column 3 → acc6/acc7
                        {
                            let p = packed4_1[3];
                            let w0 = (p & 0xF) as i8;
                            let w1 = ((p >> 4) & 0xF) as i8;
                            let w2 = ((p >> 8) & 0xF) as i8;
                            let w3 = ((p >> 12) & 0xF) as i8;

                            acc6 += (w0 as f32 * s3 - sz3) * a0;
                            acc7 += (w1 as f32 * s3 - sz3) * a1;
                            acc6 += (w2 as f32 * s3 - sz3) * a2;
                            acc7 += (w3 as f32 * s3 - sz3) * a3;

                            let w4 = ((p >> 16) & 0xF) as i8;
                            let w5 = ((p >> 20) & 0xF) as i8;
                            let w6 = ((p >> 24) & 0xF) as i8;
                            let w7 = ((p >> 28) & 0xF) as i8;

                            acc6 += (w4 as f32 * s3 - sz3) * a4;
                            acc7 += (w5 as f32 * s3 - sz3) * a5;
                            acc6 += (w6 as f32 * s3 - sz3) * a6;
                            acc7 += (w7 as f32 * s3 - sz3) * a7;
                        }
                    }
                }
            }
        }

        // Write partial sums for each of 4 columns
        let c0 = col_base;     if c0 < n_usize { partial_sums[split_idx * n_usize + c0] = acc0 + acc1; }
        let c1 = col_base + 1; if c1 < n_usize { partial_sums[split_idx * n_usize + c1] = acc2 + acc3; }
        let c2 = col_base + 2; if c2 < n_usize { partial_sums[split_idx * n_usize + c2] = acc4 + acc5; }
        let c3 = col_base + 3; if c3 < n_usize { partial_sums[split_idx * n_usize + c3] = acc6 + acc7; }
    }

    /// Reduce K-split partial sums into final bf16 output.
    #[kernel]
    #[launch_bounds(64)]
    pub fn reduce_partial_sums_bf16(
        mut output: DisjointSlice<u16>,    // [N] bf16 output
        partial_sums: &[f32],              // [K_SPLIT, N] f32
        n: u32,
        k_split: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let col = (thread::blockIdx_x() * 64u32 + thread::threadIdx_x()) as usize;
        if col >= n as usize {
            return;
        }

        let mut sum: f32 = 0.0;
        for s in 0..k_split as usize {
            sum += partial_sums[s * n as usize + col];
        }

        unsafe {
            *output.get_unchecked_mut(col) = f32_to_bf16(sum);
        }
    }

    /// Warp-cooperative INT4 GEMV for M=1 decode (AutoRound format).

    /// Each warp (32 lanes) computes one output column. Lanes split the K
    /// dimension across groups and reduce via warp shuffle — no separate
    /// reduction kernel and no partial_sums buffer in global memory.

    /// - Block: (32, WARPS_PER_BLOCK, 1) = 256 threads
    /// - Grid: (ceil(N / WARPS_PER_BLOCK), 1, 1)
    /// - Lane L handles groups L, L+32, L+64, ...
    /// - Warp shuffle reduces 32 partial sums → lane 0 writes output[col]
    #[kernel]
    #[launch_bounds(256)]
    pub fn int4_gemm_warp(
        mut output: DisjointSlice<u16>,
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        n: u32, k: u32,
        group_size: u32,
        transposed: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;
        use cuda_device::warp;

        const WARPS_PER_BLOCK: u32 = 8;

        let lane = thread::threadIdx_x() as usize;  // 0..31
        let warp_id = thread::threadIdx_y() as usize;  // 0..7
        let col = (thread::blockIdx_x() * WARPS_PER_BLOCK + warp_id as u32) as usize;

        let n_usize = n as usize;
        let k_usize = k as usize;
        let gs = group_size as usize;
        let num_groups = k_usize / gs;

        if col >= n_usize {
            return;
        }

        let mut acc: f32 = 0.0;

        // Each lane handles groups: lane, lane+32, lane+64, ...
        for g in (lane..num_groups).step_by(32) {
            let kg = g * gs;

            // Load scale for this group
            let scale_bits: u16;
            if transposed != 0 {
                scale_bits = scales[g * n_usize + col];
            } else {
                scale_bits = scales[col * num_groups + g];
            }
            let scale = f16_to_f32(scale_bits);

            // Load zero for this group
            let (zi, zsh): (usize, usize);
            if transposed != 0 {
                let n_packed = (n_usize + 7) / 8;
                zi = g * n_packed + col / 8;
                zsh = (col % 8) * 4;
            } else {
                let flat_idx = col * num_groups + g;
                zi = flat_idx / 8;
                zsh = (flat_idx % 8) * 4;
            }
            let raw_zero = ((zeros[zi] >> zsh) & 0xF) as i8;

            // Process group_size values in chunks of 8 (1 u32 load each)
            for kk in (0..gs).step_by(8) {
                let k_pos = kg + kk;
                let widx: usize;
                if transposed != 0 {
                    widx = (k_pos >> 3) * n_usize + col;
                } else {
                    widx = (col * k_usize + k_pos) / 8;
                }
                let packed = weight[widx];

                for w in 0..8usize {
                    let shift = w * 4;
                    let w_int4 = ((packed >> shift) & 0xF) as i8;
                    // AutoRound: zero = raw_zero + 1
                    let w_fp32 = (w_int4 as f32 - (raw_zero as f32 + 1.0)) * scale;
                    let a_val = f32::from_bits((input[k_pos + w] as u32) << 16);
                    acc += w_fp32 * a_val;
                }
            }
        }

        // Warp reduction: sum all 32 lanes' partial sums via shuffle_xor
        acc = acc + warp::shuffle_xor_f32(acc, 16);
        acc = acc + warp::shuffle_xor_f32(acc, 8);
        acc = acc + warp::shuffle_xor_f32(acc, 4);
        acc = acc + warp::shuffle_xor_f32(acc, 2);
        acc = acc + warp::shuffle_xor_f32(acc, 1);

        // Lane 0 writes the result (all lanes have it after shuffle)
        if lane == 0 {
            unsafe {
                *output.get_unchecked_mut(col) = f32_to_bf16(acc);
            }
        }
    }

    /// Warp-cooperative INT4 GEMV with K-splitting for M=1 decode (AutoRound format).

    /// Combines warp shuffle reduction (no block-level reduction) with K-splitting
    /// (for SM occupancy when K is large). Each warp computes one output column,
    /// and 32 lanes within the warp split the K dimension across groups.
    /// Warp shuffle reduces within each warp → lane 0 writes to partial_sums.

    /// - Block: (32, WARPS_PER_BLOCK, 1) = 256 threads
    /// - Grid: (ceil(N / WARPS_PER_BLOCK), K_SPLIT, 1)
    /// - blockIdx.x: output column tile
    /// - blockIdx.y: K-split index (0..K_SPLIT)
    /// - Each lane handles groups: group_start + lane, group_start + lane + 32, ...
    #[kernel]
    #[launch_bounds(256)]
    pub fn int4_gemm_warp_split(
        partial_sums: &mut [f32],          // [K_SPLIT, N] f32 output
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        n: u32, k: u32,
        group_size: u32,
        transposed: u32,
        k_split: u32,
    ) {
        use cuda_device::warp;

        const WARPS_PER_BLOCK: u32 = 8;

        let lane = thread::threadIdx_x() as usize;   // 0..31
        let warp_id = thread::threadIdx_y() as usize; // 0..7
        let col = (thread::blockIdx_x() * WARPS_PER_BLOCK + warp_id as u32) as usize;
        let split_idx = thread::blockIdx_y() as usize;

        let n_usize = n as usize;
        let k_usize = k as usize;
        let gs = group_size as usize;
        let num_groups_total = k_usize / gs;

        if col >= n_usize {
            return;
        }

        // K range for this split, aligned to group boundaries
        let groups_per_split = (num_groups_total + k_split as usize - 1) / k_split as usize;
        let group_start = split_idx * groups_per_split;
        let group_end = ((split_idx + 1) * groups_per_split).min(num_groups_total);

        let mut acc: f32 = 0.0;

        // Each lane handles groups: group_start + lane, group_start + lane + 32, ...
        let mut g = group_start + lane;
        while g < group_end {
            let kg = g * gs;

            // Load scale for this group
            let scale_bits: u16;
            if transposed != 0 {
                scale_bits = scales[g * n_usize + col];
            } else {
                scale_bits = scales[col * num_groups_total + g];
            }
            let scale = f16_to_f32(scale_bits);

            // Load zero for this group
            let (zi, zsh): (usize, usize);
            if transposed != 0 {
                let n_packed = (n_usize + 7) / 8;
                zi = g * n_packed + col / 8;
                zsh = (col % 8) * 4;
            } else {
                let flat_idx = col * num_groups_total + g;
                zi = flat_idx / 8;
                zsh = (flat_idx % 8) * 4;
            }
            let raw_zero = ((zeros[zi] >> zsh) & 0xF) as i8;

            // Process group_size values in chunks of 8 (1 u32 load each)
            for kk in (0..gs).step_by(8) {
                let k_pos = kg + kk;
                let widx: usize;
                if transposed != 0 {
                    widx = (k_pos >> 3) * n_usize + col;
                } else {
                    widx = (col * k_usize + k_pos) / 8;
                }
                let packed = weight[widx];

                for w in 0..8usize {
                    let shift = w * 4;
                    let w_int4 = ((packed >> shift) & 0xF) as i8;
                    // AutoRound: zero = raw_zero + 1
                    let w_fp32 = (w_int4 as f32 - (raw_zero as f32 + 1.0)) * scale;
                    let a_val = f32::from_bits((input[k_pos + w] as u32) << 16);
                    acc += w_fp32 * a_val;
                }
            }

            g += 32;
        }

        // Warp reduction: sum all 32 lanes' partial sums via shuffle_xor
        acc = acc + warp::shuffle_xor_f32(acc, 16);
        acc = acc + warp::shuffle_xor_f32(acc, 8);
        acc = acc + warp::shuffle_xor_f32(acc, 4);
        acc = acc + warp::shuffle_xor_f32(acc, 2);
        acc = acc + warp::shuffle_xor_f32(acc, 1);

        // Lane 0 writes the result to partial_sums[split_idx * N + col]
        if lane == 0 {
            partial_sums[split_idx * n_usize + col] = acc;
        }
    }

    /// INT4 GEMM kernel for GGUF format (no zero offset).
    #[kernel]
    pub fn int4_gemm_gguf(
        mut output: DisjointSlice<u16>,
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        m: u32, n: u32, k: u32,
        group_size: u32, transposed: u32,
    ) {
        int4_gemm_inner::<Gguf>(
            &mut output, weight, scales, zeros, input,
            m as i32, n as i32, k as i32,
            group_size as i32, transposed as i32,
        );
    }

    /// Dequantize INT4 AutoRound weights to BF16.
    ///
    /// Grid: 1D, one thread per output row (N dimension).
    /// Each thread reads packed INT4 values from one row, unpacks,
    /// applies scale and zero-point, and writes bf16 values.
    ///
    /// weight: [N, K/8] packed INT4 (each u32 holds 8 values)
    /// scales: [N, K/group_size] fp16 group scales
    /// zeros: [N * K/group_size / 8] packed INT4 zeros (each u32 holds 8 zeros)
    /// output: [N, K] bf16 dequantized weights
    #[kernel]
     pub fn int4_dequant_to_bf16(
        mut output: DisjointSlice<u16>,   // [N, K] bf16
        weight: &[u32],                    // [K/8, N] packed INT4 (column-major)
        scales: &[u16],                    // [K/group_size, N] fp16
        zeros: &[u32],                     // [K/group_size, N/8] packed INT4 zeros
        n: u32,
        k: u32,
        group_size: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let row = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if row >= n as usize { return; }

        let n_usize = n as usize;
        let num_groups = (k / group_size) as usize;
        let k_usize = k as usize;
        let group_size_usize = group_size as usize;

        for g in 0..num_groups {
            // Load scale (FP16 → F32)
            let scale_bits = scales[g * n_usize + row];
            let scale = f16_to_f32(scale_bits);

            // Unpack zero point (8 per u32, column-major: [K/group_size, N/8])
            let zero_packed_idx = g * (n_usize / 8) + row / 8;
            let zero_shift = (row % 8) * 4;
            let zero_packed = zeros[zero_packed_idx];
            let raw_zero = ((zero_packed >> zero_shift) & 0xF) as i8;

            // Unpack INT4 values from u32s (8 per u32)
            for i in 0..(group_size_usize / 8) {
                let weight_idx = (g * (group_size_usize / 8) + i) * n_usize + row;
                let packed = weight[weight_idx];

                for w in 0..8u32 {
                    let val = ((packed >> (w * 4)) & 0xF) as i8;
                    // AutoRound: zero = stored_zero + 1; value = (val - zero) * scale
                    let zero = raw_zero + 1;
                    let dequantized = f32::from(val - zero) * scale;
                    let bf16_val = f32_to_bf16(dequantized);
                    let out_idx = row * k_usize + g * group_size_usize + i * 8 + w as usize;
                    unsafe {
                        *output.get_unchecked_mut(out_idx) = bf16_val;
                    }
                }
            }
        }
    }
}
