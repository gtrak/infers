//! CUDA-oxide proof-of-concept: RMSNorm + reduction kernels with shared memory.
//!
//! Tests static and dynamic shared memory, parallel reduction patterns,
//! and verifies correctness against CPU reference implementations.

// Shared memory is accessed by thread-derived index, not an iterator.
#![allow(clippy::needless_range_loop)]
#![feature(f16)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, DynamicSharedArray, SharedArray, cuda_module, kernel, launch_bounds, thread};

// =============================================================================
// GENERIC KERNEL SUPPORT — Trait-based dequant dispatch
// =============================================================================

/// Trait for dequantizing packed weights in registers.
/// Each quant format implements this differently.
trait Dequant {
    /// Dequantize one group of 8 weights.
    /// `packed` is the raw packed u32 from weight memory.
    /// `scale` is the group scale (f32).
    /// `zero` is the zero point offset for this column group (i8).
    /// Returns 8 dequantized f32 values.
    fn dequant_group(packed: u32, scale: f32, zero: i8) -> [f32; 8];
}

/// INT4 dequantization: 8 values per u32, 4 bits each.
struct Int4Dequant;

impl Dequant for Int4Dequant {
    fn dequant_group(packed: u32, scale: f32, zero: i8) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        for w in 0..8 {
            let shift = (w * 4) as u32;
            let w_int4_raw: u32 = (packed >> shift) & 0xF;
            let w_int4: i8 = (w_int4_raw as i8).wrapping_sub(8);
            out[w] = f32::from(w_int4 - zero) * scale;
        }
        out
    }
}

/// INT8 dequantization: 4 values per u32, 8 bits each (pad to 8 for uniform interface).
struct Int8Dequant;

impl Dequant for Int8Dequant {
    fn dequant_group(packed: u32, scale: f32, zero: i8) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        for w in 0..4 {
            let w_int8: i8 = ((packed >> (w * 8)) & 0xFF) as i8;
            out[w] = f32::from(w_int8 - zero) * scale;
        }
        // Upper 4 remain zero for INT8 format
        out
    }
}

// =============================================================================
// KERNELS — compiled to PTX by rustc-codegen-cuda
// =============================================================================

#[cuda_module]
mod all_kernels {
    use super::*;

    /// Vector addition: c[i] = a[i] + b[i]. Grid-stride loop.
    #[kernel]
    pub fn vec_add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>, total_elements: u32) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_elements as usize;
        for i in (tid as usize..total).step_by(stride as usize) {
            unsafe { *out.get_unchecked_mut(i) = a[i] + b[i]; }
        }
    }

    /// Simpler vector addition: one element per thread.
    #[kernel]
    pub fn vec_add_simple(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) { *out_elem = a[i] + b[i]; }
    }

    /// RMSNorm using static shared memory (SharedArray<f32, 256>).
    ///
    /// One block per row, 256 threads per block.
    /// Phase 1: partial sum of squares per thread
    /// Phase 2: parallel reduction via halving stride
    /// Phase 3: apply normalization with weight
    #[kernel]
    #[launch_bounds(256)]
    pub fn rmsnorm_static_smem(x: &[f32], weight: &[f32], mut out: DisjointSlice<f32>, hidden: u32, eps: f32) {
        static mut SDATA: SharedArray<f32, 256> = SharedArray::UNINIT;
        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let total_threads = thread::blockDim_x() as usize;
        let hidden_usize = hidden as usize;
        let row_offset = row * hidden_usize;

        // Phase 1: partial sum of squares
        let mut sum_sq = 0.0f32;
        for i in (tid..hidden_usize).step_by(total_threads) { sum_sq += x[row_offset + i] * x[row_offset + i]; }
        unsafe { SDATA[tid] = sum_sq; }
        thread::sync_threads();

        // Phase 2: tree reduction
        let mut s = total_threads >> 1;
        while s > 0 {
            thread::sync_threads();
            unsafe { if tid < s { SDATA[tid] += SDATA[tid + s]; } }
            s >>= 1;
        }
        thread::sync_threads();

        // Phase 3: normalize
        let rms = unsafe { SDATA[0] };
        let scale = 1.0f32 / (rms / hidden_usize as f32 + eps).sqrt();
        for i in (tid..hidden_usize).step_by(total_threads) {
            unsafe { *out.get_unchecked_mut(row_offset + i) = x[row_offset + i] * scale * (1.0f32 + weight[i]); }
        }
    }

    /// RMSNorm using dynamic shared memory (DynamicSharedArray<f32>).
    ///
    /// Same algorithm as rmsnorm_static_smem but with runtime-sized shared memory.
    #[kernel]
    #[launch_bounds(256)]
    pub fn rmsnorm_dynamic_smem(x: &[f32], weight: &[f32], mut out: DisjointSlice<f32>, hidden: u32, eps: f32) {
        let sdata: *mut f32 = DynamicSharedArray::<f32>::get();
        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let total_threads = thread::blockDim_x() as usize;
        let hidden_usize = hidden as usize;
        let row_offset = row * hidden_usize;

        // Phase 1: partial sum of squares
        let mut sum_sq = 0.0f32;
        for i in (tid..hidden_usize).step_by(total_threads) { sum_sq += x[row_offset + i] * x[row_offset + i]; }
        unsafe { *sdata.add(tid) = sum_sq; }
        thread::sync_threads();

        // Phase 2: tree reduction
        let mut s = total_threads >> 1;
        while s > 0 {
            thread::sync_threads();
            unsafe { if tid < s { *sdata.add(tid) += *sdata.add(tid + s); } }
            s >>= 1;
        }
        thread::sync_threads();

        // Phase 3: normalize
        let rms = unsafe { *sdata.add(0) };
        let scale = 1.0f32 / (rms / hidden_usize as f32 + eps).sqrt();
        for i in (tid..hidden_usize).step_by(total_threads) {
            unsafe { *out.get_unchecked_mut(row_offset + i) = x[row_offset + i] * scale * (1.0f32 + weight[i]); }
        }
    }

    /// Shared memory reduction benchmark: sum of f32 array via tree reduction.
    /// One block, 256 threads.
    #[kernel]
    pub fn reduce_benchmark(data: &[f32], mut out: DisjointSlice<f32>, n: u32) {
        static mut SDATA_REDUCE: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let total_threads = thread::blockDim_x() as usize;
        let n_usize = n as usize;

        // Each thread sums its chunk
        let mut partial = 0.0f32;
        for i in (tid..n_usize).step_by(total_threads) { partial += data[i]; }
        unsafe { SDATA_REDUCE[tid] = partial; }
        thread::sync_threads();

        // Tree reduction
        let mut s = total_threads >> 1;
        while s > 0 {
            thread::sync_threads();
            unsafe { if tid < s { SDATA_REDUCE[tid] += SDATA_REDUCE[tid + s]; } }
            s >>= 1;
        }
        thread::sync_threads();

        // Thread 0 writes result
        unsafe { if tid == 0 { if let Some(o) = out.get_mut(thread::index_1d()) { *o = SDATA_REDUCE[0]; } } }
    }

    /// BF16 vector addition: output[i] = a[i] + b[i] (bf16 input/output, f32 compute).
    ///
    /// Input/output stored as u16 (bf16 bits). Each thread converts bf16→f32,
    /// performs the add in f32, then converts back to bf16.
    #[kernel]
    pub fn bf16_vec_add(a: &[u16], b: &[u16], mut out: DisjointSlice<u16>, total_elements: u32) {
        use cuda_device::tcgen05::f32_to_bf16;
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_elements as usize;
        for i in (tid as usize..total).step_by(stride as usize) {
            // bf16 → f32: reinterpret the 16 bits as upper 16 of f32 mantissa
            let a_f32 = f32::from_bits((a[i] as u32) << 16);
            let b_f32 = f32::from_bits((b[i] as u32) << 16);
            // f32 compute
            let sum = a_f32 + b_f32;
            // f32 → bf16: convert and store as u16
            unsafe { *out.get_unchecked_mut(i) = f32_to_bf16(sum); }
        }
    }

    /// Packed bf16x2 FMA test: c = a * b (using fma with zero accumulator).
    ///
    /// All operands are packed bf16x2 as u32. Low 16 bits = first lane,
    /// high 16 bits = second lane.
    #[kernel]
    pub fn bf16x2_fma_test(a: &[u32], b: &[u32], mut out: DisjointSlice<u32>, total_elements: u32) {
        use cuda_device::bf16x2::fma_bf16x2;
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_elements as usize;
        let zero: u32 = 0;
        for i in (tid as usize..total).step_by(stride as usize) {
            // fma_bf16x2(a, b, 0) = a * b (packed bf16x2 multiplication)
            unsafe { *out.get_unchecked_mut(i) = fma_bf16x2(a[i], b[i], zero); }
        }
    }

    /// INT4 unpack + dequantize test.
    ///
    /// Each u32 contains 8 INT4 values. For each group of 8, we extract the
    /// INT4 value, subtract the zero point, and multiply by the f16 scale.
    /// Output: 8 f32 values per input u32.
    #[kernel]
    pub fn int4_unpack_test(
        packed: &[u32],           // packed INT4 (8 values per u32)
        scales: &[u16],          // f16 scales as u16
        zeros: &[u32],           // packed zero points (INT4, 8 per u32)
        mut out: DisjointSlice<f32>, // dequantized f32 output
        total_packed: u32,       // number of u32 elements in packed input
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_packed as usize;
        for i in (tid as usize..total).step_by(stride as usize) {
            let packed_val = packed[i];
            // Each group of 8 INT4 values shares one scale and one zero point index
            // The zero point is at index i (same as the packed element index)
            let zero_packed = zeros[i / 1]; // one zero per packed u32 for simplicity
            let zero_val_i8: i8 = ((zero_packed & 0xF) as i8).wrapping_sub(8); // sign-extend INT4 to i8, then subtract group zero offset
            let scale_f16_bits = scales[i];
            let scale_f32 = (f16::from_bits(scale_f16_bits)) as f32;
            let mut k = 0;
            while k < 8 {
                let shift = (k * 4) as u32;
                let w_int4_raw: u32 = (packed_val >> shift) & 0xF;
                let w_int4: i8 = (w_int4_raw as i8).wrapping_sub(8); // sign-extend INT4 to signed i8
                let dequantized = f32::from(w_int4 - zero_val_i8) * scale_f32;
                unsafe {
                    *out.get_unchecked_mut(i * 8 + k) = dequantized;
                }
                k += 1;
            }
        }
    }

    /// INT4 GEMM: C = A @ W^T (dequant in registers, bf16 I/O).
    ///
    /// M×K bf16 input A, K×N INT4 packed weight W → M×N bf16 output C.
    /// 16×16 thread blocks, each thread computes one output element.
    #[kernel]
    pub fn int4_gemm(
        a: &[u16],              // bf16 input (M×K), stored as u16
        w_packed: &[u32],       // INT4 weight (K×N), 8 values per u32
        scales: &[u16],         // f16 scales, one per group of N_INT4 columns
        zeros: &[u32],          // packed zero points (INT4)
        mut out: DisjointSlice<u16>, // bf16 output (M×N), stored as u16
        m: u32,                 // rows of A / output
        n: u32,                 // cols of W / output
        k: u32,                 // inner dimension
        group_size: u32,       // number of columns per scale/zero group
    ) {
        use cuda_device::tcgen05::f32_to_bf16;
        let row = thread::blockIdx_y() * 16 + thread::threadIdx_y();
        let col = thread::blockIdx_x() * 16 + thread::threadIdx_x();
        let row_usize = row as usize;
        let col_usize = col as usize;
        let m_usize = m as usize;
        let n_usize = n as usize;
        let k_usize = k as usize;
        let group_size_usize = group_size as usize;
        if row_usize >= m_usize || col_usize >= n_usize {
            return;
        }
        let mut acc = 0.0f32;
        for ki in 0..k_usize {
            // Read a[row][ki] as bf16 → f32
            let a_bf16_bits = a[row_usize * k_usize + ki];
            let a_f32 = f32::from_bits((a_bf16_bits as u32) << 16);
            // Find which packed u32 and shift for w[ki][col]
            // Weight is stored as K×N, where N columns are packed 8 per u32

            let packed_col = (col_usize * k_usize + ki) / 8;
            let shift = ((col_usize * k_usize + ki) % 8) * 4;
            let w_packed_val = w_packed[packed_col];
            let w_int4_raw: u32 = (w_packed_val >> shift) & 0xF;
            let w_int4_i8 = (w_int4_raw as i8).wrapping_sub(8);
            // Find scale/zero group for this column

            let group_idx = col_usize / group_size_usize;
            let scale_f16_bits = scales[group_idx];
            let scale_f32 = (f16::from_bits(scale_f16_bits)) as f32;
            // Zero point: extract from packed zeros at group index

            let zero_raw: u32 = zeros[group_idx] & 0xF;
            let zero_i8 = (zero_raw as i8).wrapping_sub(8);
            // Dequantize and accumulate

            let w_dequantized = f32::from(w_int4_i8 - zero_i8) * scale_f32;
            acc += a_f32 * w_dequantized;
        }
        // Convert result to bf16 and write output

        let out_bf16_bits = f32_to_bf16(acc);
        unsafe {
            *out.get_unchecked_mut(row_usize * n_usize + col_usize) = out_bf16_bits;
        }
    }

    /// GDN recurrent step: single-token decode kernel.
    ///
    /// One thread per (head, v_idx) element. No shared memory.
    /// Uses libm for expf, logf, sqrtf. L2-normalizes query and key,
    /// computes decay/beta from projections, performs 5-step recurrence.
    #[kernel]
    pub fn gdn_recurrent_step(
        query: &[u16],          // [H, K] bf16
        key: &[u16],            // [H, K] bf16
        value: &[u16],          // [H, V] bf16
        a_proj: &[u16],         // [H] bf16
        b_proj: &[u16],         // [H] bf16
        A_log: &[f32],          // [H] f32
        dt_bias: &[f32],        // [H] f32
        state: &mut [f32],      // [H, K, V] f32 — read + write
        mut output: DisjointSlice<u16>, // [H, V] bf16
        num_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;
        let total = (num_heads * head_v_dim) as usize;
        let idx = thread::index_1d();
        let tid = idx.get() as usize;
        if tid >= total { return; }

        let h = tid / head_v_dim as usize;
        let v = tid % head_v_dim as usize;
        let K = head_k_dim as usize;
        let V = head_v_dim as usize;
        let rcp_sqrt_k = 1.0f32 / (K as f32).sqrt();

        // ── Compute g[h] and beta[h] from projections ──
        let decay_rate_h = libm::expf(A_log[h]);
        let a_val = f32::from_bits((a_proj[h] as u32) << 16);
        let sp_val = a_val + dt_bias[h];

        let softplus_val: f32;
        if sp_val > 20.0f32 {
            softplus_val = sp_val;
        } else if sp_val < -20.0f32 {
            softplus_val = 0.0f32;
        } else {
            softplus_val = libm::logf(1.0f32 + libm::expf(sp_val));
        }

        let g_val = -decay_rate_h * softplus_val;
        let decay = libm::expf(g_val);

        // beta[h] = sigmoid(b_proj[h])
        let b_val = f32::from_bits((b_proj[h] as u32) << 16);
        let beta_val = 1.0f32 / (1.0f32 + libm::expf(-b_val));

        // ── L2-normalize key and query ──
        let mut k_l2_sq = 0.0f32;
        let mut q_l2_sq = 0.0f32;
        for k in 0..K {
            let kv = f32::from_bits((key[h * K + k] as u32) << 16);
            let qv = f32::from_bits((query[h * K + k] as u32) << 16);
            k_l2_sq += kv * kv;
            q_l2_sq += qv * qv;
        }

        let eps = 1e-6f32;
        let k_rcp = 1.0f32 / (k_l2_sq + eps).sqrt();
        let q_rcp = 1.0f32 / (q_l2_sq + eps).sqrt();

        // ── State index: state[h*K*V + k*V + v] ──
        let state_base = h * K * V + v;

        // ── Step 1: State decay ──
        for k in 0..K {
            state[state_base + k * V] *= decay;
        }

        // ── Step 2: kv_mem = sum_k S[h][k][v] * key_normed[h][k] ──
        let mut kv_mem = 0.0f32;
        for k in 0..K {
            let s_val = state[state_base + k * V];
            let k_val = f32::from_bits((key[h * K + k] as u32) << 16) * k_rcp;
            kv_mem += s_val * k_val;
        }

        // ── Step 3: delta = beta * (value - kv_mem) ──
        let v_val = f32::from_bits((value[h * V + v] as u32) << 16);
        let delta = beta_val * (v_val - kv_mem);

        // ── Step 4: State update ──
        for k in 0..K {
            let k_val = f32::from_bits((key[h * K + k] as u32) << 16) * k_rcp;
            state[state_base + k * V] += k_val * delta;
        }

        // ── Step 5: Output ──
        let mut y_val = 0.0f32;
        for k in 0..K {
            let s_val = state[state_base + k * V];
            let q_val = f32::from_bits((query[h * K + k] as u32) << 16) * q_rcp * rcp_sqrt_k;
            y_val += s_val * q_val;
        }

        // Write output
        unsafe {
            *output.get_unchecked_mut(h * V + v) = f32_to_bf16(y_val);
        }
    }

    /// GDN Mamba2 SSM single-token update kernel.
    ///
    /// One thread per element. No shared memory.
    /// sigmoid decay, softplus delta, state update, SiLU gating.
    #[kernel]
    pub fn gdn_mamba2_update(
        x_proj: &[u16],          // [num_heads] bf16
        b_proj: &[u16],          // [num_heads] bf16
        dt_proj: &[u16],         // [total_dim] bf16
        z_gate: &[u16],          // [total_dim] bf16
        A_log: &[u16],           // [num_heads] bf16
        dt_bias: &[u16],         // [num_heads] bf16
        state: &mut [u16],       // [total_dim] bf16 — read + write
        mut output: DisjointSlice<u16>, // [total_dim] bf16
        num_heads: u32,
        head_dim: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;
        let total_dim = (num_heads * head_dim) as usize;
        let idx = thread::index_1d();
        let tid = idx.get() as usize;
        if tid >= total_dim { return; }

        let head = tid / head_dim as usize;

        // Pre-compute per-head constants
        let a_val = f32::from_bits((A_log[head] as u32) << 16);
        let decay = 1.0f32 / (1.0f32 + libm::expf(-a_val));
        let bias_val = f32::from_bits((dt_bias[head] as u32) << 16);

        // delta = softplus(dt_proj + dt_bias)
        let dt_val = f32::from_bits((dt_proj[tid] as u32) << 16) + bias_val;
        let delta: f32;
        if dt_val > 2.0f32 {
            delta = dt_val;
        } else if dt_val < -20.0f32 {
            delta = 0.0f32;
        } else {
            delta = libm::logf(1.0f32 + libm::expf(dt_val));
        }

        // b contribution (per-head, broadcast)
        let b_val = f32::from_bits((b_proj[head] as u32) << 16);

        // State update: s = decay * s + delta * b
        let mut s = f32::from_bits((state[tid] as u32) << 16);
        s = decay * s + delta * b_val;

        // Output: state * x_proj * silu(z)
        let x_val = f32::from_bits((x_proj[head] as u32) << 16);
        let z_val = f32::from_bits((z_gate[tid] as u32) << 16);

        // SiLU: numerically stable formulation
        let silu_z: f32;
        if z_val > 0.0f32 {
            silu_z = z_val / (1.0f32 + libm::expf(-z_val));
        } else {
            let exp_z = libm::expf(z_val);
            silu_z = z_val * exp_z / (1.0f32 + exp_z);
        }

        unsafe { *output.get_unchecked_mut(tid) = f32_to_bf16(s * x_val * silu_z); }

        // Store updated state back
        state[tid] = f32_to_bf16(s);
    }

    /// Dynamic shared memory size test.
    ///
    /// Takes the total number of f32 elements and the smem partition sizes,
    /// writes patterns to each partition, syncs, reads back and verifies.
    /// This answers: "Does cuda-oxide support large dynamic shared memory?"
    #[kernel]
    pub fn dynamic_smem_test(
        n_f32: u32,               // total f32 elements allocated
        size_p1: u32,            // partition 1 size in f32 elements
        size_p2: u32,            // partition 2 size (0 if not used)
        mut out: DisjointSlice<u32>, // [2] — [errors, bytes_tested]
    ) {
        let total_threads = thread::blockDim_x() as usize;
        let tid = thread::threadIdx_x() as usize;

        let smem_base: *mut f32 = DynamicSharedArray::<f32>::get();
        let n_f32_usize = n_f32 as usize;
        let size_p1_usize = size_p1 as usize;
        let size_p2_usize = size_p2 as usize;

        // Write pattern: partition 1 uses i*7+3, partition 2 uses i*13+5
        for i in (tid..size_p1_usize).step_by(total_threads) {
            unsafe { *smem_base.add(i) = (i * 7 + 3) as f32; }
        }
        if size_p2_usize > 0 {
            let p2_offset = size_p1_usize;
            for i in (tid..size_p2_usize).step_by(total_threads) {
                unsafe { *smem_base.add(p2_offset + i) = (i * 13 + 5) as f32; }
            }
        }

        thread::sync_threads();

        // Read back and verify
        let mut errors: u32 = 0;

        for i in (tid..size_p1_usize).step_by(total_threads) {
            let expected = (i * 7 + 3) as f32;
            let actual = unsafe { *smem_base.add(i) };
            if actual != expected { errors += 1; }
        }

        if size_p2_usize > 0 {
            let p2_offset = size_p1_usize;
            for i in (tid..size_p2_usize).step_by(total_threads) {
                let expected = (i * 13 + 5) as f32;
                let actual = unsafe { *smem_base.add(p2_offset + i) };
                if actual != expected { errors += 1; }
            }
        }

        thread::sync_threads();

        // Thread 0 writes result
        if tid == 0 {
            unsafe { *out.get_unchecked_mut(0) = errors; }
            unsafe { *out.get_unchecked_mut(1) = n_f32; }
        }
    }

    /// Dynamic shared memory 80KB test (full partitioned layout).
    #[kernel]
    pub fn dynamic_smem_80kb(
        mut out: DisjointSlice<u32>, // [4] — test results
    ) {
        let total_threads = thread::blockDim_x() as usize;
        let tid = thread::threadIdx_x() as usize;

        let smem_base: *mut f32 = DynamicSharedArray::<f32>::get();

        let size_k_normed = 8192;
        let size_k_beta = 4096;
        let size_attn = 6144;
        let size_beta_arr = 2048;

        let offset_k_normed: usize = 0;
        let offset_k_beta: usize = size_k_normed;
        let offset_attn: usize = size_k_normed + size_k_beta;
        let offset_beta_arr: usize = size_k_normed + size_k_beta + size_attn;

        for i in (tid..size_k_normed).step_by(total_threads) {
            unsafe { *smem_base.add(offset_k_normed + i) = (i * 7 + 3) as f32; }
        }
        for i in (tid..size_k_beta).step_by(total_threads) {
            unsafe { *smem_base.add(offset_k_beta + i) = (i * 13 + 5) as f32; }
        }
        for i in (tid..size_attn).step_by(total_threads) {
            unsafe { *smem_base.add(offset_attn + i) = (i * 19 + 7) as f32; }
        }
        for i in (tid..size_beta_arr).step_by(total_threads) {
            unsafe { *smem_base.add(offset_beta_arr + i) = (i * 23 + 11) as f32; }
        }

        thread::sync_threads();

        let mut errors: u32 = 0;

        for i in (tid..size_k_normed).step_by(total_threads) {
            let expected = (i * 7 + 3) as f32;
            let actual = unsafe { *smem_base.add(offset_k_normed + i) };
            if actual != expected { errors += 1; }
        }
        for i in (tid..size_k_beta).step_by(total_threads) {
            let expected = (i * 13 + 5) as f32;
            let actual = unsafe { *smem_base.add(offset_k_beta + i) };
            if actual != expected { errors += 1; }
        }
        for i in (tid..size_attn).step_by(total_threads) {
            let expected = (i * 19 + 7) as f32;
            let actual = unsafe { *smem_base.add(offset_attn + i) };
            if actual != expected { errors += 1; }
        }
        for i in (tid..size_beta_arr).step_by(total_threads) {
            let expected = (i * 23 + 11) as f32;
            let actual = unsafe { *smem_base.add(offset_beta_arr + i) };
            if actual != expected { errors += 1; }
        }

        thread::sync_threads();

        if tid == 0 {
            unsafe { *out.get_unchecked_mut(0) = errors; }
            unsafe { *out.get_unchecked_mut(1) = size_k_normed as u32; }
            unsafe { *out.get_unchecked_mut(2) = size_k_beta as u32; }
            unsafe { *out.get_unchecked_mut(3) = size_attn as u32; }
        }
    }

    /// Generic kernel test: scale operation with Copy+Mul traits (like cross_crate_embedded).
    /// This tests whether generic #[kernel] functions compile and run at all.
    // NOTE: Causes NoModules error at runtime because cuda_module macro switches to
    // load_all_ptx_bundles_merged() for generic kernels, but the codegen backend
    // embeds NVVM IR payloads (not PTX). See finding #2 in experiment results.
    /*
    #[kernel]
    pub fn generic_scale<T: Copy + std::ops::Mul<Output = T>>(
        factor: T,
        input: &[T],
        mut out: DisjointSlice<T>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = input[i] * factor;
        }
    }
    */

    // NOTE: Generic quantized GEMM with trait-based dequant dispatch attempted but E0282 error.
    // Rust cannot infer D because it doesn't appear in any argument type.
    // This is a fundamental limitation of generic kernels without trait-bounded args.
    // See Experiment 1a below for workaround attempts.
    /*
    #[kernel]
    pub fn quant_gemm<D: Dequant>(
        a: &[u16],               // bf16 input (M×K), stored as u16
        w_packed: &[u32],       // packed weights via D
        scales: &[u16],         // f16 scales as u16, one per group of columns
        zeros: &[i8],           // zero points, one per group of columns
        mut out: DisjointSlice<u16>, // bf16 output (M×N), stored as u16
        m: u32,                 // rows of A / output
        n: u32,                 // cols of W / output
        k: u32,                 // inner dimension
        group_size: u32,       // number of columns per scale/zero group
    ) {
        use cuda_device::tcgen05::f32_to_bf16;
        let row = thread::blockIdx_y() * 16 + thread::threadIdx_y();
        let col = thread::blockIdx_x() * 16 + thread::threadIdx_x();
        let row_usize = row as usize;
        let col_usize = col as usize;
        let m_usize = m as usize;
        let n_usize = n as usize;
        let k_usize = k as usize;
        let group_size_usize = group_size as usize;
        if row_usize >= m_usize || col_usize >= n_usize {
            return;
        }
        let mut acc = 0.0f32;
        for ki in 0..k_usize {
            // Read a[row][ki] as bf16 → f32
            let a_bf16_bits = a[row_usize * k_usize + ki];
            let a_f32 = f32::from_bits((a_bf16_bits as u32) << 16);
            // Find which packed u32 and shift for w[ki][col]
            // Weight is stored flat: W[k_i * N + col] → packed at index (k_i * N + col) / 8
            let flat_w_idx = col_usize * k_usize + ki;
            let packed_col = flat_w_idx / 8;
            let w_packed_val = w_packed[packed_col];
            // Get scale and zero for this column group
            let group_idx = col_usize / group_size_usize;
            let scale_f16_bits = scales[group_idx];
            let scale_f32 = (f16::from_bits(scale_f16_bits)) as f32;
            let zero_i8 = zeros[group_idx];
            // Dequantize this weight using the trait-dispatched method
            let dequantized = D::dequant_group(w_packed_val, scale_f32, zero_i8);
            let w_offset = flat_w_idx % 8;
            acc += a_f32 * dequantized[w_offset];
        }
        // Convert result to bf16 and write output
        unsafe {
            *out.get_unchecked_mut(row_usize * n_usize + col_usize) = f32_to_bf16(acc);
        }
    }
    */

    // ========================================================================
    // Experiment 1c: Dispatch-based kernel (workaround for E0282)
    // Instead of generic trait dispatch, use a u32 discriminant to select
    // dequant format at runtime. This avoids both E0282 and NoModules issues.
    // ========================================================================

    /// Quantized GEMM with dispatch-based dequant (no generics, no traits).
    ///
    /// `dequant_kind`: 0 = INT4, 1 = INT8
    #[kernel]
    pub fn quant_gemm_dispatch(
        a: &[u16],                  // bf16 input (M×K), stored as u16
        w_packed: &[u32],           // packed weights
        scales: &[u16],             // f16 scales as u16, one per group of columns
        zeros_i8: &[i8],            // zero points, one per group of columns
        mut out: DisjointSlice<u16>, // bf16 output (M×N), stored as u16
        m: u32,                     // rows of A / output
        n: u32,                     // cols of W / output
        k: u32,                     // inner dimension
        group_size: u32,            // number of columns per scale/zero group
        dequant_kind: u32,          // 0=INT4, 1=INT8
    ) {
        use cuda_device::tcgen05::f32_to_bf16;
        let row = thread::blockIdx_y() * 16 + thread::threadIdx_y();
        let col = thread::blockIdx_x() * 16 + thread::threadIdx_x();
        let row_usize = row as usize;
        let col_usize = col as usize;
        let m_usize = m as usize;
        let n_usize = n as usize;
        let k_usize = k as usize;
        let group_size_usize = group_size as usize;
        if row_usize >= m_usize || col_usize >= n_usize {
            return;
        }
        let mut acc = 0.0f32;

        // Dequant functions (inlined by compiler)
        #[inline(always)]
        fn dequant_int4(packed: u32, scale: f32, zero: i8) -> [f32; 8] {
            let mut out = [0.0f32; 8];
            for w in 0..8 {
                let shift = (w * 4) as u32;
                let raw: u32 = (packed >> shift) & 0xF;
                let val: i8 = (raw as i8).wrapping_sub(8);
                out[w] = f32::from(val - zero) * scale;
            }
            out
        }

        #[inline(always)]
        fn dequant_int8(packed: u32, scale: f32, zero: i8) -> [f32; 8] {
            let mut out = [0.0f32; 8];
            for w in 0..4 {
                let val: i8 = ((packed >> (w * 8)) & 0xFF) as i8;
                out[w] = f32::from(val - zero) * scale;
            }
            out
        }

        for ki in 0..k_usize {
            let a_bf16_bits = a[row_usize * k_usize + ki];
            let a_f32 = f32::from_bits((a_bf16_bits as u32) << 16);

            let flat_w_idx = col_usize * k_usize + ki;
            let packed_col = flat_w_idx / 8;
            let w_packed_val = w_packed[packed_col];

            let group_idx = col_usize / group_size_usize;
            let scale_f16_bits = scales[group_idx];
            let scale_f32 = (f16::from_bits(scale_f16_bits)) as f32;
            let zero_i8 = zeros_i8[group_idx];

            // Dispatch based on dequant_kind
            let dequantized = if dequant_kind == 0 {
                dequant_int4(w_packed_val, scale_f32, zero_i8)
            } else {
                dequant_int8(w_packed_val, scale_f32, zero_i8)
            };

            let w_offset = flat_w_idx % 8;
            acc += a_f32 * dequantized[w_offset];
        }

        unsafe {
            *out.get_unchecked_mut(row_usize * n_usize + col_usize) = f32_to_bf16(acc);
        }
    }

    // ========================================================================
    // Experiment 2: Const generics in #[kernel]
    // Tests whether const generics compile inside device kernels.
    // ========================================================================

    // Simple kernel with const generic to verify const generics support.
    // Note: only integers, bool, and char are allowed as const generic types.
    // NOTE: Causes "named symbol not found" at runtime - cuda-oxide doesn't
    // properly generate/find monomorphized symbols for const generic kernels.
    /*
    #[kernel]
    pub fn const_generic_test<const MULTIPLIER: i32>(
        input: &[f32],
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = input[i] * (MULTIPLIER as f32);
        }
    }
    */


}

// =============================================================================
// HOST CODE — compiled to native x86_64 by LLVM
// =============================================================================

/// CPU reference for RMSNorm.

fn rmsnorm_cpu(x: &[f32], weight: &[f32], hidden: usize, eps: f32) -> Vec<f32> {
    let mut sum_sq = 0.0f32;
    for i in 0..hidden { sum_sq += x[i] * x[i]; }
    let rms = (sum_sq / hidden as f32 + eps).sqrt();
    let scale = 1.0f32 / rms;
    let mut out = Vec::with_capacity(hidden);
    for i in 0..hidden { out.push(x[i] * scale * (1.0f32 + weight[i])); }
    out
}

/// CPU reference for reduction (sum).
/// CPU reference for reduction (sum).
fn reduce_cpu(data: &[f32]) -> f32 { data.iter().sum() }

/// Convert f32 to bf16 bits (truncate, matching f32_to_bf16 behavior).
fn f32_to_bf16_cpu(val: f32) -> u16 {
    ((val.to_bits() >> 16) & 0xFFFF) as u16
}

/// Convert bf16 bits to f32.
fn bf16_to_f32_cpu(bf16_bits: u16) -> f32 {
    f32::from_bits((bf16_bits as u32) << 16)
}

/// CPU reference for bf16_vec_add (bit-exact).
fn bf16_vec_add_cpu(a: &[u16], b: &[u16]) -> Vec<u16> {
    a.iter().zip(b.iter()).map(|(&a, &b)| f32_to_bf16_cpu(bf16_to_f32_cpu(a) + bf16_to_f32_cpu(b))).collect()
}

/// CPU reference for packed bf16x2 multiply (using RNE to match fma.rn.bf16x2).
fn bf16x2_mul_cpu(a: u32, b: u32) -> u32 {
    let a_lo = (a & 0xFFFF) as u16;
    let a_hi = ((a >> 16) & 0xFFFF) as u16;
    let b_lo = (b & 0xFFFF) as u16;
    let b_hi = ((b >> 16) & 0xFFFF) as u16;
    let lo_result = f32_to_bf16_rne_cpu(bf16_to_f32_cpu(a_lo) * bf16_to_f32_cpu(b_lo));
    let hi_result = f32_to_bf16_rne_cpu(bf16_to_f32_cpu(a_hi) * bf16_to_f32_cpu(b_hi));
    (lo_result as u32) | ((hi_result as u32) << 16)
}

/// Convert f32 to bf16 with round-to-nearest-even (matching PTX rn mode).
fn f32_to_bf16_rne_cpu(val: f32) -> u16 {
    let bits = val.to_bits();
    let truncated = (bits >> 16) as u16;
    let low_bits = bits & 0xFFFF;
    if low_bits == 0 {
        return truncated;
    }
    // Halfway case: round to even
    if low_bits == 0x8000 {
        if (truncated & 1) != 0 {
            return truncated + 1; // odd → round up to make even
        }
        return truncated; // already even, truncate
    }
    // More than halfway: round up
    if low_bits > 0x8000 {
        return truncated + 1;
    }
    // Less than halfway: truncate
    truncated
}

/// Sign-extend a 4-bit value to i8.
fn int4_sign_extend(val: u32) -> i8 {
    // INT4 is unsigned 0..15, subtract 8 to get signed -8..7
    (val as i8).wrapping_sub(8)
}

/// CPU reference for int4_unpack_test.
fn int4_unpack_cpu(
    packed: &[u32],
    scales: &[u16],
    zeros: &[u32],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() * 8);
    for (i, &packed_val) in packed.iter().enumerate() {
        let zero_raw = zeros[i] & 0xF;
        let zero_i8 = int4_sign_extend(zero_raw);
        let scale_f32 = (f16::from_bits(scales[i])) as f32;
        for k in 0..8 {
            let shift = (k * 4) as u32;
            let w_int4_raw: u32 = (packed_val >> shift) & 0xF;
            let w_int4 = int4_sign_extend(w_int4_raw);
            out.push(f32::from(w_int4 - zero_i8) * scale_f32);
        }
    }
    out
}

/// CPU reference for int4_gemm (naive, f32 compute).
fn int4_gemm_cpu(
    a: &[u16],      // bf16 input (M×K)
    w_packed: &[u32], // INT4 weight (K×N), 8 per u32
    scales: &[u16],   // f16 scales
    zeros: &[u32],    // zero points
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                // Read a[row][ki]
                let a_f32 = bf16_to_f32_cpu(a[row * k + ki]);
                // Read w[ki][col] — flat index is col * k + ki
                let flat_w_idx = col * k + ki;
                let packed_col = flat_w_idx / 8;
                let shift = (flat_w_idx % 8) * 4;
                let w_int4_raw: u32 = (w_packed[packed_col] >> shift) & 0xF;
                let w_int4_i8 = int4_sign_extend(w_int4_raw);
                // Scale/zero group for this column
                let group_idx = col / group_size;
                let scale_f32 = (f16::from_bits(scales[group_idx])) as f32;
                let zero_raw = zeros[group_idx] & 0xF;
                let zero_i8 = int4_sign_extend(zero_raw);
                let w_dequantized = f32::from(w_int4_i8 - zero_i8) * scale_f32;
                acc += a_f32 * w_dequantized;
            }
            out[row * n + col] = f32_to_bf16_cpu(acc);
        }
    }
    out
}

/// CPU reference for GDN recurrent step.
fn gdn_recurrent_step_cpu(
    query: &[u16],
    key: &[u16],
    value: &[u16],
    a_proj: &[u16],
    b_proj: &[u16],
    A_log: &[f32],
    dt_bias: &[f32],
    state: &mut [f32],
    num_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) -> Vec<u16> {
    let mut output = vec![0u16; num_heads * head_v_dim];
    let total = num_heads * head_v_dim;

    for idx in 0..total {
        let h = idx / head_v_dim;
        let v = idx % head_v_dim;
        let K = head_k_dim;
        let V = head_v_dim;
        let rcp_sqrt_k = 1.0f32 / (K as f32).sqrt();

        // Compute g[h] and beta[h]
        let decay_rate_h = A_log[h].exp();
        let a_val = bf16_to_f32_cpu(a_proj[h]);
        let sp_val = a_val + dt_bias[h];

        let softplus_val: f32;
        if sp_val > 20.0f32 {
            softplus_val = sp_val;
        } else if sp_val < -20.0f32 {
            softplus_val = 0.0f32;
        } else {
            softplus_val = (1.0f32 + sp_val.exp()).ln();
        }

        let g_val = -decay_rate_h * softplus_val;
        let decay = g_val.exp();

        // beta[h] = sigmoid(b_proj[h])
        let b_val = bf16_to_f32_cpu(b_proj[h]);
        let beta_val = 1.0f32 / (1.0f32 + (-b_val).exp());

        // L2-normalize key and query
        let mut k_l2_sq = 0.0f32;
        let mut q_l2_sq = 0.0f32;
        for k in 0..K {
            let kv = bf16_to_f32_cpu(key[h * K + k]);
            let qv = bf16_to_f32_cpu(query[h * K + k]);
            k_l2_sq += kv * kv;
            q_l2_sq += qv * qv;
        }

        let eps = 1e-6f32;
        let k_rcp = 1.0f32 / (k_l2_sq + eps).sqrt();
        let q_rcp = 1.0f32 / (q_l2_sq + eps).sqrt();

        let state_base = h * K * V + v;

        // Step 1: State decay
        for k in 0..K {
            state[state_base + k * V] *= decay;
        }

        // Step 2: kv_mem
        let mut kv_mem = 0.0f32;
        for k in 0..K {
            let s_val = state[state_base + k * V];
            let k_val = bf16_to_f32_cpu(key[h * K + k]) * k_rcp;
            kv_mem += s_val * k_val;
        }

        // Step 3: delta
        let v_val = bf16_to_f32_cpu(value[h * V + v]);
        let delta = beta_val * (v_val - kv_mem);

        // Step 4: State update
        for k in 0..K {
            let k_val = bf16_to_f32_cpu(key[h * K + k]) * k_rcp;
            state[state_base + k * V] += k_val * delta;
        }

        // Step 5: Output
        let mut y_val = 0.0f32;
        for k in 0..K {
            let s_val = state[state_base + k * V];
            let q_val = bf16_to_f32_cpu(query[h * K + k]) * q_rcp * rcp_sqrt_k;
            y_val += s_val * q_val;
        }

        output[h * V + v] = f32_to_bf16_cpu(y_val);
    }

    output
}

/// CPU reference for GDN Mamba2 update.
fn gdn_mamba2_update_cpu(
    x_proj: &[u16],
    b_proj: &[u16],
    dt_proj: &[u16],
    z_gate: &[u16],
    A_log: &[u16],
    dt_bias: &[u16],
    state: &mut [u16],
    num_heads: usize,
    head_dim: usize,
) -> Vec<u16> {
    let total_dim = num_heads * head_dim;
    let mut output = vec![0u16; total_dim];

    for idx in 0..total_dim {
        let head = idx / head_dim;

        // Pre-compute per-head constants
        let a_val = bf16_to_f32_cpu(A_log[head]);
        let decay = 1.0f32 / (1.0f32 + (-a_val).exp());
        let bias_val = bf16_to_f32_cpu(dt_bias[head]);

        // delta = softplus(dt_proj + dt_bias)
        let dt_val = bf16_to_f32_cpu(dt_proj[idx]) + bias_val;
        let delta: f32;
        if dt_val > 2.0f32 {
            delta = dt_val;
        } else if dt_val < -20.0f32 {
            delta = 0.0f32;
        } else {
            delta = (1.0f32 + dt_val.exp()).ln();
        }

        // b contribution
        let b_val = bf16_to_f32_cpu(b_proj[head]);

        // State update: s = decay * s + delta * b
        let mut s = bf16_to_f32_cpu(state[idx]);
        s = decay * s + delta * b_val;

        // Output: state * x_proj * silu(z)
        let x_val = bf16_to_f32_cpu(x_proj[head]);
        let z_val = bf16_to_f32_cpu(z_gate[idx]);

        // SiLU: numerically stable formulation
        let silu_z: f32;
        if z_val > 0.0f32 {
            silu_z = z_val / (1.0f32 + (-z_val).exp());
        } else {
            let exp_z = z_val.exp();
            silu_z = z_val * exp_z / (1.0f32 + exp_z);
        }

        output[idx] = f32_to_bf16_cpu(s * x_val * silu_z);

        // Store updated state back
        state[idx] = f32_to_bf16_cpu(s);
    }

    output
}


fn main() -> Result<(), Box<dyn std::error::Error>> {

    println!("=== cuda-oxide POC: RMSNorm + Reduction ===\n");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = all_kernels::load(&ctx)?;

    let mut all_pass = true;

    // Test 1: vec_add_simple (regression)
    {
        const N: usize = 1024;
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| i as f32 * 2.0).collect();
        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        module.vec_add_simple(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut c_dev)?;
        let c_host = c_dev.to_host_vec(&stream)?;
        if (0..N).filter(|&i| (c_host[i] - a_host[i] - b_host[i]).abs() > 1e-5).count() == 0 {
            println!("[PASS] vec_add_simple: {} elements correct", N);
        } else { eprintln!("[FAIL] vec_add_simple"); all_pass = false; }
    }

    // Test 2: rmsnorm_static_smem (SharedArray<f32, 256>)
    {
        const NUM_ROWS: usize = 4;
        const HIDDEN: usize = 256;
        let eps = 1e-5f32;
        let x_host: Vec<f32> = (0..NUM_ROWS * HIDDEN).map(|i| ((i % 17) as f32 + 0.5) * 3.0).collect();
        let weight_host: Vec<f32> = (0..HIDDEN).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 2.0).collect();
        let x_dev = DeviceBuffer::from_host(&stream, &x_host)?;
        let weight_dev = DeviceBuffer::from_host(&stream, &weight_host)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, NUM_ROWS * HIDDEN)?;
        module.rmsnorm_static_smem(&stream, LaunchConfig { grid_dim: (NUM_ROWS as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, &x_dev, &weight_dev, &mut out_dev, HIDDEN as u32, eps)?;
        let out_host = out_dev.to_host_vec(&stream)?;
        let mut errors = 0usize;
        for row in 0..NUM_ROWS {
            let off = row * HIDDEN;
            let expected = rmsnorm_cpu(&x_host[off..off + HIDDEN], &weight_host, HIDDEN, eps);
            for i in 0..HIDDEN { if (out_host[off + i] - expected[i]).abs() > 1e-3 { errors += 1; } }
        }
        if errors == 0 { println!("[PASS] rmsnorm_static_smem: {}x{} correct", NUM_ROWS, HIDDEN); }
        else { eprintln!("[FAIL] rmsnorm_static_smem: {} errors", errors); all_pass = false; }
    }

    // Test 3: rmsnorm_dynamic_smem (DynamicSharedArray<f32>)
    {
        const NUM_ROWS: usize = 4;
        const HIDDEN: usize = 256;
        let eps = 1e-5f32;
        let x_host: Vec<f32> = (0..NUM_ROWS * HIDDEN).map(|i| ((i % 17) as f32 + 0.5) * 3.0).collect();
        let weight_host: Vec<f32> = (0..HIDDEN).map(|i| ((i % 13) as f32 / 13.0 - 0.5) * 2.0).collect();
        let x_dev = DeviceBuffer::from_host(&stream, &x_host)?;
        let weight_dev = DeviceBuffer::from_host(&stream, &weight_host)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, NUM_ROWS * HIDDEN)?;
        module.rmsnorm_dynamic_smem(&stream, LaunchConfig { grid_dim: (NUM_ROWS as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (256 * std::mem::size_of::<f32>()) as u32 }, &x_dev, &weight_dev, &mut out_dev, HIDDEN as u32, eps)?;
        let out_host = out_dev.to_host_vec(&stream)?;
        let mut errors = 0usize;
        for row in 0..NUM_ROWS {
            let off = row * HIDDEN;
            let expected = rmsnorm_cpu(&x_host[off..off + HIDDEN], &weight_host, HIDDEN, eps);
            for i in 0..HIDDEN { if (out_host[off + i] - expected[i]).abs() > 1e-3 { errors += 1; } }
        }
        if errors == 0 { println!("[PASS] rmsnorm_dynamic_smem: {}x{} correct", NUM_ROWS, HIDDEN); }
        else { eprintln!("[FAIL] rmsnorm_dynamic_smem: {} errors", errors); all_pass = false; }
    }

    // Test 4: reduce_benchmark (sum via shared memory tree reduction)
    {
        const N: usize = 1024;
        let data_host: Vec<f32> = (0..N).map(|i| (i % 97) as f32 + 0.5).collect();
        let expected = reduce_cpu(&data_host);
        let data_dev = DeviceBuffer::from_host(&stream, &data_host)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
        module.reduce_benchmark(&stream, LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, &data_dev, &mut out_dev, N as u32)?;
        let result = out_dev.to_host_vec(&stream)?[0];
        println!("  CPU sum = {:.6}, GPU sum = {:.6} (diff={:.6})", expected, result, (result - expected).abs());
        if (result - expected).abs() < 1.0f32 { println!("[PASS] reduce_benchmark"); }
        else { eprintln!("[FAIL] reduce_benchmark"); all_pass = false; }
    }

    // Test 5: vec_add grid-stride (regression)
    {
        const N: usize = 1024;
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| i as f32 * 2.0).collect();
        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        module.vec_add(&stream, LaunchConfig::for_num_elems(256), &a_dev, &b_dev, &mut c_dev, N as u32)?;
        let c_host = c_dev.to_host_vec(&stream)?;
        if (0..N).filter(|&i| (c_host[i] - a_host[i] - b_host[i]).abs() > 1e-5).count() == 0 {
            println!("[PASS] vec_add grid-stride: {} elements correct", N);
        } else { eprintln!("[FAIL] vec_add grid-stride"); all_pass = false; }
    }

    // Test 6: bf16_vec_add (bf16→f32 add→bf16 pipeline)
    {
        const N: usize = 1024;
        let a_f32: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
        let b_f32: Vec<f32> = (0..N).map(|i| (N - i) as f32 * 0.5).collect();
        let a_bf16: Vec<u16> = a_f32.iter().map(|&x| f32_to_bf16_cpu(x)).collect();
        let b_bf16: Vec<u16> = b_f32.iter().map(|&x| f32_to_bf16_cpu(x)).collect();
        let a_dev = DeviceBuffer::from_host(&stream, &a_bf16)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_bf16)?;
        let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, N)?;
        module.bf16_vec_add(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut out_dev, N as u32)?;
        let out_host = out_dev.to_host_vec(&stream)?;
        let expected = bf16_vec_add_cpu(&a_bf16, &b_bf16);
        if (0..N).filter(|&i| out_host[i] != expected[i]).count() == 0 {
            println!("[PASS] bf16_vec_add: {} elements bit-exact", N);
        } else {
            let errors = (0..N).filter(|&i| out_host[i] != expected[i]).count();
            eprintln!("[FAIL] bf16_vec_add: {} bit mismatches out of {}", errors, N);
            all_pass = false;
        }
    }

    // Test 7: bf16x2_fma_test (packed bf16x2 multiply via FMA)
    {
        const N: usize = 256; // number of packed pairs
        let mut a_packed: Vec<u32> = Vec::with_capacity(N);
        let mut b_packed: Vec<u32> = Vec::with_capacity(N);
        for i in 0..N {
            let lo_a = f32_to_bf16_cpu(i as f32 + 1.0);
            let hi_a = f32_to_bf16_cpu((i as f32 + 1.0) * 2.0);
            a_packed.push((lo_a as u32) | ((hi_a as u32) << 16));
            let lo_b = f32_to_bf16_cpu(1.5);
            let hi_b = f32_to_bf16_cpu(0.5);
            b_packed.push((lo_b as u32) | ((hi_b as u32) << 16));
        }
        let a_dev = DeviceBuffer::from_host(&stream, &a_packed)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_packed)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.bf16x2_fma_test(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut out_dev, N as u32)?;
        let out_host = out_dev.to_host_vec(&stream)?;
        let mut errors = 0usize;
        for i in 0..N {
            let expected = bf16x2_mul_cpu(a_packed[i], b_packed[i]);
            if out_host[i] != expected { errors += 1; }
        }
        if errors == 0 { println!("[PASS] bf16x2_fma_test: {} packed pairs correct", N); }
        else { eprintln!("[FAIL] bf16x2_fma_test: {} errors out of {}", errors, N); all_pass = false; }
    }

    // Test 8: int4_unpack_test (INT4 unpack + dequantize)
    {
        const N_PACKED: usize = 64; // number of u32 elements in packed input
        let mut rng_seed: u32 = 42;
        let mut next_u32 = || { rng_seed = rng_seed.wrapping_mul(1664525u32).wrapping_add(1013904223); rng_seed };
        let packed: Vec<u32> = (0..N_PACKED).map(|_| next_u32()).collect();
        // f16 scales as u16 — use values that fit in f16 range
        let scales: Vec<u16> = (0..N_PACKED).map(|i| {
            let val = ((i % 5) as f32 + 1.0) / 4.0; // 0.25, 0.5, 0.75, 1.0, 1.25
            (val as f16).to_bits()
        }).collect();
        let zeros: Vec<u32> = (0..N_PACKED).map(|_| next_u32()).collect();
        let out_dev = DeviceBuffer::from_host(&stream, &packed)?;
        let scales_dev = DeviceBuffer::from_host(&stream, &scales)?;
        let zeros_dev = DeviceBuffer::from_host(&stream, &zeros)?;
        let mut unpacked_dev = DeviceBuffer::<f32>::zeroed(&stream, N_PACKED * 8)?;
        module.int4_unpack_test(&stream, LaunchConfig::for_num_elems(N_PACKED as u32), &out_dev, &scales_dev, &zeros_dev, &mut unpacked_dev, N_PACKED as u32)?;
        let unpacked_host = unpacked_dev.to_host_vec(&stream)?;
        let expected = int4_unpack_cpu(&packed, &scales, &zeros);
        if (0..N_PACKED * 8).filter(|&i| unpacked_host[i] != expected[i]).count() == 0 {
            println!("[PASS] int4_unpack_test: {} values correct", N_PACKED * 8);
        } else {
            let errors = (0..N_PACKED * 8).filter(|&i| unpacked_host[i] != expected[i]).count();
            eprintln!("[FAIL] int4_unpack_test: {} mismatches out of {}", errors, N_PACKED * 8);
            all_pass = false;
        }
    }

    // Test 9: int4_gemm (16×16 × 16×16 INT4 GEMM with bf16 I/O)
    {
        const M: usize = 16;
        const K: usize = 16;
        const N: usize = 16; // output columns (weight width)
        const GROUP_SIZE: usize = 8;
        let mut rng_seed: u32 = 123;
        let mut next_u32 = || { rng_seed = rng_seed.wrapping_mul(1664525u32).wrapping_add(1013904223); rng_seed };
        // bf16 input A (M×K)
        let a_f32: Vec<f32> = (0..M * K).map(|i| ((i % 7) as f32 + 1.0) / 3.0).collect();
        let a_bf16: Vec<u16> = a_f32.iter().map(|&x| f32_to_bf16_cpu(x)).collect();
        // INT4 weight W (K×N), packed 8 per u32, so need (K*N+7)/8 u32s
        let w_total = K * N;
        let w_packed_count = (w_total + 7) / 8;
        let mut w_packed: Vec<u32> = vec![0; w_packed_count];
        for i in 0..w_total {
            let val = next_u32() & 0xF;
            let packed_idx = i / 8;
            let shift = (i % 8) * 4;
            w_packed[packed_idx] |= val << shift;
        }
        // scales: f16, one per group_size columns
        let num_groups = (N + GROUP_SIZE - 1) / GROUP_SIZE;
        let scales: Vec<u16> = (0..num_groups).map(|i| {
            let val = ((i % 3) as f32 + 1.0) / 3.0; // 0.33, 0.67, 1.0
            (val as f16).to_bits()
        }).collect();
        // zeros: INT4 per group
        let zeros: Vec<u32> = (0..num_groups).map(|_| next_u32()).collect();
        // Launch kernel
        let a_dev = DeviceBuffer::from_host(&stream, &a_bf16)?;
        let w_dev = DeviceBuffer::from_host(&stream, &w_packed)?;
        let scales_dev = DeviceBuffer::from_host(&stream, &scales)?;
        let zeros_dev = DeviceBuffer::from_host(&stream, &zeros)?;
        let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, M * N)?;
        module.int4_gemm(&stream,
            LaunchConfig { grid_dim: (1, 1, 1), block_dim: (16, 16, 1), shared_mem_bytes: 0 },
            &a_dev, &w_dev, &scales_dev, &zeros_dev, &mut out_dev,
            M as u32, N as u32, K as u32, GROUP_SIZE as u32)?;
        let out_host = out_dev.to_host_vec(&stream)?;
        let expected = int4_gemm_cpu(&a_bf16, &w_packed, &scales, &zeros, M, N, K, GROUP_SIZE);
        // Check bit-exact (same bf16 truncation path)
        if (0..M * N).filter(|&i| out_host[i] != expected[i]).count() == 0 {
            println!("[PASS] int4_gemm: {}x{} correct (bit-exact)", M, N);
        } else {
            let errors = (0..M * N).filter(|&i| out_host[i] != expected[i]).count();
            eprintln!("[FAIL] int4_gemm: {} mismatches out of {}", errors, M * N);
            if errors <= 10 {
                for i in 0..M * N {
                    if out_host[i] != expected[i] {
                        eprintln!("  [{},{}] got={} (bf16 {:>5}) expected={} (bf16 {:>5})",
                            i / N, i % N,
                            bf16_to_f32_cpu(out_host[i]), out_host[i],
                            bf16_to_f32_cpu(expected[i]), expected[i]);
                    }
                }
            }
            all_pass = false;
        }
    }

    // Test 10: gdn_recurrent_step (GDN recurrent step kernel with libm math)
    {
        const H: usize = 2;
        const K: usize = 4;
        const V: usize = 4;

        let mut rng_seed: u32 = 999;
        let mut next_u32 = || { rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223); rng_seed };

        // bf16 inputs
        let query_bf16: Vec<u16> = (0..H * K).map(|_| f32_to_bf16_cpu(((next_u32() % 17) as f32 + 1.0) / 5.0)).collect();
        let key_bf16: Vec<u16> = (0..H * K).map(|_| f32_to_bf16_cpu(((next_u32() % 19) as f32 + 1.0) / 6.0)).collect();
        let value_bf16: Vec<u16> = (0..H * V).map(|_| f32_to_bf16_cpu(((next_u32() % 13) as f32 + 1.0) / 4.0)).collect();
        let a_proj_bf16: Vec<u16> = (0..H).map(|i| f32_to_bf16_cpu(i as f32 - 0.5)).collect();
        let b_proj_bf16: Vec<u16> = (0..H).map(|i| f32_to_bf16_cpu(i as f32 * 0.5 + 0.3)).collect();
        // f32 constants
        let A_log: Vec<f32> = [-0.5f32, -0.3f32].to_vec();
        let dt_bias: Vec<f32> = [0.1f32, 0.2f32].to_vec();

        // State: f32, initialize with small values
        let mut state_cpu: Vec<f32> = (0..H * K * V).map(|i| ((i % 7) as f32 + 1.0) / 10.0).collect();
        let state_gpu: Vec<f32> = state_cpu.clone();

        // Launch GPU kernel
        let query_dev = DeviceBuffer::from_host(&stream, &query_bf16)?;
        let key_dev = DeviceBuffer::from_host(&stream, &key_bf16)?;
        let value_dev = DeviceBuffer::from_host(&stream, &value_bf16)?;
        let a_proj_dev = DeviceBuffer::from_host(&stream, &a_proj_bf16)?;
        let b_proj_dev = DeviceBuffer::from_host(&stream, &b_proj_bf16)?;
        let A_log_dev = DeviceBuffer::from_host(&stream, &A_log)?;
        let dt_bias_dev = DeviceBuffer::from_host(&stream, &dt_bias)?;
        let mut state_dev = DeviceBuffer::from_host(&stream, &state_gpu)?;
        let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, H * V)?;

        let total_threads = H * V;
        module.gdn_recurrent_step(
            &stream,
            LaunchConfig::for_num_elems(total_threads as u32),
            &query_dev, &key_dev, &value_dev,
            &a_proj_dev, &b_proj_dev,
            &A_log_dev, &dt_bias_dev,
            &mut state_dev,
            &mut out_dev,
            H as u32, K as u32, V as u32,
        )?;

        let gpu_output = out_dev.to_host_vec(&stream)?;

        // CPU reference
        let cpu_output = gdn_recurrent_step_cpu(
            &query_bf16, &key_bf16, &value_bf16,
            &a_proj_bf16, &b_proj_bf16,
            &A_log, &dt_bias,
            &mut state_cpu,
            H, K, V,
        );

        // Compare outputs (bit-exact bf16)
        let mut errors = 0usize;
        for i in 0..H * V {
            if gpu_output[i] != cpu_output[i] {
                errors += 1;
                if errors <= 4 {
                    eprintln!("  output[{}] GPU={} (bf16 {:>8}) CPU={} (bf16 {:>8})",
                        i, bf16_to_f32_cpu(gpu_output[i]), gpu_output[i],
                        bf16_to_f32_cpu(cpu_output[i]), cpu_output[i]);
                }
            }
        }

        if errors == 0 {
            println!("[PASS] gdn_recurrent_step: {}x{}x{} correct (bit-exact)", H, K, V);
        } else {
            eprintln!("[FAIL] gdn_recurrent_step: {} mismatches out of {}", errors, H * V);
            all_pass = false;
        }
    }

    // Test 11: gdn_mamba2_update (Mamba2 SSM single-token with libm math)
    {
        const NUM_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        let total_dim = NUM_HEADS * HEAD_DIM; // 8

        let mut rng_seed: u32 = 777;
        let mut next_u32 = || { rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223); rng_seed };

        // bf16 inputs
        let x_proj_bf16: Vec<u16> = (0..NUM_HEADS).map(|i| f32_to_bf16_cpu(((i % 5) as f32 + 1.0) / 3.0)).collect();
        let b_proj_bf16: Vec<u16> = (0..NUM_HEADS).map(|i| f32_to_bf16_cpu(i as f32 * 0.4 + 0.5)).collect();
        let dt_proj_bf16: Vec<u16> = (0..total_dim).map(|_| f32_to_bf16_cpu(((next_u32() % 7) as f32 - 3.0) / 4.0)).collect();
        let z_gate_bf16: Vec<u16> = (0..total_dim).map(|_| f32_to_bf16_cpu(((next_u32() % 9) as f32 - 4.0) / 5.0)).collect();
        let A_log_bf16: Vec<u16> = (0..NUM_HEADS).map(|i| f32_to_bf16_cpu(-0.2f32 * (i as f32 + 1.0))).collect();
        let dt_bias_bf16: Vec<u16> = (0..NUM_HEADS).map(|_| f32_to_bf16_cpu(0.1f32)).collect();

        // State: bf16, initialize with small values
        let state_cpu: Vec<u16> = (0..total_dim).map(|i| f32_to_bf16_cpu(((i % 5) as f32 + 1.0) / 8.0)).collect();
        let state_gpu: Vec<u16> = state_cpu.clone();

        // Launch GPU kernel
        let x_proj_dev = DeviceBuffer::from_host(&stream, &x_proj_bf16)?;
        let b_proj_dev = DeviceBuffer::from_host(&stream, &b_proj_bf16)?;
        let dt_proj_dev = DeviceBuffer::from_host(&stream, &dt_proj_bf16)?;
        let z_gate_dev = DeviceBuffer::from_host(&stream, &z_gate_bf16)?;
        let A_log_dev = DeviceBuffer::from_host(&stream, &A_log_bf16)?;
        let dt_bias_dev = DeviceBuffer::from_host(&stream, &dt_bias_bf16)?;
        let mut state_dev = DeviceBuffer::from_host(&stream, &state_gpu)?;
        let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, total_dim)?;

        module.gdn_mamba2_update(
            &stream,
            LaunchConfig::for_num_elems(total_dim as u32),
            &x_proj_dev, &b_proj_dev, &dt_proj_dev, &z_gate_dev,
            &A_log_dev, &dt_bias_dev,
            &mut state_dev, &mut out_dev,
            NUM_HEADS as u32, HEAD_DIM as u32,
        )?;

        let gpu_output = out_dev.to_host_vec(&stream)?;

        // CPU reference
        let cpu_output = gdn_mamba2_update_cpu(
            &x_proj_bf16, &b_proj_bf16, &dt_proj_bf16, &z_gate_bf16,
            &A_log_bf16, &dt_bias_bf16,
            &mut state_cpu.clone(),
            NUM_HEADS, HEAD_DIM,
        );

        // Compare outputs (bit-exact bf16)
        let mut errors = 0usize;
        for i in 0..total_dim {
            if gpu_output[i] != cpu_output[i] {
                errors += 1;
                if errors <= 4 {
                    eprintln!("  output[{}] GPU={} (bf16 {:>8}) CPU={} (bf16 {:>8})",
                        i, bf16_to_f32_cpu(gpu_output[i]), gpu_output[i],
                        bf16_to_f32_cpu(cpu_output[i]), cpu_output[i]);
                }
            }
        }

        if errors == 0 {
            println!("[PASS] gdn_mamba2_update: {} heads x {} dim correct (bit-exact)", NUM_HEADS, HEAD_DIM);
        } else {
            eprintln!("[FAIL] gdn_mamba2_update: {} mismatches out of {}", errors, total_dim);
            all_pass = false;
        }
    }

    // Test 12: dynamic_smem_80kb (80KB dynamic shared memory test — baseline)
    {
        // Progressive size test using the dedicated kernel.
        // NOTE: On sm_120 (RTX 5060 Ti), cuda-oxide's default maxSharedMemoryPerBlock
        // is ~48KB. To go above this, you need cudaFuncSetAttribute which cuda-oxide
        // does not currently expose. The original gdn_chunked_gated_delta_prefill.cu
        // works because it calls cudaFuncSetAttribute before launch.
        let sizes_kb: Vec<u32> = vec![1, 4, 16, 32, 48, 56, 64, 72, 80];

        let mut largest_ok_kb: u32 = 0;
        for &size_kb in &sizes_kb {
            let smem_bytes = size_kb * 1024;
            let n_f32 = (smem_bytes / 4) as u32;
            println!("  Testing {} bytes ({:.0} KB)...", smem_bytes, size_kb as f32);

            let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
            match module.dynamic_smem_test(
                &stream,
                LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem_bytes as u32 },
                n_f32, n_f32, 0,
                &mut out_dev,
            ) {
                Ok(()) => {
                    largest_ok_kb = size_kb;
                    let results = out_dev.to_host_vec(&stream)?;
                    if results[0] == 0 {
                        println!("    ✓ OK ({} bytes verified)", smem_bytes);
                    } else {
                        println!("    ✗ DATA ERROR: {} errors", results[0]);
                        break;
                    }
                },
                Err(e) => { println!("    ✗ LAUNCH FAILED: {}", e); break; }
            }
        }

        // Report the finding (not a fail — this is a hardware + cuda-oxide API limitation)
        if largest_ok_kb >= 80 {
            println!("[PASS] dynamic_smem_80kb: up to {} KB works, data verified", largest_ok_kb);
        } else {
            // This is expected on sm_120 without cudaFuncSetAttribute support
            println!("[INFO] dynamic_smem_80kb: max = {} KB (cuda-oxide lacks cudaFuncSetAttribute for >48KB on sm_120)", largest_ok_kb);
        }
    }

    // Test 13: cuFuncSetAttribute workaround — raise max dynamic shared memory above 48KB
    {
        println!("\n=== Test 13: cuFuncSetAttribute for >48KB dynamic smem ===");

        // Access the raw CUDA driver API through cuda-core's sys re-export of cuda-bindings.
        use cuda_core::sys;

        let sizes_kb: Vec<u32> = vec![56, 80, 96];

        for &size_kb in &sizes_kb {
            let smem_bytes = size_kb * 1024usize as u32;
            let n_f32 = (smem_bytes / 4) as u32;
            println!("  Testing {} bytes ({:.0} KB) with cuFuncSetAttribute...", smem_bytes, size_kb as f32);

            // Load the kernel function by name to get a CudaFunction handle.
            let func = module.as_cuda_module().load_function("dynamic_smem_test")?;
            let raw_func: sys::CUfunction = unsafe { func.cu_function() };

            // Set max dynamic shared memory for this function to at least smem_bytes.
            // cuFuncSetAttribute signature: (CUfunction, CUfunction_attribute, int value)
            // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8 (from cuda.h line 1083)
            let result = unsafe {
                sys::cuFuncSetAttribute(
                    raw_func,
                    8, // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                    smem_bytes as i32,
                )
            };

            if result != 0 {
                println!("    ✗ cuFuncSetAttribute failed with error code {}", result);
                all_pass = false;
                break;
            }

            // Launch the kernel with the requested shared memory size.
            let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
            match module.dynamic_smem_test(
                &stream,
                LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem_bytes },
                n_f32, n_f32, 0,
                &mut out_dev,
            ) {
                Ok(()) => {
                    let results = out_dev.to_host_vec(&stream)?;
                    if results[0] == 0 {
                        println!("    ✓ OK ({} bytes verified, {} f32 elements read/written)", smem_bytes, results[1]);
                    } else {
                        println!("    ✗ DATA ERROR: {} errors", results[0]);
                        all_pass = false;
                        break;
                    }
                },
                Err(e) => {
                    println!("    ✗ LAUNCH FAILED: {}", e);
                    all_pass = false;
                    break;
                }
            }
        }
    }

    // ========================================================================
    // Experiment Results: Generic #[kernel] with trait-based Dequant dispatch
    // ========================================================================
    {
        println!("\n=== Experiment 1: Trait-based Dequant dispatch (trait generics) ===");
        // Finding: FAILS at compile time with E0282
        // quant_gemm<D: Dequant> cannot infer D because it doesn't appear in any argument type.
        // Rust requires T to be inferrable from function arguments, but D is only used
        // for trait dispatch (phantom type parameter).
        println!("[FAIL] Generic #[kernel] with Dequant trait: E0282 compile error");
        println!("       Error: 'type annotations needed - cannot infer type of D'");
        println!("       Reason: D is a phantom type param, not inferrable from kernel args");

        // Finding: Even if we could compile it, runtime would fail with NoModules error
        // because #[cuda_module] switches to load_all_ptx_bundles_merged() for generic kernels,
        // but the codegen backend embeds NVVM IR payloads (not PTX).
        println!("       Additional blocker: NoModules at runtime (NVVM IR vs PTX payload mismatch)");
    }

    // ========================================================================
    // Experiment 1c: Dispatch-based kernel workaround
    // ========================================================================
    {
        println!("\n=== Experiment 1c: Dispatch-based quant GEMM (dequant_kind param) ===");

        const M: usize = 16;
        const K: usize = 16;
        const N: usize = 16;
        const GROUP_SIZE: usize = 8;

        let mut rng_seed: u32 = 456;
        let mut next_u32 = || { rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223); rng_seed };

        // bf16 input A (M×K)
        let a_f32: Vec<f32> = (0..M * K).map(|i| ((i % 7) as f32 + 1.0) / 3.0).collect();
        let a_bf16: Vec<u16> = a_f32.iter().map(|&x| f32_to_bf16_cpu(x)).collect();

        // INT4 weight W (K×N), packed 8 per u32
        let w_total = K * N;
        let w_packed_count = (w_total + 7) / 8;
        let mut w_packed: Vec<u32> = vec![0; w_packed_count];
        for i in 0..w_total {
            let val = next_u32() & 0xF;
            let packed_idx = i / 8;
            let shift = (i % 8) * 4;
            w_packed[packed_idx] |= val << shift;
        }

        // scales: f16, one per group_size columns
        let num_groups = (N + GROUP_SIZE - 1) / GROUP_SIZE;
        let scales: Vec<u16> = (0..num_groups).map(|i| {
            let val = ((i % 3) as f32 + 1.0) / 3.0;
            (val as f16).to_bits()
        }).collect();

        // zeros: i8 per group
        let zeros_i8: Vec<i8> = (0..num_groups).map(|_| next_u32() as i8).collect();

        // Launch dispatch kernel with INT4 mode (dequant_kind=0)
        let a_dev = DeviceBuffer::from_host(&stream, &a_bf16)?;
        let w_dev = DeviceBuffer::from_host(&stream, &w_packed)?;
        let scales_dev = DeviceBuffer::from_host(&stream, &scales)?;
        let zeros_dev = DeviceBuffer::from_host(&stream, &zeros_i8)?;
        let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, M * N)?;

        module.quant_gemm_dispatch(
            &stream,
            LaunchConfig { grid_dim: (1, 1, 1), block_dim: (16, 16, 1), shared_mem_bytes: 0 },
            &a_dev, &w_dev, &scales_dev, &zeros_dev, &mut out_dev,
            M as u32, N as u32, K as u32, GROUP_SIZE as u32,
            0, // dequant_kind = INT4
        )?;

        let out_host = out_dev.to_host_vec(&stream)?;

        // CPU reference: manual INT4 dequant (matching dequant_int4 logic)
        let mut expected = vec![0u16; M * N];
        for row in 0..M {
            for col in 0..N {
                let mut acc = 0.0f32;
                for ki in 0..K {
                    let a_f32 = bf16_to_f32_cpu(a_bf16[row * K + ki]);
                    let flat_w_idx = col * K + ki;
                    let packed_col = flat_w_idx / 8;
                    let shift = (flat_w_idx % 8) * 4;
                    let raw: u32 = (w_packed[packed_col] >> shift) & 0xF;
                    let val: i8 = (raw as i8).wrapping_sub(8);
                    let group_idx = col / GROUP_SIZE;
                    let scale_f32 = (f16::from_bits(scales[group_idx])) as f32;
                    let zero_i8 = zeros_i8[group_idx];
                    acc += a_f32 * f32::from(val - zero_i8) * scale_f32;
                }
                expected[row * N + col] = f32_to_bf16_cpu(acc);
            }
        }

        if (0..M * N).filter(|&i| out_host[i] != expected[i]).count() == 0 {
            println!("[PASS] quant_gemm_dispatch INT4: {}x{} correct", M, N);
        } else {
            let errors = (0..M * N).filter(|&i| out_host[i] != expected[i]).count();
            eprintln!("[FAIL] quant_gemm_dispatch INT4: {} mismatches", errors);
            all_pass = false;
        }
    }

    // ========================================================================
    // Experiment 2: Const generics in #[kernel]
    // ========================================================================
    {
        println!("\n=== Experiment 2: Const generics in #[kernel] ===");
        // Finding 1: f32 is NOT allowed as const generic type (Rust compiler error).
        // Only integers, bool, and char are supported. So const generics can only
        // be used for integer-based parameters (group sizes, dimensions, etc.).

        // Finding 2: Even with i32 const generic, cuda-oxide fails at runtime with
        // "named symbol not found" - the codegen doesn't properly generate/find
        // monomorphized symbols for const generic kernels.
        println!("[FAIL] Const generics in #[kernel]: runtime 'named symbol not found' error");
        println!("       Finding 1: f32 forbidden as const generic type (compiler error)");
        println!("       Finding 2: i32 const generics fail at runtime (symbol resolution issue)");
    }

    println!("\n=== All tests complete ===");
    if !all_pass { std::process::exit(1); }

    Ok(())
}
