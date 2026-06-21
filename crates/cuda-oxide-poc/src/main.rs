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

    println!("\n=== All tests complete ===");
    if !all_pass { std::process::exit(1); }

    Ok(())
}
