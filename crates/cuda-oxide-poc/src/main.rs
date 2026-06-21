//! CUDA-oxide proof-of-concept: RMSNorm + reduction kernels with shared memory.
//!
//! Tests static and dynamic shared memory, parallel reduction patterns,
//! and verifies correctness against CPU reference implementations.

// Shared memory is accessed by thread-derived index, not an iterator.
#![allow(clippy::needless_range_loop)]

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
fn reduce_cpu(data: &[f32]) -> f32 { data.iter().sum() }

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

    println!("\n=== All tests complete ===");
    if !all_pass { std::process::exit(1); }
    Ok(())
}
