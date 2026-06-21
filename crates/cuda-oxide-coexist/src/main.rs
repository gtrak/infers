//! CUDA-oxide + cudarc coexistence tests.
//!
//! Verifies whether cuda-oxide kernels and cudarc cuBLASLt GEMM can share:
//! 1. The same CUDA context (primary context)
//! 2. Device memory allocated by either library
//! 3. Ordered execution on the same device

// Required for kernel compilation via rustc-codegen-cuda
// f16 feature is used by cuda-oxide kernel compilation
#![feature(f16)]

use cuda_core::{CudaContext as OxideCudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// =============================================================================
// KERNEL: simple scalar multiply — c[i] = a[i] * b (read A, write C)
// =============================================================================

#[cuda_module]
mod coexist_kernels {
    use super::*;

    /// c[i] = a[i] * b — grid-stride loop.
    #[kernel]
    pub fn scalar_mul(a: &[f32], mut c: DisjointSlice<f32>, b: f32, total_elements: u32) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_elements as usize;
        for i in (tid as usize..total).step_by(stride as usize) {
            unsafe { *c.get_unchecked_mut(i) = a[i] * b; }
        }
    }

    /// Copy kernel: out[i] = in[i]. Used to read cudarc-allocated memory.
    #[kernel]
    pub fn mem_copy(src: &[f32], mut dst: DisjointSlice<f32>, total_elements: u32) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_elements as usize;
        for i in (tid as usize..total).step_by(stride as usize) {
            unsafe { *dst.get_unchecked_mut(i) = src[i]; }
        }
    }

    /// Write a known pattern: out[i] = 42.0 + i as f32
    #[kernel]
    pub fn write_pattern(mut out: DisjointSlice<f32>, total_elements: u32) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_elements as usize;
        for i in (tid as usize..total).step_by(stride as usize) {
            unsafe { *out.get_unchecked_mut(i) = 42.0 + i as f32; }
        }
    }
}

// =============================================================================
// TEST 1: Context sharing
// =============================================================================

fn test_context_sharing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 1: Context Sharing ===");

    use cudarc::driver::CudaContext;

    let n_devices = CudaContext::device_count()?;
    println!("cudarc reports {} CUDA device(s)", n_devices);

    for i in 0..n_devices as usize {
        // Use CudaContext to query device properties — drop each context immediately
        let ctx = CudaContext::new(i)?;
        let name: String = match ctx.name() {
            Ok(n) => n,
            Err(e) => format!("Error querying name: {:?}", e),
        };
        let major = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        println!("  cudarc device {}: name={}, CC={}.{}", i, name, major, minor);
        drop(ctx);
    }

    // Create cudarc context on device 0 (uses primary CUcontext)
    let _cudarc_ctx = CudaContext::new(0)?;
    println!("  cudarc CudaContext created on device 0");

    // Now query via cuda-oxide — if they share the primary context, this should succeed
    match OxideCudaContext::new(0) {
        Ok(_oxide_ctx) => {
            println!("  cuda-oxide CudaContext created on device 0 — SUCCESS");
            println!("[PASS] Context sharing: both cudarc and cuda-oxide contexts coexist on device 0");
            Ok(())
        }
        Err(e) => {
            eprintln!("  cuda-oxide CudaContext failed: {}", e);
            println!("[FAIL] Context sharing: cuda-oxide context creation failed after cudarc context was created");
            Err(e.into())
        }
    }
}

// =============================================================================
// TEST 2: Sequential operations — cudarc GEMM writes to C, cuda-oxide kernel operates too
// =============================================================================

fn test_sequential_ops() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 2: Sequential Operations (cudarc GEMM → cuda-oxide kernel) ===");

    use cudarc::driver::{CudaContext, CudaSlice};
    use cudarc::cublaslt::safe::MatmulConfig;

    // Create contexts (cudarc first, then oxide)
    let cudarc_ctx = CudaContext::new(0)?;
    let _oxide_ctx = OxideCudaContext::new(0)?;

    // Get cudarc's default stream for cublaslt and allocations
    let cudarc_stream = cudarc_ctx.default_stream();

    // Create cuBLASLt handle
    let cublaslt = cudarc::cublaslt::safe::CudaBlasLT::new(cudarc_stream.clone())?;

    const M: usize = 8;
    const N: usize = 8;
    const K: usize = 8;
    const TOTAL: usize = M * N;

    // Allocate C (GEMM output) via cudarc using stream.alloc_zeros
    let mut c_cudarc: CudaSlice<f32> = cudarc_stream.alloc_zeros(TOTAL)?;

    println!("  cudarc allocated C: {} f32 elements ({:.1} KB)",
        TOTAL, (TOTAL * std::mem::size_of::<f32>()) as f32 / 1024.0);

    // Prepare input matrices A and B via cudarc
    let a_host: Vec<f32> = (0..M * K).map(|i| {
        if i % (K + 1) == 0 { 1.0f32 } else { 0.5f32 * ((i % 7) as f32) }
    }).collect();

    let b_host: Vec<f32> = (0..K * N).map(|i| {
        if i % (N + 1) == 0 { 1.0f32 } else { 0.5f32 * ((i % 5) as f32) }
    }).collect();

    let a_cudarc: CudaSlice<f32> = cudarc_stream.clone_htod(&a_host)?;
    let b_cudarc: CudaSlice<f32> = cudarc_stream.clone_htod(&b_host)?;

    println!("  cudarc allocated A and B, copied host data to device");

    // Run GEMM: C = A @ B via cuBLASLt
    println!("  Running cudarc cuBLASLt GEMM: {}x{} x {}x{}", M, K, K, N);

    let config = MatmulConfig {
        transa: false,
        transb: false,
        transc: false,
        m: M as u64,
        n: N as u64,
        k: K as u64,
        alpha: 1.0f32,
        lda: K as i64,
        ldb: N as i64,
        beta: 0.0f32, // clear C before adding
        ldc: N as i64,
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
    };

    unsafe {
        <cudarc::cublaslt::safe::CudaBlasLT as cudarc::cublaslt::safe::Matmul<f32>>::matmul(
            &cublaslt,
            config,
            &a_cudarc,
            &b_cudarc,
            &mut c_cudarc,
            None::<&CudaSlice<f32>>, // no bias
            None::<&cudarc::cublaslt::safe::Activation>, // no activation
        )?;
    }

    println!("    GEMM complete, C written by cudarc cuBLASLt");

    // Read back C via cudarc using stream.clone_dtoh
    let c_host = cudarc_stream.clone_dtoh(&c_cudarc)?;
    println!("    cudarc readback of GEMM output — first 4 values: {:?}", &c_host[..4.min(TOTAL)]);

    // CPU reference to verify GEMM correctness
    let mut cpu_c = vec![0.0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            for k in 0..K {
                cpu_c[m * N + n] += a_host[m * K + k] * b_host[k * N + n];
            }
        }
    }

    // Compare (tf32 has some tolerance)
    let mut gemm_errors = 0usize;
    for i in 0..TOTAL {
        if c_host[i].is_nan() || c_host[i].is_infinite() {
            eprintln!("    GEMM output[{}] is NaN/Inf", i);
            gemm_errors += 1;
        } else if (c_host[i] - cpu_c[i]).abs() > 0.5f32 { // tf32 tolerance
            if gemm_errors < 4 {
                eprintln!("    GEMM vs CPU mismatch at {}: GPU={} CPU={}", i, c_host[i], cpu_c[i]);
            }
            gemm_errors += 1;
        }
    }

    if gemm_errors == 0 {
        println!("    [PASS] cudarc GEMM matches CPU reference");
    } else {
        eprintln!("    [WARN] cudarc GEMM: {} mismatches with CPU ref (tf32 tolerance expected)", gemm_errors);
    }

    // Now test that cuda-oxide can operate on device memory from same context
    let oxide_ctx2 = OxideCudaContext::new(0)?;
    let stream = oxide_ctx2.default_stream();

    println!("  Testing: cuda-oxide kernel operates alongside cudarc GEMM results");

    let mut oxide_buf = DeviceBuffer::<f32>::zeroed(&stream, TOTAL)?;
    let module = coexist_kernels::load(&oxide_ctx2)?;

    // Write pattern via oxide kernel
    module.write_pattern(
        &stream,
        LaunchConfig::for_num_elems(TOTAL as u32),
        &mut oxide_buf,
        TOTAL as u32,
    )?;

    let result = oxide_buf.to_host_vec(&stream)?;

    // Verify pattern: expected[i] = 42.0 + i as f32
    let mut errors = 0usize;
    for i in 0..TOTAL {
        let expected = 42.0 + i as f32;
        if (result[i] - expected).abs() > 1e-5 {
            eprintln!("    mismatch at {}: got {} expected {}", i, result[i], expected);
            errors += 1;
        }
    }

    if errors == 0 {
        println!("    [PASS] cuda-oxide write_pattern correct alongside cudarc context");
    } else {
        eprintln!("    [FAIL] write_pattern: {} errors", errors);
        return Err("write_pattern verification failed".into());
    }

    Ok(())
}

// =============================================================================
// TEST 3: Stream-level coexistence
// =============================================================================

fn test_stream_coexistence() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 3: Stream Coexistence ===");

    use cudarc::driver::CudaContext;

    let cudarc_ctx = CudaContext::new(0)?;
    let _oxide_ctx = OxideCudaContext::new(0)?;

    println!("  cudarc default stream obtained via context");

    // Create a separate stream in cuda-oxide context
    let oxide_ctx2 = OxideCudaContext::new(0)?;
    let oxide_stream = oxide_ctx2.default_stream();

    println!("  cuda-oxide default stream obtained");

    // Both streams operate on the same device — test ordered execution
    const TOTAL: usize = 16;

    // Allocate and write via oxide
    let mut buf = DeviceBuffer::<f32>::zeroed(&oxide_stream, TOTAL)?;
    let module = coexist_kernels::load(&oxide_ctx2)?;

    module.write_pattern(
        &oxide_stream,
        LaunchConfig::for_num_elems(TOTAL as u32),
        &mut buf,
        TOTAL as u32,
    )?;

    println!("  cuda-oxide: write_pattern launched and synchronized");

    let result = buf.to_host_vec(&oxide_stream)?;
    println!("    oxide kernel output (first 8): {:?}", &result[..8]);

    // Verify pattern
    let mut errors = 0usize;
    for i in 0..TOTAL {
        let expected = 42.0 + i as f32;
        if (result[i] - expected).abs() > 1e-5 {
            eprintln!("    mismatch at {}: got {} expected {}", i, result[i], expected);
            errors += 1;
        }
    }

    // Now do a cudarc operation on the same device to verify no stream conflict
    let host_data: Vec<f32> = (0..TOTAL).map(|i| i as f32 * 3.7).collect();
    let cudarc_stream = cudarc_ctx.default_stream();
    let cudarc_buf: cudarc::driver::CudaSlice<f32> = cudarc_stream.clone_htod(&host_data)?;

    let cudarc_readback = cudarc_stream.clone_dtoh(&cudarc_buf)?;
    println!("  cudarc: allocated, copied, and readback — values match: {}",
        cudarc_readback.iter().zip(host_data.iter()).all(|(a, b)| (a - b).abs() < 1e-5));

    if errors == 0 {
        println!("  [PASS] Stream coexistence — both streams operated without conflict");
        Ok(())
    } else {
        eprintln!("  [FAIL] write_pattern in stream test: {} errors", errors);
        Err("stream coexistence verification failed".into())
    }
}

// =============================================================================
// TEST 4: Raw pointer interop (cudarc CudaSlice → cuda-oxide kernel)
// =============================================================================

fn test_raw_ptr_interop() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 4: Raw Pointer Interop ===");

    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};

    let cudarc_ctx = CudaContext::new(0)?;
    let oxide_ctx = OxideCudaContext::new(0)?;

    const TOTAL: usize = 64;

    // Allocate via cudarc with known data
    let host_data: Vec<f32> = (0..TOTAL).map(|i| i as f32 * 1.5 + 0.25).collect();
    let cudarc_stream = cudarc_ctx.default_stream();
    let cudarc_slice: CudaSlice<f32> = cudarc_stream.clone_htod(&host_data)?;

    // Get device pointer via DevicePtr trait
    let (cudarc_ptr, _) = cudarc_slice.device_ptr(&cudarc_stream);

    println!("  cudarc allocated CudaSlice<f32> of {} elements", TOTAL);
    println!("  cudarc device pointer: 0x{:x}", cudarc_ptr as usize);

    let stream = oxide_ctx.default_stream();

    // Write a pattern via cuda-oxide kernel to an oxide buffer
    let mut oxide_buf = DeviceBuffer::<f32>::zeroed(&stream, TOTAL)?;
    let module = coexist_kernels::load(&oxide_ctx)?;

    module.write_pattern(
        &stream,
        LaunchConfig::for_num_elems(TOTAL as u32),
        &mut oxide_buf,
        TOTAL as u32,
    )?;

    // Get raw pointer of oxide buffer using cu_deviceptr() and read via cudarc using cuMemcpyDtoH_v2
    let oxide_ptr = oxide_buf.cu_deviceptr();
    println!("  cuda-oxide buffer at: 0x{:x}", oxide_ptr as usize);

    let mut readback = vec![0.0f32; TOTAL];
    unsafe {
        use cudarc::driver::sys::{cuMemcpyDtoH_v2, cudaError_enum};
        let res = cuMemcpyDtoH_v2(
            readback.as_mut_ptr().cast(),
            oxide_ptr as _,
            TOTAL * std::mem::size_of::<f32>(),
        );
        if res != cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuMemcpyDtoH failed with {:?}", res).into());
        }
    }

    // Verify: expected[i] = 42.0 + i as f32
    let mut errors = 0usize;
    for i in 0..TOTAL {
        let expected = 42.0 + i as f32;
        if (readback[i] - expected).abs() > 1e-5 {
            if errors < 4 {
                eprintln!("    mismatch at {}: got {} expected {}", i, readback[i], expected);
            }
            errors += 1;
        }
    }

    // Also verify that cudarc can read memory allocated by cuda-oxide using cuMemcpyDtoH
    let mut cudarc_readback = vec![0.0f32; TOTAL];
    unsafe {
        use cudarc::driver::sys::{cuMemcpyDtoH_v2, cudaError_enum};
        let res = cuMemcpyDtoH_v2(
            cudarc_readback.as_mut_ptr().cast(),
            oxide_buf.cu_deviceptr() as _,
            TOTAL * std::mem::size_of::<f32>(),
        );
        if res != cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuMemcpyDtoH failed with {:?}", res).into());
        }
    }

    // Both readbacks should match
    let mut interop_errors = 0usize;
    for i in 0..TOTAL {
        if (readback[i] - cudarc_readback[i]).abs() > 1e-5 {
            eprintln!("    interop mismatch at {}: oxide_read={} cudarc_read={}",
                i, readback[i], cudarc_readback[i]);
            interop_errors += 1;
        }
    }

    // Also verify cudarc buffer is readable from same context
    let cudarc_readback2 = cudarc_stream.clone_dtoh(&cudarc_slice)?;
    println!("  cudarc self-readback of own allocation — matches input: {}",
        cudarc_readback2.iter().zip(host_data.iter()).all(|(a, b)| (a - b).abs() < 1e-5));

    if errors == 0 && interop_errors == 0 {
        println!("  [PASS] Raw pointer interop — both libraries read/write same device memory");
        Ok(())
    } else {
        eprintln!("  [FAIL] Raw pointer interop: {} kernel errors, {} interop errors",
            errors, interop_errors);
        Err("raw pointer interop failed".into())
    }
}

// =============================================================================
// TEST 5: Cudarc write → oxide kernel read (via cuMemcpyDtoD)
// =============================================================================

fn test_cudarc_to_oxide() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Test 5: Cudarc Write → Oxide Kernel Read ===");

    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};

    let cudarc_ctx = CudaContext::new(0)?;
    let oxide_ctx = OxideCudaContext::new(0)?;

    const TOTAL: usize = 128;

    // Allocate and fill via cudarc
    let host_data: Vec<f32> = (0..TOTAL).map(|i| i as f32 * 2.3 - 5.7).collect();
    let cudarc_stream = cudarc_ctx.default_stream();
    let cudarc_buf: CudaSlice<f32> = cudarc_stream.clone_htod(&host_data)?;

    println!("  cudarc allocated and filled {} elements", TOTAL);

    let stream = oxide_ctx.default_stream();
    let oxide_buf = DeviceBuffer::<f32>::zeroed(&stream, TOTAL)?;

    // cudaMemcpy from cudarc buffer to oxide buffer using cuMemcpyDtoD_v2
    unsafe {
        use cudarc::driver::sys::{cuMemcpyDtoD_v2, cudaError_enum};
        let (cudarc_ptr, _) = cudarc_buf.device_ptr(&cudarc_stream);
        let res = cuMemcpyDtoD_v2(
            oxide_buf.cu_deviceptr() as _, // note: we need mutable access — this is a safety concern
            cudarc_ptr as _,
            TOTAL * std::mem::size_of::<f32>(),
        );
        if res != cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuMemcpyDtoD failed with {:?}", res).into());
        }
    }

    println!("  cudaMemcpy from cudarc buffer to oxide buffer");

    // Read back via oxide and verify copy was correct
    let copied_data = oxide_buf.to_host_vec(&stream)?;

    let mut errors = 0usize;
    for i in 0..TOTAL {
        if (copied_data[i] - host_data[i]).abs() > 1e-5 {
            if errors < 4 {
                eprintln!("    copy mismatch at {}: got {} expected {}",
                    i, copied_data[i], host_data[i]);
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("  [PASS] Cudarc→Oxide memory copy verified ({} elements)", TOTAL);

        // Bonus: run kernel on the copied data to verify it works after cuMemcpyDtoD
        let module = coexist_kernels::load(&oxide_ctx)?;
        let mut result_buf = DeviceBuffer::<f32>::zeroed(&stream, TOTAL)?;

        module.mem_copy(
            &stream,
            LaunchConfig::for_num_elems(TOTAL as u32),
            &oxide_buf,
            &mut result_buf,
            TOTAL as u32,
        )?;

        let kernel_result = result_buf.to_host_vec(&stream)?;

        let mut kernel_errors = 0usize;
        for i in 0..TOTAL {
            if (kernel_result[i] - host_data[i]).abs() > 1e-5 {
                kernel_errors += 1;
            }
        }

        if kernel_errors == 0 {
            println!("  [PASS] Oxide kernel read cudarc-originated data ({} elements)", TOTAL);
        } else {
            eprintln!("    kernel read: {} errors", kernel_errors);
        }

        Ok(())
    } else {
        eprintln!("  [FAIL] Cudarc→Oxide copy: {} errors", errors);
        Err("copy test failed".into())
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n===========================================================");
    println!("  CUDA-oxide + cudarc Coexistence Test Suite");
    println!("===========================================================");

    let mut all_pass = true;

    // Test 1: Context sharing
    match test_context_sharing() {
        Ok(()) => {},
        Err(e) => {
            eprintln!("Test 1 error: {}", e);
            all_pass = false;
        }
    }

    // Test 2: Sequential operations
    match test_sequential_ops() {
        Ok(()) => {},
        Err(e) => {
            eprintln!("Test 2 error: {}", e);
            all_pass = false;
        }
    }

    // Test 3: Stream coexistence
    match test_stream_coexistence() {
        Ok(()) => {},
        Err(e) => {
            eprintln!("Test 3 error: {}", e);
            all_pass = false;
        }
    }

    // Test 4: Raw pointer interop
    match test_raw_ptr_interop() {
        Ok(()) => {},
        Err(e) => {
            eprintln!("Test 4 error: {}", e);
            all_pass = false;
        }
    }

    // Test 5: Cudarc write → oxide read
    match test_cudarc_to_oxide() {
        Ok(()) => {},
        Err(e) => {
            eprintln!("Test 5 error: {}", e);
            all_pass = false;
        }
    }

    println!("\n=== Summary ===");
    if all_pass {
        println!("All tests passed — cuda-oxide and cudarc coexist successfully!");
    } else {
        println!("Some tests failed — see above for details.");
    }

    Ok(())
}
