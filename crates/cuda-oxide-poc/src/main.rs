//! CUDA-oxide proof-of-concept: BF16 vector addition kernel.
//!
//! Replaces the nvcc-compiled `elementwise.cu` kernel with a Rust-compiled PTX
//! kernel using cuda-oxide's single-source compilation model.
//!
//! The original kernel (`elementwise.cu`) does:
//!   output[i] = a[i] + b[i]  (bf16)
//! via grid-stride loop: idx = threadIdx.x, stride = blockDim.x * gridDim.x
//!
//! This POC uses f32 internally (cuda-oxide supports Rust's native f16 but not
//! yet bf16 as a first-class type). The data flow is:
//!   host bf16 -> device f32 -> add -> device f32 -> host bf16
//! which matches the original elementwise.cu pattern (bf16->f32 add ->bf16).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// =============================================================================
// KERNEL — compiled to PTX by rustc-codegen-cuda
// =============================================================================

#[cuda_module]
mod vecadd_kernels {
    use super::*;

    /// Vector addition kernel: c[i] = a[i] + b[i]
    ///
    /// Grid-stride loop pattern matching the original elementwise.cu.
    #[kernel]
    pub fn vec_add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>, total_elements: u32) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();

        let total = total_elements as usize;
        for i in (tid as usize..total).step_by(stride as usize) {
            if i < total {
                unsafe {
                    *out.get_unchecked_mut(i) = a[i] + b[i];
                }
            }
        }
    }

    /// Simpler vector addition (no grid-stride, one element per thread).
    /// Used for initial pipeline validation.
    #[kernel]
    pub fn vec_add_simple(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = a[i] + b[i];
        }
    }
}

// =============================================================================
// HOST CODE — compiled to native x86_64 by LLVM
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== cuda-oxide POC: Vector Add ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 1024;

    // Test data
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| i as f32 * 2.0).collect();

    println!("Input vectors (first 5):");
    println!("  a = {:?}", &a_host[0..5]);
    println!("  b = {:?}", &b_host[0..5]);

    // Allocate device memory
    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // Load the embedded PTX bundle
    let module = vecadd_kernels::load(&ctx)?;

    // Launch: one thread per element (no grid-stride needed for this size)
    module.vec_add_simple(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &a_dev,
        &b_dev,
        &mut c_dev,
    )?;

    // Retrieve results
    let c_host = c_dev.to_host_vec(&stream)?;

    println!("\nOutput vector (first 5):");
    println!("  c = {:?}", &c_host[0..5]);

    // Verify
    let mut errors: usize = 0;
    for i in 0..N {
        let expected = a_host[i] + b_host[i];
        if (c_host[i] - expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!("  Error at [{}]: expected {}, got {}", i, expected, c_host[i]);
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("\n[PASS] All {} elements correct!", N);
    } else {
        eprintln!("\n[FAIL] {} errors out of {}", errors, N);
        return Err("Kernel verification failed".into());
    }

    // Also test grid-stride version with fewer threads than elements
    println!("\n--- Grid-stride test (256 threads for 1024 elements) ---");
    let mut c_dev2 = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    module.vec_add(
        &stream,
        LaunchConfig::for_num_elems(256), // fewer threads -> must stride
        &a_dev,
        &b_dev,
        &mut c_dev2,
        N as u32,
    )?;

    let c_host2 = c_dev2.to_host_vec(&stream)?;
    let mut errors2: usize = 0;
    for i in 0..N {
        let expected = a_host[i] + b_host[i];
        if (c_host2[i] - expected).abs() > 1e-5 {
            errors2 += 1;
        }
    }

    if errors2 == 0 {
        println!("[PASS] Grid-stride: All {} elements correct!", N);
    } else {
        eprintln!("[FAIL] Grid-stride: {} errors out of {}", errors2, N);
        return Err("Grid-stride kernel failed".into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test: allocate GPU memory, launch kernel, verify results.
    #[test]
    fn vec_add_integration() -> Result<(), Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        const N: usize = 256;
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i * 3.0)).collect();

        let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        let module = vecadd_kernels::load(&ctx)?;
        module.vec_add_simple(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )?;

        let c_host = c_dev.to_host_vec(&stream)?;
        for i in 0..N {
            let expected = a_host[i] + b_host[i];
            assert!(
                (c_host[i] - expected).abs() < 1e-5,
                "Mismatch at [{}]: expected {}, got {}",
                i,
                expected,
                c_host[i]
            );
        }

        Ok(())
    }
}
