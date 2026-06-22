//! Test binary for infers-cuda-oxide-kernels.
//!
//! Allocates test data, launches `infers_add_bf16`, and verifies the
//! result against a CPU reference (bf16→f32 add → bf16).

#![feature(f16)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

/// Convert f32 to bf16 bits (truncate — matches `cuda_device::tcgen05::f32_to_bf16`).
fn f32_to_bf16_cpu(val: f32) -> u16 {
    ((val.to_bits() >> 16) & 0xFFFF) as u16
}

/// Convert bf16 bits to f32.
fn bf16_to_f32_cpu(bf16_bits: u16) -> f32 {
    f32::from_bits((bf16_bits as u32) << 16)
}

/// CPU reference for infers_add_bf16 (bit-exact).
fn infers_add_bf16_cpu(a: &[u16], b: &[u16]) -> Vec<u16> {
    a.iter()
        .zip(b.iter())
        .map(|(&a, &b)| f32_to_bf16_cpu(bf16_to_f32_cpu(a) + bf16_to_f32_cpu(b)))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== infers-cuda-oxide-kernels: infers_add_bf16 test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(&ctx)?;

    const N: usize = 1024;

    // Generate test data: f32 values converted to bf16 bits (u16)
    let a_f32: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let b_f32: Vec<f32> = (0..N).map(|i| (N - i) as f32 * 0.5).collect();
    let a_bf16: Vec<u16> = a_f32.iter().map(|&x| f32_to_bf16_cpu(x)).collect();
    let b_bf16: Vec<u16> = b_f32.iter().map(|&x| f32_to_bf16_cpu(x)).collect();

    // Upload to device
    let a_dev = DeviceBuffer::from_host(&stream, &a_bf16)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b_bf16)?;
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, N)?;

    // Launch kernel
    module.infers_add_bf16(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &a_dev,
        &b_dev,
        &mut out_dev,
        N as u32,
    )?;

    // Read back
    let out_host = out_dev.to_host_vec(&stream)?;

    // CPU reference
    let expected = infers_add_bf16_cpu(&a_bf16, &b_bf16);

    // Verify (bit-exact comparison)
    let errors: usize = (0..N).filter(|&i| out_host[i] != expected[i]).count();

    if errors == 0 {
        println!("[PASS] infers_add_bf16: {} elements bit-exact", N);
    } else {
        eprintln!(
            "[FAIL] infers_add_bf16: {} mismatches out of {}",
            errors, N
        );
        if errors <= 5 {
            for i in 0..N {
                if out_host[i] != expected[i] {
                    eprintln!(
                        "  [{:>4}] GPU = {:>8} (bf16 {:>5})  CPU = {:>8} (bf16 {:>5})",
                        i,
                        bf16_to_f32_cpu(out_host[i]),
                        out_host[i],
                        bf16_to_f32_cpu(expected[i]),
                        expected[i]
                    );
                }
            }
        }
        std::process::exit(1);
    }

    Ok(())
}
