// Build script for compiling CUDA kernels to .cubin files.
//
// This script runs nvcc to compile .cu source files into .cubin binaries
// that are loaded at runtime by the KernelRegistry.
//
// Prerequisites:
//   - CUDA Toolkit 13.x installed (nvcc on PATH)
//   - FlashInfer kernel source files in kernels/flashinfer-gdn/ and kernels/flashinfer-attn/
//   - Run scripts/extract-kernels.sh first to populate the source files
//
// The compiled .cubin files are placed in kernels/compiled/ and can be
// loaded at runtime via cuda_core::CudaModule::load().

use std::process::Command;
use std::path::Path;

fn main() {
    // Only compile kernels if the cuda feature is enabled and source files exist
    println!("cargo:rerun-if-changed=kernels/");

    let gpu_arch = "sm_100a"; // Blackwell (RTX 5060 Ti)

    // Check if nvcc is available
    if which_nvcc().is_none() {
        println!("cargo:warning=nvcc not found on PATH, skipping kernel compilation");
        return;
    }

    let compiled_dir = Path::new("kernels/compiled");
    if !compiled_dir.exists() {
        // Create compiled directory if it doesn't exist
        let _ = std::fs::create_dir_all(compiled_dir);
    }

    // Compile GDN kernels
    compile_kernel_if_exists(
        "kernels/flashinfer-gdn/gdn_prefill.cu",
        "gdn_prefill.cubin",
        gpu_arch,
    );

    compile_kernel_if_exists(
        "kernels/flashinfer-gdn/gdn_decode.cu",
        "gdn_decode.cubin",
        gpu_arch,
    );

    // Compile standard attention kernels
    compile_kernel_if_exists(
        "kernels/flashinfer-attn/batch_prefill.cu",
        "batch_prefill.cubin",
        gpu_arch,
    );

    compile_kernel_if_exists(
        "kernels/flashinfer-attn/batch_decode.cu",
        "batch_decode.cubin",
        gpu_arch,
    );

    // Compile sampling kernels
    compile_kernel_if_exists(
        "kernels/flashinfer-attn/sampling.cu",
        "sampling.cubin",
        gpu_arch,
    );
}

fn compile_kernel_if_exists(src: &str, output: &str, arch: &str) {
    let src_path = Path::new(src);
    if !src_path.exists() {
        println!("cargo:warning=Kernel source not found, skipping: {}", src);
        return;
    }

    let output_path = format!("kernels/compiled/{}", output);
    let output_parent = Path::new(&output_path).parent().unwrap();
    if !output_parent.exists() {
        let _ = std::fs::create_dir_all(output_parent);
    }

    match compile_kernel(src, &output_path, arch) {
        Ok(_) => println!("cargo:warning=Compiled kernel: {} -> {}", src, output_path),
        Err(e) => println!("cargo:warning=Failed to compile {}: {}", src, e),
    }
}

fn compile_kernel(src: &str, output: &str, arch: &str) -> Result<(), String> {
    let status = Command::new("nvcc")
        .args([
            "-cubin",
            &format!("-arch={}", arch),
            "-O3",
            "--use_fast_math",
            "-I", "kernels/flashinfer-gdn",
            "-I", "kernels/flashinfer-attn",
            src,
            "-o",
            output,
        ])
        .status()
        .map_err(|e| format!("nvcc execution failed: {}", e))?;
    
    if status.success() {
        Ok(())
    } else {
        Err(format!("nvcc returned non-zero exit code for {}", src))
    }
}

fn which_nvcc() -> Option<String> {
    // Check PATH first
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
    }
    
    // Check common CUDA install locations
    let common_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.2/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/bin/nvcc",
    ];
    
    for path in &common_paths {
        if Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    
    None
}
