// Build script for compiling infers CUDA kernels to .cubin files.
//
// Compiles all .cu source files from kernels/infers/ into .cubin binaries
// using nvcc. Targets sm_120 (Blackwell RTX 5060 Ti) by default,
// configurable via INFERS_CUDA_ARCH environment variable.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/infers/");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_dir = manifest_dir.join("kernels/infers");
    let compiled_dir = manifest_dir.join("kernels/compiled");

    let arch_str = std::env::var("INFERS_CUDA_ARCH")
        .unwrap_or_else(|_| "sm_120".to_string());

    let nvcc_path = find_nvcc();
    if nvcc_path.is_empty() {
        println!("cargo:warning=nvcc not found, skipping kernel compilation");
        return;
    }

    if !source_dir.exists() {
        println!("cargo:warning=Source directory not found: {:?}", source_dir);
        return;
    }

    let _ = std::fs::create_dir_all(&compiled_dir);

    // Find all .cu files
    let cu_files = find_cu_files(&source_dir);
    if cu_files.is_empty() {
        println!("cargo:warning=No .cu files found in {:?}", source_dir);
        return;
    }

    for src_path in cu_files {
        let stem = src_path.file_stem().unwrap_or_default().to_string_lossy();
        let output_name = format!("{}.cubin", stem);
        let output_path = compiled_dir.join(&output_name);

        match compile_kernel(&nvcc_path, &src_path, &output_path, &arch_str, &source_dir) {
            Ok(()) => {
                println!(
                    "cargo:warning=Compiled {} -> {}",
                    src_path.display(),
                    output_path.display()
                );
            }
            Err(e) => {
                println!(
                    "cargo:warning=Failed to compile {}: {}",
                    src_path.display(),
                    e
                );
            }
        }
    }
}

fn find_cu_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|e| e == "cu") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn compile_kernel(
    nvcc: &str,
    src: &Path,
    output: &Path,
    arch: &str,
    include_dir: &Path,
) -> Result<(), String> {
    let status = Command::new(nvcc)
        .args([
            "-cubin",
            &format!("-arch={}", arch),
            "-O3",
            "--use_fast_math",
            "-I",
            include_dir.to_string_lossy().as_ref(),
            src.to_string_lossy().as_ref(),
            "-o",
            output.to_string_lossy().as_ref(),
        ])
        .status()
        .map_err(|e| format!("nvcc failed to execute: {}", e))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "nvcc returned non-zero exit code: {:?}",
            status.code()
        ))
    }
}

fn find_nvcc() -> String {
    if let Ok(output) = Command::new("which").arg("nvcc").output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return path;
        }
    }

    let common_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.2/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/bin/nvcc",
    ];

    for path in &common_paths {
        if Path::new(path).exists() {
            return path.to_string();
        }
    }

    String::new()
}
