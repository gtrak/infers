# Phase 2: CUDA Backend

**Duration:** 2 weeks  
**Goal:** Set up CUDA runtime, compile custom CUDA kernels, and establish the kernel launching pipeline.

## Deliverables

1. cuda-oxide workspace integration (`cuda-core`, `cuda-async`, `cuda-host`)
2. cudarc cuBLASLt bindings
3. NCCL communicator setup for 2 GPUs
4. Custom CUDA kernel compilation pipeline
5. Kernel loading test (load `.cubin` via cuda-oxide)
6. Memory allocator (GPU block pool, pinned host buffers)
7. Stream management for 2 GPUs
8. Context sharing between cuda-oxide and cudarc

## Technical Details

### cuda-oxide Integration

**Cargo.toml for `crates/cuda/`:**

```toml
[package]
name = "infers-cuda"
version.workspace = true
edition.workspace = true

[dependencies]
cuda-core = { workspace = true }
cuda-async = { workspace = true }
cuda-host = { workspace = true }
cudarc = { workspace = true }
anyhow = { workspace = true }
tracing = { workspace = true }

[build-dependencies]
cc = "1.0"
```

**Context creation:**

```rust
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cudarc::driver::CudaContext as CudarcContext;

pub struct CudaRuntime {
    /// cuda-oxide context (primary)
    pub oxide_ctx: CudaContext,
    
    /// cudarc context (shares primary context)
    pub cudarc_ctx: CudarcContext,
    
    /// Streams for async execution
    pub streams: Vec<CudaStream>,
    
    /// Device count
    pub num_devices: usize,
}

impl CudaRuntime {
    pub fn new() -> Result<Self> {
        let num_devices = cuda_core::device_count()?;
        assert_eq!(num_devices, 2, "Expected 2 GPUs");
        
        // Create cuda-oxide context (uses primary context)
        let oxide_ctx = CudaContext::new(0)?;
        
        // Create cudarc context (shares primary context via from_raw)
        let cudarc_ctx = CudarcContext::from_raw_context(
            oxide_ctx.raw_handle()
        )?;
        
        // Create one stream per device
        let mut streams = Vec::with_capacity(num_devices);
        for i in 0..num_devices {
            let stream = CudaStream::new(i)?;
            streams.push(stream);
        }
        
        Ok(Self {
            oxide_ctx,
            cudarc_ctx,
            streams,
            num_devices,
        })
    }
}
```

**Note:** Both cuda-oxide and cudarc use `cuDevicePrimaryCtxRetain`, so they share the same underlying `CUcontext`. We must ensure only one library manages the context lifetime.

### cudarc cuBLASLt

```rust
use cudarc::cublaslt::{CudaBlasLt, MatmulConfig};

pub struct GemmEngine {
    pub cublaslt: CudaBlasLt,
}

impl GemmEngine {
    pub fn new(stream: &CudaStream) -> Result<Self> {
        let cublaslt = CudaBlasLt::new(stream)?;
        Ok(Self { cublaslt })
    }
    
    pub fn matmul(
        &self,
        a: &DeviceBuffer<half>,
        b: &DeviceBuffer<half>,
        c: &mut DeviceBuffer<half>,
        m: usize, n: usize, k: usize,
    ) -> Result<()> {
        let config = MatmulConfig {
            m: m as i64,
            n: n as i64,
            k: k as i64,
            transa: false,
            transb: false,
            dtype: cudaDataType_t::CUDA_R_16F,
        };
        
        self.cublaslt.matmul(a, b, c, &config)?;
        Ok(())
    }
}
```

### NCCL Setup

```rust
use cudarc::nccl::{Comm, Id, ReduceOp};

pub struct NcclCommunicator {
    pub comm: Comm,
    pub rank: usize,
    pub world_size: usize,
}

impl NcclCommunicator {
    pub fn new(rank: usize, world_size: usize) -> Result<Self> {
        let id = Id::new()?;
        let comm = Comm::from_rank(&stream, rank, world_size, id)?;
        
        Ok(Self {
            comm,
            rank,
            world_size,
        })
    }
    
    pub fn all_reduce(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.comm.all_reduce(
            input.as_slice(),
            output.as_mut_slice(),
            &ReduceOp::Sum,
        )?;
        Ok(())
    }
}
```

### Custom CUDA Kernel Compilation Pipeline

**Step 1: Compile custom kernel source**

```bash
# scripts/compile-kernels.sh
# NOTE: Project uses custom CUDA kernels in kernels/infers/
# No extraction from vLLM needed

KERNEL_DIR="kernels/infers"
COMPILED_DIR="kernels/compiled"

mkdir -p $COMPILED_DIR
```

**Step 2: Build script (`crates/cuda/build.rs`)**

```rust
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    
    let cuda_arch = std::env::var("INFERS_CUDA_ARCH").unwrap_or_else(|_| "sm_120".to_string());

    // Compile all custom kernels from kernels/infers/
    let kernel_dir = Path::new("kernels/infers");
    if kernel_dir.exists() {
        for entry in fs::read_dir(kernel_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension() == Some(OsStr::new("cu")) {
                compile_kernel(&path, &cuda_arch);
            }
        }
    }
}

fn compile_kernel(src: &str, output: &str, arch: &str) {
    let status = Command::new("nvcc")
        .args(&[
            "-cubin",
            &format!("-arch={}", arch),
            "-O3",
            "--use_fast_math",
            src,
            "-o",
            &format!("kernels/compiled/{}", output),
        ])
        .status()
        .expect("nvcc failed");
    
    assert!(status.success(), "Kernel compilation failed: {}", src);
}
```

**Step 3: Load kernels in Rust**

```rust
use cuda_host::module::Module;

pub struct KernelRegistry {
    pub gdn_prefill: Kernel,
    pub gdn_decode: Kernel,
    pub attn_prefill: Kernel,
    pub attn_decode: Kernel,
    pub sampling_topk: Kernel,
    pub sampling_topp: Kernel,
}

impl KernelRegistry {
    pub fn load(ctx: &CudaContext) -> Result<Self> {
        let gdn_prefill = load_kernel(ctx, "gdn_prefill.cubin", "chunk_gated_delta_rule")?;
        let gdn_decode = load_kernel(ctx, "gdn_decode.cubin", "fused_sigmoid_gating_delta_rule_update")?;
        let attn_prefill = load_kernel(ctx, "batch_prefill.cubin", "BatchPrefillWithPagedKVCache")?;
        let attn_decode = load_kernel(ctx, "batch_decode.cubin", "BatchDecodeWithPagedKVCache")?;
        let sampling_topk = load_kernel(ctx, "sampling.cubin", "top_k_sampling_from_probs")?;
        let sampling_topp = load_kernel(ctx, "sampling.cubin", "top_p_sampling_from_probs")?;
        
        Ok(Self {
            gdn_prefill,
            gdn_decode,
            attn_prefill,
            attn_decode,
            sampling_topk,
            sampling_topp,
        })
    }
}

fn load_kernel(ctx: &CudaContext, cubin_file: &str, func_name: &str) -> Result<Kernel> {
    let bytes = std::fs::read(format!("kernels/compiled/{}", cubin_file))?;
    let module = Module::load(ctx, &bytes)?;
    let kernel = module.get_function(func_name)?;
    Ok(kernel)
}
```

### Memory Allocator

```rust
pub struct GpuAllocator {
    /// Pool of free GPU blocks
    free_blocks: Vec<DeviceBuffer<u8>>,
    
    /// Block size in bytes
    block_size: usize,
    
    /// Total allocated
    total_allocated: usize,
    
    /// Max allowed
    max_bytes: usize,
}

impl GpuAllocator {
    pub fn new(block_size: usize, max_bytes: usize) -> Result<Self> {
        Ok(Self {
            free_blocks: Vec::new(),
            block_size,
            total_allocated: 0,
            max_bytes,
        })
    }
    
    pub fn allocate(&mut self, stream: &CudaStream) -> Result<DeviceBuffer<u8>> {
        if let Some(block) = self.free_blocks.pop() {
            Ok(block)
        } else if self.total_allocated + self.block_size <= self.max_bytes {
            let block = DeviceBuffer::alloc(self.block_size, stream)?;
            self.total_allocated += self.block_size;
            Ok(block)
        } else {
            Err(anyhow!("Out of GPU memory"))
        }
    }
    
    pub fn free(&mut self, block: DeviceBuffer<u8>) {
        self.free_blocks.push(block);
    }
}
```

## File Structure

```
crates/cuda/
  Cargo.toml
  build.rs
  src/
    lib.rs
    context.rs          # CudaContext, CudaRuntime
    stream.rs           # CudaStream wrapper
    memory.rs           # GpuAllocator, DeviceBuffer helpers
    kernels.rs          # KernelRegistry, kernel loading
    gemm.rs             # GemmEngine (cuBLASLt)
    nccl.rs             # NcclCommunicator
  kernels/
    flashinfer-gdn/     # GDN .cu files (extracted from vLLM)
    flashinfer-attn/    # Attention .cu files
    compiled/           # Output .cubin files
```

## Testing

### Kernel Loading Test

```rust
#[test]
fn test_kernel_loading() {
    let ctx = CudaContext::new(0).unwrap();
    let registry = KernelRegistry::load(&ctx).unwrap();
    
    // Verify all kernels loaded
    assert!(registry.gdn_prefill.is_valid());
    assert!(registry.gdn_decode.is_valid());
    assert!(registry.attn_prefill.is_valid());
    assert!(registry.attn_decode.is_valid());
}
```

### Memory Allocation Test

```rust
#[test]
fn test_memory_allocation() {
    let stream = CudaStream::new(0).unwrap();
    let mut allocator = GpuAllocator::new(16 * 1024 * 1024, 1024 * 1024 * 1024).unwrap();
    
    let block = allocator.allocate(&stream).unwrap();
    assert_eq!(block.len(), 16 * 1024 * 1024);
    
    allocator.free(block);
}
```

### NCCL Test

```rust
#[test]
fn test_nccl_all_reduce() {
    // Requires 2 GPUs
    let comm0 = NcclCommunicator::new(0, 2).unwrap();
    let comm1 = NcclCommunicator::new(1, 2).unwrap();
    
    let input0 = DeviceBuffer::from_slice(&[1.0f32; 100]).unwrap();
    let input1 = DeviceBuffer::from_slice(&[2.0f32; 100]).unwrap();
    
    let mut output0 = DeviceBuffer::alloc(100).unwrap();
    let mut output1 = DeviceBuffer::alloc(100).unwrap();
    
    comm0.all_reduce(&input0, &mut output0).unwrap();
    comm1.all_reduce(&input1, &mut output1).unwrap();
    
    // Both should have [3.0; 100]
    let result0 = output0.to_vec().unwrap();
    let result1 = output1.to_vec().unwrap();
    
    assert!(result0.iter().all(|&x| (x - 3.0).abs() < 1e-5));
    assert!(result1.iter().all(|&x| (x - 3.0).abs() < 1e-5));
}
```

## Dependencies

### Phase 2 → Phase 1

Uses workspace and crate structure from Phase 1.

### Phase 2 → Phase 3

CUDA runtime will be used by model loading for device memory allocation.

### External Dependencies

- CUDA Toolkit 13.x (for Blackwell support)
- nvcc compiler
- cuda-oxide workspace (git)
- cudarc v0.19.7 (crates.io)
- NCCL 2.22+

## Success Criteria

1. cuda-oxide and cudarc coexist without context conflicts
2. Custom CUDA kernels compile to `.cubin` without errors
3. All `.cubin` files load successfully at runtime
4. cuBLASLt GEMM works for FP16 matrices
5. NCCL all-reduce works across 2 GPUs
6. Memory allocator can allocate/free blocks
7. All tests pass on target hardware (2× RTX 5060 Ti)

## Cross-References

- **Research:** See `../research/kernels.md` for kernel compilation strategy (deprecated FlashInfer notes)
- **Research:** See `../research/parallelism.md` for NCCL usage in TP/PP
- **Phase 3:** Model loader will use `GpuAllocator` for weight loading
- **Phase 4:** Forward pass will use `KernelRegistry`
- **Phase 5:** PP will use P2P communication primitives

## Open Questions

1. Do we need custom kernels for RMSNorm and SiLU, or can we use cuDNN?
2. Should we compile kernels at build time or runtime?
3. How to handle kernel caching across server restarts?
