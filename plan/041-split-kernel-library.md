# Phase 041: Split Kernel Library into Logical Modules

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: Phase 042 (selective module loading per configuration)
**Blocked by**: None
**Rationale**: The kernel library (`crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`) is a 3713-line monolith containing 44 `#[kernel]` functions plus traits, helper functions, and structs inside a single `#[cuda_module] pub mod kernels { ... }` block. This makes navigation difficult, bloats compile times, and forces loading ALL kernels even when a model only uses INT4 or NVFP4. The `OxideKernels` bridge (`oxide_bridge.rs`, 2674 lines) wraps every kernel with a manual `push_slice_arg` / `push_scalar_arg` / `raw_launch` method — a 1:1 copy of what cuda-oxide's `#[cuda_module]` macro already generates via typed launch methods.

The cuda-oxide `#[cuda_module]` macro supports multiple module blocks per crate, each generating its own `LoadedModule` struct with typed launch methods. The `from_module(Arc<CudaModule>)` method creates a typed wrapper from an already-loaded cubin without re-loading — just O(1) function lookups. Shared device helper functions annotated with `#[device]` and defined outside any `#[cuda_module]` block compile to PTX via transitive call-graph collection and are callable from any module's kernels.

This phase splits the monolith into 11 thematic files, replaces the manual `OxideKernels` bridge with cuda-oxide's generated typed modules, and updates all call sites in the backend.
---

## Goal

1. Split `kernel-lib/src/lib.rs` (3713 lines) into 11 files by logical function.
2. Replace `OxideKernels` (manual bridge, 2674 lines) with typed `LoadedModule` structs from cuda-oxide.
3. Update all ~50 call sites in `crates/backends/native/src/` to use typed module methods.
4. Verify identical performance (48ms INT4 decode, 105ms NVFP4 decode).
5. Verify correctness (31 kernel tests pass, "Paris" output).

## Current State

### Source layout
```
kernel-lib/src/
└── lib.rs           # 3713 lines, 44 #[kernel] functions, one #[cuda_module] block
```

### Loading architecture
```
oxide_bridge.rs (2674 lines):
  pub struct OxideKernels { ctx, module, functions: HashMap<&str, CudaFunction> }
  - KERNEL_NAMES: [&str; 42] — manually maintained list, must match cubin
  - launch_<kernel_name>() — 42 manual wrapper methods, each builds args via push_slice_arg/push_scalar_arg
  - raw_launch() — calls cudarc launch

engine.rs:
  struct PerGpuKernels { oxide: Arc<OxideKernels> }
  - load_per_gpu_kernels: OxideKernels::new(gpu_idx, cubin_path) — loads cubin per GPU

Call sites (50+ across 12 files):
  self.per_gpu_kernels[gpu_idx].oxide.launch_int4_gemm_v3_ksplit(stream, partial_sums, ...)
  self.per_gpu_kernels[gpu_idx].oxide.launch_rmsnorm_bf16(stream, hidden, weight, output, ...)
  self.per_gpu_kernels[gpu_idx].oxide.launch_silu_glu_bf16(stream, up, gate, &mut silu, ...)
```

## Target State

### Source layout
```
kernel-lib/src/
├── lib.rs                   # crate root: mod declarations + re-exports
├── shared.rs                # #[device] helpers + shared traits (no kernels)
│                             #   dev_sqrtf, f16_to_f32, fp4_e2m1_to_f32
│                             #   Dequantize trait, AutoRound, Gguf
│                             #   Fp8Format trait, Fp8E4M3, Fp8E5M2 + impls
│                             #   KvCacheFormat trait, KvBf16
├── common_kernels.rs        # #[cuda_module] — add, embedding_gather, argmax, softmax, sanitize_nan, kv_cache_write
├── norm_kernels.rs          # #[cuda_module] — rmsnorm, rms_norm_gated, l2norm
├── activation_kernels.rs    # #[cuda_module] — silu, silu_glu, attn_output_gate, conv1d_depthwise_silu
├── attention_kernels.rs     # #[cuda_module] — paged_attention_decode, paged_kv_write, paged_kv_read, rope
├── gdn_kernels.rs           # #[cuda_module] — gdn_recurrent_step, gdn_update, gdn_gated_delta_update, gdn_gated_delta_prefill, gdn_chunked_gated_delta_prefill, gdn_mamba2_update
├── int4_kernels.rs          # #[cuda_module] — int4_gemm_auto_round, _tiled, _ksplit, v3_ksplit, v4_ksplit, _warp, _warp_split, _gguf, int4_dequant_to_bf16, reduce_partial_sums_bf16
├── nvfp4_kernels.rs         # #[cuda_module] — nvfp4_dequant_to_bf16, nvfp4_gemm_fused, _ksplit, v3_ksplit
├── fp8_kernels.rs           # #[cuda_module] — fp8_quantize_e4m3, fp8_dequantize_e4m3, fp8_quantize_e5m2, fp8_dequantize_e5m2
└── bf16_kernels.rs          # #[cuda_module] — bf16_gemm_tiled
```

### Loading architecture
```
kernel-lib/src/lib.rs:
  pub mod shared;           // #[device] helpers + traits
  pub mod common_kernels;   // #[cuda_module] → LoadedModule
  pub mod norm_kernels;     // #[cuda_module] → LoadedModule
  pub mod activation_kernels;
  pub mod attention_kernels;
  pub mod gdn_kernels;
  pub mod int4_kernels;
  pub mod nvfp4_kernels;
  pub mod fp8_kernels;
  pub mod bf16_kernels;

crates/cuda/src/modules.rs (NEW — replaces oxide_bridge.rs):
  pub struct KernelModules {
      common: common_kernels::LoadedModule,
      norms: norm_kernels::LoadedModule,
      activation: activation_kernels::LoadedModule,
      attention: attention_kernels::LoadedModule,
      gdn: gdn_kernels::LoadedModule,
      int4: Option<int4_kernels::LoadedModule>,
      nvfp4: Option<nvfp4_kernels::LoadedModule>,
      fp8: Option<fp8_kernels::LoadedModule>,
      bf16_gemm: Option<bf16_kernels::LoadedModule>,
  }
  impl KernelModules {
      pub fn load(ordinal: usize, cubin_path: &str, config: &KernelConfig) -> Result<Self>
      pub fn new_from_module(module: Arc<CudaModule>, config: &KernelConfig) -> Result<Self>
  }

engine.rs:
  struct PerGpuKernels { modules: Arc<KernelModules> }
```

### Call sites (typed module methods)
```rust
// OLD:
oxide.launch_int4_gemm_v3_ksplit(stream, &mut partial_sums, &weight, &scales, &zeros, &input, n, k, gs, transposed, k_split)?;
oxide.launch_rmsnorm_bf16(stream, hidden, weight, output, hidden_size, eps)?;
oxide.launch_silu_glu_bf16(stream, &up, &gate, &mut silu, len)?;

// NEW:
modules.int4.as_ref().unwrap().int4_gemm_v3_ksplit(&stream, launch_config, &mut partial_sums, &weight, &scales, &zeros, &input, n, k, gs, transposed, k_split)?;
modules.norms.rmsnorm_bf16(&stream, launch_config, hidden, weight, output, hidden_size, eps)?;
modules.activation.silu_glu_bf16(&stream, launch_config, &up, &gate, &mut silu, len)?;
```

## Architecture

### Shared device functions

The `#[device]` attribute marks functions for device compilation. Functions annotated with `#[device]` and defined in a regular `pub mod shared { }` (outside any `#[cuda_module]` block) are compiled to PTX via transitive call-graph collection. They are callable from any `#[cuda_module]` block's kernels via `use super::shared::*;`.

```rust
// shared.rs
use cuda_device::{device, DisjointSlice, DynamicSharedArray, SharedArray};

#[device]
#[inline(always)]
pub fn dev_sqrtf(x: f32) -> f32 {
    unsafe { cuda_device::ptx_sqrtf(x) }  // or however it's currently done
}

#[device]
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// Traits and structs — no #[device] needed, they're type definitions
pub trait Dequantize {
    fn dequant_weight(packed: u32, shift: u32) -> i8;
    fn zero_offset() -> f32;
}
pub struct AutoRound;
pub struct Gguf;
impl Dequantize for AutoRound { ... }
impl Dequantize for Gguf { ... }

pub trait Fp8Format { ... }
pub struct Fp8E4M3;
pub struct Fp8E5M2;

pub trait KvCacheFormat {
    fn read_kv(pool: &[u16], offset: usize) -> f32;
}
pub struct KvBf16;
impl KvCacheFormat for KvBf16 { ... }
```

### Module file pattern

Each kernel module file follows the same pattern:

```rust
// int4_kernels.rs
use cuda_device::{cuda_module, kernel, launch_bounds, thread, DisjointSlice, DynamicSharedArray};
use super::shared::*;

#[cuda_module]
pub mod int4_kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(64)]
    pub fn int4_gemm_v3_ksplit(
        partial_sums: &mut [f32],
        weight: &[u32],
        ...
    ) {
        let scale = f16_to_f32(...);  // calls shared #[device] helper
        ...
    }
}
```

### KernelModules: selective loading

```rust
// crates/cuda/src/modules.rs
use std::sync::Arc;
use cuda_core::{CudaContext, CudaModule};
use infers_kernel_lib::{common_kernels, norm_kernels, ...};

pub enum QuantFormat { Int4, Nvfp4, Bf16 }

pub struct KernelConfig {
    pub quant_format: QuantFormat,
}

pub struct KernelModules {
    pub common: common_kernels::LoadedModule,
    pub norms: norm_kernels::LoadedModule,
    pub activation: activation_kernels::LoadedModule,
    pub attention: attention_kernels::LoadedModule,
    pub gdn: gdn_kernels::LoadedModule,
    pub int4: Option<int4_kernels::LoadedModule>,
    pub nvfp4: Option<nvfp4_kernels::LoadedModule>,
    pub fp8: Option<fp8_kernels::LoadedModule>,
    pub bf16_gemm: Option<bf16_kernels::LoadedModule>,
}

impl KernelModules {
    pub fn load(ordinal: usize, cubin_path: &str, config: &KernelConfig) -> anyhow::Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        ctx.bind_to_thread()?;
        let module = ctx.load_module_from_file(cubin_path)?;
        Self::new_from_module(module, config)
    }

    pub fn new_from_module(module: Arc<CudaModule>, config: &KernelConfig) -> anyhow::Result<Self> {
        Ok(Self {
            common: common_kernels::from_module(module.clone())?,
            norms: norm_kernels::from_module(module.clone())?,
            activation: activation_kernels::from_module(module.clone())?,
            attention: attention_kernels::from_module(module.clone())?,
            gdn: gdn_kernels::from_module(module.clone())?,
            int4: match config.quant_format {
                QuantFormat::Int4 => Some(int4_kernels::from_module(module.clone())?),
                _ => None,
            },
            nvfp4: match config.quant_format {
                QuantFormat::Nvfp4 => Some(nvfp4_kernels::from_module(module.clone())?),
                _ => None,
            },
            bf16_gemm: match config.quant_format {
                QuantFormat::Bf16 => Some(bf16_kernels::from_module(module.clone())?),
                _ => None,
            },
            fp8: None, // future: FP8 KV cache support
        })
    }
}
```

### PerGpuKernels update

```rust
// engine.rs
struct PerGpuKernels {
    modules: Arc<KernelModules>,
}

impl ForwardEngine {
    fn load_per_gpu_kernels(contexts: &[Arc<CudaContext>], num_gpus: usize, config: &KernelConfig) 
        -> Result<Vec<PerGpuKernels>> 
    {
        let cubin_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../cuda/kernels/compiled/oxide_kernels.cubin");
        let mut result = Vec::with_capacity(num_gpus);
        for gpu_idx in 0..num_gpus {
            let modules = Arc::new(KernelModules::load(gpu_idx, cubin_path, config)?);
            result.push(PerGpuKernels { modules });
        }
        Ok(result)
    }
}
```

### Call site migration patterns

Each call site changes from `oxide.launch_<name>(stream, args...)` to `modules.<group>.<name>(&stream, launch_config, args...)`.

The key difference: typed module methods require an explicit `LaunchConfig` argument, whereas the old `OxideKernels` wrappers computed it internally. Each call site must compute and pass the launch config.

**Example: rmsnorm (norm.rs)**
```rust
// OLD:
oxide.launch_rmsnorm_bf16(stream, hidden, weight, output, hidden_size as u32, eps)?;

// NEW:
let config = LaunchConfig {
    grid_dim: (1, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 256 * 4, // 256 f32 = 1KB
};
modules.norms.rmsnorm_bf16(&stream, config, hidden, weight, output, hidden_size as u32, eps)?;
```

**Example: int4_gemm_v3_ksplit (gemm_dispatch.rs)**
```rust
// OLD:
oxide.launch_int4_gemm_v3_ksplit(stream, partial_sums, weight, scales, zeros, input, n, k, gs, transposed, k_split)?;

// NEW:
let ksplit_config = LaunchConfig {
    grid_dim: ((n + 63) / 64, k_split as u32, 1),
    block_dim: (64, 1, 1),
    shared_mem_bytes: 0,
};
let reduce_config = LaunchConfig {
    grid_dim: ((n + 63) / 64, 1, 1),
    block_dim: (64, 1, 1),
    shared_mem_bytes: 0,
};
modules.int4.as_ref().unwrap().int4_gemm_v3_ksplit(&stream, ksplit_config, partial_sums, weight, scales, zeros, input, n, k, gs, transposed, k_split)?;
modules.int4.as_ref().unwrap().reduce_partial_sums_bf16(&stream, reduce_config, output, partial_sums, n, k_split)?;
```

### What gets deleted

- `crates/cuda/src/oxide_bridge.rs` — entire file (2674 lines). Replaced by `crates/cuda/src/modules.rs` (~80 lines) + cuda-oxide generated code.
- `KERNEL_NAMES` array — no longer needed (function handles resolved by `from_module`).
- All 42 `launch_*` wrapper methods — replaced by typed `LoadedModule` methods.
- `push_slice_arg`, `push_scalar_arg`, `raw_launch` helpers — no longer needed.
- `get_function` method — no longer needed (dead code was the fp8 path).
- Special chunked GDN smem setup in `OxideKernels::new` — moves to a `cuFuncSetAttribute` call after `from_module` in `KernelModules::load`.

### What stays

- `crates/cuda/src/oxide_bridge.rs` → renamed/replaced by `modules.rs`. Keep `OxideKernels` as a deprecated re-export type alias if needed for transition? No — full replacement.
- `crates/cuda/src/lib.rs` — update re-exports: remove `OxideKernels`, add `KernelModules`, `KernelConfig`, `QuantFormat`.
- `crates/cuda/src/gemm.rs` — unchanged (GemmEngine/cuBLASLt is separate).
- `crates/cuda/src/nccl.rs`, `stream.rs` — unchanged.

### Test binary impact

The test binary (`crates/cuda-oxide-kernels/src/main.rs`) currently uses `infers_kernel_lib::kernels::load(ctx)` and calls `module.kernel_name(...)`. After the split, it must load each module separately:

```rust
// OLD:
let module = infers_kernel_lib::kernels::load(ctx).unwrap();

// NEW:
let module = ctx.load_module_from_file(cubin_path).unwrap();
// OR for embedded: let module = infers_kernel_lib::load_all(ctx).unwrap();
let common = infers_kernel_lib::common_kernels::from_module(module.clone()).unwrap();
let norms = infers_kernel_lib::norm_kernels::from_module(module.clone()).unwrap();
let int4 = infers_kernel_lib::int4_kernels::from_module(module.clone()).unwrap();
...
```

The bench harness (`crates/cuda-oxide-kernels/src/bench.rs`) also needs the same update.

## Implementation Plan

### Step 1: Split source files (mechanical)

Move kernel functions from `lib.rs` into thematic files. No logic changes. The `#[cuda_module]` blocks change from one to ~10.

**Line ranges in current lib.rs:**
| Target file | Source lines | Items |
|---|---|---|
| `shared.rs` | 23-28 (dev_sqrtf), 889-908 (f16_to_f32), 668-858 (FP8 types + trait), 859-886 (Dequantize trait + AutoRound/Gguf), 2089-2110 (fp4_e2m1_to_f32), 2546-2565 (KvCacheFormat + KvBf16) | device helpers + traits + structs |
| `common_kernels.rs` | 29-65 (add), 66-90 (embedding_gather), 163-228 (argmax), 448-538 (softmax), 224-265 (kv_cache_write), 2530-2545 (sanitize_nan) | 6 kernels |
| `norm_kernels.rs` | 262-324 (rmsnorm), 325-389 (rms_norm_gated), 390-447 (l2norm) | 3 kernels |
| `activation_kernels.rs` | 91-116 (silu), 114-137 (silu_glu), 138-162 (attn_output_gate), 537-585 (conv1d_depthwise_silu) | 4 kernels |
| `attention_kernels.rs` | 583-667 (paged_kv_write), 623-667 (paged_kv_read), 2567-2721 (paged_attention_decode), 2722-2779 (rope) | 4 kernels |
| `gdn_kernels.rs` | 2780-2880 (gdn_recurrent_step), 2881-2946 (gdn_mamba2_update), 2947-3032 (gdn_update), 3033-3129 (gdn_gated_delta_update), 3130-3240 (gdn_gated_delta_prefill), 3241-3610 (gdn_chunked_gated_delta_prefill) | 6 kernels |
| `int4_kernels.rs` | 910-994 (int4_gemm_inner, NOT a kernel), 995-1012 (int4_gemm_auto_round), 1013-1133 (int4_gemm_auto_round_tiled), 1134-1246 (int4_gemm_auto_round_ksplit), 1247-1415 (int4_gemm_v3_ksplit), 1416-1755 (int4_gemm_v4_ksplit), 1756-1791 (reduce_partial_sums), 1792-1885 (int4_gemm_warp), 1886-1998 (int4_gemm_warp_split), 1999-2020 (int4_gemm_gguf), 2021-2088 (int4_dequant_to_bf16) | 11 kernels + 1 helper |
| `nvfp4_kernels.rs` | 2115-2173 (nvfp4_dequant_to_bf16), 2174-2237 (nvfp4_gemm_fused), 2238-2332 (nvfp4_gemm_fused_ksplit), 2333-2529 (nvfp4_gemm_v3_ksplit) | 4 kernels |
| `fp8_kernels.rs` | 826-858 (fp8_quantize/dequantize e4m3/e5m2) | 4 kernels |
| `bf16_kernels.rs` | 3611-3713 (bf16_gemm_tiled) | 1 kernel |

### Step 2: Create KernelModules + Config types

New file `crates/cuda/src/modules.rs` with `KernelModules`, `KernelConfig`, `QuantFormat`.

### Step 3: Update engine.rs + all call sites

Update `PerGpuKernels` to hold `Arc<KernelModules>`. Update ~50 call sites across:
- `engine.rs` (6 sites)
- `attention.rs` (25 sites)
- `gdn.rs` (8 sites)
- `gemm_dispatch.rs` (7 sites)
- `norm.rs` (5 sites)
- `add.rs` (4 sites)
- `sample.rs` (4 sites)
- `rope.rs` (2 sites)
- `embedding.rs` (2 sites)
- `prefill.rs` (1 site)
- `mlp.rs` (1 site)

Each site: `oxide.launch_<name>(stream, args...)` → `modules.<group>.<name>(&stream, launch_config, args...)`.

### Step 4: Delete oxide_bridge.rs

Remove the file, update `crates/cuda/src/lib.rs` exports.

### Step 5: Update test binary + bench harness

Update `crates/cuda-oxide-kernels/src/main.rs` and `bench.rs` to use typed modules instead of the monolithic `kernels::load()`.

### Step 6: Rebuild cubin + verify

```bash
cd /home/gary/dev/infers/crates/cuda-oxide-kernels && cargo oxide build
./target/release/infers-cuda-oxide-kernels --save-cubin ../cuda/kernels/compiled/oxide_kernels.cubin
cd /home/gary/dev/infers && cargo build --release
```

Verify:
- 31 kernel tests pass
- INT4 decode = 48ms/step
- NVFP4 decode = 105ms/step
- "Paris" output correct
- Bench harness works

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `#[device]` attribute not imported in shared.rs | Compile error | Add `use cuda_device::device;` |
| Generic kernel `int4_gemm_inner` not in `#[cuda_module]` | Trait impls not found | `int4_gemm_inner` is not a `#[kernel]`, it's a generic helper. Move it into `shared.rs` as `#[device]`. Or keep it as a plain `#[inline(always)]` function in `int4_kernels.rs` module body. |
| `from_module` fails to find kernel | Panic at load time | Kernel name mismatch — verify cubin has all kernels via `--save-cubin` + `cuobjdump` |
| LaunchConfig changes break tests | Test failures | Each test function in main.rs must pass explicit LaunchConfig |
| `f32_to_bf16` from `cuda_device::tcgen05` | Import path differs per module | Add `use cuda_device::tcgen05::f32_to_bf16;` in each kernel file that uses it |
| Chunked GDN kernel needs `cuFuncSetAttribute` for smem | Runtime error if not set | Call `cuFuncSetAttribute` in `KernelModules::load` after `from_module` |

## Acceptance Criteria

1. `kernel-lib/src/lib.rs` is ≤30 lines (just `pub mod` declarations + re-exports).
2. 10 new thematic files exist in `kernel-lib/src/`.
3. `oxide_bridge.rs` is deleted. `modules.rs` replaces it (≤100 lines).
4. No `KERNEL_NAMES` array exists. No `launch_*` wrapper methods.
5. `cargo oxide build` succeeds.
6. `cargo oxide run` — all 31 kernel tests pass.
7. `cargo build --release` succeeds.
8. INT4 decode = 48ms ± 2ms/step.
9. NVFP4 decode = 105ms ± 5ms/step.
10. "Paris" output correct for both models.
11. All 4 bench cases work with typed modules.
12. Each call site passes an explicit `LaunchConfig` to the typed module method.
