# Phase 18: cuda-oxide Exploration — Replace CUDA Kernels with Rust

---
**Status**: COMPLETE — Migration assessment: **MIGRATE LATER**
**Last Updated**: 2026-06-21
**Blocks**: Nothing (exploration only — no implementation commitment)
**Blocked by**: Nothing
**Rationale**: The project has 25 custom `.cu` kernel files compiled via nvcc's `build.rs`. cuda-oxide is a Rust→PTX compiler that could eliminate the nvcc dependency, let us write GPU kernels in Rust, and unify host+device code in one language. This phase explores feasibility, maps kernel features to cuda-oxide capabilities, and produces a migration assessment. The project already uses the exact same nightly (`nightly-2026-04-03`) and components (`rust-src`, `rustc-dev`, `llvm-tools`) that cuda-oxide requires — the toolchain alignment is already done.
---

## What is cuda-oxide?

cuda-oxide is an experimental Rust→CUDA compiler from NVIDIA Labs. It compiles standard Rust code to PTX (GPU assembly) via a custom `rustc` codegen backend → LLVM IR → `llc` NVPTX. No DSLs, no foreign language bindings — just Rust.

**Key features:**
- `#[cuda_module]` attribute: embeds PTX into host binary, generates typed launch functions
- `cuda-async`: lazy `DeviceOperation` graphs with stream pool scheduling
- `cuda-host`: host-side CUDA driver bindings (context, stream, memory)
- Same nightly pin as this project (`nightly-2026-04-03`)
- v0.1.0 alpha — "expect bugs, incomplete features, and API breakage"

**Build pipeline:**
```
Rust source (.rs)
  → rustc codegen backend (CudaCodegenBackend)
  → Pliron IR (MLIR-like)
  → LLVM IR (.ll)
  → llc NVPTX → PTX (.ptx)
  → CUDA driver loads PTX at runtime
```

## Current Kernel Inventory (25 .cu files)

### Tier 1: Simple (direct translation)

| Kernel | Lines | Features | cuda-oxide risk |
|--------|-------|----------|----------------|
| `argmax.cu` | ~30 | Block reduction, BF16→F32 | Low |
| `elementwise.cu` | ~20 | Element-wise add | Low |
| `embedding.cu` | ~25 | Index gather | Low |
| `rope.cu` | ~40 | Trig + rotation | Low |
| `silu.cu` | ~30 | SiLU + SwiGLU gating | Low |
| `sampling.cu` | ~25 | Argmax wrapper | Low |
| `kv_cache.cu` | ~35 | Scattered write by position | Low |
| `fp8_quantize.cu` | ~40 | Element-wise quantize/dequantize | Low |

### Tier 2: Medium (shared memory reductions)

| Kernel | Lines | Features | cuda-oxide risk |
|--------|-------|----------|----------------|
| `rmsnorm.cu` | 66 | Block reduction, shared mem, `rsqrtf` | Medium |
| `rms_norm_gated.cu` | ~80 | Same with gating | Medium |
| `l2norm_bf16.cu` | ~50 | L2 normalization | Medium |
| `softmax.cu` | 98 | 3-phase online softmax, causal mask | Medium |
| `conv1d_depthwise.cu` | ~60 | Causal depthwise conv | Medium |
| `paged_kv_write.cu` | ~50 | Block-table address translation | Medium |
| `paged_kv_read.cu` | ~50 | Block-table gather | Medium |

### Tier 3: Complex (advanced GPU programming)

| Kernel | Lines | Features | cuda-oxide risk |
|--------|-------|----------|----------------|
| `int4_gemm.cu` | 111 | INT4 dequant in registers, 16x16 blocks, `uint32_t` bit ops | High |
| `paged_attention_decode.cu` | 166 | 2-pass online softmax, GQA, shared mem Q reload | High |

### Tier 4: Very complex (shared memory heavy, algorithms)

| Kernel | Lines | Features | cuda-oxide risk |
|--------|-------|----------|----------------|
| `gdn_chunked_gated_delta_prefill.cu` | 405 | 80KB shared mem, WY representation, forward substitution | Very High |
| `gdn_gated_delta_prefill.cu` | ~200 | Sequential recurrence, L2 norm, softplus | High |
| `gdn_gated_delta_update.cu` | ~100 | Single-token decode | High |
| `gdn_mamba2_prefill.cu` | ~150 | SSM prefill, softplus, SiLU | High |
| `gdn_mamba2_update.cu` | ~80 | SSM decode | High |
| `gdn_recurrent_step.cu` | ~100 | Recurrent step, fp32 accumulation | High |
| `gdn_prefill.cu` | ~80 | Sequential prefill | High |
| `gdn_update.cu` | ~60 | Single-token update | High |

## Feature Mapping: CUDA → cuda-oxide

### What cuda-oxide supports (based on documentation)

| CUDA Feature | cuda-oxide Equivalent | Status |
|-------------|----------------------|--------|
| `__global__` kernel | `#[cuda_global]` fn | Supported |
| `__shared__` static | `#[cuda_shared]` static | Supported |
| `__shared__` dynamic | `extern "shared"` / dynamic | **Needs verification** |
| `__syncthreads()` | `sync_threads()` or barrier | Supported |
| `__launch_bounds__` | `#[launch_bounds]` | Supported |
| `blockIdx.x` / `threadIdx.x` | `block_idx()` / `thread_idx()` | Supported |
| `gridDim.x` / `blockDim.x` | `grid_dim()` / `block_dim()` | Supported |
| BF16 (`__nv_bfloat16`) | `bf16` type from cuda-oxide | Supported |
| `__bfloat162float` | `bf16::to_f32()` or cast | Supported |
| `__float2bfloat16` | `bf16::from_f32()` or cast | Supported |
| FP16 (`__half`) | `f16` type | Supported |
| `extern "C"` linkage | Not needed (Rust FFI) | N/A |
| `__restrict__` | Default (Rust ownership) | N/A |
| `__attribute__((maxdynamicsharedmemsize))` | Runtime config | **Needs verification** |
| `uint32_t` bit manipulation | `u32` native | Supported |
| `cuda_fp16.h` intrinsics | Rust f16 ops | Supported |

### What needs verification

1. **Dynamic shared memory**: The GDN chunked kernel uses `extern __shared__ char smem[]` with ~80KB. cuda-oxide's `#[cuda_shared]` may only support static sizes. Need to check if dynamic shared memory is supported.

2. **`maxdynamicsharedmemsize` attribute**: Used to tell the compiler the max shared memory a kernel may request. cuda-oxide may handle this via launch configuration instead.

3. **Shared memory pointer casting**: The CUDA kernels cast `char smem[]` to `float*`, `bf16*`, etc. Rust's type system may require a different pattern.

4. **Math intrinsics**: `rsqrtf()`, `expf()`, `softplusf()` — need to map to cuda-oxide's math support or use `libm`.

5. **Performance parity**: nvcc has decades of optimization. cuda-oxide's LLVM NVPTX path may produce different (possibly slower) PTX for complex kernels.

## Migration Strategy: Incremental, Kernel-by-Kernel

### Phase A: Proof of concept (1 kernel)

Migrate the simplest kernel — `elementwise.cu` (add) — to validate the build pipeline.

```rust
// Hypothetical cuda-oxide version of elementwise.cu
use cuda_core::prelude::*;

#[cuda_global]
fn infers_add_bf16(
    a: *const bf16,
    b: *const bf16,
    output: *mut bf16,
    n: i32,
) {
    let idx = block_idx() * block_dim() + thread_idx();
    if idx < n as u32 {
        let a_val = unsafe { *a.add(idx as usize) };
        let b_val = unsafe { *b.add(idx as usize) };
        unsafe { *output.add(idx as usize) = a_val + b_val; }
    }
}
```

Build integration:
```toml
# crates/cuda/Cargo.toml
[dependencies]
cuda-core = { workspace = true }
cuda-host = { workspace = true }
```

```rust
// In build.rs or via #[cuda_module]
#[cuda_module(path = "kernels/infers/")]
mod kernels {
    use cuda_core::prelude::*;
    // Kernel functions generated here
}
```

### Phase B: Simple kernels (8 kernels)

Migrate all Tier 1 kernels. These use basic features (indexing, math ops, no shared memory).

### Phase C: Shared memory kernels (7 kernels)

Migrate Tier 2 kernels. Requires verifying dynamic shared memory support.

### Phase D: Complex kernels (2 kernels)

Migrate `int4_gemm` and `paged_attention_decode`. These require bit manipulation and multi-pass algorithms.

### Phase E: GDN kernels (8 kernels)

Migrate the most complex kernels. This is the highest-risk tier — 80KB shared memory, WY representation, forward substitution.

### Phase F: Build system migration

Replace `build.rs` nvcc compilation with cuda-oxide's build pipeline. Remove `common.cuh` header.

## Benefits

1. **No nvcc dependency** — cuda-oxide compiles Rust → PTX directly. No CUDA toolkit installation needed (just the driver).
2. **Unified language** — Host and device code in the same `.rs` files. No context-switching between Rust and CUDA.
3. **Type safety** — Rust's type system catches kernel argument mismatches at compile time, not runtime.
4. **Memory safety** — Rust's ownership model prevents common CUDA bugs (use-after-free, data races in shared memory).
5. **Same toolchain** — Already using the exact same nightly and components. No toolchain changes needed.
6. **Leverages LLVM** — LLVM's NVPTX backend is actively maintained and produces good PTX.
7. **Ecosystem** — cudarc would be replaced by `cuda-host` + `cuda-async` (cuda-oxide's own runtime).

## Risks

1. **Alpha quality** — v0.1.0, "expect bugs, incomplete features, and API breakage". Production use is risky.
2. **Dynamic shared memory** — The GDN chunked kernel uses 80KB dynamic shared memory. If cuda-oxide doesn't support this, the most complex kernels can't be migrated.
3. **Performance regression** — nvcc has decades of optimization for BF16, INT4, and shared memory patterns. cuda-oxide's LLVM path may produce slower PTX for complex kernels.
4. **Debugging** — cuda-oxide's error messages for GPU code may be less mature than nvcc's.
5. **Migration cost** — 25 kernels × testing = significant effort. Each kernel needs correctness verification against the existing CUDA version.
6. **cudarc replacement** — The project uses cudarc extensively (context, streams, cuBLASLt, NCCL). cuda-oxide provides `cuda-host` and `cuda-async` but may not cover cuBLASLt or NCCL. Those would still need cudarc or a separate binding.

## Decision Framework

### Migrate NOW if:
- cuda-oxide supports dynamic shared memory (80KB+)
- Performance on Tier 1-2 kernels matches nvcc (within 5%)
- The team is comfortable with alpha-quality tooling
- The benefit of eliminating nvcc outweighs migration cost

### Migrate LATER if:
- cuda-oxide is still alpha
- Dynamic shared memory isn't supported
- Performance is significantly worse on complex kernels
- cudarc integration is incomplete

### DON'T migrate if:
- cuBLASLt and NCCL have no cuda-oxide bindings (these are critical for the project)
- Performance regression is >10% on production kernels

## Open Questions

1. Does cuda-oxide support `extern __shared__` dynamic shared memory with runtime sizing?
2. Does cuda-oxide provide cuBLASLt bindings (or can cudarc coexist)?
3. Does cuda-oxide provide NCCL bindings (or can cudarc coexist)?
4. What's the PTX quality comparison for `int4_gemm` (INT4 dequant in registers)?
5. Can `cuda-async`'s `DeviceOperation` graphs replace cudarc's stream management?
6. How does `#[cuda_module]` handle kernels that need different shared memory sizes per launch?
7. Can cuda-oxide kernels be loaded alongside cudarc-launched cuBLASLt/NCCL ops?

## Task Breakdown (Exploration Only)

### Commit 1: Install cargo-oxide and verify build

**Files**: `crates/cuda/Cargo.toml`, new `crates/cuda/src/oxide_test.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Install `cargo-oxide` | `cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide` |
| 2 | Run `cargo oxide doctor` | Verify toolchain, CUDA toolkit, LLVM, codegen backend |
| 3 | Uncomment cuda-oxide deps in Cargo.toml | Enable `cuda-core`, `cuda-host` |
| 4 | Create a minimal `#[cuda_global]` kernel | Vector add (replaces `elementwise.cu`) |
| 5 | Build with `cargo oxide build` | Verify PTX generation |
| 6 | Run on GPU | Verify correctness |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: `cargo oxide build` produces PTX. A vector-add kernel runs on GPU and produces correct results.

---

### Commit 2: Shared memory test

**Files**: `crates/cuda/src/oxide_shared.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Write a shared-memory reduction kernel in Rust | Port `rmsnorm.cu` logic |
| 2 | Test dynamic shared memory allocation | Verify `extern __shared__` equivalent works |
| 3 | Test `__launch_bounds__` equivalent | Verify launch bounds are respected |
| 4 | Compare PTX output vs nvcc version | Check for similar instruction patterns |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: Shared-memory kernel compiles and runs correctly. Dynamic shared memory works.

---

### Commit 3: BF16 + INT4 type test

**Files**: `crates/cuda/src/oxide_types.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Test BF16 arithmetic in cuda-oxide | `bf16` ops, conversions |
| 2 | Test INT4 bit manipulation | `u32` packing/unpacking, shift ops |
| 3 | Test FP16 scales | `f16` type for INT4 scales |
| 4 | Port `int4_gemm` kernel logic | Full INT4 GEMM with dequant in registers |

**Complexity**: L
**Timebox**: 3 hours
**Acceptance**: INT4 GEMM kernel compiles and runs. Output matches nvcc version within bf16 precision.

---

### Commit 4: GDN complex kernel test

**Files**: `crates/cuda/src/oxide_gdn.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Port `gdn_recurrent_step` (simplest GDN) | Test core algorithm in Rust |
| 2 | Port `gdn_mamba2_update` (single-token) | Test SSM update logic |
| 3 | Test 80KB dynamic shared memory | Port `gdn_chunked_gated_delta_prefill` shared memory layout |
| 4 | Verify forward substitution correctness | Compare output against CUDA version |

**Complexity**: XL
**Timebox**: 4 hours
**Acceptance**: GDN kernel compiles. Output matches CUDA version. 80KB shared memory works.

---

### Commit 5: cudarc coexistence test

**Files**: `crates/cuda/src/oxide_coexist.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Test cuda-oxide kernel + cudarc cuBLASLt GEMM | Launch a cuda-oxide kernel, then a cuBLASLt matmul |
| 2 | Test cuda-oxide kernel + cudarc NCCL | Launch a cuda-oxide kernel, then NCCL all-reduce |
| 3 | Verify stream sharing | Can cuda-oxide and cudarc share the same CUDA stream? |
| 4 | Document coexistence limitations | What works, what doesn't |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: cuda-oxide kernels and cudarc ops can run on the same stream. NCCL and cuBLASLt still work.

---

### Commit 6: Migration assessment document

**Files**: `plan/research/cuda-oxide.md`

| # | Task | Detail |
|---|------|--------|
| 1 | Write findings from Commits 1-5 | What works, what doesn't, performance numbers |
| 2 | Classify each kernel: migrate now / migrate later / don't migrate | Based on feature support and risk |
| 3 | Estimate migration effort per tier | Hours/days for Tier 1-4 |
| 4 | Decision recommendation | Migrate now, later, or never |

**Complexity**: S
**Timebox**: 1 hour
**Acceptance**: Document complete with clear recommendation.

---

## Key Design Decisions

### KD1: Exploration first, no implementation commitment

cuda-oxide is v0.1.0 alpha. This phase explores feasibility and produces a migration assessment. No kernels are permanently migrated until the assessment is complete and the team decides to proceed.

### KD2: cudarc coexistence is mandatory

The project uses cudarc for cuBLASLt (GEMM) and NCCL (multi-GPU). These are not provided by cuda-oxide. Any migration strategy must allow cuda-oxide kernels to coexist with cudarc-managed cuBLASLt and NCCL operations.

### KD3: Same nightly already pinned

The project uses `nightly-2026-04-03` with `rust-src`, `rustc-dev`, `llvm-tools` — the exact same toolchain cuda-oxide requires. No toolchain changes are needed.

### KD4: Kernel-by-kernel migration

Kernels are migrated one at a time, starting with the simplest. Each migration is verified against the existing CUDA version before proceeding.

### KD5: Performance gates

Each migrated kernel must match nvcc performance within 5% for Tier 1-2 kernels, 10% for Tier 3-4. If a kernel regresses beyond the threshold, it stays on nvcc.

## Success Criteria

- [ ] `cargo oxide doctor` passes
- [ ] A vector-add kernel compiles and runs via cuda-oxide
- [ ] Shared-memory reduction kernel works with dynamic allocation
- [ ] BF16 and INT4 types work correctly
- [ ] INT4 GEMM kernel produces correct output
- [ ] GDN kernel with 80KB shared memory compiles
- [ ] cuda-oxide kernels coexist with cudarc cuBLASLt and NCCL
- [ ] Migration assessment document complete with recommendation
