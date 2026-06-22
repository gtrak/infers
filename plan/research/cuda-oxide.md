# cuda-oxide Migration Assessment

**Status**: COMPLETE  
**Date**: 2026-06-21  
**Based on**: Exploration commits 1–5 in `plan/023-cuda-oxide-exploration.md`

## Executive Summary

**Recommendation: DON'T MIGRATE YET** — cuda-oxide's Rust→PTX pipeline works end-to-end for monomorphic kernels, but **trait-based generic dispatch (the primary motivation for this migration) does not work** due to two upstream bugs. The dispatch-based workaround (u32 enum parameter) works but doesn't deliver the unified quant-format design. Wait for: (1) generic kernel PTX embedding fix, (2) const generic symbol resolution fix, (3) native bf16 type.

## What Was Tested

| Category | Kernels Tested | Result |
|----------|---------------|--------|
| Simple (Tier 1) | vec_add (elementwise), bf16_vec_add, bf16x2_fma | ✅ All pass, bit-exact |
| Shared Memory (Tier 2) | rmsnorm (static + dynamic), reduce_benchmark | ✅ All pass |
| Complex (Tier 3) | int4_gemm, int4_unpack | ✅ All pass, bit-exact |
| GDN (Tier 4) | gdn_recurrent_step, gdn_mamba2_update | ✅ All pass, bit-exact |
| 80KB Shared Memory | dynamic_smem with cuFuncSetAttribute | ✅ Works (56KB, 80KB, 96KB) |
| cudarc Coexistence | context sharing, sequential ops, raw pointer interop | ✅ All pass |
| **Total** | **13 kernels across 5 commits** | **All pass** |

## Feature Compatibility Matrix

| CUDA Feature | cuda-oxide Support | Notes |
|-------------|-------------------|-------|
| `__global__` kernel | `#[kernel]` | Works |
| `__shared__` static | `SharedArray<T, N>` | Works |
| `__shared__` dynamic | `DynamicSharedArray<T>` | Works, needs `cuFuncSetAttribute` for >48KB |
| `__syncthreads()` | `thread::sync_threads()` | Works |
| `__launch_bounds__` | `#[launch_bounds(N)]` | Works (after bugfix in cuda-oxide) |
| `__nv_bfloat16` | u16 bit manipulation + `f32_to_bf16()` | No native bf16 type; works via u16 packing |
| `__half` (f16) | Rust `f16` primitive | Works with `#![feature(f16)]` |
| `uint32_t` bit ops | `u32` native | Works |
| `expf/logf/sqrtf` | `libm::expf()` etc. | Compiler intercepts → `__nv_*` libdevice |
| `rsqrtf()` | `1.0 / libm::sqrtf(x)` | No direct intrinsic; composition works |
| `sigmoid/softplus/SiLU` | Manual implementation via `libm::expf` | Works, bit-exact vs CUDA |
| `cvt.rn.bf16x2.f32` | `cvt_f32x2_bf16x2()` | Works (packed u32) |
| `fma.rn.bf16x2` | `fma_bf16x2()` | Works (sm_80+) |
| `maxdynamicsharedmemsize` | `cuFuncSetAttribute` via `cuda_core::sys` | Workaround; no cuda-oxide API |
| cuBLASLt coexistence | Same primary CUcontext | Works via raw pointer copy |
| NCCL coexistence | Same primary CUcontext | Not directly tested; same pattern should work |

## Bugs Found in cuda-oxide

1. **Missing `!"kernel"` metadata for `#[launch_bounds]` kernels** — FIXED in local checkout (commit `74ba512` in `../cuda-oxide/`). Without this, launch_bounds kernels compile as `.func` instead of `.entry`, making them invisible to `cuModuleGetFunction`. This needs to be upstreamed.

2. **Unbounded `step_by` fails in PTX translation** — `(1..).step_by(1)` triggers `Step::forward` constant assertion. Use `while` loops instead. Finite-range `step_by` works fine.

3. **Generic kernels: NVVM IR vs PTX payload mismatch** — When a `#[cuda_module]` contains generic kernels, the macro switches to `load_all_ptx_bundles_merged()` which expects PTX payloads. The codegen backend embeds NVVM IR instead. This causes a runtime error: `NoModules` or `"named symbol not found"`. The `cross_crate_embedded` example works because it uses `cargo oxide` which runs the full NVVM→PTX pipeline. The `RUSTFLAGS` codegen backend path skips NVVM linking.

4. **Const generics: symbol resolution fails** — `#[kernel] pub fn foo<const N: i32>(...)` compiles but fails at runtime with `"named symbol not found"`. The monomorphized symbol isn't found by the module loader.

## Critical Finding: Trait-Based Generic Dispatch Does NOT Work

This is the feature that motivated the entire migration. It does not work due to two independent bugs:

### What we wanted

```rust
trait Dequant {
    fn dequant_group(packed: u32, scale: f32, zero: i8) -> [f32; 8];
}

struct Int4Dequant;
impl Dequant for Int4Dequant { /* ... */ }

struct Int8Dequant;
impl Dequant for Int8Dequant { /* ... */ }

#[kernel]
pub fn quant_gemm<D: Dequant>(weights: &[u32], scales: &[u16], ...) {
    let dequantized = D::dequant_group(packed, scale, zero);
    // monomorphizes into quant_gemm::<Int4Dequant>, quant_gemm::<Int8Dequant>
}
```

Then launch as `module.quant_gemm::<Int4Dequant>(...)` for one model config, `module.quant_gemm::<Int8Dequant>(...)` for another. ONE kernel, multiple quant formats.

### Why it fails

**Bug 1: E0282 "type annotations needed"** — `D` is a phantom type parameter. It doesn't appear in any function argument, so Rust can't infer it at the call site. The `cross_crate_embedded` example's `scale<T>` works because `T` appears in `input: &[T]`. Our `D` only dispatches behavior. Workaround: add a `PhantomData<D>` to the kernel args — but that's ugly and still hits Bug 2.

**Bug 2: NoModules runtime error** — Generic kernels cause `#[cuda_module]` to use `load_all_ptx_bundles_merged()` instead of `load_embedded_module()`. The merged loader expects PTX payloads in the binary, but the `RUSTFLAGS` codegen backend path embeds NVVM IR. Result: the monomorphized symbol can't be found at runtime.

**Bug 3: Const generics also fail** — Even `#[kernel] pub fn foo<const N: i32>(...)` fails with `"named symbol not found"` at runtime. Same root cause: symbol resolution for monomorphized variants is broken.

### Workaround that works

```rust
#[kernel]
pub fn quant_gemm_dispatch(weights: &[u32], scales: &[u16], ..., dequant_kind: u32) {
    #[inline(always)]
    fn dequant_int4(packed: u32, scale: f32, zero: i8) -> [f32; 8] { ... }
    #[inline(always)]
    fn dequant_int8(packed: u32, scale: f32, zero: i8) -> [f32; 8] { ... }
    
    let dequantized = if dequant_kind == 0 {
        dequant_int4(packed, scale, zero)
    } else {
        dequant_int8(packed, scale, zero)
    };
}
```

This passes with bit-exact results. The GPU branch predictor handles the single `if` check. The `#[inline(always)]` functions get inlined by the LLVM pipeline. But it's **not** the clean trait-based design — it's a runtime switch inside one kernel.

**What this means**: The dispatch approach works but doesn't deliver the compile-time safety and ergonomics that motivated the migration. You get ONE kernel with a runtime switch instead of N monomorphized kernels with compile-time dispatch. The type system can't enforce "this model config uses INT4, that one uses FP8" — it becomes a runtime parameter that can be set wrong.

## Blockers for Production Migration

### 1. Generic kernel trait dispatch does not work (Critical — defeats the purpose)
The primary motivation for migration was trait-based generic dispatch for quant formats. This does not work due to two cuda-oxide bugs (generic PTX embedding, const generic symbol resolution). The dispatch-based workaround (u32 enum parameter) works but loses the compile-time type safety benefit. Without trait dispatch, we're writing the same per-quant kernels we have now, just in Rust instead of CUDA.

**Impact**: This was the whole point. Without it, there's no strong reason to migrate individual kernels.

### 2. No native bf16 type (Medium severity)
cuda-oxide has no `bf16` first-class type. All bf16 data must be stored as `u16` (bits) and converted to/from `f32` via bit manipulation (`f32::from_bits((u16 as u32) << 16)` and `f32_to_bf16()`). This works correctly but:
- Adds verbosity to every kernel
- Requires custom host-side packing/unpacking
- Makes the code harder to read than the CUDA equivalent

**Impact**: Every kernel in this project uses bf16 I/O. This is the single biggest ergonomics gap.

### 2. Workspace integration friction (Medium severity)
`cargo oxide build` targets the workspace root and builds ALL crates, including those using cudarc. The POC required standalone crates with `[workspace]` declarations. To integrate cuda-oxide into the main `infers-cuda` crate:
- Either: feature-gate the oxide code and use `RUSTFLAGS` for the codegen backend
- Or: keep cuda-oxide kernels in a separate crate with `path` dependencies

**Impact**: Build system complexity. Not a technical blocker but adds CI/CD friction.

### 3. Memory API mismatch with cudarc (Low severity)
cuda-oxide kernels accept `&[T]` / `DisjointSlice<T>`, not raw device pointers. cudarc-allocated `CudaSlice` must be copied to cuda-oxide `DeviceBuffer` before kernel launch. This adds a `cuMemcpyDtoD` per kernel call.

**Impact**: For kernels called per-token (GDN update, recurrent step), the extra memcpy may negate the "no nvcc" benefit. For kernels called per-layer (rmsnorm, softmax), the single copy is negligible.

### 4. Alpha quality API instability (High severity)
cuda-oxide is v0.2.1 alpha. API names changed between the plan's hypothetical examples and the actual API (`#[cuda_global]` → `#[kernel]`, `block_idx()` → `thread::index_1d()`, etc.). Future versions may break again.

**Impact**: Any migration effort could be invalidated by API changes. Wait for stabilization.

## Kernel-by-Kernel Migration Classification

| Kernel | Migration Class | Reason |
|--------|----------------|--------|
| elementwise.cu (add) | **Migrate later** | Trivial, but bf16 u16 packing overhead |
| argmax.cu | **Migrate later** | Simple, same as elementwise |
| embedding.cu | **Migrate later** | Simple gather |
| rope.cu | **Migrate later** | Trig via libm, straightforward |
| silu.cu | **Migrate later** | Simple activation |
| sampling.cu | **Migrate later** | Simple argmax wrapper |
| kv_cache.cu | **Migrate later** | Scattered write |
| fp8_quantize.cu | **Migrate later** | Element-wise quantize |
| rmsnorm.cu | **Migrate later** | Shared memory reduction works |
| rms_norm_gated.cu | **Migrate later** | Same as rmsnorm |
| l2norm_bf16.cu | **Migrate later** | Same pattern |
| softmax.cu | **Migrate later** | 3-phase reduction works |
| conv1d_depthwise.cu | **Migrate later** | Shared memory works |
| paged_kv_write.cu | **Migrate later** | Block-table indexing |
| paged_kv_read.cu | **Migrate later** | Same pattern |
| int4_gemm.cu | **Migrate later** | Bit-exact verified |
| paged_attention_decode.cu | **Migrate later** | 2-pass softmax; >48KB smem needs cuFuncSetAttribute |
| gdn_recurrent_step.cu | **Migrate later** | Bit-exact verified |
| gdn_mamba2_update.cu | **Migrate later** | Bit-exact verified |
| gdn_update.cu | **Migrate later** | Same pattern as recurrent_step |
| gdn_prefill.cu | **Migrate later** | Sequential prefill |
| gdn_gated_delta_update.cu | **Migrate later** | Single-token update |
| gdn_gated_delta_prefill.cu | **Migrate later** | Sequential prefill |
| gdn_mamba2_prefill.cu | **Migrate later** | Sequential prefill |
| gdn_chunked_gated_delta_prefill.cu | **Don't migrate yet** | 80KB smem + forward substitution + WY; too complex for alpha |

## Estimated Migration Effort

| Tier | Kernels | Effort | Risk |
|------|---------|--------|------|
| Tier 1 (Simple) | 8 | 1-2 days | Low |
| Tier 2 (Shared Memory) | 7 | 2-3 days | Low |
| Tier 3 (Complex) | 2 | 2-3 days | Medium |
| Tier 4 (GDN) | 7 | 5-7 days | Medium-High |
| Tier 5 (Chunked GDN) | 1 | 3-5 days | Very High |
| Build system | — | 1-2 days | Medium |
| **Total** | **25** | **14-22 days** | — |

Note: This does not include the time to upstream the `launch_bounds` bugfix, add bf16 type support to cuda-oxide, or handle the cudarc→oxide memory copy overhead.

## Decision

**DON'T MIGRATE YET** — the primary motivation (trait-based quant dispatch) is blocked by two upstream bugs. Migrate when:

1. **Generic kernel PTX embedding is fixed** — `#[cuda_module]` with generic kernels must work via the codegen backend path (not just `cargo oxide`). This is the critical blocker.
2. **Const generic symbol resolution is fixed** — monomorphized `foo::<N>` symbols must be loadable.
3. A native `bf16` type is added (or a `bfloat16` crate is integrated)
4. The `launch_bounds` metadata bugfix is upstreamed
5. `cuFuncSetAttribute` is wrapped in a proper cuda-oxide API

Until #1 and #2 are fixed, the migration doesn't deliver on its core promise. The dispatch-based workaround (u32 enum) works but can be implemented in CUDA C++ with a switch statement too — there's no advantage to doing it in Rust.

**What to do now**:
- Upstream the `launch_bounds` bugfix as a PR to NVlabs/cuda-oxide
- File issues for generic kernel PTX embedding and const generic symbol resolution
- Keep the POC crates as reference code
- Re-evaluate when cuda-oxide releases a version that supports generic `#[kernel]` functions via the `RUSTFLAGS` codegen backend path
