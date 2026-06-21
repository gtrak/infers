# cuda-oxide Migration Assessment

**Status**: COMPLETE  
**Date**: 2026-06-21  
**Based on**: Exploration commits 1–5 in `plan/023-cuda-oxide-exploration.md`

## Executive Summary

**Recommendation: MIGRATE LATER** — cuda-oxide's Rust→PTX pipeline works end-to-end and all kernel features are technically feasible, but the alpha quality (v0.2.1), workspace integration friction, and memory API mismatches make production migration premature. Begin a staged migration once cuda-oxide reaches v0.3+ with stabilized APIs.

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

## Blockers for Production Migration

### 1. No native bf16 type (Medium severity)
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

**MIGRATE LATER** when:
1. cuda-oxide reaches v0.3+ with stabilized API
2. A native `bf16` type is added (or a `bfloat16` crate is integrated)
3. The `launch_bounds` metadata bugfix is upstreamed
4. `cuFuncSetAttribute` is wrapped in a proper cuda-oxide API
5. The `libm` interception is documented as stable

Until then, keep the exploration code in `crates/cuda-oxide-poc/` and `crates/cuda-oxide-coexist/` as a reference for future migration. The `cuda-oxide` source checkout at `../cuda-oxide` contains our bugfix and should be upstreamed as a PR.
