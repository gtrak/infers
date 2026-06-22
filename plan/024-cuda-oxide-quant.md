# Phase 24: cuda-oxide — End-to-End Inference, All Kernels in Rust

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-22
**Blocks**: Multi-format quantization support (GGUF, AWQ, GPTQ)
**Blocked by**: Nothing
**Rationale**: Replace all 22 nvcc-compiled CUDA kernels with Rust kernels compiled via cuda-oxide. Primary motivation: trait-based quant dispatch for INT4 GEMM. Secondary motivation: unify all GPU code in one language — no more nvcc, no more .cu/.cuh files, no more build.rs with nvcc flags, no more CUDA C context-switching. Phase 023 proved all kernel features work (smem, bf16, int4, gdn, 80KB smem, generics, math intrinsics). This phase makes it production.
---

## Architecture: cuda-oxide Runtime + cudarc Coexistence

**Primary path**: Use cuda-oxide's own runtime (`CudaContext`, `DeviceBuffer`, `#[cuda_module]` typed launch) for Rust kernels. cudarc stays for cuBLASLt, NCCL, and host orchestration. Both share the same CUDA primary context (proven in coexistence POC).

**Why not extract PTX and load via cudarc?** It works but buys us almost nothing — the data copies between cudarc `CudaSlice` and cuda-oxide `DeviceBuffer` are negligible (GPU-internal memcpy at ~500 GB/s). For M=1 decode, the total copy overhead per GEMM call is ~0.04μs. The kernel reads ~13MB of weights. The copy is noise. And we might be able to avoid copies entirely by constructing cuda-oxide slice views from cudarc device pointers (same address space, same context).

The cuda-oxide runtime gives us typed launch wrappers via `#[cuda_module]`, compile-time argument checking, and the natural integration path. PTX-via-cudarc is a fallback if profiling ever shows a problem (unlikely).

### Trait dispatch via monomorphized `#[kernel]` wrappers

Generic `#[kernel]` functions monomorphize correctly via `cargo oxide`. We create named wrappers for each quant format with `PhantomData<Q>` to solve E0282:

```rust
// Generic inner kernel
fn int4_gemm_inner<Q: Dequantize>(..., _marker: PhantomData<Q>) { ... }

// Monomorphized wrappers — each becomes a separate PTX entry point
#[kernel]
fn int4_gemm_auto_round(...) {
    int4_gemm_inner::<AutoRound>(..., PhantomData)
}

#[kernel]
fn int4_gemm_gguf(...) {
    int4_gemm_inner::<Gguf>(..., PhantomData)
}
```

## Kernel Migration: All 22 Custom Kernels

We're converting everything. The POCs already validated every feature pattern needed.

### Tier 1: Simple elementwise (7 kernels) — trivially portable

| .cu Kernel | Lines | Rust Pattern | POC Precedent |
|------------|-------|-------------|---------------|
| `argmax.cu` | ~30 | Block reduction, BF16→F32 | reduce_benchmark in POC |
| `elementwise.cu` | ~20 | Element-wise add | vec_add in POC |
| `embedding.cu` | ~25 | Index gather | vec_add pattern (direct indexing) |
| `rope.cu` | ~40 | Trig + rotation | libm math in POC (expf, sqrtf) |
| `silu.cu` | ~30 | SiLU + SwiGLU gating | sigmoid in POC |
| `sampling.cu` | ~25 | Argmax wrapper | argmax pattern |
| `kv_cache.cu` | ~35 | Scattered write by position | direct indexing |

### Tier 2: Shared memory reductions (6 kernels) — POC'd patterns

| .cu Kernel | Lines | Rust Pattern | POC Precedent |
|------------|-------|-------------|---------------|
| `rmsnorm.cu` | 66 | Block reduction + shared mem + rsqrtf | rmsnorm_static_smem + rmsnorm_dynamic_smem in POC |
| `rms_norm_gated.cu` | ~80 | Same with gating | rmsnorm + silu pattern |
| `l2norm_bf16.cu` | ~50 | L2 normalization | rmsnorm pattern (sum squares instead of sum) |
| `softmax.cu` | 98 | 3-phase online softmax, causal mask | tree reduction + shared mem in POC |
| `conv1d_depthwise.cu` | ~60 | Causal depthwise conv | direct indexing + smem |
| `paged_kv_write.cu` / `paged_kv_read.cu` | ~50 each | Block-table gather/scatter | direct indexing |

### Tier 3: Quantization-sensitive (3 kernels) — primary motivation

| .cu Kernel | Lines | Why trait dispatch | POC Precedent |
|------------|-------|-------------------|---------------|
| `int4_gemm.cu` | 111 | `Dequantize` trait: AutoRound, GGUF, AWQ, GPTQ | int4_gemm + quant_gemm_dispatch in POC |
| `fp8_quantize.cu` | ~40 | `Fp8Format` trait: E4M3, E5M2, NVFP4 | bf16 conversion pipeline in POC |
| `paged_attention_decode.cu` | 166 | `KvCacheFormat` trait: BF16, FP8 | paged_kv pattern |

### Tier 4: GDN (6 active kernels) — most complex, but POC'd

| .cu Kernel | Lines | Rust Pattern | POC Precedent |
|------------|-------|-------------|---------------|
| `gdn_recurrent_step.cu` | ~100 | Single-token decode, no smem | gdn_recurrent_step in POC ✅ |
| `gdn_mamba2_update.cu` | ~80 | SSM decode, no smem | gdn_mamba2_update in POC ✅ |
| `gdn_update.cu` | ~60 | Single-token update | recurrent_step pattern |
| `gdn_gated_delta_update.cu` | ~100 | Gated delta decode | recurrent_step pattern |
| `gdn_gated_delta_prefill.cu` | ~200 | Sequential prefill | recurrent_step + token loop |
| `gdn_chunked_gated_delta_prefill.cu` | 405 | 80KB smem, WY representation, forward substitution | dynamic_smem_80kb in POC ✅ |

Note: `gdn_mamba2_prefill.cu` and `gdn_gated_delta_prefill.cu` are loaded but not in the active forward path (superseded by chunked versions). We'll port the active 6 only.

## Task Breakdown

### Commit 1: Kernel crate + build pipeline + first kernel

**Files**: new `crates/cuda-oxide-kernels/`, `crates/cuda/src/lib.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Create `crates/cuda-oxide-kernels/` as standalone workspace | Own `[workspace]`, same pattern as POCs |
| 2 | Configure deps: cuda-core, cuda-device, cuda-host (git), libm | Same as POC |
| 3 | Port `elementwise.cu` → `infers_add_bf16` Rust kernel | Simplest kernel — validates pipeline |
| 4 | Add `#[cuda_module]` with kernel, write host-side launch test | Validates load + launch via cuda-oxide runtime |
| 5 | Verify coexistence: run cuda-oxide kernel + cudarc cuBLASLt in same process | Confirms no regression vs POC |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: Rust `infers_add_bf16` kernel compiles via `cargo oxide`, loads via `#[cuda_module]`, runs on GPU, and produces correct output. cudarc cuBLASLt GEMM still works in same process.

---

### Commit 2: Tier 1 — all simple elementwise kernels

**Files**: `crates/cuda-oxide-kernels/src/`

| # | Task | Detail |
|---|------|--------|
| 1 | Port `argmax.cu` | Block reduction, BF16→F32 |
| 2 | Port `embedding.cu` | Index gather |
| 3 | Port `rope.cu` | Trig + rotation (libm sinf/cosf) |
| 4 | Port `silu.cu` | SiLU + SwiGLU gating |
| 5 | Port `sampling.cu` | Argmax wrapper |
| 6 | Port `kv_cache.cu` | Scattered write by position |
| 7 | Unit test each against CPU reference | Bit-exact or within BF16 precision |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: All 7 simple kernels compile, load, and produce correct output.

---

### Commit 3: Tier 2 — shared memory kernels

**Files**: `crates/cuda-oxide-kernels/src/`

| # | Task | Detail |
|---|------|--------|
| 1 | Port `rmsnorm.cu` | Block reduction + static shared mem (already POC'd) |
| 2 | Port `rms_norm_gated.cu` | RMSNorm + SiLU gating |
| 3 | Port `l2norm_bf16.cu` | L2 normalization |
| 4 | Port `softmax.cu` | 3-phase online softmax, causal mask |
| 5 | Port `conv1d_depthwise.cu` | Causal depthwise conv |
| 6 | Port `paged_kv_write.cu` | Block-table write |
| 7 | Port `paged_kv_read.cu` | Block-table gather |
| 8 | Unit test each against CPU reference | Within BF16 precision |

**Complexity**: M
**Timebox**: 3 hours
**Acceptance**: All 6 shared memory kernels compile, load, and produce correct output. Dynamic shared memory works via cuFuncSetAttribute workaround where needed.

---

### Commit 4: `Dequantize` trait + INT4 GEMM

**Files**: `crates/cuda-oxide-kernels/src/int4_gemm.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Define `Dequantize` trait | Group-level dequant: `fn dequant_group(packed, scale, zero, ...) -> [f32; 8]` |
| 2 | Implement `AutoRound` | `(w_int4 - (zero + 1)) * scale` — matches existing .cu kernel |
| 3 | Implement `Gguf` | `(w - z) * scale` — GGUF format |
| 4 | Write generic `int4_gemm_inner<Q: Dequantize>` | Port from `int4_gemm.cu`, preserving exact layout |
| 5 | Write monomorphized wrappers: `int4_gemm_auto_round`, `int4_gemm_gguf` | `#[kernel]` with `PhantomData<Q>` |
| 6 | Unit test: AutoRound matches .cu kernel | M=1,N=5120,K=5120 (decode) + M=4,N=17408,K=5120 (prefill) |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: Rust INT4 GEMM with `AutoRound` matches .cu kernel output. `Gguf` compiles and runs correctly. Both are separate PTX entry points.

---

### Commit 5: FP8 + paged attention with format traits

**Files**: `crates/cuda-oxide-kernels/src/fp8_quantize.rs`, `crates/cuda-oxide-kernels/src/paged_attn_decode.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Define `Fp8Format` trait | `fn quantize(val: f32) -> u8`, `fn dequantize(val: u8) -> f32` |
| 2 | Implement `Fp8E4M3` and `Fp8E5M2` | Match existing .cu modes |
| 3 | Write fp8 quantize/dequantize with format trait | Monomorphized wrappers per format |
| 4 | Define `KvCacheFormat` trait | `fn read_kv(ptr, offset) -> f32` |
| 5 | Implement `KvBf16` and `KvFp8E4M3` | BF16 direct read + FP8 dequant |
| 6 | Port `paged_attention_decode.cu` with KV format trait | 166-line kernel, 2-pass online softmax |
| 7 | Unit test all | BF16 path matches .cu kernel |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: FP8 quantize/dequantize correct for both formats. Paged attention decode with KvBf16 matches .cu kernel. KvFp8 compiles and runs.

---

### Commit 6: GDN kernels

**Files**: `crates/cuda-oxide-kernels/src/gdn/`

| # | Task | Detail |
|---|------|--------|
| 1 | Port `gdn_recurrent_step.cu` | Already POC'd ✅ — copy from POC, adapt to module structure |
| 2 | Port `gdn_mamba2_update.cu` | Already POC'd ✅ — same |
| 3 | Port `gdn_update.cu` | Simple single-token update |
| 4 | Port `gdn_gated_delta_update.cu` | Gated delta decode |
| 5 | Port `gdn_gated_delta_prefill.cu` | Sequential prefill |
| 6 | Port `gdn_chunked_gated_delta_prefill.cu` | 80KB smem + WY representation — most complex kernel, already POC'd smem pattern |
| 7 | Unit test each against CPU reference | Within 1e-3 (GDN math precision) |

**Complexity**: L
**Timebox**: 4 hours
**Acceptance**: All 6 active GDN kernels compile, load, and produce correct output. 80KB smem works via cuFuncSetAttribute.

---

### Commit 7: Host integration — wire into forward pass

**Files**: `crates/cuda/src/lib.rs`, `crates/backends/native/src/engine.rs`, `crates/backends/native/src/gemm_dispatch.rs`, all forward-pass modules

| # | Task | Detail |
|---|------|--------|
| 1 | Add cuda-oxide-kernels as a dependency of infers-cuda (or infers-backend-native) | The kernel crate provides `#[cuda_module]` module + typed launch API |
| 2 | Replace `LoadedKernelRegistry` kernel launches with cuda-oxide module launches | Where applicable — some kernels may still use cudarc for cuBLASLt interop |
| 3 | Update `gemm_dispatch.rs` to use `int4_gemm_auto_round` from Rust kernel | `CachedWeight::Int4` path calls the Rust kernel |
| 4 | Update engine forward pass to use Rust kernels for all non-cuBLASLt ops | Replace .cubin launches with cuda-oxide launches |
| 5 | End-to-end test: server produces correct tokens | `cargo run --release -p infers-server` with real model |

**Complexity**: M
**Timebox**: 3 hours
**Acceptance**: Server starts, loads model, and produces correct tokens using all-Rust kernels. Output matches the nvcc-based version (same tokens for same prompt).

---

### Commit 8: Remove nvcc build system

**Files**: `crates/cuda/build.rs`, `crates/cuda/kernels/infers/*.cu`, `crates/cuda/kernels/infers/common.cuh`, `crates/cuda/src/kernels.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Remove `build.rs` (nvcc compilation) | No more nvcc dependency |
| 2 | Remove `crates/cuda/kernels/infers/*.cu` and `common.cuh` | All source now in Rust |
| 3 | Remove `crates/cuda/kernels/compiled/*.cubin` | No more compiled cubin files |
| 4 | Remove `LoadedKernelRegistry` | No more .cubin loading at runtime |
| 5 | Remove nvcc-related code from `crates/cuda/src/kernels.rs` | Simplify module |
| 6 | Verify full build + test still works | `cargo build --release && cargo test --release` |

**Complexity**: S
**Timebox**: 1 hour
**Acceptance**: No nvcc, no .cu files, no .cubin files, no build.rs. Build succeeds. Server runs correctly.

---

### Commit 9: Latency benchmark

**Files**: `crates/cuda/benches/oxide_vs_nvcc.rs` or similar

| # | Task | Detail |
|---|------|--------|
| 1 | Benchmark critical kernels: INT4 GEMM, rmsnorm, softmax, gdn_chunked | Compare vs baseline (document numbers, not automated gate) |
| 2 | Benchmark end-to-end decode throughput | tok/s with Rust kernels vs old nvcc numbers |
| 3 | Document results | Include in plan/research/cuda-oxide.md |

**Complexity**: S
**Timebox**: 1 hour
**Acceptance**: Benchmark numbers documented. No requirement to match nvcc exactly — if there's a regression, we note it and decide whether to optimize or accept.

---

### Commit 10: Documentation + lat.md update

**Files**: `lat.md/arch.md`, `plan/research/cuda-oxide.md`

| # | Task | Detail |
|---|------|--------|
| 1 | Update lat.md/arch.md: cuda-oxide production section | Kernel crate, build pipeline, trait dispatch, all-kernels-migrated status |
| 2 | Document quantization trait design | How to add a new format: one trait impl + one `#[kernel]` wrapper |
| 3 | Update plan/research/cuda-oxide.md | Full migration complete. No nvcc dependency. |
| 4 | Run `lat check` | All links pass |

**Complexity**: XS
**Timebox**: 30 min
**Acceptance**: lat.md updated. `lat check` passes.

## Key Design Decisions

### KD1: Use cuda-oxide runtime, not PTX-via-cudarc

The cuda-oxide runtime provides typed launch wrappers, `#[cuda_module]` auto-generation, and the natural integration path. Data copies between cudarc and cuda-oxide buffers are negligible (GPU-internal memcpy at ~500 GB/s). We may even be able to avoid copies by constructing cuda-oxide slice views from cudarc device pointers (same address space). PTX-via-cudarc is a fallback if profiling ever shows a problem.

### KD2: Convert all kernels, not just quant-sensitive ones

Rust is easier to understand, modify, and debug than CUDA C. Having all GPU code in one language eliminates the cognitive overhead of context-switching and the maintenance burden of two build systems. The POCs already validated every feature pattern needed — there are no unknown unknowns.

### KD3: Standalone kernel crate, not infers workspace member

`cargo oxide build` doesn't support `-p <crate>` from a workspace root. The kernel crate must be a standalone workspace (like the POCs). Build: `cargo oxide build` from `crates/cuda-oxide-kernels/`. The compiled binary with embedded PTX is linked into the main workspace as a dependency.

### KD4: Monomorphized `#[kernel]` wrappers for trait dispatch

Generic `#[kernel]` functions monomorphize correctly via `cargo oxide`, but `PhantomData<Q>` is needed to solve E0282 (phantom type param inference). Named wrappers (`int4_gemm_auto_round`, `int4_gemm_gguf`) give us clear, predictable entry points.

### KD5: Keep cudarc for cuBLASLt + NCCL

cuda-oxide doesn't provide cuBLASLt or NCCL bindings. cudarc stays for those operations. Both runtimes coexist on the same CUDA primary context (proven in coexistence POC).

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| cuda-oxide runtime + cudarc stream interop issues | Low | Medium | Coexistence POC already validated. Test early in Commit 1. |
| PTX quality worse than nvcc for complex kernels | Medium | Low | Benchmark and document. LLVM NVPTX is mature; regressions likely minor. |
| 80KB smem in GDN chunked kernel | Low | High | POC already validated via cuFuncSetAttribute workaround. |
| GDN math precision (expf, logf vs fast-math) | Low | Medium | cuda-oxide intercepts libm to libdevice (no fast-math). Same or better precision than our nvcc non-fast-math GDN builds. |
| cargo-oxide build not reproducible | Low | Low | Pin git revision in Cargo.toml. |
| Alpha API breakage in cuda-oxide | Medium | Medium | Pin git revision. The kernel code uses stable cuda-oxide APIs (#[kernel], #[cuda_module], thread indexing, smem). |

## Success Criteria

- [ ] All 22 custom kernels compile and run via cuda-oxide
- [ ] `Dequantize` trait with `AutoRound` and `Gguf` dispatch works
- [ ] `Fp8Format` trait with E4M3 and E5M2 works
- [ ] `KvCacheFormat` trait with BF16 and FP8 works
- [ ] Server produces correct tokens end-to-end with Rust kernels
- [ ] No nvcc, no .cu files, no .cubin files, no build.rs
- [ ] lat.md updated, `lat check` passes

## Adding a New Quant Format

With the trait-based design, adding a new format (e.g., AWQ) requires only:

1. One `Dequantize` trait impl in the kernel crate
2. One 3-line `#[kernel]` wrapper that calls the generic inner function
3. One host-side launch function

No new .cu file. No nvcc. No build system changes. Just Rust.
