# Phase 24: cuda-oxide — End-to-End Inference with Trait-Based Quant Dispatch

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-22
**Blocks**: Multi-format quantization support (GGUF, AWQ, GPTQ)
**Blocked by**: Nothing
**Rationale**: Replace the nvcc-compiled INT4 GEMM kernel with a Rust kernel compiled via cuda-oxide, with `Dequantize` trait-based dispatch for quant formats. This is the primary value proposition of cuda-oxide: one generic kernel monomorphized per format at zero runtime cost. Phase 023 established that all kernel features work (smem, bf16, int4, gdn, 80KB smem, generics via `cargo oxide`). This phase makes it production.
---

## What Changed Since Plan Was Written

Phase 023's exploration corrected a critical misunderstanding: **trait-based generic dispatch IS feasible** in cuda-oxide when built via `cargo oxide` (not via the `RUSTFLAGS` codegen backend shortcut). The `cross_crate_embedded` example with `scale<T>` passes. The E0282 phantom type param issue has a simple workaround (`PhantomData<D>` in kernel args). Const generics still fail at runtime (cuda-oxide bug), but trait generics work.

The revised plan targets the actual blocker: **build system integration** — how to compile Rust kernels via `cargo oxide` and load the resulting PTX via cudarc at runtime, without memory copy overhead.

## Architecture: Rust→PTX→cudarc Pipeline

The key design: **compile Rust kernels to PTX via `cargo oxide`, load PTX via cudarc at runtime.** This eliminates the memory copy overhead of the cuda-oxide runtime coexistence approach, and requires no API changes to the existing host-side launch code.

```
Rust kernel source (crates/cuda-oxide-kernels/src/)
  → cargo oxide build
  → PTX embedded in kernel-lib binary
  → Extract PTX at build time → save as .ptx file
  → cudarc loads .ptx at runtime (cuModuleLoadDataEx)
  → Launch via existing cudarc launch_builder (same as .cubin)
```

The Rust kernel must produce the same PTX entry point name and argument layout as the existing .cu kernel. cudarc can't tell the difference.

### Why not cuda-oxide's own runtime?

The `cuda-oxide` runtime (`CudaContext`, `DeviceBuffer`, `#[cuda_module]` launch API) works but requires copying data between cudarc `CudaSlice` and cuda-oxide `DeviceBuffer` for each kernel call. With ~384 INT4 GEMM calls per forward pass (6 per layer × 64 layers), the copy overhead would be measurable. Loading PTX via cudarc avoids all copies — the Rust kernel operates directly on cudarc-allocated memory.

### Trait dispatch via monomorphized wrappers

Generic `#[kernel]` functions monomorphize to separate PTX entry points. We create named wrappers for each format:

```rust
// Generic inner kernel
fn int4_gemm_inner<Q: Dequantize>(output: *mut bf16, weight: *const u32, ..., _marker: PhantomData<Q>) { ... }

// Monomorphized wrappers with known PTX entry point names
#[kernel]
fn int4_gemm_auto_round(output: *mut bf16, weight: *const u32, ...) {
    int4_gemm_inner::<AutoRound>(output, weight, ..., PhantomData)
}

#[kernel]
fn int4_gemm_gguf(output: *mut bf16, weight: *const u32, ...) {
    int4_gemm_inner::<Gguf>(output, weight, ..., PhantomData)
}
```

cudarc loads `int4_gemm_auto_round` from the PTX, just like it loads `int4_gemm_kernel` from the .cubin today.

## Scope: What We Rewrite, What We Keep

### Rewrite in Rust via cuda-oxide (3 kernels)

| Kernel | .cu Lines | Why | Priority |
|--------|----------|-----|----------|
| `int4_gemm_kernel` | 111 | Quantization-sensitive. Trait dispatch is the main motivation. | **P0** |
| `infers_fp8_quantize_bf16` / `infers_fp8_dequantize_bf16` | ~40 | Format-sensitive (E4M3, E5M2, future NVFP4). Clean trait target. | P1 |
| `infers_paged_attention_decode_bf16` | 166 | Reads quantized KV cache (BF16, FP8). Dequant inside attention loop. | P2 |

### Keep on nvcc (22 kernels)

GDN kernels (8): format-agnostic, 80KB smem, complex algorithms — no quant dispatch benefit.
Simple elementwise (8): too small, no quant sensitivity.
Shared memory reductions (5): format-agnostic, low migration value.
cuBLASLt: stays via cudarc. NCCL: stays via cudarc.

## Task Breakdown

### Commit 1: Kernel crate + build pipeline

**Files**: new `crates/cuda-oxide-kernels/`, `crates/cuda/build.rs` (updated)

| # | Task | Detail |
|---|------|--------|
| 1 | Create `crates/cuda-oxide-kernels/` as standalone workspace | Same pattern as POCs — own `[workspace]`, not infers member |
| 2 | Configure deps: cuda-core, cuda-device, cuda-host (git), libm | Same as POC |
| 3 | Write minimal `#[kernel]` test kernel (vector-add) | Validate build pipeline |
| 4 | Run `cargo oxide build` from within the crate | Verify PTX generation |
| 5 | Determine how to extract PTX from build output | Check `target/`, intermediate files, or `#[cuda_module]` embedded bytes |
| 6 | Write build script to save PTX as `.ptx` file in `crates/cuda/kernels/compiled/` | Or embed PTX bytes via `include_bytes!` in infers-cuda |
| 7 | Add cudarc `.ptx` loading support | cudarc `CudaModule::from_ptx()` or `cuModuleLoadDataEx` with PTX string |
| 8 | Test: cudarc loads the Rust PTX and launches vector-add kernel | End-to-end: Rust kernel → PTX → cudarc load → GPU execution |

**Complexity**: M
**Timebox**: 3 hours
**Acceptance**: A Rust `#[kernel]` function compiles via `cargo oxide`, the PTX is loaded by cudarc at runtime, and the kernel runs correctly on GPU. No cuda-oxide runtime deps in infers-cuda — only cudarc loads the PTX.

---

### Commit 2: `Dequantize` trait + INT4 GEMM kernel

**Files**: `crates/cuda-oxide-kernels/src/lib.rs`, `crates/cuda-oxide-kernels/src/int4_gemm.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Define `Dequantize` trait | `fn dequant_group(packed: u32, scale_f16_bits: u16, zero_i8: i8, group_idx: u32, col: u32, n: u32, k: u32, group_size: u32, transposed: bool) -> [f32; 8]` — returns 8 dequantized f32 values from one packed u32 |
| 2 | Implement `AutoRound` | `(w_int4 - (zero + 1)) * scale` — matches existing .cu kernel |
| 3 | Implement `Gguf` | `(w - z) * scale` — GGUF format |
| 4 | Write generic `int4_gemm_inner<Q: Dequantize>` | Port from `int4_gemm.cu`, preserving exact argument order and layout |
| 5 | Write monomorphized wrappers: `int4_gemm_auto_round`, `int4_gemm_gguf` | `#[kernel]` functions with `PhantomData<Q>` workaround for E0282 |
| 6 | Verify PTX entry point names match expectations | Check PTX output for function names |
| 7 | Unit test: AutoRound output matches .cu kernel on representative shapes | M=1,N=5120,K=5120 and M=4,N=17408,K=5120 |

**Complexity**: M
**Timebox**: 3 hours
**Acceptance**: Rust INT4 GEMM with `AutoRound` produces output matching the .cu kernel within BF16 precision (FP32 accumulation, BF16 output). `Gguf` variant compiles and produces correctly dequantized values. Both are separate PTX entry points.

---

### Commit 3: Host-side integration — swap .cubin for .ptx

**Files**: `crates/cuda/src/kernels.rs`, `crates/cuda/src/gemm.rs`, `crates/cuda/build.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Add PTX file to `crates/cuda/kernels/compiled/` | From Commit 1's build output |
| 2 | Register `int4_gemm_auto_round` in KernelRegistry | Replace or supplement `int4_gemm_kernel` entry |
| 3 | Update `matmul_int4()` launch to use new kernel name | Or add a separate `matmul_int4_auto_round()` that targets the Rust kernel |
| 4 | Update `gemm_dispatch.rs` to use new launch function | `CachedWeight::Int4` path calls the Rust kernel |
| 5 | Integration test: single-layer forward pass with Rust INT4 GEMM | Compare output against nvcc version |
| 6 | End-to-end test: server produces tokens | `cargo run --release -p infers-server -- ...` with real model |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: The server starts, loads the model, and produces correct tokens with the Rust INT4 GEMM kernel replacing the .cu version. Output matches the nvcc-based version (same tokens for same prompt).

---

### Commit 4: FP8 quantize/dequantize with format trait

**Files**: `crates/cuda-oxide-kernels/src/fp8_quantize.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Define `Fp8Format` trait | `fn quantize(val: f32) -> u8`, `fn dequantize(val: u8) -> f32` |
| 2 | Implement `Fp8E4M3` | Current `infers_fp8_quantize_bf16` mode=0 behavior |
| 3 | Implement `Fp8E5M2` | Current `infers_fp8_dequantize_bf16` mode=1 behavior |
| 4 | Write generic `fp8_quantize_inner<F: Fp8Format>` + wrappers | `fp8_quantize_e4m3`, `fp8_dequantize_e4m3`, `fp8_quantize_e5m2`, `fp8_dequantize_e5m2` |
| 5 | Register in KernelRegistry, update host launch code | Wire into KV cache quantization path |
| 6 | Unit test: output matches .cu kernel for both formats | BF16→FP8→BF16 round-trip |

**Complexity**: S
**Timebox**: 1.5 hours
**Acceptance**: Rust FP8 kernels produce correct output for both E4M3 and E5M2. Loaded via cudarc.

---

### Commit 5: Paged attention decode with KV cache format trait

**Files**: `crates/cuda-oxide-kernels/src/paged_attn_decode.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Define `KvCacheFormat` trait | `fn read_kv(ptr: *const u8, offset: u32) -> f32` — dequantize on read |
| 2 | Implement `KvBf16` | Direct BF16 read (current behavior) |
| 3 | Implement `KvFp8E4M3` | FP8→BF16 dequantize on read |
| 4 | Write generic `paged_attention_decode_inner<K: KvCacheFormat>` | Port from `paged_attention_decode.cu` |
| 5 | Write monomorphized wrappers | `paged_attention_decode_bf16`, `paged_attention_decode_fp8` |
| 6 | Register in KernelRegistry | Replace `infers_paged_attention_decode_bf16` |
| 7 | Unit test: BF16 output matches .cu kernel | Decode with 2048-token context |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: Rust paged attention decode with `KvBf16` produces correct output. `KvFp8` compiles and runs. Both loaded via cudarc.

---

### Commit 6: Latency benchmark

**Files**: `crates/cuda/benches/oxide_vs_nvcc.rs`

| # | Task | Detail |
|---|------|
| 1 | Benchmark INT4 GEMM: Rust vs .cu | M=1, N=5120, K=5120 (decode shape); M=4, N=17408, K=5120 (prefill shape) |
| 2 | Benchmark FP8 quantize: Rust vs .cu | 1M elements BF16→FP8→BF16 |
| 3 | Benchmark paged attention: Rust vs .cu | seq_len=1, context=2048, GQA=6 |
| 4 | Document results | Performance parity assessment |

**Complexity**: S
**Timebox**: 1 hour
**Acceptance**: Benchmark results documented. Each Rust kernel within 10% of nvcc latency. If >10% regression, keep on nvcc and file issue.

---

### Commit 7: Documentation + lat.md update

**Files**: `lat.md/arch.md`, `plan/research/cuda-oxide.md`

| # | Task | Detail |
|---|------|--------|
| 1 | Update lat.md/arch.md: cuda-oxide production section | Kernel crate, build pipeline, trait dispatch pattern, cudarc PTX loading |
| 2 | Document quantization trait design | How to add a new format: one trait impl + one `#[kernel]` wrapper |
| 3 | Update plan/research/cuda-oxide.md | Migration complete for 3 kernels, 22 remain on nvcc |
| 4 | Run `lat check` | All links pass |

**Complexity**: XS
**Timebox**: 30 min
**Acceptance**: lat.md updated. `lat check` passes.

## Key Design Decisions

### KD1: PTX loaded via cudarc, not cuda-oxide runtime

cuda-oxide's own runtime (CudaContext, DeviceBuffer) works but requires memory copies between cudarc and cuda-oxide buffers. With ~384 INT4 GEMM calls per forward pass, this overhead is measurable. Loading the PTX via cudarc's module loading API (cuModuleLoadDataEx) avoids all copies — the Rust kernel operates directly on cudarc-allocated `CudaSlice` buffers, same as .cubin kernels.

### KD2: Monomorphized `#[kernel]` wrappers, not bare generic `#[kernel]`

Generic `#[kernel]` functions monomorphize correctly via `cargo oxide`, but the PTX entry point names are mangled. By creating explicit wrappers (`int4_gemm_auto_round`, `int4_gemm_gguf`), we get predictable PTX entry point names that can be registered in cudarc's KernelRegistry. The inner generic function (`int4_gemm_inner<Q: Dequantize>`) is not a `#[kernel]` — it's a regular `#[inline(always)]` function that gets monomorphized into each wrapper.

### KD3: Only rewrite quantization-sensitive kernels

GDN kernels: format-agnostic, 80KB smem, complex — no quant dispatch benefit. Simple elementwise: too small. The three quant-sensitive kernels give the most value for the least effort. Other 22 kernels stay on nvcc.

### KD4: E0282 workaround via PhantomData

The `Dequantize` trait parameter `Q` doesn't appear in the kernel argument list — Rust can't infer it (E0282). Workaround: add `_marker: PhantomData<Q>` as the first kernel argument. On the host side, pass `PhantomData::<AutoRound>` when launching. This is zero-cost at runtime (PhantomData is ZST).

### KD5: Benchmark gate

Each Rust kernel must match nvcc latency within 10%. If a kernel regresses beyond the threshold, it stays on nvcc. INT4 GEMM is the most performance-critical — it runs in every forward pass, every layer.

### KD6: Standalone kernel crate, not infers workspace member

`cargo oxide build` doesn't support `-p <crate>` from a workspace root. The kernel crate must be a standalone workspace (like the POCs). Build: `cargo oxide build` from `crates/cuda-oxide-kernels/`. The PTX output gets copied or included into infers-cuda.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PTX entry point names don't match expectations | Medium | Medium | Inspect PTX output before writing host code. Use `extern "C"` or named wrappers. |
| cudarc can't load PTX (only .cubin) | Low | High | cudarc supports `CudaModule::from_ptx()`. Fallback: use `cuModuleLoadDataEx` directly via cudarc's sys bindings. |
| PTX JIT compilation slower than .cubin load | Low | Low | PTX JIT happens once at module load time, not per kernel launch. Acceptable for startup. |
| INT4 GEMM PTX quality worse than nvcc | Medium | Medium | Benchmark gate. If >10% slower, keep on nvcc. |
| PhantomData argument disrupts kernel ABI | Low | High | Test in Commit 2. If PTX argument layout doesn't match cudarc expectations, use enum dispatch instead. |
| cargo-oxide build output not extractable as .ptx file | Medium | Medium | Fallback: use `#[cuda_module]` to embed PTX in a test binary, extract bytes at build time from the compiled binary's data section. |

## Success Criteria

- [ ] Rust `#[kernel]` compiles via `cargo oxide build` to PTX
- [ ] PTX loads via cudarc at runtime (no cuda-oxide runtime deps in infers-cuda)
- [ ] `int4_gemm_auto_round` produces output matching .cu kernel (BF16 precision)
- [ ] `int4_gemm_gguf` compiles, loads, and produces correctly dequantized output
- [ ] Server produces correct tokens using Rust INT4 GEMM (end-to-end)
- [ ] `fp8_quantize_e4m3` and `fp8_dequantize_e5m2` produce correct output
- [ ] `paged_attention_decode_bf16` produces correct output
- [ ] All three Rust kernels within 10% of nvcc latency
- [ ] lat.md updated with cuda-oxide production findings
- [ ] `lat check` passes

## Adding a New Quant Format

With the trait-based design, adding a new format (e.g., AWQ) is:

```rust
// 1. Implement the trait (in cuda-oxide-kernels)
struct Awq;
impl Dequantize for Awq {
    fn dequant_group(packed: u32, scale_f16_bits: u16, zero_i8: i8, ...) -> [f32; 8] {
        // AWQ-specific dequantization logic
    }
}

// 2. Add a monomorphized wrapper
#[kernel]
fn int4_gemm_awq(output: *mut bf16, weight: *const u32, ...) {
    int4_gemm_inner::<Awq>(output, weight, ..., PhantomData)
}

// 3. Register in KernelRegistry (in infers-cuda)
registry.register("int4_gemm_awq", kdir("int4_gemm_awq.ptx"));

// 4. Launch from host code
gemm_dispatch::gemm_projection_cached_awq(stream, int4_awq_kernel, ...);
```

No new CUDA kernel needed. No nvcc. No build system changes beyond registering the new PTX entry point. Just a trait impl + a 3-line wrapper.
