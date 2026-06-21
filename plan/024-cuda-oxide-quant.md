# Phase 18: cuda-oxide — Quantization-Generic Kernels + Hardware Portability

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-19
**Blocks**: Multi-format quantization support, multi-hardware portability
**Blocked by**: Nothing
**Rationale**: The project has 25 custom `.cu` kernels. Three are quantization-sensitive (INT4 GEMM, FP8 quantize, paged attention decode). Rewriting these in Rust via cuda-oxide enables: (1) supporting multiple INT4 formats (AutoRound, GGUF, AWQ, GPTQ) with one kernel via trait-based dispatch, and (2) future hardware portability — the Rust kernel source can be retargeted to SPIR-V (rust-gpu, for AMD/Intel) or HIP (amdgcn) by swapping the codegen backend. The GDN kernels (80KB shared memory, complex algorithms) are format-agnostic and can stay on nvcc for now.
---

## Why cuda-oxide for these specific kernels

### Quantization flexibility

The INT4 GEMM kernel (`int4_gemm.cu`) is hardcoded for AutoRound's format:
```c
// Today: (w_int4 - (zero + 1)) * scale  — AutoRound only
int4_val = (packed >> shift) & 0xF;
dequant = (int4_val - (zero_point + 1.0)) * scale;
```

GGUF uses `(w - z) * s` with different packing. AWQ has per-channel scales. GPTQ has different group mapping. Each format currently requires a separate kernel or a runtime branch.

In Rust, a trait-based approach:
```rust
trait Dequantize {
    fn dequant(packed: u32, scale: f16, zero: u32, group_idx: u32) -> f32;
}

struct AutoRound;
impl Dequantize for AutoRound {
    fn dequant(packed: u32, scale: f16, zero: u32, _group_idx: u32) -> f32 {
        let w = ((packed >> shift) & 0xF) as f32;
        let z = ((zero >> zshift) & 0xF) as f32;
        (w - (z + 1.0)) * scale.to_f32()
    }
}

struct Gguf;
impl Dequantize for Gguf {
    fn dequant(packed: u32, scale: f16, zero: u32, _group_idx: u32) -> f32 {
        let w = ((packed >> shift) & 0xF) as f32;
        let z = ((zero >> zshift) & 0xF) as f32;
        (w - z) * scale.to_f32()
    }
}

// One kernel, generic over format
#[cuda_global]
fn int4_gemm<Q: Dequantize>(...) { ... }
```

cuda-oxide supports generics with monomorphization — `int4_gemm::<AutoRound>` and `int4_gemm::<Gguf>` compile to separate PTX with zero runtime overhead.

### Hardware portability

The Rust kernel source is portable across codegen backends:

| Backend | Target | Quant kernel support |
|---------|--------|---------------------|
| `cuda-oxide` | NVIDIA PTX | Yes (v0.2) |
| `rust-gpu` | SPIR-V → Vulkan → NVIDIA, AMD, Intel | Possible (alpha) |
| `amdgcn` | AMD ELF via HIP | Possible (built-in) |

The INT4 GEMM, FP8 quantize, and attention decode kernels are all pure compute (no hardware-specific intrinsics beyond basic math). They could compile to multiple targets.

## Kernel selection: What to rewrite, what to keep on nvcc

### Rewrite in Rust (3 kernels)

| Kernel | Lines | Why rewrite |
|--------|-------|-------------|
| `int4_gemm.cu` | 111 | Quantization-sensitive. Different formats need different dequant. Trait-based dispatch. |
| `fp8_quantize.cu` | ~40 | Format-sensitive (E4M3, E5M2, future NVFP4). Clean trait target. |
| `paged_attention_decode.cu` | 166 | Reads quantized KV cache (BF16, FP8, NVFP4). Dequant inside attention loop. |

### Keep on nvcc (22 kernels)

| Category | Kernels | Why keep |
|----------|---------|----------|
| GDN (8 kernels) | gdn_prefill, gdn_update, gdn_recurrent_step, gdn_mamba2_prefill, gdn_mamba2_update, gdn_gated_delta_prefill, gdn_gated_delta_update, gdn_chunked_gated_delta_prefill | Format-agnostic (BF16 activations). 80KB shared memory. Complex algorithms. cuda-oxide's dynamic shared memory support is unverified. |
| Simple elementwise (8 kernels) | argmax, elementwise, embedding, rope, silu, sampling, kv_cache, conv1d_depthwise | Too small to justify migration. No quantization sensitivity. |
| Shared memory reductions (5 kernels) | rmsnorm, rms_norm_gated, l2norm_bf16, softmax, paged_kv_write, paged_kv_read | Format-agnostic. Medium complexity. Low migration value. |
| INT4 companion | — | Handled by int4_gemm (above) |

## Task Breakdown

### Commit 1: Install cargo-oxide + verify build pipeline

**Files**: `crates/cuda/Cargo.toml`, new test kernel

| # | Task | Detail |
|---|------|--------|
| 1 | Install `cargo-oxide` | `cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide` |
| 2 | Run `cargo oxide doctor` | Verify toolchain, CUDA toolkit, LLVM |
| 3 | Uncomment cuda-oxide deps in workspace + crate Cargo.toml | Enable `cuda-core`, `cuda-host` |
| 4 | Write minimal `#[cuda_global]` vector-add kernel | Verify PTX generation |
| 5 | Build and run on GPU | Verify correctness |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: `cargo oxide build` produces PTX. Vector-add kernel runs correctly on GPU.

---

### Commit 2: `Dequantize` trait + INT4 GEMM in Rust

**Files**: new `crates/cuda/src/kernels/int4_gemm.rs`, `crates/cuda/src/kernels/mod.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Define `Dequantize` trait | `fn dequant(packed: u32, scale: f16, zero: u32, shift: u32) -> f32` |
| 2 | Implement `AutoRound` | `(w - (z + 1.0)) * s` — current behavior |
| 3 | Implement `Gguf` | `(w - z) * s` — GGUF format |
| 4 | Implement `Gptq` | GPTQ format (different group mapping) |
| 5 | Write `#[cuda_global] int4_gemm_kernel<Q: Dequantize>` | Port from `int4_gemm.cu` with generic dequant |
| 6 | Add launch wrapper | `pub fn launch_int4_gemm<Q: Dequantize>(...)` |
| 7 | Unit test: AutoRound output matches CUDA version | Compare against `int4_gemm.cu` output |

**Complexity**: L
**Timebox**: 3 hours
**Acceptance**: Rust INT4 GEMM with `AutoRound` produces bit-identical output vs CUDA version. `Gguf` variant compiles and runs.

---

### Commit 3: FP8 format-generic quantize kernel

**Files**: new `crates/cuda/src/kernels/fp8_quantize.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Define `Fp8Format` trait | `fn quantize(val: f32) -> u8`, `fn dequantize(val: u8) -> f32` |
| 2 | Implement `Fp8E4M3` | Current mode=0 behavior |
| 3 | Implement `Fp8E5M2` | Current mode=1 behavior |
| 4 | Write `#[cuda_global] fp8_quantize_kernel<F: Fp8Format>` | Port from `fp8_quantize.cu` |
| 5 | Unit test: output matches CUDA version | Compare against `fp8_quantize.cu` |

**Complexity**: M
**Timebox**: 1 hour
**Acceptance**: Rust FP8 quantize produces correct output for both E4M3 and E5M2.

---

### Commit 4: Paged attention decode with KV cache format trait

**Files**: new `crates/cuda/src/kernels/paged_attn_decode.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Define `KvCacheFormat` trait | `fn read_k(page_pool: &[u8], offset: usize) -> [f32; HEAD_DIM]` |
| 2 | Implement `KvBf16` | Direct BF16 read (current behavior) |
| 3 | Implement `KvFp8` | FP8 dequantize on read |
| 4 | Write `#[cuda_global] paged_attention_decode_kernel<K: KvCacheFormat>` | Port from `paged_attention_decode.cu` |
| 5 | Unit test: BF16 output matches CUDA version | Compare against `paged_attention_decode.cu` |

**Complexity**: L
**Timebox**: 2 hours
**Acceptance**: Rust paged attention decode with `KvBf16` produces correct output. `KvFp8` compiles.

---

### Commit 5: Host-side integration + launch API

**Files**: `crates/cuda/src/oxide_kernels.rs`, `crates/cuda/src/lib.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Create `oxide_kernels` module | Host-side launch wrappers for the three Rust kernels |
| 2 | Use `#[cuda_module]` to embed PTX | Typed kernel loading |
| 3 | Add `DeviceBuffer` allocation helpers | Wrap cuda-core's memory management |
| 4 | Add launch functions matching existing API signatures | `launch_int4_gemm_auto_round(stream, ...)`, etc. |
| 5 | Integration test: launch from Rust host code | End-to-end: allocate buffers, launch kernel, read results |

**Complexity**: M
**Timebox**: 2 hours
**Acceptance**: Can launch Rust INT4 GEMM from host code via cuda-oxide's API.

---

### Commit 6: A/B performance benchmark

**Files**: new `crates/cuda/benches/oxide_vs_nvcc.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Benchmark INT4 GEMM: Rust vs CUDA | Compare latency for common shapes (M=1, N=5120, K=13888) |
| 2 | Benchmark FP8 quantize: Rust vs CUDA | Compare throughput |
| 3 | Benchmark paged attention: Rust vs CUDA | Compare latency for decode (seq_len=1, context=2048) |
| 4 | Document results | Performance parity assessment |

**Complexity**: S
**Timebox**: 1 hour
**Acceptance**: Benchmark results documented. Performance within 10% of nvcc for all three kernels.

---

### Commit 7: Documentation + migration guide

**Files**: `plan/research/cuda-oxide-quant.md`, `lat.md/lat.md`

| # | Task | Detail |
|---|------|--------|
| 1 | Document quantization trait design | How to add a new quant format |
| 2 | Document hardware portability path | How to target rust-gpu / amdgcn |
| 3 | Update lat.md with findings | CUDA crate section |
| 4 | Run `lat check` | Verify links |

**Complexity**: XS
**Timebox**: 15 min
**Acceptance**: Documentation complete. `lat check` passes.

---

## Key Design Decisions

### KD1: Only rewrite quantization-sensitive kernels

The GDN kernels are format-agnostic (BF16 activations) and use 80KB shared memory — high complexity, low migration value. The simple elementwise kernels are too small to justify. The three quant-sensitive kernels (INT4 GEMM, FP8 quantize, paged attention) give the most value for the least effort.

### KD2: Trait-based dispatch, not enum dispatch

Rust generics with monomorphization produce zero-cost abstractions. `int4_gemm::<AutoRound>` compiles to the same PTX as a hardcoded AutoRound kernel. No runtime branch, no vtable.

### KD3: cuda-oxide for NVIDIA, rust-gpu for future vendors

cuda-oxide is the pragmatic choice today (v0.2, active development, NVIDIA-only). The Rust kernel source is portable — when rust-gpu matures, the same kernel code can be compiled to SPIR-V for AMD/Intel. No code changes needed, just a different codegen backend.

### KD4: Keep cudarc for cuBLASLt + NCCL

cuda-oxide provides `cuda-core` and `cuda-async` for memory management and stream scheduling. But cuBLASLt (GEMM) and NCCL (multi-GPU) are not provided by cuda-oxide. cudarc stays for those operations. The Rust kernels and cudarc ops coexist on the same CUDA stream.

### KD5: Benchmark gate

Each Rust kernel must match nvcc performance within 10%. If a kernel regresses beyond the threshold, it stays on nvcc. The INT4 GEMM is the most performance-critical — it runs in every forward pass.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| cuda-oxide dynamic shared memory unsupported | Medium | High | Paged attention decode uses 7KB shared (OK). INT4 GEMM uses 0 shared. Only if we later port softmax would this matter. |
| Performance regression >10% | Medium | Medium | Benchmark gate. If worse, keep on nvcc. |
| Generic monomorphization doesn't work in cuda-oxide | Low | High | Test in Commit 2. Fallback: use enum dispatch instead of generics. |
| cudarc + cuda-oxide stream incompatibility | Low | High | Test in Commit 5. Fallback: use cuda-oxide's `cuda-core` for all memory ops. |
| rust-gpu never matures for compute | Medium | Low | cuda-oxide alone justifies the rewrite (quantization flexibility). |

## Adding a new quant format (future)

With the trait-based design, adding a new format (e.g., GGUF) is:

```rust
struct Gguf;

impl Dequantize for Gguf {
    fn dequant(packed: u32, scale: f16, zero: u32, shift: u32) -> f32 {
        let w = ((packed >> shift) & 0xF) as f32;
        let z = ((zero >> zshift) & 0xF) as f32;
        (w - z) * scale.to_f32()
    }
}

// That's it. The GEMM kernel is already generic.
// Just call: launch_int4_gemm::<Gguf>(stream, ...);
```

No new CUDA kernel needed. No build system changes. Just a trait impl.

## Success Criteria

- [ ] `cargo oxide build` produces PTX for Rust kernels
- [ ] `int4_gemm` in Rust with `AutoRound` matches CUDA output (bit-exact)
- [ ] `int4_gemm` in Rust with `Gguf` compiles and runs correctly
- [ ] `fp8_quantize` in Rust matches CUDA output for E4M3 and E5M2
- [ ] `paged_attention_decode` in Rust with `KvBf16` matches CUDA output
- [ ] Performance within 10% of nvcc for all three kernels
- [ ] Can launch Rust kernels from host code via cuda-oxide API
- [ ] Documentation: how to add a new quant format (1 trait impl)
- [ ] Documentation: hardware portability path (rust-gpu, amdgcn)
- [ ] `lat check` passes
