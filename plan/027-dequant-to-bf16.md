# Phase 27: Dequant-to-BF16 Unified Quantization Path

---
**Status**: PARTIAL (Phases 1-5 done, Phase 6 cleanup pending, Phase 7 testing pending)
**Last Updated**: 2026-06-22
**Blocks**: Running quantized models (INT4, NVFP4) on any hardware via bf16 GEMM
**Blocked by**: Phase 26 (NVFP4 loading)
**Rationale**: Custom per-format GEMM kernels are hard to maintain. Each new quant format (INT4, NVFP4, future GGUF Q6, etc.) requires a full GEMM implementation. Instead: dequantize to bf16 on GPU, then use cuBLAS for all GEMM. Two dequant kernels + one cuBLAS path = maintainable.
---

## Goal

Replace per-format custom GEMM kernels with a generic dequant→bf16→cuBLAS pipeline:

1. **Dequant kernel**: reads packed quantized weights + scales, writes bf16 to a temp buffer (~50 lines per format)
2. **cuBLAS GEMM**: existing `GemmEngine::matmul_bf16`, no changes
3. **NVFP4 on Blackwell**: native cuBLASLt path (best perf, separate concern)

### Why This Approach

- **Maintainability**: new quant format = one dequant kernel, zero GEMM work
- **Correctness**: cuBLAS is battle-tested; custom GEMM kernels are bug-prone
- **Portability**: dequant→bf16 works on any GPU; native cuBLASLt only on Blackwell
- **Precedent**: llama.cpp uses fused dequant+compute but for bandwidth reasons; vLLM dequantizes to fp8/bf16 for non-native formats

### Trade-offs

- **HBM bandwidth**: materializing bf16 weights costs 4× more HBM reads vs fused dequant+GEMM
- **Acceptable for**: prefill (batched GEMM, cuBLAS is fast), testing/validation, non-Blackwell hardware
- **Not optimal for**: decode (batch=1 GEMV) where fused dequant saves bandwidth — future optimization

## Architecture

### Current State

```
model forward pass
  → gemm_projection_cached()
    ├─ BF16:  GemmEngine::matmul_bf16()     ← cuBLASLt
    └─ INT4:  OxideKernels::launch_int4_gemm_auto_round()  ← custom kernel
    └─ NVFP4: (not wired yet)
```

### Target State

```
model forward pass
  → gemm_projection_cached()
    ├─ BF16:  GemmEngine::matmul_bf16()     ← cuBLASLt (unchanged)
    ├─ INT4:  dequant_int4_to_bf16() → GemmEngine::matmul_bf16()
    └─ NVFP4: dequant_nvfp4_to_bf16() → GemmEngine::matmul_bf16()
              (future: native cuBLASLt on Blackwell)
```

### Dequant Kernels

Both kernels follow the same pattern:
- **Grid**: 1D, one thread per output row (N dimension)
- **Block**: 256 threads
- **No shared memory**: each thread iterates over K independently
- **Output**: [N, K] bf16 buffer (temporary, allocated per-GEMM or cached)

#### INT4 AutoRound Dequant

```
Input:  qweight [N, K/8] u32, scales [N, K/gs] f16, qzeros packed u32
Output: bf16 [N, K]

Per thread (one row):
  for each group g in [0, K/gs):
    load scale = f16_to_f32(scales[row, g])
    load zero  = unpack(qzeros, row * num_groups + g)  // 8 zeros per u32
    for each u32 in group:
      unpack 8 × int4 values
      dequant = (val - (zero + 1)) * scale  // AutoRound offset
      write bf16 to output[row, k]
```

#### NVFP4 Dequant

```
Input:  weight_packed [N, K/2] u8, weight_scale [N, K/gs] f8_e4m3, global_scale f32
Output: bf16 [N, K]

Per thread (one row):
  for each group g in [0, K/gs):
    load scale = fp8_e4m3_dequant(weight_scale[row, g])
    for each byte in group:
      hi_nibble = fp4_e2m1_to_f32(byte >> 4)
      lo_nibble = fp4_e2m1_to_f32(byte & 0xF)
      write bf16[hi * scale * global_scale, lo * scale * global_scale]
```

### Buffer Management

The dequant output buffer is temporary. Options:
1. **Per-call allocation**: `CudaSlice::alloc(n * k * sizeof bf16)` — simple, small overhead
2. **Pooled buffer**: reuse a pre-allocated buffer of max expected size — lower overhead
3. **In-place with weight cache**: dequant directly into the GEMM output staging area — most efficient but coupling

Start with option 1 (simple). Optimize later if profiling shows allocation overhead.

## Implementation Phases

### Phase 1: Dequant Kernels ✅

**Status**: DONE — `int4_dequant_to_bf16` and `nvfp4_dequant_to_bf16` kernels written and compile.

### Phase 2: Rust Launch Wrappers ✅

**Status**: DONE — `launch_int4_dequant_to_bf16` and `launch_nvfp4_dequant_to_bf16` in oxide_bridge.rs, KERNEL_NAMES updated.

### Phase 3: GPU Buffer + Upload ✅

**Status**: DONE — `Nvfp4GpuBuffers` struct, `CachedWeight::Nvfp4` variant, NVFP4 upload path in `upload_and_cache`.

### Phase 4: Wire gemm_projection_cached ✅

**Status**: DONE — INT4 branch replaced with dequant→bf16→cuBLAS. NVFP4 branch added. `cargo check --release` passes, 126 unit tests pass.

### Phase 5: Wire NVFP4 Path ✅

**Status**: DONE — NVFP4 dispatch arm in `gemm_projection_cached`, `CachedWeight::Nvfp4` variant.

### Phase 6: Remove Old INT4 GEMM Kernel (TODO)

**Scope**: Delete `int4_gemm_auto_round`, `int4_gemm_gguf`, `int4_gemm_inner`, `Dequantize` trait, and their launch wrappers.

**Files**: `crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`, `crates/cuda/src/oxide_bridge.rs`, `crates/cuda-oxide-kernels/src/main.rs`

**Acceptance Criteria**:
- No references to deleted functions remain
- `cargo check --release` passes
- Test binary updated to use dequant kernels instead

**Complexity**: S
**Timebox**: 15 minutes

---

### Phase 7: Integration Testing (TODO)

**Scope**: Run smoke tests with real models.

**Tests**:
1. INT4 autoround model: `cargo test --release -p infers-backend-native --test smoke_test -- --ignored`
2. NVFP4 model: `cargo test --release -p infers-backend-native --test smoke_test_mmap_only -- --ignored`

**Note**: These require GPU hardware. The mmap upload path for NVFP4 (`upload_mmap_tensor`) is NOT yet wired — only the heap path (`upload_and_cache`) handles NVFP4. The mmap smoke test will fail at upload until that's addressed.

**Complexity**: M
**Timebox**: 30 minutes

---

### Phase 8: Documentation ✅

**Status**: DONE — `lat.md/lat.md` updated with dequant pipeline section. Plan doc created.

## Cross-Phase Dependencies

```
Phase 1 (kernels) ─→ Phase 2 (wrappers) ─→ Phase 3 (buffer) ─→ Phase 4 (wire INT4)
                                                          └→ Phase 5 (wire NVFP4)
Phase 4 + 5 ─→ Phase 6 (cleanup) ─→ Phase 7 (test) ─→ Phase 8 (docs)
```

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| INT4 transposition mismatch between dequant output and cuBLAS layout | Medium | High | Carefully match row-major vs column-major conventions; test with real model |
| bf16 temp buffer OOM for large layers | Low | Medium | Allocate per-GEMM with exact size; layer by layer |
| NVFP4 FP4 E2M1 decode incorrect | Medium | High | Compare dequant output against Python reference impl |
| Performance regression for INT4 decode | High | Low | Expected — this is a maintainability trade-off; optimize with fused GEMV later |

## References

- [[026-nvfp4-support]] — NVFP4 loading pipeline
- [[024-cuda-oxide-quant]] — original cuda-oxide quant exploration
- llama.cpp `dequantize_mul_mat_vec` — fused dequant+GEMV pattern
- vLLM Marlin NVFP4 kernel — fused dequant+GEMM pattern
- cuBLASLt BF16 GEMM — existing `GemmEngine::matmul_bf16`
