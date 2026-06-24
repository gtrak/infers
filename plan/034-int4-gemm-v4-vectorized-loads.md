# Phase 034: INT4 GEMM v4 — Vectorized 128-bit Loads

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: None
**Blocked by**: Phase 033 (microbench harness needed for iteration)
**Rationale**: `int4_gemm_v3_ksplit` is 37% of decode time (18.6ms wall-clock for 800 launches). At 44% of theoretical bandwidth (448 GB/s), it's the biggest single-kernel bottleneck. The v3 kernel uses 32-bit (`u32`) loads — one u32 per 8 INT4 weights. RTX 5060 Ti supports 128-bit loads (`LDG.128` / `int4` in CUDA), which can transfer 4× more data per load instruction, reducing load-issue pressure by 4×. The input vector also uses 16-bit loads and could benefit from 64-bit or 128-bit vectorization.
---

## Goal

Write `int4_gemm_v4_ksplit` that uses 128-bit loads for both weights and input activations, targeting ≥60% of theoretical bandwidth (≤12.3ms per-token, down from 18.6ms).

## Current State (v3)

```rust
// Weight load: 1 u32 = 8 INT4 weights per load instruction
let packed0: u32 = unsafe { *weight.get_unchecked(w_idx0) };

// Input load: 1 u16 = 1 BF16 value per load instruction
let a0 = f32::from_bits((*input.get_unchecked(k0 + 0) as u32) << 16);
let a1 = f32::from_bits((*input.get_unchecked(k0 + 1) as u32) << 16);
// ... 8 separate u16 loads per u32 of weights
```

Each thread issues 9 load instructions per 8-element chunk (1 weight + 8 input). At 40 SMs × 64 threads × 2 groups/split × 16 u32/group = ~80K loads per split — the load-issue unit is saturated, not the DRAM controller.

## Target State (v4)

```rust
// Weight load: 1 u128 = 32 INT4 weights (4 u32s) per load instruction
let packed4: [u32; 4] = unsafe { *(weight_ptr as *const [u32; 4]) };

// Input load: 1 u64 = 4 BF16 values per load, or 1 u128 = 8 BF16 values
let a_vec: [u16; 8] = unsafe { *(input_ptr as *const [u16; 8]) };
```

Each thread issues 2 load instructions per 32-element chunk (1 weight + 1 input) instead of 36 (4×9). The DRAM controller becomes the bottleneck, not the load-issue unit.

## Architecture

### Vectorized Load Pattern

```
Per group (group_size=128, u32s_per_group=16):
  Old v3: 8 iterations × (1 u32 load + 8 u16 loads) = 72 loads
  New v4: 4 iterations × (1 u128 load + 1 u128 load) = 8 loads  (9× fewer load instructions)
```

### Alignment Requirement

128-bit loads require 16-byte alignment. Check:
- **Weights**: `[K/8, N]` layout, `weight[(k/8) * N + col]`. For N=5120, stride = 5120 × 4 bytes = 20480 bytes (16-byte aligned ✓). But `col` varies per thread — `col * 4` must be 16-byte aligned → `col % 4 == 0`. This means we need to process 4 columns per thread (or use a different access pattern).
- **Input**: `[K]` BF16, `input[k]`. For k aligned to 8, offset is `k * 2` bytes. 16-byte alignment requires `k % 8 == 0` — already satisfied since we process 8 BF16s per u32.

### Thread Tiling: 4 Columns Per Thread

Instead of 1 thread = 1 output column, use 1 thread = 4 output columns. This aligns weight loads to 16 bytes (4 × u32 = 16 bytes). Block: (16, 1, 1) = 16 threads × 4 cols = 64 columns per block (same as v3).

```
Thread tid handles columns [tid*4, tid*4+1, tid*4+2, tid*4+3]
Weight load: weight[(k/8) * N + tid*4 .. tid*4+3] → 16 bytes = 1 u128
Input load: input[k .. k+7] → 16 bytes = 1 u128 (8 BF16 values)
```

### Accumulator Strategy

With 4 columns per thread and 4 accumulators per column = 16 accumulators. For K_SPLIT=28, each split has ~2 groups = 32 u32 weights. Process 4 u32s (1 u128) per inner step:
- 8 steps per group (32 u32s / 4 per step)
- Each step: 1 u128 weight load + 1 u128 input load + 32 FMA operations

## Implementation

### Files

1. **`crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`**: Add `int4_gemm_v4_ksplit` kernel
2. **`crates/cuda/src/oxide_bridge.rs`**: Add `launch_int4_gemm_v4_ksplit` wrapper + KERNEL_NAMES entry
3. **`crates/backends/native/src/gemm_dispatch.rs`**: Switch INT4 M=1 path to v4

### Open Questions

- **cuda-oxide `int4` type**: Does cuda-oxide support `int4` (128-bit) loads natively? Check `cuda_device` crate for vector load intrinsics. If not, use `*const [u32; 4]` pointer cast — the compiler should emit `LDG.128` for aligned 16-byte loads.
- **Alignment of GPU allocations**: cudarc `alloc` returns 256-byte aligned pointers. The weight tensor start is aligned, but per-column offsets may not be. Verify alignment at the u128 boundary.
- **Register pressure**: 16 accumulators × 4 bytes = 64 bytes of registers. Plus input/weight registers. RTX 5060 Ti has 256 registers per thread. Should be fine with launch_bounds(16).

## Acceptance Criteria

1. `int4_gemm_v4_ksplit` kernel compiles and passes correctness test (cosine > 0.999 vs v3 output).
2. Microbench (Phase 033 harness) shows ≥40% reduction in per-call latency vs v3.
3. Full decode benchmark: INT4 decode ≤ 0.040s/step (from 0.048s).
4. nsys confirms `int4_gemm_v4_ksplit` is ≤30% of decode time (down from 37%).
5. Model output still produces "Paris" for "The capital of France is".
