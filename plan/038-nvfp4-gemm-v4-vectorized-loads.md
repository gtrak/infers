# Phase 038: NVFP4 GEMM v4 — Vectorized 128-bit Loads

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: None
**Blocked by**: Phase 033 (microbench harness)
**Rationale**: `nvfp4_gemm_v3_ksplit` is the NVFP4 decode bottleneck at 105ms/token (9.4 tok/s). Same optimization as Phase 034 (INT4 v4) — vectorize loads to reduce load-issue pressure. NVFP4 weights are u8 packed (2 FP4 per byte), so a 128-bit load fetches 16 bytes = 32 FP4 values = 4 groups of 8.
---

## Goal

Write `nvfp4_gemm_v4_ksplit` with 128-bit loads, targeting NVFP4 decode ≤ 0.080s/step (from 0.105s).

## Current State (v3, lib.rs ~1978)

Same structure as INT4 v3 but with FP4 dequant:
- Weight: `weight_packed: &[u8]`, layout `[N, K/2]`. One u32 load = 8 bytes = 16 FP4 values.
- Per-group: loads 4 bytes as u32 (8 nibbles), processes 8 FP4 values.
- Effective scale precomputed per group, bf16 rounding preserved.

## Target State (v4)

Use u128 loads (16 bytes = 32 FP4 = 4 groups in one load). Thread tiling same as INT4 v4: 4 columns per thread.

Note: NVFP4 group_size=16, so 32 FP4 values = 2 groups per u128 load. Need to load 2 scales for each u128.

## Implementation

1. **`crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`**: Add `nvfp4_gemm_v4_ksplit`
2. **`crates/cuda/src/oxide_bridge.rs`**: Add launch wrapper + KERNEL_NAMES
3. **`crates/backends/native/src/gemm_dispatch.rs`**: Switch NVFP4 M=1 to v4

## Acceptance Criteria

1. Correctness: cosine > 0.999 vs v3.
2. NVFP4 decode ≤ 0.080s/step (from 0.105s).
3. Model output correct.
