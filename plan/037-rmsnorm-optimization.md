# Phase 037: RMSNorm Kernel Optimization

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: None
**Blocked by**: Phase 033 (microbench harness)
**Rationale**: `infers_rmsnorm_bf16` is 2.9% of decode time — ~1.25ms wall-clock for 6,741 calls (64 layers × 2 norms × ~53 calls). At 10µs median per call, each call processes hidden_size=5120 BF16 values (10KB). The kernel uses a 256-thread block with serial shared-memory reduction — the tree reduction has 8 sync_threads calls (`s=128→1`) which serialize the warp.
---

## Goal

Reduce rmsnorm from 10µs to ≤6µs per call, saving ~0.5ms/token. This is a small but easy win.

## Current State (lib.rs:266)

```
Block (256, 1, 1), 1 block per row
Phase 1: each thread computes partial sum_sq over hidden elements (stride 256)
Phase 2: tree reduction in shared memory — 8 iterations of s>>=1, each with sync_threads
Phase 3: thread 0 computes inv_rms
Phase 4: each thread applies norm + weight × output
```

**Inefficiencies**:
1. **Tree reduction has 8 sync_threads** — for 256 threads, s = 128, 64, 32, 16, 8, 4, 2, 1. Each sync is ~20 cycles = 160 cycles wasted.
2. **Warp-level reduction wasted** — when s ≤ 32, we still use shared memory + sync_threads instead of warp shuffle (which needs no sync).
3. **Two-pass over data** — Phase 1 (sum_sq) reads x, Phase 4 (norm apply) re-reads x. For hidden_size=5120 (20 elements per thread), this is 2× the DRAM traffic.

## Target State

### Optimization 1: Warp Shuffle Reduction (eliminate sync_threads for s ≤ 32)

When the active thread count drops to ≤32 (one warp), use `warp::shuffle_xor_f32` for reduction instead of shared memory + sync_threads:

```rust
// Tree reduction for s > 32 (shared memory)
let mut s = 128;
while s > 32 {
    if tid < s { smem[tid] += smem[tid + s]; }
    sync_threads();
    s >>= 1;
}
// Warp shuffle for s ≤ 32 (no sync needed)
if tid < 32 {
    let val = smem[tid];
    let val = warp::shuffle_xor_f32(val, 16);
    let val = warp::shuffle_xor_f32(val, 8);
    let val = warp::shuffle_xor_f32(val, 4);
    let val = warp::shuffle_xor_f32(val, 2);
    let val = warp::shuffle_xor_f32(val, 1);
    smem[tid] = val;
}
sync_threads();  // one final sync
```

Saves 5 sync_threads calls (~100 cycles).

### Optimization 2: Single-Pass (Fused Norm + Apply)

For small hidden_size (5120), the entire input fits in L2 cache. The two-pass approach re-reads from L2, which is fast but still wasteful. A single-pass approach would compute the sum_sq in shared memory, then use it for normalization — but we need the global sum_sq before we can normalize any element.

Alternative: use **warp-level RMSNorm** where each warp computes its own partial sum, reduces via shuffle, and applies norm to its chunk — all in one pass. This works if the block is small enough (1 warp = 32 threads × 160 elements/thread = 5120 > hidden_size=5120 — tight but possible).

Actually for hidden_size=5120 with 256 threads, each thread handles 20 elements. The sum_sq reduction needed before applying norm means 2 passes are inherent. Keep 2-pass but optimize the reduction.

### Optimization 3: Unroll the Input Load

For 20 elements per thread (5120/256), unroll the loop to load + compute sum_sq in groups of 4:

```rust
let mut sum_sq = 0.0;
for i in (0..20).step_by(4) {
    let v0 = bf16_to_f32(x[base + i + 0]);
    let v1 = bf16_to_f32(x[base + i + 1]);
    let v2 = bf16_to_f32(x[base + i + 2]);
    let v3 = bf16_to_f32(x[base + i + 3]);
    sum_sq += v0*v0 + v1*v1 + v2*v2 + v3*v3;
}
// Handle remainder (20 % 4 = 0, so no remainder for hidden=5120)
```

## Implementation

### Files

1. **`crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`**: Add `infers_rmsnorm_v2_bf16`
2. **`crates/cuda/src/oxide_bridge.rs`**: Add launch wrapper + KERNEL_NAMES
3. **`crates/backends/native/src/norm.rs`**: Switch to v2 in `rms_norm_into`
4. **`crates/backends/native/src/gdn.rs`**: If gdn uses rms_norm_gated inner function, optimize that too.

## Acceptance Criteria

1. `infers_rmsnorm_v2_bf16` passes correctness (output bit-identical or cosine > 0.99999).
2. Microbench shows ≥25% latency improvement (10µs → ≤7.5µs).
3. Full decode: small improvement visible (≤0.047s/step).
