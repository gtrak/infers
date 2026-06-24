# Phase 036: GDN Recurrent Step Optimization

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: None
**Blocked by**: Phase 033 (microbench harness)
**Rationale**: `infers_gdn_recurrent_step_bf16` is 6.1% of decode time — 3ms wall-clock for 48 GDN layers, 62µs per call. The kernel runs 3 serial loops over K=128 for each (head, v_dim) element. Each thread does 3×128 = 384 global memory reads of state + 256 reads of Q/K — all serial. The state `[H, K, V]` is f32 and accessed with stride V between consecutive K elements, giving poor coalescing when V=128.
---

## Goal

Reduce `infers_gdn_recurrent_step_bf16` from 62µs/call to ≤35µs/call, saving ~1.3ms per token.

## Current State (lib.rs:2444)

```
1 thread per (head, v_dim) element
Total threads = H × V = 12 × 128 = 1536 (per GPU shard) — ~24 blocks of 64

Per thread:
  Loop 1 (state decay):    for k in 0..128: state[h, k, v] *= decay      ← 128 reads + 128 writes
  Loop 2 (kv_mem):         for k in 0..128: kv_mem += state[h,k,v] * key[k]  ← 128 reads + 128 reads
  Loop 3 (state update):   for k in 0..128: state[h,k,v] += key[k] * delta  ← 128 reads + 128 writes
  Loop 4 (output):         for k in 0..128: y += state[h,k,v] * query[k]    ← 128 reads + 128 reads

Total: 384 state reads + 256 writes + 256 key reads + 128 query reads = 1024 global memory ops per thread
```

**Inefficiencies**:
1. **State read 4 times** — `state[h,k,v]` is read in loops 1, 2, 3, 4. Only loop 1 reads original, loops 2-4 read updated values. Could fuse loops 2+3 (kv_mem compute + state update).
2. **Serial K loop** — 128 iterations with serial FMA dependency on state. No ILP.
3. **Poor coalescing** — thread `tid` maps to `(h, v)` where `v = tid % V`. Adjacent threads access `state[h, k, 0], state[h, k, 1], ...` with stride V between consecutive K — memory access pattern is `state[h*K*V + k*V + v]`. For V=128 that's stride-1 across V — actually coalesced ✓. But the K loop means each thread reads 128 strided elements.
4. **Key/Query re-read** — `key[h, k]` is read in loop 2 AND loop 3. `query[h, k]` only in loop 4.

## Target State

### Optimization 1: Fuse Loops 2+3 (kv_mem + state update)

```
// Fused: compute kv_mem AND update state in one pass
let mut kv_mem = 0.0;
for k in 0..K:
    let s = state[h, k, v]
    let k_val = key[h, k] * k_rcp
    kv_mem += s * k_val
    state[h, k, v] = s * decay + k_val * delta  // combine decay + update
```

This eliminates 128 state reads (loop 3 re-read) and 128 key reads.

### Optimization 2: Register-cache Key

Load key[h, 0..128] into registers once (128 f32s = 512 bytes). Since key is reused in loop 2 (kv_mem) and loop 3 (state update), caching it saves 128 global reads. For K=128, this uses 32 registers per thread (4 f32 per register) — feasible.

### Optimization 3: ILP with 4 Accumulators for kv_mem and Output

```
let mut kv_mem0, kv_mem1, kv_mem2, kv_mem3: f32 = 0.0;
for k in (0..K).step_by(4):
    s0, s1, s2, s3 = state[h, k+0..k+3, v]  // 4 state reads
    k0, k1, k2, k3 = key[h, k+0..k+3]       // from registers
    kv_mem0 += s0 * k0
    kv_mem1 += s1 * k1
    kv_mem2 += s2 * k2
    kv_mem3 += s3 * k3
    // state update fused
    state[h, k+0, v] = s0 * decay + k0 * delta
    state[h, k+1, v] = s1 * decay + k1 * delta
    state[h, k+2, v] = s2 * decay + k2 * delta
    state[h, k+3, v] = s3 * decay + k3 * delta
kv_mem = kv_mem0 + kv_mem1 + kv_mem2 + kv_mem3
```

### Optimization 4: Fuse Output Loop

After the fused kv_mem+state_update loop, immediately compute the output using the updated state. This is already separate but can share the register-cached query.

## Implementation

### Files

1. **`crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`**: Add `infers_gdn_recurrent_step_v2_bf16`
2. **`crates/cuda/src/oxide_bridge.rs`**: Add launch wrapper + KERNEL_NAMES entry
3. **`crates/backends/native/src/gdn.rs`**: Switch decode path to v2

### Note: All GDN Layers

There are 48 GDN layers. At 62µs each = 2.98ms. A 45% reduction → ~1.6ms saved.

## Acceptance Criteria

1. `infers_gdn_recurrent_step_v2_bf16` passes correctness test (cosine > 0.99 vs v1).
2. Microbench shows ≥30% latency improvement (62µs → ≤43µs).
3. Full decode: INT4 decode ≤ 0.046s/step.
4. Model output correct.
