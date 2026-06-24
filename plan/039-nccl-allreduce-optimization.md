# Phase 039: NCCL All-Reduce Optimization

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: None
**Blocked by**: None (but Phase 033 bench harness helps measure)
**Rationale**: NCCL all-reduce is 17.6% of decode time — 8.75ms wall-clock for 268 kernel launches (avg 65µs, median 21µs). The 2-GPU ring all-reduce is dominated by kernel launch overhead, not data transfer: each all-reduce moves ~10KB (5120 × 2 bytes) but takes 21µs median. The theoretical transfer time for 10KB over NVLink is <1µs.
---

## Goal

Reduce NCCL all-reduce from 8.75ms to ≤5ms per token.

## Current State

```rust
// Per layer, 2 all-reduce calls:
// 1. Attention output (hidden_size × bf16 = 10KB)
// 2. MLP output (hidden_size × bf16 = 10KB)
// Total: 64 layers × 2 = 128 all-reduce calls per token
// Plus 140 warmup calls = 268 total in nsys
```

Each call:
1. `nccl_all_reduce` kernel launched on both GPUs
2. Ring all-reduce: GPU0 → GPU1 → GPU0 (2 hops)
3. Kernel launch overhead: ~5-10µs per call
4. Synchronization: `group_start`/`group_end` wraps all GPUs

**Inefficiencies**:
1. **140 warmup launches** — first 140 NCCL calls are 1.5ms each (JIT compilation / warmup). Only the last 128 are steady-state (~21µs each).
2. **128 separate calls** — each is a separate kernel launch with group_start/group_end overhead.
3. **No batching** — could concatenate attention + MLP outputs for the same layer and do one all-reduce.

## Target State

### Optimization 1: Batched All-Reduce (concatenate attn + mlp per layer)

Instead of 2 all-reduce calls per layer (attn_out + mlp_out), concatenate them into one buffer and do one all-reduce of 2×hidden_size. This halves kernel launches:

```
Old: 128 all-reduce calls × 21µs = 2.7ms
New: 64 all-reduce calls × 25µs = 1.6ms  (slightly more data per call, fewer launches)
```

### Optimization 2: Pre-allocate NCCL workspace

NCCL may allocate internal buffers per call. Pre-allocating a persistent workspace avoids this. Check if cudarc/NCCL supports persistent workspace configuration.

### Optimization 3: NCCL_LAUNCH_MODE=GROUP

Set `NCCL_LAUNCH_MODE=GROUP` so all enqueued operations within a `group_start`/`group_end` are launched as a single kernel. This is already the default behavior but worth verifying.

### Optimization 4: Reduce number of NCCL calls via fused layer compute

If we fuse attention + MLP for a single layer, we only need one all-reduce at the end. This requires restructuring the layer forward function — larger change, may defer.

## Implementation

### Files

1. **`crates/backends/native/src/engine.rs`**: Batch attn + mlp all-reduce into single call per layer
2. **`crates/backends/native/src/sync.rs`**: Add `all_reduce_fused` that handles 2 buffers in one NCCL group

## Acceptance Criteria

1. NCCL time drops to ≤5ms/token (from 8.75ms).
2. INT4 decode ≤ 0.044s/step (from 0.048s).
3. NVFP4 decode ≤ 0.100s/step (from 0.105s).
4. Correctness: model output unchanged.
