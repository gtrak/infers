# Phase 040: Host Dispatch Overhead Elimination

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: None
**Blocked by**: Phase 033 (microbench harness to isolate)
**Rationale**: ~6ms/token is host-side dispatch overhead — the CPU time between kernel launches. With 285+ GEMM calls per token plus attention + GDN + norm kernels, each `gemm_projection_cached` call involves cache lookups, transposition checks, and launch config computation. This is 12% of the 48ms decode budget. CUDA Graphs (Phase 032) would eliminate this entirely, but if we want to keep the kernel-by-kernel approach, we can reduce per-call overhead.
---

## Goal

Reduce host dispatch overhead from ~6ms to ≤2ms per token.

## Current State

Per decode token, the host does:
- 285 `gemm_projection_cached` calls (64 layers × ~4 GEMMs + final + misc)
- Each call: `GpuWeightCache::get(name)` HashMap lookup, transposition check, LaunchConfig compute
- `decode_forward_paged` / `gdn::decode_forward` calls with many args
- NCCL group_start/group_end wrapping

## Target State

### Optimization 1: Pre-compute kernel launch configs

The `gemm_projection_cached` function recomputes `transposed`, `LaunchConfig`, and `K_SPLIT` every call. Pre-compute these once at engine init and store alongside the weight in `GpuWeightCache`:

```rust
pub struct CachedWeight {
    Int4(Int4GpuBuffers {
        qweight, scales, qzeros,
        shape,
        // NEW: pre-computed dispatch metadata
        transposed: u32,
        k_split: u32,
        launch_config: LaunchConfig,
    }),
    ...
}
```

### Optimization 2: Avoid `Arc<CudaStream>` clone per GEMM

`gemm_projection_cached` takes `&Arc<CudaStream>`. Each call to `oxide.launch_int4_gemm_v3_ksplit` clones the Arc. With 285 calls, that's 285 atomic increments. Pass `&CudaStream` (raw reference) instead.

### Optimization 3: Inline the dispatch hot path

For the decode loop, most GEMMs are the same type (INT4 M=1 ksplit). Special-case this path with a direct kernel launch, bypassing the match arm + cache lookup for known-shape layers.

### Optimization 4: Reduce per-layer Python-like dispatch tree

The decode loop has per-layer branching:
```
match config.get_layer_type(layer_idx) {
    GatedDeltaNet => { gdn::decode_forward(...) }
    FullAttention => { attention::decode_forward_paged(...) }
}
```

Pre-compute layer type indices at init, avoid 64 match evaluations.

## Implementation

### Files

1. **`crates/backends/native/src/gpu_cache.rs`**: Add pre-computed dispatch metadata
2. **`crates/backends/native/src/gemm_dispatch.rs`**: Use pre-computed metadata, skip re-evaluation
3. **`crates/backends/native/src/engine.rs`**: Pre-compute layer type array, streamline decode loop

## Acceptance Criteria

1. Host dispatch overhead ≤2ms/token (measured via CPU-side timing around decode loop).
2. INT4 decode ≤ 0.044s/step.
3. Correctness unchanged.
