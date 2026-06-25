# Phase 045: Engine-Level Optimization — NCCL Pipeline Overlap + Batched GEMV

---
**Status**: ACTIVE — next optimization target (Phase 044 CUDA graphs deferred)
**Last Updated**: 2026-06-25
**Blocks**: None (final optimization layer)
**Blocked by**: None — Phase 044 (CUDA graphs) is deferred due to NCCL incompatibility
**Rationale**: With CUDA graphs blocked by NCCL/stream-capture incompatibility (see Phase 044 assessment), the remaining path from 0.036s to 0.025s is engine-level changes: NCCL pipeline overlap (3-5ms) and batched GEMV (1-2ms). Combined target: 4-7ms savings, closing most of the 11ms gap without requiring graph capture.
---

## Goal

Reduce decode latency from ~0.030s (post CUDA graphs) to ~0.025s (40 tok/s target).

## NCCL Pipeline Overlap

### Current serialization

```
Layer N:
  norm1 → Attn/GDN → AR(attn) → residual → norm2 → MLP → AR(mlp) → residual
                                  ^^^^^^^^                        ^^^^^^^^
                                  Must complete                    Must complete
                                  before MLP                      before next layer
```

### Target: Cross-layer overlap

```
Layer N (compute stream):  norm1 → Attn/GDN → [AR starts on NCCL stream]
Layer N (NCCL stream):                           AR(attn) → record event
Layer N (compute stream):                        wait(event) → residual → norm2 → MLP → [AR starts]
Layer N (NCCL stream):                                                                               AR(mlp) → record event
Layer N+1 (compute stream):                                                                          wait(event) → norm1 → ...
```

This overlaps the NCCL all-reduce latency with the compute of the next operation. The all-reduce of layer N's attention output overlaps with the residual add and norm2 compute.

### Implementation requirements

1. **Non-blocking compute stream**: Currently all operations use the null stream. Need a separate non-blocking CUDA stream for compute kernels, with CUDA events for synchronization.

2. **OxideKernels stream dispatch fix**: The `cc_stream` field is hardcoded to the null stream. All kernel launches go to `&self.cc_stream`. Must make the stream dynamic — either pass it per-launch or create separate `OxideKernels` instances per stream.

3. **Double-buffered hidden state**: The residual add writes to `residual_buf`, then swaps with `hidden`. With overlap, layer N+1's norm1 reads `hidden` while layer N's AR(mlp) may still be writing. Need two hidden state buffers.

4. **CUDA events for cross-stream sync**: `CudaEvent::record(nccl_stream)` after AR, then `compute_stream.wait_event(event)` before reading AR output.

### Estimated savings

NCCL is 20% of decode time (~7ms). With perfect overlap, all 7ms of NCCL latency can be hidden behind compute. Realistically, expect 50-70% overlap = 3.5-5ms savings.

## Batched INT4 GEMV

### Current pattern

For GDN layers, the decode path computes three separate INT4 GEMVs with the same input vector:
- `in_proj_qkv`: 1 × conv_dim × hidden_size
- `a_proj` (BF16 via cuBLASLt): 1 × num_v_heads × hidden_size
- `b_proj` (BF16 via cuBLASLt): 1 × num_v_heads × hidden_size

For FullAttention layers:
- `q_proj`: 1 × q_out_dim × hidden_size
- `k_proj`: 1 × kv_dim × hidden_size
- `v_proj`: 1 × kv_dim × hidden_size

Each is a separate kernel launch (~2µs CPU overhead). With 56 layers × 3 projections = 168 launches/step × 2µs = ~0.3ms. After CUDA graphs this is eliminated, but batching also reduces GPU kernel overhead.

### Target: Batched GEMV

A single kernel that computes all three projections in one launch:
- Input: shared input vector [K]
- Weights: [qweight_q, qweight_k, qweight_v] concatenated
- Output: [out_q, out_k, out_v] concatenated
- Grid: one block per output column across all three projections

### Estimated savings

~168 kernel launches → 56 launches. With CUDA graphs already eliminating launch overhead, this saves GPU-side overhead (smaller gap between kernel end and next start). Estimate: 0.5-1ms.

## Files Modified

| File | Changes |
|---|---|
| `stream.rs` | Non-blocking compute stream + events |
| `oxide_bridge.rs` | Dynamic stream dispatch (per-launch or per-instance) |
| `engine.rs` | Pipeline overlap logic, double-buffered hidden state |
| `int4_kernels.rs` | Batched GEMV kernel |
| `workspace.rs` | Double-buffered hidden state |

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Non-blocking stream races | Data corruption | Careful event placement, extensive correctness testing |
| OxideKernels stream dispatch refactor | Large change, many call sites | Incremental: first add stream param, then switch streams |
| Double-buffer memory | +20KB per GPU | Negligible |
