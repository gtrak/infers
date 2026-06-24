# Phase 035: Paged Attention Decode Optimization

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: None
**Blocked by**: Phase 033 (microbench harness needed for iteration)
**Rationale**: `infers_paged_attention_decode_bf16` is 13.5% of decode time — 6.7ms wall-clock (per-GPU) for 16 attention layers, 417µs per call. The kernel computes Q·K scores and weighted V Accumulation in two separate passes over the KV cache, doubling memory traffic. With head_dim=256 and page_size=16, each KV page is 2 × 16 × 256 × 2 bytes = 16KB. The kernel should be memory-bandwidth bound but the two-pass design wastes 50% of bandwidth re-reading K.
---

## Goal

Reduce `infers_paged_attention_decode_bf16` from 417µs/call to ≤250µs/call, saving ~2.7ms per token (6.7ms → ~4ms).

## Current State

The kernel (lib.rs:2231) has two phases:

```
Phase 1 (score computation):
  for token_pos in 0..num_cached_tokens:
    for d in 0..head_dim:  ← 256 iterations, serial
      dot += Q[d] * K[token_pos, d]
    online softmax update (max + sum)

Phase 2 (V accumulation):
  for token_pos in 0..num_cached_tokens:  ← SECOND pass over KV cache
    for d in 0..head_dim:
      dot += Q[d] * K[token_pos, d]  ← RE-COMPUTES scores!
    weight = exp(dot - global_max) * inv_sum
    out_val += weight * V[token_pos, d]
```

**Bugs/inefficiencies**:
1. **Q·K dot product recomputed in Phase 2** — scores from Phase 1 are discarded, not cached. This doubles all K reads.
2. **Serial dot product** — `for d in 0..head_dim` is a serial 256-iteration loop. No parallelism across the head_dim.
3. **One block per KV head** — only 2 blocks (2 KV heads per GPU). On 40 SMs this is 5% occupancy.

## Target State

### Optimization 1: Cache Scores (eliminate Phase 2 K re-read)

Store scores in shared memory after Phase 1, reuse in Phase 2. For decode with seq_len ≤ 4096, scores = 4096 × 4 bytes = 16KB. Since we already load Q into shared memory, adding a scores buffer is cheap.

```
Phase 1: compute scores[token_pos] for all tokens, store in smem
  Phase 2: read scores from smem, multiply by V — NO K re-read
```

Saves ~50% of KV cache DRAM traffic.

### Optimization 2: Parallel Dot Product

Instead of one thread doing 256 serial iterations for the dot product, use a warp-level reduction. Block (256, 1, 1) with 256 threads:

```
Compute dot: 256 threads × 1 element each → warp shuffle reduction → 1 dot product
Compute all token scores: stride across tokens (256 tokens per stride)
```

This parallelizes the `head_dim=256` dimension across 256 threads instead of serializing it on one thread.

### Optimization 3: Fuse Score + V Accumulation (single pass)

With cached scores, Phase 2 only reads V. We can fuse both into a single pass if we use online softmax:

```
Single pass:
  for token_chunk in 0..num_cached_tokens:
    Load K[token_chunk], compute Q·K score (parallel across head_dim)
    Online softmax: update global max, rescale running sum
    Load V[token_chunk], accumulate weighted V
```

This reads K and V once each — optimal bandwidth utilization.

## Architecture

### Block and Grid Config

Current: 1 block per KV head, 256 threads.
- Grid: (num_kv_heads, 1, 1) = (2, 1, 1) per GPU — terrible occupancy.

Proposed: 1 block per (KV head, token_chunk) pair.
- Grid: (num_kv_heads × num_token_chunks, 1, 1)
- E.g., 512 tokens / 32 tokens per chunk = 16 chunks × 2 KV heads = 32 blocks
- Each block computes partial softmax + V accumulation for its token range
- Atomic/global reduction needed for final attention output — or write partial outputs and reduce.

Alternatively: keep 1 block per KV head but increase to 256 threads and parallelize the dot product. This is simpler and doesn't need cross-block reduction.

### Shared Memory Layout

```
smem[0..256)     = Q vector (256 f32 = 1KB)
smem[256..2816)  = scores for up to 2560 tokens (10KB) — or use dynamic smem
smem[2816..3072] = reduction buffer (256 f32 = 1KB)
Total: ~12KB (fits in 48KB default smem)
```

For longer sequences (>2560 tokens), spill scores to global memory or process in chunks.

## Implementation

### Files

1. **`crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`**: Add `infers_paged_attention_decode_v2_bf16` kernel (keep original for fallback)
2. **`crates/cuda/src/oxide_bridge.rs`**: Add launch wrapper + KERNEL_NAMES entry
3. **`crates/backends/native/src/attention.rs`**: Switch `decode_forward_paged` to use v2 kernel

## Acceptance Criteria

1. `infers_paged_attention_decode_v2_bf16` passes correctness test vs v1 (cosine > 0.99).
2. Microbench shows ≥30% latency improvement per call (417µs → ≤290µs).
3. Full decode: INT4 decode ≤ 0.045s/step (from 0.048s).
4. nsys confirms attention kernel time drops to ≤10% of decode (from 13.5%).
5. Model output still correct.
