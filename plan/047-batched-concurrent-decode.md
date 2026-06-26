# Phase 047: Batched Concurrent Decode

---
**Status**: IN PROGRESS
**Last Updated**: 2026-06-25
**Blocks**: None
**Blocked by**: None (Phase 046 complete — Arc<GpuResources> + DecodeState + decode_spawnable foundation)
**Rationale**: The current INT4 GEMV is memory-bandwidth bound — each weight element is read from VRAM but used for only one token. Batching 2+ tokens from different sequences amortizes weight bandwidth across sequences, directly attacking the 57.9% GEMM cost. The v3_ksplit_sm kernel is M=1 only, but the generic `int4_gemm_auto_round` kernel already supports M>1. cuBLASLt (lm_head) already handles any M. Attention/GDN need per-sequence KV state, but their GEMMs (Q/K/V projections) are also weight-bandwidth bound. Expected: ~0.026s/token for 2 sequences (1.4x speedup), ~0.020s/token for 4 sequences (1.8x speedup).
---

## Goal

Batch multiple sequences' decode tokens into a single M>1 decode forward pass, reducing per-token latency through weight bandwidth amortization.

**Current**: 0.036s/step for M=1 (27.8 tok/s per sequence)
**Target**: ≤0.030s/step for M=2 (33+ tok/s per sequence, 66+ tok/s aggregate)
**Stretch**: ≤0.022s/step for M=4 (45+ tok/s per sequence, 180+ tok/s aggregate)

## Architecture

### Core Idea: Batched Decode

Instead of decoding one token at a time, concatenate `B` sequences' hidden states into an `[M=B, K=hidden_size]` matrix at the start of each decode step. All GEMMs become M=B instead of M=1:

```
Current (M=1, per sequence):
  For each of B sequences:
    embed[token_id] → hidden[1, K]
    For 48 layers:
      norm1 → GEMM Q/K/V (M=1, reads weight[N,K] once) → attention → AR → ...
    lm_head: GEMM[1, vocab] → sample

Batched (M=B):
  Concatenate B sequences' hidden states → hidden[B, K]
  For 48 layers:
    norm1 (batched) → GEMM Q/K/V (M=B, reads weight[N,K] once, produces B outputs)
    → attention (per-seq KV cache, shared GEMM) → AR → ...
  lm_head: GEMM[B, vocab] → per-sequence sample
```

### Weight Bandwidth Analysis

The INT4 GEMV reads `weight[N, K/8]` (packed INT4) + `scales[N, K/gs]` + `zeros[N, K/gs/8]` from VRAM. For the dominant layers (hidden_size=5120):
- Weight: `N * K / 2` bytes (INT4 = 4 bits = 0.5 bytes per element)
- For q_proj: N=per_gpu_head_dim=6144, K=5120 → 15.7 MB
- At 448 GB/s → 35µs per q_proj weight read
- With M=1: 35µs produces 1 token. With M=2: 35µs produces 2 tokens.

Total weight reads per layer per GPU (all projections): ~4 × 15.7 MB = 63 MB → ~140µs at peak bandwidth.
48 layers × 140µs = 6.7ms weight bandwidth per GPU. With M=2, this is 3.4ms/token. With M=4, 1.7ms/token.

The current 20.9ms GEMM time is 3x the bandwidth limit — likely due to non-coalesced access patterns, scale/zero overhead, and occupancy issues. With M=2, even if kernel efficiency stays constant, weight bandwidth is amortized.

### What Needs to Change

| Component | Change | Complexity |
|-----------|--------|------------|
| `decode_batched` (new) | New decode function taking `Vec<DecodeState>` + `Vec<token_id>`, concatenates hidden states, runs batched GEMM | L |
| `gemm_projection_cached` | Already handles M>1 via `int4_gemm_auto_round` — but need to verify the generic kernel works for decode tensors | S (verify only) |
| `rms_norm_into` | Needs M>1 variant (or verify existing handles strided input) | S |
| `attention::decode_forward_paged` | Needs M=B variant: B sets of Q/K/V, B KV cache lookups, but GEMM is batched | L |
| `gdn::decode_forward` | Needs M=B variant: batched GEMM, per-sequence GDN state | L |
| `add_into` (residual add) | Needs M>1 variant | XS |
| `sample` | Per-sequence sampling from `[M, vocab]` logits | S |
| `BatchedDecodeState` (new) | Container for multiple DecodeStates + batched hidden state buffer | S |
| Test | 2-sequence concurrent decode test, verify correct output + measure speedup | S |

### Attention Batched Decode

The attention path is the trickiest. Currently:
1. Q = GEMM(input, q_proj) → [1, per_gpu_head_dim]
2. K = GEMM(input, k_proj) → [1, kv_dim], stored in paged cache
3. V = GEMM(input, v_proj) → [1, kv_dim], stored in paged cache
4. Paged attention: Q[1, heads, head_dim] × K_cache[seq_len, kv_heads, head_dim] → attn_out[1, heads, head_dim]

For M=B:
1. Q = GEMM(input[B, K], q_proj) → [B, per_gpu_head_dim] — **batched GEMM, amortized weight read**
2-3. K/V = GEMM → [B, kv_dim] — **batched GEMM**. Each sequence writes to its own KV cache page.
4. **B separate paged attention kernels** — each sequence has a different KV cache, different seq_len, different block table. Cannot batch the attention kernel itself, but the Q/K/V GEMMs (the expensive part) are batched.

### GDN Batched Decode

GDN has per-sequence recurrent state. The GEMM is batched, but state update is per-sequence:
1. in_proj_qkv = GEMM(input[B, K], qkv_weight) → [B, conv_dim] — **batched GEMM**
2. Conv1d: per-sequence conv state (B separate conv1d applications)
3. a_proj/b_proj: GEMM → [B, num_v_heads] — **batched GEMM**
4. GDN recurrent update: per-sequence state
5. z_gate: GEMM → [B, z_dim] — **batched GEMM**
6. out_proj: GEMM → [B, hidden_size] — **batched GEMM**

The GEMMs (dominant cost) are batched. State operations are per-sequence but cheap.

### NCCL

NCCL all-reduce now reduces [M, hidden_size] instead of [1, hidden_size]. The NCCL time (7.1ms) is mostly latency — the data size increase from 10KB to 20KB is negligible at 48 GB/s NVLink bandwidth. **NCCL cost stays roughly constant, amortized across M tokens.**

### lm_head

cuBLASLt GEMM [M, vocab_size] instead of [1, vocab_size]. The 2.54GB lm_head weight is read once, producing M outputs. **Same weight bandwidth, M times the tokens.**

### Key Design: Concatenated Hidden States

Instead of each DecodeState having its own `hidden_states: CudaSlice<bf16>` of size [1, hidden_size], the batched decode allocates a single `batched_hidden: CudaSlice<bf16>` of size [M, hidden_size]. At each GEMM, the input is this batched buffer. For per-sequence operations (attention KV, GDN state), the code slices the batched buffer by row.

```rust
// Batched hidden: [M, hidden_size] contiguous in memory
// Stride = hidden_size
let batched_hidden = stream.alloc_zeros::<bf16>(M * hidden_size)?;

// Embed each token into its row:
for (i, state) in states.iter_mut().enumerate() {
    let offset = i * hidden_size;
    // embed token_id into batched_hidden[offset..offset+hidden_size]
}

// Batched GEMM: [M, K] × [N, K]^T → [M, N]
gemm_projection_cached(gemm, oxide, stream, cache, &q_proj_name,
    &batched_hidden, &mut batched_q_out, M, n, k, group_size, &mut ps)?;

// Attention: per-sequence, but Q is already computed via batched GEMM
for (i, state) in states.iter_mut().enumerate() {
    let q_row = batched_q_out.slice(i * q_dim..(i+1) * q_dim); // row i
    // paged attention with q_row and state's KV cache
}
```

## Implementation Plan

### Task 1: M-batched INT4 GEMM kernel (v3_ksplit_sm_m) (M)

Write a new CUDA kernel `int4_gemm_v3_ksplit_sm_m` that extends v3_ksplit_sm to handle M>1 by:
- Loading each weight element once from global memory
- Loading M input rows into shared memory (M × group_size bf16 values)
- Computing M accumulators per thread, per weight element
- Writing M output rows

This amortizes weight bandwidth across M sequences. The kernel processes 4 columns per thread (like v3_ksplit_sm), but each weight read contributes to M accumulators.

**Files**: `crates/cuda-oxide-kernels/kernel-lib/src/int4_kernels.rs` (new kernel), `crates/cuda/src/oxide_bridge.rs` (new launch method), cubin rebuild
**Acceptance criteria**:
- Kernel compiles via `cargo oxide build`
- Launch method `launch_int4_gemm_v3_ksplit_sm_m` added to OxideKernels
- Cubin rebuilt and loaded
- Correctness: for M=1, produces identical results to v3_ksplit_sm
- Correctness: for M=2, each output row matches the M=1 result for the corresponding input row
- Performance for M=2: ≤ 1.5× M=1 time (vs 2× for two separate launches)

### Task 2: Batched decode for MLP layers (S)

Write a batched MLP decode path that:
- Takes `[M, hidden_size]` hidden state as input
- Runs batched RMSNorm (already supports M>1 — one block per row)
- Runs batched gate_proj, up_proj GEMMs via the new M-batched kernel
- Runs SiLU+GLU on `[M, sharded_intermediate]` (need M>1 variant of silu_glu)
- Runs batched down_proj GEMM
- All-reduce `[M, hidden_size]` (NCCL handles any size)
- Residual add on `[M, hidden_size]` (already element-wise, works for any size)

**Files**: `crates/backends/native/src/decode.rs` (batched MLP section), `crates/cuda-oxide-kernels/kernel-lib/src/activation_kernels.rs` (batched silu_glu), `crates/cuda/src/oxide_bridge.rs` (batched launch)
**Acceptance criteria**:
- M=2 MLP decode produces correct results (same as M=1 for each row)
- Existing M=1 smoke test still passes

### Task 3: Batched decode for GDN layers (M)

Write a batched GDN decode path:
- Batched in_proj_qkv GEMM: `[M, K] → [M, conv_dim]`
- For each sequence i in 0..M: slice row i from `[M, conv_dim]`, run conv1d + recurrent step + state update using that sequence's GdnState
- Batched in_proj_a, in_proj_b, z_gate GEMMs: `[M, K] → [M, N]`
- Per-sequence RMSNormGated (slice each row, run norm)
- Batched out_proj GEMM: `[M, value_dim] → [M, hidden_size]`

The GEMMs (dominant cost) are batched. Conv1d and recurrent step are per-sequence but cheap.

**Files**: `crates/backends/native/src/gdn.rs` (new `decode_forward_batched`), `crates/backends/native/src/decode.rs` (batched GDN section)
**Acceptance criteria**:
- M=2 GDN decode produces correct results
- Existing M=1 smoke test still passes

### Task 4: Batched decode for Attention layers (M)

Write a batched attention decode path:
- Batched q_proj, k_proj, v_proj GEMMs: `[M, K] → [M, N]`
- For each sequence i: slice Q/K/V rows, run paged attention with that sequence's KV cache + block table
- Per-sequence RoPE application
- Batched o_proj GEMM: `[M, per_gpu_head_dim] → [M, hidden_size]`

**Files**: `crates/backends/native/src/attention/paged_decode.rs` (new `decode_forward_paged_batched`), `crates/backends/native/src/decode.rs` (batched attention section)
**Acceptance criteria**:
- M=2 attention decode produces correct results
- Existing M=1 smoke test still passes

### Task 5: Batched decode top-level + sampling (M)

Write `decode_batched` that:
- Takes `Vec<u32>` of token IDs and `Vec<DecodeState>` 
- Embeds M tokens into `[M, hidden_size]` contiguous buffer
- Runs 48-layer batched forward (dispatching to batched MLP/GDN/attention)
- Final norm + lm_head: `[M, hidden_size] → [M, vocab]`
- Per-sequence sampling from `[M, vocab]` logits

**Files**: `crates/backends/native/src/decode.rs` (new `decode_batched` function)
**Acceptance criteria**:
- End-to-end M=2 decode: both sequences produce correct output
- Existing M=1 smoke test still passes

### Task 6: Test + Benchmark (S)

Integration test: run 2 prompts concurrently via batched decode, verify correct output for both, measure per-token latency.

**Files**: `crates/backends/native/tests/batched_decode_test.rs`
**Acceptance criteria**:
- Both sequences produce correct output ("Paris")
- Per-token latency ≤ 0.030s (vs 0.036s baseline)
- Throughput ≥ 60 tok/s aggregate (vs 27.8 tok/s baseline)

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| int4_gemm_auto_round (M>1 kernel) slower than expected | No speedup | Profile M=2 GEMM vs 2×M=1 GEMM; if slower, write new M-batched GEMV kernel |
| Attention per-sequence loop dominates | Limited speedup | Attention is only 5% of time — even if not batched, GEMM savings dominate |
| GDN state update serialization | Limited speedup | GDN GEMMs (dominant) are batched; state update is cheap |
| NCCL with larger buffers | Slower NCCL | Data increase is tiny (10KB→20KB); NCCL latency-dominated |
| Memory for batched buffers | VRAM pressure | Each extra sequence needs ~1MB workspace (hidden_size × num_buffers × 2B). 16GB VRAM, weights are ~8GB → plenty of room |

## Profiling Breakdown (current, M=1)

| Component | Time | % | Batched (M=2) estimate |
|-----------|------|---|------------------------|
| INT4 GEMM | 20.9ms | 57.9% | ~12ms (weight read amortized) |
| NCCL | 7.1ms | 19.6% | ~7ms (latency-dominated, same size) |
| cuBLASLt (lm_head) | 3.5ms | 9.7% | ~2ms (weight read amortized) |
| GDN | 2.6ms | 7.3% | ~1.5ms (GEMM amortized, state per-seq) |
| Paged attention | 1.8ms | 5.0% | ~3.6ms (2× per-seq, no amortization) |
| RMSNorm | 0.8ms | 2.3% | ~1.6ms (2× elements) |
| Reduce | 0.5ms | 1.3% | ~0.8ms (2× data) |
| **Total** | **36ms** | | **~28ms → 14ms/token** |

The estimate shows ~22% per-token speedup for M=2 (36→28ms total, 14ms/token vs 36ms/token). Attention and norm have 2× cost because they can't batch the per-sequence parts, but GEMM/lm_head savings dominate.

For M=4: GEMM ~8ms, NCCL ~7ms, lm_head ~1.5ms, GDN ~1ms, attn ~7ms, norm ~3ms, reduce ~1.5ms → ~29ms → 7.3ms/token (4.9x speedup per token? No — NCCL and fixed overheads are constant). More realistic: ~25ms → 6.3ms/token.
