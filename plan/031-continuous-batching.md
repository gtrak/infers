# Phase 31: Continuous Batching — Multi-Sequence Decode

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: Phase 32 (CUDA graphs need stable batch shapes for capture)
**Blocked by**: Phase 28 (fused GEMM with M>1 support), Phase 29 (workspace sized for max_batch), Phase 30 (async pipeline handles M>1)
**Rationale**: The current engine processes one sequence at a time (batch=1 decode). This means GPU compute is underutilized for large weight matrices — each GEMM reads 6GB of weights to produce 1×5120 output. llama.cpp is fundamentally single-sequence; vLLM's key advantage is continuous batching where multiple sequences share weight reads. With M=4, the same 6GB of weights produces 4×5120 output, a 4x improvement in arithmetic intensity at zero extra bandwidth cost.
---

## Goal

Process multiple sequences per decode step (continuous batching):
1. Scheduler maintains a pool of active sequences
2. Each decode step processes up to `max_batch` sequences in one forward pass (M > 1 in GEMMs)
3. Paged KV cache supports independent sequences with different context lengths
4. Sampling processes all sequences in parallel (batched argmax)
5. Sequences can join/leave the batch dynamically (no full recompile needed)

## Current State

- **Batch=1 hard-coded**: `sample.rs:337` has `let batch_size_i32 = 1i32;`
- **Single-sequence KV**: block table has 1 row, positions are sequential
- **Paged KV cache exists**: `PagedKvCache`, `PagedKvManager` support multiple pages, but only used for one sequence at a time
- **GDN state per-GPU**: one state array per GPU, no batch dimension

## Target State

### Scheduler

The scheduler (`crates/scheduler/`) already has the skeleton for multi-sequence management:
- `RequestQueue` with priority scheduling
- `Session` lifecycle (prefill, decode, complete, evict)
- `BatchBuilder` constructs batches from active sessions

Extend `BatchBuilder` to group decode-ready sessions into batches of up to `max_batch`:

```rust
pub struct DecodeBatch {
    pub sessions: Vec<SessionId>,
    pub token_ids: Vec<i32>,       // [max_batch] — current token per session
    pub positions: Vec<i32>,        // [max_batch] — position per session
    pub block_tables: Vec<Vec<i32>>, // [max_batch][max_pages_per_seq]
    pub prompt_lengths: Vec<usize>, // [max_batch] — for repetition penalty
    pub token_histories: Vec<Vec<u32>>, // [max_batch][varies] — for penalties
}
```

### Batched Forward Pass

The fused GEMM kernel already handles M > 1 (the M dimension is the batch/sequence dimension). For a batch of 4 sequences with hidden_size=5120:

```
GEMM: output[4, 5120] = input[4, 5120] @ weight[5120, 5120]^T
```

The shared memory tiling from Phase 28 benefits directly: input rows staged in shared memory are reused across 4 output rows instead of 1. Arithmetic intensity increases 4x.

### Paged KV Cache for Multiple Sequences

The paged KV cache already supports logical→physical page mapping via block tables. For continuous batching:

```
GPU memory layout:
  KV cache: [num_physical_pages, page_size, kv_dim, bf16]
  Block tables: [max_batch][max_pages_per_seq] — one row per sequence

Sequence 0: pages [3, 7, 12]     → logical page 0→physical 3, 1→7, 2→12
Sequence 1: pages [1, 5]          → logical page 0→physical 1, 1→5
Sequence 2: pages [8, 9, 10, 11]  → logical page 0→physical 8, etc.
```

Each sequence's block table is independent. The paged attention kernel reads the correct pages per sequence.

### Batched Paged Attention

The existing `infers_paged_attention_decode_bf16` kernel (lib.rs:1220) processes one sequence at a time (one block per (head, sequence)). Extend it to process multiple sequences in one launch:

```
Grid: (num_heads, max_batch, 1)
Block: (256, 1, 1)
```

Each block handles one (head, sequence) pair, reading the correct block table row for that sequence. This is the standard vLLM PagedAttention v2 pattern.

### Batched Sampling

Replace single-token sampling with batched:
```rust
// Argmax over [max_batch, vocab_size] → [max_batch] token IDs
oxide.launch_batched_argmax(stream, &logits, &mut sampled_tokens, max_batch, vocab_size)?;
```

For non-greedy sampling (temperature, top-k):
```rust
// Batched top-k filtering + multinomial sampling
oxide.launch_batched_sample(stream, &logits, &mut sampled_tokens,
    max_batch, vocab_size, temperature, top_k, rng_state)?;
```

### GDN State for Batched Sequences

GDN recurrent state is per-(head, sequence). Extend the state tensor:
- Current: `[num_heads, K, V]` f32 per GPU
- Batched: `[max_batch, num_heads, K, V]` f32 per GPU

The GDN decode kernel adds a batch dimension to the grid:
```
Grid: (num_heads * max_batch, 1, 1)
```

Each block processes one (head, sequence) pair, indexing into the correct state slot.

## Implementation Plan

### Step 1: Extend paged attention kernel for multi-sequence

Modify `infers_paged_attention_decode_bf16` to accept a batch dimension:
- Add `num_sequences` parameter
- Grid becomes `(num_heads, num_sequences, 1)` 
- Block table indexing: `block_table[seq_idx * max_pages_per_seq + logical_page]`

**File**: `crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`

### Step 2: Extend GDN decode kernel for batching

Modify `infers_gdn_gated_delta_decode_bf16`:
- Add batch dimension to state array: `[max_batch, num_heads, K, V]`
- Grid becomes `(num_v_heads * max_batch, 1, 1)`
- State indexing: `state[batch_idx * num_heads * K * V + head_idx * K * V + ...]`

**File**: `crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`

### Step 3: Update Workspace for batch dimension

All workspace buffers grow from `[hidden_size]` to `[max_batch, hidden_size]`:
```rust
pub hidden: CudaSlice<bf16>,  // [max_batch, hidden_size] — was [1, hidden_size]
```

This is already the design from Phase 029 (sized for max_batch at allocation time).

### Step 4: Implement batched sampling

Add `launch_batched_argmax` to oxide_bridge.rs:
```rust
pub fn launch_batched_argmax(
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<bf16>,       // [max_batch, vocab_size]
    output: &mut CudaSlice<i32>,    // [max_batch]
    batch_size: u32,
    vocab_size: u32,
) -> anyhow::Result<()>
```

One kernel launch processes all sequences. Each thread block handles one sequence's argmax.

### Step 5: Wire scheduler to engine

The scheduler's `BatchBuilder` groups decode-ready sessions:

```rust
let batch = scheduler.build_decode_batch(max_batch=4);
let decode_input = DecodeInput {
    token_ids: batch.token_ids,     // [4]
    positions: batch.positions,     // [4]
    block_tables: batch.block_tables, // [4][max_pages]
};
engine.decode_paged_batch(decode_input).await?;
```

### Step 6: Handle dynamic batch size

When sequences complete or new ones join:
- **Sequence completes**: reduce batch size for next step (or leave slot empty with padding)
- **New sequence joins after prefill**: increase batch size
- **Padding**: pad to max_batch with zeros, mask in attention with -inf scores

For CUDA graph compatibility (Phase 32): always pad to `max_batch` and use a validity mask.

## Memory Budget (TP=2, max_batch=4)

| Component | Size per GPU | Notes |
|-----------|-------------|-------|
| INT4 weights | ~6 GB | Compressed, TP=2 sharded |
| Workspace buffers | ~200 MB | 4× sized for max_batch=4 |
| GDN state | ~96 MB | [4, 24 heads, 128, 128] f32 = 4×3MB |
| KV cache (paged) | ~8 GB | 4 sequences × ~2GB each |
| Total | ~14 GB | Fits in 16 GB |

## Verification

```bash
# Test with multiple concurrent prompt requests
# (requires server mode or batch testing harness)

# Correctness: single sequence still works
python3 scripts/compare_hidden_states.py --oracle-dir /tmp/oracle_int4 --infer-dir /tmp/infer_dumps_int4
# Target: cosine ≥ 0.99 (batching must not change single-sequence output)

# Performance: batch throughput
# Measure tokens/sec across 4 concurrent requests
# Target: 4x single-sequence throughput (weight reads amortized)
```

## Files Modified

| File | Change |
|------|--------|
| `crates/cuda-oxide-kernels/kernel-lib/src/lib.rs` | Extend paged attention + GDN decode for batch dimension |
| `crates/cuda/src/oxide_bridge.rs` | Add `launch_batched_argmax` |
| `crates/backends/native/src/engine.rs` | Add `decode_paged_batch` method; handle M>1 in forward |
| `crates/backends/native/src/workspace.rs` | Confirm buffers sized for max_batch |
| `crates/scheduler/src/batch.rs` | Implement decode batch building |
| `crates/scheduler/src/scheduler.rs` | Wire decode batching to engine |
