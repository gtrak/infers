# Phase 4.7: GPU-Resident Weight Cache

---
**Status**: in-progress
**Last Updated**: 2026-06-11
**Rationale**: Plan written but no implementation. This is the next major work item and the primary blocker for performance.
**Actual Deliverables**:
- [x] `GpuWeightCache` struct
- [x] One-time weight upload at `ForwardEngine` construction
- [x] Replace `gemm_projection` call-sites to use cached weights
- [x] Handle INT4 weights caching
- [x] Handle BF16/FP16 weights caching
- [x] Handle per-head weight slices (removed per-head K/V GEMMs, use GPU copies from k_full/v_full)
- [x] Handle attention output gate weights
- [x] Handle embedding table and LM head caching
- [x] RMSNorm weights caching
- [x] GDN SSM parameters caching
- [ ] Memory budget validation
- [ ] Benchmark before/after
---

**Duration:** 1 week  
**Goal:** Eliminate per-GEMM weight upload overhead by caching dequantized weights as GPU-resident buffers, reducing inference latency by 10–50× and enabling the model to actually fit in VRAM.

## Current Problem

The TP=2 forward pass works end-to-end but is **~100× slower than it should be** (~96 seconds for 1 prefill + 10 decode steps). Root cause: every projection GEMM re-uploads weights from CPU DRAM to GPU VRAM, then drops the GPU buffer immediately. No weights are persistent on GPU.

### Evidence

| Observation | Interpretation |
|-------------|---------------|
| Only ~8 GB total VRAM used across 2 GPUs | Weights are NOT resident (should be ~7 GB per GPU for INT4 shards) |
| ~96 seconds for 11 tokens | Each decode step uploads ~7 GB of weights per GPU (64 layers × ~110 MB of projections) |
| `nvidia-smi` shows VRAM spike then drop per layer | `CudaSlice` allocated for GEMM, used, then `Drop` frees it |
| CPU cores at 100% during inference | `bytes_to_bf16` doing per-element parsing on every forward pass |

### Upload Path Per GEMM (Current)

```
WeightRegistry (CPU Bytes) 
  → bytes_to_bf16() → Vec<bf16> on CPU 
    → clone_htod() → CudaSlice<bf16> on GPU 
      → gemm.matmul_bf16() → result 
        → [CudaSlice dropped] → GPU memory freed
```

This repeats for **every projection, every layer, every forward pass**:
- Prefill: ~64 layers × ~8 projections × 2 GPUs = ~1,000 upload+free cycles
- Decode: same count repeated for every generated token

### Upload Path Per GEMM (Target)

```
WeightRegistry (CPU Bytes)
  → [ONCE AT LOAD] dequantize + clone_htod() → GpuWeightCache 
  → [EVERY FORWARD] gemm.matmul_bf16(&cached_weight, ...) → result
```

One-time upload at model load, then zero H2D transfer during inference.

## Deliverables

- [ ] `GpuWeightCache` struct: per-GPU HashMap from weight name → `CudaSlice<bf16>` (or INT4 buffers)
- [ ] One-time weight upload at `ForwardEngine` construction (after sharding)
- [ ] Replace `gemm_projection` call-sites to use cached weights instead of `WeightData`
- [ ] Handle INT4 weights: upload qweight + scales + qzeros once, cache all three
- [ ] Handle BF16/FP16 weights: upload once, cache the `CudaSlice`
- [ ] Handle per-head weight slices: pre-extract and upload per-head slices at load time (or use full-GEMM approach)
- [ ] Handle attention output gate weights: the Q projection is doubled (12288 output), cache accordingly
- [ ] Handle embedding table and LM head: upload once, cache as GPU buffers
- [ ] RMSNorm weights: upload once, cache per-layer
- [ ] GDN SSM parameters: upload once, cache per-layer
- [ ] Memory budget validation: assert weights + KV cache + temps fit in GPU memory
- [ ] Benchmark: measure tokens/sec before and after

## Technical Details

### GpuWeightCache Design

```rust
/// Per-GPU cache of dequantized, GPU-resident weight buffers.
/// Lives as long as ForwardEngine. Dropped on engine shutdown.
pub struct GpuWeightCache {
    /// Main projection weights: "layers.3.self_attn.q_proj" → CudaSlice<bf16>
    pub main_weights: HashMap<String, CudaSlice<bf16>>,
    
    /// INT4 companion weights: "layers.3.self_attn.q_proj" → (qweight, scales, qzeros)
    pub int4_weights: HashMap<String, Int4GpuBuffers>,
    
    /// Per-head weight slices for attention: pre-extracted and uploaded at load time
    pub per_head_weights: HashMap<String, Vec<CudaSlice<bf16>>>, // per-head slices
    
    /// Small weights (norms, biases): "layers.3.input_layernorm" → CudaSlice<bf16>
    pub small_weights: HashMap<String, CudaSlice<bf16>>,
}

pub struct Int4GpuBuffers {
    pub qweight: CudaSlice<u32>,      // packed INT4 weights
    pub scales: CudaSlice<bf16>,      // per-group scales
    pub qzeros: CudaSlice<u32>,       // per-group zero points
}
```

### Lifecycle

1. **Load**: `load_safetensors()` → `WeightRegistry` (CPU `Bytes`)
2. **Shard**: `shard_weights_tp()` → `Vec<WeightRegistry>` (one per GPU, CPU `Bytes`)
3. **Build layers**: `build_main_layers()` → structured `LayerWeights` (CPU `Bytes`)
4. **Cache**: `GpuWeightCache::new(stream, &layer_weights)` → upload all weights to GPU once
5. **Forward**: `gemm_projection_cached(gemm, &cache, "q_proj", input, output, ...)` → zero H2D

### INT4 Weight Caching

For INT4 weights, the cache stores THREE GPU buffers per projection:
- `qweight`: `CudaSlice<u32>` — the packed 4-bit weights
- `scales`: `CudaSlice<bf16>` — per-group scaling factors  
- `qzeros`: `CudaSlice<u32>` — per-group zero points

The INT4 GEMM kernel (`matmul_int4`) already takes these three buffers. We just cache them instead of re-uploading.

```rust
// Current (per-call):
let (qw, sc, qz) = upload_int4_weight(stream, qweight, scales, qzeros)?;
matmul_int4(..., &qw, &sc, &qz, ...);
// qw, sc, qz dropped here

// Target (cached):
let int4 = cache.int4_weights.get("layers.3.self_attn.q_proj").unwrap();
matmul_int4(..., &int4.qweight, &int4.scales, &int4.qzeros, ...);
// buffers reused
```

### BF16 Weight Caching

For BF16 weights, the cache stores a single `CudaSlice<bf16>`:

```rust
// Current (per-call):
let weight_gpu = upload_weight(stream, weight)?;
gemm.matmul_bf16(..., &weight_gpu, ...);
// weight_gpu dropped

// Target (cached):
let weight_gpu = cache.main_weights.get("layers.3.self_attn.q_proj").unwrap();
gemm.matmul_bf16(..., weight_gpu, ...);
```

### Attention: Per-Head Slices vs Full GEMM

The current `forward_paged` does per-head Q/K/V extraction and GEMM. This requires either:

**Option A: Cache per-head slices at load time**
- Pre-extract each head's weight slice on CPU
- Upload each slice as a separate `CudaSlice<bf16>`
- During forward: use the cached slice directly (no extraction, no upload)

**Option B: Switch to full-GEMM (recommended)**
- Do a single GEMM for the entire Q projection: `[seq_len × hidden_size] × [hidden_size × num_heads_per_gpu*head_dim*2]` → `[seq_len × doubled_Q_dim]`
- Then split into Q and gate halves on GPU (via slice or copy)
- Same for K and V (single GEMM per projection, no per-head loop)

**Option B is strongly preferred** because:
- Single GEMM is orders of magnitude faster than 12 small GEMMs
- No per-head weight extraction overhead
- Simpler code path
- Better GPU utilization
- The O-projection already uses this approach (single GEMM)

### GDN: Mamba2 SSM Parameters

GDN layers have SSM parameters that are currently uploaded per-call:
- `a_log`, `dt_bias`, `D` — small vectors, but still uploaded every forward pass
- These should be cached as `CudaSlice<f32>` or `CudaSlice<bf16>`

### Embedding Table + LM Head

- `embed_tokens.weight`: shape `[vocab_size × hidden_size]` = 248320 × 5120 × 2 bytes = **2.5 GB**
- `lm_head.weight`: same shape, same size = **2.5 GB**
- Currently uploaded per-call (prefill + every decode step)
- Cache once, reuse for all forward passes

### RMSNorm Weights

- `input_layernorm.weight`: `[hidden_size]` = 5120 × 2 bytes = 10 KB per layer
- `post_attention_layernorm.weight`: same
- `q_norm.weight`, `k_norm.weight`: `[head_dim]` = 256 × 2 bytes = 512 bytes
- Small but still uploaded every layer, every pass
- Cache all of them at load time

## Memory Budget (Per GPU, TP=2, Qwen3.5-27B)

### Weight Memory

| Component | BF16 Size | INT4 Size | Per GPU (TP=2) |
|-----------|-----------|-----------|----------------|
| Embedding table | 2.5 GB | — | **2.5 GB** (replicated) |
| LM head | 2.5 GB | — | **2.5 GB** (replicated) |
| Q proj (64 layers) | 64 × 6144 × 5120 × 2 = 4.0 GB | ~0.5 GB | **~2.0 GB** (column-parallel) |
| K proj (64 layers) | 64 × 1024 × 5120 × 2 = 0.67 GB | ~0.08 GB | **~0.04 GB** (column-parallel) |
| V proj (64 layers) | 64 × 1024 × 5120 × 2 = 0.67 GB | ~0.08 GB | **~0.04 GB** (column-parallel) |
| O proj (64 layers) | 64 × 5120 × 6144 × 2 = 4.0 GB | ~0.5 GB | **~2.0 GB** (row-parallel) |
| Gate proj (64 layers) | 64 × 17408 × 5120 × 2 = 11.4 GB | ~1.4 GB | **~5.7 GB** (column-parallel) |
| Up proj (64 layers) | 64 × 17408 × 5120 × 2 = 11.4 GB | ~1.4 GB | **~5.7 GB** (column-parallel) |
| Down proj (64 layers) | 64 × 5120 × 17408 × 2 = 11.4 GB | ~1.4 GB | **~5.7 GB** (row-parallel) |
| GDN in_proj_a (48 layers) | 48 × 48 × 5120 × 2 = 23.5 MB | ~3 MB | **~12 MB** (column-parallel) |
| GDN in_proj_b (48 layers) | 48 × 6144 × 5120 × 2 = 3.0 GB | ~0.38 GB | **~1.5 GB** (column-parallel) |
| GDN out_proj (48 layers) | 48 × 5120 × 6144 × 2 = 3.0 GB | ~0.38 GB | **~1.5 GB** (row-parallel) |
| Norm weights (64 layers × 2) | 64 × 2 × 5120 × 2 = 1.3 MB | — | **1.3 MB** |
| Q/K norm (16 full-attn layers) | 16 × 2 × 256 × 2 = 16 KB | — | **16 KB** |
| **Total weights** | **~51.5 GB** | **~6.4 GB** | **~3.2 GB per GPU** |

**Key insight**: The INT4 quantized model is ~6.4 GB total. With TP=2, each GPU holds ~3.2 GB of weights. This easily fits in 16 GB VRAM.

### KV Cache Memory

| Component | Calculation | Per GPU |
|-----------|------------|---------|
| Paged KV (64 layers) | 64 × 512 pages × 16 tokens × 2 × 512 kv_dim × 2 bytes | ~1.0 GB |
| GDN SSM state (48 layers) | 48 × 24 value heads × 128 head_dim × 2 bytes | ~0.6 MB |

### Activation/Temporary Buffers

| Component | Size | Notes |
|-----------|------|-------|
| Hidden states | seq_len × hidden_size × 2 | Prefill: up to 4096 × 5120 × 2 = 40 MB |
| Attention output | seq_len × hidden_size × 2 | 40 MB |
| MLP intermediates | seq_len × sharded_intermediate × 2 × 2 | 2 × 8704 × seq_len × 2 = 68 MB (prefill) |
| Q/K/V buffers | seq_len × per_gpu_head_dim × 2 × 3 | ~120 MB |
| **Total temps (prefill, max seq_len)** | | **~270 MB** |

### Total Per-GPU Memory Budget

| Category | Size |
|----------|------|
| Cached weights | ~3.2 GB |
| KV cache | ~1.0 GB |
| Activation temps | ~0.3 GB |
| CUDA overhead | ~0.5 GB |
| **Total** | **~5.0 GB** |

**Headroom**: 16 GB - 5 GB = **11 GB free** on each GPU. This is comfortable — we could even increase KV cache pages or add MTP draft weights without OOM.

## Implementation Plan

### Phase 4.7.1: `GpuWeightCache` Struct (1 day)

**Scope**: Design and implement the cache structure.

**Tasks**:
1. Create `GpuWeightCache` in `crates/backends/native/src/gpu_cache.rs`
2. Implement `GpuWeightCache::new(stream, &LayerWeights) -> Result<Self>`
3. For each weight in `LayerWeights`:
   - If INT4: call `upload_int4_weight`, store `(qweight, scales, qzeros)`
   - If BF16/FP16: call `upload_weight`, store `CudaSlice<bf16>`
   - If replicated (embedding, norm): upload once, cache
4. Add `get_bf16(&self, name: &str) -> Option<&CudaSlice<bf16>>`
5. Add `get_int4(&self, name: &str) -> Option<&Int4GpuBuffers>`
6. Add `get_small(&self, name: &str) -> Option<&CudaSlice<bf16>>`

**Acceptance criteria**:
- `cargo check -p infers-backend-native` passes
- Cache holds all weights for a single layer without errors

### Phase 4.7.2: Engine Integration (1 day)

**Scope**: Integrate cache into `ForwardEngine` and update `prefill_paged`/`decode_paged`.

**Tasks**:
1. Add `weight_caches: Vec<GpuWeightCache>` to `ForwardEngine` (one per GPU)
2. In `ForwardEngine::new()`, after building layers, iterate over all layers and all GPUs, uploading weights to cache
3. Update `prefill_paged`:
   - Replace `gemm_projection(..., &weights.q_proj, ...)` with cached version
   - Replace all projection calls with cache lookups
   - Keep the same GEMM dimensions and configurations
4. Update `decode_paged` similarly
5. Update `gdn::forward` and `gdn::decode_forward` to use cached weights
6. Update `attention::forward_paged` and `decode_forward_paged` to use cached weights

**Acceptance criteria**:
- `cargo check -p infers-backend-native --tests` passes
- No compile errors from the integration

### Phase 4.7.3: Attention Per-Head → Full GEMM (2 days)

**Scope**: Eliminate per-head Q/K/V extraction by using single GEMMs for full projections.

**Tasks**:
1. In `forward_paged`:
   - Remove per-head Q/K/V GEMMs in Phase 3
   - Replace with single `gemm_projection` for Q (output = per_gpu_head_dim*2 if gate else per_gpu_head_dim)
   - Same for K and V (output = kv_dim)
   - Split Q output into Q and gate halves on GPU
   - Apply Q-norm and K-norm to the full Q/K buffers (not per-head)
   - RoPE: apply to the full Q buffer
   - Attention scores: compute for all heads simultaneously
   - Softmax: compute per-head or all heads
   - O-projection: already uses single GEMM, no change needed
2. In `decode_forward_paged`:
   - Same changes: single Q/K/V GEMMs instead of per-head
3. Remove `extract_head_weight_slice` and `upload_bf16_slice` if no longer needed
4. Remove `weight_to_bf16_wd` if no longer needed (or keep for backward compat)

**Acceptance criteria**:
- Smoke test passes with varied non-zero tokens
- Prefill + 10 decode steps complete without errors

### Phase 4.7.4: Benchmark + Validation (1 day)

**Scope**: Measure before/after performance and validate correctness.

**Tasks**:
1. Run smoke test with timing instrumentation:
   - Measure total time for 1 prefill + 10 decode steps
   - Break down: prefill time, per-decode-step time
2. Compare against baseline (current ~96 seconds for 11 tokens)
3. Use `nvidia-smi` to verify VRAM usage:
   - Should show ~3-5 GB persistent per GPU
   - No per-layer spikes
4. Run unit tests: `cargo test -p infers-model -p infers-backend-native`
5. Run `lat check` — all wiki links and code refs must pass
6. Update `lat.md` with the new GpuWeightCache design

**Target metrics**:
- Total time for 1 prefill + 10 decode: **<2 seconds** (50× improvement)
- Per-decode-step latency: **<100 ms** (from ~9 seconds currently)
- VRAM usage: **3–5 GB per GPU**, stable (no spikes)

## File Changes

```
crates/backends/native/src/
  gpu_cache.rs          [NEW] GpuWeightCache, Int4GpuBuffers
  engine.rs             [MOD] Integrate cache, upload at init
  attention.rs          [MOD] Use cached weights, full GEMM for Q/K/V
  gdn.rs                [MOD] Use cached weights
  gemm_dispatch.rs      [MOD] Add cached variants of gemm_projection
  upload.rs             [MOD] Keep existing functions, add cached wrappers
  lib.rs                [MOD] Export gpu_cache module

crates/backends/native/tests/smoke_test.rs
  [MOD] Add timing eprintln! for benchmark

crates/model/src/
  weights.rs            [NO CHANGE] WeightData stays as Bytes

plan/
  phase-4.7-gpu-weight-cache.md  [THIS FILE]
```

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| INT4 qzeros format varies across models | Medium | High | Verify shape against qweight, assert on mismatch |
| Per-head → full GEMM breaks attention for GQA | Medium | High | Test with real model, compare token output before/after |
| VRAM budget miscalculation (OOM) | Low | High | Add runtime memory check before upload, fail gracefully |
| `CudaSlice` lifetime issues with cache | Low | Medium | Use `Arc<CudaSlice>` or ensure cache outlives engine |
| cuBLASLt INT4 GEMM requires contiguous memory layout | Medium | Medium | Verify cached buffers are contiguous (clone_htod guarantees this) |
| GDN SSM state shape mismatch after caching | Low | Medium | Assert shapes match expected dimensions at load time |

## Dependencies

- **Phase 3 (model loading)**: WeightRegistry with Bytes data — ✅ complete
- **Phase 4 (TP forward)**: Per-GPU streams, NCCL, layer dispatch — ✅ complete
- **Phase 4.5 (attention kernels)**: GQA support in paged attention — ✅ complete
- **Phase 4.6 (paged attention)**: Paged KV read/write — ✅ complete

This phase has **no external dependencies** — it's a pure refactoring/optimization of the existing working code.

## Decisions

### DECISION: Cache BF16, not INT4, for simplicity?

**Rejected.** Caching the raw INT4 buffers (qweight + scales + qzeros) is actually simpler than dequantizing to BF16 at load time:
- INT4 is 1/4 the size of BF16 — 3.2 GB cached vs 12.8 GB
- The INT4 GEMM kernel already works with the three-buffer format
- No need to add a dequantization step at load time
- **Decision**: Cache the raw INT4 GPU buffers (qweight, scales, qzeros)

### DECISION: Use full GEMM for Q/K/V or keep per-head?

**Full GEMM.** The per-head approach is both slower and more complex. Single GEMM for Q, K, V projections is the standard approach in all production inference engines (vLLM, TensorRT-LLM, TGI). The per-head extraction was a workaround for single-GPU memory constraints that doesn't apply with TP=2.
- **Decision**: Use single GEMM for Q, K, V projections

### DECISION: Cache embedding table and LM head?

**Yes.** Both are 2.5 GB each and currently uploaded per-call (every prefill + every decode step). Caching them saves ~5 GB of H2D traffic per decode step.
- **Decision**: Cache embedding table and LM head

### DECISION: Pre-extract per-head slices or extract on GPU?

**Pre-extract at load time.** For the small number of heads per GPU (12 for Q, 2 for K/V), pre-extracting and uploading per-head slices adds negligible memory overhead (~12 × 256 × 5120 × 2 = 31 MB for Q slices per layer) but saves extraction time per forward pass.
- **Decision**: Pre-extract per-head slices at load time, upload to GPU cache

## Testing Strategy

1. **Unit test**: `GpuWeightCache::new` with dummy weights — verify all weights cached
2. **Unit test**: Cache lookup — verify correct buffer returned for each weight name
3. **Unit test**: INT4 cache — verify qweight, scales, qzeros all present
4. **Smoke test**: Real model with timing — measure total time and per-step latency
5. **Memory test**: `nvidia-smi` snapshot before/during/after — verify no per-layer spikes
6. **Correctness test**: Compare token output before/after caching — must be identical
7. **Stress test**: 100 decode steps — verify no OOM, stable VRAM

## Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Smoke test passes | ✅ Tokens varied and non-zero |
| Prefill + 10 decode time | **<2 seconds** (was ~96s) |
| Per-decode-step latency | **<100 ms** |
| VRAM usage per GPU | **3–5 GB**, stable (no spikes) |
| Token output identical to pre-cache | ✅ Same tokens for same input |
| `cargo test` pass rate | 100% (except pre-existing MTP failures) |
| `lat check` | All checks passed |

## Appendix: Why This Matters for the Roadmap

Without weight caching:
- **Cannot hit 20 tok/s single-request target** — per-GEMM upload overhead dominates
- **Cannot scale to longer sequences** — CPU parsing becomes the bottleneck
- **Wastes 11 GB of free VRAM per GPU** — weights live in DRAM instead
- **Precludes continuous batching** — repeated uploads per request multiply overhead

With weight caching:
- **Unlocks the 20 tok/s target** — compute bound instead of PCIe bound
- **Enables Phase 6 (continuous batching)** — shared cached weights across batched requests
- **Enables Phase 7 (MTP)** — cached draft weights reused across verification passes
- **Prepares for FP8/BF16 formats** — larger weight formats still fit in 16 GB with caching

This is the single highest-impact optimization in the entire roadmap.
