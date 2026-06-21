# Phase 11: Model Integration — Load Real Models and Run Inference

---
**Status**: PARTIAL
**Last Updated**: 2026-06-11
**Rationale**: Real model loads, shards, builds layers. BUT: Performance 200× off. No reference comparison against HuggingFace.
**Actual Deliverables**:
- [x] Config loading fix (`text_config` merging)
- [x] Weight loader rewrite (prefix stripping, visual filtering)
- [x] INT4 inference path (native INT4 GEMM wired)
- [~] GDN forward pass adaptation (works but not verified against reference)
- [x] KV cache and attention fixes (head_dim, dimensions)
- [ ] PrismaSCOUT NVFP4 support
- [x] Server configuration integration
- [~] Smoke test and debugging (passes but slow)
- [ ] Performance target (≥1 tok/s)
- [ ] Reference comparison against HuggingFace
---

**Duration:** 3–4 weeks  
**Goal:** Load and run inference with real Qwen3.6-27B models (AutoRound INT4 and PrismaSCOUT NVFP4) on 2× RTX 5060 Ti.

## Problem

Phase 10 wired the server pipeline, but the model loading code has never been tested against a real model. The actual model format differs from what the engine expects in several critical ways:

```
Model format (qwen3.6-27b-autoround-int4)     Engine expectation
══════════════════════════════════════          ════════════════════
Config inside text_config{}                    Flat config.json
model.language_model.layers.{i}.xxx            model.layers.{i}.xxx  
linear_attn.* (GDN layer)                      gdn.*
INT4 qweight/qzeros/scales triplets            Single BF16 weight
head_dim=256, kv_heads=4                       head_dim=128, kv_heads=40
A_log, dt_bias, conv1d extra GDN weights       Simpler GDN weight set
k_norm, q_norm in self_attn                    No k_norm/q_norm
Vision weights present (model.visual.*)        No vision handling
```

## Architecture Gaps

### 1. ModelConfig doesn't handle nested config
The model's `config.json` is a multimodal wrapper with architecture params inside `text_config`. `ModelConfig::load()` silently gives zero values.

**Fix:** Add `text_config` merging in `ModelConfig::load()`.

### 2. Weight prefix mismatch
All weight names have `model.language_model.` prefix. The engine assumes flat names or `model.layers.` prefix.

**Fix:** Add prefix stripping in `load_safetensors()`.

### 3. GDN weight structure is different
The actual model uses a Mamba2-style GDN with `A_log`, `dt_bias`, `conv1d`, `in_proj_qkv`, `in_proj_z`. The engine expects a simpler GDN with `in_proj_a`, `in_proj_b`, `x_proj_weight`, `dt_proj_weight`.

**Fix:** Either (A) adapt the GDN kernel to the actual weight layout, or (B) write an adapter that reshapes weights at load time.

### 4. INT4 quantization everywhere
All linear projections (QKV, O, MLP gate/up/down, in_proj_qkv, in_proj_z, out_proj) are INT4 quantized. Norms, biases, conv1d, A_log, dt_bias remain BF16.

The engine has `matmul_int4()` but the prefill/decode paths use `upload_weight()` which assumes BF16.

**Fix:** Either (A) add an INT4-aware upload/prefill path using the existing `int4_gemm_kernel`, or (B) dequantize INT4→BF16 on CPU at load time (increases memory 4× — infeasible for 27B).

### 5. KV cache dimensions mismatch
`head_dim=256`, `num_kv_heads=4` vs hardcoded `head_dim=128`, `kv_heads=40` in many places.

**Fix:** Use `ModelConfig` values throughout, remove hardcoded dimensions.

### 6. Full attention layers have extra QK-norm
`q_norm.weight` and `k_norm.weight` exist on full-attention layers. The engine doesn't apply QK-norm.

**Fix:** Add optional QK-norm to the attention path.

### 7. Vision weights need filtering
`model.visual.*` weights exist but are unused for LLM-only inference.

**Fix:** Filter out non-language-model tensors during loading.

### 8. MTP layer structure differs
MTP has `self_attn` (no GDN layers), INT4 quantized, and expects `pre_fc_norm_*` weights that the actual model has.

**Fix:** Adjust `build_mtp_weights()` for the actual MTP architecture.

## Deliverables

### 1. [x] Config Loading Fix (3 days)

- Modify `ModelConfig::load()` to merge `text_config` fields into root JSON before deserialization
- Add `#[serde(default)]` for fields that may be missing depending on model format
- Add a test that loads the actual model's config.json and verifies all fields parse correctly
- Verify against both INT4 and NVFP4 configs

**Files:** `crates/model/src/config.rs`

### 2. [x] Weight Loader Rewrite (1 week)

- Add `strip_prefix()` support in `load_safetensors()` — remove `model.language_model.` from tensor names
- Filter out `model.visual.*` tensors (or optionally load for multimodal support)
- Add `build_main_layers()` function that:
  - Iterates `0..config.num_hidden_layers`
  - For each layer, reads weight names matching `model.layers.{i}.*` (after prefix strip)
  - Distinguishes GDN vs full-attention layers using `config.layer_types` or weight name pattern
  - Populates `WeightRegistry.layers: Vec<LayerWeights>`
- Handle the new GDN weight structure:
  - Map `linear_attn` → GDN weight fields
  - Map `self_attn` → attention weight fields
  - Handle INT4 triplets vs BF16 weights for each submodule
- MTP weights: verify `build_mtp_weights()` works with the actual `mtp.*` tensor names

**Files:** `crates/model/src/loader.rs`, `crates/model/src/weights.rs`

### 3. [x] INT4 Inference Path (2 weeks)

This is the largest deliverable. Two sub-paths:

#### 3a. Dequantize-at-upload (quick, memory-heavy)
- Add an `upload_quantized_weight()` that takes qweight + qzeros + scales, dequantizes to BF16 on CPU, uploads to GPU
- Works for small tensors but blows memory for the full 27B model (54GB BF16 vs 13.5GB INT4)
- Useful for testing GDN conv1d, norms, and single-layer verification

#### 3b. Native INT4 GEMM in prefill/decode (real solution)
- Modify `prefill.rs` / `decode.rs` to check `WeightData.dtype` and dispatch to:
  - `gemm.matmul_bf16()` for BF16 weights (existing)
  - `gemm.matmul_int4()` for INT4 weights (existing kernel but never wired)
- Add `MatmulConfig` variant selection based on weight dtype
- Handle INT4 combined with GDN: the QKV projection for GDN uses `in_proj_qkv` (INT4) which combines Q, K, V into one tensor — need to split or handle as a single matmul
- MLP is straightforward: gate/up/down projections are standard INT4 matmuls matching `int4_gemm_kernel`

**Files:** `crates/backends/native/src/prefill.rs`, `crates/backends/native/src/decode.rs`, `crates/backends/native/src/upload.rs`

### 4. [~] GDN Forward Pass Adaptation (1 week — works but unverified)

The actual model's GDN layer uses Mamba2-style SSM with these weights per layer:
- `in_proj_a`: BF16, shape `[hidden, 2*linear_num_key_heads*key_dim]`
- `in_proj_b`: BF16, shape `[hidden, linear_num_value_heads*value_dim]`
- `in_proj_qkv` (INT4): QKV combined
- `in_proj_z` (INT4): output gate
- `conv1d.weight`: BF16 1D convolution
- `A_log`: BF16, SSM state transition
- `dt_bias`: BF16, SSM timescale bias
- `norm.weight`: BF16, normalization
- `out_proj` (INT4): output projection

The engine's `gdn.rs` has `gdn_prefill` and `gdn_update` kernels but these implement a DIFFERENT GDN variant (Gated DeltaNet with simpler projections, no SSM/conv1d).

**Fix:** Either (A) write new CUDA kernels for the actual GDN variant, or (B) fall back to a dequantized CPU-side implementation for linear/decode layers as a stopgap.

**Files:** `crates/backends/native/src/gdn.rs`, `crates/cuda/kernels/infers/`

### 5. [x] KV Cache and Attention Fixes (3 days)

- Use `config.head_dim` (256) and `config.num_key_value_heads` (4) for KV cache sizing
- Add optional QK LayerNorm (`q_norm.weight`, `k_norm.weight`) to the attention path
- Fix `KvCache` internal buffer sizing to support `head_dim=256`
- Add `seqlen_extrapolation` RoPE support if needed (Qwen3.5 uses extended RoPE)

**Files:** `crates/backends/native/src/attention.rs`, `crates/backends/native/src/rope.rs`, `crates/kv/src/`

### 6. [ ] PrismaSCOUT NVFP4 Support (1 week, parallel track)

The NVFP4 model uses a mixed-precision manifest (`mixed_native_manifest.json`) that specifies which tensors are FP4 vs BF16.

- Read `mixed_native_manifest.json` to determine per-tensor quantization format
- Add NVFP4 dequantize kernel (the `fp8_quantize_kernel` handles FP8, but NVFP4 needs its own)
- The engine already has `KvCacheDtype::Nvfp4` variant
- The matmul path for NVFP4 needs either dequantization or a native NVFP4 matmul kernel

**Files:** `crates/model/src/loader.rs`, `crates/backends/native/src/quant.rs`, `crates/cuda/kernels/infers/`

### 7. [x] Server Configuration Integration (2 days)

- Remove hardcoded defaults in `main.rs` — use `ModelConfig` values for all KV/dimension params
- Wire `--kv-cache-dtype` CLI arg to engine initialization
- Wire `--enable-mtp` with actual MTP weight loading
- Add `--tensor-parallel-size` (2) for using both GPUs

**Files:** `crates/server/src/main.rs`

### 8. [~] Smoke Test and Debugging (3 days — passes but slow)

- Write a test that loads the model, runs a single prefill step, and verifies output
- Run server with real model end-to-end
- Debug CUDA OOM, kernel launch failures, NaN outputs
- Create a benchmark script for tokens/sec

**Files:** `tests/` (new integration test), `scripts/bench.sh`

## Success Criteria

1. `cargo run -- --model ~/opt/vllm/models/qwen3.6-27b-autoround-int4/` starts without error
2. `curl POST /v1/chat/completions -d '{"model":"qwen","messages":[{"role":"user","content":"hello"}],"stream":true}'` returns a stream of real tokens (not gibberish, ideally)
3. Tokens/sec ≥ 1 for INT4 on single GPU (with TP=2 as stretch goal)
4. Session cleanup: Ctrl+C stops cleanly, `delete_sequence` frees GPU memory

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GDN kernel doesn't match model architecture | Blocking | Ship with attention-only fallback; linear layers run on CPU |
| INT4 matmul kernel has wrong format for AutoRound | Blocking | Add a dequantize-to-BF16 fallback path per tensor |
| GPU OOM (16GB insufficient) | Showstopper | Use TP=2 across both GPUs; reduce KV cache pages |
| Vision weights cause loading errors | Minor | Filter `model.visual.*` tensors |
| Tokenizer mismatch (tokenizer.json format) | Minor | Use `from_file()` directly, test with known input |

## Deferred to Phase 12

- Multi-GPU tensor parallelism (TP=2) — design exists but needs wiring
- QK-norm in attention (can ship without, quality will be slightly degraded)
- NVFP4 matmul kernel (FP4 requires new CUDA kernel, can live without)
- Vision/multimodal support
- Full paged attention pipeline (flat KV cache is adequate for single-batch)
