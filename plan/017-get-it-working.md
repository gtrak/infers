# Phase 12: Get It Working — End-to-End Real Model Inference

---
**Status**: PARTIAL
**Last Updated**: 2026-06-11
**Rationale**: End-to-end smoke test passes (varied non-zero tokens). BUT: Server not wired. No reference comparison. Performance terrible (~0.1 tok/s, 200× off target).
**Actual Deliverables**:
- [x] End-to-end smoke test passes with real model
- [x] Token validity (non-zero, in vocab range)
- [ ] Output coherence (legible text)
- [ ] Determinism check
- [~] No CUDA errors (some kernel warnings remain)
- [~] Graceful shutdown
- [ ] Server wired to engine
- [ ] Reference comparison
- [~] Performance improvement (still ~200× off target of 20 tok/s)
---

**Duration:** 3–4 weeks  
**Goal:** Run end-to-end inference with real Qwen3.6-27B AutoRound INT4 model and get coherent token output, not garbage.

## Problem

Phase 11 completed the data pipeline — config loading, weight loading, INT4 upload, dimension fixes. But the engine has never produced a single correct token with real weights. Multiple architecture gaps between the existing CUDA kernels and the actual model remain:

| Gap | Layers Affected | Severity |
|-----|----------------|----------|
| GDN kernel implements wrong architecture | 48/64 layers (all GDN) | **BLOCKING** |
| QK-norm not applied in attention | 16/64 layers (full attn) | **HIGH** |
| No integration test with real model | All | **HIGH** |
| Paged KV pipeline is stubbed | All (eventually) | Medium |
| `model.layers.{i}.norm.weight` not loaded | GDN layers | Medium |
| RoPE `partial_rotary_factor` not applied | All attention layers | Medium |
| Download-logits-to-host overhead for sampling | Per-token | Low (perf) |

## Critical Path

### 1. GDN Kernel Rewrite (2 weeks) — BLOCKING

The existing `infers_gdn_prefill_bf16` and `infers_gdn_update_bf16` kernels implement a **Gated DeltaNet** recurrence:

```
state[t] = (1 - σ(Δ)) · state[t-1] + σ(Δ) · (a ⊙ b)
output[t] = state[t] · x
```

The real Qwen3.6-27B GDN is **Mamba2-style SSM** with convolution, A_log parameterization, and a different state update:

```
z = silu(conv1d(input @ in_proj_z))
x = input @ in_proj_a
res = x @ in_proj_qkv         # fused QKV
state = σ(A_log) · state + Δ · (x @ in_proj_b)  # where Δ = softplus(input @ dt_weight + dt_bias)  
output = state @ in_proj_c * silu(z)
```

The existing kernels must be replaced with new CUDA kernels that match the Mamba2 architecture. The GDN layer weights map as:

| Weight | Shape | dtype | Role |
|--------|-------|-------|------|
| `in_proj_a` | [hidden, 2\*num_key_heads\*head_dim] | BF16 | Input projection (x) |
| `in_proj_b` | [hidden, num_value_heads\*value_dim] | BF16 | State contribution |
| `in_proj_qkv` (qweight/qzeros/scales) | [hidden, 3\*num_heads\*head_dim] | INT4 | Fused QKV for residual |
| `in_proj_z` (qweight/qzeros/scales) | [hidden, hidden] | INT4 | Output gate |
| `conv1d.weight` | [hidden, 1, conv_width] | BF16 | 1D convolution on input |
| `A_log` | [num_heads, head_dim] | BF16 | Log of state transition matrix |
| `dt_bias` | [num_heads\*head_dim] | BF16 | Bias for delta timescale |
| `dt_weight` (x_proj) | [hidden, num_heads\*head_dim] | BF16 | Delta projection |
| `norm.weight` | [num_value_heads\*value_dim] | BF16 | State normalization |
| `out_proj` (qweight/qzeros/scales) | [num_heads\*head_dim_v, hidden] | INT4 | Output projection |

#### Implementation Plan

**1a. Mamba2 GDN Prefill Kernel** (`kernels/infers/gdn_mamba2_prefill.cu`)

```
__global__ void infers_gdn_mamba2_prefill_bf16(
    input: [seq_len, hidden],           // BF16
    state: [num_heads, head_dim, num_value_heads, value_dim],  // BF16, in/out
    in_proj_a, in_proj_b,               // BF16 weights (GPU)
    conv1d_weight,                       // BF16 [hidden, 1, conv_width]
    A_log, dt_bias,                     // BF16 SSM params
    output_buf: [seq_len, hidden],      // BF16 output
    in_proj_qkv_qweight, in_proj_z_qweight,  // INT4 weights
    scales/qzeros for INT4,
    seq_len, hidden_size, head_dim, num_heads, num_kv_heads,
    conv_width: i32,
)
```

Strategy: Process one token at a time within each thread block. Each block handles one head's state dimension. For each token:
1. Load projected values (already computed by GEMMs before kernel call)
2. Apply 1D convolution to input
3. Compute delta = softplus(x @ dt_weight + dt_bias)
4. State update: state = exp(A_log) · state + delta · (b projection)
5. Output = state · c_proj
6. Write to output buffer (one element per thread)

**1b. Mamba2 GDN Decode Kernel** (`kernels/infers/gdn_mamba2_update.cu`)

Single-token variant. Same state update but processes only one token. Takes pre-computed projections as inputs.

**1c. Host-side dispatch** (`crates/backends/native/src/gdn.rs`)

Rewrite `forward()` and `decode_forward()`:
- Load all GDN weights: in_proj_a, in_proj_b, in_proj_qkv (INT4), in_proj_z (INT4), conv1d_weight, A_log, dt_bias, x_proj_weight, norm
- Upload BF16 weights directly, INT4 via `upload_int4_weight()`
- Compute input projections via `gemm_projection()`:
  - `a = input @ in_proj_a` (BF16)
  - `b = input @ in_proj_b` (BF16)  
  - `dt = input @ x_proj_weight` (BF16)
  - `qkv = input @ in_proj_qkv` (INT4) — fused QKV for residual path
  - `z_gate = input @ in_proj_z` (INT4) — for output gating
- Launch Mamba2 GDN kernel with all projected buffers
- Compute `output @ out_proj` via `gemm_projection()` (INT4)
- Add residual: `hidden = hidden + output + qkv_residual`
- Apply `norm.weight` to state (for next step)

### 2. QK-Norm in Attention (3 days) — HIGH

The 16 full-attention layers have `q_norm.weight` and `k_norm.weight` that are loaded but never applied.

**File:** `crates/backends/native/src/attention.rs`

In `forward()` and `decode_forward()`, after computing per-head Q and K but before applying RoPE:

```rust
// Apply QK-norm if weights exist
if let Some(q_norm) = &weights.q_norm {
    let q_norm_gpu = upload_weight(stream, q_norm)?;
    // Apply RMSNorm to Q head: q_norm_gpu has shape [head_dim]
    q_h = rms_norm_per_head(stream, &kernels.rmsnorm, &q_h, &q_norm_gpu, eps)?;
}
if let Some(k_norm) = &weights.k_norm {
    let k_norm_gpu = upload_weight(stream, k_norm)?;
    k_h = rms_norm_per_head(stream, &kernels.rmsnorm, &k_h, &k_norm_gpu, eps)?;
}
```

Add a `rms_norm_per_head()` function that applies RMSNorm to a flattened per-head Q/K vector `[head_dim]` using the head-specific norm weight `[head_dim]`. The existing `rms_norm()` kernel works on `[hidden_size]` — adapt it or create a small wrapper.

For `forward_paged()` and `decode_forward_paged()`, apply the same QK-norm logic.

### 3. Integration Smoke Test (4 days) — HIGH

Create an end-to-end integration test that validates the engine produces non-garbage output.

**File:** `tests/integration/smoke_test.rs` (new)

```
#[test]
fn smoke_test_real_model() -> Result<()> {
    // 1. Load model from INFERS_TEST_MODEL env var or default path
    let model_dir = Path::new(&std::env::var("INFERS_TEST_MODEL")
        .unwrap_or_else(|_| "~/opt/vllm/models/qwen3.6-27b-autoround-int4/".to_string()));
    
    // 2. Load config and weights
    let loaded = load_model(model_dir)?;
    
    // 3. Initialize CUDA + engine
    let ctx = CudaRuntime::new()?.device(0)?.clone();
    let stream = ctx.create_stream()?;
    let engine = ForwardEngine::new(
        Arc::new(loaded.config.clone()),
        vec![loaded.weights],
        ctx, kernel_registry, stream_pool, 128,
    )?;
    
    // 4. Known-good input: "Hello" → expected token ID range
    let token_ids = vec![0, 12345, 67890]; // or load from tokenizer.txt fixture
    
    // 5. Run prefill
    let sampled = engine.prefill(&stream, &token_ids)?;
    
    // 6. Verify output is valid (non-zero, in vocab range)
    assert!(sampled < loaded.config.vocab_size as u32);
    assert_ne!(sampled, 0); // not padding
    
    // 7. Run decode for N steps, verify all tokens are valid
    let mut token = sampled;
    for pos in token_ids.len()..token_ids.len() + 10 {
        token = engine.decode(&stream, token, pos as u32)?;
        assert!(token < loaded.config.vocab_size as u32);
    }
    
    Ok(())
}
```

This test requires:
- GPU hardware (skip with `#[ignore]` or `#[cfg(feature = "cuda-tests")]`)
- Model download (document how to get it)
- ~30s to load model and run inference

### 4. GDN Norm Weight Loading (1 day) — Medium

The real model's GDN layers have `model.layers.{i}.norm.weight` which is the GDN state normalization weight. This is currently loaded as `gdn_weights.norm` (the optional field we added in Phase 11 Task 5) but never uploaded or used in the GDN forward pass.

Verify the weight is correctly extracted in `build_main_layer()` and add it to the GDN kernel dispatch.

### 5. RoPE partial_rotary_factor (1 day) — Medium

Qwen3.6-27B uses `partial_rotary_factor: 0.25`, meaning only 25% of head dimensions get rotary embedding. The current `infers_rope_bf16` kernel assumes full rotation (all dims).

Fix: the kernel already accepts `partial_rotary_factor` (check the kernel source) — verify the host-side dispatch passes it correctly. In `attention.rs` `forward()` and `decode_forward()`, the `partial_rotary_factor` is already passed — verify the dimension math is correct: `rotary_dim = (head_dim * partial_rotary_factor) as usize`.

### 6. Sampling Optimization (2 days)

Current sampling downloads the full logits vector from GPU to CPU, extracts the last row, converts to f32, re-uploads to GPU, and runs argmax. For `vocab_size=248320`, this is ~500KB per decode step.

Optimization: compute argmax on GPU directly on the BF16 logits without CPU round-trip. Add an `argmax_bf16` CUDA kernel that finds the max element index in a BF16 array (same as `infers_argmax_f32` but operates on BF16 directly).

### 7. Paged Pipeline Wiring (concurrent with other tasks)

Un-stub `prefill_paged()` and `decode_paged()`:
- Wire the layer loop from `prefill.rs`/`decode.rs` into the paged methods
- Replace flat `KvCache` writes with paged KV writes using `paged_kv_write_kernel`
- Replace flat KV cache reads in decode with `paged_kv_read_kernel` + `paged_attention_decode_kernel`
- This enables continuous batching with the real engine

## Success Criteria

- [~] **Output coherence**: `cargo run` starts and returns tokens (legibility not verified)
- [x] **Token validity**: All sampled tokens are in range `[0, vocab_size)`, non-zero
- [ ] **Determinism**: Same input + seed produces same output sequence
- [~] **No CUDA errors**: No illegal memory access (some kernel warnings remain)
- [~] **Graceful shutdown**: Ctrl+C triggers cleanup (partially working)
- [ ] **Performance**: ≥20 tok/s target (currently ~0.1 tok/s, 200× off)
- [ ] **Server wired**: `curl` returns real generated tokens via server API
- [ ] **Reference comparison**: Matches HuggingFace output within tolerance

## Files to Modify

| File | Change |
|------|--------|
| `crates/cuda/kernels/infers/gdn_mamba2_prefill.cu` | NEW: Mamba2 GDN prefill kernel |
| `crates/cuda/kernels/infers/gdn_mamba2_update.cu` | NEW: Mamba2 GDN decode kernel |
| `crates/backends/native/src/gdn.rs` | Rewrite: dispatch to new Mamba2 kernels |
| `crates/backends/native/src/attention.rs` | Add QK-norm in forward/decode_forward |
| `crates/backends/native/src/norm.rs` | Add `rms_norm_per_head()` helper |
| `crates/backends/native/src/engine.rs` | Add Mamba2 kernel handles |
| `tests/integration/smoke_test.rs` | NEW: end-to-end real model test |
| `crates/backends/native/src/sample.rs` | Add `argmax_bf16` kernel or optimize path |
| `crates/backends/native/src/decode.rs` | Wire paged pipeline, verify partial_rotary_factor |
| `crates/backends/native/src/prefill.rs` | Wire paged pipeline, verify partial_rotary_factor |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mamba2 GDN kernel produces wrong math | **BLOCKING** | Write small standalone test in Python with reference impl; verify against HF output for a single layer |
| QK-norm changes don't match HF implementation | High | Trace through HF Qwen3.5 modeling code; compare norm application order (before/after RoPE) |
| INT4 dequantization mismatch | High | Verify `dequantize_int4_to_bf16` formula matches `int4_gemm_kernel` by comparing outputs for random input |
| Model too large for 16GB GPU | High | Reduce to 1-layer test; use `--num-pages 64` for minimal KV cache |
| No CUDA memory for full model weights | Showstopper | Load only first N layers; verify single-layer correctness first |

## Immediate Next Steps

1. **Week 1**: Write the Mamba2 GDN kernel (prefill + decode). Test in isolation with synthetic data.
2. **Week 2**: Wire Mamba2 dispatch in `gdn.rs`. Add QK-norm. Test single-layer correctness against HF reference.
3. **Week 3**: Full 64-layer integration test. Fix accuracy issues. Add smoke test.
4. **Week 4**: Optimize sampling. Wire paged pipeline. Benchmark tokens/sec.

## Reference: Verifying Against HuggingFace

To validate correctness at each step:

```python
# reference_single_layer.py
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", 
    device_map="cpu", torch_dtype=torch.bfloat16)

# Extract layer 0 GDN weights
state_dict = model.state_dict()
gdn_kernel_w = state_dict["model.language_model.layers.0.linear_attn.in_proj_a.weight"]
# ... run single layer in Python, compare output with infers
```

Run this Python script to extract per-layer inputs/outputs, then compare with infers layer output. Outputs should match within 1% relative error for BF16.
