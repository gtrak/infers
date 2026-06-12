# GDN Gated Delta Rule — Bug-Fix Plan

## Status

The GDN rewrite (Mamba2 SSM → Gated Delta Rule) compiles and runs, but produces all-NaN hidden states (argmax=0, all `!` tokens). Multiple formula bugs have been identified but not all are fixed yet.

## Fixed Bugs

| Bug | Fix | Status |
|-----|-----|--------|
| Qwen3_5RMSNorm uses `(1 + weight)`, kernel used `weight` | Changed rmsnorm.cu to `x * scale * (1.0f + w_val)` | ✅ Committed |
| Same issue in rms_norm_gated kernel | Added `w_scale = 1.0f + w` | ✅ Committed |
| GDN kernel used `exp(-A_log)` instead of `exp(A_log)` | Changed to `decay_rate_h = expf(A_log[h])` | ✅ Committed |
| No query/key L2 normalization before recurrence | Added L2 norm per (t,h): q /= ||q||, k /= ||k|| | ✅ Committed |
| No 1/sqrt(K) output scaling | Added `rcp_sqrt_k = rsqrtf(K)`, applied to q in output | ✅ Committed |
| Erroneous attention scale applied to state update output | Removed `scale = 1/sqrt(K)` multiplier from output (replaced with proper HF formula) | ✅ Committed |
| b_proj extraction via memcpy_dtod produced zeros | Changed to use b_proj_raw directly | ✅ Committed |
| INT4 companion scales (BF16 dtype) sharded on wrong dim during TP=2 | Extract companions from registry.tensors by name pattern during sharding; slice with same strategy as qweight | ✅ Committed |

## Remaining Phases

### Phase 1 — Fix RMSNorm NaN

The rmsnorm kernel now produces all-NaN after the `(1+weight)` fix. Need to determine if the cubin is stale or if there's a numerical issue.

- [ ] Check if cubin matches source (force delete cubin and rebuild)
- [ ] Write a Rust unit test: upload known tensor+weight, run RMSNorm, check for NaN
- [ ] Check if `hidden_states[0]` has NaN before RMSNorm (embedding issue?)
- [ ] Check if norm1_weight is valid (download from GPU cache)

### Phase 2 — Validate GDN Internals vs HF Reference

Compare each intermediate tensor against HuggingFace reference:

- [ ] mixed_qkv (in_proj_qkv output)
- [ ] conv_out (conv1d)
- [ ] q, k, v splits
- [ ] a_proj, b_proj
- [ ] gdn_output (after recurrence)
- [ ] norm_output (after RMSNormGated)
- [ ] final output (after out_proj)

### Phase 3 — Fix GDN Decode Forward

- [ ] Verify gdn_gated_delta_update kernel has same fixes as prefill
- [ ] Test single-token decode produces valid output
- [ ] Verify state persistence across decode steps

### Phase 4 — Fix out_proj INT4 GEMM NaN

- [ ] Verify if this is still an issue with non-zero norm_output
- [ ] Dump scale/qzero/qweight for failing column from inside kernel

### Phase 5 — Layer-by-Layer Validation

- [ ] Compare all 64 layer outputs against HF reference
- [ ] Identify first divergent layer

### Phase 6 — End-to-End Generation

- [ ] Test with real prompt, verify coherent text output
- [ ] Check prefill/decode performance

### Phase 7 — Clean Up

- [ ] Remove `debug_tensor_stats_bf16` and `debug_tensor_stats_f32` calls
- [ ] Remove remaining debug code
- [ ] Run `rustfmt`, `cargo clippy`, `lat check`
- [ ] Update `lat.md/` documentation
- [ ] Commit

## Key Risks

| Risk | Mitigation |
|------|-----------|
| cubin caching prevents kernel fix from taking effect | Delete `compiled/*.cubin` before each test |
| INT4 GEMM NaN persists with real input | Fall back to host-dequantized BF16 cuBLASLt |
| State explosion after all formula fixes | Add numerical clamping in kernel |
| L2 norm of zero q/k causes div-by-zero | `sqrt(l2 + 1e-6)` guard already present |

## Useful Debug Tools

- `debug_tensor_stats_bf16()` in `gdn.rs` — downloads buffer, prints min/max/mean_abs/stddev/NaN count
- `debug_tensor_stats_f32()` — same for float32 buffers
- `INFERS_DUMP_LAYER_DIR` env var — dumps per-layer hidden states for comparison
- `scripts/dump_ref_hidden.py` — captures HuggingFace reference hidden states
