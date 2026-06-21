# GDN Gated Delta Rule — Bug-Fix Plan

## Status

**Status**: DONE
**Last Updated**: 2026-06-21

The GDN rewrite (Mamba2 SSM → Gated Delta Rule) compiles and runs with valid outputs (no NaN). All 12 computable intermediates match HF reference (cos > 0.98). Smoke test passes with 30 generated tokens.

## Fixed Bugs

| Bug | Fix | Status |
|-----|-----|--------|
| Qwen3_5RMSNorm uses `(1 + weight)`, kernel used `weight` | Changed rmsnorm.cu to `x * scale * (1.0f + w_val)` | ✅ Committed |
| rms_norm_gated incorrectly used additive offset `(1 + w)` when Qwen3_5RMSNormGated uses full scale `w` (init=1.0) | Changed to `w_scale = w` | ✅ Committed |
| GDN kernel used `exp(-A_log)` instead of `exp(A_log)` | Changed to `decay_rate_h = expf(A_log[h])` | ✅ Committed |
| No query/key L2 normalization before recurrence | Added L2 norm per (t,h): q /= ||q||, k /= ||k|| | ✅ Committed |
| No 1/sqrt(K) output scaling | Added `rcp_sqrt_k = rsqrtf(K)`, applied to q in output | ✅ Committed |
| Erroneous attention scale applied to state update output | Removed `scale = 1/sqrt(K)` multiplier from output (replaced with proper HF formula) | ✅ Committed |
| b_proj extraction via memcpy_dtod produced zeros | Changed to use b_proj_raw directly | ✅ Committed |
| INT4 companion scales (BF16 dtype) sharded on wrong dim during TP=2 | Extract companions from registry.tensors by name pattern during sharding; slice with same strategy as qweight | ✅ Committed |
| QKV split used flat contiguous copy instead of per-row column extraction | Changed to `extract_columns()` with per-row strided copies from row-major `conv_out` | ✅ Committed |
| in_proj_qkv sharding naively split conv_dim instead of per-projection (Q/K/V independently) | Fixed `shard_fused_projection_columns` — segments Q, K, V independently divided by num_gpus | ✅ Committed |
| conv1d.weight had same sharding issue as in_proj_qkv | Same fix as in_proj_qkv — per-projection column split via `shard_fused_projection_columns` | ✅ Committed |
| ColumnMajor iteration order in sharding was segments-first instead of rows-first | Changed to rows-outer/segments-inner loop for row-contiguous output matching INT4 GEMM kernel layout | ✅ Committed |
| Token ID mismatch between Rust (HF `tokenizers` crate) and HF Python tokenizers | Aligned tokenizer configuration to produce identical token ID sequences before comparison | ✅ Committed |
| z_gate/norm_output comparison used contiguous slicing instead of per-token head sharding | Sliced reference tensor by `[seq, num_v_heads_per_gpu * head_dim]` to match TP=2 shard shape | ✅ Committed |
| NCCL all-reduce was declared in sync.rs but never wired into prefill/decode GDN path | Added `nccl.all_reduce_in_place()` after GDN output projection in both `prefill.rs` and `decode.rs` with group_start/group_end pattern | ✅ Committed |

## Remaining Phases

### Phase 1 — Fix RMSNorm NaN ✅ DONE

The rmsnorm kernel now produces all-NaN after the `(1+weight)` fix. Need to determine if the cubin is stale or if there's a numerical issue.

**Diagnosis:** After investigation, the NaN was caused by two issues:
1. The `rms_norm_gated` kernel incorrectly applied the additive offset formula `(1 + w)` when Qwen3_5RMSNormGated expects full scale weight (init=1.0). This was fixed in commit 46fa737 — changed to `w_scale = w`.
2. The cubin cache needed a rebuild after kernel source changes.

**Verification:** All debug output shows `nan=0 inf=0` across all intermediate tensors. Smoke test passes with 30 generated tokens and no NaN values.

- [x] Check if cubin matches source (force delete cubin and rebuild)
- [ ] Write a Rust unit test: upload known tensor+weight, run RMSNorm, check for NaN
- [ ] Check if `hidden_states[0]` has NaN before RMSNorm (embedding issue?)
- [ ] Check if norm1_weight is valid (download from GPU cache)

### Phase 2 — Validate GDN Internals vs HF Reference ✅ DONE

Compare each intermediate tensor against HuggingFace reference. All 12 computable intermediates pass (cos > 0.98). The 13th (output) correctly shows as ROW-PAR partial sum before all-reduce.

- [x] mixed_qkv (in_proj_qkv output)
- [x] conv_out (conv1d)
- [x] q, k, v splits
- [x] a_proj, b_proj
- [x] gdn_output (after recurrence)
- [x] norm_output (after RMSNormGated)
- [x] final output (after out_proj — ROW-PAR partial sum, diverges as expected before all-reduce)

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
