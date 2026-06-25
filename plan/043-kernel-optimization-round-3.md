# Phase 043: Kernel-Level Optimization Round 3 + Prerequisite Cleanup

---
**Status**: DONE
**Last Updated**: 2026-06-25
**Blocks**: Phase 044 (CUDA graphs)
**Blocked by**: Phase 042 (rounds 1-2)
**Rationale**: Round 3 tested remaining kernel-level hypotheses (warp_split, custom BF16 GEMV, Q/gate split). Also discovered RoPE Q-RoPE was uploading ~256MB/step of cos/sin tables unnecessarily — a correctness and performance bug that was also a CUDA graph blocker. Began eliminating all per-step dynamic allocations and H2D copies as CUDA graph prerequisites.
---

## Goal

1. Test remaining kernel optimization hypotheses (EXP-018, EXP-021, EXP-022)
2. Fix Q-RoPE cos/sin upload bug
3. Eliminate all per-step dynamic GPU allocations and H2D copies (CUDA graph prerequisites)

## Outcome

**Latency unchanged at 0.036s/step** — kernel-level optimization is exhausted. However, critical CUDA graph prerequisites are now in place.

## Round 3 Experiments

| Experiment | Change | Impact |
|---|---|---|
| EXP-018 | Warp-split INT4 GEMM | **REVERTED** (0.219s — 6x slower) |
| EXP-021 | Custom BF16 GEMV (2 designs) | **REVERTED** (0.037-0.040s — slower) |
| EXP-022 | Attention Q/gate split kernel | Integrated (0.036s — no change, cleaner) |

**Conclusion**: INT4 GEMM at 58% of decode time is near bandwidth limit. cuBLASLt at 10% is already efficient. NCCL at 20% is blocked by residual dependency. The remaining 11ms gap requires engine-level optimization.

## Q-RoPE Bug Fix

Discovered that all 4 Q-RoPE call sites in `decode_forward_paged` passed `None, None` for `cached_cos, cached_sin`, causing fallback path to recompute and re-upload ~8MB of cos/sin tables per attention layer per step. K-RoPE already used cached tables. Fix: pass `cached_cos, cached_sin` through.

This eliminated ~256MB/step of H2D copies — a correctness fix and CUDA graph prerequisite.

## CUDA Graph Prerequisites (Completed)

| Change | What | Why |
|---|---|---|
| LM head partial_sums | Pre-allocated in workspace (K_SPLIT × vocab_size) | Eliminates ~12MB dynamic alloc/step |
| Embedding output | Pre-allocated `embed_out` in workspace | Eliminates alloc/step |
| Token ID staging | Pre-allocated `token_ids_staging` + memcpy_htod | Replaces clone_htod |
| Position staging | Pre-allocated `position_staging` + memcpy_htod | Replaces clone_htod |
| RoPE position staging | Pre-allocated `rope_position_staging` + memcpy_htod | Replaces clone_htod |
| Block table staging | Pre-allocated `block_table_staging` + memcpy_htod | Replaces clone_htod |
| num_cached_tokens | Device buffer instead of kernel argument | Dynamic between steps |
| embed_tokens_into | Writes into pre-allocated buffer | Zero-allocation variant |
| apply_rope_with_staging | Zero-allocation RoPE variant | Uses staging buffer |

## Fresh nsys Profile (Post Round 3)

| Kernel | % | Per-step |
|---|---|---|
| int4_gemm_v3_ksplit_sm | 57.9% | ~20.8ms |
| ncclDevKernel_AllReduce | 19.6% | ~7.0ms |
| cuBLASLt bf16 gemvx | 9.7% | ~3.5ms |
| infers_gdn_recurrent_step | 7.3% | ~2.6ms |
| infers_paged_attention_decode | 5.0% | ~1.8ms |
| infers_rmsnorm_bf16 | 2.3% | ~0.8ms |
| reduce_partial_sums_bf16 | 1.3% | ~0.5ms |
| Other | ~1.1% | ~0.4ms |
| **Wall time** | | **~36ms** |
| **CPU launch overhead** | | **~6ms** |

## Files Modified

| File | Changes |
|---|---|
| `bf16_kernels.rs` | BF16 GEMV kernels (added then reverted) |
| `common_kernels.rs` | split_qgate_bf16 kernel |
| `attention_kernels.rs` | cached_tokens_count device buffer, kv_read buffer |
| `oxide_bridge.rs` | BF16 GEMV wrappers, split_qgate, cached_tokens_count param |
| `attention.rs` | Q-RoPE cached cos/sin, split_qgate, staging buffers |
| `engine.rs` | Staging buffer writes, graph field stubs |
| `workspace.rs` | All staging buffer fields + lm_head_partial_sums |
| `embedding.rs` | embed_tokens_into zero-allocation variant |
| `rope.rs` | apply_rope_with_staging zero-allocation variant |
| `gemm_dispatch.rs` | LM head partial_sums path |
