# Phase 042: Kernel-Level Optimization Rounds 1 & 2

---
**Status**: DONE
**Last Updated**: 2026-06-25
**Blocks**: Phase 043 (engine-level optimization)
**Blocked by**: Phase 041 (split kernel library — needed typed module dispatch)
**Rationale**: nsys profiling showed INT4 GEMM at 45% of decode time, NCCL at 15%, cuBLASLt at 8%. Systematic kernel-level optimization was the first lever to pull before attempting engine-level changes.
---

## Goal

Reduce INT4 decode latency from 0.050s/step (20.8 tok/s) to under 0.040s/step through kernel-level optimizations. Target: 0.025s/step (40 tok/s).

## Outcome

**0.036s/step (27.8 tok/s)** — 28% improvement from kernel-level work alone.

## Round 1 Experiments (EXP-001 through EXP-010): 0.050 → 0.038s/step

| Experiment | Change | Impact |
|---|---|---|
| EXP-001 | INT4 SM input tiling (cooperative shared mem load) | Baseline established |
| EXP-002 | INT4 vectorized weight loads (128-bit LDG) | ~2% |
| **EXP-003** | **GDN key/query shared mem caching** | **~5%** (3x→1x key reads) |
| EXP-004 | RMSNorm warp shuffle reduction | 0% (not bottleneck) |
| EXP-005 | SiLU vectorized loads | 0% (compute-bound) |
| **EXP-006** | **Paged attn K-cache (Phase 1b weight cache)** | **20%** (98.5% fewer K reads) |
| EXP-007 | GDN merge state loops | 0% (within noise) |
| EXP-008 | RMSNorm block 512 | 0% (not bottleneck) |
| EXP-009 | Fast exp (Schraudolph bit-manip) | 0% (GEMM-bound) |
| EXP-010 | Paged attn block table hoisting | ~2% |

**Key insight**: EXP-006 (paged attention K-cache) was the single highest-impact change, saving 20% by eliminating 98.5% of redundant K reads in Phase 2.

## Round 2 Experiments (EXP-011 through EXP-016): 0.038 → 0.036s/step

| Experiment | Change | Impact |
|---|---|---|
| EXP-011 | K_SPLIT sweep → optimal at 20 | ~2% |
| EXP-013 | Fused ksplit+reduce single kernel | **REVERTED** (3x slower — occupancy loss) |
| EXP-014 | NCCL grouping | **BLOCKED** (data dependency) |
| **EXP-015** | **GDN memcpy elimination** | **5%** (48 memcpy→2 kernel launches) |
| EXP-016 | v4_ksplit as production | **REVERTED** (11% slower than v3_sm) |

## Key Decisions

- **v3_ksplit_sm is the production INT4 GEMM**: Alternative block structures (v4, warp_split) are all worse due to shared memory tiling being critical for decode.
- **K_SPLIT=20**: Better occupancy on 40-SM GPU than previous 28.
- **Fused ksplit+reduce rejected**: Only ceil(N/64) blocks → very low GPU occupancy.
- **NCCL grouping blocked**: Residual add between attn and MLP AR creates strict serialization.

## Files Modified

| Kernel file | Changes |
|---|---|
| `int4_kernels.rs` | SM tiling, vectorized loads, v4, warp_split (v3_sm remains) |
| `gdn_kernels.rs` | 2D grid, shared mem key/query, merged state loops |
| `attention_kernels.rs` | K-cache Phase 1b, block table hoisting |
| `norm_kernels.rs` | Warp shuffle reduction, block 512 |
| `activation_kernels.rs` | SiLU vectorized loads |
| `common_kernels.rs` | repeat_interleave kernel (EXP-015) |
| `shared.rs` | fast_expf (Schraudolph) |

## nsys Profile (Post Round 1)

| Kernel | % | Per-step |
|---|---|---|
| int4_gemm_v3_ksplit_sm | 44.5% | 19.6ms |
| ncclDevKernel_AllReduce | 14.7% | 5.5ms |
| cuBLASLt bf16 gemvx | 7.6% | 3.5ms |
| infers_gdn_recurrent_step | 5.7% | 4.8ms |
| infers_paged_attention_decode | 4.1% | 3.5ms |
| infers_rmsnorm_bf16 | 1.9% | 0.8ms |
| reduce_partial_sums_bf16 | 1.2% | 0.5ms |
| Other | ~2.5% | ~1ms |
| **Overhead gap** | | **~8ms** |
