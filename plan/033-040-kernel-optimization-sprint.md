# Phase 033-040: Kernel-Level Optimization Sprint

## Overview

Fresh nsys profiling of INT4 v3 decode (48ms/token, 20.8 tok/s) reveals the following breakdown:

| Component | Time | % | Phase |
|-----------|------|---|-------|
| `int4_gemm_v3_ksplit` | 18.6ms | 39% | 034 |
| NCCL all-reduce | 8.75ms | 18% | 039 |
| `paged_attention_decode` | 6.7ms | 14% | 035 |
| cuBLAS gemvx | 4.8ms | 10% | (cuBLAS, skip) |
| `gdn_recurrent_step` | 3.0ms | 6% | 036 |
| Host dispatch overhead | 6.0ms | 12% | 040 |
| `rmsnorm` | 1.25ms | 3% | 037 |
| `reduce_partial_sums` | 0.5ms | 1% | ✓ done |
| Other | ~0.1ms | <1% | — |
| **Total** | **48ms** | 100% | — |

Target: 25ms/token (40 tok/s). Need to cut 23ms.

## Dependency Graph

```
Phase 033 (Bench Harness) ← blocker for 034, 035, 036, 037, 038
    │
    ├── 034 (INT4 GEMM v4)     → -10ms
    ├── 035 (Attn Decode)      → -4ms
    ├── 036 (GDN Recurrent)    → -1.3ms
    ├── 037 (RMSNorm)          → -0.5ms
    └── 038 (NVFP4 GEMM v4)   → (NVFP4 path, independent)

Phase 039 (NCCL)              → -3.7ms  (no dependency)
Phase 040 (Host Dispatch)     → -4ms    (no dependency)
```

## Execution Order

1. **Phase 033** — Build bench harness (S, delegate first, blocks everything)
2. **Phase 034** — INT4 GEMM v4 vectorized loads (M, biggest win)
3. **Phase 038** — NVFP4 GEMM v4 (M, same pattern as 034)
4. **Phase 035** — Paged attention optimization (M, second biggest)
5. **Phase 036** — GDN recurrent step (S)
6. **Phase 037** — RMSNorm (S, easiest)
7. **Phase 039** — NCCL batching (S, independent)
8. **Phase 040** — Host dispatch (S, independent)

## Projected Outcome

| After Phase | INT4 decode | tok/s |
|-------------|------------|-------|
| Baseline (now) | 48ms | 20.8 |
| +033 (harness) | 48ms | 20.8 (measurement only) |
| +034 (INT4 v4) | 38ms | 26.3 |
| +035 (attn) | 34ms | 29.4 |
| +036 (GDN) | 33ms | 30.3 |
| +037 (rmsnorm) | 32.5ms | 30.8 |
| +039 (NCCL) | 29ms | 34.5 |
| +040 (host) | 25ms | 40.0 ✓ |

This reaches the 40 tok/s target without CUDA Graphs or multi-stream overlap.
