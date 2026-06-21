# Cleanup Phase B: Medium-Risk Dead Code (Review Complete)

---
**Status**: COMPLETE
**Last Updated**: 2026-06-21
**Rationale**: Dead code items reviewed. Low-risk items cut in Phase A/B. Medium-risk items reviewed — most kept as infrastructure for planned features.
---

## Decisions

| # | What | Decision | Rationale |
|---|------|----------|-----------|
| B1 | `evict_session()` + `restore_session()` | KEEP | Phase 6.6 infrastructure — GPU→CPU data copy for eviction |
| B2 | `budget.rs` module | KEEP | Standalone utility for auto-calculating memory budgets from model config |
| B3 | `QuantizationFormat::detect()` | KEEP | Phase 8 — auto-detect quant format from model files |
| B4 | `should_use_tools()` | KEEP | Phase 9 — gate function for tool calling |
| B5 | `eviction_store` field | KEEP | Phase 6.6 — holds evicted page data during eviction |
| B6 | `num_layers` field | CUT | Dead field, no callers |
| B7 | `num_gdn_layers()` | CUT | Dead method, no callers |
| B8 | NCCL broadcast/reduce/all_gather | CUT | Multi-node never planned |
| B9 | `QuantizedKvCache` | KEEP | Phase 8 — FP8 KV cache infrastructure |
| B10 | Prefix cache methods | KEEP | Phase 17+ — LRU prefix caching infrastructure |
| B11 | COW helpers | CUT | Dead methods, no callers |
| B12 | `PageLocation::Cpu` | CUT | Dead enum variant, no callers |
| B13 | `remaining_bytes` + `clear` | CUT | Dead methods, no callers |
| B14 | Dead parallelism methods | CUT | Dead methods, no callers |
| B15 | Dead orchestrator MTP fields | CUT | Dead fields, no callers |
| B16 | `paged_kv_read` kernel handle | CUT | Dead code, superseded by paged_attention_decode |

## Cut Items (executed)

B6, B7, B8, B11, B12, B13, B14, B15, B16 — removed in cleanup commits `57aa0cb` and `c94c99a`.

## Kept Items (infrastructure for future phases)

| Item | Lines | Phase | Status |
|------|-------|-------|--------|
| B1 | 115 | 6.6 | Engine methods exist, not wired to orchestrator |
| B2 | 505 | — | Module exists, not used by server startup |
| B3 | 50 | 8 | Function exists, not called by server |
| B4 | 18 | 9 | Function exists, not called by chat handler |
| B5 | 2 | 6.6 | Field exists on orchestrator, never read |
| B9 | 63 | 8 | Struct exists, engine uses BF16 instead |
| B10 | 100 | 17+ | Prefix cache methods exist, disconnected from production |
