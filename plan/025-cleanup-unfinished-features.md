# Cleanup Phase C: Unfinished Features (Deferred)

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-20
**Blocks**: Nothing
**Blocked by**: Cleanup Phase A + B
**Rationale**: These are complete implementations that were built during earlier phases but never wired into the server's production path. They represent significant functionality (pipeline parallelism, speculative decoding, tool calls) that was implemented and tested in isolation. The decision here is: (a) delete entirely, (b) keep as-is with `#[allow(dead_code)]`, or (c) wire into the server now. Wiring is out of scope for cleanup — these items are documented here for future phase planning.
---

## Feature Inventory

### 1. Pipeline Parallelism Crate (`crates/parallelism/`)

**Lines**: ~600 across 7 files
**Phase**: 5 (PP microbatching — completed)
**Status**: Implementation complete. `PipelineEngine`, `MicrobatchScheduler`, `StageComm`, `TensorParallelEngine` all exist with tests. But the crate has zero external consumers — no other crate imports it.

**What exists**:
- `PipelineEngine`: full forward loop with microbatch scheduling, NCCL P2P send/recv between stages
- `MicrobatchScheduler`: splits prefill into microbatches, manages in-flight count
- `StageComm`: NCCL communicator wrapper for stage-to-stage communication
- `TensorParallelEngine`: TP sharding across GPUs within a stage
- `GdnStateRef`: GDN layer state management for pipeline stages

**What's missing**: Server integration. The server binary doesn't depend on `infers-parallelism`. The `ParallelismMode` enum in `main.rs` (line 105) is local to the server, not from the parallelism crate.

**Options**:
- **DELETE**: Remove entire crate. Re-implement when PP is actually needed.
- **KEEP AS-IS**: Leave the crate untouched. It compiles, has tests, and doesn't affect anything.
- **WIRE NOW**: Add `infers-parallelism` as a server dependency, implement PP mode in the orchestrator. This is a significant feature, not a cleanup.

**Recommendation**: KEEP AS-IS. The crate is self-contained, tested, and doesn't pollute the main codebase. Deleting it wastes prior work. Wiring it is a feature task, not cleanup.

---

### 2. MTP Crate (`crates/mtp/`)

**Lines**: ~500 across 5 files
**Phase**: 7 (Multi-Token Prediction — completed)
**Status**: Full implementation. `MtpEngine::generate_drafts()`, `verify_drafts()`, `accept_prefix()`, `adaptive_num_drafts()` all exist with tests. But `main.rs` sets `let mtp = None` and the orchestrator never calls any MTP methods.

**What exists**:
- `MtpHead`: forward pass for draft token prediction (normed embedding + concat, FC projection, decoder layer)
- `MtpEngine`: speculative decoding loop (generate drafts → verify → accept prefix)
- `MtpMetrics`: acceptance rate, step rate, avg draft count tracking
- `VerificationResult`: per-draft verification result

**What's missing**: Engine integration. The forward engine doesn't call MTP during decode. No speculative-config API parameter exists.

**Options**:
- **DELETE**: Remove entire crate. Re-implement when speculative decoding is needed.
- **KEEP AS-IS**: Leave untouched. It compiles, has tests, and doesn't affect anything.
- **WIRE NOW**: Integrate MTP into the engine's decode path. This is a significant feature task.

**Recommendation**: KEEP AS-IS. Same rationale as C1 — self-contained, tested, no pollution.

---

### 3. Tool Call Parser (`crates/api/src/tool_parser.rs`)

**Lines**: ~250 across 1 file
**Phase**: 9 (Tool Calls — completed)
**Status**: Complete implementation with streaming and non-streaming parsing. 6 unit tests all pass. Supports Qwen3.6 XML format. But the server's chat handler never imports or uses it.

**What exists**:
- `ToolCallParser`: `parse_streaming_delta()` for SSE streaming, `parse_complete()` for non-streaming
- `PartialToolCall`: streaming state management (accumulating XML fragments)
- Handles multiple JSON formats (Qwen3.6 XML, JSON array, function call)

**What's missing**: Chat handler integration. The handler passes tools to `template.apply()` but never parses tool calls from model output.

**Options**:
- **DELETE**: Remove the module. Re-implement when tool calls are needed.
- **KEEP AS-IS**: Leave untouched. It compiles, has tests, and is re-exported from `api::lib.rs` but unused.
- **WIRE NOW**: Integrate into the chat handler's response pipeline. This is a feature task.

**Recommendation**: KEEP AS-IS. It's a well-tested, complete module. The re-export from `lib.rs` is harmless.

---

### 4. Eviction Wiring (GPU→CPU Data Movement)

**Lines**: ~170 across 2 files
**Phase**: 6.6 (Eviction Wiring — partially completed)
**Status**: Engine methods (`evict_session`, `restore_session`) exist and work. `BackendEvictionStore` is implemented. But the orchestrator's eviction path bypasses them entirely — it calls `delete_sequence()` directly, freeing pages without preserving GPU data.

**What exists**:
- `ForwardEngine::evict_session()`: copies page data from all layers' GPU buffers to CPU store
- `ForwardEngine::restore_session()`: allocates pages, retrieves from store, copies back to GPU
- `BackendEvictionStore`: per-layer, per-page CPU data storage
- `CpuPagePool`: CPU-side page pool for evicted data (in kv crate)

**What's missing**: Orchestrator wiring. The eviction path needs to:
1. Call `engine.evict_session()` before `delete_sequence()`
2. Store the returned `EvictedSequence`
3. Call `engine.restore_session()` when a sequence is re-admitted

**Options**:
- **DELETE**: Remove engine methods + store. Eviction stays as "delete only."
- **KEEP AS-IS**: Leave the dead code in place. It's marked `#[allow(dead_code)]` on the orchestrator.
- **WIRE NOW**: Implement the orchestrator integration. This is a feature task.

**Recommendation**: Depends on Phase B decisions. If Task 1 is CUT, this is moot. If Task 1 is KEEP, consider wiring in a future phase.

---

### 5. Paged Prefill Wiring

**Lines**: N/A (no new code — wiring existing methods)
**Phase**: 17 (separate plan exists at `plan/phase-17-paged-prefill-wiring.md`)
**Status**: NOT STARTED. The orchestrator uses non-paged `engine.prefill()` for prefill and paged `engine.decode_paged()` for decode. This creates a state mismatch.

**What exists**: Everything needed — `prefill_paged()`, `decode_paged()`, `PagedKvManager`, `PagedKvCache`. Just needs wiring.

**What's missing**: The 7 commits described in `plan/phase-17-paged-prefill-wiring.md`.

**Options**:
- **DEFER**: Leave as-is. The TODO comment at orchestrator line 174 documents the gap.
- **EXECUTE**: Run Phase 17 as a separate task.

**Recommendation**: DEFER — Phase 17 is a separate, well-planned feature. Don't mix it with cleanup.

---

## Summary

| # | Feature | Lines | Status | Recommendation |
|---|---------|-------|--------|----------------|
| 1 | Pipeline Parallelism | ~600 | Complete, not wired | KEEP AS-IS |
| 2 | MTP Speculative Decoding | ~500 | Complete, not wired | KEEP AS-IS |
| 3 | Tool Call Parser | ~250 | Complete, not wired | KEEP AS-IS |
| 4 | Eviction Wiring | ~170 | Methods exist, not wired | Depends on Phase B Task 1 |
| 5 | Paged Prefill Wiring | — | Plan exists, not started | DEFER to Phase 17 |

**Total dead code from unfinished features**: ~1,520 lines across 3 crates + 2 wiring tasks.

**Key insight**: These are not "dead code" in the traditional sense — they are *completed features awaiting integration*. Deleting them wastes prior work. Keeping them costs compile time but no runtime. The right action is to leave them alone and integrate them when their respective phases are scheduled.
