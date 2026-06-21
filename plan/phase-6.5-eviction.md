# Phase 6.5: Session Eviction to CPU + Memory Pressure Handling

---
**Status**: PARTIAL — infrastructure built, not wired
**Last Updated**: 2026-06-21
**Rationale**: CPU-side eviction infrastructure exists but is disconnected from production code. Phase A/B cleanup removed dead methods (remaining_bytes, clear, PageLocation::Cpu variant).
**Actual Deliverables**:
- [x] `CpuPagePool` in `infers-kv/eviction.rs` — implemented with budget tracking
- [x] `PagedKvManager::evict_sequence` / `restore_sequence` — implemented in kv/manager.rs
- [x] Scheduler memory pressure + LRU eviction policy — implemented in scheduler/pressure.rs
- [x] Unit + integration tests — all pass
- [ ] Wire eviction path: orchestrator must call `engine.evict_session()` before `delete_sequence()` (currently bypassed — pages are freed without GPU→CPU copy)
- [ ] Wire restoration path: orchestrator must call `engine.restore_session()` when re-admitting evicted sequences
---

**Duration:** 1 week  
**Goal:** Implement session-level KV cache eviction to CPU and memory-pressure-driven eviction policy.

## Motivation

Phase 6 deferred two deliverables: session eviction to CPU/SSD (8) and memory pressure handling (9). No later phase picks them up. The scaffolding already exists — `PageLocation::Cpu`, `SessionState::Evicted`, and the `Decoding→Evicted→Prefilling` lifecycle transitions. This phase wires the data movement and policy.

## Design

### CpuPagePool (infers-kv)

A CPU-side storage pool for evicted KV page data. Since the actual GPU page data lives in `infers-backend-native`'s `PagedKvCache`, the `CpuPagePool` stores `Vec<u8>` blobs (one per evicted page) and tracks memory usage against a budget.

```rust
pub struct CpuPagePool {
    storage: Vec<Option<Vec<u8>>>,  // page_id → data
    used_bytes: usize,
    max_bytes: usize,
    page_bytes: usize,
}
```

Methods: `store(page_id, data)`, `retrieve(page_id) → Option<Vec<u8>>`, `remove(page_id)`, `num_evicted()`, `used_bytes()`, `remaining_bytes()`.

### Eviction API (PagedKvManager)

The manager coordinates eviction: given a sequence ID and per-page data from the GPU, it stores the data in `CpuPagePool` and marks the pages as evicted:

```rust
pub fn evict_sequence(&mut self, seq_id: SequenceId, page_data: Vec<Vec<u8>>) -> Result<EvictedSequence>
pub fn restore_sequence(&mut self, evicted: EvictedSequence) -> Result<SequenceId>
```

`evict_sequence` frees GPU pages back to the pool and records the mapping in an `eviction_table`. `restore_sequence` re-allocates pages and returns the CPU data for the backend to copy back to GPU.

### LRU Eviction Policy (scheduler)

The scheduler monitors `PagedKvManager::pool_utilization()` and when it exceeds a threshold (e.g., >90%), selects the LRU inactive session for eviction:

```rust
pub fn evict_idle_session(&mut self) -> Result<()>
pub fn handle_memory_pressure(&mut self) -> Result<()>
```

Restoration is triggered automatically when an evicted session would be scheduled for decode.

## File Changes

```
crates/kv/src/
  eviction.rs         # NEW: CpuPagePool, EvictedSequence
  manager.rs          # MODIFY: add evict_sequence(), restore_sequence(), eviction_table
  lib.rs              # MODIFY: export new types

crates/scheduler/src/
  pressure.rs         # NEW: MemoryPressure monitor, LRU eviction policy
  scheduler.rs        # MODIFY: integrate pressure handling
  lib.rs              # MODIFY: export pressure module
```

## Deliverables

- [ ] `CpuPagePool` in `infers-kv/eviction.rs` — CPU storage for evicted page data
- [ ] `PagedKvManager::evict_sequence` / `restore_sequence` — coordination layer
- [ ] Scheduler memory pressure + LRU eviction policy
- [ ] Unit + integration tests

## Deferred (out of scope)

Actual GPU→CPU data copy via `cudaMemcpyAsync`. The `CpuPagePool::store()` accepts `Vec<u8>` from the caller — the backend (which owns the GPU buffers and CUDA streams) will call this after performing the GPU copy. This phase builds the CPU-side infrastructure and policy; GPU data movement is a follow-up integration task in `infers-backend-native`.
