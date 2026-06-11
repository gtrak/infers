# Phase 6.6: Eviction Wiring — GPU Data Movement Through Backend

---
**Status**: NOT DONE
**Last Updated**: 2026-06-11
**Rationale**: Blocked by Phase 6 (continuous batching). GPU data movement for eviction not wired.
**Actual Deliverables**:
- [ ] `BackendEvictionStore`
- [ ] `mark_evicted()` / `allocate_for_restore()` in `PagedKvManager`
- [ ] `evict_session()` / `restore_session()` in `ForwardEngine`
---

**Duration:** 3 days  
**Goal:** Wire the eviction path through `infers-backend-native`, connecting the scheduler's eviction policy to actual GPU→CPU data movement.

## Problem

Phase 6.5 built the CPU-side eviction infrastructure (`CpuPagePool`, `PagedKvManager::evict_sequence/restore_sequence`, scheduler `handle_memory_pressure`). But the actual GPU data lives in per-layer `PagedKvCache` buffers in `infers-backend-native`, and no code copies it to/from CPU.

The CpuPagePool stores one `Vec<u8>` per `PageId`, but each page has per-layer K/V data across all full-attention layers. The backend owns the GPU buffers and CUDA streams — it must orchestrate the copy.

## Design

### BackendEvictionStore (backend-native)

A simple per-layer, per-page data store separate from `CpuPagePool`. Since the kv crate doesn't know about layers, the backend manages its own evicted data:

```rust
pub struct BackendEvictionStore {
    /// Per-layer map: PageId → Vec<u8> (page data)
    layers: Vec<HashMap<PageId, Vec<u8>>>,
    num_layers: usize,
    page_bytes: usize,
}
```

### PagedKvManager additions (infers-kv)

Add `mark_evicted()` and `allocate_for_restore()` as lightweight metadata-only APIs:

- `mark_evicted(seq_id) → EvictedSequence` — frees GPU pages, deletes the sequence, returns page table snapshot. No data storage.
- `allocate_for_restore(evicted: &EvictedSequence) → SequenceId` — creates new sequence, allocates matching pages. No data retrieval.

These complement the existing `evict_sequence/restore_sequence` which combine data storage + metadata.

### ForwardEngine additions (backend-native)

Add methods to the engine that own the actual GPU→CPU data movement:

- `evict_session(seq_id, kv_manager, store) → Result<()>` — reads each page from all layers' `PagedKvCache` GPU buffers via `cudaMemcpyAsync`, stores in `BackendEvictionStore`, calls `kv_manager.mark_evicted()`.
- `restore_session(evicted, kv_manager, store) → Result<()>` — calls `kv_manager.allocate_for_restore()`, retrieves data from `BackendEvictionStore`, copies back to GPU buffers via `cudaMemcpyAsync`.

## File Changes

```
crates/kv/src/manager.rs       # MODIFY: add mark_evicted(), allocate_for_restore()
crates/backends/native/src/
  eviction.rs                   # NEW: BackendEvictionStore
  engine.rs                     # MODIFY: add evict_session(), restore_session()
  lib.rs                        # MODIFY: add pub mod eviction;
```

## Deferred

Tying `schedule()` output to `ForwardEngine::evict_session()`. That belongs in the server crate or an orchestration layer, not in either individual crate.
