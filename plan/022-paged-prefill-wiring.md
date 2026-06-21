# Phase 17: Wire Paged Prefill into Server

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-19
**Blocks**: Prefix caching, unified KV state, memory efficiency
**Blocked by**: Nothing
**Rationale**: The orchestrator uses the non-paged `prefill()` path (flat KV buffer) while decode already uses `decode_paged()` (paged KV). This creates two problems: (1) prefix caching can't work because flat-buffer prefill doesn't populate the paged KV page table, and (2) there's an implicit state mismatch between prefill's flat KV and decode's paged KV. The fix is to wire `prefill_paged()` into the orchestrator and share the scheduler's `PagedKvManager` with the engine.
---

## Current State

| Component | Paged KV status |
|-----------|----------------|
| Scheduler | Owns `PagedKvManager`, creates sequences, allocates pages |
| Engine | Has `paged_kv_manager: Option<PagedKvManager>` (always `None` in server) |
| Engine | Has `paged_kv_caches: Vec<Vec<PagedKvCache>>` (GPU buffers, always empty in server) |
| Orchestrator prefill | Calls `engine.prefill()` (non-paged, flat KV) |
| Orchestrator decode | Calls `engine.decode_paged()` (paged KV) |
| Smoke test / infer binary | Calls `engine.init_paged()` then `engine.prefill_paged()` (correct pattern) |

**Problem**: The engine creates its own `PagedKvManager` in `init_paged()`, which is a DIFFERENT instance from the scheduler's. Two managers = two page pools = sequence IDs don't cross-reference. The scheduler allocates pages in its pool; the engine allocates in its own. They can't share cached pages.

## Architecture Decision

**Move `PagedKvManager` ownership out of the engine. Pass it as `&mut PagedKvManager` to `prefill_paged()` and `decode_paged()`.**

Why:
- The scheduler already owns the canonical `PagedKvManager` and creates sequences in it
- The engine only needs `page_size()`, `append_page()`, `block_table()`, `num_tokens()` — thin operations on the manager
- The engine's GPU-side `PagedKvCache` buffers stay in the engine (they hold `CudaSlice<bf16>`)
- No `Arc<Mutex<>>` needed — the orchestrator holds `&mut scheduler` and `&mut engine` as separate fields, so it can pass `&mut scheduler.kv_manager` to engine methods

**Borrow checker**: The orchestrator's `step()` method accesses `self.scheduler` and `self.engine` sequentially. Passing `&mut self.scheduler.kv_manager` to `self.engine.prefill_paged()` borrows two different fields simultaneously — this is safe and compiles.

## Task Breakdown

### Commit 1: Expose scheduler's kv_manager

**Files**: `crates/scheduler/src/scheduler.rs`

Add a public accessor to `RoundRobinScheduler`:

```rust
/// Access the paged KV manager (for sharing with the engine).
pub fn kv_manager(&mut self) -> &mut PagedKvManager {
    &mut self.kv_manager
}
```

Returns `&mut` (not `&`) because `append_page()` and `delete_sequence()` need mutable access.

**Complexity**: XS
**Timebox**: 5 min
**Acceptance**: `cargo check --release -p infers-scheduler` compiles.

---

### Commit 2: Refactor engine to accept external kv_manager

**Files**: `crates/backends/native/src/engine.rs`

**Changes to `ForwardEngine`:**

1. **Remove `paged_kv_manager` field** from the struct. The engine no longer owns a `PagedKvManager`.

2. **Rename `init_paged()`** to `init_paged_caches()` — it only creates `paged_kv_caches` (GPU buffers), not the manager:

```rust
pub fn init_paged_caches(
    &mut self,
    total_pages: usize,
    page_size: usize,
) -> Result<()> {
    let num_gpus = self.weights.len();
    let kv_dim_per_gpu = (self.config.num_key_value_heads / num_gpus) * self.config.head_dim;

    let caches: Vec<Vec<PagedKvCache>> = (0..num_gpus)
        .map(|_| {
            (0..self.config.num_hidden_layers)
                .map(|_| PagedKvCache::new(total_pages, page_size, kv_dim_per_gpu))
                .collect()
        })
        .collect();

    self.paged_kv_caches = caches;
    tracing::info!(
        "Paged KV caches initialized: {} pages, page_size={}, {} layers",
        total_pages, page_size, self.config.num_hidden_layers
    );
    Ok(())
}
```

3. **Change `prefill_paged()` signature** — add `kv_manager: &mut PagedKvManager` parameter:

```rust
pub fn prefill_paged(
    &mut self,
    _stream: &Arc<CudaStream>,
    token_ids: &[u32],
    seq_id: infers_kv::SequenceId,
    kv_manager: &mut PagedKvManager,  // NEW
    sampling_config: &infers_scheduler::SamplingConfig,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<(usize, u32)>
```

Inside the function, replace `self.paged_kv_manager.as_mut()` with the parameter:
```rust
// Before:
let manager = self.paged_kv_manager.as_mut()
    .ok_or_else(|| anyhow::anyhow!("Paged KV system not initialized"))?;

// After:
let manager = kv_manager;
```

4. **Change `decode_paged()` signature** — add `kv_manager: &mut PagedKvManager` parameter:

```rust
pub fn decode_paged(
    &mut self,
    _stream: &Arc<CudaStream>,
    token_id: u32,
    position: u32,
    seq_id: infers_kv::SequenceId,
    kv_manager: &mut PagedKvManager,  // NEW
    sampling_config: &infers_scheduler::SamplingConfig,
    token_history: &[u32],
    num_prompt_tokens: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<u32>
```

Inside the function, replace `self.paged_kv_manager.as_mut()` with the parameter.

5. **Remove `create_sequence()` from engine** — the scheduler already creates sequences. Remove:
```rust
pub fn create_sequence(&mut self) -> infers_kv::SequenceId {
    self.paged_kv_manager
        .as_mut()
        .map(|m| m.create_sequence())
        .unwrap_or(0)
}
```

6. **Update `new()` constructor** — remove `paged_kv_manager: None` initialization.

**Complexity**: M
**Timebox**: 1 hour
**Acceptance**: `cargo check --release -p infers-backend-native` compiles. All internal engine tests pass.

---

### Commit 3: Wire orchestrator to paged prefill

**Files**: `crates/server/src/orchestrator.rs`

**Changes to `step()` prefill section** (lines 157-188):

```rust
// Step 4: Handle prefill batch
if let Some(prefill_batch) = &work.prefill_batch {
    for &seq_id in &prefill_batch.sessions {
        let tokens = &prefill_batch.input_tokens;
        tracing::debug!(
            "Prefilling session {} with {} tokens",
            seq_id,
            tokens.len()
        );

        // Get or create per-session RNG
        if !self.session_rngs.contains_key(&seq_id) {
            let session = self.scheduler.active_sessions.iter()
                .find(|s| s.id == seq_id)
                .unwrap();
            let seed = session.sampling_config.seed
                .unwrap_or_else(infers_backend_native::sample::random_seed);
            self.session_rngs.insert(seq_id, Xoshiro256PlusPlus::from_seed(seed));
        }
        let rng = self.session_rngs.get_mut(&seq_id).unwrap();

        // Get sampling config from session
        let sampling_config = self.scheduler.active_sessions.iter()
            .find(|s| s.id == seq_id)
            .map(|s| s.sampling_config.clone())
            .unwrap_or_default();

        // Paged prefill — uses scheduler's kv_manager
        let kv_mgr = self.scheduler.kv_manager();
        let (pages_used, sampled) = self.engine.prefill_paged(
            &self.stream, tokens, seq_id, kv_mgr,
            &sampling_config, rng,
        )?;

        tracing::debug!(
            "Prefill complete: {} pages used, sampled={}",
            pages_used, sampled
        );

        // Update session state
        if let Some(session) = self.scheduler.active_sessions.iter_mut()
            .find(|s| s.id == seq_id)
        {
            let _ = lifecycle::finish_prefill(session);
            session.tokens.push(sampled);
            session.num_generated_tokens = session.num_generated_tokens.saturating_add(1);
        }

        // Send generated token
        if let Some(tx) = self.response_tx.get(&seq_id) {
            let _ = tx.try_send(sampled);
        }
    }
}
```

**Changes to decode section** — add `kv_manager` parameter to `decode_paged()`:

```rust
let kv_mgr = self.scheduler.kv_manager();
let sampled = self.engine.decode_paged(
    &self.stream, token_id, position, seq_id, kv_mgr,
    sampling_config, &session.tokens, session.num_prompt_tokens, rng,
)?;
```

**Remove TODO comment** (line 168): `// TODO: switch to prefill_paged() once paged KV is initialized in server startup`

**Complexity**: M
**Timebox**: 30 min
**Acceptance**: `cargo check --release -p infers-server` compiles. No TODO comments remain.

---

### Commit 4: Wire server main.rs — init paged caches

**Files**: `crates/server/src/main.rs`

After creating the engine and before creating the orchestrator, initialize paged KV caches:

```rust
// Step 5.5: Initialize paged KV caches on the engine
engine.init_paged_caches(total_pages, page_size)?;
tracing::info!("Paged KV caches initialized on engine");
```

Remove the `TODO: Wire kv_cache_dtype` comment and the `let _ = &kv_cache_dtype;` line — the dtype is already handled by `PagedKvManager` in the scheduler.

**Complexity**: XS
**Timebox**: 10 min
**Acceptance**: Server starts without errors. `cargo check --release -p infers-server` compiles.

---

### Commit 5: Update smoke test and infer binary

**Files**: `crates/backends/native/tests/smoke_test.rs`, `crates/backends/native/src/bin/infer.rs`

Update both to use the new API:

**Smoke test:**
```rust
// Before:
engine.init_paged(num_pages, page_size, max_cache_bytes)?;
let seq_id = engine.create_sequence();
let (pages_used, first_token) = engine.prefill_paged(&stream, &token_ids, seq_id, &sampling_config, &mut rng)?;

// After:
engine.init_paged_caches(num_pages, page_size)?;
let mut kv_manager = PagedKvManager::new(num_pages, page_size, num_kv_heads, head_dim, max_cache_bytes, eviction_max_bytes);
let seq_id = kv_manager.create_sequence();
let (pages_used, first_token) = engine.prefill_paged(&stream, &token_ids, seq_id, &mut kv_manager, &sampling_config, &mut rng)?;
```

Same pattern for `decode_paged()` calls.

**infer binary:** Same changes — create a local `PagedKvManager`, pass it to engine methods.

**Complexity**: S
**Timebox**: 30 min
**Acceptance**: Smoke test passes. `infer` binary produces tokens.

---

### Commit 6: Remove dead code

**Files**: `crates/backends/native/src/engine.rs`

Remove from `ForwardEngine`:
- `paged_kv_manager: Option<PagedKvManager>` field
- `use infers_kv::PagedKvManager;` import (if no longer needed)
- The old `init_paged()` method (replaced by `init_paged_caches()`)
- The `create_sequence()` method

**Complexity**: XS
**Timebox**: 10 min
**Acceptance**: `cargo check --release` passes with no dead code warnings.

---

### Commit 7: Documentation + lat.md

**Files**: `lat.md/lat.md`, `plan/README.md`

| # | Task | Detail |
|---|------|--------|
| 1 | Update lat.md paged prefill section | Document the shared kv_manager architecture |
| 2 | Update plan/README.md | Add Phase 17 entry |
| 3 | Run `lat check` | Verify all links pass |

**Complexity**: XS
**Timebox**: 15 min
**Acceptance**: `lat check` passes.

---

## Key Design Decisions

### KD1: Engine doesn't own PagedKvManager

The `PagedKvManager` is CPU-side bookkeeping (page allocation, block tables, prefix cache). The engine only needs it during `prefill_paged()` and `decode_paged()` — not permanently. Passing it as a parameter keeps ownership clear and eliminates the duplicate-manager problem.

### KD2: GPU-side PagedKvCache stays in engine

`PagedKvCache` holds `CudaSlice<bf16>` — GPU-resident buffers. These MUST stay in the engine because they're allocated per-GPU and used by CUDA kernels. Only the CPU-side manager is shared.

### KD3: Scheduler exposes &mut kv_manager

The scheduler needs `&mut` access to its manager for `create_sequence()` and `append_page()`. The engine also needs `&mut` for `append_page()`. Since the orchestrator calls these sequentially (never concurrently), there's no contention.

### KD4: Keep init_paged_caches() on engine

The GPU-side `PagedKvCache` buffers are engine-internal. Only the engine can allocate them (they're `CudaSlice`). The manager is external; the caches are internal.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Borrow checker conflict between scheduler and engine | Low | Medium | Separate fields in orchestrator — `self.scheduler.kv_manager()` and `self.engine.prefill_paged()` borrow different fields |
| Scheduler and engine disagree on page_size | Low | High | Both use the same `page_size` from CLI args — passed to both `PagedKvManager::new()` and `init_paged_caches()` |
| Smoke test breaks from API change | High | Low | Mechanical update — same logic, different parameter passing |
| Prefix caching still doesn't work | Low | Medium | This phase wires paged prefill; prefix caching requires additional `seal_and_cache()` calls (future work) |

## Success Criteria

- [ ] `prefill_paged()` uses scheduler's `PagedKvManager` (same instance as decode)
- [ ] `decode_paged()` uses scheduler's `PagedKvManager` (same instance as prefill)
- [ ] Engine no longer owns a `PagedKvManager`
- [ ] Orchestrator TODO comment removed
- [ ] Smoke test passes with new API
- [ ] `infer` binary produces tokens with new API
- [ ] Server starts and serves requests
- [ ] `cargo check --release` passes for all workspace crates
- [ ] `lat check` passes
