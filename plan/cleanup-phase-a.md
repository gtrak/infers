# Cleanup Phase A: Critical Fixes + Low-Hanging Fruit

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-20
**Blocks**: Nothing (standalone cleanup)
**Blocked by**: Nothing
**Rationale**: The codebase accumulated dead code, broken imports, and unused wrappers across Phases 1–16. The heap loader extraction (Phase 16) left broken imports in 3 files. Several engine wrapper methods were added during exploration but never wired. Debug code was left in the prefill path. This phase removes all verified-dead code with zero design decisions — pure mechanical cleanup.
---

## Tasks

### 1. Fix broken imports (CRITICAL)

**Files**: `crates/backends/native/src/bin/infer.rs`, `crates/backends/native/tests/smoke_test.rs`, `crates/backends/native/tests/smoke_test_mmap_only.rs`

The heap loader extraction moved `shard_weights_tp` and `load_safetensors` to `crates/model-loader-heap/` but the infer binary and smoke tests still import from `infers_model`.

**Changes**:
1. Add `infers-model-loader-heap` as a dev-dependency of `infers-backend-native`
2. In all 3 files, change:
   - `use infers_model::sharding::shard_weights_tp;` → `use infers_model_loader_heap::shard_weights_tp;`
   - `use infers_model::{load_safetensors, strip_language_model_prefix, build_main_layers};` → split into:
     - `use infers_model_loader_heap::load_safetensors;`
     - `use infers_model::{strip_language_model_prefix, build_main_layers};`

**Complexity**: XS
**Timebox**: 10 min
**Acceptance**: `cargo check --release -p infers-backend-native` compiles. All 3 files resolve imports.

---

### 2. Remove duplicate test module in model-loader-heap

**File**: `crates/model-loader-heap/src/lib.rs`

The `shard_weights_tp()` function body contains a nested `#[cfg(test)] mod tests { ... }` block (lines ~369–444) that duplicates the file-level test module (lines 622–697). The inner module is structurally wrong — tests run inside a function's scope.

**Changes**:
1. Delete the `#[cfg(test)] mod tests { ... }` block inside `shard_weights_tp()` (lines ~369–444)
2. Keep the file-level test module (lines 622–697) untouched

**Complexity**: XS
**Timebox**: 5 min
**Acceptance**: `cargo test --release -p infers-model-loader-heap` passes. Only one test module exists.

---

### 3. Remove ad-hoc debug logits dump from prefill path

**File**: `crates/backends/native/src/prefill.rs`

Lines 321–343 contain a DEBUG block that copies the full logits tensor from GPU to CPU on every prefill call, sorts them, and prints top-5 to stderr via `eprintln!`. This is wrong for two reasons:
1. **Runs unconditionally** — not gated behind `INFERS_DUMP_DIR` or any env var, so it pays the GPU→CPU copy cost on every prefill even in production
2. **Redundant** — `probe::dump("final.logits")` already dumps the full logits tensor when `INFERS_DUMP_DIR` is set (engine.rs lines 946, 1350). The top-5 summary is a post-processing view that belongs in the Python oracle comparison tools, not in the Rust engine.

**Changes**:
1. Delete the `// DEBUG: dump top-5 logits for the last token position` block (lines 321–343)
2. If the top-5 summary view is useful, add it as a Python post-processing step in `tests/compare/` that reads the probe dump files

**Complexity**: XS
**Timebox**: 5 min
**Acceptance**: `cargo check --release -p infers-backend-native` compiles. No unconditional `eprintln!` in production code paths. `probe::dump("final.logits")` still works via `INFERS_DUMP_DIR`.

---

### 4. Clean up dead code in engine.rs

**File**: `crates/backends/native/src/engine.rs`

Remove the following dead items:

1. **`_mmap_registries` field** (line 96): Always `Vec::new()`. Remove from struct definition and both constructors (`new()` line 282, `new_from_mmap()` line 368).
2. **`fp8_quantize_and_write()` method** (lines 489–513): Zero callers. Remove entire method.
3. **`fp8_dequantize_and_read()` method** (lines 519–541): Zero callers. Remove entire method.
4. **`matmul_int4()` method** (lines 547–568): Zero callers. Remove entire method.

**Complexity**: S
**Timebox**: 20 min
**Acceptance**: `cargo check --release -p infers-backend-native` compiles.

---

### 5. Clean up dead code in attention.rs

**File**: `crates/backends/native/src/attention.rs`

1. **`_add_kernel` unused parameters** in `forward_paged()` (line 821) and `decode_forward_paged()` (line 1266): Remove the `_add_kernel: &CudaFunction` parameter from both function signatures. Update callers in `engine.rs` to stop passing the argument.
2. **Duplicate doc comment line** (line 812): Remove bare `Paged prefill attention: writes K/V to paged cache, uses per-head GEMM.` line.

**Complexity**: S
**Timebox**: 15 min
**Acceptance**: `cargo check --release -p infers-backend-native` compiles. No unused parameters.

---

### 6. Clean up dead code in upload.rs

**File**: `crates/backends/native/src/upload.rs`

1. **`extract_int4()` helper** (lines 209–223): Only called by `dequantize_int4_to_bf16()`. Remove.
2. **`dequantize_int4_to_bf16()` function** (lines 225–286): Zero production callers — only called by its own `#[cfg(test)]` module. Remove function + its test module.

**Complexity**: XS
**Timebox**: 10 min
**Acceptance**: `cargo check --release -p infers-backend-native` compiles.

---

### 7. Remove entire memory.rs module from cuda crate

**Files**: `crates/cuda/src/memory.rs`, `crates/cuda/src/lib.rs`

The `GpuAllocator` and `AllocInfo` types have zero external callers. The module is self-contained with its own test suite.

**Changes**:
1. Delete `crates/cuda/src/memory.rs`
2. Remove `pub mod memory;` from `crates/cuda/src/lib.rs`

**Complexity**: XS
**Timebox**: 5 min
**Acceptance**: `cargo check --release -p infers-cuda` compiles. No references to `GpuAllocator` or `AllocInfo`.

---

### 8. Clean up dead methods in cuda crate

**Files**: `crates/cuda/src/gemm.rs`, `crates/cuda/src/context.rs`, `crates/cuda/src/nccl.rs`, `crates/cuda/src/memcpy2d.rs`, `crates/cuda/src/kernels.rs`

Remove the following dead methods:

1. **`gemm.rs`**: `matmul_f32()` (lines 56–64), `matmul_fp16()` (lines 77–86) — zero callers
2. **`context.rs`**: `CudaRuntime::new_stream()` (lines 49–53) — zero callers
3. **`nccl.rs`**: `rank()` (lines 50–52), `world_size()` (lines 55–57), `comms()` (lines 65–67) — zero callers
4. **`memcpy2d.rs`**: `clone_dtoh_raw()` (lines 78–92) — zero callers
5. **`kernels.rs`**: `KernelRegistry::len()` (lines 50–52), `is_empty()` (lines 55–57), `LoadedKernelRegistry::launch()` (lines 160–172) — zero callers
6. **`kernels.rs`**: Remove 3 dead kernel registrations from `register_infers_kernels()`: `infers_argmax_f32` (line 71), `infers_gdn_update_bf16` (line 77), `infers_gdn_prefill_bf16` (line 78)

**Complexity**: S
**Timebox**: 20 min
**Acceptance**: `cargo check --release -p infers-cuda` compiles.

---

### 9. Remove unused thiserror dependency

**File**: `crates/cuda/Cargo.toml`

Remove `thiserror = { workspace = true }` from `[dependencies]`. The crate uses `anyhow` exclusively.

**Complexity**: XS
**Timebox**: 2 min
**Acceptance**: `cargo check --release -p infers-cuda` compiles.

---

### 10. Clean up dead code in scheduler crate

**Files**: `crates/scheduler/src/scheduler.rs`, `crates/scheduler/src/lifecycle.rs`, `crates/scheduler/src/queue.rs`, `crates/scheduler/src/pressure.rs`, `crates/scheduler/src/batch.rs`, `crates/scheduler/src/session.rs`

Remove the following dead items:

1. **`scheduler.rs`**: `select_and_evict_idle_session()` (lines 136–143) — zero production callers
2. **`lifecycle.rs`**: `pause_session()` (lines 67–69), `resume_session()` (lines 72–74) — zero production callers
3. **`queue.rs`**: `peek()` (lines 132–135), `clear()` (lines 147–149), `drain()` (lines 152–154) — zero production callers
4. **`queue.rs`**: `Request.session_id` field (line 70) — always set to 0, never read
5. **`pressure.rs`**: `PressureAction` enum (lines 26–36) — zero callers. Also remove re-export from `lib.rs`
6. **`batch.rs`**: `BatchBuilder.max_tokens_per_batch` field (line 32) — never read. Remove field + constructor parameter
7. **`session.rs`**: `Session::total_tokens()` (lines 79–81) — zero callers

**Complexity**: S
**Timebox**: 20 min
**Acceptance**: `cargo check --release -p infers-scheduler` compiles.

---

### 11. Clean up dead code in server crate

**Files**: `crates/server/src/orchestrator.rs`, `crates/server/src/handlers/chat.rs`, `crates/server/src/main.rs`

Remove the following dead items:

1. **`orchestrator.rs`**: `active_count()` (lines 267–270), `pending_count()` (lines 273–276), `is_busy()` (lines 279–282) — zero callers
2. **`chat.rs`**: `should_use_tools()` (lines 259–276) — zero callers, `#[allow(dead_code)]`
3. **`main.rs`**: `kv_cache_dtype` lines (299–300) — converted then discarded with `let _ =`

**Complexity**: XS
**Timebox**: 10 min
**Acceptance**: `cargo check --release -p infers-server` compiles.

---

### 12. Clean up dead code in model crate

**Files**: `crates/model/src/config.rs`, `crates/model/src/mmap.rs`, `crates/model/src/loader.rs`

1. **`config.rs`**: `num_gdn_layers()` (lines 230–234) — zero external callers. Remove method.
2. **`mmap.rs`**: Unnecessary `unsafe` block at line 386 — remove `unsafe` wrapper.
3. **`mmap.rs`**: Stale doc comment referencing `shard_weights_tp` at line 499 — remove broken rustdoc link.
4. **`loader.rs`**: Stale comments referencing `shard_weights_tp` at lines 331, 375 — update to remove references.

**Complexity**: XS
**Timebox**: 10 min
**Acceptance**: `cargo check --release -p infers-model` compiles.

---

### 13. Clean up dead code in KV crate

**Files**: `crates/kv/src/manager.rs`, `crates/kv/src/pool.rs`

1. **`manager.rs`**: `kv_dim()` (lines 367–370) — zero external callers. Remove method.
2. **`pool.rs`**: `is_empty()` (lines 110–113) — zero external callers. Remove method.
3. **`pool.rs`**: `get_mut()` (lines 134–137) — zero external callers. Remove method.

**Complexity**: XS
**Timebox**: 10 min
**Acceptance**: `cargo check --release -p infers-kv` compiles.

---

### 14. Update lat.md

**Files**: `lat.md/lat.md`, `lat.md/misc.md`

Update documentation to reflect removed code:
- Remove references to deleted methods in tech debt section
- Update `#[allow(dead_code)]` count in lat.md
- Run `lat check` to verify all links pass

**Complexity**: XS
**Timebox**: 10 min
**Acceptance**: `lat check` passes.

---

## Execution Order

```
Task 1  (broken imports)       — CRITICAL, do first
Task 2  (duplicate tests)      — CRITICAL
Task 3  (debug dump)           — perf fix
Tasks 4-6 (engine/prefill/upload) — serialize (same crate)
Tasks 7-9 (cuda crate)         — can parallel with 4-6
Tasks 10-11 (scheduler/server) — serialize
Tasks 12-13 (model/kv)         — can parallel with 10-11
Task 14 (docs)                 — last
```

## Deferred to Phase B

- `paged_kv_read` kernel handle + dispatch function — user decision pending
- `evict_session()` + `restore_session()` on engine — user decision pending
- `_mmap_registries` field — user decision pending

## Estimated Total Time

~2 hours across 14 tasks.

## Success Criteria

- [ ] `cargo check --release` passes for all workspace crates
- [ ] `cargo test --release` passes for all crates with tests
- [ ] No broken imports
- [ ] No debug code in production paths
- [ ] `lat check` passes
