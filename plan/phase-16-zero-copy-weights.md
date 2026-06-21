# Phase 16: Zero-Copy Weight Streaming (Eliminate DRAM Residency)

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-19
**Blocks**: DRAM usage reduction, faster cold-start
**Blocked by**: Nothing
**Rationale**: The current weight loading pipeline copies every tensor from mmap'd safetensors files into heap memory (`Bytes::copy_from_slice()`), then uploads from heap to GPU. For the Qwen3.6-27B INT4 model (~14GB), this means 14GB of permanent heap residency + 14GB GPU VRAM = 28GB total. The heap copy is unnecessary — the data already exists on disk via mmap. Eliminating it halves peak DRAM usage and makes the GPU the sole long-term weight store.
---

## Current Data Flow

```
Disk (safetensors)
  → mmap (OS page cache, evictable DRAM)
  → Bytes::copy_from_slice() (permanent heap DRAM — THE PROBLEM)
  → clone_htod() (GPU VRAM via internal temp alloc)
  → GpuWeightCache (GPU VRAM, permanent)
```

**Problem**: Step 3 copies ~14GB into heap memory and holds it permanently via `WeightData.data: Bytes`. Step 4 uploads to GPU but the heap copy persists for the process lifetime. Net: 14GB heap + 14GB GPU = 28GB.

## Target Data Flow

```
Disk (safetensors)
  → mmap (OS page cache, evictable DRAM)
  → PinnedHostBuffer (256MB staging, page-locked DRAM)
  → cudaMemcpyAsync (DMA to GPU VRAM)
  → GpuWeightCache (GPU VRAM, permanent)
  → mmap dropped, pages evicted from page cache
```

**Result**: Only 256MB pinned staging in DRAM + 14GB GPU VRAM. mmap pages are evictable and released after upload.

## Architecture

### New Type: `MmapTensor`

Zero-copy reference to a tensor stored in a memory-mapped safetensors file.

```rust
pub struct MmapTensor {
    mmap: Arc<Mmap>,     // keeps the mmap alive
    data_ptr: *const u8, // pointer into mmap region
    data_len: usize,     // byte length
    shape: Vec<usize>,
    dtype: WeightDtype,
    name: String,
}
```

`MmapTensor` implements `Deref<Target = [u8]>` — callers can use it exactly like `&[u8]`. The `Arc<Mmap>` ensures the underlying file mapping stays alive as long as any tensor references it.

**Safety**: The `*const u8` pointer is derived from the mmap'd region while the `Mmap` is alive. The `Arc<Mmap>` guarantees the mapping outlives all `MmapTensor` instances. `MmapTensor` is `Send` (mmap pointers are valid across threads) but not `Sync`.

### New Type: `MmapWeightRegistry`

Drop-in alternative to `WeightRegistry` that holds mmap references instead of heap copies.

```rust
pub struct MmapWeightRegistry {
    pub tensors: HashMap<String, MmapTensor>,
    pub int4_companions: HashMap<String, MmapCompanions>,
    pub layers: Vec<MmapLayerWeights>,
    pub embedding: Option<MmapTensor>,
    pub norm: Option<MmapTensor>,
    pub lm_head: Option<MmapTensor>,
    pub mtp: Option<MmapMtpWeights>,
    _mmaps: Vec<Arc<Mmap>>,  // prevent mmap from being dropped
}

pub struct MmapCompanions {
    pub qzeros: MmapTensor,
    pub scales: MmapTensor,
}
```

Mirrors `WeightRegistry` structure but all `WeightData` fields become `MmapTensor`. The `_mmaps` field holds all mmap handles alive.

### New Type: `PinnedHostBuffer`

Page-locked (pinned) host memory for fast DMA transfers to GPU.

```rust
pub struct PinnedHostBuffer {
    ptr: *mut u8,
    size: usize,
}
```

Allocates via `cudaHostAlloc()` (page-locked). RAII: `Drop` calls `cudaFreeHost()`. One buffer is allocated at startup and reused for all weight uploads.

**Size**: 256MB default. Large enough for the biggest single weight (embedding table ~300MB for BF16, ~75MB for INT4). Configurable via `INFERS_PINNED_BUFFER_MB` env var.

**Key constraint**: `cudaHostAlloc` requires a CUDA context to be active. Must be created after CUDA initialization.

### New Function: `load_safetensors_mmap()`

Returns `MmapWeightRegistry` without copying any tensor data to heap.

```rust
pub fn load_safetensors_mmap(model_dir: &Path) -> Result<MmapWeightRegistry>
```

For each safetensors file:
1. `memmap2::Mmap::map(&file)` — zero-copy file mapping
2. `SafeTensors::deserialize(&mmap)` — parse header only (no data copy)
3. For each tensor: create `MmapTensor` with pointer into mmap region
4. Store `Arc<Mmap>` in `_mmaps` to keep mapping alive

The mmap `SafeTensors` iterator gives `&[u8]` slices pointing directly into the mmap. We store the raw pointer + length without copying.

### New Function: `GpuWeightCache::new_from_mmap()`

Streams weights from mmap → pinned buffer → GPU.

```rust
pub fn new_from_mmap(
    stream: &Arc<CudaStream>,
    registry: &MmapWeightRegistry,
    pinned: &PinnedHostBuffer,
) -> Result<Self>
```

For each weight in the registry:
1. Get `&[u8]` from `MmapTensor` (zero-copy deref)
2. **BF16**: Cast mmap `&[u8]` to `&[bf16]` via `slice::from_raw_parts` (zero-copy — mmap bytes ARE bf16 bits), then `clone_htod()` directly
3. **FP16**: Copy mmap bytes into pinned buffer, reinterpret as `&[f16]`, convert to bf16 in pinned buffer, `clone_htod()`
4. **FP32**: Copy mmap bytes into pinned buffer, reinterpret as `&[f32]`, convert to bf16 in pinned buffer, `clone_htod()`
5. **INT4**: Upload qweight bytes directly (cast `&[u8]` → `&[u32]`), upload scales via pinned buffer, upload qzeros via pinned buffer
6. Sync stream after each upload

**Key optimization for BF16 weights**: The mmap bytes ARE the bf16 bits. No conversion needed. We can upload them directly without touching the pinned buffer. This is the common case for INT4 models where qweight and scales are the bulk.

### Modified: `infers-model` exports

Add parallel exports for the mmap path:

```rust
// lib.rs additions
pub mod mmap;  // MmapTensor, MmapWeightRegistry, load_safetensors_mmap
```

The existing `WeightRegistry` path is retained for:
- TP sharding (operates on `WeightRegistry`)
- Test code
- The `infer` binary

### Modified: `infers-backend-native/src/engine.rs`

The `ForwardEngine::new()` constructor continues to accept `Vec<WeightRegistry>` for backward compatibility. Add a new constructor:

```rust
pub fn new_from_mmap(
    config: Arc<ModelConfig>,
    mmap_registries: Vec<MmapWeightRegistry>,
    contexts: Vec<Arc<CudaContext>>,
    kernel_registry: KernelRegistry,
    streams: StreamPool,
    pinned: PinnedHostBuffer,
    group_size: usize,
) -> Result<Self>
```

This constructor calls `GpuWeightCache::new_from_mmap()` instead of `GpuWeightCache::new()`.

### Modified: `infers-server/src/main.rs`

Switch the TP loading path to use mmap:

```rust
// Current:
let mut raw_weights = load_safetensors(model_path)?;  // heap copy
strip_language_model_prefix(&mut raw_weights);
let shards = shard_weights_tp(&raw_weights, &config, num_gpus)?;
// ... build registries ...
let engine = ForwardEngine::new(model_config, weight_registries, ...)?;

// New:
let mmap_reg = load_safetensors_mmap(model_path)?;  // no heap copy
let shards = shard_weights_tp_mmap(&mmap_reg, &config, num_gpus)?;
// ... build mmap registries ...
let engine = ForwardEngine::new_from_mmap(model_config, mmap_registries, ..., pinned)?;
```

### TP Sharding for Mmap

The existing `shard_weights_tp()` operates on `WeightRegistry` (heap data). For mmap, we need a parallel function:

```rust
pub fn shard_weights_tp_mmap(
    registry: &MmapWeightRegistry,
    config: &ModelConfig,
    num_gpus: usize,
) -> Result<Vec<MmapShard>>
```

This function is structurally identical to `shard_weights_tp()` but:
- Operates on `MmapTensor` references instead of `WeightData`
- Splits mmap regions by column/row dimension (same logic)
- Creates new `MmapTensor` instances pointing to sub-regions of the same mmap
- No data copying — just pointer arithmetic

For INT4 column-parallel sharding, the split is on the N dimension (rows of qweight). Each GPU gets a contiguous row range — this is a simple offset + length into the mmap.

For row-parallel sharding, the split is on the K/8 dimension (columns of qweight). Each GPU gets a contiguous column range — requires stride-aware slicing.

### Pinned Buffer Management

The `PinnedHostBuffer` is allocated once at startup and reused:

```rust
// In main.rs, after CUDA context creation:
let pinned = PinnedHostBuffer::new(256 * 1024 * 1024)?;  // 256MB

// Passed to ForwardEngine::new_from_mmap()
```

The buffer is used as a temporary staging area for:
1. FP16→BF16 conversion (scales)
2. FP32→BF16 conversion (if needed)
3. Any weight that can't be uploaded directly from mmap

For BF16 weights (the common case), the pinned buffer is NOT used — upload happens directly from mmap.

## Task Breakdown

### Commit 1: `MmapTensor` + `MmapWeightRegistry` types

**Files**: new `crates/model/src/mmap.rs`, modified `crates/model/src/lib.rs`, `crates/model/Cargo.toml`

| # | Task | Detail |
|---|------|--------|
| 1 | Create `mmap.rs` with `MmapTensor` | `Arc<Mmap>`, raw pointer, `Deref<Target=[u8]>`, `Send` impl |
| 2 | Create `MmapCompanions` struct | mirrors `Int4Companions` but with `MmapTensor` |
| 3 | Create `MmapWeightRegistry` struct | mirrors `WeightRegistry` with mmap types |
| 4 | Create `MmapLayerWeights`, `MmapGdnWeights`, etc. | mirror all layer weight structs |
| 5 | Register module in `lib.rs` | `pub mod mmap;` |
| 6 | Unit tests for `MmapTensor` | Create from synthetic mmap, verify deref, verify arc keeps mmap alive |

**Complexity**: M
**Timebox**: 1 hour
**Acceptance**: `cargo check --release -p infers-model` compiles. Unit tests pass.

---

### Commit 2: `load_safetensors_mmap()`

**Files**: `crates/model/src/mmap.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Implement `load_safetensors_mmap()` | mmap file, parse header, create `MmapTensor` references |
| 2 | Implement `strip_language_model_prefix_mmap()` | Same logic as original but on `MmapWeightRegistry` |
| 3 | Implement `build_main_layers_mmap()` | Same structure as `build_main_layers()` with mmap types |
| 4 | Implement `get_weight_mmap()` + `get_weight_or_int4_mmap()` | Mmap equivalents of registry extraction helpers |
| 5 | Unit tests | Load a synthetic safetensors file, verify mmap references point to correct data |

**Complexity**: L
**Timebox**: 2 hours
**Acceptance**: `load_safetensors_mmap()` produces a `MmapWeightRegistry` with correct tensor pointers. No heap copies. `cargo check --release -p infers-model` compiles.

---

### Commit 3: `PinnedHostBuffer`

**Files**: new `crates/cuda/src/pinned.rs`, modified `crates/cuda/src/lib.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Create `PinnedHostBuffer` with `cudaHostAlloc` FFI | Raw CUDA Driver API binding |
| 2 | Implement `as_slice()` / `as_mut_slice()` | Safe access to pinned memory |
| 3 | Implement `copy_from_slice()` | Copy data into pinned buffer |
| 4 | Implement `Drop` | Call `cudaFreeHost` |
| 5 | Implement `Send` | Pinned memory is valid across threads |
| 6 | Register module in `lib.rs` | `pub mod pinned;` |

**Complexity**: S
**Timebox**: 30 min
**Acceptance**: `PinnedHostBuffer::new(1024*1024)` allocates 1MB pinned memory. `as_mut_slice()` returns a valid mutable slice. `Drop` frees memory.

---

### Commit 4: `GpuWeightCache::new_from_mmap()`

**Files**: `crates/backends/native/src/gpu_cache.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Add `new_from_mmap()` constructor | Accept `MmapWeightRegistry` + `PinnedHostBuffer` |
| 2 | Implement BF16 direct upload | Cast `&[u8]` → `&[bf16]` from mmap, `clone_htod()` |
| 3 | Implement FP16→BF16 via pinned buffer | mmap → pinned → convert → GPU |
| 4 | Implement FP32→BF16 via pinned buffer | mmap → pinned → convert → GPU |
| 5 | Implement INT4 upload from mmap | qweight/scales/qzeros direct from mmap |
| 6 | Unit tests (CPU-only) | Verify mmap data reaches GPU cache correctly |

**Complexity**: L
**Timebox**: 2 hours
**Acceptance**: `GpuWeightCache::new_from_mmap()` produces identical GPU cache as `GpuWeightCache::new()` for the same model. `cargo check --release -p infers-backend-native` compiles.

---

### Commit 5: TP sharding for mmap

**Files**: `crates/model/src/sharding.rs`, `crates/model/src/mmap.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Implement `shard_weights_tp_mmap()` | Same split logic as `shard_weights_tp()` but on `MmapTensor` |
| 2 | Implement column-parallel split for mmap | Slice mmap region by row range (INT4 N dimension) |
| 3 | Implement row-parallel split for mmap | Stride-aware column slice (INT4 K/8 dimension) |
| 4 | Implement fused QKV sharding for mmap | Same segment logic, pointer arithmetic |
| 5 | Unit tests | Verify sharded mmap pointers point to correct data ranges |

**Complexity**: L
**Timebox**: 2 hours
**Acceptance**: Sharded mmap produces correct tensor ranges matching the heap-based path.

---

### Commit 6: Wire into server binary

**Files**: `crates/server/src/main.rs`, `crates/backends/native/src/engine.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Add `ForwardEngine::new_from_mmap()` constructor | Uses `GpuWeightCache::new_from_mmap()` |
| 2 | Add mmap loading path in `main.rs` | `load_safetensors_mmap()` → shard → build → `new_from_mmap()` |
| 3 | Allocate `PinnedHostBuffer` after CUDA init | 256MB default |
| 4 | Keep existing path as fallback | `--no-mmap` CLI flag for debugging |
| 5 | Remove debug `eprintln!` in `upload.rs` | The conv1d weight dump code |
| 6 | Integration test | Verify mmap path produces identical tokens as heap path |

**Complexity**: M
**Timebox**: 1 hour
**Acceptance**: Server starts, loads model via mmap, serves requests. Token output matches heap path.

---

### Commit 7: Drop mmap after upload + cleanup

**Files**: `crates/model/src/mmap.rs`, `crates/backends/native/src/gpu_cache.rs`

| # | Task | Detail |
|---|------|--------|
| 1 | Verify mmap handles are dropped after upload | `_mmaps` field in `MmapWeightRegistry` is dropped when registry is consumed |
| 2 | Add `mmap_dropped` flag to `GpuWeightCache` | Log when mmap references are released |
| 3 | Verify page cache eviction | After upload, mmap pages should be evictable |
| 4 | Benchmark: DRAM usage before/after | `free -m` during and after upload |

**Complexity**: XS
**Timebox**: 30 min
**Acceptance**: After `ForwardEngine::new_from_mmap()`, process RSS drops to near-GPU-VRAM-only levels.

---

### Commit 8: Documentation + lat.md

**Files**: `lat.md/lat.md`, `plan/README.md`

| # | Task | Detail |
|---|------|--------|
| 1 | Update lat.md with zero-copy weight streaming section | Architecture, types, upload pipeline |
| 2 | Update plan/README.md with Phase 16 entry | Status, description |
| 3 | Run `lat check` | Verify all links pass |

**Complexity**: XS
**Timebox**: 15 min
**Acceptance**: `lat check` passes. Documentation complete.

---

## Key Design Decisions

### KD1: Zero-copy mmap instead of streaming from disk

mmap already provides zero-copy access to file data. We don't need a separate "streaming from disk" mechanism — the OS page cache handles read-ahead and caching. Our job is to avoid copying from mmap to heap, and upload directly from mmap to GPU.

### KD2: Pinned host buffer for FP16/FP32 conversion

BF16 weights can be uploaded directly from mmap (zero-copy). FP16/FP32 weights need format conversion, which requires a writable buffer. Pinned (page-locked) host memory enables fast DMA via `cudaMemcpyAsync`. The 256MB buffer is large enough for the largest weight and reusable across all uploads.

### KD3: Direct BF16 upload without pinned buffer

For BF16 weights (the bulk of INT4 models — qweight, scales), the mmap bytes ARE the bf16/u32 bits. We cast `&[u8]` → `&[bf16]` or `&[u32]` and upload directly. This skips the pinned buffer entirely for ~90% of weight data.

### KD4: MmapWeightRegistry as parallel to WeightRegistry

Rather than modifying `WeightRegistry` (which would require changing all consumers), we create a parallel type. This keeps the existing path working and allows incremental migration. The two types have identical structure but different backing storage.

### KD5: Retain existing path with `--no-mmap` flag

The mmap path is new and may have edge cases. A `--no-mmap` CLI flag falls back to the heap-based path for debugging. This also serves as an A/B comparison for correctness.

### KD6: Pinned buffer size configurable via env var

Different models have different weight sizes. The default 256MB works for Qwen3.6-27B INT4 (~14GB total). Larger BF16 models may need 512MB or 1GB. `INFERS_PINNED_BUFFER_MB` allows tuning without recompilation.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| mmap pointer safety (dangling after drop) | Medium | High | `Arc<Mmap>` in `MmapTensor` guarantees lifetime. `Send` not `Sync` prevents concurrent mutation. |
| cudarc `clone_htod` may not accept `&[u8]` cast to `&[bf16]` | Low | Medium | Verify `DeviceRepr` alignment requirements. Fallback: copy through pinned buffer. |
| Pinned memory allocation fails (system limit) | Low | Medium | `cudaHostAlloc` returns error — log and fall back to pageable memory. |
| TP sharding on mmap produces incorrect pointer offsets | Medium | High | Unit tests comparing mmap shard bytes against heap shard bytes for known weights. |
| mmap pages evicted before upload completes | Low | Medium | mmap pages are accessed sequentially — OS read-ahead keeps them hot. If eviction occurs, it's a page fault (slow but correct). |

## Getting Started (Developer)

1. Build with mmap support: `cargo build --release -p infers-server`
2. Run with mmap: `cargo run --release --bin infers -- --model /path/to/model`
3. Run without mmap (fallback): `cargo run --release --bin infers -- --model /path/to/model --no-mmap`
4. Monitor DRAM: `watch -n1 free -m` — RSS should drop after weight upload
5. Verify correctness: token output should match between mmap and heap paths

## Success Criteria

- [ ] `MmapTensor` compiles and correctly references mmap'd data
- [ ] `load_safetensors_mmap()` produces `MmapWeightRegistry` without heap copies
- [ ] `GpuWeightCache::new_from_mmap()` uploads weights correctly
- [ ] BF16 weights uploaded directly from mmap (no pinned buffer involved)
- [ ] INT4 qweight uploaded directly from mmap
- [ ] FP16 scales converted via pinned buffer
- [ ] `shard_weights_tp_mmap()` produces correct TP shards
- [ ] `ForwardEngine::new_from_mmap()` compiles and runs
- [ ] Token output matches between mmap and heap paths
- [ ] Process RSS drops to near-GPU-VRAM-only after upload
- [ ] `--no-mmap` flag works as fallback
- [ ] `cargo check --release` passes for all workspace crates
- [ ] `lat check` passes
- [ ] Documentation in lat.md
