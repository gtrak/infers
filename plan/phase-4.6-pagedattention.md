# Phase 4.6: PagedAttention + Prefix Caching

**Duration:** 2-3 weeks
**Goal:** Replace the flat contiguous KV cache with a production-quality PagedAttention subsystem supporting prefix caching, copy-on-write page sharing, and groundwork for future CPU KV offload.

**Status:** Not started.

**Target:** Qwen 3.6, Rust runtime, cudarc, dual NVIDIA GPUs (sm_120 Blackwell)
**Workloads:** Long-context agent inference, multiple concurrent subagents, deep tree-of-thought branching

---

## What Phase 4.5 Left Behind

From `crates/backends/native/src/attention.rs`:
- Flat KV cache: single contiguous buffer `[2 * max_seq_len * kv_dim]` per layer
- Full cache downloaded to CPU during decode, sliced by head, re-uploaded per token
- No multi-session support, no block management, no page sharing
- `infers-kv` crate is an empty stub (`crates/kv/src/lib.rs` = 1 line)
- `GpuAllocator` in `infers-cuda` is bookkeeping-only, no real block pool

## What Phase 4.6 Must Replace

The current `KvCache` design:
- Wastes GPU memory for short sequences (allocates max_seq_len for all)
- Cannot share prefixes across concurrent sessions
- Requires full cache download to CPU during decode (massive PCIe bottleneck)
- Has no eviction or memory budgeting

Phase 4.6 replaces this with a true paged design inspired by vLLM.

---

## Deliverables Checklist

- [ ] `infers-kv` crate: paged block allocator, sequence page tables, prefix cache
- [ ] New CUDA kernels: paged KV cache write, paged KV cache read, paged attention decode
- [ ] Prefix cache: Blake3 hashing, page chain storage, LRU eviction
- [ ] Copy-on-write page sharing for branching prompts
- [ ] attention.rs rewrite: paged decode (no CPU round-trips)
- [ ] MemoryBudget update: block-aware KV estimation
- [ ] engine.rs update: integrate PagedKvManager
- [ ] Unit + stress tests for all subsystems
- [ ] Benchmark suite for prefill throughput, decode latency, cache hit rate
- [ ] Architecture documentation in lat.md
- [ ] `lat check` passes

---

## 1. Paged KV Allocator (infers-kv)

### Physical Page

```rust
struct PhysicalPage {
    page_id: u32,
    refcount: AtomicU32,
    state: PageState,
    location: PageLocation,
    k_ptr: DevicePtr,
    v_ptr: DevicePtr,
}

enum PageState { Mutable, Sealed }
enum PageLocation { Gpu, Cpu }
```

A sealed page is immutable. Pages become sealed when full (all 16/32 slots written).

### Page Pool

```rust
struct PagePool {
    pages: Vec<PhysicalPage>,
    free_list: Vec<u32>,
    block_size: usize,
    page_bytes: usize,
}
```

The pool pre-allocates a fixed number of pages from GPU memory at engine init.

### Sequence Page Table

```rust
struct SequencePageTable {
    page_ids: Vec<PageId>,
    num_tokens: usize,
    tail_page_id: PageId,
}
```

A sequence never owns pages directly -- it only holds a page table pointing into the shared pool.

---

## 2. Prefix Cache

### Page Content Hash

Each sealed page receives a deterministic hash:

```rust
fn hash_page(page: &PhysicalPage, model_id: &str, layer_idx: usize) -> [u8; 32]
```

Uses Blake3. The hash uniquely identifies page content across models and layers.

### Prefix Cache Map

```rust
struct PrefixCache {
    map: HashMap<[u8; 32], PageId>,
    lru: LruCache<[u8; 32], PageId>,
    max_memory_bytes: usize,
    current_memory_bytes: usize,
}
```

### Copy-On-Write

When appending to a shared page (refcount > 1):
1. Allocate a new page from the pool
2. Copy contents from the original page (GPU-side memcpy)
3. Decrement original refcount
4. Replace the page ID in the sequence table with the new page
5. Continue writing to the new page

---

## 3. PagedAttention CUDA Kernels

### Paged KV Cache Write Kernel

```c
void infers_paged_kv_write_bf16(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* page_pool,
    const int* block_table,
    const int* positions,
    int seq_len,
    int head_dim,
    int page_size,
    int kv_dim
);
```

### Paged KV Cache Read Kernel

```c
void infers_paged_kv_read_bf16(
    const __nv_bfloat16* page_pool,
    const int* block_table,
    int num_pages,
    int num_cached_tokens,
    int head_dim,
    int page_size,
    int kv_dim,
    __nv_bfloat16* k_out,
    __nv_bfloat16* v_out
);
```

Eliminates CPU round-trip during decode. All reads are GPU-side.

### PagedAttention Decode Kernel

```c
void infers_paged_attention_decode_bf16(
    const __nv_bfloat16* q,
    const __nv_bfloat16* page_pool,
    const int* block_table,
    int num_pages,
    int num_cached_tokens,
    int head_dim,
    int num_heads,
    int page_size,
    int kv_dim,
    __nv_bfloat16* output
);
```

---

## 4. Memory Layout

Chosen layout: `[page][token][head][dim]`

Within a page, tokens are contiguous. Within a token, heads are contiguous. Good for decode since each decode step needs one token's K/V across all heads.

Physical layout:
```
page_pool[page_id * page_stride + token_in_page * token_stride + head * head_dim + dim]
```

Where:
- `page_stride = page_size * num_kv_heads * head_dim`
- `token_stride = num_kv_heads * head_dim`

---

## 5. Attention Module Rewrite

### Prefill path changes:
1. Compute Q, K, V via GEMM
2. Apply RoPE to K
3. Paged write: Launch `infers_paged_kv_write_bf16` with block table
4. Per-head attention: Read K/V from page pool via `infers_paged_kv_read_bf16` into contiguous buffers, then GEMMs

### Decode path changes:
1. Compute single-token K, V via GEMM
2. Apply RoPE to K
3. Append to tail page (or allocate new if full)
4. Read cached K/V: Launch `infers_paged_kv_read_bf16` (GPU-side only, no CPU round-trip)
5. Attention: GEMM with gathered K/V

---

## 6. Engine Integration

### ForwardEngine updates:

```rust
pub struct ForwardEngine {
    // ... existing fields ...
    page_pool: PagePool,
    prefix_cache: PrefixCache,
    paged_kv_manager: PagedKvManager,
}
```

### PagedKvManager

```rust
struct PagedKvManager {
    page_pool: Arc<Mutex<PagePool>>,
    prefix_cache: Arc<Mutex<PrefixCache>>,
    page_size: usize,
}

impl PagedKvManager {
    fn create_sequence(&self) -> SequencePageTable;
    fn write_kv(&self, seq: &mut SequencePageTable, k: &CudaSlice<bf16>, v: &CudaSlice<bf16>, positions: &[u32]);
    fn read_kv(&self, seq: &SequencePageTable, num_tokens: usize) -> (CudaSlice<bf16>, CudaSlice<bf16>);
    fn seal_and_cache(&self, page_id: PageId, layer_idx: usize, model_id: &str);
    fn delete_sequence(&self, seq: &mut SequencePageTable);
}
```

---

## 7. Testing Strategy

### Unit Tests

| Test | Description |
|------|-------------|
| page_alloc_free | Allocate and free pages, verify free list |
| page_seal | Write tokens, seal page, verify immutable |
| page_refcount | Share page between sequences, verify refcount |
| prefix_cache_hit | Write identical prefix to two sequences, verify shared pages |
| prefix_cache_miss | Write different prefix, verify new pages allocated |
| cow_correctness | Append to shared page, verify COW creates new page |
| cow_immutable_original | After COW, verify original page unchanged |
| sequence_delete | Delete sequence, verify pages returned to pool |
| page_reclamation | Verify pages with refcount 0 are reclaimable |
| block_table_mapping | Verify logical token -> page ID -> physical offset |

### Stress Tests

| Test | Description |
|------|-------------|
| many_sequences | 100 concurrent sequences, random lengths |
| prefix_sharing | 50 sequences sharing a 1k-token prefix |
| deep_branching | Tree-of-thought: 1 root, 10 branches, each 5 sub-branches |
| memory_pressure | Allocate until out of pages, verify graceful failure |
| cache_eviction | Fill cache beyond memory budget, verify LRU eviction |

### Benchmarks

| Metric | Contexts |
|--------|----------|
| Prefill throughput (tok/s) | 4k, 8k, 32k, 64k, 128k |
| Decode latency (ms/token) | 4k, 8k, 32k, 64k, 128k cached |
| Cache hit rate | shared prefixes, many agent branches |
| Cache lookup cost | microseconds per page hash lookup |
| Memory usage | pages allocated vs theoretical minimum |
| Page allocator overhead | alloc + free time |

Workloads:
- Single long conversation
- Many agent branches from shared system prompt
- Shared prefixes (batch of identical prompts)
- Deep tree-of-thought branching

---

## 8. File Structure

```
crates/kv/
  src/
    lib.rs
    page.rs          # PhysicalPage, PageState, PageLocation
    pool.rs          # PagePool, allocation/free
    table.rs         # SequencePageTable
    prefix.rs        # PrefixCache, hash_page, LRU eviction
    manager.rs       # PagedKvManager, orchestrates pool + cache
    cow.rs           # CopyOnWrite logic

crates/cuda/kernels/infers/
  paged_kv_write.cu
  paged_kv_read.cu
  paged_attention_decode.cu

crates/backends/native/src/
  attention.rs       # REWRITE: paged decode, no CPU round-trips
  engine.rs          # UPDATE: integrate PagedKvManager
```

---

## 9. Cross-References

- **Phase 4.5:** Attention forward/decode implementation, kernel registry, weight upload
- **Phase 2:** CUDA backend, kernel compilation, cudarc integration
- **Phase 3:** Model loading, WeightRegistry, ModelConfig
- **Phase 6:** Continuous batching will use PagedKvManager for interleaved sequences
- **Phase 7:** MTP will share the same page pool for draft model KV cache

## Open Questions

1. Should page size be 16 or 32 tokens? Benchmark both.
2. Should prefix cache be per-layer or global? Per-layer is more accurate but uses more memory.
3. How to handle CPU offload migration? Design PageLocation::Cpu but defer implementation.
4. Should we use cudaMemcpyAsync for COW copies? Yes, for GPU-side page copying.
5. How to benchmark without a real model? Use synthetic weights and random prompts.

---

## Success Criteria

1. Decode latency < 5ms/token for 32k context (paged vs flat)
2. Prefix cache hit rate > 80% for shared-prompt workloads
3. Memory usage < 2x theoretical minimum (low fragmentation)
4. Zero CPU-GPU round-trips during decode
5. All unit tests pass
6. Stress tests run for 10 minutes without memory leaks
7. `lat check` passes
