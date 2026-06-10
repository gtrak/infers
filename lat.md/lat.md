This directory defines the high-level concepts, business logic, and architecture of this project. It is managed by [lat.md](https://www.npmjs.com/package/lat.md) — a tool that anchors source code to these definitions.

# Workspace Architecture
Rust workspace for infers, a Qwen3.6-27B inference server using nightly toolchain with cuda-oxide orchestration.

## Crates

Workspace contains 12 crates across server, API, inference backend, and utility domains.

| Crate | Path | Purpose |
|-------|------|---------|
| infers-server | crates/server | axum HTTP server, CLI, main entry point |
| infers-api | crates/api | OpenAI-compatible types + SSE protocol |
| infers-scheduler | crates/scheduler | Session lifecycle, batch construction |
| infers-kv | crates/kv | Hybrid KV state manager |
| infers-model | crates/model | Multi-format model loader |
| infers-backend-native | crates/backends/native | Custom CUDA kernels + cuBLASLt backend |
| infers-backend-gguf | crates/backends/gguf | llama.cpp backend |
| infers-cuda | crates/cuda | cuda-oxide + cudarc hybrid |
| infers-parallelism | crates/parallelism | TP=2 and PP=2 implementations |
| infers-tokenizer | crates/tokenizer | HF tokenizers wrapper |
| infers-metrics | crates/metrics | Prometheus exporter |
| infers-mtp | crates/mtp | MTP draft/verify |


## Dependency Graph
Crate dependency relationships and feature propagation between workspace members.

infers-backend-native and infers-parallelism both depend on infers-cuda for GPU kernel loading and NCCL communication respectively. cudarc is always present — no feature gating.

## Toolchain
Nightly toolchain configuration, Rust edition, and cargo-oxide requirements for CUDA support.

- Rust nightly-2026-04-03 with rust-src, rustc-dev, llvm-tools
- edition = "2024"
- cargo-oxide for CUDA crates

## Dependencies

Key workspace dependencies pinned to exact versions: tokio 1.52.3, axum 0.8.9, serde 1.0.228, clap 4.6.1, prometheus 0.14.0, thiserror 2.0.18. cudarc 0.19.7 with cublaslt, nccl, cuda-13020, f16 features. half 2.6.0 for FP16/BF16. cuda-oxide deferred.

## CUDA Crate

CUDA runtime crate for GPU inference. cudarc is always present with no feature gating or optional deps. Re-exports key cudarc types for consumer convenience.

### Module Structure

Six modules cover context, streams, memory, kernels, GEMM, and NCCL.

| Module | Purpose |
|--------|---------|
| context | CUDA device context management, CudaRuntime |
| stream | CUDA stream pool for async execution |
| memory | Block pool GPU memory allocator |
| kernels | Kernel registry for pre-compiled .cubin loading |
| gemm | cuBLASLt GEMM engine with `matmul_f32()`, `matmul_bf16()`, `matmul_fp16()` methods for FP32/BF16/FP16 matrix multiplication |
| nccl | Multi-GPU collective operations for TP/PP |

# Kernel Extraction and Build System
Pipeline for compiling infers CUDA kernel source to .cubin binaries.

### Kernel Directory Structure
Three directories hold kernel source and compiled binaries under `crates/cuda/kernels/`. All directories contain `.gitkeep` files for git tracking.

| Directory | Contents |
|-----------|----------|
| `flashinfer-gdn/` | Reserved for future custom GDN kernel source |
| `flashinfer-attn/` | Reserved for future custom attention kernel source |
| `infers/` | Custom CUDA kernel source (.cu, .cuh) for inference operations |
| `compiled/` | Compiled .cubin output from nvcc |

### Kernel Source Files
All kernels use `extern "C" __global__` so function names are directly loadable from cubin files. Launch configuration is determined by Rust dispatch code, not kernel wrappers.

Fourteen kernel implementations across 13 files for transformer forward-pass operations using BF16 data.

| File | Kernels | Description |
|------|---------|-------------|
| `common.cuh` | — | Shared utilities: `__nv_bfloat16` conversion helpers, `INFERS_BLOCK_SIZE` (256), thread indexing macros |
| `rmsnorm.cu` | `infers_rmsnorm_bf16` | RMS Layer Normalization: output = x * rsqrt(mean(x²) + eps) * weight, using float shared memory for precision-preserving reduction |
| `silu.cu` | `infers_silu_bf16`, `infers_silu_glu_bf16` | SiLU activation and SwiGLU gating: output = x * sigmoid(gate) |
| `rope.cu` | `infers_rope_bf16` | Rotary Position Embedding applied to query and key tensors |
| `embedding.cu` | `infers_embedding_gather_bf16` | Token embedding gather: gather rows from weight matrix by token ID |
| `elementwise.cu` | `infers_add_bf16` | Element-wise addition for residual connections |
| `sampling.cu` | `infers_argmax_f32` | Greedy argmax sampling from FP32 logits |
| `softmax.cu` | `infers_softmax_bf16` | Online softmax for attention scores with optional causal masking, using three-phase parallel reduction (max, sum, normalize) in shared memory |
| `kv_cache.cu` | `infers_kv_cache_write_bf16` | Scattered KV cache write using position IDs: writes K and V rows into cache at arbitrary positions via strided thread loops |
| `gdn_update.cu` | `infers_gdn_update_bf16` | Gated DeltaNet decode kernel: recurrent state update for a single token via three-phase block reduction (beta, state update, output) with one block per state row |
| `gdn_prefill.cu` | `infers_gdn_prefill_bf16` | Gated DeltaNet prefill kernel: processes all tokens in a sequence sequentially within each block, updating state and writing per-token output via shared memory reduction |
| `paged_kv_write.cu` | `infers_paged_kv_write_bf16` | Paged KV cache write using block-table address translation: writes K and V into interleaved per-page layout via strided thread loops, eliminating CPU round-trips during prefill |
| `paged_kv_read.cu` | `infers_paged_kv_read_bf16` | Paged KV cache read using block-table address translation: gathers K and V from interleaved per-page layout into contiguous output buffers via strided thread loops, eliminating CPU round-trips during decode |
| `paged_attention_decode.cu` | `infers_paged_attention_decode_bf16` | Paged attention decode: computes single-token attention over paged KV cache using two-pass online softmax and weighted V accumulation, one block per KV head — Phase 1 uses strided dot-product computation, Phase 2 loops over all tokens per thread |

### Build Script
Compiles all `.cu` files found in `kernels/infers/` to .cubin binaries using nvcc.

Targets `sm_120` (Blackwell RTX 5060 Ti) by default, configurable via the `INFERS_CUDA_ARCH` environment variable. Uses `-O3 --use_fast_math` flags with `-I kernels/infers` include path for common.cuh. The `find_nvcc()` function checks PATH first, then falls back to common CUDA install locations (`/usr/local/cuda/bin/nvcc`, `/usr/local/cuda-13.2/bin/nvcc`, `/usr/local/cuda-13.0/bin/nvcc`, `/usr/bin/nvcc`). Missing nvcc or source files produce warnings but do not fail the build. Compiled kernels are placed in `kernels/compiled/` with matching names and loaded at runtime by the KernelRegistry.

# Phase 1 Deliverables
Phase 1 (Bootstrap) creates the workspace, crate skeletons, and API scaffolding for the inference server.

- Workspace Cargo.toml with 12 crate members and pinned dependencies
- rust-toolchain.toml for nightly-2026-04-03
- OpenAI-compatible API types (request, response, streaming, error)
- Axum HTTP server with mock responses
- SSE streaming scaffold for chat completions
- Prometheus metrics endpoint at /metrics
- Health check endpoint at /health
- CLI argument parsing with clap

# Phase 2 Deliverables
Phase 2 (CUDA Backend) establishes the GPU runtime, kernel compilation pipeline, and multi-GPU communication primitives.

- CUDA crate (`infers-cuda`) with cudarc always present — no feature gating
- CudaRuntime for multi-GPU device context management (cudarc CudaContext)
- StreamPool for async CUDA stream management per device
- GpuAllocator block pool memory bookkeeper with allocate/free/reuse (5 unit tests)
- KernelRegistry for .cubin loading (14 infers kernels: rmsnorm, silu, silu_glu, rope, embedding_gather, add, argmax, softmax, kv_cache, paged_kv_write, paged_kv_read, gdn_update, gdn_prefill, paged_attention_decode) and LoadedKernelRegistry for GPU-loaded kernels with deduplication (same .cubin loaded once even when referenced by multiple kernel functions)
- GemmEngine wrapping cuBLASLt with FP16/BF16/FP32 support; `new(stream)` creates CudaBlasLT eagerly, `matmul_f32()`, `matmul_bf16()`, `matmul_fp16()` accept `GemmConfig` and `CudaSlice` buffers
- NcclCommunicator wrapping cudarc NCCL Comm with `all_reduce()`, `all_reduce_in_place()`, `broadcast()`, `reduce()`, `all_gather()` methods for TP/PP collectives across multiple GPUs
- build.rs for nvcc kernel compilation (default sm_120, configurable via INFERS_CUDA_ARCH env var)
- CUDA kernel source files in `kernels/infers/`: rmsnorm.cu, silu.cu, rope.cu, embedding.cu, elementwise.cu, sampling.cu, softmax.cu, kv_cache.cu, paged_kv_write.cu, paged_kv_read.cu, gdn_update.cu, gdn_prefill.cu, paged_attention_decode.cu, common.cuh
- Kernel directory structure (flashinfer-gdn, flashinfer-attn, compiled) preserved for organization; custom kernels use infers/

# Phase 3 Deliverables
Phase 3 (Model Loading) implements multi-format model loading with auto-detection, weight registry, TP sharding, and memory budgeting.

- Multi-format model loader (`infers-model`) with safetensors support
- ModelConfig struct for parsing `config.json` with hybrid attention layer types
- QuantizationFormat auto-detection (GGUF, PrismaSCOUT, AutoRound, BF16)
- WeightRegistry and WeightData for storing tensors as raw bytes
- SafetensorsLoader with single-file and sharded index support
- Weight sharding for TP=2 (column/row parallel) and PP=2 (layer split)
- MemoryBudget calculator for VRAM estimation across quantization formats
- LayerType enum with default pattern (every 4th layer full attention, rest GDN)
- QuantizationConfig parsing from `quantization_config.json` or embedded config

# Phase 4.5 Deliverables
Phase 4.5 (Attention, KV Cache, and GDN Kernels) adds custom CUDA kernels for attention softmax, KV cache management, and Gated DeltaNet state updates, and wires them into the prefill/decode paths.

- `softmax.cu` + `.cubin`: Online softmax with causal masking (3-phase reduction)
- `kv_cache.cu` + `.cubin`: Scattered KV cache write with position-based indexing
- `gdn_update.cu` + `.cubin`: Single-token decode recurrent state update
- `gdn_prefill.cu` + `.cubin`: Chunked prefill state update across all tokens
- `attention.rs` wired: per-head weight slicing, QKV/RoPE/KV cache/scores/softmax/O-proj/all-reduce (prefill + decode)
- `gdn.rs` wired: projection GEMMs, GDN kernel dispatch, output projection (prefill + decode)
- `prefill.rs` wired: embed → layer loop (norm1 → GDN/attention → residual → norm2 → MLP → residual) → final norm → LM head → sample
- `decode.rs` wired: embed single token → layer loop (decode variants) → final norm → LM head → sample
- `engine.rs`: 11 cached CudaFunction handles, per-layer kv_caches and gdn_states, prefill/decode delegation
- Kernel fixes: softmax max preservation (register variable), power-of-2 block rounding, attention GEMM transb correction, accumulation parity fix
- `lat check` passes

# Phase 4.6 Deliverables
Phase 4.6 (PagedAttention + Prefix Caching) replaces the flat contiguous KV cache with a paged block-pool allocator supporting prefix caching and copy-on-write page sharing. See `plan/phase-4.6-pagedattention.md` for the full design document.

## Paged KV Types

Core data structures in `infers-kv` crate for paged attention.

### PhysicalPage

CPU-side metadata for a single paged KV block.

`PhysicalPage` tracks `page_id`, `refcount` (`AtomicU32` for copy-on-write sharing), `state` (`Mutable` or `Sealed`), `location` (`Gpu` or `Cpu`), and device pointers `k_ptr`/`v_ptr`. Pages become `Sealed` when full, enabling prefix caching. See [[crates/kv/src/page.rs#PhysicalPage]].

### PageId and PageSentinel

`u32` alias for identifying physical pages in the pool.

`PageId` is a `u32` alias. `INVALID_PAGE_ID` (`u32::MAX`) serves as a sentinel for unallocated slots. See [[crates/kv/src/page.rs#PageId]].

### SequencePageTable

Maps logical token positions to physical page IDs for kernel dispatch.

`SequencePageTable` does not own pages — it holds a `Vec<PageId>` pointing into the shared `PagePool`. Methods: `push_page()`, `add_token()`, `remove_last_page()`, `is_tail_page_full()`, `tail_page_id()`, `block_table()`. See [[crates/kv/src/table.rs#SequencePageTable]].

### PagePool

Pre-allocated pool of physical pages with O(1) allocate and free via a stack-based free list.

`PagePool` pre-allocates `total_pages` `PhysicalPage` entries at engine init time. Each page starts `Mutable` on the GPU with refcount 1. The `free_list` is a `Vec<PageId>` supporting O(1) pop for allocation and push for deallocation. Methods: `allocate()`, `free()`, `seal()`, `get()`, `get_mut()`, `page_size()`, `page_bytes()`, `is_full()`, `is_empty()`, `num_free()`, `num_total()`. Page byte size is `page_size * num_kv_heads * head_dim * 2` (BF16). CPU-side bookkeeping only — GPU buffer management lives at the integration layer. See [[crates/kv/src/pool.rs#PagePool]].

### PagePoolError

Custom error type for page pool operations using thiserror.

`PagePoolError` has two variants: `PoolExhausted` when no free pages remain, and `InvalidPageId` when a lookup targets an out-of-range page ID. See [[crates/kv/src/pool.rs#PagePoolError]].

### PrefixCache

LRU-based prefix cache for sharing sealed KV cache pages across sequences.

`PrefixCache` maps content hashes (`[[PageHash]]`) to physical pages, enabling multiple sequences to share the same sealed pages when they have identical prompt prefixes. Memory is tracked against a configurable budget. When the budget is exceeded, `evict_if_needed()` evicts the least recently used entries with refcount = 0. `insert()` creates or increments entries, `lookup()` returns page IDs, `touch()` updates LRU order, and `release()` decrements refcounts. See [[crates/kv/src/prefix.rs#PrefixCache]].

### PageHash

256-bit Blake3 content hash identifying sealed page contents.

`PageHash` is a `[u8; 32]` type alias uniquely identifying page content across models and layers. See [[crates/kv/src/prefix.rs#PageHash]].

### CacheEntry

Entry in the prefix cache mapping a content hash to a physical page.

`CacheEntry` holds `page_id` (the physical page ID) and `refcount` (number of sequences currently referencing this cached entry). See [[crates/kv/src/prefix.rs#CacheEntry]].

### hash_page

Pure function that computes a Blake3 content hash for a sealed page.

`hash_page(k_data, v_data, model_id, layer_idx)` derives a 32-byte key from `model_id` using `blake3::hash`, then uses `blake3::Hasher::new_keyed` to hash `k_data || v_data || layer_idx.to_le_bytes()`. CPU-side operation — caller copies page data from GPU to host before hashing. See [[crates/kv/src/prefix.rs#hash_page]].
### CowResult

Copy-on-write result enum distinguishing in-place writes from COW operations.

`CowResult` has two variants: `NoCowNeeded { page_id }` when the tail page is exclusively owned and mutable, and `CowPerformed { new_page_id, original_page_id }` when COW allocated a new page and decremented the original refcount. See [[crates/kv/src/cow.rs#CowResult]].

### CowError

Custom error type for COW operations using thiserror.

`CowError` has two variants: `PoolExhausted` when the pool has no free pages, and `InvalidPageId` when a page lookup targets an out-of-range page ID. See [[crates/kv/src/cow.rs#CowError]].

### ensure_mutable_page

Main COW entry point that ensures a sequence's tail page is exclusively owned and mutable.

`ensure_mutable_page(pool, table)` returns `NoCowNeeded` if the tail page is exclusively owned and mutable. Returns `CowPerformed` if the tail page is shared or sealed. See [[crates/kv/src/cow.rs#ensure_mutable_page]].

### decrement_page_refcount

Helper that decrements a page's refcount and returns the previous value.

`decrement_page_refcount(pool, page_id)` calls `fetch_sub(1, SeqCst)` on the page's refcount. Returns the previous refcount before decrement. See [[crates/kv/src/cow.rs#decrement_page_refcount]].

### try_share_from_prefix_cache

Attempts to share an existing page from the prefix cache.

`try_share_from_prefix_cache(cache, hash, pool, table)` looks up `hash` in the prefix cache. If found, increments the cached page's refcount and appends the page ID to the sequence table, returning `true`. If not found, returns `false`. See [[crates/kv/src/cow.rs#try_share_from_prefix_cache]].

### PagedKvManager

Orchestrator that ties together the page pool, prefix cache, and copy-on-write logic for multi-sequence management.

`PagedKvManager` holds shared pool and cache via `Arc<Mutex<>>`, manages sequences as a `Vec<Option<SequencePageTable>>` with free-ID reuse, and coordinates page allocation, token counting, page sealing, and prefix caching. Methods: `new()` initializes pool and cache, `create_sequence()` allocates a unique `SequenceId`, `delete_sequence()` frees all pages back to the pool, `append_page()` allocates a page and pushes to the table, `ensure_writable()` delegates to COW, `add_token()` increments token count and seals pages at boundaries, `seal_and_cache()` seals and inserts into prefix cache, `block_table()` returns `&[PageId]` for kernel consumption, `num_pages()`, `num_tokens()`, `num_free_pages()`, `pool_utilization()`. See [[crates/kv/src/manager.rs#PagedKvManager]].

### ManagerError

Custom error type for manager operations using thiserror.

`ManagerError` has two variants: `InvalidSequence(seq_id)` when a sequence is not found or already deleted, and `PoolExhausted` when the page pool has no free pages. See [[crates/kv/src/manager.rs#ManagerError]].

### SequenceId

`usize` alias for identifying sequences in the manager.

`SequenceId` is a `usize` used as the index into `PagedKvManager.sequences`. Deleted IDs are recycled via a free list. See [[crates/kv/src/manager.rs#SequenceId]].

## Completed

Types, tests, and attention.rs rewrite shipped for Phase 4.6 paged KV foundations.

- `infers-kv` crate: `PhysicalPage`, `PageId`, `SequencePageTable` types with 11 unit tests
- `PageState` enum (`Mutable`, `Sealed`), `PageLocation` enum (`Gpu`, `Cpu`)
- `PagePool` with O(1) stack-based free list, `PagePoolError` (thiserror), 5 unit tests
- `PrefixCache` with Blake3 content hashing, LRU eviction, `CacheEntry`, `PageHash`, 12 unit tests
- `cow` module: `CowResult`, `CowError`, `ensure_mutable_page`, `decrement_page_refcount`, `try_share_from_prefix_cache`, 11 unit tests
- `manager` module: `PagedKvManager`, `ManagerError`, `SequenceId`, 7 unit tests
- `paged_kv_write.cu` + `.cubin`: Paged KV cache write with block-table address translation, K+V interleaved per-page layout
- `paged_kv_read.cu` + `.cubin`: Paged KV cache read with block-table address translation, gathers K and V into contiguous output buffers
- `paged_attention_decode.cu` + `.cubin`: Paged attention decode with two-pass online softmax and weighted V accumulation — Phase 1 uses strided cooperative dot-product computation for softmax stats, Phase 2 loops over all tokens per output dimension
- `attention.rs`: `PagedKvCache` struct, three kernel dispatch functions (`paged_kv_write`, `paged_kv_read`, `paged_attention_decode`), `decode_forward_paged` (zero CPU round-trips, single GEMM O-projection), `forward_paged` (paged KV write + per-head GEMM attention)
- `infers-backend-native` now depends on `infers-kv` crate

## Paged Attention Implementation

Paged attention functions and types in `attention.rs` for zero CPU round-trip decode.

### PagedKvCache

GPU-side paged KV cache replacing flat contiguous buffers. See [[crates/backends/native/src/attention.rs#PagedKvCache]].

Unlike `KvCache` which allocates `[2 * max_seq_len * kv_dim]` for a flat buffer, `PagedKvCache` allocates `[num_pages * 2 * page_size * kv_dim]` for the interleaved page layout. Per-page layout: `[K tokens | V tokens]`, each side = `page_size * kv_dim` elements. Uses lazy allocation via `ensure_allocated()`.

### Paged Kernel Dispatch

Three dispatch functions for paged attention CUDA kernels.

`paged_kv_write()` launches `infers_paged_kv_write_bf16` with grid `(seq_len * kv_dim + 255) / 256`, block `(256, 1, 1)` — writes K and V into paged cache using block-table address translation. See [[crates/backends/native/src/attention.rs#paged_kv_write]].

`paged_kv_read()` launches `infers_paged_kv_read_bf16` with grid `(num_cached_tokens * kv_dim + 255) / 256`, block `(256, 1, 1)` — gathers K and V from page pool into contiguous output buffers for GEMM consumption. See [[crates/backends/native/src/attention.rs#paged_kv_read]].

`paged_attention_decode()` launches `infers_paged_attention_decode_bf16` with grid `(num_kv_heads, 1, 1)`, block `(256, 1, 1)`, shared memory `3 * 256 * 4 = 3072` bytes — one block per KV head, computes full decode attention (score, softmax, V accumulation) in a single kernel call. See [[crates/backends/native/src/attention.rs#paged_attention_decode]].

### Paged Decode Forward

Zero CPU round-trip decode attention. See [[crates/backends/native/src/attention.rs#decode_forward_paged]].

Computes single-token K/V via GEMM, applies RoPE, writes to paged cache, computes Q with RoPE, launches paged attention decode kernel for full attention computation, then applies O-projection via a single GEMM. Replaces the per-head loop that required CPU download/re-upload of KV cache data.

### Paged Prefill Forward

Prefill with paged KV cache write. See [[crates/backends/native/src/attention.rs#forward_paged]].

Same K/V computation and RoPE as original `forward`, but writes to paged cache via `paged_kv_write` instead of flat buffer. Attention computation still uses per-head GEMMs (prefill benefits less from paged decode kernel since all tokens are processed at once).

## Remaining

Future deliverables for Phase 4.6 completion.
- GPU-side COW memcpy: actual data copy from original page to COW page in attention kernel layer
- MemoryBudget update: block-aware KV estimation vs flat-buffer model
- engine.rs integration: wire `PagedKvManager` into ForwardEngine dispatch loop (Task 10)
- Stress tests and benchmark suite

# Phase 4 Deliverables
Phase 4 (Forward Pass) implements the core inference engine with hybrid GDN/full-attention dispatch, cuBLASLt GEMM, and NCCL tensor parallelism.

## Forward Engine
Central `ForwardEngine` struct owns all GPU state and coordinates prefill/decode inference. `prefill()` and `decode()` delegate to module-level functions with cached kernel handles and per-layer KV/GDN state vectors.

### Engine Structure
`ForwardEngine` holds config, weights, 13 cached `CudaFunction` handles, GemmEngine, NcclCommunicator, StreamPool, and per-layer `kv_caches` and `gdn_states` vectors. Kernel handles are resolved from `LoadedKernelRegistry` at init time. See [[crates/backends/native/src/engine.rs#ForwardEngine]].

### Prefill Path

Embeds prompt tokens, loops through layers dispatching GDN or full attention based on `LayerType`, applies final norm + LM head, samples first token via greedy argmax.

The `prefill` function accepts a `PrefillKernels` struct holding all CUDA kernel handles (`rmsnorm`, `silu_glu`, `rope`, `embedding`, `add`, `argmax`, `softmax`, `kv_cache_write`, `gdn_prefill`), a `GemmEngine`, CUDA stream, `NcclCommunicator`, `ModelConfig`, `WeightRegistry`, token IDs, and mutable vectors of `KvCache` and `GdnState` for each layer.

**Phase 1 — Embedding**: Uploads embedding weights via `upload_weight()` (converts `WeightData` bytes to BF16 Vec and copies to GPU), then dispatches the embedding gather kernel.

**Phase 2 — Layer Loop**: For each layer, dispatches `norm::rms_norm` (norm1), then either `gdn::forward` or `attention::forward` depending on `LayerType`, followed by residual add, `norm::rms_norm` (norm2), `mlp::mlp_forward`, and another residual add. NCCL all-reduce calls are reserved for multi-GPU paths.

**Phase 3 — Final Norm + LM Head**: Applies final RMSNorm, then computes logits via `hidden @ lm_head^T` GEMM producing `[seq_len × vocab_size]` BF16 matrix.

**Phase 4 — Sampling**: Downloads BF16 logits to host, extracts last token's row, converts to FP32, uploads to GPU, and calls `sample::greedy_sample` with the argmax kernel.

See [[crates/backends/native/src/prefill.rs#prefill]], [[crates/backends/native/src/prefill.rs#PrefillKernels]], [[crates/backends/native/src/upload.rs#upload_weight]].

### Decode Path

Embeds a single token, loops through layers dispatching GDN recurrent steps or single-token attention over cached KV, applies final norm + LM head, samples next token via greedy argmax.

The `decode` function accepts a `DecodeKernels` struct holding all CUDA kernel handles (`rmsnorm`, `silu_glu`, `rope`, `embedding`, `add`, `argmax`, `softmax`, `kv_cache_write`, `gdn_update`), a `GemmEngine`, CUDA stream, `NcclCommunicator`, `ModelConfig`, `WeightRegistry`, a single `token_id`, `position`, and mutable vectors of `KvCache` and `GdnState` for each layer.

**Phase 1 — Embedding**: Uploads embedding weights via `upload_weight()` (converts `WeightData` bytes to BF16 Vec and copies to GPU), then dispatches the embedding gather kernel for a single token.

**Phase 2 — Layer Loop**: For each layer, dispatches `norm::rms_norm` (norm1), then either `gdn::decode_forward` (recurrent state update) or `attention::decode_forward` (single-token attention over cached KV) depending on `LayerType`, followed by residual add, `norm::rms_norm` (norm2), `mlp::mlp_forward`, and another residual add. All GEMMs use m=1 (single token).

**Phase 3 — Final Norm + LM Head**: Applies final RMSNorm, then computes logits via `hidden @ lm_head^T` GEMM producing `[1 × vocab_size]` BF16 vector.

**Phase 4 — Sampling**: Downloads BF16 logits to host, converts to FP32, uploads to GPU, and calls `sample::greedy_sample` with the argmax kernel. Unlike prefill, no row extraction is needed since logits are already `[1 × vocab_size]`.

See [[crates/backends/native/src/decode.rs#decode]], [[crates/backends/native/src/decode.rs#DecodeKernels]], [[crates/backends/native/src/upload.rs#upload_weight]].

## Module Structure
Thirteen modules cover forward-pass operations: engine, prefill, decode, gdn, attention, mlp, norm, rope, sample, embedding, sync, add, upload.

### Layer Operations
Per-layer CUDA kernel dispatch for transformer operations.

| Module | Purpose |
|--------|---------|
| `norm` | RMSNorm kernel dispatch (`infers_rmsnorm_bf16`) |
| `rope` | Rotary position embedding (`infers_rope_bf16`) |
| `embedding` | Token embedding gather (`infers_embedding_gather_bf16`) |
| `mlp` | SwiGLU forward via cuBLASLt GEMM + SiLU kernel |
| `sample` | Greedy argmax (`infers_argmax_f32`) + strategy enum |
| `attention` | Full attention: per-head weight slicing, QKV GEMMs, RoPE, KV cache write, softmax, O-projection |
| `gdn` | Gated DeltaNet: projection GEMMs, `infers_gdn_prefill_bf16` kernel, output projection |
| `sync` | NCCL all-reduce for TP collectives (`all_reduce_attention`, `all_reduce_mlp`) |
| `add` | Element-wise addition for residual connections (`infers_add_bf16`) |
| `upload` | Shared weight upload utility: converts `WeightData` bytes to GPU-resident BF16 buffers (`upload_weight`) |


### Attention Forward Pass

Full-attention prefill implementation using per-head weight slicing. Extracts each attention head's weight slice on CPU, uploads to GPU, and computes per-head GEMMs — avoids strided GPU sub-slices unsupported by cudarc.

#### Architecture

**Phase 1 — Full K/V for cache**: Computes full K and V via GEMM, applies RoPE to K (dummy Q buffer), writes RoPE'd K/V to KV cache.

**Phase 2 — Per-head attention**: For each head, computes Q_h/K_h/V_h via sliced weights, applies RoPE, computes attention scores with softmax (causal mask), produces partial output via O-projection.

**Alternating accumulation**: Two GPU buffers accumulate per-head partial O-projection outputs; even-indexed heads add to `accum_b`, odd-indexed to `accum_a`.

Odd number of heads leaves result in `accum_a`, even in `accum_b`.

See [[crates/backends/native/src/attention.rs#forward]].

### Decode Attention Forward Pass

Single-token attention for decode-time generation. Projects a single input token into Q/K/V, applies RoPE, appends to KV cache, then computes attention over all cached tokens.

#### Architecture

**Phase 1 — Full K/V for cache**: Computes full K and V via GEMM from single-token input, applies RoPE to K (dummy Q buffer), writes RoPE'd K/V to KV cache at current position.

**Phase 2 — Per-head cache extraction**: Downloads full KV cache from GPU to CPU, extracts per-head K and V buffers by striding through the flat `[max_seq_len × kv_dim]` layout.

**Phase 3 — Per-head attention**: For each head, computes Q_h via sliced weights, applies RoPE, computes attention scores against cached K_h via GEMM, applies softmax (no causal mask — single query attends all cache), produces partial output via O-projection.

Same alternating accumulation pattern as prefill: even-indexed heads add to `accum_b`, odd-indexed to `accum_a`.

Unlike prefill, the softmax uses `use_causal=0` because a single query token can attend to all previously cached positions.

See [[crates/backends/native/src/attention.rs#decode_forward]].

### GDN Forward Pass

Gated DeltaNet prefill: projection GEMMs feed `infers_gdn_prefill_bf16` kernel, then output projection.

#### Architecture

**Phase 1 — Projections**: Five projection weights are uploaded from CPU bytes to GPU BF16 buffers. Four GEMMs compute `a`, `b`, `x`, and `dt` projections from input. 1D convolution (`conv1d_weight`) is skipped for Phase 4.5.

**Phase 2 — State Update**: GDN state (H×H matrix) is allocated lazily on first call. The `infers_gdn_prefill_bf16` kernel runs with `hidden_size` blocks (one per state row), each block processing all `seq_len` tokens sequentially via shared memory reduction.

**Phase 3 — Output Projection**: Final GEMM multiplies kernel output by `out_proj_weight`.

Tensor-parallel all-reduce is handled by the caller in `prefill.rs`.

See [[crates/backends/native/src/gdn.rs#forward]].

### GDN Decode Forward Pass

Gated DeltaNet decode: recurrent single-token state update using `infers_gdn_update_bf16` kernel.

#### Architecture

**Phase 1 — Projections**: Four projection weights are uploaded from CPU bytes to GPU BF16 buffers. Four GEMMs (m=1) compute `a`, `b`, `x`, and `dt` projections from a single-token input `[1 × hidden_size]`. 1D convolution is skipped, same as prefill.

**Phase 2 — State Update**: GDN state is allocated lazily via `ensure_allocated()`. The `infers_gdn_update_bf16` kernel runs with `hidden_size` blocks (one per state row) and power-of-2 block size up to 256. Unlike prefill, it processes only a single token and takes 7 arguments (no `seq_len`).

**Phase 3 — Output Projection**: Final GEMM (m=1) multiplies `[hidden_size]` kernel output by `out_proj_weight`.

Tensor-parallel all-reduce is handled by the caller in `decode.rs`.

See [[crates/backends/native/src/gdn.rs#decode_forward]].

### Sampling
`SamplingStrategy` enum with `Greedy`, `Temperature`, `TopK`, `TopP` variants. `SamplingConfig` holds strategy, max tokens, and stop sequences. See [[crates/backends/native/src/sample.rs#SamplingStrategy]].

### Kernel Dispatch
Kernel dispatch functions launch pre-compiled .cubin kernels using cudarc's `LaunchArgs` API. Each function allocates output buffers, builds a `LaunchConfig`, and passes kernel arguments via the `PushKernelArg` trait. See [[crates/backends/native/src/norm.rs#rms_norm]], [[crates/backends/native/src/embedding.rs#embed_tokens]], [[crates/backends/native/src/sample.rs#greedy_sample]], [[crates/backends/native/src/mlp.rs#mlp_forward]], [[crates/backends/native/src/add.rs#add]], [[crates/backends/native/src/rope.rs#apply_rope]], [[crates/backends/native/src/attention.rs#forward]], [[crates/backends/native/src/attention.rs#decode_forward]], [[crates/backends/native/src/gdn.rs#forward]], [[crates/backends/native/src/gdn.rs#decode_forward]].

# API Types
OpenAI-compatible request, response, streaming, and error types for the inference API.

## Request Types
ChatCompletionRequest and all nested types (Message, Tool, Function, StopConfig, etc.) mirror the OpenAI chat.completions API schema.

## Response Types
ChatCompletionResponse and nested types (Choice, MessageContent, Usage, ToolCall, FunctionCall) define the synchronous API response shape.

## Streaming Types
SSE streaming chunk types (ChatCompletionChunk, ChunkChoice, Delta, ToolCallDelta, FunctionCallDelta) for incremental response delivery.

## Error Types
ApiError enum implements IntoResponse for axum, producing OpenAI-style JSON error responses with typed error codes.

## Shared Types
ToolCall and FunctionCall are defined in response and shared with request via module reference.

# Metrics
Prometheus-based metrics collection and exposure for inference server monitoring.

## Registry and Metric Definitions

All metrics use `std::sync::LazyLock` (Rust 1.80+) for lazy initialization instead of `lazy_static`. Seven metrics track inference workload and system resources.

### Counters
Metrics that track monotonically increasing values over the lifetime of the server.

#### Tokens Generated

Total count of tokens generated across all inference requests. Monotonically increasing.

### Gauges
Metrics that track instantaneous values which can go up or down.

#### Active Sessions

Current number of active inference sessions being processed.

#### KV Cache Usage Bytes

Current memory usage of the key-value cache in bytes.

#### Batch Size

Current batch size of the inference scheduler.

#### MTP Acceptance Rate

Rate at which MTP (Multi-Token Prediction) draft tokens are accepted.

#### GPU Memory Usage Bytes

Current GPU memory consumption in bytes.

### Histograms
Metrics that track the distribution of values across configurable buckets.

#### Request Latency

Request latency distribution in seconds with buckets at [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0].

## Metrics HTTP Endpoint

Axum handler exposes all registered metrics at `/metrics` in Prometheus text format.

# Server

Main binary crate for the inference server. Provides CLI argument parsing, Axum-based HTTP routing, and mock handlers for the OpenAI-compatible API.

## CLI Arguments

Uses `clap` derive API. Key arguments include model name, parallelism, KV cache dtype, context length, GPU utilization, speculative decoding, and bind address. All support environment variable override and defaults.

## AppState

Shared state struct holding the model name, wrapped in `Arc` for async-safe sharing across handler calls.

## Route Structure

All API routes registered on the Axum router with middleware layers.

| Path | Method | Handler |
|------|--------|---------|
| `/health` | GET | `health_check` |
| `/v1/models` | GET | `list_models` |
| `/v1/chat/completions` | POST | `chat_completions` |
| `/metrics` | GET | `metrics_handler` |

Routes are wrapped with `TraceLayer` for request logging and `CorsLayer::permissive()` for cross-origin access.

## Chat Completions Handler

Handles the OpenAI-compatible chat completions endpoint with both streaming and non-streaming modes.

### Non-streaming Response

Returns a single `ChatCompletionResponse` with a mock assistant message. Response includes ID, timestamp, model name, one choice with `"stop"` finish reason, and token usage stats.

### Streaming Response

Returns an SSE stream of `ChatCompletionChunk` objects following the OpenAI streaming protocol:

1. **Role delta chunk**: Sets `role: "assistant"` with empty content
2. **Token chunks**: Four incremental token chunks (`"Hello"`, `" from"`, `" infers"`, `"!"`)
3. **Finish chunk**: Empty delta with `finish_reason: "stop"`
4. **[DONE]**: Final SSE event signaling stream completion

Each chunk includes the same request ID, timestamp, and model name.

# Model Config and Format Detection

Config parser and quantization format auto-detection for the infers-model crate. Parses HuggingFace `config.json` to extract Qwen3.6-27B architecture parameters and auto-detects weight quantization format from model directory contents.

## ModelConfig

Parsed from `config.json` with architecture parameters and hybrid attention layer types. See [[crates/model/src/config.rs#ModelConfig]].

### Key Fields

Fields include `num_hidden_layers`, `hidden_size`, `intermediate_size`, `vocab_size`, attention heads, `head_dim`, `max_position_embeddings`, and MTP configuration.

### LayerType Enum

`LayerType` has two variants: `GatedDeltaNet` for linear attention and `FullAttention` for softmax attention. See [[crates/model/src/config.rs#LayerType]].

### Layer Type Pattern

Default pattern: every 4th layer (1-indexed) is full attention, others use GDN linear attention.

## Quantization Format Detection

`QuantizationFormat` enum with `Bf16`, `PrismaScout`, `AutoRound`, and `Gguf` variants. Auto-detection checks for `.gguf` files, `quantization_config.json`, and embedded config. See [[crates/model/src/formats.rs#QuantizationFormat]].

## QuantizationConfig

Parsed from quantization config JSON with arbitrary format-specific fields. See [[crates/model/src/formats.rs#QuantizationConfig]].

# Weight Registry and Tensors

Weight storage structures for model weights as raw byte data with shape metadata, ready for GPU upload.

## WeightData

Raw tensor storage holding `Vec<u8>` bytes, shape dimensions, dtype, and tensor name. Weights stay as bytes until CUDA upload time to avoid requiring GPU hardware at load time. See [[crates/model/src/weights.rs#WeightData]].

## WeightDtype

Enumeration of weight data types: BF16, FP16, FP32, INT4 packed, NVFP4, and Other. Provides `bytes_per_element()` for contiguous formats and `None` for packed layouts. See [[crates/model/src/weights.rs#WeightDtype]].

## Layer Weight Structures

Typed weight structures for GDN layers (`GdnWeights`), attention layers (`AttentionWeights`), and MLP layers (`MlpWeights`). Each contains named `WeightData` fields matching safetensors tensor names. See [[crates/model/src/weights.rs#GdnWeights]], [[crates/model/src/weights.rs#AttentionWeights]], [[crates/model/src/weights.rs#MlpWeights]].

## WeightRegistry

Complete model weight registry with embedding, layers, optional MTP head, LM head, norm, and a `HashMap<String, WeightData>` for name-based lookup and sharding. See [[crates/model/src/weights.rs#WeightRegistry]].

# Safetensors Loader

Multi-format model loader with safetensors file reading and auto-detection of single vs sharded model files.

## Loading Pipeline

The `load_model()` function is the main entry point: it reads `config.json`, detects quantization format, loads safetensors files, and constructs a `WeightRegistry`. See [[crates/model/src/loader.rs#load_model]].

## Single vs Sharded

`load_safetensors()` auto-detects whether a model uses a single `model.safetensors` file or a sharded index (`model.safetensors.index.json` with multiple shard files). Memory maps files for efficient loading. See [[crates/model/src/loader.rs#load_safetensors]].

# Weight Sharding

Weight sharding for tensor parallelism (TP=2) and pipeline parallelism (PP=2).

## Tensor Parallelism Sharding

`shard_weights_tp()` splits weights across GPUs. Column-parallel tensors are sliced along dim 0, row-parallel along last dim. Norms and embeddings are replicated. See [[crates/model/src/sharding.rs#shard_weights_tp]].

## Shard Type Detection

Tensor names determine sharding type. Projections like Q/K/V/gate/up are column-parallel; O/down are row-parallel. All others are replicated. See [[crates/model/src/sharding.rs#determine_shard_type]].

## Pipeline Parallelism Split

`split_layers_pp()` divides layers evenly across pipeline stages. For 64 layers and 2 stages: stage 0 gets layers 0-31, stage 1 gets layers 32-63. See [[crates/model/src/sharding.rs#split_layers_pp]].

# Tech Debt Fixes

Production hardening changes applied to improve error handling, safety, and code quality.

## Error Handling

Critical `.unwrap()` calls in `main.rs` replaced with `?` propagation. The `run()` function returns `anyhow::Result<()>` with proper context on bind and serve failures.

## Metrics Handler

`metrics_handler()` now returns `Result<impl IntoResponse, StatusCode>` instead of `impl IntoResponse`. Encoding errors and UTF-8 conversion errors return `INTERNAL_SERVER_ERROR` instead of panicking. See [[crates/metrics/src/lib.rs#metrics_handler]].

## Metrics Registry

Metric creation `.unwrap()` calls changed to `.expect()` with descriptive error messages. This makes it clear that metric creation failures indicate duplicate metric names or registry errors, not normal failures. See [[crates/metrics/src/registry.rs]].

## Safety Comments

`memmap2::Mmap::map()` calls have `// SAFETY:` comments explaining that the file is opened read-only, the file handle is verified to exist before mapping, and the mapping is read-only weight data. See [[crates/model/src/loader.rs#load_single]], [[crates/model/src/loader.rs#load_sharded]].

## Memory Budget Improvements

`MemoryBudget` now includes `max_position_embeddings` from the model config instead of hardcoding `262144`. The workspace size is defined as `DEFAULT_WORKSPACE_BYTES` constant. The `max_position_tokens()` helper method was removed. See [[crates/model/src/budget.rs#MemoryBudget]].

## Config Constants

`FULL_ATTENTION_INTERVAL` constant replaces the hardcoded `4` in `default_layer_type()`. See [[crates/model/src/config.rs#ModelConfig#default_layer_type]].

## Weight Registry Safety

`WeightRegistry` now uses `Option<WeightData>` for `embedding`, `lm_head`, and `norm` fields instead of empty placeholder `WeightData` structs. See [[crates/model/src/weights.rs#WeightRegistry]].

## GpuAllocator Encapsulation

`GpuAllocator` fields are now private with accessor methods. The `free()` method has overflow protection and derives `Debug` and `Clone`.

## Build Script Safety

`build.rs` uses `parent().unwrap_or(Path::new("."))` instead of `parent().unwrap()` for output path directory creation. See [[crates/cuda/build.rs]].

## GEMM Leading Dimension Fix

`GemmConfig` to `MatmulConfig` conversion in `gemm_impl` had incorrect column-major leading dimension defaults. Fixed `lda` branches (transa=true→k, transa=false→m) and `ldc` default (m, not n) to match cuBLASLt column-major convention. See [[crates/cuda/src/gemm.rs#gemm_impl]].

## SSE Constant Usage

Chat handler uses `SSE_DONE` constant from `infers_api` instead of hardcoded `"[DONE]"` string. See [[crates/server/src/handlers/chat.rs]].

## Kernel Registry Documentation

`LoadedKernelRegistry` documents its role in GPU kernel execution, loading pre-compiled .cubin files via cudarc.

## Kernel Dispatch Fixes

Six critical kernel dispatch bugs were fixed to align Rust dispatch code with CUDA kernel signatures.

### RMSNorm Shared Memory

`infers_rmsnorm_bf16` uses `extern __shared__ float shared_mem[]` for a block-wide reduction with 256 threads. Shared memory was set to 0, causing undefined behavior. Fixed to `256 * sizeof(f32) = 1024` bytes. See [[crates/backends/native/src/norm.rs#rms_norm]].

### Embedding Argument Order

Kernel dispatch passed arguments in wrong order and included an extra parameter. Fixed to match CUDA signature `infers_embedding_gather_bf16(weight, token_ids, output, seq_len, hidden_size)`. See [[crates/backends/native/src/embedding.rs#embed_tokens]].

### Argmax Missing Batch Size

`infers_argmax_f32` expects `(logits, output, batch_size, vocab_size)` but the dispatch code only passed 3 args, leaving `batch_size` unfilled (vocab_size was passed into the batch_size slot). Added `batch_size_i32 = 1` argument. See [[crates/backends/native/src/sample.rs#greedy_sample]].

### GEMM Transpose Flags

GEMM transpose flags were inverted for row-major weight storage. All three `matmul_bf16` calls in `mlp_forward` changed from `transa: false, transb: true` to `transa: true, transb: false` so cuBLASLt correctly interprets row-major inputs. See [[crates/backends/native/src/mlp.rs#mlp_forward]].

### RoPE Table Indexing

`precompute_rope_tables` allocated a compact table indexed by token index, but the CUDA kernel indexed by position value. Fixed to size table by max position and index by position. See [[crates/backends/native/src/rope.rs#precompute_rope_tables]].

### RoPE Multi-Head Rotation

RoPE kernel only rotated the first head, ignoring the `num_heads` dimension in tensor layout. Fixed by adding `num_heads` parameter and iterating all head-dimension pairs. See [[crates/backends/native/src/rope.rs#apply_rope]].

### Paged Attention Decode Kernel Fixes

Two critical bugs fixed in `infers_paged_attention_decode_bf16` in `paged_attention_decode.cu`:

1. **Strided token accumulation dropped ~99% of contributions** (CRITICAL): Phase 2 loop used `token_pos += bdim` stride, so each thread only accumulated V values for tokens at positions tid, tid+bdim, tid+2*bdim... — one thread processed one output dimension but only saw 1/bdim of the tokens. Fixed by having each thread (tid < head_dim) loop over ALL tokens independently.
2. **Block reduction summed across different output dimensions** (CRITICAL): The reduction loop added partial outputs for dimension 0, dimension 1, ..., dimension bdim-1 together into a single scalar. Fixed by removing the reduction entirely — each thread writes directly to `output[head_idx * head_dim + tid]`.

### Softmax Kernel Fixes

Three bugs fixed in `crates/cuda/kernels/infers/softmax.cu`:

1. **Max value overwritten by sum** (CRITICAL): Phase 2's sum reduction overwrote `sdata[0]`, causing Phase 3 normalization to subtract the sum instead of the max. Fixed by saving `sdata[0]` to a `max_val` register after Phase 1, then using `max_val` in Phase 2 (exp computation) and Phase 3 (normalization).
2. **Race condition** (implicit in bug 1): Reading `sdata[0]` for the max during Phase 2 created a race with concurrent sum reduction writes. Resolved by using the `max_val` register instead of shared memory.
3. **Non-power-of-2 reduction** (MINOR): Tree reduction with `stride >>= 1` silently drops tail elements for non-power-of-2 block sizes. Fixed by rounding `block_size` up to the next power of 2 and adding `tid + stride < blockDim.x` bounds checks in both reduction loops.

### GDN Kernel Fixes

Two bugs fixed in both `gdn_update.cu` and `gdn_prefill.cu`:

1. **Non-power-of-2 reduction** (MINOR): Host wrappers used `hidden_size` directly as block size, causing tree reduction to silently drop tail elements for non-power-of-2 hidden sizes. Fixed by restructuring kernels: `__global__` kernels are now placed inside `extern "C"` with names matching the registry (`infers_gdn_update_bf16`, `infers_gdn_prefill_bf16`), eliminating host wrappers. Rust-side launch configuration computes power-of-2 block sizes. Reduction loops use `tid + stride < blockDim.x` bounds checks.
2. **Kernel name mismatch** (CRITICAL): `LoadedKernelRegistry` registered `infers_gdn_update_bf16` and `infers_gdn_prefill_bf16`, but the cubins only contained C++-mangled `__global__` kernels (`gdn_update_kernel`, `gdn_prefill_kernel`) and host wrappers (not compiled into cubins). Fixed by renaming `__global__` kernels to match registered names and wrapping them in `extern "C"` for unambiguous C linkage in the cubin.

# Memory Budget

Memory budget calculator for estimating VRAM requirements across different quantization formats and parallelism configurations.

## Budget Calculation

`MemoryBudget::calculate()` estimates weight bytes, KV cache bytes, workspace bytes, and available memory per GPU. Accounts for quantization format (BF16, PrismaSCOUT, AutoRound, GGUF), GPU count, VRAM per GPU, and utilization factor. See [[crates/model/src/budget.rs#MemoryBudget#calculate]].

## Weight Size Estimation

`estimate_weight_bytes()` computes parameter count from model config (embedding, attention, MLP, norms, LM head) multiplied by bytes per element for the quantization format. See [[crates/model/src/budget.rs#MemoryBudget#estimate_weight_bytes]].

## KV Cache Estimation

`estimate_kv_cache_bytes()` calculates KV cache per GPU based on full attention layers, KV heads, head dimension, and max position embeddings. Only full attention layers use paged KV cache. See [[crates/model/src/budget.rs#MemoryBudget#estimate_kv_cache_bytes]].

## Concurrent Session Planning

`max_concurrent_sessions()` estimates how many sessions fit in available KV memory given an average context length. Scales linearly with context tokens. See [[crates/model/src/budget.rs#MemoryBudget#max_concurrent_sessions]].
