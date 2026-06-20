This directory defines the high-level concepts, business logic, and architecture of this project. It is managed by [lat.md](https://www.npmjs.com/package/lat.md) — a tool that anchors source code to these definitions.

## Index

Other documentation files:
- [[api]] — API types for the inference server
- [[arch]] — workspace architecture and kernel build system
- [[misc]] — tokenizer, MTP, FP8 quantization, tech debt fixes, phase deliverables
- [[parallel]] — tensor parallelism and pipeline parallelism implementations
- [[testing]] — reference tests, smoke tests, debugging analysis
# Model Config and Format Detection

Config parser and quantization format auto-detection for the infers-model crate. Parses HuggingFace `config.json` to extract Qwen3.6-27B architecture parameters and auto-detects weight quantization format from model directory contents.

## ModelConfig

Parsed from `config.json` with architecture parameters and hybrid attention layer types. See [[crates/model/src/config.rs#ModelConfig]].

### Key Fields

Architecture parameters: layer count, dimensions, attention heads, MTP, GDN linear attention fields, and `attn_output_gate` for Qwen3.5's doubled Q projection (Q + gate). See [[lat.md/lat#Paged Attention Implementation#Attention Output Gate]].

### LayerType Enum

`LayerType` has two variants: `GatedDeltaNet` for linear attention and `FullAttention` for softmax attention. See [[crates/model/src/config.rs#LayerType]].

### Layer Type Pattern

Default pattern: every 4th layer (1-indexed) is full attention, others use GDN linear attention.

### Text Config Merging

Multimodal model configs wrap architecture parameters inside a `text_config` object. `[[crates/model/src/config.rs#merge_text_config]]` performs a shallow merge: `text_config` fields are promoted to the root level, but root-level keys take priority. See [[crates/model/src/config.rs#ModelConfig#load]].

## Quantization Format Detection

`QuantizationFormat` enum with `Bf16`, `PrismaScout`, `AutoRound`, and `Gguf` variants. Auto-detection checks for `.gguf` files, `quantization_config.json`, and embedded config. See [[crates/model/src/formats.rs#QuantizationFormat]].

## QuantizationConfig

Parsed from quantization config JSON with arbitrary format-specific fields. See [[crates/model/src/formats.rs#QuantizationConfig]].

# Weight Registry and Tensors

Weight storage structures for model weights as raw byte data with shape metadata, ready for GPU upload.

## WeightData

Raw tensor storage holding `bytes::Bytes` bytes, shape dimensions, dtype, and tensor name. Weights stay as bytes until CUDA upload time to avoid requiring GPU hardware at load time. See [[crates/model/src/weights.rs#WeightData]].

## WeightDtype

Enumeration of weight data types: BF16, FP16, FP32, INT4 packed, NVFP4, and Other. Provides `bytes_per_element()` for contiguous formats and `None` for packed layouts. See [[crates/model/src/weights.rs#WeightDtype]].

## Layer Weight Structures

Typed weight structures for GDN layers (`GdnWeights`), attention layers (`AttentionWeights`), and MLP layers (`MlpWeights`).

`GdnWeights` has 4 required fields (in_proj_a, in_proj_b, conv1d_weight, out_proj_weight) plus 2 optional projection fields (x_proj_weight, dt_proj_weight) that may be absent in Qwen3.6 AutoRound INT4 models, and 5 optional Mamba2-style fields (in_proj_qkv, in_proj_z, a_log, dt_bias, norm) that are present in Qwen3.6 real models. See [[crates/model/src/weights.rs#GdnWeights]].

`AttentionWeights` has 4 required fields (q_proj, k_proj, v_proj, o_proj) plus 2 optional fields (q_norm, k_norm) for Q/K normalization in full attention layers. See [[crates/model/src/weights.rs#AttentionWeights]].

`MlpWeights` has 3 fields (gate_proj, up_proj, down_proj). See [[crates/model/src/weights.rs#MlpWeights]].

## MtpWeights

MTP head weights with pre-FC norms, FC projection, full transformer layers, final norm, and optional dedicated embeddings. See [[crates/model/src/weights.rs#MtpWeights]].

## WeightRegistry

Complete model weight registry with embedding, layers, optional MTP head, LM head, norm, and a `HashMap<String, WeightData>` for name-based lookup and sharding. See [[crates/model/src/weights.rs#WeightRegistry]].

## MmapTensor and MmapWeightRegistry

Zero-copy tensor references in memory-mapped safetensors files. `MmapTensor` wraps `DataOwner` (`Arc<Mmap>`) with shape, dtype, and name. Non-contiguous shards use strided metadata for cuMemcpy2D DMA upload — no CPU copies needed. See [[crates/model/src/mmap.rs#MmapTensor]].

### DataOwner

Simple wrapper around `Arc<Mmap>` keeping mmap mappings alive across tensor references. Non-contiguous shards no longer use heap-owned buffers — they store strided metadata for cuMemcpy2D DMA transfers. See [[crates/model/src/mmap.rs#DataOwner]].

### MmapCompanions

Zero-copy companion tensors (qzeros, scales) for INT4 quantized weights. See [[crates/model/src/mmap.rs#MmapCompanions]].

### MmapWeightRegistry

Memory-mapped equivalent of `WeightRegistry`: stores tensors by name in a HashMap and INT4 companions. Each MmapTensor's DataOwner keeps mmaps alive — no separate `_mmaps` tracking needed. Tracks tensor count and byte sum. See [[crates/model/src/mmap.rs#MmapWeightRegistry]].

#### clear_owned_data

After GPU upload, replaces owned (non-mmap) tensor data with empty slices to free ~2 GB of heap residency on Qwen3.6-27B. Mmap-backed tensors are preserved — their Arc<Mmap> references must stay alive. See [[crates/model/src/mmap.rs#MmapWeightRegistry#clear_owned_data]].

### shard_weights_tp_mmap

Shards MmapWeightRegistry across GPUs for tensor parallelism. All splits are zero-copy on CPU: contiguous splits use pointer offsets, non-contiguous splits produce strided tensors uploaded via cuMemcpy2D DMA. See [[crates/model/src/mmap.rs#shard_weights_tp_mmap]].

Follows the same sharding rules as `shard_weights_tp()` but operates on MmapTensor references. Dispatch table:
- **ColumnParallel + BF16**: Split dim 0 (rows) — contiguous, zero-copy via pointer offset into same mmap
- **ColumnParallel + INT4**: Split dim 1 (N, last dim) — non-contiguous, strided tensor with cuMemcpy2D DMA upload
- **RowParallel + BF16**: Split dim 1 (last dim) — non-contiguous, strided tensor with cuMemcpy2D DMA upload
- **RowParallel + INT4**: Split dim 0 (rows) — contiguous, zero-copy via pointer offset
- **Replicated**: Clone MmapTensor (cheap Arc clone)

INT4 companion tensors (.scales, .qzeros) are extracted from `registry.tensors`, sharded alongside their qweight parent (strided when needed), and stored in `int4_companions`. Fused QKV projections (`in_proj_qkv`) use per-segment column splitting — each segment (Q, K, V) is independently divided by num_gpus via [[crates/model/src/mmap.rs#mmap_shard_fused_projection_columns]]. The `conv1d.weight` tensor uses per-segment row splitting via [[crates/model/src/mmap.rs#mmap_shard_fused_projection_rows]], extracting Q/K/V row segments independently rather than naively dividing dim 0. Both match the heap path's [[lat.md/lat#Weight Sharding#Tensor Parallelism Sharding]] behavior.

MmapWeightShard holds gpu_id and the shard's MmapWeightRegistry. See [[crates/model/src/mmap.rs#MmapWeightShard]].

### build_metadata_registry

Converts an `MmapWeightRegistry` into a `WeightRegistry` with metadata only — empty data but correct shapes, dtypes, and names. Used so that `self.weights[gpu_idx]` has structure for name lookups during inference. See [[crates/model/src/mmap.rs#build_metadata_registry]].

Multi-format model loader with safetensors file reading and auto-detection of single vs sharded model files.

## Loading Pipeline

`load_model()` reads config, detects format, loads safetensors, then calls `build_mtp_weights()` if MTP is enabled. See [[crates/model-loader-heap/src/lib.rs#load_model]].

## Single vs Sharded

`load_safetensors()` auto-detects whether a model uses a single `model.safetensors` file or a sharded index (`model.safetensors.index.json` with multiple shard files). Memory maps files for efficient loading. See [[crates/model-loader-heap/src/lib.rs#load_safetensors]].

## MTP Weight Loading

`build_mtp_weights()` extracts MTP tensors from `registry.tensors` and populates `registry.mtp`. Supports GDN and full attention layers. See [[crates/model/src/loader.rs#build_mtp_weights]].

# Memory Budget

VRAM estimation for model weights, KV cache, and workspace memory based on model configuration and quantization format.

## Budget Calculation

Total VRAM budget computed as weight memory + KV cache memory + workspace overhead. Accounts for TP sharding reduction when multiple GPUs are used. See [[crates/model/src/budget.rs#MemoryBudget]].

## Weight Size Estimation

Estimates GPU weight memory from model dimensions and quantization format: BF16 (2 bytes/element), INT4 (0.5 bytes/element with companion tensors). See [[crates/model/src/budget.rs#MemoryBudget#estimate_weight_bytes]].

## KV Cache Estimation

Calculates KV cache VRAM based on hidden size, number of layers, max sequence length, and number of concurrent sessions. See [[crates/model/src/budget.rs#MemoryBudget#estimate_kv_cache_bytes]].

## Concurrent Session Planning

Determines maximum concurrent sessions given available VRAM by dividing free memory between weight cache and KV cache, then computing session capacity from per-session KV overhead. See [[crates/model/src/budget.rs#MemoryBudget#max_concurrent_sessions]].
# Paged KV Types

Paged KV cache data structures for managing key-value state in discrete memory pages with copy-on-write semantics and prefix caching.

## SequenceId

Unique identifier for a sequence (session) in the paged KV manager. See [[crates/kv/src/manager.rs#SequenceId]].

## ManagerError

Error type for PagedKvManager operations including page allocation failures and cache errors. See [[crates/kv/src/manager.rs#ManagerError]].

## PagedKvManager

Core manager for paged KV cache with page allocation, deallocation, and eviction support. Tracks sequences, pages, and prefix cache entries. See [[crates/kv/src/manager.rs#PagedKvManager]].

### Eviction

LRU-based eviction of idle or low-priority sequences when memory pressure exceeds thresholds. Configurable via PressureConfig. See [[crates/kv/src/manager.rs#PagedKvManager]].

## SequencePageTable

Per-sequence mapping from logical token positions to physical page IDs, supporting block-table address translation for paged attention kernels. See [[crates/kv/src/table.rs#SequencePageTable]].

## PagePool

Memory pool allocator for KV cache pages, tracking free and allocated pages with size constraints. See [[crates/kv/src/pool.rs#PagePool]].

## PrefixCache

Shared prefix caching: when multiple sequences share a common prefix, their KV pages are shared via copy-on-write to avoid redundant computation. See [[crates/kv/src/prefix.rs#PrefixCache]].

## CowResult

Copy-on-write result type for page operations, yielding either a shared reference or a new mutable copy. See [[crates/kv/src/cow.rs#CowResult]].

## CowError

Errors from copy-on-write page operations including allocation failures and invalid references. See [[crates/kv/src/cow.rs#CowError]].

## ensure_mutable_page

Copies a shared page to a private mutable buffer when write access is needed, preserving prefix cache sharing until mutation occurs. See [[crates/kv/src/cow.rs#ensure_mutable_page]].

## decrement_page_refcount

Decrements the reference count for a paged KV cache page and triggers deallocation when count reaches zero. See [[crates/kv/src/cow.rs#decrement_page_refcount]].

## try_share_from_prefix_cache

Attempts to share an existing page from the prefix cache instead of allocating a new one, reducing memory pressure for common prefixes. See [[crates/kv/src/cow.rs#try_share_from_prefix_cache]].
# Paged Attention Implementation

CUDA kernel implementations for paged attention with block-table address translation.

## BackendEvictionStore

Callback-based eviction store connecting the backend engine to the scheduler's eviction decisions. See [[crates/backends/native/src/eviction.rs]].

## PagedKvCache

GPU-resident paged KV cache storage with interleaved per-page layout for K and V tensors, addressing via block tables in paged attention kernels. See [[crates/backends/native/src/attention.rs#PagedKvCache]].

## Paged Kernel Dispatch

Dispatch logic for paged attention CUDA kernels, translating logical positions to physical page offsets via block tables before kernel launch. See [[crates/backends/native/src/attention.rs]].

## Attention Output Gate

Qwen3.5-specific attention output gating: Q projection is doubled (Q + gate) with per-head interleaved layout. The gate is applied as `attn_out * sigmoid(gate)` to produce the final attention output. See [[crates/backends/native/src/attention.rs]].

# Forward Engine

The main inference engine coordinating prefill, decode, and weight upload operations across GPUs.

## Prefill Path

Full-sequence prefill path: runs embedding lookup through all layers (GDN or full attention) for a batch of input tokens. See [[crates/backends/native/src/prefill.rs]].

## Paged Prefill Path

Prefill using paged KV cache: writes K/V into pages during the forward pass via block-table address translation instead of contiguous cache buffers. See [[crates/backends/native/src/engine.rs]].

## Paged Decode Path

Decode using paged KV cache: reads K/V from pages during attention computation via block tables, supports single-token generation with paged attention kernels. See [[crates/backends/native/src/engine.rs]].

## INT4 Triplet Upload

GPU weight upload for INT4 quantized weights: handles qweight, scales, and qzeros as a triplet with proper dequantization layout. See [[crates/backends/native/src/upload.rs]].

## General Instrumentation Probe

Per-layer probe infrastructure for dumping intermediate tensors during inference via `INFERS_DUMP_DIR`, `INFERS_DUMP_LAYERS`, and `INFERS_DUMP_STAGES` environment variables. Writes raw bf16 bytes plus JSON metadata sidecars. See [[crates/backends/native/src/probe.rs]].

# Mmap Weight Upload

Zero-copy weight upload from memory-mapped safetensors files using cuMemcpy2D DMA transfers. Handles contiguous tensors and strided non-contiguous shards, eliminating CPU buffer allocations. See [[crates/cuda/src/memcpy2d.rs]].

# GpuWeightCache

GPU-resident weight cache managing tensor lifetimes across CUDA streams with download support for debugging.

## GPU Buffer Download

Readback of GPU-resident weight tensors from device memory to host CPU buffers. Used in smoke tests to verify INT4 qweight data between heap and mmap engine paths. See [[crates/backends/native/src/gpu_cache.rs]].

# Metrics

Prometheus-based metrics collection and export for monitoring inference server performance.

## Registry and Metric Definitions

Centralized metric registry with typed counter, gauge, and histogram definitions.

### Counters

Monotonically increasing metrics tracking total events.

#### Tokens Generated

Counter tracking total tokens generated across all sessions. See [[crates/metrics/src/registry.rs]].

### Gauges

Point-in-time value metrics tracking current state.

#### Active Sessions

Gauge tracking number of currently active inference sessions. See [[crates/metrics/src/registry.rs]].

#### KV Cache Usage Bytes

Gauge tracking bytes consumed by the KV cache across all pages. See [[crates/metrics/src/registry.rs]].

#### Batch Size

Gauge tracking current batch size during decode steps. See [[crates/metrics/src/registry.rs]].

#### MTP Acceptance Rate

Gauge tracking speculative token acceptance rate for MTP drafting. See [[crates/metrics/src/registry.rs]].

#### GPU Memory Usage Bytes

Gauge tracking GPU memory consumption by the weight cache and KV cache combined. See [[crates/metrics/src/registry.rs]].

### Histograms

Distribution metrics capturing value ranges over time.

#### Request Latency

Histogram tracking end-to-end request latency in milliseconds. See [[crates/metrics/src/registry.rs]].

## Metrics HTTP Endpoint

HTTP handler at `/metrics` exposing Prometheus-formatted metric data with proper Content-Type headers. See [[crates/metrics/src/lib.rs#metrics_handler]].

# Scheduler

Scheduler types and configuration for managing inference sessions, batching, and resource allocation.

## Sampling Strategy

Strategy for selecting next-token sampling method: greedy (argmax) or stochastic. See [[crates/scheduler/src/queue.rs#SamplingStrategy]].

## SamplingConfig

Configuration for token sampling including temperature, top_p, top_k parameters. See [[crates/scheduler/src/queue.rs#SamplingConfig]].

## Request

Scheduler request type wrapping a session ID, prompt tokens, and sampling configuration. See [[crates/scheduler/src/queue.rs#Request]].

## RequestQueue

Priority-based queue managing incoming inference requests with preemption support under memory pressure. See [[crates/scheduler/src/queue.rs#RequestQueue]].

## Session

Scheduler session tracking state for each active inference request, including token buffer and sequence metadata. See [[crates/scheduler/src/session.rs#Session]].

### Session State

Per-session state machine: New → Prefilling → Decoding → Complete / Evicted. See [[crates/scheduler/src/session.rs#SessionState]].
# Weight Sharding

Weight tensor sharding strategies for tensor parallelism and pipeline parallelism across multiple GPUs.

## Tensor Parallelism Sharding

Splits weight tensors across GPUs for tensor parallelism using column-parallel (dim 0) and row-parallel (dim 1) strategies. INT4 quantized weights produce non-contiguous strided tensors requiring cuMemcpy2D DMA uploads. See [[crates/model/src/mmap.rs#shard_weights_tp_mmap]].
## Shard Type Detection

Automatic detection of how to shard each weight tensor based on layer type, tensor name pattern, and quantization format. Maps column-parallel projections (dim 0 split), row-parallel projections (dim 1 split), and replicated weights. See [[crates/model/src/sharding.rs#ShardType]].

## Pipeline Parallelism Split

Splits the model into pipeline stages by dividing layers evenly across GPUs. Each stage gets a contiguous range of layers with its own weight subset and KV cache. See [[crates/model/src/sharding.rs]].

## Sharding Equivalence Tests

Comprehensive tests verifying that sharded weights produce identical results to full-precision references.

### TP=2 All Weights

Verifies BF16 weight sharding produces correct column and row splits across 2 GPUs. See [[crates/model/tests/shard_equiv.rs]].

### TP=1 All Weights

Verifies that TP=1 (no sharding) preserves all weights unchanged. See [[crates/model/tests/shard_equiv.rs]].

### TP=2 conv1d Fused QKV

Tests per-segment column splitting for fused QKV projections in conv1d layers. See [[crates/model/tests/shard_equiv.rs]].

### TP=2 INT4 Column-Parallel

Verifies INT4 quantized weight sharding with non-contiguous strided tensor handling. See [[crates/model/tests/shard_equiv.rs]].

### TP=2 INT4 Row-Parallel

Tests INT4 row-parallel splits across GPUs. See [[crates/model/tests/shard_equiv.rs]].

### TP=2 GDN Fused QKV in_proj

Tests fused QKV sharding for GDN layers with Mamba2-style projections. See [[crates/model/tests/shard_equiv.rs]].

### TP=2 Strided Metadata Verification

Verifies strided tensor metadata (stride, offset) is computed correctly for non-contiguous splits. See [[crates/model/tests/shard_equiv.rs]].

### TP=2 Strided Data Materialization

Tests that strided tensors produce correct data when read via cuMemcpy2D DMA. See [[crates/model/tests/shard_equiv.rs]].
