This directory defines the high-level concepts, business logic, and architecture of this project. It is managed by [lat.md](https://www.npmjs.com/package/lat.md) — a tool that anchors source code to these definitions.

## Index

Other documentation files:
- [[api]] — API types for the inference server
- [[arch]] — workspace architecture and kernel build system
- [[misc]] — tokenizer, MTP, FP8 quantization, tech debt fixes, phase deliverables
- [[parallel]] — tensor parallelism and pipeline parallelism implementations
- [[testing]] — reference tests, smoke tests, debugging analysis
- [[tests]] — CUDA kernel regression tests (gemm_compare)
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

## QuantTargetMap

Resolved per-tensor quantization assignment from `QuantizationConfig`. Parses `config_groups` target regexes and `ignore` lists to determine which tensors are NVFP4, INT4, or BF16 passthrough.

Resolves stripped tensor names (after `strip_language_model_prefix`) by reconstructing the unstripped form for matching against original config targets. See [[crates/model/src/formats.rs#QuantTargetMap]].

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

## Quantized Companion Tensors

Companion tensors for quantized weights, stored in `HashMap<String, QuantCompanions>` on WeightRegistry. The `[[crates/model/src/weights.rs#QuantCompanions]]` enum unifies INT4 (AutoRound) and NVFP4 (PrismaSCOUT) formats.

### Int4Companions

INT4 quantization companions: packed zero points (`qzeros`) and BF16 group scales. See [[crates/model/src/weights.rs#Int4Companions]].

### Nvfp4Companions

NVFP4 (PrismaSCOUT) quantization companions: per-block weight scale (FP8 E4M3), global tensor scale (BF16), and input activation global scale (BF16). See [[crates/model/src/weights.rs#Nvfp4Companions]].

### QuantCompanions

Unified companion enum supporting both INT4 (AutoRound) and NVFP4 (PrismaSCOUT) quantization formats. See [[crates/model/src/weights.rs#QuantCompanions]].

## WeightRegistry

Complete model weight registry with embedding, layers, optional MTP head, LM head, norm, a flat tensor map, and `quant_companions` for INT4/NVFP4 companion tensors keyed by packed weight name. See [[crates/model/src/weights.rs#WeightRegistry]].

## MmapTensor and MmapWeightRegistry

Zero-copy tensor references in memory-mapped safetensors files. `MmapTensor` wraps `DataOwner` (`Arc<Mmap>`) with shape, dtype, and name. Non-contiguous shards use strided metadata for cuMemcpy2D DMA upload — no CPU copies needed. See [[crates/model/src/mmap.rs#MmapTensor]].

### DataOwner

Simple wrapper around `Arc<Mmap>` keeping mmap mappings alive across tensor references. Non-contiguous shards no longer use heap-owned buffers — they store strided metadata for cuMemcpy2D DMA transfers. See [[crates/model/src/mmap.rs#DataOwner]].

### MmapCompanions

Zero-copy companion tensors (qzeros, scales) for INT4 quantized weights. See [[crates/model/src/mmap.rs#MmapCompanions]].

### MmapNvfp4Companions

Zero-copy companion tensors (weight_scale, weight_global_scale, input_global_scale) for NVFP4 quantized weights in the PrismaSCOUT format. See [[crates/model/src/mmap.rs#MmapNvfp4Companions]].

### MmapQuantCompanions

Unified enum wrapping either `MmapCompanions` (INT4/AutoRound) or `MmapNvfp4Companions` (NVFP4/PrismaSCOUT). Stored in `quant_companions` HashMap of `MmapWeightRegistry`. See [[crates/model/src/mmap.rs#MmapQuantCompanions]].

### MmapWeightRegistry

 Memory-mapped equivalent of `WeightRegistry`: stores tensors by name in a HashMap and quantized companions via `quant_companions: HashMap<String, MmapQuantCompanions>` — unified enum for INT4 or NVFP4. DataOwner keeps mmaps alive. See [[crates/model/src/mmap.rs#MmapWeightRegistry]].

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

INT4 companion tensors (.scales, .qzeros) are extracted from `registry.tensors`, sharded alongside their parent weight (strided when needed), and stored in `quant_companions`. For NVFP4 companions, only `weight_scale` (2D matrix) is sharded with the same strategy; `weight_global_scale` and `input_global_scale` (1D scalars, shape `[1]`) are replicated to all GPUs. Fused QKV projections (`in_proj_qkv`) use per-segment splitting with **generic layout detection** — see [[lat.md/lat#Weight Sharding#Tensor Parallelism Sharding#Fused QKV Layout Detection]]. The `conv1d.weight` tensor uses the same generic detection. Both match the heap path's [[lat.md/lat#Weight Sharding#Tensor Parallelism Sharding]] behavior.
MmapWeightShard holds gpu_id and the shard's MmapWeightRegistry. See [[crates/model/src/mmap.rs#MmapWeightShard]].

### build_metadata_registry

Converts an `MmapWeightRegistry` into a `WeightRegistry` with metadata only — empty data but correct shapes, dtypes, and names. Used so that `self.weights[gpu_idx]` has structure for name lookups during inference. See [[crates/model/src/mmap.rs#build_metadata_registry]].

Multi-format model loader with safetensors file reading and auto-detection of single vs sharded model files.

## Loading Pipeline

`load_model()` reads config, detects format, loads safetensors, then calls `build_mtp_weights()` if MTP is enabled. See [[crates/model-loader-heap/src/lib.rs#load_model]].

## Metadata-Driven Weight Loading

`get_weight_with_quant()` consults `QuantTargetMap` to determine each tensor's quantization format (NVFP4, INT4, or BF16) and extracts the appropriate weight and companion tensors from the registry.

For NVFP4 it extracts `weight_packed` + scale companions; for INT4 it extracts `qweight` + qzeros/scales; for BF16/GGUF it falls through to passthrough. `build_main_layers()` and `build_mtp_weights()` accept a `QuantTargetMap` reference and pass it through the layer-building chain.

See [[crates/model/src/loader.rs#get_weight_with_quant]], [[crates/model/src/loader.rs#build_main_layers]], [[crates/model/src/loader.rs#build_mtp_weights]].

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

### Prefill Debug Checkpoints

eprintln! checkpoints at key stages in [[crates/backends/native/src/engine.rs]] to pinpoint CUDA_ERROR_ILLEGAL_ADDRESS crashes.

Run with `cargo test -- --nocapture`. Seven checkpoints: events recorded, pages allocated, block tables uploaded, page pools allocated, embedding complete, per-layer norm1, and per-layer GDN/attention.

## Paged Decode Path

Paged decode: reads K/V from pages via block tables, single-token generation with paged attention kernels. Zero-allocation — intermediate buffers use pre-allocated [[lat.md/lat#Forward Engine#Decode Workspace]]. See [[crates/backends/native/src/engine.rs]].

## INT4 Triplet Upload

GPU weight upload for INT4 quantized weights: handles qweight, scales, and qzeros as a triplet with proper dequantization layout. See [[crates/backends/native/src/upload.rs]].

## Quantized GEMM Dispatch (Fused INT4, Dequant NVFP4)

GEMM dispatch uses two strategies. INT4 uses the fused `int4_gemm_auto_round` kernel — no intermediate buffer. NVFP4 follows dequantize-to-bf16-then-tiled-kernel pipeline (bypassing cuBLAS for workspace safety).

For **INT4 (AutoRound)**: `gemm_projection_cached` calls `oxide.launch_int4_gemm_auto_round` directly with packed qweight (u32), f16 scales, u32 qzeros, and bf16 input. The kernel dequantizes in registers and accumulates into the output buffer — eliminating ~896 allocations and dequant kernel launches per decode token. Transposition is determined from `int4_bufs.shape` at call time: if shape[1] == n the weight is transposed [K/8, N] layout (AutoRound convention). See [[crates/backends/native/src/gemm_dispatch.rs#gemm_projection_cached]].

### K-Split GEMM for M=1 Decode

K-splitting divides the K dimension across thread blocks, multiplying occupancy by K_SPLIT (28). Two-phase: v1 computes f32 partial sums, then `reduce_partial_sums_bf16` combines them. Production uses **v3 kernels** with four bandwidth optimizations.

#### INT4 V3 Optimizations

Four optimizations in `int4_gemm_v3_ksplit`:

(1) 4 independent f32 accumulators (acc0..acc3) exposing FMA pipeline depth. (2) Group-aligned ceil-grouped K-split — full quantization groups distributed across splits for branchless inner loop, handling non-divisible K_SPLIT. (3) Two-u32 outer-loop stride per group hiding DRAM latency. (4) Per-group `scaled_zero = (raw_zero+1)*scale` hoist saving one FMA per dequantized weight. INT4 decode: 61ms -> 48ms/step (16.4 to 20.8 tok/s).

#### NVFP4 V3 Kernel

`nvfp4_gemm_v3_ksplit` applies identical optimizations to NVFP4 dequant logic (group_size=16). Result: 117ms -> 105ms/step (8.5 to 9.4 tok/s). For **M>1**, fused `nvfp4_gemm_fused` is used directly — no intermediate bf16 buffer. See [[crates/backends/native/src/gemm_dispatch.rs#gemm_projection_cached]].

#### Partial Sums Buffer

Pre-allocated in `DecodeWorkspace` as `partial_sums: CudaSlice<f32>` (K_SPLIT × sharded_intermediate, ~1.4MB). Threaded via `partial_sums_buf`. Decode passes `Some(&mut ws.partial_sums)`, prefill/LM head `None`. Eliminates ~800 alloc_zeros per token.
#### Warp-Cooperative Kernels (Benchmark Result)
`int4_gemm_warp` and `int4_gemm_warp_split` were measured slower than naive ksplit: 158ms and 161ms vs 61ms — with group_size=128 and K=5120 there are only 40 groups, too few for 32-lane shuffle reduction. Kept compiled but not wired into dispatch.

When `m==1` (single-token decode), the naive INT4 kernel has only `ceil(N/64)` thread blocks — e.g., N=5120 gives 80 blocks = 1.6 blocks/SM = 3.2 warps/SM on RTX 5090. Insufficient to hide VRAM latency (~400ns).
### eprintln Gating in GEMM Dispatch

GEMM debug output is gated behind `INFERS_DEBUG` env var via `OnceLock<bool>`, eliminating ~400 stderr writes per decode step. See [[crates/backends/native/src/gemm_dispatch.rs#gemm_projection_cached]].

## RoPE Table Caching

Precomputed cos/sin tables are uploaded to GPU at engine init and passed through the decode path, removing 3 synchronize calls per RoPE invocation. See [[crates/backends/native/src/engine.rs#ForwardEngine]], [[crates/backends/native/src/rope.rs#apply_rope]].

## General Instrumentation Probe

Per-layer probe infrastructure for dumping intermediate tensors during inference via `INFERS_DUMP_DIR`, `INFERS_DUMP_LAYERS`, and `INFERS_DUMP_STAGES` environment variables. Writes raw bf16 bytes plus JSON metadata sidecars. See [[crates/backends/native/src/probe.rs]].

## Logit Dump Debug Tool

Debug feature for diagnosing stuck-token issues by printing top-5 logits at each decode step.

Enabled via `INFERS_DUMP_LOGITS=1` environment variable. Downloads the full logits tensor from GPU 0 to CPU after LM head projection, computes statistics (max, min, standard deviation), sorts descending to find top-5 tokens, and prints to stderr in the format `[LOGIT-DUMP] step={step} top5=[(token_id, logit_value), ...] max_logit={max} min_logit={min} logit_std={std}`. See [[crates/backends/native/src/engine.rs]].

## Zero-Allocation _into Variants

Zero-allocation variants of `rms_norm` and `add` writing into caller-provided buffers instead of allocating. Eliminates per-token GPU malloc in the decode hot path.

Original functions (`rms_norm`, `add`) allocate new GPU buffers each call. The `_into` variants accept a pre-allocated `output` parameter, reducing allocation pressure from hundreds of calls per token to zero.
`[[crates/backends/native/src/norm.rs#rms_norm_into]]` — RMSNorm writing into a pre-allocated buffer. Validates output size before kernel launch. See `[[crates/backends/native/src/norm.rs#rms_norm_into]]`.

`[[crates/backends/native/src/add.rs#add_into]]` — element-wise addition writing into a pre-allocated buffer. Validates both input sizes and output capacity. See `[[crates/backends/native/src/add.rs#add_into]]`.

## Decode Workspace

Pre-allocated GPU workspace buffers for the decode hot path. `DecodeWorkspace` holds all intermediate buffers allocated once per GPU at engine init, eliminating hundreds of `alloc_zeros` calls per token. Fully wired into `decode_paged`. See [[crates/backends/native/src/workspace.rs#DecodeWorkspace]].

New `attn_out` field provides a shared output buffer for both GDN and attention decode outputs, eliminating the `attn_outputs: Vec<CudaSlice>` and its per-layer allocation. The `gdn` field holds a nested `GdnWorkspace` with 16 pre-allocated buffers covering all GDN intermediate allocations (mixed_qkv, conv_input, conv_out, query/key/value, expanded heads, projection outputs, recurrent step output, z-gate, norm output, and fallback zero buffers for a_log/dt_bias). See [[crates/backends/native/src/workspace.rs#GdnWorkspace]].

The `attn` field holds a nested `AttnWorkspace` with 11 pre-allocated buffers covering all attention decode intermediate allocations (k_single, v_single, k_norm_out, q_dummy, q_full, q_buf, gate_buf, q_norm_out, k_rope_dummy, attn_output, gated). See [[crates/backends/native/src/workspace.rs#AttnWorkspace]].

The `partial_sums: CudaSlice<f32>` field (sized K_SPLIT x sharded_intermediate, ~1.4MB) is pre-allocated once at engine init and threaded through GEMM dispatch, eliminating ~800 alloc_zeros calls per token. See [[lat.md/lat#Forward Engine#Quantized GEMM Dispatch (Fused INT4, Dequant NVFP4)#K-Split GEMM for M=1 Decode]].
`gdn::decode_forward` is now fully wired to use workspace buffers: takes `ws: &mut GdnWorkspace` and `output: &mut CudaSlice<bf16>` parameters, returns `Result<()>`, and eliminates all per-token allocations (zero `alloc_zeros`, zero `.clone()` calls). The `_into` helper variants (`copy_view_into`, `repeat_interleave_heads_into`) replace their allocating counterparts. NCCL all-reduce and residual add operate directly on `workspace.attn_out`. See [[crates/backends/native/src/gdn.rs#decode_forward]].

`attention::decode_forward_paged` is now fully wired to use workspace buffers: takes `ws: &mut AttnWorkspace` and `output: &mut CudaSlice<bf16>` parameters, returns `Result<()>`, and eliminates all per-token allocations (zero `alloc_zeros` calls). The Q/gate extraction uses device-to-device memcpy into workspace buffers instead of allocating new ones. K-norm and Q-norm use `rms_norm_into` writing into `ws.k_norm_out`/`ws.q_norm_out`. Paged attention decode uses `paged_attention_decode_into` writing directly into `ws.attn_output`. The final O-projection writes directly into `output` (i.e., `workspace.attn_out`), eliminating the post-attention `memcpy_dtod` copy in engine.rs. See [[crates/backends/native/src/attention.rs#decode_forward_paged]].

# Mmap Weight Upload

Zero-copy weight upload from memory-mapped safetensors files using cuMemcpy2D DMA. Supports BF16, FP16, FP32, INT4 packed, and NVFP4 quantized weights via companion detection in `quant_companions` map. See [[crates/cuda/src/memcpy2d.rs]].

## NVFP4 Mmap Upload

NVFP4 weights are detected when their key has `MmapQuantCompanions::Nvfp4(_)` companions, since mmap dtype mapping classifies U8 as `WeightDtype::Other`.

Both contiguous and strided paths handle: weight_packed (u8 bytes), weight_scale (FP8 E4M3 as u8 bytes), weight_global_scale (f32 scalar read from companion tensor), and input_global_scale (f32 scalar read from companion tensor). Strided path copies data to contiguous buffers before GPU upload, same pattern as INT4. See [[crates/backends/native/src/gpu_cache.rs#upload_strided_mmap_tensor]], [[crates/backends/native/src/gpu_cache.rs#upload_contiguous_mmap_tensor]].

# GpuWeightCache

GPU-resident weight cache with BF16/INT4/NVFP4 weights and pre-converted f32 GDN parameters.

## F32 Buffers for GDN Parameters

The `f32_buffers` map stores a_log/dt_bias as f32 (converted at load time), eliminating ~96 syncs per token. See [[crates/backends/native/src/gpu_cache.rs#GpuWeightCache]], [[crates/backends/native/src/gdn.rs#decode_forward]].

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

Scheduler request type wrapping prompt tokens, sampling configuration, priority, and routing ID. See [[crates/scheduler/src/queue.rs#Request]].

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

### NVFP4 Companion Sharding
Companion tensors go into companion_skip. Only `weight_scale` (2D) is sharded with its parent strategy. Scalar companions (`weight_global_scale`, `input_global_scale`) are **replicated** — cloned to all GPUs, never sliced.
### Fused QKV Layout Detection

Generic detection of which axis holds the fused output dimension by comparing weight shape against segment endpoints. INT4 qweight `(K/8, N=fused_dim)` splits dim 1; NVFP4 weight_packed `(N=fused_dim, K/2)` splits dim 0.

This eliminates hardcoded layout branching on tensor names and prevents out-of-bounds panics when new quantization formats appear.
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
