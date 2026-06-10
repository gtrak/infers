This directory defines the high-level concepts, business logic, and architecture of this project using markdown. It is managed by [lat.md](https://www.npmjs.com/package/lat.md) — a tool that anchors source code to these definitions. Install the `lat` command with `npm i -g lat.md` and run `lat --help`.

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
| infers-backend-native | crates/backends/native | FlashInfer + cuBLASLt backend |
| infers-backend-gguf | crates/backends/gguf | llama.cpp backend |
| infers-cuda | crates/cuda | cuda-oxide + cudarc hybrid |
| infers-parallelism | crates/parallelism | TP=2 and PP=2 implementations |
| infers-tokenizer | crates/tokenizer | HF tokenizers wrapper |
| infers-metrics | crates/metrics | Prometheus exporter |
| infers-mtp | crates/mtp | MTP draft/verify |


## Dependency Graph
Crate dependency relationships and feature propagation between workspace members.

infers-backend-native and infers-parallelism both depend on infers-cuda for GPU kernel loading and NCCL communication respectively. The `cuda` feature propagates through: `infers-backend-native/cuda` -> `infers-cuda/cuda` -> `cudarc`. Same propagation applies to `infers-parallelism/cuda`.

## Toolchain
Nightly toolchain configuration, Rust edition, and cargo-oxide requirements for CUDA support.

- Rust nightly-2026-04-03 with rust-src, rustc-dev, llvm-tools
- edition = "2024"
- cargo-oxide for CUDA crates

## Dependencies

Key workspace dependencies pinned to exact versions: tokio 1.52.3, axum 0.8.9, serde 1.0.228, clap 4.6.1, prometheus 0.14.0, thiserror 2.0.18. cudarc 0.19.7 with cublaslt, nccl, cuda-13020 features. cuda-oxide crates deferred to future phase.

## CUDA Crate

Feature-gated CUDA runtime crate for GPU inference. The `cuda` feature gates cudarc so the crate compiles on non-CUDA machines, providing stub types that panic at runtime when the feature is disabled.

### Feature Gate Pattern

The `cuda` feature enables cudarc. Without it, all GPU types are stubs that bail on construction with a descriptive error.

### Module Structure

Six modules cover context, streams, memory, kernels, GEMM, and NCCL.

| Module | Purpose |
|--------|---------|
| context | CUDA device context management, CudaRuntime |
| stream | CUDA stream pool for async execution |
| memory | Block pool GPU memory allocator (pure Rust, no cfg) |
| kernels | Kernel registry for pre-compiled .cubin loading |
| gemm | cuBLASLt GEMM engine with `matmul()` method for matrix multiplication |
| nccl | Multi-GPU collective operations for TP/PP |

# Kernel Extraction and Build System
Pipeline for extracting FlashInfer kernel source from vLLM and compiling to .cubin binaries.

### Kernel Directory Structure
Three directories hold kernel source and compiled binaries under `crates/cuda/kernels/`. All directories contain `.gitkeep` files for git tracking.

| Directory | Contents |
|-----------|----------|
| `flashinfer-gdn/` | GDN (Gated DeltaNet) kernel source (.cu, .cuh, include/) |
| `flashinfer-attn/` | Standard attention kernel source (prefill, decode, sampling) |
| `compiled/` | Compiled .cubin output from nvcc |

### Extraction Script
Copies kernel source from a local vLLM checkout into `crates/cuda/kernels/`.

Handles three sources: GDN kernels, standard attention kernels, and FlashInfer submodule headers. Missing directories produce warnings but do not abort.

### Build Script
Compiles kernel source files to .cubin binaries using nvcc.

Targets `sm_100a` (Blackwell) by default, configurable via the `INFERS_CUDA_ARCH` environment variable. With `-O3 --use_fast_math`. The `which_nvcc()` function checks PATH first, then falls back to common CUDA install locations (`/usr/local/cuda/bin/nvcc`, `/usr/local/cuda-13.2/bin/nvcc`, `/usr/local/cuda-13.0/bin/nvcc`, `/usr/bin/nvcc`). nvcc args include `-I` flags for `kernels/flashinfer-gdn` and `kernels/flashinfer-attn` include paths. Missing nvcc or source files produce warnings but do not fail the build. Compiled kernels are placed in `kernels/compiled/` and loaded at runtime by the KernelRegistry.

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

- Feature-gated CUDA crate (`infers-cuda`) that compiles with and without cudarc
- CudaRuntime for multi-GPU device context management (cudarc CudaContext)
- StreamPool for async CUDA stream management per device
- GpuAllocator block pool memory bookkeeper with allocate/free/reuse (5 unit tests)
- KernelRegistry for .cubin loading (5 kernels: gdn_prefill, gdn_decode, batch_prefill, batch_decode, sampling) and LoadedKernelRegistry for GPU-loaded kernels
- GemmEngine wrapping cuBLASLt with FP16/BF16/FP32/NVFP4 support; `matmul()` method accepts `GemmConfig` (placeholder implementation)
- NcclCommunicator for TP all-reduce and PP send/recv operations
- build.rs for nvcc kernel compilation (default sm_100a, configurable via INFERS_CUDA_ARCH env var)
- scripts/extract-kernels.sh for pulling FlashInfer kernels from vLLM
- Kernel directory structure (flashinfer-gdn, flashinfer-attn, compiled)
- Feature propagation: backends/native and parallelism crates forward `cuda` feature to infers-cuda

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

## SSE Constant Usage

Chat handler uses `SSE_DONE` constant from `infers_api` instead of hardcoded `"[DONE]"` string. See [[crates/server/src/handlers/chat.rs]].

## Kernel Registry Documentation

`LoadedKernelRegistry` has documentation explaining its use with the `cuda` feature for GPU kernel execution.

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
