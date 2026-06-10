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
