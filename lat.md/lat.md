This directory defines the high-level concepts, business logic, and architecture of this project using markdown. It is managed by [lat.md](https://www.npmjs.com/package/lat.md) — a tool that anchors source code to these definitions. Install the `lat` command with `npm i -g lat.md` and run `lat --help`.

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

All metrics are registered in a single lazy_static Prometheus Registry. Seven metrics track inference workload and system resources.

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
