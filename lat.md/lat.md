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
