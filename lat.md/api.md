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
