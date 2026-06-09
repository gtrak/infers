# OpenAI API Compatibility Research

## Required Endpoints

### POST /v1/chat/completions

**Request body fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | Yes | Model ID (e.g., "Qwen3.6-27B") |
| `messages` | array | Yes | Conversation history |
| `stream` | boolean | No | Enable SSE streaming (default: false) |
| `temperature` | float | No | Sampling temperature (default: 1.0) |
| `top_p` | float | No | Nucleus sampling (default: 1.0) |
| `top_k` | integer | No | Top-k sampling (Qwen extension) |
| `max_tokens` | integer | No | Max completion tokens |
| `max_completion_tokens` | integer | No | Newer field, same purpose |
| `stop` | string/array | No | Stop sequences |
| `tools` | array | No | Available tools/functions |
| `tool_choice` | string/object | No | "none", "auto", "required", or specific function |
| `parallel_tool_calls` | boolean | No | Allow multiple tool calls |
| `presence_penalty` | float | No | Penalize repeated tokens |
| `frequency_penalty` | float | No | Penalize frequent tokens |
| `repetition_penalty` | float | No | Qwen-specific (default: 1.0) |
| `seed` | integer | No | Random seed for reproducibility |
| `response_format` | object | No | `{ "type": "text" }` or `{ "type": "json_object" }` |
| `stream_options` | object | No | `{ "include_usage": true }` |
| `chat_template_kwargs` | object | No | Qwen-specific: `{ "enable_thinking": false }` |
| `speculative_config` | object | No | MTP config: `{ "method": "mtp", "num_speculative_tokens": 2 }` |

**Non-streaming response:**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1690174221,
  "model": "Qwen3.6-27B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?",
        "tool_calls": [...]
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

**Streaming response (SSE):**

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1690174221,"model":"Qwen3.6-27B","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1690174221,"model":"Qwen3.6-27B","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1690174221,"model":"Qwen3.6-27B","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### GET /v1/models

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.6-27B",
      "object": "model",
      "created": 1686935002,
      "owned_by": "infers"
    }
  ]
}
```

## Tool Calls

### Request Format

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": { "type": "string" },
            "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### Tool Call Streaming

Tool calls arrive incrementally via `delta.tool_calls`:

```
chunk 1: delta.tool_calls = [{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather"}}]
chunk 2: delta.tool_calls = [{"index":0,"function":{"arguments":"{\""}}]
chunk 3: delta.tool_calls = [{"index":0,"function":{"arguments":"location"}}]
...
chunk N: finish_reason = "tool_calls", delta = {}
```

**Key rules:**
- `index`: position in tool_calls array (required)
- `id`: appears on first chunk for that index
- `function.name`: may appear in early chunks
- `function.arguments`: partial JSON string, accumulates across chunks
- Parse complete JSON only when `finish_reason: "tool_calls"`

### Response Format (Non-Streaming)

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"San Francisco\",\"unit\":\"celsius\"}"
          }
        }
      ]
    },
    "finish_reason": "tool_calls"
  }]
}
```

## Qwen3.6-Specific Parameters

### Thinking Mode

**Default:** Thinking enabled  
**Disable:** `chat_template_kwargs: { "enable_thinking": false }`

**Streaming thinking tokens:**
- Thinking content wrapped in `<thinking>...</thinking>`
- Must be streamed as regular content (no special field)
- Client responsibility to parse and hide/show

### Preserve Thinking

```json
{
  "chat_template_kwargs": {
    "preserve_thinking": true
  }
}
```

- Retains thinking blocks from historical messages
- Beneficial for agent scenarios
- Improves KV cache utilization

### Speculative Config (MTP)

```json
{
  "speculative_config": {
    "method": "mtp",
    "num_speculative_tokens": 2
  }
}
```

- `method`: "mtp" (Qwen3.6 native) or "eagle" or "medusa"
- `num_speculative_tokens`: 1-4 (2 recommended)
- Only valid for models with native MTP (Qwen3.6)

## Chat Template

Qwen3.6 uses Jinja2 chat template with these special tokens:

| Token | ID | Purpose |
|---|---|---|
| `<|im_start|>` | 151644 | Message start |
| `<|im_end|>` | 151645 | Message end |
| `<|endoftext|>` | 151643 | Pad/BOS |
| `<|tool_call|>` | 151657 | Tool call start |
| `</tool_call>` | 151658 | Tool call end |
| `<thinking>` | implicit | Thinking block (via chat template) |

**Template format:**
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<thinking>
{reasoning}
</thinking>
{response}<|im_end|>
```

## SSE Implementation

```rust
use axum::response::sse::{Event, Sse};
use futures::stream::Stream;

async fn chat_completions_stream(
    req: ChatCompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        // Send role delta
        yield Event::default().data(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "choices": [{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]
        }).to_string());
        
        // Stream tokens
        for token in token_generator {
            yield Event::default().data(json!({
                "choices": [{"index":0,"delta":{"content":token},"finish_reason":null}]
            }).to_string());
        }
        
        // Send finish
        yield Event::default().data(json!({
            "choices": [{"index":0,"delta":{},"finish_reason":"stop"}]
        }).to_string());
        
        // Send [DONE]
        yield Event::default().data("[DONE]".to_string());
    };
    
    Sse::new(stream)
}
```

## References

1. OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat
2. OpenAI SSE Streaming: https://platform.openai.com/docs/api-reference/streaming
3. vLLM OpenAI-Compatible Server: https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/
4. Qwen3.6 Chat Template: `tokenizer_config.json` in model repo

## Cross-References

- See `architecture.md` for Qwen3.6 chat template and thinking mode details
- See Phase 1 (Bootstrap) for API types and server scaffold
- See Phase 9 (Tool Calls) for tool call streaming implementation
