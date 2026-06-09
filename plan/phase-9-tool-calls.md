# Phase 9: Tool Calls + Final Polish

**Duration:** 2 weeks  
**Goal:** Implement tool calling, SSE streaming with tools, end-to-end benchmarking, and stability testing.

## Deliverables

1. Qwen3.6 chat template integration (thinking tokens, tool calls)
2. Tool call streaming (delta format for `tool_calls`)
3. `enable-auto-tool-choice` API parameter
4. Tool call parser (`qwen3_xml` or `qwen3_coder`)
5. OpenAI-compatible tool call responses
6. End-to-end benchmark vs vLLM baseline
7. 24-hour stability test
8. Documentation
9. Performance optimization pass

## Technical Details

### Chat Template Integration

Qwen3.6 uses a Jinja2 template with special handling for thinking and tools:

```rust
pub struct QwenChatTemplate {
    pub tokenizer: Tokenizer,
    pub enable_thinking: bool,
    pub preserve_thinking: bool,
}

impl QwenChatTemplate {
    pub fn apply(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
    ) -> Result<String> {
        let mut formatted = String::new();
        
        // Add tools if present
        if let Some(tools) = tools {
            formatted.push_str("<|im_start|>system\n");
            formatted.push_str("You are a helpful assistant.\n");
            formatted.push_str(&self.format_tools(tools));
            formatted.push_str("<|im_end|>\n");
        }
        
        for message in messages {
            match message.role.as_str() {
                "system" => {
                    formatted.push_str(&format!(
                        "<|im_start|>system\n{}<|im_end|>\n",
                        message.content
                    ));
                }
                "user" => {
                    formatted.push_str(&format!(
                        "<|im_start|>user\n{}<|im_end|>\n",
                        message.content
                    ));
                }
                "assistant" => {
                    formatted.push_str("<|im_start|>assistant\n");
                    
                    if let Some(thinking) = &message.reasoning_content {
                        formatted.push_str(&format!("<thinking>{}</thinking>\n", thinking));
                    }
                    
                    if let Some(tool_calls) = &message.tool_calls {
                        for tool_call in tool_calls {
                            formatted.push_str(&format!(
                                "<tool_call>\n{}\n</tool_call>\n",
                                serde_json::to_string(&tool_call.function.arguments)?
                            ));
                        }
                    }
                    
                    if let Some(content) = &message.content {
                        formatted.push_str(content);
                    }
                    
                    formatted.push_str("<|im_end|>\n");
                }
                "tool" => {
                    formatted.push_str(&format!(
                        "<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n",
                        message.content
                    ));
                }
                _ => {}
            }
        }
        
        // Add assistant prompt
        formatted.push_str("<|im_start|>assistant\n");
        
        Ok(formatted)
    }
    
    fn format_tools(&self, tools: &[Tool]) -> String {
        let mut result = String::from("<tools>\n");
        for tool in tools {
            result.push_str(&format!(
                "<tool>\n<name>{}</name>\n<parameters>{}</parameters>\n</tool>\n",
                tool.function.name,
                serde_json::to_string(&tool.function.parameters).unwrap_or_default()
            ));
        }
        result.push_str("</tools>\n");
        result
    }
}
```

### Tool Call Parser

```rust
pub struct ToolCallParser;

impl ToolCallParser {
    pub fn parse_streaming_delta(
        &self,
        accumulated: &mut PartialToolCall,
        new_text: &str,
    ) -> Result<Option<ToolCall>> {
        accumulated.buffer.push_str(new_text);
        
        // Check if we have a complete tool call
        if let Some(start) = accumulated.buffer.find("<tool_call>") {
            if let Some(end) = accumulated.buffer.find("</tool_call>") {
                let tool_json = &accumulated.buffer[start + "<tool_call>".len()..end];
                
                let tool_call: ToolCall = serde_json::from_str(tool_json)?;
                
                // Reset buffer
                accumulated.buffer.clear();
                
                return Ok(Some(tool_call));
            }
        }
        
        Ok(None)
    }
    
    pub fn parse_complete(
        &self,
        text: &str,
    ) -> Result<Vec<ToolCall>> {
        let mut tool_calls = Vec::new();
        
        // Find all <tool_call>...</tool_call> blocks
        let mut start = 0;
        while let Some(begin) = text[start..].find("<tool_call>") {
            let begin = start + begin;
            if let Some(end) = text[begin..].find("</tool_call>") {
                let end = begin + end + "</tool_call>".len();
                let tool_json = &text[begin + "<tool_call>".len()..end - "</tool_call>".len()];
                
                let tool_call: ToolCall = serde_json::from_str(tool_json)?;
                tool_calls.push(tool_call);
                
                start = end;
            } else {
                break;
            }
        }
        
        Ok(tool_calls)
    }
}

pub struct PartialToolCall {
    pub buffer: String,
    pub index: usize,
}
```

### Tool Call Streaming

```rust
async fn stream_tool_call_response(
    engine: &BackendRouter,
    req: ChatCompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut parser = ToolCallParser;
    let mut partial = PartialToolCall {
        buffer: String::new(),
        index: 0,
    };
    
    let stream = async_stream::stream! {
        let mut tokens = engine.generate_streaming(&req).await?;
        
        while let Some(token) = tokens.next().await {
            // Try to parse tool call from accumulated text
            if let Some(tool_call) = parser.parse_streaming_delta(&mut partial, &token)? {
                yield Event::default().data(json!({
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": partial.index,
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }]
                        },
                        "finish_reason": null
                    }]
                }).to_string());
                
                partial.index += 1;
            } else {
                // Regular content
                yield Event::default().data(json!({
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": token
                        },
                        "finish_reason": null
                    }]
                }).to_string());
            }
        }
        
        // Final chunk
        yield Event::default().data(json!({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls"
            }]
        }).to_string());
        
        yield Event::default().data("[DONE]".to_string());
    };
    
    Sse::new(stream)
}
```

### End-to-End Benchmark

```rust
pub struct Benchmark {
    pub engine: BackendRouter,
    pub vllm_baseline: Option<f32>,  // tok/s from vLLM
}

impl Benchmark {
    pub async fn run(&self) -> Result<BenchmarkResults> {
        let mut results = BenchmarkResults::new();
        
        // Test 1: Single request latency
        let prompt = "Explain quantum computing in one paragraph";
        let start = Instant::now();
        let _ = self.engine.chat_completions(ChatCompletionRequest {
            messages: vec![Message::user(prompt.to_string())],
            max_tokens: Some(256),
            ..Default::default()
        }).await?;
        let latency = start.elapsed();
        results.single_latency = latency;
        
        // Test 2: Throughput (concurrent requests)
        let prompts = vec![prompt; 10];
        let start = Instant::now();
        let futures: Vec<_> = prompts.into_iter().map(|p| {
            self.engine.chat_completions(ChatCompletionRequest {
                messages: vec![Message::user(p.to_string())],
                max_tokens: Some(256),
                ..Default::default()
            })
        }).collect();
        let _ = futures::future::join_all(futures).await;
        let throughput_time = start.elapsed();
        results.throughput_10 = throughput_time;
        
        // Test 3: Long context (128K)
        let long_prompt = " ".repeat(128000);  // dummy
        let start = Instant::now();
        let _ = self.engine.chat_completions(ChatCompletionRequest {
            messages: vec![Message::user(long_prompt)],
            max_tokens: Some(10),
            ..Default::default()
        }).await?;
        results.long_context_latency = start.elapsed();
        
        // Test 4: Tool calling
        let tool_prompt = "What's the weather in Paris?";
        let start = Instant::now();
        let _ = self.engine.chat_completions(ChatCompletionRequest {
            messages: vec![Message::user(tool_prompt.to_string())],
            tools: Some(vec![weather_tool()]),
            tool_choice: Some(ToolChoice::String("auto".to_string())),
            ..Default::default()
        }).await?;
        results.tool_call_latency = start.elapsed();
        
        // Test 5: MTP speedup
        let start = Instant::now();
        let _ = self.engine.chat_completions(ChatCompletionRequest {
            messages: vec![Message::user(prompt.to_string())],
            max_tokens: Some(512),
            speculative_config: Some(SpeculativeConfig {
                method: "mtp".to_string(),
                num_speculative_tokens: 2,
            }),
            ..Default::default()
        }).await?;
        let mtp_time = start.elapsed();
        results.mtp_speedup = results.single_latency.as_secs_f32() / mtp_time.as_secs_f32();
        
        Ok(results)
    }
}
```

### Stability Test

```rust
#[tokio::test]
async fn test_24_hour_stability() {
    let engine = BackendRouter::load("/models/Qwen3.6-27B-PrismaSCOUT").unwrap();
    
    let prompts = vec![
        "Hello, world!",
        "What is the capital of France?",
        "Explain quantum computing",
        "Write a Python function",
        "Tell me a joke",
    ];
    
    let start = Instant::now();
    let mut iteration = 0;
    
    while start.elapsed() < Duration::from_secs(24 * 3600) {
        let prompt = &prompts[iteration % prompts.len()];
        
        let result = engine.chat_completions(ChatCompletionRequest {
            messages: vec![Message::user(prompt.to_string())],
            max_tokens: Some(100),
            ..Default::default()
        }).await;
        
        assert!(result.is_ok(), "Failed at iteration {}", iteration);
        
        // Memory check every 100 iterations
        if iteration % 100 == 0 {
            let mem_usage = get_gpu_memory_usage();
            assert!(mem_usage < 0.95, "Memory leak detected: {}%", mem_usage * 100.0);
        }
        
        iteration += 1;
    }
    
    println!("Completed {} iterations over 24 hours", iteration);
}
```

## File Structure

```
crates/api/
  src/
    template.rs         # QwenChatTemplate
    tool_parser.rs      # ToolCallParser
    
crates/server/
  src/
    handlers/
      chat.rs           # Updated with tool call streaming
      
docs/
  README.md
  ARCHITECTURE.md
  API.md
  DEPLOYMENT.md
  
benchmarks/
  throughput.rs
  latency.rs
  memory.rs
```

## Performance Targets

| Metric | Target | vLLM Baseline |
|---|---|---|
| Single request decode | >20 tok/s | ~15 tok/s (PrismaSCOUT) |
| Throughput (10 concurrent) | >100 tok/s total | ~80 tok/s |
| First token latency (1K prompt) | <500ms | ~400ms |
| First token latency (128K prompt) | <5s | ~4s |
| MTP speedup | >1.5x | ~1.8x |
| Memory overhead | <10% vs vLLM | baseline |
| 24-hour stability | 0 OOM, 0 crashes | baseline |

## Dependencies

### Phase 9 → All previous phases

Integrates everything.

## Success Criteria

1. Tool calls work end-to-end (request → parse → response)
2. SSE streaming handles tool call deltas correctly
3. Benchmarks within 90% of vLLM performance
4. 24-hour stability test passes
5. Documentation is complete
6. All tests pass

## Cross-References

- **Research:** See `../research/api.md` for OpenAI tool call schema
- **Phase 1:** API types and server scaffold
- **Phase 7:** MTP speedup measurement
- **Phase 8:** All quantization formats tested

## Final Checklist

Before shipping:

- [ ] All 9 phases complete
- [ ] Unit tests for all crates
- [ ] Integration tests for end-to-end flow
- [ ] Benchmark results documented
- [ ] Memory leak check passed
- [ ] Documentation complete
- [ ] Docker deployment ready
- [ ] Monitoring (Prometheus) working
- [ ] Error handling robust
- [ ] Graceful shutdown

## Open Questions

1. Should we add WebSocket support alongside SSE?
2. Should we implement request queuing with backpressure?
3. Should we add OpenTelemetry tracing?
