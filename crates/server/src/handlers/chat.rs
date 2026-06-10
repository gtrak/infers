use axum::{
    extract::State,
    response::{IntoResponse, sse::{Event, Sse}},
    Json,
};
use futures::stream::{self, Stream};
use futures::StreamExt;
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use infers_api::{
    ApiError, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    Choice, ChunkChoice, Delta, FunctionCall, FunctionCallDelta, MessageContent,
    QwenChatTemplate, ToolCall, ToolCallDelta, Usage, SSE_DONE,
};
use crate::state::SharedState;

pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ApiError> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Build QwenChatTemplate from chat_template_kwargs
    let template = build_template_from_kwargs(&req);

    // Apply template to format the prompt (useful for engine integration / logging)
    let _formatted_prompt = template.apply(
        &req.messages,
        req.tools.as_deref(),
    );

    // Determine whether tools should be used, respecting enable_auto_tool_choice
    let should_use_tools = should_use_tools(&req);

    // Extract tool name if tools will be used
    let tool_name = if should_use_tools {
        req.tools
            .as_ref()
            .and_then(|t| t.first())
            .map(|t| t.function.name.clone())
    } else {
        None
    };

    if req.stream {
        if let Some(name) = tool_name {
            let stream = create_mock_tool_call_stream(state.model_name.clone(), now, name);
            let sse = Sse::new(stream).keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(std::time::Duration::from_secs(5)),
            );
            Ok(sse.into_response())
        } else {
            let stream = create_mock_stream(state.model_name.clone(), now);
            let sse = Sse::new(stream).keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(std::time::Duration::from_secs(5)),
            );
            Ok(sse.into_response())
        }
    } else {
        if let Some(ref name) = tool_name {
            let response = create_mock_tool_call_response(&state.model_name, now, name);
            Ok(Json(response).into_response())
        } else {
            let response = ChatCompletionResponse {
                id: format!("chatcmpl-{}", generate_id()),
                object: "chat.completion".to_string(),
                created: now as i64,
                model: state.model_name.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: MessageContent {
                        role: "assistant".to_string(),
                        content: Some("Hello! I am a mock response from infers.".to_string()),
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 10,
                    total_tokens: 20,
                }),
            };
            Ok(Json(response).into_response())
        }
    }
}

/// Determine whether tools should be used based on request parameters.
///
/// Respects `enable_auto_tool_choice`: when false, tools are only activated
/// if `tool_choice` explicitly requires them (i.e., is not "none"). When true
/// (default), tools are automatically available for the model to call.
fn should_use_tools(req: &ChatCompletionRequest) -> bool {
    let has_tools = req.tools.as_ref().map_or(false, |t| !t.is_empty());
    if !has_tools {
        return false;
    }

    // If tool_choice is explicitly set to a string other than "none", use tools
    if let Some(tool_choice) = &req.tool_choice {
        match tool_choice {
            infers_api::ToolChoice::String(s) if s == "none" => return false,
            _ => return true,
        }
    }

    // Otherwise, use tools only if enable_auto_tool_choice is true
    req.enable_auto_tool_choice
}

/// Build a QwenChatTemplate from the request's chat_template_kwargs.
///
/// Reads `enable_thinking` (default: true) and `preserve_thinking` (default: false)
/// from the chat_template_kwargs JSON object. Falls back to defaults if kwargs
/// are absent or malformed.
fn build_template_from_kwargs(req: &ChatCompletionRequest) -> QwenChatTemplate {
    let default_enable_thinking = true;
    let default_preserve_thinking = false;

    let (enable_thinking, preserve_thinking) = match &req.chat_template_kwargs {
        Some(kwargs) if kwargs.is_object() => {
            let enable = kwargs
                .get("enable_thinking")
                .and_then(|v| v.as_bool())
                .unwrap_or(default_enable_thinking);
            let preserve = kwargs
                .get("preserve_thinking")
                .and_then(|v| v.as_bool())
                .unwrap_or(default_preserve_thinking);
            (enable, preserve)
        }
        _ => (default_enable_thinking, default_preserve_thinking),
    };

    QwenChatTemplate::new(enable_thinking, preserve_thinking)
}

/// Create a mock non-streaming response with tool calls.
fn create_mock_tool_call_response(
    model: &str,
    created: u64,
    tool_name: &str,
) -> ChatCompletionResponse {
    let tool_call = ToolCall {
        id: format!("call_{}", generate_id()),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: tool_name.to_string(),
            arguments: r#"{"location":"Paris","unit":"celsius"}"#.to_string(),
        },
    };

    ChatCompletionResponse {
        id: format!("chatcmpl-{}", generate_id()),
        object: "chat.completion".to_string(),
        created: created as i64,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: MessageContent {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![tool_call]),
                reasoning_content: None,
            },
            finish_reason: Some("tool_calls".to_string()),
        }],
        usage: Some(Usage {
            prompt_tokens: 15,
            completion_tokens: 20,
            total_tokens: 35,
        }),
    }
}

/// Create a mock SSE stream with tool call deltas.
fn create_mock_tool_call_stream(
    model: String,
    created: u64,
    tool_name: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let id = format!("chatcmpl-{}", generate_id());
    let call_id = format!("call_{}", generate_id());

    // Simulated incremental arguments to stream
    let arg_chunks: Vec<&str> = vec![
        "{\"",
        "location",
        "\":\"",
        "Paris",
        "\",\"",
        "unit",
        "\":\"",
        "celsius",
        "\"}",
    ];

    let model_c = model.clone();
    let created_c = created;

    // 1. Role delta: set role to "assistant"
    let role_chunk = {
        let id_c = id.clone();
        let model_c = model_c.clone();
        stream::once(async move {
            Ok(Event::default().data(
                serde_json::to_string(&ChatCompletionChunk {
                    id: id_c,
                    object: "chat.completion.chunk".to_string(),
                    created: created_c as i64,
                    model: model_c,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: Some("assistant".to_string()),
                            content: None,
                            tool_calls: None,
                            reasoning_content: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                })
                .unwrap_or_default(),
            ))
        })
    };

    // 2. Tool call name chunk: emit id + type + function name
    let name_chunk = {
        let id_c = id.clone();
        let model_c = model_c.clone();
        let tool_name_c = tool_name.clone();
        let call_id_c = call_id.clone();
        stream::once(async move {
            Ok(Event::default().data(
                serde_json::to_string(&ChatCompletionChunk {
                    id: id_c,
                    object: "chat.completion.chunk".to_string(),
                    created: created_c as i64,
                    model: model_c,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                            tool_calls: Some(vec![ToolCallDelta {
                                index: 0,
                                id: Some(call_id_c),
                                tool_type: Some("function".to_string()),
                                function: Some(FunctionCallDelta {
                                    name: Some(tool_name_c),
                                    arguments: None,
                                }),
                            }]),
                            reasoning_content: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                })
                .unwrap_or_default(),
            ))
        })
    };

    // Pre-clone for move closures
    let id_for_args = id.clone();
    let model_for_args = model_c.clone();

    // 3. Argument chunks: stream incremental arguments
    let arg_stream = stream::iter(arg_chunks.into_iter().map(move |arg| {
        let id_c = id_for_args.clone();
        let model_c = model_for_args.clone();
        Ok(Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: id_c,
                object: "chat.completion.chunk".to_string(),
                created: created_c as i64,
                model: model_c,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                        tool_calls: Some(vec![ToolCallDelta {
                            index: 0,
                            id: None,
                            tool_type: None,
                            function: Some(FunctionCallDelta {
                                name: None,
                                arguments: Some(arg.to_string()),
                            }),
                        }]),
                        reasoning_content: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            })
            .unwrap_or_default(),
        ))
    }));

    // 4. Finish chunk: empty delta with finish_reason "tool_calls"
    let finish_chunk = {
        let id_c = id.clone();
        let model_c = model_c.clone();
        stream::once(async move {
            Ok(Event::default().data(
                serde_json::to_string(&ChatCompletionChunk {
                    id: id_c,
                    object: "chat.completion.chunk".to_string(),
                    created: created_c as i64,
                    model: model_c,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                            tool_calls: None,
                            reasoning_content: None,
                        },
                        finish_reason: Some("tool_calls".to_string()),
                    }],
                    usage: None,
                })
                .unwrap_or_default(),
            ))
        })
    };

    // 5. [DONE] event
    let end_stream = stream::once(async move {
        Ok(Event::default().data(SSE_DONE))
    });

    role_chunk
        .chain(name_chunk)
        .chain(arg_stream)
        .chain(finish_chunk)
        .chain(end_stream)
}

/// Create a mock regular text SSE stream.
fn create_mock_stream(
    model: String,
    created: u64,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let id = format!("chatcmpl-{}", generate_id());
    let tokens = vec!["Hello", " from", " infers", "!"];

    // 1. Role delta chunk
    let id1 = id.clone();
    let model1 = model.clone();
    let intro_stream = stream::once(async move {
        Ok(Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: id1,
                object: "chat.completion.chunk".to_string(),
                created: created as i64,
                model: model1,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: Some("assistant".to_string()),
                        content: None,
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            })
            .unwrap_or_default(),
        ))
    });

    // 2. Token chunks
    let id2 = id.clone();
    let model2 = model.clone();
    let token_stream = stream::iter(
        tokens.into_iter().map(move |token| {
            let id_c = id2.clone();
            let model_c = model2.clone();
            Ok(Event::default().data(
                serde_json::to_string(&ChatCompletionChunk {
                    id: id_c,
                    object: "chat.completion.chunk".to_string(),
                    created: created as i64,
                    model: model_c,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(token.to_string()),
                            tool_calls: None,
                            reasoning_content: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                })
                .unwrap_or_default(),
            ))
        }),
    );

    // 3. Finish reason chunk
    let id3 = id.clone();
    let model3 = model.clone();
    let done_stream = stream::once(async move {
        Ok(Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: id3,
                object: "chat.completion.chunk".to_string(),
                created: created as i64,
                model: model3,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: None,
            })
            .unwrap_or_default(),
        ))
    });

    // 4. [DONE] event
    let end_stream = stream::once(async move {
        Ok(Event::default().data(SSE_DONE))
    });

    intro_stream
        .chain(token_stream)
        .chain(done_stream)
        .chain(end_stream)
}

fn generate_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed).to_string()
}
