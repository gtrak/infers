use axum::{
    extract::State,
    response::{IntoResponse, sse::{Event, Sse}},
    Json,
};
use futures::stream::{self, Stream, StreamExt};
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use infers_api::{
    ApiError, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    Choice, ChunkChoice, Delta, MessageContent, Usage, SSE_DONE, QwenChatTemplate,
};
use infers_scheduler::{
    SamplingConfig, SamplingStrategy,
};

use crate::state::SharedState;

pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ApiError> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Build QwenChatTemplate from chat_template_kwargs
    let template = build_template_from_kwargs(&req);

    // Format the prompt
    let formatted = template.apply(&req.messages, req.tools.as_deref());

    // Tokenize the prompt
    let prompt_tokens = state.tokenizer
        .encode(&formatted)
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    let prompt_token_count = prompt_tokens.len();

    // Determine max_tokens from request
    let max_tokens = req.max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(512) as usize;

    // Build SamplingConfig
    let sampling_config = SamplingConfig {
        strategy: SamplingStrategy::Greedy,
        max_tokens,
        stop_sequences: Vec::new(),
    };

    if req.stream {
        // Create mpsc channel for token responses
        let (tx, rx) = mpsc::channel::<u32>(256);

        // Enqueue request and register response channel
        let routing_id = {
            let mut orchestrator = state.orchestrator.lock().await;
            let routing_id = orchestrator.enqueue_request(prompt_tokens, sampling_config);
            orchestrator.register_response_channel(routing_id, tx);
            routing_id
        };

        let id = format!("chatcmpl-{}", routing_id);
        let model = state.model_name.clone();

        let stream = create_token_stream(
            rx, id, created, model, state.tokenizer.clone(),
        );

        let sse = Sse::new(stream).keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(std::time::Duration::from_secs(5)),
        );
        Ok(sse.into_response())
    } else {
        // Non-streaming: collect all tokens
        let (tx, rx) = mpsc::channel::<u32>(256);

        // Enqueue request and register response channel
        let routing_id = {
            let mut orchestrator = state.orchestrator.lock().await;
            let routing_id = orchestrator.enqueue_request(prompt_tokens, sampling_config);
            orchestrator.register_response_channel(routing_id, tx);
            routing_id
        };

        // Collect all generated tokens
        let completion_tokens = ReceiverStream::new(rx)
            .collect::<Vec<u32>>().await;

        // Decode the full token sequence
        let text = state.tokenizer
            .decode(&completion_tokens)
            .map_err(|e| ApiError::Internal(e.to_string()))?;

        let completion_token_count = completion_tokens.len();

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", routing_id),
            object: "chat.completion".to_string(),
            created: created as i64,
            model: state.model_name.clone(),
            choices: vec![Choice {
                index: 0,
                message: MessageContent {
                    role: "assistant".to_string(),
                    content: Some(text),
                    tool_calls: None,
                    reasoning_content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: prompt_token_count as i32,
                completion_tokens: completion_token_count as i32,
                total_tokens: (prompt_token_count + completion_token_count) as i32,
            }),
        };

        Ok(Json(response).into_response())
    }
}

/// Create an SSE stream from a token receiver channel.
///
/// Emits: role delta → token chunks → finish chunk → [DONE]
fn create_token_stream(
    rx: mpsc::Receiver<u32>,
    id: String,
    created: u64,
    model: String,
    tokenizer: infers_tokenizer::Tokenizer,
) -> impl Stream<Item = Result<Event, Infallible>> {
    // 1. Role delta chunk
    let id_c = id.clone();
    let model_c = model.clone();
    let role_chunk = stream::once(async move {
        Ok(Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: id_c,
                object: "chat.completion.chunk".to_string(),
                created: created as i64,
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
            }).unwrap_or_default()
        ))
    });

    // 2. Token stream from mpsc receiver
    let id_c = id.clone();
    let model_c = model.clone();
    let token_stream = ReceiverStream::new(rx)
        .map(move |token| {
            let text = tokenizer.decode(&[token]).unwrap_or_default();
            Ok(Event::default().data(
                serde_json::to_string(&ChatCompletionChunk {
                    id: id_c.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: created as i64,
                    model: model_c.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(text),
                            tool_calls: None,
                            reasoning_content: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                }).unwrap_or_default()
            ))
        });

    // 3. Finish chunk with usage
    let finish_chunk = stream::once({
        let id_c = id.clone();
        let model_c = model.clone();
        async move {
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
                            content: None,
                            tool_calls: None,
                            reasoning_content: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: None,
                }).unwrap_or_default()
            ))
        }
    });

    // 4. [DONE] event
    let done_stream = stream::once(async move {
        Ok(Event::default().data(SSE_DONE))
    });

    role_chunk
        .chain(token_stream)
        .chain(finish_chunk)
        .chain(done_stream)
}

/// Determine whether tools should be used based on request parameters.
///
/// Respects `enable_auto_tool_choice`: when false, tools are only activated
/// if `tool_choice` explicitly requires them (i.e., is not "none"). When true
/// (default), tools are automatically available for the model to call.
#[allow(dead_code)]
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
