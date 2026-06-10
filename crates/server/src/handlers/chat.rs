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
    Choice, ChunkChoice, Delta, MessageContent, Usage, SSE_DONE,
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

    if req.stream {
        let stream = create_mock_stream(state.model_name.clone(), now);
        let sse = Sse::new(stream).keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(std::time::Duration::from_secs(5)),
        );
        Ok(sse.into_response())
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
