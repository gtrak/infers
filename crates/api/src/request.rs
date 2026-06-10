use serde::{Deserialize, Serialize};

use super::response::ToolCall;

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_presence_penalty() -> f32 {
    0.0
}

fn default_frequency_penalty() -> f32 {
    0.0
}

/// Request body for the chat completions endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: bool,
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f32,
    #[serde(default = "default_frequency_penalty")]
    pub frequency_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative_config: Option<SpeculativeConfig>,
}

/// Conversation message in a chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Stop token configuration for chat completions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopConfig {
    String(String),
    Array(Vec<String>),
}

/// Tool definition for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Function,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// Tool choice strategy for chat completions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String),
    Object {
        #[serde(rename = "type")]
        tool_type: String,
        function: FunctionChoice,
    },
}

/// Explicit function selection for tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionChoice {
    pub name: String,
}

/// Output format specification for chat completions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

/// Streaming options for chat completions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// Configuration for speculative decoding (MTP).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    pub method: String,
    pub num_speculative_tokens: usize,
}