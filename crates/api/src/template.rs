use crate::{Message, Tool};

/// Qwen3.6 chat template for formatting messages with thinking tokens and tool calls.
///
/// Uses `<|im_start|>` / `<|im_end|>` tokens as per Qwen3.6 format.
pub struct QwenChatTemplate {
    pub enable_thinking: bool,
    pub preserve_thinking: bool,
}

impl QwenChatTemplate {
    /// Create a new QwenChatTemplate.
    pub fn new(enable_thinking: bool, preserve_thinking: bool) -> Self {
        Self {
            enable_thinking,
            preserve_thinking,
        }
    }

    /// Apply the chat template to format messages into a prompt string.
    ///
    /// If tools are provided, a system message with `<tools>` XML block is prepended.
    /// Assistant messages may include `<thinking>` blocks and `<tool_call>` blocks.
    /// Tool responses are wrapped in `<tool_response>` blocks.
    /// The final assistant prompt is appended for generation.
    pub fn apply(&self, messages: &[Message], tools: Option<&[Tool]>) -> String {
        let mut formatted = String::new();

        // Add tools block as a system preamble
        if let Some(tools_list) = tools {
            if !tools_list.is_empty() {
                formatted.push_str("<|im_start|>system\n");
                formatted.push_str("You are a helpful assistant.\n");
                formatted.push_str(&self.format_tools(tools_list));
                formatted.push_str("<|im_end|>\n");
            }
        }

        for message in messages {
            match message.role.as_str() {
                "system" => {
                    formatted.push_str(&format!(
                        "<|im_start|>system\n{}<|im_end|>\n",
                        message.content_string()
                    ));
                }
                "user" => {
                    formatted.push_str(&format!(
                        "<|im_start|>user\n{}<|im_end|>\n",
                        message.content_string()
                    ));
                }
                "assistant" => {
                    formatted.push_str("<|im_start|>assistant\n");

                    // Add thinking/reasoning block
                    if let Some(thinking) = &message.reasoning_content {
                        if self.enable_thinking {
                            formatted.push_str(&format!("<thinking>{}</thinking>\n", thinking));
                        }
                    }

                    // Add tool calls
                    if let Some(tool_calls) = &message.tool_calls {
                        for tool_call in tool_calls {
                            formatted.push_str(&format!(
                                "<tool_call>\n{}\n</tool_call>\n",
                                serde_json::to_string(&tool_call.function.arguments)
                                    .unwrap_or_default()
                            ));
                        }
                    }

                    // Add text content
                    if let Some(content) = &message.content {
                        formatted.push_str(&content.to_string());
                    }

                    formatted.push_str("<|im_end|>\n");
                }
                "tool" => {
                    formatted.push_str(&format!(
                        "<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n",
                        message.content_string()
                    ));
                }
                _ => {}
            }
        }

        // Append assistant prompt for generation
        formatted.push_str("<|im_start|>assistant\n");

        formatted
    }

    /// Format tools as an XML block for the system prompt.
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

impl Message {
    /// Helper to get the content as a string, regardless of whether it's a JSON value or string.
    fn content_string(&self) -> String {
        match &self.content {
            Some(val) => {
                if let Some(s) = val.as_str() {
                    s.to_string()
                } else {
                    val.to_string()
                }
            }
            None => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::{FunctionCall, ToolCall};
    use serde_json::json;

    fn make_system_msg(content: &str) -> Message {
        Message {
            role: "system".to_string(),
            content: Some(json!(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn make_user_msg(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Some(json!(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn make_assistant_msg(content: Option<&str>, reasoning: Option<&str>, tool_calls: Option<Vec<ToolCall>>) -> Message {
        Message {
            role: "assistant".to_string(),
            content: content.map(|c| json!(c)),
            name: None,
            tool_calls,
            tool_call_id: None,
            reasoning_content: reasoning.map(|s| s.to_string()),
        }
    }

    fn make_tool_msg(content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: Some(json!(content)),
            name: None,
            tool_calls: None,
            tool_call_id: Some("call_123".to_string()),
            reasoning_content: None,
        }
    }

    fn make_weather_tool() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: crate::request::Function {
                name: "get_weather".to_string(),
                description: Some("Get weather for a location".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" },
                        "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
                    },
                    "required": ["location"]
                }),
            },
        }
    }

    #[test]
    fn test_system_and_user() {
        let template = QwenChatTemplate::new(false, false);
        let messages = vec![
            make_system_msg("You are a helpful assistant."),
            make_user_msg("Hello!"),
        ];
        let result = template.apply(&messages, None);
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_assistant_with_thinking() {
        let template = QwenChatTemplate::new(true, false);
        let messages = vec![
            make_user_msg("What is 2+2?"),
            make_assistant_msg(Some("4"), Some("This is simple math"), None),
        ];
        let result = template.apply(&messages, None);
        assert!(result.contains("<thinking>"));
        assert!(result.contains("This is simple math"));
        assert!(result.contains("4"));
    }

    #[test]
    fn test_thinking_disabled() {
        let template = QwenChatTemplate::new(false, false);
        let messages = vec![
            make_user_msg("Hi"),
            make_assistant_msg(Some("Hello"), Some("Should not appear"), None),
        ];
        let result = template.apply(&messages, None);
        assert!(!result.contains("<thinking>"), "thinking should be disabled");
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_tool_call_formatting() {
        let template = QwenChatTemplate::new(false, false);

        // First add tool definition
        let tools = vec![make_weather_tool()];

        let tool_call = ToolCall {
            id: "call_abc".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"Paris","unit":"celsius"}"#.to_string(),
            },
        };

        let messages = vec![
            make_user_msg("What's the weather in Paris?"),
            make_assistant_msg(None, None, Some(vec![tool_call])),
        ];
        let result = template.apply(&messages, Some(&tools));

        // Should have tools block
        assert!(result.contains("<tools>"));
        assert!(result.contains("get_weather"));
        assert!(result.contains("<tool_call>"));
        assert!(result.contains("Paris"));
    }

    #[test]
    fn test_tool_response() {
        let template = QwenChatTemplate::new(false, false);
        let messages = vec![
            make_user_msg("What's the weather?"),
            make_assistant_msg(None, None, Some(vec![ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "get_weather".to_string(),
                    arguments: r#"{"location":"Paris"}"#.to_string(),
                },
            }])),
            make_tool_msg(r#"{"temperature": 22, "unit": "celsius"}"#),
        ];
        let result = template.apply(&messages, None);
        assert!(result.contains("<tool_response>"));
        assert!(result.contains("temperature"));
    }
}
