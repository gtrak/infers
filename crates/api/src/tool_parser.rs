use crate::response::{FunctionCall, ToolCall};

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

/// Tracks partial tool call state during streaming.
pub struct PartialToolCall {
    pub buffer: String,
    pub index: i32,
}

impl PartialToolCall {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            index: 0,
        }
    }
}

impl Default for PartialToolCall {
    fn default() -> Self {
        Self::new()
    }
}

/// Parser for Qwen3.6 XML-format tool calls.
///
/// Parses `<tool_call>...</tool_call>` blocks from model output text,
/// supporting both streaming (incremental) and complete text modes.
pub struct ToolCallParser;

impl ToolCallParser {
    /// Parse a streaming delta, accumulating text in `partial`.
    ///
    /// Returns `Some(ToolCall)` when a complete `<tool_call>...</tool_call>` block
    /// is found. The buffer is cleared after a successful parse.
    pub fn parse_streaming_delta(
        &self,
        partial: &mut PartialToolCall,
        new_text: &str,
    ) -> Result<Option<ToolCall>, String> {
        partial.buffer.push_str(new_text);

        // Check if we have a complete tool call
        if let Some(start) = partial.buffer.find(TOOL_OPEN) {
            if let Some(end) = partial.buffer.find(TOOL_CLOSE) {
                if end > start {
                    let tool_json =
                        &partial.buffer[start + TOOL_OPEN.len()..end];
                    let trimmed = tool_json.trim();
                    let tool_call = self.parse_tool_call_json(trimmed)?;

                    // Clear the consumed portion of the buffer
                    partial.buffer.drain(..end + TOOL_CLOSE.len());
                    partial.index += 1;

                    return Ok(Some(tool_call));
                }
            }
        }

        Ok(None)
    }

    /// Parse all complete tool calls from a full text response.
    ///
    /// Returns all `<tool_call>...</tool_call>` blocks found in order.
    pub fn parse_complete(&self, text: &str) -> Result<Vec<ToolCall>, String> {
        let mut tool_calls = Vec::new();
        let mut search_start = 0;
        let mut id_counter = 0i32;

        while let Some(begin) = text[search_start..].find(TOOL_OPEN) {
            let abs_begin = search_start + begin;
            let after_open = abs_begin + TOOL_OPEN.len();
            if let Some(relative_end) = text[after_open..].find(TOOL_CLOSE) {
                let abs_end = after_open + relative_end;
                let tool_json = &text[after_open..abs_end];
                let trimmed = tool_json.trim();
                let mut tool_call = self.parse_tool_call_json(trimmed)?;
                // Assign auto-incrementing ID if not already present
                if tool_call.id.is_empty() {
                    tool_call.id = format!("call_{}", id_counter);
                }
                tool_calls.push(tool_call);
                id_counter += 1;
                search_start = abs_end + TOOL_CLOSE.len();
            } else {
                break;
            }
        }

        Ok(tool_calls)
    }

    /// Parse a JSON string into a ToolCall.
    ///
    /// Handles multiple input shapes:
    /// - Full `{"id": "...", "type": "function", "function": {...}}`
    /// - Function-only `{"name": "...", "arguments": "..."}`
    /// - Raw string (fallback — wraps with placeholder)
    fn parse_tool_call_json(&self, json_str: &str) -> Result<ToolCall, String> {
        // Try parsing as serde_json::Value first
        let v: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| format!("Invalid tool call JSON: {}", e))?;

        match v {
            // Full tool call format with id, type, function
            Value::Object(ref map) if map.contains_key("function") => {
                let id = map
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let tool_type = map
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("function")
                    .to_string();
                let func = &map["function"];
                let name = func
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let arguments = func
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}")
                    .to_string();

                Ok(ToolCall {
                    id,
                    tool_type,
                    function: FunctionCall { name, arguments },
                })
            }
            // Function-only format { "name": "...", "arguments": "..." }
            Value::Object(ref map) if map.contains_key("name") => {
                let name = map["name"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string();
                let arguments = map
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}")
                    .to_string();

                Ok(ToolCall {
                    id: String::new(),
                    tool_type: "function".to_string(),
                    function: FunctionCall { name, arguments },
                })
            }
            // Fallback: wrap the entire JSON as a string argument
            Value::Object(_) | Value::String(_) => {
                Ok(ToolCall {
                    id: String::new(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "unknown".to_string(),
                        arguments: json_str.to_string(),
                    },
                })
            }
            _ => Err(format!("Unexpected tool call JSON value: {}", v)),
        }
    }
}

use serde_json::Value;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_complete_single() {
        let parser = ToolCallParser;
        let text = r#"Some text <tool_call>{"name":"get_weather","arguments":"{\"location\":\"Paris\"}"}</tool_call> more text"#;
        let calls = parser.parse_complete(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Paris"));
    }

    #[test]
    fn test_parse_complete_multiple() {
        let parser = ToolCallParser;
        let text = concat!(
            r#"<tool_call>{"name":"get_weather","arguments":"{\"location\":\"Paris\"}"}</tool_call>"#,
            r#"<tool_call>{"name":"get_time","arguments":"{\"timezone\":\"UTC\"}"}</tool_call>"#,
        );
        let calls = parser.parse_complete(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_time");
    }

    #[test]
    fn test_parse_streaming_delta_partial() {
        let parser = ToolCallParser;
        let mut partial = PartialToolCall::new();

        // Incomplete — only opening tag
        let result = parser
            .parse_streaming_delta(&mut partial, "some text <tool_call>")
            .unwrap();
        assert!(result.is_none());
        assert!(!partial.buffer.is_empty());
    }

    #[test]
    fn test_parse_streaming_delta_complete() {
        let parser = ToolCallParser;
        let mut partial = PartialToolCall::new();

        // Stream part 1
        let r1 = parser
            .parse_streaming_delta(&mut partial, r#"<tool_call>{"name":"get_weath"#)
            .unwrap();
        assert!(r1.is_none());

        // Stream part 2 — completes the tool call
        let r2 = parser
            .parse_streaming_delta(&mut partial, r#"er","arguments":"{}"}</tool_call>"#)
            .unwrap();
        assert!(r2.is_some());
        let call = r2.unwrap();
        assert_eq!(call.function.name, "get_weather");
        assert_eq!(partial.index, 1);
        assert!(partial.buffer.is_empty());
    }

    #[test]
    fn test_parse_complete_empty() {
        let parser = ToolCallParser;
        let text = "Just a regular response without any tool calls.";
        let calls = parser.parse_complete(text).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_streaming_reset() {
        let parser = ToolCallParser;
        let mut partial = PartialToolCall::new();

        // First tool call
        let r1 = parser
            .parse_streaming_delta(&mut partial, r#"<tool_call>{"name":"fn1","arguments":"{}"}</tool_call>"#)
            .unwrap();
        assert!(r1.is_some());

        // Second tool call
        let r2 = parser
            .parse_streaming_delta(&mut partial, r#"<tool_call>{"name":"fn2","arguments":"{}"}</tool_call>"#)
            .unwrap();
        assert!(r2.is_some());
        assert_eq!(r2.unwrap().function.name, "fn2");
        assert_eq!(partial.index, 2);
    }
}
