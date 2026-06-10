//! OpenAI-compatible API types for inference server.
//!
//! Provides request, response, streaming, and error types for the
//! Chat Completions API.

pub mod request;
pub mod response;
pub mod streaming;
pub mod error;
pub mod template;
pub mod tool_parser;

pub use request::*;
pub use response::*;
pub use streaming::*;
pub use error::*;
pub use template::QwenChatTemplate;
pub use tool_parser::ToolCallParser;
