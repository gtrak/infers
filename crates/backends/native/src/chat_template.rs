//! Chat template rendering using minijinja.
//!
//! Renders HuggingFace Jinja chat templates in pure Rust, replacing the
//! Python subprocess dependency. Extracts the template from the model via
//! a one-time Python call, then renders subsequent messages locally.

use std::path::Path;

use anyhow::{Context, Result};
use minijinja::{AutoEscape, Environment, Error as JinjaError, ErrorKind, State, Value};
use minijinja::value::ValueKind;

/// Extract the raw Jinja chat template from a HuggingFace model directory.
///
/// This is a one-time Python call — the returned template can then be rendered
/// entirely in Rust via [`ChatTemplate`].
pub fn extract_template(model_dir: &Path) -> Result<String> {
    let script = "from transformers import AutoTokenizer; \
        print(AutoTokenizer.from_pretrained(__import__('sys').argv[1], trust_remote_code=True).chat_template, end='')";

    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(model_dir.to_str().context("model dir must be valid UTF-8")?)
        .output()
        .context("Failed to run Python for template extraction")?;

    if !output.status.success() {
        anyhow::bail!(
            "Template extraction failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(String::from_utf8(output.stdout)?)
}

/// A chat template renderer backed by minijinja.
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    /// Create a new renderer from the raw Jinja template source.
    pub fn new(template_source: String) -> Result<Self> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_auto_escape_callback(|_| AutoEscape::None);

        // Register raise_exception — used by some templates for error handling
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<Value, JinjaError> {
                Err(JinjaError::new(ErrorKind::InvalidOperation, message))
            },
        );

        // Handle Python string methods that some templates (e.g. Qwen) call on str values
        env.set_unknown_method_callback(
            |state: &State, value: &Value, method: &str, args: &[Value]| {
                if value.kind() != ValueKind::String {
                    return Err(JinjaError::new(
                        ErrorKind::UnknownMethod,
                        format!("{} has no method named {}", value.kind(), method),
                    ));
                }

                match method {
                    "split" => {
                        let separator = args
                            .first()
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        state.apply_filter("split", &[value.clone(), Value::from(separator)])
                    }
                    "startswith" => {
                        let prefix = args.first().and_then(|v| v.as_str()).unwrap_or("");
                        let s = value.as_str().unwrap_or("");
                        Ok(Value::from(s.starts_with(prefix)))
                    }
                    "endswith" => {
                        let suffix = args.first().and_then(|v| v.as_str()).unwrap_or("");
                        let s = value.as_str().unwrap_or("");
                        Ok(Value::from(s.ends_with(suffix)))
                    }
                    "lstrip" => {
                        let chars = args.first().and_then(|v| v.as_str());
                        let s = value.as_str().unwrap_or("");
                        if let Some(chars) = chars {
                            Ok(Value::from(s.trim_start_matches(chars)))
                        } else {
                            Ok(Value::from(s.trim_start()))
                        }
                    }
                    "rstrip" => {
                        let chars = args.first().and_then(|v| v.as_str());
                        let s = value.as_str().unwrap_or("");
                        if let Some(chars) = chars {
                            Ok(Value::from(s.trim_end_matches(chars)))
                        } else {
                            Ok(Value::from(s.trim_end()))
                        }
                    }
                    _ => Err(JinjaError::new(
                        ErrorKind::UnknownMethod,
                        format!("string has no method named {}", method),
                    )),
                }
            },
        );

        env.add_template_owned("chat_template", template_source)
            .context("Failed to parse chat template")?;

        Ok(Self { env })
    }

    /// Render a single user message with add_generation_prompt=true.
    ///
    /// Returns the rendered text — the caller is responsible for tokenizing it.
    pub fn render_user(&self, prompt: &str) -> Result<String> {
        let context = serde_json::json!({
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": true
        });

        let tmpl = self.env.get_template("chat_template")
            .expect("chat_template should exist");
        tmpl.render(context)
            .context("Failed to render chat template")
    }
}
