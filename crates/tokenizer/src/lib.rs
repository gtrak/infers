//! Tokenizer for model input/output.
//!
//! Provides a `Tokenizer` wrapper around the [`tokenizers`] crate for encoding
//! text into token ID sequences and decoding token IDs back to text.

use anyhow::{Context, Result};
use tokenizers::Tokenizer as T;

/// A tokenizer that converts text into token IDs and vice-versa.
///
/// Wraps the [`tokenizers::Tokenizer`](T) type and provides convenient
/// constructors for loading from a local file or a HuggingFace model ID.
#[derive(Clone)]
pub struct Tokenizer {
    inner: T,
}

impl Tokenizer {
    /// Load a tokenizer from a local `tokenizer.json` file.
    ///
    /// # Arguments
    ///
    /// * `path` — Path to a `tokenizer.json` file on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not a valid tokenizer.
    pub fn from_file(path: &str) -> Result<Self> {
        let inner = T::from_file(path)
            .map_err(|e| anyhow::Error::msg(e.to_string()))
            .with_context(|| format!("failed to load tokenizer from {path}"))?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from a HuggingFace Hub model ID.
    ///
    /// # Arguments
    ///
    /// * `model_id` — A HuggingFace model identifier (e.g. `"meta-llama/Llama-3.2-1B"`).
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be found or downloaded.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        let inner = T::from_pretrained(model_id, None::<tokenizers::FromPretrainedParameters>)
            .map_err(|e| anyhow::Error::msg(e.to_string()))
            .with_context(|| format!("failed to load pretrained tokenizer {model_id}"))?;
        Ok(Self { inner })
    }

    /// Encode a string into a sequence of token IDs.
    ///
    /// # Arguments
    ///
    /// * `text` — The text to tokenize.
    ///
    /// # Errors
    ///
    /// Returns an error if encoding fails.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::Error::msg(e.to_string()))
            .with_context(|| format!("failed to encode text (len={})", text.len()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a sequence of token IDs back into a string.
    ///
    /// # Arguments
    ///
    /// * `tokens` — A slice of token IDs to decode.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let result = self
            .inner
            .decode(tokens, false)
            .map_err(|e| anyhow::Error::msg(e.to_string()))
            .context("failed to decode token IDs")?;
        Ok(result)
    }

    /// Return the size of the tokenizer's vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
