//! Model configuration deserialization from `config.json`.
//!
//! Parses the HuggingFace model config for Qwen3.6-27B architecture parameters,
//! including hybrid attention layer types (GDN vs full attention), MTP settings,
//! and quantization configuration.

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::formats::QuantizationConfig;

/// Type of attention mechanism used in a transformer layer.
// @lat: [[lat#Model Config and Format Detection#ModelConfig#Layer Type Pattern]]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// Gated DeltaNet: linear attention with recurrent state.
    GatedDeltaNet,
    /// Standard softmax attention with KV cache.
    FullAttention,
}

/// Parsed model configuration from `config.json`.
///
/// Contains architecture parameters for Qwen3.6-27B, including the hybrid
/// attention pattern (GDN + full attention), MTP head configuration,
/// and optional quantization settings.
// @lat: [[lat#Model Config and Format Detection#ModelConfig]]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture class names (e.g. `["Qwen3_5ForConditionalGeneration"]`).
    pub architectures: Vec<String>,

    /// Model type identifier (e.g. `"qwen3_5"`).
    pub model_type: String,

    /// Number of transformer layers.
    pub num_hidden_layers: usize,

    /// Hidden layer dimension.
    pub hidden_size: usize,

    /// FFN intermediate dimension.
    pub intermediate_size: usize,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Number of query attention heads.
    pub num_attention_heads: usize,

    /// Number of key/value heads (grouped attention).
    pub num_key_value_heads: usize,

    /// Per-head dimension.
    pub head_dim: usize,

    /// Maximum position embeddings.
    pub max_position_embeddings: usize,

    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,

    /// Hidden activation function (e.g. `"silu"`).
    pub hidden_act: String,

    /// Whether word embeddings and output embeddings are tied.
    pub tie_word_embeddings: bool,

    /// RoPE base frequency.
    pub rope_theta: f64,

    /// Fraction of head dimensions using rotary embedding.
    pub partial_rotary_factor: f32,

    /// Whether mRoPE sections are interleaved.
    pub mrope_interleaved: bool,

    /// mRoPE section sizes (e.g. `[11, 11, 10]`).
    pub mrope_section: Vec<usize>,

    /// Number of MTP hidden layers (defaults to 0).
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,

    /// Whether MTP uses dedicated embeddings (defaults to false).
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,

    /// Optional quantization configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_config: Option<QuantizationConfig>,

    /// Explicit layer type overrides. Falls back to default pattern if absent.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
}

/// Default interval for full-attention layers in the hybrid attention pattern.
/// Every Nth layer (0-indexed, 1-based modulo) is full attention.
const FULL_ATTENTION_INTERVAL: usize = 4;

/// Shallow-merge `text_config` into the root object.
///
/// If the root JSON contains a `text_config` object, its keys are copied into
/// the root level. Root-level keys are preserved (they take priority over
/// text_config keys). Returns the original value unchanged if `text_config`
/// is absent or not an object.
fn merge_text_config(value: serde_json::Value) -> serde_json::Value {
    let mut obj = match value {
        serde_json::Value::Object(map) => map,
        other => return other,
    };

    if let Some(serde_json::Value::Object(text_config)) = obj.remove("text_config") {
        for (k, v) in text_config {
            // Root takes priority — only insert if key not already present
            obj.entry(k).or_insert(v);
        }
    }

    serde_json::Value::Object(obj)
}

impl ModelConfig {
    /// Load model config from `config.json` in the given directory.
    ///
    /// Handles both flat configs and multimodal wrappers where architecture
    /// parameters are nested inside a `text_config` object. When `text_config`
    /// is present, its fields are merged into the root (shallow merge: root
    /// keys take priority over text_config keys).
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let config_path = model_dir.join("config.json");
        let contents = std::fs::read_to_string(&config_path)?;
        Self::from_str(&contents).map_err(|e| anyhow::anyhow!(e))
    }

    /// Deserialize a `ModelConfig` from a JSON string, handling optional
    /// `text_config` wrapper for multimodal model configs.
    pub fn from_str(contents: &str) -> serde_json::Result<Self> {
        let value: serde_json::Value = serde_json::from_str(contents)?;
        let merged = merge_text_config(value);
        serde_json::from_value(merged)
    }

    /// Return the layer type for a given layer index.
    ///
    /// If `layer_types` is set explicitly, uses that. Otherwise falls back
    /// to the default pattern: every Nth layer (0, 4, 8, …) is full attention,
    /// all others are GDN.
    pub fn get_layer_type(&self, layer_idx: usize) -> LayerType {
        if let Some(ref types) = self.layer_types {
            if layer_idx < types.len() {
                match types[layer_idx].as_str() {
                    "full_attention" => LayerType::FullAttention,
                    "linear_attention" => LayerType::GatedDeltaNet,
                    other => {
                        tracing::warn!(
                            layer = layer_idx,
                            type_name = other,
                            "Unknown layer type string; defaulting to GDN"
                        );
                        LayerType::GatedDeltaNet
                    }
                }
            } else {
                self.default_layer_type(layer_idx)
            }
        } else {
            self.default_layer_type(layer_idx)
        }
    }

    /// Default layer type pattern: every Nth layer (0-indexed) is full attention.
    fn default_layer_type(&self, layer_idx: usize) -> LayerType {
        if (layer_idx + 1).is_multiple_of(FULL_ATTENTION_INTERVAL) {
            LayerType::FullAttention
        } else {
            LayerType::GatedDeltaNet
        }
    }

    /// Number of full-attention layers in the model.
    pub fn num_full_attention_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.get_layer_type(i) == LayerType::FullAttention)
            .count()
    }

    /// Number of GDN layers in the model.
    pub fn num_gdn_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.get_layer_type(i) == LayerType::GatedDeltaNet)
            .count()
    }

    /// Whether MTP (Multi-Token Prediction) is enabled.
    pub fn has_mtp(&self) -> bool {
        self.mtp_num_hidden_layers > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_6_config_json() -> String {
        serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false
        })
        .to_string()
    }

    #[test]
    fn deserialize_qwen3_6_config() {
        let config: ModelConfig = serde_json::from_str(&qwen3_6_config_json()).unwrap();

        assert_eq!(config.architectures, vec!["Qwen3_5ForConditionalGeneration"]);
        assert_eq!(config.model_type, "qwen3_5");
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.intermediate_size, 17408);
        assert_eq!(config.vocab_size, 248320);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.max_position_embeddings, 262144);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.hidden_act, "silu");
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert_eq!(config.partial_rotary_factor, 0.25);
        assert!(config.mrope_interleaved);
        assert_eq!(config.mrope_section, vec![11, 11, 10]);
        assert_eq!(config.mtp_num_hidden_layers, 1);
        assert!(!config.mtp_use_dedicated_embeddings);
        assert!(config.quantization_config.is_none());
        assert!(config.layer_types.is_none());
    }

    #[test]
    fn deserialize_defaults_for_optional_fields() {
        // Minimal config with only required fields — defaults should kick in.
        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "num_hidden_layers": 32,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "vocab_size": 152064,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
        })
        .to_string();

        let config: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.mtp_num_hidden_layers, 0);
        assert!(!config.mtp_use_dedicated_embeddings);
        assert!(config.layer_types.is_none());
        assert!(!config.has_mtp());
    }

    #[test]
    fn get_layer_type_explicit_layer_types() {
        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "num_hidden_layers": 8,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ]
        })
        .to_string();

        let config: ModelConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.get_layer_type(0), LayerType::GatedDeltaNet);
        assert_eq!(config.get_layer_type(1), LayerType::GatedDeltaNet);
        assert_eq!(config.get_layer_type(2), LayerType::GatedDeltaNet);
        assert_eq!(config.get_layer_type(3), LayerType::FullAttention);
        assert_eq!(config.get_layer_type(6), LayerType::GatedDeltaNet);
        assert_eq!(config.get_layer_type(7), LayerType::FullAttention);
    }

    #[test]
    fn get_layer_type_default_pattern() {
        // No layer_types — uses (i+1) % 4 == 0 pattern
        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "num_hidden_layers": 16,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
        })
        .to_string();

        let config: ModelConfig = serde_json::from_str(&json).unwrap();

        // Layer 0: (0+1)%4 = 1 → GDN
        assert_eq!(config.get_layer_type(0), LayerType::GatedDeltaNet);
        // Layer 1: (1+1)%4 = 2 → GDN
        assert_eq!(config.get_layer_type(1), LayerType::GatedDeltaNet);
        // Layer 2: (2+1)%4 = 3 → GDN
        assert_eq!(config.get_layer_type(2), LayerType::GatedDeltaNet);
        // Layer 3: (3+1)%4 = 0 → FullAttention
        assert_eq!(config.get_layer_type(3), LayerType::FullAttention);
        // Layer 4: (4+1)%4 = 1 → GDN
        assert_eq!(config.get_layer_type(4), LayerType::GatedDeltaNet);
        // Layer 7: (7+1)%4 = 0 → FullAttention
        assert_eq!(config.get_layer_type(7), LayerType::FullAttention);
    }

    #[test]
    fn layer_count_methods_default_pattern() {
        // 64 layers, default pattern: every 4th is full attention
        // Full attention at indices 3, 7, 11, 15, ..., 63 → 16 layers
        // GDN: 64 - 16 = 48 layers
        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
        })
        .to_string();

        let config: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.num_full_attention_layers(), 16);
        assert_eq!(config.num_gdn_layers(), 48);
    }

    #[test]
    fn layer_count_methods_explicit_layer_types() {
        // 8 layers: 2 full attention, 6 GDN
        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "num_hidden_layers": 8,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ]
        })
        .to_string();

        let config: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.num_full_attention_layers(), 2);
        assert_eq!(config.num_gdn_layers(), 6);
    }

    #[test]
    fn deserialize_with_text_config() {
        // Multimodal wrapper: architecture params are inside text_config.
        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "vocab_size": 248320,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
                "tie_word_embeddings": false,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false
            }
        })
        .to_string();

        let config = ModelConfig::from_str(&json).unwrap();

        assert_eq!(config.architectures, vec!["Qwen3_5ForConditionalGeneration"]);
        assert_eq!(config.model_type, "qwen3_5");
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.intermediate_size, 17408);
        assert_eq!(config.vocab_size, 248320);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.max_position_embeddings, 262144);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.hidden_act, "silu");
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert_eq!(config.partial_rotary_factor, 0.25);
        assert!(config.mrope_interleaved);
        assert_eq!(config.mrope_section, vec![11, 11, 10]);
        assert_eq!(config.mtp_num_hidden_layers, 1);
        assert!(!config.mtp_use_dedicated_embeddings);
    }

    #[test]
    fn deserialize_text_config_with_root_overrides() {
        // Root-level fields must override text_config fields.
        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "num_hidden_layers": 32,
            "hidden_size": 2048,
            "text_config": {
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "vocab_size": 248320,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
                "tie_word_embeddings": false,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            }
        })
        .to_string();

        let config = ModelConfig::from_str(&json).unwrap();

        // Root values must win
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.hidden_size, 2048);

        // text_config values fill in where root is missing
        assert_eq!(config.intermediate_size, 17408);
        assert_eq!(config.vocab_size, 248320);
        assert_eq!(config.num_attention_heads, 24);
    }

    #[test]
    fn load_with_text_config_from_file() {
        use std::io::Write;

        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.json");

        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "vocab_size": 248320,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu",
                "tie_word_embeddings": false,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "mtp_num_hidden_layers": 1
            }
        })
        .to_string();

        let mut file = std::fs::File::create(&config_path).unwrap();
        file.write_all(json.as_bytes()).unwrap();

        let config = ModelConfig::load(tmp.path()).unwrap();
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.mtp_num_hidden_layers, 1);
        assert!(config.has_mtp());
    }

    #[test]
    fn has_mtp_enabled() {
        let config: ModelConfig = serde_json::from_str(&qwen3_6_config_json()).unwrap();
        assert!(config.has_mtp());
    }

    #[test]
    fn has_mtp_disabled() {
        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "num_hidden_layers": 32,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "vocab_size": 152064,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
        })
        .to_string();

        let config: ModelConfig = serde_json::from_str(&json).unwrap();
        assert!(!config.has_mtp());
    }

    #[test]
    fn unknown_layer_type_falls_back_to_gdn() {
        let json = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "num_hidden_layers": 4,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "layer_types": ["linear_attention", "unknown_type", "full_attention", "linear_attention"]
        })
        .to_string();

        let config: ModelConfig = serde_json::from_str(&json).unwrap();
        // unknown_type should fall back to GDN
        assert_eq!(config.get_layer_type(1), LayerType::GatedDeltaNet);
    }
}
