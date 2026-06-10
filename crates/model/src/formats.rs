//! Quantization format detection and configuration.
//!
//! Supports PrismaSCOUT (NVFP4), AutoRound (INT4), GGUF, and BF16 formats
//! with automatic detection from model directory contents.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Detected quantization format for a model.
// @lat: [[lat#Model Config and Format Detection#Quantization Format Detection]]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationFormat {
    /// No quantization — full BF16 weights.
    Bf16,
    /// PrismaSCOUT: mixed NVFP4/BF16 via compressed-tensors.
    PrismaScout,
    /// AutoRound: INT4 weights via auto-round method.
    AutoRound,
    /// GGUF: llama.cpp quantized format.
    Gguf,
}

impl QuantizationFormat {
    /// Auto-detect quantization format from a model directory.
    ///
    /// Detection order:
    /// 1. `.gguf` files → `Gguf`
    /// 2. `quantization_config.json` → inspect `quant_method`
    /// 3. `config.json` embedded `quantization_config` → inspect `quant_method`
    /// 4. Fallback → `Bf16`
    pub fn detect(model_dir: &Path) -> anyhow::Result<Self> {
        // 1. Check for GGUF files
        if Self::has_gguf_files(model_dir) {
            return Ok(Self::Gguf);
        }

        // 2. Check quantization_config.json
        let quant_config_path = model_dir.join("quantization_config.json");
        if quant_config_path.exists() {
            let contents = std::fs::read_to_string(&quant_config_path)?;
            if let Ok(config) = serde_json::from_str::<QuantizationConfig>(&contents) {
                return Ok(Self::from_quant_method(&config.quant_method));
            }
        }

        // 3. Check config.json for embedded quantization_config
        let config_path = model_dir.join("config.json");
        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&contents)?;
            if let Some(method) = config
                .get("quantization_config")
                .and_then(|q| q.get("quant_method"))
                .and_then(|v| v.as_str())
            {
                return Ok(Self::from_quant_method(method));
            }
        }

        // 4. Fallback
        Ok(Self::Bf16)
    }

    fn has_gguf_files(model_dir: &Path) -> bool {
        match std::fs::read_dir(model_dir) {
            Ok(entries) => {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
            }
            Err(_) => false,
        }
    }

    fn from_quant_method(method: &str) -> Self {
        match method {
            "compressed-tensors" => Self::PrismaScout,
            "auto-round" => Self::AutoRound,
            _ => Self::Bf16,
        }
    }
}

/// Quantization configuration parsed from JSON.
///
/// Stores the quantization method and any format-specific fields as
/// arbitrary JSON for flexibility across PrismaSCOUT and AutoRound.
// @lat: [[lat#Model Config and Format Detection#QuantizationConfig]]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization method identifier (e.g. `"compressed-tensors"`, `"auto-round"`).
    pub quant_method: String,

    /// Format-specific configuration data (arbitrary JSON).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl QuantizationConfig {
    /// Return the detected [`QuantizationFormat`] for this config.
    pub fn format(&self) -> QuantizationFormat {
        QuantizationFormat::from_quant_method(&self.quant_method)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_json(path: &Path, value: &serde_json::Value) {
        fs::write(path, serde_json::to_string(value).unwrap()).unwrap();
    }

    #[test]
    fn detect_gguf_format() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.gguf"), b"fake").unwrap();

        let format = QuantizationFormat::detect(dir.path()).unwrap();
        assert_eq!(format, QuantizationFormat::Gguf);
    }

    #[test]
    fn detect_prisma_scout_via_quantization_config_json() {
        let dir = tempfile::tempdir().unwrap();
        write_json(
            dir.path().join("quantization_config.json").as_ref(),
            &serde_json::json!({
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "type": "float"}
                    }
                }
            }),
        );

        let format = QuantizationFormat::detect(dir.path()).unwrap();
        assert_eq!(format, QuantizationFormat::PrismaScout);
    }

    #[test]
    fn detect_auto_round_via_quantization_config_json() {
        let dir = tempfile::tempdir().unwrap();
        write_json(
            dir.path().join("quantization_config.json").as_ref(),
            &serde_json::json!({
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 128,
                "sym": true
            }),
        );

        let format = QuantizationFormat::detect(dir.path()).unwrap();
        assert_eq!(format, QuantizationFormat::AutoRound);
    }

    #[test]
    fn detect_prisma_scout_via_config_json_embedded() {
        let dir = tempfile::tempdir().unwrap();
        write_json(
            dir.path().join("config.json").as_ref(),
            &serde_json::json!({
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
                "quantization_config": {
                    "quant_method": "compressed-tensors"
                }
            }),
        );

        let format = QuantizationFormat::detect(dir.path()).unwrap();
        assert_eq!(format, QuantizationFormat::PrismaScout);
    }

    #[test]
    fn detect_bf16_fallback() {
        let dir = tempfile::tempdir().unwrap();
        // No quantization config, no GGUF files
        let format = QuantizationFormat::detect(dir.path()).unwrap();
        assert_eq!(format, QuantizationFormat::Bf16);
    }

    #[test]
    fn detect_gguf_takes_priority_over_quantization_config() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.gguf"), b"fake").unwrap();
        write_json(
            dir.path().join("quantization_config.json").as_ref(),
            &serde_json::json!({
                "quant_method": "compressed-tensors"
            }),
        );

        let format = QuantizationFormat::detect(dir.path()).unwrap();
        // GGUF should win
        assert_eq!(format, QuantizationFormat::Gguf);
    }

    #[test]
    fn quant_config_deserialize() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "float", "strategy": "tensor"}
                }
            },
            "format_version": "1.0"
        }"#;
        let config: QuantizationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.quant_method, "compressed-tensors");
        assert!(config.extra.contains_key("config_groups"));
        assert!(config.extra.contains_key("format_version"));
        assert_eq!(config.format(), QuantizationFormat::PrismaScout);
    }

    #[test]
    fn quant_config_auto_round() {
        let json = r#"{
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": true
        }"#;
        let config: QuantizationConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.quant_method, "auto-round");
        assert_eq!(config.format(), QuantizationFormat::AutoRound);
    }
}
