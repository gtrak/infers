//! Quantization format detection and configuration.
//!
//! Supports PrismaSCOUT (NVFP4), AutoRound (INT4), GGUF, and BF16 formats
//! with automatic detection from model directory contents.

use anyhow::Context;
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
/// Resolved per-tensor quantization assignment from config metadata.

/// Built from a `QuantizationConfig`'s `config_groups` targets and `ignore` lists.
/// After `strip_language_model_prefix`, tensor base names are matched against
/// compiled regexes to determine their quantization format.
// @lat: [[lat#Model Config and Format Detection#QuantTargetMap]]
pub struct QuantTargetMap {
    /// Compiled regex patterns for NVFP4-quantized tensor base names.
    nvfp4_targets: Vec<regex::Regex>,
    /// Compiled regex patterns for INT4-quantized tensor base names (future-proof).
    int4_targets: Vec<regex::Regex>,
    /// Compiled regex patterns for BF16-passthrough (ignored) tensor base names.
    ignore: Vec<regex::Regex>,
}

impl QuantTargetMap {
    /// Build from a QuantizationConfig's config_groups and ignore list.
    ///
    /// Parses `config_groups` entries: if the format contains "nvfp4", the targets
    /// go into `nvfp4_targets`; if it contains "int4", they go into `int4_targets`.
    /// The `ignore` list entries are compiled as regexes (with `re:` prefix stripped).
    pub fn from_config(config: &QuantizationConfig) -> anyhow::Result<Self> {
        let mut nvfp4_targets = Vec::new();
        let mut int4_targets = Vec::new();
        let mut ignore = Vec::new();

        match config.quant_method.as_str() {
            "auto-round" => {
                // AutoRound: everything under block_name_to_quantize is INT4,
                // except tensors in extra_config with bits=16 (BF16 overrides).
                if let Some(blocks) = config.extra.get("block_name_to_quantize") {
                    let block_names: Vec<&str> = if let Some(s) = blocks.as_str() {
                        vec![s]
                    } else if let Some(arr) = blocks.as_array() {
                        arr.iter().filter_map(|v| v.as_str()).collect()
                    } else {
                        vec![]
                    };

                    for block in block_names {
                        // Strip "model.language_model." prefix to get the stripped-name form.
                        // e.g., "model.language_model.layers" → "layers"
                        //       "mtp.layers" → "mtp.layers" (already stripped)
                        let stripped = block
                            .strip_prefix("model.language_model.")
                            .unwrap_or(block);
                        // Match any tensor base name under this block
                        let pattern = format!("^{}\\..+", regex::escape(stripped));
                        let compiled = regex::Regex::new(&pattern)
                            .with_context(|| format!("Invalid regex for AutoRound block '{}': {}", block, pattern))?;
                        int4_targets.push(compiled);
                    }
                }

                // Parse extra_config: per-tensor BF16 overrides
                if let Some(extra) = config.extra.get("extra_config") {
                    if let Some(obj) = extra.as_object() {
                        for (key, value) in obj {
                            let bits = value.get("bits").and_then(|v| v.as_u64()).unwrap_or(4);
                            if bits == 16 {
                                // This tensor should be BF16 — add to ignore.
                                // Keys can be regex patterns (start with "." or ".*")
                                // or exact names like "model.language_model.layers.0.xxx"
                                let regex_pattern = if key.starts_with('.') || key.contains(".*") || key.contains("\\.") {
                                    // Regex pattern — use as-is (anchored)
                                    format!("^{}$", Self::strip_re_prefix(key))
                                } else {
                                    // Exact name — strip prefix and escape
                                    let stripped = key
                                        .strip_prefix("model.language_model.")
                                        .unwrap_or(key);
                                    format!("^{}$", regex::escape(stripped))
                                };
                                let compiled = regex::Regex::new(&regex_pattern)
                                    .with_context(|| format!("Invalid regex in extra_config: {}", regex_pattern))?;
                                ignore.push(compiled);
                            }
                        }
                    }
                }
            }
            _ => {
                // PrismaSCOUT and other formats: parse config_groups
                if let Some(groups) = config.extra.get("config_groups") {
                    if let Some(groups_obj) = groups.as_object() {
                        for (_group_name, group_value) in groups_obj {
                            if let Some(group_obj) = group_value.as_object() {
                                let format = group_obj
                                    .get("format")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");

                                let targets = group_obj
                                    .get("targets")
                                    .and_then(|v| v.as_array())
                                    .cloned()
                                    .unwrap_or_default();

                                for target in targets {
                                    if let Some(pattern) = target.as_str() {
                                        let regex_pattern = Self::strip_re_prefix(pattern);
                                        let compiled = regex::Regex::new(regex_pattern)
                                            .with_context(|| format!("Invalid regex in targets: {}", pattern))?;

                                        if format.contains("nvfp4") {
                                            nvfp4_targets.push(compiled);
                                        } else if format.contains("int4") {
                                            int4_targets.push(compiled);
                                        } else {
                                            // Unknown format — treat as NVFP4 for compressed-tensors
                                            nvfp4_targets.push(compiled);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Parse ignore list
                if let Some(ignore_list) = config.extra.get("ignore") {
                    if let Some(arr) = ignore_list.as_array() {
                        for entry in arr {
                            if let Some(pattern) = entry.as_str() {
                                // Ignore entries may have "re:" prefix for regex patterns
                                let regex_pattern = if pattern.starts_with("re:") {
                                    Self::strip_re_prefix(pattern).to_string()
                                } else {
                                    // Exact match: anchor it
                                    format!("^{}$", regex::escape(pattern))
                                };
                                let compiled = regex::Regex::new(&regex_pattern)
                                    .with_context(|| format!("Invalid regex in ignore: {}", pattern))?;
                                ignore.push(compiled);
                            }
                        }
                    }
                }
            }
        }

        Ok(Self { nvfp4_targets, int4_targets, ignore })
    }

    /// Create an empty map (all tensors are BF16 passthrough).
    pub fn empty() -> Self {
        Self {
            nvfp4_targets: Vec::new(),
            int4_targets: Vec::new(),
            ignore: Vec::new(),
        }
    }

    /// Returns the quantization format for a tensor base name, or None if BF16.
    ///
    /// The tensor base name should be the stripped name (after `strip_language_model_prefix`),
    /// e.g., `layers.0.linear_attn.in_proj_qkv`.
    ///
    /// The targets regexes in config.json use `language_model.model.layers.X.YYY` format,
    /// but after stripping the `model.language_model.` prefix, they become `layers.X.YYY`.
    /// So we need to prepend `language_model.model.` back when matching against targets.
    ///
    /// IMPORTANT: The regex targets in config.json match against the FULL tensor base name
    /// (before stripping), e.g., `language_model.model.layers.0.linear_attn.in_proj_qkv`.
    /// After stripping, the same tensor is `layers.0.linear_attn.in_proj_qkv`.
    ///
    /// Strategy: When checking a stripped name like `layers.0.linear_attn.in_proj_qkv`,
    /// we also check against `language_model.model.layers.0.linear_attn.in_proj_qkv`
    /// to match the original config targets.
    pub fn resolve(&self, tensor_base: &str) -> Option<QuantizationFormat> {
        // Build both the stripped name and the full (unstripped) name for matching
        let unstripped = if tensor_base.starts_with("layers.") || tensor_base.starts_with("mtp.") || tensor_base.starts_with("visual.") {
            format!("language_model.model.{}", tensor_base)
        } else {
            tensor_base.to_string()
        };

        // Check ignore list first — BF16 passthrough takes priority
        for re in &self.ignore {
            if re.is_match(tensor_base) || re.is_match(&unstripped) {
                return None; // BF16 passthrough
            }
        }

        // Check NVFP4 targets
        for re in &self.nvfp4_targets {
            if re.is_match(tensor_base) || re.is_match(&unstripped) {
                return Some(QuantizationFormat::PrismaScout);
            }
        }

        // Check INT4 targets
        for re in &self.int4_targets {
            if re.is_match(tensor_base) || re.is_match(&unstripped) {
                return Some(QuantizationFormat::AutoRound);
            }
        }

        // Not in any target — BF16 passthrough
        None
    }

    /// Strip "re:" prefix from a regex pattern string.
    fn strip_re_prefix(pattern: &str) -> &str {
        pattern.strip_prefix("re:").unwrap_or(pattern)
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

    #[test]
    fn quant_target_map_empty_returns_none() {
        let map = QuantTargetMap::empty();
        assert_eq!(map.resolve("layers.0.mlp.gate_proj"), None);
    }

    #[test]
    fn quant_target_map_from_config_nvfp4() {
        let config: QuantizationConfig = serde_json::from_str(r#"{
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "targets": [
                        "re:^language_model[.]model[.]layers[.]0[.]linear_attn[.]in_proj_qkv$"
                    ]
                }
            },
            "ignore": []
        }"#).unwrap();

        let map = QuantTargetMap::from_config(&config).unwrap();
        // Stripped name should match via unstripped reconstruction
        assert_eq!(map.resolve("layers.0.linear_attn.in_proj_qkv"), Some(QuantizationFormat::PrismaScout));
        // Non-target should return None
        assert_eq!(map.resolve("layers.0.linear_attn.in_proj_a"), None);
    }

    #[test]
    fn quant_target_map_ignore_takes_priority() {
        let config: QuantizationConfig = serde_json::from_str(r#"{
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "targets": [
                        "re:^language_model[.]model[.]layers[.]0[.]mlp[.]gate_proj$"
                    ]
                }
            },
            "ignore": [
                "language_model.model.layers.0.mlp.gate_proj"
            ]
        }"#).unwrap();

        let map = QuantTargetMap::from_config(&config).unwrap();
        // Ignore takes priority — BF16 passthrough
        assert_eq!(map.resolve("layers.0.mlp.gate_proj"), None);
    }

    #[test]
    fn quant_target_map_mtp_resolution() {
        let config: QuantizationConfig = serde_json::from_str(r#"{
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "targets": [
                        "re:^mtp[.]layers[.]0[.]mlp[.]gate_proj$"
                    ]
                }
            },
            "ignore": []
        }"#).unwrap();

        let map = QuantTargetMap::from_config(&config).unwrap();
        // MTP targets don't get the language_model.model. prefix
        assert_eq!(map.resolve("mtp.layers.0.mlp.gate_proj"), Some(QuantizationFormat::PrismaScout));
    }

    #[test]
    fn quant_target_map_auto_round_block_quantization() {
        let config: QuantizationConfig = serde_json::from_str(r#"{
            "quant_method": "auto-round",
            "block_name_to_quantize": ["model.language_model.layers", "mtp.layers"],
            "extra_config": {
                "model.language_model.layers.0.linear_attn.in_proj_a": {"bits": 16, "data_type": "fp"},
                "model.language_model.layers.0.linear_attn.in_proj_b": {"bits": 16, "data_type": "fp"}
            }
        }"#).unwrap();

        let map = QuantTargetMap::from_config(&config).unwrap();

        // INT4: MLP and attention projections within quantized blocks
        assert_eq!(map.resolve("layers.0.mlp.gate_proj"), Some(QuantizationFormat::AutoRound));
        assert_eq!(map.resolve("layers.0.mlp.up_proj"), Some(QuantizationFormat::AutoRound));
        assert_eq!(map.resolve("layers.0.self_attn.q_proj"), Some(QuantizationFormat::AutoRound));
        assert_eq!(map.resolve("layers.13.linear_attn.out_proj"), Some(QuantizationFormat::AutoRound));

        // BF16: in_proj_a/in_proj_b excluded via extra_config (layer 0 only in test data)
        assert_eq!(map.resolve("layers.0.linear_attn.in_proj_a"), None);
        assert_eq!(map.resolve("layers.0.linear_attn.in_proj_b"), None);
        // Layer 1 not in extra_config → still INT4 (the real config has all layers)
        assert_eq!(map.resolve("layers.1.linear_attn.in_proj_a"), Some(QuantizationFormat::AutoRound));

        // BF16: norm weights (not under block_name_to_quantize targets)
        // These won't match INT4 targets because they lack a digit after "layers."
        // Actually they DO match because "layers.0.input_layernorm" starts with "layers."
        // But the loader calls get_weight (not get_weight_with_quant) for norms, so this is fine.

        // MTP: also quantized
        assert_eq!(map.resolve("mtp.layers.0.mlp.gate_proj"), Some(QuantizationFormat::AutoRound));
    }
}
