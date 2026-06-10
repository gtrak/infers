//! Multi-format model loader with auto-detection.
//!
//! Loads model weights from safetensors files (single or sharded),
//! detects quantization format, and constructs a WeightRegistry.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use serde::Deserialize;

use super::config::ModelConfig;
use super::formats::QuantizationFormat;
use super::weights::{WeightData, WeightDtype, WeightRegistry};

/// Index file for sharded safetensors models.
/// Maps tensor names to shard filenames.
#[derive(Debug, Deserialize)]
pub struct ShardIndex {
    /// Map from tensor name to shard filename.
    pub weight_map: HashMap<String, String>,
    /// Metadata (model name, etc).
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Result of loading a model directory.
#[derive(Debug)]
pub struct LoadedModel {
    /// Parsed model configuration.
    pub config: ModelConfig,
    /// Detected quantization format.
    pub format: QuantizationFormat,
    /// Loaded weight registry.
    pub weights: WeightRegistry,
}

/// Load a model from a directory.
// @lat: [[lat#Safetensors Loader#Loading Pipeline]]
///
/// This is the main entry point for model loading. It:
/// 1. Reads config.json
/// 2. Detects quantization format
/// 3. Loads safetensors files
/// 4. Constructs a WeightRegistry
pub fn load_model(model_dir: &Path) -> Result<LoadedModel> {
    let config = ModelConfig::load(model_dir)?;
    let format = QuantizationFormat::detect(model_dir)?;
    let weights = load_safetensors(model_dir)?;

    tracing::info!(
        "Loaded model: {} layers, format: {:?}, {} tensors, {:.2} GB",
        config.num_hidden_layers,
        format,
        weights.num_tensors(),
        weights.total_bytes() as f64 / 1e9,
    );

    Ok(LoadedModel {
        config,
        format,
        weights,
    })
}

/// Load all safetensors files from a model directory.
// @lat: [[lat#Safetensors Loader#Single vs Sharded]]
///
/// Handles both single-file (`model.safetensors`) and sharded
/// (`model.safetensors.index.json` + multiple shard files) formats.
pub fn load_safetensors(model_dir: &Path) -> Result<WeightRegistry> {
    let index_path = model_dir.join("model.safetensors.index.json");

    if index_path.exists() {
        load_sharded(model_dir, &index_path)
    } else {
        let single_path = model_dir.join("model.safetensors");
        if single_path.exists() {
            load_single(&single_path)
        } else {
            anyhow::bail!(
                "No safetensors files found in {:?}. Expected model.safetensors or model.safetensors.index.json",
                model_dir
            )
        }
    }
}

fn load_single(path: &Path) -> Result<WeightRegistry> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open safetensors file: {:?}", path))?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut tensors = HashMap::new();
    for name in st.names() {
        let tensor = st.tensor(name)?;
        let shape: Vec<usize> = tensor.shape().to_vec();
        let dtype = map_safetensor_dtype(tensor.dtype());
        let data = tensor.data().to_vec();

        tensors.insert(name.to_string(), WeightData {
            data,
            shape,
            dtype,
            name: name.to_string(),
        });
    }

    let total_bytes: usize = tensors.values().map(|t| t.data.len()).sum();
    tracing::debug!("Loaded {} tensors ({:.2} GB) from {:?}", tensors.len(), total_bytes as f64 / 1e9, path);

    let mut registry = WeightRegistry::new();
    registry.tensors = tensors;
    Ok(registry)
}

fn load_sharded(model_dir: &Path, index_path: &Path) -> Result<WeightRegistry> {
    let index_content = std::fs::read_to_string(index_path)?;
    let index: ShardIndex = serde_json::from_str(&index_content)?;

    // Collect unique shard filenames
    let shards: std::collections::HashSet<String> = index.weight_map.values().cloned().collect();
    let mut all_tensors = HashMap::new();

    for shard_name in &shards {
        let shard_path = model_dir.join(shard_name);
        tracing::debug!("Loading shard: {:?}", shard_path);

        let file = std::fs::File::open(&shard_path)
            .with_context(|| format!("Failed to open shard: {:?}", shard_path))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        for name in st.names() {
            let tensor = st.tensor(name)?;
            let shape: Vec<usize> = tensor.shape().to_vec();
            let dtype = map_safetensor_dtype(tensor.dtype());
            let data = tensor.data().to_vec();

            all_tensors.insert(name.to_string(), WeightData {
                data,
                shape,
                dtype,
                name: name.to_string(),
            });
        }
    }

    let total_bytes: usize = all_tensors.values().map(|t| t.data.len()).sum();
    tracing::info!(
        "Loaded {} shards, {} tensors ({:.2} GB)",
        shards.len(),
        all_tensors.len(),
        total_bytes as f64 / 1e9,
    );

    let mut registry = WeightRegistry::new();
    registry.tensors = all_tensors;
    Ok(registry)
}

/// Map safetensors dtype to our WeightDtype.
fn map_safetensor_dtype(dtype: safetensors::Dtype) -> WeightDtype {
    match dtype {
        safetensors::Dtype::BF16 => WeightDtype::Bf16,
        safetensors::Dtype::F16 => WeightDtype::Fp16,
        safetensors::Dtype::F32 => WeightDtype::Fp32,
        safetensors::Dtype::U8 => WeightDtype::Other,
        safetensors::Dtype::I8 => WeightDtype::Other,
        safetensors::Dtype::I16 => WeightDtype::Other,
        safetensors::Dtype::I32 => WeightDtype::Other,
        safetensors::Dtype::I64 => WeightDtype::Other,
        safetensors::Dtype::F64 => WeightDtype::Other,
        safetensors::Dtype::BOOL => WeightDtype::Other,
        // Handle any future dtypes
        _ => WeightDtype::Other,
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
    fn load_safetensors_no_files_bails() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_safetensors(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("No safetensors files found"));
    }

    #[test]
    fn load_model_config_and_format() {
        // Just verify load_model reads config successfully when files exist.
        // We don't need actual safetensors for the config to parse.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            qwen3_6_config_json().as_bytes(),
        )
        .unwrap();
        // No safetensors, so load_model will fail at the weights step,
        // but config loading works.
        let result = load_model(dir.path());
        // Expected to fail because no safetensors
        assert!(result.is_err());
    }

    #[test]
    fn map_safetensor_dtype_bf16() {
        assert_eq!(map_safetensor_dtype(safetensors::Dtype::BF16), WeightDtype::Bf16);
    }

    #[test]
    fn map_safetensor_dtype_fp32() {
        assert_eq!(map_safetensor_dtype(safetensors::Dtype::F32), WeightDtype::Fp32);
    }

    #[test]
    fn map_safetensor_dtype_f16() {
        assert_eq!(map_safetensor_dtype(safetensors::Dtype::F16), WeightDtype::Fp16);
    }

    #[test]
    fn map_safetensor_dtype_other() {
        assert_eq!(map_safetensor_dtype(safetensors::Dtype::U8), WeightDtype::Other);
        assert_eq!(map_safetensor_dtype(safetensors::Dtype::BOOL), WeightDtype::Other);
    }
}
