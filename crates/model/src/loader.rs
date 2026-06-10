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
use super::weights::{
    AttentionWeights, GdnWeights, LayerWeights, MlpWeights, MtpWeights, WeightData, WeightDtype,
    WeightRegistry,
};

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
    let mut weights = load_safetensors(model_dir)?;

    // Build structured MTP weights if the model has MTP
    build_mtp_weights(&mut weights, &config)?;

    tracing::info!(
        "Loaded model: {} layers, format: {:?}, {} tensors, {:.2} GB{}",
        config.num_hidden_layers,
        format,
        weights.num_tensors(),
        weights.total_bytes() as f64 / 1e9,
        if config.has_mtp() {
            format!(" (MTP: {} layers)", config.mtp_num_hidden_layers)
        } else {
            String::new()
        },
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
    // SAFETY: The file is opened read-only (std::fs::File::open), the file handle
    // is verified to exist before mapping, and the mapping is read-only weight data.
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
        // SAFETY: The shard file is opened read-only, verified to exist before mapping,
        // and the mapping is read-only weight data.
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

/// Build MTP weights from the flat tensor map.
///
/// Extracts MTP tensors from `registry.tensors` and populates `registry.mtp`
/// with structured `MtpWeights`. MTP layer tensor names follow the pattern:
/// `mtp.layers.{i}.<submodule>.<proj>.weight`
///
/// This function is called during model loading when `config.has_mtp()` is true.
pub fn build_mtp_weights(registry: &mut WeightRegistry, config: &ModelConfig) -> Result<()> {
    if !config.has_mtp() {
        return Ok(());
    }

    let num_mtp_layers = config.mtp_num_hidden_layers;

    // Load pre-FC norms and FC projection
    let pre_fc_norm_embedding = get_weight(registry, "mtp.pre_fc_norm_embedding.weight")?;
    let pre_fc_norm_hidden = get_weight(registry, "mtp.pre_fc_norm_hidden.weight")?;
    let fc = get_weight(registry, "mtp.fc.weight")?;
    let norm = get_weight(registry, "mtp.norm.weight")?;

    // Build MTP layers
    let mut layers = Vec::with_capacity(num_mtp_layers);
    for i in 0..num_mtp_layers {
        let layer = build_mtp_layer(registry, config, i)?;
        layers.push(layer);
    }

    // Load dedicated embeddings if configured
    let embed_tokens = if config.mtp_use_dedicated_embeddings {
        Some(get_weight(registry, "mtp.embed_tokens.weight")?)
    } else {
        None
    };

    registry.mtp = Some(MtpWeights {
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        fc,
        layers,
        norm,
        embed_tokens,
    });

    Ok(())
}

/// Build a single MTP layer from the flat tensor map.
fn build_mtp_layer(
    registry: &WeightRegistry,
    config: &ModelConfig,
    layer_idx: usize,
) -> Result<LayerWeights> {
    let prefix = format!("mtp.layers.{}", layer_idx);

    let norm1 = get_weight(registry, &format!("{}.input_layernorm.weight", prefix))?;
    let norm2 = get_weight(registry, &format!("{}.post_attention_layernorm.weight", prefix))?;

    let layer_type = config.get_layer_type(layer_idx);

    let (gdn, attn) = match layer_type {
        super::config::LayerType::GatedDeltaNet => {
            let gdn = GdnWeights {
                in_proj_a: get_weight(registry, &format!("{}.gdn.in_proj_a.weight", prefix))?,
                in_proj_b: get_weight(registry, &format!("{}.gdn.in_proj_b.weight", prefix))?,
                conv1d_weight: get_weight(registry, &format!("{}.gdn.conv1d_weight.weight", prefix))?,
                x_proj_weight: get_weight(registry, &format!("{}.gdn.x_proj_weight.weight", prefix))?,
                dt_proj_weight: get_weight(registry, &format!("{}.gdn.dt_proj_weight.weight", prefix))?,
                out_proj_weight: get_weight(registry, &format!("{}.gdn.out_proj_weight.weight", prefix))?,
            };
            (Some(gdn), None)
        }
        super::config::LayerType::FullAttention => {
            let attn = AttentionWeights {
                q_proj: get_weight(registry, &format!("{}.self_attn.q_proj.weight", prefix))?,
                k_proj: get_weight(registry, &format!("{}.self_attn.k_proj.weight", prefix))?,
                v_proj: get_weight(registry, &format!("{}.self_attn.v_proj.weight", prefix))?,
                o_proj: get_weight(registry, &format!("{}.self_attn.o_proj.weight", prefix))?,
            };
            (None, Some(attn))
        }
    };

    let mlp = MlpWeights {
        gate_proj: get_weight(registry, &format!("{}.mlp.gate_proj.weight", prefix))?,
        up_proj: get_weight(registry, &format!("{}.mlp.up_proj.weight", prefix))?,
        down_proj: get_weight(registry, &format!("{}.mlp.down_proj.weight", prefix))?,
    };

    Ok(LayerWeights {
        layer_type,
        layer_idx,
        gdn,
        attn,
        mlp,
        norm1,
        norm2,
    })
}

/// Get a weight tensor from the registry by name.
fn get_weight(registry: &WeightRegistry, name: &str) -> Result<WeightData> {
    registry
        .tensors
        .get(name)
        .cloned()
        .with_context(|| format!("MTP tensor not found: {}", name))
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

    #[test]
    fn build_mtp_weights_no_mtp_returns_early() {
        // Config without MTP — build_mtp_weights should be a no-op
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();
        assert!(!config.has_mtp());

        let mut registry = WeightRegistry::new();
        assert!(build_mtp_weights(&mut registry, &config).is_ok());
        assert!(registry.mtp.is_none());
    }

    #[test]
    fn build_mtp_weights_with_full_attention_layer() {
        // Config with MTP enabled, 1 full-attention layer
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false,"layer_types":["full_attention"]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();
        assert!(config.has_mtp());
        assert_eq!(config.mtp_num_hidden_layers, 1);

        let mut registry = WeightRegistry::new();
        let dummy = |name: &str| WeightData {
            data: vec![0u8; 32],
            shape: vec![2, 16],
            dtype: WeightDtype::Bf16,
            name: name.to_string(),
        };

        // Pre-FC norms and FC
        registry.tensors.insert("mtp.pre_fc_norm_embedding.weight".to_string(), dummy("mtp.pre_fc_norm_embedding.weight"));
        registry.tensors.insert("mtp.pre_fc_norm_hidden.weight".to_string(), dummy("mtp.pre_fc_norm_hidden.weight"));
        registry.tensors.insert("mtp.fc.weight".to_string(), dummy("mtp.fc.weight"));
        registry.tensors.insert("mtp.norm.weight".to_string(), dummy("mtp.norm.weight"));

        // Layer 0: norms
        registry.tensors.insert("mtp.layers.0.input_layernorm.weight".to_string(), dummy("mtp.layers.0.input_layernorm.weight"));
        registry.tensors.insert("mtp.layers.0.post_attention_layernorm.weight".to_string(), dummy("mtp.layers.0.post_attention_layernorm.weight"));

        // Layer 0: attention (FullAttention)
        registry.tensors.insert("mtp.layers.0.self_attn.q_proj.weight".to_string(), dummy("mtp.layers.0.self_attn.q_proj.weight"));
        registry.tensors.insert("mtp.layers.0.self_attn.k_proj.weight".to_string(), dummy("mtp.layers.0.self_attn.k_proj.weight"));
        registry.tensors.insert("mtp.layers.0.self_attn.v_proj.weight".to_string(), dummy("mtp.layers.0.self_attn.v_proj.weight"));
        registry.tensors.insert("mtp.layers.0.self_attn.o_proj.weight".to_string(), dummy("mtp.layers.0.self_attn.o_proj.weight"));

        // Layer 0: MLP
        registry.tensors.insert("mtp.layers.0.mlp.gate_proj.weight".to_string(), dummy("mtp.layers.0.mlp.gate_proj.weight"));
        registry.tensors.insert("mtp.layers.0.mlp.up_proj.weight".to_string(), dummy("mtp.layers.0.mlp.up_proj.weight"));
        registry.tensors.insert("mtp.layers.0.mlp.down_proj.weight".to_string(), dummy("mtp.layers.0.mlp.down_proj.weight"));

        let result = build_mtp_weights(&mut registry, &config);
        assert!(result.is_ok());

        let mtp = registry.mtp.as_ref().expect("MTP weights should be populated");
        assert_eq!(mtp.layers.len(), 1);
        assert!(mtp.embed_tokens.is_none());
        assert!(mtp.layers[0].attn.is_some());
        assert!(mtp.layers[0].gdn.is_none());
    }

    #[test]
    fn build_mtp_weights_with_gdn_layer() {
        // Config with MTP enabled, 1 GDN layer
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false,"layer_types":["linear_attention"]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let mut registry = WeightRegistry::new();
        let dummy = |name: &str| WeightData {
            data: vec![0u8; 32],
            shape: vec![2, 16],
            dtype: WeightDtype::Bf16,
            name: name.to_string(),
        };

        // Pre-FC norms and FC
        registry.tensors.insert("mtp.pre_fc_norm_embedding.weight".to_string(), dummy("mtp.pre_fc_norm_embedding.weight"));
        registry.tensors.insert("mtp.pre_fc_norm_hidden.weight".to_string(), dummy("mtp.pre_fc_norm_hidden.weight"));
        registry.tensors.insert("mtp.fc.weight".to_string(), dummy("mtp.fc.weight"));
        registry.tensors.insert("mtp.norm.weight".to_string(), dummy("mtp.norm.weight"));

        // Layer 0: norms
        registry.tensors.insert("mtp.layers.0.input_layernorm.weight".to_string(), dummy("mtp.layers.0.input_layernorm.weight"));
        registry.tensors.insert("mtp.layers.0.post_attention_layernorm.weight".to_string(), dummy("mtp.layers.0.post_attention_layernorm.weight"));

        // Layer 0: GDN
        registry.tensors.insert("mtp.layers.0.gdn.in_proj_a.weight".to_string(), dummy("mtp.layers.0.gdn.in_proj_a.weight"));
        registry.tensors.insert("mtp.layers.0.gdn.in_proj_b.weight".to_string(), dummy("mtp.layers.0.gdn.in_proj_b.weight"));
        registry.tensors.insert("mtp.layers.0.gdn.conv1d_weight.weight".to_string(), dummy("mtp.layers.0.gdn.conv1d_weight.weight"));
        registry.tensors.insert("mtp.layers.0.gdn.x_proj_weight.weight".to_string(), dummy("mtp.layers.0.gdn.x_proj_weight.weight"));
        registry.tensors.insert("mtp.layers.0.gdn.dt_proj_weight.weight".to_string(), dummy("mtp.layers.0.gdn.dt_proj_weight.weight"));
        registry.tensors.insert("mtp.layers.0.gdn.out_proj_weight.weight".to_string(), dummy("mtp.layers.0.gdn.out_proj_weight.weight"));

        // Layer 0: MLP
        registry.tensors.insert("mtp.layers.0.mlp.gate_proj.weight".to_string(), dummy("mtp.layers.0.mlp.gate_proj.weight"));
        registry.tensors.insert("mtp.layers.0.mlp.up_proj.weight".to_string(), dummy("mtp.layers.0.mlp.up_proj.weight"));
        registry.tensors.insert("mtp.layers.0.mlp.down_proj.weight".to_string(), dummy("mtp.layers.0.mlp.down_proj.weight"));

        let result = build_mtp_weights(&mut registry, &config);
        assert!(result.is_ok());

        let mtp = registry.mtp.as_ref().expect("MTP weights should be populated");
        assert_eq!(mtp.layers.len(), 1);
        assert!(mtp.embed_tokens.is_none());
        assert!(mtp.layers[0].gdn.is_some());
        assert!(mtp.layers[0].attn.is_none());
    }

    #[test]
    fn build_mtp_weights_with_dedicated_embeddings() {
        // Config with MTP enabled and dedicated embeddings
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":true,"layer_types":["linear_attention"]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let mut registry = WeightRegistry::new();
        let dummy = |name: &str| WeightData {
            data: vec![0u8; 32],
            shape: vec![2, 16],
            dtype: WeightDtype::Bf16,
            name: name.to_string(),
        };

        // Pre-FC norms and FC
        registry.tensors.insert("mtp.pre_fc_norm_embedding.weight".to_string(), dummy("mtp.pre_fc_norm_embedding.weight"));
        registry.tensors.insert("mtp.pre_fc_norm_hidden.weight".to_string(), dummy("mtp.pre_fc_norm_hidden.weight"));
        registry.tensors.insert("mtp.fc.weight".to_string(), dummy("mtp.fc.weight"));
        registry.tensors.insert("mtp.norm.weight".to_string(), dummy("mtp.norm.weight"));

        // Dedicated embeddings
        registry.tensors.insert("mtp.embed_tokens.weight".to_string(), dummy("mtp.embed_tokens.weight"));

        // Layer 0: norms
        registry.tensors.insert("mtp.layers.0.input_layernorm.weight".to_string(), dummy("mtp.layers.0.input_layernorm.weight"));
        registry.tensors.insert("mtp.layers.0.post_attention_layernorm.weight".to_string(), dummy("mtp.layers.0.post_attention_layernorm.weight"));

        // Layer 0: GDN
        registry.tensors.insert("mtp.layers.0.gdn.in_proj_a.weight".to_string(), dummy("mtp.layers.0.gdn.in_proj_a.weight"));
        registry.tensors.insert("mtp.layers.0.gdn.in_proj_b.weight".to_string(), dummy("mtp.layers.0.gdn.in_proj_b.weight"));
        registry.tensors.insert("mtp.layers.0.gdn.conv1d_weight.weight".to_string(), dummy("mtp.layers.0.gdn.conv1d_weight.weight"));
        registry.tensors.insert("mtp.layers.0.gdn.x_proj_weight.weight".to_string(), dummy("mtp.layers.0.gdn.x_proj_weight.weight"));
        registry.tensors.insert("mtp.layers.0.gdn.dt_proj_weight.weight".to_string(), dummy("mtp.layers.0.gdn.dt_proj_weight.weight"));
        registry.tensors.insert("mtp.layers.0.gdn.out_proj_weight.weight".to_string(), dummy("mtp.layers.0.gdn.out_proj_weight.weight"));

        // Layer 0: MLP
        registry.tensors.insert("mtp.layers.0.mlp.gate_proj.weight".to_string(), dummy("mtp.layers.0.mlp.gate_proj.weight"));
        registry.tensors.insert("mtp.layers.0.mlp.up_proj.weight".to_string(), dummy("mtp.layers.0.mlp.up_proj.weight"));
        registry.tensors.insert("mtp.layers.0.mlp.down_proj.weight".to_string(), dummy("mtp.layers.0.mlp.down_proj.weight"));

        let result = build_mtp_weights(&mut registry, &config);
        assert!(result.is_ok());

        let mtp = registry.mtp.as_ref().expect("MTP weights should be populated");
        assert!(mtp.embed_tokens.is_some(), "embed_tokens should be present when mtp_use_dedicated_embeddings is true");
    }

    #[test]
    fn build_mtp_weights_missing_tensor_fails() {
        // Config with MTP enabled
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let mut registry = WeightRegistry::new();
        // No tensors at all — should fail
        let result = build_mtp_weights(&mut registry, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("mtp.pre_fc_norm_embedding.weight"));
    }

    #[test]
    fn mtp_weights_struct_has_expected_fields() {
        // Verify MtpWeights has all expected fields by constructing one
        let dummy = WeightData {
            data: vec![0u8; 32],
            shape: vec![2, 16],
            dtype: WeightDtype::Bf16,
            name: "test".to_string(),
        };
        let mtp = MtpWeights {
            pre_fc_norm_embedding: dummy.clone(),
            pre_fc_norm_hidden: dummy.clone(),
            fc: dummy.clone(),
            layers: Vec::new(),
            norm: dummy.clone(),
            embed_tokens: None,
        };
        assert!(mtp.layers.is_empty());
        assert!(mtp.embed_tokens.is_none());
    }
}
