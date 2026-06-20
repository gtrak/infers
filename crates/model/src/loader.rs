//! Multi-format model loader with auto-detection.
//!
//! Loads model weights from safetensors files (single or sharded),
//! detects quantization format, and constructs a WeightRegistry.

use anyhow::{Context, Result};
use super::config::ModelConfig;
use super::weights::{
    AttentionWeights, GdnWeights, Int4Companions, LayerWeights, MlpWeights, MtpWeights,
    WeightData, WeightRegistry,
};

/// Strip `model.language_model.` prefix from tensor names and remove vision tensors.
///
/// Tensors starting with `model.language_model.` get that prefix stripped, so
/// `model.language_model.layers.0.input_layernorm.weight` becomes
/// `layers.0.input_layernorm.weight`. Tensors starting with `model.visual.`
/// are removed entirely. Tensors starting with `mtp.` and all other tensors
/// are kept as-is.
pub fn strip_language_model_prefix(registry: &mut WeightRegistry) {
    let lang_prefix = "model.language_model.";
    let vis_prefix = "model.visual.";
    let mut to_remove = Vec::new();
    let mut to_rename = Vec::new();
    for key in registry.tensors.keys() {
        if key.starts_with(vis_prefix) {
            to_remove.push(key.clone());
        } else if key.starts_with(lang_prefix) {
            let new_key = key.strip_prefix(lang_prefix).unwrap().to_string();
            if new_key != *key {
                to_rename.push((key.clone(), new_key));
            }
        }
    }
    for key in to_remove {
        registry.tensors.remove(&key);
    }
    for (old_key, new_key) in to_rename {
        if let Some(mut weight) = registry.tensors.remove(&old_key) {
            // Also update the internal name field so companion lookups match
            if weight.name == old_key || weight.name.starts_with("model.language_model.") {
                weight.name = new_key.clone();
            }
            registry.tensors.insert(new_key, weight);
        }
    }
}

/// Build main model layers from the flat tensor map.
///
/// Populates `registry.layers`, `registry.embedding`, `registry.norm`, and
/// `registry.lm_head` from tensors in `registry.tensors`. Tensors are removed
/// from the flat map during extraction to halve memory usage.
pub fn build_main_layers(registry: &mut WeightRegistry, config: &ModelConfig) -> Result<()> {
    // Extract scalar weights: embedding, norm, lm_head
    registry.embedding = get_weight(registry, "embed_tokens.weight").ok();
    registry.norm = get_weight(registry, "norm.weight").ok();
    registry.lm_head = get_weight(registry, "lm_head.weight").ok();
    // Build per-layer weights
    let num_layers = config.num_hidden_layers;
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let layer = build_main_layer(registry, config, i)?;
        layers.push(layer);
    }
    registry.layers = layers;
    Ok(())
}

/// Build a single main model layer from the flat tensor map.
fn build_main_layer(
    registry: &mut WeightRegistry,
    config: &ModelConfig,
    layer_idx: usize,
) -> Result<LayerWeights> {
    let prefix = format!("layers.{}", layer_idx);
    let norm1 = get_weight(registry, &format!("{}.input_layernorm.weight", prefix))?;
    let norm2 = get_weight(registry, &format!("{}.post_attention_layernorm.weight", prefix))?;
    let layer_type = config.get_layer_type(layer_idx);
    // Determine GDN sub-prefix: check if linear_attn. exists, fall back to gdn.
    let gdn_sub = if registry.tensors.contains_key(&format!("{}.linear_attn.in_proj_a.weight", prefix)) {
        "linear_attn"
    } else {
        "gdn"
    };
    let (gdn, attn) = match layer_type {
        super::config::LayerType::GatedDeltaNet => {
            let p = &prefix;
            let sub = gdn_sub;
            let in_proj_a = get_weight_or_int4(registry, &format!("{p}.{sub}.in_proj_a.weight"), &format!("{p}.{sub}.in_proj_a"))?;
            let in_proj_b = get_weight_or_int4(registry, &format!("{p}.{sub}.in_proj_b.weight"), &format!("{p}.{sub}.in_proj_b"))?;
            let conv1d_weight = get_weight(registry, &format!("{p}.{sub}.conv1d.weight"))?;
            // x_proj_weight and dt_proj_weight are optional — not present in Qwen3.6
            let x_proj_weight = get_weight_or_int4_optional(registry, &format!("{p}.{sub}.x_proj_weight.weight"), &format!("{p}.{sub}.x_proj_weight"))?;
            let dt_proj_weight = get_weight_or_int4_optional(registry, &format!("{p}.{sub}.dt_proj_weight.weight"), &format!("{p}.{sub}.dt_proj_weight"))?;
            // out_proj (not out_proj_weight) — matches real Qwen3.6 tensor names
            let out_proj_weight = get_weight_or_int4(registry, &format!("{p}.{sub}.out_proj.weight"), &format!("{p}.{sub}.out_proj"))?;
            // Optional Mamba2-style weights
            let a_log = registry.tensors.remove(&format!("{p}.{sub}.A_log"));
            let dt_bias = registry.tensors.remove(&format!("{p}.{sub}.dt_bias"));
            let norm = registry.tensors.remove(&format!("{p}.{sub}.norm.weight"));
            let in_proj_qkv = get_weight_or_int4_optional(registry, &format!("{p}.{sub}.in_proj_qkv.weight"), &format!("{p}.{sub}.in_proj_qkv"))?;
            let in_proj_z = get_weight_or_int4_optional(registry, &format!("{p}.{sub}.in_proj_z.weight"), &format!("{p}.{sub}.in_proj_z"))?;
            let gdn = GdnWeights {
                in_proj_a,
                in_proj_b,
                conv1d_weight,
                x_proj_weight,
                dt_proj_weight,
                out_proj_weight,
                in_proj_qkv,
                in_proj_z,
                a_log,
                dt_bias,
                norm,
            };
            (Some(gdn), None)
        }
        super::config::LayerType::FullAttention => {
            let p = &prefix;
            let q_proj = get_weight_or_int4(registry, &format!("{p}.self_attn.q_proj.weight"), &format!("{p}.self_attn.q_proj"))?;
            let k_proj = get_weight_or_int4(registry, &format!("{p}.self_attn.k_proj.weight"), &format!("{p}.self_attn.k_proj"))?;
            let v_proj = get_weight_or_int4(registry, &format!("{p}.self_attn.v_proj.weight"), &format!("{p}.self_attn.v_proj"))?;
            let o_proj = get_weight_or_int4(registry, &format!("{p}.self_attn.o_proj.weight"), &format!("{p}.self_attn.o_proj"))?;
            // Optional Q/K norm weights
            let q_norm = registry.tensors.remove(&format!("{p}.self_attn.q_norm.weight"));
            let k_norm = registry.tensors.remove(&format!("{p}.self_attn.k_norm.weight"));
            let attn = AttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
            };
            (None, Some(attn))
        }
    };
    let mlp = MlpWeights {
        gate_proj: get_weight_or_int4(registry, &format!("{}.mlp.gate_proj.weight", prefix), &format!("{}.mlp.gate_proj", prefix))?,
        up_proj: get_weight_or_int4(registry, &format!("{}.mlp.up_proj.weight", prefix), &format!("{}.mlp.up_proj", prefix))?,
        down_proj: get_weight_or_int4(registry, &format!("{}.mlp.down_proj.weight", prefix), &format!("{}.mlp.down_proj", prefix))?,
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
    registry: &mut WeightRegistry,
    config: &ModelConfig,
    layer_idx: usize,
) -> Result<LayerWeights> {
    let prefix = format!("mtp.layers.{}", layer_idx);

    let norm1 = get_weight(registry, &format!("{}.input_layernorm.weight", prefix))?;
    let norm2 = get_weight(registry, &format!("{}.post_attention_layernorm.weight", prefix))?;

    // MTP layers use full_attention by default (detect from available tensors)
    let layer_type = if registry.tensors.contains_key(&format!("{}.linear_attn.in_proj_a.weight", prefix)) {
        super::config::LayerType::GatedDeltaNet
    } else {
        super::config::LayerType::FullAttention
    };

    let (gdn, attn) = match layer_type {
        super::config::LayerType::GatedDeltaNet => {
            let p = &prefix;
            let sub = "linear_attn";  // MTP GDN layers always use linear_attn prefix
            let in_proj_a = get_weight_or_int4(registry, &format!("{p}.{sub}.in_proj_a.weight"), &format!("{p}.{sub}.in_proj_a"))?;
            let in_proj_b = get_weight_or_int4(registry, &format!("{p}.{sub}.in_proj_b.weight"), &format!("{p}.{sub}.in_proj_b"))?;
            let conv1d_weight = get_weight(registry, &format!("{p}.{sub}.conv1d.weight"))?;
            // x_proj_weight and dt_proj_weight are optional — not present in Qwen3.6
            let x_proj_weight = get_weight_or_int4_optional(registry, &format!("{p}.{sub}.x_proj_weight.weight"), &format!("{p}.{sub}.x_proj_weight"))?;
            let dt_proj_weight = get_weight_or_int4_optional(registry, &format!("{p}.{sub}.dt_proj_weight.weight"), &format!("{p}.{sub}.dt_proj_weight"))?;
            // out_proj (not out_proj_weight) — matches real Qwen3.6 tensor names
            let out_proj_weight = get_weight_or_int4(registry, &format!("{p}.{sub}.out_proj.weight"), &format!("{p}.{sub}.out_proj"))?;
            // Optional Mamba2-style weights
            let a_log = registry.tensors.remove(&format!("{p}.{sub}.A_log"));
            let dt_bias = registry.tensors.remove(&format!("{p}.{sub}.dt_bias"));
            let norm = registry.tensors.remove(&format!("{p}.{sub}.norm.weight"));
            let in_proj_qkv = get_weight_or_int4_optional(registry, &format!("{p}.{sub}.in_proj_qkv.weight"), &format!("{p}.{sub}.in_proj_qkv"))?;
            let in_proj_z = get_weight_or_int4_optional(registry, &format!("{p}.{sub}.in_proj_z.weight"), &format!("{p}.{sub}.in_proj_z"))?;
            let gdn = GdnWeights {
                in_proj_a,
                in_proj_b,
                conv1d_weight,
                x_proj_weight,
                dt_proj_weight,
                out_proj_weight,
                in_proj_qkv,
                in_proj_z,
                a_log,
                dt_bias,
                norm,
            };
            (Some(gdn), None)
        }
        super::config::LayerType::FullAttention => {
            let p = &prefix;
            let q_proj = get_weight_or_int4(registry, &format!("{p}.self_attn.q_proj.weight"), &format!("{p}.self_attn.q_proj"))?;
            let k_proj = get_weight_or_int4(registry, &format!("{p}.self_attn.k_proj.weight"), &format!("{p}.self_attn.k_proj"))?;
            let v_proj = get_weight_or_int4(registry, &format!("{p}.self_attn.v_proj.weight"), &format!("{p}.self_attn.v_proj"))?;
            let o_proj = get_weight_or_int4(registry, &format!("{p}.self_attn.o_proj.weight"), &format!("{p}.self_attn.o_proj"))?;
            // Optional Q/K norm weights
            let q_norm = registry.tensors.remove(&format!("{p}.self_attn.q_norm.weight"));
            let k_norm = registry.tensors.remove(&format!("{p}.self_attn.k_norm.weight"));
            let attn = AttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
            };
            (None, Some(attn))
        }
    };

    let mlp = MlpWeights {
        gate_proj: get_weight_or_int4(registry, &format!("{}.mlp.gate_proj.weight", prefix), &format!("{}.mlp.gate_proj", prefix))?,
        up_proj: get_weight_or_int4(registry, &format!("{}.mlp.up_proj.weight", prefix), &format!("{}.mlp.up_proj", prefix))?,
        down_proj: get_weight_or_int4(registry, &format!("{}.mlp.down_proj.weight", prefix), &format!("{}.mlp.down_proj", prefix))?,
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

/// Get a weight tensor from the registry by name. Removes the tensor from the flat map.
/// Returns ownership of the weight data to halve memory usage during model loading.
fn get_weight(registry: &mut WeightRegistry, name: &str) -> Result<WeightData> {
    registry
        .tensors
        .remove(name)
        .with_context(|| format!("tensor not found: {}", name))
}

/// Get a weight tensor, falling back to INT4 quantized triplet.
///
/// First tries `{name}` (BF16/FP16/FP32 weight). If not found, tries
/// `{name_base}.qweight` (INT4 packed). When INT4 is found, also extracts
/// `qzeros` and `scales` companions and stores them in `registry.int4_companions`.
///
/// # Arguments
/// * `registry` — Weight registry (mutated: tensors removed, companions added)
/// * `bf16_name` — Full tensor name for BF16 weight (e.g., `layers.0.mlp.gate_proj.weight`)
/// * `int4_base` — Base name for INT4 triplet (e.g., `layers.0.mlp.gate_proj`)
fn get_weight_or_int4(
    registry: &mut WeightRegistry,
    bf16_name: &str,
    int4_base: &str,
) -> Result<WeightData> {
    // Try BF16 weight first
    if let Some(w) = registry.tensors.remove(bf16_name) {
        return Ok(w);
    }

    // Fall back to INT4 qweight
    let qweight_name = format!("{}.qweight", int4_base);
    let qweight = registry
        .tensors
        .remove(&qweight_name)
        .with_context(|| format!("neither '{}' nor '{}' found", bf16_name, qweight_name))?;

   // Extract companion tensors — check int4_companions first (populated by sharding),
    // then fall back to registry.tensors (non-sharded path).
    let qweight_name_ref = &qweight_name;
    if registry.int4_companions.contains_key(qweight_name_ref) {
        // Sharded path: companions already populated by shard_weights_tp
        return Ok(qweight);
    }

    // Non-sharded path: extract from tensors and store in int4_companions
    let qzeros_name = format!("{}.qzeros", int4_base);
    let scales_name = format!("{}.scales", int4_base);

    let qzeros = registry
        .tensors
        .remove(&qzeros_name)
        .with_context(|| format!("INT4 qzeros '{}' not found (qweight '{}')", qzeros_name, qweight_name))?;
    let scales = registry
        .tensors
        .remove(&scales_name)
        .with_context(|| format!("INT4 scales '{}' not found (qweight '{}')", scales_name, qweight_name))?;

    // Store companions keyed by qweight name
    registry.int4_companions.insert(qweight_name.clone(), Int4Companions { qzeros, scales });

    Ok(qweight)
}

/// Optional version of `get_weight_or_int4` that returns `Ok(None)` when
/// neither BF16 nor INT4 weights are found, instead of erroring.
/// INT4 companions are stored in `registry.int4_companions` when INT4 is used.
fn get_weight_or_int4_optional(
    registry: &mut WeightRegistry,
    bf16_name: &str,
    int4_base: &str,
) -> Result<Option<WeightData>> {
    // Try BF16 weight first
    if let Some(w) = registry.tensors.remove(bf16_name) {
        return Ok(Some(w));
    }

  // Fall back to INT4 qweight (optional — return None if missing)
    let qweight_name = format!("{}.qweight", int4_base);
    let qweight = match registry.tensors.remove(&qweight_name) {
        Some(w) => w,
        None => return Ok(None),
    };

    // Extract companion tensors — check int4_companions first (populated by sharding),
    // then fall back to registry.tensors (non-sharded path).
    if registry.int4_companions.contains_key(&qweight_name) {
        // Sharded path: companions already populated by shard_weights_tp
        return Ok(Some(qweight));
    }

    // Non-sharded path: extract from tensors and store in int4_companions
    let qzeros_name = format!("{}.qzeros", int4_base);
    let scales_name = format!("{}.scales", int4_base);
    let qzeros = registry.tensors.remove(&qzeros_name)
        .with_context(|| format!("INT4 qzeros '{}' not found (qweight '{}')", qzeros_name, qweight_name))?;
    let scales = registry.tensors.remove(&scales_name)
        .with_context(|| format!("INT4 scales '{}' not found (qweight '{}')", scales_name, qweight_name))?;

    // Store companions keyed by qweight name
    registry.int4_companions.insert(qweight_name.clone(), Int4Companions { qzeros, scales });

    Ok(Some(qweight))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LayerType;
    use crate::weights::{WeightData, WeightRegistry, WeightDtype, MtpWeights};
    use bytes::Bytes;

    fn dummy_weight(name: &str) -> WeightData {
        WeightData { data: Bytes::from(vec![0u8; 32]), shape: vec![2, 16], dtype: WeightDtype::Bf16, name: name.to_string() }
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
            data: Bytes::from(vec![0u8; 32]),
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
    fn mtp_weights_struct_has_expected_fields() {
        // Verify MtpWeights has all expected fields by constructing one
        let dummy = WeightData {
            data: Bytes::from(vec![0u8; 32]),
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

    #[test]
    fn strip_language_model_prefix_removes_prefix() {
        let mut registry = WeightRegistry::new();
        registry.tensors.insert(
            "model.language_model.layers.0.input_layernorm.weight".to_string(),
            WeightData { data: Bytes::from(vec![1; 8]), shape: vec![4, 2], dtype: WeightDtype::Bf16, name: String::new() },
        );
        registry.tensors.insert(
            "model.language_model.norm.weight".to_string(),
            WeightData { data: Bytes::from(vec![2; 8]), shape: vec![4, 2], dtype: WeightDtype::Bf16, name: String::new() },
        );
        strip_language_model_prefix(&mut registry);
        assert!(registry.tensors.contains_key("layers.0.input_layernorm.weight"));
        assert!(registry.tensors.contains_key("norm.weight"));
        assert!(!registry.tensors.contains_key("model.language_model.layers.0.input_layernorm.weight"));
        assert!(!registry.tensors.contains_key("model.language_model.norm.weight"));
    }

    #[test]
    fn strip_language_model_prefix_filters_visual() {
        let mut registry = WeightRegistry::new();
        registry.tensors.insert(
            "model.visual.patch_embed.proj.weight".to_string(),
            WeightData { data: Bytes::from(vec![3; 8]), shape: vec![4, 2], dtype: WeightDtype::Bf16, name: String::new() },
        );
        registry.tensors.insert(
            "layers.0.input_layernorm.weight".to_string(),
            WeightData { data: Bytes::from(vec![4; 8]), shape: vec![4, 2], dtype: WeightDtype::Bf16, name: String::new() },
        );
        strip_language_model_prefix(&mut registry);
        assert!(!registry.tensors.contains_key("model.visual.patch_embed.proj.weight"));
        assert!(registry.tensors.contains_key("layers.0.input_layernorm.weight"));
        assert_eq!(registry.tensors.len(), 1);
    }

    #[test]
    fn strip_language_model_prefix_keeps_mtp() {
        let mut registry = WeightRegistry::new();
        registry.tensors.insert(
            "mtp.layers.0.self_attn.q_proj.weight".to_string(),
            WeightData { data: Bytes::from(vec![5; 8]), shape: vec![4, 2], dtype: WeightDtype::Bf16, name: String::new() },
        );
        strip_language_model_prefix(&mut registry);
        assert!(registry.tensors.contains_key("mtp.layers.0.self_attn.q_proj.weight"));
        assert_eq!(registry.tensors.len(), 1);
    }

    #[test]
    fn build_main_layers_basic() {
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":4,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();
        let mut registry = WeightRegistry::new();
        registry.tensors.insert("embed_tokens.weight".to_string(), dummy_weight("embed_tokens.weight"));
        registry.tensors.insert("norm.weight".to_string(), dummy_weight("norm.weight"));
        registry.tensors.insert("lm_head.weight".to_string(), dummy_weight("lm_head.weight"));
        for i in 0..4 {
            let prefix = format!("layers.{}", i);
            registry.tensors.insert(format!("{}.input_layernorm.weight", prefix), dummy_weight(&format!("{}.input_layernorm.weight", prefix)));
            registry.tensors.insert(format!("{}.post_attention_layernorm.weight", prefix), dummy_weight(&format!("{}.post_attention_layernorm.weight", prefix)));
            if i < 3 {
                for sub in ["gdn.in_proj_a.weight", "gdn.in_proj_b.weight", "gdn.conv1d.weight", "gdn.out_proj.weight"] {
                    registry.tensors.insert(format!("{}.{}", prefix, sub), dummy_weight(&format!("{}.{}", prefix, sub)));
                }
            } else {
                for sub in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"] {
                    registry.tensors.insert(format!("{}.{}", prefix, sub), dummy_weight(&format!("{}.{}", prefix, sub)));
                }
            }
            for sub in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"] {
                registry.tensors.insert(format!("{}.{}", prefix, sub), dummy_weight(&format!("{}.{}", prefix, sub)));
            }
        }
        let result = build_main_layers(&mut registry, &config);
        assert!(result.is_ok());
        assert_eq!(registry.layers.len(), 4);
        for i in 0..3 {
            assert_eq!(registry.layers[i].layer_type, LayerType::GatedDeltaNet);
            assert!(registry.layers[i].gdn.is_some(), "layer {} should have GDN", i);
            assert!(registry.layers[i].attn.is_none(), "layer {} should not have attention", i);
        }
        assert_eq!(registry.layers[3].layer_type, LayerType::FullAttention);
        assert!(registry.layers[3].attn.is_some());
        assert!(registry.layers[3].gdn.is_none());
        assert!(registry.embedding.is_some());
        assert!(registry.norm.is_some());
        assert!(registry.lm_head.is_some());
        assert_eq!(registry.embedding.as_ref().unwrap().name, "embed_tokens.weight");
        assert_eq!(registry.norm.as_ref().unwrap().name, "norm.weight");
        assert_eq!(registry.lm_head.as_ref().unwrap().name, "lm_head.weight");
    }

    #[test]
    fn build_main_layers_with_linear_attn() {
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":1,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"layer_types":["linear_attention"]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();
        let mut registry = WeightRegistry::new();
        registry.tensors.insert("embed_tokens.weight".to_string(), dummy_weight("embed_tokens.weight"));
        registry.tensors.insert("norm.weight".to_string(), dummy_weight("norm.weight"));
        registry.tensors.insert("lm_head.weight".to_string(), dummy_weight("lm_head.weight"));
        registry.tensors.insert("layers.0.input_layernorm.weight".to_string(), dummy_weight("layers.0.input_layernorm.weight"));
        registry.tensors.insert("layers.0.post_attention_layernorm.weight".to_string(), dummy_weight("layers.0.post_attention_layernorm.weight"));
        for sub in ["linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight", "linear_attn.conv1d.weight", "linear_attn.out_proj.weight"] {
            registry.tensors.insert(format!("layers.0.{}", sub), dummy_weight(&format!("layers.0.{}", sub)));
        }
        for sub in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"] {
            registry.tensors.insert(format!("layers.0.{}", sub), dummy_weight(&format!("layers.0.{}", sub)));
        }
        let result = build_main_layers(&mut registry, &config);
        assert!(result.is_ok());
        assert!(registry.layers[0].gdn.is_some());
        let gdn = registry.layers[0].gdn.as_ref().unwrap();
        assert_eq!(gdn.in_proj_a.name, "layers.0.linear_attn.in_proj_a.weight");
        assert_eq!(gdn.out_proj_weight.name, "layers.0.linear_attn.out_proj.weight");
    }

    #[test]
    fn build_main_layers_extracts_scalar_weights() {
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":1,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"layer_types":["linear_attention"]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();
        let mut registry = WeightRegistry::new();
        registry.tensors.insert("embed_tokens.weight".to_string(), dummy_weight("embed_tokens.weight"));
        registry.tensors.insert("norm.weight".to_string(), dummy_weight("norm.weight"));
        registry.tensors.insert("lm_head.weight".to_string(), dummy_weight("lm_head.weight"));
        registry.tensors.insert("layers.0.input_layernorm.weight".to_string(), dummy_weight("layers.0.input_layernorm.weight"));
        registry.tensors.insert("layers.0.post_attention_layernorm.weight".to_string(), dummy_weight("layers.0.post_attention_layernorm.weight"));
        for sub in ["gdn.in_proj_a.weight", "gdn.in_proj_b.weight", "gdn.conv1d.weight", "gdn.out_proj.weight"] {
            registry.tensors.insert(format!("layers.0.{}", sub), dummy_weight(&format!("layers.0.{}", sub)));
        }
        for sub in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"] {
            registry.tensors.insert(format!("layers.0.{}", sub), dummy_weight(&format!("layers.0.{}", sub)));
        }
        let result = build_main_layers(&mut registry, &config);
        assert!(result.is_ok());
        assert!(registry.embedding.is_some());
        assert!(registry.norm.is_some());
        assert!(registry.lm_head.is_some());
        assert_eq!(registry.embedding.as_ref().unwrap().name, "embed_tokens.weight");
        assert_eq!(registry.norm.as_ref().unwrap().name, "norm.weight");
        assert_eq!(registry.lm_head.as_ref().unwrap().name, "lm_head.weight");
        // Tensors removed from flat map (not cloned)
        assert!(!registry.tensors.contains_key("embed_tokens.weight"));
        assert!(!registry.tensors.contains_key("norm.weight"));
        assert!(!registry.tensors.contains_key("lm_head.weight"));
    }

    #[test]
    fn load_model_populates_layers() {
        // Integration test: strip + build_main_layers correctly populates the registry.
        // We test the pipeline without needing actual safetensors files.
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":2,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"layer_types":["linear_attention","full_attention"]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();
        let mut registry = WeightRegistry::new();
        // Simulate what happens after safetensors loading: tensors have model.language_model. prefix
        registry.tensors.insert("model.language_model.layers.0.input_layernorm.weight".to_string(), dummy_weight("model.language_model.layers.0.input_layernorm.weight"));
        registry.tensors.insert("model.language_model.layers.0.post_attention_layernorm.weight".to_string(), dummy_weight("model.language_model.layers.0.post_attention_layernorm.weight"));
        registry.tensors.insert("model.language_model.layers.0.gdn.in_proj_a.weight".to_string(), dummy_weight("model.language_model.layers.0.gdn.in_proj_a.weight"));
        registry.tensors.insert("model.language_model.layers.0.gdn.in_proj_b.weight".to_string(), dummy_weight("model.language_model.layers.0.gdn.in_proj_b.weight"));
        registry.tensors.insert("model.language_model.layers.0.gdn.conv1d.weight".to_string(), dummy_weight("model.language_model.layers.0.gdn.conv1d.weight"));
        registry.tensors.insert("model.language_model.layers.0.gdn.out_proj.weight".to_string(), dummy_weight("model.language_model.layers.0.gdn.out_proj.weight"));
        registry.tensors.insert("model.language_model.layers.0.mlp.gate_proj.weight".to_string(), dummy_weight("model.language_model.layers.0.mlp.gate_proj.weight"));
        registry.tensors.insert("model.language_model.layers.0.mlp.up_proj.weight".to_string(), dummy_weight("model.language_model.layers.0.mlp.up_proj.weight"));
        registry.tensors.insert("model.language_model.layers.0.mlp.down_proj.weight".to_string(), dummy_weight("model.language_model.layers.0.mlp.down_proj.weight"));
        registry.tensors.insert("model.language_model.layers.1.input_layernorm.weight".to_string(), dummy_weight("model.language_model.layers.1.input_layernorm.weight"));
        registry.tensors.insert("model.language_model.layers.1.post_attention_layernorm.weight".to_string(), dummy_weight("model.language_model.layers.1.post_attention_layernorm.weight"));
        registry.tensors.insert("model.language_model.layers.1.self_attn.q_proj.weight".to_string(), dummy_weight("model.language_model.layers.1.self_attn.q_proj.weight"));
        registry.tensors.insert("model.language_model.layers.1.self_attn.k_proj.weight".to_string(), dummy_weight("model.language_model.layers.1.self_attn.k_proj.weight"));
        registry.tensors.insert("model.language_model.layers.1.self_attn.v_proj.weight".to_string(), dummy_weight("model.language_model.layers.1.self_attn.v_proj.weight"));
        registry.tensors.insert("model.language_model.layers.1.self_attn.o_proj.weight".to_string(), dummy_weight("model.language_model.layers.1.self_attn.o_proj.weight"));
        registry.tensors.insert("model.language_model.layers.1.mlp.gate_proj.weight".to_string(), dummy_weight("model.language_model.layers.1.mlp.gate_proj.weight"));
        registry.tensors.insert("model.language_model.layers.1.mlp.up_proj.weight".to_string(), dummy_weight("model.language_model.layers.1.mlp.up_proj.weight"));
        registry.tensors.insert("model.language_model.layers.1.mlp.down_proj.weight".to_string(), dummy_weight("model.language_model.layers.1.mlp.down_proj.weight"));
        registry.tensors.insert("model.language_model.embed_tokens.weight".to_string(), dummy_weight("model.language_model.embed_tokens.weight"));
        registry.tensors.insert("model.language_model.norm.weight".to_string(), dummy_weight("model.language_model.norm.weight"));
        registry.tensors.insert("model.language_model.lm_head.weight".to_string(), dummy_weight("model.language_model.lm_head.weight"));
        // Visual tensor that should be removed
        registry.tensors.insert("model.visual.patch_embed.proj.weight".to_string(), dummy_weight("model.visual.patch_embed.proj.weight"));
        // Simulate strip_language_model_prefix then build_main_layers
        strip_language_model_prefix(&mut registry);
        let result = build_main_layers(&mut registry, &config);
        assert!(result.is_ok(), "build_main_layers should succeed");
        assert_eq!(registry.layers.len(), 2);
        assert!(registry.embedding.is_some());
        assert!(registry.norm.is_some());
        assert!(registry.lm_head.is_some());
        assert!(registry.layers[0].gdn.is_some());
        assert!(registry.layers[1].attn.is_some());
        // Visual tensor should be gone
        assert!(!registry.tensors.contains_key("model.visual.patch_embed.proj.weight"));
        // Old prefixed keys should be gone
        assert!(!registry.tensors.contains_key("model.language_model.layers.0.input_layernorm.weight"));
    }
}
