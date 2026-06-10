//! Weight sharding for tensor parallelism (TP=2) and pipeline parallelism (PP=2).
//!
//! TP=2: column-parallel for Q/K/V/gate/up projections, row-parallel for O/down.
//! PP=2: split layers into two stages (0-31, 32-63).

use anyhow::{Context, Result};

use super::config::ModelConfig;
use super::weights::{WeightData, WeightShard};

/// Shard model weights across `num_gpus` devices for tensor parallelism.
///
/// Column-parallel: Q, K, V, gate, up projections are split along the output dimension.
/// Row-parallel: O, down projections are replicated on each GPU.
// @lat: [[lat#Weight Sharding#Tensor Parallelism Sharding]]
pub fn shard_weights_tp(
    registry: &super::weights::WeightRegistry,
    _config: &ModelConfig,
    num_gpus: usize,
) -> Result<Vec<WeightShard>> {
    anyhow::ensure!(num_gpus >= 1, "num_gpus must be >= 1");

    if num_gpus == 1 {
        // No sharding needed for single GPU
        return Ok(vec![WeightShard {
            gpu_id: 0,
            registry: registry.clone(),
        }]);
    }

    // For TP=2+, shard each layer's weights
    let mut shards: Vec<WeightShard> = (0..num_gpus)
        .map(|gpu_id| WeightShard {
            gpu_id,
            registry: super::weights::WeightRegistry::new(),
        })
        .collect();

    // For each tensor, determine if it should be column-parallel or row-parallel
    // and slice accordingly.
    for (name, weight) in &registry.tensors {
        let shard_type = determine_shard_type(name);

        match shard_type {
            ShardType::ColumnParallel => {
                // Split along dimension 0
                for (gpu_id, shard) in shards.iter_mut().enumerate() {
                    let sliced = slice_weight_dim0(weight, gpu_id, num_gpus)
                        .context(format!("Failed to shard column-parallel weight: {}", name))?;
                    shard.registry.tensors.insert(name.clone(), sliced);
                }
            }
            ShardType::RowParallel => {
                // Split along last dimension
                for (gpu_id, shard) in shards.iter_mut().enumerate() {
                    let sliced = slice_weight_last_dim(weight, gpu_id, num_gpus)
                        .context(format!("Failed to shard row-parallel weight: {}", name))?;
                    shard.registry.tensors.insert(name.clone(), sliced);
                }
            }
            ShardType::Replicated => {
                // Replicate on all GPUs
                for shard in shards.iter_mut() {
                    shard.registry.tensors.insert(name.clone(), weight.clone());
                }
            }
        }
    }

    Ok(shards)
}

/// How a weight tensor is distributed across GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardType {
    /// Split along dimension 0 (output dim for Q/K/V/gate/up).
    ColumnParallel,
    /// Split along last dimension (input dim for O/down).
    RowParallel,
    /// Replicated on all GPUs (norms, embeddings with TP).
    Replicated,
}

/// Determine sharding type for a weight tensor by its name.
// @lat: [[lat#Weight Sharding#Shard Type Detection]]
fn determine_shard_type(name: &str) -> ShardType {
    // Column-parallel: projections that produce per-head outputs
    if name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj") {
        return ShardType::ColumnParallel;
    }
    if name.contains("gate_proj") || name.contains("up_proj") {
        return ShardType::ColumnParallel;
    }
    // GDN projections
    if name.contains("in_proj_a") || name.contains("in_proj_b") {
        return ShardType::ColumnParallel;
    }

    // Row-parallel: projections that reduce across heads
    if name.contains("o_proj") || name.contains("down_proj") {
        return ShardType::RowParallel;
    }
    if name.contains("out_proj") {
        return ShardType::RowParallel;
    }

    // Everything else is replicated (norms, embeddings, lm_head)
    ShardType::Replicated
}

/// Slice a weight tensor along dimension 0 for a specific GPU.
fn slice_weight_dim0(weight: &WeightData, gpu_id: usize, num_gpus: usize) -> Result<WeightData> {
    anyhow::ensure!(
        !weight.shape.is_empty(),
        "Weight {} has no dimensions",
        weight.name
    );

    let dim0 = weight.shape[0];
    let shard_size = dim0 / num_gpus;
    let start = gpu_id * shard_size;
    let end = start + shard_size;

    let bytes_per_row = weight.data.len() / dim0;
    let start_byte = start * bytes_per_row;
    let end_byte = end * bytes_per_row;

    let mut new_shape = weight.shape.clone();
    new_shape[0] = shard_size;

    Ok(WeightData {
        data: weight.data[start_byte..end_byte].to_vec(),
        shape: new_shape,
        dtype: weight.dtype,
        name: weight.name.clone(),
    })
}

/// Slice a weight tensor along the last dimension for a specific GPU.
fn slice_weight_last_dim(weight: &WeightData, gpu_id: usize, num_gpus: usize) -> Result<WeightData> {
    let last_dim = *weight.shape.last().with_context(|| format!("Weight {} has no dimensions", weight.name))?;
    let shard_size = last_dim / num_gpus;
    let start = gpu_id * shard_size;

    // For row-parallel on last dim, we need to handle the stride
    // Each row of the matrix has `last_dim` elements
    let num_rows: usize = weight.shape.iter().take(weight.shape.len() - 1).product();
    let bytes_per_element = weight.data.len() / weight.shape.iter().product::<usize>();

    // For now, if the weight is 2D we can slice efficiently
    // Higher-dimensional tensors need more careful handling
    anyhow::ensure!(
        weight.shape.len() == 2,
        "Row-parallel slicing only supports 2D weights, got {}D",
        weight.shape.len()
    );

    let cols = last_dim;
    let shard_cols = shard_size;
    let mut new_data = Vec::with_capacity(num_rows * shard_cols * bytes_per_element);

    for row in 0..num_rows {
        let row_start = row * cols * bytes_per_element + start * bytes_per_element;
        let row_end = row_start + shard_cols * bytes_per_element;
        new_data.extend_from_slice(&weight.data[row_start..row_end]);
    }

    let mut new_shape = weight.shape.clone();
    *new_shape.last_mut().unwrap() = shard_size;

    Ok(WeightData {
        data: new_data,
        shape: new_shape,
        dtype: weight.dtype,
        name: weight.name.clone(),
    })
}

/// Split model layers across pipeline stages for PP=2.
// @lat: [[lat#Weight Sharding#Pipeline Parallelism Split]]
///
/// Stage 0: layers 0 to (num_layers / 2 - 1)
/// Stage 1: layers (num_layers / 2) to (num_layers - 1)
pub fn split_layers_pp(config: &ModelConfig, num_stages: usize) -> Vec<std::ops::Range<usize>> {
    let layers_per_stage = config.num_hidden_layers / num_stages;
    (0..num_stages)
        .map(|stage| {
            let start = stage * layers_per_stage;
            let end = start + layers_per_stage;
            start..end
        })
        .collect()
}

/// Filter a WeightRegistry to only contain weights for a specific stage's layers.
///
/// For PP=2, stage 0 gets layers 0-31 and stage 1 gets layers 32-63.
/// Shared weights (embedding, norm, lm_head, mtp) are kept in both stages.
/// Layer-specific tensors not in the given range are removed from the
/// `tensors` HashMap, and the `layers` Vec is filtered to only contain
/// layers in the range.
// @lat: [[lat#Weight Sharding#Pipeline Parallelism Split]]
pub fn shard_weights_for_stage(
    registry: &super::weights::WeightRegistry,
    layer_range: &std::ops::Range<usize>,
) -> super::weights::WeightRegistry {
    let mut shard = registry.clone();

    // Filter layer-specific weights to only this stage's range
    shard.layers.retain(|layer| layer_range.contains(&layer.layer_idx));

    // Keep only tensors whose layer index is in range, plus global tensors
    // Global tensors are those that don't contain "layers." in their name
    shard.tensors.retain(|name, _| {
        // Check if this is a layer-specific tensor
        if let Some(layer_str) = name.strip_prefix("model.layers.") {
            // Extract the layer index from e.g. "0.self_attn.q_proj.weight"
            if let Some(idx_str) = layer_str.split('.').next()
                && let Ok(idx) = idx_str.parse::<usize>()
            {
                return layer_range.contains(&idx);
            }
        }
        // Global tensors (embedding, norm, lm_head, etc.) are always kept
        true
    });

    shard
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{WeightDtype, WeightRegistry};

    #[test]
    fn test_determine_shard_type() {
        assert_eq!(
            determine_shard_type("model.layers.0.self_attn.q_proj.weight"),
            ShardType::ColumnParallel
        );
        assert_eq!(
            determine_shard_type("model.layers.0.self_attn.k_proj.weight"),
            ShardType::ColumnParallel
        );
        assert_eq!(
            determine_shard_type("model.layers.0.self_attn.v_proj.weight"),
            ShardType::ColumnParallel
        );
        assert_eq!(
            determine_shard_type("model.layers.0.self_attn.o_proj.weight"),
            ShardType::RowParallel
        );
        assert_eq!(
            determine_shard_type("model.layers.0.mlp.gate_proj.weight"),
            ShardType::ColumnParallel
        );
        assert_eq!(
            determine_shard_type("model.layers.0.mlp.up_proj.weight"),
            ShardType::ColumnParallel
        );
        assert_eq!(
            determine_shard_type("model.layers.0.mlp.down_proj.weight"),
            ShardType::RowParallel
        );
        assert_eq!(
            determine_shard_type("model.layers.0.input_layernorm.weight"),
            ShardType::Replicated
        );
        assert_eq!(
            determine_shard_type("model.embed_tokens.weight"),
            ShardType::Replicated
        );
        assert_eq!(determine_shard_type("lm_head.weight"), ShardType::Replicated);
    }

    #[test]
    fn test_slice_weight_dim0() {
        // 4x8 matrix, 2 elements per row (BF16 = 2 bytes)
        let weight = WeightData {
            data: vec![0u8; 64], // 4 rows * 8 cols * 2 bytes = 64 bytes
            shape: vec![4, 8],
            dtype: WeightDtype::Bf16,
            name: "test.weight".to_string(),
        };

        // Shard 0 of 2: rows 0-1
        let shard0 = slice_weight_dim0(&weight, 0, 2).unwrap();
        assert_eq!(shard0.shape, vec![2, 8]);
        assert_eq!(
            shard0.data.len(),
            32 // 2 rows * 8 cols * 2 bytes
        );

        // Shard 1 of 2: rows 2-3
        let shard1 = slice_weight_dim0(&weight, 1, 2).unwrap();
        assert_eq!(shard1.shape, vec![2, 8]);
        assert_eq!(shard1.data.len(), 32);
    }

    #[test]
    fn test_slice_weight_last_dim() {
        // 4x8 matrix, 2 bytes per element
        let weight = WeightData {
            data: vec![0u8; 64], // 4 rows * 8 cols * 2 bytes
            shape: vec![4, 8],
            dtype: WeightDtype::Bf16,
            name: "test.weight".to_string(),
        };

        // Shard 0 of 2: columns 0-3
        let shard0 = slice_weight_last_dim(&weight, 0, 2).unwrap();
        assert_eq!(shard0.shape, vec![4, 4]);
        assert_eq!(
            shard0.data.len(),
            32 // 4 rows * 4 cols * 2 bytes
        );
    }

    #[test]
    fn test_pp_stage_split() {
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let stages = split_layers_pp(&config, 2);
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0], 0..32);
        assert_eq!(stages[1], 32..64);
    }

    #[test]
    fn test_shard_weights_tp_single_gpu() {
        let mut registry = WeightRegistry::new();
        registry.tensors.insert(
            "test.weight".to_string(),
            WeightData {
                data: vec![1u8; 100],
                shape: vec![10, 5],
                dtype: WeightDtype::Bf16,
                name: "test.weight".to_string(),
            },
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 1).unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].gpu_id, 0);
        assert_eq!(shards[0].registry.num_tensors(), 1);
    }

    #[test]
    fn test_shard_weights_tp_rejected_zero_gpus() {
        let registry = WeightRegistry::new();
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();
        let result = shard_weights_tp(&registry, &config, 0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("num_gpus must be >= 1"));
    }
}
