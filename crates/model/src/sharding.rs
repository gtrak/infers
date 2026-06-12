//! Weight sharding for tensor parallelism (TP=2) and pipeline parallelism (PP=2).
//!
//! TP=2: column-parallel for Q/K/V/gate/up projections, row-parallel for O/down.
//! PP=2: split layers into two stages (0-31, 32-63).

use anyhow::{Context, Result};
use bytes::Bytes;
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
    // and slice accordingly. INT4 weights (GPTQ/AutoRound) store qweights as
    // (K/8, N) — swapped relative to BF16's [N, K] — so the split dimension
    // differs between formats.
    for (name, weight) in &registry.tensors {
        let shard_type = determine_shard_type(name);
        let is_int4 = weight.dtype == super::weights::WeightDtype::Int4Packed;

        match shard_type {
            ShardType::ColumnParallel => {
                // Column-parallel: split along the output dimension (N).
                // BF16 [N, K]: N on dim 0 → slice_weight_dim0
                // INT4 (K/8, N): N on dim 1 → slice_weight_last_dim
                for (gpu_id, shard) in shards.iter_mut().enumerate() {
                    let sliced = if is_int4 {
                        slice_weight_last_dim(weight, gpu_id, num_gpus)
                            .context(format!("Failed to shard INT4 column-parallel weight: {}", name))?
                    } else {
                        slice_weight_dim0(weight, gpu_id, num_gpus)
                            .context(format!("Failed to shard column-parallel weight: {}", name))?
                    };
                    shard.registry.tensors.insert(name.clone(), sliced);

                    // Shard INT4 companion weights — scales [N, groups] and qzeros split along dim 0 (N)
                    if is_int4 {
                        if let Some(companions) = registry.int4_companions.get(name) {
                            let scaled_scales = slice_weight_dim0(&companions.scales, gpu_id, num_gpus)
                                .with_context(|| format!("Failed to shard INT4 scales: {}", name))?;
                            let sliced_qzeros = slice_weight_dim0(&companions.qzeros, gpu_id, num_gpus)
                                .with_context(|| format!("Failed to shard INT4 qzeros: {}", name))?;
                            shard.registry.int4_companions.insert(
                                name.clone(),
                                super::weights::Int4Companions {
                                    scales: scaled_scales,
                                    qzeros: sliced_qzeros,
                                },
                            );
                        }
                    }
                }
            }
            ShardType::RowParallel => {
                // Row-parallel: split along the input dimension (K).
                // BF16 [N, K]: K on dim 1 → slice_weight_last_dim
                // INT4 (K/8, N): K/8 on dim 0 → slice_weight_dim0
                for (gpu_id, shard) in shards.iter_mut().enumerate() {
                    let sliced = if is_int4 {
                        slice_weight_dim0(weight, gpu_id, num_gpus)
                            .context(format!("Failed to shard INT4 row-parallel weight: {}", name))?
                    } else {
                        slice_weight_last_dim(weight, gpu_id, num_gpus)
                            .context(format!("Failed to shard row-parallel weight: {}", name))?
                    };
                    shard.registry.tensors.insert(name.clone(), sliced);

                    // Shard INT4 companion weights — scales [N, groups] split along last dim (groups = K/group_size)
                    if is_int4 {
                        if let Some(companions) = registry.int4_companions.get(name) {
                            let sliced_scales = slice_weight_last_dim(&companions.scales, gpu_id, num_gpus)
                                .with_context(|| format!("Failed to shard INT4 scales: {}", name))?;
                            let sliced_qzeros = slice_weight_last_dim(&companions.qzeros, gpu_id, num_gpus)
                                .with_context(|| format!("Failed to shard INT4 qzeros: {}", name))?;
                            shard.registry.int4_companions.insert(
                                name.clone(),
                                super::weights::Int4Companions {
                                    scales: sliced_scales,
                                    qzeros: sliced_qzeros,
                                },
                            );
                        }
                    }
                }
            }
            ShardType::Replicated => {
                // Replicate on all GPUs
                for shard in shards.iter_mut() {
                    shard.registry.tensors.insert(name.clone(), weight.clone());

                    // Replicate INT4 companion weights as well
                    if is_int4 {
                        if let Some(companions) = registry.int4_companions.get(name) {
                            shard.registry.int4_companions.insert(
                                name.clone(),
                                companions.clone(),
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(shards)
}

/// How a weight tensor is distributed across GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardType {
    /// Split along the output dimension (N). For BF16 \[N,K\] this is dim 0;
    /// for INT4 packed (K/8,N) this is dim 1.
    ColumnParallel,
    /// Split along the input dimension (K). For BF16 \[N,K\] this is dim 1;
    /// for INT4 packed (K/8,N) this is dim 0.
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
    // GDN projections — column-parallel (per-head output)
    if name.contains("in_proj_qkv") || name.contains("in_proj_z")
        || name.contains("in_proj_a") || name.contains("in_proj_b")
        || name.contains("conv1d.weight")
    {
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
        data: weight.data.slice(start_byte..end_byte),
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
        data: Bytes::from(new_data),
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
    use bytes::Bytes;
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

        // GDN QKV and Z projections — column-parallel
        assert_eq!(
            determine_shard_type("model.layers.3.gdn.in_proj_qkv.qweight"),
            ShardType::ColumnParallel
        );
        assert_eq!(
            determine_shard_type("model.layers.7.gdn.in_proj_z.qweight"),
            ShardType::ColumnParallel
        );
        // GDN conv1d weight — column-parallel (split along conv_dim)
        assert_eq!(
            determine_shard_type("model.layers.0.gdn.conv1d.weight"),
            ShardType::ColumnParallel
        );
    }

    #[test]
    fn test_slice_weight_dim0() {
        // 4x8 matrix, 2 elements per row (BF16 = 2 bytes)
        let weight = WeightData {
            data: Bytes::from(vec![0u8; 64]), // 4 rows * 8 cols * 2 bytes = 64 bytes
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
            data: Bytes::from(vec![0u8; 64]), // 4 rows * 8 cols * 2 bytes
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
                data: Bytes::from(vec![1u8; 100]),
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

    #[test]
    fn test_shard_weights_tp_int4_column_parallel() {
        // INT4 qweight shape (K/8, N) = (4, 8). Column-parallel should split dim 1 (N).
        // After splitting with 2 GPUs: GPU 0 gets (4, 4), GPU 1 gets (4, 4).
        let mut registry = WeightRegistry::new();
        let qweight = WeightData {
            data: Bytes::from(vec![0u8; 128]), // 4*8 u32 * 4 bytes = 128
            shape: vec![4, 8],
            dtype: WeightDtype::Int4Packed,
            name: "layers.0.self_attn.q_proj.qweight".to_string(),
        };
        registry.tensors.insert("layers.0.self_attn.q_proj.qweight".to_string(), qweight);

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        // Column-parallel: split dim 1 (N), so shape becomes [4, 4]
        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("layers.0.self_attn.q_proj.qweight").unwrap();
            assert_eq!(w.shape, vec![4, 4], "GPU {} INT4 column-parallel should have shape [4, 4]", gpu_id);
        }
    }

    #[test]
    fn test_shard_weights_tp_int4_row_parallel() {
        // INT4 qweight shape (K/8, N) = (4, 8). Row-parallel should split dim 0 (K/8).
        // After splitting with 2 GPUs: GPU 0 gets (2, 8), GPU 1 gets (2, 8).
        let mut registry = WeightRegistry::new();
        let qweight = WeightData {
            data: Bytes::from(vec![0u8; 128]), // 4*8 u32 * 4 bytes = 128
            shape: vec![4, 8],
            dtype: WeightDtype::Int4Packed,
            name: "layers.0.self_attn.o_proj.qweight".to_string(),
        };
        registry.tensors.insert("layers.0.self_attn.o_proj.qweight".to_string(), qweight);

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        // Row-parallel: split dim 0 (K/8), so shape becomes [2, 8]
        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("layers.0.self_attn.o_proj.qweight").unwrap();
            assert_eq!(w.shape, vec![2, 8], "GPU {} INT4 row-parallel should have shape [2, 8]", gpu_id);
        }
    }

    #[test]
    fn test_shard_weights_tp_int4_companions_column_parallel() {
        // Verify companion weights (scales, qzeros) are also sharded correctly.
        let mut registry = WeightRegistry::new();
        let qweight_name = "layers.0.mlp.gate_proj.qweight".to_string();
        registry.tensors.insert(
            qweight_name.clone(),
            WeightData {
                data: Bytes::from(vec![0u8; 128]), // shape [4, 8] * 4 bytes
                shape: vec![4, 8],
                dtype: WeightDtype::Int4Packed,
                name: qweight_name.clone(),
            },
        );

        // Scales: same shape as (K/group_size, N) — here [4, 8]
        registry.int4_companions.insert(
            qweight_name.clone(),
            crate::weights::Int4Companions {
                scales: WeightData {
                    data: Bytes::from(vec![0u8; 64]), // shape [4, 8] * 2 bytes (BF16)
                    shape: vec![4, 8],
                    dtype: WeightDtype::Bf16,
                    name: "layers.0.mlp.gate_proj.scales".to_string(),
                },
                qzeros: WeightData {
                    data: Bytes::from(vec![0u8; 128]), // shape [4, 8] * 4 bytes (u32)
                    shape: vec![4, 8],
                    dtype: WeightDtype::Int4Packed,
                    name: "layers.0.mlp.gate_proj.qzeros".to_string(),
                },
            },
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();

        for gpu_id in 0..2 {
            // qweight should be sharded along dim 1 (N) → [4, 4]
            let w = shards[gpu_id].registry.tensors.get(&qweight_name).unwrap();
            assert_eq!(w.shape, vec![4, 4]);

            // scales [N, groups] and qzeros — split along dim 0 (N) → [2, 8]
            let companions = shards[gpu_id].registry.int4_companions.get(&qweight_name).unwrap();
            assert_eq!(companions.scales.shape, vec![2, 8], "scales should be sharded on dim 0 for column-parallel");
            assert_eq!(companions.qzeros.shape, vec![2, 8], "qzeros should be sharded on dim 0 for column-parallel");
        }
    }

    #[test]
    fn test_shard_weights_tp_int4_companions_row_parallel() {
        // Verify companion weights are sharded correctly for row-parallel.
        let mut registry = WeightRegistry::new();
        let qweight_name = "layers.0.mlp.down_proj.qweight".to_string();
        registry.tensors.insert(
            qweight_name.clone(),
            WeightData {
                data: Bytes::from(vec![0u8; 128]), // shape [4, 8] * 4 bytes
                shape: vec![4, 8],
                dtype: WeightDtype::Int4Packed,
                name: qweight_name.clone(),
            },
        );

        registry.int4_companions.insert(
            qweight_name.clone(),
            crate::weights::Int4Companions {
                scales: WeightData {
                    data: Bytes::from(vec![0u8; 64]), // shape [4, 8] * 2 bytes (BF16)
                    shape: vec![4, 8],
                    dtype: WeightDtype::Bf16,
                    name: "layers.0.mlp.down_proj.scales".to_string(),
                },
                qzeros: WeightData {
                    data: Bytes::from(vec![0u8; 128]), // shape [4, 8] * 4 bytes (u32)
                    shape: vec![4, 8],
                    dtype: WeightDtype::Int4Packed,
                    name: "layers.0.mlp.down_proj.qzeros".to_string(),
                },
            },
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();

        for gpu_id in 0..2 {
            // qweight should be sharded along dim 0 (K/8) → [2, 8]
            let w = shards[gpu_id].registry.tensors.get(&qweight_name).unwrap();
            assert_eq!(w.shape, vec![2, 8]);

            // scales [N, groups] — split along last_dim (groups) → [4, 4]
            let companions = shards[gpu_id].registry.int4_companions.get(&qweight_name).unwrap();
            assert_eq!(companions.scales.shape, vec![4, 4], "scales should be sharded on last dim for row-parallel");
            assert_eq!(companions.qzeros.shape, vec![4, 4], "qzeros should be sharded on last dim for row-parallel");
        }
    }

    #[test]
    fn test_shard_weights_tp_bf16_unchanged() {
        // Ensure BF16 weight sharding is NOT affected by INT4 changes.
        // Column-parallel: split dim 0 → [2, 8] from [4, 8]
        let mut registry = WeightRegistry::new();
        registry.tensors.insert(
            "layers.0.self_attn.q_proj.weight".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 64]), // shape [4, 8] * 2 bytes
                shape: vec![4, 8],
                dtype: WeightDtype::Bf16,
                name: "layers.0.self_attn.q_proj.weight".to_string(),
            },
        );
        // Row-parallel: split dim 1 → [4, 4] from [4, 8]
        registry.tensors.insert(
            "layers.0.self_attn.o_proj.weight".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 64]), // shape [4, 8] * 2 bytes
                shape: vec![4, 8],
                dtype: WeightDtype::Bf16,
                name: "layers.0.self_attn.o_proj.weight".to_string(),
            },
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();

        // BF16 column-parallel: split dim 0 → [2, 8]
        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("layers.0.self_attn.q_proj.weight").unwrap();
            assert_eq!(w.shape, vec![2, 8], "BF16 column-parallel should split dim 0");
        }

        // BF16 row-parallel: split dim 1 → [4, 4]
        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("layers.0.self_attn.o_proj.weight").unwrap();
            assert_eq!(w.shape, vec![4, 4], "BF16 row-parallel should split dim 1");
        }
    }

    #[test]
    fn test_shard_weights_tp_int4_replicated() {
        // INT4 weights that are replicated (e.g., lm_head if INT4 quantized) should not be split.
        let mut registry = WeightRegistry::new();
        registry.tensors.insert(
            "lm_head.qweight".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 128]), // shape [4, 8] * 4 bytes
                shape: vec![4, 8],
                dtype: WeightDtype::Int4Packed,
                name: "lm_head.qweight".to_string(),
            },
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();

        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("lm_head.qweight").unwrap();
            assert_eq!(w.shape, vec![4, 8], "INT4 replicated weight should not be split");
        }
    }
}
