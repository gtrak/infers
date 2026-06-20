//! Weight sharding for tensor parallelism (TP=2) and pipeline parallelism (PP=2).
//!
//! TP=2: column-parallel for Q/K/V/gate/up projections, row-parallel for O/down.
//! PP=2: split layers into two stages (0-31, 32-63).

use super::config::ModelConfig;

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
pub fn determine_shard_type(name: &str) -> ShardType {
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

    #[test]
    fn test_determine_shard_type() {
        assert_eq!(
            determine_shard_type("model.layers.0.self_attn.q_proj.weight"),
            ShardType::ColumnParallel
        );
        assert_eq!(determine_shard_type("lm_head.weight"), ShardType::Replicated);
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
}
