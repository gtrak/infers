//! Weight sharding for tensor parallelism (TP=2) and pipeline parallelism (PP=2).
//!
//! TP=2: column-parallel for Q/K/V/gate/up projections, row-parallel for O/down.
//! PP=2: split layers into two stages (0-31, 32-63).

use std::collections::HashSet;

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
    config: &ModelConfig,
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

    // Pre-scan: companion tensors (.scales, .qzeros) are skipped during sharding
    // since they are processed together with their qweight parent.
    let mut companion_skip: HashSet<String> = HashSet::new();
    for name in registry.tensors.keys() {
        if name.ends_with(".scales") || name.ends_with(".qzeros") {
            companion_skip.insert(name.clone());
        }
    }

    for (name, weight) in &registry.tensors {
        // Skip companion tensors that were already processed with their qweight.
        if companion_skip.contains(name) {
            continue;
        }
        let is_int4 = weight.dtype == super::weights::WeightDtype::Int4Packed;

        // Check if this is a fused QKV projection that needs per-projection sharding
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let qkv_segments: &[(usize, usize)] = &[
            (0, key_dim),               // Q
            (key_dim, 2 * key_dim),     // K
            (2 * key_dim, conv_dim),    // V
        ];

        if name.contains("in_proj_qkv") {
            // Shard each sub-projection independently (INT4 column-major layout)
            let layout = FusedProjectionLayout::ColumnMajor;

            // Scaled segments for qzeros (conv_dim/8 instead of conv_dim)
            let qzeros_segments: Vec<(usize, usize)> =
                qkv_segments.iter().map(|&(s, e)| (s / 8, e / 8)).collect();

            for (gpu_id, shard) in shards.iter_mut().enumerate() {
                let sliced = shard_fused_projection_columns(weight, gpu_id, num_gpus, qkv_segments, layout)
                    .context(format!("Failed to shard fused QKV projection: {}", name))?;
                shard.registry.tensors.insert(name.clone(), sliced);

                // Shard INT4 companion weights with the same segment structure (also column-major)
                if is_int4 && name.ends_with(".qweight") {
                    let base = name.strip_suffix(".qweight").unwrap_or(name.as_str());
                    let scales_name = format!("{}.scales", base);
                    let qzeros_name = format!("{}.qzeros", base);

                    if let Some(scales) = registry.tensors.get(&scales_name)
                        && let Some(qzeros) = registry.tensors.get(&qzeros_name) {
                        companion_skip.insert(scales_name.clone());
                        companion_skip.insert(qzeros_name.clone());
                        let sliced_scales = shard_fused_projection_columns(
                            scales,
                            gpu_id,
                            num_gpus,
                            qkv_segments,
                            layout,
                        )
                        .context(format!("Failed to shard INT4 scales: {}", scales_name))?;
                        // qzeros has last_dim = conv_dim/8, so segments must be scaled by 1/8
                        let sliced_qzeros = shard_fused_projection_columns(
                            qzeros,
                            gpu_id,
                            num_gpus,
                            &qzeros_segments,
                            layout,
                        )
                        .context(format!("Failed to shard INT4 qzeros: {}", qzeros_name))?;
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
            continue;
        }

        if name.contains("conv1d.weight") {
            // Shard conv1d weight with row-major layout (BF16)
            let layout = FusedProjectionLayout::RowMajor;
            for (gpu_id, shard) in shards.iter_mut().enumerate() {
                let sliced = shard_fused_projection_columns(weight, gpu_id, num_gpus, qkv_segments, layout)
                    .context(format!("Failed to shard conv1d weight: {}", name))?;
                shard.registry.tensors.insert(name.clone(), sliced);
            }
            continue;
        }

        let shard_type = determine_shard_type(name);

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

                    // Shard INT4 companion weights — extract from registry.tensors by name pattern.
                    if is_int4 && name.ends_with(".qweight") {
                        let base = name.strip_suffix(".qweight").unwrap_or(name.as_str());
                        let scales_name = format!("{}.scales", base);
                        let qzeros_name = format!("{}.qzeros", base);

                        if let Some(scales) = registry.tensors.get(&scales_name)
                            && let Some(qzeros) = registry.tensors.get(&qzeros_name) {
                            companion_skip.insert(scales_name.clone());
                            companion_skip.insert(qzeros_name.clone());
                            let sliced_scales = slice_weight_last_dim(scales, gpu_id, num_gpus)
                                .context(format!("Failed to shard INT4 scales: {}", scales_name))?;
                            let sliced_qzeros = slice_weight_last_dim(qzeros, gpu_id, num_gpus)
                                .context(format!("Failed to shard INT4 qzeros: {}", qzeros_name))?;
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

                    // Shard INT4 companion weights — extract from registry.tensors by name pattern.
                    if is_int4 && name.ends_with(".qweight") {
                        let base = name.strip_suffix(".qweight").unwrap_or(name.as_str());
                        let scales_name = format!("{}.scales", base);
                        let qzeros_name = format!("{}.qzeros", base);

                        if let Some(scales) = registry.tensors.get(&scales_name)
                            && let Some(qzeros) = registry.tensors.get(&qzeros_name) {
                            companion_skip.insert(scales_name.clone());
                            companion_skip.insert(qzeros_name.clone());
                            let sliced_scales = slice_weight_dim0(scales, gpu_id, num_gpus)
                                .context(format!("Failed to shard INT4 scales: {}", scales_name))?;
                            let sliced_qzeros = slice_weight_dim0(qzeros, gpu_id, num_gpus)
                                .context(format!("Failed to shard INT4 qzeros: {}", qzeros_name))?;
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

                    // Replicate INT4 companion weights — extract from registry.tensors by name pattern.
                    if is_int4 && name.ends_with(".qweight") {
                        let base = name.strip_suffix(".qweight").unwrap_or(name.as_str());
                        let scales_name = format!("{}.scales", base);
                        let qzeros_name = format!("{}.qzeros", base);

                        if let Some(scales) = registry.tensors.get(&scales_name)
                            && let Some(qzeros) = registry.tensors.get(&qzeros_name) {
                            companion_skip.insert(scales_name.clone());
                            companion_skip.insert(qzeros_name.clone());
                            shard.registry.int4_companions.insert(
                                name.clone(),
                                super::weights::Int4Companions {
                                    scales: scales.clone(),
                                    qzeros: qzeros.clone(),
                                },
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

/// Layout mode for splitting fused projection weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FusedProjectionLayout {
    /// INT4 column-major: shape (M, N), split on last dim (N).
    /// Used by qweight and companion tensors (scales, qzeros) in GPTQ/AutoRound format.
    ColumnMajor,
    /// BF16 row-major: shape [N, K], split on first dim (N).
    /// Used by conv1d.weight (shape [conv_dim, 1, kernel_size]).
    RowMajor,
}

// @lat: [[lat#Weight Sharding#Tensor Parallelism Sharding#Fused QKV Projection Sharding]]
/// Shard a fused projection weight by splitting each sub-projection independently.
///
/// For `in_proj_qkv`, the output dimension (last dim for INT4, first dim for BF16)
/// contains concatenated sub-projections: [Q, K, V]. Each sub-projection must be
/// independently split across GPUs, then re-concatenated.
///
/// `segments` is a list of (start, end) column ranges within the full output dimension.
/// For in_proj_qkv: [(0, key_dim), (key_dim, 2*key_dim), (2*key_dim, conv_dim)]
///
/// `layout` determines which dimension to split: ColumnMajor splits dim -1 (N),
/// RowMajor splits dim 0 (N). Companion tensors of INT4 weights use ColumnMajor
/// even if their dtype is BF16.
fn shard_fused_projection_columns(
    weight: &WeightData,
    gpu_id: usize,
    num_gpus: usize,
    segments: &[(usize, usize)],
    layout: FusedProjectionLayout,
) -> Result<WeightData> {
    anyhow::ensure!(weight.shape.len() >= 2, "Fused projection weight must be at least 2D");

    match layout {
        FusedProjectionLayout::ColumnMajor => {
            // Shape (M, N) — split on last dim (N).
            // Used by INT4 qweight, scales, and qzeros companions.
            let rows = weight.shape[0];
            let full_n = weight.shape[1];
            let bytes_per_element = weight.data.len() / (rows * full_n);

            let shard_n: usize = segments.iter().map(|&(s, e)| (e - s) / num_gpus).sum();
            let mut shard_data = Vec::new();

            for row in 0..rows {
                let row_offset = row * full_n * bytes_per_element;

                for &(start, end) in segments {
                    let seg_len = end - start;
                    let shard_size = seg_len / num_gpus;
                    let shard_start = start + gpu_id * shard_size;
                    let shard_end = shard_start + shard_size;

                    shard_data.extend_from_slice(
                        &weight.data[row_offset + shard_start * bytes_per_element
                            ..row_offset + shard_end * bytes_per_element],
                    );
                }
            }

            let mut new_shape = weight.shape.clone();
            new_shape[1] = shard_n;

            Ok(WeightData {
                data: Bytes::from(shard_data),
                shape: new_shape,
                dtype: weight.dtype,
                name: weight.name.clone(),
            })
        }
        FusedProjectionLayout::RowMajor => {
            // Shape (N, K...) — split on dim 0.
            // Used by conv1d.weight with BF16 dtype.
            let full_n = weight.shape[0];
            let bytes_per_row = weight.data.len() / full_n;

            let mut shard_n: usize = 0;
            let mut shard_data = Vec::new();

            for &(start, end) in segments {
                let seg_len = end - start;
                let shard_size = seg_len / num_gpus;
                let shard_start = start + gpu_id * shard_size;
                let shard_end = shard_start + shard_size;
                shard_n += shard_size;

                // Copy rows [shard_start, shard_end) from this segment
                let start_byte = shard_start * bytes_per_row;
                let end_byte = shard_end * bytes_per_row;
                shard_data.extend_from_slice(&weight.data[start_byte..end_byte]);
            }

            let mut new_shape = weight.shape.clone();
            new_shape[0] = shard_n;

            Ok(WeightData {
                data: Bytes::from(shard_data),
                shape: new_shape,
                dtype: weight.dtype,
                name: weight.name.clone(),
            })
        }
    }
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
        // For transposed INT4 format: qweight [K/8, N], scales [K/group_size, N].
        // Column-parallel splits N on the last dim (same as qweight).
        // Companions must be present as separate tensors in registry.tensors.
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

        // Scales and qzeros must be in registry.tensors (not int4_companions)
        registry.tensors.insert(
            "layers.0.mlp.gate_proj.scales".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 64]), // shape [4, 8] * 2 bytes (BF16)
                shape: vec![4, 8],
                dtype: WeightDtype::Bf16,
                name: "layers.0.mlp.gate_proj.scales".to_string(),
            },
        );
        registry.tensors.insert(
            "layers.0.mlp.gate_proj.qzeros".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 128]), // shape [4, 8] * 4 bytes (u32)
                shape: vec![4, 8],
                dtype: WeightDtype::Int4Packed,
                name: "layers.0.mlp.gate_proj.qzeros".to_string(),
            },
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();

        for gpu_id in 0..2 {
            // qweight should be sharded along dim 1 (N) → [4, 4]
            let w = shards[gpu_id].registry.tensors.get(&qweight_name).unwrap();
            assert_eq!(w.shape, vec![4, 4]);

            // scales [K/group_size, N] and qzeros — split along last dim (N) → [4, 4]
            let companions = shards[gpu_id].registry.int4_companions.get(&qweight_name).unwrap();
            assert_eq!(companions.scales.shape, vec![4, 4], "scales should be sharded on last dim for column-parallel (transposed format)");
            assert_eq!(companions.qzeros.shape, vec![4, 4], "qzeros should be sharded on last dim for column-parallel (transposed format)");
        }
    }

    #[test]
    fn test_shard_weights_tp_int4_companions_row_parallel() {
        // Verify companion weights are sharded correctly for row-parallel.
        // For transposed INT4 format: qweight [K/8, N], scales [K/group_size, N].
        // Row-parallel splits K/group_size on dim 0 (same as qweight splits K/8 on dim 0).
        // Companions must be present as separate tensors in registry.tensors.
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

        // Scales and qzeros must be in registry.tensors (not int4_companions)
        registry.tensors.insert(
            "layers.0.mlp.down_proj.scales".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 64]), // shape [4, 8] * 2 bytes (BF16)
                shape: vec![4, 8],
                dtype: WeightDtype::Bf16,
                name: "layers.0.mlp.down_proj.scales".to_string(),
            },
        );
        registry.tensors.insert(
            "layers.0.mlp.down_proj.qzeros".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 128]), // shape [4, 8] * 4 bytes (u32)
                shape: vec![4, 8],
                dtype: WeightDtype::Int4Packed,
                name: "layers.0.mlp.down_proj.qzeros".to_string(),
            },
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();

        for gpu_id in 0..2 {
            // qweight should be sharded along dim 0 (K/8) → [2, 8]
            let w = shards[gpu_id].registry.tensors.get(&qweight_name).unwrap();
            assert_eq!(w.shape, vec![2, 8]);

            // scales [K/group_size, N] — split along dim 0 (groups) → [2, 8]
            let companions = shards[gpu_id].registry.int4_companions.get(&qweight_name).unwrap();
            assert_eq!(companions.scales.shape, vec![2, 8], "scales should be sharded on dim 0 for row-parallel (transposed format)");
            assert_eq!(companions.qzeros.shape, vec![2, 8], "qzeros should be sharded on dim 0 for row-parallel (transposed format)");
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

    #[test]
    fn test_shard_weights_tp_int4_companions_from_tensors_column_parallel() {
        // Verify companion weights extracted from registry.tensors (not int4_companions)
        // are sharded correctly for column-parallel. Companion tensors must be separate
        // entries in registry.tensors, not pre-populated in int4_companions.
        let mut registry = WeightRegistry::new();

        // qweight: shape [4, 8] (K/8=4, N=8) — column-parallel splits dim 1 → [4, 4]
        registry.tensors.insert(
            "model.layers.0.mlp.gate_proj.qweight".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 128]),
                shape: vec![4, 8],
                dtype: WeightDtype::Int4Packed,
                name: "model.layers.0.mlp.gate_proj.qweight".to_string(),
            },
        );

        // scales: shape [4, 8] (K/group_size=4, N=8) — should be split along last dim → [4, 4]
        registry.tensors.insert(
            "model.layers.0.mlp.gate_proj.scales".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 64]),
                shape: vec![4, 8],
                dtype: WeightDtype::Bf16,
                name: "model.layers.0.mlp.gate_proj.scales".to_string(),
            },
        );

        // qzeros: shape [4, 8] — should be split along last dim → [4, 4]
        registry.tensors.insert(
            "model.layers.0.mlp.gate_proj.qzeros".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 128]),
                shape: vec![4, 8],
                dtype: WeightDtype::Int4Packed,
                name: "model.layers.0.mlp.gate_proj.qzeros".to_string(),
            },
        );

        // Ensure int4_companions is empty — companions must come from registry.tensors
        assert!(registry.int4_companions.is_empty());

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        for gpu_id in 0..2 {
            // qweight sharded along last dim → [4, 4]
            let w = shards[gpu_id].registry.tensors.get("model.layers.0.mlp.gate_proj.qweight").unwrap();
            assert_eq!(w.shape, vec![4, 4], "GPU {} qweight should be [4,4]", gpu_id);

            // Companions extracted from registry.tensors and sharded along last dim → [4, 4]
            let companions = shards[gpu_id]
                .registry
                .int4_companions
                .get("model.layers.0.mlp.gate_proj.qweight")
                .unwrap_or_else(|| panic!("GPU {} should have companions in int4_companions", gpu_id));
            assert_eq!(companions.scales.shape, vec![4, 4], "GPU {} scales should be [4,4]", gpu_id);
            assert_eq!(companions.qzeros.shape, vec![4, 4], "GPU {} qzeros should be [4,4]", gpu_id);

            // Companion tensors should NOT appear in shards[gpu_id].registry.tensors (skipped)
            assert!(
                shards[gpu_id]
                    .registry
                    .tensors
                    .get("model.layers.0.mlp.gate_proj.scales")
                    .is_none(),
                "GPU {} scales should not be in tensors HashMap",
                gpu_id
            );
            assert!(
                shards[gpu_id]
                    .registry
                    .tensors
                    .get("model.layers.0.mlp.gate_proj.qzeros")
                    .is_none(),
                "GPU {} qzeros should not be in tensors HashMap",
                gpu_id
            );
        }
    }

    #[test]
    fn test_shard_weights_tp_int4_companions_from_tensors_row_parallel() {
        // Verify companion weights extracted from registry.tensors (not int4_companions)
        // are sharded correctly for row-parallel. Companion tensors must be separate
        // entries in registry.tensors, not pre-populated in int4_companions.
        let mut registry = WeightRegistry::new();

        // qweight: shape [4, 8] (K/8=4, N=8) — row-parallel splits dim 0 → [2, 8]
        registry.tensors.insert(
            "model.layers.0.mlp.down_proj.qweight".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 128]),
                shape: vec![4, 8],
                dtype: WeightDtype::Int4Packed,
                name: "model.layers.0.mlp.down_proj.qweight".to_string(),
            },
        );

        // scales: shape [4, 8] — should be split along dim 0 → [2, 8]
        registry.tensors.insert(
            "model.layers.0.mlp.down_proj.scales".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 64]),
                shape: vec![4, 8],
                dtype: WeightDtype::Bf16,
                name: "model.layers.0.mlp.down_proj.scales".to_string(),
            },
        );

        // qzeros: shape [4, 8] — should be split along dim 0 → [2, 8]
        registry.tensors.insert(
            "model.layers.0.mlp.down_proj.qzeros".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 128]),
                shape: vec![4, 8],
                dtype: WeightDtype::Int4Packed,
                name: "model.layers.0.mlp.down_proj.qzeros".to_string(),
            },
        );

        // Ensure int4_companions is empty — companions must come from registry.tensors
        assert!(registry.int4_companions.is_empty());

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        for gpu_id in 0..2 {
            // qweight sharded along dim 0 → [2, 8]
            let w = shards[gpu_id].registry.tensors.get("model.layers.0.mlp.down_proj.qweight").unwrap();
            assert_eq!(w.shape, vec![2, 8], "GPU {} qweight should be [2,8]", gpu_id);

            // Companions extracted from registry.tensors and sharded along dim 0 → [2, 8]
            let companions = shards[gpu_id]
                .registry
                .int4_companions
                .get("model.layers.0.mlp.down_proj.qweight")
                .unwrap_or_else(|| panic!("GPU {} should have companions in int4_companions", gpu_id));
            assert_eq!(companions.scales.shape, vec![2, 8], "GPU {} scales should be [2,8]", gpu_id);
            assert_eq!(companions.qzeros.shape, vec![2, 8], "GPU {} qzeros should be [2,8]", gpu_id);

            // Companion tensors should NOT appear in shards[gpu_id].registry.tensors (skipped)
            assert!(
                shards[gpu_id]
                    .registry
                    .tensors
                    .get("model.layers.0.mlp.down_proj.scales")
                    .is_none(),
                "GPU {} scales should not be in tensors HashMap",
                gpu_id
            );
            assert!(
                shards[gpu_id]
                    .registry
                    .tensors
                    .get("model.layers.0.mlp.down_proj.qzeros")
                    .is_none(),
                "GPU {} qzeros should not be in tensors HashMap",
                gpu_id
            );
        }
    }

    #[test]
    fn test_shard_fused_projection_columns_int4() {
        // Simulate in_proj_qkv qweight: INT4 shape (K/8, N) = (5, 12)
        // Segments: Q=[0,4), K=[4,8), V=[8,12)
        // TP=2: GPU0 gets Q[0,2)+K[4,6)+V[8,10) = cols 0,2,4 → N=6
        //        GPU1 gets Q[2,4)+K[6,8)+V[10,12) = cols 5,7,9 → N=6
        let weight = WeightData {
            data: Bytes::from(vec![0u8; 5 * 12 * 4]), // u32
            shape: vec![5, 12],
            dtype: WeightDtype::Int4Packed,
            name: "test.in_proj_qkv.qweight".to_string(),
        };
        let segments = &[(0, 4), (4, 8), (8, 12)];

        let shard0 = shard_fused_projection_columns(&weight, 0, 2, segments, FusedProjectionLayout::ColumnMajor).unwrap();
        assert_eq!(shard0.shape, vec![5, 6], "GPU 0 should have shape [5, 6]");
        assert_eq!(shard0.data.len(), 5 * 6 * 4);

        let shard1 = shard_fused_projection_columns(&weight, 1, 2, segments, FusedProjectionLayout::ColumnMajor).unwrap();
        assert_eq!(shard1.shape, vec![5, 6], "GPU 1 should have shape [5, 6]");
        assert_eq!(shard1.data.len(), 5 * 6 * 4);
    }

    #[test]
    fn test_shard_fused_projection_columns_bf16() {
        // Simulate conv1d.weight: BF16 shape [N, 1, kernel_size] = [12, 1, 3]
        // Segments: Q=[0,4), K=[4,8), V=[8,12)
        // TP=2: GPU0 gets rows Q[0,2)+K[4,6)+V[8,10) = 2+2+2=6 rows
        //        GPU1 gets rows Q[2,4)+K[6,8)+V[10,12) = 2+2+2=6 rows
        let weight = WeightData {
            data: Bytes::from(vec![0u8; 12 * 1 * 3 * 2]), // bf16
            shape: vec![12, 1, 3],
            dtype: WeightDtype::Bf16,
            name: "test.conv1d.weight".to_string(),
        };
        let segments = &[(0, 4), (4, 8), (8, 12)];

        let shard0 = shard_fused_projection_columns(&weight, 0, 2, segments, FusedProjectionLayout::RowMajor).unwrap();
        assert_eq!(shard0.shape, vec![6, 1, 3], "GPU 0 should have shape [6, 1, 3]");
        assert_eq!(shard0.data.len(), 6 * 1 * 3 * 2);

        let shard1 = shard_fused_projection_columns(&weight, 1, 2, segments, FusedProjectionLayout::RowMajor).unwrap();
        assert_eq!(shard1.shape, vec![6, 1, 3], "GPU 1 should have shape [6, 1, 3]");
        assert_eq!(shard1.data.len(), 6 * 1 * 3 * 2);
    }

    #[test]
    fn test_shard_fused_projection_columns_data_correctness() {
        // Use distinct values to verify data is extracted from correct segments.
        // ColumnMajor layout (INT4-style): shape (2, 8), segments: Q=[0,2), K=[2,5), V=[5,8)
        // TP=2: GPU0 gets Q[0,1)+K[2,3)+V[5,6) → 1 col from each segment = 3 cols
        //        GPU1 gets Q[1,2)+K[3,4)+V[6,7) → 1 col from each segment = 3 cols
        // Data layout (1 byte per element): row 0: [0,1,2,3,4,5,6,7], row 1: [10,11,12,13,14,15,16,17]
        let weight = WeightData {
            data: Bytes::from(vec![
                0u8, 1, 2, 3, 4, 5, 6, 7, // row 0
                10, 11, 12, 13, 14, 15, 16, 17, // row 1
            ]),
            shape: vec![2, 8],
            dtype: WeightDtype::Int4Packed,
            name: "test.qweight".to_string(),
        };
        let segments = &[(0, 2), (2, 5), (5, 8)];

        // The function iterates rows first, then segments within each row.
        // So data is in (row × segment) order: [row0_Q + row0_K + row0_V, row1_Q + row1_K + row1_V]
        // GPU 0: col 0 from Q, col 2 from K, col 5 from V
        // [byte(col0,row0), byte(col2,row0), byte(col5,row0), byte(col0,row1), byte(col2,row1), byte(col5,row1)]
        // = [0, 2, 5, 10, 12, 15]
        let shard0 = shard_fused_projection_columns(&weight, 0, 2, segments, FusedProjectionLayout::ColumnMajor).unwrap();
        assert_eq!(shard0.shape, vec![2, 3]);
        assert_eq!(&shard0.data[..], &[0u8, 2, 5, 10, 12, 15]);

        // GPU 1: col 1 from Q, col 3 from K, col 6 from V
        // [byte(col1,row0), byte(col3,row0), byte(col6,row0), byte(col1,row1), byte(col3,row1), byte(col6,row1)]
        // = [1, 3, 6, 11, 13, 16]
        let shard1 = shard_fused_projection_columns(&weight, 1, 2, segments, FusedProjectionLayout::ColumnMajor).unwrap();
        assert_eq!(shard1.shape, vec![2, 3]);
        assert_eq!(&shard1.data[..], &[1u8, 3, 6, 11, 13, 16]);
    }

    #[test]
    fn test_shard_fused_projection_column_major_row_order() {
        // Verify ColumnMajor produces row-major layout: each row contains GPU's
        // portion of all segments contiguously (Q+K+V columns interleaved per row).
        // 2×12 matrix with values 0..=23, segments Q=[0,4), K=[4,8), V=[8,12).
        // TP=2: GPU 0 gets cols [0,1] from Q, [4,5] from K, [8,9] from V.
        // GPU 0 row 0: [0,1, 4,5, 8,9]  GPU 0 row 1: [12,13, 16,17, 20,21]
        // GPU 1 row 0: [2,3, 6,7, 10,11]  GPU 1 row 1: [14,15, 18,19, 22,23]
        let weight = WeightData {
            data: Bytes::from(vec![
                0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, // row 0
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, // row 1
            ]),
            shape: vec![2, 12],
            dtype: WeightDtype::Int4Packed,
            name: "test.qweight".to_string(),
        };
        let segments = &[(0, 4), (4, 8), (8, 12)];

        let shard0 = shard_fused_projection_columns(&weight, 0, 2, segments, FusedProjectionLayout::ColumnMajor).unwrap();
        assert_eq!(shard0.shape, vec![2, 6]);
        assert_eq!(&shard0.data[..], &[0u8, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21]);

        let shard1 = shard_fused_projection_columns(&weight, 1, 2, segments, FusedProjectionLayout::ColumnMajor).unwrap();
        assert_eq!(shard1.shape, vec![2, 6]);
        assert_eq!(&shard1.data[..], &[2u8, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23]);
    }

    #[test]
    fn test_in_proj_qkv_int4_sharding() {
        // Full integration test: INT4 in_proj_qkv with proper dimensions.
        // key_dim = 2*16 = 32, value_dim = 4*16 = 64, conv_dim = 32 + 32 + 64 = 128
        // Segments: Q=[0,32), K=[32,64), V=[64,128)
        // TP=2: GPU0 gets Q[0,16)+K[32,48)+V[64,96) → N=96
        //        GPU1 gets Q[16,32)+K[48,64)+V[96,128) → N=96
        let mut registry = WeightRegistry::new();
        let qweight_name = "layers.0.linear_attn.in_proj_qkv.qweight".to_string();

        // qweight: shape (K/8, conv_dim) = (4, 128), K=32
        registry.tensors.insert(
            qweight_name.clone(),
            WeightData {
                data: Bytes::from(vec![0u8; 4 * 128 * 4]),
                shape: vec![4, 128],
                dtype: WeightDtype::Int4Packed,
                name: qweight_name.clone(),
            },
        );

        // scales: shape (groups, conv_dim) = (4, 128)
        registry.tensors.insert(
            "layers.0.linear_attn.in_proj_qkv.scales".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 4 * 128 * 2]),
                shape: vec![4, 128],
                dtype: WeightDtype::Bf16,
                name: "layers.0.linear_attn.in_proj_qkv.scales".to_string(),
            },
        );

        // qzeros: shape (groups, conv_dim/8) = (4, 16)
        registry.tensors.insert(
            "layers.0.linear_attn.in_proj_qkv.qzeros".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 4 * 16 * 4]),
                shape: vec![4, 16],
                dtype: WeightDtype::Int4Packed,
                name: "layers.0.linear_attn.in_proj_qkv.qzeros".to_string(),
            },
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_num_value_heads":4,"linear_value_head_dim":16}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        // Verify dimensions
        assert_eq!(config.linear_num_key_heads * config.linear_key_head_dim, 32); // key_dim
        assert_eq!(config.linear_num_value_heads * config.linear_value_head_dim, 64); // value_dim

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        // GPU 0: Q[0,16)+K[32,48)+V[64,96) → N=16+16+32=64
        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get(&qweight_name).unwrap();
            assert_eq!(
                w.shape,
                vec![4, 64],
                "GPU {} qweight should be [4, 64]",
                gpu_id
            );

            let companions = shards[gpu_id]
                .registry
                .int4_companions
                .get(&qweight_name)
                .unwrap();
            assert_eq!(
                companions.scales.shape, vec![4, 64],
                "GPU {} scales should be [4, 64]",
                gpu_id
            );
            // qzeros with scaled segments: N/8 = 64/8 = 8
            // Scaled: Q[0,2)+K[4,6)+V[8,16) → 2+2+8=12... wait, let me recalculate.
            // Full segments for qzeros: scaled by /8
            // key_dim=32, so scaled: Q=[0,4), K=[4,8), V=[8,16)
            // GPU 0: Q[0,2)+K[4,6)+V[8,12) → 2+2+4=8
            assert_eq!(
                companions.qzeros.shape, vec![4, 8],
                "GPU {} qzeros should be [4, 8]",
                gpu_id
            );
        }
    }

    #[test]
    fn test_conv1d_weight_sharding() {
        // BF16 conv1d.weight: shape [conv_dim, 1, kernel_size] = [128, 1, 4]
        // key_dim=32, value_dim=64, conv_dim=128
        // Segments: Q=[0,32), K=[32,64), V=[64,128)
        // TP=2: GPU0 gets 16+16+32=64 rows, GPU1 gets 16+16+32=64 rows
        let mut registry = WeightRegistry::new();

        registry.tensors.insert(
            "layers.0.linear_attn.conv1d.weight".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 128 * 1 * 4 * 2]), // bf16
                shape: vec![128, 1, 4],
                dtype: WeightDtype::Bf16,
                name: "layers.0.linear_attn.conv1d.weight".to_string(),
            },
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_num_value_heads":4,"linear_value_head_dim":16}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp(&registry, &config, 2).unwrap();

        for gpu_id in 0..2 {
            let w = shards[gpu_id]
                .registry
                .tensors
                .get("layers.0.linear_attn.conv1d.weight")
                .unwrap();
            assert_eq!(
                w.shape,
                vec![64, 1, 4],
                "GPU {} conv1d.weight should be [64, 1, 4]",
                gpu_id
            );
        }
    }
}
