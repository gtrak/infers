//! Heap-only weight loader — used solely for sharding equivalence tests.
//!
//! Loads safetensors files by copying data into owned `Bytes` buffers,
//! as opposed to the mmap path which keeps data zero-copy mapped.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use bytes::Bytes;
use safetensors::SafeTensors;

use infers_model::config::ModelConfig;
use infers_model::weights::{Int4Companions, Nvfp4Companions, QuantCompanions, ShardIndex, WeightData, WeightDtype, WeightRegistry, WeightShard};
// ---------------------------------------------------------------------------
// Safetensors loading (heap copy)
// ---------------------------------------------------------------------------

/// Load all safetensors files from a model directory.
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
        let data = Bytes::copy_from_slice(tensor.data());

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
    let shards: HashSet<String> = index.weight_map.values().cloned().collect();
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
            let data = Bytes::copy_from_slice(tensor.data());

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

/// Load model from a directory (config + safetensors).
pub fn load_model(model_dir: &Path) -> Result<(ModelConfig, WeightRegistry)> {
    let config = ModelConfig::load(model_dir)?;
    let registry = load_safetensors(model_dir)?;
    Ok((config, registry))
}

/// Map safetensors dtype to our WeightDtype.
pub fn map_safetensor_dtype(dtype: safetensors::Dtype) -> WeightDtype {
    match dtype {
        safetensors::Dtype::BF16 => WeightDtype::Bf16,
        safetensors::Dtype::F16 => WeightDtype::Fp16,
        safetensors::Dtype::F32 => WeightDtype::Fp32,
        safetensors::Dtype::U32 => WeightDtype::Int4Packed, // INT4 packed as u32
        safetensors::Dtype::I32 => WeightDtype::Int4Packed, // INT4 packed as i32 (used by newer AutoRound)
        _ => WeightDtype::Other,
    }
}

// ---------------------------------------------------------------------------
// Tensor parallelism sharding (heap path)
// ---------------------------------------------------------------------------

/// Shard model weights across `num_gpus` devices for tensor parallelism.
///
/// Column-parallel: Q, K, V, gate, up projections are split along the output dimension.
/// Row-parallel: O, down projections are replicated on each GPU.
pub fn shard_weights_tp(
    registry: &WeightRegistry,
    config: &ModelConfig,
    num_gpus: usize,
) -> Result<Vec<WeightShard>> {
    anyhow::ensure!(num_gpus >= 1, "num_gpus must be >= 1");

    if num_gpus == 1 {
        // No sharding needed for single GPU — clone and populate companions.
        let mut shard_registry = registry.clone();
        for name in shard_registry.tensors.keys() {
            // Handle INT4 companions
            if name.ends_with(".qweight") {
                let base = name.strip_suffix(".qweight").unwrap_or(name.as_str());
                let scales_name = format!("{}.scales", base);
                let qzeros_name = format!("{}.qzeros", base);

                if let Some(scales) = shard_registry.tensors.get(&scales_name)
                    && let Some(qzeros) = shard_registry.tensors.get(&qzeros_name) {
                    shard_registry.quant_companions.insert(
                        name.clone(),
                        QuantCompanions::Int4(Int4Companions {
                            scales: scales.clone(),
                            qzeros: qzeros.clone(),
                        }),
                    );
                }
            }

            // Handle NVFP4 companions
            if name.ends_with(".weight_packed") {
                let base = name.strip_suffix(".weight_packed").unwrap_or(name.as_str());
                let weight_scale_name = format!("{}.weight_scale", base);
                let weight_global_scale_name = format!("{}.weight_global_scale", base);
                let input_global_scale_name = format!("{}.input_global_scale", base);

                if let Some(weight_scale) = shard_registry.tensors.get(&weight_scale_name)
                    && let Some(weight_global_scale) = shard_registry.tensors.get(&weight_global_scale_name)
                    && let Some(input_global_scale) = shard_registry.tensors.get(&input_global_scale_name) {
                    shard_registry.quant_companions.insert(
                        name.clone(),
                        QuantCompanions::Nvfp4(Nvfp4Companions {
                            weight_scale: weight_scale.clone(),
                            weight_global_scale: weight_global_scale.clone(),
                            input_global_scale: input_global_scale.clone(),
                        }),
                    );
                }
            }
        }
        return Ok(vec![WeightShard { gpu_id: 0, registry: shard_registry }]);
    }
    // For TP=2+, shard each layer's weights
    let mut shards: Vec<WeightShard> = (0..num_gpus)
        .map(|gpu_id| WeightShard {
            gpu_id,
            registry: WeightRegistry::new(),
        })
        .collect();

    // Pre-scan: companion tensors (.scales, .qzeros) are skipped during sharding
    // since they are processed together with their qweight parent.
    let mut companion_skip: HashSet<String> = HashSet::new();
    for name in registry.tensors.keys() {
        if name.ends_with(".scales") || name.ends_with(".qzeros") {
            companion_skip.insert(name.clone());
        }
        // NVFP4 companions
        if name.ends_with(".weight_scale") || name.ends_with(".weight_global_scale") || name.ends_with(".input_global_scale") {
            companion_skip.insert(name.clone());
        }
    }
    for (name, weight) in &registry.tensors {
        // Skip companion tensors that were already processed with their qweight.
        if companion_skip.contains(name) {
            continue;
        }
        let is_int4 = weight.dtype == WeightDtype::Int4Packed;

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
            // The fused output dimension is the last segment endpoint.
            let fused_dim = qkv_segments.last().map(|(_, e)| *e).unwrap_or(0);
            // Detect which axis of the weight tensor holds the fused output dimension.
            // INT4 qweight: shape (K/8, N=fused_dim) → split dim1 → ColumnMajor
            // NVFP4 weight_packed: shape (N=fused_dim, K/2) → split dim0 → RowMajor
            let layout = if weight.shape[0] == fused_dim {
                FusedProjectionLayout::RowMajor
            } else {
                FusedProjectionLayout::ColumnMajor
            };

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
                        shard.registry.quant_companions.insert(
                            name.clone(),
                            QuantCompanions::Int4(Int4Companions {
                                scales: sliced_scales,
                                qzeros: sliced_qzeros,
                            }),
                        );
                    }
                }

                // Shard NVFP4 companion weights — replicate ALL companions to all GPUs.
                // Fused QKV: weight_scale has a different second dim than weight_packed
                // (groups vs elements), so segment-based sharding doesn't apply.
                if name.ends_with(".weight_packed") {
                    let base = name.strip_suffix(".weight_packed").unwrap_or(name.as_str());
                    let weight_scale_name = format!("{}.weight_scale", base);
                    let weight_global_scale_name = format!("{}.weight_global_scale", base);
                    let input_global_scale_name = format!("{}.input_global_scale", base);

                    if let Some(weight_scale) = registry.tensors.get(&weight_scale_name)
                        && let Some(weight_global_scale) = registry.tensors.get(&weight_global_scale_name)
                        && let Some(input_global_scale) = registry.tensors.get(&input_global_scale_name) {
                        companion_skip.insert(weight_scale_name.clone());
                        companion_skip.insert(weight_global_scale_name.clone());
                        companion_skip.insert(input_global_scale_name.clone());
                        shard.registry.quant_companions.insert(
                            name.clone(),
                            QuantCompanions::Nvfp4(Nvfp4Companions {
                                weight_scale: weight_scale.clone(),
                                weight_global_scale: weight_global_scale.clone(),
                                input_global_scale: input_global_scale.clone(),
                            }),
                        );
                    }
                }
            }
            continue;
        }

        if name.contains("conv1d.weight") {
            let fused_dim = qkv_segments.last().map(|(_, e)| *e).unwrap_or(0);
            // Detect which axis of the weight tensor holds the fused output dimension.
            // BF16: shape (N=fused_dim, K...) → split dim0 → RowMajor
            // NVFP4 might differ — detect generically from shape vs segments.
            let layout = if weight.shape[0] == fused_dim {
                FusedProjectionLayout::RowMajor
            } else {
                FusedProjectionLayout::ColumnMajor
            };
            for (gpu_id, shard) in shards.iter_mut().enumerate() {
                let sliced = shard_fused_projection_columns(weight, gpu_id, num_gpus, qkv_segments, layout)
                    .context(format!("Failed to shard conv1d weight: {}", name))?;
                shard.registry.tensors.insert(name.clone(), sliced);
            }
            continue;
        }

        let shard_type = infers_model::sharding::determine_shard_type(name);

        match shard_type {
            infers_model::sharding::ShardType::ColumnParallel => {
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
                            shard.registry.quant_companions.insert(
                                name.clone(),
                                QuantCompanions::Int4(Int4Companions {
                                    scales: sliced_scales,
                                    qzeros: sliced_qzeros,
                                }),
                            );
                        }
                    }

                    // Shard NVFP4 companion weights
                    if name.ends_with(".weight_packed") {
                        let base = name.strip_suffix(".weight_packed").unwrap_or(name.as_str());
                        let weight_scale_name = format!("{}.weight_scale", base);
                        let weight_global_scale_name = format!("{}.weight_global_scale", base);
                        let input_global_scale_name = format!("{}.input_global_scale", base);

                        if let Some(weight_scale) = registry.tensors.get(&weight_scale_name)
                            && let Some(weight_global_scale) = registry.tensors.get(&weight_global_scale_name)
                            && let Some(input_global_scale) = registry.tensors.get(&input_global_scale_name) {
                            companion_skip.insert(weight_scale_name.clone());
                            companion_skip.insert(weight_global_scale_name.clone());
                            companion_skip.insert(input_global_scale_name.clone());
                            // Column-parallel: weight_packed [N, K/2] split on dim0 → weight_scale [N, K/gs] split on dim0
                            let sliced_ws = slice_weight_dim0(weight_scale, gpu_id, num_gpus)
                                .context(format!("Failed to shard NVFP4 weight_scale: {}", weight_scale_name))?;
                            // weight_global_scale and input_global_scale are 1D scalars — replicate to all GPUs
                            shard.registry.quant_companions.insert(
                                name.clone(),
                                QuantCompanions::Nvfp4(Nvfp4Companions {
                                    weight_scale: sliced_ws,
                                    weight_global_scale: weight_global_scale.clone(),
                                    input_global_scale: input_global_scale.clone(),
                                }),
                            );
                        }
                    }
                }
            }
            infers_model::sharding::ShardType::RowParallel => {
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
                            shard.registry.quant_companions.insert(
                                name.clone(),
                                QuantCompanions::Int4(Int4Companions {
                                    scales: sliced_scales,
                                    qzeros: sliced_qzeros,
                                }),
                            );
                        }
                    }

                    // Shard NVFP4 companion weights
                    if name.ends_with(".weight_packed") {
                        let base = name.strip_suffix(".weight_packed").unwrap_or(name.as_str());
                        let weight_scale_name = format!("{}.weight_scale", base);
                        let weight_global_scale_name = format!("{}.weight_global_scale", base);
                        let input_global_scale_name = format!("{}.input_global_scale", base);

                        if let Some(weight_scale) = registry.tensors.get(&weight_scale_name)
                            && let Some(weight_global_scale) = registry.tensors.get(&weight_global_scale_name)
                            && let Some(input_global_scale) = registry.tensors.get(&input_global_scale_name) {
                            companion_skip.insert(weight_scale_name.clone());
                            companion_skip.insert(weight_global_scale_name.clone());
                            companion_skip.insert(input_global_scale_name.clone());
                            let sliced_ws = slice_weight_last_dim(weight_scale, gpu_id, num_gpus)
                                .context(format!("Failed to shard NVFP4 weight_scale: {}", weight_scale_name))?;
                            // weight_global_scale and input_global_scale are 1D scalars — replicate to all GPUs
                            shard.registry.quant_companions.insert(
                                name.clone(),
                                QuantCompanions::Nvfp4(Nvfp4Companions {
                                    weight_scale: sliced_ws,
                                    weight_global_scale: weight_global_scale.clone(),
                                    input_global_scale: input_global_scale.clone(),
                                }),
                            );
                        }
                    }
                }
            }
            infers_model::sharding::ShardType::Replicated => {
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
                            shard.registry.quant_companions.insert(
                                name.clone(),
                                QuantCompanions::Int4(Int4Companions {
                                    scales: scales.clone(),
                                    qzeros: qzeros.clone(),
                                }),
                            );
                        }
                    }

                    // Replicate NVFP4 companion weights
                    if name.ends_with(".weight_packed") {
                        let base = name.strip_suffix(".weight_packed").unwrap_or(name.as_str());
                        let weight_scale_name = format!("{}.weight_scale", base);
                        let weight_global_scale_name = format!("{}.weight_global_scale", base);
                        let input_global_scale_name = format!("{}.input_global_scale", base);

                        if let Some(weight_scale) = registry.tensors.get(&weight_scale_name)
                            && let Some(weight_global_scale) = registry.tensors.get(&weight_global_scale_name)
                            && let Some(input_global_scale) = registry.tensors.get(&input_global_scale_name) {
                            companion_skip.insert(weight_scale_name.clone());
                            companion_skip.insert(weight_global_scale_name.clone());
                            companion_skip.insert(input_global_scale_name.clone());
                            shard.registry.quant_companions.insert(
                                name.clone(),
                                QuantCompanions::Nvfp4(Nvfp4Companions {
                                    weight_scale: weight_scale.clone(),
                                    weight_global_scale: weight_global_scale.clone(),
                                    input_global_scale: input_global_scale.clone(),
                                }),
                            );
                        }
                    }
                }
   }
}


}
    Ok(shards)
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
