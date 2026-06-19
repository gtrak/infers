//! GPU-resident weight cache for cached dequantized tensors.
//!
//! Holds per-GPU caches of weights in either BF16 or INT4-quantized form,
//! keyed by tensor name.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use half::{bf16, f16};
use infers_cuda::{CudaSlice, CudaStream};
use infers_model::{Int4Companions, WeightData, WeightDtype, WeightRegistry};

/// A weight stored on the GPU, either as raw BF16/FP16/FP32 or INT4 quantized.
pub enum CachedWeight {
    /// BF16/FP16/FP32 weight uploaded as CudaSlice<bf16>
    Bf16(CudaSlice<bf16>),
    /// INT4 quantized weight triplet: qweight (u32 packed) + scales (fp16) + qzeros (u32 packed)
    Int4(Int4GpuBuffers),
}

/// GPU buffers for an INT4 quantized weight tensor.
pub struct Int4GpuBuffers {
    pub qweight: CudaSlice<u32>,
    pub scales: CudaSlice<f16>,
    pub qzeros: CudaSlice<u32>,
    /// Original shape of the INT4 weight tensor, used by GEMM dispatch to
    /// determine transposition at call time (depends on the K dimension).
    pub shape: Vec<usize>,
}

/// Per-GPU cache of dequantized, GPU-resident weight buffers.
/// All weights for one GPU shard, keyed by tensor name.
pub struct GpuWeightCache {
    weights: HashMap<String, CachedWeight>,
}

impl GpuWeightCache {
    /// General lookup by tensor name.
    pub fn get(&self, name: &str) -> Option<&CachedWeight> {
        self.weights.get(name)
    }

    /// Lookup BF16 weight. Returns None if the cached weight is INT4.
    pub fn get_bf16(&self, name: &str) -> Option<&CudaSlice<bf16>> {
        match self.weights.get(name)? {
            CachedWeight::Bf16(slice) => Some(slice),
            CachedWeight::Int4(_) => None,
        }
    }

    /// Lookup INT4 weight. Returns None if the cached weight is BF16.
    pub fn get_int4(&self, name: &str) -> Option<&Int4GpuBuffers> {
        match self.weights.get(name)? {
            CachedWeight::Bf16(_) => None,
            CachedWeight::Int4(buffers) => Some(buffers),
        }
    }

    /// Number of cached weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Upload all weights from a `WeightRegistry` to GPU memory.
    ///
    /// Iterates over every tensor in the registry, classifying each as either
    /// BF16/FP16/FP32 (stored as `CachedWeight::Bf16`) or INT4 packed
    /// (stored as `CachedWeight::Int4` with shape for transposition detection).
    /// Unsupported dtypes are skipped with a warning.
    pub fn new(
        stream: &Arc<CudaStream>,
        registry: &WeightRegistry,
    ) -> Result<Self> {
        let mut weights = HashMap::new();

        // 1. Upload embedding table (if present)
        if let Some(embed) = &registry.embedding {
            upload_and_cache(stream, embed, &registry.int4_companions, &mut weights)?;
        }

        // 2. Upload LM head (if present, may be tied with embedding)
        if let Some(lm_head) = &registry.lm_head {
            upload_and_cache(stream, lm_head, &registry.int4_companions, &mut weights)?;
        }

        // 3. Upload final norm (if present)
        if let Some(norm) = &registry.norm {
            upload_and_cache(stream, norm, &registry.int4_companions, &mut weights)?;
        }

        // 4. Upload MTP head weights (if present)
        if let Some(mtp) = &registry.mtp {
            upload_and_cache(stream, &mtp.pre_fc_norm_embedding, &registry.int4_companions, &mut weights)?;
            upload_and_cache(stream, &mtp.pre_fc_norm_hidden, &registry.int4_companions, &mut weights)?;
            upload_and_cache(stream, &mtp.fc, &registry.int4_companions, &mut weights)?;
            if let Some(embed) = &mtp.embed_tokens {
                upload_and_cache(stream, embed, &registry.int4_companions, &mut weights)?;
            }
            for layer in &mtp.layers {
                upload_and_cache(stream, &layer.norm1, &registry.int4_companions, &mut weights)?;
                upload_and_cache(stream, &layer.norm2, &registry.int4_companions, &mut weights)?;

                if let Some(attn) = &layer.attn {
                    upload_and_cache(stream, &attn.q_proj, &registry.int4_companions, &mut weights)?;
                    upload_and_cache(stream, &attn.k_proj, &registry.int4_companions, &mut weights)?;
                    upload_and_cache(stream, &attn.v_proj, &registry.int4_companions, &mut weights)?;
                    upload_and_cache(stream, &attn.o_proj, &registry.int4_companions, &mut weights)?;
                    if let Some(q_norm) = &attn.q_norm {
                        upload_and_cache(stream, q_norm, &registry.int4_companions, &mut weights)?;
                    }
                    if let Some(k_norm) = &attn.k_norm {
                        upload_and_cache(stream, k_norm, &registry.int4_companions, &mut weights)?;
                    }
                }

                if let Some(gdn) = &layer.gdn {
                    upload_and_cache(stream, &gdn.in_proj_a, &registry.int4_companions, &mut weights)?;
                    upload_and_cache(stream, &gdn.in_proj_b, &registry.int4_companions, &mut weights)?;
                    upload_and_cache(stream, &gdn.out_proj_weight, &registry.int4_companions, &mut weights)?;
                    if let Some(x_proj) = &gdn.x_proj_weight {
                        upload_and_cache(stream, x_proj, &registry.int4_companions, &mut weights)?;
                    }
                    if let Some(dt_proj) = &gdn.dt_proj_weight {
                        upload_and_cache(stream, dt_proj, &registry.int4_companions, &mut weights)?;
                    }
                    if let Some(z) = &gdn.in_proj_z {
                        upload_and_cache(stream, z, &registry.int4_companions, &mut weights)?;
                    }
                    if let Some(qkv) = &gdn.in_proj_qkv {
                        upload_and_cache(stream, qkv, &registry.int4_companions, &mut weights)?;
                    }
                    if let Some(a_log) = &gdn.a_log {
                        upload_and_cache(stream, a_log, &registry.int4_companions, &mut weights)?;
                    }
                    if let Some(dt_bias) = &gdn.dt_bias {
                        upload_and_cache(stream, dt_bias, &registry.int4_companions, &mut weights)?;
                    }
                    if let Some(norm) = &gdn.norm {
                        upload_and_cache(stream, norm, &registry.int4_companions, &mut weights)?;
                    }
                    upload_and_cache(stream, &gdn.conv1d_weight, &registry.int4_companions, &mut weights)?;
                }

                upload_and_cache(stream, &layer.mlp.gate_proj, &registry.int4_companions, &mut weights)?;
                upload_and_cache(stream, &layer.mlp.up_proj, &registry.int4_companions, &mut weights)?;
                upload_and_cache(stream, &layer.mlp.down_proj, &registry.int4_companions, &mut weights)?;
            }
            upload_and_cache(stream, &mtp.norm, &registry.int4_companions, &mut weights)?;
        }

        // 5. Upload all layer weights
        for layer in &registry.layers {
            // Norm1 and Norm2 (always present)
            upload_and_cache(stream, &layer.norm1, &registry.int4_companions, &mut weights)?;
            upload_and_cache(stream, &layer.norm2, &registry.int4_companions, &mut weights)?;

            // Attention weights (if present)
            if let Some(attn) = &layer.attn {
                upload_and_cache(stream, &attn.q_proj, &registry.int4_companions, &mut weights)?;
                upload_and_cache(stream, &attn.k_proj, &registry.int4_companions, &mut weights)?;
                upload_and_cache(stream, &attn.v_proj, &registry.int4_companions, &mut weights)?;
                upload_and_cache(stream, &attn.o_proj, &registry.int4_companions, &mut weights)?;
                if let Some(q_norm) = &attn.q_norm {
                    upload_and_cache(stream, q_norm, &registry.int4_companions, &mut weights)?;
                }
                if let Some(k_norm) = &attn.k_norm {
                    upload_and_cache(stream, k_norm, &registry.int4_companions, &mut weights)?;
                }
            }

            // GDN weights (if present)
            if let Some(gdn) = &layer.gdn {
                upload_and_cache(stream, &gdn.in_proj_a, &registry.int4_companions, &mut weights)?;
                upload_and_cache(stream, &gdn.in_proj_b, &registry.int4_companions, &mut weights)?;
                upload_and_cache(stream, &gdn.out_proj_weight, &registry.int4_companions, &mut weights)?;
                if let Some(x_proj) = &gdn.x_proj_weight {
                    upload_and_cache(stream, x_proj, &registry.int4_companions, &mut weights)?;
                }
                if let Some(dt_proj) = &gdn.dt_proj_weight {
                    upload_and_cache(stream, dt_proj, &registry.int4_companions, &mut weights)?;
                }
                if let Some(z) = &gdn.in_proj_z {
                    upload_and_cache(stream, z, &registry.int4_companions, &mut weights)?;
                }
                if let Some(qkv) = &gdn.in_proj_qkv {
                    upload_and_cache(stream, qkv, &registry.int4_companions, &mut weights)?;
                }
                // SSM parameters (a_log, dt_bias) — always BF16, small
                if let Some(a_log) = &gdn.a_log {
                    upload_and_cache(stream, a_log, &registry.int4_companions, &mut weights)?;
                }
                if let Some(dt_bias) = &gdn.dt_bias {
                    upload_and_cache(stream, dt_bias, &registry.int4_companions, &mut weights)?;
                }
                if let Some(norm) = &gdn.norm {
                    upload_and_cache(stream, norm, &registry.int4_companions, &mut weights)?;
                }
                // conv1d_weight is uploaded but currently not used in forward pass
                upload_and_cache(stream, &gdn.conv1d_weight, &registry.int4_companions, &mut weights)?;
            }

            // MLP weights (always present)
            upload_and_cache(stream, &layer.mlp.gate_proj, &registry.int4_companions, &mut weights)?;
            upload_and_cache(stream, &layer.mlp.up_proj, &registry.int4_companions, &mut weights)?;
            upload_and_cache(stream, &layer.mlp.down_proj, &registry.int4_companions, &mut weights)?;
        }

        Ok(Self { weights })
    }
}

/// Upload a single weight to GPU and cache it in the `weights` map.
///
/// Classifies the weight by dtype:
/// - **Int4Packed**: look up companions from `int4_companions`, call
///   `upload_int4_weight`, store as `CachedWeight::Int4` with shape info.
/// - **Bf16 / Fp16 / Fp32**: call `upload_weight`, store as `CachedWeight::Bf16`.
/// - **Other (Nvfp4, Other)**: skip with a warning log.
fn upload_and_cache(
    stream: &Arc<CudaStream>,
    weight: &WeightData,
    int4_companions: &HashMap<String, Int4Companions>,
    weights: &mut HashMap<String, CachedWeight>,
) -> Result<()> {
    match weight.dtype {
        WeightDtype::Int4Packed => {
            let companions = int4_companions
                .get(&weight.name)
                .ok_or_else(|| anyhow::anyhow!("INT4 companions not found for weight '{}'", weight.name))?;

            let (qweight_gpu, scales_gpu, qzeros_gpu) = crate::upload::upload_int4_weight(
                stream,
                weight,
                &companions.scales,
                &companions.qzeros,
            )?;

            weights.insert(
                weight.name.clone(),
                CachedWeight::Int4(Int4GpuBuffers {
                    qweight: qweight_gpu,
                    scales: scales_gpu,
                    qzeros: qzeros_gpu,
                    shape: weight.shape.clone(),
                }),
            );
        }
        WeightDtype::Bf16 | WeightDtype::Fp16 | WeightDtype::Fp32 => {
            let gpu_slice = crate::upload::upload_weight(stream, weight)?;
            weights.insert(weight.name.clone(), CachedWeight::Bf16(gpu_slice));
        }
        _ => {
            tracing::warn!(
                "Skipping weight '{}' with unsupported dtype {:?} during GPU cache build",
                weight.name,
                weight.dtype
            );
        }
    }
    Ok(())
}
