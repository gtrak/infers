//! MTP integration helpers for the native backend.
//!
//! Provides `forward_layer_pass` and `full_forward_logits` functions
//! that the ForwardEngine uses to build MTP operation callbacks.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::{CudaSlice, CudaStream};
use infers_model::{LayerType, LayerWeights, ModelConfig, WeightRegistry};

use crate::add;
use crate::attention::{self, KvCache};
use crate::embedding;
use crate::gdn::{self, GdnState};
use crate::mlp;
use crate::norm;
use crate::decode::DecodeKernels;

/// Run a single full decoder layer forward pass.
///
/// Performs: norm1 → attention/GDN dispatch → residual → norm2 → MLP → residual
///
/// This is equivalent to one iteration of the decode layer loop.
///
/// # Arguments
/// * `layer` — Layer weights (norms, attention/GDN, MLP)
/// * `input` — Input hidden state `[1 × hidden_size]`
/// * `gemm` — cuBLASLt engine
/// * `stream` — CUDA stream
/// * `kernels` — Loaded kernel handles
/// * `config` — Model configuration
/// * `kv_caches` — Per-layer KV caches
/// * `gdn_states` — Per-layer GDN states
/// * `position` — Current sequence position (for RoPE)
/// * `layer_idx` — Layer index
///
/// # Returns
/// Output hidden state `[1 × hidden_size]` after the full decoder layer
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_pass(
    layer: &LayerWeights,
    input: &CudaSlice<bf16>,
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    kernels: &DecodeKernels,
    config: &ModelConfig,
    kv_caches: &mut [KvCache],
    gdn_states: &mut [GdnState],
    position: u32,
    layer_idx: usize,
) -> Result<CudaSlice<bf16>> {
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let max_seq_len = config.max_position_embeddings;
    let rms_norm_eps = config.rms_norm_eps;
    let rope_theta = config.rope_theta;
    let partial_rotary_factor = config.partial_rotary_factor;

    // Norm1 (pre-attention/GDN)
    let norm1_weight = crate::upload::upload_weight(stream, &layer.norm1)?;
    let norm1_out = norm::rms_norm(
        stream,
        &kernels.rmsnorm,
        input,
        &norm1_weight,
        rms_norm_eps,
        hidden_size,
    )?;

    // Attention / GDN dispatch
    // MTP is optional/deferred: pass empty HashMap for int4_companions.
    let int4_companions = HashMap::<String, infers_model::Int4Companions>::new();
    let attn_or_gdn_out = match layer.layer_type {
        LayerType::GatedDeltaNet => {
            let gdn_weights = layer.gdn.as_ref()
                .ok_or_else(|| anyhow::anyhow!("GDN weights not found for MTP layer {}", layer_idx))?;
            gdn::decode_forward(
                gemm,
                &kernels.int4_gemm,
                stream,
                &kernels.gdn_update,
                gdn_weights,
                &norm1_out,
                &mut gdn_states[layer_idx],
                hidden_size,
                128, // group_size default
                &int4_companions,
            )?
        }
        LayerType::FullAttention => {
            let attn_weights = layer.attn.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Attention weights not found for MTP layer {}", layer_idx))?;
            attention::decode_forward(
                gemm,
                &kernels.int4_gemm,
                stream,
                &kernels.softmax,
                &kernels.kv_cache_write,
                &kernels.rope,
                &kernels.rmsnorm,
                &kernels.add,
                attn_weights,
                &norm1_out,
                &mut kv_caches[layer_idx],
                position,
                head_dim,
                num_heads,
                num_kv_heads,
                max_seq_len,
                rope_theta,
                partial_rotary_factor,
                rms_norm_eps,
                128, // group_size default
                &int4_companions,
            )?
        }
    };

    // Residual add (attention/GDN output + input)
    let mut hidden = add::add(stream, &kernels.add, input, &attn_or_gdn_out)?;

    // Norm2 (pre-MLP)
    let norm2_weight = crate::upload::upload_weight(stream, &layer.norm2)?;
    let norm2_out = norm::rms_norm(
        stream,
        &kernels.rmsnorm,
        &hidden,
        &norm2_weight,
        rms_norm_eps,
        hidden_size,
    )?;

    // MLP
    let mlp_weights = &layer.mlp;
    let gate_proj = crate::upload::upload_weight(stream, &mlp_weights.gate_proj)?;
    let up_proj = crate::upload::upload_weight(stream, &mlp_weights.up_proj)?;
    let down_proj = crate::upload::upload_weight(stream, &mlp_weights.down_proj)?;
    let mlp_out = mlp::mlp_forward(
        gemm,
        stream,
        &kernels.silu_glu,
        &gate_proj,
        &up_proj,
        &down_proj,
        &norm2_out,
        hidden_size,
        intermediate_size,
    )?;

    // Residual add (MLP output + hidden)
    hidden = add::add(stream, &kernels.add, &hidden, &mlp_out)?;

    Ok(hidden)
}

/// Run a full forward pass for a single token, returning LM head logits.
///
/// Embed → all layers → final norm → LM head → logits
///
/// # Arguments
/// * `token_id` — Token to process
/// * `gemm` — cuBLASLt engine
/// * `stream` — CUDA stream
/// * `kernels` — Loaded kernel handles
/// * `_nccl` — NCCL communicator
/// * `config` — Model configuration
/// * `weights` — Weight registry
/// * `kv_caches` — Per-layer KV caches
/// * `gdn_states` — Per-layer GDN states
/// * `position` — Current sequence position
///
/// # Returns
/// LM head logits `[vocab_size]`
#[allow(clippy::too_many_arguments)]
pub fn full_forward_logits(
    token_id: u32,
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    kernels: &DecodeKernels,
    _nccl: &NcclCommunicator,
    config: &ModelConfig,
    weights: &WeightRegistry,
    kv_caches: &mut [KvCache],
    gdn_states: &mut [GdnState],
    position: u32,
) -> Result<CudaSlice<bf16>> {
    let hidden_size = config.hidden_size;
    let num_layers = config.num_hidden_layers;

    // Embed single token
    let embed_weight = weights.embedding.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
    let embed_table = crate::upload::upload_weight(stream, embed_weight)?;

    let mut hidden = embedding::embed_tokens(
        stream,
        &kernels.embedding,
        &[token_id],
        &embed_table,
        hidden_size,
        config.vocab_size,
    )?;

    // Layer loop
    for layer_idx in 0..num_layers {
        hidden = forward_layer_pass(
            &weights.layers[layer_idx],
            &hidden,
            gemm,
            stream,
            kernels,
            config,
            kv_caches,
            gdn_states,
            position,
            layer_idx,
        )?;
    }

    // Final norm
    let final_norm_weight = weights.norm.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
    let final_norm_gpu = crate::upload::upload_weight(stream, final_norm_weight)?;
    let hidden = norm::rms_norm(
        stream,
        &kernels.rmsnorm,
        &hidden,
        &final_norm_gpu,
        config.rms_norm_eps,
        hidden_size,
    )?;

    // LM head projection
    // The `weights.lm_head` is optional (None when tied embeddings).
    // If None, use the embedding weight transposed.
    let lm_head_weight = weights.lm_head.as_ref()
        .or_else(|| weights.embedding.as_ref())
        .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;
    let lm_head_gpu = crate::upload::upload_weight(stream, lm_head_weight)?;

    let mut logits = stream
        .alloc_zeros::<bf16>(config.vocab_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate logits buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: 1,
            n: config.vocab_size,
            k: hidden_size,
            transa: true,
            transb: false,
            alpha: 1.0,
            beta: 0.0,
            lda: None,
            ldb: None,
            ldc: None,
            activation: None,
        },
        &hidden,
        &lm_head_gpu,
        &mut logits,
    )?;

    Ok(logits)
}
