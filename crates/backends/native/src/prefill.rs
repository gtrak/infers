//! Prefill path: full forward pass over a prompt sequence.
//!
//! Tokenize → embed → layer loop → final norm → LM head → sample

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaStream};
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_cuda::nccl::NcclCommunicator;
use infers_model::{LayerType, ModelConfig, WeightRegistry};

use crate::add;
use crate::attention::{self, KvCache};
use crate::embedding;
use crate::gdn::{self, GdnState};
use crate::mlp;
use crate::norm;
use crate::sample;

// @lat: [[lat.md/lat#Phase 4 Deliverables#Forward Engine#Prefill Path]]
/// Kernel handles needed for the prefill pass.
pub struct PrefillKernels {
    pub rmsnorm: CudaFunction,
    pub silu_glu: CudaFunction,
    pub rope: CudaFunction,
    pub embedding: CudaFunction,
    pub add: CudaFunction,
    pub argmax: CudaFunction,
    pub softmax: CudaFunction,
    pub kv_cache_write: CudaFunction,
    pub gdn_prefill: CudaFunction,
}

// @lat: [[lat.md/lat#Phase 4 Deliverables#Forward Engine#Prefill Path]]
/// Execute the full prefill forward pass.
///
/// 1. Embed all prompt tokens
/// 2. For each layer:
///    - RMSNorm (norm1)
///    - If GDN: GDN forward pass
///    - If FullAttention: attention forward pass + KV cache write
///    - Residual add
///    - RMSNorm (norm2)
///    - SwiGLU MLP forward
///    - Residual add
/// 3. Final RMSNorm
/// 4. LM head projection
/// 5. Sample first token
pub fn prefill(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    kernels: &PrefillKernels,
    _nccl: &NcclCommunicator,
    config: &ModelConfig,
    weights: &WeightRegistry,
    token_ids: &[u32],
    kv_caches: &mut Vec<KvCache>,
    gdn_states: &mut Vec<GdnState>,
) -> Result<u32> {
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let num_layers = config.num_hidden_layers;
    let max_seq_len = config.max_position_embeddings;
    let rms_norm_eps = config.rms_norm_eps;
    let rope_theta = config.rope_theta;
    let partial_rotary_factor = config.partial_rotary_factor;

    // Initialize caches/states if empty
    if kv_caches.is_empty() {
        kv_caches.resize_with(num_layers, KvCache::new);
    }
    if gdn_states.is_empty() {
        gdn_states.resize_with(num_layers, GdnState::new);
    }

    // Create position indices: 0, 1, 2, ..., seq_len-1
    let seq_len = token_ids.len();
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    // =========================================================================
    // Phase 1: Embed tokens
    // =========================================================================

    let embed_weight = weights.embedding.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
    let embed_table = crate::upload::upload_weight(stream, embed_weight)?;

    let mut hidden = embedding::embed_tokens(
        stream,
        &kernels.embedding,
        token_ids,
        &embed_table,
        hidden_size,
        config.vocab_size,
    )?;

    // =========================================================================
    // Phase 2: Layer loop
    // =========================================================================

    for layer_idx in 0..num_layers {
        let layer = &weights.layers[layer_idx];
        let layer_type = config.get_layer_type(layer_idx);

        // --- Norm1 (pre-attention/GDN) ---
        let norm1_weight = crate::upload::upload_weight(stream, &layer.norm1)?;
        let norm1_out = norm::rms_norm(
            stream,
            &kernels.rmsnorm,
            &hidden,
            &norm1_weight,
            rms_norm_eps,
            hidden_size,
        )?;

        // --- Attention / GDN dispatch ---
        let attn_or_gdn_out = match layer_type {
            LayerType::GatedDeltaNet => {
                let gdn_weights = layer.gdn.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("GDN weights not found for layer {}", layer_idx))?;
                gdn::forward(
                    gemm,
                    stream,
                    &kernels.gdn_prefill,
                    gdn_weights,
                    &norm1_out,
                    &mut gdn_states[layer_idx],
                    hidden_size,
                )?
            }
            LayerType::FullAttention => {
                let attn_weights = layer.attn.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
                attention::forward(
                    gemm,
                    stream,
                    &kernels.softmax,
                    &kernels.kv_cache_write,
                    &kernels.rope,
                    &kernels.add,
                    attn_weights,
                    &norm1_out,
                    &mut kv_caches[layer_idx],
                    &positions,
                    head_dim,
                    num_heads,
                    num_kv_heads,
                    max_seq_len,
                    rope_theta,
                    partial_rotary_factor,
                )?
            }
        };

        // --- Residual add (attention/GDN output + hidden) ---
        hidden = add::add(stream, &kernels.add, &hidden, &attn_or_gdn_out)?;

        // --- Norm2 (pre-MLP) ---
        let norm2_weight = crate::upload::upload_weight(stream, &layer.norm2)?;
        let norm2_out = norm::rms_norm(
            stream,
            &kernels.rmsnorm,
            &hidden,
            &norm2_weight,
            rms_norm_eps,
            hidden_size,
        )?;

        // --- MLP ---
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

        // --- Residual add (MLP output + hidden) ---
        hidden = add::add(stream, &kernels.add, &hidden, &mlp_out)?;
    }

    // =========================================================================
    // Phase 3: Final norm + LM head
    // =========================================================================

    let final_norm_weight = weights.norm.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
    let final_norm_gpu = crate::upload::upload_weight(stream, final_norm_weight)?;
    let hidden = norm::rms_norm(
        stream,
        &kernels.rmsnorm,
        &hidden,
        &final_norm_gpu,
        rms_norm_eps,
        hidden_size,
    )?;

    let lm_head_weight = weights.lm_head.as_ref()
        .ok_or_else(|| anyhow::anyhow!("LM head weights not found"))?;
    let lm_head_gpu = crate::upload::upload_weight(stream, lm_head_weight)?;

    // LM head: logits = hidden @ lm_head^T → [seq_len × vocab_size]
    let logits_size = seq_len * config.vocab_size;
    let mut logits = stream
        .alloc_zeros::<bf16>(logits_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate logits buffer: {e}"))?;
    gemm.matmul_bf16(
        &GemmConfig {
            m: seq_len,
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

    // =========================================================================
    // Phase 4: Sample first token (greedy — argmax of last token's logits)
    // =========================================================================

    // Download bf16 logits, extract last row, convert to f32 for argmax kernel
    let logits_host: Vec<bf16> = stream
        .clone_dtoh(&logits)
        .map_err(|e| anyhow::anyhow!("Failed to download logits from device: {e}"))?;
    let last_row_start = (seq_len - 1) * config.vocab_size;
    let last_row_bf16 =
        &logits_host[last_row_start..last_row_start + config.vocab_size];
    let last_row_f32: Vec<f32> = last_row_bf16.iter().map(|v| v.to_f32()).collect();

    // Upload f32 logits for argmax kernel
    let logits_f32_gpu = stream
        .clone_htod(&last_row_f32)
        .map_err(|e| anyhow::anyhow!("Failed to upload f32 logits: {e}"))?;

    sample::greedy_sample(stream, &kernels.argmax, &logits_f32_gpu)
}
