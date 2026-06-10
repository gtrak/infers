//! Decode path: single-token forward pass.
//!
//! Embed token → layer loop → norm → LM head → sample

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream};
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_cuda::nccl::NcclCommunicator;
use infers_model::{LayerType, ModelConfig, WeightData, WeightRegistry};

use crate::add;
use crate::attention::{self, KvCache};
use crate::embedding;
use crate::gdn::{self, GdnState};
use crate::mlp;
use crate::norm;
use crate::sample;

/// Convert [WeightData] bytes to [Vec<bf16>] and upload to GPU.
fn upload_weight(
    stream: &Arc<CudaStream>,
    weight: &WeightData,
) -> Result<CudaSlice<bf16>> {
    let bf16_vec: Vec<bf16> = {
        let count = weight.data.len() / 2;
        let mut v = Vec::with_capacity(count);
        for chunk in weight.data.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            v.push(bf16::from_bits(bits));
        }
        v
    };
    stream
        .clone_htod(&bf16_vec)
        .map_err(|e| anyhow::anyhow!("Failed to upload weight '{}': {}", weight.name, e))
}

/// Kernel handles needed for the decode pass.
pub struct DecodeKernels {
    pub rmsnorm: CudaFunction,
    pub silu_glu: CudaFunction,
    pub rope: CudaFunction,
    pub embedding: CudaFunction,
    pub add: CudaFunction,
    pub argmax: CudaFunction,
    pub softmax: CudaFunction,
    pub kv_cache_write: CudaFunction,
    pub gdn_update: CudaFunction,
}

/// Execute a single-token decode step.
///
/// 1. Embed the current token
/// 2. For each layer:
///    - RMSNorm (norm1)
///    - If GDN: GDN recurrent step
///    - If FullAttention: single-token attention over cached KV
///    - Residual add
///    - RMSNorm (norm2)
///    - SwiGLU MLP forward
///    - Residual add
/// 3. Final RMSNorm
/// 4. LM head projection
/// 5. Sample next token
///
/// # Arguments
/// * `gemm` — cuBLASLt engine
/// * `stream` — CUDA stream
/// * `kernels` — Loaded CUDA kernels
/// * `nccl` — NCCL communicator (unused for single GPU)
/// * `config` — Model configuration
/// * `weights` — Weight registry
/// * `token_id` — Current token to process
/// * `position` — Position index for RoPE
/// * `kv_caches` — KV cache state for each attention layer
/// * `gdn_states` — GDN recurrent state for each GDN layer
///
/// # Returns
/// Sampled token ID for the next generated token
pub fn decode(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    kernels: &DecodeKernels,
    _nccl: &NcclCommunicator,
    config: &ModelConfig,
    weights: &WeightRegistry,
    token_id: u32,
    position: u32,
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

    // =========================================================================
    // Phase 1: Embed single token
    // =========================================================================

    let embed_weight = weights.embedding.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
    let embed_table = upload_weight(stream, embed_weight)?;

    let mut hidden = embedding::embed_tokens(
        stream,
        &kernels.embedding,
        &[token_id],
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
        let norm1_weight = upload_weight(stream, &layer.norm1)?;
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
                gdn::decode_forward(
                    gemm,
                    stream,
                    &kernels.gdn_update,
                    gdn_weights,
                    &norm1_out,
                    &mut gdn_states[layer_idx],
                    hidden_size,
                )?
            }
            LayerType::FullAttention => {
                let attn_weights = layer.attn.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
                attention::decode_forward(
                    gemm,
                    stream,
                    &kernels.softmax,
                    &kernels.kv_cache_write,
                    &kernels.rope,
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
                )?
            }
        };

        // --- Residual add (attention/GDN output + hidden) ---
        hidden = add::add(stream, &kernels.add, &hidden, &attn_or_gdn_out)?;

        // --- Norm2 (pre-MLP) ---
        let norm2_weight = upload_weight(stream, &layer.norm2)?;
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
        let gate_proj = upload_weight(stream, &mlp_weights.gate_proj)?;
        let up_proj = upload_weight(stream, &mlp_weights.up_proj)?;
        let down_proj = upload_weight(stream, &mlp_weights.down_proj)?;
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
    let final_norm_gpu = upload_weight(stream, final_norm_weight)?;
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
    let lm_head_gpu = upload_weight(stream, lm_head_weight)?;

    // LM head: logits = hidden @ lm_head^T → [1 × vocab_size]
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

    // =========================================================================
    // Phase 4: Sample token (greedy — argmax of logits)
    // =========================================================================

    // Download bf16 logits, convert to f32 for argmax kernel
    let logits_bf16: Vec<bf16> = stream
        .clone_dtoh(&logits)
        .map_err(|e| anyhow::anyhow!("Failed to download logits from device: {e}"))?;
    let logits_f32: Vec<f32> = logits_bf16.iter().map(|v| v.to_f32()).collect();

    // Upload f32 logits for argmax kernel
    let logits_f32_gpu = stream
        .clone_htod(&logits_f32)
        .map_err(|e| anyhow::anyhow!("Failed to upload f32 logits: {e}"))?;

    sample::greedy_sample(stream, &kernels.argmax, &logits_f32_gpu)
}
