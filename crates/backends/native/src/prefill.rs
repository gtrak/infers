//! Prefill path: full forward pass over a prompt sequence.
//!
//! Tokenize → embed → layer loop → final norm → LM head → sample

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaStream, OxideKernels};
use infers_cuda::gemm::GemmEngine;
use infers_cuda::nccl::NcclCommunicator;
use infers_model::{LayerType, ModelConfig, WeightRegistry};

use crate::add;
use crate::attention::{self, KvCache};
use crate::embedding;
use crate::gdn::{self, GdnState};
use crate::gpu_cache::GpuWeightCache;
use crate::norm;
use crate::sample;
use crate::sync;

// @lat: [[lat.md/lat#Forward Engine#Prefill Path]]
/// Kernel handles needed for the prefill pass.
pub struct PrefillKernels {
    /// Oxide bridge for all kernel launches.
    pub oxide: Arc<OxideKernels>,
}

// @lat: [[lat.md/lat#Forward Engine#Prefill Path]]
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
///
/// # Arguments
/// * `nccl` — NCCL communicator for TP=2 multi-GPU all-reduce
/// * `group_size` — INT4 quantization group size (typically 128)
pub fn prefill(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    kernels: &PrefillKernels,
    nccl: &NcclCommunicator,
    config: &ModelConfig,
    weights: &WeightRegistry,
    cache: &GpuWeightCache,
    token_ids: &[u32],
    kv_caches: &mut Vec<KvCache>,
    gdn_states: &mut Vec<GdnState>,
    group_size: usize,
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
    let embed_table = cache.get_bf16(&embed_weight.name)
        .ok_or_else(|| anyhow::anyhow!("Embedding weight '{}' not in cache", embed_weight.name))?;

    let mut hidden = embedding::embed_tokens(
        stream,
        &kernels.oxide,
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
        let norm1_weight = cache.get_bf16(&layer.norm1.name)
            .ok_or_else(|| anyhow::anyhow!("Norm1 weight '{}' not in cache", layer.norm1.name))?;
        let norm1_out = norm::rms_norm(
            stream,
            &kernels.oxide,
            &hidden,
            &norm1_weight,
            rms_norm_eps,
            hidden_size,
        )?;

        // --- Attention / GDN dispatch ---
        let mut attn_or_gdn_out = match layer_type {
            LayerType::GatedDeltaNet => {
                let gdn_weights = layer.gdn.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("GDN weights not found for layer {}", layer_idx))?;
              gdn::forward(
                     gemm,
        stream,
                      &kernels.oxide,
                     gdn_weights,
                    &norm1_out,
                    &mut gdn_states[layer_idx],
                    hidden_size,
                    config,
                    group_size,
                    cache,
                    layer_idx,
                    0, // gpu_idx: single-GPU prefill path
                    &crate::probe::ProbeConfig::disabled(),
                )?
            }
            LayerType::FullAttention => {
                let attn_weights = layer.attn.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
                attention::forward(
                    gemm,
             stream,
                     &kernels.oxide,
                    attn_weights,
                    &norm1_out,
                    &mut kv_caches[layer_idx],
                    &positions,
                    hidden_size,
                    head_dim,
                    num_heads,
                    num_kv_heads,
                    max_seq_len,
                    rope_theta,
                    partial_rotary_factor,
                    rms_norm_eps,
                    group_size,
                    cache,
                    config.attn_output_gate,
                )?
            }
        };

        // --- All-reduce after row-parallel projection (TP=2) ---
        sync::all_reduce_attention(nccl, stream, &mut attn_or_gdn_out)?;

        // --- Residual add (attention/GDN output + hidden) ---
        hidden = add::add(stream, &kernels.oxide, &hidden, &attn_or_gdn_out)?;

        // --- Norm2 (pre-MLP) ---
        let norm2_weight = cache.get_bf16(&layer.norm2.name)
            .ok_or_else(|| anyhow::anyhow!("Norm2 weight '{}' not in cache", layer.norm2.name))?;
        let norm2_out = norm::rms_norm(
            stream,
            &kernels.oxide,
            &hidden,
            &norm2_weight,
            rms_norm_eps,
            hidden_size,
        )?;

        // --- MLP (INT4-aware) ---
        let mlp_weights = &layer.mlp;

        // gate = GEMM(norm2_out, gate_proj)  [seq_len × intermediate_size]
        let gate_size = seq_len * intermediate_size;
        let mut gate = stream
            .alloc_zeros::<bf16>(gate_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate gate buffer: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, &kernels.oxide, stream,
            cache, &mlp_weights.gate_proj.name, &norm2_out, &mut gate,
            seq_len, intermediate_size, hidden_size, group_size,
        )?;

        // up = GEMM(norm2_out, up_proj)  [seq_len × intermediate_size]
        let mut up = stream
            .alloc_zeros::<bf16>(gate_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate up buffer: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, &kernels.oxide, stream,
            cache, &mlp_weights.up_proj.name, &norm2_out, &mut up,
            seq_len, intermediate_size, hidden_size, group_size,
        )?;

        // silu_out = SiLU(gate) ⊗ up
        let mut silu_out = stream
            .alloc_zeros::<bf16>(gate_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate silu_out buffer: {e}"))?;
        kernels.oxide.launch_silu_glu_bf16(stream, &up, &gate, &mut silu_out, gate_size as u32)
            .map_err(|e| anyhow::anyhow!("SiLU+GLU kernel launch failed: {e}"))?;

        // output = GEMM(silu_out, down_proj^T)  [seq_len × hidden_size]
        let output_size = seq_len * hidden_size;
        let mut mlp_out = stream
            .alloc_zeros::<bf16>(output_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate MLP output: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, &kernels.oxide, stream,
            cache, &mlp_weights.down_proj.name, &silu_out, &mut mlp_out,
            seq_len, hidden_size, intermediate_size, group_size,
        )?;

        // --- All-reduce after row-parallel MLP down projection (TP=2) ---
        sync::all_reduce_mlp(nccl, stream, &mut mlp_out)?;

        // --- Residual add (MLP output + hidden) ---
        hidden = add::add(stream, &kernels.oxide, &hidden, &mlp_out)?;
    }

    // =========================================================================
    // Phase 3: Final norm + LM head
    // =========================================================================

    let final_norm_weight = weights.norm.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Final norm weights not found"))?;
    let final_norm_gpu = cache.get_bf16(&final_norm_weight.name)
        .ok_or_else(|| anyhow::anyhow!("Final norm weight '{}' not in cache", final_norm_weight.name))?;
    let hidden = norm::rms_norm(
        stream,
        &kernels.oxide,
        &hidden,
        &final_norm_gpu,
        rms_norm_eps,
        hidden_size,
    )?;

    let lm_head_weight = weights.lm_head.as_ref()
        .or_else(|| weights.embedding.as_ref())
        .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;

    // LM head: logits = hidden @ lm_head^T → [seq_len × vocab_size]
    let logits_size = seq_len * config.vocab_size;
    let mut logits = stream
        .alloc_zeros::<bf16>(logits_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate logits buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, &kernels.oxide, stream,
        cache, &lm_head_weight.name, &hidden, &mut logits,
        seq_len, config.vocab_size, hidden_size, group_size,
    )?;

    // =========================================================================
    // Phase 4: Sample first token (greedy — argmax of last token's logits)
    // =========================================================================

    // Extract last row's logits on GPU directly (no CPU round-trip)
    let last_row_start = (seq_len - 1) * config.vocab_size;
    let last_row_logits = logits.slice(last_row_start..last_row_start + config.vocab_size);

    // Argmax directly on BF16 logits on GPU
        let sampled = sample::greedy_sample_bf16(stream, &kernels.oxide, &last_row_logits)?;

    Ok(sampled)
}
