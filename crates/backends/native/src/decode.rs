//! Decode path: single-token forward pass.
//!
//! Embed token → layer loop → norm → LM head → sample

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, PushKernelArg};
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
    pub gdn_gated_delta_update: CudaFunction,
    pub conv1d_depthwise: CudaFunction,
    pub rms_norm_gated: CudaFunction,
    /// INT4 GEMM kernel for quantized weight dispatch.
    pub int4_gemm: CudaFunction,
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
/// * `_nccl` — NCCL communicator (reserved for future TP=2 multi-GPU all-reduce)
/// * `config` — Model configuration
/// * `weights` — Weight registry
/// * `token_id` — Current token to process
/// * `position` — Position index for RoPE
/// * `kv_caches` — KV cache state for each attention layer
/// * `gdn_states` — GDN recurrent state for each GDN layer
/// * `group_size` — INT4 quantization group size (typically 128)
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
    cache: &GpuWeightCache,
    token_id: u32,
    position: u32,
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

    // =========================================================================
    // Phase 1: Embed single token
    // =========================================================================

    let embed_weight = weights.embedding.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Embedding weights not found"))?;
    let embed_table = cache.get_bf16(&embed_weight.name)
        .ok_or_else(|| anyhow::anyhow!("Embedding weight '{}' not in cache", embed_weight.name))?;

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
        let norm1_weight = cache.get_bf16(&layer.norm1.name)
            .ok_or_else(|| anyhow::anyhow!("Norm1 weight '{}' not in cache", layer.norm1.name))?;
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
                    &kernels.int4_gemm,
                    stream,
                    &kernels.gdn_gated_delta_update,
                    &kernels.conv1d_depthwise,
                    &kernels.rms_norm_gated,
                    gdn_weights,
                    &norm1_out,
                    &mut gdn_states[layer_idx],
                    hidden_size,
                    config,
                    group_size,
                    cache,
                )?
            }
            LayerType::FullAttention => {
                let attn_weights = layer.attn.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Attention weights not found for layer {}", layer_idx))?;
                attention::decode_forward(
                    gemm,
                    &kernels.int4_gemm,
                    stream,
                    &kernels.softmax,
                    &kernels.kv_cache_write,
                    &kernels.rope,
                    &kernels.rmsnorm,
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
                    group_size,
                    cache,
                )?
            }
        };

        // --- Residual add (attention/GDN output + hidden) ---
        hidden = add::add(stream, &kernels.add, &hidden, &attn_or_gdn_out)?;

        // --- Norm2 (pre-MLP) ---
        let norm2_weight = cache.get_bf16(&layer.norm2.name)
            .ok_or_else(|| anyhow::anyhow!("Norm2 weight '{}' not in cache", layer.norm2.name))?;
        let norm2_out = norm::rms_norm(
            stream,
            &kernels.rmsnorm,
            &hidden,
            &norm2_weight,
            rms_norm_eps,
            hidden_size,
        )?;

        // --- MLP (INT4-aware) ---
        let mlp_weights = &layer.mlp;

        // gate = GEMM(norm2_out, gate_proj)  [1 × intermediate_size]
        let gate_size = intermediate_size;
        let mut gate = stream
            .alloc_zeros::<bf16>(gate_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate gate buffer: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, &kernels.int4_gemm, stream,
            cache, &mlp_weights.gate_proj.name, &norm2_out, &mut gate,
            1, intermediate_size, hidden_size, group_size,
        )?;

        // up = GEMM(norm2_out, up_proj)  [1 × intermediate_size]
        let mut up = stream
            .alloc_zeros::<bf16>(gate_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate up buffer: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, &kernels.int4_gemm, stream,
            cache, &mlp_weights.up_proj.name, &norm2_out, &mut up,
            1, intermediate_size, hidden_size, group_size,
        )?;

        // silu_out = SiLU(gate) ⊗ up
        let mut silu_out = stream
            .alloc_zeros::<bf16>(gate_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate silu_out buffer: {e}"))?;
        {
            let elem_count_i32 = gate_size as i32;
            let config = infers_cuda::LaunchConfig {
                grid_dim: ((gate_size as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&kernels.silu_glu)
                    .arg(&gate)
                    .arg(&up)
                    .arg(&mut silu_out)
                    .arg(&elem_count_i32)
                    .launch(config)
                    .map_err(|e| anyhow::anyhow!("SiLU+GLU kernel launch failed: {e}"))?;
            }
        }

        // output = GEMM(silu_out, down_proj^T)  [1 × hidden_size]
        let mut mlp_out = stream
            .alloc_zeros::<bf16>(hidden_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate MLP output: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, &kernels.int4_gemm, stream,
            cache, &mlp_weights.down_proj.name, &silu_out, &mut mlp_out,
            1, hidden_size, intermediate_size, group_size,
        )?;

        // --- Residual add (MLP output + hidden) ---
        hidden = add::add(stream, &kernels.add, &hidden, &mlp_out)?;
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
        &kernels.rmsnorm,
        &hidden,
        &final_norm_gpu,
        rms_norm_eps,
        hidden_size,
    )?;

    let lm_head_weight = weights.lm_head.as_ref()
        .or_else(|| weights.embedding.as_ref())
        .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;

    // LM head: logits = hidden @ lm_head^T → [1 × vocab_size]
    let mut logits = stream
        .alloc_zeros::<bf16>(config.vocab_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate logits buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, &kernels.int4_gemm, stream,
        cache, &lm_head_weight.name, &hidden, &mut logits,
        1, config.vocab_size, hidden_size, group_size,
    )?;

    // =========================================================================
    // Phase 4: Sample token (greedy — argmax of logits)
    // =========================================================================

    // Argmax directly on BF16 logits on GPU (no CPU round-trip)
    sample::greedy_sample_bf16(stream, &kernels.argmax, &logits.as_view())
}

/// Execute a single-token decode step and return both the sampled token
/// and the final hidden state (pre-LM-head) for MTP speculative decoding.
///
/// Same as `decode()` but additionally returns the hidden state after
/// the final RMSNorm, before the LM head projection. This hidden state
/// is needed by the MTP head for draft token generation.
///
/// # Returns
/// `(sampled_token, hidden_state)` where `hidden_state` is the output
/// of the final RMSNorm (`[hidden_size]`), preserved before LM head projection.
pub fn decode_with_hidden(
    gemm: &mut GemmEngine,
    stream: &Arc<CudaStream>,
    kernels: &DecodeKernels,
    _nccl: &NcclCommunicator,
    config: &ModelConfig,
    weights: &WeightRegistry,
    cache: &GpuWeightCache,
    token_id: u32,
    position: u32,
    kv_caches: &mut Vec<KvCache>,
    gdn_states: &mut Vec<GdnState>,
    group_size: usize,
) -> Result<(u32, CudaSlice<bf16>)> {
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
    let embed_table = cache.get_bf16(&embed_weight.name)
        .ok_or_else(|| anyhow::anyhow!("Embedding weight '{}' not in cache", embed_weight.name))?;

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
        let norm1_weight = cache.get_bf16(&layer.norm1.name)
            .ok_or_else(|| anyhow::anyhow!("Norm1 weight '{}' not in cache", layer.norm1.name))?;
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
                    .ok_or_else(|| anyhow::anyhow!("GDN weights not found for MTP layer {}", layer_idx))?;
                gdn::decode_forward(
                    gemm,
                    &kernels.int4_gemm,
                    stream,
                    &kernels.gdn_gated_delta_update,
                    &kernels.conv1d_depthwise,
                    &kernels.rms_norm_gated,
                    gdn_weights,
                    &norm1_out,
                    &mut gdn_states[layer_idx],
                    hidden_size,
                    config,
                    group_size,
                    cache,
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
                    group_size,
                    cache,
                )?
            }
        };

        // --- Residual add (attention/GDN output + hidden) ---
        hidden = add::add(stream, &kernels.add, &hidden, &attn_or_gdn_out)?;

        // --- Norm2 (pre-MLP) ---
        let norm2_weight = cache.get_bf16(&layer.norm2.name)
            .ok_or_else(|| anyhow::anyhow!("Norm2 weight '{}' not in cache", layer.norm2.name))?;
        let norm2_out = norm::rms_norm(
            stream,
            &kernels.rmsnorm,
            &hidden,
            &norm2_weight,
            rms_norm_eps,
            hidden_size,
        )?;

        // --- MLP (INT4-aware) ---
        let mlp_weights = &layer.mlp;

        // gate = GEMM(norm2_out, gate_proj)  [1 × intermediate_size]
        let gate_size = intermediate_size;
        let mut gate = stream
            .alloc_zeros::<bf16>(gate_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate gate buffer: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, &kernels.int4_gemm, stream,
            cache, &mlp_weights.gate_proj.name, &norm2_out, &mut gate,
            1, intermediate_size, hidden_size, group_size,
        )?;

        // up = GEMM(norm2_out, up_proj)  [1 × intermediate_size]
        let mut up = stream
            .alloc_zeros::<bf16>(gate_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate up buffer: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, &kernels.int4_gemm, stream,
            cache, &mlp_weights.up_proj.name, &norm2_out, &mut up,
            1, intermediate_size, hidden_size, group_size,
        )?;

        // silu_out = SiLU(gate) ⊗ up
        let mut silu_out = stream
            .alloc_zeros::<bf16>(gate_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate silu_out buffer: {e}"))?;
        {
            let elem_count_i32 = gate_size as i32;
            let config = infers_cuda::LaunchConfig {
                grid_dim: ((gate_size as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream
                    .launch_builder(&kernels.silu_glu)
                    .arg(&gate)
                    .arg(&up)
                    .arg(&mut silu_out)
                    .arg(&elem_count_i32)
                    .launch(config)
                    .map_err(|e| anyhow::anyhow!("SiLU+GLU kernel launch failed: {e}"))?;
            }
        }

        // output = GEMM(silu_out, down_proj^T)  [1 × hidden_size]
        let mut mlp_out = stream
            .alloc_zeros::<bf16>(hidden_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate MLP output: {e}"))?;
        crate::gemm_dispatch::gemm_projection_cached(
            gemm, &kernels.int4_gemm, stream,
            cache, &mlp_weights.down_proj.name, &silu_out, &mut mlp_out,
            1, hidden_size, intermediate_size, group_size,
        )?;

        // --- Residual add (MLP output + hidden) ---
        hidden = add::add(stream, &kernels.add, &hidden, &mlp_out)?;
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
        &kernels.rmsnorm,
        &hidden,
        &final_norm_gpu,
        rms_norm_eps,
        hidden_size,
    )?;

    // Preserve hidden state before LM head projection for MTP
    let mtp_hidden = hidden.clone();

    let lm_head_weight = weights.lm_head.as_ref()
        .or_else(|| weights.embedding.as_ref())
        .ok_or_else(|| anyhow::anyhow!("Neither LM head nor embedding weights found"))?;

    // LM head: logits = hidden @ lm_head^T → [1 × vocab_size]
    let mut logits = stream
        .alloc_zeros::<bf16>(config.vocab_size)
        .map_err(|e| anyhow::anyhow!("Failed to allocate logits buffer: {e}"))?;
    crate::gemm_dispatch::gemm_projection_cached(
        gemm, &kernels.int4_gemm, stream,
        cache, &lm_head_weight.name, &hidden, &mut logits,
        1, config.vocab_size, hidden_size, group_size,
    )?;

    // =========================================================================
    // Phase 4: Sample token (greedy — argmax of logits)
    // =========================================================================

    // Argmax directly on BF16 logits on GPU (no CPU round-trip)
    let sampled_token = sample::greedy_sample_bf16(stream, &kernels.argmax, &logits.as_view())?;

    Ok((sampled_token, mtp_hidden))
}
