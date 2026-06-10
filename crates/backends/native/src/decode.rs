//! Decode path: single-token forward pass.
//!
//! Embed token → layer loop → norm → LM head → sample

use std::sync::Arc;

use anyhow::Result;
use infers_cuda::CudaStream;
use infers_cuda::gemm::GemmEngine;
use infers_model::LayerType;

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
/// * `token_id` — Current token to process
/// * `position` — Position index for RoPE
/// * `num_layers` — Total transformer layers
/// * `get_layer_type` — Closure that returns layer type by index
///
/// # Returns
/// Sampled token ID for the next generated token
pub fn decode(
    _gemm: &mut GemmEngine,
    _stream: &Arc<CudaStream>,
    _token_id: u32,
    _position: u32,
    _num_layers: usize,
    _get_layer_type: impl Fn(usize) -> LayerType,
) -> Result<u32> {
    // Phase 1: Embed single token
    // let hidden = embedding::embed_tokens(stream, &embed_kernel, &[token_id], &embed_table, hidden_size, vocab_size)?;

    // Phase 2: Decode layer loop
    // for layer_idx in 0..num_layers {
    //     hidden = norm::rms_norm(stream, &norm_kernel, &hidden, &layer_weights.norm1, eps, hidden_size)?;

    //     match get_layer_type(layer_idx) {
    //         LayerType::GatedDeltaNet => {
    //             hidden = gdn::decode_forward(gemm, stream, &layer_weights.gdn, &hidden, hidden_size)?;
    //         }
    //         LayerType::FullAttention => {
    //             hidden = attention::decode_forward(gemm, stream, &layer_weights.attn, &hidden, &mut kv_cache, position, head_dim, num_heads)?;
    //         }
    //     }

    //     hidden = norm::rms_norm(stream, &norm_kernel, &hidden, &layer_weights.norm2, eps, hidden_size)?;
    //     hidden = mlp::mlp_forward(gemm, stream, &silu_kernel, ..., &hidden, hidden_size, intermediate_size)?;
    // }

    // Phase 3: Final norm + LM head + sample
    // hidden = norm::rms_norm(...)?;
    // logits = gemm.matmul_bf16(...)?;
    // sample::greedy_sample(stream, &argmax_kernel, &logits)

    todo!("decode: embed → layer loop (norm1 → GDN/attention → norm2 → MLP) → final norm → LM head → sample")
}
