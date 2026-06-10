//! Prefill path: full forward pass over a prompt sequence.
//!
//! Tokenize → embed → layer loop → final norm → LM head → sample

use std::sync::Arc;

use anyhow::Result;
use infers_cuda::CudaStream;
use infers_cuda::gemm::GemmEngine;
use infers_model::LayerType;

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
/// * `gemm` — cuBLASLt engine
/// * `stream` — CUDA stream
/// * `token_ids` — Prompt token IDs
/// * `num_layers` — Total transformer layers
/// * `get_layer_type` — Closure that returns layer type by index
///
/// # Returns
/// Sampled token ID for the first generated token
pub fn prefill(
    _gemm: &mut GemmEngine,
    _stream: &Arc<CudaStream>,
    _token_ids: &[u32],
    _num_layers: usize,
    _get_layer_type: impl Fn(usize) -> LayerType,
) -> Result<u32> {
    // Phase 1: Embedding
    // let hidden = embedding::embed_tokens(stream, &embed_kernel, token_ids, &embed_table, hidden_size, vocab_size)?;

    // Phase 2: Layer loop
    // for layer_idx in 0..num_layers {
    //     // Norm1
    //     hidden = norm::rms_norm(stream, &norm_kernel, &hidden, &layer_weights.norm1, eps, hidden_size)?;

    //     // Attention / GDN dispatch
    //     match get_layer_type(layer_idx) {
    //         LayerType::GatedDeltaNet => {
    //             hidden = gdn::forward(gemm, stream, &layer_weights.gdn, &hidden, hidden_size)?;
    //         }
    //         LayerType::FullAttention => {
    //             hidden = attention::forward(gemm, stream, &layer_weights.attn, &hidden, &mut kv_cache, positions, head_dim, num_heads)?;
    //         }
    //     }

    //     // Norm2
    //     hidden = norm::rms_norm(stream, &norm_kernel, &hidden, &layer_weights.norm2, eps, hidden_size)?;

    //     // MLP
    //     hidden = mlp::mlp_forward(gemm, stream, &silu_kernel, &layer_weights.mlp.gate_proj, &layer_weights.mlp.up_proj, &layer_weights.mlp.down_proj, &hidden, hidden_size, intermediate_size)?;
    // }

    // Phase 3: Final norm + LM head
    // hidden = norm::rms_norm(stream, &norm_kernel, &hidden, &final_norm, eps, hidden_size)?;
    // logits = gemm.matmul_bf16(...)?;

    // Phase 4: Sample
    // sample::greedy_sample(stream, &argmax_kernel, &logits)

    todo!("prefill: embed → layer loop (norm1 → GDN/attention → norm2 → MLP) → final norm → LM head → sample")
}
