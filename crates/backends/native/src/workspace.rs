//! Pre-allocated workspace buffers for the decode hot path.
//!
//! All intermediate buffers needed during a single decode step are allocated once
//! at engine init and reused every token. This eliminates hundreds of `alloc_zeros`
//! calls per token that cause GPU memory management overhead.

use std::sync::Arc;
use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaSlice, CudaStream};
use infers_model::ModelConfig;

/// Pre-allocated GPU buffers for the decode path (per-GPU).
///
/// Allocated once at `ForwardEngine::new()` time, these buffers replace the
/// per-token `alloc_zeros` calls in `decode_paged`. The decode loop writes into
/// these buffers via `&mut` references; no allocation happens during steady-state decode.
///
/// ## Buffer Lifecycle (per layer, per GPU)
///
/// 1. `norm1_out` ← `rms_norm_into(hidden, norm1_weight)` — input to GDN/attention
  /// 2. GDN writes into attn_out via workspace; attention writes directly into attn_out via workspace
/// 3. `residual_buf` ← `add_into(hidden, attn_out)` — then `swap(hidden, residual_buf)`
/// 4. `norm2_out` ← `rms_norm_into(hidden, norm2_weight)` — input to MLP gate/up
/// 5. `mlp_gate` ← GEMM(norm2_out, gate_proj)
/// 6. `mlp_up` ← GEMM(norm2_out, up_proj)
/// 7. `mlp_silu` ← silu_glu(mlp_up, mlp_gate)
/// 8. `mlp_out` ← GEMM(mlp_silu, down_proj)
/// 9. `residual_buf` ← `add_into(hidden, mlp_out)` — then `swap(hidden, residual_buf)`
pub struct DecodeWorkspace {
    /// RMSNorm output for norm1 (before attention/GDN). Size: hidden_size.
    pub norm1_out: CudaSlice<bf16>,
    /// RMSNorm output for norm2 (before MLP). Size: hidden_size.
    pub norm2_out: CudaSlice<bf16>,
    /// Residual add output buffer. Double-buffered with hidden_states via mem::swap.
    /// Size: hidden_size.
    pub residual_buf: CudaSlice<bf16>,
    /// MLP gate projection output. Size: sharded_intermediate.
    pub mlp_gate: CudaSlice<bf16>,
    /// MLP up projection output. Size: sharded_intermediate.
    pub mlp_up: CudaSlice<bf16>,
    /// MLP SiLU+GLU output. Size: sharded_intermediate.
    pub mlp_silu: CudaSlice<bf16>,
    /// MLP down projection output. Size: hidden_size.
    pub mlp_out: CudaSlice<bf16>,
    /// Final logits buffer. Size: vocab_size.
    pub logits: CudaSlice<bf16>,
    /// Shared output buffer for GDN and attention decode outputs. Size: hidden_size.
    /// Both GDN and attention write their final output here, eliminating the
    /// `attn_outputs: Vec` and its per-layer allocation.
    pub attn_out: CudaSlice<bf16>,
    /// GDN-specific workspace buffers (allocated once, reused every GDN layer).
    pub gdn: GdnWorkspace,
    /// Attention-specific workspace buffers (allocated once, reused every attention layer).
    pub attn: AttnWorkspace,
}

impl DecodeWorkspace {
    /// Allocate all workspace buffers on the given stream.
    ///
    /// # Arguments
    /// * `stream` — CUDA stream to allocate on
    /// * `config` — Model configuration
    /// * `hidden_size` — Model hidden dimension (e.g., 5120)
    /// * `sharded_intermediate` — Intermediate size / num_gpus (e.g., ~14336/2)
    /// * `vocab_size` — Vocabulary size for logits (e.g., 151936)
    /// * `num_gpus` — Number of GPUs for tensor-parallel sharding
    pub fn new(
        stream: &Arc<CudaStream>,
        config: &ModelConfig,
        hidden_size: usize,
        sharded_intermediate: usize,
        vocab_size: usize,
        num_gpus: usize,
    ) -> Result<Self> {
        Ok(Self {
            norm1_out: stream.alloc_zeros::<bf16>(hidden_size)?,
            norm2_out: stream.alloc_zeros::<bf16>(hidden_size)?,
            residual_buf: stream.alloc_zeros::<bf16>(hidden_size)?,
            mlp_gate: stream.alloc_zeros::<bf16>(sharded_intermediate)?,
            mlp_up: stream.alloc_zeros::<bf16>(sharded_intermediate)?,
            mlp_silu: stream.alloc_zeros::<bf16>(sharded_intermediate)?,
            mlp_out: stream.alloc_zeros::<bf16>(hidden_size)?,
            logits: stream.alloc_zeros::<bf16>(vocab_size)?,
            attn_out: stream.alloc_zeros::<bf16>(hidden_size)?,
            gdn: GdnWorkspace::new(stream, config, num_gpus)?,
            attn: AttnWorkspace::new(stream, config, num_gpus)?,
        })
    }
}

/// Pre-allocated GPU buffers for GDN decode intermediates (per-GPU).
///
/// These buffers replace the ~15 `alloc_zeros` calls per GDN layer per token
/// in `gdn::decode_forward`. Allocated once at engine init.
pub struct GdnWorkspace {
    /// Mixed QKV projection output. Size: conv_dim.
    pub mixed_qkv: CudaSlice<bf16>,
    /// Conv1d input buffer [conv_state | mixed_qkv]. Size: kernel_size * conv_dim.
    pub conv_input: CudaSlice<bf16>,
    /// Conv1d output buffer. Size: kernel_size * conv_dim.
    pub conv_out: CudaSlice<bf16>,
    /// Last row of conv_out (the result for the current token). Size: conv_dim.
    pub conv_out_last: CudaSlice<bf16>,
    /// Query sub-slice from conv_out_last. Size: key_dim.
    pub query: CudaSlice<bf16>,
    /// Key sub-slice from conv_out_last. Size: key_dim.
    pub key: CudaSlice<bf16>,
    /// Value sub-slice from conv_out_last. Size: value_dim.
    pub value: CudaSlice<bf16>,
    /// Repeat-interleaved query (num_v_heads * head_k_dim). Only used when kv_ratio > 1.
    pub query_expanded: CudaSlice<bf16>,
    /// Repeat-interleaved key (num_v_heads * head_k_dim). Only used when kv_ratio > 1.
    pub key_expanded: CudaSlice<bf16>,
    /// in_proj_a GEMM output. Size: num_v_heads.
    pub a_proj: CudaSlice<bf16>,
    /// in_proj_b GEMM output. Size: num_v_heads (= b_dim).
    pub b_proj: CudaSlice<bf16>,
    /// GDN recurrent step output. Size: num_v_heads * head_v_dim.
    pub gdn_output: CudaSlice<bf16>,
    /// Z-gate GEMM output. Size: num_v_heads * head_v_dim (= z_dim).
    pub z_gate: CudaSlice<bf16>,
    /// RMSNormGated output. Size: num_v_heads * head_v_dim.
    pub norm_out: CudaSlice<bf16>,
    /// Pre-allocated zeros for a_log fallback (when weight is None). Size: num_v_heads (f32).
    pub a_log_zeros: CudaSlice<f32>,
    /// Pre-allocated zeros for dt_bias fallback (when weight is None). Size: num_v_heads (f32).
    pub dt_bias_zeros: CudaSlice<f32>,
}

impl GdnWorkspace {
    /// Allocate all GDN workspace buffers on the given stream.
    ///
    /// Sizes are computed from model config and GPU count for tensor-parallel sharding.
    pub fn new(
        stream: &Arc<CudaStream>,
        config: &ModelConfig,
        num_gpus: usize,
    ) -> Result<Self> {
        let num_v_heads = config.linear_num_value_heads / num_gpus;
        let kv_ratio = config.linear_num_value_heads / config.linear_num_key_heads;
        let num_k_heads = num_v_heads / kv_ratio;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let kernel_size = config.linear_conv_kernel_dim as usize;
        let v_heads_times_k_dim = num_v_heads * head_k_dim;
        let v_heads_times_v_dim = num_v_heads * head_v_dim;

        Ok(Self {
            mixed_qkv: stream.alloc_zeros::<bf16>(conv_dim)?,
            conv_input: stream.alloc_zeros::<bf16>(kernel_size * conv_dim)?,
            conv_out: stream.alloc_zeros::<bf16>(kernel_size * conv_dim)?,
            conv_out_last: stream.alloc_zeros::<bf16>(conv_dim)?,
            query: stream.alloc_zeros::<bf16>(key_dim)?,
            key: stream.alloc_zeros::<bf16>(key_dim)?,
            value: stream.alloc_zeros::<bf16>(value_dim)?,
            query_expanded: stream.alloc_zeros::<bf16>(v_heads_times_k_dim)?,
            key_expanded: stream.alloc_zeros::<bf16>(v_heads_times_k_dim)?,
            a_proj: stream.alloc_zeros::<bf16>(num_v_heads)?,
            b_proj: stream.alloc_zeros::<bf16>(num_v_heads)?,
            gdn_output: stream.alloc_zeros::<bf16>(v_heads_times_v_dim)?,
            z_gate: stream.alloc_zeros::<bf16>(v_heads_times_v_dim)?,
            norm_out: stream.alloc_zeros::<bf16>(v_heads_times_v_dim)?,
            a_log_zeros: stream.alloc_zeros::<f32>(num_v_heads)?,
            dt_bias_zeros: stream.alloc_zeros::<f32>(num_v_heads)?,
        })
    }
}

/// Pre-allocated GPU buffers for attention decode intermediates (per-GPU).
pub struct AttnWorkspace {
    /// K projection output. Size: kv_dim.
    pub k_single: CudaSlice<bf16>,
    /// V projection output. Size: kv_dim.
    pub v_single: CudaSlice<bf16>,
    /// K-norm RMSNorm output (when k_norm weight exists). Size: kv_dim.
    pub k_norm_out: CudaSlice<bf16>,
    /// RoPE dummy buffer for K. Size: kv_dim.
    pub q_dummy: CudaSlice<bf16>,
    /// Q projection output (doubled when gate). Size: per_gpu_head_dim * 2.
    pub q_full: CudaSlice<bf16>,
    /// Extracted Q from q_full. Size: per_gpu_head_dim.
    pub q_buf: CudaSlice<bf16>,
    /// Extracted gate from q_full. Size: per_gpu_head_dim.
    pub gate_buf: CudaSlice<bf16>,
    /// Q-norm RMSNorm output. Size: per_gpu_head_dim.
    pub q_norm_out: CudaSlice<bf16>,
    /// RoPE dummy buffer for Q. Size: per_gpu_head_dim.
    pub k_rope_dummy: CudaSlice<bf16>,
    /// Paged attention decode output. Size: per_gpu_head_dim.
    pub attn_output: CudaSlice<bf16>,
    /// Gate application output. Size: per_gpu_head_dim.
    pub gated: CudaSlice<bf16>,
}

impl AttnWorkspace {
    /// Allocate all attention workspace buffers on the given stream.
    ///
    /// Sizes are computed from model config and GPU count for tensor-parallel sharding.
    pub fn new(
        stream: &Arc<CudaStream>,
        config: &ModelConfig,
        num_gpus: usize,
    ) -> Result<Self> {
        let head_dim = config.head_dim;
        let num_kv_heads = config.num_key_value_heads / num_gpus;
        let num_heads = config.num_attention_heads / num_gpus;
        let kv_dim = num_kv_heads * head_dim;
        let per_gpu_head_dim = num_heads * head_dim;
        let q_out_dim_max = per_gpu_head_dim * 2; // worst case: gate enabled

        Ok(Self {
            k_single: stream.alloc_zeros::<bf16>(kv_dim)?,
            v_single: stream.alloc_zeros::<bf16>(kv_dim)?,
            k_norm_out: stream.alloc_zeros::<bf16>(kv_dim)?,
            q_dummy: stream.alloc_zeros::<bf16>(kv_dim)?,
            q_full: stream.alloc_zeros::<bf16>(q_out_dim_max)?,
            q_buf: stream.alloc_zeros::<bf16>(per_gpu_head_dim)?,
            gate_buf: stream.alloc_zeros::<bf16>(per_gpu_head_dim)?,
            q_norm_out: stream.alloc_zeros::<bf16>(per_gpu_head_dim)?,
            k_rope_dummy: stream.alloc_zeros::<bf16>(per_gpu_head_dim)?,
            attn_output: stream.alloc_zeros::<bf16>(per_gpu_head_dim)?,
            gated: stream.alloc_zeros::<bf16>(per_gpu_head_dim)?,
        })
    }
}
