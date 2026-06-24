//! Pre-allocated workspace buffers for the decode hot path.
//!
//! All intermediate buffers needed during a single decode step are allocated once
//! at engine init and reused every token. This eliminates hundreds of `alloc_zeros`
//! calls per token that cause GPU memory management overhead.

use std::sync::Arc;
use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaSlice, CudaStream};

/// Pre-allocated GPU buffers for the decode path (per-GPU).
///
/// Allocated once at `ForwardEngine::new()` time, these buffers replace the
/// per-token `alloc_zeros` calls in `decode_paged`. The decode loop writes into
/// these buffers via `&mut` references; no allocation happens during steady-state decode.
///
/// ## Buffer Lifecycle (per layer, per GPU)
///
/// 1. `norm1_out` ← `rms_norm_into(hidden, norm1_weight)` — input to GDN/attention
/// 2. GDN/attention returns output (still allocates internally for now)
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
}

impl DecodeWorkspace {
    /// Allocate all workspace buffers on the given stream.
    ///
    /// # Arguments
    /// * `stream` — CUDA stream to allocate on
    /// * `hidden_size` — Model hidden dimension (e.g., 5120)
    /// * `sharded_intermediate` — Intermediate size / num_gpus (e.g., ~14336/2)
    /// * `vocab_size` — Vocabulary size for logits (e.g., 151936)
    pub fn new(
        stream: &Arc<CudaStream>,
        hidden_size: usize,
        sharded_intermediate: usize,
        vocab_size: usize,
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
        })
    }
}
