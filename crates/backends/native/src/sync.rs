//! Tensor-parallel synchronization wrappers.
//!
//! Provides `all_reduce` helpers for attention and MLP outputs
//! across tensor-parallel GPUs via NCCL.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaSlice, CudaStream, NcclReduceOp};
use infers_cuda::nccl::NcclCommunicator;

/// All-reduce attention output across tensor-parallel ranks.
///
/// Used after the attention O-projection to sum contributions from
/// all tensor-parallel shards.
///
/// # Arguments
/// * `nccl` — NCCL communicator for collectives
/// * `stream` — CUDA stream for async NCCL operations
/// * `buffer` — GPU buffer to reduce in-place
pub fn all_reduce_attention(
    nccl: &NcclCommunicator,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<bf16>,
) -> Result<()> {
    let rank = stream.context().ordinal();
    nccl.all_reduce_in_place(rank, buffer, NcclReduceOp::Sum)
}

/// All-reduce MLP output across tensor-parallel ranks.
///
/// Used after the MLP down-projection to sum contributions from
/// all tensor-parallel shards.
///
/// # Arguments
/// * `nccl` — NCCL communicator for collectives
/// * `stream` — CUDA stream for async NCCL operations
/// * `buffer` — GPU buffer to reduce in-place
pub fn all_reduce_mlp(
    nccl: &NcclCommunicator,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<bf16>,
) -> Result<()> {
    let rank = stream.context().ordinal();
    nccl.all_reduce_in_place(rank, buffer, NcclReduceOp::Sum)
}
