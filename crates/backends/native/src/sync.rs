//! Tensor-parallel synchronization wrappers.
//!
//! Provides `all_reduce` helpers for attention and MLP outputs
//! across tensor-parallel GPUs via NCCL.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaSlice, CudaStream};
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
    _stream: &Arc<CudaStream>,
    _buffer: &mut CudaSlice<bf16>,
) -> Result<()> {
    todo!("all_reduce_attention: call nccl.all_reduce(rank, send, recv, ReduceOp::Sum), sync stream")
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
    _stream: &Arc<CudaStream>,
    _buffer: &mut CudaSlice<bf16>,
) -> Result<()> {
    todo!("all_reduce_mlp: call nccl.all_reduce(rank, send, recv, ReduceOp::Sum), sync stream")
}
