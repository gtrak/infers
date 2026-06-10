//! NCCL communicator for multi-GPU collective operations.
//!
//! Provides all-reduce (TP) and send/recv (PP) primitives
//! across 2× RTX 5060 Ti GPUs.

/// Rank of this process in the NCCL communicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NcclRank(pub usize);

/// World size (total number of processes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NcclWorldSize(pub usize);

/// NCCL collective operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum.
    Sum,
    /// Element-wise product.
    Prod,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
}

use cudarc::nccl;
use std::sync::Arc;
use cudarc::driver::CudaStream;

/// NCCL communicator managing collective operations across GPUs.
pub struct NcclCommunicator {
    /// The NCCL communicator handles (one per GPU stream).
    pub comms: Vec<nccl::safe::Comm>,
    /// Rank of this process.
    pub rank: NcclRank,
    /// Total number of processes.
    pub world_size: NcclWorldSize,
}

impl NcclCommunicator {
    /// Create a new NCCL communicator using the provided streams.
    /// All ranks must coordinate to create communicators from the same unique ID.
    pub fn new(
        rank: NcclRank,
        world_size: NcclWorldSize,
        streams: Vec<Arc<CudaStream>>,
    ) -> anyhow::Result<Self> {
        let comms = nccl::safe::Comm::from_devices(streams)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        tracing::info!(
            "NCCL communicator created: rank {}/{}",
            rank.0,
            world_size.0
        );
        Ok(Self { comms, rank, world_size })
    }
}
