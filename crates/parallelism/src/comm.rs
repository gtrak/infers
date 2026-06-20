//! Stage-to-stage hidden state transfer via NCCL P2P.
//!
//! Each pipeline stage uses its own NCCL communicator rank to send/recv
//! hidden states (BF16 tensors) to/from the adjacent stage.
//!
//! For PP=2 with microbatching, stage 0 sends and stage 1 receives.
//! For multiple microbatches, sends and receives are interleaved to keep
//! both GPUs busy.

use std::sync::Arc;

use infers_cuda::nccl::NcclCommunicator;

/// Stage-to-stage hidden state transfer via NCCL P2P.
///
/// Each `StageComm` is initialized with the stage's NCCL communicator and
/// knows the peer rank. Hidden states are BF16 tensors of shape
/// `[microbatch_size × seq_len × hidden_size]`.
///
/// For PP=2: rank 0 (stage 0) sends to peer rank 1; rank 1 (stage 1) receives
/// from peer rank 0.
#[derive(Debug)]
pub struct StageComm {
    /// NCCL communicator shared across stages (wrapped in Arc for sharing).
    pub nccl: Arc<NcclCommunicator>,
    /// Rank of this stage within the NCCL communicator.
    pub rank: usize,
    /// Peer rank (the adjacent stage).
    pub peer_rank: usize,
}

impl StageComm {
    /// Create a new stage communicator.
    pub fn new(nccl: Arc<NcclCommunicator>, rank: usize, peer_rank: usize) -> Self {
        Self { nccl, rank, peer_rank }
    }


    /// The NCCL communicator for this stage.
    pub fn comm(&self) -> &NcclCommunicator {
        &self.nccl
    }
}
