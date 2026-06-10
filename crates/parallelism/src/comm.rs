//! Stage-to-stage hidden state transfer via NCCL P2P.
//!
//! Each pipeline stage uses its own NCCL communicator rank to send/recv
//! hidden states (BF16 tensors) to/from the adjacent stage.
//!
//! For PP=2 with microbatching, stage 0 sends and stage 1 receives.
//! For multiple microbatches, sends and receives are interleaved to keep
//! both GPUs busy.

use anyhow::Result;
use half::bf16;
use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::CudaSlice;

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
    /// NCCL communicator shared across stages.
    pub nccl: NcclCommunicator,
    /// Rank of this stage within the NCCL communicator.
    pub rank: usize,
    /// Peer rank (the adjacent stage).
    pub peer_rank: usize,
}

impl StageComm {
    /// Create a new stage communicator.
    pub fn new(nccl: NcclCommunicator, rank: usize, peer_rank: usize) -> Self {
        Self { nccl, rank, peer_rank }
    }

    /// Send hidden states to the next pipeline stage.
    ///
    /// `hidden` is a BF16 tensor on the GPU, typically the output of
    /// `forward_stage0` before being passed to `forward_stage1`.
    pub fn send_hidden(&self, hidden: &CudaSlice<bf16>) -> Result<()> {
        self.nccl
            .send(self.rank, hidden, self.peer_rank as i32)
            .map_err(|e| anyhow::anyhow!("StageComm send failed: {e}"))
    }

    /// Receive hidden states from the previous pipeline stage.
    ///
    /// `hidden` is a pre-allocated mutable BF16 tensor on the GPU that
    /// will be filled with the received hidden states.
    pub fn recv_hidden(&self, hidden: &mut CudaSlice<bf16>) -> Result<()> {
        self.nccl
            .recv(self.rank, hidden, self.peer_rank as i32)
            .map_err(|e| anyhow::anyhow!("StageComm recv failed: {e}"))
    }

    /// The NCCL communicator for this stage.
    pub fn comm(&self) -> &NcclCommunicator {
        &self.nccl
    }
}
