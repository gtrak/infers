//! NCCL communicator for multi-GPU collective operations.
//!
//! Provides all-reduce, broadcast, reduce, and all-gather primitives
//! across multiple GPUs via cudarc's safe NCCL bindings.

use cudarc::driver::CudaSlice;
use cudarc::nccl::safe::{Comm, NcclType, ReduceOp};
use std::sync::Arc;

/// NCCL communicator managing collective operations across GPUs.
///
/// Created via `NcclCommunicator::new()` using `Comm::from_devices()`,
/// which returns one `Comm` per GPU stream.
pub struct NcclCommunicator {
    /// One NCCL communicator per GPU stream.
    comms: Vec<Comm>,
    /// Rank of this process (always 0 for single-process multi-GPU).
    rank: usize,
    /// Total number of GPUs in the communicator.
    world_size: usize,
}

impl NcclCommunicator {
    /// Create a new NCCL communicator using the provided streams.
    ///
    /// All streams must be on different GPUs and belong to contexts
    /// that are bound to the correct CUDA devices.
    pub fn new(streams: Vec<Arc<cudarc::driver::CudaStream>>) -> anyhow::Result<Self> {
        let comms =
            Comm::from_devices(streams).map_err(|e| anyhow::anyhow!("NCCL init failed: {:?}", e))?;
        let world_size = if comms.is_empty() { 0 } else { comms[0].world_size() };
        let rank = if comms.is_empty() { 0 } else { comms[0].rank() };

        tracing::info!(
            "NCCL communicator created: rank {}/{}, {} devices",
            rank,
            world_size,
            world_size
        );
        Ok(Self { comms, rank, world_size })
    }

    /// Get the rank of this communicator.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the world size (total GPUs).
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Get the comm for a specific rank/GPU.
    pub fn comm(&self, rank: usize) -> Option<&Comm> {
        self.comms.get(rank)
    }

    /// Get all comms.
    pub fn comms(&self) -> &[Comm] {
        &self.comms
    }

    /// All-reduce across all ranks for a specific GPU's comm.
    pub fn all_reduce<T: NcclType>(
        &self,
        rank: usize,
        send: &CudaSlice<T>,
        recv: &mut CudaSlice<T>,
        op: ReduceOp,
    ) -> anyhow::Result<()> {
        let comm = self
            .comms
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("Rank {} out of range (world_size={})", rank, self.world_size))?;
        comm.all_reduce(send, recv, &op)
            .map_err(|e| anyhow::anyhow!("NCCL all_reduce failed: {:?}", e))?;
        Ok(())
    }

    /// Broadcast from a root rank to all other ranks for a specific GPU's comm.
    pub fn broadcast<T: NcclType>(
        &self,
        rank: usize,
        send: Option<&CudaSlice<T>>,
        recv: &mut CudaSlice<T>,
        root: i32,
    ) -> anyhow::Result<()> {
        let comm = self
            .comms
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("Rank {} out of range", rank))?;
        comm.broadcast(send, recv, root)
            .map_err(|e| anyhow::anyhow!("NCCL broadcast failed: {:?}", e))?;
        Ok(())
    }

    /// Reduce to a single root rank.
    pub fn reduce<T: NcclType>(
        &self,
        rank: usize,
        send: &CudaSlice<T>,
        recv: Option<&mut CudaSlice<T>>,
        op: ReduceOp,
        root: i32,
    ) -> anyhow::Result<()> {
        let comm = self
            .comms
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("Rank {} out of range", rank))?;
        comm.reduce(send, recv, &op, root)
            .map_err(|e| anyhow::anyhow!("NCCL reduce failed: {:?}", e))?;
        Ok(())
    }

    /// All-reduce in-place across all ranks for a specific GPU's comm.
    pub fn all_reduce_in_place<T: NcclType>(
        &self,
        rank: usize,
        buffer: &mut CudaSlice<T>,
        op: ReduceOp,
    ) -> anyhow::Result<()> {
        let comm = self
            .comms
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("Rank {} out of range (world_size={})", rank, self.world_size))?;
        comm.all_reduce_in_place(buffer, &op)
            .map_err(|e| anyhow::anyhow!("NCCL all_reduce_in_place failed: {:?}", e))?;
        Ok(())
    }

    /// All-gather: gather data from all ranks into recv buffer.
    pub fn all_gather<T: NcclType>(
        &self,
        rank: usize,
        send: &CudaSlice<T>,
        recv: &mut CudaSlice<T>,
    ) -> anyhow::Result<()> {
        let comm = self
            .comms
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("Rank {} out of range", rank))?;
        comm.all_gather(send, recv)
            .map_err(|e| anyhow::anyhow!("NCCL all_gather failed: {:?}", e))?;
        Ok(())
    }
}
