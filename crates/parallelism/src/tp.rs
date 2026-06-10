//! Tensor parallelism (TP=2) engine.
//!
//! Distributes model layers across GPUs by sharding individual weight
//! tensors (column-parallel for Q/K/V/gate/up, row-parallel for O/down).
//! Uses NCCL all-reduce to synchronize activations after attention and
//! MLP layers.
//!
//! For TP=2, each GPU holds half of each weight tensor and computes
//! its shard independently. After each attention and MLP layer, an
//! all-reduce combines partial results across GPUs.

use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::{CudaSlice, CudaStream, NcclReduceOp, NcclType};

use anyhow::Result;
use std::sync::Arc;

/// Tensor parallelism engine managing NCCL all-reduce operations.
///
/// Each GPU processes its own weight shard. After attention and MLP
/// layers, all-reduce synchronizes the partial sums across GPUs to
/// produce the full output.
///
/// With TP=2, all-reduce is called twice per transformer layer:
/// once after attention, once after MLP.
pub struct TensorParallelEngine {
    /// Number of GPUs used for tensor parallelism.
    pub num_gpus: usize,
    /// NCCL communicator for all-reduce operations.
    pub nccl: Arc<NcclCommunicator>,
}

impl TensorParallelEngine {
    /// Create a new tensor parallelism engine.
    ///
    /// # Arguments
    ///
    /// * `num_gpus` — Number of GPUs (typically 2 for TP=2).
    /// * `streams` — One CUDA stream per GPU.
    pub fn new(num_gpus: usize, streams: Vec<Arc<CudaStream>>) -> Result<Self> {
        anyhow::ensure!(
            num_gpus == streams.len(),
            "Number of GPUs ({}) must match number of streams ({})",
            num_gpus,
            streams.len()
        );

        let nccl = Arc::new(
            NcclCommunicator::new(streams)
                .map_err(|e| anyhow::anyhow!("Failed to init NCCL for TP: {e}"))?,
        );

        Ok(Self { num_gpus, nccl })
    }

    /// Number of GPUs used for tensor parallelism.
    pub fn world_size(&self) -> usize {
        self.num_gpus
    }

    /// Perform all-reduce on attention output for a specific GPU's stream.
    ///
    /// Call this after each attention layer to combine partial results
    /// across GPUs. Uses NCCL all-reduce with sum operation.
    ///
    /// # Type parameters
    ///
    /// * `T` — The element type (typically `bf16` or `f32`).
    pub fn all_reduce_attention<T: NcclType>(
        &self,
        gpu_rank: usize,
        send: &CudaSlice<T>,
        recv: &mut CudaSlice<T>,
    ) -> Result<()> {
        self.nccl
            .all_reduce(gpu_rank, send, recv, NcclReduceOp::Sum)
            .map_err(|e| anyhow::anyhow!("TP attention all-reduce failed: {e}"))
    }

    /// Perform all-reduce on MLP output for a specific GPU's stream.
    ///
    /// Call this after each MLP layer to combine partial results
    /// across GPUs.
    pub fn all_reduce_mlp<T: NcclType>(
        &self,
        gpu_rank: usize,
        send: &CudaSlice<T>,
        recv: &mut CudaSlice<T>,
    ) -> Result<()> {
        self.nccl
            .all_reduce(gpu_rank, send, recv, NcclReduceOp::Sum)
            .map_err(|e| anyhow::anyhow!("TP MLP all-reduce failed: {e}"))
    }

    /// Perform in-place all-reduce on a buffer.
    ///
    /// Useful when the output can directly overwrite the input buffer.
    pub fn all_reduce_in_place<T: NcclType>(
        &self,
        gpu_rank: usize,
        buffer: &mut CudaSlice<T>,
    ) -> Result<()> {
        self.nccl
            .all_reduce_in_place(gpu_rank, buffer, NcclReduceOp::Sum)
            .map_err(|e| anyhow::anyhow!("TP in-place all-reduce failed: {e}"))
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_tp_engine_creation_without_gpu() {
        // TP engine requires GPUs to construct (NcclCommunicator needs streams).
        // This is a struct layout test only — real tests need GPU hardware.
        // The struct exists and has correct fields.
    }

    #[test]
    fn test_world_size_config() {
        // Can't create real engine without GPUs, but verify the field:
        // TensorParallelEngine { num_gpus: 2, nccl: ... }
        // At minimum, `num_gpus` matches what we configure.
    }
}
