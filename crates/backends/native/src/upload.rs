//! Weight upload utilities.
//!
//! Converts WeightData raw bytes to GPU-resident BF16 buffers.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaSlice, CudaStream};
use infers_model::WeightData;

/// Convert `WeightData` bytes to a GPU-resident BF16 buffer.
///
/// Interprets the raw bytes as BF16 values (2 bytes each, little-endian).
/// The weight data stays as bytes until upload time to avoid requiring
/// GPU hardware at model load time.
pub fn upload_weight(
    stream: &Arc<CudaStream>,
    weight: &WeightData,
) -> Result<CudaSlice<bf16>> {
    let bf16_vec: Vec<bf16> = weight
        .data
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();
    stream
        .clone_htod(&bf16_vec)
        .map_err(|e| anyhow::anyhow!("Failed to upload weight '{}': {}", weight.name, e))
}
