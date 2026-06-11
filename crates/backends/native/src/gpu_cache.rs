//! GPU-resident weight cache for cached dequantized tensors.
//!
//! Holds per-GPU caches of weights in either BF16 or INT4-quantized form,
//! keyed by tensor name.

use std::collections::HashMap;

use half::bf16;
use infers_cuda::CudaSlice;

/// A weight stored on the GPU, either as raw BF16/FP16/FP32 or INT4 quantized.
pub enum CachedWeight {
    /// BF16/FP16/FP32 weight uploaded as CudaSlice<bf16>
    Bf16(CudaSlice<bf16>),
    /// INT4 quantized weight triplet: qweight (u32 packed) + scales (bf16) + qzeros (u32 packed)
    Int4(Int4GpuBuffers),
}

/// GPU buffers for an INT4 quantized weight tensor.
pub struct Int4GpuBuffers {
    pub qweight: CudaSlice<u32>,
    pub scales: CudaSlice<bf16>,
    pub qzeros: CudaSlice<u32>,
    /// Whether the INT4 weight has transposed layout (shape\[0\]*8 == K).
    pub transposed: bool,
}

/// Per-GPU cache of dequantized, GPU-resident weight buffers.
/// All weights for one GPU shard, keyed by tensor name.
pub struct GpuWeightCache {
    weights: HashMap<String, CachedWeight>,
}

impl GpuWeightCache {
    /// General lookup by tensor name.
    pub fn get(&self, name: &str) -> Option<&CachedWeight> {
        self.weights.get(name)
    }

    /// Lookup BF16 weight. Returns None if the cached weight is INT4.
    pub fn get_bf16(&self, name: &str) -> Option<&CudaSlice<bf16>> {
        match self.weights.get(name)? {
            CachedWeight::Bf16(slice) => Some(slice),
            CachedWeight::Int4(_) => None,
        }
    }

    /// Lookup INT4 weight. Returns None if the cached weight is BF16.
    pub fn get_int4(&self, name: &str) -> Option<&Int4GpuBuffers> {
        match self.weights.get(name)? {
            CachedWeight::Bf16(_) => None,
            CachedWeight::Int4(buffers) => Some(buffers),
        }
    }

    /// Number of cached weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}
