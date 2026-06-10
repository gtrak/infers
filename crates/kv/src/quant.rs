/// KV cache quantization data types.
///
/// Defines the supported precision formats for storing key/value
/// activation tensors in the paged attention cache. Each variant
/// carries a different trade-off between memory footprint and
/// numerical fidelity.

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheDtype {
    /// Brain float 16 — 2 bytes per element.
    Bf16,
    /// FP8 with 4 exponent bits, 3 mantissa bits — 1 byte per element.
    Fp8E4M3,
    /// FP8 with 5 exponent bits, 2 mantissa bits — 1 byte per element.
    Fp8E5M2,
    /// NVIDIA 4-bit float — packed with block scales, 1 byte per element.
    Nvfp4,
}

impl KvCacheDtype {
    /// Returns the number of bytes used for a single element.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8E4M3 => 1,
            Self::Fp8E5M2 => 1,
            Self::Nvfp4 => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_bytes_per_element() {
        assert_eq!(KvCacheDtype::Bf16.bytes_per_element(), 2);
    }

    #[test]
    fn fp8_e4m3_bytes_per_element() {
        assert_eq!(KvCacheDtype::Fp8E4M3.bytes_per_element(), 1);
    }

    #[test]
    fn fp8_e5m2_bytes_per_element() {
        assert_eq!(KvCacheDtype::Fp8E5M2.bytes_per_element(), 1);
    }

    #[test]
    fn nvfp4_bytes_per_element() {
        assert_eq!(KvCacheDtype::Nvfp4.bytes_per_element(), 1);
    }

    #[test]
    fn all_variants_are_distinct() {
        assert_ne!(KvCacheDtype::Bf16, KvCacheDtype::Fp8E4M3);
        assert_ne!(KvCacheDtype::Fp8E4M3, KvCacheDtype::Fp8E5M2);
        assert_ne!(KvCacheDtype::Fp8E5M2, KvCacheDtype::Nvfp4);
    }
}

/// Quantized paged KV cache using interleaved K+V per page layout.
///
/// Matches the PagedKvCache layout but stores quantized values:
/// page_pool[page_id * page_stride + side * page_size * kv_dim + ...]
/// where side=0 for K, side=1 for V.
#[derive(Debug)]
pub struct QuantizedKvCache {
    /// Quantized data type for cache entries.
    pub dtype: KvCacheDtype,
    /// Interleaved page pool (K then V per page), quantized to dtype.
    pub page_pool: CudaSlice<u8>,
    /// Number of pages in the pool.
    pub num_pages: usize,
    /// Page size (tokens per page).
    pub page_size: usize,
    /// KV dimension (num_kv_heads * head_dim).
    pub kv_dim: usize,
    /// Block scales for NVFP4 (one per 128-element block).
    /// Two scales per block: one for K, one for V.
    pub scales: Option<CudaSlice<half::bf16>>,
}

impl QuantizedKvCache {
    /// Allocate a new quantized KV cache on the GPU.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream for memory allocation
    /// * `num_pages` - Total number of physical pages
    /// * `page_size` - Number of tokens per page
    /// * `kv_dim` - KV dimension (num_kv_heads * head_dim)
    /// * `dtype` - Quantized data type for cache entries
    ///
    /// # Returns
    /// A new `QuantizedKvCache` with GPU memory allocated and zeroed.
    pub fn allocate(
        stream: &std::sync::Arc<CudaStream>,
        num_pages: usize,
        page_size: usize,
        kv_dim: usize,
        dtype: KvCacheDtype,
    ) -> anyhow::Result<Self> {
        let bytes_per_elem = dtype.bytes_per_element();
        let page_stride = 2 * page_size * kv_dim; // elements per page (K + V)
        let page_bytes = page_stride * bytes_per_elem; // bytes per page
        let total_bytes = num_pages * page_bytes;

        let page_pool = stream
            .alloc_zeros::<u8>(total_bytes)?;

        let scales = if matches!(dtype, KvCacheDtype::Nvfp4) {
            // NVFP4 uses block scales: one scale per 128 elements
            // Each page has 2 * page_size * kv_dim elements, so:
            let num_blocks = (num_pages * 2 * page_size * kv_dim).div_ceil(128);
            // Two scale values per block (one for K-side, one for V-side)
            Some(stream.alloc_zeros::<half::bf16>(num_blocks * 2)?)
        } else {
            None
        };

        Ok(Self {
            dtype,
            page_pool,
            num_pages,
            page_size,
            kv_dim,
            scales,
        })
    }
}
