/// KV cache quantization data types.
///
/// Defines the supported precision formats for storing key/value
/// activation tensors in the paged attention cache. Each variant
/// carries a different trade-off between memory footprint and
/// numerical fidelity.

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
