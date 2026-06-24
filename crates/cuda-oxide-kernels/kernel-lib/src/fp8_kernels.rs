//! FP8 quantize/dequantize kernels.

use cuda_device::{cuda_module, kernel, launch_bounds, thread, DisjointSlice};
use super::shared::*;

#[cuda_module]
pub mod fp8 {
    use super::*;

    /// FP8 quantize kernel: E4M3 format (BF16 → u8).
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_fp8_quantize_e4m3(input: &[u16], mut output: DisjointSlice<u8>, n: u32) {
        fp8_quantize_inner::<Fp8E4M3>(input, output, n);
    }

    /// FP8 dequantize kernel: E4M3 format (u8 → BF16).
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_fp8_dequantize_e4m3(input: &[u8], mut output: DisjointSlice<u16>, n: u32) {
        fp8_dequantize_inner::<Fp8E4M3>(input, output, n);
    }

    /// FP8 quantize kernel: E5M2 format (BF16 → u8).
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_fp8_quantize_e5m2(input: &[u16], mut output: DisjointSlice<u8>, n: u32) {
        fp8_quantize_inner::<Fp8E5M2>(input, output, n);
    }

    /// FP8 dequantize kernel: E5M2 format (u8 → BF16).
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_fp8_dequantize_e5m2(input: &[u8], mut output: DisjointSlice<u16>, n: u32) {
        fp8_dequantize_inner::<Fp8E5M2>(input, output, n);
    }
}
