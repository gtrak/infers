---
lat:
  require-code-mention: false
---
# NVFP4 Fused vs Dequant+GEMM Compare Test

Direct GPU-side comparison of `nvfp4_gemm_fused` against the two-step `nvfp4_dequant_to_bf16` + `bf16_gemm_tiled` path, plus a CPU reference.

The fused kernel applies bf16 rounding after dequantization to match the dequant+GEMM path: `(fp4_val * scale) / global_scale` is computed identically in both paths, and the result is rounded through bf16 before multiplication. Non-finite values (NaN/infinity) from bf16 overflow are sanitized to 0.0 before accumulation, preventing inf-inf subtraction that would produce NaN.

## Small Dimensions (M=2, N=16, K=64)

Verifies that with small random data both paths produce identical results (max diff = 0). Confirms no obvious indexing bug in the fused kernel at low dimensions.

## Large Dimensions (M=2, N=512, K=1024)

Stress-test with larger dimensions closer to real inference workloads. Both paths match exactly (max diff = 0), ruling out tile-boundary or grid-sizing bugs.
