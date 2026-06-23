---
lat:
  require-code-mention: true
---
# Gemm Compare Test

Regression test comparing bf16_gemm_tiled against cuBLAS on known small data (4×8 output).

Both kernels compute C = A @ B^T where all buffers are row-major BF16. The test allocates input A [M=4, K=16] and weight B [N=8, K=16], runs both GEMMs, computes ground truth in fp32 on CPU, and reports max absolute error (asserts < 0.01), mean absolute error, cosine similarity, and deviation from gold standard per kernel.

The test currently **fails** because bf16_gemm_tiled produces larger deviations than cuBLAS, indicating a bug in the tiled kernel implementation. The cosine similarity is 0.999996 — vectors point nearly the same direction but element-level errors reach 4.0 units.

## Multi-Tile GEMM Bug

bf16_gemm_tiled only computes the first tile (blockIdx_x=0) correctly; subsequent tiles produce all-zero output. The failure threshold is N=64. Details from diagnostic tests in [[crates/cuda/tests/nvfp4_debug.rs#test_dequant_small_n]] and [[crates/cuda/tests/nvfp4_debug.rs#test_tile_boundary]].

This bug explains why in_proj_qkv (N=5120) fails via the NVFP4 dequant→bf16_gemm_tiled path, while a_proj/b_proj (N=48, single tile) work because they use cuBLAS for BF16 weights.


