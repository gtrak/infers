---
lat:
  require-code-mention: true
---
# RMSNorm Gated Kernel Test

Unit test validating the `infers_rms_norm_gated_bf16` CUDA kernel against CPU reference implementations in both Rust and Python.

The algorithm normalizes each row via RMS normalization, applies a SiLU gate, and multiplies by weight: for row r, compute inv_rms = 1/sqrt(mean(input^2) + eps), then output[r][i] = weight[i] * input[r][i] * inv_rms * silu(gate[r][i]).

The test generates deterministic BF16 inputs using a seeded LCG (seed=42, values in [-1, 1]) for input [n_rows=24, d=128], gate [24, 128], and weight [128]. The Rust test computes both GPU output (via `launch_rms_norm_gated_bf16`) and CPU reference output (same algorithm in f32), then compares per-row cosine similarity and max absolute error. All inputs and outputs are saved to `/tmp/rms_norm_test_inputs/` as raw binary with a JSON manifest for external verification.

The Python reference script (`/tmp/rms_norm_gated_ref.py`) independently loads the same inputs, implements the exact kernel algorithm, and compares against GPU output. Cross-verification confirms: global cosine similarity > 0.999998 (GPU vs Python), max abs error < 0.007, and all per-row cosines > 0.999997.
