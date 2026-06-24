---
lat:
  require-code-mention: true
---
# GDN Recurrent Step Kernel Test

Unit test validating the `infers_gdn_recurrent_step_bf16` CUDA kernel against CPU reference implementations in both Rust and Python.

The test generates deterministic BF16 inputs using a seeded LCG (seed=42, values in [-1, 1]) for query [H=24, K=128], key [24, 128], value [24, V=128], a_proj [24], and b_proj [24]. Model weights from Qwen3.6-27B-AutoRound-INT4 layer 0 provide actual a_log and dt_bias f32 values (shape [24]). State is initialized to zeros [24, 128, 128] f32.

The Rust test computes both GPU output (via `launch_gdn_recurrent_step_bf16`) and CPU reference output (same algorithm in f32), then compares per-head cosine similarity and max absolute error. All inputs and outputs are saved to `/tmp/gdn_test_inputs/` as raw binary with a JSON manifest for external verification.

The Python reference script (`/tmp/gdn_recurrent_step_ref.py`) independently loads the same inputs, implements the exact kernel algorithm (L2 normalization, softplus-clamped decay, sigmoid beta, state update via outer product), and compares against GPU output. Cross-verification confirms: global cosine similarity > 0.999998 (GPU vs Python), max abs error < 0.0001, and all per-head cosines > 0.999997.

The bf16→f32 conversion uses the kernel's exact method: `f32::from_bits(bits << 16)`, ensuring bit-level compatibility between all three implementations (GPU kernel, Rust CPU ref, Python CPU ref).
