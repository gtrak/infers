---
lat:
  require-code-mention: true
---
# GDN Chunked Prefill Kernel Test

Unit test validating the `infers_gdn_chunked_gated_delta_prefill_bf16` CUDA kernel against a sequential CPU reference in both Rust and Python, with per-token and per-head diagnostics.

The test generates deterministic BF16 inputs using a seeded LCG (seed=42, values in [-1, 1]) for query [S=8, H=4, K=32], key [8, 4, 32], value [8, 4, V=32], a_proj [8, 4], and b_proj [8, 4]. Model weights from Qwen3.6-27B-AutoRound-INT4 layer 0 provide actual a_log and dt_bias f32 values for the first 4 heads. State is initialized to zeros [4, 32, 32] f32. Chunk size is 4 (two chunks for 8 tokens).

The Rust test computes both GPU output (via `launch_gdn_chunked_gated_delta_prefill_bf16`) and CPU sequential reference output (token-by-token recurrence in f32), then compares global and per-token cosine similarity plus max absolute error. All inputs, outputs, and states are saved to `/tmp/gdn_prefill_test_inputs/` as raw binary with a JSON manifest for external verification.

The Python reference script (`/tmp/gdn_chunked_prefill_ref.py`) independently loads the same inputs, implements the exact sequential recurrence (L2 normalization, softplus-clamped decay, sigmoid beta, state update via outer product), and compares against GPU output. The bf16→f32 conversion uses the kernel's exact method: `f32::from_bits(bits << 16)`.

The chunked kernel used bf16 intermediate round-trips on query normalization (converting f32 to bf16 and back), introducing numerical divergence from pure f32 arithmetic. This was fixed by removing the bf16 round-trip in lines 2144 and 2158 of `kernel-lib/src/lib.rs`, aligning with the recurrent step kernel's approach of using f32 directly. The threshold is set to cosine > 0.95 for both output and state comparisons. Typical results: output cosine ~0.983, state cosine ~0.978, with Python vs Rust CPU reference at cosine 1.0 (perfect match confirming the sequential algorithm).

## rcp_sqrt_k Double Application Bug Fix

Phase 4 query scaling was off: `rcp_sqrt_k` (1/sqrt(K)) was applied redundantly, causing GDN outputs to be under-scaled by K instead of sqrt(K). Model generation became degenerate and repetitive on INT4 models.

The fix removed redundant `rcp_sqrt_k` multiplications from Phase 4 of `[[src/cuda-oxide-kernels/kernel-lib/src/lib.rs#infers_gdn_chunked_gated_delta_prefill_bf16]]`:
- Line ~2144: `q_scl = q_normed_f32 * exp_g_row` (removed extra `* rcp_sqrt_k`)
- Line ~2157: `qk_dot_j += q_normed_f32 * k_normed[d]` (removed extra `* rcp_sqrt_k`)

Without this fix, the query contribution to GDN output was scaled by 1/K instead of the correct scaling factor, causing model outputs to become repetitive. With the fix, the INT4 Qwen3.6-27B-AutoRound model produces coherent non-repeating text on TP=2 inference.
