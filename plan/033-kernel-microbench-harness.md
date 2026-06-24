# Phase 033: Kernel Microbenchmark Harness with Real Dumped Inputs

---
**Status**: IN PROGRESS
**Last Updated**: 2026-06-24
**Blocks**: Phases 034-039 (all per-kernel optimization phases depend on this)
**Blocked by**: None
**Rationale**: nsys profiling shows INT4 decode at 48ms/token (target 25ms). Individual kernel timings from the profile: `int4_gemm_v3_ksplit` 37%, NCCL 18%, `paged_attention_decode` 13%, cuBLAS gemvx 6%, `gdn_recurrent_step` 6%, rmsnorm 3%. To iterate on each kernel we need to isolate it, feed it realistic inputs, and measure with CUDA events — without running the full 5-second inference loop. The existing probe dump infra (`INFERS_DUMP_DIR`) already writes BF16 intermediates to disk; the `cuda-oxide-kernels` binary already has test functions that launch individual kernels. This phase connects the two: dump once, bench many times.
---

## Goal

Build a `--bench` mode into `infers-cuda-oxide-kernels` that:
1. Loads dumped `.raw` activations + individual weight tensors from safetensors (mmap, zero-copy)
2. Uploads to GPU, runs the kernel N times with CUDA events, reports median/mean/min latency
3. Verifies output against dumped reference output (if available) or CPU reference
4. Supports per-kernel selection so we can iterate on one kernel in isolation

## Design

### Dump-Once, Bench-Many Workflow

```bash
# One-time dump (5s) — captures all decode intermediates at realistic dimensions:
INFERS_DUMP_DIR=/tmp/decode_dump INFERS_DUMP_PHASE=decode \
  INFERS_DUMP_LAYERS=all INFERS_DUMP_STAGES=all \
  ./target/release/infer --model /home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4 \
  --max-tokens 2 --no-chat --prompt "The capital of France is Paris"

# Iterate (sub-second):
./target/release/infers-cuda-oxide-kernels --bench int4_gemm_v3_ksplit \
  --dump-dir /tmp/decode_dump --layer 0 --gpu 0 --stage mlp.gate_proj
```

### CLI Interface

```
infers-cuda-oxide-kernels --bench <KERNEL_NAME> [options]

Options:
  --dump-dir <DIR>     Directory containing probe dumps (required for real inputs)
  --layer <N>          Layer index to use for inputs (default: 0)
  --gpu <N>            GPU index (default: 0)
  --stage <NAME>       Stage name prefix (e.g., "mlp.gate_proj", "attn.q_proj")
  --model-dir <DIR>    Path to model safetensors (for weight loading)
  --iterations <N>     Number of timed iterations (default: 100)
  --warmup <N>         Warmup iterations before timing (default: 10)
  --verify             Verify output against dumped reference or CPU ref (default: true)
  --no-verify          Skip verification
```

### Bench Cases

Each bench case maps a kernel + stage to:
- **Input tensor(s)**: loaded from `.raw` dump files (e.g., `layer_0/decode/mlp.norm2_gpu0.raw` for gate_proj input)
- **Weight tensor(s)**: loaded from safetensors via mmap (e.g., `model.layers.0.mlp.gate_proj.weight`)
- **Output reference**: dumped `.raw` if available (e.g., `layer_0/decode/mlp.gate_proj_gpu0.raw`)
- **Kernel launch config**: block/grid dims matching production dispatch

| Kernel | Stage | Input dump | Weight (safetensors) | Output dump |
|--------|-------|-----------|---------------------|-------------|
| `int4_gemm_v3_ksplit` | `mlp.gate_proj` | `mlp.norm2_gpu0` | `model.layers.{L}.mlp.gate_proj.*` | `mlp.gate_proj_gpu0` |
| `int4_gemm_v3_ksplit` | `mlp.up_proj` | `mlp.norm2_gpu0` | `model.layers.{L}.mlp.up_proj.*` | `mlp.up_proj_gpu0` |
| `int4_gemm_v3_ksplit` | `mlp.down_proj` | `mlp.silu_gpu0` | `model.layers.{L}.mlp.down_proj.*` | `mlp.down_raw_gpu0` |
| `int4_gemm_v3_ksplit` | `attn.q_proj` | `{stage}.norm1_gpu0` | `model.layers.{L}.self_attn.q_proj.*` | (varies) |
| `infers_paged_attention_decode_bf16` | `attn.decode` | Q/K/V from dumps | N/A (uses KV cache) | `attn.o_proj` input |
| `infers_gdn_recurrent_step_bf16` | `gdn.recurrent` | Q/K/V/a/b from dumps | N/A | `gdn.gdn_output_gpu0` |
| `infers_rmsnorm_bf16` | `attn.norm1` | hidden input | norm weight | `attn.norm1_gpu0` |
| `reduce_partial_sums_bf16` | `gemm.reduce` | partial_sums (synthetic) | N/A | GEMM output |

### Weight Loading

Use `infers_model::load_safetensors_mmap` to get zero-copy access to the full model. Individual tensors are paged in by the OS on first access. For INT4, load the triplet: `qweight` (u32 packed), `scales` (f16), `qzeros` (u32 packed). For NVFP4, load `weight_packed` (u8), `weight_scale` (u8 fp8), `weight_global_scale` (f32).

TP sharding: for 2-GPU models, each GPU holds half the output columns. The bench harness loads the full weight and shards it the same way production does (column-parallel: `weight[:, :N/2]` for GPU 0).

### Timing Protocol

```rust
// Warmup
for _ in 0..warmup { launch_kernel(...); }
stream.synchronize()?;

// Timed
let start = ctx.new_event(Some(CU_EVENT_DEFAULT))?;
let end = ctx.new_event(Some(CU_EVENT_DEFAULT))?;
start.record(&stream)?;
for _ in 0..iterations { launch_kernel(...); }
end.record(&stream)?;
end.synchronize()?;
let ms = start.elapsed_ms(&end)?;
let per_call = ms / iterations as f32;
```

Report: `kernel=X, iters=N, total={ms}ms, per_call={per_call}us, median={median}us`

## Implementation

### Files Modified

1. **`crates/cuda-oxide-kernels/src/main.rs`**: Add `--bench` CLI parsing + bench dispatcher
2. **`crates/cuda-oxide-kernels/src/bench.rs`** (NEW): Bench harness module — dump loading, weight loading, event timing, per-kernel bench functions
3. **`crates/cuda-oxide-kernels/Cargo.toml`**: Add `infers-model` dependency (for mmap weight loading)

### Dependency: CudaEvent in cuda-core

The test binary uses `cuda_core` (not `cudarc` directly). Check if `cuda_core::CudaEvent` exists; if not, add it via the cudarc re-export or use `cudarc::driver::safe::CudaEvent` directly.

## Current State

```
infers-cuda-oxide-kernels binary:
  ├─ Default: run all test functions (31 tests, ~5s)
  ├─ --save-cubin <path>: extract cubin
  └─ --verify-cubin <path>: verify cubin signatures
```

## Target State

```
infers-cuda-oxide-kernels binary:
  ├─ Default: run all tests
  ├─ --save-cubin <path>
  ├─ --verify-cubin <path>
  └─ --bench <kernel> --dump-dir <dir> --layer <N> --stage <name>
      ├─ int4_gemm_v3_ksplit (with int4 weight triplet from safetensors)
      ├─ nvfp4_gemm_v3_ksplit (with nvfp4 weight triplet)
      ├─ infers_paged_attention_decode_bf16 (with KV cache pages)
      ├─ infers_gdn_recurrent_step_bf16 (with recurrent state)
      ├─ infers_rmsnorm_bf16 (with norm weight)
      ├─ infers_silu_glu_bf16 (elementwise)
      ├─ reduce_partial_sums_bf16 (synthetic input)
      └─ infers_conv1d_depthwise_silu_bf16 (with conv weight + state)
```

## Acceptance Criteria

1. `--bench int4_gemm_v3_ksplit --dump-dir /tmp/decode_dump --layer 0 --stage mlp.gate_proj --model-dir ...` runs and prints timing.
2. Timing is stable across runs (±5% noise).
3. Per-call latency matches nsys per-kernel average (within 20%).
4. `--verify` flag confirms output matches dumped reference (cosine > 0.99 or max_diff < threshold).
5. At least 4 kernels are benchable: int4_gemm_v3_ksplit, infers_rmsnorm_bf16, infers_silu_glu_bf16, reduce_partial_sums_bf16.
6. Full bench run (all benchable kernels) completes in <5 seconds.
