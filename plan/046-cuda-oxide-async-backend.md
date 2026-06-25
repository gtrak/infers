# Phase 046: cuda-oxide Async Backend

---
**Status**: IN PROGRESS
**Last Updated**: 2026-06-25
**Blocks**: Phase 031 (continuous batching), Phase 032 (CUDA graphs — may be obsoleted)
**Blocked by**: None
**Rationale**: The current engine uses a single null stream per GPU with synchronous kernel launches. CPU launch overhead is ~6ms/step (672+ kernel launches × ~2µs each + cuBLASLt/NCCL overhead). CUDA graph capture is blocked by NCCL/stream-capture incompatibility (Phase 044). The cuda-oxide `DeviceOperation` model with `and_then` chains offers an alternative: build the entire 48-layer decode as a lazy pipeline, submit all kernels in one `execute()` call via `unsafe async_on(&stream)`, then synchronize once. This eliminates host-side launch overhead without graph capture.
---

## Goal

Transform the synchronous per-kernel decode loop into a lazy `DeviceOperation` pipeline that submits all GPU work in one burst, eliminating ~6ms CPU launch overhead per decode step.

**Current**: 0.036s/step (27.8 tok/s) — GPU-bound, no CPU overhead gap
**Target**: 0.036s/step (27.8 tok/s) for single-token decode — **parity, not speedup**

The GPU is fully utilized (36ms GPU time = 36ms wall time). The async pipeline does NOT improve single-token decode latency. The value is architectural:
- Clean buffer ownership (replace CudaSliceView hack)
- Composable pipelines (and_then / zip! for future prefill overlap)
- Foundation for continuous batching (multiple sequences on different streams)
- Future multi-stream scheduling via round-robin pool

## Architecture

### The Key Insight: `and_then::execute()` is Synchronous on the Host

The `AndThen::execute()` method (in `cuda-async/src/device_operation.rs:339-345`) recursively traverses all nested `and_then` nodes, calling each operation's `execute()` in sequence on the same thread. All kernel launches are enqueued on the same stream. No `cuLaunchHostFunc` callback is needed — that's only in the `DeviceFuture` polling path.

`unsafe async_on(&stream)` calls `self.execute(&ctx)` which submits all work to the stream and returns immediately. The caller syncs the stream afterward. This is functionally equivalent to CUDA graph replay:

```
Current (synchronous):
  for layer in 0..48:
    launch kernel → launch kernel → launch NCCL → launch kernel → ...
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    672 individual host→GPU round trips, ~6ms total CPU overhead

Async pipeline:
  pipeline.async_on(&stream)   // submits ALL 672 kernels at once
  stream.synchronize()          // one sync, zero host involvement during execution
```

### What We Keep

| Component | Status | Notes |
|-----------|--------|-------|
| Model loading (`crates/model/`) | Unchanged | safetensors loading, weight registry, metadata |
| Weight cache (`GpuWeightCache`) | Unchanged | cudarc `CudaSlice<T>` storage, accessed via `cu_deviceptr()` |
| Kernel library (`crates/cuda-oxide-kernels/`) | Add `async` feature | Same cubin, same kernel source |
| Workspace buffers (`DecodeWorkspace`) | Unchanged | cudarc `CudaSlice<T>`, accessed via `cu_deviceptr()` |
| KV cache, GDN state | Unchanged | |
| Tokenizer, scheduler, server | Unchanged | |

### What Changes

| File | Change |
|------|--------|
| `crates/cuda/Cargo.toml` | Add `cuda-async` dependency, enable `async` feature on `cuda-host` |
| `crates/backends/native/Cargo.toml` | Add `cuda-async` dependency |
| `crates/cuda/src/oxide_bridge.rs` | Add async launch wrappers via `module.kernel_async()` |
| `crates/cuda/src/oxide_bridge.rs` | Add `with_context` wrappers for cuBLASLt GEMMs and NCCL all-reduces |
| `crates/cuda/src/stream.rs` | Non-blocking streams instead of null stream |
| `crates/backends/native/src/decode.rs` | Rewrite from imperative to `and_then` chain builder |
| `crates/backends/native/src/engine.rs` | Use `async_on` + `synchronize` instead of per-kernel sync |
| `crates/backends/native/src/sync.rs` | NCCL wrappers return `DeviceOperation`s |

### What We Don't Do (Yet)

- **No buffer ownership migration**: Keep cudarc `CudaSlice<T>`, not `DeviceBox<[T]>`. Weights and workspace buffers stay as-is, accessed via `cu_deviceptr()` in async launches.
- **No round-robin scheduling**: Single non-blocking stream per GPU, pinned via `async_on`. The round-robin scheduler's value is for multi-stream overlap, which Phase 045 established is not feasible for M=1 decode.
- **No continuous batching**: Future phase. The async pipeline foundation enables it, but decode-only first.
- **No prefill migration**: Decode path only. Prefill stays synchronous.

## Architecture: Arc'd GPU Resources + with_context Closures

### The Problem with and_then + Mutable State

The `DeviceOperation` trait requires `Send + 'static` on outputs and `Send` on closures.
`ForwardEngine` holds all GPU state as `&mut self` fields. Multiple `and_then` closures
can't each capture `&mut ForwardEngine` — Rust's borrow checker prevents multiple mutable borrows,
even though they execute sequentially.

### The Solution: Split GPU Resources into Arc'd Components

Refactor the GPU resources that `ForwardEngine` holds into `Arc`'d, `Send + Sync` components:

1. **`Arc<GpuResources>`**: Holds `OxideKernels`, `GemmEngine`, `NcclCommunicator`, `GpuWeightCache` — all immutable, shared across sequences
2. **`DecodeWorkspace`**: Per-sequence mutable workspace buffers (already separate, just needs to be `Arc<Mutex<DecodeWorkspace>>` or allocated per-sequence)
3. **`Arc<ModelConfig>`**: Already `Arc`'d

Each layer's forward pass is a `with_context` closure that captures `Arc<GpuResources>` and `&mut DecodeWorkspace`:

```rust
fn build_layer_decode(
    res: Arc<GpuResources>,
    ws: Arc<Mutex<DecodeWorkspace>>,
    layer_idx: usize,
    config: Arc<ModelConfig>,
) -> impl DeviceOperation<Output = ()> {
    with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let mut ws = ws.lock().unwrap();
        
        // Call existing sync methods — they use the stream parameter
        res.norm1(stream, &mut ws, layer_idx)?;
        res.attn_decode(stream, &mut ws, layer_idx)?;
        res.all_reduce(stream, &mut ws.attn_out)?;
        res.residual_add(stream, &mut ws)?;
        res.norm2(stream, &mut ws, layer_idx)?;
        res.mlp(stream, &mut ws, layer_idx)?;
        res.all_reduce(stream, &mut ws.mlp_out)?;
        res.residual_add(stream, &mut ws)?;
        
        value(())
    })
}
```

### Stream Handling

The `with_context` closure gets `ctx.get_cuda_stream()` — a `&Arc<cuda_core::CudaStream>`.
But our sync methods use cudarc streams (`&Arc<cudarc::driver::CudaStream>`).

**Solution**: Create a cudarc `CudaStream` and a cuda-core `CudaStream` that share the same
underlying `CUstream` handle. The `OxideKernels.cc_stream` and the cudarc `StreamPool` stream
both wrap the same `CUstream`. The `with_context` closure gets the cuda-core stream for
`module.kernel_async()` calls, while sync methods use the cudarc stream.

For the MVP (single stream), both use the null stream — no new stream needed.

### Execution

```rust
let pipeline = value(())
    .and_then(move |()| build_layer_decode(res.clone(), ws.clone(), 0, config.clone()))
    .and_then(move |()| build_layer_decode(res.clone(), ws.clone(), 1, config.clone()))
    // ... 48 layers
    .and_then(move |()| build_final_norm_and_lm_head(res, ws, config));

// Execute
pipeline.sync()?;
```

For continuous batching later: each sequence gets its own `DecodeWorkspace` + `and_then` chain,
spawned as `tokio::spawn(pipeline.into_future())`.

## Implementation Plan

### Task 1: Enable async feature + add dependencies (XS) ✅

Done. `cuda-async` added, `async` feature enabled on `cuda-host` and `infers-kernel-lib`.

### Task 2: Split GPU resources into Arc'd components (M)

Refactor `ForwardEngine` to extract an `Arc<GpuResources>` struct that holds all the
immutable GPU resources (kernels, GEMM engines, weight caches, NCCL communicator, config).
Workspace buffers remain per-sequence mutable state.

**Files**:
- `crates/backends/native/src/engine.rs` — extract `GpuResources` struct
- `crates/backends/native/src/resources.rs` (new) — `GpuResources` definition

**Acceptance criteria**:
- `GpuResources` is `Send + Sync` (all fields `Arc`'d or `Send + Sync`)
- `ForwardEngine` holds `Arc<GpuResources>` instead of individual fields
- Existing sync code still compiles and passes smoke test
- `DecodeWorkspace` referenced via `Arc<Mutex<DecodeWorkspace>>`

### Task 3: Make OxideKernels stream dynamic (S)

Change all 44 launch methods to accept a `&cuda_core::CudaStream` parameter for kernel dispatch,
instead of using `&self.cc_stream`. The cudarc stream parameter (for `CudaSliceView` guards) stays.

**Files**:
- `crates/cuda/src/oxide_bridge.rs`

**Acceptance criteria**:
- All 44 launch methods accept `dispatch_stream: &cuda_core::CudaStream` parameter
- Kernel dispatch uses `dispatch_stream` instead of `&self.cc_stream`
- Smoke test passes (parity with 0.036s/step)

### Task 4: Build per-layer decode pipeline as and_then chain (M)

Rewrite `decode.rs` to build the decode forward pass as `with_context` closures chained
with `and_then`. Each closure calls the existing sync methods (norm, GEMM, attention/GDN,
NCCL, residual_add).

**Files**:
- `crates/backends/native/src/decode.rs` — rewrite as pipeline builder

**Acceptance criteria**:
- `decode_paged()` builds `and_then` chain, executes via `pipeline.sync()`
- Smoke test produces "Paris"
- Decode latency: 0.036s/step (parity)

### Task 5: Benchmark + verify (XS)

Run smoke test with timing, verify parity.

**Acceptance criteria**:
- Smoke test: "Paris" output, 30 tokens generated
- Decode latency: ≤ 0.037s/step (parity ± 1ms)
- No regressions in prefill path

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Async feature changes macro codegen | Compilation failure | Test incrementally — enable feature first, check compilation |
| cuBLASLt stream binding | GEMM launches on wrong stream | Use `with_context` to get stream from ExecutionContext, create GemmEngine bound to that stream |
| NCCL on non-blocking stream | Deadlock or corruption | NCCL works on non-blocking streams; test with smoke test |
| `and_then` closure captures | Borrow checker fights | Use `value()` baton pattern, `Arc` for shared weights |
| Recursive `execute()` stack depth | Stack overflow for 48 layers × 12 ops = 576 deep | Test — Rust's default stack is 8MB, each frame is ~100 bytes, 576 frames = ~60KB, well within limits |
