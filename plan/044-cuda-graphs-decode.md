# Phase 044: CUDA Graph Capture/Replay for Decode

---
**Status**: IN PROGRESS
**Last Updated**: 2026-06-25
**Blocks**: Phase 045 (NCCL pipeline overlap)
**Blocked by**: Phase 043 (graph prerequisites — H2D elimination, pre-allocated buffers, dynamic params)
**Rationale**: The decode step executes ~3,400 kernel launches per token. Each launch has ~2µs CPU overhead. That's ~6.8ms of pure CPU dispatch overhead — the single largest remaining optimization target. CUDA graphs capture the entire computation graph once, then replay it with a single API call (~10µs), eliminating all per-kernel launch overhead. This is what vLLM does for its decode step.
---

## Goal

Reduce CPU dispatch overhead from ~6.8ms/step to near-zero by capturing the decode computation graph and replaying it each step.

**Target**: 0.036s → 0.030s (6ms savings) → 0.029s with warm-up optimization.

## Architecture

### Graph Lifecycle

```
Step 0 (warm-up): Execute decode normally. GPU caches warm up.
Step 1 (capture): Begin capture → execute decode loop → end capture → CudaGraph
Step 2+ (replay): Write staging buffers → graph.launch() → read output
```

### What Gets Captured

All operations on the null stream:
- cudarc kernel launches (cuBLASLt, memcpy, etc.)
- cuda-oxide kernel launches (OxideKernels via cc_stream)
- NCCL all-reduce (group_start/end)

All use the same null/default stream, so a single capture captures everything.

### What Changes Between Steps

Dynamic data written to pre-allocated device staging buffers before graph launch:
- `token_ids_staging` — current input token
- `position_staging` — current position
- `block_table_staging` — current page mapping
- `rope_position_staging` — position for RoPE
- `num_cached_tokens_staging` — cached token count

These are written via `memcpy_htod` into fixed-address device buffers. The captured graph references these buffer addresses, so when the data changes, the kernels read the new values automatically.

### Graph Invalidation

The graph becomes invalid if:
- Buffer addresses change (they don't — pre-allocated workspace)
- Grid dimensions change (they don't — all constant)
- Shared memory sizes change (they don't — fixed max)

For our decode loop, the graph **never** becomes invalid during steady-state decode. We capture once and replay indefinitely.

KV page allocation (every page_size=16 steps) changes `num_pages` but this kernel argument is effectively unused by the paged attention decode kernel. The block table data is written to the staging buffer before each launch.

## Implementation Plan

### Step 1: Expose cudarc CudaGraph API

In `crates/cuda/src/lib.rs`, re-export:
```rust
pub use cudarc::driver::safe::graph::CudaGraph;
```

cudarc 0.19.x provides:
- `CudaStream::begin_capture(mode)` / `end_capture(flags)` → `Option<CudaGraph>`
- `CudaGraph::launch()` — replays on the captured stream
- `CudaGraph::upload()` — pre-uploads resources (optional warmup)

### Step 2: Add graph state to ForwardEngine

```rust
pub struct ForwardEngine {
    // ... existing ...
    decode_graphs: Vec<Option<Arc<CudaGraph>>>,
    graph_capture_step: usize,  // 0 = warm-up, 1 = capture, 2+ = replay
}
```

### Step 3: Modify decode_paged

```rust
fn decode_paged(&mut self, ...) -> Result<Vec<u32>> {
    // Write staging buffers (always — both capture and replay need current data)
    for gpu_idx in 0..num_gpus {
        self.write_staging_buffers(gpu_idx, position, token_ids, ...)?;
    }

    match self.graph_capture_step {
        0 => {
            // Warm-up: execute normally
            self.execute_decode_step(...)?;
            self.graph_capture_step = 1;
        }
        1 => {
            // Capture: record the computation graph
            for gpu_idx in 0..num_gpus {
                self.streams.get(gpu_idx)?.begin_capture(CU_STREAM_CAPTURE_MODE_GLOBAL)?;
            }
            self.execute_decode_step(...)?;
            for gpu_idx in 0..num_gpus {
                let graph = self.streams.get(gpu_idx)?.end_capture(0)?;
                self.decode_graphs[gpu_idx] = graph.map(Arc::new);
            }
            self.graph_capture_step = 2;
        }
        _ => {
            // Replay: launch captured graph
            for gpu_idx in 0..num_gpus {
                if let Some(ref graph) = self.decode_graphs[gpu_idx] {
                    graph.launch()?;
                }
            }
        }
    }
    
    // Read output (sampled token)
    self.sample_and_update(...)?;
}
```

### Step 4: Handle sampling outside graph

The sampling step (argmax + D→H readback) happens AFTER the graph launch. The logits are in `ws.logits` which was computed inside the graph. The argmax kernel could also be part of the graph, but the D→H copy to read the result must happen outside.

Strategy: Keep sampling outside the graph. After `graph.launch()`, synchronize the stream, then read the argmax result.

### Step 5: Handle NCCL within graph

NCCL operations are just kernel launches on the stream. During capture, they are recorded as graph nodes. During replay, they execute correctly. No special handling needed.

The `group_start()`/`group_end()` calls are part of the capture. They define the all-reduce scope within the graph.

### Step 6: Remove GPU timing events from captured path

Move `gpu_start_events`/`gpu_end_events` creation and recording to outside the graph capture window. Only time warm-up and capture steps — replay steps use `CudaEvent::record()` before/after `graph.launch()`.

## Verification

```bash
cargo test --release -p infers-backend-native -- smoke_test_real_model --ignored --nocapture
# Target: PASSED, decode avg improved by ~5-7ms

nsys profile — check that replay shows a single cuGraphLaunch call instead of 3400 individual kernel launches
```

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| NCCL capture fails | Graph creation error | NCCL 2.23+ supports graphs; check version. Fall back to non-graph if capture fails. |
| Warm-up step produces wrong output | Correctness | Warm-up output is discarded (first token only). |
| Shared memory overflow | Kernel crash | Fixed max shared memory (19KB for paged attn) must fit GPU limits. |
| Graph capture on null stream | May capture cross-stream ops | We only use the null stream, so this is correct. |
| CudaSliceView SyncOnDrop conflicts | Capture breaks | CudaSliceView must not insert sync points during capture. Verify that cc_stream is the same null stream being captured. |
