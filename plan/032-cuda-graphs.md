# Phase 32: CUDA Graph Capture/Replay for Zero-Overhead Decode

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: None (final optimization layer)
**Blocked by**: Phase 28 (fused GEMM — stable kernel count), Phase 29 (pre-allocated workspace — stable buffer addresses), Phase 30 (async pipeline — stable execution graph), Phase 31 (continuous batching — stable max_batch)
**Rationale**: The decode step currently launches ~9000 kernels per token. Even with async pipelining, each kernel launch has 2-5µs CPU overhead. At 9000 launches, that's 18-45ms of pure CPU overhead per token. CUDA graphs capture the entire kernel sequence once, then replay it with a single API call (~10µs), eliminating all launch overhead. This is what vLLM does for its decode step. Combined with mapped pinned memory for input updates, the decode loop becomes: update mapped memory → replay graph → read output.
---

## Goal

After this phase, the decode loop at steady state is:

```rust
// Per decode step (steady state):
1. Write token_ids + positions + block_tables to mapped pinned memory (CPU, ~1µs)
2. cuda_graph.replay() (~10µs — all 64 layers, NCCL, sampling in one call)
3. Read sampled tokens from mapped memory (CPU, ~1µs)
// Total CPU overhead: ~12µs (vs ~18-45ms with 9000 launches)
```

## Current State

- No CUDA graph infrastructure exists
- ~9000 kernel launches per decode step at 2-5µs each = 18-45ms CPU overhead
- With Phase 30's async pipeline, launches are still submitted individually
- Buffer addresses are allocated fresh each step (Phase 029 fixes this)

## Target Architecture

### Graph Lifecycle

```
1. First decode step: "warm-up" — execute pipeline normally, building graph
   - cuda_stream_begin_capture()
   - Execute all kernels (async pipeline from Phase 30)
   - cuda_stream_end_capture() → CudaGraph
   - Instantiate: cuda_graph_instantiate() → CudaGraphExec

2. Subsequent steps: "replay"
   - Update inputs via mapped pinned memory (CPU writes, no GPU sync)
   - cuda_graph_launch(graph_exec, stream)
   - Stream sync only when result needed (before reading output)

3. Batch size change: "recapture"
   - If batch size changes (sequence joins/leaves), recapture the graph
   - New graph instantiated for the new batch shape
   - Cache per batch size (e.g., graphs for batch=1, 2, 3, 4)
```

### Mapped Pinned Memory

CUDA mapped memory is simultaneously accessible from CPU and GPU:
- CPU writes to the pinned host buffer
- GPU reads from the device pointer (same physical memory, different address)
- No `clone_htod`, no `synchronize` — the write is visible to the next kernel

```rust
pub struct MappedMemory<T> {
    host_ptr: *mut T,      // CPU address
    device_ptr: CUdeviceptr, // GPU address (same memory)
    len: usize,
}

impl<T: Copy> MappedMemory<T> {
    pub fn new(len: usize) -> Result<Self> {
        // Allocate pinned, mapped memory
        let num_bytes = len * std::mem::size_of::<T>();
        let mut host_ptr = std::ptr::null_mut();
        let mut flags = CUDALLOCATION::HOST_REGISTER_MAPPED;
        cuda_host_alloc(&mut host_ptr, num_bytes, flags)?;
        let mut device_ptr: CUdeviceptr = 0;
        cuda_host_get_device_pointer(&mut device_ptr, host_ptr, 0)?;
        Ok(Self { host_ptr: host_ptr as *mut T, device_ptr, len })
    }

    pub fn write(&self, data: &[T]) {
        // CPU write — visible to GPU without explicit copy
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.host_ptr, data.len());
        }
    }

    pub fn device_ptr(&self) -> CUdeviceptr {
        self.device_ptr
    }
}
```

### What Gets Captured in the Graph

1. Embedding lookup (reads token_ids from mapped memory)
2. All 64 layer forward passes (norm1 → GEMMs → attention/GDN → all_reduce → residual → norm2 → MLP → all_reduce → residual)
3. Final norm + lm_head GEMM
4. Argmax sampling (writes sampled tokens to mapped memory)

NCCL all-reduce operations are captured in the graph (they're just kernel launches on the CUDA stream). The graph includes both GPU 0 and GPU 1 streams.

### What Does NOT Get Captured

- Token ID update (CPU writes to mapped pinned memory)
- Position update (CPU writes to mapped pinned memory)
- Block table update (CPU writes to mapped pinned memory)
- Reading sampled tokens (CPU reads from mapped pinned memory)
- KV cache page allocation (happens during prefill, not decode)
- Sequence scheduling decisions (CPU-only logic)

## Implementation Plan

### Step 1: Add CUDA graph API to cudarc wrapper

cudarc already has `cuStreamBeginCapture`/`cuStreamEndCapture` in its CUDA driver bindings. Add a high-level wrapper:

```rust
// crates/cuda/src/graph.rs
pub struct CudaGraph {
    graph: CUgraph,
    exec: CUgraphExec,
}

impl CudaGraph {
    pub fn capture<F>(stream: &CudaStream, f: F) -> Result<Self>
    where
        F: FnOnce() -> Result<()>
    {
        stream.begin_capture()?;
        f()?;
        let graph = stream.end_capture()?;
        let exec = graph.instantiate()?;
        Ok(Self { graph, exec })
    }

    pub fn replay(&self, stream: &CudaStream) -> Result<()> {
        unsafe {
            cuGraphLaunch(self.exec, stream.handle())?;
        }
        Ok(())
    }
}
```

### Step 2: Add mapped pinned memory

```rust
// crates/cuda/src/mapped.rs
pub struct MappedMemory<T> { /* as above */ }
```

### Step 3: Convert decode inputs to mapped memory

In `ForwardEngine`, replace per-step H2D copies with mapped memory:

```rust
pub struct ForwardEngine {
    // ... existing ...
    token_ids_mapped: MappedMemory<i32>,      // [max_batch]
    positions_mapped: MappedMemory<i32>,     // [max_batch]
    block_tables_mapped: MappedMemory<i32>,  // [max_batch, max_pages_per_seq]
    sampled_tokens_mapped: MappedMemory<i32>, // [max_batch] — GPU writes, CPU reads
}
```

### Step 4: Capture decode graph

On the first decode step after a batch size change:
```rust
fn capture_decode_graph(&mut self, batch_size: usize) -> Result<CudaGraph> {
    // Ensure workspace buffers are sized for this batch size
    // Write test values to mapped memory
    self.token_ids_mapped.write(&vec![0; batch_size]);

    let graph = CudaGraph::capture(&self.stream, || {
        // Execute the full async pipeline (Phase 30)
        let pipeline = self.build_decode_pipeline(batch_size);
        pipeline.block_on()  // execute and capture
    })?;

    Ok(graph)
}
```

### Step 5: Replay in decode loop

```rust
fn decode_paged(&mut self, batch: &DecodeBatch) -> Result<Vec<u32>> {
    // If batch size changed, recapture graph
    if self.cached_batch_size != batch.len() {
        self.decode_graph = self.capture_decode_graph(batch.len())?;
        self.cached_batch_size = batch.len();
    }

    // Update inputs via mapped memory (zero-copy)
    self.token_ids_mapped.write(&batch.token_ids);
    self.positions_mapped.write(&batch.positions);
    self.block_tables_mapped.write(&batch.block_tables_flattened);

    // Replay captured graph
    self.decode_graph.replay(&self.stream)?;

    // Read results (may need stream sync for the read to be correct)
    self.stream.synchronize()?;
    let tokens: Vec<i32> = self.sampled_tokens_mapped.read(batch.len());

    Ok(tokens.iter().map(|&t| t as u32).collect())
}
```

### Step 6: Graph cache for different batch sizes

```rust
pub struct ForwardEngine {
    decode_graphs: HashMap<usize, CudaGraph>,  // keyed by batch_size
}
```

When a sequence joins/leaves, the batch size changes. On first occurrence of a new batch size, capture a new graph. Subsequent steps with the same batch size replay the cached graph.

## Considerations

### Graph Recapture Triggers
- Batch size change (sequence joins/leaves)
- Max sequence length exceeding previous max (block table grows — requires updating mapped memory size)
- Sampling parameters change (temperature, top-k — different sampling kernel path)

### NCCL in Graphs
NCCL collectives in graph mode require `nccl_group_start()` / `nccl_group_end()` to be inside the capture. The existing code already uses group_start/end. The graph replay will re-execute the NCCL calls, which is correct as long as the communicator state is stable.

### KV Cache Writes
The paged KV cache write (storing K/V for the current token) is part of the attention decode forward pass. It's captured in the graph. The block table in mapped memory tells the kernel which physical page to write to.

### GDN State
GDN recurrent state is mutable and persists across decode steps. The state update is part of the captured graph — it reads the previous state and writes the updated state. Since the state buffer address is fixed (pre-allocated in Phase 029), the graph correctly references it.

## Verification

```bash
# Correctness
python3 scripts/compare_hidden_states.py --oracle-dir /tmp/oracle_int4 --infer-dir /tmp/infer_dumps_int4
# Target: cosine ≥ 0.99 (graph replay must produce identical results)

# Performance
nsys profile ./target/release/infer --model ... --max-tokens 50 --no-chat
# Check nsys: graph launch should be a single cuGraphLaunch call
# CPU time per token should drop dramatically

time ./target/release/infer --model ... --max-tokens 50 --no-chat
# Target: close to theoretical minimum (weight bandwidth limited)
# For INT4 on RTX 5060 Ti: ~14ms/token theoretical → ~20-30ms/target with overhead
```

## Files Modified

| File | Change |
|------|--------|
| new: `crates/cuda/src/graph.rs` | CudaGraph capture/replay wrapper |
| new: `crates/cuda/src/mapped.rs` | Mapped pinned memory |
| `crates/backends/native/src/engine.rs` | Graph capture on first step; replay on subsequent; mapped memory for inputs |
| `crates/cuda/src/lib.rs` | Export graph + mapped modules |

## Expected Performance Breakthrough

After Phases 28-32, the decode path should look like:

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Time per token | ~2600ms | ~20-30ms | ~100x |
| Kernel launches per token | ~9000 | 1 (graph replay) | 9000x |
| CPU-GPU syncs per token | ~144 | 0 (steady state) | ∞ |
| Memory allocations per token | ~1000+ | 0 (steady state) | ∞ |
| GPU0 utilization | 100% (memory-bound, 80W) | ~95% (compute-bound, 160W+) | Higher efficiency |
| GPU1 utilization | 80% (sync bubbles) | ~95% (overlapped) | +15% |
| Weight reads per GEMM | 52MB (BF16 dequant) | 13MB (compressed INT4) | 4x |
| Batch support | 1 | Up to 4 (configurable) | 4x throughput |
