# Phase 15: Tracing Integration + OTLP Export

---
**Status**: DONE
**Last Updated**: 2026-06-21
**Blocks**: Performance profiling, bottleneck identification
**Blocked by**: Nothing
**Rationale**: OTLP tracing integrated. Layer spans, GPU timing via CudaEvent, I/O latency spans, scheduler/orchestrator spans all wired. Integration test passes.
---

## Goals

1. **Export structured traces via OpenTelemetry OTLP/gRPC** — every inference step appears as a span in Jaeger/Tempo/Grafana with timing, attributes, and parent-child relationships.
2. **Measure CPU wall-clock vs GPU execution time** — CUDA event timing reveals CPU launch overhead, GPU saturation, and async gaps. This is the key diagnostic for a 200× performance gap.
3. **Identify I/O bottlenecks** — weight upload, logits download, and CUDA synchronization are instrumented with spans showing their latency contribution.
4. **Zero cost when disabled** — when `--otlp-enabled=false` (default), all spans compile away to no-ops via `tracing`'s macro layer. No allocations, no FFI, no overhead.

## Non-Goals

- GPU-side profiling (Nsight Compute, CUPTI) — those are external tools, not application-level instrumentation.
- Replacing the `probe` module (Phase 13) — probe dumps *tensor data* for correctness; tracing records *timing* for performance. They are complementary.
- Custom OpenTelemetry metrics exporter — the existing `prometheus` endpoint at `/metrics` covers that. OTLP here is traces/spans only.
- Per-operation (per-GEMM, per-kernel) spans in the forward pass — layer-level granularity first. Operation-level can be added later behind an env gate.

## Architecture

### Layered Tracing Subscriber

The `infers-server` binary initializes a `tracing_subscriber::Registry` with two optional layers:

```
tracing_subscriber::registry()
    .with(fmt::Layer::default().with_env_filter(...))   // always on (console logs)
    .with(otlp_layer)                                    // conditional on --otlp-enabled
```

The OTLP layer uses `tracing-opentelemetry` to convert `tracing` spans into OpenTelemetry spans, then exports them via `opentelemetry-otlp` over gRPC (tonic) to a collector (Jaeger, Tempo, Grafana OTLP, etc.).

**Dependency chain**:

```
tracing (emit spans)
  → tracing-subscriber (Registry + layers)
    → tracing-opentelemetry (Bridge: tracing span → OTel span)
      → opentelemetry-sdk (SpanProcessor, Exporter pipeline)
        → opentelemetry-otlp (OTLP exporter)
          → tonic (gRPC transport)
```

### Span Taxonomy

Spans form a hierarchy. The parent-child relationship is established by `tracing`'s `#[instrument]` attribute and explicit `enter()` calls.

```
prefill (info_span)                        # Top-level request trace
  ├── weight_cache_build (info_span)       # One-time at startup
  ├── scheduler_step (info_span)           # Scheduling iteration
  │   ├── schedule_batch (debug_span)
  │   └── evict_session (debug_span)
  ├── prefill_session (info_span)           # Per-session prefill
  │   ├── embed (debug_span)
  │   ├── layer_0 (info_span)              # Per-layer, with attributes
  │   │   ├── norm1 (debug_span)
  │   │   ├── attention | gdn (info_span)  # Dispatched by layer type
  │   │   ├── nccl_all_reduce (debug_span) # kind="attention"|"mlp"
  │   │   ├── residual (debug_span)
  │   │   ├── norm2 (debug_span)
  │   │   ├── mlp (info_span)
  │   │   │   ├── gate_proj (debug_span)
  │   │   │   ├── up_proj (debug_span)
  │   │   │   ├── silu (debug_span)
  │   │   │   ├── down_proj (debug_span)
  │   │   │   ├── nccl_all_reduce (debug_span)
  │   │   │   └── residual (debug_span)
  │   │   └── ...
  │   ├── layer_1 ...
  │   ├── final_norm (debug_span)
  │   ├── lm_head (debug_span)
  │   └── sample (debug_span)
  └── decode_session (info_span)           # Per-session decode
      └── (same layer structure as prefill)
```

**Span attribute conventions**:

| Attribute | Type | Example | Source |
|-----------|------|---------|--------|
| `layer_idx` | u64 | 3 | Per-layer span |
| `layer_type` | string | `"full_attn"` / `"gdn"` | Per-layer span |
| `gpu_idx` | u64 | 0 | Per-GPU operations |
| `phase` | string | `"prefill"` / `"decode"` | Top-level span |
| `num_tokens` | u64 | 14 | Prefill span |
| `session_id` | u64 | 7 | Session span |
| `kind` | string | `"attention"` / `"mlp"` | NCCL all-reduce |
| `tensor_name` | string | `"model.layers.3.self_attn.q_proj"` | Weight upload |
| `gpu_time_ms` | f64 | 155.3 | CUDA event timing |
| `cpu_wall_ms` | f64 | 162.1 | Derived from span duration |
| `launch_overhead_ms` | f64 | 6.8 | `gpu_time_ms - cpu_wall_ms` |

### CUDA Event Timing

CPU wall-clock time (from `tracing` spans) measures *launch + wait* time. GPU execution time (from CUDA events) measures *actual kernel execution*. The difference reveals:

- **CPU launch overhead** — `cpu_wall_ms > gpu_time_ms` means the CPU is doing significant work between kernel launches (e.g., GEMM config, buffer allocation).
- **GPU saturation** — `cpu_wall_ms ≈ gpu_time_ms` means the GPU is fully occupied.
- **Async gaps** — `cpu_wall_ms < gpu_time_ms` should not happen (CPU launched, GPU still running) unless the CPU moves ahead and later synchronization catches it.

**Implementation**: A `CudaEvent` wrapper in `infers-cuda`:

```rust
pub struct CudaEvent {
    event: cudaEvent_t,  // raw CUDA driver API handle
}

impl CudaEvent {
    pub fn new() -> Result<Self>;
    pub fn record(&self, stream: &CudaStream);
    pub fn synchronize(&self);
    pub fn elapsed_ms(start: &CudaEvent, end: &CudaEvent) -> Result<f32>;
}

impl Drop for CudaEvent {
    fn drop(&mut self) { unsafe { cudaEventDestroy(self.event); } }
}
```

Uses raw CUDA Driver API FFI (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`) — these are in `libcuda.so` and always available. No cudarc wrapper exists for events, so we add our own.

**Timing integration in engine.rs**:

```rust
let start_event = CudaEvent::new()?;
let end_event = CudaEvent::new()?;

start_event.record(&stream);
// ... entire layer loop ...
end_event.record(&stream);
end_event.synchronize();

let gpu_time_ms = CudaEvent::elapsed_ms(&start_event, &end_event)?;
tracing::info!(gpu_time_ms, phase = "prefill", "GPU timing complete");
```

For per-layer timing, wrap each layer dispatch with its own start/end events. The overhead of `cudaEventRecord` is ~1μs — negligible compared to per-layer execution time (~3-5ms target).

**Key constraint**: CUDA events are recorded on a specific stream. `CudaEvent::elapsed_ms` measures time between two events *on the same stream*. For TP=2, each GPU has its own stream, so per-GPU timing requires per-GPU event pairs. The top-level span reports the max across GPUs (bottleneck GPU).

## Task Breakdown

### Commit 1: Workspace + server OTLP dependencies

**Files**: `Cargo.toml`, `crates/server/Cargo.toml`

Add workspace dependencies:

```toml
# [workspace.dependencies] additions
opentelemetry = "0.27"
opentelemetry_sdk = "0.27"
opentelemetry-otlp = { version = "0.27", features = ["grpc-tonic", "metrics"] }
tracing-opentelemetry = "0.28"
tonic = "0.12"
```

Add to `crates/server/Cargo.toml`:

```toml
opentelemetry = { workspace = true }
opentelemetry_sdk = { workspace = true }
opentelemetry-otlp = { workspace = true }
tracing-opentelemetry = { workspace = true }
tonic = { workspace = true }
```

**Complexity**: XS
**Timebox**: 15 min
**Acceptance**: `cargo check --release -p infers-server` compiles.

---

### Commit 2: OTLP CLI arguments

**Files**: `crates/server/src/main.rs`

Add three new CLI args to `Args`:

```rust
/// Enable OTLP trace export
#[arg(long, default_value = "false")]
pub otlp_enabled: bool,

/// OTLP gRPC endpoint
#[arg(long, default_value = "http://localhost:4317")]
pub otlp_endpoint: String,

/// OTLP service name
#[arg(long, default_value = "infers")]
pub otlp_service_name: String,
```

**Complexity**: XS
**Timebox**: 10 min
**Acceptance**: `--help` shows the new args. `cargo check --release` compiles.

---

### Commit 3: Layered tracing subscriber

**Files**: `crates/server/src/main.rs`

Replace the existing `tracing_subscriber::fmt()` init with a layered subscriber:

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Registry};

let env_filter = EnvFilter::try_from_default_env()
    .unwrap_or_else(|_| EnvFilter::new(&args.log_level));

let fmt_layer = tracing_subscriber::fmt::layer()
    .with_target(false)
    .with_thread_ids(true);

if args.otlp_enabled {
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(&args.otlp_endpoint);
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .install_batch(opentelemetry_sdk::runtime::Tokio)
        .expect("Failed to install OTLP tracer");
    let otlp_layer = tracing_opentelemetry::layer()
        .with_tracer(tracer);

    Registry::default()
        .with(env_filter)
        .with(fmt_layer)
        .with(otlp_layer)
        .init();
} else {
    Registry::default()
        .with(env_filter)
        .with(fmt_layer)
        .init();
}
```

Key design decisions:
- `Registry` (not `fmt()`) as the base subscriber — required for layering.
- `fmt_layer` always present — console output is always useful.
- `otlp_layer` conditional on `--otlp-enabled` — zero overhead when off.
- `install_batch(Tokio)` — batch export is more efficient than simple (spans buffered, sent in batches).
- The tracer is installed once at startup. All crates that `use tracing` emit spans automatically.

**Complexity**: S
**Timebox**: 30 min
**Acceptance**:
- With `--otlp-enabled=false` (default), server logs work as before.
- With `--otlp-enabled=true --otlp-endpoint=http://localhost:4317` and a Jaeger backend running, traces appear in Jaeger UI.
- No compile errors.

---

### Commit 4: `CudaEvent` FFI wrapper

**Files**: `crates/cuda/src/event.rs`, `crates/cuda/src/lib.rs`, `crates/cuda/Cargo.toml`

New module `event.rs` in `infers-cuda`:

```rust
//! CUDA event wrappers for GPU-side timing measurement.

use std::ffi::c_void;
use anyhow::Result;

/// Opaque handle to a CUDA event.
pub struct CudaEvent {
    event: cudaEvent_t,
}

// Raw CUDA driver API types
type cudaEvent_t = *mut c_void;

extern "C" {
    fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
    // Need stream type from cudarc
}

type cudaStream_t = *mut c_void;
type cudaError_t = i32;

const CUDA_SUCCESS: cudaError_t = 0;

impl CudaEvent {
    /// Create a new CUDA event.
    pub fn new() -> Result<Self> {
        let mut event: cudaEvent_t = std::ptr::null_mut();
        let err = unsafe { cudaEventCreate(&mut event) };
        if err != CUDA_SUCCESS {
            anyhow::bail!("cudaEventCreate failed: error {}", err);
        }
        Ok(Self { event })
    }

    /// Record the event on the given stream.
    pub fn record(&self, stream: &cudarc::driver::CudaStream) {
        // Extract raw stream pointer from cudarc's CudaStream
        // cudarc exposes .stream() for the raw custream_t
        let raw_stream = stream.stream() as cudaStream_t;
        let err = unsafe { cudaEventRecord(self.event, raw_stream) };
        if err != CUDA_SUCCESS {
            tracing::warn!("cudaEventRecord failed: error {}", err);
        }
    }

    /// Synchronize on this event (block CPU until event completes on GPU).
    pub fn synchronize(&self) -> Result<()> {
        let err = unsafe { cudaEventSynchronize(self.event) };
        if err != CUDA_SUCCESS {
            anyhow::bail!("cudaEventSynchronize failed: error {}", err);
        }
        Ok(())
    }

    /// Compute elapsed milliseconds between two events.
    ///
    /// Both events must be recorded on the same stream, and `end` must
    /// complete after `start`. Call `end.synchronize()` before calling this.
    pub fn elapsed_ms(start: &CudaEvent, end: &CudaEvent) -> Result<f32> {
        let mut ms: f32 = 0.0;
        let err = unsafe { cudaEventElapsedTime(&mut ms, start.event, end.event) };
        if err != CUDA_SUCCESS {
            anyhow::bail!("cudaEventElapsedTime failed: error {}", err);
        }
        Ok(ms)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let err = unsafe { cudaEventDestroy(self.event) };
        if err != CUDA_SUCCESS {
            tracing::warn!("cudaEventDestroy failed: error {}", err);
        }
    }
}

// Safety: CudaEvent owns a raw CUDA event handle and cleans it up in Drop.
// It is not Send/Sync by default because CUDA events are stream-bound.
// However, once recorded, reading elapsed_ms is thread-safe (read-only query).
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}
```

**cudarc stream access**: The `cudarc::driver::CudaStream` type has a `stream()` method returning the raw `CUstream` pointer. We use this to pass to `cudaEventRecord`. Verify by checking cudarc 0.19.7 API.

**Linking**: `libcuda.so` is dynamically linked by the CUDA driver — always available on CUDA systems. No extra `-l` flags needed.

Register the module in `lib.rs`:
```rust
pub mod event;
pub use event::CudaEvent;
```

**Complexity**: M
**Timebox**: 45 min
**Acceptance**:
- `cargo check --release -p infers-cuda` compiles.
- `CudaEvent::new()` creates an event.
- `CudaEvent::elapsed_ms()` signature matches usage pattern.
- No runtime test yet (requires GPU) — just compilation check.

---

### Commit 5: Engine-level prefill/decode spans

**Files**: `crates/backends/native/src/engine.rs`

Add `tracing` spans to `prefill_paged()` and `decode_paged()`. These are the highest-value instrumentation points — they show the full request latency breakdown.

```rust
pub fn prefill_paged(&mut self, ...) -> Result<(usize, u32)> {
    let span = tracing::info_span!("prefill", num_tokens = token_ids.len(), num_layers = config.num_hidden_layers);
    let _enter = span.enter();

    // ... existing code ...

    // Per-layer loop
    for layer_idx in 0..config.num_hidden_layers {
        let layer_type = config.get_layer_type(layer_idx);
        let layer_span = tracing::info_span!(
            "layer",
            layer_idx,
            layer_type = match layer_type {
                LayerType::FullAttention => "full_attn",
                LayerType::GatedDeltaNet => "gdn",
            }
        );
        let _layer_enter = layer_span.enter();

        // norm1
        let norm_span = tracing::debug_span!("norm1");
        let _norm_enter = norm_span.enter();
        // ... rms_norm dispatch ...
        drop(_norm_enter);

        // attention or GDN
        match layer_type {
            LayerType::FullAttention => {
                let attn_span = tracing::info_span!("attention", gpu_idx);
                let _attn_enter = attn_span.enter();
                // ... attention dispatch ...
            }
            LayerType::GatedDeltaNet => {
                let gdn_span = tracing::info_span!("gdn", gpu_idx);
                let _gdn_enter = gdn_span.enter();
                // ... GDN dispatch ...
            }
        }

        // NCCL all-reduce
        {
            let ar_span = tracing::debug_span!("nccl_all_reduce", kind = "attention");
            let _ar_enter = ar_span.enter();
            // ... group_start, all_reduce_in_place, group_end ...
        }

        // residual
        {
            let res_span = tracing::debug_span!("residual");
            let _res_enter = res_span.enter();
            // ... add kernel ...
        }

        // norm2 + MLP
        {
            let norm2_span = tracing::debug_span!("norm2");
            let _n2_enter = norm2_span.enter();
            // ... rms_norm ...
        }
        {
            let mlp_span = tracing::info_span!("mlp");
            let _mlp_enter = mlp_span.enter();
            // gate_proj, up_proj, silu, down_proj, ar, residual
        }
    }

    // Final norm + LM head + sample
    {
        let final_span = tracing::debug_span!("final");
        let _f_enter = final_span.enter();
        // ... norm, lm_head, sample ...
    }
}
```

Same structure for `decode_paged()`.

**Important**: All spans use `skip_all` (no span-level string formatting). Attributes are set via structured fields (`layer_idx = 3`, `layer_type = "full_attn"`). This ensures OTLP exports them as key-value attributes, not concatenated strings.

**Complexity**: M
**Timebox**: 1 hour
**Acceptance**:
- `cargo check --release -p infers-backend-native` compiles.
- Running `infer` binary with `--otlp-enabled` produces traces showing the layer hierarchy.
- Existing `eprintln!` progress markers are replaced or supplemented with `tracing::info!`.

---

### Commit 6: CUDA event timing in engine

**Files**: `crates/backends/native/src/engine.rs`

Add GPU timing to `prefill_paged()` and `decode_paged()`:

```rust
use infers_cuda::CudaEvent;

// Before layer loop:
let gpu_start = CudaEvent::new()?;
let gpu_end = CudaEvent::new()?;

gpu_start.record(&stream);
// ... layer loop ...
gpu_end.record(&stream);
gpu_end.synchronize()?;

let gpu_time_ms = CudaEvent::elapsed_ms(&gpu_start, &gpu_end)?;
tracing::info!(gpu_time_ms, phase = "prefill", "GPU execution complete");
```

For per-layer timing (optional, lower priority):

```rust
// Inside per-layer loop, before dispatch:
let layer_start = CudaEvent::new()?;
let layer_end = CudaEvent::new()?;
layer_start.record(&stream);
// ... layer dispatch ...
layer_end.record(&stream);
layer_end.synchronize()?;
let layer_gpu_ms = CudaEvent::elapsed_ms(&layer_start, &layer_end)?;
tracing::debug!(layer_gpu_ms, layer_idx, "Layer GPU time");
```

**Per-layer event overhead**: Each `CudaEvent::new()` + `record()` + `synchronize()` adds ~5μs per layer. At 48 layers, that's ~240μs — negligible compared to the current 10s/step. But for production (20 tok/s target, 50ms/step), per-layer events add 0.5% overhead — acceptable for profiling, too expensive for always-on. Make per-layer events conditional on an env var:

```rust
let per_layer_timing = std::env::var("INFERS_TRACE_LAYER_TIMING").is_ok();
```

**TP=2 consideration**: For multi-GPU, record events on each GPU's stream. Report `gpu_time_ms` as the max across GPUs (bottleneck determines throughput).

```rust
// Per-GPU events
let mut gpu_starts = Vec::new();
let mut gpu_ends = Vec::new();
for gpu_idx in 0..num_gpus {
    gpu_starts.push(CudaEvent::new()?);
    gpu_ends.push(CudaEvent::new()?);
    gpu_starts[gpu_idx].record(&gpu_streams[gpu_idx]);
}
// ... layer loop ...
for gpu_idx in 0..num_gpus {
    gpu_ends[gpu_idx].record(&gpu_streams[gpu_idx]);
    gpu_ends[gpu_idx].synchronize()?;
}
let max_gpu_ms = (0..num_gpus)
    .map(|i| CudaEvent::elapsed_ms(&gpu_starts[i], &gpu_ends[i]).unwrap_or(0.0))
    .fold(0.0f32, f32::max);
tracing::info!(gpu_time_ms = max_gpu_ms as f64, phase = "prefill", "GPU execution complete (max across GPUs)");
```

**Complexity**: M
**Timebox**: 1 hour
**Acceptance**:
- `cargo check --release -p infers-backend-native` compiles.
- With `INFERS_TRACE_LAYER_TIMING=1`, per-layer GPU timing appears in trace spans.
- Top-level `gpu_time_ms` attribute appears on the prefill/decode span.

---

### Commit 7: I/O latency spans

**Files**: `crates/backends/native/src/upload.rs`, `crates/backends/native/src/sample.rs`, `crates/cuda/src/nccl.rs`

**Weight upload** (`upload.rs`):
The `upload_weight()` and `upload_int4_weight()` functions are called during `GpuWeightCache::new()` — this is the cold-start path. Instrument to show how much time is spent uploading weights:

```rust
pub fn upload_weight(...) -> Result<CudaSlice<bf16>> {
    let span = tracing::debug_span!("weight_upload", tensor = %name, bytes = bytes.len());
    let _enter = span.enter();
    // ... existing upload code + stream.synchronize() ...
    tracing::debug!(elapsed_ms = /* measure */, "weight uploaded");
    Ok(slice)
}
```

Note: `GpuWeightCache::new()` already logs `tracing::info!("GPU {}: cached {} weights", ...)` — this adds per-tensor timing.

**Logits download** (`sample.rs` — `sample_with_config`):
Non-greedy sampling requires downloading logits from GPU to CPU:

```rust
// In sample_with_config(), after clone_dtoh:
let span = tracing::debug_span!("logits_download", vocab_size = logits.len());
let _enter = span.enter();
let cpu_logits: Vec<bf16> = stream.clone_dtoh(logits)?;
```

**CUDA synchronization** — explicit `stream.synchronize()` calls:
In `upload.rs`, `upload_weight()` calls `stream.synchronize()` after each upload to prevent async allocator aliasing. This is a known synchronization point. Wrap it:

```rust
{
    let sync_span = tracing::debug_span!("cuda_sync", reason = "weight_upload");
    let _enter = sync_span.enter();
    stream.synchronize()?;
}
```

**NCCL all-reduce** (`sync.rs`):
Already structured with group_start/group_end. Add spans:

```rust
pub fn all_reduce_attention(nccl: &NcclCommunicator, stream: &CudaStream, tensor: &mut CudaSlice<bf16>) -> Result<()> {
    let span = tracing::debug_span!("nccl_all_reduce", kind = "attention");
    let _enter = span.enter();
    nccl.all_reduce_in_place(tensor, NcclReduceOp::Sum, stream)?;
    Ok(())
}
```

**Complexity**: S
**Timebox**: 30 min
**Acceptance**:
- `cargo check --release` compiles.
- Spans appear in OTLP traces for weight upload, logits download, and NCCL ops.
- No behavioral changes — purely additive instrumentation.

---

### Commit 8: Scheduler + orchestrator spans

**Files**: `crates/server/src/orchestrator.rs`, `crates/server/src/server.rs`, `crates/scheduler/src/lib.rs` (or `queue.rs`)

**Orchestrator `step()`**:

```rust
pub fn step(&mut self, ...) -> Result<()> {
    let span = tracing::info_span!("scheduler_step");
    let _enter = span.enter();
    // ... existing scheduling logic ...
}
```

**Session spans** (already has `tracing::info!` calls — add structured spans):

```rust
// When running prefill:
let span = tracing::info_span!("prefill_session", session_id = seq_id);
let _enter = span.enter();
// ... prefill_paged() call ...

// When running decode:
let span = tracing::info_span!("decode_session", session_id = seq_id);
let _enter = span.enter();
// ... decode_paged() call ...
```

**Eviction span** (already has `tracing::info!`):

```rust
let span = tracing::debug_span!("eviction", session_id = evicted_id);
let _enter = span.enter();
// ... eviction logic ...
```

**Background scheduler loop** (`server.rs`):

```rust
loop {
    let span = tracing::debug_span!("scheduler_loop_tick");
    let _enter = span.enter();
    match orchestrator.step() { ... }
}
```

**Complexity**: S
**Timebox**: 30 min
**Acceptance**:
- `cargo check --release` compiles.
- Traces show scheduler_step → prefill_session/decode_session hierarchy.
- No behavioral changes.

---

### Commit 9: Replace `eprintln!` progress markers with `tracing`

**Files**: `crates/backends/native/src/engine.rs`, `crates/backends/native/src/gdn.rs`

The engine uses `eprintln!` for progress logging (e.g., "Layer 8/48 (phase A)"). Replace these with structured `tracing::debug!` calls so they appear in OTLP traces:

```rust
// Before:
eprintln!("Layer {}/{} (phase A)", layer_idx + 1, config.num_hidden_layers);

// After:
tracing::debug!(layer_idx, total_layers = config.num_hidden_layers, phase = "A", "Layer dispatch");
```

This ensures progress markers are:
- Visible in console output (via the `fmt` layer with appropriate log level)
- Captured in OTLP traces as span events
- Filterable via `RUST_LOG` or `INFERS_LOG_LEVEL`

**Complexity**: XS
**Timebox**: 15 min
**Acceptance**:
- `cargo check --release` compiles.
- No `eprintln!` calls remain in engine.rs or gdn.rs (except in `probe.rs` which is env-gated debug tooling).
- Progress markers appear via `tracing::debug!`.

---

### Commit 10: Integration test + documentation

**Files**: New `crates/server/tests/tracing.rs`, `lat.md/lat.md`

**Integration test** (manual / smoke test style — requires OTLP backend):

```rust
// Test that OTLP-enabled server starts and emits at least one trace.
// This is a smoke test — run with a Jaeger backend at localhost:4317.
#[tokio::test]
#[ignore] // requires OTLP backend
async fn otlp_smoke_test() {
    // Spawn server with --otlp-enabled
    // Send a request
    // Verify server didn't crash
    // (Manual verification: check Jaeger UI for traces)
}
```

**lat.md documentation**: Add a new top-level section:

```markdown
# Tracing and Observability

Distributed tracing via OpenTelemetry OTLP for performance profiling and bottleneck identification.

## Architecture

Layered tracing subscriber with optional OTLP gRPC export. When OTLP is disabled (default), all spans compile to no-ops. CUDA event timing provides GPU execution time distinct from CPU wall-clock time.

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--otlp-enabled` | `false` | Enable OTLP trace export |
| `--otlp-endpoint` | `http://localhost:4317` | OTLP gRPC collector endpoint |
| `--otlp-service-name` | `infers` | Service name in OTLP traces |

## Span Taxonomy

[Table of all spans with parent relationships and key attributes]

## CUDA Event Timing

[Description of CudaEvent FFI, per-layer and top-level timing, TP=2 max-across-GPUs approach]
```

**Complexity**: S
**Timebox**: 30 min
**Acceptance**:
- `cargo check --release` and `lat check` pass.
- Tracing architecture documented in lat.md.

---

## Dependency Version Compatibility

OpenTelemetry Rust crates have a complex version matrix. The versions below are pinned to a known-working set as of 2026-06:

| Crate | Version | Notes |
|-------|---------|-------|
| `opentelemetry` | `0.27` | API crate (stable) |
| `opentelemetry_sdk` | `0.27` | SDK (batch processor, runtime) |
| `opentelemetry-otlp` | `0.27` | OTLP exporter with `grpc-tonic` feature |
| `tracing-opentelemetry` | `0.28` | Bridge: `tracing` span → OTel span |
| `tonic` | `0.12` | gRPC transport (pulled by opentelemetry-otlp) |
| `tracing` | `0.1.44` | Already in workspace |
| `tracing-subscriber` | `0.3.23` | Already in workspace (needs `registry` feature — verify) |

**Note on `tracing-subscriber` features**: The existing workspace dep has `features = ["env-filter", "json"]`. The layered subscriber approach requires the `registry` feature. Check if it's already included by default or needs adding:

```toml
tracing-subscriber = { version = "0.3.23", features = ["env-filter", "json", "registry"] }
```

The `registry` feature is likely already enabled transitively, but should be made explicit.

## Key Design Decisions

### KD1: OTLP transport = gRPC only

gRPC is the standard OTLP transport for production profiling (Jaeger, Tempo, Grafana). HTTP is simpler for local dev but adds a feature gate and dependency complexity for minimal benefit. If HTTP is needed later, add `opentelemetry-otlp/http-proto` feature behind a CLI flag.

### KD2: Tracing subscriber lives in infers-server

Only the server binary needs the subscriber init. Other crates (`infers-backend-native`, `infers-cuda`, `infers-scheduler`) emit `tracing` spans/events which are no-ops when no subscriber is installed. This keeps the dependency graph clean — only `infers-server` depends on `opentelemetry-*` and `tonic`.

### KD3: Layer-level span granularity (default)

Per-layer spans give enough detail to identify which layers are slow (GDN vs attention, early vs late layers) without excessive overhead. Operation-level spans (per-GEMM, per-kernel) can be added later behind `INFERS_TRACE_DETAIL=ops` but are not in this phase.

### KD4: CUDA events for GPU timing, not CPU timers

CPU timers (`std::time::Instant`) measure launch + wait time, not GPU execution time. CUDA events measure actual GPU execution, which is what matters for identifying compute-bound vs CPU-bound bottlenecks. The `CudaEvent` FFI wrapper is small and self-contained.

### KD5: Per-layer GPU timing is env-gated

`INFERS_TRACE_LAYER_TIMING=1` enables per-layer CUDA event timing. This adds ~5μs overhead per layer (event record + synchronize). At 48 layers that's ~240μs — acceptable for profiling but too expensive for always-on in production (0.5% of a 50ms target step). Top-level GPU timing (start/end events around the full layer loop) is always-on when OTLP is enabled — its overhead is ~10μs total.

### KD6: Structured attributes, not string interpolation

All span fields use structured key-value pairs: `layer_idx = 3`, `gpu_time_ms = 155.3`, `phase = "prefill"`. This ensures OTLP exports them as queryable attributes in Jaeger/Tempo/Grafana, not opaque log lines. Never use `format!` in span names or field values.

### KD7: No custom OTLP metrics

The existing Prometheus endpoint at `/metrics` handles metrics (counters, gauges, histograms). OTLP in this phase is traces only. If OTLP metrics export is needed later (e.g., to unify metrics + traces in Grafana Tempo), add `opentelemetry-otlp`'s `metrics` feature in a follow-up.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OpenTelemetry crate version conflicts | Medium | High | Pin exact versions in workspace. Test `cargo check --release` before implementation. |
| `cudarc::driver::CudaStream` doesn't expose raw stream pointer | Low | High | Check cudarc 0.19.7 API. If not exposed, use cudarc's `sys` module for raw `CUstream`. |
| OTLP export adds latency to hot path | Low | Medium | Batch processor (not simple) + separate export thread. Export is async, doesn't block inference. |
| CUDA event FFI bindings break on non-CUDA systems | Medium | Low | Guard with `cfg(feature = "cuda")` or just don't compile `infers-cuda` on non-CUDA — already the case. |
| `tracing-subscriber` `registry` feature not enabled | Low | Low | Add `registry` to feature list explicitly. |

## Getting Started (Developer)

1. Install Jaeger all-in-one: `docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/jaeger`
2. Build server: `cargo build --release -p infers-server`
3. Run with OTLP: `cargo run --release --bin infers -- --model /path/to/model --otlp-enabled --otlp-endpoint http://localhost:4317`
4. Send a request: `curl -X POST http://localhost:8000/v1/chat/completions -d '{"model":"infers","messages":[{"role":"user","content":"hello"}]}'`
5. Open Jaeger UI: http://localhost:16686 — search for service "infers"
6. Examine flame graph: prefill → layer_0 → attention → nccl_all_reduce → ...
7. Check `gpu_time_ms` vs span duration to identify CPU vs GPU bottlenecks

## Success Criteria

- [ ] `--otlp-enabled` flag exists and defaults to `false`
- [ ] With `--otlp-enabled`, traces appear in Jaeger/Tempo/Grafana via OTLP gRPC
- [ ] Without `--otlp-enabled`, no OTLP dependency is exercised (zero overhead)
- [ ] `prefill` and `decode` top-level spans appear in traces with `gpu_time_ms` attribute
- [ ] Per-layer spans show `layer_idx`, `layer_type`, and hierarchical structure
- [ ] `CudaEvent::elapsed_ms()` produces accurate GPU timing (±1ms vs Nsight)
- [ ] I/O spans (`weight_upload`, `logits_download`, `cuda_sync`) appear with timing
- [ ] NCCL all-reduce spans appear with `kind` attribute
- [ ] `eprintln!` progress markers in engine.rs replaced with `tracing::debug!`
- [ ] `cargo check --release` passes for all workspace crates
- [ ] `lat check` passes with updated documentation
- [ ] Tracing architecture documented in `lat.md/lat.md`
