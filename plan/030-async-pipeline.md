# Phase 30: cuda-oxide Async Pipeline with Multi-Stream Scheduling

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: Phase 31 (continuous batching needs async for multi-sequence overlap), Phase 32 (CUDA graphs need stable pipeline)
**Blocked by**: Phase 28 (fused GEMM), Phase 29 (pre-allocated workspace — DeviceOperation closures need stable buffer addresses)
**Rationale**: The current engine uses a single CUDA stream per GPU with synchronous NCCL all-reduce after every layer (128 blocking collectives per token). GPU1 sits at 80% utilization because of sync bubbles between layers. The cuda-oxide `cuda-async` crate provides `DeviceOperation` — a lazy GPU computation graph with stream-pool scheduling. Building each layer's forward pass as a `DeviceOperation` allows the scheduler to overlap NCCL all-reduce (stream 1) with the next layer's norm+GEMM (stream 0), filling idle bubbles.
---

## Goal

Transform the synchronous per-layer decode loop into a lazy `DeviceOperation` graph that the cuda-oxide scheduler executes across multiple streams, overlapping:
- Layer N's NCCL all-reduce with Layer N+1's norm1 + first GEMM
- Attention compute with MLP norm2 preparation
- Independent GEMMs (gate_proj, up_proj) in parallel via `zip!`

## Current State

```
Decode loop (synchronous, single stream per GPU):
  for layer in 0..64:
    norm1 → GEMMs (sequential) → attention/GDN kernel → NCCL all_reduce (BLOCKING)
    → residual add → norm2 → GEMMs (sequential) → SiLU → GEMM → NCCL all_reduce (BLOCKING)
    → residual add
```

Problems:
1. Single stream: all kernels serialized, no overlap possible
2. NCCL all-reduce blocks: GPU idle waiting for the other GPU
3. GEMMs within a layer are sequential even when independent (gate_proj + up_proj are independent)
4. No async NCCL: `comm.all_reduce_in_place` blocks until complete

## Target State

```
Layer pipeline (lazy DeviceOperation, multi-stream):
  Stream 0:  norm1 → zip!(gemm_q, gemm_k, gemm_v) → gdn/attn → gemm_o
  Stream 1:                                          all_reduce_async ──┐
  Stream 0:                              ←── all_reduce complete ──────┘
             add_residual → norm2 → zip!(gemm_gate, gemm_up) → silu → gemm_down
  Stream 1:                                                              all_reduce_async
  Stream 0: next_layer(norm1) ← overlaps with previous all_reduce ──────┘
```

## Architecture

### cuda-async API

From the [cuda-oxide async docs](https://nvlabs.github.io/cuda-oxide/projects/async-mlp-pipeline.html):

- **`DeviceOperation<T>`**: A lazy description of GPU work. Built with combinators, executed when polled.
- **`with_context(|ctx| { ... })`**: Create a DeviceOperation that receives the scheduler's stream at execution time.
- **`and_then(|prev_output| { ... })`**: Chain operations, threading device handles through.
- **`zip!(op1, op2, ...)`**: Execute independent operations in parallel on separate streams.
- **`value(x)`**: Wrap a value as a completed DeviceOperation (the "baton" that passes data between stages).
- **`.arc()`**: Wrap result in `Arc` for cheap cloning across multiple consumers.
- **`.into_future()`**: Convert to a `DeviceFuture` for polling with tokio.
- **Stream pool**: Round-robin stream assignment. Configurable via `init_device_contexts`.

### Per-Layer Pipeline

Each layer's forward pass becomes a `DeviceOperation`:

```rust
fn build_layer_decode_pipeline(
    engine: &Engine,
    layer_idx: usize,
    workspace: &Workspace,
) -> impl DeviceOperation<Output = ()> {
    let weights = engine.layer_weights(layer_idx);
    let config = engine.config();

    // Phase 1: Norm1 + Attention/GDN
    load_hidden(workspace, layer_idx)
        .and_then(move |hidden| {
            // Norm1 (stream assigned by scheduler)
            rms_norm_into(hidden, &workspace.norm1_out, &weights.norm1)
                .and_then(move |()| {
                    // Branch: GDN or Full Attention
                    match config.get_layer_type(layer_idx) {
                        GatedDeltaNet => build_gdn_decode_pipeline(workspace, weights),
                        FullAttention => build_attn_decode_pipeline(workspace, weights),
                    }
                })
        })
        // Phase 2: NCCL all-reduce (async, separate stream)
        .and_then(move |attn_out| {
            all_reduce_async(attn_out, engine.nccl)
        })
        // Phase 3: Residual + Norm2 + MLP
        .and_then(move |attn_reduced| {
            add_residual_into(&workspace.hidden, attn_reduced, &workspace.hidden_next)
                .and_then(move |()| {
                    rms_norm_into(&workspace.hidden_next, &workspace.norm2_out, &weights.norm2)
                })
                .and_then(move |()| {
                    // gate_proj and up_proj are independent — run in parallel
                    zip!(
                        fused_quant_gemm(&workspace.norm2_out, &weights.gate, &workspace.mlp_gate),
                        fused_quant_gemm(&workspace.norm2_out, &weights.up, &workspace.mlp_up),
                    )
                    .and_then(move |((), ())| {
                        silu_glu_into(&workspace.mlp_gate, &workspace.mlp_up, &workspace.mlp_silu)
                            .and_then(move |()| {
                                fused_quant_gemm(&workspace.mlp_silu, &weights.down, &workspace.mlp_out)
                            })
                    })
                })
        })
        // Phase 4: NCCL all-reduce MLP (async, overlaps with next layer's Phase 1)
        .and_then(move |mlp_out| {
            all_reduce_async(mlp_out, engine.nccl)
        })
        .and_then(move |mlp_reduced| {
            add_residual_into(&workspace.hidden_next, mlp_reduced, &workspace.hidden)
        })
}
```

### Full Model Pipeline

The full 64-layer decode step is a chain of layer pipelines:

```rust
fn build_decode_pipeline(engine: &Engine, workspace: &Workspace) -> impl DeviceOperation<Output = ()> {
    let mut pipeline = build_layer_decode_pipeline(engine, 0, workspace);
    for layer_idx in 1..engine.config().num_hidden_layers {
        pipeline = pipeline.and_then(move |()| {
            build_layer_decode_pipeline(engine, layer_idx, workspace)
        });
    }
    // Final: norm + lm_head + sample
    pipeline
        .and_then(|()| rms_norm_into(&workspace.hidden, &workspace.final_norm, &weights.norm))
        .and_then(|()| fused_quant_gemm(&workspace.final_norm, &weights.lm_head, &workspace.logits))
        .and_then(|()| argmax(&workspace.logits, &workspace.sampled_token))
}
```

The scheduler's stream pool handles overlap:
- The `and_then` chain ensures ordering within a single sequence
- `zip!` enables parallelism for independent operations
- NCCL `all_reduce_async` runs on a separate stream, overlapping with the next `and_then`

### NCCL Async Integration

The current NCCL wrapper is synchronous:
```rust
// crates/cuda/src/nccl.rs:81
comm.all_reduce_in_place(&slice, ReductionOp::Sum, stream)?;
stream.synchronize()?;  // BLOCKING
```

Wrap as async:
```rust
fn all_reduce_async(
    tensor: CudaSlice<bf16>,
    nccl: &NcclCommunicator,
) -> impl DeviceOperation<Output = CudaSlice<bf16>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let mut buf = tensor;  // take ownership
        nccl.all_reduce_in_place(&mut buf, ReductionOp::Sum, stream)?;
        value(buf)  // return ownership for next stage
    })
}
```

### GDN Decode Sub-Pipeline

The GDN decode path (gdn.rs:468-708) becomes a sub-pipeline:

```rust
fn build_gdn_decode_pipeline(ws: &Workspace, w: &LayerWeights) -> impl DeviceOperation<Output = CudaSlice<bf16>> {
    rms_norm_into(ws.hidden, &ws.norm1_out, &w.norm1)
        .and_then(|()| {
            // in_proj_qkv (fused GEMM)
            fused_quant_gemm(&ws.norm1_out, &w.in_proj_qkv, &ws.qkv_proj)
        })
        .and_then(|()| {
            // conv1d + split Q/K/V (single kernel)
            conv1d_silu_split(&ws.qkv_proj, &w.conv1d, &ws.conv_out,
                &ws.query, &ws.key, &ws.value)
        })
        .and_then(|()| {
            // a_proj, b_proj (independent, parallel)
            zip!(
                fused_quant_gemm(&ws.norm1_out, &w.in_proj_a, &ws.a_proj),
                fused_quant_gemm(&ws.norm1_out, &w.in_proj_b, &ws.b_proj),
            )
        })
        .and_then(|((), ())| {
            // GDN recurrent step (single kernel, uses GPU-resident a_log, dt_bias)
            gdn_recurrent_step(
                &ws.query, &ws.key, &ws.value,
                &ws.a_proj, &ws.b_proj,
                &w.a_log_f32, &w.dt_bias_f32,  // GPU-resident f32
                &mut ws.gdn_state,
                &ws.gdn_output,
            )
        })
        .and_then(|()| {
            // z_gate (if present) + RMSNormGated + out_proj
            match &w.in_proj_z {
                Some(z_weight) => {
                    fused_quant_gemm(&ws.norm1_out, z_weight, &ws.z_gate_raw)
                        .and_then(|()| {
                            rms_norm_gated(&ws.gdn_output, &ws.z_gate_raw, &w.norm,
                                &ws.norm_out)
                        })
                        .and_then(|()| {
                            fused_quant_gemm(&ws.norm_out, &w.out_proj, &ws.attn_out)
                        })
                }
                None => {
                    rms_norm_gated(&ws.gdn_output, &ws.z_gate_raw, &w.norm,
                        &ws.norm_out)
                        .and_then(|()| {
                            fused_quant_gemm(&ws.norm_out, &w.out_proj, &ws.attn_out)
                        })
                }
            }
        })
}
```

### Attention Decode Sub-Pipeline

```rust
fn build_attn_decode_pipeline(ws: &Workspace, w: &LayerWeights) -> impl DeviceOperation<Output = CudaSlice<bf16>> {
    rms_norm_into(ws.hidden, &ws.norm1_out, &w.norm1)
        .and_then(|()| {
            // k_proj, v_proj, q_proj (independent — parallel)
            zip!(
                fused_quant_gemm(&ws.norm1_out, &w.k_proj, &ws.k_proj),
                fused_quant_gemm(&ws.norm1_out, &w.v_proj, &ws.v_proj),
                fused_quant_gemm(&ws.norm1_out, &w.q_proj, &ws.q_proj),
            )
        })
        .and_then(|((), (), ())| {
            // k_norm + RoPE (single kernel, uses GPU-resident rope tables)
            rope_k_and_q(&mut ws.k_proj, &mut ws.q_proj,
                &ws.rope_cos, &ws.rope_sin, &ws.positions)
        })
        .and_then(|()| {
            // Write K, V to paged cache
            paged_kv_write(&ws.k_proj, &ws.v_proj, &ws.block_table, ws.position)
        })
        .and_then(|()| {
            // Paged attention decode (single kernel)
            paged_attention_decode(&ws.q_proj, &ws.block_table, &ws.kv_cache,
                &ws.attn_combined)
        })
        .and_then(|()| {
            // o_proj
            fused_quant_gemm(&ws.attn_combined, &w.o_proj, &ws.attn_out)
        })
}
```

## Implementation Plan

### Step 1: Add cuda-async dependency

```toml
# crates/cuda/Cargo.toml
[dependencies]
cuda-async = { path = "../../cuda-oxide/crates/cuda-async" }
```

Verify the crate exists in the cuda-oxide checkout.

### Step 2: Create async NCCL wrapper

Wrap `nccl.all_reduce_in_place` as a `DeviceOperation`:

```rust
// crates/cuda/src/nccl_async.rs
pub fn all_reduce_async(
    tensor: CudaSlice<bf16>,
    nccl: &Arc<NcclCommunicator>,
    op: ReductionOp,
) -> impl DeviceOperation<Output = CudaSlice<bf16>> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        let mut buf = tensor;
        nccl.comm.all_reduce_in_place(&mut buf, op, stream)?;
        value(buf)
    })
}
```

### Step 3: Create fused_quant_gemm as DeviceOperation

```rust
pub fn fused_quant_gemm(
    input: Arc<CudaSlice<bf16>>,
    weight: &QuantWeight,
    output: &mut CudaSlice<bf16>,
) -> impl DeviceOperation<Output = ()> {
    device_operation::with_context(move |ctx| {
        let stream = ctx.get_cuda_stream();
        match weight {
            QuantWeight::Int4(bufs) => {
                engine.oxide.launch_fused_int4_gemm(stream, output, &input,
                    &bufs.qweight, &bufs.scales, &bufs.qzeros, m, n, k, group_size)?;
            }
            QuantWeight::Nvfp4(bufs) => {
                engine.oxide.launch_fused_nvfp4_gemm(stream, output, &input,
                    &bufs.weight_packed, &bufs.weight_scale, bufs.weight_global_scale,
                    m, n, k)?;
            }
            QuantWeight::Bf16(buf) => {
                engine.gemm.matmul_bf16(...)?;
            }
        }
        value(())
    })
}
```

### Step 4: Build per-layer pipeline functions

Implement `build_gdn_decode_pipeline`, `build_attn_decode_pipeline`, and the full `build_decode_pipeline`.

### Step 5: Replace synchronous decode loop with async execution

In `engine.rs decode_paged`:
```rust
// Before: synchronous for loop over 64 layers
for layer_idx in 0..num_layers { ... }

// After: build and execute async pipeline
let pipeline = build_decode_pipeline(self, &self.workspace);
pipeline.into_future().await?;
```

### Step 6: Configure stream pool

```rust
// At engine init
init_device_contexts(0, 2)?;  // 2 streams per GPU
```

Profile with `nsys` to verify stream overlap.

## Verification

```bash
# Correctness
python3 scripts/compare_hidden_states.py --oracle-dir /tmp/oracle_int4 --infer-dir /tmp/infer_dumps_int4
# Target: cosine ≥ 0.99 (async ordering must produce same results)

# Performance
nsys profile ./target/release/infer --model ... --max-tokens 20 --no-chat
# Check nsys timeline: should see parallel stream rows
# GPU1 utilization should increase from 80% → 95%+

# Timing
time ./target/release/infer ...
# Target: significant speedup from NCCL overlap
```

## Files Modified

| File | Change |
|------|--------|
| new: `crates/cuda/src/nccl_async.rs` | Async NCCL wrapper using DeviceOperation |
| new: `crates/backends/native/src/pipeline.rs` | Build per-layer and full decode pipelines |
| `crates/backends/native/src/engine.rs` | Replace synchronous loop with async execution |
| `crates/cuda/Cargo.toml` | Add `cuda-async` dependency |
| `crates/backends/native/Cargo.toml` | Add `cuda-async` dependency |

## Considerations

- **Double-buffering**: The `hidden` and `hidden_next` workspace buffers enable overlap. While NCCL all-reduce writes to `hidden_next`, the next layer can read from `hidden` (previous layer's output).
- **Stream ordering**: `DeviceOperation`'s `and_then` guarantees ordering. `zip!` enables parallelism. The scheduler's stream pool assigns streams round-robin.
- **NCCL on separate stream**: The async NCCL wrapper runs on whatever stream the scheduler assigns. To force it on a specific stream, use `device_operation::with_stream(stream_id, || { ... })`.
- **Tokio runtime**: The engine will need a tokio runtime to poll the DeviceFuture. This is already available (the server uses axum/tokio).
