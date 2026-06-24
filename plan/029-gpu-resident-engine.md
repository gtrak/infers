# Phase 29: GPU-Resident Engine — Zero-Alloc, Zero-Sync Steady State

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: Phase 30 (async pipeline needs pre-allocated buffers)
**Blocked by**: Phase 28 (fused GEMM eliminates dequant_buf allocations)
**Rationale**: The current decode path has ~100+ `alloc_zeros` calls per token, ~48 blocking `synchronize()` calls per token (RoPE tables), and ~96 D2H→CPU→H2D round-trips per token (a_log/dt_bias). These CPU-GPU sync stalls cause GPU0 to sit at 100% utilization but only 80W (memory-bound, stalled on syncs) and GPU1 at 80% (sync bubbles between layers). This phase eliminates all steady-state allocations and CPU-GPU synchronization.
---

## Goal

After this phase, the decode loop at steady state performs:
- **Zero** `alloc_zeros` or `alloc` calls
- **Zero** `clone_htod` or `clone_dtoh` copies
- **Zero** `stream.synchronize()` calls
- **Zero** `eprintln!` calls
- All metadata GPU-resident, converted at load time

## Current Sync Stalls (Must Eliminate All)

### 1. RoPE Table Uploads (3 syncs × 16 attention layers = 48 syncs/token)

**File**: `crates/backends/native/src/rope.rs:86-110`

```rust
// TODO: Cache RoPE tables in ForwardEngine at init time
let (cos_table, sin_table) = precompute_rope_tables(max_position, head_dim, rope_theta, partial_rotary_factor);
let positions_gpu = stream.clone_htod(&positions_i32)?;
stream.synchronize()?;  // ← BLOCKING SYNC #1
let cos_gpu = stream.clone_htod(&cos_table)?;
stream.synchronize()?;  // ← BLOCKING SYNC #2
let sin_gpu = stream.clone_htod(&sin_table)?;
stream.synchronize()?;  // ← BLOCKING SYNC #3
```

### 2. a_log/dt_bias D2H Round-Trip (2 syncs × 48 GDN layers = 96 syncs/token)

**File**: `crates/backends/native/src/gdn.rs:151-164`

```rust
fn bf16_to_f32_gpu(stream, src, _count) -> Result<CudaSlice<f32>> {
    let cpu_bf16 = stream.clone_dtoh(src)?;    // D2H copy
    let cpu_f32 = cpu_bf16.iter().map(|v| v.to_f32()).collect();  // CPU convert
    let dst = stream.clone_htod(&cpu_f32)?;     // H2D copy
    stream.synchronize()?;                       // ← BLOCKING SYNC
    Ok(dst)
}
```

### 3. eprintln in GEMM Dispatch (~400 blocking I/O calls/token)

**File**: `crates/backends/native/src/gemm_dispatch.rs:52,73`

```rust
eprintln!("[GEMM-DISPATCH] Bf16 weight '{}': len={}", weight_name, weight_gpu.len());
eprintln!("[GEMM-DISPATCH] Int4 weight '{}': n={}, k={}", weight_name, n, k);
```

### 4. Per-Token Allocations (~100+ alloc_zeros/token)

**Files**: `engine.rs` (decode_paged), `gdn.rs` (decode_forward), `attention.rs` (decode_forward_paged), `norm.rs`, `add.rs`

Every intermediate buffer is allocated fresh and dropped every layer, every token:
- `gate`, `up`, `silu_out`, `mlp_out` (engine.rs MLP path)
- `k_single`, `v_single`, `q_full`, `attn_combined` (attention.rs)
- `mixed_qkv`, `conv_out`, `gdn_output`, `z_gate_raw`, `norm_out` (gdn.rs)
- `output` in `norm.rs`, `add.rs` — allocated and dropped every call

## Target Architecture

### Pre-Allocated Workspace Buffer

At `ForwardEngine::new_from_mmap()` time, allocate a `Workspace` struct:

```rust
pub struct DecodeWorkspace {
    // Residual stream (double-buffered for overlap)
    pub hidden: CudaSlice<bf16>,              // [max_batch, hidden_size]
    pub hidden_next: CudaSlice<bf16>,         // [max_batch, hidden_size]

    // Norm outputs
    pub norm1_out: CudaSlice<bf16>,           // [max_batch, hidden_size]
    pub norm2_out: CudaSlice<bf16>,           // [max_batch, hidden_size]

    // Attention/GDN intermediates
    pub attn_out: CudaSlice<bf16>,            // [max_batch, hidden_size]
    pub qkv_proj: CudaSlice<bf16>,            // [max_batch, conv_dim_shard]
    pub conv_out: CudaSlice<bf16>,            // [max_batch, conv_dim_shard]
    pub gdn_output: CudaSlice<bf16>,          // [max_batch, v_dim_shard]
    pub a_proj: CudaSlice<bf16>,              // [max_batch, num_v_heads_shard]
    pub b_proj: CudaSlice<bf16>,              // [max_batch, num_v_heads_shard]

    // MLP intermediates
    pub mlp_gate: CudaSlice<bf16>,            // [max_batch, intermediate_shard]
    pub mlp_up: CudaSlice<bf16>,              // [max_batch, intermediate_shard]
    pub mlp_silu: CudaSlice<bf16>,            // [max_batch, intermediate_shard]
    pub mlp_out: CudaSlice<bf16>,             // [max_batch, hidden_size]

    // Self-attention intermediates
    pub q_proj: CudaSlice<bf16>,              // [max_batch, q_dim_shard]
    pub k_proj: CudaSlice<bf16>,              // [max_batch, kv_dim_shard]
    pub v_proj: CudaSlice<bf16>,              // [max_batch, kv_dim_shard]
    pub attn_combined: CudaSlice<bf16>,       // [max_batch, per_gpu_head_dim]

    // Output
    pub logits: CudaSlice<bf16>,              // [max_batch, vocab_size]

    // Per-sequence inputs (updated in-place, no realloc)
    pub token_ids: CudaSlice<i32>,            // [max_batch]
    pub positions: CudaSlice<i32>,            // [max_batch]
}
```

All buffers sized for `max_batch` (e.g., 4 for continuous batching). Allocated once, reused every decode step. The decode loop writes into these buffers; no allocation happens.

### GPU-Resident Metadata

Converted at load time, stored in `ForwardEngine`:

```rust
pub struct GpuMetadata {
    // RoPE tables (precomputed, uploaded once)
    pub rope_cos: CudaSlice<f32>,             // [max_position+1, half_dim]
    pub rope_sin: CudaSlice<f32>,             // [max_position+1, half_dim]
    pub rope_positions: CudaSlice<i32>,       // [max_batch] — updated in-place per step

    // GDN params (converted bf16→f32 at load time)
    // Stored as a HashMap<layer_idx, CudaSlice<f32>>
    pub a_log_f32: HashMap<usize, CudaSlice<f32>>,      // [num_v_heads_shard]
    pub dt_bias_f32: HashMap<usize, CudaSlice<f32>>,    // [num_v_heads_shard]

    // Block tables (persistent, updated in-place)
    pub block_tables: CudaSlice<i32>,         // [max_batch, max_pages_per_seq]
}
```

### What Gets Eliminated

| Current Pattern | Count/Token | Replacement |
|----------------|-------------|-------------|
| `alloc_zeros` in gemm_dispatch.rs | ~896 (INT4 + NVFP4 GEMMs) | Eliminated by fused kernel (Phase 28) |
| `alloc_zeros` in engine.rs MLP path | ~256 (4 per layer × 64) | Pre-allocated workspace buffers |
| `alloc_zeros` in norm.rs / add.rs | ~256 (2 per layer × 64) | Pre-allocated workspace buffers |
| `alloc_zeros` in gdn.rs | ~480 (10 per GDN layer × 48) | Pre-allocated workspace buffers |
| `alloc_zeros` in attention.rs | ~256 (4 per attn layer × 16) | Pre-allocated workspace buffers |
| RoPE table H2D + sync | ~48 syncs | Pre-allocated `rope_cos`/`rope_sin` |
| a_log/dt_bias D2H→H2D + sync | ~96 syncs | Pre-converted `a_log_f32`/`dt_bias_f32` |
| `eprintln!` in GEMM dispatch | ~400 calls | Gate behind `INFERS_DEBUG` env var |
| Block table/position H2D per step | ~4 copies | Mapped pinned memory (zero-copy) |
| `gpu_end_events.synchronize()` | 2 per step | Remove or make non-blocking |

## Implementation Plan

### Step 1: Gate eprintln in gemm_dispatch.rs

Quick win. Add `use std::sync::OnceLock;` and only print on first call per weight name, or gate behind env var.

**File**: `crates/backends/native/src/gemm_dispatch.rs`

### Step 2: Pre-convert a_log and dt_bias to f32

Add f32 variants to `GpuWeightCache`:

```rust
pub enum CachedWeight {
    Bf16(CudaSlice<bf16>),
    Int4(Int4GpuBuffers),
    Nvfp4(Nvfp4GpuBuffers),
    F32(CudaSlice<f32>),  // NEW: for a_log, dt_bias
}
```

At load time, when uploading GDN params:
```rust
if let Some(a_log) = &gdn.a_log {
    // Convert bf16→f32 on GPU (simple kernel or CPU convert at upload time)
    let f32_data: Vec<f32> = a_log.data.iter().map(|b| b.to_f32()).collect();
    let f32_gpu = stream.clone_htod(&f32_data)?;
    cache.insert(name, CachedWeight::F32(f32_gpu));
}
```

Then in `gdn.rs decode_forward`, replace `bf16_to_f32_gpu` calls with:
```rust
let a_log = cache.get_f32("layer.N.linear_attn.A_log");  // O(1) lookup, no conversion
```

**Files**: `crates/backends/native/src/gpu_cache.rs`, `crates/backends/native/src/gdn.rs`

### Step 3: Cache RoPE tables at init

Add RoPE tables to `ForwardEngine`:

```rust
pub struct ForwardEngine {
    // ... existing fields ...
    rope_cos: CudaSlice<f32>,
    rope_sin: CudaSlice<f32>,
}
```

At `new_from_mmap()`, precompute and upload:
```rust
let (cos, sin) = precompute_rope_tables(
    config.max_position_embeddings,
    head_dim, config.rope_theta, partial_rotary_factor,
);
self.rope_cos = stream.clone_htod(&cos)?;
self.rope_sin = stream.clone_htod(&sin)?;
```

Modify `apply_rope()` to accept pre-computed GPU tables instead of recomputing.

**Files**: `crates/backends/native/src/engine.rs`, `crates/backends/native/src/rope.rs`

### Step 4: Pre-allocate workspace buffers

Add `DecodeWorkspace` to `ForwardEngine`:

```rust
pub struct ForwardEngine {
    // ... existing fields ...
    workspace: DecodeWorkspace,
}
```

In the layer loop, pass workspace buffers by mutable reference instead of allocating:
```rust
// Before:
let gate = stream.alloc_zeros::<bf16>(m * n)?;
gemm_projection_cached(..., &mut gate)?;
let silu_out = stream.alloc_zeros::<bf16>(m * n)?;

// After:
gemm_projection_cached(..., &mut self.workspace.mlp_gate)?;
silu_glu(&self.workspace.mlp_gate, &self.workspace.mlp_up, &mut self.workspace.mlp_silu)?;
```

**Files**: `crates/backends/native/src/engine.rs`, new file `crates/backends/native/src/workspace.rs`

### Step 5: Update kernel functions to take output buffers

`norm.rs`, `add.rs`, `silu.rs` — change from allocating-and-returning to taking `&mut CudaSlice<bf16>` output:

```rust
// Before:
pub fn rms_norm(stream, oxide, input, weight, eps) -> Result<CudaSlice<bf16>> {
    let mut output = stream.alloc_zeros::<bf16>(...)?;
    oxide.launch_rmsnorm_bf16(stream, &mut output, input, weight, ...)?;
    Ok(output)
}

// After:
pub fn rms_norm_into(stream, oxide, output: &mut CudaSlice<bf16>, input, weight, eps) -> Result<()> {
    oxide.launch_rmsnorm_bf16(stream, output, input, weight, ...)?;
    Ok(())
}
```

**Files**: `crates/backends/native/src/norm.rs`, `crates/backends/native/src/add.rs`, `crates/backends/native/src/silu.rs`

### Step 6: Mapped pinned memory for inputs

Use CUDA mapped memory for `token_ids` and `positions`:
```rust
let token_ids_pinned = pinned_host.alloc::<i32>(max_batch)?;
// CPU writes directly, GPU reads via PCIe — no H2D copy, no sync
```

**File**: `crates/backends/native/src/engine.rs`

## Verification

```bash
# Verify correctness
python3 scripts/dump_oracle_hidden.py --model ... --output-dir /tmp/oracle_int4
INFERS_DUMP_DIR=/tmp/infer_dumps_int4 ./target/release/infer ...
python3 scripts/compare_hidden_states.py --oracle-dir /tmp/oracle_int4 --infer-dir /tmp/infer_dumps_int4
# Target: cosine ≥ 0.99 (no regression from pre-allocation)

# Verify no syncs in steady state
# Add timing around decode_paged and check for syncs
grep -r "synchronize" crates/backends/native/src/ | grep -v test | grep -v init
# Target: zero synchronize calls in decode/forward hot path

# Time decode
time ./target/release/infer --model ... --max-tokens 20 --no-chat
# Target: measurably faster than before (with Phase 28 fused GEMM)
```

## Files Modified

| File | Change |
|------|--------|
| `crates/backends/native/src/gpu_cache.rs` | Add `F32` variant to `CachedWeight`; add `get_f32()` |
| `crates/backends/native/src/gdn.rs` | Remove `bf16_to_f32_gpu`; use cached f32 params |
| `crates/backends/native/src/rope.rs` | Accept pre-computed GPU tables; remove H2D + sync |
| `crates/backends/native/src/engine.rs` | Add `DecodeWorkspace`; pre-allocate at init; use workspace in decode loop |
| `crates/backends/native/src/gemm_dispatch.rs` | Gate eprintln behind env var |
| `crates/backends/native/src/norm.rs` | Add `rms_norm_into` variant |
| `crates/backends/native/src/add.rs` | Add `add_into` variant |
| `crates/backends/native/src/silu.rs` | Add `silu_glu_into` variant |
| new: `crates/backends/native/src/workspace.rs` | `DecodeWorkspace` struct definition |
