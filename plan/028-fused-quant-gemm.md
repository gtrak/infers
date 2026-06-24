# Phase 28: Fused Quantized GEMM Kernel with Shared Memory Tiling

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-24
**Blocks**: Phases 29-32 (all depend on efficient GEMM)
**Blocked by**: None (softmax fix is done, oracle test framework exists)
**Rationale**: The current decode path dequantizes INT4 weights to a full BF16 buffer in global memory, then runs a separate GEMM — 4 kernel launches + 52MB allocation per GEMM, ~896 GEMMs per token. Both vLLM and llama.cpp keep weights compressed in VRAM and dequant on-the-fly inside the GEMM kernel using shared memory tiling. The existing `int4_gemm_auto_round` kernel already fuses dequant+GEMM but is naive (1 thread per output element, no shared memory, no K-tiling). This phase adds proper shared memory tiling to the fused kernel and extends it to NVFP4 and GGUF formats.
---

## Goal

Replace the dequant→sanitize→bf16_gemm_tiled pipeline with a single fused kernel that:
1. Reads compressed weights directly from VRAM (INT4, NVFP4, or GGUF)
2. Stages input activations in shared memory for reuse across output columns
3. Dequantizes weights in registers (never writes to global memory)
4. Accumulates in f32, outputs BF16
5. Supports all quant formats via the existing `Dequantize` trait

## Current State

```
gemm_projection_cached()
  ├─ BF16:  GemmEngine::matmul_bf16()           ← cuBLASLt (good)
  ├─ INT4:  alloc_zeros → int4_dequant_to_bf16 → sanitize → bf16_gemm_tiled
  │         (4 operations, 52MB alloc, ~46GB mem traffic/token)
  └─ NVFP4: alloc_zeros → nvfp4_dequant_to_bf16 → sanitize → bf16_gemm_tiled
            (4 operations, 52MB alloc, ~46GB mem traffic/token)
```

The existing fused kernel `int4_gemm_auto_round` (lib.rs:997) IS never called. It has:
- 1 thread per output element
- Grid: `(ceil(n/64), ceil(m/4))`, Block: `(64, 4)`
- No shared memory (`shared_mem_bytes: 0`)
- Reads input activations from global memory per-thread, per-K-step (no reuse)
- Works correctly but has terrible arithmetic intensity (~2 FLOP/byte)

## Target State

```
gemm_projection_cached()
  ├─ BF16:  GemmEngine::matmul_bf16()           ← cuBLASLt (unchanged)
  ├─ INT4:  launch_fused_quant_gemm()          ← NEW: tiled fused kernel
  └─ NVFP4: launch_fused_quant_gemm()            ← NEW: tiled fused kernel
```

## Architecture

### Tiling Strategy

For M=1 (single-token decode), the bottleneck is weight reads from VRAM. The key optimization is **input staging in shared memory** so each weight element is loaded once and multiplied against the input vector:

```
Block (256 threads, 16×16 thread tile) computes a 16×16 output tile
├── Shared memory: input_vec[K_TILE] in BF16 (e.g., K_TILE=128 → 256 bytes)
│
├── For each K_TILE chunk of the K dimension:
│   ├── Cooperatively load input[row, k_tile_start..k_tile_end] into smem
│   │   (256 threads load 128 BF16 values = 2 values per thread, coalesced)
│   ├── sync_threads()
│   │
│   └── Each thread (tx, ty) computes output[block_m+ty, block_n+tx]:
│       ├── Load 1 u32 from qweight (gives 8 INT4 values)
│       │   Weight layout [K/8, N] → qweight[(k/8), col] — coalesced by N
│       ├── Load scale + zero for this group (1 per group_size=128)
│       ├── Dequant 8 INT4 → 8 f32 values in registers
│       ├── Load 8 BF16 input values from shared memory → f32
│       ├── 8× FMA: acc += w_dequant[i] * input_from_smem[i]
│       └── (advance to next u32 in K)
│
├── sync_threads() (ensure smem is done before next tile load)
└── Write f32 accumulator → BF16 output[row, col]
```

**Why this helps:**
- Without tiling: each of N threads reads its own copy of the input (N × K reads)
- With tiling: input is loaded once per block (K reads), shared by N threads
- For a [1, 5120] × [5120, 5120] GEMM: input reads drop from 5120×5120=26M to 5120 (per block), then 5120 blocks = 26M total reads, but now **coalesced and cached in smem** rather than random global accesses
- Weight reads stay the same (must read all weights), but they're coalesced (adjacent columns = adjacent u32s)

### For M > 1 (Batched Decode, Future Phase 31)

When M > 1, the tile becomes `M_TILE × N_TILE` (e.g., 4×16):
- Load `M_TILE` input rows into shared memory (4 rows × K_TILE = 4×128 = 512 BF16 values = 1KB)
- Each thread computes one element of the M_TILE × N_TILE output sub-tile
- Input reuse increases: same input loaded once, used by N_TILE threads
- Arithmetic intensity improves linearly with M

### Quant Format Abstraction

The existing `Dequantize` trait at lib.rs:861 is the right abstraction:

```rust
pub trait Dequantize {
    fn dequant(w_int4: i8, raw_zero: i8, scale: f32) -> f32;
}
```

The tiled kernel structure is identical for all formats — only the inner dequant changes:

| Format | Weight Storage | Scale Storage | Zero Storage | Dequant Formula |
|--------|---------------|---------------|--------------|-----------------|
| AutoRound INT4 | u32 packed [K/8, N] | fp16 [K/gs, N] | u32 packed [K/gs, N/8] | `(w - (zero+1)) * f16_to_f32(scale)` |
| GGUF Q4_0 | u32 packed [K/8, N] | fp16 [N] | u32 packed [N/8] | `(w - zero) * scale` |
| GGUF Q4_K | u32 packed [K/8, N] | super-block scales | sub-block scales | `(w - block_zero) * block_scale * sub_scale` |
| NVFP4 | u8 packed [N, K/2] | fp8_e4m3 [N, K/gs] | n/a | `fp4_to_f32(w) * fp8_to_f32(scale) / global_scale` |
| NVFP4 (signed) | u8 packed [N, K/2] | fp8_e4m3 [N, K/gs] | n/a | `fp4_signed_to_f32(w) * fp8_to_f32(scale) / global_scale` |

The `Dequantize` trait will be extended to take more context (packed weight, scale arrays, group indices) since NVFP4 has a different unpacking path from INT4. The trait method signature will change to:

```rust
pub trait QuantFormat {
    /// Unpack and dequantize one group of weights.
    /// `packed` is the raw packed bytes/u32s for this group.
    /// `scales` points to the scale data for this group.
    /// `out` receives the dequantized f32 values.
    fn dequant_group(
        packed: &[u32],      // 8 INT4 values packed in u32 (or equivalent)
        scales: &[u8],       // FP16 or FP8 scale bytes
        global_scale: f32,  // NVFP4 only, 1.0 for INT4
        group_size: usize,
        out: &mut [f32],    // output: group_size f32 values
    );
}
```

### Weight Layout Analysis

INT4 AutoRound weights have layout `[K/8, N]` (column-major transposed). This means:
- Fixing K and scanning N: sequential addresses = coalesced reads ✓
- Fixing N and scanning K: stride-N addresses = uncoalesced ✗

The kernel must iterate K in the inner loop (each thread fixes its N column, iterates over K groups). This is already the structure of `int4_gemm_inner`.

NVFP4 weights have layout `[N, K/2]` (row-major). Each thread fixes its N row, iterates over K — this is naturally coalesced.

### Launch Configuration

```
Grid:  (ceil(N / N_TILE), ceil(M / M_TILE), 1)
Block: (N_TILE, M_TILE, 1)  e.g., (16, 1, 1) for M=1 decode, (16, 4, 1) for M=4
Smem:  M_TILE × K_TILE × sizeof(bf16)  e.g., 1×128×2 = 256 bytes
```

For M=1 decode: Block=(64, 1, 1), each block computes 64 output elements for one row.

### Shared Memory Budget

- RTX 5060 Ti (Blackwell SM 12.0): 228KB per SM
- RTX 4090 (Ada SM 8.9): 164KB per SM  
- RTX 3000 (Ampere SM 8.6): 164KB per SM

Our shared memory usage is tiny:
- Input staging: K_TILE × sizeof(bf16) = 128×2 = 256 bytes (M=1)
- Weight staging: optional, K_TILE/8 × sizeof(u32) = 16×4 = 64 bytes
- Total: <1KB per block — trivially fits

## Implementation Plan

### Step 1: Write the tiled INT4 AutoRound kernel

Replace the naive `int4_gemm_auto_round` with a shared-memory-tiled version. Keep the same kernel name (or add a new one like `int4_gemm_auto_round_tiled` for testing).

**File**: `crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`

Key changes to `int4_gemm_inner`:
1. Add shared memory declaration: `let smem = DynamicSharedArray::<u16>::get();`
2. Add K-tiling loop: iterate K in chunks of K_TILE=128
3. Cooperative load: all threads in block load input row chunk into smem
4. sync_threads() barrier
5. Each thread reads its qweight column from global memory, dequants in registers, reads input from smem
6. sync_threads() before next K tile

### Step 2: Add NVFP4 fused GEMM kernel

Create `nvfp4_gemm_fused` kernel using the same tiling structure but with FP4 dequant in the inner loop. The dequant takes packed u8 bytes (2 FP4 values per byte), unpacks to f32, multiplies by FP8 scale and f32 global scale.

**File**: `crates/cuda-oxide-kernels/kernel-lib/src/lib.rs`

### Step 3: Add launch wrappers

**File**: `crates/cuda/src/oxide_bridge.rs`

- `launch_fused_int4_gemm()` — wraps the tiled INT4 kernel
- `launch_fused_nvfp4_gemm()` — wraps the tiled NVFP4 kernel

Both take the same args as the current dequant+gemm path plus shared memory size.

### Step 4: Wire into gemm_dispatch.rs

Replace the INT4/NVFP4 paths in `gemm_projection_cached()`:

```rust
Some(crate::gpu_cache::CachedWeight::Int4(int4_bufs)) => {
    oxide.launch_fused_int4_gemm(
        stream, output, input,
        &int4_bufs.qweight, &int4_bufs.scales, &int4_bufs.qzeros,
        m as u32, n as u32, k as u32, group_size as u32,
    )?;
}
Some(crate::gpu_cache::CachedWeight::Nvfp4(nvfp4_bufs)) => {
    oxide.launch_fused_nvfp4_gemm(
        stream, output, input,
        &nvfp4_bufs.weight_packed, &nvfp4_bufs.weight_scale,
        nvfp4_bufs.weight_global_scale,
        m as u32, n as u32, k as u32,
    )?;
}
```

No allocations, no sanitize, no intermediate buffer. Single kernel launch per GEMM.

### Step 5: Rebuild cubin

```bash
cargo oxide build  # in crates/cuda-oxide-kernels/
./target/release/infers-cuda-oxide-kernels --save-cubin crates/cuda/kernels/compiled/oxide_kernels.cubin
```

### Step 6: Verify correctness with oracle

```bash
# Dump oracle
python3 scripts/dump_oracle_hidden.py \
  --model /home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4 \
  --prompt "The capital of France is" \
  --output-dir /tmp/oracle_int4

# Run engine with probes
INFERS_DUMP_DIR=/tmp/infer_dumps_int4 INFERS_DUMP_PHASE=prefill ./target/release/infer \
  --model /home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4 \
  --tp 2 --prompt "The capital of France is" --max-tokens 1 --no-chat

# Compare
python3 scripts/compare_hidden_states.py \
  --oracle-dir /tmp/oracle_int4 --infer-dir /tmp/infer_dumps_int4
```

Target: cosine ≥ 0.99 on all layers (matching current dequant→GEMM accuracy).

### Step 7: Profile performance

```bash
# Time decode
time ./target/release/infer \
  --model /home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4 \
  --tp 2 --prompt "The capital of France is" --max-tokens 20 --no-chat
```

Target: < 500ms per token (vs current ~2.6s). Theoretical minimum with fused kernel: ~14ms per token (weight bandwidth limited).

## GGUF Extension (Future)

The `Dequantize` trait supports adding GGUF formats. The key difference is the scale layout:
- AutoRound: per-group scales (one scale per 128 weights)
- GGUF Q4_0: per-column scales (one scale per N)
- GGUF Q4_K: super-block + sub-block scales (6-bit min, 6-bit delta)

The tiling structure stays the same — only the scale loading changes. Add a `GgufQ4K` impl of the quant trait when GGUF support is needed.

## Files Modified

| File | Change |
|------|--------|
| `crates/cuda-oxide-kernels/kernel-lib/src/lib.rs` | Rewrite `int4_gemm_inner` with shared memory tiling; add `nvfp4_gemm_fused` |
| `crates/cuda/src/oxide_bridge.rs` | Add `launch_fused_int4_gemm`, `launch_fused_nvfp4_gemm` |
| `crates/backends/native/src/gemm_dispatch.rs` | Replace dequant→GEMM path with fused kernel dispatch |

## Testing

- Run existing CUDA kernel tests (gemm_compare, nvfp4_debug, gdn tests)
- Run oracle comparison (INT4 and NVFP4) — cosine ≥ 0.99
- Time decode before/after
- Check GPU utilization with `nvidia-smi` — should show higher power draw (compute-bound)
