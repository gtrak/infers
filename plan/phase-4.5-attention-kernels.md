# Phase 4.5: Attention, KV Cache, and GDN Kernels (Custom Build)

**Duration:** 1-2 weeks
**Goal:** Complete the forward pass by building custom CUDA kernels for attention softmax, KV cache management, and Gated DeltaNet state updates, and wire them into the prefill/decode paths.

**Status:** Partially complete from Phase 4. This phase captures the remaining work.

---

## What Phase 4 Accomplished

From `git log`:
- ✅ CUDA feature gate removed (cudarc unconditional)
- ✅ Real cudarc API wired: GemmEngine, NcclCommunicator, kernel loading
- ✅ 7 custom CUDA kernels compiled for sm_120 (Blackwell): RMSNorm, SiLU, RoPE, embedding, add, sampling
- ✅ Forward pass skeleton (11 modules in `backends/native/src/`)
- ✅ Review found & fixed 9 critical bugs: lda/ldc inversion, cubin dedup, rmsnorm precision, arg order, GEMM transpose, RoPE table indexing, multi-head rotation
- ✅ Kernel dispatches implemented: RMSNorm, embedding, sampling, SiLU/GLU, add, RoPE, NCCL sync

## What Phase 4.5 Must Complete

### 1. Attention Kernels

**Online Softmax Kernel** (`kernels/infers/softmax.cu`)
- Numerically stable online softmax for attention scores
- Takes Q*K^T matrix [seq_len × seq_len] (or [1 × seq_len] for decode) as input
- Computes row-wise softmax with `-inf` masking for causal attention
- Launched as: `infers_softmax_bf16(scores, output, seq_len, use_causal)`

**KV Cache Write Kernel** (`kernels/infers/kv_cache.cu`)
- For prefill: writes K and V tensors into paged KV cache blocks
- For decode: appends single K/V pair at current position
- Launched as: `infers_kv_cache_write_bf16(k, v, kv_cache, positions, head_dim)`

**KV Cache Read Kernel** (optional — decode only)
- Reads cached K/V for single-token attention computation

### 2. Gated DeltaNet (GDN) Kernels

**Gated Delta Rule Update** (`kernels/infers/gdn_update.cu`)
- Applies the gated delta rule: `state = state + delta_gate(state, in_proj_a, in_proj_b, dt_proj, x_proj)`
- Recurrent state update for decode path
- Launched as: `infers_gdn_update_bf16(state, in_proj_a, in_proj_b, dt_proj, x_proj, hidden_size, seq_len)`

**GDN Prefill** (`kernels/infers/gdn_prefill.cu`)
- Chunked gated delta rule for prefill path (processes full prompt sequence)
- Applies gated delta rule across all positions in parallel where possible
- Updates Mamba-style SSM state

### 3. Attention Forward Implementation (Rust)

File: `crates/backends/native/src/attention.rs`

Implement the full attention layer using cuBLASLt + custom kernels:

1. **QKV projections**: 3 GEMMs (input @ q_proj^T, input @ k_proj^T, input @ v_proj^T)
2. **RoPE**: apply_rope() on Q and K
3. **KV cache write**: launch `infers_kv_cache_write_bf16`
4. **Attention scores**: Q @ K^T via cuBLASLt GEMM
5. **Softmax**: launch `infers_softmax_bf16` (causal mask for prefill, full mask for decode)
6. **Attention @ V**: scores @ V via cuBLASLt GEMM
7. **O-projection**: attention @ o_proj^T via cuBLASLt GEMM
8. **TP all-reduce**: NCCL all_reduce

For decode:
- Q, K, V projections on single token
- KV cache append (single position)
- Attention over ALL cached positions: scores = Q @ K_cache^T
- Same softmax + attention @ V + O-projection + all_reduce

### 4. GDN Forward Implementation (Rust)

File: `crates/backends/native/src/gdn.rs`

Implement Gated DeltaNet layer:

1. **in_proj_a, in_proj_b**: 2 GEMMs (input @ in_proj_a^T, input @ in_proj_b^T)
2. **x_proj, dt_proj**: 2 GEMMs (input @ x_proj^T, input @ dt_proj^T)
3. **conv1d**: gated 1D convolution (or GEMM with convolution matrix)
4. **Gated delta rule**: launch `infers_gdn_update_bf16` or `infers_gdn_prefill_bf16`
5. **out_proj**: GEMM (gdn_out @ out_proj^T)
6. **TP all-reduce**: NCCL all_reduce

### 5. Prefill/Decode Wiring (Rust)

Files: `crates/backends/native/src/prefill.rs`, `decode.rs`

Complete the end-to-end prefill and decode functions:

**Prefill:**
1. `embed_tokens()` → hidden [seq_len × hidden_size]
2. For each layer (dispatch by LayerType):
   - `rms_norm()` on input
   - If GDN: `gdn::forward()`
   - If Attention: `attention::forward()`
   - `add()` for residual
   - `rms_norm()` for norm2
   - `mlp_forward()` for SwiGLU MLP
   - `add()` for residual
3. Final `rms_norm()`
4. `gemm.matmul_bf16()` for LM head
5. `sample::greedy_sample()` for first token

**Decode:**
1. `embed_tokens()` for single token
2. Same layer loop but with decode variants:
   - `gdn::decode_forward()`
   - `attention::decode_forward()`
3. Final norm + LM head + sample

### 6. ForwardEngine Wiring

File: `crates/backends/native/src/engine.rs`

- Store cached kernel function handles (`CudaFunction`) for fast lookup
- Cache RoPE sin/cos tables across calls
- Wire `prefill()` and `decode()` methods to call the module-level functions
- Handle TP rank selection for multi-GPU dispatch

### 7. Build System

- Update `build.rs` to compile new `.cu` files: `softmax.cu`, `kv_cache.cu`, `gdn_update.cu`, `gdn_prefill.cu`
- Update `KernelRegistry::register_infers_kernels()` with new kernel names
- Verify all `.cubin` files compile cleanly at `sm_120`

### 8. Testing

**Kernel Unit Tests** (CPU-side reference comparison):
- Softmax vs CPU reference (float precision)
- KV cache write/read roundtrip
- GDN state update vs reference PyTorch implementation

**End-to-End Tests**:
- Prefill produces deterministic first token for known prompt
- Decode produces deterministic next token
- TP=2 produces same results as single-GPU for same weights

### 9. Review Cycle

Follow the strict review-fix-commit cycle for each sub-task:
1. Implement → delegate to worker
2. Review → delegate to general agent for bug finding
3. Fix → delegate to worker for corrections
4. Commit after each review pass

---

## Deliverables Checklist

- [ ] `kernels/infers/softmax.cu` + `.cubin`
- [ ] `kernels/infers/kv_cache.cu` + `.cubin`
- [ ] `kernels/infers/gdn_update.cu` + `.cubin`
- [ ] `kernels/infers/gdn_prefill.cu` + `.cubin`
- [ ] `attention.rs` wired (QKV, RoPE, KV cache, scores, softmax, O-proj, all-reduce)
- [ ] `gdn.rs` wired (projections, conv1d, delta rule, out-proj, all-reduce)
- [ ] `prefill.rs` wired end-to-end
- [ ] `decode.rs` wired end-to-end
- [ ] `engine.rs` owns kernel handles and caches
- [ ] All kernel dispatches reviewed for arg order, shared mem, launch config
- [ ] All GEMM configs reviewed for transpose correctness
- [ ] `lat.md` updated with Phase 4.5 deliverables
- [ ] `lat check` passes

---

## Cross-References

- **Phase 4:** Builds on the forward pass skeleton and basic kernels already in place
- **Phase 5:** Pipeline parallelism will split the same layer logic across stages
- **Phase 6:** Continuous batching will call `prefill` and `decode` in loops
- **Phase 2:** Uses cuBLASLt GEMM and NCCL from `infers-cuda`
- **Phase 3:** Uses `ModelConfig` for layer dispatch, `WeightRegistry` for weights

## File Structure

```
crates/backends/native/src/
  prefill.rs          # NOW: wire end-to-end (was skeleton)
  decode.rs           # NOW: wire end-to-end (was skeleton)
  attention.rs          # NOW: full attention implementation (was todo)
  gdn.rs              # NOW: full GDN implementation (was todo)
  engine.rs             # NOW: kernel handle caching + prefill/decode methods
crates/cuda/kernels/infers/
  softmax.cu            # NEW: online softmax kernel
  kv_cache.cu           # NEW: KV cache write/read kernels
  gdn_update.cu         # NEW: decode-time GDN recurrent update
  gdn_prefill.cu        # NEW: prefill-time GDN chunked update
```
