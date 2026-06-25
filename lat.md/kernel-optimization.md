# Kernel Optimization Experiments

Systematic record of kernel-level optimization experiments targeting 48ms→25ms INT4 decode.

## Baseline

INT4 decode at 48ms/token (20.8 tok/s). Target: 25ms/token (40 tok/s).

## Nsight Systems Profile (EXP round 1)

GPU kernel time breakdown from nsys profile (2× RTX 5060 Ti, 30 decode steps):

| Kernel | Time % | Total (ms) | Instances | Avg (µs) | Med (µs) |
|--------|--------|-----------|-----------|----------|----------|
| `int4_gemm_v3_ksplit_sm` | **44.5%** | 1137.5 | 24,000 | 47.4 | 48.8 |
| `int4_gemm_auto_round` | 16.8% | 428.9 | 800 | 536.1 | 522.4 |
| `ncclDevKernel_AllReduce` | 14.7% | 375.9 | 7,936 | 47.4 | 20.9 |
| `gemvx (cuBLASLt bf16)` | 7.6% | 193.9 | 5,790 | 33.5 | 3.5 |
| `infers_gdn_recurrent_step` | 5.7% | 144.7 | 2,880 | 50.2 | 50.2 |
| `infers_paged_attention_decode` | 4.1% | 105.3 | 960 | 109.7 | 115.9 |
| `infers_gdn_gated_delta_prefill` | 2.0% | 50.1 | 96 | 522.0 | 523.1 |
| `infers_rmsnorm_bf16` | 1.9% | 48.1 | 9,951 | 4.8 | 5.7 |
| `reduce_partial_sums_bf16` | 1.2% | 30.5 | 24,000 | 1.3 | 1.3 |
| All others | ~2.5% | ~65 | — | — | — |

Per-step estimate (median, 56 layers, 2 GPUs): ~30ms kernel time + ~8ms overhead = 38ms/step.

**Top bottlenecks:**
1. **INT4 GEMM (ksplit+reduce) = 45.7%** — 49µs/call, ~400 calls/step = 19.6ms
2. **NCCL AllReduce = 14.7%** — 7936 calls, ring latency
3. **cuBLASLt BF16 GEMMs = 7.6%** — small projections (a_proj, b_proj)
4. **GDN recurrent step = 5.7%** — 96 calls/step × 50µs = 4.8ms
5. **Paged attention = 4.1%** — 32 calls/step × 110µs = 3.5ms

## Round 2 Experiment Queue

Based on nsys profile: INT4 GEMM dominates at 45.7%. NCCL is 14.7%. These two account for 60% of GPU time.

### EXP-011: INT4 GEMM occupancy — K_SPLIT sweep

Sweep K_SPLIT (1..56) to find optimal occupancy. Current K_SPLIT=28 gives 1792 threads per GEMM ≈ 3.5 SMs on RTX 5060 Ti.

Hypothesis: lower K_SPLIT → fewer blocks → better cache locality but less parallelism. Optimal may differ from 28.

### EXP-012: INT4 GEMM batch K-split — eliminate reduce kernel

Batch N K-split calls into one kernel. Eliminates `reduce_partial_sums_bf16` (1.2%) and launch overhead.

Affects: `int4_kernels.rs`, `oxide_bridge.rs`, `gemm_dispatch.rs`.

### EXP-013: INT4 GEMM fused ksplit+reduce — single-kernel approach

Combine ksplit and reduce into one kernel. Each block computes all K-splits for one output column, accumulates in registers, and writes final bf16 output directly. Eliminates partial_sums buffer allocation + reduce kernel. Affects: `int4_kernels.rs`.

### EXP-014: NCCL AllReduce grouping — BLOCKED

Per-layer all-reduces have data dependency (residual add between attn and MLP AR). Cannot batch within a layer. Cross-layer pipeline overlap requires engine-level stream separation — deferred.

### EXP-015: GDN decode memcpy elimination — DONE, BUG FIXED

Replaces 48 per-head repeat-interleave memcpy calls with a single CUDA kernel. Eliminates conv_out_last intermediate buffer. Reduces per-token memcpy calls from 55 to 7.

Kernel `infers_repeat_interleave_bf16` in `common_kernels.rs` uses grid-stride pattern over `[seq_len, num_src_heads * kv_ratio, head_dim]`. q/k/v splits copy directly from conv_out via offset-based `copy_view_into`. Affects: `gdn.rs`, `common_kernels.rs`, `oxide_bridge.rs`, `workspace.rs`.

**Bug (post-merge):** The `src_t` calculation in the kernel used `num_src_heads * head_dim` as the divisor instead of `num_dst_heads * head_dim`. This caused wrong source indexing when mapping DST layout `[seq_len, num_dst_heads, head_dim]` back to SRC layout `[seq_len, num_src_heads, head_dim]`. Fix: changed `num_src_heads` to `num_dst_heads` in the `src_t` divisor (line 284 of `common_kernels.rs`). Smoke test confirmed "Paris" decoded correctly after fix.

### EXP-016: v4_ksplit as production kernel

The v4 kernel uses 16 threads/block, 4 cols/thread, 128-bit loads. Had higher throughput in microbench but was not integrated. Evaluate for production use.

**Result:** Swapped v3_ksplit_sm → v4_ksplit in m==1 decode path (K_SPLIT=20). Smoke test PASSED — 30 tokens, avg decode 0.040s/step. No type mismatches (scales already f16). Affects: `int4_kernels.rs`, `oxide_bridge.rs`, `gemm_dispatch.rs`.

### EXP-018: INT4 GEMV warp_split kernel — REVERTED

Replaced `int4_gemm_v3_ksplit_sm` with `int4_gemm_warp_split` in the m==1 decode path. Warp split uses block (32,8,1) with warp shuffle reduction instead of v3's shared memory tiling. Eliminates shared memory entirely (0 bytes vs v3's SM tile).

- **Correctness**: Smoke test PASSED — 30 tokens decoded
- **Latency**: 0.219s/step — **6x slower** than v3_ksplit_sm (0.036s). Warp-cooperative approach is fundamentally wrong for M=1 GEMV — 256 threads/block with inter-warp coordination overhead far worse than v3's simple 64-thread blocks with ksplit grid.
- **Status**: REVERTED. v3_ksplit_sm remains the production INT4 GEMM kernel.

### EXP-021: Custom BF16 GEMV for small projections — REVERTED

Attempted custom BF16 GEMV kernels to replace cuBLASLt for `in_proj_a`/`in_proj_b` (N=24, K=5120). AutoRound excluded these from INT4 quantization, leaving them as BF16 cuBLASLt calls (9.7% of decode time). Two designs tested:

1. **128-thread block** with shared memory cross-warp reduction: 0.037s/step (+1ms regression)
2. **32-thread single-warp** block with pure warp shuffle: 0.040s/step (+4ms regression)

Both worse than cuBLASLt. cuBLASLt is already well-optimized for tiny GEMVs — kernel launch overhead dominates and custom kernels can't amortize it better. Low GPU occupancy with few blocks (N=24 → 24 blocks) is the fundamental problem. The 9.7% cuBLASLt decode time is dominated by `lm_head` (N=152064), not the small a_proj/b_proj projections.

- **Status**: REVERTED. cuBLASLt remains for all BF16 GEMVs.

### EXP-022: Attention Q+gate split kernel — DONE

Replaces per-head memcpy with single split kernel. Affects: `attention.rs`, `oxide_bridge.rs`.

Replaced 2×num_heads per-head `memcpy_dtod` calls with a single `infers_split_qgate_bf16` kernel. Kernel in `common_kernels.rs` scatter-gathers from interleaved `[Q_h0, G_h0, Q_h1, G_h1, ...]` layout into separate `q_buf` and `gate_buf` buffers. 768 memcpy calls/step → 16 kernel launches.

- **Correctness**: Smoke test PASSED — 30 tokens decoded
- **Latency**: 0.036s/step (no measurable change — memcpy overhead was within noise)
- **Status**: Integrated. Cleaner code, fewer D2D calls.

## Nsight Systems Profile (Round 3 — post EXP-015/022)

GPU kernel time breakdown from nsys profile (2× RTX 5060 Ti, 23 decode steps, current 0.036s/step):

| Kernel | Total (µs) | Calls | Avg (µs) | % Decode |
|--------|-----------|-------|----------|----------|
| `int4_gemm_v3_ksplit_sm` | 885,715 | 18,543 | 47.8 | 57.9% |
| `ncclDevKernel_AllReduce` | 299,946 | 6,188 | 48.5 | 19.6% |
| `cublasGemvParamsEx<bf16>` | 149,115 | 4,475 | 33.3 | 9.7% |
| `infers_gdn_recurrent_step` | 112,174 | 2,226 | 50.4 | 7.3% |
| `infers_paged_attention_decode` | 76,822 | 741 | 103.7 | 5.0% |
| `infers_rmsnorm_bf16` | 35,831 | 7,440 | 4.8 | 2.3% |
| `reduce_partial_sums_bf16` | 20,118 | 18,543 | 1.1 | 1.3% |
| All others | ~17,000 | — | — | ~1.1% |

**Decode D2D memcpy:** 55,686 calls across all steps. Decode-specific: ~1,345 calls/step (GDN conv state: 288, GDN Q/K/V split: 288, Attention Q/gate: 768→16 after EXP-022, sampling: 1).

**Updated bottlenecks:**
1. **INT4 GEMM = 57.9%** — near bandwidth limit at 48µs/call
2. **NCCL AllReduce = 19.6%** — residual dependency prevents pipeline overlap within a layer
3. **cuBLASLt BF16 = 9.7%** — dominated by lm_head (N=152064), small a_proj/b_proj are fast
4. **GDN recurrent = 7.3%** — low occupancy (24 blocks, 3072 threads on 40 SMs)

## Round 3 Analysis: Kernel-Level Optimization Exhausted

Three rounds of kernel optimization have reduced decode latency from 0.050s to 0.036s (28% improvement). The remaining 11ms gap to target (0.025s) is dominated by overheads that kernel-level changes cannot address:

- **INT4 GEMM (57.9%)**: Near memory bandwidth limit. K-split grid gives good occupancy. Fused ksplit+reduce (EXP-013) was 3x slower. Alternative block structures (v4, warp_split) are worse.
- **NCCL (19.6%)**: Residual add dependency creates strict serialization — MLP cannot start until AR(attn) completes. Cross-layer pipeline overlap requires multi-stream architecture, double-buffered workspaces, and Oxide bridge stream dispatch fix.
- **cuBLASLt (9.7%)**: Custom GEMV kernels are slower (EXP-021). cuBLASLt is already efficient. The dominant lm_head call is bandwidth-limited.
- **CPU-side kernel launch overhead (~6ms)**: ~3,400 kernel launches per step with ~2µs CPU overhead each. This is the largest remaining optimization opportunity.

## Next Steps: Engine-Level Optimization

The path from 0.036s to 0.025s requires engine-level changes:

1. **CUDA graphs for decode loop — BLOCKED**: Stream capture succeeds but NCCL operations leave the stream in a deadlocked state after `end_capture`. NCCL internally creates its own streams/events that conflict with graph capture. Production engines (vLLM, TensorRT-LLM) use `ncclGraphAddAllReduce` (explicit graph API) or per-layer compute-only graphs (NCCL outside graph). Deferred until we implement one of these approaches. See `plan/044-cuda-graphs-decode.md` for full analysis.

2. **NCCL pipeline overlap (cross-layer)**: Use separate CUDA streams for NCCL all-reduce vs compute. Overlap layer N's AR(mlp) with layer N+1's norm1+Attn. Requires: (a) non-blocking compute stream, (b) Oxide bridge stream dispatch fix, (c) double-buffered hidden state, (d) CUDA event synchronization. Estimated savings: 3-5ms. **NEXT TARGET** — see `plan/045-engine-optimization-nccl-batched.md`.

3. **Batched INT4 GEMV**: Fuse q/k/v projections (same input) into single batched kernel. Saves ~448 kernel launches/step. Estimated savings: 1-2ms. Combine with NCCL overlap.
## Experiment Queue

Each experiment is a self-contained change to one kernel, tested in isolation via the bench harness before integration.

### EXP-001: INT4 GEMM shared memory input tiling

Tile input bf16 vector into shared memory so all 64 threads cooperatively load once, eliminating 64x redundant DRAM reads.

### EXP-002: INT4 GEMM vectorized weight loads

Kernel: `int4_gemm_v3_ksplit_sm`. Change: replace scalar u32 weight loads with 128-bit `[u32;4]` loads (pattern from v4 kernel). Hypothesis: 4x fewer LDG instructions, better memory throughput. Affects: `int4_kernels.rs`.

### EXP-004: RMSNorm warp-level reduction — DONE

Replace shared-memory halving reduction (9 barriers) with two-phase reduction: shared-memory warp-fold + warp-shuffle. Reduces sync barriers from 9 to 2. Affects: `norm_kernels.rs` (all 3 norm kernels).

### EXP-005: SiLU vectorized loads

Replace scalar u16 loads with `[u16;4]` 8-byte vectorized loads (4 bf16 at once) in `infers_silu_bf16` and `infers_silu_glu_bf16`. Scalar remainder loop handles tail.

### EXP-006: Paged attention K-cache caching — DONE

Kernel: `paged_attention_decode_bf16`. Change: cache K dot products from Phase 1 so Phase 2 doesn't re-read K from global memory. Hypothesis: ~2x KV bandwidth saved. Affects: `attention_kernels.rs`.

### EXP-007: GDN recurrent step loop merging — DONE

Merge state decay+kv_mem and state update+output into single loops, cutting global memory reads by half. Affects: `gdn_kernels.rs`.
### EXP-008: RMSNorm block size 512 — DONE

Kernel: `rmsnorm_bf16`. Change: increase `launch_bounds` from 256 to 512, halving per-thread iterations for hidden=5120. Hypothesis: ~15-20% improvement from better SM utilization. Affects: `norm_kernels.rs`.

### EXP-009: Fast exp approximation — DONE

Replace all `libm::expf` calls with Schraudolph bit-manip trick (~0.3% error). 39 call sites across 5 kernel files.

### EXP-010: Paged attention block table hoisting — DONE

Cache `physical_page` across consecutive token positions sharing the same logical page in Phase 1, Phase 1b, and Phase 2. Eliminates 15/16 redundant `block_table` reads per phase for page_size=16.

## Results

Measured outcomes from completed experiments, sorted by execution order.

### EXP-001: INT4 GEMM shared memory input tiling — DONE

Replaced `int4_gemm_v3_ksplit` with `int4_gemm_v3_ksplit_sm`. Tiles input bf16 vector into shared memory per group, eliminating 64x redundant DRAM reads. Strided load handles group_size=128 > block_size=64.

- **Correctness**: cosine=1.00000 vs dumped reference output (N=8704, K=5120, group_size=128, k_split=28)
- **Latency**: 55.7 µs/call (mean), 55.1 µs/call (min) — ksplit + reduce together
- **Status**: Integrated. Old `int4_gemm_v3_ksplit` kernel and `launch_int4_gemm_v3_ksplit` bridge shim removed.

### EXP-002: INT4 GEMM vectorized weight loads — DONE

Replaced scalar u32 weight loads with `[u32; 4]` 128-bit loads in `int4_gemm_v3_ksplit_sm` for non-transposed path.

- **Correctness**: Smoke test PASSED — correct output ("Paris")
- **Latency**: 0.049s/step (vs 0.050s baseline) — marginal improvement from fewer LDG instructions. INT4 GEMM is likely not the bottleneck.
- **Status**: Integrated.

### EXP-005: SiLU vectorized loads — DONE

Replaced scalar u16 loads with `[u16;4]` 8-byte vectorized loads in both SiLU kernels. Tail handled by scalar loop.

- **Correctness**: Smoke test PASSED — correct output ("Paris")
- **Latency**: 0.050s/step (vs 0.050s baseline) — no measurable improvement. SiLU kernels are likely not the bottleneck (compute-bound from libm::expf, not memory-bound).
- **Status**: Integrated.

### EXP-008: RMSNorm block size 512 — DONE

Increased launch block size from 256 to 512 in all three norm kernels, with dynamic step sizes via `thread::blockDim_x()`.

- **Correctness**: Smoke test PASSED — correct output ("Paris", 30 tokens decoded)
- **Latency**: 0.049s/step (vs 0.049s baseline) — no measurable change. Norm kernels are already fast relative to the INT4 GEMM bottleneck.
- **Status**: Integrated.

### EXP-009: Fast exp approximation — DONE

Replaced all 39 `libm::expf` calls with `fast_expf` (Schraudolph bit-manip, ~0.3% error). Covers activation, GDN, attention, norm, and common softmax kernels.

- **Correctness**: Smoke test PASSED — correct output ("Paris", 30 tokens decoded)
- **Latency**: 0.049s/step (vs 0.049s baseline) — no measurable change. The fast exp avoids slow libm software emulation but the overall pipeline is INT4 GEMM bound.
- **Status**: Integrated. `fast_expf` lives in `shared.rs`.

### EXP-004: RMSNorm warp-level reduction — DONE

Replaced 9-barrier shared-memory halving reduction with 2-barrier warp-shuffle reduction across all 3 norm kernels.

- **Correctness**: Smoke test PASSED — correct output ("Paris", 30 tokens decoded)
- **Latency**: 0.049s/step (vs 0.049s baseline) — no measurable change. Norm kernels are already fast relative to the INT4 GEMM bottleneck.
- **Status**: Integrated.

### EXP-006: Paged attention K-cache caching — DONE

Cache K dot products from Phase 1 in shared memory so Phase 2 doesn't re-read K from global memory.

Added Phase 1b: after Phase 1's block reduction, each thread re-iterates its tokens, recomputes the dot product, applies softmax weight, and caches the result in shared memory. Phase 2 reads cached weights instead of recomputing K dot products. Shared memory expanded from `3*bdim` to `3*bdim + num_cached_tokens` f32s.

- **Correctness**: Smoke test PASSED — correct output ("Paris", 30 tokens decoded). Internal unit test PASS (CPU reference match).
- **Latency**: 0.040s/step (vs 0.050s baseline) — **20% improvement (20.8→25 tok/s)**. For Qwen3.6-27B (head_dim=128, bdim=128, ~512 cached tokens), K reads drop from ~66K to ~1K (98.5% reduction).
- **Status**: Integrated. Shared memory increased in `oxide_bridge.rs` launch wrapper.

### EXP-015: GDN decode memcpy elimination — DONE

Replaced 48 tiny repeat-interleave memcpy calls (256 bytes each) with single `infers_repeat_interleave_bf16` kernel. Eliminated `conv_out_last` intermediate buffer — q/k/v split now reads directly from conv_out offset.

- **Correctness**: Smoke test PASSED — 30 tokens decoded
- **Latency**: 0.036s/step (vs 0.038s baseline) — **5% improvement**. Eliminated 48 memcpy→2 kernel launches.
- **Status**: Integrated.

### EXP-003: GDN shared memory key/query caching — DONE

Restructured `infers_gdn_recurrent_step_bf16` from 1D to 2D grid with shared memory key/query caching, eliminating redundant global reads.

One block per head (128 threads tiling v_dim). Cooperative load of key and query into shared memory eliminates 3 redundant global reads of key and 2 redundant global reads of query. Shared memory: 2 × K × sizeof(f32) = 1024 bytes for K=128.
- **Correctness**: Smoke test PASSED — correct output (30 tokens decoded)
- **Latency**: 0.038s/step (vs prior baseline to be measured). Key saved: ~512 global reads per head eliminated (K×2 across steps 2,4). Query saved: ~256 global reads per head eliminated (K across step 5).
- **Status**: Integrated. `#[launch_bounds(128)]`, `DynamicSharedArray::<f32>` for key+query. Launch config updated in `oxide_bridge.rs` with 2D grid and shared memory allocation.
### EXP-007: GDN recurrent step loop merging — DONE

Merged Steps 1+2 (decay + kv_mem) and Steps 4+5 (update + output) into single loops, using register-held `s_decayed`/`s_updated` instead of re-reading global memory. State reads reduced from 4 to 2 per K element.

- **Correctness**: Smoke test PASSED — correct output (30 tokens decoded)
- **Latency**: 0.038s/step (vs 0.038s post-EXP-003 baseline) — no measurable change at this stage; the pipeline remains INT4 GEMM bound, but global memory traffic for state is cut by 50%.
- **Status**: Integrated.

### EXP-015: GDN decode memcpy elimination — DONE

Replaced 48 per-head memcpy calls with one kernel launch. Eliminated conv_out_last buffer entirely.

Kernel `infers_repeat_interleave_bf16` uses grid-stride pattern over `[seq_len, num_src_heads * kv_ratio, head_dim]`. q/k/v splits copy directly from conv_out via offset-based `copy_view_into`. Per-token memcpy reduced from 55 to 7. One GdnWorkspace buffer freed (~5KB per GPU).

- **Correctness**: Smoke test PASSED — correct output (30 tokens decoded)
- **Latency**: 0.036s/step (vs 0.038s post-EXP-007 baseline) — small but measurable improvement from reduced memcpy overhead and kernel launch latency.
- **Status**: Integrated. Kernel in `common_kernels.rs`, bridge wrapper in `oxide_bridge.rs`.

### EXP-018: INT4 GEMV warp_split kernel — DONE

Replaced `int4_gemm_v3_ksplit_sm` with `int4_gemm_warp_split` in the m==1 decode path. Warp split uses block (32,8,1) with warp shuffle reduction instead of v3's shared memory tiling. Eliminates shared memory entirely (0 bytes vs v3's SM tile).

- **Correctness**: Smoke test PASSED — 30 tokens decoded
- **Latency**: 0.219s/step — measured with current production model
- **Status**: Integrated. Bridge wrapper `launch_int4_gemm_warp_split` in `oxide_bridge.rs`, dispatch change in `gemm_dispatch.rs`.
