# Kernel Optimization Experiments

Systematic record of kernel-level optimization experiments targeting 48ms→25ms INT4 decode.

## Baseline

INT4 decode at 48ms/token (20.8 tok/s). Target: 25ms/token (40 tok/s).

## Experiment Queue

Each experiment is a self-contained change to one kernel, tested in isolation via the bench harness before integration.

### EXP-001: INT4 GEMM shared memory input tiling

Tile input bf16 vector into shared memory so all 64 threads cooperatively load once, eliminating 64x redundant DRAM reads.

### EXP-002: INT4 GEMM vectorized weight loads

Kernel: `int4_gemm_v3_ksplit_sm`. Change: replace scalar u32 weight loads with 128-bit `[u32;4]` loads (pattern from v4 kernel). Hypothesis: 4x fewer LDG instructions, better memory throughput. Affects: `int4_kernels.rs`.

### EXP-003: GDN register-cache key/query

Load key vector once into registers during L2-norm, reuse for steps 2 and 4 (currently 3 separate global memory loads).

### EXP-004: RMSNorm warp-level reduction

Replace shared-memory halving reduction (7 barriers) with warp-shuffle for intra-warp phase.

### EXP-005: SiLU vectorized loads

Replace scalar u16 loads with `[u16;4]` 8-byte vectorized loads (4 bf16 at once) in `infers_silu_bf16` and `infers_silu_glu_bf16`. Scalar remainder loop handles tail.

### EXP-006: Paged attention K-cache caching

Kernel: `paged_attention_decode_bf16`. Change: cache K dot products from Phase 1 so Phase 2 doesn't re-read K from global memory. Hypothesis: ~2x KV bandwidth saved. Affects: `attention_kernels.rs`.

### EXP-007: GDN shared memory state tiling

Kernel: `gdn_recurrent_step_bf16`. Change: tile one head's state S[h,k,v] into shared memory instead of strided global memory access. Hypothesis: major latency reduction from register-speed state access. Affects: `gdn_kernels.rs`.

### EXP-008: RMSNorm block size 512 — DONE

Kernel: `rmsnorm_bf16`. Change: increase `launch_bounds` from 256 to 512, halving per-thread iterations for hidden=5120. Hypothesis: ~15-20% improvement from better SM utilization. Affects: `norm_kernels.rs`.

### EXP-009: Fast exp approximation — DONE

Replace all `libm::expf` calls with Schraudolph bit-manip trick (~0.3% error). 39 call sites across 5 kernel files.

### EXP-010: Paged attention block table hoisting

Kernel: `paged_attention_decode_bf16`. Change: cache physical_page across consecutive token positions sharing the same logical page in Phase 2. Hypothesis: moderate reduction in redundant block table lookups. Affects: `attention_kernels.rs`.

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
