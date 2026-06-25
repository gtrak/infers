# Kernel Optimization Experiments

Systematic record of kernel-level optimization experiments targeting 48ms→25ms INT4 decode.

## Baseline

INT4 decode at 48ms/token (20.8 tok/s). Target: 25ms/token (40 tok/s).

## Experiment Queue

Each experiment is a self-contained change to one kernel, tested in isolation via the bench harness before integration.

### EXP-001: INT4 GEMM shared memory input tiling

Tile input bf16 vector into shared memory so all 64 threads cooperatively load once, eliminating 64x redundant DRAM reads.

### EXP-002: INT4 GEMM vectorized weight loads

Kernel: `int4_gemm_v3_ksplit`. Change: replace scalar u32 weight loads with 128-bit `[u32;4]` loads (pattern from v4 kernel). Hypothesis: 4x fewer LDG instructions, better memory throughput. Affects: `int4_kernels.rs`.

### EXP-003: GDN register-cache key/query

Load key vector once into registers during L2-norm, reuse for steps 2 and 4 (currently 3 separate global memory loads).

### EXP-004: RMSNorm warp-level reduction

Replace shared-memory halving reduction (7 barriers) with warp-shuffle for intra-warp phase.

### EXP-005: SiLU vectorized loads

Kernel: `silu_glu_bf16`. Change: replace scalar u16 loads with u64 vectorized loads (4 bf16 elements at once). Hypothesis: 4x fewer memory transactions. Affects: `activation_kernels.rs`.

### EXP-006: Paged attention K-cache caching

Kernel: `paged_attention_decode_bf16`. Change: cache K dot products from Phase 1 so Phase 2 doesn't re-read K from global memory. Hypothesis: ~2x KV bandwidth saved. Affects: `attention_kernels.rs`.

### EXP-007: GDN shared memory state tiling

Kernel: `gdn_recurrent_step_bf16`. Change: tile one head's state S[h,k,v] into shared memory instead of strided global memory access. Hypothesis: major latency reduction from register-speed state access. Affects: `gdn_kernels.rs`.

### EXP-008: RMSNorm block size 512

Kernel: `rmsnorm_bf16`. Change: increase `launch_bounds` from 256 to 512, halving per-thread iterations for hidden=5120. Hypothesis: ~15-20% improvement from better SM utilization. Affects: `norm_kernels.rs`.

### EXP-009: SiLU fast exp approximation

Kernel: `silu_glu_bf16`. Change: replace `libm::expf` with GPU-native fast exp or tanh-based approximation. Hypothesis: ~2x faster sigmoid computation. Affects: `activation_kernels.rs`.

### EXP-010: Paged attention block table hoisting

Kernel: `paged_attention_decode_bf16`. Change: cache physical_page across consecutive token positions sharing the same logical page in Phase 2. Hypothesis: moderate reduction in redundant block table lookups. Affects: `attention_kernels.rs`.

## Results

Measured outcomes from completed experiments, sorted by execution order.

<!-- Append results here as experiments complete -->
