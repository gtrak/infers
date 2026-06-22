# Workspace Architecture

Rust workspace for infers, a Qwen3.6-27B inference server using nightly toolchain with cuda-oxide orchestration.

## Crates

Workspace contains 12 crates across server, API, inference backend, and utility domains.

| Crate | Path | Purpose |
|-------|------|---------|
| infers-server | crates/server | axum HTTP server, CLI, main entry point |
| infers-api | crates/api | OpenAI-compatible types + SSE protocol |
| infers-scheduler | crates/scheduler | Session lifecycle, batch construction |
| infers-kv | crates/kv | Hybrid KV state manager |
| infers-model | crates/model | Multi-format model loader |
| infers-backend-native | crates/backends/native | Custom CUDA kernels + cuBLASLt backend |
| infers-backend-gguf | crates/backends/gguf | llama.cpp backend |
| infers-cuda | crates/cuda | cuda-oxide + cudarc hybrid |
| infers-parallelism | crates/parallelism | TP=2 and PP=2 implementations |
| infers-tokenizer | crates/tokenizer | HF tokenizers wrapper |
| infers-metrics | crates/metrics | Prometheus exporter |
| infers-mtp | crates/mtp | MTP draft/verify |

## Dependency Graph

Crate dependency relationships and feature propagation between workspace members.

infers-backend-native and infers-parallelism both depend on infers-cuda for GPU kernel loading and NCCL communication respectively. cudarc is always present — no feature gating.

## Toolchain

Nightly toolchain configuration, Rust edition, and cargo-oxide requirements for CUDA support.

- Rust nightly-2026-04-03 with rust-src, rustc-dev, llvm-tools
- edition = "2024"
- cargo-oxide for CUDA crates

## Dependencies

Key workspace dependencies pinned to exact versions: tokio 1.52.3, axum 0.8.9, serde 1.0.228, clap 4.6.1, prometheus 0.14.0, thiserror 2.0.18. cudarc 0.19.7 with cublaslt, nccl, cuda-13020, f16 features. half 2.6.0 for FP16/BF16. cuda-oxide deferred.

## CUDA Crate

CUDA runtime crate for GPU inference. cudarc is always present with no feature gating or optional deps. Re-exports key cudarc types for consumer convenience.

### Module Structure

Six modules cover context, streams, kernels, GEMM, pinned, and NCCL.

| Module | Purpose |
|--------|---------|
| context | CUDA device context management, CudaRuntime |
| stream | CUDA stream pool for async execution |
| kernels | Kernel registry for pre-compiled .cubin loading |
| gemm | cuBLASLt GEMM engine with `matmul_bf16()` method for BF16 matrix multiplication, plus `matmul_int4()` for INT4-packed weight GEMM with per-group dequantization and native transposed layout support via `Int4GemmConfig.transposed` |
| pinned | Page-locked host memory (`PinnedHostBuffer`) for fast DMA transfers to GPU — Phase 16 Zero-Copy Weight Streaming |
| nccl | Multi-GPU collective operations for TP/PP |

### cuda-oxide: Quantization-Generic Kernels (Phase 18)

Rust→PTX compiler for three quantization-sensitive kernels with trait-based dispatch. Enables AutoRound, GGUF, AWQ, GPTQ with one kernel. Rust source is portable to rust-gpu (SPIR-V) and amdgcn (HIP) for multi-hardware.

### cuda-oxide POC: Vector Add Kernel (Exploration Complete)

End-to-end pipeline validated: Rust kernel → PTX → GPU launch. Standalone crate at `crates/cuda-oxide-poc/` isolated from parent workspace to avoid codegen conflicts with stable builds.

| Attribute | Value | Description |
|-----------|-------|-------------|
| `#[kernel]` | marks function as CUDA kernel | Compiles to PTX via rustc-codegen-cuda |
| `#[cuda_module]` | wraps kernel functions | Generates typed module loader with per-kernel launch methods |
| `thread::index_1d()` | thread index | Type-safe `ThreadIndex<'kernel, Index1D>` witness for DisjointSlice access |
| `DisjointSlice<T>` | GPU output buffer | Bounds-checked writes; each thread gets unique memory location via ThreadIndex |
| `thread::blockDim_x()`, `thread::gridDim_x()` | grid-stride intrinsics | CamelCase (not snake_case) — key API discovery finding |
| `CudaContext::new(0)` | host CUDA context | Device 0, creates default stream |
| `DeviceBuffer::from_host()` / `zeroed()` | GPU memory | Host-to-device transfer or zero-initialized allocation |
| `LaunchConfig::for_num_elems(N)` | launch config | Auto-calculates block/grid from element count (256 threads/block) |

**Build command**: `RUSTFLAGS="-Z codegen-backend=/home/gary/.cargo/cuda-oxide/librustc_codegen_cuda.so" cargo build --release` from within the POC crate directory. **Not** `cargo oxide build -p cuda-oxide-poc` — that targets the workspace root and builds everything.

**Test results**: Both simple kernel (1 thread/element) and grid-stride kernel (256 threads for 1024 elements) pass verification with f32 data. BF16 arithmetic now validated via u16 bit manipulation — see [[lat.md/arch#Workspace Architecture#CUDA Crate#cuda-oxide POC: BF16 and INT4 Kernels (Exploration Complete)]].

**Existing build unaffected**: `cargo build --release -p infers-cuda` (without oxide) still compiles successfully with cudarc + nvcc pipeline.

**Key findings**:

| Finding | Status | Details |
|----------|--------|---------|
| `SharedArray<T, N>` (static smem) | ✅ Works | Declare as `static mut` in kernel body; access via unsafe indexing |
| `DynamicSharedArray<T>::get()` (dynamic smem) | ✅ Works | Returns raw `*mut T`; requires `LaunchConfig.shared_mem_bytes`. Default ~48KB limit on sm_120; >48KB achievable via cuFuncSetAttribute workaround — see [[lat.md/arch#Workspace Architecture#CUDA Crate#cuda-oxide POC: GDN Kernels and 80KB Dynamic Shared Memory (Exploration Complete)]] |
| Tree reduction in shared memory | ✅ Works | Halving stride pattern: `let mut s = total_threads >> 1; while s > 0 { ... }` |
| RMSNorm via shared memory | ✅ Correct | GPU output matches CPU reference within 1e-3 for f32 data |
| Multiple kernels in single `#[cuda_module]` | ✅ Works | 13+ kernels coexist: vec_add, rmsnorm_static_smem, rmsnorm_dynamic_smem, reduce_benchmark, bf16_vec_add, bf16x2_fma_test, int4_unpack_test, int4_gemm, gdn_recurrent_step, gdn_mamba2_update, dynamic_smem_test, dynamic_smem_80kb |
| `#[launch_bounds(N)]` with DynamicSharedArray | ✅ Fixed | Was a cuda-oxide bug: `llvm-export/metadata.rs` omitted `!"kernel"` annotation for launch_bounds kernels, so NVPTX backend didn't emit `.entry` — fixed by adding kernel metadata in the launch_bounds loop |
| `(1..).step_by(1)` iterator pattern | ✅ Works | Finite-range `step_by` works in all POC kernels; unbounded `(1..).step_by(N)` may still fail on `Step::forward` constant asserts — use explicit `while` loop for that case |

### cuda-oxide POC: BF16 and INT4 Kernels (Exploration Complete)

Four additional kernels validating BF16 arithmetic, packed BF16x2 FMA, INT4 bit manipulation, and full INT4 GEMM with dequantization — all passing with bit-exact CPU verification.

**BF16 conversion pipeline**: cuda-oxide has no native `bf16` type. bf16 values are stored as `u16` on host/device, converted via:

| Direction | Method | Details |
|-----------|--------|---------|
| bf16→f32 (read) | `f32::from_bits((u16_bits as u32) << 16)` | Reinterpret upper 16 bits of f32; exact, no loss |
| f32→bf16 (write) | `cuda_device::tcgen05::f32_to_bf16(val)` | Truncate mode — matches CUDA's `__float2bfloat16` |
| f32→bf16 (RNE) | `cuda_device::tcgen05::f32_to_bf16_rne(val)` | Round-to-nearest-even — matches PTX `rn` rounding mode |
| f32→packed bf16x2 | `cvt_f32x2_bf16x2(lo, hi)` | Single PTX instruction, two f32s into one u32 |

**Packed bf16x2 FMA**: `cuda_device::bf16x2::fma_bf16x2(a, b, c)` compiles to `fma.rn.bf16x2` (sm_80+). Multiplication via FMA with zero accumulator: `fma_bf16x2(a, b, 0)`.

**INT4 dequantization pattern**: Extract from packed u32 (8 INT4 per u32), sign-extend via `(val as i8).wrapping_sub(8)`, dequantize as `f32::from(w_int4 - zero_i8) * scale_f32`. All in registers, no shared memory.

**Native f16 type**: Rust's unstable `#![feature(f16)]` works inside cuda-oxide kernels. f16→f32 via `(f16_val as f32)` — **not** via `f32::from(f16)` (trait not implemented). Requires `#![feature(f16)]` at crate root for both host and device code.

**Kernels added**:

| Kernel | Purpose | Test Result |
|--------|---------|-------------|
| `bf16_vec_add` | bf16→f32 add→bf16 pipeline, bit-exact | ✅ 1024 elements |
| `bf16x2_fma_test` | Packed bf16x2 multiply via FMA | ✅ 256 packed pairs |
| `int4_unpack_test` | INT4 unpack + dequantize (f16 scales) | ✅ 512 values |
| `int4_gemm` | Full INT4 GEMM (16×16, bf16 I/O) | ✅ Bit-exact |

**Key discovery — RNE vs truncate mismatch**: The `fma.rn.bf16x2` PTX instruction uses round-to-nearest-even. CPU verification must use the same rounding mode (`f32_to_bf16_rne`) rather than truncation (`f32_to_bf16`) to match GPU results.

**Key discovery — f16 in device code**: `f16::from_bits()` works inside kernels. Casting `(f16_val as f32)` works for f16→f32 conversion. The `From<f16>` trait is not implemented on `f32` — use explicit cast instead.
### cuda-oxide POC: GDN Kernels and 80KB Dynamic Shared Memory (Exploration Complete)

Three additional kernels validating GDN math patterns with `libm` math functions, plus a progressive dynamic shared memory sizing test.

**Math functions in kernels**: cuda-oxide intercepts `libm::expf()`, `libm::logf()`, `libm::sqrtf()` and maps them to NVIDIA libdevice intrinsics (`__nv_expf`, etc.). `f32::sqrt()` also works directly. No native `rsqrtf` — use `1.0f32 / libm::sqrtf(x)`. Sigmoid: `1.0f32 / (1.0f32 + libm::expf(-x))`. Softplus uses piecewise clamping (x>20 → x; x<-20 → 0; else `libm::logf(1.0f32 + libm::expf(x))`).

**Kernels added**:

| Kernel | Purpose | Test Result |
|--------|---------|-------------|
| `gdn_recurrent_step` | GDN single-token decode: L2 norm, softplus decay, sigmoid beta, 5-step recurrence | ✅ H=2, K=4, V=4 bit-exact |
| `gdn_mamba2_update` | Mamba2 SSM single-token: sigmoid decay, softplus delta, SiLU gating | ✅ H=2, head_dim=4 bit-exact |
| `dynamic_smem_test` | Progressive dynamic shared memory sizing test | ✅ 48KB default; >48KB via cuFuncSetAttribute workaround |
| `dynamic_smem_80kb` | Full partitioned 80KB layout (4 partitions: k_normed, k_beta, attn, beta_arr) | ✅ Works with cuFuncSetAttribute |

**Key discovery — dynamic shared memory limit on sm_120**: On RTX 5060 Ti (sm_120/Blackwell), cuda-oxide's default `maxSharedMemoryPerBlock` is approximately 48KB. Progressive testing shows launches succeed up to 49152 bytes (48KB) but fail with `DriverError(1, "invalid argument")` at 57344 bytes (56KB). **Workaround**: `cuFuncSetAttribute` IS accessible through `cuda_core::sys` (re-export of cuda-bindings). Load the function via `module.as_cuda_module().load_function("kernel_name")`, call `unsafe { func.cu_function() }` for the raw CUfunction, then call `unsafe { sys::cuFuncSetAttribute(raw_func, 8, size as i32) }` where `8` is `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`. After setting this attribute, dynamic shared memory up to at least 96KB works.

**Test results with cuFuncSetAttribute**:
| Size | Result |
|------|--------|
| 56 KB (57344 bytes) | ✅ Pass — data verified |
| 80 KB (81920 bytes) | ✅ Pass — data verified |
| 96 KB (98304 bytes) | ✅ Pass — data verified |

**Implication for gdn_chunked_gated_delta_prefill**: Can be ported to cuda-oxide using the cuFuncSetAttribute workaround. No additional feature request needed.

### cuda-oxide + cudarc Coexistence (Validated)

cuda-oxide kernels and cudarc cuBLASLt/NCCL coexist on the same CUDA primary context, sharing device memory via raw pointers.

**Test crate**: `crates/cuda-oxide-coexist/` — standalone workspace with both dependencies: cuda-core, cuda-device, cuda-host (git) + cudarc 0.19 (cublaslt, cuda-13020, f16 features). Build requires the cuda-oxide codegen backend.

**Key findings**:

| Question | Answer | Details |
|----------|--------|---------|
| Same CUDA context? | ✅ Yes | Both libraries operate on the primary CUcontext for device 0; no conflict |
| Shared device memory? | ✅ Yes | cudarc `CudaSlice` and cuda-oxide `DeviceBuffer` allocate from same pool; raw pointers via `cuMemcpyDtoH_v2` / `cuMemcpyDtoD_v2` cross library boundaries |
| Stream coexistence? | ✅ Yes | Separate streams from each library's context default_stream() operate without conflict on the same device |
| cuBLASLt GEMM alongside oxide kernels? | ✅ Yes | cudarc cuBLASLt GEMM (tf32) runs correctly in same process as cuda-oxide kernel launches |
| Raw pointer interop? | ✅ Yes | `CudaSlice.device_ptr()` + `DeviceBuffer.cu_deviceptr()` allow cross-library memcpy and verification |

**Coexistence pattern**: Create cudarc context first, then cuda-oxide context on the same device. Allocate buffers via whichever library's allocator fits. Use cuMemcpyDtoD_v2 for cross-library buffer transfers when kernel arguments require it (cuda-oxide kernels accept `&[T]` / `DisjointSlice<T>`, not raw pointers — so cudarc data must be copied into a cuda-oxide DeviceBuffer before kernel launch).

**API differences requiring workarounds**:
- cudarc allocates via `stream.alloc_zeros()` / `stream.clone_htod()`, reads back via `stream.clone_dtoh()` and `CudaSlice.to_cuda_vec()`
- cuda-oxide allocates via `DeviceBuffer::zeroed(&stream, N)` / `DeviceBuffer::from_host()`, reads back via `to_host_vec(&stream)`
- Raw device pointers: cudarc via `DevicePtr` trait's `device_ptr(&stream)`, cuda-oxide via `cu_deviceptr()`
- No direct raw pointer passing from cudarc into cuda-oxide kernels — data must be copied to a DeviceBuffer first

**Build command**: `RUSTFLAGS="-Z codegen-backend=/home/gary/.cargo/cuda-oxide/librustc_codegen_cuda.so" cargo run --release` from within the coexist crate directory.
# Kernel Extraction and Build System

Pipeline for compiling infers CUDA kernel source to .cubin binaries.

### Kernel Directory Structure

Three directories hold kernel source and compiled binaries under `crates/cuda/kernels/`. All directories contain `.gitkeep` files for git tracking.

| Directory | Contents |
|-----------|----------|
| `flashinfer-gdn/` | Reserved for future custom GDN kernel source |
| `flashinfer-attn/` | Reserved for future custom attention kernel source |
| `infers/` | Custom CUDA kernel source (.cu, .cuh) for inference operations |
| `compiled/` | Compiled .cubin output from nvcc |

### Kernel Source Files

All kernels use `extern "C" __global__` so function names are directly loadable from cubin files. Launch configuration is determined by Rust dispatch code, not kernel wrappers.

Twenty-two kernel implementations across 20 files for transformer forward-pass operations using BF16 data, plus INT4 GEMM for AutoRound quantization.

| File | Kernels | Description |
|------|---------|-------------|
| `common.cuh` | — | Shared utilities: `__nv_bfloat16` conversion helpers, `INFERS_BLOCK_SIZE` (256), thread indexing macros |
| `rmsnorm.cu` | `infers_rmsnorm_bf16` | RMS Layer Normalization: output = x * rsqrt(mean(x²) + eps) * weight, using float shared memory for precision-preserving reduction. Qwen3_5RMSNorm stores multiplicative scale weight (init=1). Gated variant uses full scale — see `rms_norm_gated.cu` |
| `silu.cu` | `infers_silu_bf16`, `infers_silu_glu_bf16` | SiLU activation and SwiGLU gating: output = x * sigmoid(gate) |
| `rope.cu` | `infers_rope_bf16` | Rotary Position Embedding applied to query and key tensors |
| `embedding.cu` | `infers_embedding_gather_bf16` | Token embedding gather: gather rows from weight matrix by token ID |
| `elementwise.cu` | `infers_add_bf16` | Element-wise addition for residual connections |
| `sampling.cu` | `infers_argmax_f32`, `infers_argmax_bf16` | Greedy argmax sampling: F32 variant for legacy CPU round-trip path, BF16 variant operates directly on BF16 logits on GPU eliminating download→convert→upload cycle |
| `softmax.cu` | `infers_softmax_bf16` | Online softmax for attention scores with optional causal masking, using three-phase parallel reduction (max, sum, normalize) in shared memory |
| `kv_cache.cu` | `infers_kv_cache_write_bf16` | Scattered KV cache write using position IDs: writes K and V rows into cache at arbitrary positions via strided thread loops |
| `gdn_update.cu` | `infers_gdn_update_bf16` | Gated DeltaNet decode kernel: recurrent state update for a single token via three-phase block reduction (beta, state update, output) with one block per state row |
| `gdn_recurrent_step.cu` | `infers_gdn_recurrent_step_bf16` | Gated DeltaNet single-token decode: L2-normalize q/k with 1/sqrt(K) scaling, softplus-clamped decay, sigmoid beta, state update with outer product — one thread per (head, v_dim) element, no shared memory |
| `gdn_gated_delta_prefill.cu` | `infers_gdn_gated_delta_prefill_bf16` | Gated DeltaNet sequential prefill: per-token recurrence with L2 normalization, softplus decay, sigmoid beta — one thread per (head, v_dim) element, no shared memory, sequential token loop |
| `gdn_chunked_gated_delta_prefill.cu` | `infers_gdn_chunked_gated_delta_prefill_bf16` | Gated DeltaNet chunked parallel prefill: replaces the per-token loop with intra-chunk WY representation via attn matrix + forward substitution, followed by inter-chunk state recurrence — one block per head, 256 threads, ~80KB shared memory for k_normed, k_beta, and attn buffers |
| `gdn_gated_delta_update.cu` | `infers_gdn_gated_delta_update_bf16` | Gated Delta Rule single-token decode: L2-normalize q/k, softplus decay, sigmoid beta, state update — variant of recurrent_step with gated delta rule specific logic |
| `gdn_mamba2_prefill.cu` | `infers_gdn_mamba2_prefill_bf16` | Mamba2 SSM prefill kernel: element-wise SSM recurrence with softplus delta, state update, SiLU gating — one thread per total_dim element (total_dim = num_heads × head_dim), per-head signals (x_proj, b_proj, A_log, dt_bias) broadcast across head_dim, sequential token loop, no shared memory |
| `gdn_mamba2_update.cu` | `infers_gdn_mamba2_update_bf16` | Mamba2 SSM decode kernel: single-token state update with sigmoid decay, softplus delta, SiLU gating — one thread per total_dim element (total_dim = num_heads × head_dim), per-head signals broadcast across head_dim, no token loop, no shared memory |
| `paged_kv_write.cu` | `infers_paged_kv_write_bf16` | Paged KV cache write using block-table address translation: writes K and V into interleaved per-page layout via strided thread loops, eliminating CPU round-trips during prefill |
| `paged_kv_read.cu` | `infers_paged_kv_read_bf16` | Paged KV cache read using block-table address translation: gathers K and V from interleaved per-page layout into contiguous output buffers via strided thread loops, eliminating CPU round-trips during decode |
| `paged_attention_decode.cu` | `infers_paged_attention_decode_bf16` | Paged attention decode: computes single-token attention over paged KV cache using two-pass online softmax and weighted V accumulation, one block per KV head — Phase 1 uses strided dot-product computation, Phase 2 loops over all tokens per thread |
| `fp8_quantize.cu` | `infers_fp8_quantize_bf16`, `infers_fp8_dequantize_bf16` | FP8 quantize (BF16→FP8) and dequantize (FP8→BF16) for KV cache quantization, supporting both E4M3 (mode=0) and E5M2 (mode=1) formats — one thread per element, 256 threads per block |
| `int4_gemm.cu` | `int4_gemm_kernel` | INT4 GEMM with per-group dequantization in registers and native transposed [K/8, N] layout support via `transposed` flag: weights stay packed as INT4 (8 per uint32), dequantize `(w_int4 - (zero + 1)) * scale` on-the-fly during inner loop (AutoRound uses biased zero points — stored `z` represents actual zero point `z+1`), accumulate in FP32, output BF16 — 16×16 thread blocks, one thread per output element |

### Build Script

Compiles `.cu` in `kernels/infers/` to .cubin via nvcc `-O3`. Non-GDN kernels use `--use_fast_math`; GDN kernels are excluded from `--use_fast_math` due to precision requirements. Targets `sm_120` by default (`INFERS_CUDA_ARCH` override).

**Precision policy**: `--use_fast_math` causes `expf()`/`logf()`/`rsqrtf()` to use reduced-precision approximations (~2 ULP vs ~1 ULP). In the GDN recurrence kernel (`gdn_gated_delta_prefill.cu`), these small per-step errors compound through the sequential state update, causing cosine similarity of only ~0.94 vs PyTorch reference after 15 tokens (token 0 matches perfectly at 1.0, worst at token 9 = 0.84). To prevent this, all GDN kernel files (`gdn_*.cu`) are compiled **without** `--use_fast_math`, while the remaining kernels (softmax, silu, conv1d_depthwise, etc.) retain the flag for performance. The build script determines this by checking whether the file stem starts with `"gdn"` in `compile_kernel()`.

The `find_nvcc()` function checks PATH first, then falls back to common CUDA install locations (`/usr/local/cuda/bin/nvcc`, `/usr/local/cuda-13.2/bin/nvcc`, `/usr/local/cuda-13.0/bin/nvcc`, `/usr/bin/nvcc`). Missing nvcc or source files produce warnings but do not fail the build. Compiled kernels are placed in `kernels/compiled/` with matching names and loaded at runtime by the KernelRegistry.

### cuda-oxide Migration Assessment (Phase 18 Complete)

Migration assessment: **MIGRATE LATER.** All kernel features technically feasible (13 POC kernels pass), but alpha quality, no native bf16 type, and workspace friction make production migration premature. Full analysis in `plan/research/cuda-oxide.md`.

**Key results**: 13 kernels across 5 exploration commits pass on RTX 5060 Ti (sm_120): vec_add, rmsnorm (static+dynamic smem), reduce, bf16_vec_add, bf16x2_fma, int4_unpack, int4_gemm, gdn_recurrent_step, gdn_mamba2_update, dynamic_smem_test, dynamic_smem_80kb, plus 5 cudarc coexistence tests. Bugfix in cuda-oxide `llvm-export/metadata.rs` for launch_bounds kernel metadata (upstream pending). 80KB+ dynamic shared memory works via `cuFuncSetAttribute` workaround.

**Blockers**: (1) No native bf16 type — all bf16 I/O via u16 bit manipulation. (2) Workspace integration friction — standalone crate required. (3) cudarc→oxide memory copy overhead per kernel call. (4) Alpha API instability (v0.2.1). See `plan/research/cuda-oxide.md` for full analysis.

### cuda-oxide Generic Kernel Experiments (Phase 19 Complete — Assessment Corrected)

Experimented whether trait-based dequant dispatch is possible in cuda-oxide generic kernels, as envisioned in [[lat.md/arch#Workspace Architecture#CUDA Crate#cuda-oxide: Quantization-Generic Kernels (Phase 18)]]. **Result: FEASIBLE via `cargo oxide` build path.** Initial experiments via `RUSTFLAGS` codegen backend path failed, but `cargo oxide` (full NVVM→PTX pipeline) handles generics correctly.

**Experiment 1 — RUSTFLAGS path (FAILS)**: Generic `#[kernel]` with `Dequant` trait, compiled via `RUSTFLAGS="-Z codegen-backend=..."`. Fails with E0282 (phantom type param) and NoModules (NVVM IR vs PTX payload mismatch). The RUSTFLAGS path skips NVVM linking and embeds NVVM IR directly — incompatible with generic kernels.

**Experiment 1a — cargo oxide path (WORKS)**: `cargo oxide run cross_crate_embedded` with `scale<T>` generic kernel — **PASSES**. The `cargo oxide` command runs the full NVVM→PTX pipeline, which correctly monomorphizes generic kernels. This is the supported build path.

**Experiment 1b — E0282 workaround**: The Rust inference error for phantom type params is solvable via `_marker: PhantomData<D>` argument. Zero-cost at runtime (ZST). Launch from host with `PhantomData::<AutoRound>`.

**Experiment 1c — Monomorphized wrappers for cudarc loading**: To get predictable PTX entry point names for cudarc's KernelRegistry, create named `#[kernel]` wrappers around the generic inner function. Each wrapper monomorphizes to a separate PTX entry point.

**Experiment 2 — Const generics**: Still fail at runtime (`"named symbol not found"`). Not needed for trait dispatch — use type parameters instead.

**Revised assessment**: Trait-based generic dispatch IS the right approach. Use `cargo oxide` as the build tool, `PhantomData<D>` for E0282, and named monomorphized wrappers for cudarc PTX loading. See [[plan/024-cuda-oxide-quant.md]] for the production plan.

### cuda-oxide Kernel Library (Phase 18 — Build Pipeline Validated)

Standalone workspace at `crates/cuda-oxide-kernels/` for production kernels compiled via cuda-oxide. Cross-crate kernel library pattern with `kernel-lib` subcrate and host test binary.

**Workspace structure**: `Cargo.toml` defines `[workspace]` with `members = ["kernel-lib"]`. Root crate is the host test binary; `kernel-lib/` holds the `#[cuda_module]` kernel definitions. Not a member of the infers parent workspace — avoids codegen backend conflicts during stable builds.

**Kernel: infers_add_bf16**: Ported from `crates/cuda/kernels/infers/elementwise.cu`. Grid-stride loop, 256 threads per block (`#[launch_bounds(256)]`). bf16 stored as u16 — converts to f32 for compute, truncates back via `cuda_device::tcgen05::f32_to_bf16()`. Bit-exact verification against CPU reference passes (1024 elements).

**Build commands**:
```bash
cd crates/cuda-oxide-kernels
cargo oxide build    # compile kernels to PTX
cargo oxide run      # build + run test binary
```
