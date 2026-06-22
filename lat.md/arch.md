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

Seven modules cover context, streams, kernels, GEMM, pinned, NCCL, and oxide_bridge.

| Module | Purpose |
|--------|---------|
| context | CUDA device context management, CudaRuntime |
| stream | CUDA stream pool for async execution |
| kernels | Kernel registry for pre-compiled .cubin loading |
| gemm | cuBLASLt GEMM engine with `matmul_bf16()` method for BF16 matrix multiplication, plus `matmul_int4()` for INT4-packed weight GEMM with per-group dequantization and native transposed layout support via `Int4GemmConfig.transposed` |
| pinned | Page-locked host memory (`PinnedHostBuffer`) for fast DMA transfers to GPU — Phase 16 Zero-Copy Weight Streaming |
| nccl | Multi-GPU collective operations for TP/PP |
| oxide_bridge | Loads pre-compiled cuda-oxide kernels from `.cubin` at runtime and launches them via cudarc `CudaSlice<T>` buffers and `CudaStream`

### Oxide Bridge: Runtime Kernel Loading

One `OxideKernels` instance per GPU loads the cubin on the correct device's primary context, preventing cross-GPU context errors in tensor-parallel inference.

  Resolves all kernel function handles into a `HashMap<&str, CudaFunction>`. Type-safe launch wrappers accept cudarc `CudaSlice<T>` buffers — the bridge casts `CUdeviceptr` between cudarc and cuda-oxide type namespaces while keeping `SyncOnDrop` guards alive during launches. Proven via `launch_add_bf16` test: cudarc allocates bf16 buffers, bridge launches kernel, result verified on CPU.

### Oxide Bridge: Launch Wrapper Methods (27 Kernels)

Twenty-seven launch wrapper methods on the `OxideKernels` impl block. Each follows the same pattern as `launch_add_bf16`: device pointers from cudarc, cast to cuda-oxide CUdeviceptr, pack args, call `raw_launch`.

**Element-wise kernels** (use `LaunchConfig::for_num_elems(n)`):

| Method | Kernel | Parameters |
|--------|--------|------------|
| `launch_embedding_gather_bf16` | `infers_embedding_gather_bf16` | weight: bf16, token_ids: i32, output: bf16, seq_len, hidden_size |
| `launch_silu_bf16` | `infers_silu_bf16` | x: bf16, output: bf16, total_elements |
| `launch_silu_glu_bf16` | `infers_silu_glu_bf16` | x: bf16, gate: bf16, output: bf16, total_elements |
| `launch_attn_output_gate_bf16` | `infers_attn_output_gate_bf16` | x: bf16, gate: bf16, output: bf16, total_elements |
| `launch_kv_cache_write_bf16` | `infers_kv_cache_write_bf16` | k: bf16, v: bf16, kv_cache: bf16, positions: i32, seq_len, head_dim, max_seq_len |
| `launch_conv1d_depthwise_silu_bf16` | `infers_conv1d_depthwise_silu_bf16` | input: bf16, weight: bf16, output: bf16, batch_size, conv_dim, seq_len, kernel_size |
| `launch_paged_kv_write_bf16` | `infers_paged_kv_write_bf16` | k: bf16, v: bf16, page_pool: bf16 (write), block_table: i32, positions: i32, seq_len, head_dim, page_size, kv_dim |
| `launch_paged_kv_read_bf16` | `infers_paged_kv_read_bf16` | page_pool: bf16, block_table: i32, num_pages, num_cached_tokens, head_dim, page_size, kv_dim, k_out: bf16 (write), v_out: bf16 (write) |
| `launch_fp8_quantize_e4m3` | `infers_fp8_quantize_e4m3` | input: bf16, output: u8 (write), n |
| `launch_fp8_dequantize_e4m3` | `infers_fp8_dequantize_e4m3` | input: u8, output: bf16 (write), n |
| `launch_fp8_quantize_e5m2` | `infers_fp8_quantize_e5m2` | input: bf16, output: u8 (write), n |
| `launch_fp8_dequantize_e5m2` | `infers_fp8_dequantize_e5m2` | input: u8, output: bf16 (write), n |

**Per-row kernels** (explicit grid/block config with dynamic shared memory):

| Method | Kernel | Config |
|--------|--------|--------|
| `launch_argmax_bf16` | `infers_argmax_bf16` | grid=(batch_size), block=(256), smem=2048 bytes (static arrays) |
| `launch_rmsnorm_bf16` | `infers_rmsnorm_bf16` | grid=(num_rows), block=(min(hidden,256)), smem=block*4 bytes |
| `launch_rms_norm_gated_bf16` | `infers_rms_norm_gated_bf16` | grid=(n), block=(min(d,256)), smem=block*4 bytes |
| `launch_l2norm_bf16` | `infers_l2norm_bf16` | grid=(num_rows), block=(min(dim,256)), smem=block*4 bytes |
| `launch_softmax_bf16` | `infers_softmax_bf16` | grid=(num_rows), block=(min(seq_len,1024)), smem=block*4 bytes |
| `launch_gdn_update_bf16` | `infers_gdn_update_bf16` | grid=(hidden_size), block=(256), smem=1024 bytes (two-phase reduction) |

**Tile-based kernels** (2D grid, INT4 GEMM with 64x4 blocks):

| Method | Kernel | Config |
|--------|--------|--------|
| `launch_int4_gemm_auto_round` | `int4_gemm_auto_round` | grid=(ceil(n/64), ceil(m/4)), block=(64,4), smem=0 |
| `launch_int4_gemm_gguf` | `int4_gemm_gguf` | grid=(ceil(n/64), ceil(m/4)), block=(64,4), smem=0 |

**In-place write kernels** (q and k are writable `DisjointSlice<u16>`):

| Method | Kernel | Notes |
|--------|--------|-------|
| `launch_rope_bf16` | `infers_rope_bf16` | q: bf16, k: bf16 (both mutable), cos: f32, sin: f32, positions: i32, total_tokens, num_heads, head_dim, rotary_dim |

**Paged attention** (one block per KV head):

| Method | Kernel | Config |
|--------|--------|--------|
| `launch_paged_attention_decode_bf16` | `infers_paged_attention_decode_bf16` | grid=(num_kv_heads), block=(256), smem=head_dim*4 bytes (Q storage) |

**GDN kernels** (state as f32 for recurrent/gated_delta; bf16 for mamba2/update):

| Method | Kernel | Config | State type |
|--------|--------|--------|------------|
| `launch_gdn_recurrent_step_bf16` | `infers_gdn_recurrent_step_bf16` | LaunchConfig::for_num_elems(H*V) | f32 (read+write) |
| `launch_gdn_mamba2_update_bf16` | `infers_gdn_mamba2_update_bf16` | LaunchConfig::for_num_elems(nh*d) | bf16 (read+write) |
| `launch_gdn_gated_delta_update_bf16` | `infers_gdn_gated_delta_update_bf16` | LaunchConfig::for_num_elems(H*V) | f32 (read+write) |
| `launch_gdn_gated_delta_prefill_bf16` | `infers_gdn_gated_delta_prefill_bf16` | LaunchConfig::for_num_elems(H*V) | f32 (read+write) |
| `launch_gdn_chunked_gated_delta_prefill_bf16` | `infers_gdn_chunked_gated_delta_prefill_bf16` | grid=(num_heads), block=(256), smem=(2*C*K + C*C + 3*C)*4 bytes | f32 (read+write) |

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

### cuda-oxide: INT4 GEMM with Trait-Based Dequantization Dispatch

Flagship kernel in `infers-kernel-lib` — computes `output[M][N] = dequant(weight) @ input[M][K]` with per-group FP16 scales and packed INT4 zero points, using Rust trait dispatch for multi-format support.

**Dequantize trait**: Defines `dequant(w_int4: i8, raw_zero: i8, scale: f32) -> f32` — the single point of format-specific logic. Two implementations:

| Format | Zero Point | Formula |
|--------|-----------|---------|
| AutoRound | `zero = stored_zero + 1` | `(w - (stored_zero + 1)) * scale` |
| GGUF | `zero = stored_zero` | `(w - stored_zero) * scale` |

**Architecture**: Generic inner function `int4_gemm_inner<Q: Dequantize>` handles all kernel logic (thread indexing, transposed/non-transposed layouts, group iteration, weight unpacking). Two `#[kernel]` wrappers (`int4_gemm_auto_round`, `int4_gemm_gguf`) monomorphize the inner function for each format. The inner function is **not** a `#[kernel]` — it's `#[inline(always)]` and inlined into each wrapper at compile time.

**Layout support**: Two weight layouts via `transposed` flag:
- `transposed=0`: weight [N, K/8], scales [N, K/group_size], zeros flat-packed
- `transposed=1`: weight [K/8, N], scales [K/group_size, N], zeros [K/group_size, ceil(N/8)]

**FP16 scale conversion**: Custom `f16_to_f32` function handles subnormals, normals, and inf/NaN without depending on Rust's unstable f16 type. Bias adjustment: 15→127, mantissa shift: 10→23 bits.

**Launch configuration**: Prefill (M>1): 16×16 blocks, tiled grid. Decode (M=1): 256 threads flat. Each thread computes one output element with fp32 accumulation, writing bf16 result via `f32_to_bf16`.

**Kernels added**:

| Kernel | Purpose | Test Result |
|--------|---------|-------------|
| `int4_gemm_auto_round` | AutoRound INT4 GEMM (transposed=1) | ✅ M=2, N=16, K=64 vs CPU reference |
| `int4_gemm_gguf` | GGUF INT4 GEMM (transposed=0) | ✅ M=2, N=16, K=64 vs CPU reference |

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

### cuda-oxide Kernel Library: Tier 1 Kernels (Phase 18 — 6 Kernels Ported)

Six additional Tier 1 CUDA kernels ported from nvcc to Rust in cuda-oxide-kernel-lib. All pass bit-exact verification.

**Kernels added**:

| Kernel | Source File | Description | Test Result |
|--------|-------------|-------------|-------------|
| `infers_embedding_gather_bf16` | `embedding.cu` | Token embedding gather: `output[i] = weight[token_ids[pos] * hidden_size + dim]` | ✅ 4 tokens × 8 hidden, positions [0, 2] |
| `infers_silu_bf16` | `silu.cu` | SiLU activation: `x * sigmoid(x)` using `libm::expf(-val)` | ✅ 256 elements bit-exact |
| `infers_silu_glu_bf16` | `silu.cu` | SwiGLU gating: `x * gate * sigmoid(gate)` | ✅ 256 elements bit-exact |
| `infers_attn_output_gate_bf16` | `silu.cu` | Attention output gate: `x * sigmoid(gate)` (no gate multiply) | ✅ 256 elements bit-exact |
| `infers_argmax_bf16` | `sampling.cu` | Per-row argmax via shared memory halving-stride reduction, one block per row | ✅ 2 rows × 16 vocab, known max positions |
| `infers_kv_cache_write_bf16` | `kv_cache.cu` | Scattered KV cache write by position: K at `pos * head_dim + dim`, V at offset + same | ✅ 2 tokens × 4 head_dim, positions [2, 5] |

**Shared memory patterns**: `infers_argmax_bf16` uses two `static mut SharedArray<f32, 256>` (one for values, one for indices stored as f32). Thread 0 writes the final argmax index. Launch config: one block per row via direct `LaunchConfig { grid_dim, block_dim }` construction (no `for_num_elems` convenience for multi-block launches).

**Sigmoid implementation**: All three sigmoid kernels (`silu_bf16`, `silu_glu_bf16`, `attn_output_gate_bf16`) use the same pattern: bf16→f32 conversion, `libm::expf(-val)` for sigmoid denominator, f32→bf16 truncation via `cuda_device::tcgen05::f32_to_bf16()`. CPU verification uses matching `libm::expf()` for bit-exact comparison.

### cuda-oxide Kernel Library: Tier 2 Kernels (Phase 18 — 8 Kernels Ported)

Eight shared memory and advanced compute kernels ported from nvcc to Rust in cuda-oxide-kernel-lib. All pass verification via tolerance-based CPU reference.

**Kernels added**:

| Kernel | Source File | Description | Test Result |
|--------|-------------|-------------|-------------|
| `infers_rmsnorm_bf16` | New | RMSNorm with dynamic shared memory tree reduction: `x * rsqrt(mean(x²) + eps) * (1 + weight)` | ✅ 2 rows × 8 hidden, bit-exact |
| `infers_rms_norm_gated_bf16` | New | RMSNorm + SiLU gate: `weight * x_norm * SiLU(gate)` with shared memory reduction | ✅ 2 rows × 8 dim, tolerance 0.5 |
| `infers_l2norm_bf16` | `l2norm_bf16.cu` | L2 normalize per row: `input / sqrt(sum(input²) + eps)` via shared memory reduction | ✅ 2 rows × 8 dim, unit length verified |
| `infers_softmax_bf16` | `softmax.cu` | 3-phase softmax (max→exp-sum→normalize) with optional causal mask | ✅ 4×4 matrix, rows sum to ~1.0 |
| `infers_conv1d_depthwise_silu_bf16` | `conv1d_depthwise.cu` | Depthwise 1D conv + SiLU activation, padding = kernel_size - 1 | ✅ 1 batch × 2 dim × 4 seq, tolerance 1.0 |
| `infers_paged_kv_write_bf16` | `paged_kv_write.cu` | Paged KV cache write with block-table address translation | ✅ Round-trip: write then read back matches |
| `infers_paged_kv_read_bf16` | `paged_kv_read.cu` | Mirror of paged KV write — reads from page_pool using block_table | ✅ Same round-trip test |
| `infers_rope_bf16` | `rope.cu` | Rotary Position Embedding with half-split pairing (rotate_half/GPT-NeoX) | ✅ 2 tokens × 2 heads × 4 head_dim, tolerance 1.0 |

**Dynamic shared memory reduction pattern**: Kernels using `DynamicSharedArray::<f32>::get()` follow a consistent pattern:

1. **Phase 1 — Partial reduction**: Each thread accumulates partial results over its grid-stride chunk, writes to `smem[tid]`, then `sync_threads()`.
2. **Phase 2 — Halving reduction**: Loop from stride = 128 down to 1, adding (or maxing) adjacent pairs: `if tid < stride { smem[tid] += smem[tid + stride] }` with `sync_threads()` between steps.
3. **Phase 3 — Scalar computation**: Thread 0 reads the reduced value from `smem[0]`, computes a scalar (e.g., inverse RMS, inverse norm), writes back to `smem[0]`, then `sync_threads()`.
4. **Phase 4 — Apply transformation**: All threads read the scalar from `smem[0]` and apply it to their respective elements in a grid-stride loop.

**Device sqrt**: `f32::sqrt()` compiles directly to PTX `sqrt.rn.f32` (validated in POC and kernel-lib build). Replaces the initial `dev_sqrtf()` bit-hack + Newton-Raphson implementation. The bit-hack was introduced because `libm::sqrtf()` uses x86 inline assembly that fails PTX generation, but `f32::sqrt()` is a compiler intrinsic that nvptx backend handles natively. Used by `rmsnorm_bf16`, `rms_norm_gated_bf16`, and `l2norm_bf16`.

**Index layout for conv1d**: Output decomposition uses `[batch][seq_len][conv_dim]` layout (D innermost, matches nvcc `conv1d_depthwise.cu`): `d = i % conv_dim`, `t = (i / conv_dim) % seq_len`, `b = i / (seq_len * conv_dim)`. Input indexing: `inp_idx = b * seq_len * conv_dim + adj_t * conv_dim + d`. Bounds check avoids usize underflow: `input_t >= pad && input_t < seq_len + pad`. Original port had incorrect [B,D,T] layout (T innermost) — fixed to match nvcc.

**Paged KV address translation**: Both read and write use block_table to map logical pages to physical pages. Write takes a `positions` array for scattered writes; read uses contiguous positions (0..num_cached_tokens) for sequential reads. Page stride = 2 × page_size × kv_dim (K and V stored back-to-back in each physical page).

**Rope half-split pairing**: Each rotary pair pairs dimension `dim_pair` with `dim_pair + half_rotary` (rotate_half/GPT-NeoX convention). Q and K are rotated in-place using the same cos/sin values for a given token position.

### cuda-oxide Kernel Library: Tier 3 — INT4 GEMM (Phase 18 — Trait-Based Dispatch Validated)

INT4 GEMM with trait-based dequantization dispatch validates the monomorphized wrapper pattern from Experiment 1c in [[lat.md/arch#Workspace Architecture#CUDA Crate#cuda-oxide Generic Kernel Experiments (Phase 19 Complete — Assessment Corrected)]].

**Trait: Dequantize**: Defines `dequant(w_int4: i8, raw_zero: i8, scale: f32) -> f32` — the single point of format-specific logic. Two implementations:

| Format | Zero Point | Formula |
|--------|-----------|---------|
| AutoRound | `zero = stored_zero + 1` | `(w - (stored_zero + 1)) * scale` |
| GGUF | `zero = stored_zero` | `(w - stored_zero) * scale` |

**Architecture**: Generic inner function `int4_gemm_inner<Q: Dequantize>` is `#[inline(always)]` and inlined into each wrapper at compile time. Two `#[kernel]` wrappers (`int4_gemm_auto_round`, `int4_gemm_gguf`) monomorphize the inner function per format.

**FP16 scale conversion**: Custom `f16_to_f32` function handles subnormals, normals, and inf/NaN without Rust's unstable f16 type. Bias adjustment: 15→127, mantissa shift: 10→23 bits.

| Kernel | Test Result |
|--------|-------------|
| `int4_gemm_auto_round` | ✅ M=2, N=16, K=64, transposed=1 vs CPU reference |
| `int4_gemm_gguf` | ✅ M=2, N=16, K=64, transposed=0 vs CPU reference |

### cuda-oxide Kernel Library: Tier 4 — FP8 and Paged Attention (Phase 18 — 5 Kernels Ported)

Five kernels ported from nvcc to Rust: two FP8 format traits with four quantize/dequantize wrappers, plus the paged attention decode kernel with KvCacheFormat trait.

**Trait: Fp8Format**: Defines `quantize(val: f32) -> u8` and `dequantize(val: u8) -> f32` — the single point of format-specific bit layout logic. Two implementations:

| Format | Exponent | Bias | Mantissa | Max Finite |
|--------|----------|------|----------|------------|
| E4M3 | 4 bits | 7 | 3 bits | +0x77 / -0xF7 |
| E5M2 | 5 bits | 15 | 2 bits | +0x7B / -0xFB |

E4M3 provides better precision (smaller quantization error ~25%) while E5M2 supports a wider dynamic range. NaN handling: E4M3 maps NaN→0x7F, Inf→max finite; E5M2 uses sign-preserving NaN/Inf (NaN→0x7F/0xFF, Inf→0x7C/0xFC). Both handle zero, clamp-to-max-finite, and underflow-to-zero. Subnormal dequantization: when fp8 exp=0 with nonzero mantissa, the output fp32 exponent is 0 (subnormal path) rather than adding the bias offset. Generic inner functions `fp8_quantize_inner<F: Fp8Format>` and `fp8_dequantize_inner<F: Fp8Format>` compute grid-stride thread indices via `thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()` (not `thread::index_1d()` which resolves to a host-only stub outside `#[kernel]` context).
**Trait: KvCacheFormat**: Defines `read_kv(pool: &[u16], offset: usize) -> f32` for reading KV cache values with format-specific dequantization. Current implementation: `KvBf16` reads bf16 directly via `(pool[offset] as u32) << 16`. Designed to support future FP8 cache variants (e.g., `KvFp8E4M3`).

| Kernel | Source File | Description | Test Result |
|--------|-------------|-------------|-------------|
| `infers_fp8_quantize_e4m3` | `fp8_quantize.cu` | BF16→FP8 E4M3 quantization | ✅ 256 elements bit-exact vs CPU reference |
| `infers_fp8_dequantize_e4m3` | `fp8_quantize.cu` | FP8 E4M3→BF16 dequantization | ✅ Round-trip rel error < 0.25 |
| `infers_fp8_quantize_e5m2` | `fp8_quantize.cu` | BF16→FP8 E5M2 quantization | ✅ 256 elements bit-exact vs CPU reference |
| `infers_fp8_dequantize_e5m2` | `fp8_quantize.cu` | FP8 E5M2→BF16 dequantization | ✅ Round-trip rel error < 0.5 |
| `infers_paged_attention_decode_bf16` | `paged_attention_decode.cu` | Two-pass attention: online softmax + weighted V accumulation, one block per KV head, GQA support via q_per_kv loop | ✅ 2 KV heads, 4 query heads, 8 cached tokens vs CPU reference (tolerance 2.0) |

**Paged attention decode algorithm**: Two-pass approach using dynamic shared memory (`3 * bdim * sizeof(f32)`):
1. **Phase 1 — Online softmax**: Each thread processes strided tokens, computing Q·K dot products with per-thread online softmax (tracking local_max and local_sum via incremental update). Block reduction: global max via fmax halving, then adjusted sum reduction.
2. **Phase 2 — Weighted V accumulation**: Threads with `tid < head_dim` loop over ALL tokens, recomputing Q·K dot products, applying softmax weights, and accumulating weighted V values.

**Launch configuration**: Grid = `num_kv_heads` blocks (one per KV head), block = `min(head_dim, 256)` threads. Dynamic shared memory: `3 * bdim * sizeof(f32)`. For GQA, each block iterates over `q_per_kv = num_query_heads / num_kv_heads` query heads.

**Key insight — thread::index_1d() limitation**: Inside `#[inline(always)]` helper functions that are inlined into kernels, `thread::index_1d()` resolves to a host-only stub. Must compute index manually: `(thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize`.

### cuda-oxide Kernel Library: Tier 5 — GDN Kernels (Phase 18 — 4 Kernels Ported)

Four GDN (Gated DeltaNet) kernels ported from nvcc to Rust in cuda-oxide-kernel-lib, covering the core SSM recurrence patterns used by Qwen3.6-27B inference. All compute in f32 with bf16 I/O for precision.

**Shared math patterns**: Each kernel implements softplus (`log(1 + exp(x))` with clamping at ±20), sigmoid (`1/(1+exp(-x))`), and bf16↔f32 conversion via `(bits as u32) << 16`. These are computed inline in each kernel rather than as shared helpers — cuda-oxide's `#[cuda_module]` does not support cross-scope device functions outside the module.

| Kernel | Source File | Description | Test Result |
|--------|-------------|-------------|-------------|
| `infers_gdn_recurrent_step_bf16` | `gdn_recurrent_step.cu` | L2-norm Q/K, softplus decay, sigmoid beta, 5-step recurrence (decay state, kv_mem, delta, update, output) — f32 state, bf16 I/O | ✅ H=2, K=4, V=4 bit-exact vs CPU reference |
| `infers_gdn_mamba2_update_bf16` | `gdn_mamba2_update.cu` | Sigmoid decay, softplus delta (threshold 2.0), state update, SiLU gating — bf16 state and I/O | ✅ H=2, head_dim=4 bit-exact vs CPU reference |
| `infers_gdn_update_bf16` | `gdn_update.cu` | Shared memory reductions: Phase 1 beta = dot(state_row, b), Phase 2 state update with dt*a*beta, Phase 3 output = dot(updated_state_row, a) | ✅ hidden_size=8 tolerance 0.5 vs CPU reference |
| `infers_gdn_gated_delta_update_bf16` | `gdn_gated_delta_update.cu` | Same algorithm as recurrent_step (L2-norm, softplus decay, sigmoid beta, 5-step recurrence) — f32 state, bf16 I/O | ✅ H=2, K=4, V=4 bit-exact vs CPU reference |

**Recurrent step and gated delta update**: One thread per (head, v_dim) element. No shared memory. Algorithm: L2-normalize query and key, compute decay via `exp(-exp(A_log[h]) * softplus(a_proj[h] + dt_bias[h]))`, compute beta via `sigmoid(b_proj[h])`. Five sequential steps: (1) multiply state by decay, (2) accumulate kv_mem = sum of state × normalized key, (3) delta = beta × (value - kv_mem), (4) update state += normalized_key × delta, (5) output = sum of updated state × normalized query × 1/sqrt(K).

**GDN update**: One block per state row using dynamic shared memory (256 × 4 bytes for reduction buffer). Three phases with halving reductions: Phase 1 computes beta via dot product reduction, Phase 2 updates state row, Phase 3 computes output via dot product reduction.

**Mamba2 update**: One thread per total_dim element. Sigmoid-based decay, softplus delta (with threshold at 2.0 instead of 20.0), SiLU gating for output computation. All bf16 storage including state.
