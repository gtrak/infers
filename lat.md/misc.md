# Tokenizer

HF `tokenizers` crate wrapper for encoding prompts into token IDs and decoding IDs back to text.

`infers-tokenizer` wraps `tokenizers::Tokenizer` with `anyhow::Result` error handling. The crate provides a single public `Tokenizer` struct with two constructors (`from_file` and `from_pretrained`) and three core methods (`encode`, `decode`, `vocab_size`). It depends on `tokenizers 0.21` with `onig` and `http` features enabled.

## Tokenizer

Wrapper around `tokenizers::Tokenizer` with `anyhow::Result` error handling. See [[crates/tokenizer/src/lib.rs#Tokenizer]].

`Tokenizer` holds a single `inner` field (`tokenizers::Tokenizer`). Implements `Clone`. `from_file(path)` loads from a local `tokenizer.json`. `from_pretrained(model_id)` downloads from HuggingFace Hub. `encode(text)` returns `Vec<u32>` of token IDs with add_special_tokens=true. `decode(tokens)` returns decoded string with skip_special_tokens=false. `vocab_size()` returns vocabulary size with add_special_tokens=true.

## Error Handling

All public methods convert `tokenizers` errors (which return `Box<dyn Error + Send + Sync>`) into `anyhow::Error` via `map_err(|e| anyhow::Error::msg(e.to_string()))`, then attach context using anyhow's `with_context`. See [[crates/tokenizer/src/lib.rs#Tokenizer#from_file]], [[crates/tokenizer/src/lib.rs#Tokenizer#from_pretrained]], [[crates/tokenizer/src/lib.rs#Tokenizer#encode]], [[crates/tokenizer/src/lib.rs#Tokenizer#decode]].

# MTP Engine

MTP speculative decoding engine coordinating draft generation, verification, and adaptive token counts.

## MtpEngine

Orchestrates the full MTP speculative decoding lifecycle: draft generation, verification, acceptance, and adaptive draft count.

`MtpEngine` wraps an `MtpHead` and manages the speculative decoding flow. It tracks acceptance history for adaptive draft count management. Fields: `mtp_head` (MTP prediction head), `num_draft_tokens` (1-4), `acceptance_history` (recent results), `rms_norm_eps`, `hidden_size`. Methods: `new()` constructs from weights/config, `generate_drafts()` iteratively runs MTP head forward + LM head + sampling, `verify_drafts()` runs full model forward for each draft and compares predictions, `accept_prefix()` returns accepted tokens plus correction, `adaptive_num_drafts()` adjusts count based on rolling 10-step acceptance rate (>80% increase, <30% decrease). See [[crates/mtp/src/engine.rs#MtpEngine]].

## MtpOperations

Callback bundle for GPU operations required by the MTP engine.

`MtpOperations` bundles callbacks for embedding lookup, RMSNorm, layer forward, LM head projection, greedy sampling, and full model forward. By passing callbacks instead of direct dependencies, the MTP crate avoids coupling to the backend crate's kernel dispatch and CUDA resource management. See [[crates/mtp/src/engine.rs#MtpOperations]].

## MtpHead

Single-layer transformer head that predicts the next token from the main model's hidden state.

`MtpHead` stores GPU-resident weight buffers for the MTP prediction head (norms, FC projections, decoder layer, final norm). The `forward()` method takes callbacks for embedding lookup, RMSNorm, and decoder layer execution. Architecture: normalize embedding and hidden state, project via FC layers, add element-wise, run decoder layer, final norm. See [[crates/mtp/src/head.rs#MtpHead]].

# MTP Metrics

Lightweight metrics collection for tracking MTP speculative decoding performance.

## MtpMetrics

Collects running statistics on draft token acceptance rate, tokens saved, and provides helpers for speedup estimation.

`MtpMetrics` tracks `total_drafts`, `total_accepted`, `verification_steps`, and `rate_sum` (for rolling average). `record_step()` updates counters per verification step. `acceptance_rate()` returns overall acceptance (accepted/drafts), `average_step_rate()` returns rolling mean of per-step rates. `estimated_speedup()` computes rough speedup factor: `1 / (1 - r + r/k)` where `r` is acceptance rate and `k` is draft count. `tokens_saved()` returns total accepted tokens. `reset()` zeroes all counters. See [[crates/mtp/src/metrics.rs#MtpMetrics]].

# FP8 Quantization

CPU reference implementations for FP8 (E4M3 and E5M2) quantization and dequantization of BF16 data.

The `quant` module provides CPU-based reference implementations for converting between `half::bf16` slices and `u8` slices (raw FP8 bytes). E4M3 uses 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits with range ±240. E5M2 uses 1 sign bit, 5 exponent bits (bias 15), 2 mantissa bits with range ±57344. Overflow clamps to max finite; underflow maps to zero. Production paths will use CUDA kernel implementations.

## E4M3 Format

FP8 E4M3 encoding: 1 sign, 4 exponent (bias 7), 3 mantissa bits.

Max finite value is exp=14, mant=7 → 240.0. Values beyond this range are clamped to the max finite (0x77 positive, 0xF7 negative). NaN is encoded as exp=0xF with non-zero mantissa (0x7F positive). See [[crates/kv/src/quant.rs#f32_to_fp8_e4m3]], [[crates/kv/src/quant.rs#fp8_e4m3_to_f32]].

## E5M2 Format

FP8 E5M2 encoding: 1 sign, 5 exponent (bias 15), 2 mantissa bits.

Max finite value is exp=30, mant=3 → 57344.0. Overflow clamps to 0x7B (positive) or 0xFB (negative). NaN encodes as exp=0x1F with non-zero mantissa (0x7F positive, 0xFF negative). See [[crates/kv/src/quant.rs#f32_to_fp8_e5m2]], [[crates/kv/src/quant.rs#fp8_e5m2_to_f32]].

## Public API

Four public functions provide quantize/dequantize pairs for both formats.

| Function | Description |
|----------|-------------|
| `quantize_fp8_e4m3` | Convert `&[bf16]` to `Vec<u8>` (E4M3 bytes) |
| `dequantize_fp8_e4m3` | Convert `&[u8]` (E4M3) back to `Vec<bf16>` |
| `quantize_fp8_e5m2` | Convert `&[bf16]` to `Vec<u8>` (E5M2 bytes) |
| `dequantize_fp8_e5m2` | Convert `&[u8]` (E5M2) back to `Vec<bf16>` |

See [[crates/kv/src/quant.rs#quantize_fp8_e4m3]], [[crates/kv/src/quant.rs#dequantize_fp8_e4m3]], [[crates/kv/src/quant.rs#quantize_fp8_e5m2]], [[crates/kv/src/quant.rs#dequantize_fp8_e5m2]].

## Delegation Pattern

FP8 helpers live in `infers-kv::quant` and are re-exported by `infers-backend-native::quant` to eliminate duplication. `KvCacheDtype` and `QuantizedKvCache` are additionally re-exported from the `infers-kv` crate root. See [[crates/kv/src/lib.rs]], [[crates/backends/native/src/quant.rs]].

# Tech Debt Fixes

Production hardening changes applied to improve error handling, safety, and code quality.

## Error Handling

Critical `.unwrap()` calls in `main.rs` replaced with `?` propagation. The `run()` function returns `anyhow::Result<()>` with proper context on bind and serve failures.

## Metrics Handler

`metrics_handler()` now returns `Result<impl IntoResponse, StatusCode>` instead of `impl IntoResponse`. Encoding errors and UTF-8 conversion errors return `INTERNAL_SERVER_ERROR` instead of panicking. See [[crates/metrics/src/lib.rs#metrics_handler]].

## Metrics Registry

Metric creation `.unwrap()` calls changed to `.expect()` with descriptive error messages. This makes it clear that metric creation failures indicate duplicate metric names or registry errors, not normal failures. See [[crates/metrics/src/registry.rs]].

## Safety Comments

`memmap2::Mmap::map()` calls have `// SAFETY:` comments explaining that the file is opened read-only, the file handle is verified to exist before mapping, and the mapping is read-only weight data. See [[crates/model-loader-heap/src/lib.rs#load_single]], [[crates/model-loader-heap/src/lib.rs#load_sharded]].

## Memory Budget Improvements

`MemoryBudget` now includes `max_position_embeddings` from the model config instead of hardcoding `262144`. The workspace size is defined as `DEFAULT_WORKSPACE_BYTES` constant. The `max_position_tokens()` helper method was removed. See [[crates/model/src/budget.rs#MemoryBudget]].

## Config Constants

`FULL_ATTENTION_INTERVAL` constant replaces the hardcoded `4` in `default_layer_type()`. See [[crates/model/src/config.rs#ModelConfig#default_layer_type]].

## Weight Registry Safety

`WeightRegistry` now uses `Option<WeightData>` for `embedding`, `lm_head`, and `norm` fields instead of empty placeholder `WeightData` structs. See [[crates/model/src/weights.rs#WeightRegistry]].

## CPU Weight Data Clearing after GPU Upload

`WeightRegistry::clear_data()` frees ~5 GB per GPU of persistent heap residency by dropping CPU-side weight data after GPU upload. See [[crates/model/src/weights.rs#WeightRegistry#clear_data]], [[crates/backends/native/src/engine.rs#ForwardEngine#new]].

### malloc_trim after Weight Clearing

After `clear_data()` (heap path) and `clear_owned_data()` (mmap path), `trim_memory()` calls `malloc_trim(0)` to force glibc to return freed memory to the OS, reducing VmData. On Linux only — no-op on other platforms. See [[crates/backends/native/src/engine.rs#trim_memory]].

## GpuAllocator Encapsulation

`GpuAllocator` fields are now private with accessor methods. The `free()` method has overflow protection and derives `Debug` and `Clone`.

## Build Script Safety

`build.rs` uses `parent().unwrap_or(Path::new("."))` instead of `parent().unwrap()` for output path directory creation. See [[crates/cuda/build.rs]].

## GEMM Leading Dimension Fix

`GemmConfig` to `MatmulConfig` conversion in `gemm_impl` had incorrect column-major leading dimension defaults. Fixed `lda` branches (transa=true→k, transa=false→m) and `ldc` default (m, not n) to match cuBLASLt column-major convention. See [[crates/cuda/src/gemm.rs#gemm_impl]].

## cuBLAS Column-Major Output Fix in Attention GEMMs

cuBLAS writes GEMM output in column-major order, but downstream code reads buffers as row-major. The bug was already fixed for projection GEMMs but persisted in per-head attention loop GEMMs.

**Scores GEMM**: `Q @ K^T = scores[i,j]` — cuBLAS output in column-major means softmax reads transposed scores (`Q[j]·K[i]` instead of `Q[m]·K[n]`). Fix: swap `q_h` and `k_h` arguments; dot product is commutative so `K(n)·Q(m) = Q(m)·K(n)` after the column-to-row-major read.
**Attention output GEMM**: `softmax @ V` — cuBLAS output in column-major means downstream reads transposed. Fix: swap m/n dimensions, change `transa`/`transb` to false, and swap `v_h` with `softmax_out_h`. Computes `V^T @ S^T = [head_dim × seq_len]` in column-major; reading as row-major `[seq_len × head_dim]` gives correct layout because offset `m*head_dim + k = k + m*head_dim` matches column-major offset `k + m*head_dim`.

Applied in 4 GEMM calls across `forward()` and `forward_paged()` prefill paths. The decode path uses `m=1` so column-major and row-major are identical for single-row output — correct by coincidence. After applying, prefill attention verification improved from cos=0.008 to cos>0.9996 across all 34 stages (attention + MLP) on both GPUs. See [[crates/backends/native/src/attention.rs#forward]], [[crates/backends/native/src/attention.rs#forward_paged]].

The scores GEMM swap (`q_h`↔`k_h`) was NOT applied because scores already match reference (cos=0.993) — the transpose produces `Q[j]·K[i]` but the softmax kernel reads row-by-row, effectively computing the same causal attention. The attention output GEMM fix (swap m/n, transa→false, transb→false, swap A/B) WAS the critical fix.

## Reference Comparison Bugs Found

The Python reference comparison framework (`tests/compare/`) had two bugs that masked the GEMM fix verification.

**Einsum bug** in `_scaled_dot_product_attention()`: used `"sah,sbh->sab"` which computes head-to-head dot products at each position, not per-head position-to-position attention. Fixed to `"sah,tah->sat"` for scores and `"sat,tad->sad"` for V multiplication. Causal mask needed `.unsqueeze(1)` for broadcast with `[S,H,S]` shape.
**Missing residual** in `Norm2Stage.compute()`: computed `rms_norm(attn.after_ar)` instead of `rms_norm(hidden_input + attn.after_ar)`. The residual connection was missing, causing cos=0.982 and max_diff=13.16 for norm2. After fix, cos=1.000021. See [[tests/compare/stages/attention.py#_scaled_dot_product_attention]], [[tests/compare/stages/mlp.py#Norm2Stage]].

## Paged Attention Decode Kernel Divergence

The paged attention decode kernel produces output with cos≈0.97 against PyTorch reference, causing garbage model output. Root cause under investigation.

The kernel `infers_paged_attention_decode_bf16` computes attention for one query token against all cached K/V tokens. When the kernel algorithm is simulated in Python with the same inputs (K/V from dumps, Q from dumps), the result matches PyTorch reference exactly (cos=1.0). But the engine's actual kernel output diverges (cos=0.97). This points to the KV cache contents differing from the dumped K/V values, possibly due to a write-ordering or layout issue in the paged KV cache.

## SSE Constant Usage

Chat handler uses `SSE_DONE` constant from `infers_api` instead of hardcoded `"[DONE]"` string. See [[crates/server/src/handlers/chat.rs]].

## Kernel Registry Documentation

`LoadedKernelRegistry` documents its role in GPU kernel execution, loading pre-compiled .cubin files via cudarc.

## Kernel Dispatch Fixes

Six critical kernel dispatch bugs were fixed to align Rust dispatch code with CUDA kernel signatures.

### RMSNorm Shared Memory

`infers_rmsnorm_bf16` uses `extern __shared__ float shared_mem[]` for a block-wide reduction with 256 threads. Shared memory was set to 0, causing undefined behavior. Fixed to `256 * sizeof(f32) = 1024` bytes. See [[crates/backends/native/src/norm.rs#rms_norm]].

### Embedding Argument Order

Kernel dispatch passed arguments in wrong order and included an extra parameter. Fixed to match CUDA signature `infers_embedding_gather_bf16(weight, token_ids, output, seq_len, hidden_size)`. See [[crates/backends/native/src/embedding.rs#embed_tokens]].

### Argmax Missing Batch Size

`infers_argmax_f32` expects `(logits, output, batch_size, vocab_size)` but the dispatch code only passed 3 args, leaving `batch_size` unfilled (vocab_size was passed into the batch_size slot). Added `batch_size_i32 = 1` argument. See [[crates/backends/native/src/sample.rs#greedy_sample]].

### GEMM Transpose Flags

GEMM transpose flags were inverted for row-major weight storage. All three `matmul_bf16` calls in `mlp_forward` changed from `transa: false, transb: true` to `transa: true, transb: false` so cuBLASLt correctly interprets row-major inputs. See [[crates/backends/native/src/mlp.rs#mlp_forward]].

### RoPE Table Indexing

`precompute_rope_tables` allocated a compact table indexed by token index, but the CUDA kernel indexed by position value. Fixed to size table by max position and index by position. See [[crates/backends/native/src/rope.rs#precompute_rope_tables]].

### RoPE Multi-Head Rotation

RoPE kernel only rotated the first head, ignoring the `num_heads` dimension in tensor layout. Fixed by adding `num_heads` parameter and iterating all head-dimension pairs. See [[crates/backends/native/src/rope.rs#apply_rope]].

### Paged Attention Decode Kernel Fixes

Two critical bugs fixed in `infers_paged_attention_decode_bf16` in `paged_attention_decode.cu`:
1. **Strided token accumulation dropped ~99% of contributions** (CRITICAL): Phase 2 loop used `token_pos += bdim` stride, so each thread only accumulated V values for tokens at positions tid, tid+bdim, tid+2*bdim... — one thread processed one output dimension but only saw 1/bdim of the tokens. Fixed by having each thread (tid < head_dim) loop over ALL tokens independently.
2. **Block reduction summed across different output dimensions** (CRITICAL): The reduction loop added partial outputs for dimension 0, dimension 1, ..., dimension bdim-1 together into a single scalar. Fixed by removing the reduction entirely — each thread writes directly to `output[head_idx * head_dim + tid]`.

### Softmax Kernel Fixes

Three bugs fixed in `crates/cuda/kernels/infers/softmax.cu`:
1. **Max value overwritten by sum** (CRITICAL): Phase 2's sum reduction overwrote `sdata[0]`, causing Phase 3 normalization to subtract the sum instead of the max. Fixed by saving `sdata[0]` to a `max_val` register after Phase 1, then using `max_val` in Phase 2 (exp computation) and Phase 3 (normalization).
2. **Race condition** (implicit in bug 1): Reading `sdata[0]` for the max during Phase 2 created a race with concurrent sum reduction writes. Resolved by using the `max_val` register instead of shared memory.
3. **Non-power-of-2 reduction** (MINOR): Tree reduction with `stride >>= 1` silently drops tail elements for non-power-of-2 block sizes. Fixed by rounding `block_size` up to the next power of 2 and adding `tid + stride < blockDim.x` bounds checks in both reduction loops.

### GDN Kernel Fixes

Two bugs fixed in both `gdn_update.cu` and `gdn_prefill.cu`:
1. **Non-power-of-2 reduction** (MINOR): Host wrappers used `hidden_size` directly as block size, causing tree reduction to silently drop tail elements for non-power-of-2 hidden sizes. Fixed by restructuring kernels: `__global__` kernels are now placed inside `extern "C"` with names matching the registry (`infers_gdn_update_bf16`, `infers_gdn_prefill_bf16`), eliminating host wrappers. Rust-side launch configuration computes power-of-2 block sizes. Reduction loops use `tid + stride < blockDim.x` bounds checks.
2. **Kernel name mismatch** (CRITICAL): `LoadedKernelRegistry` registered `infers_gdn_update_bf16` and `infers_gdn_prefill_bf16`, but the cubins only contained C++-mangled `__global__` kernels (`gdn_update_kernel`, `gdn_prefill_kernel`) and host wrappers (not compiled into cubins). Fixed by renaming `__global__` kernels to match registered names and wrapping them in `extern "C"` for unambiguous C linkage in the cubin.

### GDN QKV Split Bug Fix

`clone_view_to_slice` did a contiguous flat copy instead of per-row column extraction from row-major `conv_out`, causing catastrophic query/key/value cosine similarity. Fixed by `extract_columns()` with per-row strided copies.

For example at TP=2 with seq_len=15 and conv_dim=5120:
- Range `0..15*1024 = 0..15360` copies the first 15360 flat elements = all of row 0 (5120) + all of row 1 (5120) + first 5120 of row 2
- NOT the first 1024 columns of each row, which requires per-row strided copies

See [[crates/backends/native/src/gdn.rs#extract_columns]].

### Attention Q Extraction and Q-norm Fixes

Three bugs in `forward_paged` and `decode_forward_paged` when `attn_output_gate` is true:
1. **Per-head Q extraction used wrong row stride** (CRITICAL): The per-head loop extracted Q with `src_offset = s * per_gpu_head_dim + head_idx * head_dim`, but `q_full` has width `q_out_dim = per_gpu_head_dim * 2`. For token s=1, this read from offset `1*3072 + head_idx*256` — the gate portion of token 0, not the Q portion of token 1. Fixed to `s * q_out_dim + head_idx * head_dim`.
2. **Q-norm applied to entire buffer including gate**: The RMSNorm was applied to the full `q_full` buffer `[seq, heads*dim*2]`, normalizing both Q and gate values. The HuggingFace reference (`self.q_norm(q)`) only normalizes the Q portion. Fixed by extracting just the Q portion (`[seq, heads*dim]`), applying RMSNorm, and writing back.
3. **q_full extraction assumed contiguous layout** (CRITICAL): The q_proj GEMM output uses per-head interleaved layout `[Q_h0(256), G_h0(256), Q_h1(256), G_h1(256), ...]` per row (matching HuggingFace's `repeat_interleave` semantics). The engine assumed contiguous `[Q_all(3072), G_all(3072)]`. This caused Q-norm extraction, gate extraction, and per-head Q extraction in the attention loop to read from wrong offsets. Fixed by using per-head interleaved extraction with stride `head_dim * 2` for all three operations: (a) Q-norm copies Q from `h * (head_dim * 2)` instead of first half of row; (b) gate copies G from `h * (head_dim * 2) + head_dim`; (c) per-head attention loop uses `head_stride = head_dim * 2` for Q offset calculation. The same fix was applied to `decode_forward_paged` (m=1, no seq_len loop).

See [[crates/backends/native/src/attention.rs#forward_paged]], [[crates/backends/native/src/attention.rs#decode_forward_paged]].

## Dead Code Cleanup

Post-Phase 10 cleanup removed dead code and annotated deferred-feature fields with `#[allow(dead_code)]` to eliminate all compiler warnings in the server crate.

### Chat Handler Dead Code

`generate_id()` function removed from `chat_completions` handler. This function was used by removed mock functions. The handler now uses `routing_id` from the orchestrator for IDs. See [[crates/server/src/handlers/chat.rs]].

### Orchestrator Deferred Fields

Five deferred-feature fields and three monitoring methods in `InferenceOrchestrator` are annotated with `#[allow(dead_code)]`. Paged eviction and MTP speculative decoding support is deferred. See [[crates/server/src/orchestrator.rs#InferenceOrchestrator]].

### Native Crate Warning Fixes

Fixed all compiler warnings in `infers-backend-native` crate (TD-5).

Unused `stream` parameters in `prefill_paged` and `decode_paged` prefixed with `_`. The `paged_kv_read` field in `PerGpuKernels` annotated with `#[allow(dead_code)]` — kernel loaded but unused, retained for potential future use. `a_log_zeros`/`dt_bias_zeros` double-assignment warnings suppressed via `#[allow(unused_assignments)]` on both `forward` and `decode_forward` functions; the Option<None> initialization is required because zero-allocated buffers must live at function scope to satisfy borrow checker constraints across if/else branches. See [[crates/backends/native/src/engine.rs#PerGpuKernels]], [[crates/backends/native/src/gdn.rs#forward]], [[crates/backends/native/src/gdn.rs#decode_forward]].

# Phase 1 Deliverables

Phase 1 (Bootstrap) creates the workspace, crate skeletons, and API scaffolding for the inference server.

- Workspace Cargo.toml with 12 crate members and pinned dependencies
- rust-toolchain.toml for nightly-2026-04-03
- OpenAI-compatible API types (request, response, streaming, error)
- Axum HTTP server with mock responses
- SSE streaming scaffold for chat completions
- Prometheus metrics endpoint at /metrics
- Health check endpoint at /health
- CLI argument parsing with clap

# Phase 2 Deliverables

Phase 2 (CUDA Backend) establishes the GPU runtime, kernel compilation pipeline, and multi-GPU communication primitives.

- CUDA crate (`infers-cuda`) with cudarc always present — no feature gating
- CudaRuntime for multi-GPU device context management (cudarc CudaContext)
- StreamPool for async CUDA stream management per device
- KernelRegistry for .cubin loading (20 infers kernels: rmsnorm, silu, silu_glu, rope, embedding_gather, add, argmax_bf16, softmax, kv_cache, paged_kv_write, paged_kv_read, gdn_recurrent_step, gdn_gated_delta_prefill, gdn_chunked_gated_delta_prefill, gdn_mamba2_prefill, gdn_mamba2_update, paged_attention_decode, fp8_quantize, fp8_dequantize, int4_gemm) and LoadedKernelRegistry for GPU-loaded kernels with deduplication (same .cubin loaded once even when referenced by multiple kernel functions)
- GemmEngine wrapping cuBLASLt with BF16 support; `new(stream)` creates CudaBlasLT eagerly, `matmul_bf16()` accepts `GemmConfig` and `CudaSlice` buffers; `matmul_int4()` accepts `Int4GemmConfig` for INT4-packed weight GEMM with per-group dequantization (group_size=128, FP32 accumulation, BF16 output)
- NcclCommunicator wrapping cudarc NCCL Comm with `all_reduce()`, `all_reduce_in_place()`, `broadcast()`, `reduce()`, `all_gather()`, `send()`, `recv()` methods for TP/PP collectives and P2P hidden state transfer across multiple GPUs
- build.rs for nvcc kernel compilation (default sm_120, configurable via INFERS_CUDA_ARCH env var)
- CUDA kernel source files in `kernels/infers/`: rmsnorm.cu, silu.cu, rope.cu, embedding.cu, elementwise.cu, sampling.cu, softmax.cu, kv_cache.cu, paged_kv_write.cu, paged_kv_read.cu, gdn_update.cu, gdn_prefill.cu, gdn_recurrent_step.cu, gdn_gated_delta_prefill.cu, gdn_chunked_gated_delta_prefill.cu, gdn_mamba2_prefill.cu, gdn_mamba2_update.cu, paged_attention_decode.cu, fp8_quantize.cu, int4_gemm.cu, common.cuh
- Kernel directory structure (flashinfer-gdn, flashinfer-attn, compiled) preserved for organization; custom kernels use infers/
- Kernel fixes: softmax max preservation (register variable), power-of-2 block rounding, attention GEMM transb correction, accumulation parity fix

# Phase 3 Deliverables

Phase 3 (Model Loading) implements multi-format model loading with auto-detection, weight registry, TP sharding, and memory budgeting.

- Multi-format model loader (`infers-model`) with safetensors support
- ModelConfig struct for parsing `config.json` with hybrid attention layer types
- QuantizationFormat auto-detection (GGUF, PrismaSCOUT, AutoRound, BF16)
- WeightRegistry and WeightData for storing tensors as raw bytes
- SafetensorsLoader with single-file and sharded index support
- Weight sharding for TP=2 (column/row parallel) and PP=2 (layer split)
- MemoryBudget calculator for VRAM estimation across quantization formats
- LayerType enum with default pattern (every 4th layer full attention, rest GDN)
- QuantizationConfig parsing from `quantization_config.json` or embedded config
- `strip_language_model_prefix()` strips `model.language_model.` prefix from tensor names and removes `model.visual.*` tensors for Qwen3.6 multimodal models
- `build_main_layers()` populates `WeightRegistry.layers`, `embedding`, `norm`, `lm_head` from the flat tensor map using `HashMap::remove()` (not clone) to halve memory usage
- `build_mtp_weights()` populates `WeightRegistry.mtp` from MTP tensor map
- `get_weight()` helper uses `remove()` to transfer ownership from flat map to structured fields, avoiding expensive clones of large tensors

# Phase 9 Deliverables

Phase 9 (Tool Calls + Final Polish) adds Qwen3.6 chat template formatting, tool call parsing, tool call streaming in SSE format, and the `enable_auto_tool_choice` API parameter.

## Chat Template

Qwen3.6 chat template for formatting messages with thinking tokens and tool calls using `</think>` / `<![CDATA[<thinking>` tokens.

`QwenChatTemplate` holds `enable_thinking` and `preserve_thinking` flags. `apply()` formats system, user, assistant, and tool messages into a prompt string. Tools are formatted as `<tools>` XML blocks in a prepended system message. Assistant reasoning content is wrapped in `<thinking>` tags. Tool calls use `<tool_call>` XML blocks. Tool responses use `</think>` wrappers. See [[crates/api/src/template.rs#QwenChatTemplate]].

**`Message` reasoning_content**: The `request::Message` type now supports `reasoning_content` for assistant messages containing thinking/reasoning tokens. `response::MessageContent` and `streaming::Delta` also carry `reasoning_content` for output. See [[crates/api/src/request.rs#Message]], [[crates/api/src/response.rs#MessageContent]], [[crates/api/src/streaming.rs#Delta]].

## Tool Call Parser

Parser for Qwen3.6 XML-format tool calls (`<tool_call>...</tool_call>` blocks) supporting both streaming and complete text modes.

`ToolCallParser` provides `parse_streaming_delta()` for incremental text accumulation with `PartialToolCall` state tracking, and `parse_complete()` for processing full response text. Handles multiple JSON shapes: full `ToolCall` JSON, function-only `{"name", "arguments"}`, and fallback wrappers. See [[crates/api/src/tool_parser.rs#ToolCallParser]].

## Tool Call Streaming

The chat handler now produces OpenAI-compatible tool call responses when `tools` are provided in the request.

`create_mock_tool_call_response()` returns a non-streaming response with `tool_calls` in the message and `finish_reason: "tool_calls"`. `create_mock_tool_call_stream()` emits SSE deltas following the OpenAI protocol: role delta, tool call name/ID, incremental argument chunks, finish reason, and `[DONE]`. See [[crates/server/src/handlers/chat.rs#chat_completions]].

## API Parameters

`ChatCompletionRequest` gained `enable_auto_tool_choice` (boolean, default `false`) for automatic tool choice when tools are present. `streaming::ToolCallDelta` gained `id` field for the OpenAI streaming tool call identifier. See [[crates/api/src/request.rs#ChatCompletionRequest]], [[crates/api/src/streaming.rs#ToolCallDelta]].

## Tool Call Response Schema

Non-streaming tool responses set `content` null and populate `tool_calls`. Streaming emits deltas with `index`, `id`, `type`, and partial `function` arguments. See `plan/research/api.md#Tool Calls` for the full schema.

# Phase 10 Deliverables

Phase 10 (Server Wiring) connects all crates into a working inference server by initializing CUDA runtime, model loading, KV management, scheduler, engine, tokenizer, and the background scheduler loop.

## Server Initialization

`main.rs` wires all crates into a single `run()` function that creates the full inference pipeline.

The initialization sequence: (1) parse CLI args, (2) initialize CUDA runtime via `CudaRuntime::new()`, (3) create `StreamPool` with one stream per device, (4) load model config and weights from disk (or create defaults if path missing), (5) register and load CUDA kernels, (6) create `ForwardEngine`, (7) create `PagedKvManager`, (8) create `RoundRobinScheduler`, (9) create `BackendEvictionStore`, (10) create `InferenceOrchestrator`, (11) create `Tokenizer`, (12) build `AppState` with orchestrator and tokenizer, (13) spawn background scheduler loop, (14) start HTTP server.

### CUDA Runtime

`CudaRuntime::new()` enumerates all CUDA devices and creates contexts. `device(0)` returns `&Arc<CudaContext>` for the first GPU. See [[crates/cuda/src/context.rs#CudaRuntime]].

### Stream Pool

`StreamPool::new()` creates one async CUDA stream per context. `get(0)` returns the first stream for orchestrator use. See [[crates/cuda/src/stream.rs#StreamPool]].

### Model Loading

If the model path exists on disk, `load_model()` reads `config.json` and safetensors files, returning `(ModelConfig, WeightRegistry)`. If the path is missing, a default Qwen3.6-27B config is created with empty weights for wiring validation. See [[crates/model-loader-heap/src/lib.rs#load_model]].

### Kernel Registration

`KernelRegistry::register_infers_kernels()` registers all 15 infers kernels by name and cubin path. `ForwardEngine::new()` loads them into GPU memory via `LoadedKernelRegistry::load_all()`. See [[crates/cuda/src/kernels.rs#KernelRegistry#register_infers_kernels]].

### AppState Evolution

`AppState` now holds three fields: `model_name`, `orchestrator` (`Arc<Mutex<InferenceOrchestrator>>`), and `tokenizer` (`Tokenizer`). The `SharedState` type alias is `Arc<AppState>`. See [[crates/server/src/state.rs#AppState]].

### Scheduler Loop

`server::spawn_scheduler_loop()` spawns a background task that continuously calls `orchestrator.step()`.

`server::spawn_scheduler_loop()` spawns a tokio task that continuously calls `orchestrator.step()` with a 1ms sleep between iterations. Each iteration locks the orchestrator, runs scheduling (admit, cleanup, eviction, prefill, decode), then releases the lock. Errors are logged with `tracing::error`. See [[crates/server/src/server.rs#spawn_scheduler_loop]].

## Default Model Config

When the model path doesn't exist, `main.rs` creates a hardcoded default `ModelConfig` for wiring validation.

When the model path doesn't exist, `main.rs` creates a hardcoded default `ModelConfig` for Qwen3.6-27B wiring validation: 48 layers, 5120 hidden size, 13888 intermediate size, 152064 vocab, 40 attention heads, 40 KV heads, 128 head dim, 262144 max position embeddings, silu activation, rope_theta=10000000, partial_rotary=0.25, mRoPE interleaved, no MTP, linear_num_value_heads=48, linear_value_head_dim=128. Empty `WeightRegistry` is used — engine wiring succeeds without actual weights since kernels won't be dispatched.
