# Rust Inference Server for Qwen3.6-27B
## Product Requirements Document (PRD) + Implementation Specification

Version: 0.2
Target Audience: Single engineer
Primary Language: Rust
Target Hardware: 2x NVIDIA RTX 5060 Ti
Model Family: Qwen3.6-27B
Scope: Personal inference server optimized for agent workloads

---

# 1. Goals

Build a high-performance inference server focused on:

- Qwen3.6-27B support
- Tensor parallelism across 2 GPUs
- Pipeline parallelism with microbatching across 2 GPUs
- Hybrid KV cache (Mamba state + paged)
- Continuous batching
- Native Multi-Token Prediction (MTP)
- OpenAI-compatible API with tool calls
- Support for PrismaSCOUT (NVFP4), AutoRound (INT4), GGUF, and BF16 weights
- KV cache quantization (BF16, FP8, NVFP4)
- Long-lived agent sessions
- Predictable latency
- Minimal operational complexity

Non-goals:

- Multi-tenant hosting
- Kubernetes
- Dynamic model loading
- LoRA support
- Multi-model serving
- Distributed clusters
- Vision/text-to-image (text-only serving)

---

# 2. Success Metrics

## Functional

- Load Qwen3.6-27B
- Generate tokens correctly
- Support concurrent sessions
- Support native MTP
- Support tool calls
- Support OpenAI-compatible chat completions API
- Support multiple quantization formats
- Survive multi-hour agent runs

## Performance

- >= 90% of vLLM throughput
- <= 10% memory overhead vs vLLM
- Stable operation for 24 hours

---

# 3. Architecture

Layers:

1. API Layer (OpenAI-compatible)
2. Scheduler
3. Batch Builder
4. KV Manager (hybrid: Mamba state + paged)
5. Model Runner
6. CUDA Backend

## Technology Choices

### Rust

- nightly Rust (nightly-2026-04-03)
- tokio
- tracing
- anyhow
- serde

### CUDA Orchestration

- cuda-oxide (NVlabs) for memory management, streams, and kernel launching
- cudarc for cuBLASLt and NCCL bindings

### Kernels

- Custom CUDA kernels: GDN prefill/decode, attention, RMSNorm, RoPE, sampling
- cuBLASLt: GEMM (supports NVFP4 on Blackwell)
- llama.cpp: GGUF backend via FFI

### Parallelism

- TP=2: all-reduce/all-gather via NCCL
- PP=2: P2P send/recv with microbatching

---

# 4. Repository Layout

```text
crates/
  server/      # axum HTTP server, OpenAI routes, SSE streaming
  api/         # OpenAI API types + streaming protocol
  scheduler/   # Session lifecycle, batch construction, continuous batching
  kv/          # Hybrid KV cache (Mamba state + paged blocks), quantization
  model/       # Multi-format loader (safetensors/GGUF), config.json, weight registry
  backends/
    native/    # Custom CUDA kernels + cuBLASLt (PrismaSCOUT, AutoRound, BF16)
    gguf/      # llama.cpp wrapper (GGUF models)
  cuda/        # Tensor orchestration, cuda-oxide runtime, kernel launching
  parallelism/ # TP=2 and PP=2 implementations
  tokenizer/   # HuggingFace tokenizers (Qwen2Tokenizer class)
  metrics/     # Prometheus exporter
  mtp/         # Native MTP draft generation, verification, acceptance
```

---

# 5. Model Loader

Responsibilities:

- read config.json
- read tokenizer
- read safetensors or GGUF
- detect quantization format automatically (PrismaSCOUT / AutoRound / GGUF / BF16)
- build weight registry

Interface:

```rust
trait ModelLoader {
    fn load(path: &Path) -> LoadedModel;
}
```

---

# 6. Parallelism

## Tensor Parallelism (TP=2)

Startup flag: `--parallelism tp`

Weight sharding: column-parallel for Q/K/V/gate/up, row-parallel for O/down

Collectives: `ncclAllReduce` after attention and MLP

## Pipeline Parallelism (PP=2)

Startup flag: `--parallelism pp --pp-microbatch-size N`

Stage partitioning:
- GPU0: layers 0-31
- GPU1: layers 32-63

Communication: P2P send/recv of hidden states between stages

Microbatching: split batch into N microbatches, pipeline through stages to reduce GPU bubbles

---

# 7. KV Cache Design

## Two State Types Required

Qwen3.6 has a hybrid architecture:
- 48 layers use Gated DeltaNet (linear attention) → Mamba-style recurrent state
- 16 layers use full softmax attention → paged KV cache

### Mamba State (GDN layers)

```rust
struct MambaState {
    conv_state: DeviceBuffer<f32>,
    ssm_state: DeviceBuffer<f32>,
}
```

- Not paged in the traditional sense
- Updated incrementally per decode step
- Cannot be easily evicted to CPU/SSD

### Paged KV (full attention layers)

```rust
struct PagedKvCache {
    k_blocks: DeviceBuffer<half>,
    v_blocks: DeviceBuffer<half>,
    block_table: Vec<BlockTable>,
}
```

- Standard attention with custom CUDA kernels
- Supports GPU → CPU → SSD eviction
- Block size: 16 tokens (configurable)

## KV Cache Quantization

Supported formats:
- BF16 (baseline)
- FP8 (E4M3 / E5M2)
- NVFP4 (Blackwell only)

Format selectable at startup via `--kv-cache-dtype`

---

# 8. Custom CUDA Kernel Integration

Strategy: Write custom CUDA kernels in `crates/cuda/kernels/infers/`, compile to `.cubin` with nvcc, load at runtime via cudarc.

Kernels:
- GDN prefill: `infers_gdn_prefill_bf16`
- GDN decode: `infers_gdn_update_bf16`
- Attention: custom softmax + KV cache kernels
- Sampling: `infers_argmax_f32`

---

# 9. Session Lifecycle

States:

```text
Created
Prefill
Decoding
Paused
Evicted
Completed
```

Transitions:

```text
Created -> Prefill
Prefill -> Decoding
Decoding -> Paused
Paused -> Decoding
Decoding -> Completed
```

---

# 10. Scheduler

Objective: Latency fairness.

Algorithm: round robin.

Each decode iteration:

```text
collect active sessions
build batch
run decode
distribute outputs
```

---

# 11. Continuous Batching

Batch contains:

```rust
struct DecodeBatch {
    requests: Vec<RequestId>
}
```

Requirements:

- join existing batches
- leave existing batches
- dynamic batch growth

---

# 12. Prefill Path

Input: prompt tokens

Steps:

1. tokenize
2. allocate blocks (Mamba state + paged KV)
3. run transformer (dispatch GDN vs full attention per layer)
4. write KV
5. emit first token

---

# 13. Decode Path

Input: 1 token/session

Steps:

1. build decode batch
2. run attention (GDN or standard)
3. sample
4. append KV
5. stream output

---

# 14. MTP Support

Goal: Exploit Qwen3.6 native MTP head (`mtp_num_hidden_layers: 1`).

Flow:

```text
main forward (get hidden state)
draft tokens from MTP head (greedy)
verification: main model checks draft tokens in single forward pass
accept/reject: accept longest valid prefix
commit KV: write accepted tokens' KV cache
```

API parameter:

```json
{
  "speculative_config": {
    "method": "mtp",
    "num_speculative_tokens": 2
  }
}
```

## Acceptance Logic

Track:

```rust
struct MtpResult {
    accepted: usize,
}
```

Metrics:

- acceptance rate
- tokens saved

---

# 15. Sampling

Support:

- greedy
- temperature
- top-k
- top-p
- repetition penalty
- presence penalty
- frequency penalty

Trait:

```rust
trait Sampler {
    fn sample(&self, logits: &[f32]) -> TokenId;
}
```

---

# 16. KV Eviction

Priority:

```text
GPU
 -> pinned RAM
 -> SSD
```

Policies:

```text
LRU
manual pinning
```

Note: Mamba states (GDN layers) cannot be easily evicted. Only paged KV (full attention layers) supports eviction.

---

# 17. KV Serialization

Format:

```text
session header
block metadata
raw kv bytes
```

Goal: Resume sessions after restart.

---

# 18. Memory Budgeting

Startup:

```text
discover VRAM
reserve weights
reserve workspace
allocate remaining to KV
```

Outputs:

```rust
struct MemoryPlan {
    weight_bytes: u64,
    kv_bytes: u64,
}
```

---

# 19. API (OpenAI-Compatible)

Endpoints:

## Chat Completions

```http
POST /v1/chat/completions
```

Supports streaming via SSE (`stream: true`).

## Models

```http
GET /v1/models
```

## Tool Calls

Request format:

```json
{
  "tools": [...],
  "tool_choice": "auto"
}
```

Streaming tool calls via `delta.tool_calls` array.

No custom API (`/generate`, `/stream`, `/session` replaced by OpenAI compatibility).

---

# 20. Metrics

Expose:

```text
tokens/sec
active sessions
kv usage
batch size
mtp acceptance
gpu utilization
```

Prometheus format preferred.

---

# 21. Logging

Use:

```rust
tracing
```

Levels:

```text
error
warn
info
debug
```

---

# 22. Failure Recovery

Detect:

- OOM
- NCCL failure
- kernel failure

Recovery:

```text
pause scheduler
free failed batch
resume
```

---

# 23. Development Phases

## Phase 1: Bootstrap

- workspace, nightly toolchain, crate skeletons
- OpenAI API types and axum server scaffold
- SSE streaming
- Prometheus metrics

Expected: 2 weeks

## Phase 2: CUDA Backend

- cuda-oxide workspace setup
- Custom CUDA kernel compilation pipeline
- cudarc cuBLASLt + NCCL
- Memory allocator

Expected: 2 weeks

## Phase 3: Model Loading

- Safetensors loader, multi-format detection
- PrismaSCOUT, AutoRound, BF16 loaders
- Weight sharding for TP=2
- Memory budget calculator

Expected: 3 weeks

## Phase 4: TP=2 Forward Pass

- GDN prefill/decode (custom CUDA kernels)
- Standard attention prefill/decode (custom CUDA kernels)
- Layer dispatch per `layer_type`
- GEMM: cuBLASLt (NVFP4, FP16, BF16)
- NCCL all-reduce

Expected: 3 weeks

## Phase 5: PP=2 + Microbatching

- Stage partitioning (layers 0-31 vs 32-63)
- P2P communication
- Microbatch scheduler
- Pipeline bubble minimization

Expected: 3 weeks

## Phase 6: Continuous Batching

- Hybrid KV state manager (Mamba + paged)
- Block allocator, free lists
- Session lifecycle
- Batch builder with dynamic join/leave
- Round-robin scheduler

Expected: 3 weeks

## Phase 7: MTP

- MTP head weight loading
- MTP forward pass
- Draft generation, verification, acceptance
- `speculative-config` API parameter

Expected: 2 weeks

## Phase 8: Quantization Polish

- AutoRound INT4 end-to-end
- GGUF parser + llama.cpp integration
- KV cache quantization (FP8, NVFP4)
- Cross-format benchmarking
- Backend routing

Expected: 3 weeks

## Phase 9: Tool Calls + Final Polish

- Qwen3.6 chat template + tool parsing
- Tool call streaming (delta format)
- End-to-end benchmark vs vLLM
- 24-hour stability test
- Documentation

Expected: 2 weeks

---

# 24. Testing Strategy

Unit:

- block allocator
- sampler
- scheduler
- kernel loading
- format detection

Integration:

- prefill correctness
- decode correctness
- TP correctness
- PP correctness
- MTP acceptance rate
- tool call parsing

Golden tests:

Compare outputs against reference implementation (vLLM).

---

# 25. Stretch Goals

- CUDA Graphs
- speculative scheduling
- flash decoding
- prefix caching
- multi-node inference

---

# 26. Explicit Anti-Goals

Do not build:

- generic framework
- plugin ecosystem
- model zoo
- training support
- custom non-OpenAI API surface

The server exists solely to run Qwen3.6-27B efficiently on a fixed hardware configuration.

---

# 27. Key Decisions

| Decision | Choice |
|---|---|
| Target model | Qwen3.6-27B |
| Primary quant | PrismaSCOUT (NVFP4+BF16) |
| Secondary quant | AutoRound INT4 (W4A16) |
| Tertiary quant | GGUF (all standard Q2_K–Q8_0) |
| GGUF dequant | On-the-fly in llama.cpp kernels |
| AutoRound dequant | On-the-fly in custom INT4 GEMM kernels (weights stay packed in VRAM) |
| CUDA orchestration | cuda-oxide (NVlabs workspace) + cudarc |
| Math kernels | Custom CUDA kernels (GDN + standard attention), cuBLASLt (GEMM) |
| API | OpenAI-compatible only |
| Vision | Text-only |
| Thinking tokens | Stream as regular content |
| TP | 2 GPUs via NCCL |
| PP | 2 GPUs via P2P with microbatching |
| Context | 262K native (no YaRN) |
| MTP | Native Qwen3.6 MTP heads |
| Tool calls | Yes |
| Format detection | Auto-detect from model directory |
| KV cache quants | BF16, FP8 (E4M3/E5M2), NVFP4 |
| Rust toolchain | nightly-2026-04-03 |
