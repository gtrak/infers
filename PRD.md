# Rust Inference Server for Qwen 3.6 27B
## Product Requirements Document (PRD) + Implementation Specification

Version: 0.1
Target Audience: Single engineer
Primary Language: Rust
Target Hardware: 2x NVIDIA RTX 5060 Ti
Model Family: Qwen 3.6 (initially 27B)
Scope: Personal inference server optimized for agent workloads

---

# 1. Goals

Build a high-performance inference server focused on:

- Qwen 3.6 support
- Tensor parallelism across 2 GPUs
- Paged KV cache
- Continuous batching
- Multi-Token Prediction (MTP)
- Long-lived agent sessions
- Predictable latency
- Minimal operational complexity

Non-goals:

- OpenAI API compatibility
- Multi-tenant hosting
- Kubernetes
- Dynamic model loading
- LoRA support
- Multi-model serving
- Distributed clusters
- Vision models (phase 1)

---

# 2. Success Metrics

## Functional

- Load Qwen 3.6 27B
- Generate tokens correctly
- Support concurrent sessions
- Support MTP
- Survive multi-hour agent runs

## Performance

- >= 90% of vLLM throughput
- <= 10% memory overhead vs vLLM
- Stable operation for 24 hours

---

# 3. Architecture

Layers:

1. API Layer
2. Scheduler
3. Batch Builder
4. KV Manager
5. Model Runner
6. CUDA Backend

---

# 4. Technology Choices

## Rust

- stable Rust
- tokio
- tracing
- anyhow
- serde

## CUDA

Preferred:

- cuBLASLt
- NCCL
- FlashInfer kernels

Fallback:

- FlashAttention kernels

Do not implement GEMMs manually.

---

# 5. Repository Layout

```text
crates/

api/
scheduler/
kv/
model/
cuda/
mtp/
tokenizer/
metrics/
server/
```

---

# 6. Model Loader

Responsibilities:

- read config.json
- read tokenizer
- read safetensors
- build weight registry

Interface:

```rust
trait ModelLoader {
    fn load(path: &Path) -> LoadedModel;
}
```

---

# 7. Tensor Parallelism

Phase 1:

TP = 2 only

Assumptions:

```text
GPU0 = shard 0
GPU1 = shard 1
```

Collectives:

```text
all_reduce
all_gather
```

Transport:

```text
NCCL
```

---

# 8. KV Cache Design

## Block Size

Initial:

```text
16 tokens
```

Configurable later.

## Physical Block

```rust
struct PhysicalBlock {
    id: u32,
    device: DeviceId,
    ptr: DevicePtr,
}
```

## Logical Mapping

```rust
struct Sequence {
    blocks: Vec<BlockId>
}
```

## Free Lists

```rust
gpu_free
cpu_free
```

---

# 9. PagedAttention

Core abstraction:

```rust
struct BlockTable {
    physical_blocks: Vec<u32>
}
```

Attention kernel receives:

```text
query
block table
kv blocks
```

Kernel performs:

```text
lookup
gather
attention
output
```

Reuse FlashInfer implementation where possible.

---

# 10. Session Lifecycle

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

# 11. Scheduler

Objective:

Latency fairness.

Algorithm:

```text
round robin
```

Each decode iteration:

```text
collect active sessions
build batch
run decode
distribute outputs
```

---

# 12. Continuous Batching

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

# 13. Prefill Path

Input:

```text
prompt tokens
```

Steps:

1. tokenize
2. allocate blocks
3. run transformer
4. write KV
5. emit first token

---

# 14. Decode Path

Input:

```text
1 token/session
```

Steps:

1. build decode batch
2. run attention
3. sample
4. append KV
5. stream output

---

# 15. MTP Support

Goal:

Exploit Qwen native MTP heads.

## Flow

```text
main forward
draft tokens
verification
accept/reject
commit KV
```

## Acceptance Logic

Track:

```rust
struct MtpResult {
    accepted: usize,
}
```

Metrics:

```text
acceptance rate
tokens saved
```

---

# 16. Sampling

Support:

- greedy
- temperature
- top-k
- top-p
- repetition penalty

Trait:

```rust
trait Sampler {
    fn sample(&self, logits: &[f32]) -> TokenId;
}
```

---

# 17. KV Eviction

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

---

# 18. KV Serialization

Format:

```text
session header
block metadata
raw kv bytes
```

Goal:

Resume sessions after restart.

---

# 19. Memory Budgeting

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

# 20. API

Phase 1:

## Generate

```http
POST /generate
```

## Stream

```http
GET /stream/{id}
```

## Session

```http
POST /session
DELETE /session/{id}
```

---

# 21. Metrics

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

# 22. Logging

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

# 23. Failure Recovery

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

# 24. Development Phases

## Phase 1

- model loading
- tokenizer
- single GPU inference

Expected:
4-6 weeks

## Phase 2

- TP=2
- NCCL

Expected:
2-4 weeks

## Phase 3

- paged KV
- continuous batching

Expected:
4-6 weeks

## Phase 4

- MTP

Expected:
2-3 weeks

## Phase 5

- KV offload
- persistence

Expected:
3-5 weeks

---

# 25. Testing Strategy

Unit:

- block allocator
- sampler
- scheduler

Integration:

- prefill correctness
- decode correctness
- TP correctness

Golden tests:

Compare outputs against reference implementation.

---

# 26. Stretch Goals

- CUDA Graphs
- speculative scheduling
- flash decoding
- prefix caching
- multi-node inference

---

# 27. Explicit Anti-Goals

Do not build:

- generic framework
- plugin ecosystem
- model zoo
- training support

The server exists solely to run Qwen efficiently on a fixed hardware configuration.
