# Infers Implementation Plan

## Overview

This plan describes the implementation of **infers**, a high-performance inference server for Qwen3.5-27B on 2× NVIDIA RTX 5060 Ti GPUs (16 GB VRAM each).

**Current State (2026-06-11)**: TP=2 forward pass works end-to-end with real Qwen3.5-27B AutoRound INT4 model, producing varied non-zero tokens. Performance is ~0.1 tok/s (200× off the 20 tok/s target) because weights are uploaded per-GEMM and immediately dropped — no GPU-resident weight cache exists.

## Goals

- Serve Qwen3.5-27B with OpenAI-compatible API
- Support AutoRound INT4 (working) — BF16, PrismaSCOUT, GGUF are **not yet implemented**
- Tensor Parallelism (TP=2) — **working but 200× too slow**
- Pipeline Parallelism (PP=2) — **out of scope** for 2 GPUs (TP is optimal)
- Native Multi-Token Prediction (MTP) for speculative decoding — **not started**
- Continuous batching with hybrid KV cache (Mamba state + paged KV) — **blocked by weight cache**
- Tool calls with streaming support — **not started**
- KV cache quantization (BF16, FP8, NVFP4) — **not started**

## Target Hardware

- 2× NVIDIA RTX 5060 Ti (16 GB VRAM each — **not 32 GB**, corrected)
- PCIe 4.0 x16 interconnect (no NVLink)
- CUDA 12.x (not 13.x — corrected)

## Architecture Summary

```
Rust Host (cudarc orchestration)
  ├─ Scheduler: continuous batching, round-robin — NOT IMPLEMENTED
  ├─ KV Manager: hybrid (Mamba + paged) — paged works, Mamba works
  ├─ Model Loader: AutoRound INT4 (safetensors) — GGUF, BF16 not implemented
  ├─ Parallelism: TP=2 working (PP out of scope for 2 GPUs)
  ├─ MTP: draft generation, verification — NOT IMPLEMENTED
  ├─ API: OpenAI chat completions — scaffold exists, not wired to inference
  └─ Server: axum HTTP — mock responses only (not wired to real engine)

CUDA Kernels
  ├─ Custom CUDA kernels: GDN prefill/decode, standard attention — WORKING
  ├─ cuBLASLt: GEMM (BF16, INT4) — WORKING
  ├─ Custom: RMSNorm, RoPE, INT4 dequant, SiLU, softmax, embedding — WORKING
  └─ llama.cpp: GGUF backend — NOT IMPLEMENTED
```

## Phases

| Phase | Status | Duration | Description | Key Deliverable |
|-------|--------|----------|-------------|-----------------|
| [Phase 1](phase-1-bootstrap.md) | PARTIAL | 2 weeks | Workspace, crates, API types, server scaffold | HTTP server with **mock** responses |
| [Phase 2](phase-2-cuda-backend.md) | PARTIAL | 2 weeks | CUDA runtime, kernel compilation, memory allocator | cudarc works, kernels compile, NCCL works |
| [Phase 3](phase-3-model-loading.md) | PARTIAL | 3 weeks | Multi-format loader, weight sharding, memory budget | **AutoRound INT4** loaded and sharded for TP=2. BF16/GGUF/PrismaSCOUT **not implemented** |
| [Phase 4](phase-4-tp-forward.md) | PARTIAL | 3 weeks | Single GPU forward pass with GDN + standard attention | **Tokens produced but 200× off performance target** |
| [Phase 4.5](phase-4.5-attention-kernels.md) | PARTIAL | — | Attention kernels: GDN prefill/decode, standard attention | Kernels compile and run. Bugs fixed. No standalone tests. |
| [Phase 4.6](phase-4.6-pagedattention.md) | PARTIAL | — | Paged KV cache read/write, block tables | Paged KV works. No standalone correctness tests. |
| [Phase 4.7](phase-4.7-gpu-weight-cache.md) | **NOT DONE** | 1 week | Cache weights as GPU-resident buffers | **Next critical blocker** |
| ~~Phase 5~~ | ~~OUT OF SCOPE~~ | ~~3 weeks~~ | ~~Pipeline parallelism with microbatching~~ | ~~Moved to [done/](done/)~~ |
| [Phase 6](phase-6-continuous-batching.md) | NOT DONE | 3 weeks | Continuous batching, hybrid KV, session lifecycle | **Blocked by Phase 4.7** |
| [Phase 6.5](phase-6.5-eviction.md) | NOT DONE | — | KV cache eviction policy | **Blocked by Phase 6** |
| [Phase 6.6](phase-6.6-eviction-wiring.md) | NOT DONE | — | Eviction policy wired to scheduler | **Blocked by Phase 6** |
| [Phase 7](phase-7-mtp.md) | NOT DONE | 2 weeks | Multi-Token Prediction speculative decoding | **Blocked by Phase 4.7 + 6** |
| [Phase 8](phase-8-quantization.md) | PARTIAL | 3 weeks | AutoRound, GGUF, KV cache quantization, FP8/INT4 CUDA kernels | **AutoRound works**. GGUF/PrismaSCOUT/FP8 **not implemented** |
| [Phase 9](phase-9-tool-calls.md) | NOT DONE | 2 weeks | Tool calls, benchmarking, stability | API scaffold exists |
| [Phase 10](phase-10-server-orchestration.md) | PARTIAL | 1.5 weeks | Server orchestration — wire everything together | **Mock model only**. Not wired to real inference engine. |
| [Phase 11](phase-11-model-integration.md) | PARTIAL | 3–4 weeks | Model integration — real model loading and inference | Real model loads and produces tokens. **Performance 200× off.** |
| [Phase 12](phase-12-get-it-working.md) | PARTIAL | 3–4 weeks | End-to-end real model inference | Smoke test passes. **Server unwired. No HF reference.** |

**Critical Path**: Phase 4.7 (GPU Weight Cache) → Phase 4 (hit 20 tok/s) → Phase 6 (continuous batching) → Phase 7 (MTP) → Phase 10 (wire server)

**Total Time to Production**: ~33 weeks original estimate. With current state, **~4-6 weeks remaining** if focused on critical path (4.7 → 4 perf → 6 → 7 → 10).

## Research Documents

| Document | Contents |
|----------|----------|
| [architecture.md](research/architecture.md) | Qwen3.6-27B architecture, hybrid attention |
| [quantization.md](research/quantization.md) | PrismaSCOUT, AutoRound, GGUF formats, memory calculations |
| [kernels.md](research/kernels.md) | Kernel compilation strategy (deprecated — custom kernels used instead) |
| [api.md](research/api.md) | OpenAI chat completions API, tool calls, SSE streaming |
| [parallelism.md](research/parallelism.md) | TP=2, PP=2, NCCL, P2P, memory distribution |

## Decisions

See individual phase documents for detailed decisions. Key ones:

- **CUDA orchestration:** **cudarc** (not cuda-oxide — out of scope). cudarc handles context, streams, cuBLASLt, NCCL.
- **Kernel compilation:** Compile at build time via `build.rs` (nvcc). No runtime compilation.
- **Parallelism:** TP=2 only. PP is out of scope for 2 GPUs. No DP.
- **Quantization formats:** AutoRound INT4 works. GGUF, PrismaSCOUT, BF16 base model **not yet implemented**.
- **KV cache:** BF16 only. FP8/NVFP4 KV cache quantization **not yet implemented**.
- **Context:** 262K native (no YaRN extension)
- **Vision:** Text-only (skip vision encoder)
- **Thinking tokens:** Stream as regular content

## Dependencies

### Rust Toolchain
- Nightly: `nightly-2026-04-03`
- Components: `rust-src`, `rustc-dev`, `llvm-tools`

### System
- CUDA Toolkit 13.x
- NCCL 2.22+
- nvcc compiler
- Rust 1.85+

### External Crates
- **cudarc v0.19.7** (crates.io) — CUDA runtime, cuBLASLt, NCCL
- axum, tokio, serde, tracing, clap, prometheus
- safetensors, tokenizers
- half (bf16/f16 types)
- axum, tokio, serde, tracing, clap, prometheus
- safetensors, tokenizers

## Metrics

### Performance Targets

| Metric | Target |
|--------|--------|
| Single request decode | >20 tok/s |
| Throughput (10 concurrent) | >100 tok/s total |
| First token latency (1K prompt) | <500ms |
| MTP speedup | >1.5x |
| 24-hour stability | 0 OOM, 0 crashes |

### Exposed Metrics (Prometheus)

- `infers_tokens_generated_total`
- `infers_active_sessions`
- `infers_kv_cache_usage_bytes`
- `infers_batch_size`
- `infers_mtp_acceptance_rate`
- `infers_request_latency_seconds`
- `infers_gpu_memory_usage_bytes`

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Weight cache OOM on 16 GB GPUs | Medium | High | Benchmark memory, assert before upload, fallback to per-call upload |
| INT4 dequantization accuracy drift | Medium | High | Compare layer outputs against HF reference (Phase 12) |
| Custom CUDA kernel bugs (GDN math) | Medium | High | Write Python reference impl, compare per-layer |
| Per-head → full GEMM refactor breaks attention | Medium | High | A/B token comparison before/after |
| 2× RTX 5060 Ti insufficient for 262K context | Medium | High | Benchmark early, limit context to 32K or 64K |
| cudarc API changes | Low | Medium | Pin to 0.19.7 in Cargo.lock |

## Getting Started

1. Install nightly Rust: `rustup install nightly-2026-04-03`
2. Install CUDA 13.x and NCCL
3. Clone and build: `cargo build --release`
4. Run smoke test: `cargo test --release -p infers-backend-native --test smoke_test -- --ignored`
5. Run server (mock mode): `cargo run --bin infers-server -- --model /path/to/model`

## Contributing

Each phase is designed to be delegatable with fresh context. See individual phase documents for:
- Detailed technical specifications
- File structures
- Testing strategies
- Success criteria
- Cross-references to research

## License

MIT OR Apache-2.0

## Contact

For questions about this plan, refer to the research documents or the PRD (`../PRD.md`).
