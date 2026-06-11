# Infers Implementation Plan

## Overview

This plan describes the implementation of **infers**, a high-performance inference server for Qwen3.6-27B on 2× NVIDIA RTX 5060 Ti GPUs.

## Goals

- Serve Qwen3.6-27B with OpenAI-compatible API
- Support multiple quantization formats (PrismaSCOUT, AutoRound, GGUF, BF16)
- Tensor Parallelism (TP=2) and Pipeline Parallelism (PP=2) with microbatching
- Native Multi-Token Prediction (MTP) for speculative decoding
- Continuous batching with hybrid KV cache (Mamba state + paged KV)
- Tool calls with streaming support
- KV cache quantization (BF16, FP8, NVFP4)

## Target Hardware

- 2× NVIDIA RTX 5060 Ti (32 GB VRAM each, Blackwell architecture)
- PCIe 4.0 x16 interconnect (no NVLink)
- CUDA 13.x

## Architecture Summary

```
Rust Host (cuda-oxide orchestration)
  ├─ Scheduler: continuous batching, round-robin
  ├─ KV Manager: hybrid (Mamba + paged)
  ├─ Model Loader: multi-format (safetensors, GGUF)
  ├─ Parallelism: TP=2 or PP=2 with microbatching
  ├─ MTP: draft generation, verification
  ├─ API: OpenAI chat completions
  └─ Server: axum HTTP

CUDA Kernels
  ├─ Custom CUDA kernels: GDN prefill/decode, standard attention
  ├─ cuBLASLt: GEMM (NVFP4, FP16, BF16)
  ├─ Custom: RMSNorm, RoPE, INT4 dequant
  └─ llama.cpp: GGUF backend
```

## Phases

| Phase | Duration | Description | Key Deliverable |
|-------|----------|-------------|-----------------|
| [Phase 1](phase-1-bootstrap.md) | 2 weeks | Workspace, crates, API types, server scaffold | Running HTTP server with mock responses |
| [Phase 2](phase-2-cuda-backend.md) | 2 weeks | CUDA runtime, kernel compilation, memory allocator | Loaded custom CUDA kernels on GPU |
| [Phase 3](phase-3-model-loading.md) | 3 weeks | Multi-format loader, weight sharding, memory budget | Weights loaded and sharded for TP=2 |
| [Phase 4](phase-4-tp-forward.md) | 3 weeks | Single GPU forward pass with GDN + standard attention | Correct token generation |
| [Phase 5](phase-5-pp-microbatching.md) | 3 weeks | Pipeline parallelism with microbatching | PP=2 producing correct results |
| [Phase 6](phase-6-continuous-batching.md) | 3 weeks | Continuous batching, hybrid KV, session lifecycle | 3+ concurrent sessions |
| [Phase 7](phase-7-mtp.md) | 2 weeks | Multi-Token Prediction speculative decoding | >1.2x speedup with MTP |
| [Phase 8](phase-8-quantization.md) | 3 weeks | AutoRound, GGUF, KV cache quantization, FP8/INT4 CUDA kernels | All formats working end-to-end |
| [Phase 9](phase-9-tool-calls.md) | 2 weeks | Tool calls, benchmarking, stability | Production-ready server |
| [Phase 10](phase-10-server-orchestration.md) | 1.5 weeks | Server orchestration — wire everything together | End-to-end pipeline (mock model) |
| [Phase 11](phase-11-model-integration.md) | 3–4 weeks | Model integration — real model loading and inference | Working inference with real Qwen3.6-27B |
| [Phase 12](phase-12-get-it-working.md) | 3–4 weeks | End-to-end real model inference | Correct token output with real Qwen3.6-27B |

**Total: ~33 weeks (8 months)**

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

- **CUDA orchestration:** cuda-oxide (NVlabs) for memory/streams, cudarc for cuBLASLt/NCCL
- **Parallelism:** Manual selection (TP or PP), no auto-switching, no DP
- **KV cache:** BF16, FP8 (E4M3/E5M2), NVFP4 (no TurboQuant)
- **Context:** 262K native (no YaRN extension)
- **Vision:** Text-only (skip vision encoder)
- **Thinking tokens:** Stream as regular content

## Dependencies

### Rust Toolchain
- Nightly: `nightly-2026-04-03`
- Components: `rust-src`, `rustc-dev`, `llvm-tools`
- cargo-oxide (custom cargo subcommand)

### System
- CUDA Toolkit 13.x
- NCCL 2.22+
- nvcc compiler
- Rust 1.85+

### External Crates
- cuda-oxide (git: NVlabs/cuda-oxide)
- cudarc v0.19.7 (crates.io)
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
| Custom CUDA kernel development | Medium | Medium | Prototype early, benchmark against targets |
| cuda-oxide alpha breakage | Medium | High | Pin to specific commit |
| llama.cpp Qwen3.6 support | Low | Medium | Verify before Phase 8 |
| 2× RTX 5060 Ti insufficient for 262K | Medium | High | Benchmark early, limit context |
| PP=2 P2P slow on PCIe | Medium | Medium | Fallback to TP if latency unacceptable |
| INT4 on-the-fly GEMM complex | Medium | Medium | Dequantize at load time |

## Getting Started

1. Install nightly Rust: `rustup install nightly-2026-04-03`
2. Install cargo-oxide: `cargo +nightly install --git https://github.com/NVlabs/cuda-oxide cargo-oxide`
3. Install CUDA 13.x and NCCL
4. Clone and build: `cargo oxide build`
5. Run server: `cargo run --bin server -- --model /path/to/model --parallelism tp`

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
