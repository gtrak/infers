# Phase 1: Bootstrap

---
**Status**: DONE
**Last Updated**: 2026-06-21
**Rationale**: Workspace, crates, toolchain, API types, server skeleton all exist. Superseded by later phases that wired real inference.
**Actual Deliverables**:
- [x] Workspace `Cargo.toml` with nightly toolchain
- [x] `rust-toolchain.toml`
- [x] Crate skeletons for all crates
- [~] OpenAI API types (partial — basic structs exist)
- [x] Basic axum HTTP server with mock responses
- [~] SSE streaming scaffold
- [x] Prometheus metrics endpoint
- [x] CLI argument parsing
- [x] Health check endpoint
- [ ] Server wired to real inference engine
---

**Duration:** 2 weeks  
**Goal:** Create the Rust workspace, crate skeletons, and API scaffolding.

## Deliverables

- [x] Workspace `Cargo.toml` with nightly toolchain
- [x] `rust-toolchain.toml` (nightly-2026-04-03, rust-src, rustc-dev, llvm-tools)
- [x] Crate skeletons for all crates
- [~] OpenAI API types (partial — basic structs exist)
- [x] Basic axum HTTP server with mock responses
- [~] SSE streaming scaffold
- [x] Prometheus metrics endpoint
- [x] CLI argument parsing
- [x] Health check endpoint
- [ ] Server wired to real inference engine

## Crate Structure

```
crates/
  server/              # axum HTTP server, CLI, main entry point
  api/                 # OpenAI-compatible types + SSE protocol
  scheduler/           # Session lifecycle, batch construction
  kv/                  # Hybrid KV state manager
  model/               # Multi-format model loader
  backends/
    native/            # Custom CUDA kernels + cuBLASLt backend
    gguf/              # llama.cpp backend
  cuda/                # cuda-oxide + cudarc hybrid
  parallelism/         # TP=2 and PP=2 implementations
  tokenizer/           # HF tokenizers wrapper
  metrics/             # Prometheus exporter
  mtp/                 # MTP draft/verify
```

## Technical Details

### Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "crates/server",
    "crates/api",
    "crates/scheduler",
    "crates/kv",
    "crates/model",
    "crates/backends/native",
    "crates/backends/gguf",
    "crates/cuda",
    "crates/parallelism",
    "crates/tokenizer",
    "crates/metrics",
    "crates/mtp",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.85"
authors = ["infers"]
license = "MIT OR Apache-2.0"

[workspace.dependencies]
# Core async
tokio = { version = "1.43", features = ["full"] }
tokio-util = "0.7"
futures = "0.3"
async-stream = "0.3"
pin-project = "1"

# HTTP / API
axum = { version = "0.8", features = ["ws"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "trace"] }
hyper = "1"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_with = "3"

# Error handling
anyhow = "1.0"
thiserror = "2.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }

# CLI
clap = { version = "4.5", features = ["derive", "env"] }

# Metrics
prometheus = "0.13"

# CUDA (cudarc - stable)
cudarc = { version = "0.19.7", features = ["cublaslt", "nccl", "cuda-13020"] }

# CUDA (cuda-oxide - nightly)
cuda-core = { git = "https://github.com/NVlabs/cuda-oxide" }
cuda-async = { git = "https://github.com/NVlabs/cuda-oxide" }
cuda-host = { git = "https://github.com/NVlabs/cuda-oxide" }

# ML / Data
safetensors = "0.5"
memmap2 = "0.9"

# Testing
proptest = "1"
tempfile = "3"

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
panic = "abort"

[profile.bench]
inherits = "release"
debug = true
```

### rust-toolchain.toml

```toml
[toolchain]
channel = "nightly-2026-04-03"
components = ["rust-src", "rustc-dev", "llvm-tools"]
profile = "minimal"
```

### Key Types (in `crates/api/`)

```rust
// Request types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub parallel_tool_calls: bool,
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f32,
    #[serde(default = "default_frequency_penalty")]
    pub frequency_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative_config: Option<SpeculativeConfig>,
}

// Response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// Tool types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String),  // "none", "auto", "required"
    Object {  // {"type": "function", "function": {"name": "..."}}
        #[serde(rename = "type")]
        tool_type: String,
        function: FunctionChoice,
    },
}
```

### Server Scaffold (`crates/server/`)

```rust
#[derive(Parser, Debug, Clone)]
#[command(name = "infers")]
#[command(about = "Qwen3.6-27B Inference Server")]
pub struct Args {
    /// Model path or HuggingFace model ID
    #[arg(long, env = "INFERS_MODEL")]
    pub model: String,
    
    /// Parallelism mode
    #[arg(long, value_enum, default_value = "tp")]
    pub parallelism: ParallelismMode,
    
    /// PP microbatch size
    #[arg(long, default_value = "1")]
    pub pp_microbatch_size: usize,
    
    /// KV cache data type
    #[arg(long, value_enum, default_value = "fp8")]
    pub kv_cache_dtype: KvCacheDtype,
    
    /// Max model context length
    #[arg(long, default_value = "262144")]
    pub max_model_len: usize,
    
    /// GPU memory utilization (0.0 - 1.0)
    #[arg(long, default_value = "0.85")]
    pub gpu_memory_utilization: f32,
    
    /// Enable speculative decoding (MTP)
    #[arg(long, default_value = "false")]
    pub enable_mtp: bool,
    
    /// Number of speculative tokens
    #[arg(long, default_value = "2")]
    pub num_speculative_tokens: usize,
    
    /// Server port
    #[arg(short, long, default_value = "8000")]
    pub port: u16,
    
    /// Server host
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
    
    /// Log level
    #[arg(long, default_value = "info")]
    pub log_level: String,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum ParallelismMode {
    Tp,
    Pp,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum KvCacheDtype {
    Bf16,
    Fp8,
    Nvfp4,
}
```

### Mock Handlers

```rust
async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if req.stream {
        // Return SSE stream
        let stream = mock_stream();
        Ok(Sse::new(stream).into_response())
    } else {
        // Return mock response
        Ok(Json(mock_response()).into_response())
    }
}

async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![Model {
            id: "Qwen3.6-27B".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "infers".to_string(),
        }],
    })
}
```

### Metrics Endpoint

```rust
use prometheus::{Counter, Gauge, Histogram, Registry};

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();
    
    pub static ref TOKENS_GENERATED: Counter = Counter::new(
        "infers_tokens_generated_total",
        "Total tokens generated"
    ).unwrap();
    
    pub static ref ACTIVE_SESSIONS: Gauge = Gauge::new(
        "infers_active_sessions",
        "Number of active inference sessions"
    ).unwrap();
    
    pub static ref BATCH_SIZE: Gauge = Gauge::new(
        "infers_batch_size",
        "Current batch size"
    ).unwrap();
    
    pub static ref REQUEST_LATENCY: Histogram = Histogram::with_opts(
        HistogramOpts::new("infers_request_latency_seconds", "Request latency")
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    ).unwrap();
}
```

## File Structure

```
Cargo.toml                  # Workspace root
rust-toolchain.toml         # Nightly toolchain

Cargo.lock                  # Generated after first build

crates/
  server/
    Cargo.toml
    src/
      main.rs              # Entry point, CLI parsing
      server.rs            # axum routes
      state.rs             # AppState
      handlers/
        chat.rs            # POST /v1/chat/completions
        models.rs          # GET /v1/models
        health.rs          # GET /health
    
  api/
    Cargo.toml
    src/
      lib.rs
      request.rs           # ChatCompletionRequest, Message, Tool, etc.
      response.rs          # ChatCompletionResponse, Choice, Usage
      streaming.rs         # SSE streaming types and helpers
      error.rs             # ApiError, error responses
      
  scheduler/
    Cargo.toml
    src/
      lib.rs
      session.rs           # Session struct, lifecycle
      batch.rs             # BatchBuilder, DecodeBatch
      
  kv/
    Cargo.toml
    src/
      lib.rs
      manager.rs           # HybridKvManager
      paged.rs             # PagedKvCache, BlockTable
      mamba.rs             # MambaState
      
  model/
    Cargo.toml
    src/
      lib.rs
      loader.rs            # Multi-format loader
      config.rs            # Qwen3.6 config parser
      weights.rs           # Weight registry, sharding
      formats.rs           # QuantizationFormat enum
      
  backends/
    native/
      Cargo.toml
      src/
        lib.rs
        forward.rs         # Native forward pass
        
    gguf/
      Cargo.toml
      src/
        lib.rs
        
  cuda/
    Cargo.toml
    src/
      lib.rs
      context.rs            # CudaContext wrapper
      stream.rs             # CudaStream wrapper
      memory.rs             # DeviceBuffer, PinnedBuffer
      
  parallelism/
    Cargo.toml
    src/
      lib.rs
      tp.rs                 # TensorParallel
      pp.rs                 # PipelineParallel
      microbatch.rs         # Microbatch scheduler
      
  tokenizer/
    Cargo.toml
    src/
      lib.rs
      
  metrics/
    Cargo.toml
    src/
      lib.rs
      registry.rs           # Prometheus registry setup
      
  mtp/
    Cargo.toml
    src/
      lib.rs
      draft.rs              # Draft token generation
      verify.rs             # Verification logic
```

## Dependencies

### Phase 1 → Phase 2

Phase 1 creates the workspace and crate structure. Phase 2 will fill in the CUDA implementation.

### External Dependencies

- cargo-oxide (install via `cargo +nightly install --git https://github.com/NVlabs/cuda-oxide cargo-oxide`)
- axum, tokio, serde, tracing, clap, prometheus
- cuda-oxide workspace (git dependency)
- cudarc (crates.io)

## Success Criteria

1. `cargo check` passes on nightly
2. `cargo oxide check` works for CUDA crates
3. Server starts and responds to `GET /health`
4. `POST /v1/chat/completions` returns mock response
5. `GET /v1/models` returns model list
6. SSE streaming works (mock tokens)
7. Prometheus metrics endpoint at `/metrics`

## Cross-References

- **Research:** See `../research/api.md` for OpenAI API schema details
- **Research:** See `../research/quantization.md` for format detection logic
- **Phase 2:** CUDA backend will use crate skeletons from this phase
- **Phase 3:** Model loader will use `model/` crate structure
- **Phase 9:** Tool calls will extend `api/` types

## Open Questions

1. Should we use `utoipa` for OpenAPI spec generation?
2. Should we implement request validation middleware?
3. Rate limiting: built-in or external (e.g., nginx)?
