mod handlers;
mod orchestrator;
mod server;
mod state;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::sync::Mutex;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt, Registry};

use infers_backend_native::{BackendEvictionStore, ForwardEngine};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;

use infers_cuda::context::CudaRuntime;
use infers_cuda::kernels::KernelRegistry;
use infers_cuda::stream::StreamPool;
use infers_kv::PagedKvManager;
use infers_model::sharding::shard_weights_tp;
use infers_model::{load_model, load_safetensors, strip_language_model_prefix, build_main_layers, ModelConfig, WeightRegistry};
use infers_scheduler::RoundRobinScheduler;
use infers_tokenizer::Tokenizer;

use crate::orchestrator::InferenceOrchestrator;
use crate::state::AppState;

/// Qwen3.6-27B Inference Server
#[derive(Parser, Debug, Clone)]
#[command(name = "infers")]
#[command(about = "Qwen3.6-27B Inference Server")]
pub struct Args {
    /// Model path or HuggingFace model ID
    #[arg(long, env = "INFERS_MODEL", default_value = "Qwen3.6-27B")]
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

    /// Tensor parallelism degree (number of GPUs)
    #[arg(long, default_value = "1")]
    pub tensor_parallel_size: usize,

    /// Number of KV cache pages in the pool
    #[arg(long, default_value = "2048")]
    pub num_pages: usize,

    /// Number of tokens per KV cache page
    #[arg(long, default_value = "16")]
    pub page_size: usize,

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

    /// Enable OTLP trace export
    #[arg(long, default_value = "false")]
    pub otlp_enabled: bool,

    /// OTLP gRPC endpoint
    #[arg(long, default_value = "http://localhost:4317")]
    pub otlp_endpoint: String,

    /// OTLP service name
    #[arg(long, default_value = "infers")]
    pub otlp_service_name: String,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum ParallelismMode {
    Tp,
    Pp,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum KvCacheDtype {
    Bf16,
    Fp8,
    Nvfp4,
}

impl From<KvCacheDtype> for infers_kv::KvCacheDtype {
    fn from(dtype: KvCacheDtype) -> Self {
        match dtype {
            KvCacheDtype::Bf16 => infers_kv::KvCacheDtype::Bf16,
            KvCacheDtype::Fp8 => infers_kv::KvCacheDtype::Fp8E4M3,
            KvCacheDtype::Nvfp4 => infers_kv::KvCacheDtype::Nvfp4,
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Server failed: {}", e);
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args = Args::parse();

    // Initialize layered tracing subscriber
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&args.log_level));

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_thread_ids(true);

    if args.otlp_enabled {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(&args.otlp_endpoint)
            .build()
            .expect("Failed to build OTLP span exporter");
        let tracer_provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
            .build();
        let tracer = tracer_provider.tracer(args.otlp_service_name);

        use opentelemetry::global;
        global::set_tracer_provider(tracer_provider);

        let otlp_layer = tracing_opentelemetry::layer()
            .with_tracer(tracer);

        Registry::default()
            .with(env_filter)
            .with(fmt_layer)
            .with(otlp_layer)
            .init();
    } else {
        Registry::default()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    }

    tracing::info!("Starting infers server");
    tracing::info!("Model: {}", args.model);
    tracing::info!("Parallelism: {:?}", args.parallelism);
    tracing::info!("Tensor parallel size: {}", args.tensor_parallel_size);
    tracing::info!("KV cache dtype: {:?}", args.kv_cache_dtype);
    tracing::info!("Num pages: {}", args.num_pages);
    tracing::info!("Page size: {}", args.page_size);

    // Step 1: Initialize CUDA runtime with tensor parallelism support
    let num_gpus = args.tensor_parallel_size;
    let cuda_runtime = CudaRuntime::new()
        .context("Failed to initialize CUDA runtime")?;
    let mut contexts = Vec::new();
    for i in 0..num_gpus {
        contexts.push(
            cuda_runtime.device(i)
                .with_context(|| format!("Failed to get CUDA device {i}"))?
                .clone()
        );
    }

    // Step 2: Create CUDA stream pool and get a stream for the orchestrator
    let streams = StreamPool::new(&contexts)
        .context("Failed to create CUDA stream pool")?;
    let stream = streams.get(0)
        .context("Failed to get CUDA stream")?
        .clone();

    // Step 3: Load model config and weights
    let model_path = Path::new(&args.model);
    let (model_config, weight_registries) = if model_path.exists() && num_gpus > 1 {
        // TP path: load raw safetensors, shard, build per-GPU layers
        tracing::info!("Loading model from {} with TP={}", args.model, num_gpus);
        let config = infers_model::config::ModelConfig::load(model_path)
            .with_context(|| format!("Failed to load model config from {}", args.model))?;
        let mut raw_weights = load_safetensors(model_path)
            .with_context(|| format!("Failed to load safetensors from {}", args.model))?;
        strip_language_model_prefix(&mut raw_weights);
        let shards = shard_weights_tp(&raw_weights, &config, num_gpus)
            .with_context(|| format!("Failed to shard weights for TP={num_gpus}"))?;
        let mut registries = Vec::new();
        for shard in shards {
            let mut registry = shard.registry;
            build_main_layers(&mut registry, &config)
                .with_context(|| format!("Failed to build layers for GPU {}", shard.gpu_id))?;
            registries.push(registry);
        }
        (Arc::new(config), registries)
    } else if model_path.exists() {
        // TP=1 path: use load_model as before
        tracing::info!("Loading model from {}", args.model);
        let loaded = load_model(model_path)
            .with_context(|| format!("Failed to load model from {}", args.model))?;
        (Arc::new(loaded.config), vec![loaded.weights])
    } else {
        // If no model path, create a minimal config for wiring
        tracing::warn!("Model path {} not found, using default config for wiring", args.model);
        let config = ModelConfig {
            architectures: vec!["Qwen3_5ForConditionalGeneration".to_string()],
            model_type: "qwen3_5".to_string(),
            num_hidden_layers: 48,
            hidden_size: 5120,
            intermediate_size: 17408,
            vocab_size: 152064,
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: false,
            rope_theta: 10000000.0,
            partial_rotary_factor: 0.25,
            mrope_interleaved: true,
            mrope_section: vec![11, 11, 10],
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            attn_output_gate: true,
            quantization_config: None,
            layer_types: None,
            linear_num_key_heads: 1,
            linear_key_head_dim: 1,
            linear_conv_kernel_dim: 4,
            linear_num_value_heads: 48,
            linear_value_head_dim: 128,
        };
        let weights = WeightRegistry::new();
        (Arc::new(config), vec![weights])
    };

    let num_layers = model_config.num_hidden_layers;

    // Step 4: Register and load CUDA kernels
    let mut kernel_registry = KernelRegistry::new();
    kernel_registry.register_infers_kernels();

    // Step 5: Create ForwardEngine
    let engine = ForwardEngine::new(
        model_config.clone(),
        weight_registries,
        contexts,
        kernel_registry,
        streams,
        128, // group_size (default AutoRound)
    ).context("Failed to create ForwardEngine")?;

    // Step 6: Create PagedKvManager for the scheduler
    let kv_cache_dtype = infers_kv::KvCacheDtype::from(args.kv_cache_dtype);
    // TODO: Wire kv_cache_dtype ({:?}) to PagedKvManager when quantized KV cache support is ready.
    // The dtype determines bytes-per-element for buffer sizing (bf16=2, fp8=1, nvfp4=1).
    // For now, dtype conversion validates the CLI value is recognized.
    let _ = &kv_cache_dtype;

    let page_size = args.page_size;
    let num_kv_heads = model_config.num_key_value_heads;
    let head_dim = model_config.head_dim;
    let total_pages = args.num_pages;
    let max_cache_bytes = 64 * 1024 * 1024; // 64 MB
    let eviction_max_bytes = 32 * 1024 * 1024; // 32 MB

    let kv_manager = PagedKvManager::new(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        max_cache_bytes,
        eviction_max_bytes,
    );

    // Step 7: Create RoundRobinScheduler
    let max_concurrent_sessions = 4;
    let max_batch_size = 4;
    let max_tokens_per_batch = 1024;

    let scheduler = RoundRobinScheduler::new(
        max_concurrent_sessions,
        max_batch_size,
        max_tokens_per_batch,
        kv_manager,
    );

    // Step 8: Create BackendEvictionStore
    let eviction_store = BackendEvictionStore::new(num_layers);

    // Step 9: Create InferenceOrchestrator
    let enable_mtp = args.enable_mtp;
    let mtp = None; // MTP initialization deferred — requires MtpWeights from model loading
    let mtp_metrics = None;

    let orchestrator = InferenceOrchestrator::new(
        scheduler,
        engine,
        eviction_store,
        stream,
        num_layers,
        enable_mtp,
        mtp,
        mtp_metrics,
    );

    // Step 10: Create Tokenizer
    let tokenizer = if model_path.exists() {
        let tokenizer_path = model_path.join("tokenizer.json");
        if tokenizer_path.exists() {
            Tokenizer::from_file(tokenizer_path.to_str().unwrap())
                .context("Failed to load tokenizer")?
        } else {
            tracing::warn!("tokenizer.json not found at {:?}, using pretrained tokenizer", tokenizer_path);
            Tokenizer::from_pretrained("Qwen/Qwen3.6-27B")
                .context("Failed to load pretrained tokenizer")?
        }
    } else {
        tracing::warn!("No model path, loading pretrained tokenizer");
        Tokenizer::from_pretrained("Qwen/Qwen3.6-27B")
            .context("Failed to load pretrained tokenizer")?
    };

    // Step 11: Build AppState
    let state = Arc::new(AppState {
        model_name: args.model.clone(),
        orchestrator: Arc::new(Mutex::new(orchestrator)),
        tokenizer,
    });

    // Step 12: Spawn background scheduler loop
    server::spawn_scheduler_loop(state.orchestrator.clone());

    // Step 13: Build and start HTTP server
    let app = server::build_router(state);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("Failed to bind to {}", addr))?;
    tracing::info!("Listening on {}", addr);

    axum::serve(listener, app)
        .await
        .with_context(|| "Server shutdown unexpectedly")?;
    Ok(())
}
