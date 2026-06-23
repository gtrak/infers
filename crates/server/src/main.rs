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
use infers_cuda::stream::StreamPool;
use infers_kv::PagedKvManager;
use infers_model::mmap::{load_safetensors_mmap, strip_language_model_prefix_mmap, shard_weights_tp_mmap};
use infers_model::{build_main_layers, build_metadata_registry, ModelConfig, QuantTargetMap, WeightRegistry};
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
    let use_mmap = model_path.exists();

    let (model_config, engine, num_layers) = if use_mmap {
        // Mmap path: load via zero-copy memory-mapped access
        tracing::info!("Loading model from {} with TP={} (mmap)", args.model, num_gpus);
        let config = infers_model::config::ModelConfig::load(model_path)
            .with_context(|| format!("Failed to load model config from {}", args.model))?;
        let mut mmap_reg = load_safetensors_mmap(model_path)
            .with_context(|| format!("Failed to load safetensors (mmap) from {}", args.model))?;
        strip_language_model_prefix_mmap(&mut mmap_reg);
        let shards = shard_weights_tp_mmap(&mmap_reg, &config, num_gpus)
            .with_context(|| format!("Failed to shard weights (mmap) for TP={num_gpus}"))?;
        let quant_map = if let Some(ref quant_config) = config.quantization_config {
            QuantTargetMap::from_config(quant_config)
                .unwrap_or_else(|e| {
                    tracing::warn!("Failed to parse quantization config, using empty map: {}", e);
                    QuantTargetMap::empty()
                })
        } else {
            QuantTargetMap::empty()
        };
        let mut mmap_registries = Vec::new();
        let mut metadata_registries = Vec::new();
        for shard in shards {
            mmap_registries.push(shard.registry.clone());
            let mut meta_registry = build_metadata_registry(&shard.registry);
            build_main_layers(&mut meta_registry, &config, &quant_map)
                .with_context(|| format!("Failed to build metadata layers for GPU {}", shard.gpu_id))?;
            metadata_registries.push(meta_registry);
        }

        let num_layers = config.num_hidden_layers;
        let mut pinned = infers_cuda::PinnedHostBuffer::new(256 * 1024 * 1024)
            .context("Failed to allocate pinned host buffer")?;

        let model_config = Arc::new(config);
        let engine = ForwardEngine::new_from_mmap(
            model_config.clone(),
            mmap_registries,
            metadata_registries,
            contexts,
            streams,
            &mut pinned,
            128,
        ).context("Failed to create ForwardEngine (mmap)")?;

        (model_config, engine, num_layers)
    } else {
        // No model path: create minimal config for wiring
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
        let num_layers = config.num_hidden_layers;

        let model_config = Arc::new(config);
        let engine = ForwardEngine::new(
            model_config.clone(),
            vec![weights],
            contexts,
            streams,
            128,
        ).context("Failed to create ForwardEngine")?;

        (model_config, engine, num_layers)
    };

    // Step 4: Create PagedKvManager for the scheduler

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

    // Step 5: Create RoundRobinScheduler
    let max_concurrent_sessions = 4;
    let max_batch_size = 4;
    let scheduler = RoundRobinScheduler::new(
        max_concurrent_sessions,
        max_batch_size,
        kv_manager,
    );

    // Step 6: Create BackendEvictionStore
    let eviction_store = BackendEvictionStore::new(num_layers);

    // Step 7: Create InferenceOrchestrator

    let orchestrator = InferenceOrchestrator::new(
        scheduler,
        engine,
        eviction_store,
        stream,
    );

    // Step 8: Create Tokenizer
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

    // Step 9: Build AppState
    let state = Arc::new(AppState {
        model_name: args.model.clone(),
        orchestrator: Arc::new(Mutex::new(orchestrator)),
        tokenizer,
    });

    // Step 10: Spawn background scheduler loop
    server::spawn_scheduler_loop(state.orchestrator.clone());

    // Step 11: Build and start HTTP server
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
