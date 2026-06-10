mod handlers;
mod server;
mod state;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::EnvFilter;

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

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Server failed: {}", e);
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&args.log_level)),
        )
        .init();

    tracing::info!("Starting infers server");
    tracing::info!("Model: {}", args.model);
    tracing::info!("Parallelism: {:?}", args.parallelism);

    let state = std::sync::Arc::new(AppState {
        model_name: args.model.clone(),
    });

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
