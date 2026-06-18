//! Standalone one-shot inference binary.
//!
//! Loads a model, runs prefill + decode with optional probe dumps.
//! No server, no scheduler — just raw inference with TP sharding.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use infers_backend_native::ForwardEngine;
use infers_cuda::context::CudaRuntime;
use infers_cuda::kernels::KernelRegistry;
use infers_cuda::stream::StreamPool;
use infers_model::config::ModelConfig;
use infers_model::sharding::shard_weights_tp;
use infers_model::{load_safetensors, strip_language_model_prefix, build_main_layers};

/// Standalone inference binary for prefill + decode.
#[derive(Parser, Debug)]
#[command(name = "infer", about = "One-shot inference with optional probe dumps")]
struct Args {
    /// Path to the model directory (containing config.json and safetensors)
    #[arg(short, long)]
    model: String,

    /// Number of GPUs for tensor parallelism
    #[arg(short, long, default_value = "2")]
    tp: usize,

    /// Input prompt text (tokenized locally)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Comma-separated token IDs to use directly (bypasses tokenizer)
    #[arg(long, conflicts_with = "prompt")]
    token_ids: Option<String>,

    /// Maximum number of tokens to generate (exclusive of prompt)
    #[arg(short, long, default_value = "64")]
    max_tokens: usize,

    /// Directory for probe dump files (enables dumping when set)
    #[arg(long)]
    dump_dir: Option<String>,

    /// Comma-separated layer indices to dump (e.g. "0,3,10"), or "all"
    #[arg(long)]
    dump_layers: Option<String>,

    /// Group size for quantization (default: 128 for AutoRound INT4)
    #[arg(long, default_value = "128")]
    group_size: usize,

    /// Override max sequence length (default: min(4096, config.max_position_embeddings))
    #[arg(long)]
    max_seq_len: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // --- Resolve model path (expand ~) ---
    let model_dir = if args.model.starts_with('~') {
        let home = std::env::var("HOME").unwrap_or_default();
        Path::new(&home).join(args.model.strip_prefix("~/").unwrap())
    } else {
        Path::new(&args.model).to_path_buf()
    };

    eprintln!("=== Standalone Inference ===");
    eprintln!("Model: {}", model_dir.display());
    eprintln!("TP: {}, Max tokens: {}", args.tp, args.max_tokens);

    // --- Enable probe dumps if requested (must be set BEFORE engine creation) ---
    if let Some(ref dump_dir) = args.dump_dir {
        unsafe { std::env::set_var("INFERS_DUMP_DIR", dump_dir) };
        eprintln!("Probe dump dir: {}", dump_dir);
    }
    if let Some(ref dump_layers) = args.dump_layers {
        unsafe { std::env::set_var("INFERS_DUMP_LAYERS", dump_layers) };
        eprintln!("Probe dump layers: {}", dump_layers);
    }

    // --- 1. Load model config ---
    let mut config = ModelConfig::load(&model_dir)?;
    eprintln!("Config loaded: {} layers, vocab_size={}", config.num_hidden_layers, config.vocab_size);

    // Optionally override max position embeddings
    if let Some(max_seq_len) = args.max_seq_len {
        config.max_position_embeddings = max_seq_len;
    } else {
        let default_max = 4096usize.min(config.max_position_embeddings);
        config.max_position_embeddings = default_max;
        eprintln!("Using max_seq_len={}", default_max);
    }

    // --- 2. Load raw safetensors and strip language_model prefix ---
    let mut raw_weights = load_safetensors(&model_dir)?;
    strip_language_model_prefix(&mut raw_weights);
    eprintln!("Raw tensors loaded: {}", raw_weights.num_tensors());

    // --- 3. Shard weights for TP ---
    let num_gpus = args.tp;
    let shards = shard_weights_tp(&raw_weights, &config, num_gpus)?;
    assert_eq!(shards.len(), num_gpus, "Expected {num_gpus} shards, got {}", shards.len());

    // --- 4. Build per-GPU weight registries ---
    let mut weight_registries: Vec<infers_model::WeightRegistry> = Vec::new();
    for shard in shards {
        let gpu_id = shard.gpu_id;
        let mut registry = shard.registry;
        build_main_layers(&mut registry, &config)?;
        eprintln!(
            "Shard {}: layers={}, embedding={}, norm={}, lm_head={}",
            gpu_id,
            registry.layers.len(),
            registry.embedding.is_some(),
            registry.norm.is_some(),
            registry.lm_head.is_some(),
        );
        weight_registries.push(registry);
    }

    // --- 5. Initialize CUDA runtime ---
    let runtime = CudaRuntime::new()?;
    if runtime.num_devices < num_gpus {
        anyhow::bail!(
            "TP={} requires at least {} GPUs, found {}",
            num_gpus,
            num_gpus,
            runtime.num_devices
        );
    }

    // Collect device contexts
    let mut gpu_contexts = Vec::with_capacity(num_gpus);
    for i in 0..num_gpus {
        let ctx = runtime.device(i)?.clone();
        eprintln!("GPU {}: available", i);
        gpu_contexts.push(ctx);
    }

    // --- 6. Create stream pool ---
    let stream_pool = StreamPool::new(&gpu_contexts)?;

    // --- 7. Register and load kernels ---
    let mut kernel_registry = KernelRegistry::new();
    kernel_registry.register_infers_kernels();

    // --- 8. Create the forward engine ---
    let config = Arc::new(config);
    let engine_start = Instant::now();
    let mut engine = ForwardEngine::new(
        config.clone(),
        weight_registries,
        gpu_contexts,
        kernel_registry,
        stream_pool,
        args.group_size,
    )?;
    let engine_elapsed = engine_start.elapsed();
    eprintln!("Engine init (including weight cache upload): {:.2}s", engine_elapsed.as_secs_f64());

    // --- 9. Initialize paged KV cache ---
    let page_size = 16;
    let max_seq_len = config.max_position_embeddings;
    let num_pages = (max_seq_len / page_size) * 4; // headroom for multiple sequences
    let max_cache_bytes = 512 * 1024 * 1024; // 512 MB
    engine.init_paged(num_pages, page_size, max_cache_bytes)?;
    eprintln!("Paged KV cache: {} pages, page_size={}", num_pages, page_size);

    // --- 10. Create sequence and get external stream ---
    let seq_id = engine.create_sequence();
    let external_stream = runtime.default_stream(0)?;

  // --- 11. Load tokenizer (for decoding output) and get prompt tokens ---
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Some(infers_tokenizer::Tokenizer::from_file(tokenizer_path.to_str().unwrap())?)
    } else {
        None
    };

    let token_ids: Vec<u32> = if let Some(ref ids_str) = args.token_ids {
        ids_str.split(',')
            .map(|s| u32::from_str_radix(s.trim(), 10).unwrap())
            .collect()
    } else if let Some(ref prompt) = args.prompt {
        if let Some(ref tok) = tokenizer {
            tok.encode(prompt)?
        } else {
            eprintln!("No tokenizer found, using fallback token [151644]");
            vec![151644u32]
        }
    } else {
        anyhow::bail!("either --prompt or --token-ids must be specified");
    };
    eprintln!("Prompt tokens ({}): {:?}", token_ids.len(), &token_ids[..10.min(token_ids.len())]);

    // --- 12. Run prefill ---
    let prefill_start = Instant::now();
    let pages_used = engine.prefill_paged(&external_stream, &token_ids, seq_id)?;
    let prefill_elapsed = prefill_start.elapsed();
    eprintln!(
        "Prefill: {} pages used, {:.3}s",
        pages_used,
        prefill_elapsed.as_secs_f64()
    );

    // --- 13. Run decode ---
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut total_decode_time = std::time::Duration::ZERO;

    for step in 0..args.max_tokens {
        let decode_start = Instant::now();
        let pos = (token_ids.len() + step) as u32;
        let token = engine.decode_paged(&external_stream, token_ids[0], pos, seq_id)?;
        let decode_elapsed = decode_start.elapsed();
        total_decode_time += decode_elapsed;
        generated_tokens.push(token);

        // Print per-step info (verbose to stderr)
        eprintln!(
            "Decode step {}: token={}, {:.3}s",
            step,
            token,
            decode_elapsed.as_secs_f64()
        );

        assert!(
            token < config.vocab_size as u32,
            "Decode token {} >= vocab_size {} at step {}",
            token,
            config.vocab_size,
            step
        );
    }

    let num_decode = generated_tokens.len();
    eprintln!(
        "Total decode time: {:.3}s, avg per step: {:.3}s",
        total_decode_time.as_secs_f64(),
        total_decode_time.as_secs_f64() / num_decode as f64,
    );

    // --- 14. Decode tokens to text ---
    if let Some(ref tok) = tokenizer {
        let text = tok.decode(&generated_tokens)?;
        println!("--- Generated text ({} tokens) ---", generated_tokens.len());
        println!("{}", text.trim());
    } else {
        println!(
            "--- Generated tokens ({}) ---\n{:?}",
            generated_tokens.len(),
            generated_tokens
        );
    }

    // --- Summary ---
    eprintln!(
        "\n=== Inference complete ===",
    );
    eprintln!(
        "Engine: {:.2}s | Prefill: {:.3}s | Decode avg: {:.3}s/step",
        engine_elapsed.as_secs_f64(),
        prefill_elapsed.as_secs_f64(),
        total_decode_time.as_secs_f64() / num_decode as f64,
    );

    Ok(())
}
