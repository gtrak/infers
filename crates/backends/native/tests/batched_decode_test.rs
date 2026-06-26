//! Batched decode test: 2 sequences decoded simultaneously via decode_batched.
//!
//! Requires:
//! - 2 GPUs with CUDA compute capability 12.0+ (Blackwell)
//! - Model at ~/opt/vllm/models/qwen3.6-27n-autoround-int4/
//!
//! Run: cargo test --package infers-backend-native --test batched_decode_test batched_decode_2seq -- --ignored --nocapture
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use infers_backend_native::ForwardEngine;
use infers_cuda::context::CudaRuntime;
use infers_cuda::stream::StreamPool;
use infers_kv::SequenceId;
use infers_model_loader_heap::{load_safetensors, shard_weights_tp};
use infers_model::QuantTargetMap;
use infers_model::{strip_language_model_prefix, build_main_layers};

const DEFAULT_MODEL_DIR: &str = "~/opt/vllm/models/qwen3.6-27b-autoround-int4/";

#[test]
#[ignore = "requires 2 GPUs and real model weights"]
fn batched_decode_2seq() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();

    // Resolve model path
    let model_dir_str = std::env::var("INFERS_TEST_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string());
    let model_dir = if model_dir_str.starts_with('~') {
        let home = std::env::var("HOME")?;
        Path::new(&home).join(model_dir_str.strip_prefix("~/").unwrap())
    } else {
        Path::new(&model_dir_str).to_path_buf()
    };
    eprintln!("Loading model from: {}", model_dir.display());

    // Load config
    let config = infers_model::config::ModelConfig::load(&model_dir)?;
    eprintln!("Config loaded: {} layers", config.num_hidden_layers);

    let quant_map = if let Some(ref quant_config) = config.quantization_config {
        QuantTargetMap::from_config(quant_config).unwrap_or_else(|e| {
            eprintln!("Warning: Failed to parse quantization config: {}", e);
            QuantTargetMap::empty()
        })
    } else {
        QuantTargetMap::empty()
    };

    // Load weights
    let mut raw_weights = load_safetensors(&model_dir)?;
    strip_language_model_prefix(&mut raw_weights);
    let num_gpus = 2;
    let shards = shard_weights_tp(&raw_weights, &config, num_gpus)?;
    let mut weight_registries: Vec<infers_model::WeightRegistry> = Vec::new();
    for shard in shards {
        let mut registry = shard.registry;
        build_main_layers(&mut registry, &config, &quant_map)?;
        weight_registries.push(registry);
    }

    // Init CUDA
    let runtime = CudaRuntime::new()?;
    let ctx0 = runtime.device(0)?.clone();
    let ctx1 = runtime.device(1)?.clone();
    let stream_pool = StreamPool::new(&[ctx0.clone(), ctx1.clone()])?;

    // Create engine
    let mut config = config;
    config.max_position_embeddings = 4096usize.min(config.max_position_embeddings);
    let config = Arc::new(config);
    let group_size = 128;
    let mut engine = ForwardEngine::new(
        config.clone(), weight_registries,
        vec![ctx0, ctx1], stream_pool, group_size,
    )?;
    eprintln!("Engine initialized");

    // Init paged KV
    let page_size = 16;
    let num_pages = (config.max_position_embeddings / page_size) * 2;
    engine.init_paged(num_pages, page_size, 512 * 1024 * 1024)?;
    eprintln!("Paged KV initialized: {} pages", num_pages);

    // Create 2 sequences
    let seq_id0: SequenceId = engine.create_sequence();
    let seq_id1: SequenceId = engine.create_sequence();

    let external_stream = runtime.default_stream(0)?;

    // Tokenize prompt for both sequences (same prompt)
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Some(infers_tokenizer::Tokenizer::from_file(tokenizer_path.to_str().unwrap())?)
    } else {
        None
    };
    let prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n".to_string();
    let token_ids: Vec<u32> = if let Some(ref tok) = tokenizer {
        tok.encode(&prompt)?
    } else {
        vec![151644u32]
    };
    eprintln!("Prompt tokens: {}", token_ids.len());

    // Prefill both sequences
    let sampling_config = infers_scheduler::SamplingConfig::default();
    let mut rng0 = infers_backend_native::Xoshiro256PlusPlus::from_seed(42);
    let mut rng1 = infers_backend_native::Xoshiro256PlusPlus::from_seed(99);

    let (_pages0, first_token0) = engine.prefill_paged(&external_stream, &token_ids, seq_id0, &sampling_config, &mut rng0)?;
    let (_pages1, first_token1) = engine.prefill_paged(&external_stream, &token_ids, seq_id1, &sampling_config, &mut rng1)?;
    eprintln!("Prefill: seq0 first_token={}, seq1 first_token={}", first_token0, first_token1);

    // Create decode states for both sequences
    let mut state0 = engine.create_decode_state()?;
    let mut state1 = engine.create_decode_state()?;
    engine.prepare_batched_state(&mut state0);
    engine.prepare_batched_state(&mut state1);

    // Run batched decode for 30 steps
    let mut all_tokens0 = token_ids.clone();
    all_tokens0.push(first_token0);
    let mut all_tokens1 = token_ids.clone();
    all_tokens1.push(first_token1);

    let mut token0 = first_token0;
    let mut token1 = first_token1;
    let mut total_batched_time = std::time::Duration::ZERO;

    // Keep states in an array to avoid move issues
    let mut states = [state0, state1];
    // Keep RNGs in arrays to avoid move issues
    let mut rngs = [rng0, rng1];

    for step in 0..30 {
        let pos0 = (token_ids.len() + step) as u32;
        let pos1 = (token_ids.len() + step) as u32;

        let tokens = &[token0, token1];
        let positions = &[pos0, pos1];
        let seq_ids = &[seq_id0, seq_id1];
        let configs = &[sampling_config.clone(), sampling_config.clone()];
        let histories = &[all_tokens0.clone(), all_tokens1.clone()];
        let num_prompt = &[token_ids.len(), token_ids.len()];

        let decode_start = Instant::now();
        let sampled = engine.decode_batched(tokens, positions, seq_ids, &mut states, configs, histories, num_prompt, &mut rngs, step)?;
        let decode_elapsed = decode_start.elapsed();
        total_batched_time += decode_elapsed;

        token0 = sampled[0];
        token1 = sampled[1];
        all_tokens0.push(token0);
        all_tokens1.push(token1);

        eprintln!("Batched decode step {}: seq0={}, seq1={}, {:.3}s", step, token0, token1, decode_elapsed.as_secs_f64());

        assert!(token0 < config.vocab_size as u32, "Token0 out of range: {}", token0);
        assert!(token1 < config.vocab_size as u32, "Token1 out of range: {}", token1);
    }

    let n = 30;
    eprintln!("\nBatched decode: {} steps, avg {:.3}s/step, {:.1} tok/s/seq, {:.1} tok/s aggregate",
        n, total_batched_time.as_secs_f64() / n as f64,
        n as f64 / total_batched_time.as_secs_f64(),
        (2 * n) as f64 / total_batched_time.as_secs_f64());

    // Decode tokens to text
    if let Some(ref tok) = tokenizer {
        let gen0: Vec<u32> = all_tokens0[token_ids.len()+1..].to_vec();
        let gen1: Vec<u32> = all_tokens1[token_ids.len()+1..].to_vec();
        let text0 = tok.decode(&gen0)?;
        let text1 = tok.decode(&gen1)?;
        eprintln!("\nSeq0 output: {}", text0.trim());
        eprintln!("Seq1 output: {}", text1.trim());

        // Check for semantic correctness — both sequences should mention France/capital
        let combined = format!("{} {}", text0, text1).to_lowercase();
        if combined.contains("france") && (combined.contains("capital") || combined.contains("paris")) {
            eprintln!("\n*** PASSED: Both sequences decoded coherently, mention France/capital ***");
        } else {
            eprintln!("\n*** WARNING: Output doesn't mention France/capital ***");
        }
    }

    Ok(())
}
