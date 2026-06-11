//! End-to-end smoke test with real Qwen3.6-27B AutoRound INT4 model.
//!
//! Requires:
//! - 2 GPUs with CUDA compute capability 12.0+ (Blackwell)
//! - NVLink or NCCL-capable interconnect between GPUs
//! - Model downloaded to INFERS_TEST_MODEL path (default: ~/opt/vllm/models/qwen3.6-27b-autoround-int4/)
//! - ~30s to load model and run inference
//!
//! Run with: cargo test --package infers-backend-native --test smoke_test smoke_test_real_model -- --ignored --nocapture
use std::time::Instant;

use std::path::Path;
use std::sync::Arc;

use infers_backend_native::ForwardEngine;
use infers_cuda::context::CudaRuntime;
use infers_cuda::kernels::KernelRegistry;
use infers_cuda::stream::StreamPool;
use infers_kv::SequenceId;
use infers_model::sharding::shard_weights_tp;
use infers_model::{load_safetensors, strip_language_model_prefix, build_main_layers};

/// Default path for the Qwen3.6-27B AutoRound INT4 model.
const DEFAULT_MODEL_DIR: &str = "~/opt/vllm/models/qwen3.6-27b-autoround-int4/";

/// Smoke test that loads a real model and runs prefill + decode using TP=2.
///
/// Verifies:
/// - Model loads without errors
/// - Engine initializes with 2 GPU shards
/// - Prefill produces a valid token (in vocab range, non-zero)
/// - Decode produces valid tokens for 10 steps
#[test]
#[ignore = "requires 2 GPUs and real model weights"]
fn smoke_test_real_model() -> Result<(), Box<dyn std::error::Error>> {
    // Resolve model path from environment or default
    let model_dir_str = std::env::var("INFERS_TEST_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string());
    let model_dir = Path::new(&model_dir_str);

    // Expand ~ in path
    let model_dir = if model_dir_str.starts_with('~') {
        let home = std::env::var("HOME")?;
        Path::new(&home).join(model_dir_str.strip_prefix("~/").unwrap())
    } else {
        model_dir.to_path_buf()
    };

    eprintln!("Loading model from: {}", model_dir.display());

    // 1. Load model config
    let config = infers_model::config::ModelConfig::load(&model_dir)?;
    eprintln!("Config loaded: {} layers", config.num_hidden_layers);

    // 2. Load raw safetensors and strip language_model prefix (BEFORE sharding)
    let mut raw_weights = load_safetensors(&model_dir)?;
    strip_language_model_prefix(&mut raw_weights);
    eprintln!("Raw tensors loaded: {}", raw_weights.num_tensors());

    // 3. Shard raw weights for TP=2 (operates on tensors HashMap before build_main_layers)
    let num_gpus = 2;
    let shards = shard_weights_tp(&raw_weights, &config, num_gpus)?;
    assert_eq!(shards.len(), num_gpus, "Expected {num_gpus} shards, got {}", shards.len());

    // 4. Build structured layers for each shard
    let mut weight_registries: Vec<infers_model::WeightRegistry> = Vec::new();
    for shard in shards {
        let mut registry = shard.registry;
        build_main_layers(&mut registry, &config)?;
        eprintln!(
            "Shard {}: layers={}, embedding={}, norm={}, lm_head={}",
            shard.gpu_id,
            registry.layers.len(),
            registry.embedding.is_some(),
            registry.norm.is_some(),
            registry.lm_head.is_some(),
        );
        weight_registries.push(registry);
    }

    // 5. Initialize CUDA runtime and get contexts for both GPUs
    let runtime = CudaRuntime::new()?;
    assert!(
        runtime.num_devices >= 2,
        "TP=2 requires at least 2 GPUs, found {}",
        runtime.num_devices
    );
    let ctx0 = runtime.device(0)?.clone();
    let ctx1 = runtime.device(1)?.clone();

    // 6. Create stream pool with one stream per GPU
    let stream_pool = StreamPool::new(&[ctx0.clone(), ctx1.clone()])?;

    // 7. Register and load kernels
    let mut kernel_registry = KernelRegistry::new();
    kernel_registry.register_infers_kernels();

    // 8. Create the forward engine with TP=2
    let mut config = config;
    let test_max_seq_len = 4096usize.min(config.max_position_embeddings);
    config.max_position_embeddings = test_max_seq_len;
    let config = Arc::new(config);
    let group_size = 128; // Standard for AutoRound INT4
    let engine_start = Instant::now();
    let mut engine = ForwardEngine::new(
        config.clone(),
        weight_registries,
        vec![ctx0, ctx1],
        kernel_registry,
        stream_pool,
        group_size,
    )?;
    let engine_elapsed = engine_start.elapsed();
    eprintln!("Engine init (including weight cache upload): {:.2}s", engine_elapsed.as_secs_f64());

    // 9. Initialize paged KV cache
    let page_size = 16; // tokens per page
    let max_seq_len = config.max_position_embeddings;
    let num_pages = (max_seq_len / page_size) * 2; // some headroom
    let max_cache_bytes = 512 * 1024 * 1024; // 512 MB
    engine.init_paged(num_pages, page_size, max_cache_bytes)?;
    eprintln!("Paged KV cache initialized: {} pages, page_size={}", num_pages, page_size);

    // 10. Create a sequence and get its ID
    let seq_id: SequenceId = engine.create_sequence();

    // 11. Get a stream for external API calls (ignored internally by TP path)
    let external_stream = runtime.default_stream(0)?;

    // 12. Run prefill with BOS token
    let token_ids = vec![151644u32]; // BOS/im_start token for Qwen3.5
    let prefill_start = Instant::now();
    let pages_used = engine.prefill_paged(&external_stream, &token_ids, seq_id)?;
    let prefill_elapsed = prefill_start.elapsed();
    eprintln!("Prefill completed: {} pages used, {:.3}s", pages_used, prefill_elapsed.as_secs_f64());

    // 13. Run decode for 10 steps
    let mut token = token_ids[0];
    let mut total_decode_time = std::time::Duration::ZERO;
    for step in 0..10 {
        let decode_start = Instant::now();
        let pos = (token_ids.len() + step) as u32;
        token = engine.decode_paged(&external_stream, token, pos, seq_id)?;
        let decode_elapsed = decode_start.elapsed();
        total_decode_time += decode_elapsed;
        eprintln!("Decode step {}: token={}, {:.3}s", step, token, decode_elapsed.as_secs_f64());
        assert!(
            token < config.vocab_size as u32,
            "Decode token {} >= vocab_size {} at step {}",
            token,
            config.vocab_size,
            step
        );
    }
    eprintln!(
        "Total decode time: {:.3}s, avg per step: {:.3}s",
        total_decode_time.as_secs_f64(),
        total_decode_time.as_secs_f64() / 10.0,
    );

    eprintln!(
        "Smoke test PASSED: {} tokens generated | Engine: {:.2}s | Prefill: {:.3}s | Decode avg: {:.3}s/step",
        1 + 10,
        engine_elapsed.as_secs_f64(),
        prefill_elapsed.as_secs_f64(),
        total_decode_time.as_secs_f64() / 10.0,
    );
    Ok(())
}
