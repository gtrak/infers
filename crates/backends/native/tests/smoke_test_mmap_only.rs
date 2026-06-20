//! Minimal smoke tests for heap-only and mmap-only paths, without cross-comparison.
//!
//! Each test loads a real model via exactly one path (heap or mmap), runs
//! prefill + 30 decode steps, prints the generated text, and asserts all
//! tokens are within vocab range. This eliminates the complexity of running
//! both engines sequentially in the same process.
//!
//! Requires:
//! - 2 GPUs with CUDA compute capability 12.0+ (Blackwell)
//! - NVLink or NCCL-capable interconnect between GPUs
//! - Model downloaded to INFERS_TEST_MODEL path (default: ~/opt/vllm/models/qwen3.6-27b-autoround-int4/)
//!
//! Run heap-only:  cargo test --package infers-backend-native --test smoke_test_mmap_only smoke_test_heap_only -- --ignored --nocapture
//! Run mmap-only:  cargo test --package infers-backend-native --test smoke_test_mmap_only smoke_test_mmap_only -- --ignored --nocapture

use std::path::Path;
use std::sync::Arc;

use half::bf16;
use infers_backend_native::ForwardEngine;
use infers_cuda::context::CudaRuntime;
use infers_cuda::kernels::KernelRegistry;
use infers_cuda::stream::StreamPool;
use infers_kv::SequenceId;
use infers_model::sharding::shard_weights_tp;
use infers_model::{load_safetensors, strip_language_model_prefix, build_main_layers};

/// Default path for the Qwen3.6-27B AutoRound INT4 model.
const DEFAULT_MODEL_DIR: &str = "~/opt/vllm/models/qwen3.6-27b-autoround-int4/";

/// Resolve model directory from environment or default, expanding `~`.
fn resolve_model_dir() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let model_dir_str = std::env::var("INFERS_TEST_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string());
    if model_dir_str.starts_with('~') {
        let home = std::env::var("HOME")?;
        Ok(Path::new(&home).join(model_dir_str.strip_prefix("~/").unwrap()))
    } else {
        Ok(Path::new(&model_dir_str).to_path_buf())
    }
}

/// Shared prompt text used by both tests.
fn build_prompt() -> String {
    "user\nWhat is the capital of France?\n\n".to_string()
}

/// Read current process RSS (resident set size) in bytes from /proc/self/status.
fn current_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                // VmRSS is in kB
                let kb: u64 = parts[1].parse().ok()?;
                return Some(kb * 1024);
            }
        }
    }
    None
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    } else {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    }
}

/// Run prefill + 30 decode steps on an initialized engine.
/// Returns (first_token, generated_tokens).
fn run_prefill_decode(
    engine: &mut ForwardEngine,
    runtime: &CudaRuntime,
    config: Arc<infers_model::config::ModelConfig>,
    token_ids: &[u32],
) -> Result<(u32, Vec<u32>), Box<dyn std::error::Error>> {
    // Initialize paged KV cache
    let page_size = 16;
    let num_pages = (config.max_position_embeddings / page_size) * 2;
    engine.init_paged(num_pages, page_size, 512 * 1024 * 1024)?;

    // Create a sequence
    let seq_id: SequenceId = engine.create_sequence();

    // Get external stream (GPU 0)
    let ext_stream = runtime.default_stream(0)?;

    // Sampling config + RNG
    let sampling_config = infers_scheduler::SamplingConfig::default();
    let mut rng = infers_backend_native::Xoshiro256PlusPlus::from_seed(42);

    // Prefill
    let (_, first_token) = engine.prefill_paged(&ext_stream, token_ids, seq_id, &sampling_config, &mut rng)?;

    // Decode for 30 steps
    let mut generated: Vec<u32> = Vec::with_capacity(30);
    let mut all_tokens = token_ids.to_vec();
    all_tokens.push(first_token);
    let mut current = first_token;
    for step in 0..30 {
        let pos = (token_ids.len() + step) as u32;
        current = engine.decode_paged(&ext_stream, current, pos, seq_id, &sampling_config, &all_tokens, token_ids.len(), &mut rng)?;
        all_tokens.push(current);
        assert!(current < config.vocab_size as u32, "Token {} >= vocab_size at step {}", current, step);
        generated.push(current);
    }

    Ok((first_token, generated))
}

/// Heap-only smoke test: load model via heap path, run prefill + 30 decode steps.
#[test]
#[ignore = "requires 2 GPUs and real model weights"]
fn smoke_test_heap_only() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = resolve_model_dir()?;
    eprintln!("[heap-only] Loading model from: {}", model_dir.display());

    // 1. Load config
    let config = infers_model::config::ModelConfig::load(&model_dir)?;
    eprintln!("[heap-only] Config loaded: {} layers", config.num_hidden_layers);
    let rss = current_rss_bytes().unwrap_or(0);
    eprintln!("[heap-only] RSS after config: {}", format_bytes(rss));
    // 2. Load safetensors via heap path
    let mut raw_weights = load_safetensors(&model_dir)?;
    strip_language_model_prefix(&mut raw_weights);
    eprintln!("[heap-only] Raw tensors loaded (heap): {}", raw_weights.num_tensors());

    // 3. Shard for TP=2
    let num_gpus = 2;
    let shards = shard_weights_tp(&raw_weights, &config, num_gpus)?;
    assert_eq!(shards.len(), num_gpus);

    // 4. Build structured layers per shard
    let mut weight_registries: Vec<infers_model::WeightRegistry> = Vec::new();
    for shard in shards {
        let mut registry = shard.registry;
        build_main_layers(&mut registry, &config)?;
        weight_registries.push(registry);
    }

    let rss = current_rss_bytes().unwrap_or(0);
    eprintln!("[heap-only] RSS after load+shard: {}", format_bytes(rss));

    // 5. CUDA init
    let runtime = CudaRuntime::new()?;
    assert!(runtime.num_devices >= 2, "TP=2 requires at least 2 GPUs");
    let ctx0 = runtime.device(0)?.clone();
    let ctx1 = runtime.device(1)?.clone();

    let streams = StreamPool::new(&[ctx0.clone(), ctx1.clone()])?;

    let mut kr = KernelRegistry::new();
    kr.register_infers_kernels();

    // 6. Limit max_position_embeddings for the test
    let mut config = config;
    config.max_position_embeddings = 4096.min(config.max_position_embeddings);
    let config = Arc::new(config);

    // 7. Create engine via heap path
    eprintln!("[heap-only] Creating ForwardEngine...");
    let group_size = 128;
    let mut engine = ForwardEngine::new(
        config.clone(),
        weight_registries,
        vec![ctx0, ctx1],
        kr,
        streams,
        group_size,
    )?;
    eprintln!("[heap-only] Engine created");
    let rss = current_rss_bytes().unwrap_or(0);
    eprintln!("[heap-only] RSS after engine: {}", format_bytes(rss));

    // 8. Tokenize prompt
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Some(infers_tokenizer::Tokenizer::from_file(tokenizer_path.to_str().unwrap())?)
    } else {
        None
    };
    let prompt = build_prompt();
    let token_ids: Vec<u32> = if let Ok(ids_str) = std::env::var("INFERS_TEST_TOKEN_IDS") {
        ids_str.split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    } else if let Some(ref tok) = tokenizer {
        tok.encode(&prompt)?
    } else {
        vec![151644u32]
    };
    eprintln!("[heap-only] Prompt tokens: {}", token_ids.len());

    // 9. Run inference
    let (first_token, generated) = run_prefill_decode(&mut engine, &runtime, config.clone(), &token_ids)?;
    eprintln!("[heap-only] First token: {}", first_token);
    eprintln!("[heap-only] Tokens: {:?}", generated);

    // 10. Decode to text
    if let Some(ref tok) = tokenizer {
        let text = tok.decode(&generated)?;
        eprintln!("[heap-only] Generated text: {}", text.trim());
    }
    let rss = current_rss_bytes().unwrap_or(0);
    eprintln!("[heap-only] RSS after inference: {}", format_bytes(rss));

    Ok(())
}

/// Mmap-only smoke test: load model via mmap path, run prefill + 30 decode steps.
#[test]
#[ignore = "requires 2 GPUs and real model weights"]
fn smoke_test_mmap_only() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = resolve_model_dir()?;
    eprintln!("[mmap-only] Loading model from: {}", model_dir.display());

    // 1. Load config
    let config = infers_model::config::ModelConfig::load(&model_dir)?;
    eprintln!("[mmap-only] Config loaded: {} layers", config.num_hidden_layers);
    let rss = current_rss_bytes().unwrap_or(0);
    eprintln!("[mmap-only] RSS after config: {}", format_bytes(rss));

    // 2. Load safetensors via mmap path
    use infers_model::mmap::{load_safetensors_mmap, strip_language_model_prefix_mmap, shard_weights_tp_mmap, build_metadata_registry};

    let mut mmap_reg = load_safetensors_mmap(&model_dir)?;

    strip_language_model_prefix_mmap(&mut mmap_reg);

    // 3. Shard for TP=2
    let num_gpus = 2;
    let shards = shard_weights_tp_mmap(&mmap_reg, &config, num_gpus)?;
    assert_eq!(shards.len(), num_gpus);


    // 4. Build metadata registries per shard
    let mut mmap_registries: Vec<infers_model::MmapWeightRegistry> = Vec::new();
    let mut metadata_registries: Vec<infers_model::WeightRegistry> = Vec::new();
    for shard in shards {
        mmap_registries.push(shard.registry.clone());
        let mut meta = build_metadata_registry(&shard.registry);
        build_main_layers(&mut meta, &config)?;
        metadata_registries.push(meta);
    }

    let rss = current_rss_bytes().unwrap_or(0);
    eprintln!("[mmap-only] RSS after load+shard: {}", format_bytes(rss));

    // 5. CUDA init
    let runtime = CudaRuntime::new()?;
    assert!(runtime.num_devices >= 2, "TP=2 requires at least 2 GPUs");
    let ctx0 = runtime.device(0)?.clone();
    let ctx1 = runtime.device(1)?.clone();

    let streams = StreamPool::new(&[ctx0.clone(), ctx1.clone()])?;

    let mut kr = KernelRegistry::new();
    kr.register_infers_kernels();

    // 6. Limit max_position_embeddings for the test
    let mut config = config;
    config.max_position_embeddings = 4096.min(config.max_position_embeddings);
    let config = Arc::new(config);

    // 7. Pinned buffer + create engine via mmap path
    eprintln!("[mmap-only] Creating ForwardEngine (mmap)...");
    let group_size = 128;
    let mut pinned = infers_cuda::PinnedHostBuffer::new(256 * 1024 * 1024)?;
    let mut engine = ForwardEngine::new_from_mmap(
        config.clone(),
        mmap_registries,
        metadata_registries,
        vec![ctx0, ctx1],
        kr,
        streams,
        &mut pinned,
        group_size,
    )?;
    eprintln!("[mmap-only] Engine created");
    let rss = current_rss_bytes().unwrap_or(0);
    eprintln!("[mmap-only] RSS after engine: {}", format_bytes(rss));

    // 8. Tokenize prompt
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Some(infers_tokenizer::Tokenizer::from_file(tokenizer_path.to_str().unwrap())?)
    } else {
        None
    };
    let prompt = build_prompt();
    let token_ids: Vec<u32> = if let Ok(ids_str) = std::env::var("INFERS_TEST_TOKEN_IDS") {
        ids_str.split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    } else if let Some(ref tok) = tokenizer {
        tok.encode(&prompt)?
    } else {
        vec![151644u32]
    };
    eprintln!("[mmap-only] Prompt tokens: {}", token_ids.len());

    // 9. Run inference
    let (first_token, generated) = run_prefill_decode(&mut engine, &runtime, config.clone(), &token_ids)?;
    eprintln!("[mmap-only] First token: {}", first_token);
    eprintln!("[mmap-only] Tokens: {:?}", generated);

    // 10. Decode to text
    if let Some(ref tok) = tokenizer {
        let text = tok.decode(&generated)?;
        eprintln!("[mmap-only] Generated text: {}", text.trim());
    }
    let rss = current_rss_bytes().unwrap_or(0);
    eprintln!("[mmap-only] RSS after inference: {}", format_bytes(rss));

    Ok(())
}

/// Diagnostic test: compare INT4 GPU data between heap and mmap engines.
///
/// Loads the model via both paths sequentially (StreamPool is not clonable,
/// NCCL can only be initialized once per process). Creates the heap engine
/// first, downloads an INT4 qweight from its GPU 0 cache, drops it, then
/// creates the mmap engine and compares. Compares both a row-parallel weight
/// (contiguous) and a column-parallel weight (may be strided in mmap path).
#[test]
#[ignore = "requires 2 GPUs and real model weights"]
fn smoke_test_mmap_vs_heap_gpu_data() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = resolve_model_dir()?;
    eprintln!("[mmap-vs-heap] Loading model from: {}", model_dir.display());

    // 1. Load config
    let config = infers_model::config::ModelConfig::load(&model_dir)?;
    eprintln!("[mmap-vs-heap] Config loaded: {} layers", config.num_hidden_layers);

    // ---- Heap path: load, create engine, download GPU data, drop engine ----
    eprintln!("[mmap-vs-heap] Loading via heap path...");
    let mut raw_weights = load_safetensors(&model_dir)?;
    strip_language_model_prefix(&mut raw_weights);
    eprintln!("[mmap-vs-heap] Raw tensors loaded (heap): {}", raw_weights.num_tensors());

    let num_gpus = 2;
    let shards = shard_weights_tp(&raw_weights, &config, num_gpus)?;
    assert_eq!(shards.len(), num_gpus);

    let mut weight_registries: Vec<infers_model::WeightRegistry> = Vec::new();
    for shard in shards {
        let mut registry = shard.registry;
        build_main_layers(&mut registry, &config)?;
        weight_registries.push(registry);
    }

    eprintln!("[mmap-vs-heap] Initializing CUDA (heap)...");
    let runtime_heap = CudaRuntime::new()?;
    assert!(runtime_heap.num_devices >= 2, "TP=2 requires at least 2 GPUs");
    let ctx0_h = runtime_heap.device(0)?.clone();
    let ctx1_h = runtime_heap.device(1)?.clone();

    let streams_heap = StreamPool::new(&[ctx0_h.clone(), ctx1_h.clone()])?;

    let mut kr_heap = KernelRegistry::new();
    kr_heap.register_infers_kernels();

    // Limit max_position_embeddings for the test
    let mut config = config;
    config.max_position_embeddings = 4096.min(config.max_position_embeddings);
    let config = Arc::new(config);

    eprintln!("[mmap-vs-heap] Creating ForwardEngine (heap)...");
    let group_size = 128;
    let engine_heap = ForwardEngine::new(
        config.clone(),
        weight_registries,
        vec![ctx0_h, ctx1_h],
        kr_heap,
        streams_heap,
        group_size,
    )?;
    eprintln!("[mmap-vs-heap] Heap engine created");

    // Download INT4 qweights from heap engine's GPU 0 cache
    let gpu0_stream_h = runtime_heap.default_stream(0)?;
    let row_parallel_name = "layers.63.self_attn.o_proj.qweight";
    let col_parallel_name = "layers.63.mlp.gate_proj.qweight";

    eprintln!("[mmap-vs-heap] Downloading from heap engine GPU 0 cache...");
    let heap_row_data = engine_heap.weight_caches()[0]
        .download_int4_qweight(row_parallel_name, &gpu0_stream_h);
    let heap_col_data = engine_heap.weight_caches()[0]
        .download_int4_qweight(col_parallel_name, &gpu0_stream_h);

    eprintln!("[mmap-vs-heap] Heap data for '{}': {:?}", row_parallel_name,
        heap_row_data.as_ref().map(|d| d.len()));
    eprintln!("[mmap-vs-heap] Heap data for '{}': {:?}", col_parallel_name,
        heap_col_data.as_ref().map(|d| d.len()));

    // Download scales and qzeros from heap engine
    let heap_row_scales = engine_heap.weight_caches()[0]
        .download_int4_scales(row_parallel_name, &gpu0_stream_h);
    let heap_col_scales = engine_heap.weight_caches()[0]
        .download_int4_scales(col_parallel_name, &gpu0_stream_h);
    let heap_row_qzeros = engine_heap.weight_caches()[0]
        .download_int4_qzeros(row_parallel_name, &gpu0_stream_h);
    let heap_col_qzeros = engine_heap.weight_caches()[0]
        .download_int4_qzeros(col_parallel_name, &gpu0_stream_h);

    eprintln!("[mmap-vs-heap] Heap scales for '{}': {:?}", row_parallel_name,
        heap_row_scales.as_ref().map(|d| d.len()));
    eprintln!("[mmap-vs-heap] Heap qzeros for '{}': {:?}", row_parallel_name,
        heap_row_qzeros.as_ref().map(|d| d.len()));

    // Collect heap cache key data before dropping the engine (both engines can't coexist)
    let heap_cache_keys: Vec<String> = engine_heap.weight_caches()[0].keys().map(|s| s.to_string()).collect();
    let mut heap_key_types: Vec<(String, &'static str)> = Vec::new();
    for key in &heap_cache_keys {
        let wtype = match engine_heap.weight_caches()[0].get(key) {
            Some(infers_backend_native::CachedWeight::Bf16(_)) => "Bf16",
            Some(infers_backend_native::CachedWeight::Int4(_)) => "Int4",
            None => "???",
        };
        heap_key_types.push((key.to_string(), wtype));
    }

    // Download BF16 conv1d weights from heap engine before dropping it
    let conv1d_0_name = "layers.0.linear_attn.conv1d.weight";
    let conv1d_30_name = "layers.30.linear_attn.conv1d.weight";
    eprintln!("[mmap-vs-heap] Downloading BF16 conv1d weights from heap engine...");
    let heap_conv1d_0 = engine_heap.weight_caches()[0].download_bf16(conv1d_0_name, &gpu0_stream_h);
    let heap_conv1d_30 = engine_heap.weight_caches()[0].download_bf16(conv1d_30_name, &gpu0_stream_h);
    eprintln!("[mmap-vs-heap] Heap conv1d weight (layer 0): {:?}", heap_conv1d_0.as_ref().map(|d| d.len()));
    eprintln!("[mmap-vs-heap] Heap conv1d weight (layer 30): {:?}", heap_conv1d_30.as_ref().map(|d| d.len()));

    // Drop the heap engine to free GPU memory before creating mmap engine
    drop(engine_heap);
    drop(runtime_heap);
    eprintln!("[mmap-vs-heap] Heap engine and runtime dropped");

    // ---- Mmap path: load, create engine, download GPU data, compare ----
    eprintln!("[mmap-vs-heap] Initializing CUDA (mmap)...");
    let runtime_mmap = CudaRuntime::new()?;
    assert!(runtime_mmap.num_devices >= 2, "TP=2 requires at least 2 GPUs");
    let ctx0_m = runtime_mmap.device(0)?.clone();
    let ctx1_m = runtime_mmap.device(1)?.clone();

    let streams_mmap = StreamPool::new(&[ctx0_m.clone(), ctx1_m.clone()])?;

    let mut kr_mmap = KernelRegistry::new();
    kr_mmap.register_infers_kernels();

    use infers_model::mmap::{
        load_safetensors_mmap, strip_language_model_prefix_mmap, shard_weights_tp_mmap, build_metadata_registry,
    };

    eprintln!("[mmap-vs-heap] Loading via mmap path...");
    let mut mmap_reg = load_safetensors_mmap(&model_dir)?;
    eprintln!("[mmap-vs-heap] Mmap tensors loaded: {}", mmap_reg.tensors.len());
    strip_language_model_prefix_mmap(&mut mmap_reg);

    let shards_mmap = shard_weights_tp_mmap(&mmap_reg, &config, num_gpus)?;
    assert_eq!(shards_mmap.len(), num_gpus);

    let mut mmap_registries: Vec<infers_model::MmapWeightRegistry> = Vec::new();
    let mut metadata_registries: Vec<infers_model::WeightRegistry> = Vec::new();
    for shard in shards_mmap {
        mmap_registries.push(shard.registry.clone());
        let mut meta = build_metadata_registry(&shard.registry);
        build_main_layers(&mut meta, &config)?;
        metadata_registries.push(meta);
    }

    eprintln!("[mmap-vs-heap] Creating ForwardEngine (mmap)...");
    let mut pinned = infers_cuda::PinnedHostBuffer::new(256 * 1024 * 1024)?;
    let engine_mmap = ForwardEngine::new_from_mmap(
        config.clone(),
        mmap_registries,
        metadata_registries,
        vec![ctx0_m, ctx1_m],
        kr_mmap,
        streams_mmap,
        &mut pinned,
        group_size,
    )?;
    eprintln!("[mmap-vs-heap] Mmap engine created");

    let gpu0_stream_m = runtime_mmap.default_stream(0)?;

    // Download BF16 conv1d weights from mmap engine
    eprintln!("[mmap-vs-heap] Downloading BF16 conv1d weights from mmap engine...");
    let mmap_conv1d_0 = engine_mmap.weight_caches()[0].download_bf16(conv1d_0_name, &gpu0_stream_m);
    let mmap_conv1d_30 = engine_mmap.weight_caches()[0].download_bf16(conv1d_30_name, &gpu0_stream_m);
    eprintln!("[mmap-vs-heap] Mmap conv1d weight (layer 0): {:?}", mmap_conv1d_0.as_ref().map(|d| d.len()));
    eprintln!("[mmap-vs-heap] Mmap conv1d weight (layer 30): {:?}", mmap_conv1d_30.as_ref().map(|d| d.len()));

    eprintln!("[mmap-vs-heap] Downloading from mmap engine GPU 0 cache...");
    let mmap_row_data = engine_mmap.weight_caches()[0]
        .download_int4_qweight(row_parallel_name, &gpu0_stream_m);
    let mmap_col_data = engine_mmap.weight_caches()[0]
        .download_int4_qweight(col_parallel_name, &gpu0_stream_m);

    eprintln!("[mmap-vs-heap] Mmap data for '{}': {:?}", row_parallel_name,
        mmap_row_data.as_ref().map(|d| d.len()));
    eprintln!("[mmap-vs-heap] Mmap data for '{}': {:?}", col_parallel_name,
        mmap_col_data.as_ref().map(|d| d.len()));

    // Download scales and qzeros from mmap engine
    let mmap_row_scales = engine_mmap.weight_caches()[0]
        .download_int4_scales(row_parallel_name, &gpu0_stream_m);
    let mmap_col_scales = engine_mmap.weight_caches()[0]
        .download_int4_scales(col_parallel_name, &gpu0_stream_m);
    let mmap_row_qzeros = engine_mmap.weight_caches()[0]
        .download_int4_qzeros(row_parallel_name, &gpu0_stream_m);
    let mmap_col_qzeros = engine_mmap.weight_caches()[0]
        .download_int4_qzeros(col_parallel_name, &gpu0_stream_m);

    eprintln!("[mmap-vs-heap] Mmap scales for '{}': {:?}", row_parallel_name,
        mmap_row_scales.as_ref().map(|d| d.len()));
    eprintln!("[mmap-vs-heap] Mmap qzeros for '{}': {:?}", row_parallel_name,
        mmap_row_qzeros.as_ref().map(|d| d.len()));

    // ---- Cache key coverage diagnostic ----
    {
        let mmap_cache_keys: Vec<String> = engine_mmap.weight_caches()[0].keys().map(|s| s.to_string()).collect();

        let mut heap_only: Vec<&String> = Vec::new();
        let mut mmap_only: Vec<&String> = Vec::new();
        let mut common: Vec<&String> = Vec::new();

        for key in &heap_cache_keys {
            if mmap_cache_keys.contains(key) {
                common.push(key);
            } else {
                heap_only.push(key);
            }
        }
        for key in &mmap_cache_keys {
            if !heap_cache_keys.contains(key) {
                mmap_only.push(key);
            }
        }

        eprintln!(
            "[mmap-vs-heap] Cache key coverage: {} common, {} heap-only, {} mmap-only",
            common.len(),
            heap_only.len(),
            mmap_only.len()
        );

        if !heap_only.is_empty() {
            eprintln!("[mmap-vs-heap] Heap-only keys (first 10):");
            for key in heap_only.iter().take(10) {
                eprintln!("    {}", key);
            }
        }
        if !mmap_only.is_empty() {
            eprintln!("[mmap-vs-heap] Mmap-only keys (first 10):");
            for key in mmap_only.iter().take(10) {
                eprintln!("    {}", key);
            }
        }

        // For the first 20 common keys, check that CachedWeight type matches
        if !common.is_empty() {
            eprintln!("[mmap-vs-heap] Type match for first {} common keys:", common.len().min(20));
            for key in common.iter().take(20) {
                let heap_type = heap_key_types.iter()
                    .find(|(k, _)| k == key.as_str())
                    .map(|(_, t)| *t);

                let mmap_type = match engine_mmap.weight_caches()[0].get(key) {
                    Some(infers_backend_native::CachedWeight::Bf16(_)) => Some("Bf16"),
                    Some(infers_backend_native::CachedWeight::Int4(_)) => Some("Int4"),
                    None => None,
                };

                let match_str = match (heap_type, mmap_type) {
                    (Some(ht), Some(mt)) if ht == mt => "MATCH".to_string(),
                    (Some(ht), Some(mt)) => format!("MISMATCH (heap={}, mmap={})", ht, mt),
                    _ => "???".to_string(),
                };
                eprintln!("    {}: {}", key, match_str);
            }
        }

        // Explicitly verify inference-critical weight names exist in both caches
        let critical_names = [
            "embed_tokens.weight",
            "norm.weight",
            "lm_head.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.self_attn.q_proj.qweight",
            "layers.0.linear_attn.in_proj_qkv.qweight",
            "layers.0.mlp.gate_proj.qweight",
        ];

        eprintln!("[mmap-vs-heap] Inference-critical weight presence:");
        for name in &critical_names {
            let heap_has = heap_cache_keys.contains(&name.to_string());
            let mmap_has = engine_mmap.weight_caches()[0].get(name).is_some();

            let status = match (heap_has, mmap_has) {
                (true, true) => "present in both",
                (true, false) => "heap only",
                (false, true) => "mmap only",
                (false, false) => "missing from both",
            };
            eprintln!("    {}: {}", name, status);
        }
    }

    // ---- Compare row-parallel INT4 weight (o_proj.qweight — should be contiguous) ----
    {
        match (heap_row_data, mmap_row_data) {
            (Some(h), Some(m)) => {
                let compare_len = 64.min(h.len()).min(m.len());
                eprintln!("[mmap-vs-heap] Heap data len: {}, Mmap data len: {}, comparing first {} u32",
                    h.len(), m.len(), compare_len);

                let mut mismatch_count = 0usize;
                for i in 0..compare_len {
                    if h[i] != m[i] {
                        mismatch_count += 1;
                        eprintln!("[mmap-vs-heap] MISMATCH at index {}: heap={}, mmap={}", i, h[i], m[i]);
                    }
                }

                if mismatch_count == 0 {
                    eprintln!("[mmap-vs-heap] row-parallel: MATCH (all {} values identical)", compare_len);
                } else {
                    eprintln!("[mmap-vs-heap] row-parallel: MISMATCH ({} of {} values differ)",
                        mismatch_count, compare_len);
                }
            }
            (None, None) => {
                eprintln!("[mmap-vs-heap] Weight '{}' not found in either engine's GPU 0 cache", row_parallel_name);
            }
            (None, Some(_)) => {
                eprintln!("[mmap-vs-heap] Weight '{}' missing from heap engine, present in mmap engine", row_parallel_name);
            }
            (Some(_), None) => {
                eprintln!("[mmap-vs-heap] Weight '{}' present in heap engine, missing from mmap engine", row_parallel_name);
            }
        }
    }

    // ---- Compare row-parallel scales ----
    {
        match (heap_row_scales, mmap_row_scales) {
            (Some(h), Some(m)) => {
                let compare_len = 64.min(h.len()).min(m.len());
                let mut mismatch_count = 0usize;
                for i in 0..compare_len {
                    if h[i].to_bits() != m[i].to_bits() {
                        mismatch_count += 1;
                    }
                }
                if mismatch_count == 0 {
                    eprintln!("[mmap-vs-heap] scales: MATCH ({} of {})", mismatch_count, compare_len);
                } else {
                    eprintln!("[mmap-vs-heap] scales: MISMATCH ({} of {} differ)", mismatch_count, compare_len);
                }
            }
            (None, None) => {
                eprintln!("[mmap-vs-heap] scales for '{}' not found in either engine", row_parallel_name);
            }
            _ => {
                eprintln!("[mmap-vs-heap] scales for '{}' present in one but not both engines", row_parallel_name);
            }
        }
    }

    // ---- Compare row-parallel qzeros ----
    {
        match (heap_row_qzeros, mmap_row_qzeros) {
            (Some(h), Some(m)) => {
                let compare_len = 64.min(h.len()).min(m.len());
                let mut mismatch_count = 0usize;
                for i in 0..compare_len {
                    if h[i] != m[i] {
                        mismatch_count += 1;
                    }
                }
                if mismatch_count == 0 {
                    eprintln!("[mmap-vs-heap] qzeros: MATCH ({} of {})", mismatch_count, compare_len);
                } else {
                    eprintln!("[mmap-vs-heap] qzeros: MISMATCH ({} of {} differ)", mismatch_count, compare_len);
                }
            }
            (None, None) => {
                eprintln!("[mmap-vs-heap] qzeros for '{}' not found in either engine", row_parallel_name);
            }
            _ => {
                eprintln!("[mmap-vs-heap] qzeros for '{}' present in one but not both engines", row_parallel_name);
            }
        }
    }

    // ---- Compare column-parallel INT4 weight (gate_proj.qweight — may be strided in mmap path) ----
    {
        match (heap_col_data, mmap_col_data) {
            (Some(h), Some(m)) => {
                let compare_len = 64.min(h.len()).min(m.len());
                eprintln!("[mmap-vs-heap] Heap data len: {}, Mmap data len: {}, comparing first {} u32",
                    h.len(), m.len(), compare_len);

                let mut mismatch_count = 0usize;
                for i in 0..compare_len {
                    if h[i] != m[i] {
                        mismatch_count += 1;
                        eprintln!("[mmap-vs-heap] MISMATCH at index {}: heap={}, mmap={}", i, h[i], m[i]);
                    }
                }

                if mismatch_count == 0 {
                    eprintln!("[mmap-vs-heap] column-parallel: MATCH (all {} values identical)", compare_len);
                } else {
                    eprintln!("[mmap-vs-heap] column-parallel: MISMATCH ({} of {} values differ)",
                        mismatch_count, compare_len);
                }
            }
            (None, None) => {
                eprintln!("[mmap-vs-heap] Weight '{}' not found in either engine's GPU 0 cache", col_parallel_name);
            }
            (None, Some(_)) => {
                eprintln!("[mmap-vs-heap] Weight '{}' missing from heap engine, present in mmap engine", col_parallel_name);
            }
            (Some(_), None) => {
                eprintln!("[mmap-vs-heap] Weight '{}' present in heap engine, missing from mmap engine", col_parallel_name);
            }
        }
    }

    // ---- Compare column-parallel scales ----
    {
        match (heap_col_scales, mmap_col_scales) {
            (Some(h), Some(m)) => {
                let compare_len = 64.min(h.len()).min(m.len());
                let mut mismatch_count = 0usize;
                for i in 0..compare_len {
                    if h[i].to_bits() != m[i].to_bits() {
                        mismatch_count += 1;
                    }
                }
                if mismatch_count == 0 {
                    eprintln!("[mmap-vs-heap] col-parallel scales: MATCH ({} of {})", mismatch_count, compare_len);
                } else {
                    eprintln!("[mmap-vs-heap] col-parallel scales: MISMATCH ({} of {} differ)", mismatch_count, compare_len);
                }
            }
            (None, None) => {
                eprintln!("[mmap-vs-heap] col-parallel scales for '{}' not found in either engine", col_parallel_name);
            }
            _ => {
                eprintln!("[mmap-vs-heap] col-parallel scales for '{}' present in one but not both engines", col_parallel_name);
            }
        }
    }

    // ---- Compare column-parallel qzeros ----
    {
        match (heap_col_qzeros, mmap_col_qzeros) {
            (Some(h), Some(m)) => {
                let compare_len = 64.min(h.len()).min(m.len());
                let mut mismatch_count = 0usize;
                for i in 0..compare_len {
                    if h[i] != m[i] {
                        mismatch_count += 1;
                    }
                }
                if mismatch_count == 0 {
                    eprintln!("[mmap-vs-heap] col-parallel qzeros: MATCH ({} of {})", mismatch_count, compare_len);
                } else {
                    eprintln!("[mmap-vs-heap] col-parallel qzeros: MISMATCH ({} of {} differ)", mismatch_count, compare_len);
                }
            }
            (None, None) => {
                eprintln!("[mmap-vs-heap] col-parallel qzeros for '{}' not found in either engine", col_parallel_name);
            }
            _ => {
                eprintln!("[mmap-vs-heap] col-parallel qzeros for '{}' present in one but not both engines", col_parallel_name);
            }
        }
    }

    // ---- Compare BF16 conv1d weights (layer 0 and layer 30) ----
    fn compare_bf16_conv1d(name: &str, heap_data: Option<Vec<bf16>>, mmap_data: Option<Vec<bf16>>) {
        match (heap_data, mmap_data) {
            (Some(h), Some(m)) => {
                let n = h.len().min(m.len());
                eprintln!("[mmap-vs-heap] {} BF16 conv1d: heap len={}, mmap len={}, comparing {} values", name, h.len(), m.len(), n);
                let mut mismatch_count = 0usize;
                let mut mismatches: Vec<(usize, u16, u16)> = Vec::new();
                for i in 0..n {
                    if h[i].to_bits() != m[i].to_bits() {
                        mismatch_count += 1;
                        if mismatches.len() < 5 {
                            mismatches.push((i, h[i].to_bits(), m[i].to_bits()));
                        }
                    }
                }
                if mismatch_count == 0 {
                    eprintln!("[mmap-vs-heap] {} BF16 conv1d: MATCH (all {} values identical)", name, n);
                } else {
                    eprintln!("[mmap-vs-heap] {} BF16 conv1d: MISMATCH ({} of {} differ)", name, mismatch_count, n);
                    for (idx, hv, mv) in mismatches {
                        eprintln!("[mmap-vs-heap]   index {}: heap=0x{:04x}, mmap=0x{:04x}", idx, hv, mv);
                    }
                }
            }
            (None, None) => {
                eprintln!("[mmap-vs-heap] BF16 weight '{}' not found in either engine's GPU 0 cache", name);
            }
            (None, Some(_)) => {
                eprintln!("[mmap-vs-heap] BF16 weight '{}' missing from heap engine, present in mmap engine", name);
            }
            (Some(_), None) => {
                eprintln!("[mmap-vs-heap] BF16 weight '{}' present in heap engine, missing from mmap engine", name);
            }
        }
    }

    compare_bf16_conv1d("layers.0.linear_attn.conv1d.weight", heap_conv1d_0, mmap_conv1d_0);
    compare_bf16_conv1d("layers.30.linear_attn.conv1d.weight", heap_conv1d_30, mmap_conv1d_30);

    // Drop mmap engine
    drop(engine_mmap);
    drop(runtime_mmap);

    eprintln!("[mmap-vs-heap] Diagnostic comparison complete");

    Ok(())
}
