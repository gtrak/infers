//! Smoke test comparing heap-loaded vs mmap-loaded model inference.
//!
//! Loads the same real model via both paths sequentially (since both need
//! the same GPUs), runs prefill + 30 decode steps on each, and asserts
//! the generated token sequences match exactly.
//!
//! Requires:
//! - 2 GPUs with CUDA compute capability 12.0+ (Blackwell)
//! - NVLink or NCCL-capable interconnect between GPUs
//! - Model downloaded to INFERS_TEST_MODEL path (default: ~/opt/vllm/models/qwen3.6-27b-autoround-int4/)
//!
//! Run with: cargo test --package infers-backend-native --test smoke_test_mmap smoke_test_mmap_matches_heap -- --ignored --nocapture
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use infers_backend_native::{CachedWeight, ForwardEngine};
use infers_cuda::context::CudaRuntime;
use infers_cuda::kernels::KernelRegistry;
use infers_cuda::stream::StreamPool;
use infers_kv::SequenceId;
use infers_model::mmap::{load_safetensors_mmap, strip_language_model_prefix_mmap, shard_weights_tp_mmap, build_metadata_registry};
use infers_model::sharding::shard_weights_tp;
use infers_model::{load_safetensors, strip_language_model_prefix, build_main_layers};

/// Default path for the Qwen3.6-27B AutoRound INT4 model.
const DEFAULT_MODEL_DIR: &str = "~/opt/vllm/models/qwen3.6-27b-autoround-int4/";

/// Run prefill + decode inference, returning (engine_init_time, prefill_time, token_sequence).
fn run_inference(
    engine: &mut ForwardEngine,
    runtime: &CudaRuntime,
    num_pages: usize,
    page_size: usize,
    max_cache_bytes: usize,
    token_ids: &[u32],
) -> Result<(f64, f64, Vec<u32>), Box<dyn std::error::Error>> {
    // Initialize paged KV cache
    engine.init_paged(num_pages, page_size, max_cache_bytes)?;

    // Create a sequence
    let seq_id: SequenceId = engine.create_sequence();

    // Get external stream
    let external_stream = runtime.default_stream(0)?;

    // Run prefill
    let prefill_start = Instant::now();
    let sampling_config = infers_scheduler::SamplingConfig::default();
    let mut rng = infers_backend_native::Xoshiro256PlusPlus::from_seed(42);
    let (_, first_token) = engine.prefill_paged(&external_stream, token_ids, seq_id, &sampling_config, &mut rng)?;
    let prefill_time = prefill_start.elapsed().as_secs_f64();

    // Run decode for 30 steps
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut all_tokens = token_ids.to_vec();
    all_tokens.push(first_token);
    let mut token = token_ids[0];
    for step in 0..30 {
        let pos = (token_ids.len() + step) as u32;
        token = engine.decode_paged(&external_stream, token, pos, seq_id, &sampling_config, &all_tokens, token_ids.len(), &mut rng)?;
        all_tokens.push(token);
        generated_tokens.push(token);
    }

    // We don't actually need the engine init time here since it's measured by caller,
    // but return 0 as placeholder
    Ok((0.0, prefill_time, generated_tokens))
}

/// Smoke test that loads a real model via both heap and mmap paths, runs inference on each,
/// and asserts the generated token sequences are identical.
#[test]
#[ignore = "requires 2 GPUs and real model weights"]
fn smoke_test_mmap_matches_heap() -> Result<(), Box<dyn std::error::Error>> {
    // Resolve model path from environment or default
    let model_dir_str = std::env::var("INFERS_TEST_MODEL")
        .unwrap_or_else(|_| DEFAULT_MODEL_DIR.to_string());
    let model_dir = Path::new(&model_dir_str);

    let model_dir = if model_dir_str.starts_with('~') {
        let home = std::env::var("HOME")?;
        Path::new(&home).join(model_dir_str.strip_prefix("~/").unwrap())
    } else {
        model_dir.to_path_buf()
    };

    eprintln!("Loading model from: {}", model_dir.display());

    // =========================================================
    // PHASE 1: Heap path — load, run inference, drop everything
    // =========================================================
    eprintln!("\n=== PHASE 1: Heap path ===");

    let config = infers_model::config::ModelConfig::load(&model_dir)?;
    eprintln!("Config loaded: {} layers", config.num_hidden_layers);

    let mut raw_weights = load_safetensors(&model_dir)?;
    strip_language_model_prefix(&mut raw_weights);
    eprintln!("Raw tensors loaded (heap): {}", raw_weights.num_tensors());

    let num_gpus = 2;
    let shards = shard_weights_tp(&raw_weights, &config, num_gpus)?;
    assert_eq!(shards.len(), num_gpus);

    let mut weight_registries: Vec<infers_model::WeightRegistry> = Vec::new();
    for shard in shards {
        let gpu_id = shard.gpu_id;
        let mut registry = shard.registry;
        build_main_layers(&mut registry, &config)?;
        eprintln!(
            "Heap Shard {}: layers={}, embedding={}, norm={}, lm_head={}",
            gpu_id,
            registry.layers.len(),
            registry.embedding.is_some(),
            registry.norm.is_some(),
            registry.lm_head.is_some(),
        );
        weight_registries.push(registry);
    }

    // --- Capture heap layer structure BEFORE ForwardEngine consumes weight_registries ---
    let reg = &weight_registries[0];
    let heap_layer_count = reg.layers.len();
    let heap_layer_types: Vec<String> = reg.layers.iter()
        .map(|l| format!("{:?} idx={}", l.layer_type, l.layer_idx))
        .collect();
    // Weight names and shapes in layer 0
    let heap_l0_weight_info: Vec<(String, String)> = {
        let mut info = Vec::new();
        if let Some(layer) = reg.layers.first() {
            info.push(("norm1".to_string(), layer.norm1.name.clone()));
            info.push(("norm2".to_string(), layer.norm2.name.clone()));
            if let Some(ref gdn) = layer.gdn {
                info.push(("gdn.in_proj_a".to_string(), gdn.in_proj_a.name.clone()));
                info.push(("gdn.in_proj_b".to_string(), gdn.in_proj_b.name.clone()));
                info.push(("gdn.conv1d_weight".to_string(), gdn.conv1d_weight.name.clone()));
                if let Some(ref w) = gdn.in_proj_qkv { info.push(("gdn.in_proj_qkv".to_string(), w.name.clone())); }
                if let Some(ref w) = gdn.in_proj_z   { info.push(("gdn.in_proj_z".to_string(), w.name.clone())); }
                info.push(("gdn.out_proj_weight".to_string(), gdn.out_proj_weight.name.clone()));
            }
            if let Some(ref attn) = layer.attn {
                info.push(("attn.q_proj".to_string(), attn.q_proj.name.clone()));
                info.push(("attn.k_proj".to_string(), attn.k_proj.name.clone()));
                info.push(("attn.v_proj".to_string(), attn.v_proj.name.clone()));
                info.push(("attn.o_proj".to_string(), attn.o_proj.name.clone()));
            }
            info.push(("mlp.gate_proj".to_string(), layer.mlp.gate_proj.name.clone()));
            info.push(("mlp.up_proj".to_string(), layer.mlp.up_proj.name.clone()));
            info.push(("mlp.down_proj".to_string(), layer.mlp.down_proj.name.clone()));
        }
        info
    };
    let heap_l0_shape_info: Vec<(String, String)> = {
        let mut info = Vec::new();
        if let Some(layer) = reg.layers.first() {
            info.push(("norm1".to_string(), format!("{:?}", layer.norm1.shape)));
            info.push(("norm2".to_string(), format!("{:?}", layer.norm2.shape)));
            if let Some(ref gdn) = layer.gdn {
                info.push(("gdn.in_proj_a".to_string(), format!("{:?}", gdn.in_proj_a.shape)));
                if let Some(ref w) = gdn.in_proj_qkv { info.push(("gdn.in_proj_qkv".to_string(), format!("{:?}", w.shape))); }
                if let Some(ref w) = gdn.in_proj_z   { info.push(("gdn.in_proj_z".to_string(), format!("{:?}", w.shape))); }
            }
            if let Some(ref attn) = layer.attn {
                info.push(("attn.q_proj".to_string(), format!("{:?}", attn.q_proj.shape)));
            }
            info.push(("mlp.gate_proj".to_string(), format!("{:?}", layer.mlp.gate_proj.shape)));
        }
        info
    };
    let heap_int4_companion_count = reg.int4_companions.len();
    let heap_int4_companion_keys: Vec<String> = reg.int4_companions.keys().cloned().take(5).collect();
    // Flat tensors map — first 10 entries after build_main_layers
    let heap_tensor_keys: Vec<String> = reg.tensors.keys().cloned().take(10).collect();
    let heap_tensor_count = reg.tensors.len();
    // Capture shapes for first 10 tensor keys for later comparison
    let heap_tensor_shapes: Vec<(String, String)> = reg.tensors.iter()
        .map(|(k, t)| (k.clone(), format!("{:?}", t.shape)))
        .take(10)
        .collect();

    let runtime = CudaRuntime::new()?;
    assert!(runtime.num_devices >= 2, "TP=2 requires at least 2 GPUs, found {}", runtime.num_devices);
    let ctx0 = runtime.device(0)?.clone();
    let ctx1 = runtime.device(1)?.clone();

    let stream_pool = StreamPool::new(&[ctx0.clone(), ctx1.clone()])?;

    let mut kernel_registry = KernelRegistry::new();
    kernel_registry.register_infers_kernels();

    let mut config_heap = config.clone();
    let test_max_seq_len = 4096usize.min(config.max_position_embeddings);
    config_heap.max_position_embeddings = test_max_seq_len;
    let config_heap = Arc::new(config_heap);

    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Some(infers_tokenizer::Tokenizer::from_file(tokenizer_path.to_str().unwrap())?)
    } else {
        None
    };
    let prompt = "user\nWhat is the capital of France?\n\n".to_string();
    let token_ids: Vec<u32> = if let Ok(ids_str) = std::env::var("INFERS_TEST_TOKEN_IDS") {
        ids_str.split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    } else if let Some(ref tok) = tokenizer {
        tok.encode(&prompt)?
    } else {
        vec![151644u32]
    };

    let group_size = 128;

// --- Capture heap INT4 companion shapes before ForwardEngine consumes weight_registries ---
    let heap_int4_shapes: std::collections::HashMap<String, (Vec<usize>, Vec<usize>)> = weight_registries[0].int4_companions.iter()
        .map(|(name, comp)| (name.clone(), (comp.scales.shape.to_vec(), comp.qzeros.shape.to_vec())))
        .collect();

    // Capture heap tensor data bytes for later comparison with mmap
    let compare_key = "layers.0.linear_attn.in_proj_qkv.qweight";
    let mut heap_tensor_bytes: Option<Vec<u8>> = None;
    if let Some(layer) = weight_registries[0].layers.first() {
        if let Some(gdn) = &layer.gdn {
            if let Some(qkv) = &gdn.in_proj_qkv {
                eprintln!("[diagnostic] Heap '{}' dtype={:?} shape={:?} first 16 bytes: {:?}", 
                    compare_key, qkv.dtype, qkv.shape, &qkv.data[..16.min(qkv.data.len())]);
                heap_tensor_bytes = Some(qkv.data[0..16].to_vec());
            }
        }
    }

    let engine_start = Instant::now();
    let mut heap_engine = ForwardEngine::new(
        config_heap.clone(),
        weight_registries,
        vec![ctx0, ctx1],
        kernel_registry,
        stream_pool,
        group_size,
    )?;
    let heap_engine_time = engine_start.elapsed().as_secs_f64();
    eprintln!("Heap engine init: {:.2}s", heap_engine_time);

  // --- Diagnostic: capture heap cache info before it gets dropped ---
    let heap_keys_gpu0: Vec<String> = heap_engine.weight_caches()[0].keys().map(|s| s.to_string()).collect();
    let _heap_keys_gpu1: Vec<String> = heap_engine.weight_caches()[1].keys().map(|s| s.to_string()).collect();
    eprintln!("[diagnostic] Heap cache[0]: {} weights", heap_engine.weight_caches()[0].len());
    eprintln!("[diagnostic] Heap cache[1]: {} weights", heap_engine.weight_caches()[1].len());

    // Sample a few actual keys from the heap cache to understand naming convention
    eprintln!("[diagnostic] First 20 heap keys: {:?}", &heap_keys_gpu0[..20.min(heap_keys_gpu0.len())]);

    // Check if keys contain "layers.0" with any suffix pattern
    let layers0_keys: Vec<_> = heap_keys_gpu0.iter()
        .filter(|k| k.starts_with("layers.0"))
        .collect();
    eprintln!("[diagnostic] Heap keys starting with 'layers.0': {}", layers0_keys.len());
    if !layers0_keys.is_empty() {
        eprintln!("[diagnostic] Sample layers.0 keys: {:?}", &layers0_keys[..10.min(layers0_keys.len())]);
    }

    // Check for q_proj specifically in any form
    let q_proj_keys: Vec<_> = heap_keys_gpu0.iter()
        .filter(|k| k.contains("q_proj"))
        .collect();
    eprintln!("[diagnostic] Heap keys containing 'q_proj': {}", q_proj_keys.len());
    if !q_proj_keys.is_empty() {
        eprintln!("[diagnostic] Sample q_proj keys: {:?}", &q_proj_keys[..10.min(q_proj_keys.len())]);
    }

    // Check for embedding key in heap cache
    let embed_keys_heap: Vec<_> = heap_keys_gpu0.iter()
        .filter(|k| k.contains("embed"))
        .collect();
    eprintln!("[diagnostic] Heap keys containing 'embed': {:?}", embed_keys_heap);

    // Capture dtype mapping for all heap keys as owned data (engine will be dropped later)
    let heap_dtype_gpu0: Vec<(String, String)> = heap_engine.weight_caches()[0].keys()
        .map(|k| {
            let dtype = match heap_engine.weight_caches()[0].get(k) {
                Some(CachedWeight::Bf16(_)) => "Bf16".to_string(),
                Some(CachedWeight::Int4(_)) => "Int4".to_string(),
                None => "???".to_string(),
            };
            (k.to_string(), dtype)
        })
        .collect();

    let heap_int4_count_gpu0: usize = heap_engine.weight_caches()[0].keys()
        .filter(|k| matches!(heap_engine.weight_caches()[0].get(k), Some(CachedWeight::Int4(_))))
        .count();

    // Check specific weight exists and its dtype in heap path (try both with/without .qweight suffix)
    for &name in &["layers.0.self_attn.q_proj.qweight", "layers.0.self_attn.q_proj"] {
        if let Some(w) = heap_engine.weight_caches()[0].get(name) {
            match w {
                CachedWeight::Bf16(_) => eprintln!("[diagnostic] Heap {} → Bf16", name),
                CachedWeight::Int4(b) => eprintln!("[diagnostic] Heap {} → Int4 (shape={:?})", name, b.shape),
            }
        }
    }
   // Check a few more critical keys exist in heap
    for &name in &[
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.norm1",
    ] {
        let exists = heap_engine.weight_caches()[0].get(name).is_some();
        eprintln!("[diagnostic] Heap '{}' exists: {}", name, exists);
    }

    // Check embedding weight buffer size in heap cache
    if let Some(w) = heap_engine.weight_caches()[0].get("embed_tokens.weight") {
        match w {
            CachedWeight::Bf16(s) => eprintln!("[diagnostic] Heap embed_tokens Bf16 len: {}", s.len()),
            CachedWeight::Int4(_) => eprintln!("[diagnostic] WARNING: Heap embed_tokens is Int4?"),
        }
    } else {
        eprintln!("[diagnostic] Heap missing: embed_tokens.weight");
    }

    // Check BF16 norm weight sizes
    for test_key in &["input_layernorm", "post_attention_layernorm"] {
        let key = format!("layers.0.{}", test_key);
        if let Some(w) = heap_engine.weight_caches()[0].get(&key) {
            match w {
                CachedWeight::Bf16(s) => eprintln!("[diagnostic] Heap '{}' Bf16 len: {}", key, s.len()),
                CachedWeight::Int4(_) => eprintln!("[diagnostic] WARNING: Heap '{}' is Int4?", key),
            }
        } else {
            eprintln!("[diagnostic] Heap missing: {}", key);
        }
    }

    // Run inference via heap path
    let num_pages = (test_max_seq_len / 16) * 2;
    let max_cache_bytes = 512 * 1024 * 1024;
    let (_heap_init, heap_prefill_time, heap_tokens) = run_inference(
        &mut heap_engine,
        &runtime,
        num_pages,
        16,
        max_cache_bytes,
        &token_ids,
    )?;

    eprintln!("Heap tokens ({}): {:?}", heap_tokens.len(), heap_tokens);
    if let Some(ref tok) = tokenizer {
        let text = tok.decode(&heap_tokens)?;
        eprintln!("Heap decoded: {}", text.trim());
    }

    // Drop heap engine and all associated resources to free GPU memory
    drop(heap_engine);
    drop(config_heap);
    drop(raw_weights);
    // weight_registries was already consumed by ForwardEngine::new
    eprintln!("Heap engine dropped — GPU memory freed");

    // =========================================================
    // PHASE 2: Mmap path — load, run inference, compare tokens
    // =========================================================
    eprintln!("\n=== PHASE 2: Mmap path ===");

    let mmap_config = infers_model::config::ModelConfig::load(&model_dir)?;

    let mut mmap_registry = load_safetensors_mmap(&model_dir)?;
    strip_language_model_prefix_mmap(&mut mmap_registry);
    eprintln!("Mmap registry loaded: {} tensors", mmap_registry.tensors.len());

    let mmap_shards = shard_weights_tp_mmap(&mmap_registry, &mmap_config, num_gpus)?;
    assert_eq!(mmap_shards.len(), num_gpus);

    // Build metadata registries per shard, then build main layers (embedding/norm/layers) for inference lookup
    let mut mmap_weight_registries: Vec<infers_model::MmapWeightRegistry> = Vec::new();
    let mut metadata_registries: Vec<infers_model::WeightRegistry> = Vec::new();
    for shard in mmap_shards {
        let gpu_id = shard.gpu_id;
        let registry = shard.registry;

        let mut meta = build_metadata_registry(&registry);
        // Build main layers so that embedding/norm/lm_head/layers are populated (required by engine)
        infers_model::build_main_layers(&mut meta, &mmap_config)?;
        eprintln!(
            "Mmap Shard {}: tensors={}, int4_companions={}, embedding={}, norm={}, lm_head={}",
            gpu_id,
            registry.tensors.len(),
            registry.int4_companions.len(),
            meta.embedding.is_some(),
            meta.norm.is_some(),
            meta.lm_head.is_some(),
        );
        mmap_weight_registries.push(registry);
        metadata_registries.push(meta);
    }

    // === LAYER STRUCTURE COMPARISON (heap vs mmap) ===
    {
        let mm = &metadata_registries[0];
        let mm_layer_count = mm.layers.len();
        let mm_layer_types: Vec<String> = mm.layers.iter()
            .map(|l| format!("{:?} idx={}", l.layer_type, l.layer_idx))
            .collect();
        let mm_l0_weight_info: Vec<(String, String)> = {
            let mut info = Vec::new();
            if let Some(layer) = mm.layers.first() {
                info.push(("norm1".to_string(), layer.norm1.name.clone()));
                info.push(("norm2".to_string(), layer.norm2.name.clone()));
                if let Some(ref gdn) = layer.gdn {
                    info.push(("gdn.in_proj_a".to_string(), gdn.in_proj_a.name.clone()));
                    info.push(("gdn.in_proj_b".to_string(), gdn.in_proj_b.name.clone()));
                    info.push(("gdn.conv1d_weight".to_string(), gdn.conv1d_weight.name.clone()));
                    if let Some(ref w) = gdn.in_proj_qkv { info.push(("gdn.in_proj_qkv".to_string(), w.name.clone())); }
                    if let Some(ref w) = gdn.in_proj_z   { info.push(("gdn.in_proj_z".to_string(), w.name.clone())); }
                    info.push(("gdn.out_proj_weight".to_string(), gdn.out_proj_weight.name.clone()));
                }
                if let Some(ref attn) = layer.attn {
                    info.push(("attn.q_proj".to_string(), attn.q_proj.name.clone()));
                    info.push(("attn.k_proj".to_string(), attn.k_proj.name.clone()));
                    info.push(("attn.v_proj".to_string(), attn.v_proj.name.clone()));
                    info.push(("attn.o_proj".to_string(), attn.o_proj.name.clone()));
                }
                info.push(("mlp.gate_proj".to_string(), layer.mlp.gate_proj.name.clone()));
                info.push(("mlp.up_proj".to_string(), layer.mlp.up_proj.name.clone()));
                info.push(("mlp.down_proj".to_string(), layer.mlp.down_proj.name.clone()));
            }
            info
        };
        let mm_l0_shape_info: Vec<(String, String)> = {
            let mut info = Vec::new();
            if let Some(layer) = mm.layers.first() {
                info.push(("norm1".to_string(), format!("{:?}", layer.norm1.shape)));
                info.push(("norm2".to_string(), format!("{:?}", layer.norm2.shape)));
                if let Some(ref gdn) = layer.gdn {
                    info.push(("gdn.in_proj_a".to_string(), format!("{:?}", gdn.in_proj_a.shape)));
                    if let Some(ref w) = gdn.in_proj_qkv { info.push(("gdn.in_proj_qkv".to_string(), format!("{:?}", w.shape))); }
                    if let Some(ref w) = gdn.in_proj_z   { info.push(("gdn.in_proj_z".to_string(), format!("{:?}", w.shape))); }
                }
                if let Some(ref attn) = layer.attn {
                    info.push(("attn.q_proj".to_string(), format!("{:?}", attn.q_proj.shape)));
                }
                info.push(("mlp.gate_proj".to_string(), format!("{:?}", layer.mlp.gate_proj.shape)));
            }
            info
        };
        let mm_int4_companion_count = mm.int4_companions.len();
        let mm_int4_companion_keys: Vec<String> = mm.int4_companions.keys().cloned().take(5).collect();
        let mm_tensor_keys: Vec<String> = mm.tensors.keys().cloned().take(10).collect();
        let mm_tensor_count = mm.tensors.len();

        eprintln!("\n[layer-compare] ===============================");
        eprintln!("[layer-compare] Layer count:      heap={} mmap={}", heap_layer_count, mm_layer_count);

        // Compare layer types
        if heap_layer_types != mm_layer_types {
            eprintln!("[layer-compare] LAYER TYPES DIFFER!");
            for (i, (h, m)) in heap_layer_types.iter().zip(mm_layer_types.iter()).enumerate() {
                if h != m {
                    eprintln!("[layer-compare]   layer[{}]: heap={} mmap={}", i, h, m);
                }
            }
        } else {
            eprintln!("[layer-compare] Layer types: MATCH");
        }

        // Compare first 5 layer types explicitly
        for i in 0..5.min(heap_layer_count) {
            let h_type = &heap_layer_types[i];
            let m_type = &mm_layer_types[i];
            eprintln!("[layer-compare] layer[{}] type: heap={} mmap={}", i, h_type, m_type);
        }

        // Compare weight names in layer 0
        eprintln!("[layer-compare] Layer 0 weight names:");
        for (h_name, h_val) in &heap_l0_weight_info {
            let m_val = mm_l0_weight_info.iter().find(|(n, _)| n == h_name).map(|(_, v)| v.as_str());
            match m_val {
                Some(mv) if mv == h_val => {},
                Some(mv) => eprintln!("[layer-compare]   weight_name {} MISMATCH: heap={} mmap={}", h_name, h_val, mv),
                None => eprintln!("[layer-compare]   weight_name {} MISSING in mmap (heap={})", h_name, h_val),
            }
        }
        // Check for extra keys in mmap not present in heap
        for (m_name, m_val) in &mm_l0_weight_info {
            if !heap_l0_weight_info.iter().any(|(n, _)| n == m_name) {
                eprintln!("[layer-compare]   weight_name {} EXTRA in mmap (mmap={})", m_name, m_val);
            }
        }

        // Compare shapes in layer 0
        eprintln!("[layer-compare] Layer 0 weight shapes:");
        for (h_name, h_shape) in &heap_l0_shape_info {
            let m_shape = mm_l0_shape_info.iter().find(|(n, _)| n == h_name).map(|(_, v)| v.as_str());
            match m_shape {
                Some(ms) if ms == h_shape => {},
                Some(ms) => eprintln!("[layer-compare]   shape {} MISMATCH: heap={} mmap={}", h_name, h_shape, ms),
                None => eprintln!("[layer-compare]   shape {} MISSING in mmap (heap={})", h_name, h_shape),
            }
        }

        // INT4 companion count
        eprintln!("[layer-compare] INT4 companions: heap={} mmap={}", heap_int4_companion_count, mm_int4_companion_count);

        // Compare first 5 companion keys
        eprintln!("[layer-compare] Companion keys (first 5):");
        for key in &heap_int4_companion_keys {
            if !mm.int4_companions.contains_key(key.as_str()) {
                eprintln!("[layer-compare]   companion {} MISSING in mmap", key);
            }
        }
        for key in &mm_int4_companion_keys {
            if !heap_int4_companion_keys.iter().any(|k| k == key) {
                eprintln!("[layer-compare]   companion {} EXTRA in mmap", key);
            }
        }

        // Flat tensors comparison
        eprintln!("[layer-compare] Flat tensors count: heap={} mmap={}", heap_tensor_count, mm_tensor_count);
        eprintln!("[layer-compare] First 10 tensor keys (heap):");
        for k in &heap_tensor_keys {
            let present = mm.tensors.contains_key(k.as_str());
            if !present {
                eprintln!("[layer-compare]   {} MISSING in mmap", k);
            } else {
                // Compare shape
                let h_shape = heap_tensor_shapes.iter().find(|(name, _)| name == k).map(|(_, v)| v.as_str());
                let m_shape = mm.tensors.get(k.as_str()).map(|t| format!("{:?}", t.shape));
                match (h_shape, m_shape) {
                    (Some(hs), Some(ms)) if hs != ms => eprintln!("[layer-compare]   {} shape MISMATCH: heap={} mmap={}", k, hs, ms),
                    _ => {}, // present and shapes match or both missing
                }
            }
        }
        eprintln!("[layer-compare] First 10 tensor keys (mmap):");
        for k in &mm_tensor_keys {
            if !heap_tensor_keys.iter().any(|h| h == k) {
                eprintln!("[layer-compare]   {} EXTRA in mmap", k);
            }
        }

        eprintln!("[layer-compare] ===============================\n");
    }

    // Fresh CUDA contexts for the second engine run
    let runtime2 = CudaRuntime::new()?;
    let ctx0 = runtime2.device(0)?.clone();
    let ctx1 = runtime2.device(1)?.clone();

    let stream_pool2 = StreamPool::new(&[ctx0.clone(), ctx1.clone()])?;

    let mut kernel_registry2 = KernelRegistry::new();
    kernel_registry2.register_infers_kernels();

    let mut config_mmap = mmap_config.clone();
    config_mmap.max_position_embeddings = test_max_seq_len;
    let config_mmap = Arc::new(config_mmap);

   // --- Capture mmap INT4 companion shapes before ForwardEngine consumes mmap_weight_registries ---
    let mmap_int4_shapes: std::collections::HashMap<String, (Vec<usize>, Vec<usize>)> = mmap_weight_registries[0].int4_companions.iter()
        .map(|(name, comp)| (name.clone(), (comp.scales.shape().to_vec(), comp.qzeros.shape().to_vec())))
        .collect();

    // Check embedding key name in mmap registry vs heap keys
    let embed_keys_mmap: Vec<_> = mmap_weight_registries[0].tensors.keys()
        .filter(|k| k.contains("embed"))
        .collect();
    eprintln!("[diagnostic] Mmap keys containing 'embed': {:?}", embed_keys_mmap);

    // Check which tensors are strided in the mmap registry (only these go through memcpy2d upload)
    let strided_keys: Vec<_> = mmap_weight_registries[0].tensors.iter()
        .filter(|(_, t)| t.is_strided())
        .map(|(name, _)| name.clone())
        .collect();
    eprintln!("[diagnostic] Mmap shard 0 has {} strided tensors (memcpy2d upload)", strided_keys.len());
    if !strided_keys.is_empty() {
        eprintln!("[diagnostic] First 10 strided keys: {:?}", &strided_keys[..10.min(strided_keys.len())]);
    }

// Compare raw data bytes between heap and mmap for the same tensor
    let compare_key = "layers.0.linear_attn.in_proj_qkv.qweight";

    // Read from mmap registry (before engine consumes it) — compare with captured heap bytes
    if let Some(mmap_tensor) = mmap_weight_registries[0].tensors.get(compare_key) {
        let mmap_bytes = &mmap_tensor.data()[..16.min(mmap_tensor.data().len())];
        eprintln!("[diagnostic] Mmap '{}' dtype={:?} shape={:?} first 16 bytes: {:?}", 
            compare_key, mmap_tensor.dtype(), mmap_tensor.shape(), mmap_bytes);

        // Compare with captured heap bytes
        if let Some(ref heap_bytes) = heap_tensor_bytes {
            if *heap_bytes == *mmap_bytes {
                eprintln!("[diagnostic] Byte comparison '{}' — MATCH!", compare_key);
            } else {
                eprintln!("[diagnostic] Byte comparison '{}' — MISMATCH!!!", compare_key);
                for (i, (&h, &m)) in heap_bytes.iter().zip(mmap_bytes).enumerate() {
                    if h != m {
                        eprintln!("[diagnostic]   first diff at byte {}: heap={}, mmap={}", i, h, m);
                    }
                }
            }
        }
    }

    // Compare INT4 companion shapes between heap and mmap for the same weight
    if let Some(heap_companion) = heap_int4_shapes.get(compare_key) {
        eprintln!("[diagnostic] Heap companion scales shape={:?}, qzeros shape={:?}", 
            heap_companion.0, heap_companion.1);
    }
    if let Some(mmap_tensor) = mmap_weight_registries[0].int4_companions.get(compare_key) {
        eprintln!("[diagnostic] Mmap companion scales shape={:?}, qzeros shape={:?}", 
            mmap_tensor.scales.shape(), mmap_tensor.qzeros.shape());
        let scales_data = &mmap_tensor.scales.data()[..16.min(mmap_tensor.scales.data().len())];
        eprintln!("[diagnostic] Mmap companion scales first 16 bytes: {:?}", scales_data);
    }

    // Check element count for strided tensors
    let check_key = "layers.0.mlp.gate_proj.qweight";
    if let Some(mmap_tensor) = mmap_weight_registries[0].tensors.get(check_key) {
        let dtype = mmap_tensor.dtype();
        let elem_size = match dtype {
            infers_model::WeightDtype::Int4Packed => 4u64,
            infers_model::WeightDtype::Bf16 | infers_model::WeightDtype::Fp16 => 2,
            _ => 4,
        };
        if mmap_tensor.is_strided() {
            let strided_elems = (mmap_tensor.strided_width() as u64 * mmap_tensor.strided_rows() as u64) / elem_size;
            let shape_elems: u64 = mmap_tensor.shape().iter().map(|&d| d as u64).product();
            eprintln!("[diagnostic] Mmap '{}' strided: strided_elems={}, shape_elems={} {}", 
                check_key, strided_elems, shape_elems, if strided_elems == shape_elems { "MATCH" } else { "MISMATCH!!!" });
        }
    }

    // Capture BF16 source bytes from mmap registry before engine consumes it
    let bf16_test_name = "layers.0.input_layernorm.weight";
    let mut mmap_bf16_source: Option<Vec<u8>> = None;
    if let Some(t) = mmap_weight_registries[0].tensors.get(bf16_test_name) {
        eprintln!("[diagnostic] Mmap '{}' dtype={:?} shape={:?} first 32 bytes: {:?}", 
            bf16_test_name, t.dtype(), t.shape(), &t.data()[..32.min(t.data().len())]);
        mmap_bf16_source = Some(t.data()[..].to_vec());
    }

    // Capture INT4 qweight source bytes from mmap registry before engine consumes it
    let int4_test_name = "layers.0.mlp.gate_proj.qweight";
    let mut mmap_int4_source: Option<Vec<u8>> = None;
    if let Some(t) = mmap_weight_registries[0].tensors.get(int4_test_name) {
        eprintln!("[diagnostic] Mmap '{}' dtype={:?} shape={:?} first 32 bytes: {:?}", 
            int4_test_name, t.dtype(), t.shape(), &t.data()[..32.min(t.data().len())]);
        mmap_int4_source = Some(t.data()[..].to_vec());
    }

    // DIAGNOSTIC: Heap upload path does not need a pinned buffer.

    // =========================================================
    // DIAGNOSTIC: Convert mmap registry to heap WeightRegistry,
    // then upload via the HEAP path (GpuWeightCache::new).
    // If this produces matching tokens, the issue is in mmap upload code.
    // If still wrong, the issue is elsewhere (metadata, inference, etc.).
    // =========================================================
    eprintln!("\n=== PHASE 2: Mmap data via HEAP upload path ===");

    use bytes::Bytes;

    // Convert each MmapWeightRegistry shard to a WeightRegistry with heap-backed data
    let mut heap_from_mmap_registries: Vec<infers_model::WeightRegistry> = Vec::new();
    for mmap_shard in &mmap_weight_registries {
        let mut heap_reg = infers_model::WeightRegistry::new();

        // Convert MmapTensor → WeightData (copies bytes from mmap to heap)
        for (name, tensor) in &mmap_shard.tensors {
            let data = Bytes::copy_from_slice(tensor.data());
            heap_reg.tensors.insert(name.clone(), infers_model::WeightData {
                data,
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype(),
                name: name.clone(),
            });
        }

        // Convert MmapCompanions → Int4Companions
        for (name, companions) in &mmap_shard.int4_companions {
            let scales_data = Bytes::copy_from_slice(companions.scales.data());
            let qzeros_data = Bytes::copy_from_slice(companions.qzeros.data());
            heap_reg.int4_companions.insert(name.clone(), infers_model::Int4Companions {
                scales: infers_model::WeightData {
                    data: scales_data,
                    shape: companions.scales.shape().to_vec(),
                    dtype: companions.scales.dtype(),
                    name: companions.scales.name().to_string(),
                },
                qzeros: infers_model::WeightData {
                    data: qzeros_data,
                    shape: companions.qzeros.shape().to_vec(),
                    dtype: companions.qzeros.dtype(),
                    name: companions.qzeros.name().to_string(),
                },
            });
        }

        // Build main layers (embedding/norm/lm_head/layers) for inference lookup
        infers_model::build_main_layers(&mut heap_reg, &mmap_config)?;
        heap_from_mmap_registries.push(heap_reg);
    }

    let engine_start = Instant::now();
    let mut mmap_engine = ForwardEngine::new(
        config_mmap.clone(),
        heap_from_mmap_registries,
        vec![ctx0, ctx1],
        kernel_registry2,
        stream_pool2,
        group_size,
    )?;
    let mmap_engine_time = engine_start.elapsed().as_secs_f64();
    eprintln!("Mmap engine init: {:.2}s", mmap_engine_time);

    // --- GPU buffer diagnostic: download weights from GPU and compare with source ---
    {
        let stream = runtime2.default_stream(0)?;
        let cache = &mmap_engine.weight_caches()[0];

        // BF16 comparison: layers.0.input_layernorm.weight
        if let Some(ref src) = mmap_bf16_source {
            if let Some(gpu_data) = cache.download_bf16(bf16_test_name, &stream) {
                eprintln!("[diagnostic] BF16 download '{}' GPU len={}, source len={}", 
                    bf16_test_name, gpu_data.len(), src.len() / 2);
                
                let src_bf16: Vec<half::bf16> = src.chunks_exact(2)
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                
                if src_bf16.len() == gpu_data.len() {
                    let mut mismatches = 0;
                    for (i, (s, g)) in src_bf16.iter().zip(gpu_data.iter()).enumerate() {
                        if s.to_bits() != g.to_bits() {
                            mismatches += 1;
                            if mismatches <= 5 {
                                eprintln!("[diagnostic]   BF16 MISMATCH at {}: source=0x{:04x} gpu=0x{:04x}", 
                                    i, s.to_bits(), g.to_bits());
                            }
                        }
                    }
                    eprintln!("[diagnostic] BF16 '{}' comparison: {}/{} mismatches", bf16_test_name, mismatches, src_bf16.len());
                } else {
                    eprintln!("[diagnostic] BF16 '{}' length mismatch: GPU={}, source={}", 
                        bf16_test_name, gpu_data.len(), src_bf16.len());
                }
            } else {
                eprintln!("[diagnostic] BF16 download failed for '{}'", bf16_test_name);
            }
        }

        // INT4 comparison: layers.0.mlp.gate_proj.qweight
        if let Some(ref src) = mmap_int4_source {
            // Debug: check what type the weight actually is in cache
            match cache.get(int4_test_name) {
                Some(CachedWeight::Int4(b)) => eprintln!("[diagnostic] INT4 '{}' found as Int4, shape={:?}", int4_test_name, b.shape),
                Some(CachedWeight::Bf16(s)) => eprintln!("[diagnostic] WARNING: '{}' stored as Bf16 (len={}) instead of Int4!", int4_test_name, s.len()),
                None => eprintln!("[diagnostic] ERROR: '{}' NOT FOUND in mmap cache!", int4_test_name),
            }
            if let Some(gpu_data) = cache.download_int4_qweight(int4_test_name, &stream) {
                eprintln!("[diagnostic] INT4 download '{}' GPU len={}, source len={}", 
                    int4_test_name, gpu_data.len(), src.len() / 4);
                
                let src_u32: Vec<u32> = src.chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                
                if src_u32.len() == gpu_data.len() {
                    let mut mismatches = 0;
                    for (i, (s, g)) in src_u32.iter().zip(gpu_data.iter()).enumerate() {
                        if s != g {
                            mismatches += 1;
                            if mismatches <= 5 {
                                eprintln!("[diagnostic]   INT4 MISMATCH at {}: source=0x{:08x} gpu=0x{:08x}", 
                                    i, s, g);
                            }
                        }
                    }
                    eprintln!("[diagnostic] INT4 '{}' comparison: {}/{} mismatches", int4_test_name, mismatches, src_u32.len());
                } else {
                    eprintln!("[diagnostic] INT4 '{}' length mismatch: GPU={}, source={}", 
                        int4_test_name, gpu_data.len(), src_u32.len());
                }
            } else {
                eprintln!("[diagnostic] INT4 download failed for '{}'", int4_test_name);
            }
        }
    }

    // --- Diagnostic: compare mmap cache with heap cache ---
    let mmap_keys_gpu0: Vec<String> = mmap_engine.weight_caches()[0].keys().map(|s| s.to_string()).collect();
    eprintln!("[diagnostic] Mmap cache[0]: {} weights", mmap_engine.weight_caches()[0].len());

    // Key set comparison
    let heap_key_set: std::collections::HashSet<_> = heap_keys_gpu0.iter().cloned().collect();
    let mmap_key_set: std::collections::HashSet<_> = mmap_keys_gpu0.iter().cloned().collect();
    let only_heap: Vec<&String> = heap_key_set.difference(&mmap_key_set).collect();
    let only_mmap: Vec<&String> = mmap_key_set.difference(&heap_key_set).collect();

    if !only_heap.is_empty() {
        eprintln!("[diagnostic] Keys ONLY in heap (count={}): {:?}", only_heap.len(), only_heap);
    } else {
        eprintln!("[diagnostic] No keys only in heap");
    }
    if !only_mmap.is_empty() {
        eprintln!("[diagnostic] Keys ONLY in mmap (count={}): {:?}", only_mmap.len(), only_mmap);
    } else {
        eprintln!("[diagnostic] No keys only in mmap");
    }

    // Dtype comparison for common keys — use captured heap_dtype_gpu0 (engine_heap is dropped)
    let heap_dtype_map: std::collections::HashMap<String, String> = heap_dtype_gpu0.into_iter().collect();
    let common_keys: Vec<_> = heap_key_set.intersection(&mmap_key_set).collect();
    eprintln!("[diagnostic] Common keys: {}", common_keys.len());

   let mut dtype_mismatches = 0;
    for key in &common_keys {
        let heap_dtype = heap_dtype_map.get(key.as_str()).map(|s| s.as_str()).unwrap_or("???");
        let mmap_dtype = match mmap_engine.weight_caches()[0].get(key) {
            Some(CachedWeight::Bf16(_)) => "Bf16",
            Some(CachedWeight::Int4(_)) => "Int4",
            None => continue,
        };
       if heap_dtype != mmap_dtype {
            eprintln!("[diagnostic] Dtype mismatch '{}': heap={}, mmap={}", key, heap_dtype, mmap_dtype);
            dtype_mismatches += 1;
        }
    }
    eprintln!("[diagnostic] Dtype mismatches: {}", dtype_mismatches);

    // Compare INT4 companion shapes between heap and mmap for a specific weight
    let test_weight = "layers.0.linear_attn.in_proj_qkv.qweight";
    if let Some(heap_comp) = heap_int4_shapes.get(test_weight) {
        eprintln!("[diagnostic] Heap companion for '{}': scales={:?}, qzeros={:?}", test_weight, heap_comp.0, heap_comp.1);
    } else {
        eprintln!("[diagnostic] Heap no companion for '{}'", test_weight);
    }

    // Check the mmap registry's INT4 companions directly
    let mmap_int4_count = mmap_int4_shapes.len();
    let heap_int4_count = heap_int4_shapes.len();
    eprintln!("[diagnostic] Heap int4_companions entries: {}, Mmap int4_companions entries: {}", heap_int4_count, mmap_int4_count);

    if let Some(mmap_comp) = mmap_int4_shapes.get(test_weight) {
        eprintln!("[diagnostic] Mmap companion for '{}': scales={:?}, qzeros={:?}", 
            test_weight, mmap_comp.0, mmap_comp.1);

        // Check if shapes match
        if let Some(heap_comp) = heap_int4_shapes.get(test_weight) {
            if heap_comp.0 != mmap_comp.0 {
                eprintln!("[diagnostic] SCALE SHAPE MISMATCH for '{}': heap={:?}, mmap={:?}", test_weight, heap_comp.0, mmap_comp.0);
            } else {
                eprintln!("[diagnostic] Scale shapes MATCH for '{}'", test_weight);
            }
            if heap_comp.1 != mmap_comp.1 {
                eprintln!("[diagnostic] QZERO SHAPE MISMATCH for '{}': heap={:?}, mmap={:?}", test_weight, heap_comp.1, mmap_comp.1);
            } else {
                eprintln!("[diagnostic] Qzero shapes MATCH for '{}'", test_weight);
            }
        }
    } else {
        eprintln!("[diagnostic] Mmap NO companion for '{}'", test_weight);
    }

    // Specific weight checks (try both with/without .qweight suffix)
    for &name in &["layers.0.self_attn.q_proj.qweight", "layers.0.self_attn.q_proj"] {
        if let Some(w) = mmap_engine.weight_caches()[0].get(name) {
            match w {
                CachedWeight::Bf16(_) => eprintln!("[diagnostic] Mmap {} → Bf16", name),
                CachedWeight::Int4(b) => eprintln!("[diagnostic] Mmap {} → Int4 (shape={:?})", name, b.shape),
            }
        } else {
            eprintln!("[diagnostic] Mmap missing: {}", name);
        }
    }
    // Check a few more critical keys exist in mmap
    for &name in &[
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.mlp.gate_proj",
        "layers.0.norm1",
    ] {
        let exists = mmap_engine.weight_caches()[0].get(name).is_some();
        eprintln!("[diagnostic] Mmap '{}' exists: {}", name, exists);
    }

   // Check embedding weight buffer size in mmap cache
    if let Some(w) = mmap_engine.weight_caches()[0].get("embed_tokens.weight") {
        match w {
            CachedWeight::Bf16(s) => eprintln!("[diagnostic] Mmap embed_tokens Bf16 len: {}", s.len()),
            CachedWeight::Int4(_) => eprintln!("[diagnostic] WARNING: Mmap embed_tokens is Int4?"),
        }
    } else {
        eprintln!("[diagnostic] Mmap missing: embed_tokens.weight");
    }

    // Check BF16 norm weight sizes for mmap
    for test_key in &["input_layernorm", "post_attention_layernorm"] {
        let key = format!("layers.0.{}", test_key);
        if let Some(w) = mmap_engine.weight_caches()[0].get(&key) {
            match w {
                CachedWeight::Bf16(s) => eprintln!("[diagnostic] Mmap '{}' Bf16 len: {}", key, s.len()),
                CachedWeight::Int4(_) => eprintln!("[diagnostic] WARNING: Mmap '{}' is Int4?", key),
            }
        } else {
            eprintln!("[diagnostic] Mmap missing: {}", key);
        }
    }

    // Check dtype of the extra MTP keys to understand the INT4 count difference
    for key in &only_mmap {
        let mtp_dtype = match mmap_engine.weight_caches()[0].get(key) {
            Some(CachedWeight::Bf16(_)) => "Bf16",
            Some(CachedWeight::Int4(_)) => "Int4",
            None => continue,
        };
        eprintln!("[diagnostic] MTP key '{}' → {}", key, mtp_dtype);
    }

    // Also check cache[1] (GPU 1) for comparison
    let mmap_keys_gpu1: Vec<String> = mmap_engine.weight_caches()[1].keys().map(|s| s.to_string()).collect();
    eprintln!("[diagnostic] Mmap cache[1]: {} weights", mmap_keys_gpu1.len());

    // Check for missing INT4 companion lookups by counting Int4 entries
    let mmap_int4_count: usize = mmap_engine.weight_caches()[0].keys()
        .filter(|k| matches!(mmap_engine.weight_caches()[0].get(k), Some(CachedWeight::Int4(_))))
        .count();
    eprintln!("[diagnostic] INT4 weights — heap: {}, mmap: {}", heap_int4_count_gpu0, mmap_int4_count);

    // Run inference via mmap path
    let (_mmap_init, mmap_prefill_time, mmap_tokens) = run_inference(
        &mut mmap_engine,
        &runtime2,
        num_pages,
        16,
        max_cache_bytes,
        &token_ids,
    )?;

    eprintln!("Mmap tokens ({}): {:?}", mmap_tokens.len(), mmap_tokens);
    if let Some(ref tok) = tokenizer {
        let text = tok.decode(&mmap_tokens)?;
        eprintln!("Mmap decoded: {}", text.trim());
    }

    // =========================================================
    // PHASE 3: Compare results
    // =========================================================
    eprintln!("\n=== Comparison ===");
    eprintln!("[compare] Heap engine init: {:.2}s, Prefill: {:.3}s", heap_engine_time, heap_prefill_time);
    eprintln!("[compare] Mmap engine init: {:.2}s, Prefill: {:.3}s", mmap_engine_time, mmap_prefill_time);
    eprintln!("[compare] Tokens match: {}", heap_tokens == mmap_tokens);

    if heap_tokens != mmap_tokens {
        for (i, (h, m)) in heap_tokens.iter().zip(&mmap_tokens).enumerate() {
            if h != m {
                eprintln!("[compare] First mismatch at position {}: heap={}, mmap={}", i, h, m);
            }
        }
    }

    assert_eq!(heap_tokens, mmap_tokens, "mmap path produced different tokens than heap path");

    eprintln!("\nSmoke test PASSED: heap and mmap token sequences match!");
    Ok(())
}
