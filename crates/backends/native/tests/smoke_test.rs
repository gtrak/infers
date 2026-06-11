//! End-to-end smoke test with real Qwen3.6-27B AutoRound INT4 model.
//!
//! Requires:
//! - GPU with CUDA compute capability 12.0+ (Blackwell)
//! - Model downloaded to INFERS_TEST_MODEL path (default: ~/opt/vllm/models/qwen3.6-27b-autoround-int4/)
//! - ~30s to load model and run inference
//!
//! Run with: cargo test --package infers-backend-native --test smoke_test smoke_test_real_model -- --ignored --nocapture

use std::path::Path;
use std::sync::Arc;

use infers_backend_native::ForwardEngine;
use infers_cuda::context::CudaRuntime;
use infers_cuda::kernels::KernelRegistry;
use infers_cuda::stream::StreamPool;
use infers_model::load_model;

/// Default path for the Qwen3.6-27B AutoRound INT4 model.
const DEFAULT_MODEL_DIR: &str = "~/opt/vllm/models/qwen3.6-27b-autoround-int4/";

/// Smoke test that loads a real model and runs prefill + decode.
///
/// Verifies:
/// - Model loads without errors
/// - Engine initializes without CUDA errors
/// - Prefill produces a valid token (in vocab range, non-zero)
/// - Decode produces valid tokens for 10 steps
#[test]
#[ignore = "requires GPU and real model weights"]
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

    // 1. Load model config and weights
    let loaded = load_model(&model_dir)?;
    eprintln!(
        "Model loaded: {} layers, {} tensors",
        loaded.config.num_hidden_layers,
        loaded.weights.num_tensors(),
    );

    // 2. Initialize CUDA runtime and get context for device 0
    let runtime = CudaRuntime::new()?;
    let ctx = runtime.device(0)?.clone();

    // 3. Create a CUDA stream for kernel launches
    let stream = runtime.default_stream(0)?;

    // 4. Register and load kernels
    let mut kernel_registry = KernelRegistry::new();
    kernel_registry.register_infers_kernels();

    // 5. Create stream pool from the CUDA context
    let stream_pool = StreamPool::new(&[ctx.clone()])?;

    // 6. Create the forward engine
    let config = Arc::new(loaded.config.clone());
    let group_size = 128; // Standard for AutoRound INT4
    let mut engine = ForwardEngine::new(
        config.clone(),
        vec![loaded.weights],
        ctx,
        kernel_registry,
        stream_pool,
        group_size,
    )?;

    eprintln!("Engine initialized successfully");

    // 7. Run prefill with a known-good prompt prefix
    // Using BOS token (typically 151643 for Qwen) + a few known tokens
    // In a real scenario, these would come from the tokenizer
    let token_ids = vec![0u32]; // BOS or padding token

    let sampled = engine.prefill(&stream, &token_ids)?;
    eprintln!("Prefill sampled token: {}", sampled);

    // 8. Verify prefill output is valid
    assert!(
        sampled < config.vocab_size as u32,
        "Sampled token {} >= vocab_size {}",
        sampled,
        config.vocab_size
    );
    assert_ne!(sampled, 0, "Sampled token should not be padding");

    // 9. Run decode for 10 steps, verify all tokens are valid
    let mut token = sampled;
    for pos in token_ids.len()..token_ids.len() + 10 {
        token = engine.decode(&stream, token, pos as u32)?;
        eprintln!("Decode step {}: token={}", pos, token);

        assert!(
            token < config.vocab_size as u32,
            "Decode token {} >= vocab_size {} at position {}",
            token,
            config.vocab_size,
            pos
        );
    }

    eprintln!("Smoke test PASSED: {} tokens generated", 1 + 10);
    Ok(())
}
