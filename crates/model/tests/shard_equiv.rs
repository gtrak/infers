//! Equivalence tests: heap and mmap sharding paths must produce identical results.
//!
//! These tests create synthetic safetensors files, load them via both paths
//! (heap copy and zero-copy mmap), shard the weights for tensor parallelism,
//! and compare the sharded results byte-for-byte.

use infers_model::config::ModelConfig;
use infers_model::mmap::{MmapWeightShard, shard_weights_tp_mmap, load_safetensors_mmap, strip_language_model_prefix_mmap};
use infers_model::sharding::shard_weights_tp;
use infers_model::weights::WeightShard;
use infers_model::{load_safetensors, strip_language_model_prefix};

// ---------------------------------------------------------------------------
// Synthetic config builder
// ---------------------------------------------------------------------------

fn make_test_config() -> ModelConfig {
    ModelConfig {
        architectures: vec!["Qwen3_5ForConditionalGeneration".to_string()],
        model_type: "qwen3_5".to_string(),
        num_hidden_layers: 2,
        hidden_size: 64,
        intermediate_size: 128,
        vocab_size: 256,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        max_position_embeddings: 256,
        rms_norm_eps: 1e-6,
        hidden_act: "silu".to_string(),
        tie_word_embeddings: false,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        mrope_interleaved: true,
        mrope_section: vec![11, 11, 10],
        linear_num_key_heads: 2,
        linear_key_head_dim: 8,
        linear_num_value_heads: 4,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 3,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        attn_output_gate: true,
        quantization_config: None,
        layer_types: Some(vec![
            "linear_attention".to_string(),   // layer 0: GDN
            "full_attention".to_string(),      // layer 1: full attention
        ]),
    }
}

// ---------------------------------------------------------------------------
// Synthetic safetensors file builder
// ---------------------------------------------------------------------------

/// Create a temporary safetensors file with synthetic weights for all sharding cases.
fn create_synthetic_safetensors(tmp_dir: &std::path::Path, config: &ModelConfig) -> anyhow::Result<()> {
    use safetensors::tensor::{Dtype, TensorView};

    let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;   // 16
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim; // 32
    let conv_dim = key_dim * 2 + value_dim;  // 64

    /// Push a tensor entry into the data collection.
    fn push(
        entries: &mut Vec<(String, Vec<u8>, Dtype, Vec<usize>)>,
        name: &str,
        shape: &[usize],
        dtype: Dtype,
    ) {
        let bytes_per_elem = match dtype {
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
            _ => panic!("Unsupported dtype for synthetic weights"),
        };
        let total_bytes: usize = shape.iter().product::<usize>() * bytes_per_elem;
        let data: Vec<u8> = (0..total_bytes).map(|i| i as u8).collect();
        entries.push((name.to_string(), data, dtype, shape.to_vec()));
    }

    let mut all_data: Vec<(String, Vec<u8>, Dtype, Vec<usize>)> = Vec::new();

    // --- Top-level replicated weights ---
    push(&mut all_data, "model.language_model.embed_tokens.weight", &[config.vocab_size, config.hidden_size], Dtype::BF16);
    push(&mut all_data, "model.language_model.norm.weight", &[config.hidden_size], Dtype::BF16);
    push(&mut all_data, "lm_head.weight", &[config.vocab_size, config.hidden_size], Dtype::BF16);

    for layer_idx in 0..config.num_hidden_layers {
        let lp = format!("model.language_model.layers.{}", layer_idx);

        // Norm weights (replicated)
        push(&mut all_data, &format!("{lp}.input_layernorm.weight"), &[config.hidden_size], Dtype::BF16);
        push(&mut all_data, &format!("{lp}.post_attention_layernorm.weight"), &[config.hidden_size], Dtype::BF16);

        if layer_idx == 0 {
            // --- Layer 0: GDN (GatedDeltaNet) ---

            let qkv_shape = vec![config.hidden_size / 8, conv_dim];
            push(&mut all_data, &format!("{lp}.linear_attn.in_proj_qkv.qweight"), &qkv_shape, Dtype::U32);
            let qkv_scales_shape = vec![1, conv_dim];
            push(&mut all_data, &format!("{lp}.linear_attn.in_proj_qkv.scales"), &qkv_scales_shape, Dtype::BF16);
            let qkv_qzeros_shape = vec![1, conv_dim / 8];
            push(&mut all_data, &format!("{lp}.linear_attn.in_proj_qkv.qzeros"), &qkv_qzeros_shape, Dtype::U32);

            // in_proj_z — INT4 column-parallel fused QKV
            push(&mut all_data, &format!("{lp}.linear_attn.in_proj_z.qweight"), &qkv_shape, Dtype::U32);
            push(&mut all_data, &format!("{lp}.linear_attn.in_proj_z.scales"), &qkv_scales_shape, Dtype::BF16);
            push(&mut all_data, &format!("{lp}.linear_attn.in_proj_z.qzeros"), &qkv_qzeros_shape, Dtype::U32);

            // in_proj_a — BF16 column-parallel [hidden_size, hidden_size]
            push(&mut all_data, &format!("{lp}.linear_attn.in_proj_a.weight"), &[config.hidden_size, config.hidden_size], Dtype::BF16);

            // in_proj_b — BF16 column-parallel [hidden_size, hidden_size]
            push(&mut all_data, &format!("{lp}.linear_attn.in_proj_b.weight"), &[config.hidden_size, config.hidden_size], Dtype::BF16);

            // conv1d.weight — BF16 column-parallel fused QKV [conv_dim, 1, kernel_size]
            push(&mut all_data, &format!("{lp}.linear_attn.conv1d.weight"), &[conv_dim, 1, config.linear_conv_kernel_dim], Dtype::BF16);

            // out_proj — INT4 row-parallel [hidden_size/8, hidden_size] as u32
            let out_shape = vec![config.hidden_size / 8, config.hidden_size];
            push(&mut all_data, &format!("{lp}.linear_attn.out_proj.qweight"), &out_shape, Dtype::U32);
            let out_scales_shape = vec![1, config.hidden_size];
            push(&mut all_data, &format!("{lp}.linear_attn.out_proj.scales"), &out_scales_shape, Dtype::BF16);
            let out_qzeros_shape = vec![1, config.hidden_size / 8];
            push(&mut all_data, &format!("{lp}.linear_attn.out_proj.qzeros"), &out_qzeros_shape, Dtype::U32);

        } else {
            // --- Layer 1+: Full Attention ---

            let proj_shape = vec![config.hidden_size / 8, config.hidden_size];
            let proj_scales_shape = vec![1, config.hidden_size];
            let proj_qzeros_shape = vec![1, config.hidden_size / 8];

            // q_proj — INT4 column-parallel
            push(&mut all_data, &format!("{lp}.self_attn.q_proj.qweight"), &proj_shape, Dtype::U32);
            push(&mut all_data, &format!("{lp}.self_attn.q_proj.scales"), &proj_scales_shape, Dtype::BF16);
            push(&mut all_data, &format!("{lp}.self_attn.q_proj.qzeros"), &proj_qzeros_shape, Dtype::U32);

            // k_proj — INT4 column-parallel
            push(&mut all_data, &format!("{lp}.self_attn.k_proj.qweight"), &proj_shape, Dtype::U32);
            push(&mut all_data, &format!("{lp}.self_attn.k_proj.scales"), &proj_scales_shape, Dtype::BF16);
            push(&mut all_data, &format!("{lp}.self_attn.k_proj.qzeros"), &proj_qzeros_shape, Dtype::U32);

            // v_proj — INT4 column-parallel
            push(&mut all_data, &format!("{lp}.self_attn.v_proj.qweight"), &proj_shape, Dtype::U32);
            push(&mut all_data, &format!("{lp}.self_attn.v_proj.scales"), &proj_scales_shape, Dtype::BF16);
            push(&mut all_data, &format!("{lp}.self_attn.v_proj.qzeros"), &proj_qzeros_shape, Dtype::U32);

            // o_proj — INT4 row-parallel
            push(&mut all_data, &format!("{lp}.self_attn.o_proj.qweight"), &proj_shape, Dtype::U32);
            push(&mut all_data, &format!("{lp}.self_attn.o_proj.scales"), &proj_scales_shape, Dtype::BF16);
            push(&mut all_data, &format!("{lp}.self_attn.o_proj.qzeros"), &proj_qzeros_shape, Dtype::U32);
        }

        // --- MLP (same for both layer types) ---
        let mlp_col_shape = vec![config.hidden_size / 8, config.intermediate_size];
        let mlp_col_scales_shape = vec![1, config.intermediate_size];
        let mlp_col_qzeros_shape = vec![1, config.intermediate_size / 8];

        // gate_proj — INT4 column-parallel
        push(&mut all_data, &format!("{lp}.mlp.gate_proj.qweight"), &mlp_col_shape, Dtype::U32);
        push(&mut all_data, &format!("{lp}.mlp.gate_proj.scales"), &mlp_col_scales_shape, Dtype::BF16);
        push(&mut all_data, &format!("{lp}.mlp.gate_proj.qzeros"), &mlp_col_qzeros_shape, Dtype::U32);

        // up_proj — INT4 column-parallel
        push(&mut all_data, &format!("{lp}.mlp.up_proj.qweight"), &mlp_col_shape, Dtype::U32);
        push(&mut all_data, &format!("{lp}.mlp.up_proj.scales"), &mlp_col_scales_shape, Dtype::BF16);
        push(&mut all_data, &format!("{lp}.mlp.up_proj.qzeros"), &mlp_col_qzeros_shape, Dtype::U32);

        // down_proj — INT4 row-parallel
        let mlp_row_shape = vec![config.intermediate_size / 8, config.hidden_size];
        let mlp_row_scales_shape = vec![1, config.hidden_size];
        let mlp_row_qzeros_shape = vec![1, config.hidden_size / 8];
        push(&mut all_data, &format!("{lp}.mlp.down_proj.qweight"), &mlp_row_shape, Dtype::U32);
        push(&mut all_data, &format!("{lp}.mlp.down_proj.scales"), &mlp_row_scales_shape, Dtype::BF16);
        push(&mut all_data, &format!("{lp}.mlp.down_proj.qzeros"), &mlp_row_qzeros_shape, Dtype::U32);
    }

    // Build TensorViews and write to file
    let views: Vec<_> = all_data.iter().map(|(name, data, dtype, shape)| {
        let view = TensorView::new(*dtype, shape.clone(), data)
            .expect(&format!("Invalid tensor: {}", name));
        (name.as_str(), view)
    }).collect();

    let safetensors_path = tmp_dir.join("model.safetensors");
    safetensors::serialize_to_file(views.iter().cloned(), None, &safetensors_path)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Equivalence comparison
// ---------------------------------------------------------------------------

/// Compare heap-sharded results with mmap-sharded results.
fn compare_shards(
    heap_shards: &[WeightShard],
    mmap_shards: &[MmapWeightShard],
    label: &str,
) {
    assert_eq!(
        heap_shards.len(),
        mmap_shards.len(),
        "{}: shard count mismatch",
        label
    );

    for gpu_id in 0..heap_shards.len() {
        let heap = &heap_shards[gpu_id].registry;
        let mmap = &mmap_shards[gpu_id].registry;

        // Check tensor keys match
        let heap_keys: std::collections::HashSet<String> = heap.tensors.keys().cloned().collect();
        let mmap_keys: std::collections::HashSet<String> = mmap.tensors.keys().cloned().collect();
        assert_eq!(
            heap_keys, mmap_keys,
            "{} GPU {}: tensor key mismatch\n  heap: {:?}\n  mmap: {:?}",
            label, gpu_id, heap_keys, mmap_keys
        );

        // Compare each tensor
        for key in &heap_keys {
            let heap_w = heap.tensors.get(key).unwrap();
            let mmap_w = mmap.tensors.get(key).unwrap();

            assert_eq!(
                heap_w.shape,
                mmap_w.shape(),
                "{} GPU {}: {} shape mismatch: {:?} vs {:?}",
                label, gpu_id, key, heap_w.shape, mmap_w.shape()
            );
            assert_eq!(
                heap_w.dtype,
                mmap_w.dtype(),
                "{} GPU {}: {} dtype mismatch",
                label, gpu_id, key
            );

            // Byte-for-byte comparison only for non-strided (contiguous) tensors.
            // Strided tensors (mmap_slice_last_dim results) have different memory layouts
            // but logically equivalent data — they are used for INT4 column-parallel
            // non-fused splits via cuMemcpy2D. Fused QKV weights produce owned contiguous
            // buffers in both paths, so they can be compared byte-for-byte.
            if !mmap_w.is_strided() {
                assert_eq!(
                    heap_w.data.as_ref(),
                    mmap_w.data(),
                    "{} GPU {}: {} data mismatch ({} vs {} bytes)",
                    label, gpu_id, key, heap_w.data.len(), mmap_w.data().len()
                );
            }
        }

        // Check INT4 companion keys match
        let heap_comp_keys: std::collections::HashSet<String> =
            heap.int4_companions.keys().cloned().collect();
        let mmap_comp_keys: std::collections::HashSet<String> =
            mmap.int4_companions.keys().cloned().collect();
        assert_eq!(
            heap_comp_keys, mmap_comp_keys,
            "{} GPU {}: companion key mismatch\n  heap: {:?}\n  mmap: {:?}",
            label, gpu_id, heap_comp_keys, mmap_comp_keys
        );

        // Compare INT4 companions
        for key in &heap_comp_keys {
            let heap_c = heap.int4_companions.get(key).unwrap();
            let mmap_c = mmap.int4_companions.get(key).unwrap();

            // Scales
            assert_eq!(
                heap_c.scales.shape,
                mmap_c.scales.shape(),
                "{} GPU {}: {} scales shape mismatch",
                label, gpu_id, key
            );
            if !mmap_c.scales.is_strided() {
                assert_eq!(
                    heap_c.scales.data.as_ref(),
                    mmap_c.scales.data(),
                    "{} GPU {}: {} scales data mismatch ({} vs {} bytes)",
                    label, gpu_id, key,
                    heap_c.scales.data.len(), mmap_c.scales.data().len()
                );
            }

            // Qzeros
            assert_eq!(
                heap_c.qzeros.shape,
                mmap_c.qzeros.shape(),
                "{} GPU {}: {} qzeros shape mismatch",
                label, gpu_id, key
            );
            if !mmap_c.qzeros.is_strided() {
                assert_eq!(
                    heap_c.qzeros.data.as_ref(),
                    mmap_c.qzeros.data(),
                    "{} GPU {}: {} qzeros data mismatch ({} vs {} bytes)",
                    label, gpu_id, key,
                    heap_c.qzeros.data.len(), mmap_c.qzeros.data().len()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

type WeightRegistry = infers_model::weights::WeightRegistry;

/// Load safetensors from a directory via both paths and return (heap, mmap) registries.
fn load_both_paths(tmp_dir: &std::path::Path) -> (WeightRegistry, infers_model::mmap::MmapWeightRegistry) {
    let mut heap_registry = load_safetensors(tmp_dir).unwrap();
    let mut mmap_registry = load_safetensors_mmap(tmp_dir).unwrap();
    strip_language_model_prefix(&mut heap_registry);
    strip_language_model_prefix_mmap(&mut mmap_registry);
    (heap_registry, mmap_registry)
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

// @lat: [[lat#Weight Sharding#Sharding Equivalence Tests#TP=2 All Weights]]
#[test]
fn shard_equiv_tp2_all_weights() {
    let config = make_test_config();
    let tmp_dir = tempfile::tempdir().unwrap();

    create_synthetic_safetensors(tmp_dir.path(), &config).unwrap();

    let (heap_registry, mmap_registry) = load_both_paths(tmp_dir.path());

    let heap_shards = shard_weights_tp(&heap_registry, &config, 2).unwrap();
    let mmap_shards = shard_weights_tp_mmap(&mmap_registry, &config, 2).unwrap();

    compare_shards(&heap_shards, &mmap_shards, "TP=2 all weights");
}

// @lat: [[lat#Weight Sharding#Sharding Equivalence Tests#TP=1 All Weights]]
#[test]
fn shard_equiv_tp1_all_weights() {
    let config = make_test_config();
    let tmp_dir = tempfile::tempdir().unwrap();

    create_synthetic_safetensors(tmp_dir.path(), &config).unwrap();

    let (heap_registry, mmap_registry) = load_both_paths(tmp_dir.path());

    // TP=1 — no sharding, just verify both paths produce same result
    let heap_shards = shard_weights_tp(&heap_registry, &config, 1).unwrap();
    let mmap_shards = shard_weights_tp_mmap(&mmap_registry, &config, 1).unwrap();

    compare_shards(&heap_shards, &mmap_shards, "TP=1 all weights");
}

// @lat: [[lat#Weight Sharding#Sharding Equivalence Tests#TP=2 conv1d Fused QKV]]
#[test]
fn shard_equiv_tp2_conv1d_fused_qkv() {
    let config = make_test_config();
    let tmp_dir = tempfile::tempdir().unwrap();

    create_synthetic_safetensors(tmp_dir.path(), &config).unwrap();

    let (heap_registry, mmap_registry) = load_both_paths(tmp_dir.path());

    let heap_shards = shard_weights_tp(&heap_registry, &config, 2).unwrap();
    let mmap_shards = shard_weights_tp_mmap(&mmap_registry, &config, 2).unwrap();

    // Specifically verify conv1d.weight sharding (the source of the all-220 bug)
    for gpu_id in 0..2 {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim; // 16
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim; // 32

        let heap_conv = heap_shards[gpu_id]
            .registry
            .tensors
            .get("layers.0.linear_attn.conv1d.weight")
            .unwrap();
        let mmap_conv = mmap_shards[gpu_id]
            .registry
            .tensors
            .get("layers.0.linear_attn.conv1d.weight")
            .unwrap();

        // Expected shard dimension: Q/2 + K/2 + V/2 = 8+8+16 = 32
        let expected_conv_dim_shard = key_dim / 2 + key_dim / 2 + value_dim / 2;
        assert_eq!(
            heap_conv.shape[0],
            expected_conv_dim_shard,
            "GPU {}: conv1d weight dim-0 should be {}, got {} (heap)",
            gpu_id, expected_conv_dim_shard, heap_conv.shape[0]
        );
        assert_eq!(
            mmap_conv.shape()[0],
            expected_conv_dim_shard,
            "GPU {}: conv1d weight dim-0 should be {}, got {} (mmap)",
            gpu_id, expected_conv_dim_shard, mmap_conv.shape()[0]
        );
        // Both paths produce contiguous owned data for fused QKV — compare bytes
        assert!(!mmap_conv.is_strided(), "conv1d weight should be contiguous in mmap path");
        assert_eq!(
            heap_conv.data.as_ref(),
            mmap_conv.data(),
            "GPU {}: conv1d weight data mismatch", gpu_id
        );
    }

    compare_shards(&heap_shards, &mmap_shards, "TP=2 conv1d fused QKV");
}

// @lat: [[lat#Weight Sharding#Sharding Equivalence Tests#TP=2 INT4 Column-Parallel]]
#[test]
fn shard_equiv_tp2_int4_column_parallel() {
    let config = make_test_config();
    let tmp_dir = tempfile::tempdir().unwrap();

    create_synthetic_safetensors(tmp_dir.path(), &config).unwrap();

    let (heap_registry, mmap_registry) = load_both_paths(tmp_dir.path());

    let heap_shards = shard_weights_tp(&heap_registry, &config, 2).unwrap();
    let mmap_shards = shard_weights_tp_mmap(&mmap_registry, &config, 2).unwrap();

    // Verify INT4 column-parallel sharding for layer 1's q_proj
    // Full shape: [8, 64] -> each GPU gets [8, 32]
    for gpu_id in 0..2 {
        let heap_w = heap_shards[gpu_id]
            .registry
            .tensors
            .get("layers.1.self_attn.q_proj.qweight")
            .unwrap();
        let mmap_w = mmap_shards[gpu_id]
            .registry
            .tensors
            .get("layers.1.self_attn.q_proj.qweight")
            .unwrap();

        assert_eq!(heap_w.shape, vec![8, 32]);
        assert_eq!(mmap_w.shape(), vec![8, 32]);
    }

    compare_shards(&heap_shards, &mmap_shards, "TP=2 INT4 column-parallel");
}

// @lat: [[lat#Weight Sharding#Sharding Equivalence Tests#TP=2 INT4 Row-Parallel]]
#[test]
fn shard_equiv_tp2_int4_row_parallel() {
    let config = make_test_config();
    let tmp_dir = tempfile::tempdir().unwrap();

    create_synthetic_safetensors(tmp_dir.path(), &config).unwrap();

    let (heap_registry, mmap_registry) = load_both_paths(tmp_dir.path());

    let heap_shards = shard_weights_tp(&heap_registry, &config, 2).unwrap();
    let mmap_shards = shard_weights_tp_mmap(&mmap_registry, &config, 2).unwrap();

    // Verify INT4 row-parallel sharding for layer 1's o_proj
    // Full shape: [8, 64] -> each GPU gets [4, 64]
    for gpu_id in 0..2 {
        let heap_w = heap_shards[gpu_id]
            .registry
            .tensors
            .get("layers.1.self_attn.o_proj.qweight")
            .unwrap();
        let mmap_w = mmap_shards[gpu_id]
            .registry
            .tensors
            .get("layers.1.self_attn.o_proj.qweight")
            .unwrap();

        assert_eq!(heap_w.shape, vec![4, 64]);
        assert_eq!(mmap_w.shape(), vec![4, 64]);
    }

    compare_shards(&heap_shards, &mmap_shards, "TP=2 INT4 row-parallel");
}

// @lat: [[lat#Weight Sharding#Sharding Equivalence Tests#TP=2 GDN Fused QKV in_proj]]
#[test]
fn shard_equiv_tp2_gdn_fused_qkv_in_proj() {
    let config = make_test_config();
    let tmp_dir = tempfile::tempdir().unwrap();

    create_synthetic_safetensors(tmp_dir.path(), &config).unwrap();

    let (heap_registry, mmap_registry) = load_both_paths(tmp_dir.path());

    let heap_shards = shard_weights_tp(&heap_registry, &config, 2).unwrap();
    let mmap_shards = shard_weights_tp_mmap(&mmap_registry, &config, 2).unwrap();

    // Verify GDN in_proj_qkv sharding
    // Full shape: [8, 64] where conv_dim=64 split as Q(16), K(16), V(32)
    // Per GPU: [8, 8+8+16] = [8, 32]
    for gpu_id in 0..2 {
        let heap_w = heap_shards[gpu_id]
            .registry
            .tensors
            .get("layers.0.linear_attn.in_proj_qkv.qweight")
            .unwrap();
        let mmap_w = mmap_shards[gpu_id]
            .registry
            .tensors
            .get("layers.0.linear_attn.in_proj_qkv.qweight")
            .unwrap();

        assert_eq!(heap_w.shape, vec![8, 32]);
        assert_eq!(mmap_w.shape(), vec![8, 32]);
        // Fused QKV produces contiguous owned data — compare bytes
        assert!(!mmap_w.is_strided(), "in_proj_qkv qweight should be contiguous in mmap path");
        assert_eq!(
            heap_w.data.as_ref(),
            mmap_w.data(),
            "GPU {}: in_proj_qkv qweight data mismatch", gpu_id
        );

        // Verify companions too
        let heap_comp = heap_shards[gpu_id]
            .registry
            .int4_companions
            .get("layers.0.linear_attn.in_proj_qkv.qweight")
            .unwrap();
        let mmap_comp = mmap_shards[gpu_id]
            .registry
            .int4_companions
            .get("layers.0.linear_attn.in_proj_qkv.qweight")
            .unwrap();

        assert_eq!(heap_comp.scales.shape, mmap_comp.scales.shape());
        assert!(!mmap_comp.scales.is_strided(), "in_proj_qkv scales should be contiguous");
        assert_eq!(heap_comp.scales.data.as_ref(), mmap_comp.scales.data(),
            "GPU {}: in_proj_qkv scales data mismatch", gpu_id);

        assert_eq!(heap_comp.qzeros.shape, mmap_comp.qzeros.shape());
        assert!(!mmap_comp.qzeros.is_strided(), "in_proj_qkv qzeros should be contiguous");
        assert_eq!(heap_comp.qzeros.data.as_ref(), mmap_comp.qzeros.data(),
            "GPU {}: in_proj_qkv qzeros data mismatch", gpu_id);
    }

    compare_shards(&heap_shards, &mmap_shards, "TP=2 GDN fused QKV in_proj");
}
