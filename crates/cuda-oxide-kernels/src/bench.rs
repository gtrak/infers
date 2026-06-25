//! Kernel microbenchmark harness with real dumped inputs.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use cuda_core::{sys, CudaContext, CudaEvent, CudaStream, DeviceBuffer, LaunchConfig};
use infers_model::load_safetensors_mmap;

pub struct BenchConfig {
    pub dump_dir: PathBuf,
    pub model_dir: PathBuf,
    pub layer: usize,
    pub gpu: usize,
    pub stage: String,
    pub iterations: usize,
    pub warmup: usize,
    pub verify: bool,
}

/// Load a BF16 dump file from disk. Returns raw u16 bytes and shape.
pub fn load_dump_bf16(
    dump_dir: &Path,
    layer: usize,
    phase: &str,
    stage: &str,
    gpu: usize,
) -> Result<(Vec<u16>, Vec<usize>)> {
    // The .raw files are stored as: layer_{L}/{phase}/{stage}_gpu{G}.raw
    let raw_path = dump_dir.join(format!(
        "layer_{}/{}/{}_gpu{}.raw",
        layer, phase, stage, gpu
    ));

    if !raw_path.exists() {
        anyhow::bail!("Dump file not found: {:?}", raw_path);
    }

    load_dump_file(&raw_path)
}

fn load_dump_file(
    path: &Path,
) -> Result<(Vec<u16>, Vec<usize>)> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read dump file {:?}", path))?;

    // Try reading shape from .meta sidecar
    let meta_path = path.with_extension("meta");
    let shape = if meta_path.exists() {
        let meta_str = std::fs::read_to_string(&meta_path)
            .with_context(|| format!("Failed to read meta file {:?}", meta_path))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_str)
            .with_context(|| format!("Failed to parse meta file {:?}", meta_path))?;
        meta["shape"]
            .as_array()
            .map(|arr| arr.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default()
    } else {
        // Default: infer from data length (BF16 = 2 bytes per element)
        vec![1, data.len() / 2]
    };

    let bf16_data: Vec<u16> = data
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    Ok((bf16_data, shape))
}

/// Load a tensor from safetensors by name. Returns raw bytes and shape.
pub fn load_safetensor(
    model_dir: &Path,
    tensor_name: &str,
) -> Result<(Vec<u8>, Vec<usize>)> {
    let registry = load_safetensors_mmap(model_dir)
        .with_context(|| format!("Failed to load safetensors from {:?}", model_dir))?;

    let tensor = registry
        .tensors
        .get(tensor_name)
        .with_context(|| format!("Tensor not found: {}", tensor_name))?;

    let shape = tensor.shape().to_vec();
    let data: Vec<u8> = tensor.data().to_vec();
    Ok((data, shape))
}

/// Time a closure that launches kernels with per-iteration timing.
/// Returns (mean_us, median_us, min_us).
pub fn time_kernel_per_iter(
    stream: &Arc<CudaStream>,
    ctx: &Arc<CudaContext>,
    warmup: usize,
    iterations: usize,
    mut launch: impl FnMut(),
) -> Result<(f64, f64, f64)> {
    // Warmup
    for _ in 0..warmup {
        launch();
    }
    stream.synchronize()?;

    let mut times_us: Vec<f64> = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let flags = sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
        let start = ctx.new_event(Some(flags))?;
        let end = ctx.new_event(Some(flags))?;

        start.record(stream)?;
        launch();
        end.record(stream)?;
        end.synchronize()?;

        let ms = start.elapsed_ms(&end)?;
        times_us.push((ms as f64) * 1_000.0);
    }

    let mean = times_us.iter().sum::<f64>() / times_us.len() as f64;
    let min = times_us.iter().cloned().fold(f64::INFINITY, f64::min);

    // Median
    let mut sorted = times_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    Ok((mean, median, min))
}

/// Compute cosine similarity between two BF16 vectors (as f32).
fn cosine_similarity(a: &[u16], b: &[u16]) -> f64 {
    let a_f32: Vec<f32> = a.iter().map(|&v| f32::from_bits((v as u32) << 16)).collect();
    let b_f32: Vec<f32> = b.iter().map(|&v| f32::from_bits((v as u32) << 16)).collect();

    let dot: f64 = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| (*x * *y) as f64).sum();
    let norm_a: f64 = a_f32.iter().map(|x| (*x * *x) as f64).sum::<f64>().sqrt();
    let norm_b: f64 = b_f32.iter().map(|x| (*x * *x) as f64).sum::<f64>().sqrt();

    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ─── Bench cases ──────────────────────────────────────────

/// Benchmark infers_rmsnorm_bf16 with real dumped inputs.
pub fn bench_rmsnorm(ctx: &Arc<CudaContext>, cfg: &BenchConfig) -> Result<()> {
    let stream = ctx.default_stream();
    let module = crate::load_modules(ctx)?;

    println!("=== BENCH: infers_rmsnorm_bf16 ({}) ===", cfg.stage);

    // Input: residual.attn for norm2 input (GPU0)
    let (input_data, input_shape) = load_dump_bf16(
        &cfg.dump_dir,
        cfg.layer,
        "decode",
        "residual.attn",
        cfg.gpu,
    )?;
    println!("Input shape: {:?}", input_shape);

    // Norm2 (post-attention layernorm) weight from safetensors
    let weight_name = format!(
        "model.language_model.layers.{}.post_attention_layernorm.weight",
        cfg.layer
    );
    let (weight_bytes, weight_shape) = load_safetensor(&cfg.model_dir, &weight_name)?;
    println!("Weight shape: {:?}", weight_shape);

    let weight_data: Vec<u16> = weight_bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    // Upload to device
    let input_dev = DeviceBuffer::from_host(&stream, &input_data)?;
    let w_dev = DeviceBuffer::from_host(&stream, &weight_data)?;
    let rows = input_shape.get(0).copied().unwrap_or(1);
    let hidden = input_shape.get(1).copied().unwrap_or(weight_data.len());

    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, rows * hidden)?;

    let eps = 1e-6f32;
    let launch = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    // Warmup
    for _ in 0..cfg.warmup {
        module.norm.infers_rmsnorm_bf16(
            &stream,
            launch.clone(),
            &input_dev,
            &w_dev,
            &mut out_dev,
            hidden as u32,
            eps,
        )?;
    }
    stream.synchronize()?;

    // Time
    let times = time_kernel_per_iter(
        &stream,
        ctx,
        cfg.warmup,
        cfg.iterations,
        || {
            module.norm.infers_rmsnorm_bf16(
                &stream,
                launch.clone(),
                &input_dev,
                &w_dev,
                &mut out_dev,
                hidden as u32,
                eps,
            ).unwrap();
        },
    )?;

    println!("Timed: {} iters", cfg.iterations);
    println!("  mean: {:.1} µs/call", times.0);
    println!("  median: {:.1} µs/call", times.1);
    println!("  min: {:.1} µs/call", times.2);

    // Verify against dumped output
    if cfg.verify {
        let (ref_data, _) = load_dump_bf16(
            &cfg.dump_dir,
            cfg.layer,
            "decode",
            "mlp.norm2",
            cfg.gpu,
        )?;

        let out_host = out_dev.to_host_vec(&stream)?;
        let cos_sim = cosine_similarity(&out_host, &ref_data);
        println!(
            "Verify: cosine={:.5} {}",
            cos_sim,
            if cos_sim > 0.99 { "PASS" } else { "FAIL" }
        );
    }

    Ok(())
}

/// Benchmark int4_gemm_v3_ksplit_sm with real dumped inputs and weights.
pub fn bench_int4_gemm_ksplit_sm(ctx: &Arc<CudaContext>, cfg: &BenchConfig) -> Result<()> {
    let stream = ctx.default_stream();
    let module = crate::load_modules(ctx)?;

    println!("=== BENCH: int4_gemm_v3_ksplit_sm ({}) ===", cfg.stage);

    // Input: norm2 output (input to mlp.gate_proj) — for mlp.gate_proj stage
    let input_stage = match cfg.stage.as_str() {
        "mlp.gate_proj" | "mlp.up_proj" => "mlp.norm2",
        other => other, // fallback: use the same stage name
    };

    let (input_data, input_shape) = load_dump_bf16(
        &cfg.dump_dir,
        cfg.layer,
        "decode",
        input_stage,
        cfg.gpu,
    )?;
    println!("Input shape: {:?}", input_shape);

    // Determine K from input shape (hidden dim = 5120)
    let k = input_shape.get(1).copied().unwrap_or(5120);

    // Parse the weight name from the stage
    let base_name = cfg.stage.split('.').nth(1).unwrap_or("gate_proj");

    let qweight_name = format!(
        "model.language_model.layers.{}.mlp.{}.qweight",
        cfg.layer, base_name
    );
    let scales_name = format!(
        "model.language_model.layers.{}.mlp.{}.scales",
        cfg.layer, base_name
    );
    let qzeros_name = format!(
        "model.language_model.layers.{}.mlp.{}.qzeros",
        cfg.layer, base_name
    );

    let (qweight_bytes, qweight_shape) = load_safetensor(&cfg.model_dir, &qweight_name)?;
    println!("qweight shape: {:?}", qweight_shape);

    let (scales_bytes, scales_shape) = load_safetensor(&cfg.model_dir, &scales_name)?;
    println!("scales shape: {:?}", scales_shape);

    let (qzeros_bytes, qzeros_shape) = load_safetensor(&cfg.model_dir, &qzeros_name)?;
    println!("qzeros shape: {:?}", qzeros_shape);

    // Safetensors stores full model (TP=1). For GPU0 (TP=2), we need first half of N columns.
    let full_n = qweight_shape.get(1).copied().unwrap_or(17408);
    let n = full_n / 2; // GPU0 shard
    println!("Full N={}, GPU0 N={}", full_n, n);

    // Group size from scales: num_groups = scales_shape[0], so group_size = K / num_groups
    let num_groups_safetensor = scales_shape[0];
    let group_size = k / num_groups_safetensor;
    if group_size == 0 {
        anyhow::bail!("Cannot determine group_size");
    }

    // Slice qweight: [K/8, N_full] -> [K/8, n]
    let k8 = qweight_shape[0];
    let full_n_i32 = qweight_shape[1];
    let qweight_data: Vec<u32> = qweight_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Slice to GPU0's N columns: for each row of K/8, take first n i32 values
    let qweight_shard: Vec<u32> = (0..k8)
        .flat_map(|row| qweight_data[row * full_n_i32 .. row * full_n_i32 + n].iter().cloned())
        .collect();

    // Slice scales: [K/group_size, N_full] f16 -> [K/group_size, n] f16
    let num_groups = scales_shape[0];
    let full_n_f16 = scales_shape[1];
    let scales_data: Vec<u16> = scales_bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    let scales_shard: Vec<u16> = (0..num_groups)
        .flat_map(|row| scales_data[row * full_n_f16 .. row * full_n_f16 + n].iter().cloned())
        .collect();

    // Slice qzeros: [K/group_size, N_full/8] i32 -> [K/group_size, n/8] i32
    let zeros_rows = qzeros_shape[0];
    let full_n_zeros = qzeros_shape[1];
    let zeros_data: Vec<u32> = qzeros_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let n_zeros = n / 8;
    let qzeros_shard: Vec<u32> = (0..zeros_rows)
        .flat_map(|row| zeros_data[row * full_n_zeros .. row * full_n_zeros + n_zeros].iter().cloned())
        .collect();

    println!(
        "Sharded weights: qweight [{}, {}], scales [{}, {}], qzeros [{}, {}]",
        k8, n, num_groups, n, zeros_rows, n_zeros
    );

    // K_split for production kernels — typically 28 for Qwen3.6 with K=5120
    let k_split: u32 = 28;

    // Upload to device
    let input_dev = DeviceBuffer::from_host(&stream, &input_data)?;
    let weight_dev = DeviceBuffer::from_host(&stream, &qweight_shard)?;
    let scales_dev = DeviceBuffer::from_host(&stream, &scales_shard)?;
    let zeros_dev = DeviceBuffer::from_host(&stream, &qzeros_shard)?;

    // partial_sums: [K_SPLIT, N] f32
    let partial_sums_size = (k_split as usize) * n;
    let mut partial_sums_dev = DeviceBuffer::<f32>::zeroed(&stream, partial_sums_size)?;

    // output: [N] bf16
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, n)?;

    // Launch config for ksplit kernel with shared memory
    let launch_ksplit = LaunchConfig {
        grid_dim: (((n + 63) / 64) as u32, k_split, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: (group_size as usize * 2) as u32,
    };

    // Launch config for reduce kernel
    let launch_reduce = LaunchConfig {
        grid_dim: (((n + 63) / 64) as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warmup
    for _ in 0..cfg.warmup {
        module.int4.int4_gemm_v3_ksplit_sm(
            &stream,
            launch_ksplit.clone(),
            &mut partial_sums_dev,
            &weight_dev,
            &scales_dev,
            &zeros_dev,
            &input_dev,
            n as u32,
            k as u32,
            group_size as u32,
            1u32, // transposed
            k_split,
        )?;

        module.int4.reduce_partial_sums_bf16(
            &stream,
            launch_reduce.clone(),
            &mut out_dev,
            &partial_sums_dev,
            n as u32,
            k_split,
        )?;
    }
    stream.synchronize()?;

    // Time (ksplit + reduce together)
    let times = time_kernel_per_iter(
        &stream,
        ctx,
        cfg.warmup,
        cfg.iterations,
        || {
            module.int4.int4_gemm_v3_ksplit_sm(
                &stream,
                launch_ksplit.clone(),
                &mut partial_sums_dev,
                &weight_dev,
                &scales_dev,
                &zeros_dev,
                &input_dev,
                n as u32,
                k as u32,
                group_size as u32,
                1u32, // transposed
                k_split,
            ).unwrap();

            module.int4.reduce_partial_sums_bf16(
                &stream,
                launch_reduce.clone(),
                &mut out_dev,
                &partial_sums_dev,
                n as u32,
                k_split,
            ).unwrap();
        },
    )?;

    println!("Timed: {} iters (ksplit + reduce)", cfg.iterations);
    println!("  mean: {:.1} µs/call", times.0);
    println!("  median: {:.1} µs/call", times.1);
    println!("  min: {:.1} µs/call", times.2);

    // Verify against dumped output
    if cfg.verify {
        let (ref_data, _) = load_dump_bf16(
            &cfg.dump_dir,
            cfg.layer,
            "decode",
            "mlp.gate_proj",
            cfg.gpu,
        )?;

        let out_host = out_dev.to_host_vec(&stream)?;
        let cos_sim = cosine_similarity(&out_host, &ref_data);
        println!(
            "Verify: cosine={:.5} {}",
            cos_sim,
            if cos_sim > 0.99 { "PASS" } else { "FAIL" }
        );
    }

    Ok(())
}

/// Benchmark reduce_partial_sums_bf16 with synthetic input.
pub fn bench_reduce_partial_sums(ctx: &Arc<CudaContext>, cfg: &BenchConfig) -> Result<()> {
    let stream = ctx.default_stream();
    let module = crate::load_modules(ctx)?;

    println!("=== BENCH: reduce_partial_sums_bf16 ({}) ===", cfg.stage);

    // Synthetic input: [K_SPLIT, N] f32 with random-ish data
    let n: usize = 8704;
    let k_split: u32 = 28;
    let partial_sums_size = (k_split as usize) * n;

    let mut rng_state = 42u64;
    fn next_f32(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (*state >> 40) as u32;
        f32::from_bits((bits & 0x7F7FFFFFu32) + 0x3F000000u32) - 128.0
    }

    let partial_sums: Vec<f32> = (0..partial_sums_size)
        .map(|_| next_f32(&mut rng_state))
        .collect();

    println!("Synthetic input: [{}, {}]", k_split, n);

    // Upload to device
    let partial_sums_dev = DeviceBuffer::from_host(&stream, &partial_sums)?;
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, n)?;

    let launch_reduce = LaunchConfig {
        grid_dim: (((n + 63) / 64) as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warmup
    for _ in 0..cfg.warmup {
        module.int4.reduce_partial_sums_bf16(
            &stream,
            launch_reduce.clone(),
            &mut out_dev,
            &partial_sums_dev,
            n as u32,
            k_split,
        )?;
    }
    stream.synchronize()?;

    // Time
    let times = time_kernel_per_iter(
        &stream,
        ctx,
        cfg.warmup,
        cfg.iterations,
        || {
            module.int4.reduce_partial_sums_bf16(
                &stream,
                launch_reduce.clone(),
                &mut out_dev,
                &partial_sums_dev,
                n as u32,
                k_split,
            ).unwrap();
        },
    )?;

    println!("Timed: {} iters", cfg.iterations);
    println!("  mean: {:.1} µs/call", times.0);
    println!("  median: {:.1} µs/call", times.1);
    println!("  min: {:.1} µs/call", times.2);

    // No verification for synthetic data
    if cfg.verify {
        println!("Verify: N/A (synthetic input)");
    }

    Ok(())
}

/// Benchmark infers_silu_glu_bf16 with real dumped inputs.
pub fn bench_silu_glu(ctx: &Arc<CudaContext>, cfg: &BenchConfig) -> Result<()> {
    let stream = ctx.default_stream();
    let module = crate::load_modules(ctx)?;

    println!("=== BENCH: infers_silu_glu_bf16 ({}) ===", cfg.stage);

    // Input 1 (x): mlp.up_proj output for GPU0
    let (up_data, up_shape) = load_dump_bf16(
        &cfg.dump_dir,
        cfg.layer,
        "decode",
        "mlp.up_proj",
        cfg.gpu,
    )?;

    // Input 2 (gate): mlp.gate_proj output for GPU0
    let (gate_data, gate_shape) = load_dump_bf16(
        &cfg.dump_dir,
        cfg.layer,
        "decode",
        "mlp.gate_proj",
        cfg.gpu,
    )?;

    println!("up_proj shape: {:?}", up_shape);
    println!("gate_proj shape: {:?}", gate_shape);

    let n = up_data.len();

    // Upload to device
    let x_dev = DeviceBuffer::from_host(&stream, &up_data)?;
    let g_dev = DeviceBuffer::from_host(&stream, &gate_data)?;
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, n)?;

    // Warmup
    for _ in 0..cfg.warmup {
        module.activation.infers_silu_glu_bf16(
            &stream,
            LaunchConfig::for_num_elems(n as u32),
            &x_dev,
            &g_dev,
            &mut out_dev,
            n as u32,
        )?;
    }
    stream.synchronize()?;

    // Time
    let times = time_kernel_per_iter(
        &stream,
        ctx,
        cfg.warmup,
        cfg.iterations,
        || {
            module.activation.infers_silu_glu_bf16(
                &stream,
                LaunchConfig::for_num_elems(n as u32),
                &x_dev,
                &g_dev,
                &mut out_dev,
                n as u32,
            ).unwrap();
        },
    )?;

    println!("Timed: {} iters", cfg.iterations);
    println!("  mean: {:.1} µs/call", times.0);
    println!("  median: {:.1} µs/call", times.1);
    println!("  min: {:.1} µs/call", times.2);

    // Verify against dumped output
    if cfg.verify {
        let (ref_data, _) = load_dump_bf16(
            &cfg.dump_dir,
            cfg.layer,
            "decode",
            "mlp.silu",
            cfg.gpu,
        )?;

        let out_host = out_dev.to_host_vec(&stream)?;
        let cos_sim = cosine_similarity(&out_host, &ref_data);
        println!(
            "Verify: cosine={:.5} {}",
            cos_sim,
            if cos_sim > 0.99 { "PASS" } else { "FAIL" }
        );
    }

    Ok(())
}
