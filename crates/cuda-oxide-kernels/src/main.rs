//! Test binary for infers-cuda-oxide-kernels.
//!
//! Allocates test data, launches each kernel, and verifies the result
//! against a CPU reference.

#![feature(f16)]

use std::sync::Arc;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};

/// Convert f32 to bf16 bits (truncate — matches `cuda_device::tcgen05::f32_to_bf16`).
fn f32_to_bf16_cpu(val: f32) -> u16 {
    ((val.to_bits() >> 16) & 0xFFFF) as u16
}

/// Convert bf16 bits to f32.
fn bf16_to_f32_cpu(bf16_bits: u16) -> f32 {
    f32::from_bits((bf16_bits as u32) << 16)
}

type Stream = Arc<CudaStream>;

// ─── infers_add_bf16 ─────────────────────────────────────────────

fn test_add(ctx: &Arc<CudaContext>) -> bool {
    let stream: Stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const N: usize = 1024;
    let a_f32: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let b_f32: Vec<f32> = (0..N).map(|i| (N - i) as f32 * 0.5).collect();
    let a_bf16: Vec<u16> = a_f32.iter().map(|&x| f32_to_bf16_cpu(x)).collect();
    let b_bf16: Vec<u16> = b_f32.iter().map(|&x| f32_to_bf16_cpu(x)).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_bf16).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();

    module.infers_add_bf16(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &a_dev, &b_dev, &mut out_dev, N as u32,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();
    // Match kernel: bf16→f32, add in f32, convert back to bf16
    let expected: Vec<u16> = a_bf16.iter()
        .zip(b_bf16.iter())
        .map(|(&a, &b)| f32_to_bf16_cpu(bf16_to_f32_cpu(a) + bf16_to_f32_cpu(b)))
        .collect();

    out_host == expected
}

// ─── infers_embedding_gather_bf16 ────────────────────────────────

fn test_embedding_gather(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    // 4 tokens × 8 hidden, bf16 bits filled with token_id*hidden + dim
    const NUM_TOKENS: usize = 4;
    const HIDDEN: usize = 8;
    let weight: Vec<u16> = (0..NUM_TOKENS * HIDDEN)
        .map(|i| i as u16)
        .collect();

    // Gather token_ids [0, 2] → output should be weight[0*8+dim], weight[2*8+dim]
    let token_ids: Vec<i32> = vec![0, 2];
    let seq_len = token_ids.len() as u32;

    let weight_dev = DeviceBuffer::from_host(&stream, &weight).unwrap();
    let tid_dev = DeviceBuffer::from_host(&stream, &token_ids).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, (seq_len as usize) * HIDDEN).unwrap();

    module.infers_embedding_gather_bf16(
        &stream,
        LaunchConfig::for_num_elems((seq_len as u32) * (HIDDEN as u32)),
        &weight_dev, &tid_dev, &mut out_dev, seq_len, HIDDEN as u32,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: gather token 0 → [0..8], token 2 → [16..24]
    let mut expected: Vec<u16> = Vec::with_capacity((seq_len as usize) * HIDDEN);
    for pos in 0..seq_len as usize {
        let tid = token_ids[pos] as usize;
        for dim in 0..HIDDEN {
            expected.push(weight[tid * HIDDEN + dim]);
        }
    }

    out_host == expected
}

// ─── infers_silu_bf16 ────────────────────────────────────────────

fn test_silu(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const N: usize = 256;
    let x_f32: Vec<f32> = (0..N).map(|i| (i as f32 - 128.0) / 50.0).collect();
    let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();

    module.infers_silu_bf16(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &x_dev, &mut out_dev, N as u32,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

   // CPU reference: match kernel — read from bf16, convert to f32, compute sigmoid
    let expected: Vec<u16> = x_bf16.iter()
        .map(|&v| {
            let val = bf16_to_f32_cpu(v);
            f32_to_bf16_cpu(val / (1.0 + libm::expf(-val)))
        })
        .collect();

    out_host == expected
}

// ─── infers_silu_glu_bf16 ────────────────────────────────────────

fn test_silu_glu(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const N: usize = 256;
    let x_bf16: Vec<u16> = (0..N).map(|i| f32_to_bf16_cpu((i as f32 - 128.0) / 50.0)).collect();
    let g_bf16: Vec<u16> = (0..N).map(|i| f32_to_bf16_cpu((i as f32 + 64.0) / 40.0)).collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_bf16).unwrap();
    let g_dev = DeviceBuffer::from_host(&stream, &g_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();

    module.infers_silu_glu_bf16(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &x_dev, &g_dev, &mut out_dev, N as u32,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

   // CPU reference: match kernel — read from bf16, convert to f32, compute sigmoid
    let expected: Vec<u16> = x_bf16.iter()
        .zip(g_bf16.iter())
        .map(|(&xv_bits, &gv_bits)| {
            let xv = bf16_to_f32_cpu(xv_bits);
            let gv = bf16_to_f32_cpu(gv_bits);
            f32_to_bf16_cpu(xv * gv / (1.0 + libm::expf(-gv)))
        })
        .collect();

    out_host == expected
}

// ─── infers_attn_output_gate_bf16 ────────────────────────────────

fn test_attn_output_gate(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const N: usize = 256;
    let x_bf16: Vec<u16> = (0..N).map(|i| f32_to_bf16_cpu((i as f32 - 128.0) / 50.0)).collect();
    let g_bf16: Vec<u16> = (0..N).map(|i| f32_to_bf16_cpu((i as f32 + 64.0) / 40.0)).collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_bf16).unwrap();
    let g_dev = DeviceBuffer::from_host(&stream, &g_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();

    module.infers_attn_output_gate_bf16(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &x_dev, &g_dev, &mut out_dev, N as u32,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

   // CPU reference: match kernel — read from bf16, convert to f32, compute sigmoid
    let expected: Vec<u16> = x_bf16.iter()
        .zip(g_bf16.iter())
        .map(|(&xv_bits, &gv_bits)| {
            let xv = bf16_to_f32_cpu(xv_bits);
            let gv = bf16_to_f32_cpu(gv_bits);
            f32_to_bf16_cpu(xv / (1.0 + libm::expf(-gv)))
        })
        .collect();

    out_host == expected
}

// ─── infers_argmax_bf16 ──────────────────────────────────────────

fn test_argmax(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const BATCH: usize = 2;
    const VOCAB: usize = 16;

    // Row 0: max at index 5 (value 5.0), rest < 5
    // Row 1: max at index 11 (value 11.0), rest < 11
    let logits_f32: Vec<f32> = (0..BATCH * VOCAB)
        .map(|i| {
            let row = i / VOCAB;
            let col = i % VOCAB;
            if row == 0 && col == 5 { 5.0 }
            else if row == 1 && col == 11 { 11.0 }
            else { col as f32 * 0.1 } // all < 5 and < 11 respectively
        })
        .collect();

    let logits_bf16: Vec<u16> = logits_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let logits_dev = DeviceBuffer::from_host(&stream, &logits_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, BATCH).unwrap();

    // Launch: batch_size blocks, 256 threads each
    let launch = LaunchConfig {
        grid_dim: (BATCH as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    module.infers_argmax_bf16(
        &stream,
        launch,
        &logits_dev, &mut out_dev, BATCH as u32, VOCAB as u32,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();
    out_host == vec![5i32, 11i32]
}

// ─── infers_kv_cache_write_bf16 ──────────────────────────────────

fn test_kv_cache(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const SEQ_LEN: usize = 2;
    const HEAD_DIM: usize = 4;
    const MAX_SEQ_LEN: usize = 8;

    // K values: token 0 → [10, 11, 12, 13], token 1 → [14, 15, 16, 17]
    let k_bf16: Vec<u16> = (0..SEQ_LEN * HEAD_DIM).map(|i| (10 + i) as u16).collect();
    // V values: token 0 → [20, 21, 22, 23], token 1 → [24, 25, 26, 27]
    let v_bf16: Vec<u16> = (0..SEQ_LEN * HEAD_DIM).map(|i| (20 + i) as u16).collect();
    // positions[0] = 2, positions[1] = 5
    let positions: Vec<i32> = vec![2, 5];

    // kv_cache size: max_seq_len * head_dim for K part + same for V part
    let kv_size = MAX_SEQ_LEN * HEAD_DIM * 2;
    let k_dev = DeviceBuffer::from_host(&stream, &k_bf16).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v_bf16).unwrap();
    let pos_dev = DeviceBuffer::from_host(&stream, &positions).unwrap();
    let mut kv_dev = DeviceBuffer::<u16>::zeroed(&stream, kv_size).unwrap();

    module.infers_kv_cache_write_bf16(
        &stream,
        LaunchConfig::for_num_elems((SEQ_LEN as u32) * (HEAD_DIM as u32)),
        &k_dev, &v_dev, &mut kv_dev, &pos_dev,
        SEQ_LEN as u32, HEAD_DIM as u32, MAX_SEQ_LEN as u32,
    ).unwrap();

    let kv_host = kv_dev.to_host_vec(&stream).unwrap();

    // K part: position 2 → offset 2*4=8, position 5 → offset 5*4=20
    // V part: offset max_seq_len*head_dim + pos*head_dim+dim
    //   position 2 → offset 32+8=40, position 5 → offset 32+20=52

    kv_host[8..12] == [10, 11, 12, 13] &&
    kv_host[20..24] == [14, 15, 16, 17] &&
    kv_host[40..44] == [20, 21, 22, 23] &&
    kv_host[52..56] == [24, 25, 26, 27]
}

// ─── main ────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== infers-cuda-oxide-kernels: all kernel tests ===\n");

    let ctx: Arc<CudaContext> = CudaContext::new(0)?;
    let mut fail_count = 0u32;

    if test_add(&ctx) {
        println!("[PASS] infers_add_bf16");
    } else {
        eprintln!("[FAIL] infers_add_bf16");
        fail_count += 1;
    }

    if test_embedding_gather(&ctx) {
        println!("[PASS] infers_embedding_gather_bf16");
    } else {
        eprintln!("[FAIL] infers_embedding_gather_bf16");
        fail_count += 1;
    }

    if test_silu(&ctx) {
        println!("[PASS] infers_silu_bf16");
    } else {
        eprintln!("[FAIL] infers_silu_bf16");
        fail_count += 1;
    }

    if test_silu_glu(&ctx) {
        println!("[PASS] infers_silu_glu_bf16");
    } else {
        eprintln!("[FAIL] infers_silu_glu_bf16");
        fail_count += 1;
    }

    if test_attn_output_gate(&ctx) {
        println!("[PASS] infers_attn_output_gate_bf16");
    } else {
        eprintln!("[FAIL] infers_attn_output_gate_bf16");
        fail_count += 1;
    }

    if test_argmax(&ctx) {
        println!("[PASS] infers_argmax_bf16");
    } else {
        eprintln!("[FAIL] infers_argmax_bf16");
        fail_count += 1;
    }

    if test_kv_cache(&ctx) {
        println!("[PASS] infers_kv_cache_write_bf16");
    } else {
        eprintln!("[FAIL] infers_kv_cache_write_bf16");
        fail_count += 1;
    }

    println!("\n=== Summary: {} tests, {} failed ===", 7, fail_count);

    if fail_count > 0 {
        std::process::exit(1);
    }

    Ok(())
}
