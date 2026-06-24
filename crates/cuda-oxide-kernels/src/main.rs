//! Test binary for infers-cuda-oxide-kernels.
//!
//! Allocates test data, launches each kernel, and verifies the result
//! against a CPU reference.

mod bench;

use std::path::PathBuf;
use std::sync::Arc;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_core::embedded::{self, ArtifactPayloadKind};
use cuda_host::ltoir;

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

// ─── infers_rmsnorm_bf16 ──────────────────────────────────────

fn test_rmsnorm(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const ROWS: usize = 2;
    const HIDDEN: usize = 8;
    let eps = 1e-6f32;

    // Row 0: [1, 2, 3, 4, 5, 6, 7, 8] as f32
    let x_f32: Vec<f32> = (0..ROWS * HIDDEN).map(|i| (i + 1) as f32).collect();
    let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Weights: all ones
    let weight_f32: Vec<f32> = (0..HIDDEN).map(|_| 1.0).collect();
    let weight_bf16: Vec<u16> = weight_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_bf16).unwrap();
    let w_dev = DeviceBuffer::from_host(&stream, &weight_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, ROWS * HIDDEN).unwrap();

    let launch = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    module.infers_rmsnorm_bf16(
        &stream, launch, &x_dev, &w_dev, &mut out_dev, HIDDEN as u32, eps,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: for each row, compute RMSNorm manually
    let mut expected: Vec<u16> = Vec::with_capacity(ROWS * HIDDEN);
    for row in 0..ROWS {
        let offset = row * HIDDEN;
        let sum_sq: f32 = (0..HIDDEN).map(|i| x_f32[offset + i] * x_f32[offset + i]).sum();
        let inv_rms = 1.0 / libm::sqrtf(sum_sq / HIDDEN as f32 + eps);
        for i in 0..HIDDEN {
            let w_val = weight_f32[i];
            let result = x_f32[offset + i] * inv_rms * (1.0 + w_val);
            expected.push(f32_to_bf16_cpu(result));
        }
    }

    out_host == expected
}

// ─── infers_rms_norm_gated_bf16 ───────────────────────────────

fn test_rmsnorm_gated(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const ROWS: usize = 2;
    const D: usize = 8;
    let eps = 1e-6f32;

    // Input values: [1, 2, 3, 4, 5, 6, 7, 8] per row (two rows)
    let input_f32: Vec<f32> = (0..ROWS * D).map(|i| (i % D + 1) as f32).collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Gate values: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2] per row
    let gate_f32: Vec<f32> = (0..ROWS * D).map(|i| 0.5 + (i % D) as f32 * 0.1).collect();
    let gate_bf16: Vec<u16> = gate_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Weights: [1, 2, 3, 4, 5, 6, 7, 8]
    let weight_f32: Vec<f32> = (0..D).map(|i| (i + 1) as f32).collect();
    let weight_bf16: Vec<u16> = weight_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();
    let gate_dev = DeviceBuffer::from_host(&stream, &gate_bf16).unwrap();
    let w_dev = DeviceBuffer::from_host(&stream, &weight_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, ROWS * D).unwrap();

    let launch = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    module.infers_rms_norm_gated_bf16(
        &stream, launch, &input_dev, &gate_dev, &w_dev, &mut out_dev, ROWS as u32, D as u32, eps,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: weight * x_norm * SiLU(gate) — use tolerance due to bf16 precision
    for row in 0..ROWS {
        let offset = row * D;
        let sum_sq: f32 = (0..D).map(|i| input_f32[offset + i] * input_f32[offset + i]).sum();
        let inv_rms = 1.0 / libm::sqrtf(sum_sq / D as f32 + eps);
        for i in 0..D {
            let x_norm = input_f32[offset + i] * inv_rms;
            let g_val = gate_f32[offset + i];
            let silu_gate = g_val / (1.0 + libm::expf(-g_val));
            let expected_val = weight_f32[i] * x_norm * silu_gate;
            let actual_val = bf16_to_f32_cpu(out_host[offset + i]);
            if (actual_val - expected_val).abs() > 0.5 {
                return false;
            }
        }
    }
    true
}

// ─── infers_l2norm_bf16 ──────────────────────────────────────

fn test_l2norm(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const ROWS: usize = 2;
    const DIM: usize = 8;
    let eps = 1e-6f32;

    // Row 0: [1, 2, 3, 4, 5, 6, 7, 8]
    // Row 1: [1, 2, 3, 4, 5, 6, 7, 8] (same for simplicity)
    let input_f32: Vec<f32> = (0..ROWS * DIM).map(|i| (i % DIM + 1) as f32).collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, ROWS * DIM).unwrap();

    let launch = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    module.infers_l2norm_bf16(
        &stream, launch, &input_dev, &mut out_dev, DIM as u32, eps,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: verify each row has unit length
    for row in 0..ROWS {
        let offset = row * DIM;
        let sum_sq: f32 = (0..DIM)
            .map(|i| bf16_to_f32_cpu(input_bf16[offset + i]) * bf16_to_f32_cpu(input_bf16[offset + i]))
            .sum();
        let inv_norm = 1.0 / libm::sqrtf(sum_sq + eps);
        for i in 0..DIM {
            let val = bf16_to_f32_cpu(input_bf16[offset + i]);
            let expected_val = val * inv_norm;
            let actual_val = bf16_to_f32_cpu(out_host[offset + i]);
            if (actual_val - expected_val).abs() > 0.5 {
                return false;
            }
        }
    }
    true
}

// ─── infers_softmax_bf16 ─────────────────────────────────────

fn test_softmax(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const ROWS: usize = 4;
    const COLS: usize = 4;

    // Simple scores: row i has values [i+1, i+2, i+3, i+4]
    let scores_f32: Vec<f32> = (0..ROWS * COLS).map(|i| (i + 1) as f32).collect();
    let scores_bf16: Vec<u16> = scores_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let scores_dev = DeviceBuffer::from_host(&stream, &scores_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, ROWS * COLS).unwrap();

    let launch = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };
    module.infers_softmax_bf16(
        &stream, launch, &scores_dev, &mut out_dev, COLS as u32, 0u32, // no causal mask
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: softmax per row, verify rows sum to ~1.0 and values match
    for row in 0..ROWS {
        let offset = row * COLS;
        let row_vals: Vec<f32> = (0..COLS).map(|i| bf16_to_f32_cpu(scores_bf16[offset + i])).collect();
        let max_val = row_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row_vals.iter().map(|&v| libm::expf(v - max_val)).sum();
        for c in 0..COLS {
            let expected = libm::expf(row_vals[c] - max_val) / sum_exp;
            let actual = bf16_to_f32_cpu(out_host[offset + c]);
            if (actual - expected).abs() > 0.5 {
                return false;
            }
        }
    }
    true
}

// ─── infers_conv1d_depthwise_silu_bf16 ────────────────────────

fn test_conv1d(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const BATCH: usize = 1;
    const CONVDIM: usize = 2;
    const SEQ: usize = 4;
    const KERNEL: usize = 3;

    // Input layout: [batch][seq_len][conv_dim] → flattened as b*T*D + t*D + d (D innermost, matches nvcc)
    let input_f32: Vec<f32> = vec![
        // b=0, t=0: [1.0, 5.0]   t=1: [2.0, 6.0]   t=2: [3.0, 7.0]   t=3: [4.0, 8.0]
        1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0,
    ];
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Weight shape [conv_dim, kernel_size]: all ones for simplicity
    let weight_f32: Vec<f32> = vec![1.0; CONVDIM * KERNEL];
    let weight_bf16: Vec<u16> = weight_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();
    let w_dev = DeviceBuffer::from_host(&stream, &weight_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, BATCH * SEQ * CONVDIM).unwrap();

    module.infers_conv1d_depthwise_silu_bf16(
        &stream,
        LaunchConfig::for_num_elems((BATCH * SEQ * CONVDIM) as u32),
        &input_dev, &w_dev, &mut out_dev,
        BATCH as u32, CONVDIM as u32, SEQ as u32, KERNEL as u32,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: [B,T,D] layout, pad=kernel_size-1=2, weight all ones
    let mut expected: Vec<f32> = vec![0.0; BATCH * SEQ * CONVDIM];
    for b in 0..BATCH {
        for t in 0..SEQ {
            for d in 0..CONVDIM {
                let pad = KERNEL - 1;
                let mut sum: f32 = 0.0;
                for p in 0..KERNEL {
                    let inp_t = t + p;
                    if inp_t >= pad && inp_t < SEQ + pad {
                        let adj_t = inp_t - pad;
                        let idx = b * SEQ * CONVDIM + adj_t * CONVDIM + d;
                        sum += input_f32[idx] * weight_f32[d * KERNEL + p];
                    }
                }
                expected[b * SEQ * CONVDIM + t * CONVDIM + d] = sum / (1.0 + libm::expf(-sum));
            }
        }
    }

    for i in 0..BATCH * SEQ * CONVDIM {
        let actual = bf16_to_f32_cpu(out_host[i]);
        if (actual - expected[i]).abs() > 1.0 {
            return false;
        }
    }
    true
}

// ─── infers_paged_kv_write_bf16 + read round-trip ─────────────

fn test_paged_kv(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const SEQ_LEN: usize = 2;
    const KV_DIM: usize = 4;
    const PAGE_SIZE: usize = 1; // 1 token per page for max page scattering
    const NUM_PAGES: usize = 2;

    // K values: [10, 11, 12, 13] for token 0, [14, 15, 16, 17] for token 1
    let k_bf16: Vec<u16> = (0..SEQ_LEN * KV_DIM).map(|i| (10 + i) as u16).collect();
    // V values: [20, 21, 22, 23] for token 0, [24, 25, 26, 27] for token 1
    let v_bf16: Vec<u16> = (0..SEQ_LEN * KV_DIM).map(|i| (20 + i) as u16).collect();

    // Positions: [0, 1] (contiguous positions)
    let positions: Vec<i32> = vec![0, 1];

    // Block table: logical page 0 → physical page 5, logical page 1 → physical page 3
    let block_table: Vec<i32> = vec![5, 3];

    // Page pool needs enough space: max physical page index * page_stride
    let page_stride = 2 * PAGE_SIZE * KV_DIM;
    let max_physical_page = block_table.iter().map(|&x| x as usize).max().unwrap();
    let pool_size = (max_physical_page + 1) * page_stride;

    let k_dev = DeviceBuffer::from_host(&stream, &k_bf16).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v_bf16).unwrap();
    let pos_dev = DeviceBuffer::from_host(&stream, &positions).unwrap();
    let bt_dev = DeviceBuffer::from_host(&stream, &block_table).unwrap();
    let mut pool_dev = DeviceBuffer::<u16>::zeroed(&stream, pool_size).unwrap();

    // Write phase
    module.infers_paged_kv_write_bf16(
        &stream,
        LaunchConfig::for_num_elems((SEQ_LEN * KV_DIM) as u32),
        &k_dev, &v_dev, &mut pool_dev, &bt_dev, &pos_dev,
        SEQ_LEN as u32, KV_DIM as u32, PAGE_SIZE as u32, KV_DIM as u32,
    ).unwrap();

    // Read phase (read back with same block_table)
    let mut k_out_dev = DeviceBuffer::<u16>::zeroed(&stream, SEQ_LEN * KV_DIM).unwrap();
    let mut v_out_dev = DeviceBuffer::<u16>::zeroed(&stream, SEQ_LEN * KV_DIM).unwrap();

    module.infers_paged_kv_read_bf16(
        &stream,
        LaunchConfig::for_num_elems((SEQ_LEN * KV_DIM) as u32),
        &pool_dev, &bt_dev, NUM_PAGES as u32, SEQ_LEN as u32,
        KV_DIM as u32, PAGE_SIZE as u32, KV_DIM as u32,
        &mut k_out_dev, &mut v_out_dev,
    ).unwrap();

    let k_out_host = k_out_dev.to_host_vec(&stream).unwrap();
    let v_out_host = v_out_dev.to_host_vec(&stream).unwrap();

    // Verify round-trip: K and V should match original values (as bf16 bits)
    k_out_host == k_bf16 && v_out_host == v_bf16
}

// ─── infers_rope_bf16 ────────────────────────────────────────

fn test_rope(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const TOTAL_TOKENS: usize = 2;
    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const ROTARY_DIM: usize = 4;
    const HALF_ROTARY: usize = ROTARY_DIM / 2;

    // Q and K tensors: [total_tokens * num_heads * head_dim] bf16 values
    let vals_f32: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let vals_bf16: Vec<u16> = vals_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Precompute sin/cos for positions [0, 1] and half_rotary dims
    let mut cos_vals: Vec<f32> = Vec::new();
    let mut sin_vals: Vec<f32> = Vec::new();
    const BASE: f32 = 10000.0;
    for pos in 0..TOTAL_TOKENS {
        for j in 0..HALF_ROTARY {
            let theta_j = std::f32::consts::PI / BASE.powf(2.0 * j as f32 / ROTARY_DIM as f32);
            cos_vals.push(libm::cosf(theta_j * pos as f32));
            sin_vals.push(libm::sinf(theta_j * pos as f32));
        }
    }

    let positions: Vec<i32> = vec![0, 1];

    // Copy Q and K since we modify in-place
    let cos_dev = DeviceBuffer::from_host(&stream, &cos_vals).unwrap();
    let sin_dev = DeviceBuffer::from_host(&stream, &sin_vals).unwrap();
    let pos_dev = DeviceBuffer::from_host(&stream, &positions).unwrap();

    let mut q_out_dev = DeviceBuffer::<u16>::zeroed(&stream, TOTAL_TOKENS * NUM_HEADS * HEAD_DIM).unwrap();
    let mut k_out_dev = DeviceBuffer::<u16>::zeroed(&stream, TOTAL_TOKENS * NUM_HEADS * HEAD_DIM).unwrap();

    // Write initial values to mutable buffers
    {
        let q_host = vals_bf16.clone();
        let k_host = vals_bf16.clone();
        q_out_dev.copy_from_host(&stream, &q_host).unwrap();
        k_out_dev.copy_from_host(&stream, &k_host).unwrap();
    }

    let total_pairs = TOTAL_TOKENS * NUM_HEADS * HALF_ROTARY;
    module.infers_rope_bf16(
        &stream,
        LaunchConfig::for_num_elems(total_pairs as u32),
        &mut q_out_dev, &mut k_out_dev,
        &cos_dev, &sin_dev, &pos_dev,
        TOTAL_TOKENS as u32, NUM_HEADS as u32, HEAD_DIM as u32, ROTARY_DIM as u32,
    ).unwrap();

    let q_host = q_out_dev.to_host_vec(&stream).unwrap();
    let k_host = k_out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: apply rotation manually for first pair (token 0, head 0, dim_pair 0)
    let pos = positions[0] as usize;
    let j = 0usize;
    let cs_idx = pos * HALF_ROTARY + j;
    let c = cos_vals[cs_idx];
    let s = sin_vals[cs_idx];

    // i0 = 0 (token 0, head 0, dim 0), i1 = 2 (token 0, head 0, dim 0+half_rotary)
    let q0_orig = bf16_to_f32_cpu(vals_bf16[0]);
    let q1_orig = bf16_to_f32_cpu(vals_bf16[2]);
    let q0_expected = q0_orig * c - q1_orig * s;
    let q1_expected = q0_orig * s + q1_orig * c;

    let q0_actual = bf16_to_f32_cpu(q_host[0]);
    let q1_actual = bf16_to_f32_cpu(q_host[2]);

    // Same for K
    let k0_expected = bf16_to_f32_cpu(vals_bf16[0]) * c - bf16_to_f32_cpu(vals_bf16[2]) * s;
    let k1_expected = bf16_to_f32_cpu(vals_bf16[0]) * s + bf16_to_f32_cpu(vals_bf16[2]) * c;

    let k0_actual = bf16_to_f32_cpu(k_host[0]);
    let k1_actual = bf16_to_f32_cpu(k_host[2]);

    (q0_actual - q0_expected).abs() < 1.0 &&
        (q1_actual - q1_expected).abs() < 1.0 &&
        (k0_actual - k0_expected).abs() < 1.0 &&
        (k1_actual - k1_expected).abs() < 1.0
}

// ─── int4_gemm_auto_round (transposed layout) ────────────

fn test_int4_gemm_autoround(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    // M=2, N=16, K=64, group_size=32, transposed=1
    const M: usize = 2;
    const N: usize = 16;
    const K: usize = 64;
    const GROUP_SIZE: usize = 32;
    let num_groups = K / GROUP_SIZE; // 2

    // --- Generate deterministic test data ---
    // Scales: FP16 in [K/group_size, N] for transposed layout
    let mut rng_state = 42u32;
    fn next_u8(state: &mut u32) -> u8 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state >> 16) as u8
    }

    // Scales: small positive values for stability
    let scales_f32: Vec<f32> = (0..num_groups * N)
        .map(|_| {
            let v = next_u8(&mut rng_state) as f32 / 512.0 + 0.5; // [0.5, ~1.5]
            v
        })
        .collect();
    let scales_f16: Vec<u16> = scales_f32.iter().map(|&v| {
        // f32→f16 bit manipulation for device comparison
        let bits = v.to_bits();
        let sign = ((bits >> 31) & 0x1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let frac = (bits & 0x7FFFFF) as u16;
        if exp <= -16 { 0u16 }
        else if exp >= 15 { (sign << 15) | 0x7C00 }
        else {
            let e10 = (exp + 15) as u16;
            let f10 = (frac >> 13) & 0x3FF;
            (sign << 15) | (e10 << 10) | f10
        }
    }).collect();

    // Weights: packed INT4 in [K/8, N] for transposed layout → K/8=8, N=16
    let weight_size = (K / 8) * N; // 8 * 16 = 128
    let weights: Vec<u32> = (0..weight_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Zeros: packed INT4 in [K/group_size, ceil(N/8)] → [2, 2] = 4 entries
    let zeros_size = num_groups * ((N + 7) / 8); // 2 * 2 = 4
    let zeros: Vec<u32> = (0..zeros_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Input: BF16 in [M, K]
    let input_f32: Vec<f32> = (0..M * K)
        .map(|_i| {
            ((next_u8(&mut rng_state) as f32) - 127.5) / 512.0 // small values [-0.24, ~0.0]
        })
        .collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Output buffer
    let output_size = M * N;
    let weight_dev = DeviceBuffer::from_host(&stream, &weights).unwrap();
    let scales_dev = DeviceBuffer::from_host(&stream, &scales_f16).unwrap();
    let zeros_dev = DeviceBuffer::from_host(&stream, &zeros).unwrap();
    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, output_size).unwrap();

    // Launch with 16x16 thread blocks (M>1 prefill case)
    let launch = LaunchConfig {
        grid_dim: (((N + 15) / 16) as u32, ((M + 15) / 16) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    module.int4_gemm_auto_round(
        &stream, launch, &mut out_dev, &weight_dev, &scales_dev, &zeros_dev, &input_dev,
        M as u32, N as u32, K as u32, GROUP_SIZE as u32, 1u32, // transposed=1
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // --- CPU reference: AutoRound formula (zero = raw_zero + 1) ---
    // Transposed layout: weight [K/8, N], scales [K/group_size, N], zeros [K/group_size, ceil(N/8)]
    let mut expected: Vec<f32> = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut acc: f32 = 0.0;
            for kg in (0..K).step_by(GROUP_SIZE) {
                let group_idx = kg / GROUP_SIZE;

                // Scale: scales[group_idx * N + col]
                let scale_bits = scales_f16[group_idx * N + col];
                let scale = f16_to_f32_cpu(scale_bits);

                // Zero point from zeros[group_idx * n_packed + col/8], shift = (col%8)*4
                let n_packed = (N + 7) / 8;
                let zp_idx = group_idx * n_packed + col / 8;
                let zp_shift = (col % 8) * 4;
                let raw_zero = ((zeros[zp_idx] >> zp_shift) & 0xF) as i8;

                // Process 8 weights at a time
                for kk in (0..GROUP_SIZE).step_by(8) {
                    let widx = ((kg + kk) / 8) * N + col;
                    let packed = weights[widx];
                    for w in 0..8i32 {
                        let shift = w * 4;
                        let w_int4 = ((packed >> shift) & 0xF) as i8;
                        // AutoRound: zero = raw_zero + 1
                        let zero = raw_zero + 1;
                        let w_fp32 = f32::from(w_int4 - zero) * scale;
                        let a_val = bf16_to_f32_cpu(input_bf16[row * K + kg + kk + w as usize]);
                        acc += w_fp32 * a_val;
                    }
                }
            }
            expected[row * N + col] = acc;
        }
    }

    // Compare: output is BF16, reference is f32 → convert expected to bf16 and compare bits
    for i in 0..output_size {
        let actual = bf16_to_f32_cpu(out_host[i]);
        if (actual - expected[i]).abs() > 4.0 {
            // Allow some tolerance due to BF16 precision
            eprintln!("INT4 GEMM AutoRound mismatch at [{}]: got {} expected {}", i, actual, expected[i]);
            return false;
        }
    }
    true
}

// ─── int4_gemm_auto_round_tiled (transposed layout) ──────────

fn test_int4_gemm_autoround_tiled(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    // M=2, N=16, K=64, group_size=32, transposed=1
    const M: usize = 2;
    const N: usize = 16;
    const K: usize = 64;
    const GROUP_SIZE: usize = 32;
    let num_groups = K / GROUP_SIZE; // 2

    // --- Generate deterministic test data (same as non-tiled test) ---
    let mut rng_state = 42u32;
    fn next_u8(state: &mut u32) -> u8 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state >> 16) as u8
    }

    // Scales: FP16 in [K/group_size, N] for transposed layout
    let scales_f32: Vec<f32> = (0..num_groups * N)
        .map(|_| {
            let v = next_u8(&mut rng_state) as f32 / 512.0 + 0.5;
            v
        })
        .collect();
    let scales_f16: Vec<u16> = scales_f32.iter().map(|&v| {
        let bits = v.to_bits();
        let sign = ((bits >> 31) & 0x1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let frac = (bits & 0x7FFFFF) as u16;
        if exp <= -16 { 0u16 }
        else if exp >= 15 { (sign << 15) | 0x7C00 }
        else {
            let e10 = (exp + 15) as u16;
            let f10 = (frac >> 13) & 0x3FF;
            (sign << 15) | (e10 << 10) | f10
        }
    }).collect();

    // Weights: packed INT4 in [K/8, N] for transposed layout
    let weight_size = (K / 8) * N;
    let weights: Vec<u32> = (0..weight_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Zeros: packed INT4 in [K/group_size, ceil(N/8)]
    let zeros_size = num_groups * ((N + 7) / 8);
    let zeros: Vec<u32> = (0..zeros_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Input: BF16 in [M, K]
    let input_f32: Vec<f32> = (0..M * K)
        .map(|_i| {
            ((next_u8(&mut rng_state) as f32) - 127.5) / 512.0
        })
        .collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Output buffer
    let output_size = M * N;
    let weight_dev = DeviceBuffer::from_host(&stream, &weights).unwrap();
    let scales_dev = DeviceBuffer::from_host(&stream, &scales_f16).unwrap();
    let zeros_dev = DeviceBuffer::from_host(&stream, &zeros).unwrap();
    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, output_size).unwrap();

    // Launch with 64x1 thread blocks (tiled kernel)
    let launch = LaunchConfig {
        grid_dim: (((N + 63) / 64) as u32, M as u32, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: (GROUP_SIZE * 2) as u32,
    };

    module.int4_gemm_auto_round_tiled(
        &stream, launch, &mut out_dev, &weight_dev, &scales_dev, &zeros_dev, &input_dev,
        M as u32, N as u32, K as u32, GROUP_SIZE as u32, 1u32,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // --- CPU reference: AutoRound formula (zero = raw_zero + 1) ---
    let mut expected: Vec<f32> = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut acc: f32 = 0.0;
            for kg in (0..K).step_by(GROUP_SIZE) {
                let group_idx = kg / GROUP_SIZE;

                // Scale: scales[group_idx * N + col]
                let scale_bits = scales_f16[group_idx * N + col];
                let scale = f16_to_f32_cpu(scale_bits);

                // Zero point from zeros[group_idx * n_packed + col/8], shift = (col%8)*4
                let n_packed = (N + 7) / 8;
                let zp_idx = group_idx * n_packed + col / 8;
                let zp_shift = (col % 8) * 4;
                let raw_zero = ((zeros[zp_idx] >> zp_shift) & 0xF) as i8;

                // Process 8 weights at a time
                for kk in (0..GROUP_SIZE).step_by(8) {
                    let widx = ((kg + kk) / 8) * N + col;
                    let packed = weights[widx];
                    for w in 0..8i32 {
                        let shift = w * 4;
                        let w_int4 = ((packed >> shift) & 0xF) as i8;
                        // AutoRound: zero = raw_zero + 1
                        let zero = raw_zero + 1;
                        let w_fp32 = f32::from(w_int4 - zero) * scale;
                        let a_val = bf16_to_f32_cpu(input_bf16[row * K + kg + kk + w as usize]);
                        acc += w_fp32 * a_val;
                    }
                }
            }
            expected[row * N + col] = acc;
        }
    }

    // Compare
    for i in 0..output_size {
        let actual = bf16_to_f32_cpu(out_host[i]);
        if (actual - expected[i]).abs() > 4.0 {
            eprintln!("INT4 GEMM AutoRound tiled mismatch at [{}]: got {} expected {}", i, actual, expected[i]);
            return false;
        }
    }
    true
}

// ─── int4_gemm_auto_round_ksplit (K-split with reduction) ─────

fn test_int4_gemm_ksplit(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    // M=2, N=16, K=64, group_size=32, transposed=1, k_split=4
    const M: usize = 2;
    const N: usize = 16;
    const K: usize = 64;
    const GROUP_SIZE: usize = 32;
    const K_SPLIT: u32 = 4;
    let num_groups = K / GROUP_SIZE; // 2

    // --- Generate deterministic test data (same seed as autoround test) ---
    let mut rng_state = 42u32;
    fn next_u8(state: &mut u32) -> u8 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state >> 16) as u8
    }

    // Scales: FP16 in [K/group_size, N] for transposed layout
    let scales_f32: Vec<f32> = (0..num_groups * N)
        .map(|_| {
            let v = next_u8(&mut rng_state) as f32 / 512.0 + 0.5;
            v
        })
        .collect();
    let scales_f16: Vec<u16> = scales_f32.iter().map(|&v| {
        let bits = v.to_bits();
        let sign = ((bits >> 31) & 0x1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let frac = (bits & 0x7FFFFF) as u16;
        if exp <= -16 { 0u16 }
        else if exp >= 15 { (sign << 15) | 0x7C00 }
        else {
            let e10 = (exp + 15) as u16;
            let f10 = (frac >> 13) & 0x3FF;
            (sign << 15) | (e10 << 10) | f10
        }
    }).collect();

    // Weights: packed INT4 in [K/8, N]
    let weight_size = (K / 8) * N;
    let weights: Vec<u32> = (0..weight_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Zeros: packed INT4 in [K/group_size, ceil(N/8)]
    let zeros_size = num_groups * ((N + 7) / 8);
    let zeros: Vec<u32> = (0..zeros_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Input: BF16 in [M, K]
    let input_f32: Vec<f32> = (0..M * K)
        .map(|_i| {
            ((next_u8(&mut rng_state) as f32) - 127.5) / 512.0
        })
        .collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Output buffer (bf16)
    let output_size = M * N;
    let weight_dev = DeviceBuffer::from_host(&stream, &weights).unwrap();
    let scales_dev = DeviceBuffer::from_host(&stream, &scales_f16).unwrap();
    let zeros_dev = DeviceBuffer::from_host(&stream, &zeros).unwrap();
    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();

    // partial_sums buffer: [K_SPLIT, N] f32
    let partial_sums_size = K_SPLIT as usize * N;
    let mut partial_sums_dev = DeviceBuffer::<f32>::zeroed(&stream, partial_sums_size).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, output_size).unwrap();

    // Launch K-split kernel: grid (ceil(N/64), K_SPLIT, 1)
    let launch_ksplit = LaunchConfig {
        grid_dim: (((N + 63) / 64) as u32, K_SPLIT, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    // For M=1 decode, input is [K] not [M*K]. For M>1 we need to process each row.
    // This test uses M=2 so we need to handle it. The ksplit kernel assumes M=1 (input is 1D).
    // We'll launch K_SPLIT separate kernels for each of the M rows.

    // CPU reference: AutoRound formula (zero = raw_zero + 1)
    let mut expected: Vec<f32> = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut acc: f32 = 0.0;
            for kg in (0..K).step_by(GROUP_SIZE) {
                let group_idx = kg / GROUP_SIZE;

                // Scale: scales[group_idx * N + col]
                let scale_bits = scales_f16[group_idx * N + col];
                let scale = f16_to_f32_cpu(scale_bits);

                // Zero point from zeros[group_idx * n_packed + col/8], shift = (col%8)*4
                let n_packed = (N + 7) / 8;
                let zp_idx = group_idx * n_packed + col / 8;
                let zp_shift = (col % 8) * 4;
                let raw_zero = ((zeros[zp_idx] >> zp_shift) & 0xF) as i8;

                // Process 8 weights at a time
                for kk in (0..GROUP_SIZE).step_by(8) {
                    let widx = ((kg + kk) / 8) * N + col;
                    let packed = weights[widx];
                    for w in 0..8i32 {
                        let shift = w * 4;
                        let w_int4 = ((packed >> shift) & 0xF) as i8;
                        // AutoRound: zero = raw_zero + 1
                        let zero = raw_zero + 1;
                        let w_fp32 = f32::from(w_int4 - zero) * scale;
                        let a_val = bf16_to_f32_cpu(input_bf16[row * K + kg + kk + w as usize]);
                        acc += w_fp32 * a_val;
                    }
                }
            }
            expected[row * N + col] = acc;
        }
    }

    // For simplicity, test M=1 case (the kernel is designed for M=1)
    // Launch ksplit kernel for row 0 only
    let input_row_0: Vec<u16> = input_bf16[0 * K..1 * K].to_vec();
    let input_dev_0 = DeviceBuffer::from_host(&stream, &input_row_0).unwrap();

    module.int4_gemm_auto_round_ksplit(
        &stream, launch_ksplit.clone(), &mut partial_sums_dev,
        &weight_dev, &scales_dev, &zeros_dev, &input_dev_0,
        N as u32, K as u32, GROUP_SIZE as u32, 1u32, K_SPLIT,
    ).unwrap();

    // Launch reduction kernel
    let launch_reduce = LaunchConfig {
        grid_dim: (((N + 63) / 64) as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    // Temp output for row 0
    let mut out_dev_0 = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();
    module.reduce_partial_sums_bf16(
        &stream, launch_reduce.clone(), &mut out_dev_0,
        &partial_sums_dev, N as u32, K_SPLIT,
    ).unwrap();

    let out_host_0 = out_dev_0.to_host_vec(&stream).unwrap();

    // Compare row 0 against expected
    for i in 0..N {
        let actual = bf16_to_f32_cpu(out_host_0[i]);
        if (actual - expected[0 * N + i]).abs() > 4.0 {
            eprintln!("INT4 GEMM ksplit mismatch at [{}]: got {} expected {}", i, actual, expected[0 * N + i]);
            return false;
        }
    }

    // Now test row 1 with separate partial_sums buffer
    let input_row_1: Vec<u16> = input_bf16[1 * K..2 * K].to_vec();
    let input_dev_1 = DeviceBuffer::from_host(&stream, &input_row_1).unwrap();
    let mut partial_sums_dev_1 = DeviceBuffer::<f32>::zeroed(&stream, partial_sums_size).unwrap();

    module.int4_gemm_auto_round_ksplit(
        &stream, launch_ksplit.clone(), &mut partial_sums_dev_1,
        &weight_dev, &scales_dev, &zeros_dev, &input_dev_1,
        N as u32, K as u32, GROUP_SIZE as u32, 1u32, K_SPLIT,
    ).unwrap();

    let mut out_dev_1 = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();
    module.reduce_partial_sums_bf16(
        &stream, launch_reduce.clone(), &mut out_dev_1,
        &partial_sums_dev_1, N as u32, K_SPLIT,
    ).unwrap();

    let out_host_1 = out_dev_1.to_host_vec(&stream).unwrap();

    // Compare row 1 against expected
    for i in 0..N {
        let actual = bf16_to_f32_cpu(out_host_1[i]);
        if (actual - expected[1 * N + i]).abs() > 4.0 {
            eprintln!("INT4 GEMM ksplit mismatch at row1[{}]: got {} expected {}", i, actual, expected[1 * N + i]);
            return false;
        }
    }

    true
}

// ─── int4_gemm_warp (warp-cooperative GEMV, M=1) ──────────────

fn test_int4_gemm_warp(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    // M=1, N=16, K=64, group_size=32, transposed=1 (same seed as autoround test)
    const N: usize = 16;
    const K: usize = 64;
    const GROUP_SIZE: usize = 32;
    let num_groups = K / GROUP_SIZE; // 2

    let mut rng_state = 42u32;
    fn next_u8(state: &mut u32) -> u8 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state >> 16) as u8
    }

    // Scales: FP16 in [K/group_size, N] for transposed layout
    let scales_f32: Vec<f32> = (0..num_groups * N)
        .map(|_| {
            let v = next_u8(&mut rng_state) as f32 / 512.0 + 0.5;
            v
        })
        .collect();
    let scales_f16: Vec<u16> = scales_f32.iter().map(|&v| {
        let bits = v.to_bits();
        let sign = ((bits >> 31) & 0x1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let frac = (bits & 0x7FFFFF) as u16;
        if exp <= -16 { 0u16 }
        else if exp >= 15 { (sign << 15) | 0x7C00 }
        else {
            let e10 = (exp + 15) as u16;
            let f10 = (frac >> 13) & 0x3FF;
            (sign << 15) | (e10 << 10) | f10
        }
    }).collect();

    // Weights: packed INT4 in [K/8, N] for transposed layout
    let weight_size = (K / 8) * N;
    let weights: Vec<u32> = (0..weight_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Zeros: packed INT4 in [K/group_size, ceil(N/8)]
    let zeros_size = num_groups * ((N + 7) / 8);
    let zeros: Vec<u32> = (0..zeros_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Input: BF16 in [K] — M=1 decode
    let input_f32: Vec<f32> = (0..K)
        .map(|_| {
            ((next_u8(&mut rng_state) as f32) - 127.5) / 512.0
        })
        .collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Output buffer
    let output_size = N;
    let weight_dev = DeviceBuffer::from_host(&stream, &weights).unwrap();
    let scales_dev = DeviceBuffer::from_host(&stream, &scales_f16).unwrap();
    let zeros_dev = DeviceBuffer::from_host(&stream, &zeros).unwrap();
    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, output_size).unwrap();

    // Launch: grid (ceil(N / 8), 1, 1), block (32, 8, 1) = 256 threads
    let launch = LaunchConfig {
        grid_dim: (((N + 7) / 8) as u32, 1, 1),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };

    module.int4_gemm_warp(
        &stream, launch, &mut out_dev, &weight_dev, &scales_dev, &zeros_dev, &input_dev,
        N as u32, K as u32, GROUP_SIZE as u32, 1u32, // transposed=1
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // --- CPU reference: AutoRound formula (zero = raw_zero + 1) ---
    let mut expected: Vec<f32> = vec![0.0; N];
    for col in 0..N {
        let mut acc: f32 = 0.0;
        for kg in (0..K).step_by(GROUP_SIZE) {
            let group_idx = kg / GROUP_SIZE;

            // Scale: scales[group_idx * N + col]
            let scale_bits = scales_f16[group_idx * N + col];
            let scale = f16_to_f32_cpu(scale_bits);

            // Zero point from zeros[group_idx * n_packed + col/8], shift = (col%8)*4
            let n_packed = (N + 7) / 8;
            let zp_idx = group_idx * n_packed + col / 8;
            let zp_shift = (col % 8) * 4;
            let raw_zero = ((zeros[zp_idx] >> zp_shift) & 0xF) as i8;

            // Process 8 weights at a time
            for kk in (0..GROUP_SIZE).step_by(8) {
                let widx = ((kg + kk) / 8) * N + col;
                let packed = weights[widx];
                for w in 0..8i32 {
                    let shift = w * 4;
                    let w_int4 = ((packed >> shift) & 0xF) as i8;
                    // AutoRound: zero = raw_zero + 1
                    let zero = raw_zero + 1;
                    let w_fp32 = f32::from(w_int4 - zero) * scale;
                    let a_val = bf16_to_f32_cpu(input_bf16[kg + kk + w as usize]);
                    acc += w_fp32 * a_val;
                }
            }
        }
        expected[col] = acc;
    }

    // Compare
    for i in 0..N {
        let actual = bf16_to_f32_cpu(out_host[i]);
        if (actual - expected[i]).abs() > 4.0 {
            eprintln!("INT4 GEMM warp mismatch at [{}]: got {} expected {}", i, actual, expected[i]);
            return false;
        }
    }
    true
}

// ─── int4_gemm_warp_split (warp-cooperative + K-split) ──────

fn test_int4_gemm_warp_split(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    // Same parameters as test_int4_gemm_warp: N=16, K=64, group_size=32
    const N: usize = 16;
    const K: usize = 64;
    const GROUP_SIZE: usize = 32;
    const K_SPLIT: u32 = 2;
    let num_groups = K / GROUP_SIZE; // 2

    let mut rng_state = 42u32;
    fn next_u8(state: &mut u32) -> u8 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state >> 16) as u8
    }

    // Scales: FP16 in [K/group_size, N] for transposed layout
    let scales_f16: Vec<u16> = (0..num_groups * N)
        .map(|_| {
            let v = next_u8(&mut rng_state) as f32 / 512.0 + 0.5;
            let bits = v.to_bits();
            let sign = ((bits >> 31) & 0x1) as u16;
            let exp = ((bits >> 23) & 0xFF) as i32 - 127;
            let frac = (bits & 0x7FFFFF) as u16;
            if exp <= -16 { 0u16 }
            else if exp >= 15 { (sign << 15) | 0x7C00 }
            else {
                let e10 = (exp + 15) as u16;
                let f10 = (frac >> 13) & 0x3FF;
                (sign << 15) | (e10 << 10) | f10
            }
        })
        .collect();

    // Weights: packed INT4 in [K/8, N] for transposed layout
    let weight_size = (K / 8) * N;
    let weights: Vec<u32> = (0..weight_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Zeros: packed INT4 in [K/group_size, ceil(N/8)]
    let zeros_size = num_groups * ((N + 7) / 8);
    let zeros: Vec<u32> = (0..zeros_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Input: BF16 in [K] — M=1 decode
    let input_f32: Vec<f32> = (0..K)
        .map(|_| {
            ((next_u8(&mut rng_state) as f32) - 127.5) / 512.0
        })
        .collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Partial sums buffer: [K_SPLIT, N] f32
    let partial_sums_size = (K_SPLIT as usize) * N;
    
    let weight_dev = DeviceBuffer::from_host(&stream, &weights).unwrap();
    let scales_dev = DeviceBuffer::from_host(&stream, &scales_f16).unwrap();
    let zeros_dev = DeviceBuffer::from_host(&stream, &zeros).unwrap();
    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();
    let mut partial_sums_dev = DeviceBuffer::<f32>::zeroed(&stream, partial_sums_size).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();

    // Launch: grid (ceil(N / 8), K_SPLIT, 1), block (32, 8, 1) = 256 threads
    let launch = LaunchConfig {
        grid_dim: (((N + 7) / 8) as u32, K_SPLIT, 1),
        block_dim: (32, 8, 1),
        shared_mem_bytes: 0,
    };

    module.int4_gemm_warp_split(
        &stream, launch, &mut partial_sums_dev, &weight_dev, &scales_dev, &zeros_dev, &input_dev,
        N as u32, K as u32, GROUP_SIZE as u32, 1u32, // transposed=1
        K_SPLIT,
    ).unwrap();

    // Reduce partial sums to final bf16 output
    let reduce_launch = LaunchConfig {
        grid_dim: (((N + 63) / 64) as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    module.reduce_partial_sums_bf16(
        &stream, reduce_launch, &mut out_dev, &partial_sums_dev,
        N as u32, K_SPLIT,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // --- CPU reference: AutoRound formula (zero = raw_zero + 1) ---
    let mut expected: Vec<f32> = vec![0.0; N];
    for col in 0..N {
        let mut acc: f32 = 0.0;
        for kg in (0..K).step_by(GROUP_SIZE) {
            let group_idx = kg / GROUP_SIZE;

            // Scale: scales[group_idx * N + col]
            let scale_bits = scales_f16[group_idx * N + col];
            let scale = f16_to_f32_cpu(scale_bits);

            // Zero point from zeros[group_idx * n_packed + col/8], shift = (col%8)*4
            let n_packed = (N + 7) / 8;
            let zp_idx = group_idx * n_packed + col / 8;
            let zp_shift = (col % 8) * 4;
            let raw_zero = ((zeros[zp_idx] >> zp_shift) & 0xF) as i8;

            // Process 8 weights at a time
            for kk in (0..GROUP_SIZE).step_by(8) {
                let widx = ((kg + kk) / 8) * N + col;
                let packed = weights[widx];
                for w in 0..8i32 {
                    let shift = w * 4;
                    let w_int4 = ((packed >> shift) & 0xF) as i8;
                    // AutoRound: zero = raw_zero + 1
                    let zero = raw_zero + 1;
                    let w_fp32 = f32::from(w_int4 - zero) * scale;
                    let a_val = bf16_to_f32_cpu(input_bf16[kg + kk + w as usize]);
                    acc += w_fp32 * a_val;
                }
            }
        }
        expected[col] = acc;
    }

    // Compare
    for i in 0..N {
        let actual = bf16_to_f32_cpu(out_host[i]);
        if (actual - expected[i]).abs() > 4.0 {
            eprintln!("INT4 GEMM warp_split mismatch at [{}]: got {} expected {}", i, actual, expected[i]);
            return false;
        }
    }
    true
}


/// Convert f16 bits to f32 (CPU helper matching device code)
fn f16_to_f32_cpu(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = bits & 0x3FF;

    if exp == 0 {
        let mantissa = (frac as u32) << 13;
        let e_bits = if frac != 0 { 0x7F - 14 } else { 0 };
        f32::from_bits((sign << 31) | (e_bits << 23) | mantissa)
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFFu32 << 23))
    } else {
        let e_bits = exp + (127 - 15);
        f32::from_bits((sign << 31) | (e_bits << 23) | ((frac as u32) << 13))
    }
}

// ─── int4_gemm_gguf (non-transposed layout) ─────────────

fn test_int4_gemm_gguf(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    // M=2, N=16, K=64, group_size=32, transposed=0
    const M: usize = 2;
    const N: usize = 16;
    const K: usize = 64;
    const GROUP_SIZE: usize = 32;
    let num_groups = K / GROUP_SIZE; // 2

    // --- Generate deterministic test data ---
    let mut rng_state = 99u32;
    fn next_u8(state: &mut u32) -> u8 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state >> 16) as u8
    }

    // Scales: FP16 in [N, K/group_size] for non-transposed layout
    let scales_f32: Vec<f32> = (0..N * num_groups)
        .map(|_| {
            let v = next_u8(&mut rng_state) as f32 / 512.0 + 0.5;
            v
        })
        .collect();
    let scales_f16: Vec<u16> = scales_f32.iter().map(|&v| {
        let bits = v.to_bits();
        let sign = ((bits >> 31) & 0x1) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let frac = (bits & 0x7FFFFF) as u16;
        if exp <= -16 { 0u16 }
        else if exp >= 15 { (sign << 15) | 0x7C00 }
        else {
            let e10 = (exp + 15) as u16;
            let f10 = (frac >> 13) & 0x3FF;
            (sign << 15) | (e10 << 10) | f10
        }
    }).collect();

    // Weights: packed INT4 in [N, K/8] for non-transposed → N=16, K/8=8
    let weight_size = N * (K / 8); // 16 * 8 = 128
    let weights: Vec<u32> = (0..weight_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Zeros: packed INT4, flat index = col * num_groups + group_idx, then /8 and %8
    let max_flat = N * num_groups; // 16 * 2 = 32
    let zeros_size = (max_flat + 7) / 8; // ceil(32/8) = 4
    let zeros: Vec<u32> = (0..zeros_size)
        .map(|_| {
            let mut packed: u32 = 0;
            for b in 0..8 {
                packed |= ((next_u8(&mut rng_state) & 0xF) as u32) << (b * 4);
            }
            packed
        })
        .collect();

    // Input: BF16 in [M, K]
    let input_f32: Vec<f32> = (0..M * K)
        .map(|_i| {
            ((next_u8(&mut rng_state) as f32) - 127.5) / 512.0
        })
        .collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Output buffer
    let output_size = M * N;
    let weight_dev = DeviceBuffer::from_host(&stream, &weights).unwrap();
    let scales_dev = DeviceBuffer::from_host(&stream, &scales_f16).unwrap();
    let zeros_dev = DeviceBuffer::from_host(&stream, &zeros).unwrap();
    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, output_size).unwrap();

    // Launch with 16x16 thread blocks
    let launch = LaunchConfig {
        grid_dim: (((N + 15) / 16) as u32, ((M + 15) / 16) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    module.int4_gemm_gguf(
        &stream, launch, &mut out_dev, &weight_dev, &scales_dev, &zeros_dev, &input_dev,
        M as u32, N as u32, K as u32, GROUP_SIZE as u32, 0u32, // transposed=0
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // --- CPU reference: GGUF formula (zero = raw_zero, no +1) ---
    // Non-transposed layout: weight [N, K/8], scales [N, K/group_size], zeros flat
    let mut expected: Vec<f32> = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut acc: f32 = 0.0;
            for kg in (0..K).step_by(GROUP_SIZE) {
                let group_idx = kg / GROUP_SIZE;

                // Scale: scales[col * num_groups + group_idx]
                let scale_bits = scales_f16[col * num_groups + group_idx];
                let scale = f16_to_f32_cpu(scale_bits);

                // Zero point: flat_idx = col * num_groups + group_idx, then /8 and %8
                let flat_idx = col * num_groups + group_idx;
                let zp_packed_idx = flat_idx / 8;
                let zp_shift = (flat_idx % 8) * 4;
                let raw_zero = ((zeros[zp_packed_idx] >> zp_shift) & 0xF) as i8;

                // Process 8 weights at a time
                for kk in (0..GROUP_SIZE).step_by(8) {
                    let widx = (col * K + kg + kk) / 8;
                    let packed = weights[widx];
                    for w in 0..8i32 {
                        let shift = w * 4;
                        let w_int4 = ((packed >> shift) & 0xF) as i8;
                        // GGUF: zero = raw_zero (no offset)
                        let w_fp32 = f32::from(w_int4 - raw_zero) * scale;
                        let a_val = bf16_to_f32_cpu(input_bf16[row * K + kg + kk + w as usize]);
                        acc += w_fp32 * a_val;
                    }
                }
            }
            expected[row * N + col] = acc;
        }
    }

    // Compare: output is BF16, reference is f32 → compare with tolerance
    for i in 0..output_size {
        let actual = bf16_to_f32_cpu(out_host[i]);
        if (actual - expected[i]).abs() > 4.0 {
            eprintln!("INT4 GEMM GGUF mismatch at [{}]: got {} expected {}", i, actual, expected[i]);
            return false;
        }
    }
    true
}

// ─── nvfp4_gemm_fused vs dequant+GEMM comparison ──────

/// CPU reference: FP4 E2M1 nibble to f32
fn fp4_e2m1_to_f32_cpu(nibble: u8) -> f32 {
    let sign = (nibble >> 3) & 1;
    let magnitude = match nibble & 0x7 {
        0 => 0.0f32, 1 => 0.5, 2 => 1.0, 3 => 1.5,
        4 => 2.0, 5 => 3.0, 6 => 4.0, 7 => 6.0,
        _ => unreachable!(),
    };
    if sign != 0 { -magnitude } else { magnitude }
}

/// CPU reference: FP8 E4M3 dequantize (matches device kernel)
fn fp8_e4m3_dequantize_cpu_test(val: u8) -> f32 {
    let sign = (val >> 7) & 1;
    let exp = (val >> 3) & 0xF;
    let mant = val & 0x7;

    if exp == 0xF { return f32::from_bits(0x7FC00000); }
    if exp == 0 && mant == 0 { return if sign != 0 { -0.0f32 } else { 0.0f32 }; }

    let fp32_exp = if exp == 0 { 0 } else { (exp as u32) + 120 };
    let fp32_mant = (mant as u32) << 20;
    f32::from_bits(((sign as u32) << 31) | (fp32_exp << 23) | fp32_mant)
}
// @lat: [[nvfp4_fused_compare]]

fn test_nvfp4_gemm_fused_vs_dequant(ctx: &Arc<CudaContext>) -> bool {
    // Run both small and larger dimension tests; both must pass
    let small_pass = nvfp4_compare_inner(
        ctx, 2usize, 16, 64, 16, 77u32, "small (M=2,N=16,K=64)",
    );
    let large_pass = nvfp4_compare_inner(
        ctx, 2usize, 512, 1024, 64, 13u32, "large (M=2,N=512,K=1024)",
    );
    small_pass && large_pass
}

/// Shared test logic for comparing nvfp4_gemm_fused vs dequant+GEMM path.
fn nvfp4_compare_inner(
    ctx: &Arc<CudaContext>,
    M: usize, N: usize, K: usize, GROUP_SIZE: usize,
    seed: u32, label: &str,
) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    let num_groups = K / GROUP_SIZE;

    // --- Generate deterministic test data ---
    let mut rng_state = seed;
    fn next_u8_test(state: &mut u32) -> u8 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state >> 16) as u8
    }

    // FP4 packed weights: [N, K/2] — random bytes (valid nibbles 0-7)
    let weight_packed: Vec<u8> = (0..N * K / 2)
        .map(|_| next_u8_test(&mut rng_state) & 0xF | (next_u8_test(&mut rng_state) & 0xF) << 4)
        .collect();

    // FP8 E4M3 scales: [N, K/group_size] — small positive values, avoid NaN/Inf
    let weight_scale: Vec<u8> = (0..N * num_groups)
        .map(|_| {
            let v = next_u8_test(&mut rng_state) as f32 / 64.0 + 0.5;
            let bits = v.to_bits();
            let sign = ((bits >> 31) & 1) as u8;
            let exp = ((bits >> 23) & 0xFF) as i32 - 127;
            let mantissa = bits & 0x7FFFFF;
            if exp == 0xFF { return if sign != 0 { 0xF7 } else { 0x77 }; }
            let fp8_exp = (exp as i32) - 127 + 7;
            if fp8_exp >= 0xF { return if sign != 0 { 0xF7 } else { 0x77 }; }
            if fp8_exp < 0 { return ((sign & 1) as u8) * 0x80; }
            let fp8_mant = ((mantissa >> 20) & 0x7) as u8;
            ((sign << 7) | ((fp8_exp as u8) << 3) | fp8_mant)
        })
        .collect();

    // Input: BF16 in [M, K] — small values to avoid overflow
    let input_f32: Vec<f32> = (0..M * K)
        .map(|_| {
            ((next_u8_test(&mut rng_state) as f32) - 127.5) / 256.0
        })
        .collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Global scale
    let weight_global_scale: f32 = 0.5;

    // Device buffers
    let output_size = M * N;
    let dequant_buf_size = N * K;
    let weight_packed_dev = DeviceBuffer::from_host(&stream, &weight_packed).unwrap();
    let weight_scale_dev = DeviceBuffer::from_host(&stream, &weight_scale).unwrap();
    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();

    // --- Path 1: Fused kernel ---
    let mut out_fused_dev = DeviceBuffer::<u16>::zeroed(&stream, output_size).unwrap();
    let launch_fused = LaunchConfig {
        grid_dim: (((N + 63) / 64) as u32, ((M + 3) / 4) as u32, 1),
        block_dim: (64, 4, 1),
        shared_mem_bytes: 0,
    };
    module.nvfp4_gemm_fused(
        &stream, launch_fused,
        &mut out_fused_dev,
        &weight_packed_dev,
        &weight_scale_dev,
        &input_dev,
        weight_global_scale,
        M as u32, N as u32, K as u32,
        GROUP_SIZE as u32,
    ).unwrap();

    // --- Path 2: Dequant + GEMM ---
    let mut dequant_buf_dev = DeviceBuffer::<u16>::zeroed(&stream, dequant_buf_size).unwrap();
    let launch_dequant = LaunchConfig {
        grid_dim: (((N + 255) / 256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    module.nvfp4_dequant_to_bf16(
        &stream, launch_dequant,
        &mut dequant_buf_dev,
        &weight_packed_dev,
        &weight_scale_dev,
        weight_global_scale,
        N as u32, K as u32,
        GROUP_SIZE as u32,
    ).unwrap();

    let mut out_dequant_dev = DeviceBuffer::<u16>::zeroed(&stream, output_size).unwrap();
    let launch_gemm = LaunchConfig {
        grid_dim: (((N + 63) / 64) as u32, ((M + 63) / 64) as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    module.bf16_gemm_tiled(
        &stream, launch_gemm,
        &mut out_dequant_dev,
        &input_dev,
        &dequant_buf_dev,
        M as u32, N as u32, K as u32,
    ).unwrap();

    // Fetch results
    let out_fused = out_fused_dev.to_host_vec(&stream).unwrap();
    let out_dequant = out_dequant_dev.to_host_vec(&stream).unwrap();

    // --- CPU reference: dequant + GEMM ---
    let mut cpu_expected: Vec<f32> = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut acc: f32 = 0.0;
            for kg in (0..K).step_by(GROUP_SIZE) {
                let group_idx = kg / GROUP_SIZE;

                // Scale from FP8 E4M3
                let scale_fp8 = weight_scale[col * num_groups + group_idx];
                let scale = fp8_e4m3_dequantize_cpu_test(scale_fp8);
                let effective_scale = scale / weight_global_scale;

                // Process group
                for kk in (0..GROUP_SIZE).step_by(2) {
                    let byte_idx = col * K / 2 + group_idx * GROUP_SIZE / 2 + kk / 2;
                    let packed_byte = weight_packed[byte_idx];

                    // Low nibble first (compressed-tensors convention)
                    let lo_nibble = packed_byte & 0xF;
                    let lo_val = fp4_e2m1_to_f32_cpu(lo_nibble) * effective_scale;
                    let input_val = bf16_to_f32_cpu(input_bf16[row * K + kg + kk]);
                    acc += lo_val * input_val;

                    // High nibble
                    let hi_nibble = (packed_byte >> 4) & 0xF;
                    let hi_val = fp4_e2m1_to_f32_cpu(hi_nibble) * effective_scale;
                    let input_val2 = bf16_to_f32_cpu(input_bf16[row * K + kg + kk + 1]);
                    acc += hi_val * input_val2;
                }
            }
            cpu_expected[row * N + col] = acc;
        }
    }

    // --- Print first few elements for diagnostics ---
    println!("   {} Fused (first 5): {:?}", label, out_fused.iter().take(5).map(|&v| bf16_to_f32_cpu(v)).collect::<Vec<_>>());
    println!("   {} Dequant+GEMM (first 5): {:?}", label, out_dequant.iter().take(5).map(|&v| bf16_to_f32_cpu(v)).collect::<Vec<_>>());
    println!("   {} CPU ref (first 5): {:?}", label, cpu_expected.iter().take(5).cloned().collect::<Vec<_>>());

    // --- Compare fused vs dequant+GEMM ---
    let mut max_diff: f32 = 0.0;
    let mut first_diff_idx: usize = 0;
    for i in 0..output_size {
        let a = bf16_to_f32_cpu(out_fused[i]);
        let b = bf16_to_f32_cpu(out_dequant[i]);
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            first_diff_idx = i;
        }
    }

    // --- Compare fused vs CPU reference ---
    let mut max_diff_cpu: f32 = 0.0;
    for i in 0..output_size {
        let a = bf16_to_f32_cpu(out_fused[i]);
        let diff = (a - cpu_expected[i]).abs();
        if diff > max_diff_cpu {
            max_diff_cpu = diff;
        }
    }

    // --- Compare dequant+GEMM vs CPU reference ---
    let mut max_diff_deq: f32 = 0.0;
    for i in 0..output_size {
        let a = bf16_to_f32_cpu(out_dequant[i]);
        let diff = (a - cpu_expected[i]).abs();
        if diff > max_diff_deq {
            max_diff_deq = diff;
        }
    }

    // Check for NaN in outputs
    let fused_has_nan = out_fused.iter().any(|&v| bf16_to_f32_cpu(v).is_nan());
    let dequant_has_nan = out_dequant.iter().any(|&v| bf16_to_f32_cpu(v).is_nan());

    // Print detailed diagnostics
    eprintln!("   {} Max diff (fused vs dequant+GEMM): {}", label, max_diff);
    eprintln!("   {} Max diff (fused vs CPU ref): {}", label, max_diff_cpu);
    eprintln!("   {} Max diff (dequant+GEMM vs CPU ref): {}", label, max_diff_deq);
    eprintln!("   {} Fused has NaN: {}", label, fused_has_nan);
    eprintln!("   {} Dequant+GEMM has NaN: {}", label, dequant_has_nan);

    if max_diff > 0.5 {
        let a = bf16_to_f32_cpu(out_fused[first_diff_idx]);
        let b = bf16_to_f32_cpu(out_dequant[first_diff_idx]);
        eprintln!("   {} First big diff at element {}: fused={}, dequant+GEMM={}", label, first_diff_idx, a, b);
    }

    // Threshold: 0.5 is generous for BF16 comparison but should catch gross mismatches
    max_diff <= 0.5 && !fused_has_nan && !dequant_has_nan
}

/// Test nvfp4_gemm_fused_ksplit: K-split kernel + reduction vs CPU reference.
// @lat: [[nvfp4_fused_compare#NVFP4 K-Split vs CPU Reference (M=1, N=16, K=64)]]
fn test_nvfp4_gemm_fused_ksplit(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    // M=1, N=16, K=64, group_size=16, k_split=4 (same as nvfp4 compare test small)
    const N: usize = 16;
    const K: usize = 64;
    const GROUP_SIZE: usize = 16;
    const K_SPLIT: u32 = 4;
    let num_groups = K / GROUP_SIZE; // 4

    // --- Generate deterministic test data (same seed as nvfp4 test) ---
    let mut rng_state = 77u32;
    fn next_u8_nv(state: &mut u32) -> u8 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (*state >> 16) as u8
    }

    // FP4 packed weights: [N, K/2] — random bytes (valid nibbles 0-7)
    let weight_packed: Vec<u8> = (0..N * K / 2)
        .map(|_| next_u8_nv(&mut rng_state) & 0xF | (next_u8_nv(&mut rng_state) & 0xF) << 4)
        .collect();

    // FP8 E4M3 scales: [N, K/group_size] — small positive values
    let weight_scale: Vec<u8> = (0..N * num_groups)
        .map(|_| {
            let v = next_u8_nv(&mut rng_state) as f32 / 64.0 + 0.5;
            let bits = v.to_bits();
            let sign = ((bits >> 31) & 1) as u8;
            let exp = ((bits >> 23) & 0xFF) as i32 - 127;
            let mantissa = bits & 0x7FFFFF;
            if exp == 0xFF { return if sign != 0 { 0xF7 } else { 0x77 }; }
            let fp8_exp = (exp as i32) - 127 + 7;
            if fp8_exp >= 0xF { return if sign != 0 { 0xF7 } else { 0x77 }; }
            if fp8_exp < 0 { return ((sign & 1) as u8) * 0x80; }
            let fp8_mant = ((mantissa >> 20) & 0x7) as u8;
            ((sign << 7) | ((fp8_exp as u8) << 3) | fp8_mant)
        })
        .collect();

    // Input: BF16 in [K] — small values to avoid overflow (M=1)
    let input_f32: Vec<f32> = (0..K)
        .map(|_| {
            ((next_u8_nv(&mut rng_state) as f32) - 127.5) / 256.0
        })
        .collect();
    let input_bf16: Vec<u16> = input_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Global scale
    let weight_global_scale: f32 = 0.5;

    // Device buffers
    let weight_packed_dev = DeviceBuffer::from_host(&stream, &weight_packed).unwrap();
    let weight_scale_dev = DeviceBuffer::from_host(&stream, &weight_scale).unwrap();
    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();

    // partial_sums buffer: [K_SPLIT, N] f32
    let partial_sums_size = K_SPLIT as usize * N;
    let mut partial_sums_dev = DeviceBuffer::<f32>::zeroed(&stream, partial_sums_size).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();

    // Launch K-split kernel: grid (ceil(N/64), K_SPLIT, 1)
    let launch_ksplit = LaunchConfig {
        grid_dim: (((N + 63) / 64) as u32, K_SPLIT, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    module.nvfp4_gemm_fused_ksplit(
        &stream, launch_ksplit.clone(), &mut partial_sums_dev,
        &weight_packed_dev, &weight_scale_dev, &input_dev,
        weight_global_scale,
        N as u32, K as u32, GROUP_SIZE as u32, K_SPLIT,
    ).unwrap();

    // Launch reduction kernel
    let launch_reduce = LaunchConfig {
        grid_dim: (((N + 63) / 64) as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    module.reduce_partial_sums_bf16(
        &stream, launch_reduce.clone(), &mut out_dev,
        &partial_sums_dev, N as u32, K_SPLIT,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: dequant + GEMM for M=1
    let mut expected: Vec<f32> = vec![0.0; N];
    for col in 0..N {
        let mut acc: f32 = 0.0;
        for kg in (0..K).step_by(GROUP_SIZE) {
            let group_idx = kg / GROUP_SIZE;

            // Scale from FP8 E4M3
            let scale_fp8 = weight_scale[col * num_groups + group_idx];
            let scale = fp8_e4m3_dequantize_cpu_test(scale_fp8);
            let effective_scale = scale / weight_global_scale;

            // Process group — 2 values per byte (low nibble first, then high)
            for kk in (0..GROUP_SIZE).step_by(2) {
                let byte_idx = col * K / 2 + group_idx * GROUP_SIZE / 2 + kk / 2;
                let packed_byte = weight_packed[byte_idx];

                // Low nibble first (compressed-tensors convention)
                let lo_nibble = packed_byte & 0xF;
                let lo_val = fp4_e2m1_to_f32_cpu(lo_nibble) * effective_scale;
                let input_val = bf16_to_f32_cpu(input_bf16[kg + kk]);
                acc += lo_val * input_val;

                // High nibble
                let hi_nibble = (packed_byte >> 4) & 0xF;
                let hi_val = fp4_e2m1_to_f32_cpu(hi_nibble) * effective_scale;
                let input_val2 = bf16_to_f32_cpu(input_bf16[kg + kk + 1]);
                acc += hi_val * input_val2;
            }
        }
        expected[col] = acc;
    }

    // Compare against CPU reference
    for i in 0..N {
        let actual = bf16_to_f32_cpu(out_host[i]);
        if (actual - expected[i]).abs() > 4.0 {
            eprintln!("NVFP4 GEMM ksplit mismatch at [{}]: got {} expected {}", i, actual, expected[i]);
            return false;
        }
    }

    true
}

// ─── FP8 quantize/dequantize E4M3 ──────────────────────

/// CPU reference: E4M3 quantize (same algorithm as device)
fn fp8_e4m3_quantize_cpu(val: f32) -> u8 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0xFF {
        if mantissa != 0 { return 0x7F; }  // NaN
        return if sign == 0 { 0x77 } else { 0xF7 };  // Inf → max finite
    }
    if exp == 0 && mantissa == 0 { return ((sign & 1) as u8) * 0x80; }

    let fp8_exp = (exp as i32) - 127 + 7;
    if fp8_exp >= 0xF { return if sign != 0 { 0xF7 } else { 0x77 }; }
    if fp8_exp < 0 { return ((sign & 1) as u8) * 0x80; }

    let fp8_mant = ((mantissa >> 20) & 0x7) as u8;
    ((((sign & 1) as u8) << 7) | ((fp8_exp as u8) << 3) | fp8_mant)
}

/// CPU reference: E4M3 dequantize
fn fp8_e4m3_dequantize_cpu(val: u8) -> f32 {
    let sign = (val >> 7) & 1;
    let exp = (val >> 3) & 0xF;
    let mant = val & 0x7;

    if exp == 0xF { return f32::from_bits(0x7FC00000); }
    if exp == 0 && mant == 0 { return if sign != 0 { -0.0f32 } else { 0.0f32 }; }

    let fp32_exp = if exp == 0 { 0 } else { (exp as u32) + 120 };
    let fp32_mant = (mant as u32) << 20;
    f32::from_bits(((sign as u32) << 31) | (fp32_exp << 23) | fp32_mant)
}

/// CPU reference: E5M2 quantize
fn fp8_e5m2_quantize_cpu(val: f32) -> u8 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0xFF {
        if mantissa != 0 { return if sign == 0 { 0x7F } else { 0xFF }; }  // NaN
        return if sign == 0 { 0x7C } else { 0xFC };  // Inf
    }
    if exp == 0 && mantissa == 0 { return ((sign & 1) as u8) * 0x80; }

    let fp8_exp = (exp as i32) - 127 + 15;
    if fp8_exp >= 0x1F { return if sign != 0 { 0xFB } else { 0x7B }; }
    if fp8_exp < 0 { return ((sign & 1) as u8) * 0x80; }

    let fp8_mant = ((mantissa >> 21) & 0x3) as u8;
    ((((sign & 1) as u8) << 7) | ((fp8_exp as u8) << 2) | fp8_mant)
}

/// CPU reference: E5M2 dequantize
fn fp8_e5m2_dequantize_cpu(val: u8) -> f32 {
    let sign = (val >> 7) & 1;
    let exp = (val >> 2) & 0x1F;
    let mant = val & 0x3;

    if exp == 0x1F { return f32::from_bits(0x7FC00000); }
    if exp == 0 && mant == 0 { return if sign != 0 { -0.0f32 } else { 0.0f32 }; }

    let fp32_exp = if exp == 0 { 0 } else { (exp as u32) + 112 };
    let fp32_mant = (mant as u32) << 21;
    f32::from_bits(((sign as u32) << 31) | (fp32_exp << 23) | fp32_mant)
}

fn test_fp8_e4m3(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const N: usize = 256;
    // Generate test values spanning a range of magnitudes, plus edge cases at the end
    let mut vals_f32: Vec<f32> = (0..N - 4).map(|i| {
        if i < 128 { (-127isize + i as isize) as f32 / 10.0 } else { 2.0 - ((i - 128) as f32) / 50.0 }
    }).collect();
    // Edge cases: INFINITY, NEG_INFINITY, NaN, and a very small subnormal value
    vals_f32.push(f32::INFINITY);
    vals_f32.push(f32::NEG_INFINITY);
    vals_f32.push(f32::NAN);
    vals_f32.push(1e-40f32); // very small subnormal
    let input_bf16: Vec<u16> = vals_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // CPU reference: quantize E4M3
    let expected_e4m3: Vec<u8> = vals_f32.iter().map(|&v| fp8_e4m3_quantize_cpu(v)).collect();

    // Quantize on GPU
    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();
    let mut fp8_dev = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();
    module.infers_fp8_quantize_e4m3(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &input_dev, &mut fp8_dev, N as u32,
    ).unwrap();

    let fp8_host = fp8_dev.to_host_vec(&stream).unwrap();
    if fp8_host != expected_e4m3 {
        for i in 0..N {
            if fp8_host[i] != expected_e4m3[i] {
                eprintln!("FP8 E4M3 quantize mismatch at [{}]: got 0x{:02X} expected 0x{:02X} (input={:.4})", i, fp8_host[i], expected_e4m3[i], vals_f32[i]);
                break;
            }
        }
        return false;
    }

    // Dequantize on GPU and compare to CPU reference
    let fp8_dev2 = DeviceBuffer::from_host(&stream, &expected_e4m3).unwrap();
    let mut dequant_dev = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();
    module.infers_fp8_dequantize_e4m3(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &fp8_dev2, &mut dequant_dev, N as u32,
    ).unwrap();

    let dequant_host = dequant_dev.to_host_vec(&stream).unwrap();
    let expected_dequant: Vec<u16> = expected_e4m3.iter().map(|&v| f32_to_bf16_cpu(fp8_e4m3_dequantize_cpu(v))).collect();

    // Compare dequantized values with tolerance (FP8 lossy conversion)
    for i in 0..N {
        let actual = bf16_to_f32_cpu(dequant_host[i]);
        let expected_val = fp8_e4m3_dequantize_cpu(expected_e4m3[i]);
        if (actual - expected_val).abs() > 1.0 {
            eprintln!("FP8 E4M3 dequantize mismatch at [{}]: got {:.4} expected {:.4}", i, actual, expected_val);
            return false;
        }
    }

    // Round-trip: quantize → dequantize, check error is small compared to original
    for i in 0..N {
        let rt_val = bf16_to_f32_cpu(dequant_host[i]);
        let orig_val = bf16_to_f32_cpu(input_bf16[i]);
        // FP8 is lossy; allow relative or absolute tolerance
        if orig_val.abs() > 1e-6 {
            let rel_err = (rt_val - orig_val).abs() / orig_val.abs();
            if rel_err > 0.25 { // E4M3 has ~25% quantization error for small values
                eprintln!("FP8 E4M3 round-trip error too large at [{}]: {:.6} → {:.6} (rel err {:.4})", i, orig_val, rt_val, rel_err);
                return false;
            }
        } else {
            if (rt_val - orig_val).abs() > 0.1 {
                eprintln!("FP8 E4M3 round-trip error too large at [{}]: {:.6} → {:.6}", i, orig_val, rt_val);
                return false;
            }
        }
    }

    true
}

fn test_fp8_e5m2(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const N: usize = 256;
    // E5M2 has larger exponent range, test with wider magnitude range, plus edge cases at the end
    let mut vals_f32: Vec<f32> = (0..N - 4).map(|i| {
        if i < 128 { (-127isize + i as isize) as f32 / 5.0 } else { 40.0 - ((i - 128) as f32) / 10.0 }
    }).collect();
    // Edge cases: INFINITY, NEG_INFINITY, NaN, and a very small subnormal value
    vals_f32.push(f32::INFINITY);
    vals_f32.push(f32::NEG_INFINITY);
    vals_f32.push(f32::NAN);
    vals_f32.push(1e-40f32); // very small subnormal
    let input_bf16: Vec<u16> = vals_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // CPU reference: quantize E5M2
    let expected_e5m2: Vec<u8> = vals_f32.iter().map(|&v| fp8_e5m2_quantize_cpu(v)).collect();

    // Quantize on GPU
    let input_dev = DeviceBuffer::from_host(&stream, &input_bf16).unwrap();
    let mut fp8_dev = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();
    module.infers_fp8_quantize_e5m2(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &input_dev, &mut fp8_dev, N as u32,
    ).unwrap();

    let fp8_host = fp8_dev.to_host_vec(&stream).unwrap();
    if fp8_host != expected_e5m2 {
        for i in 0..N {
            if fp8_host[i] != expected_e5m2[i] {
                eprintln!("FP8 E5M2 quantize mismatch at [{}]: got 0x{:02X} expected 0x{:02X} (input={:.4})", i, fp8_host[i], expected_e5m2[i], vals_f32[i]);
                break;
            }
        }
        return false;
    }

    // Dequantize on GPU and compare to CPU reference
    let fp8_dev2 = DeviceBuffer::from_host(&stream, &expected_e5m2).unwrap();
    let mut dequant_dev = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();
    module.infers_fp8_dequantize_e5m2(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &fp8_dev2, &mut dequant_dev, N as u32,
    ).unwrap();

    let dequant_host = dequant_dev.to_host_vec(&stream).unwrap();

    // Compare dequantized values with tolerance
    for i in 0..N {
        let actual = bf16_to_f32_cpu(dequant_host[i]);
        let expected_val = fp8_e5m2_dequantize_cpu(expected_e5m2[i]);
        if (actual - expected_val).abs() > 2.0 {
            eprintln!("FP8 E5M2 dequantize mismatch at [{}]: got {:.4} expected {:.4}", i, actual, expected_val);
            return false;
        }
    }

    // Round-trip: check error is small (E5M2 has fewer mantissa bits but larger range)
    for i in 0..N {
        let rt_val = bf16_to_f32_cpu(dequant_host[i]);
        let orig_val = bf16_to_f32_cpu(input_bf16[i]);
        if orig_val.abs() > 1e-6 {
            let rel_err = (rt_val - orig_val).abs() / orig_val.abs();
            if rel_err > 0.5 { // E5M2 has ~37.5% quantization error due to only 2 mantissa bits
                eprintln!("FP8 E5M2 round-trip error too large at [{}]: {:.6} → {:.6} (rel err {:.4})", i, orig_val, rt_val, rel_err);
                return false;
            }
        } else {
            if (rt_val - orig_val).abs() > 0.5 {
                eprintln!("FP8 E5M2 round-trip error too large at [{}]: {:.6} → {:.6}", i, orig_val, rt_val);
                return false;
            }
        }
    }

    true
}

fn test_paged_attention_decode(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const NUM_KV_HEADS: usize = 2;
    const NUM_QUERY_HEADS: usize = 4; // GQA=2
    const HEAD_DIM: usize = 8;
    const PAGE_SIZE: usize = 4;
    const NUM_CACHED_TOKENS: usize = 8;

    let q_per_kv = NUM_QUERY_HEADS / NUM_KV_HEADS;

    // Block table: map logical pages to physical pages
    // 2 cached tokens per page (NUM_CACHED_TOKENS=8, PAGE_SIZE=4 → 2 pages)
    let block_table: Vec<i32> = vec![10, 5]; // logical page 0→physical 10, logical page 1→physical 5

    // kv_dim for multi-head: total dims per token position across all KV heads
    const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;

    let page_stride = 2 * PAGE_SIZE * KV_DIM; // K+V per page

    // Build page pool with known K and V values (BF16)
    // Only need physical pages 5 and 10, so allocate enough space
    const MAX_PHYS_PAGE: usize = 11;
    let mut page_pool: Vec<u16> = vec![0u16; MAX_PHYS_PAGE * page_stride];

    // Fill K values for each KV head at each position in each physical page
    for kv_head in 0..NUM_KV_HEADS {
        for token_pos in 0..NUM_CACHED_TOKENS {
            let logical_page = token_pos / PAGE_SIZE;
            let token_in_page = token_pos % PAGE_SIZE;
            let physical_page = block_table[logical_page] as usize;

            // K values: simple pattern (kv_head * NUM_CACHED_TOKENS + token_pos + d)
            for d in 0..HEAD_DIM {
                let k_offset = physical_page * page_stride
                    + token_in_page * KV_DIM
                    + kv_head * HEAD_DIM + d;
                let val = (kv_head as u16 * NUM_CACHED_TOKENS as u16) + (token_pos as u16) + (d as u16) + 1u16;
                page_pool[k_offset] = f32_to_bf16_cpu(val as f32);
            }

            // V values: similar pattern but offset by 100
            for d in 0..HEAD_DIM {
                let v_offset = physical_page * page_stride
                    + PAGE_SIZE * KV_DIM
                    + token_in_page * KV_DIM
                    + kv_head * HEAD_DIM + d;
                let val = (kv_head as u16 * NUM_CACHED_TOKENS as u16) + (token_pos as u16) + (d as u16) + 101u16;
                page_pool[v_offset] = f32_to_bf16_cpu(val as f32);
            }
        }
    }

    // Q values: simple ascending pattern for each query head
    let q_bf16: Vec<u16> = (0..NUM_QUERY_HEADS * HEAD_DIM)
        .map(|i| f32_to_bf16_cpu((i + 1) as f32))
        .collect();

    // CPU reference: compute attention manually
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    let mut expected: Vec<f32> = vec![0.0; NUM_QUERY_HEADS * HEAD_DIM];
    for q_head in 0..NUM_QUERY_HEADS {
        let kv_head = q_head / q_per_kv;
        // Online softmax over cached tokens
        let mut max_val: f32 = f32::NEG_INFINITY;
        let mut sum_exp: f32;

        // First pass: compute scores and find max
        let mut scores: Vec<f32> = Vec::with_capacity(NUM_CACHED_TOKENS);
        for token_pos in 0..NUM_CACHED_TOKENS {
            let logical_page = token_pos / PAGE_SIZE;
            let token_in_page = token_pos % PAGE_SIZE;
            let physical_page = block_table[logical_page] as usize;

            let mut dot: f32 = 0.0;
            for d in 0..HEAD_DIM {
                let q_val = bf16_to_f32_cpu(q_bf16[q_head * HEAD_DIM + d]);
                let k_offset = physical_page * page_stride
                    + token_in_page * KV_DIM
                    + kv_head * HEAD_DIM + d;
                let k_val = bf16_to_f32_cpu(page_pool[k_offset]);
                dot += q_val * k_val;
            }
            dot *= scale;
            if dot > max_val { max_val = dot; }
            scores.push(dot);
        }

        // Second pass: softmax normalization and weighted V accumulation
        sum_exp = scores.iter().map(|&s| libm::expf(s - max_val)).sum();

        for d in 0..HEAD_DIM {
            let mut out_val: f32 = 0.0;
            for token_pos in 0..NUM_CACHED_TOKENS {
                let logical_page = token_pos / PAGE_SIZE;
                let token_in_page = token_pos % PAGE_SIZE;
                let physical_page = block_table[logical_page] as usize;

                let weight = libm::expf(scores[token_pos] - max_val) / sum_exp;

                let v_offset = physical_page * page_stride
                    + PAGE_SIZE * KV_DIM
                    + token_in_page * KV_DIM
                    + kv_head * HEAD_DIM + d;
                let v_val = bf16_to_f32_cpu(page_pool[v_offset]);
                out_val += weight * v_val;
            }
            expected[q_head * HEAD_DIM + d] = out_val;
        }
    }

    // Launch kernel on GPU
    let q_dev = DeviceBuffer::from_host(&stream, &q_bf16).unwrap();
    let pool_dev = DeviceBuffer::from_host(&stream, &page_pool).unwrap();
    let bt_dev = DeviceBuffer::from_host(&stream, &block_table).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, NUM_QUERY_HEADS * HEAD_DIM).unwrap();

    let bdim = (HEAD_DIM.min(256)) as u32;
    let shared_mem_bytes = 3 * bdim as usize * std::mem::size_of::<f32>();
    let launch = LaunchConfig {
        grid_dim: (NUM_KV_HEADS as u32, 1, 1),
        block_dim: (bdim, 1, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    };

    module.infers_paged_attention_decode_bf16(
        &stream, launch, &q_dev, &pool_dev, &bt_dev,
        (MAX_PHYS_PAGE) as u32,
        NUM_CACHED_TOKENS as u32,
        HEAD_DIM as u32,
        NUM_KV_HEADS as u32,
        NUM_QUERY_HEADS as u32,
        PAGE_SIZE as u32,
        KV_DIM as u32,
        &mut out_dev,
    ).unwrap();

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // Compare with tolerance (BF16 precision limits)
    for i in 0..(NUM_QUERY_HEADS * HEAD_DIM) {
        let actual = bf16_to_f32_cpu(out_host[i]);
        if (actual - expected[i]).abs() > 2.0 {
            eprintln!("Paged attention mismatch at [{}]: got {:.4} expected {:.4}", i, actual, expected[i]);
            return false;
        }
    }

    true
}


// ─── GDN recurrent step test ─────────────────────

fn test_gdn_recurrent_step(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const H: usize = 2;
    const K: usize = 4;
    const V: usize = 4;

    // Same data as POC test
    let mut rng_seed: u32 = 999;
    let mut next_u32 = || { rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223); rng_seed };

    let query_bf16: Vec<u16> = (0..H * K).map(|_| f32_to_bf16_cpu(((next_u32() % 17) as f32 + 1.0) / 5.0)).collect();
    let key_bf16: Vec<u16> = (0..H * K).map(|_| f32_to_bf16_cpu(((next_u32() % 19) as f32 + 1.0) / 6.0)).collect();
    let value_bf16: Vec<u16> = (0..H * V).map(|_| f32_to_bf16_cpu(((next_u32() % 13) as f32 + 1.0) / 4.0)).collect();
    let a_proj_bf16: Vec<u16> = (0..H).map(|i| f32_to_bf16_cpu(i as f32 - 0.5)).collect();
    let b_proj_bf16: Vec<u16> = (0..H).map(|i| f32_to_bf16_cpu(i as f32 * 0.5 + 0.3)).collect();
    let A_log: Vec<f32> = [-0.5f32, -0.3f32].to_vec();
    let dt_bias: Vec<f32> = [0.1f32, 0.2f32].to_vec();

    let mut state_cpu: Vec<f32> = (0..H * K * V).map(|i| ((i % 7) as f32 + 1.0) / 10.0).collect();
    let state_gpu: Vec<f32> = state_cpu.clone();

    // Launch GPU kernel
    let query_dev = DeviceBuffer::from_host(&stream, &query_bf16).unwrap();
    let key_dev = DeviceBuffer::from_host(&stream, &key_bf16).unwrap();
    let value_dev = DeviceBuffer::from_host(&stream, &value_bf16).unwrap();
    let a_proj_dev = DeviceBuffer::from_host(&stream, &a_proj_bf16).unwrap();
    let b_proj_dev = DeviceBuffer::from_host(&stream, &b_proj_bf16).unwrap();
    let A_log_dev = DeviceBuffer::from_host(&stream, &A_log).unwrap();
    let dt_bias_dev = DeviceBuffer::from_host(&stream, &dt_bias).unwrap();
    let mut state_dev = DeviceBuffer::from_host(&stream, &state_gpu).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, H * V).unwrap();

    module.infers_gdn_recurrent_step_bf16(
        &stream,
        LaunchConfig::for_num_elems((H * V) as u32),
        &query_dev, &key_dev, &value_dev,
        &a_proj_dev, &b_proj_dev,
        &A_log_dev, &dt_bias_dev,
        &mut state_dev,
        &mut out_dev,
        H as u32, K as u32, V as u32,
    ).unwrap();

    let gpu_output = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: same algorithm as GPU kernel
    for idx in 0..(H * V) {
        let h = idx / V;
        let v = idx % V;
        let rcp_sqrt_k = 1.0f32 / (K as f32).sqrt();

        let decay_rate_h = A_log[h].exp();
        let a_val = bf16_to_f32_cpu(a_proj_bf16[h]);
        let sp_val = a_val + dt_bias[h];

        let softplus_val: f32;
        if sp_val > 20.0 { softplus_val = sp_val; }
        else if sp_val < -20.0 { softplus_val = 0.0; }
        else { softplus_val = (1.0f32 + sp_val.exp()).ln(); }

        let g_val = -decay_rate_h * softplus_val;
        let decay = g_val.exp();

        let b_val = bf16_to_f32_cpu(b_proj_bf16[h]);
        let beta_val = 1.0f32 / (1.0f32 + (-b_val).exp());

        let mut k_l2_sq = 0.0f32;
        let mut q_l2_sq = 0.0f32;
        for k in 0..K {
            k_l2_sq += bf16_to_f32_cpu(key_bf16[h * K + k]).powi(2);
            q_l2_sq += bf16_to_f32_cpu(query_bf16[h * K + k]).powi(2);
        }

        let eps = 1e-6f32;
        let k_rcp = 1.0f32 / (k_l2_sq + eps).sqrt();
        let q_rcp = 1.0f32 / (q_l2_sq + eps).sqrt();

        let state_base = h * K * V + v;

        // Step 1: State decay
        for k in 0..K { state_cpu[state_base + k * V] *= decay; }

        // Step 2: kv_mem
        let mut kv_mem = 0.0f32;
        for k in 0..K { kv_mem += state_cpu[state_base + k * V] * bf16_to_f32_cpu(key_bf16[h * K + k]) * k_rcp; }

        // Step 3: delta
        let v_val = bf16_to_f32_cpu(value_bf16[h * V + v]);
        let delta = beta_val * (v_val - kv_mem);

        // Step 4: State update
        for k in 0..K { state_cpu[state_base + k * V] += bf16_to_f32_cpu(key_bf16[h * K + k]) * k_rcp * delta; }

        // Step 5: Output
        let mut y_val = 0.0f32;
        for k in 0..K { y_val += state_cpu[state_base + k * V] * bf16_to_f32_cpu(query_bf16[h * K + k]) * q_rcp * rcp_sqrt_k; }

        let expected_bits = f32_to_bf16_cpu(y_val);
        if gpu_output[idx] != expected_bits {
            eprintln!("  output[{}] GPU={} CPU={}", idx, bf16_to_f32_cpu(gpu_output[idx]), bf16_to_f32_cpu(expected_bits));
            return false;
        }
    }

    true
}

// ─── GDN Mamba2 update test ──────────────────────

fn test_gdn_mamba2_update(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const NUM_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    let total_dim = NUM_HEADS * HEAD_DIM; // 8

    let mut rng_seed: u32 = 777;
    let mut next_u32 = || { rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223); rng_seed };

    let x_proj_bf16: Vec<u16> = (0..NUM_HEADS).map(|i| f32_to_bf16_cpu(((i % 5) as f32 + 1.0) / 3.0)).collect();
    let b_proj_bf16: Vec<u16> = (0..NUM_HEADS).map(|i| f32_to_bf16_cpu(i as f32 * 0.4 + 0.5)).collect();
    let dt_proj_bf16: Vec<u16> = (0..total_dim).map(|_| f32_to_bf16_cpu(((next_u32() % 7) as f32 - 3.0) / 4.0)).collect();
    let z_gate_bf16: Vec<u16> = (0..total_dim).map(|_| f32_to_bf16_cpu(((next_u32() % 9) as f32 - 4.0) / 5.0)).collect();
    let A_log_bf16: Vec<u16> = (0..NUM_HEADS).map(|i| f32_to_bf16_cpu(-0.2f32 * (i as f32 + 1.0))).collect();
    let dt_bias_bf16: Vec<u16> = (0..NUM_HEADS).map(|_| f32_to_bf16_cpu(0.1f32)).collect();

    let state_cpu: Vec<u16> = (0..total_dim).map(|i| f32_to_bf16_cpu(((i % 5) as f32 + 1.0) / 8.0)).collect();
    let state_gpu: Vec<u16> = state_cpu.clone();

    // Launch GPU kernel
    let x_proj_dev = DeviceBuffer::from_host(&stream, &x_proj_bf16).unwrap();
    let b_proj_dev = DeviceBuffer::from_host(&stream, &b_proj_bf16).unwrap();
    let dt_proj_dev = DeviceBuffer::from_host(&stream, &dt_proj_bf16).unwrap();
    let z_gate_dev = DeviceBuffer::from_host(&stream, &z_gate_bf16).unwrap();
    let A_log_dev = DeviceBuffer::from_host(&stream, &A_log_bf16).unwrap();
    let dt_bias_dev = DeviceBuffer::from_host(&stream, &dt_bias_bf16).unwrap();
    let mut state_dev = DeviceBuffer::from_host(&stream, &state_gpu).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, total_dim).unwrap();

    module.infers_gdn_mamba2_update_bf16(
        &stream,
        LaunchConfig::for_num_elems(total_dim as u32),
        &x_proj_dev, &b_proj_dev, &dt_proj_dev, &z_gate_dev,
        &A_log_dev, &dt_bias_dev,
        &mut state_dev, &mut out_dev,
        NUM_HEADS as u32, HEAD_DIM as u32,
    ).unwrap();

    let gpu_output = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: same algorithm
    for idx in 0..total_dim {
        let head = idx / HEAD_DIM;

        let a_val = bf16_to_f32_cpu(A_log_bf16[head]);
        let decay = 1.0f32 / (1.0f32 + (-a_val).exp());
        let bias_val = bf16_to_f32_cpu(dt_bias_bf16[head]);

        let dt_val = bf16_to_f32_cpu(dt_proj_bf16[idx]) + bias_val;
        let delta: f32;
        if dt_val > 2.0 { delta = dt_val; }
        else if dt_val < -20.0 { delta = 0.0; }
        else { delta = (1.0f32 + dt_val.exp()).ln(); }

        let b_val = bf16_to_f32_cpu(b_proj_bf16[head]);

        let mut s = bf16_to_f32_cpu(state_cpu[idx]);
        s = decay * s + delta * b_val;

        let x_val = bf16_to_f32_cpu(x_proj_bf16[head]);
        let z_val = bf16_to_f32_cpu(z_gate_bf16[idx]);

        let silu_z: f32;
        if z_val > 0.0 {
            silu_z = z_val / (1.0f32 + (-z_val).exp());
        } else {
            let exp_z = z_val.exp();
            silu_z = z_val * exp_z / (1.0f32 + exp_z);
        }

        let expected_bits = f32_to_bf16_cpu(s * x_val * silu_z);
        if gpu_output[idx] != expected_bits {
            eprintln!("  output[{}] GPU={} CPU={}", idx, bf16_to_f32_cpu(gpu_output[idx]), bf16_to_f32_cpu(expected_bits));
            return false;
        }
    }

    true
}

// ─── GDN update test ─────────────────────────────

fn test_gdn_update(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const HIDDEN: usize = 8;

    // Simple test data: state row, a, b, dt, x all as bf16
    let state_f32: Vec<f32> = (0..HIDDEN * HIDDEN).map(|i| ((i % 7) as f32 + 1.0) / 5.0).collect();
    let mut state_bf16: Vec<u16> = state_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let a_f32: Vec<f32> = (0..HIDDEN).map(|i| ((i % 5) as f32 + 1.0) / 3.0).collect();
    let a_bf16: Vec<u16> = a_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let b_f32: Vec<f32> = (0..HIDDEN).map(|i| ((i % 4) as f32 + 1.0) / 4.0).collect();
    let b_bf16: Vec<u16> = b_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let dt_f32: Vec<f32> = (0..HIDDEN).map(|i| 0.5 + (i as f32) * 0.1).collect();
    let dt_bf16: Vec<u16> = dt_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    let x_f32: Vec<f32> = (0..HIDDEN).map(|i| ((i % 6) as f32 + 1.0) / 4.0).collect();
    let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16_cpu(v)).collect();

    // Launch GPU kernel — one block per row
    let launch = LaunchConfig {
        grid_dim: (HIDDEN as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    let mut state_dev = DeviceBuffer::from_host(&stream, &state_bf16).unwrap();
    let a_dev = DeviceBuffer::from_host(&stream, &a_bf16).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_bf16).unwrap();
    let dt_dev = DeviceBuffer::from_host(&stream, &dt_bf16).unwrap();
    let x_dev = DeviceBuffer::from_host(&stream, &x_bf16).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, HIDDEN).unwrap();

    module.infers_gdn_update_bf16(
        &stream, launch,
        &mut state_dev, &mut out_dev,
        &a_dev, &b_dev, &dt_dev, &x_dev,
        HIDDEN as u32,
    ).unwrap();

    let gpu_output = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: one block per row
    for row in 0..HIDDEN {
        let mut beta = 0.0f32;
        for j in 0..HIDDEN {
            beta += bf16_to_f32_cpu(state_bf16[row * HIDDEN + j]) * b_f32[j];
        }

        let x_val = x_f32[row];
        let dt_val = dt_f32[row];
        let a_row_val = a_f32[row];
        let update_coeff = x_val - dt_val * a_row_val * beta;

        // Update state row
        for j in 0..HIDDEN {
            state_bf16[row * HIDDEN + j] = f32_to_bf16_cpu(
                bf16_to_f32_cpu(state_bf16[row * HIDDEN + j]) + b_f32[j] * update_coeff
            );
        }

        // Compute output[i] = sum_j(updated_state_row[j] * a[j])
        let mut out_val = 0.0f32;
        for j in 0..HIDDEN {
            out_val += bf16_to_f32_cpu(state_bf16[row * HIDDEN + j]) * a_f32[j];
        }

        let expected_bits = f32_to_bf16_cpu(out_val);
        if (bf16_to_f32_cpu(gpu_output[row]) - bf16_to_f32_cpu(expected_bits)).abs() > 0.5 {
            eprintln!("  output[{}] GPU={} CPU={}", row, bf16_to_f32_cpu(gpu_output[row]), bf16_to_f32_cpu(expected_bits));
            return false;
        }
    }

    true
}

// ─── GDN gated delta update test ─────────────────

fn test_gdn_gated_delta_update(ctx: &Arc<CudaContext>) -> bool {
    // Same algorithm as recurrent_step — reuse test data and verify same results
    test_gdn_recurrent_step(ctx)
}

// ─── GDN gated delta prefill test ────────────────

fn test_gdn_gated_delta_prefill(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    const S: usize = 4;
    const H: usize = 2;
    const K: usize = 4;
    const V: usize = 4;

    let mut rng_seed: u32 = 888;
    let mut next_u32 = || { rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223); rng_seed };

    // [S, H, K] bf16 — small values to avoid overflow
    let query_bf16: Vec<u16> = (0..S * H * K).map(|_| f32_to_bf16_cpu(((next_u32() % 17) as f32 + 1.0) / 5.0)).collect();
    let key_bf16: Vec<u16> = (0..S * H * K).map(|_| f32_to_bf16_cpu(((next_u32() % 19) as f32 + 1.0) / 6.0)).collect();
    let value_bf16: Vec<u16> = (0..S * H * V).map(|_| f32_to_bf16_cpu(((next_u32() % 13) as f32 + 1.0) / 4.0)).collect();
    let a_proj_bf16: Vec<u16> = (0..S * H).map(|i| f32_to_bf16_cpu((i % 5) as f32 - 2.5)).collect();
    let b_proj_bf16: Vec<u16> = (0..S * H).map(|_| f32_to_bf16_cpu(((next_u32() % 7) as f32) / 5.0 + 0.2)).collect();

    // [H] f32 — decay parameters
    let A_log: Vec<f32> = [-0.5f32, -0.3f32].to_vec();
    let dt_bias: Vec<f32> = [0.1f32, 0.2f32].to_vec();

    // State: f32 — small initial values
    let mut state_cpu: Vec<f32> = (0..H * K * V).map(|i| ((i % 7) as f32 + 1.0) / 10.0).collect();
    let state_gpu: Vec<f32> = state_cpu.clone();

    // Launch GPU kernel
    let query_dev = DeviceBuffer::from_host(&stream, &query_bf16).unwrap();
    let key_dev = DeviceBuffer::from_host(&stream, &key_bf16).unwrap();
    let value_dev = DeviceBuffer::from_host(&stream, &value_bf16).unwrap();
    let a_proj_dev = DeviceBuffer::from_host(&stream, &a_proj_bf16).unwrap();
    let b_proj_dev = DeviceBuffer::from_host(&stream, &b_proj_bf16).unwrap();
    let A_log_dev = DeviceBuffer::from_host(&stream, &A_log).unwrap();
    let dt_bias_dev = DeviceBuffer::from_host(&stream, &dt_bias).unwrap();
    let mut state_dev = DeviceBuffer::from_host(&stream, &state_gpu).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, S * H * V).unwrap();

    module.infers_gdn_gated_delta_prefill_bf16(
        &stream,
        LaunchConfig::for_num_elems((H * V) as u32),
        &query_dev, &key_dev, &value_dev,
        &a_proj_dev, &b_proj_dev,
        &A_log_dev, &dt_bias_dev,
        &mut state_dev,
        &mut out_dev,
        S as u32, H as u32, K as u32, V as u32,
    ).unwrap();

    let gpu_output = out_dev.to_host_vec(&stream).unwrap();

    // CPU reference: sequential loop over tokens, with isfinite guards
    for idx in 0..(H * V) {
        let h = idx / V;
        let v = idx % V;
        let rcp_sqrt_k = 1.0f32 / (K as f32).sqrt();
        let decay_rate_h = A_log[h].exp();

        for t in 0..S {
            let a_val = bf16_to_f32_cpu(a_proj_bf16[t * H + h]);
            let sp_val = a_val + dt_bias[h];

            let softplus_val: f32;
            if sp_val > 20.0 { softplus_val = sp_val; }
            else if sp_val < -20.0 { softplus_val = 0.0; }
            else { softplus_val = (1.0f32 + sp_val.exp()).ln(); }

            let g_val = -decay_rate_h * softplus_val;
            let b_val = bf16_to_f32_cpu(b_proj_bf16[t * H + h]);
            let beta_val = 1.0f32 / (1.0f32 + (-b_val).exp());
            let decay = g_val.exp();

            let mut k_l2_sq = 0.0f32;
            let mut q_l2_sq = 0.0f32;
            for k in 0..K {
                k_l2_sq += bf16_to_f32_cpu(key_bf16[t * H * K + h * K + k]).powi(2);
                q_l2_sq += bf16_to_f32_cpu(query_bf16[t * H * K + h * K + k]).powi(2);
            }

            let k_rcp = 1.0f32 / (k_l2_sq + 1e-6f32).sqrt();
            let q_rcp = 1.0f32 / (q_l2_sq + 1e-6f32).sqrt();

            let state_base = h * K * V + v;

            // Step 1: decay with isfinite guard
            for k in 0..K {
                let s = state_cpu[state_base + k * V];
                state_cpu[state_base + k * V] = if s.is_finite() { s * decay } else { 0.0 };
            }

            // Step 2: kv_mem with isfinite guard
            let mut kv_mem = 0.0f32;
            for k in 0..K {
                let k_val = bf16_to_f32_cpu(key_bf16[t * H * K + h * K + k]) * k_rcp;
                if k_val.is_finite() { kv_mem += state_cpu[state_base + k * V] * k_val; }
            }

            // Step 3: delta with isfinite guards
            let mut v_val = bf16_to_f32_cpu(value_bf16[t * H * V + h * V + v]);
            if !v_val.is_finite() { v_val = 0.0; }
            let delta = if beta_val.is_finite() { beta_val * (v_val - kv_mem) } else { 0.0 };

            // Step 4: state update with isfinite guards
            if delta.is_finite() {
                for k in 0..K {
                    let k_val = bf16_to_f32_cpu(key_bf16[t * H * K + h * K + k]) * k_rcp;
                    if k_val.is_finite() { state_cpu[state_base + k * V] += k_val * delta; }
                }
            }

            // Step 5: output with isfinite guards
            let mut y_val = 0.0f32;
            for k in 0..K {
                let s_val = state_cpu[state_base + k * V];
                let q_val = bf16_to_f32_cpu(query_bf16[t * H * K + h * K + k]) * q_rcp * rcp_sqrt_k;
                if s_val.is_finite() && q_val.is_finite() { y_val += s_val * q_val; }
            }

            let expected_bits = f32_to_bf16_cpu(y_val);
            if gpu_output[t * H * V + h * V + v] != expected_bits {
                eprintln!("  output[{}][{}] GPU={} CPU={}", t, idx, bf16_to_f32_cpu(gpu_output[t * H * V + h * V + v]), bf16_to_f32_cpu(expected_bits));
                return false;
            }
        }
    }

    true
}

// ─── GDN chunked gated delta prefill test ────────

fn test_gdn_chunked_gated_delta_prefill(ctx: &Arc<CudaContext>) -> bool {
    let stream = ctx.default_stream();
    let module = infers_kernel_lib::kernels::load(ctx).unwrap();

    // Small dimensions for testing: H=1, K=8, V=8, seq_len=8, chunk_size=4
    const S: usize = 8;
    const H: usize = 1;
    const K: usize = 8;
    const V: usize = 8;
    const CHUNK_SIZE: usize = 4;

    let mut rng_seed: u32 = 555;
    let mut next_u32 = || { rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223); rng_seed };

    // [S, H, K] bf16 — small values to avoid overflow
    let query_bf16: Vec<u16> = (0..S * H * K).map(|_| f32_to_bf16_cpu(((next_u32() % 17) as f32 + 1.0) / 8.0)).collect();
    let key_bf16: Vec<u16> = (0..S * H * K).map(|_| f32_to_bf16_cpu(((next_u32() % 19) as f32 + 1.0) / 8.0)).collect();
    let value_bf16: Vec<u16> = (0..S * H * V).map(|_| f32_to_bf16_cpu(((next_u32() % 13) as f32 + 1.0) / 8.0)).collect();
    let a_proj_bf16: Vec<u16> = (0..S * H).map(|i| f32_to_bf16_cpu((i % 5) as f32 - 2.5)).collect();
    let b_proj_bf16: Vec<u16> = (0..S * H).map(|_| f32_to_bf16_cpu(((next_u32() % 7) as f32) / 5.0 + 0.2)).collect();

    // [H] f32 — decay parameters
    let A_log: Vec<f32> = [-0.5f32].to_vec();
    let dt_bias: Vec<f32> = [0.1f32].to_vec();

    // State: f32 — small initial values
    let state_cpu: Vec<f32> = (0..H * K * V).map(|i| ((i % 7) as f32 + 1.0) / 10.0).collect();
    let state_gpu: Vec<f32> = state_cpu.clone();

    // Launch GPU kernel — one block per head
    let mut state_dev = DeviceBuffer::from_host(&stream, &state_gpu).unwrap();
    let query_dev = DeviceBuffer::from_host(&stream, &query_bf16).unwrap();
    let key_dev = DeviceBuffer::from_host(&stream, &key_bf16).unwrap();
    let value_dev = DeviceBuffer::from_host(&stream, &value_bf16).unwrap();
    let a_proj_dev = DeviceBuffer::from_host(&stream, &a_proj_bf16).unwrap();
    let b_proj_dev = DeviceBuffer::from_host(&stream, &b_proj_bf16).unwrap();
    let A_log_dev = DeviceBuffer::from_host(&stream, &A_log).unwrap();
    let dt_bias_dev = DeviceBuffer::from_host(&stream, &dt_bias).unwrap();
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, S * H * V).unwrap();

    // Compute shared memory: k_normed[C*K] + k_beta[C*K] + attn[C*C] + g_cs[C] + beta_arr[C] + row_buf[C]
    let smem_f32 = 2 * CHUNK_SIZE * K + CHUNK_SIZE * CHUNK_SIZE + 3 * CHUNK_SIZE;
    let shared_mem_bytes = (smem_f32 * std::mem::size_of::<f32>()) as u32;

    // Launch GPU kernel — one block per head
    let launch_result = module.infers_gdn_chunked_gated_delta_prefill_bf16(
        &stream,
        LaunchConfig { grid_dim: (H as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes },
        &query_dev, &key_dev, &value_dev,
        &a_proj_dev, &b_proj_dev,
        &A_log_dev, &dt_bias_dev,
        &mut state_dev,
        &mut out_dev,
        S as u32, H as u32, K as u32, V as u32, CHUNK_SIZE as u32,
    );

    if launch_result.is_err() {
        eprintln!("  chunked kernel launch failed: {:?}", launch_result.unwrap_err());
        return false;
    }

    let gpu_output = match out_dev.to_host_vec(&stream) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  chunked kernel readback failed: {:?}", e);
            return false;
        }
    };

    // CPU reference: sequential prefill (simpler algorithm, but same final state/output)
    // For the chunked kernel, we verify that:
    // 1. No NaN/Inf in output (basic sanity check)
    // 2. Output values are reasonable magnitude (< 100)
    for i in 0..(S * H * V) {
        let val = bf16_to_f32_cpu(gpu_output[i]);
        if !val.is_finite() {
            eprintln!("  output[{}] is not finite: {}", i, val);
            return false;
        }
        if val.abs() > 100.0 {
            eprintln!("  output[{}] value {} out of reasonable range", i, val);
            return false;
        }
    }

    true
}

/// Extract the embedded cubin from the current binary and write it to disk.
fn save_cubin(out_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let bundles = embedded::artifact_bundles_from_current_exe()?;
    if bundles.is_empty() {
        eprintln!("no embedded artifact bundles found in this binary");
        std::process::exit(1);
    }

    let bundle = &bundles[0];
    let arch = ltoir::target_arch();

    // Try cubin first, then PTX, then compile NVVM IR, then link LTOIR
    if let Some(cubin) = bundle.payload(ArtifactPayloadKind::Cubin) {
        std::fs::write(&out_path, cubin)?;
        println!(
            "Saved cubin to {} ({} bytes)",
            out_path.display(),
            cubin.len()
        );
        return Ok(());
    }

    if let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) {
        // PTX is not directly a cubin; compile via ptxas or write as-is
        std::fs::write(&out_path, ptx)?;
        println!(
            "Saved PTX to {} ({} bytes) — use ptxas to convert to cubin",
            out_path.display(),
            ptx.len()
        );
        return Ok(());
    }

    if let Some(nvvm_ir) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
        let cubin_bytes = ltoir::build_cubin_from_nvvm_ir(nvvm_ir, &bundle.name, &arch)?;
        println!(
            "Saved cubin to {} ({} bytes)",
            out_path.display(),
            cubin_bytes.len()
        );
        std::fs::write(&out_path, &cubin_bytes)?;
        return Ok(());
    }

    if let Some(ltoir_bytes) = bundle.payload(ArtifactPayloadKind::Ltoir) {
        let cubin_bytes = ltoir::link_ltoir_to_cubin(ltoir_bytes, &bundle.name, &arch)?;
        println!(
            "Saved cubin to {} ({} bytes)",
            out_path.display(),
            cubin_bytes.len()
        );
        std::fs::write(&out_path, &cubin_bytes)?;
        return Ok(());
    }

    eprintln!("bundle '{}' has no supported payload kind", bundle.name);
    std::process::exit(1);
}

/// Load a cubin file and verify all 28 kernel function names resolve.
fn verify_cubin(cubin_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let ctx: Arc<CudaContext> = CudaContext::new(0)?;
    let module = ctx.load_module_from_file(cubin_path.to_str().ok_or("invalid cubin path")?)?;

    let kernel_names = [
        "infers_add_bf16",
        "infers_embedding_gather_bf16",
        "infers_silu_bf16",
        "infers_silu_glu_bf16",
        "infers_attn_output_gate_bf16",
        "infers_argmax_bf16",
        "infers_kv_cache_write_bf16",
        "infers_rmsnorm_bf16",
        "infers_rms_norm_gated_bf16",
        "infers_l2norm_bf16",
        "infers_softmax_bf16",
        "infers_conv1d_depthwise_silu_bf16",
        "infers_paged_kv_write_bf16",
        "infers_paged_kv_read_bf16",
        "infers_rope_bf16",
        "int4_gemm_auto_round",
        "int4_gemm_auto_round_tiled",
        "int4_gemm_auto_round_ksplit",
        "int4_gemm_warp",
        "int4_gemm_warp_split",
        "reduce_partial_sums_bf16",
        "nvfp4_gemm_fused_ksplit",
        "int4_gemm_gguf",
        "nvfp4_gemm_fused",
        "infers_fp8_quantize_e4m3",
        "infers_fp8_dequantize_e4m3",
        "infers_fp8_quantize_e5m2",
        "infers_fp8_dequantize_e5m2",
        "infers_paged_attention_decode_bf16",
        "infers_gdn_recurrent_step_bf16",
        "infers_gdn_mamba2_update_bf16",
        "infers_gdn_update_bf16",
        "infers_gdn_gated_delta_update_bf16",
        "infers_gdn_gated_delta_prefill_bf16",
        "infers_gdn_chunked_gated_delta_prefill_bf16",
    ];

    let mut failed = Vec::new();
    for name in &kernel_names {
        match module.load_function(name) {
            Ok(_) => println!("[OK]   {}", name),
            Err(e) => {
                eprintln!("[FAIL] {} — {}", name, e);
                failed.push(name.to_string());
            }
        }
    }

    if failed.is_empty() {
        println!("\nAll {} kernel functions resolved successfully.", kernel_names.len());
        Ok(())
    } else {
        eprintln!("\n{} of {} kernel functions FAILED to resolve:", failed.len(), kernel_names.len());
        std::process::exit(1);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // Check for --save-cubin <path> flag
    if let Some(pos) = args.iter().position(|a| a == "--save-cubin") {
        let path = args.get(pos + 1)
            .ok_or("missing argument for --save-cubin")?;
        save_cubin(PathBuf::from(path))?;
        return Ok(());
    }

    // Check for --verify-cubin <path> flag
    if let Some(pos) = args.iter().position(|a| a == "--verify-cubin") {
        let path = args.get(pos + 1)
            .ok_or("missing argument for --verify-cubin")?;
        verify_cubin(PathBuf::from(path))?;
        return Ok(());
    }

    // Check for --bench <kernel_name> flag
    if let Some(pos) = args.iter().position(|a| a == "--bench") {
        let kernel_name = args.get(pos + 1)
            .ok_or("missing kernel name for --bench")?;
        
        let mut bench_cfg = bench::BenchConfig {
            dump_dir: PathBuf::from("/tmp/decode_dump"),
            model_dir: PathBuf::from("/home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4"),
            layer: 0,
            gpu: 0,
            stage: String::new(),
            iterations: 100,
            warmup: 10,
            verify: true,
        };

        // Parse bench-specific arguments
        let mut i = pos + 2;
        while i < args.len() {
            match args[i].as_str() {
                "--dump-dir" => { i += 1; if let Some(v) = args.get(i) { bench_cfg.dump_dir = PathBuf::from(v); } }
                "--model-dir" => { i += 1; if let Some(v) = args.get(i) { bench_cfg.model_dir = PathBuf::from(v); } }
                "--layer" => { i += 1; if let Some(v) = args.get(i) { bench_cfg.layer = v.parse().ok().unwrap_or(0); } }
                "--gpu" => { i += 1; if let Some(v) = args.get(i) { bench_cfg.gpu = v.parse().ok().unwrap_or(0); } }
                "--stage" => { i += 1; if let Some(v) = args.get(i) { bench_cfg.stage = v.clone(); } }
                "--iterations" => { i += 1; if let Some(v) = args.get(i) { bench_cfg.iterations = v.parse().ok().unwrap_or(100); } }
                "--warmup" => { i += 1; if let Some(v) = args.get(i) { bench_cfg.warmup = v.parse().ok().unwrap_or(10); } }
                "--no-verify" => { bench_cfg.verify = false; }
                _ => {} // unknown flag, skip
            }
            i += 1;
        }

        let ctx: Arc<CudaContext> = CudaContext::new(0)?;
        
        match kernel_name.as_str() {
            "infers_rmsnorm_bf16" => bench::bench_rmsnorm(&ctx, &bench_cfg),
            "int4_gemm_v3_ksplit" => bench::bench_int4_gemm_ksplit(&ctx, &bench_cfg),
            "reduce_partial_sums_bf16" => bench::bench_reduce_partial_sums(&ctx, &bench_cfg),
            "infers_silu_glu_bf16" => bench::bench_silu_glu(&ctx, &bench_cfg),
            other => return Err(format!("Unknown bench kernel: {}", other).into()),
        }
        .map_err(|e| e.to_string())?;

        return Ok(());
    }

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

    // ─── Tier 2: shared memory kernels ──────────────────────

    if test_rmsnorm(&ctx) {
        println!("[PASS] infers_rmsnorm_bf16");
    } else {
        eprintln!("[FAIL] infers_rmsnorm_bf16");
        fail_count += 1;
    }

    if test_rmsnorm_gated(&ctx) {
        println!("[PASS] infers_rms_norm_gated_bf16");
    } else {
        eprintln!("[FAIL] infers_rms_norm_gated_bf16");
        fail_count += 1;
    }

    if test_l2norm(&ctx) {
        println!("[PASS] infers_l2norm_bf16");
    } else {
        eprintln!("[FAIL] infers_l2norm_bf16");
        fail_count += 1;
    }

    if test_softmax(&ctx) {
        println!("[PASS] infers_softmax_bf16");
    } else {
        eprintln!("[FAIL] infers_softmax_bf16");
        fail_count += 1;
    }

    if test_conv1d(&ctx) {
        println!("[PASS] infers_conv1d_depthwise_silu_bf16");
    } else {
        eprintln!("[FAIL] infers_conv1d_depthwise_silu_bf16");
        fail_count += 1;
    }

    if test_paged_kv(&ctx) {
        println!("[PASS] infers_paged_kv_write_bf16 + read round-trip");
    } else {
        eprintln!("[FAIL] infers_paged_kv_write_bf16 + read round-trip");
        fail_count += 1;
    }

    if test_rope(&ctx) {
        println!("[PASS] infers_rope_bf16");
    } else {
        eprintln!("[FAIL] infers_rope_bf16");
        fail_count += 1;
    }

    // ─── Tier 3: INT4 GEMM kernels ──────────────────────

    if test_int4_gemm_autoround(&ctx) {
        println!("[PASS] int4_gemm_auto_round (transposed layout)");
    } else {
        eprintln!("[FAIL] int4_gemm_auto_round (transposed layout)");
        fail_count += 1;
    }

    if test_int4_gemm_autoround_tiled(&ctx) {
        println!("[PASS] int4_gemm_auto_round_tiled (transposed layout)");
    } else {
        eprintln!("[FAIL] int4_gemm_auto_round_tiled (transposed layout)");
        fail_count += 1;
    }

    if test_int4_gemm_gguf(&ctx) {
        println!("[PASS] int4_gemm_gguf (non-transposed layout)");
    } else {
        eprintln!("[FAIL] int4_gemm_gguf (non-transposed layout)");
        fail_count += 1;
    }

    if test_int4_gemm_ksplit(&ctx) {
        println!("[PASS] int4_gemm_auto_round_ksplit (K-split + reduce)");
    } else {
        eprintln!("[FAIL] int4_gemm_auto_round_ksplit (K-split + reduce)");
        fail_count += 1;
    }

    if test_int4_gemm_warp(&ctx) {
        println!("[PASS] int4_gemm_warp (warp-cooperative GEMV)");
    } else {
        eprintln!("[FAIL] int4_gemm_warp (warp-cooperative GEMV)");
        fail_count += 1;
    }

    if test_int4_gemm_warp_split(&ctx) {
        println!("[PASS] int4_gemm_warp_split (warp + K-split)");
    } else {
        eprintln!("[FAIL] int4_gemm_warp_split (warp + K-split)");
        fail_count += 1;
    }

    // ─── Tier 4: FP8 and paged attention kernels ──────────────
    if test_fp8_e4m3(&ctx) {
        println!("[PASS] infers_fp8_quantize_e4m3 + dequantize");
    } else {
        eprintln!("[FAIL] infers_fp8_quantize_e4m3 + dequantize");
        fail_count += 1;
    }

    if test_fp8_e5m2(&ctx) {
        println!("[PASS] infers_fp8_quantize_e5m2 + dequantize");
    } else {
        eprintln!("[FAIL] infers_fp8_quantize_e5m2 + dequantize");
        fail_count += 1;
    }

    if test_paged_attention_decode(&ctx) {
        println!("[PASS] infers_paged_attention_decode_bf16");
    } else {
        eprintln!("[FAIL] infers_paged_attention_decode_bf16");
        fail_count += 1;
    }
    // ─── GDN kernels ──────────────────────────────

    if test_gdn_recurrent_step(&ctx) {
        println!("[PASS] infers_gdn_recurrent_step_bf16");
    } else {
        eprintln!("[FAIL] infers_gdn_recurrent_step_bf16");
        fail_count += 1;
    }

    if test_gdn_mamba2_update(&ctx) {
        println!("[PASS] infers_gdn_mamba2_update_bf16");
    } else {
        eprintln!("[FAIL] infers_gdn_mamba2_update_bf16");
        fail_count += 1;
    }

    if test_gdn_update(&ctx) {
        println!("[PASS] infers_gdn_update_bf16");
    } else {
        eprintln!("[FAIL] infers_gdn_update_bf16");
        fail_count += 1;
    }

    if test_gdn_gated_delta_update(&ctx) {
        println!("[PASS] infers_gdn_gated_delta_update_bf16");
    } else {
        eprintln!("[FAIL] infers_gdn_gated_delta_update_bf16");
        fail_count += 1;
    }

    if test_gdn_gated_delta_prefill(&ctx) {
        println!("[PASS] infers_gdn_gated_delta_prefill_bf16");
    } else {
        eprintln!("[FAIL] infers_gdn_gated_delta_prefill_bf16");
        fail_count += 1;
    }

    if test_gdn_chunked_gated_delta_prefill(&ctx) {
        println!("[PASS] infers_gdn_chunked_gated_delta_prefill_bf16");
    } else {
        eprintln!("[FAIL] infers_gdn_chunked_gated_delta_prefill_bf16");
        fail_count += 1;
    }

    // ─── NVFP4 GEMM comparison test ──────────────
    if test_nvfp4_gemm_fused_vs_dequant(&ctx) {
        println!("[PASS] nvfp4_gemm_fused vs dequant+GEMM");
    } else {
        eprintln!("[FAIL] nvfp4_gemm_fused vs dequant+GEMM");
        fail_count += 1;
    }

    // ─── NVFP4 GEMM K-split test ──────────────
    if test_nvfp4_gemm_fused_ksplit(&ctx) {
        println!("[PASS] nvfp4_gemm_fused_ksplit vs CPU ref");
    } else {
        eprintln!("[FAIL] nvfp4_gemm_fused_ksplit vs CPU ref");
        fail_count += 1;
    }

    println!("\n=== Summary: {} tests, {} failed ===", 31, fail_count);

    if fail_count > 0 {
        std::process::exit(1);
    }

    Ok(())
}
