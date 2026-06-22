//! Test binary for infers-cuda-oxide-kernels.
//!
//! Allocates test data, launches each kernel, and verifies the result
//! against a CPU reference.


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

// ─── main ────────────────────────────────────────────────

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

    println!("\n=== Summary: {} tests, {} failed ===", 15, fail_count);

    if fail_count > 0 {
        std::process::exit(1);
    }

    Ok(())
}
