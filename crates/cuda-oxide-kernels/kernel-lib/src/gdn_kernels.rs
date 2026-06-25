//! GDN (Gated DeltaNet) kernels — recurrent step, Mamba2 update, gated delta, chunked prefill.

use cuda_device::{cuda_module, kernel, launch_bounds, thread, DisjointSlice, DynamicSharedArray};
use super::shared::*;

#[cuda_module]
pub mod gdn {
    use super::*;

    /// GDN recurrent step: single-token decode kernel.
    /// 2D grid: blockIdx.y = head, blockIdx.x tiles over v_dim.
    /// All threads in a block share the same head → key/query cached in shared memory.
    #[kernel]
    #[launch_bounds(128)]
    pub fn infers_gdn_recurrent_step_bf16(
        query: &[u16],           // [H, K] bf16
        key: &[u16],             // [H, K] bf16
        value: &[u16],           // [H, V] bf16
        a_proj: &[u16],          // [H] bf16
        b_proj: &[u16],          // [H] bf16
        a_log: &[f32],           // [H] f32
        dt_bias: &[f32],         // [H] f32
        state: &mut [f32],       // [H, K, V] f32 — read + write
        mut output: DisjointSlice<u16>, // [H, V] bf16
        num_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let h = thread::blockIdx_y() as usize;
        let v = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let K = head_k_dim as usize;
        let V = head_v_dim as usize;
        let H = num_heads as usize;
        let rcp_sqrt_k = 1.0f32 / (K as f32).sqrt();

        if h >= H || v >= V { return; }

        // Shared memory: key[K] + query[K]
        let smem = DynamicSharedArray::<f32>::get();
        let smem_key = smem;
        let smem_query = unsafe { smem.add(K) };

        // Cooperative load: all threads in block load key and query
        let block_size = thread::blockDim_x() as usize;
        let tid = thread::threadIdx_x() as usize;

        for i in (tid..K).step_by(block_size) {
            unsafe {
                *smem_key.add(i) = f32::from_bits((key[h * K + i] as u32) << 16);
                *smem_query.add(i) = f32::from_bits((query[h * K + i] as u32) << 16);
            }
        }
        cuda_device::sync_threads();

        // L2-normalize key and query from shared memory
        let mut k_l2_sq = 0.0f32;
        let mut q_l2_sq = 0.0f32;
        for k in 0..K {
            let kv = unsafe { *smem_key.add(k) };
            let qv = unsafe { *smem_query.add(k) };
            k_l2_sq += kv * kv;
            q_l2_sq += qv * qv;
        }

        let k_rcp = 1.0f32 / (k_l2_sq + 1e-6f32).sqrt();
        let q_rcp = 1.0f32 / (q_l2_sq + 1e-6f32).sqrt() * rcp_sqrt_k;

        // Compute g[h] and beta[h]
        let decay_rate_h = fast_expf(a_log[h]);
        let a_val = f32::from_bits((a_proj[h] as u32) << 16);
        let sp_val = a_val + dt_bias[h];

        let softplus_val: f32;
        if sp_val > 20.0f32 {
            softplus_val = sp_val;
        } else if sp_val < -20.0f32 {
            softplus_val = 0.0f32;
        } else {
            softplus_val = libm::logf(1.0f32 + fast_expf(sp_val));
        }

        let g_val = -decay_rate_h * softplus_val;
        let decay = fast_expf(g_val);

        let b_val = f32::from_bits((b_proj[h] as u32) << 16);
        let beta_val = 1.0f32 / (1.0f32 + fast_expf(-b_val));

        let state_base = h * K * V + v;

        // Step 1: State decay (no key/query needed)
        for k in 0..K {
            state[state_base + k * V] *= decay;
        }

        // Step 2: kv_mem = sum_k S[h][k][v] * key_normed[h][k] (from smem)
        let mut kv_mem = 0.0f32;
        for k in 0..K {
            let s_val = state[state_base + k * V];
            let k_val = unsafe { *smem_key.add(k) } * k_rcp;
            kv_mem += s_val * k_val;
        }

        // Step 3: delta = beta * (value - kv_mem)
        let v_val = f32::from_bits((value[h * V + v] as u32) << 16);
        let delta = beta_val * (v_val - kv_mem);

        // Step 4: State update (from smem)
        for k in 0..K {
            let k_val = unsafe { *smem_key.add(k) } * k_rcp;
            state[state_base + k * V] += k_val * delta;
        }

        // Step 5: Output (from smem)
        let mut y_val = 0.0f32;
        for k in 0..K {
            let s_val = state[state_base + k * V];
            let q_val = unsafe { *smem_query.add(k) } * q_rcp;
            y_val += s_val * q_val;
        }

        unsafe { *output.get_unchecked_mut(h * V + v) = f32_to_bf16(y_val); }
    }

    /// GDN Mamba2 SSM single-token update kernel.
    /// One thread per total_dim element. No shared memory.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_gdn_mamba2_update_bf16(
        x_proj: &[u16],           // [num_heads] bf16
        b_proj: &[u16],           // [num_heads] bf16
        dt_proj: &[u16],          // [total_dim] bf16
        z_gate: &[u16],           // [total_dim] bf16
        a_log: &[u16],            // [num_heads] bf16
        dt_bias: &[u16],          // [num_heads] bf16
        state: &mut [u16],       // [total_dim] bf16 — read + write
        mut output: DisjointSlice<u16>, // [total_dim] bf16
        num_heads: u32,
        head_dim: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;
        let total_dim = (num_heads * head_dim) as usize;
        let idx = thread::index_1d();
        let tid = idx.get() as usize;
        if tid >= total_dim { return; }

        let head = tid / head_dim as usize;

        // Pre-compute per-head constants
        let a_val = f32::from_bits((a_log[head] as u32) << 16);
        let decay = 1.0f32 / (1.0f32 + fast_expf(-a_val));
        let bias_val = f32::from_bits((dt_bias[head] as u32) << 16);

        // delta = softplus(dt_proj + dt_bias)
        let dt_val = f32::from_bits((dt_proj[tid] as u32) << 16) + bias_val;
        let delta: f32;
        if dt_val > 2.0f32 {
            delta = dt_val;
        } else if dt_val < -20.0f32 {
            delta = 0.0f32;
        } else {
            delta = libm::logf(1.0f32 + fast_expf(dt_val));
        }

        // b contribution
        let b_val = f32::from_bits((b_proj[head] as u32) << 16);

        // State update: s = decay * s + delta * b
        let mut s = f32::from_bits((state[tid] as u32) << 16);
        s = decay * s + delta * b_val;

        // Output: state * x_proj * silu(z)
        let x_val = f32::from_bits((x_proj[head] as u32) << 16);
        let z_val = f32::from_bits((z_gate[tid] as u32) << 16);

        // SiLU: numerically stable formulation
        let silu_z: f32;
        if z_val > 0.0f32 {
            silu_z = z_val / (1.0f32 + fast_expf(-z_val));
        } else {
            let exp_z = fast_expf(z_val);
            silu_z = z_val * exp_z / (1.0f32 + exp_z);
        }

        unsafe { *output.get_unchecked_mut(tid) = f32_to_bf16(s * x_val * silu_z); }

        // Store updated state back
        state[tid] = f32_to_bf16(s);
    }

    /// GDN single-token update kernel — one block per state row.
    /// Uses dynamic shared memory for reductions.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_gdn_update_bf16(
        state: &mut [u16],         // [hidden_size, hidden_size] bf16 — read + write
        mut output: DisjointSlice<u16>, // [hidden_size] bf16
        a: &[u16],                // [hidden_size] bf16
        b: &[u16],                // [hidden_size] bf16
        dt: &[u16],               // [hidden_size] bf16
        x: &[u16],                // [hidden_size] bf16
        hidden_size: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let total_threads = thread::blockDim_x() as usize;
        let H = hidden_size as usize;

        // Dynamic shared memory for reductions
        let smem = DynamicSharedArray::<f32>::get();

        // Load per-row scalars in f32 for precision
        let x_val = f32::from_bits((x[row] as u32) << 16);
        let dt_val = f32::from_bits((dt[row] as u32) << 16);
        let a_row_val = f32::from_bits((a[row] as u32) << 16);

        // Phase 1: Compute beta = sum_j(state[row*H + j] * b[j])
        let mut beta = 0.0f32;
        for j in (tid..H).step_by(total_threads) {
            let s_val = f32::from_bits((state[row * H + j] as u32) << 16);
            let b_val = f32::from_bits((b[j] as u32) << 16);
            beta += s_val * b_val;
        }

        unsafe { *smem.add(tid) = beta; }
        cuda_device::sync_threads();

        let mut stride = total_threads / 2;
        while stride > 0 {
            cuda_device::sync_threads();
            if tid < stride && tid + stride < total_threads {
                unsafe { *smem.add(tid) += *smem.add(tid + stride); }
            }
            stride >>= 1;
        }
        cuda_device::sync_threads();

        let beta_all = unsafe { *smem.add(0) };

        // Phase 2: Update state row — state[i][j] += b[j] * (x[i] - dt[i]*a[i]*beta_i)
        let update_coeff = x_val - dt_val * a_row_val * beta_all;
        for j in (tid..H).step_by(total_threads) {
            let s_val = f32::from_bits((state[row * H + j] as u32) << 16);
            let b_val = f32::from_bits((b[j] as u32) << 16);
            unsafe { *state.get_unchecked_mut(row * H + j) = f32_to_bf16(s_val + b_val * update_coeff); }
        }
        cuda_device::sync_threads();

        // Phase 3: Compute output[i] = sum_j(updated_state_row[j] * a[j])
        let mut out_val = 0.0f32;
        for j in (tid..H).step_by(total_threads) {
            let s_val = f32::from_bits((state[row * H + j] as u32) << 16);
            let a_val = f32::from_bits((a[j] as u32) << 16);
            out_val += s_val * a_val;
        }

        unsafe { *smem.add(tid) = out_val; }
        cuda_device::sync_threads();

        stride = total_threads / 2;
        while stride > 0 {
            cuda_device::sync_threads();
            if tid < stride && tid + stride < total_threads {
                unsafe { *smem.add(tid) += *smem.add(tid + stride); }
            }
            stride >>= 1;
        }
        cuda_device::sync_threads();

        if tid == 0 {
            unsafe { *output.get_unchecked_mut(row) = f32_to_bf16(smem.add(0).read()); }
        }
    }

    /// GDN gated delta update: single-token decode (same algorithm as recurrent_step).
    /// One thread per (head, v_dim) element. No shared memory.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_gdn_gated_delta_update_bf16(
        query: &[u16],           // [H, K] bf16
        key: &[u16],             // [H, K] bf16
        value: &[u16],           // [H, V] bf16
        a_proj: &[u16],          // [H] bf16
        b_proj: &[u16],          // [H] bf16
        a_log: &[f32],           // [H] f32
        dt_bias: &[f32],         // [H] f32
        state: &mut [f32],       // [H, K, V] f32 — read + write
        mut output: DisjointSlice<u16>, // [H, V] bf16
        num_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;
        let total = (num_heads * head_v_dim) as usize;
        let idx = thread::index_1d();
        let tid = idx.get() as usize;
        if tid >= total { return; }

        let h = tid / head_v_dim as usize;
        let v = tid % head_v_dim as usize;
        let K = head_k_dim as usize;
        let V = head_v_dim as usize;
        let rcp_sqrt_k = 1.0f32 / (K as f32).sqrt();

        // Compute g[h] and beta[h]
        let decay_rate_h = fast_expf(a_log[h]);
        let a_val = f32::from_bits((a_proj[h] as u32) << 16);
        let sp_val = a_val + dt_bias[h];

        let softplus_val: f32;
        if sp_val > 20.0f32 {
            softplus_val = sp_val;
        } else if sp_val < -20.0f32 {
            softplus_val = 0.0f32;
        } else {
            softplus_val = libm::logf(1.0f32 + fast_expf(sp_val));
        }

        let g_val = -decay_rate_h * softplus_val;
        let b_val = f32::from_bits((b_proj[h] as u32) << 16);
        let beta_val = 1.0f32 / (1.0f32 + fast_expf(-b_val));
        let decay = fast_expf(g_val);

        // L2-normalize key and query
        let mut k_l2_sq = 0.0f32;
        let mut q_l2_sq = 0.0f32;
        for k in 0..K {
            let kv = f32::from_bits((key[h * K + k] as u32) << 16);
            let qv = f32::from_bits((query[h * K + k] as u32) << 16);
            k_l2_sq += kv * kv;
            q_l2_sq += qv * qv;
        }
        let k_rcp = 1.0f32 / (k_l2_sq + 1e-6f32).sqrt();
        let q_rcp = 1.0f32 / (q_l2_sq + 1e-6f32).sqrt();

        let state_base = h * K * V + v;

        // Step 1: S *= exp(g)
        for k in 0..K {
            state[state_base + k * V] *= decay;
        }

        // Step 2: kv_mem = sum_k S[k][v] * key[h][k] (key L2-normalized)
        let mut kv_mem = 0.0f32;
        for k in 0..K {
            let s_val = state[state_base + k * V];
            let k_val = f32::from_bits((key[h * K + k] as u32) << 16) * k_rcp;
            kv_mem += s_val * k_val;
        }

        // Step 3: delta = beta * (value - kv_mem)
        let v_val = f32::from_bits((value[h * V + v] as u32) << 16);
        let delta = beta_val * (v_val - kv_mem);

        // Step 4: State update (key L2-normalized)
        for k in 0..K {
            let k_val = f32::from_bits((key[h * K + k] as u32) << 16) * k_rcp;
            state[state_base + k * V] += k_val * delta;
        }

        // Step 5: y[v] = sum_k S[k][v] * (query[h][k] * q_l2 * rcp_sqrt_k)
        let mut y_val = 0.0f32;
        for k in 0..K {
            let s_val = state[state_base + k * V];
            let q_val = f32::from_bits((query[h * K + k] as u32) << 16) * q_rcp * rcp_sqrt_k;
            y_val += s_val * q_val;
        }

        unsafe { *output.get_unchecked_mut(h * V + v) = f32_to_bf16(y_val); }
    }

    /// GDN gated delta prefill: sequential token loop with isfinite guards.
    /// One thread per (head, v_dim) element. No shared memory.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_gdn_gated_delta_prefill_bf16(
        query: &[u16],           // [S, H, K] bf16
        key: &[u16],             // [S, H, K] bf16
        value: &[u16],           // [S, H, V] bf16
        a_proj: &[u16],          // [S, H] bf16
        b_proj: &[u16],          // [S, H] bf16
        a_log: &[f32],           // [H] f32
        dt_bias: &[f32],         // [H] f32
        state: &mut [f32],       // [H, K, V] f32 — read + write
        mut output: DisjointSlice<u16>, // [S, H, V] bf16
        seq_len: u32,
        num_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;
        let total = (num_heads * head_v_dim) as usize;
        let idx = thread::index_1d();
        let tid = idx.get() as usize;
        if tid >= total { return; }

        let h = tid / head_v_dim as usize;
        let v = tid % head_v_dim as usize;
        let K = head_k_dim as usize;
        let V = head_v_dim as usize;
        let S = seq_len as usize;
        let H = num_heads as usize;
        let rcp_sqrt_k = 1.0f32 / (K as f32).sqrt();

        let decay_rate_h = fast_expf(a_log[h]);

        for t in 0..S {
            // Compute g[t][h] and beta[t][h]
            let a_val = f32::from_bits((a_proj[t * H + h] as u32) << 16);
            let sp_val = a_val + dt_bias[h];

            let softplus_val: f32;
            if sp_val > 20.0f32 {
                softplus_val = sp_val;
            } else if sp_val < -20.0f32 {
                softplus_val = 0.0f32;
            } else {
                softplus_val = libm::logf(1.0f32 + fast_expf(sp_val));
            }

            let g_val = -decay_rate_h * softplus_val;
            let b_val = f32::from_bits((b_proj[t * H + h] as u32) << 16);
            let beta_val = 1.0f32 / (1.0f32 + fast_expf(-b_val));
            let decay = fast_expf(g_val);

            // L2-normalize query and key
            let mut k_l2_sq = 0.0f32;
            let mut q_l2_sq = 0.0f32;
            for k in 0..K {
                let kv = f32::from_bits((key[t * H * K + h * K + k] as u32) << 16);
                let qv = f32::from_bits((query[t * H * K + h * K + k] as u32) << 16);
                k_l2_sq += kv * kv;
                q_l2_sq += qv * qv;
            }
            let k_rcp = 1.0f32 / (k_l2_sq + 1e-6f32).sqrt();
            let q_rcp = 1.0f32 / (q_l2_sq + 1e-6f32).sqrt() * rcp_sqrt_k;

            let state_base = h * K * V + v;

            // Step 1: S *= exp(g) with isfinite guard
            for k in 0..K {
                let s = state[state_base + k * V];
                state[state_base + k * V] = if s.is_finite() { s * decay } else { 0.0f32 };
            }

            // Step 2: kv_mem with isfinite guard on k_val
            let mut kv_mem = 0.0f32;
            for k in 0..K {
                let s_val = state[state_base + k * V];
                let k_val = f32::from_bits((key[t * H * K + h * K + k] as u32) << 16) * k_rcp;
                if k_val.is_finite() { kv_mem += s_val * k_val; }
            }

            // Step 3: delta with isfinite guards
            let mut v_val = f32::from_bits((value[t * H * V + h * V + v] as u32) << 16);
            if !v_val.is_finite() { v_val = 0.0f32; }
            let delta = if beta_val.is_finite() { beta_val * (v_val - kv_mem) } else { 0.0f32 };

            // Step 4: State update with isfinite guard on delta and k_val
            if delta.is_finite() {
                for k in 0..K {
                    let k_val = f32::from_bits((key[t * H * K + h * K + k] as u32) << 16) * k_rcp;
                    if k_val.is_finite() {
                        state[state_base + k * V] += k_val * delta;
                    }
                }
            }

            // Step 5: Output with isfinite guards
            let mut y_val = 0.0f32;
            for k in 0..K {
                let s_val = state[state_base + k * V];
                let q_val = f32::from_bits((query[t * H * K + h * K + k] as u32) << 16) * q_rcp;
                if s_val.is_finite() && q_val.is_finite() { y_val += s_val * q_val; }
            }

            unsafe {
                *output.get_unchecked_mut(t * H * V + h * V + v) = f32_to_bf16(y_val);
            }
        }
    }

    /// GDN chunked gated delta prefill: parallel attention with WY representation.
    /// One block per head, 256 threads. ~80KB dynamic shared memory.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_gdn_chunked_gated_delta_prefill_bf16(
        query: &[u16],           // [S, H, K] bf16
        key: &[u16],             // [S, H, K] bf16
        value: &[u16],           // [S, H, V] bf16
        a_proj: &[u16],          // [S, H] bf16
        b_proj: &[u16],          // [S, H] bf16
        a_log: &[f32],           // [H] f32
        dt_bias: &[f32],         // [H] f32
        state: &mut [f32],       // [H, K, V] f32 — read + write
        mut output: DisjointSlice<u16>, // [S, H, V] bf16
        seq_len: u32,
        num_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        chunk_size: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let h = thread::blockIdx_x() as usize;           // head index
        let C = chunk_size as usize;                     // chunk size
        let K = head_k_dim as usize;                     // key dimension
        let V = head_v_dim as usize;                     // value dimension
        let num_chunks = (seq_len as usize + C - 1) / C;

        let rcp_sqrt_k = 1.0f32 / (K as f32).sqrt();
        let decay_rate = fast_expf(a_log[h]);            // A = exp(A_log[h])

        // Shared memory layout:
        // k_normed[C*K], k_beta[C*K], attn[C*C], g_cs[C], beta_arr[C], row_buf[C]
        let smem = DynamicSharedArray::<f32>::get();
        let k_normed = smem;                              // [C*K] f32
        let k_beta = unsafe { smem.add(C * K) };          // [C*K] f32
        let attn = unsafe { smem.add(2 * C * K) };        // [C*C] f32
        let g_cs = unsafe { smem.add(2 * C * K + C * C) };// [C] f32
        let beta_arr = unsafe { g_cs.add(C) };            // [C] f32
        let row_buf = unsafe { beta_arr.add(C) };         // [C] f32

        let state_base = h * K * V;                       // base index into state[h][k][v]

        // Per-chunk loop: sequential across chunks (state recurrence)
        for c in 0..num_chunks {
            let chunk_start = c * C;
            let actual_len = C.min(seq_len as usize - chunk_start);

            // ═══════════ PHASE 1: Compute g/beta/g_cumsum, L2-norm key, k_beta ═══════════

            // ── Compute g[i] and beta[i] for each position in the chunk ─────
            {
                let tid = thread::threadIdx_x() as usize;
                for offset in (0..C).step_by(thread::blockDim_x() as usize) {
                    let i = tid + offset;
                    if i >= actual_len { continue; }
                    let seq_pos = chunk_start + i;

                    // g = -exp(A_log[h]) * softplus(a_proj[seq_pos][h] + dt_bias[h])
                    let a_val = f32::from_bits((a_proj[seq_pos * num_heads as usize + h] as u32) << 16);
                    let sp_val = a_val + dt_bias[h];

                    let softplus: f32;
                    if sp_val > 20.0f32 {
                        softplus = sp_val;
                    } else if sp_val < -20.0f32 {
                        softplus = 0.0f32;
                    } else {
                        softplus = libm::logf(1.0f32 + fast_expf(sp_val));
                    }

                    let g_val = -decay_rate * softplus;

                    // beta = sigmoid(b_proj[seq_pos][h])
                    let b_val = f32::from_bits((b_proj[seq_pos * num_heads as usize + h] as u32) << 16);
                    let beta_v = 1.0f32 / (1.0f32 + fast_expf(-b_val));

                    unsafe { *g_cs.add(i) = g_val; }
                    unsafe { *beta_arr.add(i) = beta_v; }
                }
            }
            cuda_device::sync_threads();

            // ── Zero-init padded positions ────────────────────────────────────
            {
                let tid = thread::threadIdx_x() as usize;
                for i in (tid..C).step_by(thread::blockDim_x() as usize) {
                    if i >= actual_len {
                        unsafe { *g_cs.add(i) = 0.0f32; }
                        unsafe { *beta_arr.add(i) = 0.0f32; }
                    }
                }
            }
            cuda_device::sync_threads();

            // ── Cumulative sum of g over chunk positions (sequential scan) ───────
            if thread::threadIdx_x() == 0 {
                let mut running = 0.0f32;
                for i in 0..C {
                    unsafe { running += *g_cs.add(i); }
                    unsafe { *g_cs.add(i) = running; }
                }
            }
            cuda_device::sync_threads();

            // ── L2-normalize key and compute k_beta ─────────────────────────────
            {
                let tid = thread::threadIdx_x() as usize;

                // Step 1: load key into k_normed shared buffer
                for offset in (0..C * K).step_by(thread::blockDim_x() as usize) {
                    let idx = tid + offset;
                    if idx >= C * K { continue; }
                    let row = idx / K;
                    let col = idx % K;
                    if row >= actual_len {
                        unsafe { *k_normed.add(row * K + col) = 0.0f32; }
                        continue;
                    }
                    let seq_pos = chunk_start + row;
                    let k_idx = seq_pos * num_heads as usize * K + h * K + col;
                    unsafe { *k_normed.add(row * K + col) = f32::from_bits((key[k_idx] as u32) << 16); }
                }

                // Step 2: per-row L2 norm and normalize
                if tid < actual_len {
                    let mut sum_sq = 0.0f32;
                    for d in 0..K {
                        unsafe {
                            let kv = *k_normed.add(tid * K + d);
                            sum_sq += kv * kv;
                        }
                    }
                    let rcp_norm = 1.0f32 / (sum_sq + 1e-6f32).sqrt();
                    for d in 0..K {
                        unsafe {
                            let idx = tid * K + d;
                            *k_normed.add(idx) = *k_normed.add(idx) * rcp_norm;
                        }
                    }
                }

                cuda_device::sync_threads();

                // Step 3: compute k_beta = k_normed * beta, zero padded rows
                for offset in (0..C * K).step_by(thread::blockDim_x() as usize) {
                    let idx = tid + offset;
                    if idx >= C * K { continue; }
                    let row = idx / K;
                    let col = idx % K;

                    unsafe {
                        let kv = *k_normed.add(row * K + col);
                        let btv = if row < actual_len { *beta_arr.add(row) } else { 0.0f32 };
                        *k_beta.add(row * K + col) = kv * btv;
                    }
                }
            }
            cuda_device::sync_threads();

            // ═══════════ PHASE 2: Intra-chunk GEMM — attn matrix ═══════════
            {
                let tid = thread::threadIdx_x() as usize;
                let total = C * C;
                let per_thread = (total + thread::blockDim_x() as usize - 1) / thread::blockDim_x() as usize;

                for flat in (tid * per_thread)..(tid + 1) * per_thread {
                    if flat >= total { break; }
                    let row = flat / C;
                    let col = flat % C;

                    // Dot product: k_beta[row] · k_normed[col]
                    let mut sum = 0.0f32;
                    for d in 0..K {
                        unsafe {
                            sum += *k_beta.add(row * K + d) * *k_normed.add(col * K + d);
                        }
                    }

                    // Apply decay mask: exp(g_cs[row] - g_cs[col]) for row > col
                    let attn_val: f32;
                    if row > col {
                        let g_diff = unsafe { *g_cs.add(row) - *g_cs.add(col) };
                        attn_val = -sum * fast_expf(g_diff);
                    } else {
                        attn_val = 0.0f32;
                    }

                    unsafe { *attn.add(row * C + col) = attn_val; }
                }
            }
            cuda_device::sync_threads();

            // ═══════════ PHASE 3: Forward substitution + identity ═══════════
            if thread::threadIdx_x() == 0 {
                // Forward substitution: for each row i, update attn[i][j]
                for i in 1..actual_len {
                    // Save original row values before update
                    for j in 0..i {
                        unsafe { *row_buf.add(j) = *attn.add(i * C + j); }
                    }

                    // Update: attn[i][j] = row[j] + sum_m(row[m] * attn[m][j])
                    for j in 0..i {
                        let mut accum = 0.0f32;
                        for m in 0..i {
                            unsafe {
                                accum += *row_buf.add(m) * *attn.add(m * C + j);
                            }
                        }
                        unsafe {
                            *attn.add(i * C + j) = *row_buf.add(j) + accum;
                        }
                    }
                }

                // Add identity (only for valid positions)
                for i in 0..actual_len {
                    unsafe { *attn.add(i * C + i) += 1.0f32; }
                }
            }
            cuda_device::sync_threads();

            // ═══════════ PHASE 4: Compute output per (row, col_v) ═══════════
            {
                let tid = thread::threadIdx_x() as usize;
                let total = C * V;
                let per_thread = (total + thread::blockDim_x() as usize - 1) / thread::blockDim_x() as usize;

                for flat_v in (tid * per_thread)..(tid + 1) * per_thread {
                    if flat_v >= total { break; }

                    let row = flat_v / V;
                    let col_v = flat_v % V;

                    if row >= actual_len || col_v >= V { continue; }

                    let seq_row = chunk_start + row;
                    let q_base = seq_row * num_heads as usize * K + h * K;

                    // ── Load and L2-normalize query for this row ─────
                    let mut q_l2_sq = 0.0f32;
                    let mut q_reg: [f32; 128] = [0.0f32; 128];
                    for d in 0..K {
                        let qv = f32::from_bits((query[q_base + d] as u32) << 16);
                        q_reg[d] = qv;
                        q_l2_sq += qv * qv;
                    }
                    let q_norm_rational = 1.0f32 / (q_l2_sq + 1e-6f32).sqrt();
                    let exp_g_row = unsafe { fast_expf(*g_cs.add(row)) };
                    // ── attn_inter[row][col_v] = (q_normed * exp(g_cs) @ S) ──
                    // @lat: [[tests/gdn_chunked_prefill_test#GDN Chunked Prefill Kernel Test#rcp_sqrt_k Double Application Bug Fix]]
                    let mut attn_inter_val = 0.0f32;
                    for d in 0..K {
                        let q_normed_f32 = q_reg[d] * q_norm_rational;
                        let q_scl = q_normed_f32 * exp_g_row;
                        attn_inter_val += q_scl * state[state_base + d * V + col_v];
                    }

                    // ── Compute output contribution from attn_qk ─────
                    let mut output_from_qk = 0.0f32;

                    for j in 0..=row {
                        // qk_dot: query-key dot product with decay
                        let mut qk_dot_j = 0.0f32;
                        for d in 0..K {
                            unsafe {
                                let q_normed_f32 = q_reg[d] * q_norm_rational;
                                qk_dot_j += q_normed_f32 * *k_normed.add(j * K + d);
                            }
                        }
                        let g_diff = unsafe { *g_cs.add(row) - *g_cs.add(j) };
                        let attn_qk_val = qk_dot_j * fast_expf(g_diff);
                        // v_new[j][col_v] = attn @ (v * beta)
                        let mut v_new_j = 0.0f32;
                        for ii in 0..actual_len {
                            let seq_pos = chunk_start + ii;
                            let v_idx = seq_pos * num_heads as usize * V + h * V + col_v;
                            unsafe {
                                let v_val = f32::from_bits((value[v_idx] as u32) << 16) * *beta_arr.add(ii);
                                v_new_j += *attn.add(j * C + ii) * v_val;
                            }
                        }

                        // v_prime[j][col_v] = k_cumdecay @ S
                        let mut v_prime_j = 0.0f32;
                        for d in 0..K {
                            let mut k_cd_j = 0.0f32;
                            for ii in 0..actual_len {
                                unsafe {
                                    k_cd_j += *attn.add(j * C + ii)
                                        * *k_beta.add(ii * K + d)
                                        * fast_expf(*g_cs.add(ii));
                                }
                            }
                            v_prime_j += k_cd_j * state[state_base + d * V + col_v];
                        }

                        let v_nc_j = v_new_j - v_prime_j;
                        output_from_qk += attn_qk_val * v_nc_j;
                    }

                    // ── Write final output ─────
                    let out_val = attn_inter_val + output_from_qk;
                    let out_idx = seq_row * num_heads as usize * V + h * V + col_v;
                    unsafe { *output.get_unchecked_mut(out_idx) = f32_to_bf16(out_val); }
                }
            }

            // ═══════════ PHASE 5: State update ═══════════
            {
                let tid = thread::threadIdx_x() as usize;
                let total = K * V;
                let per_thread = (total + thread::blockDim_x() as usize - 1) / thread::blockDim_x() as usize;

                for flat_s in (tid * per_thread)..(tid + 1) * per_thread {
                    if flat_s >= total { break; }

                    let d = flat_s / V;
                    let col_v = flat_s % V;
                    let exp_g_last = unsafe { fast_expf(*g_cs.add(actual_len - 1)) };
                    // S[d][col_v] *= exp(g_cs[-1])
                    let mut s_val = state[state_base + d * V + col_v] * exp_g_last;

                    // Add: sum_j exp_diff[j] * k_normed[j][d] * v_nc[j][col_v]
                    for j in 0..actual_len {
                        let exp_diff_j = unsafe { fast_expf(*g_cs.add(actual_len - 1) - *g_cs.add(j)) };
                        // v_new[j][col_v] = attn @ (v * beta)
                        let mut v_new_j = 0.0f32;
                        for ii in 0..actual_len {
                            let seq_pos = chunk_start + ii;
                            let v_idx = seq_pos * num_heads as usize * V + h * V + col_v;
                            unsafe {
                                let v_val = f32::from_bits((value[v_idx] as u32) << 16) * *beta_arr.add(ii);
                                v_new_j += *attn.add(j * C + ii) * v_val;
                            }
                        }

                        // v_prime[j][col_v] = k_cumdecay @ S
                        let mut v_prime_j = 0.0f32;
                        for dd in 0..K {
                            let mut k_cd_j = 0.0f32;
                            for ii in 0..actual_len {
                                unsafe {
                                    k_cd_j += *attn.add(j * C + ii)
                                        * *k_beta.add(ii * K + dd)
                                        * fast_expf(*g_cs.add(ii));
                                }
                            }
                            v_prime_j += k_cd_j * state[state_base + dd * V + col_v];
                        }

                        let v_nc_j = v_new_j - v_prime_j;
                        unsafe { s_val += exp_diff_j * *k_normed.add(j * K + d) * v_nc_j; }
                    }

                    state[state_base + d * V + col_v] = s_val;
                }
            }

        } // end chunk loop
    }
}
