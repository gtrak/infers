//! Kernel library for infers — cuda-oxide PTX kernels.
//!
//! Kernels are compiled to PTX by rustc-codegen-cuda via `#[cuda_module]`.
//! Host code loads the module with `kernels::load(&ctx)`.


use cuda_device::{SharedArray, DisjointSlice, DynamicSharedArray, cuda_module, kernel, launch_bounds, thread};

/// All device kernels — compiled to PTX by cuda-oxide.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Device sqrt: f32::sqrt() compiles to PTX sqrt.rn.f32 (validated in POC).
    #[inline(always)]
    fn dev_sqrtf(x: f32) -> f32 {
        x.sqrt()
    }

    /// Element-wise addition kernel: output[i] = a[i] + b[i] in BF16.
    ///
    /// Inputs and output are stored as u16 (bf16 bit representation).
    /// Each thread converts bf16→f32, performs the add in f32, then
    /// converts back to bf16. Grid-stride loop pattern.
    ///
    /// # Launch configuration
    /// * grid: derived from `LaunchConfig::for_num_elems(total_elements)`
    /// * block: 256 threads (via `#[launch_bounds(256)]`)
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_add_bf16(
        a: &[u16],
        b: &[u16],
        mut out: DisjointSlice<u16>,
        total_elements: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_elements as usize;

        for i in (tid as usize..total).step_by(stride as usize) {
            // bf16 → f32: reinterpret the 16 bits as upper 16 of f32
            let a_f32 = f32::from_bits((a[i] as u32) << 16);
            let b_f32 = f32::from_bits((b[i] as u32) << 16);

            // f32 compute
            let sum = a_f32 + b_f32;

            // f32 → bf16: convert and store as u16
            unsafe { *out.get_unchecked_mut(i) = f32_to_bf16(sum); }
        }
    }

    /// Embedding gather kernel: output[i] = weight[token_ids[pos] * hidden_size + dim]
    /// where pos = i / hidden_size, dim = i % hidden_size.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_embedding_gather_bf16(
        weight: &[u16],
        token_ids: &[i32],
        mut output: DisjointSlice<u16>,
        seq_len: u32,
        hidden_size: u32,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = (seq_len as usize) * (hidden_size as usize);

        for i in (tid as usize..total).step_by(stride as usize) {
            let pos = i / hidden_size as usize;
            let dim = i % hidden_size as usize;
            let token_id = token_ids[pos] as usize;
            let w_idx = token_id * hidden_size as usize + dim;
            unsafe { *output.get_unchecked_mut(i) = weight[w_idx]; }
        }
    }

    /// SiLU activation: output[i] = x[i] * sigmoid(x[i])
    /// where sigmoid(v) = 1.0 / (1.0 + exp(-v))
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_silu_bf16(
        x: &[u16],
        mut output: DisjointSlice<u16>,
        total_elements: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_elements as usize;

        for i in (tid as usize..total).step_by(stride as usize) {
            let val = f32::from_bits((x[i] as u32) << 16);
            let sigmoid = 1.0 / (1.0 + libm::expf(-val));
            unsafe { *output.get_unchecked_mut(i) = f32_to_bf16(val * sigmoid); }
        }
    }

    /// SiLU Gated Linear Unit: output[i] = x[i] * gate[i] * sigmoid(gate[i])
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_silu_glu_bf16(
        x: &[u16],
        gate: &[u16],
        mut output: DisjointSlice<u16>,
        total_elements: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_elements as usize;

        for i in (tid as usize..total).step_by(stride as usize) {
            let x_val = f32::from_bits((x[i] as u32) << 16);
            let g_val = f32::from_bits((gate[i] as u32) << 16);
            let sigmoid_g = 1.0 / (1.0 + libm::expf(-g_val));
            unsafe { *output.get_unchecked_mut(i) = f32_to_bf16(x_val * g_val * sigmoid_g); }
        }
    }

    /// Attention output gate: output[i] = x[i] * sigmoid(gate[i])
    /// Unlike SwiGLU, does NOT multiply by gate.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_attn_output_gate_bf16(
        x: &[u16],
        gate: &[u16],
        mut output: DisjointSlice<u16>,
        total_elements: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = total_elements as usize;

        for i in (tid as usize..total).step_by(stride as usize) {
            let x_val = f32::from_bits((x[i] as u32) << 16);
            let g_val = f32::from_bits((gate[i] as u32) << 16);
            let sigmoid_g = 1.0 / (1.0 + libm::expf(-g_val));
            unsafe { *output.get_unchecked_mut(i) = f32_to_bf16(x_val * sigmoid_g); }
        }
    }

    /// Argmax per row using shared memory reduction.
    /// Launch: batch_size blocks, 256 threads each.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_argmax_bf16(
        logits: &[u16],
        mut output: DisjointSlice<i32>,
       _batch_size: u32,
        vocab_size: u32,
    ) {
        static mut SVALS: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut SINDS: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let vocab = vocab_size as usize;

        // Phase 1: each thread scans its chunk of the vocabulary
        let mut local_max: f32 = f32::NEG_INFINITY;
        let mut local_idx: f32 = -1.0;

        for i in (tid as usize..vocab).step_by(256) {
            let val = f32::from_bits((logits[(row as usize) * vocab + i] as u32) << 16);
            if val > local_max {
                local_max = val;
                local_idx = i as f32;
            }
        }

        // Write to shared memory
        unsafe {
            SVALS[tid as usize] = local_max;
            SINDS[tid as usize] = local_idx;
        }
        cuda_device::sync_threads();

        // Phase 2: halving stride reduction
        let mut s = 128u32;
        while s > 0 {
            unsafe {
                if tid < s {
                    let other_val = SVALS[(tid + s) as usize];
                    let own_val = SVALS[tid as usize];
                    if other_val > own_val {
                        SVALS[tid as usize] = other_val;
                        SINDS[tid as usize] = SINDS[(tid + s) as usize];
                    }
                }
            }
            cuda_device::sync_threads();
            s >>= 1;
        }

        // Phase 3: thread 0 writes result
        if tid == 0 {
            unsafe {
                *output.get_unchecked_mut(row as usize) = SINDS[0] as i32;
            }
        }
    }

    /// KV cache write kernel: scattered write by position.
    /// K part: output[pos * head_dim + dim] = k[token * head_dim + dim]
    /// V part: output[max_seq_len * head_dim + pos * head_dim + dim] = v[token * head_dim + dim]
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_kv_cache_write_bf16(
        k: &[u16],
        v: &[u16],
        mut kv_cache: DisjointSlice<u16>,
        positions: &[i32],
        seq_len: u32,
        head_dim: u32,
        max_seq_len: u32,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = (seq_len as usize) * (head_dim as usize);

        for i in (tid as usize..total).step_by(stride as usize) {
            let token = i / head_dim as usize;
            let dim = i % head_dim as usize;
            let pos = positions[token] as usize;

            // Write K: kv_cache[pos * head_dim + dim] = k[token * head_dim + dim]
            let k_offset = pos * head_dim as usize + dim;
            unsafe {
                *kv_cache.get_unchecked_mut(k_offset) = k[i];
            }

            // Write V: kv_cache[max_seq_len * head_dim + pos * head_dim + dim] = v[token * head_dim + dim]
            let v_offset = (max_seq_len as usize) * (head_dim as usize) + pos * head_dim as usize + dim;
            unsafe {
                *kv_cache.get_unchecked_mut(v_offset) = v[i];
            }
        }
    }

    /// RMSNorm: output = x * rsqrt(mean(x²) + eps) * (1 + weight)
    /// One block per row, shared memory tree reduction for sum-of-squares.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_rmsnorm_bf16(
        x: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<u16>,
        hidden: u32,
        eps: f32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let hidden_usize = hidden as usize;

        // Dynamic shared memory for reduction buffer (256 f32s)
        let smem = DynamicSharedArray::<f32>::get();

        // Phase 1: partial sum-of-squares per thread
        let mut local_sum_sq: f32 = 0.0;
        let row_offset = row * hidden_usize;
        for i in (tid..hidden_usize).step_by(256) {
            let val = f32::from_bits((x[row_offset + i] as u32) << 16);
            local_sum_sq += val * val;
        }

        unsafe { *smem.add(tid) = local_sum_sq; }
        cuda_device::sync_threads();

        // Phase 2: halving reduction in shared memory
        let mut s = 128u32;
        while s > 0 {
            if tid < s as usize {
                unsafe {
                    let val = *smem.add(tid);
                    let other = *smem.add(tid + s as usize);
                    *smem.add(tid) = val + other;
                }
            }
            cuda_device::sync_threads();
            s >>= 1;
        }

        // Phase 3: thread 0 computes inverse RMS and writes to smem[0]
        if tid == 0 {
            let sum_sq = unsafe { *smem.add(0) };
            let inv_rms = 1.0 / dev_sqrtf(sum_sq / hidden_usize as f32 + eps);
            unsafe { *smem.add(0) = inv_rms; }
        }
        cuda_device::sync_threads();

        // Phase 4: apply normalization — each thread handles its chunk
        let inv_rms = unsafe { *smem.add(0) };
        for i in (tid..hidden_usize).step_by(256) {
            let x_val = f32::from_bits((x[row_offset + i] as u32) << 16);
            let w_val = f32::from_bits((weight[i] as u32) << 16);
            let result = x_val * inv_rms * (1.0 + w_val);
            unsafe { *output.get_unchecked_mut(row_offset + i) = f32_to_bf16(result); }
        }
    }

    /// RMSNorm with SiLU gate: output = weight * x_norm * SiLU(gate)
    /// where x_norm = x * rsqrt(mean(x²) + eps) and SiLU(g) = g / (1 + exp(-g))
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_rms_norm_gated_bf16(
        input: &[u16],
        gate: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<u16>,
        _n: u32,
        d: u32,
        eps: f32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let d_usize = d as usize;

        let smem = DynamicSharedArray::<f32>::get();

        // Phase 1: partial sum-of-squares per thread
        let mut local_sum_sq: f32 = 0.0;
        let row_offset = row * d_usize;
        for i in (tid..d_usize).step_by(256) {
            let val = f32::from_bits((input[row_offset + i] as u32) << 16);
            local_sum_sq += val * val;
        }

        unsafe { *smem.add(tid) = local_sum_sq; }
        cuda_device::sync_threads();

        // Phase 2: halving reduction
        let mut s = 128u32;
        while s > 0 {
            if tid < s as usize {
                unsafe {
                    *smem.add(tid) = *smem.add(tid) + *smem.add(tid + s as usize);
                }
            }
            cuda_device::sync_threads();
            s >>= 1;
        }

        // Phase 3: thread 0 computes inverse RMS
        if tid == 0 {
            let sum_sq = unsafe { *smem.add(0) };
            let inv_rms = 1.0 / dev_sqrtf(sum_sq / d_usize as f32 + eps);
            unsafe { *smem.add(0) = inv_rms; }
        }
        cuda_device::sync_threads();

        // Phase 4: apply normalization and SiLU gate
        let inv_rms = unsafe { *smem.add(0) };
        for i in (tid..d_usize).step_by(256) {
            let x_val = f32::from_bits((input[row_offset + i] as u32) << 16);
            let g_val = f32::from_bits((gate[row_offset + i] as u32) << 16);
            let w_val = f32::from_bits((weight[i] as u32) << 16);
            let x_norm = x_val * inv_rms;
            let silu_gate = g_val / (1.0 + libm::expf(-g_val));
            let result = w_val * x_norm * silu_gate;
            unsafe { *output.get_unchecked_mut(row_offset + i) = f32_to_bf16(result); }
        }
    }

    /// L2 norm: output[i] = input[i] / sqrt(sum(input²) + eps) per row
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_l2norm_bf16(
        input: &[u16],
        mut output: DisjointSlice<u16>,
        dim: u32,
        eps: f32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let dim_usize = dim as usize;

        let smem = DynamicSharedArray::<f32>::get();

        // Phase 1: partial sum-of-squares per thread
        let mut local_sum_sq: f32 = 0.0;
        let row_offset = row * dim_usize;
        for i in (tid..dim_usize).step_by(256) {
            let val = f32::from_bits((input[row_offset + i] as u32) << 16);
            local_sum_sq += val * val;
        }

        unsafe { *smem.add(tid) = local_sum_sq; }
        cuda_device::sync_threads();

        // Phase 2: halving reduction
        let mut s = 128u32;
        while s > 0 {
            if tid < s as usize {
                unsafe {
                    *smem.add(tid) = *smem.add(tid) + *smem.add(tid + s as usize);
                }
            }
            cuda_device::sync_threads();
            s >>= 1;
        }

        // Phase 3: thread 0 computes inverse L2 norm
        if tid == 0 {
            let sum_sq = unsafe { *smem.add(0) };
            let inv_norm = 1.0 / dev_sqrtf(sum_sq + eps);
            unsafe { *smem.add(0) = inv_norm; }
        }
        cuda_device::sync_threads();

        // Phase 4: apply normalization
        let inv_norm = unsafe { *smem.add(0) };
        for i in (tid..dim_usize).step_by(256) {
            let val = f32::from_bits((input[row_offset + i] as u32) << 16);
            let result = val * inv_norm;
            unsafe { *output.get_unchecked_mut(row_offset + i) = f32_to_bf16(result); }
        }
    }

    /// Softmax with optional causal mask.
    /// 3-phase: max reduction, exp(x-max) sum reduction, normalize.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_softmax_bf16(
        scores: &[u16],
        mut output: DisjointSlice<u16>,
        seq_len: u32,
        use_causal: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let sl_usize = seq_len as usize;

        let smem = DynamicSharedArray::<f32>::get();

        // Phase 1: max reduction per row
        let mut local_max: f32 = f32::NEG_INFINITY;
        for c in (tid..sl_usize).step_by(256) {
            let val = f32::from_bits((scores[row * sl_usize + c] as u32) << 16);
            if use_causal == 0 || c <= row {
                if val > local_max {
                    local_max = val;
                }
            }
        }

        unsafe { *smem.add(tid) = local_max; }
        cuda_device::sync_threads();

        // Halving max reduction
        let mut s = 128u32;
        while s > 0 {
            if tid < s as usize {
                unsafe {
                    let own = *smem.add(tid);
                    let other = *smem.add(tid + s as usize);
                    *smem.add(tid) = if other > own { other } else { own };
                }
            }
            cuda_device::sync_threads();
            s >>= 1;
        }

        let row_max = unsafe { *smem.add(0) };

        // Phase 2: sum of exp(x - max) reduction
        let mut local_sum: f32 = 0.0;
        for c in (tid..sl_usize).step_by(256) {
            if use_causal == 0 || c <= row {
                let val = f32::from_bits((scores[row * sl_usize + c] as u32) << 16);
                local_sum += libm::expf(val - row_max);
            }
        }

        unsafe { *smem.add(tid) = local_sum; }
        cuda_device::sync_threads();

        // Halving sum reduction
        let mut s2 = 128u32;
        while s2 > 0 {
            if tid < s2 as usize {
                unsafe {
                    *smem.add(tid) = *smem.add(tid) + *smem.add(tid + s2 as usize);
                }
            }
            cuda_device::sync_threads();
            s2 >>= 1;
        }

        let row_sum = unsafe { *smem.add(0) };

        // Phase 3: write normalized output
        for c in (tid..sl_usize).step_by(256) {
            let out_idx = row * sl_usize + c;
            if use_causal == 0 || c <= row {
                let val = f32::from_bits((scores[out_idx] as u32) << 16);
                let result = libm::expf(val - row_max) / row_sum;
                unsafe { *output.get_unchecked_mut(out_idx) = f32_to_bf16(result); }
            } else {
                // Masked position → zero
                unsafe { *output.get_unchecked_mut(out_idx) = 0u16; }
            }
        }
    }

    /// Depthwise 1D convolution with SiLU activation.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_conv1d_depthwise_silu_bf16(
        input: &[u16],
        weight: &[u16],
        mut output: DisjointSlice<u16>,
        batch_size: u32,
        conv_dim: u32,
        seq_len: u32,
        kernel_size: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = (batch_size as usize) * (seq_len as usize) * (conv_dim as usize);

        for i in (tid..total).step_by(stride as usize) {
            // Decompose output index: [batch][seq_len][conv_dim] layout (D innermost, matches nvcc)
            let d = i % conv_dim as usize;
            let t = (i / conv_dim as usize) % seq_len as usize;
            let b = i / (seq_len as usize * conv_dim as usize);

            let pad = (kernel_size - 1) as usize;
            let mut sum: f32 = 0.0;

            for p in 0..kernel_size as usize {
                let input_t = t + p;
                if input_t >= pad && input_t < seq_len as usize + pad {
                    let adj_t = input_t - pad;
                    let inp_idx = b * seq_len as usize * conv_dim as usize + adj_t * conv_dim as usize + d;
                    let w_idx = d * kernel_size as usize + p;
                    let inp_val = f32::from_bits((input[inp_idx] as u32) << 16);
                    let w_val = f32::from_bits((weight[w_idx] as u32) << 16);
                    sum += inp_val * w_val;
                }
            }

            // SiLU activation: sum / (1 + exp(-sum))
            let silu = sum / (1.0 + libm::expf(-sum));
            unsafe { *output.get_unchecked_mut(i) = f32_to_bf16(silu); }
        }
    }

    /// Paged KV cache write with block-table address translation.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_paged_kv_write_bf16(
        k: &[u16],
        v: &[u16],
        mut page_pool: DisjointSlice<u16>,
        block_table: &[i32],
        positions: &[i32],
        seq_len: u32,
        _head_dim: u32,
        page_size: u32,
        kv_dim: u32,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = (seq_len as usize) * (kv_dim as usize);

        for i in (tid..total).step_by(stride as usize) {
            let pos_idx = i / kv_dim as usize;
            let dim = i % kv_dim as usize;
            let pos = positions[pos_idx] as usize;

            let logical_page = pos / page_size as usize;
            let token_in_page = pos % page_size as usize;
            let physical_page = block_table[logical_page] as usize;

            let page_stride = 2 * (page_size as usize) * (kv_dim as usize);

            // Write K
            let k_offset = physical_page * page_stride + token_in_page * kv_dim as usize + dim;
            unsafe { *page_pool.get_unchecked_mut(k_offset) = k[i]; }

            // Write V
            let v_offset = physical_page * page_stride + (page_size as usize) * (kv_dim as usize) + token_in_page * kv_dim as usize + dim;
            unsafe { *page_pool.get_unchecked_mut(v_offset) = v[i]; }
        }
    }

    /// Paged KV cache read (mirror of write).
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_paged_kv_read_bf16(
        page_pool: &[u16],
        block_table: &[i32],
        _num_pages: u32,
        num_cached_tokens: u32,
        _head_dim: u32,
        page_size: u32,
        kv_dim: u32,
        mut k_out: DisjointSlice<u16>,
        mut v_out: DisjointSlice<u16>,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let total = (num_cached_tokens as usize) * (kv_dim as usize);

        for i in (tid..total).step_by(stride as usize) {
            let pos_idx = i / kv_dim as usize;
            let dim = i % kv_dim as usize;
            let pos = pos_idx; // Contiguous read from position 0

            let logical_page = pos / page_size as usize;
            let token_in_page = pos % page_size as usize;
            let physical_page = block_table[logical_page] as usize;

            let page_stride = 2 * (page_size as usize) * (kv_dim as usize);

            // Read K
            let k_offset = physical_page * page_stride + token_in_page * kv_dim as usize + dim;
            unsafe { *k_out.get_unchecked_mut(i) = page_pool[k_offset]; }

            // Read V
            let v_offset = physical_page * page_stride + (page_size as usize) * (kv_dim as usize) + token_in_page * kv_dim as usize + dim;
            unsafe { *v_out.get_unchecked_mut(i) = page_pool[v_offset]; }
        }
    }

    /// Rotary Position Embedding with precomputed sin/cos.
    /// Applies rotation to both Q and K in-place using half-split pairing (rotate_half).
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_rope_bf16(
        mut q: DisjointSlice<u16>,
        mut k_tensor: DisjointSlice<u16>,
        cos: &[f32],
        sin: &[f32],
        positions: &[i32],
        total_tokens: u32,
        num_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let half_rotary = (rotary_dim / 2) as usize;
        let total_pairs = (total_tokens as usize) * (num_heads as usize) * half_rotary;

        let idx = thread::index_1d();
        let tid = idx.get();
        let stride = thread::blockDim_x() * thread::gridDim_x();

        for t in (tid..total_pairs).step_by(stride as usize) {
            let token_idx = t / ((num_heads * rotary_dim / 2) as usize);
            let head_idx = (t / half_rotary) % num_heads as usize;
            let dim_pair = t % half_rotary;

            let pos = positions[token_idx] as usize;
            let cs_idx = pos * half_rotary + dim_pair;
            let cos_val = cos[cs_idx];
            let sin_val = sin[cs_idx];

            let base = token_idx * num_heads as usize * head_dim as usize + head_idx * head_dim as usize;
            let i0 = base + dim_pair;
            let i1 = base + dim_pair + half_rotary;

            // Apply rotation to Q
            let q0 = unsafe { f32::from_bits((*q.get_unchecked_mut(i0) as u32) << 16) };
            let q1 = unsafe { f32::from_bits((*q.get_unchecked_mut(i1) as u32) << 16) };
            unsafe {
                *q.get_unchecked_mut(i0) = f32_to_bf16(q0 * cos_val - q1 * sin_val);
                *q.get_unchecked_mut(i1) = f32_to_bf16(q0 * sin_val + q1 * cos_val);
            }

            // Apply rotation to K
            let k0 = unsafe { f32::from_bits((*k_tensor.get_unchecked_mut(i0) as u32) << 16) };
            let k1 = unsafe { f32::from_bits((*k_tensor.get_unchecked_mut(i1) as u32) << 16) };
            unsafe {
                *k_tensor.get_unchecked_mut(i0) = f32_to_bf16(k0 * cos_val - k1 * sin_val);
                *k_tensor.get_unchecked_mut(i1) = f32_to_bf16(k0 * sin_val + k1 * cos_val);
            }
        }
    }
}
