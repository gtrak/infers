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

        // Phase 2: halving reduction (start from d/2, not hardcoded 128)
        let mut s = d_usize / 2;
        while s > 0 {
            if tid < s {
                unsafe {
                    *smem.add(tid) = *smem.add(tid) + *smem.add(tid + s);
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

    // ─── FP8 quantize/dequantize with Fp8Format trait ─────────────

    /// Trait for FP8 quantization format. Each format (E4M3, E5M2) implements
    /// the quantize and dequantize methods with its specific bit layout.
    pub trait Fp8Format {
        fn quantize(val: f32) -> u8;
        fn dequantize(val: u8) -> f32;
    }

    /// Fp8E4M3 — 1 sign, 4 exponent (bias 7), 3 mantissa bits.
    /// Max finite: 0x77 positive / 0xF7 negative.
    pub struct Fp8E4M3;
    impl Fp8Format for Fp8E4M3 {
        #[inline(always)]
        fn quantize(val: f32) -> u8 {
            let bits = val.to_bits();
            let sign = (bits >> 31) & 1;
            let exp = (bits >> 23) & 0xFF;
            let mantissa = bits & 0x7FFFFF;

            // NaN → 0x7F, Inf → max finite (0x77 / 0xF7)
            if exp == 0xFF {
                if mantissa != 0 { return 0x7F; }  // NaN
                return if sign == 0 { 0x77 } else { 0xF7 };  // Inf → max finite
            }
            // Zero/subnormal
            if exp == 0 && mantissa == 0 {
                return (sign & 1) as u8 * 0x80;
            }

            let fp8_exp = (exp as i32) - 127 + 7;
            // Clamp to max finite
            if fp8_exp >= 0xF {
                return if sign != 0 { 0xF7 } else { 0x77 };
            }
            // Underflow to zero
            if fp8_exp < 0 {
                return (sign & 1) as u8 * 0x80;
            }

            let fp8_mant = ((mantissa >> 20) & 0x7) as u8;
            ((((sign & 1) as u8) << 7) | ((fp8_exp as u8) << 3) | fp8_mant)
        }

        #[inline(always)]
        fn dequantize(val: u8) -> f32 {
            let sign = (val >> 7) & 1;
            let exp = (val >> 3) & 0xF;
            let mant = val & 0x7;

            // NaN
            if exp == 0xF {
                return f32::from_bits(0x7FC00000);
            }
            // Zero
            if exp == 0 && mant == 0 {
                return if sign != 0 { -0.0f32 } else { 0.0f32 };
            }

            let fp32_exp = if exp == 0 { 0 } else { (exp as u32) + 120 }; // 127 - 7 = 120
            let fp32_mant = (mant as u32) << 20;
            f32::from_bits(((sign as u32) << 31) | (fp32_exp << 23) | fp32_mant)
        }
    }

    /// Fp8E5M2 — 1 sign, 5 exponent (bias 15), 2 mantissa bits.
    /// Max finite: 0x7B positive / 0xFB negative.
    pub struct Fp8E5M2;
    impl Fp8Format for Fp8E5M2 {
        #[inline(always)]
        fn quantize(val: f32) -> u8 {
            let bits = val.to_bits();
            let sign = (bits >> 31) & 1;
            let exp = (bits >> 23) & 0xFF;
            let mantissa = bits & 0x7FFFFF;

            // NaN/Inf — sign-preserving
            if exp == 0xFF {
                if mantissa != 0 { return if sign == 0 { 0x7F } else { 0xFF }; }  // NaN
                return if sign == 0 { 0x7C } else { 0xFC };  // Inf
            }
            // Zero/subnormal
            if exp == 0 && mantissa == 0 {
                return (sign & 1) as u8 * 0x80;
            }

            let fp8_exp = (exp as i32) - 127 + 15;
            // Clamp to max finite
            if fp8_exp >= 0x1F {
                return if sign != 0 { 0xFB } else { 0x7B };
            }
            // Underflow to zero
            if fp8_exp < 0 {
                return (sign & 1) as u8 * 0x80;
            }

            let fp8_mant = ((mantissa >> 21) & 0x3) as u8;
            ((((sign & 1) as u8) << 7) | ((fp8_exp as u8) << 2) | fp8_mant)
        }

        #[inline(always)]
        fn dequantize(val: u8) -> f32 {
            let sign = (val >> 7) & 1;
            let exp = (val >> 2) & 0x1F;
            let mant = val & 0x3;

            // NaN
            if exp == 0x1F {
                return f32::from_bits(0x7FC00000);
            }
            // Zero
            if exp == 0 && mant == 0 {
                return if sign != 0 { -0.0f32 } else { 0.0f32 };
            }

            let fp32_exp = if exp == 0 { 0 } else { (exp as u32) + 112 }; // 127 - 15 = 112
            let fp32_mant = (mant as u32) << 21;
            f32::from_bits(((sign as u32) << 31) | (fp32_exp << 23) | fp32_mant)
        }
    }

    /// Generic FP8 quantize inner function. Monomorphized per Fp8Format impl.
    #[inline(always)]
    fn fp8_quantize_inner<F: Fp8Format>(
        input: &[u16],
        mut output: DisjointSlice<u8>,
        n: u32,
    ) {
        let tid = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let total = n as usize;

        for i in (tid..total).step_by(stride) {
            // bf16 → f32
            let val = f32::from_bits((input[i] as u32) << 16);
            let fp8 = F::quantize(val);
            unsafe { *output.get_unchecked_mut(i) = fp8; }
        }
    }

    /// Generic FP8 dequantize inner function. Monomorphized per Fp8Format impl.
    #[inline(always)]
    fn fp8_dequantize_inner<F: Fp8Format>(
        input: &[u8],
        mut output: DisjointSlice<u16>,
        n: u32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let tid = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let stride = (thread::blockDim_x() * thread::gridDim_x()) as usize;
        let total = n as usize;

        for i in (tid..total).step_by(stride) {
            let fp8 = input[i];
            let val = F::dequantize(fp8);
            unsafe { *output.get_unchecked_mut(i) = f32_to_bf16(val); }
        }
    }

    /// FP8 quantize kernel: E4M3 format (BF16 → u8).
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_fp8_quantize_e4m3(input: &[u16], mut output: DisjointSlice<u8>, n: u32) {
        fp8_quantize_inner::<Fp8E4M3>(input, output, n);
    }

    /// FP8 dequantize kernel: E4M3 format (u8 → BF16).
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_fp8_dequantize_e4m3(input: &[u8], mut output: DisjointSlice<u16>, n: u32) {
        fp8_dequantize_inner::<Fp8E4M3>(input, output, n);
    }

    /// FP8 quantize kernel: E5M2 format (BF16 → u8).
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_fp8_quantize_e5m2(input: &[u16], mut output: DisjointSlice<u8>, n: u32) {
        fp8_quantize_inner::<Fp8E5M2>(input, output, n);
    }

    /// FP8 dequantize kernel: E5M2 format (u8 → BF16).
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_fp8_dequantize_e5m2(input: &[u8], mut output: DisjointSlice<u16>, n: u32) {
        fp8_dequantize_inner::<Fp8E5M2>(input, output, n);
    }

    // ─── INT4 GEMM with trait-based dequantization dispatch ─────────────

    /// Trait for dequantizing INT4 weights. Each quant format implements this
    /// with its specific zero-point offset and dequant formula.
    pub trait Dequantize {
        /// Dequantize one INT4 value.
        /// `w_int4` is the raw 4-bit value [0, 15] cast to i8.
        /// `raw_zero` is the raw 4-bit zero point [0, 15] extracted from packed zeros.
        /// `scale` is the FP16 group scale converted to f32.
        /// Returns the dequantized f32 value.
        fn dequant(w_int4: i8, raw_zero: i8, scale: f32) -> f32;
    }

    /// AutoRound INT4: zero = stored_zero + 1
    /// Formula: (w - (stored_zero + 1)) * scale
    pub struct AutoRound;
    impl Dequantize for AutoRound {
        fn dequant(w_int4: i8, raw_zero: i8, scale: f32) -> f32 {
            let zero = raw_zero + 1;
            f32::from(w_int4 - zero) * scale
        }
    }

    /// GGUF INT4: zero = stored_zero (no offset)
    /// Formula: (w - stored_zero) * scale
    pub struct Gguf;
    impl Dequantize for Gguf {
        fn dequant(w_int4: i8, raw_zero: i8, scale: f32) -> f32 {
            f32::from(w_int4 - raw_zero) * scale
        }
    }

    /// Convert half-precision (FP16) bits to f32.
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = (bits >> 15) as u32;
        let exp = ((bits >> 10) & 0x1F) as u32;
        let frac = bits & 0x3FF;

        if exp == 0 {
            // Subnormal or zero: convert to normal f32 with exponent -14
            let mantissa = (frac as u32) << 13;
            let e_bits = if frac != 0 { 0x7F - 14 } else { 0 };
            f32::from_bits((sign << 31) | (e_bits << 23) | mantissa)
        } else if exp == 31 {
            // Inf or NaN
            f32::from_bits((sign << 31) | (0xFFu32 << 23))
        } else {
            // Normal: bias adjustment (15→127), shift mantissa by 13
            let e_bits = exp + (127 - 15);
            f32::from_bits((sign << 31) | (e_bits << 23) | ((frac as u32) << 13))
        }
    }

    /// Generic INT4 GEMM inner function. Monomorphized per Dequantize impl.
    /// NOT a #[kernel] — called from #[kernel] wrappers.
    #[inline(always)]
    fn int4_gemm_inner<Q: Dequantize>(
        output: &mut DisjointSlice<u16>,
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        m: i32, n: i32, k: i32,
        group_size: i32,
        transposed: i32,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let row = (thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y()) as i32;
        let col = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as i32;

        if row >= m || col >= n {
            return;
        }

        let mut acc: f32 = 0.0;
        let k_usize = k as usize;
        let n_usize = n as usize;
        let group_size_usize = group_size as usize;

        for kg in (0i32..k).step_by(group_size as usize) {
            let group_idx = (kg / group_size) as usize;

            // Load scale (FP16 → F32)
            let scale_bits: u16;
            if transposed != 0 {
                scale_bits = scales[group_idx * n_usize + col as usize];
            } else {
                let num_groups = k_usize / group_size_usize;
                scale_bits = scales[col as usize * num_groups + group_idx];
            }
            let scale = f16_to_f32(scale_bits);

            // Unpack zero point (8 per u32)
            let (zero_packed_idx, zero_shift): (usize, usize);
            if transposed != 0 {
                let n_packed = (n_usize + 7) / 8;
                zero_packed_idx = group_idx * n_packed + col as usize / 8;
                zero_shift = (col % 8) as usize * 4;
            } else {
                let num_groups = k_usize / group_size_usize;
                let flat_idx = col as usize * num_groups + group_idx;
                zero_packed_idx = flat_idx / 8;
                zero_shift = (flat_idx % 8) * 4;
            }
            let zero_packed = zeros[zero_packed_idx];
            let raw_zero = ((zero_packed >> zero_shift) & 0xF) as i8;

            for kk in (0i32..group_size).step_by(8) {
                // Load 8 INT4 weights from one u32
                let weight_idx: usize;
                if transposed != 0 {
                    weight_idx = ((kg + kk) >> 3) as usize * n_usize + col as usize;
                } else {
                    weight_idx = (col as usize * k_usize + kg as usize + kk as usize) / 8;
                }
                let packed = weight[weight_idx];

                for w in 0..8i32 {
                    let shift = w * 4;
                    let w_int4 = ((packed >> shift) & 0xF) as i8;
                    let w_fp32 = Q::dequant(w_int4, raw_zero, scale);

                    // Load activation (BF16 → f32)
                    let a_val = f32::from_bits((input[row as usize * k_usize + kg as usize + kk as usize + w as usize] as u32) << 16);

                    // Multiply and accumulate
                    acc += w_fp32 * a_val;
                }
            }
        }

        // Write output in BF16
        unsafe {
            *output.get_unchecked_mut(row as usize * n_usize + col as usize) = f32_to_bf16(acc);
        }
    }

    /// INT4 GEMM kernel for AutoRound format (zero offset +1).
    #[kernel]
    pub fn int4_gemm_auto_round(
        mut output: DisjointSlice<u16>,
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        m: u32, n: u32, k: u32,
        group_size: u32, transposed: u32,
    ) {
        int4_gemm_inner::<AutoRound>(
            &mut output, weight, scales, zeros, input,
            m as i32, n as i32, k as i32,
            group_size as i32, transposed as i32,
        );
    }

    /// INT4 GEMM kernel for GGUF format (no zero offset).
    #[kernel]
    pub fn int4_gemm_gguf(
        mut output: DisjointSlice<u16>,
        weight: &[u32],
        scales: &[u16],
        zeros: &[u32],
        input: &[u16],
        m: u32, n: u32, k: u32,
        group_size: u32, transposed: u32,
    ) {
        int4_gemm_inner::<Gguf>(
            &mut output, weight, scales, zeros, input,
            m as i32, n as i32, k as i32,
            group_size as i32, transposed as i32,
        );
    }

    // ─── Paged attention decode with KvCacheFormat trait ─────────────

    /// Trait for reading K/V values from the KV cache. Each format (BF16, FP8)
    /// implements the read method with its specific dequantization.
    pub trait KvCacheFormat {
        /// Read a value from the page pool at the given offset.
        /// Returns the dequantized f32 value.
        fn read_kv(pool: &[u16], offset: usize) -> f32;
    }

    /// KV cache stored as BF16 (default).
    pub struct KvBf16;
    impl KvCacheFormat for KvBf16 {
        #[inline(always)]
        fn read_kv(pool: &[u16], offset: usize) -> f32 {
            f32::from_bits((pool[offset] as u32) << 16)
        }
    }



    /// Paged attention decode kernel (BF16 KV cache).
    /// One block per KV head, supports GQA via q_per_kv query heads per KV head.
    #[kernel]
    #[launch_bounds(256)]
    pub fn infers_paged_attention_decode_bf16(
        q: &[u16],
        page_pool: &[u16],
        block_table: &[i32],
        num_pages: u32,
        num_cached_tokens: u32,
        head_dim: u32,
        num_kv_heads: u32,
        num_query_heads: u32,
        page_size: u32,
        kv_dim: u32,
        mut output: DisjointSlice<u16>,
    ) {
        use cuda_device::tcgen05::f32_to_bf16;

        let kv_head_idx = thread::blockIdx_x() as usize;
        if kv_head_idx >= num_kv_heads as usize {
            return;
        }

        let q_per_kv = (num_query_heads / num_kv_heads) as usize;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let page_stride = 2 * (page_size as usize) * (kv_dim as usize);
        let scale = 1.0f32 / dev_sqrtf(head_dim as f32);

        // Dynamic shared memory: 3 * bdim f32s
        let smem = DynamicSharedArray::<f32>::get();

        // Each block handles all query heads that share this KV head
        for local_q in 0..q_per_kv {
            let q_idx = kv_head_idx * q_per_kv + local_q;

            // ================================================================
            // Load Q_{q_idx} into shared memory [0..bdim)
            // ================================================================
            if tid < head_dim as usize {
                unsafe {
                    *smem.add(tid) = f32::from_bits((q[q_idx * (head_dim as usize) + tid] as u32) << 16);
                }
            }
            cuda_device::sync_threads();

            // ================================================================
            // Phase 1: Compute attention scores with per-thread online softmax
            // ================================================================
            let mut local_max: f32 = f32::NEG_INFINITY;
            let mut local_sum: f32 = 0.0;

            for token_pos in (tid as usize..num_cached_tokens as usize).step_by(bdim) {
                let logical_page = token_pos / page_size as usize;
                let token_in_page = token_pos % page_size as usize;
                let physical_page = block_table[logical_page] as usize;

                let mut dot: f32 = 0.0;
                for d in 0..head_dim as usize {
                    let q_v = unsafe { *smem.add(d) };
                    let k_off = physical_page * page_stride
                        + token_in_page * (kv_dim as usize)
                        + kv_head_idx * (head_dim as usize)
                        + d;
                    let k_v = KvBf16::read_kv(page_pool, k_off);
                    dot += q_v * k_v;
                }
                dot *= scale;

                let new_max = local_max.max(dot);
                let correction = libm::expf(local_max - new_max);
                local_sum = local_sum * correction + libm::expf(dot - new_max);
                local_max = new_max;
            }

            // --- Block reduction: global max ---
            unsafe { *smem.add(bdim + tid) = local_max; }
            cuda_device::sync_threads();

            let mut s = bdim / 2;
            while s > 0 {
                if tid < s {
                    unsafe {
                        let a = *smem.add(bdim + tid);
                        let b = *smem.add(bdim + tid + s);
                        *smem.add(bdim + tid) = a.max(b);
                    }
                }
                cuda_device::sync_threads();
                s >>= 1;
            }
            let global_max = unsafe { *smem.add(bdim) };

            // --- Adjust per-thread sums to global max, then reduce ---
            let adjusted_sum = local_sum * libm::expf(local_max - global_max);
            unsafe { *smem.add(2 * bdim + tid) = adjusted_sum; }
            cuda_device::sync_threads();

            s = bdim / 2;
            while s > 0 {
                if tid < s {
                    unsafe {
                        let a = *smem.add(2 * bdim + tid);
                        let b = *smem.add(2 * bdim + tid + s);
                        *smem.add(2 * bdim + tid) = a + b;
                    }
                }
                cuda_device::sync_threads();
                s >>= 1;
            }
            let global_sum = unsafe { *smem.add(2 * bdim) };

            // ================================================================
            // Phase 2: Compute weighted V accumulation
            // ================================================================
            let inv_sum = if global_sum > 0.0f32 { 1.0 / global_sum } else { 0.0 };

            if tid < head_dim as usize {
                let mut out_val: f32 = 0.0;
                for token_pos in 0..num_cached_tokens as usize {
                    let logical_page = token_pos / page_size as usize;
                    let token_in_page = token_pos % page_size as usize;
                    let physical_page = block_table[logical_page] as usize;

                    let mut dot: f32 = 0.0;
                    for d in 0..head_dim as usize {
                        let q_v = unsafe { *smem.add(d) };
                        let k_off = physical_page * page_stride
                            + token_in_page * (kv_dim as usize)
                            + kv_head_idx * (head_dim as usize)
                            + d;
                        let k_v = KvBf16::read_kv(page_pool, k_off);
                        dot += q_v * k_v;
                    }
                    dot *= scale;

                    let weight = libm::expf(dot - global_max) * inv_sum;
                    let v_off = physical_page * page_stride
                        + (page_size as usize) * (kv_dim as usize)
                        + token_in_page * (kv_dim as usize)
                        + kv_head_idx * (head_dim as usize)
                        + tid;
                    let v_val = KvBf16::read_kv(page_pool, v_off);
                    out_val += weight * v_val;
                }
                unsafe {
                    *output.get_unchecked_mut(q_idx * (head_dim as usize) + tid) = f32_to_bf16(out_val);
                }
            }

            cuda_device::sync_threads();
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

    // ─── GDN (Gated DeltaNet) kernels ─────────────

    /// GDN recurrent step: single-token decode kernel.
    /// One thread per (head, v_dim) element. No shared memory.
    #[kernel]
    #[launch_bounds(256)]
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
        let decay_rate_h = libm::expf(a_log[h]);
        let a_val = f32::from_bits((a_proj[h] as u32) << 16);
        let sp_val = a_val + dt_bias[h];

        let softplus_val: f32;
        if sp_val > 20.0f32 {
            softplus_val = sp_val;
        } else if sp_val < -20.0f32 {
            softplus_val = 0.0f32;
        } else {
            softplus_val = libm::logf(1.0f32 + libm::expf(sp_val));
        }

        let g_val = -decay_rate_h * softplus_val;
        let decay = libm::expf(g_val);

        // beta[h] = sigmoid(b_proj[h])
        let b_val = f32::from_bits((b_proj[h] as u32) << 16);
        let beta_val = 1.0f32 / (1.0f32 + libm::expf(-b_val));

        // L2-normalize key and query
        let mut k_l2_sq = 0.0f32;
        let mut q_l2_sq = 0.0f32;
        for k in 0..K {
            let kv = f32::from_bits((key[h * K + k] as u32) << 16);
            let qv = f32::from_bits((query[h * K + k] as u32) << 16);
            k_l2_sq += kv * kv;
            q_l2_sq += qv * qv;
        }

        let eps = 1e-6f32;
        let k_rcp = 1.0f32 / (k_l2_sq + eps).sqrt();
        let q_rcp = 1.0f32 / (q_l2_sq + eps).sqrt();

        let state_base = h * K * V + v;

        // Step 1: State decay
        for k in 0..K {
            state[state_base + k * V] *= decay;
        }

        // Step 2: kv_mem = sum_k S[h][k][v] * key_normed[h][k]
        let mut kv_mem = 0.0f32;
        for k in 0..K {
            let s_val = state[state_base + k * V];
            let k_val = f32::from_bits((key[h * K + k] as u32) << 16) * k_rcp;
            kv_mem += s_val * k_val;
        }

        // Step 3: delta = beta * (value - kv_mem)
        let v_val = f32::from_bits((value[h * V + v] as u32) << 16);
        let delta = beta_val * (v_val - kv_mem);

        // Step 4: State update
        for k in 0..K {
            let k_val = f32::from_bits((key[h * K + k] as u32) << 16) * k_rcp;
            state[state_base + k * V] += k_val * delta;
        }

        // Step 5: Output
        let mut y_val = 0.0f32;
        for k in 0..K {
            let s_val = state[state_base + k * V];
            let q_val = f32::from_bits((query[h * K + k] as u32) << 16) * q_rcp * rcp_sqrt_k;
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
        let decay = 1.0f32 / (1.0f32 + libm::expf(-a_val));
        let bias_val = f32::from_bits((dt_bias[head] as u32) << 16);

        // delta = softplus(dt_proj + dt_bias)
        let dt_val = f32::from_bits((dt_proj[tid] as u32) << 16) + bias_val;
        let delta: f32;
        if dt_val > 2.0f32 {
            delta = dt_val;
        } else if dt_val < -20.0f32 {
            delta = 0.0f32;
        } else {
            delta = libm::logf(1.0f32 + libm::expf(dt_val));
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
            silu_z = z_val / (1.0f32 + libm::expf(-z_val));
        } else {
            let exp_z = libm::expf(z_val);
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
        let decay_rate_h = libm::expf(a_log[h]);
        let a_val = f32::from_bits((a_proj[h] as u32) << 16);
        let sp_val = a_val + dt_bias[h];

        let softplus_val: f32;
        if sp_val > 20.0f32 {
            softplus_val = sp_val;
        } else if sp_val < -20.0f32 {
            softplus_val = 0.0f32;
        } else {
            softplus_val = libm::logf(1.0f32 + libm::expf(sp_val));
        }

        let g_val = -decay_rate_h * softplus_val;
        let b_val = f32::from_bits((b_proj[h] as u32) << 16);
        let beta_val = 1.0f32 / (1.0f32 + libm::expf(-b_val));
        let decay = libm::expf(g_val);

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

        let decay_rate_h = libm::expf(a_log[h]);

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
                softplus_val = libm::logf(1.0f32 + libm::expf(sp_val));
            }

            let g_val = -decay_rate_h * softplus_val;
            let b_val = f32::from_bits((b_proj[t * H + h] as u32) << 16);
            let beta_val = 1.0f32 / (1.0f32 + libm::expf(-b_val));
            let decay = libm::expf(g_val);

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
            let q_rcp = 1.0f32 / (q_l2_sq + 1e-6f32).sqrt();

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
                let q_val = f32::from_bits((query[t * H * K + h * K + k] as u32) << 16) * q_rcp * rcp_sqrt_k;
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
        let decay_rate = libm::expf(a_log[h]);            // A = exp(A_log[h])

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
                        softplus = libm::logf(1.0f32 + libm::expf(sp_val));
                    }

                    let g_val = -decay_rate * softplus;

                    // beta = sigmoid(b_proj[seq_pos][h])
                    let b_val = f32::from_bits((b_proj[seq_pos * num_heads as usize + h] as u32) << 16);
                    let beta_v = 1.0f32 / (1.0f32 + libm::expf(-b_val));

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
                            *k_normed.add(idx) *= rcp_norm;
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
                        attn_val = -sum * libm::expf(g_diff);
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
                    let q_rcp = 1.0f32 / (q_l2_sq + 1e-6f32).sqrt() * rcp_sqrt_k;
                    let exp_g_row = unsafe { libm::expf(*g_cs.add(row)) };
                    // ── attn_inter[row][col_v] = (q_normed * exp(g_cs) @ S) ──
                    let mut attn_inter_val = 0.0f32;
                    for d in 0..K {
                        let q_scl = q_reg[d] * q_rcp * exp_g_row;
                        attn_inter_val += q_scl * state[state_base + d * V + col_v];
                    }

                    // ── Compute output contribution from attn_qk ─────
                    let mut output_from_qk = 0.0f32;

                    for j in 0..=row {
                        // qk_dot: query-key dot product with decay
                        let mut qk_dot_j = 0.0f32;
                        for d in 0..K {
                            unsafe {
                                qk_dot_j += q_reg[d] * q_rcp * *k_normed.add(j * K + d);
                            }
                        }
                        let g_diff = unsafe { *g_cs.add(row) - *g_cs.add(j) };
                        let attn_qk_val = qk_dot_j * libm::expf(g_diff);
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
                                        * libm::expf(*g_cs.add(ii));
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
                    let exp_g_last = unsafe { libm::expf(*g_cs.add(actual_len - 1)) };
                    // S[d][col_v] *= exp(g_cs[-1])
                    let mut s_val = state[state_base + d * V + col_v] * exp_g_last;

                    // Add: sum_j exp_diff[j] * k_normed[j][d] * v_nc[j][col_v]
                    for j in 0..actual_len {
                        let exp_diff_j = unsafe { libm::expf(*g_cs.add(actual_len - 1) - *g_cs.add(j)) };
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
                                        * libm::expf(*g_cs.add(ii));
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
