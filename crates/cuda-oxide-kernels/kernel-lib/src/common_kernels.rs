//! Common utility kernels — add, embedding gather, argmax, softmax, KV cache write, sanitize_nan.

use cuda_device::{cuda_module, kernel, launch_bounds, thread, DisjointSlice, DynamicSharedArray, SharedArray};
use super::shared::*;

#[cuda_module]
pub mod common {
    use super::*;

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
        let block_dim_x = thread::blockDim_x() as usize;
        let mut s = block_dim_x / 2;
        while s > 0 {
            if tid < s && tid + s < block_dim_x {
                unsafe {
                    let own = *smem.add(tid);
                    let other = *smem.add(tid + s);
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
                local_sum += fast_expf(val - row_max);
            }
        }

        unsafe { *smem.add(tid) = local_sum; }
        cuda_device::sync_threads();

        // Halving sum reduction
        let mut s2 = block_dim_x / 2;
        while s2 > 0 {
            if tid < s2 && tid + s2 < block_dim_x {
                unsafe {
                    *smem.add(tid) = *smem.add(tid) + *smem.add(tid + s2);
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
                let result = fast_expf(val - row_max) / row_sum;
                unsafe { *output.get_unchecked_mut(out_idx) = f32_to_bf16(result); }
            } else {
                // Masked position → zero
                unsafe { *output.get_unchecked_mut(out_idx) = 0u16; }
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

    /// Replace NaN values in a bf16 buffer with 0.0.
    #[kernel]
    pub fn sanitize_nan_bf16(buf: &mut [u16], len: u32) {
        let idx = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if idx >= len as usize { return; }
        let val = unsafe { *buf.get_unchecked(idx) };
        let f = f32::from_bits((val as u32) << 16);
        if f.is_nan() {
            unsafe { *buf.get_unchecked_mut(idx) = 0u16; }
        }
    }
}
