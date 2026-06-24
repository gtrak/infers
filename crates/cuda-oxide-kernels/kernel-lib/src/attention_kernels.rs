//! Attention kernels — paged KV write/read, paged attention decode, RoPE.

use cuda_device::{cuda_module, kernel, launch_bounds, thread, DisjointSlice, DynamicSharedArray};
use super::shared::*;

#[cuda_module]
pub mod attention {
    use super::*;

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
}
