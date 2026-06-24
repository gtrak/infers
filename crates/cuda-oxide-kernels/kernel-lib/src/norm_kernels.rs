//! Norm kernels — RMSNorm, gated RMSNorm, L2 norm.

use cuda_device::{cuda_module, kernel, launch_bounds, thread, DisjointSlice, DynamicSharedArray};
use super::shared::*;

#[cuda_module]
pub mod norm {
    use super::*;

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
        let block_dim_x = thread::blockDim_x() as usize;
        let mut s = block_dim_x / 2;
        while s > 0 {
            if tid < s && tid + s < block_dim_x {
                unsafe {
                    *smem.add(tid) = *smem.add(tid) + *smem.add(tid + s);
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
}
