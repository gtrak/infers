//! Activation kernels — SiLU, SiLU-GLU, attention output gate, conv1d depthwise silu.

use cuda_device::{cuda_module, kernel, launch_bounds, thread, DisjointSlice};
use super::shared::*;

#[cuda_module]
pub mod activation {
    use super::*;

    /// SiLU activation: output[i] = x[i] * sigmoid(x[i])
    /// where sigmoid(v) = 1.0 / (1.0 + exp(-v))
    // @lat: [[kernel-optimization#Kernel Optimization Experiments#Experiment Queue#EXP-005: SiLU vectorized loads]]
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

        let vec_total = total / 4 * 4;
        let vec_stride = stride as usize * 4;
        for base in (tid as usize * 4..vec_total).step_by(vec_stride) {
            let x4: [u16; 4] = unsafe { *(x.as_ptr().add(base) as *const [u16; 4]) };
            let mut o4 = [0u16; 4];
            for j in 0..4 {
                let val = f32::from_bits((x4[j] as u32) << 16);
                let sigmoid = 1.0 / (1.0 + fast_expf(-val));
                o4[j] = f32_to_bf16(val * sigmoid);
            }
            unsafe {
                let ptr = output.get_unchecked_mut(base) as *mut u16 as *mut [u16; 4];
                *ptr = o4;
            }
        }
        for i in (vec_total + tid as usize..total).step_by(stride as usize) {
            let val = f32::from_bits((x[i] as u32) << 16);
            let sigmoid = 1.0 / (1.0 + fast_expf(-val));
            unsafe { *output.get_unchecked_mut(i) = f32_to_bf16(val * sigmoid); }
        }
    }

    /// SiLU Gated Linear Unit: output[i] = x[i] * gate[i] * sigmoid(gate[i])
    // @lat: [[kernel-optimization#Kernel Optimization Experiments#Experiment Queue#EXP-005: SiLU vectorized loads]]
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

        let vec_total = total / 4 * 4;
        let vec_stride = stride as usize * 4;
        for base in (tid as usize * 4..vec_total).step_by(vec_stride) {
            let x4: [u16; 4] = unsafe { *(x.as_ptr().add(base) as *const [u16; 4]) };
            let g4: [u16; 4] = unsafe { *(gate.as_ptr().add(base) as *const [u16; 4]) };
            let mut o4 = [0u16; 4];
            for j in 0..4 {
                let x_val = f32::from_bits((x4[j] as u32) << 16);
                let g_val = f32::from_bits((g4[j] as u32) << 16);
                let sigmoid_g = 1.0 / (1.0 + fast_expf(-g_val));
                o4[j] = f32_to_bf16(x_val * g_val * sigmoid_g);
            }
            unsafe {
                let ptr = output.get_unchecked_mut(base) as *mut u16 as *mut [u16; 4];
                *ptr = o4;
            }
        }
        for i in (vec_total + tid as usize..total).step_by(stride as usize) {
            let x_val = f32::from_bits((x[i] as u32) << 16);
            let g_val = f32::from_bits((gate[i] as u32) << 16);
            let sigmoid_g = 1.0 / (1.0 + fast_expf(-g_val));
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
            let sigmoid_g = 1.0 / (1.0 + fast_expf(-g_val));
            unsafe { *output.get_unchecked_mut(i) = f32_to_bf16(x_val * sigmoid_g); }
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
            let silu = sum / (1.0 + fast_expf(-sum));
            unsafe { *output.get_unchecked_mut(i) = f32_to_bf16(silu); }
        }
    }
}
