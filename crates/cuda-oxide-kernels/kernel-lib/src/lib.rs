//! Kernel library for infers — cuda-oxide PTX kernels.
//!
//! Kernels are compiled to PTX by rustc-codegen-cuda via `#[cuda_module]`.
//! Host code loads the module with `kernels::load(&ctx)`.

#![feature(f16)]

use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, thread};

/// All device kernels — compiled to PTX by cuda-oxide.
#[cuda_module]
pub mod kernels {
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
}
