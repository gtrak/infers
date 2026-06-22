//! Bridge module for loading cuda-oxide compiled kernels and launching them
//! using cudarc buffers and streams.
//!
//! Loads a pre-compiled `.cubin` file via `cuda-core`'s `CudaContext::load_module_from_file`,
//! resolves kernel function handles, and provides type-safe launch wrappers that accept
//! cudarc `CudaSlice<T>` buffers.

use std::collections::HashMap;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaModule, CudaFunction, LaunchConfig};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

/// Pre-loaded cuda-oxide kernels from a compiled cubin file.
pub struct OxideKernels {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    functions: HashMap<&'static str, CudaFunction>,
}

impl OxideKernels {
    /// Load all kernels from the given cubin file path.
    pub fn new(cubin_path: &str) -> anyhow::Result<Self> {
        // Create cuda-oxide context on device 0 (primary context shared with cudarc)
        let ctx = CudaContext::new(0)?;
        // Bind the context to the current thread before loading
        ctx.bind_to_thread()?;

        // Load the pre-compiled cubin
        let module = ctx.load_module_from_file(cubin_path)?;
        // Resolve all kernel function handles
        let mut functions: HashMap<&'static str, CudaFunction> = HashMap::new();
        for name in KERNEL_NAMES {
            functions.insert(name, module.load_function(name)?);
        }

        Ok(Self { ctx, module, functions })
    }

    /// Internal helper: push a slice argument (ptr + len) for the kernel ABI.
    fn push_slice_arg(
        args: &mut Vec<*mut std::ffi::c_void>,
        ptr: &mut cuda_core::sys::CUdeviceptr,
        len: &mut u64,
    ) {
        args.push(ptr as *mut _ as *mut std::ffi::c_void);
        args.push(len as *mut _ as *mut std::ffi::c_void);
    }

    /// Internal helper: push a scalar argument for the kernel ABI.
    fn push_scalar_arg<T: Copy>(args: &mut Vec<*mut std::ffi::c_void>, val: &mut T) {
        if std::mem::size_of::<T>() > 0 {
            args.push(val as *mut _ as *mut std::ffi::c_void);
        }
    }

    /// Internal raw kernel launch using cuda-oxide's low-level API.
    fn raw_launch(
        &self,
        func_name: &str,
        stream: &Arc<CudaStream>,
        config: LaunchConfig,
        args: &mut Vec<*mut std::ffi::c_void>,
    ) -> anyhow::Result<()> {
        self.ctx.bind_to_thread()?;

        let func = self.functions.get(func_name)
            .ok_or_else(|| anyhow::anyhow!("kernel '{}' not found", func_name))?;

        // Convert cudarc CUstream to cuda-oxide CUstream
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void as cuda_core::sys::CUstream;

        unsafe {
            cuda_core::launch_kernel(
                func.cu_function(),
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                cu_stream,
                args,
            ).map_err(|e| anyhow::anyhow!("kernel launch '{}' failed: {}", func_name, e))?;
        }

        Ok(())
    }

    /// Launch the `infers_add_bf16` kernel: element-wise addition of two bf16 buffers.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_add_bf16(a: &[u16], b: &[u16], mut out: DisjointSlice<u16>, total_elements: u32)
    /// ```
    pub fn launch_add_bf16(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaSlice<half::bf16>,
        b: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
    ) -> anyhow::Result<()> {
        let n = a.len() as u32;

        // Compute lengths first (immutable borrows) before getting device pointers
        let a_len_val = a.len() as u64;
        let b_len_val = b.len() as u64;
        let out_len_val = output.len() as u64;

        // Get device pointers from cudarc buffers (read-only for inputs)
        let (a_ptr, a_guard) = a.device_ptr(stream);
        let (b_ptr, b_guard) = b.device_ptr(stream);
        // Mutable access for output buffer
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        // Keep guards alive until after the launch call
        let _guards = (a_guard, b_guard, out_guard);

        // Cast cudarc CUdeviceptr to cuda-oxide CUdeviceptr
        let mut a_ptr = a_ptr as cuda_core::sys::CUdeviceptr;
        let mut b_ptr = b_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut a_len = a_len_val;
        let mut b_len = b_len_val;
        let mut out_len = out_len_val;
        let mut total = n;

        // Pack arguments according to the cuda-oxide kernel ABI:
        // &mut CUdeviceptr + &mut u64 for each slice, &mut T for scalars
        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut a_ptr, &mut a_len);   // a: &[u16]
        Self::push_slice_arg(&mut args, &mut b_ptr, &mut b_len);   // b: &[u16]
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len); // out: DisjointSlice<u16>
        Self::push_scalar_arg(&mut args, &mut total);              // total_elements: u32

        let config = LaunchConfig::for_num_elems(n);
        self.raw_launch("infers_add_bf16", stream, config, &mut args)
    }

    /// Access the underlying cuda-oxide context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Access the loaded module.
    pub fn module(&self) -> &Arc<CudaModule> {
        &self.module
    }

    /// Get a resolved kernel function by name.
    pub fn get_function(&self, name: &str) -> Option<&CudaFunction> {
        self.functions.get(name)
    }
}

/// All 28 kernel names compiled into the oxide_kernels.cubin file.
const KERNEL_NAMES: [&str; 28] = [
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
    "int4_gemm_gguf",
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

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext as CudarcCtx;

    #[test]
    fn test_add_bf16_bridge() {
        // 1. Create cudarc context + stream
        let cudarc_ctx = CudarcCtx::new(0).unwrap();
        let stream = cudarc_ctx.default_stream();

        // 2. Load oxide kernels from cubin
        let cubin_path = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/compiled/oxide_kernels.cubin");
        let oxide = OxideKernels::new(cubin_path).unwrap();

        // 3. Alloc + fill test data (use values exactly representable in bf16)
        let n = 128;
        let a_data: Vec<half::bf16> = (0..n).map(|i| half::bf16::from_f32(i as f32)).collect();
        let b_data: Vec<half::bf16> = (0..n).map(|i| half::bf16::from_f32((i + 1) as f32)).collect();
        let a_gpu = stream.clone_htod(&a_data).unwrap();
        let b_gpu = stream.clone_htod(&b_data).unwrap();
        let mut out_gpu = stream.alloc_zeros::<half::bf16>(n).unwrap();

        // 4. Launch via bridge
        oxide.launch_add_bf16(&stream, &a_gpu, &b_gpu, &mut out_gpu).unwrap();

        // 5. Read back + verify
        let result: Vec<half::bf16> = stream.clone_dtoh(&out_gpu).unwrap();
        // Expected: a[i] + b[i] = i + (i+1) = 2*i + 1, converted through bf16
        let expected: Vec<half::bf16> = a_data.iter()
            .zip(b_data.iter())
            .map(|(&a, &b)| half::bf16::from_f32(a.to_f32() + b.to_f32()))
            .collect();

        // Allow for bf16 precision: compare with tolerance
        for (i, (&r, &e)) in result.iter().zip(expected.iter()).enumerate() {
            let diff = (r.to_f32() - e.to_f32()).abs();
            assert!(diff < 0.01, "Mismatch at {}: got {}, expected {}", i, r, e);
        }
    }
}
