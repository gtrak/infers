//! Bridge module for loading cuda-oxide compiled kernels and launching them
//! using cudarc buffers and streams.
//!
//! Loads a pre-compiled `.cubin` file via `cuda-core`'s `CudaContext::load_module_from_file`,
//! resolves kernel function handles, and provides type-safe launch wrappers that accept
//! cudarc `CudaSlice<T>` buffers.

use std::collections::HashMap;
use std::sync::Arc;

use cuda_core::sys;
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
    pub fn new(ordinal: usize, cubin_path: &str) -> anyhow::Result<Self> {
        // Create cuda-oxide context on the specified device (primary context shared with cudarc)
        let ctx = CudaContext::new(ordinal)?;
        // Bind the context to the current thread before loading
        ctx.bind_to_thread()?;

        // Load the pre-compiled cubin
        let module = ctx.load_module_from_file(cubin_path)?;
        // Resolve all kernel function handles
        let mut functions: HashMap<&'static str, CudaFunction> = HashMap::new();
        for name in KERNEL_NAMES {
            functions.insert(name, module.load_function(name)?);
        }

        // Set max dynamic shared memory for chunked GDN kernel (~80KB needed, exceeds 48KB default)
        if let Some(chunked_gdn_func) = functions.get("infers_gdn_chunked_gated_delta_prefill_bf16") {
            let raw_func: sys::CUfunction = unsafe { chunked_gdn_func.cu_function() };
            let result = unsafe {
                sys::cuFuncSetAttribute(
                    raw_func,
                    8, // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                    100_000, // match the .cu source's maxdynamicsharedmemsize(100000)
                )
            };
            if result != 0 {
                anyhow::bail!(
                    "cuFuncSetAttribute failed for infers_gdn_chunked_gated_delta_prefill_bf16 (error code {})",
                    result
                );
            }
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

    /// Launch the `infers_embedding_gather_bf16` kernel: gather embeddings by token ids.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_embedding_gather_bf16(weight: &[u16], token_ids: &[i32], mut output: DisjointSlice<u16>, seq_len: u32, hidden_size: u32)
    /// ```
    pub fn launch_embedding_gather_bf16(
        &self,
        stream: &Arc<CudaStream>,
        weight: &CudaSlice<half::bf16>,
        token_ids: &CudaSlice<i32>,
        output: &mut CudaSlice<half::bf16>,
        seq_len: u32,
        hidden_size: u32,
    ) -> anyhow::Result<()> {
        let total = (seq_len as usize) * (hidden_size as usize);

        let w_len_val = weight.len() as u64;
        let t_len_val = token_ids.len() as u64;
        let out_len_val = output.len() as u64;

        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (t_ptr, t_guard) = token_ids.device_ptr(stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        let _guards = (w_guard, t_guard, out_guard);

        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut t_ptr = t_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_len = w_len_val;
        let mut t_len = t_len_val;
        let mut out_len = out_len_val;
        let mut seq_len_v = seq_len;
        let mut hidden_size_v = hidden_size;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);
        Self::push_slice_arg(&mut args, &mut t_ptr, &mut t_len);
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len);
        Self::push_scalar_arg(&mut args, &mut seq_len_v);
        Self::push_scalar_arg(&mut args, &mut hidden_size_v);

        let config = LaunchConfig::for_num_elems(total as u32);
        self.raw_launch("infers_embedding_gather_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_silu_bf16` kernel: SiLU activation.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_silu_bf16(x: &[u16], mut output: DisjointSlice<u16>, total_elements: u32)
    /// ```
    pub fn launch_silu_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        total: u32,
    ) -> anyhow::Result<()> {
        let x_len_val = x.len() as u64;
        let out_len_val = output.len() as u64;

        let (x_ptr, x_guard) = x.device_ptr(stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        let _guards = (x_guard, out_guard);

        let mut x_ptr = x_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut x_len = x_len_val;
        let mut out_len = out_len_val;
        let mut total_v = total;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut x_ptr, &mut x_len);
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len);
        Self::push_scalar_arg(&mut args, &mut total_v);

        let config = LaunchConfig::for_num_elems(total);
        self.raw_launch("infers_silu_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_silu_glu_bf16` kernel: SiLU Gated Linear Unit.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_silu_glu_bf16(x: &[u16], gate: &[u16], mut output: DisjointSlice<u16>, total_elements: u32)
    /// ```
    pub fn launch_silu_glu_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<half::bf16>,
        gate: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        total: u32,
    ) -> anyhow::Result<()> {
        let x_len_val = x.len() as u64;
        let g_len_val = gate.len() as u64;
        let out_len_val = output.len() as u64;

        let (x_ptr, x_guard) = x.device_ptr(stream);
        let (g_ptr, g_guard) = gate.device_ptr(stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        let _guards = (x_guard, g_guard, out_guard);

        let mut x_ptr = x_ptr as cuda_core::sys::CUdeviceptr;
        let mut g_ptr = g_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut x_len = x_len_val;
        let mut g_len = g_len_val;
        let mut out_len = out_len_val;
        let mut total_v = total;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut x_ptr, &mut x_len);
        Self::push_slice_arg(&mut args, &mut g_ptr, &mut g_len);
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len);
        Self::push_scalar_arg(&mut args, &mut total_v);

        let config = LaunchConfig::for_num_elems(total);
        self.raw_launch("infers_silu_glu_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_attn_output_gate_bf16` kernel: attention output gate.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_attn_output_gate_bf16(x: &[u16], gate: &[u16], mut output: DisjointSlice<u16>, total_elements: u32)
    /// ```
    pub fn launch_attn_output_gate_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<half::bf16>,
        gate: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        total: u32,
    ) -> anyhow::Result<()> {
        let x_len_val = x.len() as u64;
        let g_len_val = gate.len() as u64;
        let out_len_val = output.len() as u64;

        let (x_ptr, x_guard) = x.device_ptr(stream);
        let (g_ptr, g_guard) = gate.device_ptr(stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        let _guards = (x_guard, g_guard, out_guard);

        let mut x_ptr = x_ptr as cuda_core::sys::CUdeviceptr;
        let mut g_ptr = g_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut x_len = x_len_val;
        let mut g_len = g_len_val;
        let mut out_len = out_len_val;
        let mut total_v = total;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut x_ptr, &mut x_len);
        Self::push_slice_arg(&mut args, &mut g_ptr, &mut g_len);
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len);
        Self::push_scalar_arg(&mut args, &mut total_v);

        let config = LaunchConfig::for_num_elems(total);
        self.raw_launch("infers_attn_output_gate_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_argmax_bf16` kernel: argmax per row using shared memory reduction.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_argmax_bf16(logits: &[u16], mut output: DisjointSlice<i32>, _batch_size: u32, vocab_size: u32)
    /// ```
    pub fn launch_argmax_bf16(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<i32>,
        batch_size: u32,
        vocab_size: u32,
    ) -> anyhow::Result<()> {
        let l_len_val = logits.len() as u64;
        let out_len_val = output.len() as u64;

        let (l_ptr, l_guard) = logits.device_ptr(stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        let _guards = (l_guard, out_guard);

        let mut l_ptr = l_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut l_len = l_len_val;
        let mut out_len = out_len_val;
        let mut batch_size_v = batch_size;
        let mut vocab_size_v = vocab_size;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut l_ptr, &mut l_len);
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len);
        Self::push_scalar_arg(&mut args, &mut batch_size_v);
        Self::push_scalar_arg(&mut args, &mut vocab_size_v);

        // 2 static shared arrays of 256 f32s each = 2048 bytes
        let config = LaunchConfig {
            grid_dim: (batch_size, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4 * 2,
        };
        self.raw_launch("infers_argmax_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_kv_cache_write_bf16` kernel: scattered KV cache write.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_kv_cache_write_bf16(k: &[u16], v: &[u16], mut kv_cache: DisjointSlice<u16>, positions: &[i32], seq_len: u32, head_dim: u32, max_seq_len: u32)
    /// ```
    pub fn launch_kv_cache_write_bf16(
        &self,
        stream: &Arc<CudaStream>,
        k: &CudaSlice<half::bf16>,
        v: &CudaSlice<half::bf16>,
        kv_cache: &mut CudaSlice<half::bf16>,
        positions: &CudaSlice<i32>,
        seq_len: u32,
        head_dim: u32,
        max_seq_len: u32,
    ) -> anyhow::Result<()> {
        let total = (seq_len as usize) * (head_dim as usize);

        let k_len_val = k.len() as u64;
        let v_len_val = v.len() as u64;
        let kv_len_val = kv_cache.len() as u64;
        let p_len_val = positions.len() as u64;

        let (k_ptr, k_guard) = k.device_ptr(stream);
        let (v_ptr, v_guard) = v.device_ptr(stream);
        let (kv_ptr, kv_guard) = kv_cache.device_ptr_mut(stream);
        let (p_ptr, p_guard) = positions.device_ptr(stream);

        let _guards = (k_guard, v_guard, kv_guard, p_guard);

        let mut k_ptr = k_ptr as cuda_core::sys::CUdeviceptr;
        let mut v_ptr = v_ptr as cuda_core::sys::CUdeviceptr;
        let mut kv_ptr = kv_ptr as cuda_core::sys::CUdeviceptr;
        let mut p_ptr = p_ptr as cuda_core::sys::CUdeviceptr;
        let mut k_len = k_len_val;
        let mut v_len = v_len_val;
        let mut kv_len = kv_len_val;
        let mut p_len = p_len_val;
        let mut seq_len_v = seq_len;
        let mut head_dim_v = head_dim;
        let mut max_seq_len_v = max_seq_len;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut k_ptr, &mut k_len);
        Self::push_slice_arg(&mut args, &mut v_ptr, &mut v_len);
        Self::push_slice_arg(&mut args, &mut kv_ptr, &mut kv_len);
        Self::push_slice_arg(&mut args, &mut p_ptr, &mut p_len);
        Self::push_scalar_arg(&mut args, &mut seq_len_v);
        Self::push_scalar_arg(&mut args, &mut head_dim_v);
        Self::push_scalar_arg(&mut args, &mut max_seq_len_v);

        let config = LaunchConfig::for_num_elems(total as u32);
        self.raw_launch("infers_kv_cache_write_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_rmsnorm_bf16` kernel: RMS normalization.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_rmsnorm_bf16(x: &[u16], weight: &[u16], mut output: DisjointSlice<u16>, hidden: u32, eps: f32)
    /// ```
    pub fn launch_rmsnorm_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<half::bf16>,
        weight: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        hidden: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        let num_rows = x.len() / hidden as usize;
        let block_size = (hidden.min(256)) as u32;

        let x_len_val = x.len() as u64;
        let w_len_val = weight.len() as u64;
        let out_len_val = output.len() as u64;

        let (x_ptr, x_guard) = x.device_ptr(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        let _guards = (x_guard, w_guard, out_guard);

        let mut x_ptr = x_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut x_len = x_len_val;
        let mut w_len = w_len_val;
        let mut out_len = out_len_val;
        let mut hidden_v = hidden;
        let mut eps_v = eps;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut x_ptr, &mut x_len);
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len);
        Self::push_scalar_arg(&mut args, &mut hidden_v);
        Self::push_scalar_arg(&mut args, &mut eps_v);

        let config = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };
        self.raw_launch("infers_rmsnorm_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_rms_norm_gated_bf16` kernel: RMSNorm with SiLU gate.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_rms_norm_gated_bf16(input: &[u16], gate: &[u16], weight: &[u16], mut output: DisjointSlice<u16>, _n: u32, d: u32, eps: f32)
    /// ```
    pub fn launch_rms_norm_gated_bf16(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<half::bf16>,
        gate: &CudaSlice<half::bf16>,
        weight: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        n: u32,
        d: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        let block_size = (d.min(256)) as u32;

        let i_len_val = input.len() as u64;
        let g_len_val = gate.len() as u64;
        let w_len_val = weight.len() as u64;
        let out_len_val = output.len() as u64;

        let (i_ptr, i_guard) = input.device_ptr(stream);
        let (g_ptr, g_guard) = gate.device_ptr(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        let _guards = (i_guard, g_guard, w_guard, out_guard);

        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut g_ptr = g_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_len = i_len_val;
        let mut g_len = g_len_val;
        let mut w_len = w_len_val;
        let mut out_len = out_len_val;
        let mut n_v = n;
        let mut d_v = d;
        let mut eps_v = eps;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);
        Self::push_slice_arg(&mut args, &mut g_ptr, &mut g_len);
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len);
        Self::push_scalar_arg(&mut args, &mut n_v);
        Self::push_scalar_arg(&mut args, &mut d_v);
        Self::push_scalar_arg(&mut args, &mut eps_v);

        let config = LaunchConfig {
            grid_dim: (n, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };
        self.raw_launch("infers_rms_norm_gated_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_l2norm_bf16` kernel: L2 normalization.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_l2norm_bf16(input: &[u16], mut output: DisjointSlice<u16>, dim: u32, eps: f32)
    /// ```
    pub fn launch_l2norm_bf16(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        dim: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        let num_rows = input.len() as u32 / dim;
        let block_size = (dim.min(256)) as u32;

        let i_len_val = input.len() as u64;
        let out_len_val = output.len() as u64;

        let (i_ptr, i_guard) = input.device_ptr(stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        let _guards = (i_guard, out_guard);

        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_len = i_len_val;
        let mut out_len = out_len_val;
        let mut dim_v = dim;
        let mut eps_v = eps;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len);
        Self::push_scalar_arg(&mut args, &mut dim_v);
        Self::push_scalar_arg(&mut args, &mut eps_v);

        let config = LaunchConfig {
            grid_dim: (num_rows, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };
        self.raw_launch("infers_l2norm_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_softmax_bf16` kernel: softmax with optional causal mask.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_softmax_bf16(scores: &[u16], mut output: DisjointSlice<u16>, seq_len: u32, use_causal: u32)
    /// ```
    pub fn launch_softmax_bf16(
        &self,
        stream: &Arc<CudaStream>,
        scores: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        seq_len: u32,
        use_causal: u32,
    ) -> anyhow::Result<()> {
        let num_rows = scores.len() as u32 / seq_len;
        let block_size: u32 = 256;

        let s_len_val = scores.len() as u64;
        let out_len_val = output.len() as u64;

        let (s_ptr, s_guard) = scores.device_ptr(stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        let _guards = (s_guard, out_guard);

        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_len = s_len_val;
        let mut out_len = out_len_val;
        let mut seq_len_v = seq_len;
        let mut use_causal_v = use_causal;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len);
        Self::push_scalar_arg(&mut args, &mut seq_len_v);
        Self::push_scalar_arg(&mut args, &mut use_causal_v);

        let config = LaunchConfig {
            grid_dim: (num_rows, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };
        self.raw_launch("infers_softmax_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_conv1d_depthwise_silu_bf16` kernel: depthwise 1D convolution with SiLU.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_conv1d_depthwise_silu_bf16(input: &[u16], weight: &[u16], mut output: DisjointSlice<u16>, batch_size: u32, conv_dim: u32, seq_len: u32, kernel_size: u32)
    /// ```
    pub fn launch_conv1d_depthwise_silu_bf16(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<half::bf16>,
        weight: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        batch_size: u32,
        conv_dim: u32,
        seq_len: u32,
        kernel_size: u32,
    ) -> anyhow::Result<()> {
        let total = (batch_size as usize) * (seq_len as usize) * (conv_dim as usize);

        let i_len_val = input.len() as u64;
        let w_len_val = weight.len() as u64;
        let out_len_val = output.len() as u64;

        let (i_ptr, i_guard) = input.device_ptr(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (out_ptr, out_guard) = output.device_ptr_mut(stream);

        let _guards = (i_guard, w_guard, out_guard);

        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut out_ptr = out_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_len = i_len_val;
        let mut w_len = w_len_val;
        let mut out_len = out_len_val;
        let mut batch_size_v = batch_size;
        let mut conv_dim_v = conv_dim;
        let mut seq_len_v = seq_len;
        let mut kernel_size_v = kernel_size;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);
        Self::push_slice_arg(&mut args, &mut out_ptr, &mut out_len);
        Self::push_scalar_arg(&mut args, &mut batch_size_v);
        Self::push_scalar_arg(&mut args, &mut conv_dim_v);
        Self::push_scalar_arg(&mut args, &mut seq_len_v);
        Self::push_scalar_arg(&mut args, &mut kernel_size_v);

        let config = LaunchConfig::for_num_elems(total as u32);
        self.raw_launch("infers_conv1d_depthwise_silu_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_rope_bf16` kernel: rotary position embedding.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_rope_bf16(mut q: DisjointSlice<u16>, mut k_tensor: DisjointSlice<u16>, cos: &[f32], sin: &[f32], positions: &[i32], total_tokens: u32, num_heads: u32, head_dim: u32, rotary_dim: u32)
    /// ```
    pub fn launch_rope_bf16(
        &self,
        stream: &Arc<CudaStream>,
        q: &mut CudaSlice<half::bf16>,
        k: &mut CudaSlice<half::bf16>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        total_tokens: u32,
        num_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
    ) -> anyhow::Result<()> {
        let total = (total_tokens as usize) * (num_heads as usize) * (head_dim as usize);

        let q_len_val = q.len() as u64;
        let k_len_val = k.len() as u64;
        let cos_len_val = cos.len() as u64;
        let sin_len_val = sin.len() as u64;
        let p_len_val = positions.len() as u64;

        let (q_ptr, q_guard) = q.device_ptr_mut(stream);
        let (k_ptr, k_guard) = k.device_ptr_mut(stream);
        let (cos_ptr, cos_guard) = cos.device_ptr(stream);
        let (sin_ptr, sin_guard) = sin.device_ptr(stream);
        let (p_ptr, p_guard) = positions.device_ptr(stream);

        let _guards = (q_guard, k_guard, cos_guard, sin_guard, p_guard);

        let mut q_ptr = q_ptr as cuda_core::sys::CUdeviceptr;
        let mut k_ptr = k_ptr as cuda_core::sys::CUdeviceptr;
        let mut cos_ptr = cos_ptr as cuda_core::sys::CUdeviceptr;
        let mut sin_ptr = sin_ptr as cuda_core::sys::CUdeviceptr;
        let mut p_ptr = p_ptr as cuda_core::sys::CUdeviceptr;
        let mut q_len = q_len_val;
        let mut k_len = k_len_val;
        let mut cos_len = cos_len_val;
        let mut sin_len = sin_len_val;
        let mut p_len = p_len_val;
        let mut total_tokens_v = total_tokens;
        let mut num_heads_v = num_heads;
        let mut head_dim_v = head_dim;
        let mut rotary_dim_v = rotary_dim;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut q_ptr, &mut q_len);
        Self::push_slice_arg(&mut args, &mut k_ptr, &mut k_len);
        Self::push_slice_arg(&mut args, &mut cos_ptr, &mut cos_len);
        Self::push_slice_arg(&mut args, &mut sin_ptr, &mut sin_len);
        Self::push_slice_arg(&mut args, &mut p_ptr, &mut p_len);
        Self::push_scalar_arg(&mut args, &mut total_tokens_v);
        Self::push_scalar_arg(&mut args, &mut num_heads_v);
        Self::push_scalar_arg(&mut args, &mut head_dim_v);
        Self::push_scalar_arg(&mut args, &mut rotary_dim_v);

        let config = LaunchConfig::for_num_elems(total as u32);
        self.raw_launch("infers_rope_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_paged_kv_write_bf16` kernel: paged KV cache write.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_paged_kv_write_bf16(k: &[u16], v: &[u16], mut page_pool: DisjointSlice<u16>, block_table: &[i32], positions: &[i32], seq_len: u32, _head_dim: u32, page_size: u32, kv_dim: u32)
    /// ```
    pub fn launch_paged_kv_write_bf16(
        &self,
        stream: &Arc<CudaStream>,
        k: &CudaSlice<half::bf16>,
        v: &CudaSlice<half::bf16>,
        page_pool: &mut CudaSlice<half::bf16>,
        block_table: &CudaSlice<i32>,
        positions: &CudaSlice<i32>,
        seq_len: u32,
        head_dim: u32,
        page_size: u32,
        kv_dim: u32,
    ) -> anyhow::Result<()> {
        let total = (seq_len as usize) * (kv_dim as usize);

        let k_len_val = k.len() as u64;
        let v_len_val = v.len() as u64;
        let pp_len_val = page_pool.len() as u64;
        let bt_len_val = block_table.len() as u64;
        let p_len_val = positions.len() as u64;

        let (k_ptr, k_guard) = k.device_ptr(stream);
        let (v_ptr, v_guard) = v.device_ptr(stream);
        let (pp_ptr, pp_guard) = page_pool.device_ptr_mut(stream);
        let (bt_ptr, bt_guard) = block_table.device_ptr(stream);
        let (p_ptr, p_guard) = positions.device_ptr(stream);

        let _guards = (k_guard, v_guard, pp_guard, bt_guard, p_guard);

        let mut k_ptr = k_ptr as cuda_core::sys::CUdeviceptr;
        let mut v_ptr = v_ptr as cuda_core::sys::CUdeviceptr;
        let mut pp_ptr = pp_ptr as cuda_core::sys::CUdeviceptr;
        let mut bt_ptr = bt_ptr as cuda_core::sys::CUdeviceptr;
        let mut p_ptr = p_ptr as cuda_core::sys::CUdeviceptr;
        let mut k_len = k_len_val;
        let mut v_len = v_len_val;
        let mut pp_len = pp_len_val;
        let mut bt_len = bt_len_val;
        let mut p_len = p_len_val;
        let mut seq_len_v = seq_len;
        let mut head_dim_v = head_dim;
        let mut page_size_v = page_size;
        let mut kv_dim_v = kv_dim;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut k_ptr, &mut k_len);     // k: &[u16]
        Self::push_slice_arg(&mut args, &mut v_ptr, &mut v_len);     // v: &[u16]
        Self::push_slice_arg(&mut args, &mut pp_ptr, &mut pp_len);  // mut page_pool: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut bt_ptr, &mut bt_len);  // block_table: &[i32]
        Self::push_slice_arg(&mut args, &mut p_ptr, &mut p_len);    // positions: &[i32]
        Self::push_scalar_arg(&mut args, &mut seq_len_v);           // seq_len: u32
        Self::push_scalar_arg(&mut args, &mut head_dim_v);          // _head_dim: u32
        Self::push_scalar_arg(&mut args, &mut page_size_v);         // page_size: u32
        Self::push_scalar_arg(&mut args, &mut kv_dim_v);            // kv_dim: u32

        let config = LaunchConfig::for_num_elems(total as u32);
        self.raw_launch("infers_paged_kv_write_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_paged_kv_read_bf16` kernel: paged KV cache read.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_paged_kv_read_bf16(page_pool: &[u16], block_table: &[i32], _num_pages: u32, num_cached_tokens: u32, _head_dim: u32, page_size: u32, kv_dim: u32, mut k_out: DisjointSlice<u16>, mut v_out: DisjointSlice<u16>)
    /// ```
    pub fn launch_paged_kv_read_bf16(
        &self,
        stream: &Arc<CudaStream>,
        page_pool: &CudaSlice<half::bf16>,
        block_table: &CudaSlice<i32>,
        num_pages: u32,
        num_cached_tokens: u32,
        head_dim: u32,
        page_size: u32,
        kv_dim: u32,
        k_out: &mut CudaSlice<half::bf16>,
        v_out: &mut CudaSlice<half::bf16>,
    ) -> anyhow::Result<()> {
        let total = (num_cached_tokens as usize) * (kv_dim as usize);

        let pp_len_val = page_pool.len() as u64;
        let bt_len_val = block_table.len() as u64;
        let ko_len_val = k_out.len() as u64;
        let vo_len_val = v_out.len() as u64;

        let (pp_ptr, pp_guard) = page_pool.device_ptr(stream);
        let (bt_ptr, bt_guard) = block_table.device_ptr(stream);
        let (ko_ptr, ko_guard) = k_out.device_ptr_mut(stream);
        let (vo_ptr, vo_guard) = v_out.device_ptr_mut(stream);

        let _guards = (pp_guard, bt_guard, ko_guard, vo_guard);

        let mut pp_ptr = pp_ptr as cuda_core::sys::CUdeviceptr;
        let mut bt_ptr = bt_ptr as cuda_core::sys::CUdeviceptr;
        let mut ko_ptr = ko_ptr as cuda_core::sys::CUdeviceptr;
        let mut vo_ptr = vo_ptr as cuda_core::sys::CUdeviceptr;
        let mut pp_len = pp_len_val;
        let mut bt_len = bt_len_val;
        let mut ko_len = ko_len_val;
        let mut vo_len = vo_len_val;
        let mut num_pages_v = num_pages;
        let mut num_cached_tokens_v = num_cached_tokens;
        let mut head_dim_v = head_dim;
        let mut page_size_v = page_size;
        let mut kv_dim_v = kv_dim;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut pp_ptr, &mut pp_len);  // page_pool: &[u16]
        Self::push_slice_arg(&mut args, &mut bt_ptr, &mut bt_len);  // block_table: &[i32]
        Self::push_scalar_arg(&mut args, &mut num_pages_v);         // _num_pages: u32
        Self::push_scalar_arg(&mut args, &mut num_cached_tokens_v); // num_cached_tokens: u32
        Self::push_scalar_arg(&mut args, &mut head_dim_v);          // _head_dim: u32
        Self::push_scalar_arg(&mut args, &mut page_size_v);         // page_size: u32
        Self::push_scalar_arg(&mut args, &mut kv_dim_v);            // kv_dim: u32
        Self::push_slice_arg(&mut args, &mut ko_ptr, &mut ko_len);  // mut k_out: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut vo_ptr, &mut vo_len);  // mut v_out: DisjointSlice<u16>

        let config = LaunchConfig::for_num_elems(total as u32);
        self.raw_launch("infers_paged_kv_read_bf16", stream, config, &mut args)
    }

    /// Launch the `int4_gemm_auto_round` kernel: INT4 GEMM with AutoRound dequant.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// int4_gemm_auto_round(mut output: DisjointSlice<u16>, weight: &[u32], scales: &[u16], zeros: &[u32], input: &[u16], m: u32, n: u32, k: u32, group_size: u32, transposed: u32)
    /// ```
    pub fn launch_int4_gemm_auto_round(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut CudaSlice<half::bf16>,
        weight: &CudaSlice<u32>,
        scales: &CudaSlice<half::f16>,
        zeros: &CudaSlice<u32>,
        input: &CudaSlice<half::bf16>,
        m: u32,
        n: u32,
        k: u32,
        group_size: u32,
        transposed: u32,
    ) -> anyhow::Result<()> {
        let o_len_val = output.len() as u64;
        let w_len_val = weight.len() as u64;
        let s_len_val = scales.len() as u64;
        let z_len_val = zeros.len() as u64;
        let i_len_val = input.len() as u64;

        let (o_ptr, o_guard) = output.device_ptr_mut(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (s_ptr, s_guard) = scales.device_ptr(stream);
        let (z_ptr, z_guard) = zeros.device_ptr(stream);
        let (i_ptr, i_guard) = input.device_ptr(stream);

        let _guards = (o_guard, w_guard, s_guard, z_guard, i_guard);

        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut z_ptr = z_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_len = o_len_val;
        let mut w_len = w_len_val;
        let mut s_len = s_len_val;
        let mut z_len = z_len_val;
        let mut i_len = i_len_val;
        let mut m_v = m;
        let mut n_v = n;
        let mut k_v = k;
        let mut group_size_v = group_size;
        let mut transposed_v = transposed;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);  // mut output: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);  // weight: &[u32]
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);  // scales: &[u16]
        Self::push_slice_arg(&mut args, &mut z_ptr, &mut z_len);  // zeros: &[u32]
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);  // input: &[u16]
        Self::push_scalar_arg(&mut args, &mut m_v);               // m: u32
        Self::push_scalar_arg(&mut args, &mut n_v);               // n: u32
        Self::push_scalar_arg(&mut args, &mut k_v);               // k: u32
        Self::push_scalar_arg(&mut args, &mut group_size_v);      // group_size: u32
        Self::push_scalar_arg(&mut args, &mut transposed_v);      // transposed: u32

        let config = if m <= 1 {
            LaunchConfig {
                grid_dim: ((n + 63) / 64, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: ((n + 63) / 64, (m + 3) / 4, 1),
                block_dim: (64, 4, 1),
                shared_mem_bytes: 0,
            }
        };
        self.raw_launch("int4_gemm_auto_round", stream, config, &mut args)
    }

    /// Launch the `int4_gemm_auto_round_tiled` kernel: INT4 GEMM with AutoRound dequant
    /// and shared memory input tiling for optimized M=1 decode.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// int4_gemm_auto_round_tiled(mut output: DisjointSlice<u16>, weight: &[u32], scales: &[u16], zeros: &[u32], input: &[u16], m: u32, n: u32, k: u32, group_size: u32, transposed: u32)
    /// ```
    pub fn launch_int4_gemm_auto_round_tiled(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut CudaSlice<half::bf16>,
        weight: &CudaSlice<u32>,
        scales: &CudaSlice<half::f16>,
        zeros: &CudaSlice<u32>,
        input: &CudaSlice<half::bf16>,
        m: u32,
        n: u32,
        k: u32,
        group_size: u32,
        transposed: u32,
    ) -> anyhow::Result<()> {
        let o_len_val = output.len() as u64;
        let w_len_val = weight.len() as u64;
        let s_len_val = scales.len() as u64;
        let z_len_val = zeros.len() as u64;
        let i_len_val = input.len() as u64;

        let (o_ptr, o_guard) = output.device_ptr_mut(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (s_ptr, s_guard) = scales.device_ptr(stream);
        let (z_ptr, z_guard) = zeros.device_ptr(stream);
        let (i_ptr, i_guard) = input.device_ptr(stream);

        let _guards = (o_guard, w_guard, s_guard, z_guard, i_guard);

        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut z_ptr = z_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_len = o_len_val;
        let mut w_len = w_len_val;
        let mut s_len = s_len_val;
        let mut z_len = z_len_val;
        let mut i_len = i_len_val;
        let mut m_v = m;
        let mut n_v = n;
        let mut k_v = k;
        let mut group_size_v = group_size;
        let mut transposed_v = transposed;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);  // mut output: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);  // weight: &[u32]
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);  // scales: &[u16]
        Self::push_slice_arg(&mut args, &mut z_ptr, &mut z_len);  // zeros: &[u32]
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);  // input: &[u16]
        Self::push_scalar_arg(&mut args, &mut m_v);               // m: u32
        Self::push_scalar_arg(&mut args, &mut n_v);               // n: u32
        Self::push_scalar_arg(&mut args, &mut k_v);               // k: u32
        Self::push_scalar_arg(&mut args, &mut group_size_v);      // group_size: u32
        Self::push_scalar_arg(&mut args, &mut transposed_v);      // transposed: u32

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, m, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: (group_size * 2),
        };
        self.raw_launch("int4_gemm_auto_round_tiled", stream, config, &mut args)
    }


    /// Launch the `int4_gemm_auto_round_ksplit` kernel: INT4 GEMM with K-splitting.
    /// Each block computes partial sums for 64 output columns over a portion of K.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// int4_gemm_auto_round_ksplit(partial_sums: &mut [f32], weight: &[u32], scales: &[u16], zeros: &[u32], input: &[u16], n: u32, k: u32, group_size: u32, transposed: u32, k_split: u32)
    /// ```
    pub fn launch_int4_gemm_auto_round_ksplit(
        &self,
        stream: &Arc<CudaStream>,
        partial_sums: &mut CudaSlice<f32>,
        weight: &CudaSlice<u32>,
        scales: &CudaSlice<half::f16>,
        zeros: &CudaSlice<u32>,
        input: &CudaSlice<half::bf16>,
        n: u32,
        k: u32,
        group_size: u32,
        transposed: u32,
        k_split: u32,
    ) -> anyhow::Result<()> {
        let ps_len_val = partial_sums.len() as u64;
        let w_len_val = weight.len() as u64;
        let s_len_val = scales.len() as u64;
        let z_len_val = zeros.len() as u64;
        let i_len_val = input.len() as u64;

        let (ps_ptr, ps_guard) = partial_sums.device_ptr_mut(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (s_ptr, s_guard) = scales.device_ptr(stream);
        let (z_ptr, z_guard) = zeros.device_ptr(stream);
        let (i_ptr, i_guard) = input.device_ptr(stream);

        let _guards = (ps_guard, w_guard, s_guard, z_guard, i_guard);

        let mut ps_ptr = ps_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut z_ptr = z_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut ps_len = ps_len_val;
        let mut w_len = w_len_val;
        let mut s_len = s_len_val;
        let mut z_len = z_len_val;
        let mut i_len = i_len_val;
        let mut n_v = n;
        let mut k_v = k;
        let mut group_size_v = group_size;
        let mut transposed_v = transposed;
        let mut k_split_v = k_split;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut ps_ptr, &mut ps_len); // partial_sums: &mut [f32]
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);  // weight: &[u32]
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);  // scales: &[u16]
        Self::push_slice_arg(&mut args, &mut z_ptr, &mut z_len);  // zeros: &[u32]
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);  // input: &[u16]
        Self::push_scalar_arg(&mut args, &mut n_v);               // n: u32
        Self::push_scalar_arg(&mut args, &mut k_v);               // k: u32
        Self::push_scalar_arg(&mut args, &mut group_size_v);      // group_size: u32
        Self::push_scalar_arg(&mut args, &mut transposed_v);      // transposed: u32
        Self::push_scalar_arg(&mut args, &mut k_split_v);         // k_split: u32

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, k_split, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        self.raw_launch("int4_gemm_auto_round_ksplit", stream, config, &mut args)
    }

    /// Launch the `int4_gemm_warp_split` kernel: warp-cooperative INT4 GEMV with K-splitting.
    /// Same interface as ksplit but uses block (32,8,1) with warp shuffle reduction.
    pub fn launch_int4_gemm_warp_split(
        &self,
        stream: &Arc<CudaStream>,
        partial_sums: &mut CudaSlice<f32>,
        weight: &CudaSlice<u32>,
        scales: &CudaSlice<half::f16>,
        zeros: &CudaSlice<u32>,
        input: &CudaSlice<half::bf16>,
        n: u32,
        k: u32,
        group_size: u32,
        transposed: u32,
        k_split: u32,
    ) -> anyhow::Result<()> {
        let ps_len_val = partial_sums.len() as u64;
        let w_len_val = weight.len() as u64;
        let s_len_val = scales.len() as u64;
        let z_len_val = zeros.len() as u64;
        let i_len_val = input.len() as u64;

        let (ps_ptr, ps_guard) = partial_sums.device_ptr_mut(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (s_ptr, s_guard) = scales.device_ptr(stream);
        let (z_ptr, z_guard) = zeros.device_ptr(stream);
        let (i_ptr, i_guard) = input.device_ptr(stream);

        let _guards = (ps_guard, w_guard, s_guard, z_guard, i_guard);

        let mut ps_ptr = ps_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut z_ptr = z_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut ps_len = ps_len_val;
        let mut w_len = w_len_val;
        let mut s_len = s_len_val;
        let mut z_len = z_len_val;
        let mut i_len = i_len_val;
        let mut n_v = n;
        let mut k_v = k;
        let mut group_size_v = group_size;
        let mut transposed_v = transposed;
        let mut k_split_v = k_split;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut ps_ptr, &mut ps_len);
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);
        Self::push_slice_arg(&mut args, &mut z_ptr, &mut z_len);
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);
        Self::push_scalar_arg(&mut args, &mut n_v);
        Self::push_scalar_arg(&mut args, &mut k_v);
        Self::push_scalar_arg(&mut args, &mut group_size_v);
        Self::push_scalar_arg(&mut args, &mut transposed_v);
        Self::push_scalar_arg(&mut args, &mut k_split_v);

        let config = LaunchConfig {
            grid_dim: ((n + 7) / 8, k_split, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        self.raw_launch("int4_gemm_warp_split", stream, config, &mut args)
    }

    /// Launch the `reduce_partial_sums_bf16` kernel: reduce K-split partial sums to bf16.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// reduce_partial_sums_bf16(mut output: DisjointSlice<u16>, partial_sums: &[f32], n: u32, k_split: u32)
    /// ```
    pub fn launch_reduce_partial_sums_bf16(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut CudaSlice<half::bf16>,
        partial_sums: &CudaSlice<f32>,
        n: u32,
        k_split: u32,
    ) -> anyhow::Result<()> {
        let o_len_val = output.len() as u64;
        let ps_len_val = partial_sums.len() as u64;

        let (o_ptr, o_guard) = output.device_ptr_mut(stream);
        let (ps_ptr, ps_guard) = partial_sums.device_ptr(stream);

        let _guards = (o_guard, ps_guard);

        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut ps_ptr = ps_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_len = o_len_val;
        let mut ps_len = ps_len_val;
        let mut n_v = n;
        let mut k_split_v = k_split;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);    // output: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut ps_ptr, &mut ps_len); // partial_sums: &[f32]
        Self::push_scalar_arg(&mut args, &mut n_v);                // n: u32
        Self::push_scalar_arg(&mut args, &mut k_split_v);          // k_split: u32

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        self.raw_launch("reduce_partial_sums_bf16", stream, config, &mut args)
    }

    /// Launch the `int4_gemm_warp` kernel: warp-cooperative INT4 GEMV for M=1 decode.

    /// Each warp (32 lanes) computes one output column. Lanes split the K
    /// dimension across groups and reduce via warp shuffle — no separate
    /// reduction kernel or partial_sums buffer in global memory.

    /// The kernel signature is:
    /// ```ignore
    /// int4_gemm_warp(mut output: DisjointSlice<u16>, weight: &[u32], scales: &[u16], zeros: &[u32], input: &[u16], n: u32, k: u32, group_size: u32, transposed: u32)
    /// ```
    pub fn launch_int4_gemm_warp(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut CudaSlice<half::bf16>,
        weight: &CudaSlice<u32>,
        scales: &CudaSlice<half::f16>,
        zeros: &CudaSlice<u32>,
        input: &CudaSlice<half::bf16>,
        n: u32,
        k: u32,
        group_size: u32,
        transposed: u32,
    ) -> anyhow::Result<()> {
        let o_len_val = output.len() as u64;
        let w_len_val = weight.len() as u64;
        let s_len_val = scales.len() as u64;
        let z_len_val = zeros.len() as u64;
        let i_len_val = input.len() as u64;

        let (o_ptr, o_guard) = output.device_ptr_mut(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (s_ptr, s_guard) = scales.device_ptr(stream);
        let (z_ptr, z_guard) = zeros.device_ptr(stream);
        let (i_ptr, i_guard) = input.device_ptr(stream);

        let _guards = (o_guard, w_guard, s_guard, z_guard, i_guard);

        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut z_ptr = z_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_len = o_len_val;
        let mut w_len = w_len_val;
        let mut s_len = s_len_val;
        let mut z_len = z_len_val;
        let mut i_len = i_len_val;
        let mut n_v = n;
        let mut k_v = k;
        let mut gs_v = group_size;
        let mut trans_v = transposed;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);  // output: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);  // weight: &[u32]
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);  // scales: &[u16]
        Self::push_slice_arg(&mut args, &mut z_ptr, &mut z_len);  // zeros: &[u32]
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);  // input: &[u16]
        Self::push_scalar_arg(&mut args, &mut n_v);              // n: u32
        Self::push_scalar_arg(&mut args, &mut k_v);              // k: u32
        Self::push_scalar_arg(&mut args, &mut gs_v);             // group_size: u32
        Self::push_scalar_arg(&mut args, &mut trans_v);          // transposed: u32

        let config = LaunchConfig {
            grid_dim: ((n + 7) / 8, 1, 1),   // ceil(N / 8)
            block_dim: (32, 8, 1),           // 32 lanes × 8 warps = 256 threads
            shared_mem_bytes: 0,
        };
        self.raw_launch("int4_gemm_warp", stream, config, &mut args)
    }


    /// Launch INT4 AutoRound dequantize-to-BF16 kernel.
    ///
    /// Reads packed INT4 weights + FP16 scales + packed zeros,
    /// writes bf16 output to `output` buffer.
    pub fn launch_int4_dequant_to_bf16(
        &self,
        stream: &Arc<CudaStream>,
        output: &CudaSlice<half::bf16>,     // [N, K] bf16 output
        weight: &CudaSlice<u32>,            // [N, K/8] packed INT4
        scales: &CudaSlice<half::f16>,      // [N, K/group_size] fp16 scales
        zeros: &CudaSlice<u32>,             // packed zeros
        n: u32,
        k: u32,
        group_size: u32,
    ) -> anyhow::Result<()> {
        let o_len_val = output.len() as u64;
        let w_len_val = weight.len() as u64;
        let s_len_val = scales.len() as u64;
        let z_len_val = zeros.len() as u64;

        let (o_ptr, o_guard) = output.device_ptr(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (s_ptr, s_guard) = scales.device_ptr(stream);
        let (z_ptr, z_guard) = zeros.device_ptr(stream);

        let _guards = (o_guard, w_guard, s_guard, z_guard);

        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut z_ptr = z_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_len = o_len_val;
        let mut w_len = w_len_val;
        let mut s_len = s_len_val;
        let mut z_len = z_len_val;
        let mut n_v = n;
        let mut k_v = k;
        let mut group_size_v = group_size;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);  // output: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);  // weight: &[u32]
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);  // scales: &[u16]
        Self::push_slice_arg(&mut args, &mut z_ptr, &mut z_len);  // zeros: &[u32]
        Self::push_scalar_arg(&mut args, &mut n_v);               // n: u32
        Self::push_scalar_arg(&mut args, &mut k_v);               // k: u32
        Self::push_scalar_arg(&mut args, &mut group_size_v);      // group_size: u32

        let config = LaunchConfig {
            grid_dim: ((n + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.raw_launch("int4_dequant_to_bf16", stream, config, &mut args)
    }

    /// Launch NVFP4 dequantize-to-BF16 kernel.
    /// Reads packed NVFP4 weights + FP8 scales + weight global scale scalar,
    /// writes bf16 output to `output` buffer.
    pub fn launch_nvfp4_dequant_to_bf16(
        &self,
        stream: &Arc<CudaStream>,
        output: &CudaSlice<half::bf16>,     // [N, K] bf16 output
        weight_packed: &CudaSlice<u8>,       // [N, K/2] packed FP4
        weight_scale: &CudaSlice<u8>,        // [N, K/group_size] fp8_e4m3
        weight_global_scale: f32,            // scalar global scale
        n: u32,
        k: u32,
        group_size: u32,
    ) -> anyhow::Result<()> {
        let o_len_val = output.len() as u64;
        let w_len_val = weight_packed.len() as u64;
        let s_len_val = weight_scale.len() as u64;

        let (o_ptr, o_guard) = output.device_ptr(stream);
        let (w_ptr, w_guard) = weight_packed.device_ptr(stream);
        let (s_ptr, s_guard) = weight_scale.device_ptr(stream);

        let _guards = (o_guard, w_guard, s_guard);

        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_len = o_len_val;
        let mut w_len = w_len_val;
        let mut s_len = s_len_val;
        let mut n_v = n;
        let mut k_v = k;
        let mut group_size_v = group_size;
        let mut gscale = weight_global_scale;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);  // output: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);  // weight_packed: &[u8]
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);  // weight_scale: &[u8]
        Self::push_scalar_arg(&mut args, &mut gscale);            // weight_global_scale: f32
        Self::push_scalar_arg(&mut args, &mut n_v);               // n: u32
        Self::push_scalar_arg(&mut args, &mut k_v);               // k: u32
        Self::push_scalar_arg(&mut args, &mut group_size_v);      // group_size: u32

        let config = LaunchConfig {
            grid_dim: ((n + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.raw_launch("nvfp4_dequant_to_bf16", stream, config, &mut args)
    }

    /// Launch the `nvfp4_gemm_fused` kernel: fused NVFP4 dequant + GEMM.
    ///
    /// Reads compressed FP4 weights directly from GPU, dequantizes in registers,
    /// and multiplies with BF16 activations — no intermediate bf16 buffer needed.
    pub fn launch_nvfp4_gemm_fused(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut CudaSlice<half::bf16>,     // [M, N] bf16
        weight_packed: &CudaSlice<u8>,          // [N, K/2] packed FP4
        weight_scale: &CudaSlice<u8>,           // [N, K/group_size] fp8_e4m3
        input: &CudaSlice<half::bf16>,            // [M, K] bf16
        weight_global_scale: f32,                // scalar global scale
        m: u32,
        n: u32,
        k: u32,
        group_size: u32,
    ) -> anyhow::Result<()> {
        let o_len_val = output.len() as u64;
        let w_len_val = weight_packed.len() as u64;
        let s_len_val = weight_scale.len() as u64;
        let i_len_val = input.len() as u64;

        let (o_ptr, o_guard) = output.device_ptr_mut(stream);
        let (w_ptr, w_guard) = weight_packed.device_ptr(stream);
        let (s_ptr, s_guard) = weight_scale.device_ptr(stream);
        let (i_ptr, i_guard) = input.device_ptr(stream);

        let _guards = (o_guard, w_guard, s_guard, i_guard);

        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_len = o_len_val;
        let mut w_len = w_len_val;
        let mut s_len = s_len_val;
        let mut i_len = i_len_val;
        let mut m_v = m;
        let mut n_v = n;
        let mut k_v = k;
        let mut group_size_v = group_size;
        let mut gscale = weight_global_scale;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);  // mut output: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);  // weight_packed: &[u8]
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);  // weight_scale: &[u8]
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);  // input: &[u16]
        Self::push_scalar_arg(&mut args, &mut gscale);            // weight_global_scale: f32
        Self::push_scalar_arg(&mut args, &mut m_v);               // m: u32
        Self::push_scalar_arg(&mut args, &mut n_v);               // n: u32
        Self::push_scalar_arg(&mut args, &mut k_v);               // k: u32
        Self::push_scalar_arg(&mut args, &mut group_size_v);      // group_size: u32

        let config = if m <= 1 {
            LaunchConfig {
                grid_dim: ((n + 63) / 64, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: ((n + 63) / 64, (m + 3) / 4, 1),
                block_dim: (64, 4, 1),
                shared_mem_bytes: 0,
            }
        };
        self.raw_launch("nvfp4_gemm_fused", stream, config, &mut args)
    }

    /// Launch the `nvfp4_gemm_fused_ksplit` kernel: fused NVFP4 GEMM with K-splitting.
    /// Each block computes partial sums for 64 output columns over a portion of K (M=1 only).
    ///
    /// The kernel signature is:
    /// ```ignore
    /// nvfp4_gemm_fused_ksplit(partial_sums: &mut [f32], weight_packed: &[u8], weight_scale: &[u8], input: &[u16], weight_global_scale: f32, n: u32, k: u32, group_size: u32, k_split: u32)
    /// ```
    pub fn launch_nvfp4_gemm_fused_ksplit(
        &self,
        stream: &Arc<CudaStream>,
        partial_sums: &mut CudaSlice<f32>,         // [K_SPLIT, N] f32
        weight_packed: &CudaSlice<u8>,              // [N, K/2] packed FP4
        weight_scale: &CudaSlice<u8>,               // [N, K/group_size] fp8_e4m3
        input: &CudaSlice<half::bf16>,                // [K] bf16 (M=1)
        weight_global_scale: f32,                    // scalar global scale
        n: u32,
        k: u32,
        group_size: u32,
        k_split: u32,
    ) -> anyhow::Result<()> {
        let ps_len_val = partial_sums.len() as u64;
        let w_len_val = weight_packed.len() as u64;
        let s_len_val = weight_scale.len() as u64;
        let i_len_val = input.len() as u64;

        let (ps_ptr, ps_guard) = partial_sums.device_ptr_mut(stream);
        let (w_ptr, w_guard) = weight_packed.device_ptr(stream);
        let (s_ptr, s_guard) = weight_scale.device_ptr(stream);
        let (i_ptr, i_guard) = input.device_ptr(stream);

        let _guards = (ps_guard, w_guard, s_guard, i_guard);

        let mut ps_ptr = ps_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut ps_len = ps_len_val;
        let mut w_len = w_len_val;
        let mut s_len = s_len_val;
        let mut i_len = i_len_val;
        let mut n_v = n;
        let mut k_v = k;
        let mut group_size_v = group_size;
        let mut gscale = weight_global_scale;
        let mut k_split_v = k_split;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut ps_ptr, &mut ps_len);    // partial_sums: &mut [f32]
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);     // weight_packed: &[u8]
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);     // weight_scale: &[u8]
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);     // input: &[u16]
        Self::push_scalar_arg(&mut args, &mut gscale);               // weight_global_scale: f32
        Self::push_scalar_arg(&mut args, &mut n_v);                  // n: u32
        Self::push_scalar_arg(&mut args, &mut k_v);                  // k: u32
        Self::push_scalar_arg(&mut args, &mut group_size_v);         // group_size: u32
        Self::push_scalar_arg(&mut args, &mut k_split_v);            // k_split: u32

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, k_split, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        self.raw_launch("nvfp4_gemm_fused_ksplit", stream, config, &mut args)
    }

    /// Launch the `sanitize_nan_bf16` kernel: replace NaN values in a bf16 buffer with 0.0.
    pub fn launch_sanitize_nan_bf16(
        &self,
        stream: &Arc<CudaStream>,
        buf: &mut CudaSlice<half::bf16>,
    ) -> anyhow::Result<()> {
        let len_val = buf.len() as u64;
        let mut len_scalar = buf.len() as u32;
        let block_size = 256u32;
        let grid_size = ((buf.len() as u32 + block_size - 1) / block_size, 1, 1);
        let (ptr, guard) = buf.device_ptr_mut(stream);
        let _guard = guard;
        let mut d_ptr = ptr as cuda_core::sys::CUdeviceptr;
        let mut len_v = len_val;
        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut d_ptr, &mut len_v);
        Self::push_scalar_arg(&mut args, &mut len_scalar);
        let config = LaunchConfig {
            grid_dim: grid_size,
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        self.raw_launch("sanitize_nan_bf16", stream, config, &mut args)
    }

    /// Launch the `bf16_gemm_tiled` kernel: tiled bf16 GEMM with shared memory.
    ///
    /// Computes C[M,N] = A[M,K] @ B[N,K]^T where all buffers are row-major bf16.
    /// Used as a replacement for cuBLAS in the NVFP4 path to avoid workspace corruption.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// bf16_gemm_tiled(mut output: DisjointSlice<u16>, input: &[u16], weight: &[u16], m: u32, n: u32, k: u32)
    /// ```
   pub fn launch_bf16_gemm_tiled(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut CudaSlice<half::bf16>,
        input: &CudaSlice<half::bf16>,
        weight: &CudaSlice<half::bf16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let o_len_val = output.len() as u64;
        let i_len_val = input.len() as u64;
        let w_len_val = weight.len() as u64;

        let (o_ptr, o_guard) = output.device_ptr_mut(stream);
        let (i_ptr, i_guard) = input.device_ptr(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);

        let _guards = (o_guard, i_guard, w_guard);

        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_len = o_len_val;
        let mut i_len = i_len_val;
        let mut w_len = w_len_val;
        let mut m_v = m;
        let mut n_v = n;
        let mut k_v = k;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);  // output: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);  // input: &[u16]
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);  // weight: &[u16]
        Self::push_scalar_arg(&mut args, &mut m_v);               // m: u32
        Self::push_scalar_arg(&mut args, &mut n_v);               // n: u32
        Self::push_scalar_arg(&mut args, &mut k_v);               // k: u32

        // Grid: (N/64, M/64, 1), Block: (256, 1, 1)
        let grid_x = (n + 63) / 64;
        let grid_y = (m + 63) / 64;
        let config = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,  // no shared memory used in current version
        };
        self.raw_launch("bf16_gemm_tiled", stream, config, &mut args)
    }

    /// Launch the `int4_gemm_gguf` kernel: INT4 GEMM with GGUF dequant.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// int4_gemm_gguf(mut output: DisjointSlice<u16>, weight: &[u32], scales: &[u16], zeros: &[u32], input: &[u16], m: u32, n: u32, k: u32, group_size: u32, transposed: u32)
    /// ```
    pub fn launch_int4_gemm_gguf(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut CudaSlice<half::bf16>,
        weight: &CudaSlice<u32>,
        scales: &CudaSlice<half::bf16>,
        zeros: &CudaSlice<u32>,
        input: &CudaSlice<half::bf16>,
        m: u32,
        n: u32,
        k: u32,
        group_size: u32,
        transposed: u32,
    ) -> anyhow::Result<()> {
        let o_len_val = output.len() as u64;
        let w_len_val = weight.len() as u64;
        let s_len_val = scales.len() as u64;
        let z_len_val = zeros.len() as u64;
        let i_len_val = input.len() as u64;

        let (o_ptr, o_guard) = output.device_ptr_mut(stream);
        let (w_ptr, w_guard) = weight.device_ptr(stream);
        let (s_ptr, s_guard) = scales.device_ptr(stream);
        let (z_ptr, z_guard) = zeros.device_ptr(stream);
        let (i_ptr, i_guard) = input.device_ptr(stream);

        let _guards = (o_guard, w_guard, s_guard, z_guard, i_guard);

        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut w_ptr = w_ptr as cuda_core::sys::CUdeviceptr;
        let mut s_ptr = s_ptr as cuda_core::sys::CUdeviceptr;
        let mut z_ptr = z_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_len = o_len_val;
        let mut w_len = w_len_val;
        let mut s_len = s_len_val;
        let mut z_len = z_len_val;
        let mut i_len = i_len_val;
        let mut m_v = m;
        let mut n_v = n;
        let mut k_v = k;
        let mut group_size_v = group_size;
        let mut transposed_v = transposed;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);  // mut output: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut w_ptr, &mut w_len);  // weight: &[u32]
        Self::push_slice_arg(&mut args, &mut s_ptr, &mut s_len);  // scales: &[u16]
        Self::push_slice_arg(&mut args, &mut z_ptr, &mut z_len);  // zeros: &[u32]
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);  // input: &[u16]
        Self::push_scalar_arg(&mut args, &mut m_v);               // m: u32
        Self::push_scalar_arg(&mut args, &mut n_v);               // n: u32
        Self::push_scalar_arg(&mut args, &mut k_v);               // k: u32
        Self::push_scalar_arg(&mut args, &mut group_size_v);      // group_size: u32
        Self::push_scalar_arg(&mut args, &mut transposed_v);      // transposed: u32

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, (m + 3) / 4, 1),
            block_dim: (64, 4, 1),
            shared_mem_bytes: 0,
        };
        self.raw_launch("int4_gemm_gguf", stream, config, &mut args)
    }

    /// Launch the `infers_fp8_quantize_e4m3` kernel: BF16 -> FP8 E4M3.
    pub fn launch_fp8_quantize_e4m3(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<u8>,
        n: u32,
    ) -> anyhow::Result<()> {
        let i_len_val = input.len() as u64;
        let o_len_val = output.len() as u64;

        let (i_ptr, i_guard) = input.device_ptr(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);

        let _guards = (i_guard, o_guard);

        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_len = i_len_val;
        let mut o_len = o_len_val;
        let mut n_v = n;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);
        Self::push_scalar_arg(&mut args, &mut n_v);

        let config = LaunchConfig::for_num_elems(n);
        self.raw_launch("infers_fp8_quantize_e4m3", stream, config, &mut args)
    }

    /// Launch the `infers_fp8_dequantize_e4m3` kernel: FP8 E4M3 -> BF16.
    pub fn launch_fp8_dequantize_e4m3(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<half::bf16>,
        n: u32,
    ) -> anyhow::Result<()> {
        let i_len_val = input.len() as u64;
        let o_len_val = output.len() as u64;

        let (i_ptr, i_guard) = input.device_ptr(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);

        let _guards = (i_guard, o_guard);

        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_len = i_len_val;
        let mut o_len = o_len_val;
        let mut n_v = n;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);
        Self::push_scalar_arg(&mut args, &mut n_v);

        let config = LaunchConfig::for_num_elems(n);
        self.raw_launch("infers_fp8_dequantize_e4m3", stream, config, &mut args)
    }

    /// Launch the `infers_fp8_quantize_e5m2` kernel: BF16 -> FP8 E5M2.
    pub fn launch_fp8_quantize_e5m2(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<u8>,
        n: u32,
    ) -> anyhow::Result<()> {
        let i_len_val = input.len() as u64;
        let o_len_val = output.len() as u64;

        let (i_ptr, i_guard) = input.device_ptr(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);

        let _guards = (i_guard, o_guard);

        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_len = i_len_val;
        let mut o_len = o_len_val;
        let mut n_v = n;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);
        Self::push_scalar_arg(&mut args, &mut n_v);

        let config = LaunchConfig::for_num_elems(n);
        self.raw_launch("infers_fp8_quantize_e5m2", stream, config, &mut args)
    }

    /// Launch the `infers_fp8_dequantize_e5m2` kernel: FP8 E5M2 -> BF16.
    pub fn launch_fp8_dequantize_e5m2(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<half::bf16>,
        n: u32,
    ) -> anyhow::Result<()> {
        let i_len_val = input.len() as u64;
        let o_len_val = output.len() as u64;

        let (i_ptr, i_guard) = input.device_ptr(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);

        let _guards = (i_guard, o_guard);

        let mut i_ptr = i_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut i_len = i_len_val;
        let mut o_len = o_len_val;
        let mut n_v = n;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut i_ptr, &mut i_len);
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);
        Self::push_scalar_arg(&mut args, &mut n_v);

        let config = LaunchConfig::for_num_elems(n);
        self.raw_launch("infers_fp8_dequantize_e5m2", stream, config, &mut args)
    }

    /// Launch the `infers_paged_attention_decode_bf16` kernel: paged attention decode.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_paged_attention_decode_bf16(q: &[u16], page_pool: &[u16], block_table: &[i32], num_pages: u32, num_cached_tokens: u32, head_dim: u32, num_kv_heads: u32, num_query_heads: u32, page_size: u32, kv_dim: u32, mut output: DisjointSlice<u16>)
    /// ```
    pub fn launch_paged_attention_decode_bf16(
        &self,
        stream: &Arc<CudaStream>,
        q: &CudaSlice<half::bf16>,
        page_pool: &CudaSlice<half::bf16>,
        block_table: &CudaSlice<i32>,
        output: &mut CudaSlice<half::bf16>,
        num_pages: u32,
        num_cached_tokens: u32,
        head_dim: u32,
        num_kv_heads: u32,
        num_query_heads: u32,
        page_size: u32,
        kv_dim: u32,
    ) -> anyhow::Result<()> {
        let q_len_val = q.len() as u64;
        let pp_len_val = page_pool.len() as u64;
        let bt_len_val = block_table.len() as u64;
        let o_len_val = output.len() as u64;

        let (q_ptr, q_guard) = q.device_ptr(stream);
        let (pp_ptr, pp_guard) = page_pool.device_ptr(stream);
        let (bt_ptr, bt_guard) = block_table.device_ptr(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);

        let _guards = (q_guard, pp_guard, bt_guard, o_guard);

        let mut q_ptr = q_ptr as cuda_core::sys::CUdeviceptr;
        let mut pp_ptr = pp_ptr as cuda_core::sys::CUdeviceptr;
        let mut bt_ptr = bt_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut q_len = q_len_val;
        let mut pp_len = pp_len_val;
        let mut bt_len = bt_len_val;
        let mut o_len = o_len_val;
        let mut num_pages_v = num_pages;
        let mut num_cached_tokens_v = num_cached_tokens;
        let mut head_dim_v = head_dim;
        let mut num_kv_heads_v = num_kv_heads;
        let mut num_query_heads_v = num_query_heads;
        let mut page_size_v = page_size;
        let mut kv_dim_v = kv_dim;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut q_ptr, &mut q_len);           // q: &[u16]
        Self::push_slice_arg(&mut args, &mut pp_ptr, &mut pp_len);         // page_pool: &[u16]
        Self::push_slice_arg(&mut args, &mut bt_ptr, &mut bt_len);         // block_table: &[i32]
        Self::push_scalar_arg(&mut args, &mut num_pages_v);                // num_pages: u32
        Self::push_scalar_arg(&mut args, &mut num_cached_tokens_v);        // num_cached_tokens: u32
        Self::push_scalar_arg(&mut args, &mut head_dim_v);                 // head_dim: u32
        Self::push_scalar_arg(&mut args, &mut num_kv_heads_v);             // num_kv_heads: u32
        Self::push_scalar_arg(&mut args, &mut num_query_heads_v);          // num_query_heads: u32
        Self::push_scalar_arg(&mut args, &mut page_size_v);                // page_size: u32
        Self::push_scalar_arg(&mut args, &mut kv_dim_v);                   // kv_dim: u32
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);           // mut output: DisjointSlice<u16>

        let config = LaunchConfig {
            grid_dim: (num_kv_heads, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 3 * 256 * 4,  // 3 regions: Q values + max scratch + sum scratch
        };
        self.raw_launch("infers_paged_attention_decode_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_gdn_recurrent_step_bf16` kernel: GDN recurrent step.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_gdn_recurrent_step_bf16(query: &[u16], key: &[u16], value: &[u16], a_proj: &[u16], b_proj: &[u16], a_log: &[f32], dt_bias: &[f32], state: &mut [f32], mut output: DisjointSlice<u16>, num_heads: u32, head_k_dim: u32, head_v_dim: u32)
    /// ```
    pub fn launch_gdn_recurrent_step_bf16(
        &self,
        stream: &Arc<CudaStream>,
        query: &CudaSlice<half::bf16>,
        key: &CudaSlice<half::bf16>,
        value: &CudaSlice<half::bf16>,
        a_proj: &CudaSlice<half::bf16>,
        b_proj: &CudaSlice<half::bf16>,
        a_log: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        output: &mut CudaSlice<half::bf16>,
        num_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
    ) -> anyhow::Result<()> {
        let total = (num_heads as usize) * (head_v_dim as usize);

        let q_len_val = query.len() as u64;
        let k_len_val = key.len() as u64;
        let v_len_val = value.len() as u64;
        let ap_len_val = a_proj.len() as u64;
        let bp_len_val = b_proj.len() as u64;
        let al_len_val = a_log.len() as u64;
        let db_len_val = dt_bias.len() as u64;
        let st_len_val = state.len() as u64;
        let o_len_val = output.len() as u64;

        let (q_ptr, q_guard) = query.device_ptr(stream);
        let (k_ptr, k_guard) = key.device_ptr(stream);
        let (v_ptr, v_guard) = value.device_ptr(stream);
        let (ap_ptr, ap_guard) = a_proj.device_ptr(stream);
        let (bp_ptr, bp_guard) = b_proj.device_ptr(stream);
        let (al_ptr, al_guard) = a_log.device_ptr(stream);
        let (db_ptr, db_guard) = dt_bias.device_ptr(stream);
        let (st_ptr, st_guard) = state.device_ptr_mut(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);

        let _guards = (q_guard, k_guard, v_guard, ap_guard, bp_guard, al_guard, db_guard, st_guard, o_guard);

        let mut q_ptr = q_ptr as cuda_core::sys::CUdeviceptr;
        let mut k_ptr = k_ptr as cuda_core::sys::CUdeviceptr;
        let mut v_ptr = v_ptr as cuda_core::sys::CUdeviceptr;
        let mut ap_ptr = ap_ptr as cuda_core::sys::CUdeviceptr;
        let mut bp_ptr = bp_ptr as cuda_core::sys::CUdeviceptr;
        let mut al_ptr = al_ptr as cuda_core::sys::CUdeviceptr;
        let mut db_ptr = db_ptr as cuda_core::sys::CUdeviceptr;
        let mut st_ptr = st_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut q_len = q_len_val;
        let mut k_len = k_len_val;
        let mut v_len = v_len_val;
        let mut ap_len = ap_len_val;
        let mut bp_len = bp_len_val;
        let mut al_len = al_len_val;
        let mut db_len = db_len_val;
        let mut st_len = st_len_val;
        let mut o_len = o_len_val;
        let mut num_heads_v = num_heads;
        let mut head_k_dim_v = head_k_dim;
        let mut head_v_dim_v = head_v_dim;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut q_ptr, &mut q_len);   // query: &[u16]
        Self::push_slice_arg(&mut args, &mut k_ptr, &mut k_len);   // key: &[u16]
        Self::push_slice_arg(&mut args, &mut v_ptr, &mut v_len);   // value: &[u16]
        Self::push_slice_arg(&mut args, &mut ap_ptr, &mut ap_len); // a_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut bp_ptr, &mut bp_len); // b_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut al_ptr, &mut al_len); // a_log: &[f32]
        Self::push_slice_arg(&mut args, &mut db_ptr, &mut db_len); // dt_bias: &[f32]
        Self::push_slice_arg(&mut args, &mut st_ptr, &mut st_len); // state: &mut [f32]
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);   // mut output: DisjointSlice<u16>
        Self::push_scalar_arg(&mut args, &mut num_heads_v);        // num_heads: u32
        Self::push_scalar_arg(&mut args, &mut head_k_dim_v);       // head_k_dim: u32
        Self::push_scalar_arg(&mut args, &mut head_v_dim_v);       // head_v_dim: u32

        let config = LaunchConfig::for_num_elems(total as u32);
        self.raw_launch("infers_gdn_recurrent_step_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_gdn_mamba2_update_bf16` kernel: GDN Mamba-2 update.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_gdn_mamba2_update_bf16(x_proj: &[u16], b_proj: &[u16], dt_proj: &[u16], z_gate: &[u16], a_log: &[u16], dt_bias: &[u16], state: &mut [u16], mut output: DisjointSlice<u16>, num_heads: u32, head_dim: u32)
    /// ```
    pub fn launch_gdn_mamba2_update_bf16(
        &self,
        stream: &Arc<CudaStream>,
        x_proj: &CudaSlice<half::bf16>,
        b_proj: &CudaSlice<half::bf16>,
        dt_proj: &CudaSlice<half::bf16>,
        z_gate: &CudaSlice<half::bf16>,
        a_log: &CudaSlice<half::bf16>,
        dt_bias: &CudaSlice<half::bf16>,
        state: &mut CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        num_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<()> {
        let total = (num_heads as usize) * (head_dim as usize);

        let xp_len_val = x_proj.len() as u64;
        let bp_len_val = b_proj.len() as u64;
        let dp_len_val = dt_proj.len() as u64;
        let zg_len_val = z_gate.len() as u64;
        let al_len_val = a_log.len() as u64;
        let db_len_val = dt_bias.len() as u64;
        let st_len_val = state.len() as u64;
        let o_len_val = output.len() as u64;

        let (xp_ptr, xp_guard) = x_proj.device_ptr(stream);
        let (bp_ptr, bp_guard) = b_proj.device_ptr(stream);
        let (dp_ptr, dp_guard) = dt_proj.device_ptr(stream);
        let (zg_ptr, zg_guard) = z_gate.device_ptr(stream);
        let (al_ptr, al_guard) = a_log.device_ptr(stream);
        let (db_ptr, db_guard) = dt_bias.device_ptr(stream);
        let (st_ptr, st_guard) = state.device_ptr_mut(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);

        let _guards = (xp_guard, bp_guard, dp_guard, zg_guard, al_guard, db_guard, st_guard, o_guard);

        let mut xp_ptr = xp_ptr as cuda_core::sys::CUdeviceptr;
        let mut bp_ptr = bp_ptr as cuda_core::sys::CUdeviceptr;
        let mut dp_ptr = dp_ptr as cuda_core::sys::CUdeviceptr;
        let mut zg_ptr = zg_ptr as cuda_core::sys::CUdeviceptr;
        let mut al_ptr = al_ptr as cuda_core::sys::CUdeviceptr;
        let mut db_ptr = db_ptr as cuda_core::sys::CUdeviceptr;
        let mut st_ptr = st_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut xp_len = xp_len_val;
        let mut bp_len = bp_len_val;
        let mut dp_len = dp_len_val;
        let mut zg_len = zg_len_val;
        let mut al_len = al_len_val;
        let mut db_len = db_len_val;
        let mut st_len = st_len_val;
        let mut o_len = o_len_val;
        let mut num_heads_v = num_heads;
        let mut head_dim_v = head_dim;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut xp_ptr, &mut xp_len); // x_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut bp_ptr, &mut bp_len); // b_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut dp_ptr, &mut dp_len); // dt_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut zg_ptr, &mut zg_len); // z_gate: &[u16]
        Self::push_slice_arg(&mut args, &mut al_ptr, &mut al_len); // a_log: &[u16]
        Self::push_slice_arg(&mut args, &mut db_ptr, &mut db_len); // dt_bias: &[u16]
        Self::push_slice_arg(&mut args, &mut st_ptr, &mut st_len); // state: &mut [u16]
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);   // mut output: DisjointSlice<u16>
        Self::push_scalar_arg(&mut args, &mut num_heads_v);        // num_heads: u32
        Self::push_scalar_arg(&mut args, &mut head_dim_v);         // head_dim: u32

        let config = LaunchConfig::for_num_elems(total as u32);
        self.raw_launch("infers_gdn_mamba2_update_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_gdn_update_bf16` kernel: GDN single-token update.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_gdn_update_bf16(state: &mut [u16], mut output: DisjointSlice<u16>, a: &[u16], b: &[u16], dt: &[u16], x: &[u16], hidden_size: u32)
    /// ```
    pub fn launch_gdn_update_bf16(
        &self,
        stream: &Arc<CudaStream>,
        state: &mut CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        a: &CudaSlice<half::bf16>,
        b: &CudaSlice<half::bf16>,
        dt: &CudaSlice<half::bf16>,
        x: &CudaSlice<half::bf16>,
        hidden_size: u32,
    ) -> anyhow::Result<()> {
        let st_len_val = state.len() as u64;
        let o_len_val = output.len() as u64;
        let a_len_val = a.len() as u64;
        let b_len_val = b.len() as u64;
        let dt_len_val = dt.len() as u64;
        let x_len_val = x.len() as u64;

        let (st_ptr, st_guard) = state.device_ptr_mut(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);
        let (a_ptr, a_guard) = a.device_ptr(stream);
        let (b_ptr, b_guard) = b.device_ptr(stream);
        let (dt_ptr, dt_guard) = dt.device_ptr(stream);
        let (x_ptr, x_guard) = x.device_ptr(stream);

        let _guards = (st_guard, o_guard, a_guard, b_guard, dt_guard, x_guard);

        let mut st_ptr = st_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut a_ptr = a_ptr as cuda_core::sys::CUdeviceptr;
        let mut b_ptr = b_ptr as cuda_core::sys::CUdeviceptr;
        let mut dt_ptr = dt_ptr as cuda_core::sys::CUdeviceptr;
        let mut x_ptr = x_ptr as cuda_core::sys::CUdeviceptr;
        let mut st_len = st_len_val;
        let mut o_len = o_len_val;
        let mut a_len = a_len_val;
        let mut b_len = b_len_val;
        let mut dt_len = dt_len_val;
        let mut x_len = x_len_val;
        let mut hidden_size_v = hidden_size;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut st_ptr, &mut st_len); // state: &mut [u16]
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);   // mut output: DisjointSlice<u16>
        Self::push_slice_arg(&mut args, &mut a_ptr, &mut a_len);   // a: &[u16]
        Self::push_slice_arg(&mut args, &mut b_ptr, &mut b_len);   // b: &[u16]
        Self::push_slice_arg(&mut args, &mut dt_ptr, &mut dt_len); // dt: &[u16]
        Self::push_slice_arg(&mut args, &mut x_ptr, &mut x_len);   // x: &[u16]
        Self::push_scalar_arg(&mut args, &mut hidden_size_v);      // hidden_size: u32

        let config = LaunchConfig {
            grid_dim: (hidden_size, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (256 * 4) as u32,
        };
        self.raw_launch("infers_gdn_update_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_gdn_gated_delta_update_bf16` kernel: GDN gated delta update.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_gdn_gated_delta_update_bf16(query: &[u16], key: &[u16], value: &[u16], a_proj: &[u16], b_proj: &[u16], a_log: &[f32], dt_bias: &[f32], state: &mut [f32], mut output: DisjointSlice<u16>, num_heads: u32, head_k_dim: u32, head_v_dim: u32)
    /// ```
    pub fn launch_gdn_gated_delta_update_bf16(
        &self,
        stream: &Arc<CudaStream>,
        query: &CudaSlice<half::bf16>,
        key: &CudaSlice<half::bf16>,
        value: &CudaSlice<half::bf16>,
        a_proj: &CudaSlice<half::bf16>,
        b_proj: &CudaSlice<half::bf16>,
        a_log: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        output: &mut CudaSlice<half::bf16>,
        num_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
    ) -> anyhow::Result<()> {
        let total = (num_heads as usize) * (head_v_dim as usize);

        let q_len_val = query.len() as u64;
        let k_len_val = key.len() as u64;
        let v_len_val = value.len() as u64;
        let ap_len_val = a_proj.len() as u64;
        let bp_len_val = b_proj.len() as u64;
        let al_len_val = a_log.len() as u64;
        let db_len_val = dt_bias.len() as u64;
        let st_len_val = state.len() as u64;
        let o_len_val = output.len() as u64;

        let (q_ptr, q_guard) = query.device_ptr(stream);
        let (k_ptr, k_guard) = key.device_ptr(stream);
        let (v_ptr, v_guard) = value.device_ptr(stream);
        let (ap_ptr, ap_guard) = a_proj.device_ptr(stream);
        let (bp_ptr, bp_guard) = b_proj.device_ptr(stream);
        let (al_ptr, al_guard) = a_log.device_ptr(stream);
        let (db_ptr, db_guard) = dt_bias.device_ptr(stream);
        let (st_ptr, st_guard) = state.device_ptr_mut(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);

        let _guards = (q_guard, k_guard, v_guard, ap_guard, bp_guard, al_guard, db_guard, st_guard, o_guard);

        let mut q_ptr = q_ptr as cuda_core::sys::CUdeviceptr;
        let mut k_ptr = k_ptr as cuda_core::sys::CUdeviceptr;
        let mut v_ptr = v_ptr as cuda_core::sys::CUdeviceptr;
        let mut ap_ptr = ap_ptr as cuda_core::sys::CUdeviceptr;
        let mut bp_ptr = bp_ptr as cuda_core::sys::CUdeviceptr;
        let mut al_ptr = al_ptr as cuda_core::sys::CUdeviceptr;
        let mut db_ptr = db_ptr as cuda_core::sys::CUdeviceptr;
        let mut st_ptr = st_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut q_len = q_len_val;
        let mut k_len = k_len_val;
        let mut v_len = v_len_val;
        let mut ap_len = ap_len_val;
        let mut bp_len = bp_len_val;
        let mut al_len = al_len_val;
        let mut db_len = db_len_val;
        let mut st_len = st_len_val;
        let mut o_len = o_len_val;
        let mut num_heads_v = num_heads;
        let mut head_k_dim_v = head_k_dim;
        let mut head_v_dim_v = head_v_dim;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut q_ptr, &mut q_len);   // query: &[u16]
        Self::push_slice_arg(&mut args, &mut k_ptr, &mut k_len);   // key: &[u16]
        Self::push_slice_arg(&mut args, &mut v_ptr, &mut v_len);   // value: &[u16]
        Self::push_slice_arg(&mut args, &mut ap_ptr, &mut ap_len); // a_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut bp_ptr, &mut bp_len); // b_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut al_ptr, &mut al_len); // a_log: &[f32]
        Self::push_slice_arg(&mut args, &mut db_ptr, &mut db_len); // dt_bias: &[f32]
        Self::push_slice_arg(&mut args, &mut st_ptr, &mut st_len); // state: &mut [f32]
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);   // mut output: DisjointSlice<u16>
        Self::push_scalar_arg(&mut args, &mut num_heads_v);        // num_heads: u32
        Self::push_scalar_arg(&mut args, &mut head_k_dim_v);       // head_k_dim: u32
        Self::push_scalar_arg(&mut args, &mut head_v_dim_v);       // head_v_dim: u32

        let config = LaunchConfig::for_num_elems(total as u32);
        self.raw_launch("infers_gdn_gated_delta_update_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_gdn_gated_delta_prefill_bf16` kernel: GDN gated delta prefill.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_gdn_gated_delta_prefill_bf16(query: &[u16], key: &[u16], value: &[u16], a_proj: &[u16], b_proj: &[u16], a_log: &[f32], dt_bias: &[f32], state: &mut [f32], mut output: DisjointSlice<u16>, seq_len: u32, num_heads: u32, head_k_dim: u32, head_v_dim: u32)
    /// ```
    pub fn launch_gdn_gated_delta_prefill_bf16(
        &self,
        stream: &Arc<CudaStream>,
        query: &CudaSlice<half::bf16>,
        key: &CudaSlice<half::bf16>,
        value: &CudaSlice<half::bf16>,
        a_proj: &CudaSlice<half::bf16>,
        b_proj: &CudaSlice<half::bf16>,
        a_log: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        output: &mut CudaSlice<half::bf16>,
        seq_len: u32,
        num_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
    ) -> anyhow::Result<()> {
        let total = (num_heads as usize) * (head_v_dim as usize);

        let q_len_val = query.len() as u64;
        let k_len_val = key.len() as u64;
        let v_len_val = value.len() as u64;
        let ap_len_val = a_proj.len() as u64;
        let bp_len_val = b_proj.len() as u64;
        let al_len_val = a_log.len() as u64;
        let db_len_val = dt_bias.len() as u64;
        let st_len_val = state.len() as u64;
        let o_len_val = output.len() as u64;

        let (q_ptr, q_guard) = query.device_ptr(stream);
        let (k_ptr, k_guard) = key.device_ptr(stream);
        let (v_ptr, v_guard) = value.device_ptr(stream);
        let (ap_ptr, ap_guard) = a_proj.device_ptr(stream);
        let (bp_ptr, bp_guard) = b_proj.device_ptr(stream);
        let (al_ptr, al_guard) = a_log.device_ptr(stream);
        let (db_ptr, db_guard) = dt_bias.device_ptr(stream);
        let (st_ptr, st_guard) = state.device_ptr_mut(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);

        let _guards = (q_guard, k_guard, v_guard, ap_guard, bp_guard, al_guard, db_guard, st_guard, o_guard);

        let mut q_ptr = q_ptr as cuda_core::sys::CUdeviceptr;
        let mut k_ptr = k_ptr as cuda_core::sys::CUdeviceptr;
        let mut v_ptr = v_ptr as cuda_core::sys::CUdeviceptr;
        let mut ap_ptr = ap_ptr as cuda_core::sys::CUdeviceptr;
        let mut bp_ptr = bp_ptr as cuda_core::sys::CUdeviceptr;
        let mut al_ptr = al_ptr as cuda_core::sys::CUdeviceptr;
        let mut db_ptr = db_ptr as cuda_core::sys::CUdeviceptr;
        let mut st_ptr = st_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut q_len = q_len_val;
        let mut k_len = k_len_val;
        let mut v_len = v_len_val;
        let mut ap_len = ap_len_val;
        let mut bp_len = bp_len_val;
        let mut al_len = al_len_val;
        let mut db_len = db_len_val;
        let mut st_len = st_len_val;
        let mut o_len = o_len_val;
        let mut seq_len_v = seq_len;
        let mut num_heads_v = num_heads;
        let mut head_k_dim_v = head_k_dim;
        let mut head_v_dim_v = head_v_dim;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut q_ptr, &mut q_len);   // query: &[u16]
        Self::push_slice_arg(&mut args, &mut k_ptr, &mut k_len);   // key: &[u16]
        Self::push_slice_arg(&mut args, &mut v_ptr, &mut v_len);   // value: &[u16]
        Self::push_slice_arg(&mut args, &mut ap_ptr, &mut ap_len); // a_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut bp_ptr, &mut bp_len); // b_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut al_ptr, &mut al_len); // a_log: &[f32]
        Self::push_slice_arg(&mut args, &mut db_ptr, &mut db_len); // dt_bias: &[f32]
        Self::push_slice_arg(&mut args, &mut st_ptr, &mut st_len); // state: &mut [f32]
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);   // mut output: DisjointSlice<u16>
        Self::push_scalar_arg(&mut args, &mut seq_len_v);          // seq_len: u32
        Self::push_scalar_arg(&mut args, &mut num_heads_v);        // num_heads: u32
        Self::push_scalar_arg(&mut args, &mut head_k_dim_v);       // head_k_dim: u32
        Self::push_scalar_arg(&mut args, &mut head_v_dim_v);       // head_v_dim: u32

        let config = LaunchConfig::for_num_elems(total as u32);
        self.raw_launch("infers_gdn_gated_delta_prefill_bf16", stream, config, &mut args)
    }

    /// Launch the `infers_gdn_chunked_gated_delta_prefill_bf16` kernel: GDN chunked gated delta prefill.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_gdn_chunked_gated_delta_prefill_bf16(query: &[u16], key: &[u16], value: &[u16], a_proj: &[u16], b_proj: &[u16], a_log: &[f32], dt_bias: &[f32], state: &mut [f32], mut output: DisjointSlice<u16>, seq_len: u32, num_heads: u32, head_k_dim: u32, head_v_dim: u32, chunk_size: u32)
    /// ```
    pub fn launch_gdn_chunked_gated_delta_prefill_bf16(
        &self,
        stream: &Arc<CudaStream>,
        query: &CudaSlice<half::bf16>,
        key: &CudaSlice<half::bf16>,
        value: &CudaSlice<half::bf16>,
        a_proj: &CudaSlice<half::bf16>,
        b_proj: &CudaSlice<half::bf16>,
        a_log: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        output: &mut CudaSlice<half::bf16>,
        seq_len: u32,
        num_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
        chunk_size: u32,
    ) -> anyhow::Result<()> {
        let q_len_val = query.len() as u64;
        let k_len_val = key.len() as u64;
        let v_len_val = value.len() as u64;
        let ap_len_val = a_proj.len() as u64;
        let bp_len_val = b_proj.len() as u64;
        let al_len_val = a_log.len() as u64;
        let db_len_val = dt_bias.len() as u64;
        let st_len_val = state.len() as u64;
        let o_len_val = output.len() as u64;

        let (q_ptr, q_guard) = query.device_ptr(stream);
        let (k_ptr, k_guard) = key.device_ptr(stream);
        let (v_ptr, v_guard) = value.device_ptr(stream);
        let (ap_ptr, ap_guard) = a_proj.device_ptr(stream);
        let (bp_ptr, bp_guard) = b_proj.device_ptr(stream);
        let (al_ptr, al_guard) = a_log.device_ptr(stream);
        let (db_ptr, db_guard) = dt_bias.device_ptr(stream);
        let (st_ptr, st_guard) = state.device_ptr_mut(stream);
        let (o_ptr, o_guard) = output.device_ptr_mut(stream);

        let _guards = (q_guard, k_guard, v_guard, ap_guard, bp_guard, al_guard, db_guard, st_guard, o_guard);

        let mut q_ptr = q_ptr as cuda_core::sys::CUdeviceptr;
        let mut k_ptr = k_ptr as cuda_core::sys::CUdeviceptr;
        let mut v_ptr = v_ptr as cuda_core::sys::CUdeviceptr;
        let mut ap_ptr = ap_ptr as cuda_core::sys::CUdeviceptr;
        let mut bp_ptr = bp_ptr as cuda_core::sys::CUdeviceptr;
        let mut al_ptr = al_ptr as cuda_core::sys::CUdeviceptr;
        let mut db_ptr = db_ptr as cuda_core::sys::CUdeviceptr;
        let mut st_ptr = st_ptr as cuda_core::sys::CUdeviceptr;
        let mut o_ptr = o_ptr as cuda_core::sys::CUdeviceptr;
        let mut q_len = q_len_val;
        let mut k_len = k_len_val;
        let mut v_len = v_len_val;
        let mut ap_len = ap_len_val;
        let mut bp_len = bp_len_val;
        let mut al_len = al_len_val;
        let mut db_len = db_len_val;
        let mut st_len = st_len_val;
        let mut o_len = o_len_val;
        let mut seq_len_v = seq_len;
        let mut num_heads_v = num_heads;
        let mut head_k_dim_v = head_k_dim;
        let mut head_v_dim_v = head_v_dim;
        let mut chunk_size_v = chunk_size;

        let mut args: Vec<*mut std::ffi::c_void> = Vec::new();
        Self::push_slice_arg(&mut args, &mut q_ptr, &mut q_len);   // query: &[u16]
        Self::push_slice_arg(&mut args, &mut k_ptr, &mut k_len);   // key: &[u16]
        Self::push_slice_arg(&mut args, &mut v_ptr, &mut v_len);   // value: &[u16]
        Self::push_slice_arg(&mut args, &mut ap_ptr, &mut ap_len); // a_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut bp_ptr, &mut bp_len); // b_proj: &[u16]
        Self::push_slice_arg(&mut args, &mut al_ptr, &mut al_len); // a_log: &[f32]
        Self::push_slice_arg(&mut args, &mut db_ptr, &mut db_len); // dt_bias: &[f32]
        Self::push_slice_arg(&mut args, &mut st_ptr, &mut st_len); // state: &mut [f32]
        Self::push_slice_arg(&mut args, &mut o_ptr, &mut o_len);   // mut output: DisjointSlice<u16>
        Self::push_scalar_arg(&mut args, &mut seq_len_v);          // seq_len: u32
        Self::push_scalar_arg(&mut args, &mut num_heads_v);        // num_heads: u32
        Self::push_scalar_arg(&mut args, &mut head_k_dim_v);       // head_k_dim: u32
        Self::push_scalar_arg(&mut args, &mut head_v_dim_v);       // head_v_dim: u32
        Self::push_scalar_arg(&mut args, &mut chunk_size_v);       // chunk_size: u32

        let c = chunk_size as usize;
        let k = head_k_dim as usize;
        let smem_f32s = 2 * c * k + c * c + 3 * c;
        let shared_mem_bytes = (smem_f32s * 4) as u32;

        let config = LaunchConfig {
            grid_dim: (num_heads, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes,
        };
        self.raw_launch("infers_gdn_chunked_gated_delta_prefill_bf16", stream, config, &mut args)
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

   /// All 38 kernel names compiled into the oxide_kernels.cubin file.
const KERNEL_NAMES: [&str; 39] = [
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
    "int4_gemm_auto_round_tiled",
      "int4_gemm_auto_round_ksplit",
    "int4_gemm_warp",
    "int4_gemm_warp_split",
    "reduce_partial_sums_bf16",
    "nvfp4_gemm_fused_ksplit",
    "int4_dequant_to_bf16",
    "nvfp4_dequant_to_bf16",
    "nvfp4_gemm_fused",
    "sanitize_nan_bf16",
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
    "bf16_gemm_tiled",
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
        let oxide = OxideKernels::new(0, cubin_path).unwrap();

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
