//! Bridge module for loading cuda-oxide compiled kernels and launching them
//! using cudarc buffers and streams.
//!
//! Loads a pre-compiled `.cubin` file via `cuda-core`'s `CudaContext::load_module_from_file`,
//! resolves kernel function handles, and provides type-safe launch wrappers that accept
//! cudarc `CudaSlice<T>` buffers.

use std::mem::ManuallyDrop;
use std::marker::PhantomData;
use std::sync::Arc;

use cuda_core::sys;
use cuda_core::{CudaContext, CudaModule, LaunchConfig, DeviceBuffer};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, SyncOnDrop};

use crate::modules::KernelModules;

/// Non-owning bridge from a cudarc CudaSlice to a cuda-core DeviceBuffer.
/// Wraps the same device pointer without taking ownership. The SyncOnDrop
/// guard keeps the cudarc stream synchronization alive.
///
/// `T` is the CudaSlice element type (e.g., half::bf16).
/// `U` is the DeviceBuffer element type expected by the typed kernel method (e.g., u16).
pub(crate) struct CudaSliceView<'a, T, U> {
    db: ManuallyDrop<DeviceBuffer<U>>,
    _guard: SyncOnDrop<'a>,
    _marker: PhantomData<T>,
}

impl<'a, T, U> CudaSliceView<'a, T, U> {
    pub fn new(slice: &'a CudaSlice<T>, stream: &'a Arc<CudaStream>, ctx: &Arc<CudaContext>) -> Self {
        let (ptr, guard) = slice.device_ptr(stream);
        let db = unsafe { DeviceBuffer::from_raw_parts(ptr as _, slice.len(), ctx.clone()) };
        Self { db: ManuallyDrop::new(db), _guard: guard, _marker: PhantomData }
    }

    pub fn new_mut(slice: &'a mut CudaSlice<T>, stream: &'a Arc<CudaStream>, ctx: &Arc<CudaContext>) -> Self {
        let len = slice.len();
        let (ptr, guard) = slice.device_ptr_mut(stream);
        let db = unsafe { DeviceBuffer::from_raw_parts(ptr as _, len, ctx.clone()) };
        Self { db: ManuallyDrop::new(db), _guard: guard, _marker: PhantomData }
    }
}

impl<'a, T, U> std::ops::Deref for CudaSliceView<'a, T, U> {
    type Target = DeviceBuffer<U>;
    fn deref(&self) -> &DeviceBuffer<U> { &self.db }
}

impl<'a, T, U> std::ops::DerefMut for CudaSliceView<'a, T, U> {
    fn deref_mut(&mut self) -> &mut DeviceBuffer<U> { &mut self.db }
}

/// Pre-loaded cuda-oxide kernels from a compiled cubin file.
pub struct OxideKernels {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    /// Typed kernel modules for direct dispatch (avoids raw arg packing).
    modules: KernelModules,
    /// cuda-core stream used by typed module dispatch. This is the null/default
    /// stream (stream 0). StreamPool must also use default streams so both
    /// systems share the null stream — non-blocking streams do NOT synchronize
    /// with the null stream.

    cc_stream: Arc<cuda_core::CudaStream>,
}

impl OxideKernels {
    /// Load all kernels from the given cubin file path.
    pub fn new(ordinal: usize, cubin_path: &str) -> anyhow::Result<Self> {
        // Create cuda-oxide context on the specified device (primary context shared with cudarc)
        let ctx = CudaContext::new(ordinal)?;

        // Save the current thread context so we can restore it after loading.
        // OxideKernels::new() is often called in a loop for multiple GPUs, and
        // bind_to_thread() changes the current context. Leaving GPU N's context
        // active breaks cuBLASLt which was initialized with a different context.
        let saved_ctx = {
            let mut current = std::mem::MaybeUninit::uninit();
            let result = unsafe { sys::cuCtxGetCurrent(current.as_mut_ptr()) };
            if result != 0 {
                anyhow::bail!("cuCtxGetCurrent failed (error code {})", result);
            }
            unsafe { current.assume_init() }
        };

        // Bind the context to the current thread before loading
        ctx.bind_to_thread()?;

        // Load the pre-compiled cubin
        let module = ctx.load_module_from_file(cubin_path)?;
        // Set max dynamic shared memory for chunked GDN kernel (~80KB needed, exceeds 48KB default)
        let chunked_gdn_func = module.load_function("infers_gdn_chunked_gated_delta_prefill_bf16")?;
        {
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

        // Create typed kernel modules from the same loaded module (no double-load)
        let modules = KernelModules::from_module(module.clone())?;

        // Create cuda-core default stream for typed module dispatch.
        // StreamPool also uses default streams so both share the null stream.
        // This ensures cross-library synchronization: cudarc memcpy_dtod calls
        // on the null stream are correctly ordered with kernel dispatches on
        // the same null stream. Using a separate non-blocking stream would
        // break ordering between cudarc operations and cuda-core kernel launches.
        let cc_stream = ctx.default_stream();

        // Restore the previous thread context.
        if !saved_ctx.is_null() {
            let _ = unsafe { sys::cuCtxSetCurrent(saved_ctx) };
        }

        Ok(Self { ctx, module, modules, cc_stream })
    }


    /// Access to typed kernel modules for direct dispatch.
    pub fn modules(&self) -> &KernelModules {
        &self.modules
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
        dispatch_stream: &cuda_core::CudaStream,
        a: &CudaSlice<half::bf16>,
        b: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
    ) -> anyhow::Result<()> {
        let n = a.len() as u32;

        // Create CudaSliceViews wrapping the cudarc slices
        let a_view = CudaSliceView::new(&a, stream, &self.ctx);
        let b_view = CudaSliceView::new(&b, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig::for_num_elems(n);
        self.modules.common.infers_add_bf16(
            dispatch_stream, config, &a_view, &b_view, &mut output_view, n,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_add_bf16' failed: {:?}", e))?;

        Ok(())
    }

    /// Launch the `infers_repeat_interleave_bf16` kernel: replicate each head kv_ratio times.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_repeat_interleave_bf16(src: &[u16], mut dst: DisjointSlice<u16>, seq_len: u32, num_src_heads: u32, head_dim: u32, kv_ratio: u32)
    /// ```
    pub fn launch_repeat_interleave_bf16(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        src: &CudaSlice<half::bf16>,
        dst: &mut CudaSlice<half::bf16>,
        seq_len: u32,
        num_src_heads: u32,
        head_dim: u32,
        kv_ratio: u32,
    ) -> anyhow::Result<()> {
        let num_dst_heads = (num_src_heads as usize) * (kv_ratio as usize);
        let total = (seq_len as usize) * num_dst_heads * (head_dim as usize);

        // Create CudaSliceViews wrapping the cudarc slices
        let src_view = CudaSliceView::new(&src, stream, &self.ctx);
        let mut dst_view = CudaSliceView::new_mut(dst, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig::for_num_elems(total as u32);
        self.modules.common.infers_repeat_interleave_bf16(
            dispatch_stream, config, &src_view, &mut dst_view, seq_len, num_src_heads, head_dim, kv_ratio,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_repeat_interleave_bf16' failed: {:?}", e))?;

        Ok(())
    }

    /// Launch the `infers_split_qgate_bf16` kernel: split interleaved Q+gate into separate buffers.

    /// Replaces 2×num_heads per-head memcpy calls with a single kernel launch.
    /// Grid: LaunchConfig::for_num_elems(num_heads * head_dim * 2), Block: (256, 1, 1).
    pub fn launch_split_qgate_bf16(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        q_full: &CudaSlice<half::bf16>,
        q_buf: &mut CudaSlice<half::bf16>,
        gate_buf: &mut CudaSlice<half::bf16>,
        num_heads: u32,
        head_dim: u32,
    ) -> anyhow::Result<()> {
        let q_full_view = CudaSliceView::new(&q_full, stream, &self.ctx);
        let mut q_buf_view = CudaSliceView::new_mut(q_buf, stream, &self.ctx);
        let mut gate_buf_view = CudaSliceView::new_mut(gate_buf, stream, &self.ctx);

        let total = (num_heads as u32) * (head_dim as u32) * 2;
        let config = LaunchConfig::for_num_elems(total);
        self.modules.common.infers_split_qgate_bf16(
            dispatch_stream, config, &q_full_view, &mut q_buf_view, &mut gate_buf_view,
            num_heads, head_dim,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_split_qgate_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
        weight: &CudaSlice<half::bf16>,
        token_ids: &CudaSlice<i32>,
        output: &mut CudaSlice<half::bf16>,
        seq_len: u32,
        hidden_size: u32,
    ) -> anyhow::Result<()> {
        let total = (seq_len as usize) * (hidden_size as usize);

        // Create CudaSliceViews wrapping the cudarc slices
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let token_ids_view = CudaSliceView::new(&token_ids, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig::for_num_elems(total as u32);
        self.modules.common.infers_embedding_gather_bf16(
            dispatch_stream, config, &weight_view, &token_ids_view, &mut output_view, seq_len, hidden_size,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_embedding_gather_bf16' failed: {:?}", e))?;

        Ok(())
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
        dispatch_stream: &cuda_core::CudaStream,
        x: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        total: u32,
    ) -> anyhow::Result<()> {
        let n = total;

        // Create CudaSliceViews wrapping the cudarc slices
        let x_view = CudaSliceView::new(&x, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig::for_num_elems(n);
        self.modules.activation.infers_silu_bf16(
            dispatch_stream, config, &x_view, &mut output_view, n,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_silu_bf16' failed: {:?}", e))?;

        Ok(())
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
        dispatch_stream: &cuda_core::CudaStream,
        x: &CudaSlice<half::bf16>,
        gate: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        total: u32,
    ) -> anyhow::Result<()> {
        let n = total;

        // Create CudaSliceViews wrapping the cudarc slices
        let x_view = CudaSliceView::new(&x, stream, &self.ctx);
        let gate_view = CudaSliceView::new(&gate, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig::for_num_elems(n);
        self.modules.activation.infers_silu_glu_bf16(
            dispatch_stream, config, &x_view, &gate_view, &mut output_view, n,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_silu_glu_bf16' failed: {:?}", e))?;

        Ok(())
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
        dispatch_stream: &cuda_core::CudaStream,
        x: &CudaSlice<half::bf16>,
        gate: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        total: u32,
    ) -> anyhow::Result<()> {
        let n = total;

        // Create CudaSliceViews wrapping the cudarc slices
        let x_view = CudaSliceView::new(&x, stream, &self.ctx);
        let gate_view = CudaSliceView::new(&gate, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig::for_num_elems(n);
        self.modules.activation.infers_attn_output_gate_bf16(
            dispatch_stream, config, &x_view, &gate_view, &mut output_view, n,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_attn_output_gate_bf16' failed: {:?}", e))?;

        Ok(())
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
        dispatch_stream: &cuda_core::CudaStream,
        logits: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<i32>,
        batch_size: u32,
        vocab_size: u32,
    ) -> anyhow::Result<()> {
        // Create CudaSliceViews wrapping the cudarc slices
        let logits_view = CudaSliceView::new(&logits, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // 2 static shared arrays of 256 f32s each = 2048 bytes
        let config = LaunchConfig {
            grid_dim: (batch_size, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4 * 2,
        };

        // Dispatch via typed module
        self.modules.common.infers_argmax_bf16(
            dispatch_stream, config, &logits_view, &mut output_view, batch_size, vocab_size,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_argmax_bf16' failed: {:?}", e))?;

        Ok(())
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
        dispatch_stream: &cuda_core::CudaStream,
        k: &CudaSlice<half::bf16>,
        v: &CudaSlice<half::bf16>,
        kv_cache: &mut CudaSlice<half::bf16>,
        positions: &CudaSlice<i32>,
        seq_len: u32,
        head_dim: u32,
        max_seq_len: u32,
    ) -> anyhow::Result<()> {
        let total = (seq_len as usize) * (head_dim as usize);

        let k_view = CudaSliceView::new(&k, stream, &self.ctx);
        let v_view = CudaSliceView::new(&v, stream, &self.ctx);
        let mut kv_cache_view = CudaSliceView::new_mut(kv_cache, stream, &self.ctx);
        let positions_view = CudaSliceView::new(&positions, stream, &self.ctx);

        let config = LaunchConfig::for_num_elems(total as u32);
        self.modules.common.infers_kv_cache_write_bf16(
            dispatch_stream, config, &k_view, &v_view, &mut kv_cache_view, &positions_view, seq_len, head_dim, max_seq_len,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_kv_cache_write_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
        x: &CudaSlice<half::bf16>,
        weight: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        hidden: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        let num_rows = x.len() / hidden as usize;
        let block_size = (hidden.min(512)) as u32;

        // Create CudaSliceViews wrapping the cudarc slices
        let x_view = CudaSliceView::new(&x, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };

        self.modules.norm.infers_rmsnorm_bf16(
            dispatch_stream, config, &x_view, &weight_view, &mut output_view, hidden, eps,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_rmsnorm_bf16' failed: {:?}", e))?;

        Ok(())
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
        dispatch_stream: &cuda_core::CudaStream,
        input: &CudaSlice<half::bf16>,
        gate: &CudaSlice<half::bf16>,
        weight: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        n: u32,
        d: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        let block_size = (d.min(512)) as u32;

        // Create CudaSliceViews wrapping the cudarc slices
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);
        let gate_view = CudaSliceView::new(&gate, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig {
            grid_dim: (n, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };

        self.modules.norm.infers_rms_norm_gated_bf16(
            dispatch_stream, config, &input_view, &gate_view, &weight_view, &mut output_view, n, d, eps,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_rms_norm_gated_bf16' failed: {:?}", e))?;

        Ok(())
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
        dispatch_stream: &cuda_core::CudaStream,
        input: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        dim: u32,
        eps: f32,
    ) -> anyhow::Result<()> {
        let num_rows = input.len() as u32 / dim;
        let block_size = (dim.min(512)) as u32;

        // Create CudaSliceViews wrapping the cudarc slices
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig {
            grid_dim: (num_rows, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };

        self.modules.norm.infers_l2norm_bf16(
            dispatch_stream, config, &input_view, &mut output_view, dim, eps,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_l2norm_bf16' failed: {:?}", e))?;

        Ok(())
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
        dispatch_stream: &cuda_core::CudaStream,
        scores: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        seq_len: u32,
        use_causal: u32,
    ) -> anyhow::Result<()> {
        let num_rows = scores.len() as u32 / seq_len;
        let block_size: u32 = 256;

        // Create CudaSliceViews wrapping the cudarc slices
        let scores_view = CudaSliceView::new(&scores, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig {
            grid_dim: (num_rows, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: (block_size * 4) as u32,
        };

        self.modules.common.infers_softmax_bf16(
            dispatch_stream, config, &scores_view, &mut output_view, seq_len, use_causal,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_softmax_bf16' failed: {:?}", e))?;

        Ok(())
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
        dispatch_stream: &cuda_core::CudaStream,
        input: &CudaSlice<half::bf16>,
        weight: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        batch_size: u32,
        conv_dim: u32,
        seq_len: u32,
        kernel_size: u32,
    ) -> anyhow::Result<()> {
        let total = (batch_size as usize) * (seq_len as usize) * (conv_dim as usize);

        let input_view = CudaSliceView::new(&input, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        let config = LaunchConfig::for_num_elems(total as u32);
        self.modules.activation.infers_conv1d_depthwise_silu_bf16(
            dispatch_stream, config, &input_view, &weight_view, &mut output_view, batch_size, conv_dim, seq_len, kernel_size,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_conv1d_depthwise_silu_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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

        let mut q_view = CudaSliceView::new_mut(q, stream, &self.ctx);
        let mut k_view = CudaSliceView::new_mut(k, stream, &self.ctx);
        let cos_view = CudaSliceView::new(&cos, stream, &self.ctx);
        let sin_view = CudaSliceView::new(&sin, stream, &self.ctx);
        let positions_view = CudaSliceView::new(&positions, stream, &self.ctx);

        let config = LaunchConfig::for_num_elems(total as u32);
        self.modules.attention.infers_rope_bf16(
            dispatch_stream, config, &mut q_view, &mut k_view, &cos_view, &sin_view, &positions_view, total_tokens, num_heads, head_dim, rotary_dim,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_rope_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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

        let k_view = CudaSliceView::new(&k, stream, &self.ctx);
        let v_view = CudaSliceView::new(&v, stream, &self.ctx);
        let mut page_pool_view = CudaSliceView::new_mut(page_pool, stream, &self.ctx);
        let block_table_view = CudaSliceView::new(&block_table, stream, &self.ctx);
        let positions_view = CudaSliceView::new(&positions, stream, &self.ctx);

        let config = LaunchConfig::for_num_elems(total as u32);
        self.modules.attention.infers_paged_kv_write_bf16(
            dispatch_stream, config, &k_view, &v_view, &mut page_pool_view, &block_table_view, &positions_view, seq_len, head_dim, page_size, kv_dim,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_paged_kv_write_bf16' failed: {:?}", e))
    }

    /// Launch the `infers_paged_kv_read_bf16` kernel: paged KV cache read.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_paged_kv_read_bf16(page_pool: &[u16], block_table: &[i32], _num_pages: u32, cached_tokens_count: &[u32], _head_dim: u32, page_size: u32, kv_dim: u32, mut k_out: DisjointSlice<u16>, mut v_out: DisjointSlice<u16>)
    /// ```
    pub fn launch_paged_kv_read_bf16(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        page_pool: &CudaSlice<half::bf16>,
        block_table: &CudaSlice<i32>,
        num_pages: u32,
        cached_tokens_count: &CudaSlice<u32>,
        head_dim: u32,
        page_size: u32,
        kv_dim: u32,
        k_out: &mut CudaSlice<half::bf16>,
        v_out: &mut CudaSlice<half::bf16>,
    ) -> anyhow::Result<()> {
        let page_pool_view = CudaSliceView::new(&page_pool, stream, &self.ctx);
        let block_table_view = CudaSliceView::new(&block_table, stream, &self.ctx);
        let cached_tokens_count_view = CudaSliceView::new(cached_tokens_count, stream, &self.ctx);
        let mut k_out_view = CudaSliceView::new_mut(k_out, stream, &self.ctx);
        let mut v_out_view = CudaSliceView::new_mut(v_out, stream, &self.ctx);

        // Use a fixed max total for launch config (CUDA graph compatible).
        // The kernel reads actual count from device and uses grid-stride loop.
        const MAX_KV_READ_TOTAL: u32 = 4096 * 16384; // up to 4096 tokens × kv_dim 16384
        let config = LaunchConfig::for_num_elems(MAX_KV_READ_TOTAL);
        self.modules.attention.infers_paged_kv_read_bf16(
            dispatch_stream, config, &page_pool_view, &block_table_view, num_pages, &cached_tokens_count_view, head_dim, page_size, kv_dim, &mut k_out_view, &mut v_out_view,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_paged_kv_read_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let scales_view = CudaSliceView::new(&scales, stream, &self.ctx);
        let zeros_view = CudaSliceView::new(&zeros, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

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
        self.modules.int4.int4_gemm_auto_round(
            dispatch_stream, config, &mut output_view, &weight_view, &scales_view, &zeros_view, &input_view,
            m, n, k, group_size, transposed,
        ).map_err(|e| anyhow::anyhow!("kernel 'int4_gemm_auto_round' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let scales_view = CudaSliceView::new(&scales, stream, &self.ctx);
        let zeros_view = CudaSliceView::new(&zeros, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, m, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: (group_size * 2),
        };
        self.modules.int4.int4_gemm_auto_round_tiled(
            dispatch_stream, config, &mut output_view, &weight_view, &scales_view, &zeros_view, &input_view,
            m, n, k, group_size, transposed,
        ).map_err(|e| anyhow::anyhow!("kernel 'int4_gemm_auto_round_tiled' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut ps_view = CudaSliceView::new_mut(partial_sums, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let scales_view = CudaSliceView::new(&scales, stream, &self.ctx);
        let zeros_view = CudaSliceView::new(&zeros, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, k_split, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        self.modules.int4.int4_gemm_auto_round_ksplit(
            dispatch_stream, config, &mut ps_view, &weight_view, &scales_view, &zeros_view, &input_view,
            n, k, group_size, transposed, k_split,
        ).map_err(|e| anyhow::anyhow!("kernel 'int4_gemm_auto_round_ksplit' failed: {:?}", e))
    }

    /// Launch the `int4_gemm_v3_ksplit_sm` kernel: same as v3 but with shared memory
    /// input tiling. Tiles group_size bf16 values per group into shared memory to
    /// eliminate redundant DRAM reads.
    pub fn launch_int4_gemm_v3_ksplit_sm(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut ps_view = CudaSliceView::new_mut(partial_sums, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let scales_view = CudaSliceView::new(&scales, stream, &self.ctx);
        let zeros_view = CudaSliceView::new(&zeros, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, k_split, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: (group_size as u32) * 2, // group_size bf16 values
        };
        self.modules.int4.int4_gemm_v3_ksplit_sm(
            dispatch_stream, config, &mut ps_view, &weight_view, &scales_view, &zeros_view, &input_view,
            n, k, group_size, transposed, k_split,
        ).map_err(|e| anyhow::anyhow!("kernel 'int4_gemm_v3_ksplit_sm' failed: {:?}", e))
    }
    /// Launch the `int4_gemm_v4_ksplit` kernel: 128-bit load-optimized INT4 GEMM
    /// with K-splitting (4 columns per thread, block_dim (16,1,1)).
    /// Same interface as `launch_int4_gemm_v3_ksplit`. Precondition:
    /// `k_split` must divide `k / group_size` evenly.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// int4_gemm_v4_ksplit(partial_sums: &mut [f32], weight: &[u32], scales: &[u16], zeros: &[u32], input: &[u16], n: u32, k: u32, group_size: u32, transposed: u32, k_split: u32)
    /// ```
    pub fn launch_int4_gemm_v4_ksplit(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut ps_view = CudaSliceView::new_mut(partial_sums, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let scales_view = CudaSliceView::new(&scales, stream, &self.ctx);
        let zeros_view = CudaSliceView::new(&zeros, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, k_split, 1),
            block_dim: (16, 1, 1),
            shared_mem_bytes: 0,
        };
        self.modules.int4.int4_gemm_v4_ksplit(
            dispatch_stream, config, &mut ps_view, &weight_view, &scales_view, &zeros_view, &input_view,
            n, k, group_size, transposed, k_split,
        ).map_err(|e| anyhow::anyhow!("kernel 'int4_gemm_v4_ksplit' failed: {:?}", e))
    }

    /// Launch the `int4_gemm_warp_split` kernel: warp-cooperative INT4 GEMV with K-splitting.
    /// Same interface as ksplit but uses block (32,8,1) with warp shuffle reduction.
    pub fn launch_int4_gemm_warp_split(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut ps_view = CudaSliceView::new_mut(partial_sums, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let scales_view = CudaSliceView::new(&scales, stream, &self.ctx);
        let zeros_view = CudaSliceView::new(&zeros, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 7) / 8, k_split, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        self.modules.int4.int4_gemm_warp_split(
            dispatch_stream, config, &mut ps_view, &weight_view, &scales_view, &zeros_view, &input_view,
            n, k, group_size, transposed, k_split,
        ).map_err(|e| anyhow::anyhow!("kernel 'int4_gemm_warp_split' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
        output: &mut CudaSlice<half::bf16>,
        partial_sums: &CudaSlice<f32>,
        n: u32,
        k_split: u32,
    ) -> anyhow::Result<()> {
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let partial_sums_view = CudaSliceView::new(&partial_sums, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        self.modules.int4.reduce_partial_sums_bf16(
            dispatch_stream, config, &mut output_view, &partial_sums_view, n, k_split,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'reduce_partial_sums_bf16' failed: {:?}", e))
    }

    /// Launch the `int4_gemm_v3_ksplit_sm_m` kernel: M-batched INT4 GEMM with shared memory
    /// input tiling and K-splitting. Same as v3_ksplit_sm but processes M rows simultaneously,
    /// amortizing weight bandwidth.
    pub fn launch_int4_gemm_v3_ksplit_sm_m(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        partial_sums: &mut CudaSlice<f32>,       // [k_split, N, M] f32
        weight: &CudaSlice<u32>,
        scales: &CudaSlice<half::f16>,
        zeros: &CudaSlice<u32>,
        input: &CudaSlice<half::bf16>,            // [M, K] row-major
        m: u32,
        n: u32,
        k: u32,
        group_size: u32,
        transposed: u32,
        k_split: u32,
    ) -> anyhow::Result<()> {
        let mut ps_view = CudaSliceView::new_mut(partial_sums, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let scales_view = CudaSliceView::new(&scales, stream, &self.ctx);
        let zeros_view = CudaSliceView::new(&zeros, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, k_split, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: m * group_size * 2, // M * group_size bf16 values
        };
        self.modules.int4.int4_gemm_v3_ksplit_sm_m(
            dispatch_stream, config, &mut ps_view, &weight_view, &scales_view, &zeros_view, &input_view,
            m, n, k, group_size, transposed, k_split,
        ).map_err(|e| anyhow::anyhow!("kernel 'int4_gemm_v3_ksplit_sm_m' failed: {:?}", e))
    }

    /// Launch the `reduce_partial_sums_bf16_m` kernel: reduce K-split partial sums to bf16 for M-batched GEMM.
    pub fn launch_reduce_partial_sums_bf16_m(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        output: &mut CudaSlice<half::bf16>,       // [M * N] bf16, row-major
        partial_sums: &CudaSlice<f32>,             // [k_split, N, M] f32
        n: u32,
        m: u32,
        k_split: u32,
    ) -> anyhow::Result<()> {
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let partial_sums_view = CudaSliceView::new(&partial_sums, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, m, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        self.modules.int4.reduce_partial_sums_bf16_m(
            dispatch_stream, config, &mut output_view, &partial_sums_view, n, m, k_split,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'reduce_partial_sums_bf16_m' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let scales_view = CudaSliceView::new(&scales, stream, &self.ctx);
        let zeros_view = CudaSliceView::new(&zeros, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 7) / 8, 1, 1),   // ceil(N / 8)
            block_dim: (32, 8, 1),           // 32 lanes x8 warps = 256 threads
            shared_mem_bytes: 0,
        };
        self.modules.int4.int4_gemm_warp(
            dispatch_stream, config, &mut output_view, &weight_view, &scales_view, &zeros_view, &input_view,
            n, k, group_size, transposed,
        ).map_err(|e| anyhow::anyhow!("kernel 'int4_gemm_warp' failed: {:?}", e))
    }


    /// Launch INT4 AutoRound dequantize-to-BF16 kernel.
    ///
    /// Reads packed INT4 weights + FP16 scales + packed zeros,
    /// writes bf16 output to `output` buffer.
    pub fn launch_int4_dequant_to_bf16(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        output: &mut CudaSlice<half::bf16>,     // [N, K] bf16 output
        weight: &CudaSlice<u32>,            // [N, K/8] packed INT4
        scales: &CudaSlice<half::f16>,      // [N, K/group_size] fp16 scales
        zeros: &CudaSlice<u32>,             // packed zeros
        n: u32,
        k: u32,
        group_size: u32,
    ) -> anyhow::Result<()> {
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let scales_view = CudaSliceView::new(&scales, stream, &self.ctx);
        let zeros_view = CudaSliceView::new(&zeros, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.modules.int4.int4_dequant_to_bf16(
            dispatch_stream, config, &mut output_view, &weight_view, &scales_view, &zeros_view,
            n, k, group_size,
        ).map_err(|e| anyhow::anyhow!("kernel 'int4_dequant_to_bf16' failed: {:?}", e))
    }

    /// Launch NVFP4 dequantize-to-BF16 kernel.
    /// Reads packed NVFP4 weights + FP8 scales + weight global scale scalar,
    /// writes bf16 output to `output` buffer.
    pub fn launch_nvfp4_dequant_to_bf16(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        output: &mut CudaSlice<half::bf16>,  // [N, K] bf16 output
        weight_packed: &CudaSlice<u8>,       // [N, K/2] packed FP4
        weight_scale: &CudaSlice<u8>,        // [N, K/group_size] fp8_e4m3
        weight_global_scale: f32,            // scalar global scale
        n: u32,
        k: u32,
        group_size: u32,
    ) -> anyhow::Result<()> {
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let weight_packed_view = CudaSliceView::new(&weight_packed, stream, &self.ctx);
        let weight_scale_view = CudaSliceView::new(&weight_scale, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.modules.nvfp4.nvfp4_dequant_to_bf16(
            dispatch_stream, config, &mut output_view, &weight_packed_view, &weight_scale_view,
            weight_global_scale, n, k, group_size,
        ).map_err(|e| anyhow::anyhow!("kernel 'nvfp4_dequant_to_bf16' failed: {:?}", e))
    }

    /// Launch the `nvfp4_gemm_fused` kernel: fused NVFP4 dequant + GEMM.
    ///
    /// Reads compressed FP4 weights directly from GPU, dequantizes in registers,
    /// and multiplies with BF16 activations — no intermediate bf16 buffer needed.
    pub fn launch_nvfp4_gemm_fused(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let weight_packed_view = CudaSliceView::new(&weight_packed, stream, &self.ctx);
        let weight_scale_view = CudaSliceView::new(&weight_scale, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

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
        self.modules.nvfp4.nvfp4_gemm_fused(
            dispatch_stream, config, &mut output_view, &weight_packed_view, &weight_scale_view, &input_view,
            weight_global_scale, m, n, k, group_size,
        ).map_err(|e| anyhow::anyhow!("kernel 'nvfp4_gemm_fused' failed: {:?}", e))
    }

    /// Launch the `nvfp4_gemm_fused_ksplit` kernel: fused NVFP4 GEMM with K-splitting.
    /// Each block computes partial sums for 64 output columns over a portion of K (M=1 only).
    pub fn launch_nvfp4_gemm_fused_ksplit(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut ps_view = CudaSliceView::new_mut(partial_sums, stream, &self.ctx);
        let weight_packed_view = CudaSliceView::new(&weight_packed, stream, &self.ctx);
        let weight_scale_view = CudaSliceView::new(&weight_scale, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, k_split, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        self.modules.nvfp4.nvfp4_gemm_fused_ksplit(
            dispatch_stream, config, &mut ps_view, &weight_packed_view, &weight_scale_view, &input_view,
            weight_global_scale, n, k, group_size, k_split,
        ).map_err(|e| anyhow::anyhow!("kernel 'nvfp4_gemm_fused_ksplit' failed: {:?}", e))
    }

    /// Launch the `nvfp4_gemm_v3_ksplit` kernel: fused NVFP4 GEMM v3 with K-splitting.
    /// Four independent accumulators + ceil-grouped K-split + 2-u32 stride (M=1 only).
    pub fn launch_nvfp4_gemm_v3_ksplit(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut ps_view = CudaSliceView::new_mut(partial_sums, stream, &self.ctx);
        let weight_packed_view = CudaSliceView::new(&weight_packed, stream, &self.ctx);
        let weight_scale_view = CudaSliceView::new(&weight_scale, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, k_split, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        self.modules.nvfp4.nvfp4_gemm_v3_ksplit(
            dispatch_stream, config, &mut ps_view, &weight_packed_view, &weight_scale_view, &input_view,
            weight_global_scale, n, k, group_size, k_split,
        ).map_err(|e| anyhow::anyhow!("kernel 'nvfp4_gemm_v3_ksplit' failed: {:?}", e))
    }

    /// Launch the `sanitize_nan_bf16` kernel: replace NaN values in a bf16 buffer with 0.0.
    pub fn launch_sanitize_nan_bf16(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        buf: &mut CudaSlice<half::bf16>,
    ) -> anyhow::Result<()> {
        let len_scalar = buf.len() as u32;
        let block_size = 256u32;
        let grid_size = ((buf.len() as u32 + block_size - 1) / block_size, 1, 1);

        // Create CudaSliceViews wrapping the cudarc slices
        let mut buf_view = CudaSliceView::new_mut(buf, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig {
            grid_dim: grid_size,
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        self.modules.common.sanitize_nan_bf16(
            dispatch_stream, config, &mut buf_view, len_scalar,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'sanitize_nan_bf16' failed: {:?}", e))?;

        Ok(())
    }

    /// Launch the `bf16_gemm_tiled` kernel: tiled bf16 GEMM with shared memory.
    ///
    /// Computes C[M,N] = A[M,K] @ B[N,K]^T where all buffers are row-major bf16.
    /// Used as a replacement for cuBLAS in the NVFP4 path to avoid workspace corruption.
   pub fn launch_bf16_gemm_tiled(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        output: &mut CudaSlice<half::bf16>,
        input: &CudaSlice<half::bf16>,
        weight: &CudaSlice<half::bf16>,
        m: u32,
        n: u32,
        k: u32,
    ) -> anyhow::Result<()> {
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);

        // Grid: (N/64, M/64, 1), Block: (256, 1, 1)
        let grid_x = (n + 63) / 64;
        let grid_y = (m + 63) / 64;
        let config = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,  // no shared memory used in current version
        };
        self.modules.bf16.bf16_gemm_tiled(
            dispatch_stream, config, &mut output_view, &input_view, &weight_view,
            m, n, k,
        ).map_err(|e| anyhow::anyhow!("kernel 'bf16_gemm_tiled' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let weight_view = CudaSliceView::new(&weight, stream, &self.ctx);
        let scales_view = CudaSliceView::new(&scales, stream, &self.ctx);
        let zeros_view = CudaSliceView::new(&zeros, stream, &self.ctx);
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);

        let config = LaunchConfig {
            grid_dim: ((n + 63) / 64, (m + 3) / 4, 1),
            block_dim: (64, 4, 1),
            shared_mem_bytes: 0,
        };
        self.modules.int4.int4_gemm_gguf(
            dispatch_stream, config, &mut output_view, &weight_view, &scales_view, &zeros_view, &input_view,
            m, n, k, group_size, transposed,
        ).map_err(|e| anyhow::anyhow!("kernel 'int4_gemm_gguf' failed: {:?}", e))
    }

    /// Launch the `infers_fp8_quantize_e4m3` kernel: BF16 -> FP8 E4M3.
    pub fn launch_fp8_quantize_e4m3(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        input: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<u8>,
        n: u32,
    ) -> anyhow::Result<()> {
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        let config = LaunchConfig::for_num_elems(n);
        self.modules.fp8.infers_fp8_quantize_e4m3(
            dispatch_stream, config, &input_view, &mut output_view, n,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_fp8_quantize_e4m3' failed: {:?}", e))
    }

    /// Launch the `infers_fp8_dequantize_e4m3` kernel: FP8 E4M3 -> BF16.
    pub fn launch_fp8_dequantize_e4m3(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<half::bf16>,
        n: u32,
    ) -> anyhow::Result<()> {
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        let config = LaunchConfig::for_num_elems(n);
        self.modules.fp8.infers_fp8_dequantize_e4m3(
            dispatch_stream, config, &input_view, &mut output_view, n,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_fp8_dequantize_e4m3' failed: {:?}", e))
    }

    /// Launch the `infers_fp8_quantize_e5m2` kernel: BF16 -> FP8 E5M2.
    pub fn launch_fp8_quantize_e5m2(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        input: &CudaSlice<half::bf16>,
        output: &mut CudaSlice<u8>,
        n: u32,
    ) -> anyhow::Result<()> {
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        let config = LaunchConfig::for_num_elems(n);
        self.modules.fp8.infers_fp8_quantize_e5m2(
            dispatch_stream, config, &input_view, &mut output_view, n,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_fp8_quantize_e5m2' failed: {:?}", e))
    }

    /// Launch the `infers_fp8_dequantize_e5m2` kernel: FP8 E5M2 -> BF16.
    pub fn launch_fp8_dequantize_e5m2(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        input: &CudaSlice<u8>,
        output: &mut CudaSlice<half::bf16>,
        n: u32,
    ) -> anyhow::Result<()> {
        let input_view = CudaSliceView::new(&input, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        let config = LaunchConfig::for_num_elems(n);
        self.modules.fp8.infers_fp8_dequantize_e5m2(
            dispatch_stream, config, &input_view, &mut output_view, n,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_fp8_dequantize_e5m2' failed: {:?}", e))
    }

    /// Launch the `infers_paged_attention_decode_bf16` kernel: paged attention decode.
    ///
    /// The kernel signature is:
    /// ```ignore
    /// infers_paged_attention_decode_bf16(q: &[u16], page_pool: &[u16], block_table: &[i32], num_pages: u32, cached_tokens_count: &[u32], head_dim: u32, num_kv_heads: u32, num_query_heads: u32, page_size: u32, kv_dim: u32, mut output: DisjointSlice<u16>)
    /// ```
    pub fn launch_paged_attention_decode_bf16(
        &self,
        stream: &Arc<CudaStream>,
        dispatch_stream: &cuda_core::CudaStream,
        q: &CudaSlice<half::bf16>,
        page_pool: &CudaSlice<half::bf16>,
        block_table: &CudaSlice<i32>,
        cached_tokens_count: &CudaSlice<u32>,
        output: &mut CudaSlice<half::bf16>,
        num_pages: u32,
        head_dim: u32,
        num_kv_heads: u32,
        num_query_heads: u32,
        page_size: u32,
        kv_dim: u32,
    ) -> anyhow::Result<()> {
        let q_view = CudaSliceView::new(&q, stream, &self.ctx);
        let page_pool_view = CudaSliceView::new(&page_pool, stream, &self.ctx);
        let block_table_view = CudaSliceView::new(&block_table, stream, &self.ctx);
        let cached_tokens_count_view = CudaSliceView::new(cached_tokens_count, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        const MAX_CACHED_TOKENS_FOR_SHARED_MEM: usize = 4096;
        let config = LaunchConfig {
            grid_dim: (num_kv_heads, 1, 1),
            block_dim: (256, 1, 1),
            // 3 regions: Q values + max scratch + sum scratch + cached attention weights
            shared_mem_bytes: ((3 * 256 + MAX_CACHED_TOKENS_FOR_SHARED_MEM) * 4) as u32,
        };
        self.modules.attention.infers_paged_attention_decode_bf16(
            dispatch_stream, config, &q_view, &page_pool_view, &block_table_view, num_pages, &cached_tokens_count_view, head_dim, num_kv_heads, num_query_heads, page_size, kv_dim, &mut output_view,
        ).map_err(|e| anyhow::anyhow!("kernel launch 'infers_paged_attention_decode_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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
        // Create CudaSliceViews wrapping the cudarc slices
        let query_view = CudaSliceView::new(&query, stream, &self.ctx);
        let key_view = CudaSliceView::new(&key, stream, &self.ctx);
        let value_view = CudaSliceView::new(&value, stream, &self.ctx);
        let a_proj_view = CudaSliceView::new(&a_proj, stream, &self.ctx);
        let b_proj_view = CudaSliceView::new(&b_proj, stream, &self.ctx);
        let a_log_view = CudaSliceView::new(&a_log, stream, &self.ctx);
        let dt_bias_view = CudaSliceView::new(&dt_bias, stream, &self.ctx);
        let mut state_view = CudaSliceView::new_mut(state, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // 2D grid: blockIdx.y = head, blockIdx.x tiles over v_dim
        // Shared memory: 2*K f32s for key + query caching
        let block_dim = 128u32;
        let grid_x = (head_v_dim + block_dim - 1) / block_dim;
        let smem_bytes = 2 * (head_k_dim as usize) * std::mem::size_of::<f32>();
        let config = LaunchConfig {
            grid_dim: (grid_x, num_heads, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: smem_bytes as u32,
        };
        self.modules.gdn.infers_gdn_recurrent_step_bf16(
            dispatch_stream, config,
            &query_view, &key_view, &value_view,
            &a_proj_view, &b_proj_view,
            &a_log_view, &dt_bias_view,
            &mut state_view, &mut output_view,
            num_heads, head_k_dim, head_v_dim,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_gdn_recurrent_step_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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

        // Create CudaSliceViews wrapping the cudarc slices
        let x_proj_view = CudaSliceView::new(&x_proj, stream, &self.ctx);
        let b_proj_view = CudaSliceView::new(&b_proj, stream, &self.ctx);
        let dt_proj_view = CudaSliceView::new(&dt_proj, stream, &self.ctx);
        let z_gate_view = CudaSliceView::new(&z_gate, stream, &self.ctx);
        let a_log_view = CudaSliceView::new(&a_log, stream, &self.ctx);
        let dt_bias_view = CudaSliceView::new(&dt_bias, stream, &self.ctx);
        let mut state_view = CudaSliceView::new_mut(state, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig::for_num_elems(total as u32);
        self.modules.gdn.infers_gdn_mamba2_update_bf16(
            dispatch_stream, config,
            &x_proj_view, &b_proj_view, &dt_proj_view,
            &z_gate_view, &a_log_view, &dt_bias_view,
            &mut state_view, &mut output_view,
            num_heads, head_dim,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_gdn_mamba2_update_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
        state: &mut CudaSlice<half::bf16>,
        output: &mut CudaSlice<half::bf16>,
        a: &CudaSlice<half::bf16>,
        b: &CudaSlice<half::bf16>,
        dt: &CudaSlice<half::bf16>,
        x: &CudaSlice<half::bf16>,
        hidden_size: u32,
    ) -> anyhow::Result<()> {
        // Create CudaSliceViews wrapping the cudarc slices
        let mut state_view = CudaSliceView::new_mut(state, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);
        let a_view = CudaSliceView::new(&a, stream, &self.ctx);
        let b_view = CudaSliceView::new(&b, stream, &self.ctx);
        let dt_view = CudaSliceView::new(&dt, stream, &self.ctx);
        let x_view = CudaSliceView::new(&x, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig {
            grid_dim: (hidden_size, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (256 * 4) as u32,
        };
        self.modules.gdn.infers_gdn_update_bf16(
            dispatch_stream, config,
            &mut state_view, &mut output_view,
            &a_view, &b_view, &dt_view, &x_view,
            hidden_size,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_gdn_update_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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

        // Create CudaSliceViews wrapping the cudarc slices
        let query_view = CudaSliceView::new(&query, stream, &self.ctx);
        let key_view = CudaSliceView::new(&key, stream, &self.ctx);
        let value_view = CudaSliceView::new(&value, stream, &self.ctx);
        let a_proj_view = CudaSliceView::new(&a_proj, stream, &self.ctx);
        let b_proj_view = CudaSliceView::new(&b_proj, stream, &self.ctx);
        let a_log_view = CudaSliceView::new(&a_log, stream, &self.ctx);
        let dt_bias_view = CudaSliceView::new(&dt_bias, stream, &self.ctx);
        let mut state_view = CudaSliceView::new_mut(state, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig::for_num_elems(total as u32);
        self.modules.gdn.infers_gdn_gated_delta_update_bf16(
            dispatch_stream, config,
            &query_view, &key_view, &value_view,
            &a_proj_view, &b_proj_view,
            &a_log_view, &dt_bias_view,
            &mut state_view, &mut output_view,
            num_heads, head_k_dim, head_v_dim,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_gdn_gated_delta_update_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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

        // Create CudaSliceViews wrapping the cudarc slices
        let query_view = CudaSliceView::new(&query, stream, &self.ctx);
        let key_view = CudaSliceView::new(&key, stream, &self.ctx);
        let value_view = CudaSliceView::new(&value, stream, &self.ctx);
        let a_proj_view = CudaSliceView::new(&a_proj, stream, &self.ctx);
        let b_proj_view = CudaSliceView::new(&b_proj, stream, &self.ctx);
        let a_log_view = CudaSliceView::new(&a_log, stream, &self.ctx);
        let dt_bias_view = CudaSliceView::new(&dt_bias, stream, &self.ctx);
        let mut state_view = CudaSliceView::new_mut(state, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dispatch via typed module
        let config = LaunchConfig::for_num_elems(total as u32);
        self.modules.gdn.infers_gdn_gated_delta_prefill_bf16(
            dispatch_stream, config,
            &query_view, &key_view, &value_view,
            &a_proj_view, &b_proj_view,
            &a_log_view, &dt_bias_view,
            &mut state_view, &mut output_view,
            seq_len, num_heads, head_k_dim, head_v_dim,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_gdn_gated_delta_prefill_bf16' failed: {:?}", e))
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
        dispatch_stream: &cuda_core::CudaStream,
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
        // Create CudaSliceViews wrapping the cudarc slices
        let query_view = CudaSliceView::new(&query, stream, &self.ctx);
        let key_view = CudaSliceView::new(&key, stream, &self.ctx);
        let value_view = CudaSliceView::new(&value, stream, &self.ctx);
        let a_proj_view = CudaSliceView::new(&a_proj, stream, &self.ctx);
        let b_proj_view = CudaSliceView::new(&b_proj, stream, &self.ctx);
        let a_log_view = CudaSliceView::new(&a_log, stream, &self.ctx);
        let dt_bias_view = CudaSliceView::new(&dt_bias, stream, &self.ctx);
        let mut state_view = CudaSliceView::new_mut(state, stream, &self.ctx);
        let mut output_view = CudaSliceView::new_mut(output, stream, &self.ctx);

        // Dynamic shared memory for chunked gated delta (~80KB needed)
        let c = chunk_size as usize;
        let k = head_k_dim as usize;
        let smem_f32s = 2 * c * k + c * c + 3 * c;
        let shared_mem_bytes = (smem_f32s * 4) as u32;

        // Dispatch via typed module
        let config = LaunchConfig {
            grid_dim: (num_heads, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes,
        };
        self.modules.gdn.infers_gdn_chunked_gated_delta_prefill_bf16(
            dispatch_stream, config,
            &query_view, &key_view, &value_view,
            &a_proj_view, &b_proj_view,
            &a_log_view, &dt_bias_view,
            &mut state_view, &mut output_view,
            seq_len, num_heads, head_k_dim, head_v_dim, chunk_size,
        ).map_err(|e| anyhow::anyhow!("kernel 'infers_gdn_chunked_gated_delta_prefill_bf16' failed: {:?}", e))
    }


    /// Access the underlying cuda-oxide context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Access the loaded module.
    pub fn module(&self) -> &Arc<CudaModule> {
        &self.module
    }

    /// Access the cuda-core stream used by typed module dispatch.
    pub fn cc_stream(&self) -> &cuda_core::CudaStream {
        &self.cc_stream
    }
}

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
        oxide.launch_add_bf16(&stream, &oxide.cc_stream(), &a_gpu, &b_gpu, &mut out_gpu).unwrap();

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
