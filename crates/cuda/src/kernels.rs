//! Kernel registry and loading infrastructure.
//!
//! Loads pre-compiled `.cubin` files and extracts kernel function handles.

/// A handle to a loaded CUDA kernel function.
///
/// Stores the module name and function name for identification.
/// Actual GPU function handle requires the `cuda` feature.
#[derive(Debug, Clone)]
pub struct KernelHandle {
    /// Name of the kernel function (e.g., "chunk_gated_delta_rule").
    pub name: String,
    /// Path to the .cubin file this kernel was loaded from.
    pub cubin_path: String,
}

/// Registry of loaded CUDA kernels.
///
/// Maps kernel names to their handles, supporting dynamic loading
/// of pre-compiled .cubin files.
#[derive(Debug, Clone)]
pub struct KernelRegistry {
    /// Loaded kernel handles indexed by name.
    kernels: std::collections::HashMap<String, KernelHandle>,
}

impl KernelRegistry {
    /// Create an empty kernel registry.
    pub fn new() -> Self {
        Self {
            kernels: std::collections::HashMap::new(),
        }
    }

    /// Register a kernel by name and cubin path.
    /// Actual GPU loading requires the `cuda` feature.
    pub fn register(&mut self, name: impl Into<String>, cubin_path: impl Into<String>) {
        let name = name.into();
        let cubin_path = cubin_path.into();
        self.kernels.insert(name.clone(), KernelHandle { name, cubin_path });
    }

    /// Get a kernel handle by name.
    pub fn get(&self, name: &str) -> Option<&KernelHandle> {
        self.kernels.get(name)
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Register the standard set of FlashInfer kernels.
    /// Paths point to .cubin files in the kernels/compiled/ directory.
    pub fn register_flashinfer_kernels(&mut self) {
        self.register("gdn_prefill", "kernels/compiled/gdn_prefill.cubin");
        self.register("gdn_decode", "kernels/compiled/gdn_decode.cubin");
        self.register("batch_prefill", "kernels/compiled/batch_prefill.cubin");
        self.register("batch_decode", "kernels/compiled/batch_decode.cubin");
        self.register("sampling_topk", "kernels/compiled/sampling_topk.cubin");
        self.register("sampling_topp", "kernels/compiled/sampling_topp.cubin");
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use cudarc::driver::{CudaContext, CudaModule};
    use cudarc::nvrtc::Ptx;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// GPU-loaded kernel registry that holds actual CUDA module and function handles.
    #[allow(dead_code)]
    pub struct LoadedKernelRegistry {
        /// Map from kernel name to (module, function_name).
        modules: HashMap<String, (Arc<CudaModule>, String)>,
        /// The CUDA context these kernels are loaded into.
        _ctx: Arc<CudaContext>,
    }

    impl LoadedKernelRegistry {
        /// Load all kernels from a KernelRegistry into GPU memory.
        pub fn load_all(
            ctx: Arc<CudaContext>,
            registry: &KernelRegistry,
        ) -> anyhow::Result<Self> {
            let mut modules = HashMap::new();
            for (name, handle) in &registry.kernels {
                let cubin_bytes = std::fs::read(&handle.cubin_path)?;
                let ptx = Ptx::from_binary(cubin_bytes);
                let module = ctx.load_module(ptx)?;
                modules.insert(name.clone(), (module, handle.name.clone()));
            }
            Ok(Self { modules, _ctx: ctx })
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::LoadedKernelRegistry;
