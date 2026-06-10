//! Kernel registry and loading infrastructure.
//!
//! Loads pre-compiled `.cubin` files and extracts kernel function handles.

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::Arc;

/// A handle to a loaded CUDA kernel function.
///
/// Stores the kernel name and cubin path for identification.
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
    kernels: HashMap<String, KernelHandle>,
}

impl KernelRegistry {
    /// Create an empty kernel registry.
    pub fn new() -> Self {
        Self { kernels: HashMap::new() }
    }

    /// Register a kernel by name and cubin path.
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

    /// Register the standard set of infers CUDA kernels.
    /// Paths point to .cubin files in the kernels/compiled/ directory.
    pub fn register_infers_kernels(&mut self) {
        self.register("infers_rmsnorm_bf16", "kernels/compiled/rmsnorm.cubin");
        self.register("infers_silu_bf16", "kernels/compiled/silu.cubin");
        self.register("infers_silu_glu_bf16", "kernels/compiled/silu.cubin");
        self.register("infers_rope_bf16", "kernels/compiled/rope.cubin");
        self.register("infers_embedding_gather_bf16", "kernels/compiled/embedding.cubin");
        self.register("infers_add_bf16", "kernels/compiled/elementwise.cubin");
        self.register("infers_argmax_f32", "kernels/compiled/sampling.cubin");
        self.register("infers_softmax_bf16", "kernels/compiled/softmax.cubin");
        self.register("infers_kv_cache_write_bf16", "kernels/compiled/kv_cache.cubin");
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU-loaded kernel registry that holds actual CUDA module and function handles.
///
/// Deduplicates module loading so that the same .cubin file is loaded only once,
/// even when multiple kernel functions reference it.
pub struct LoadedKernelRegistry {
    /// Map from kernel name to (cubin_path, function_name).
    kernels: HashMap<String, (String, String)>,
    /// Map from cubin path to loaded module — deduplicated so same .cubin is loaded once.
    modules: HashMap<String, Arc<CudaModule>>,
    /// The CUDA context these kernels are loaded into.
    _ctx: Arc<CudaContext>,
}

impl LoadedKernelRegistry {
    /// Load all kernels from a KernelRegistry into GPU memory.
    ///
    /// Deduplicates by cubin path so that the same .cubin file is loaded only once.
    pub fn load_all(
        ctx: Arc<CudaContext>,
        registry: &KernelRegistry,
    ) -> anyhow::Result<Self> {
        let mut kernels = HashMap::new();
        let mut modules = HashMap::new();
        for (name, handle) in &registry.kernels {
            // Deduplicate: load module only once per cubin path
            if !modules.contains_key(&handle.cubin_path) {
                let cubin_bytes = std::fs::read(&handle.cubin_path)
                    .map_err(|e| anyhow::anyhow!("Failed to read {}: {:?}", handle.cubin_path, e))?;
                let ptx = Ptx::from_binary(cubin_bytes);
                let module = ctx.load_module(ptx)
                    .map_err(|e| anyhow::anyhow!("Failed to load module '{}': {:?}", handle.cubin_path, e))?;
                modules.insert(handle.cubin_path.clone(), module);
            }
            kernels.insert(name.clone(), (handle.cubin_path.clone(), handle.name.clone()));
        }
        Ok(Self { kernels, modules, _ctx: ctx })
    }

    /// Get a `CudaFunction` for a kernel by name.
    ///
    /// Looks up the deduplicated module by cubin path and loads the function from it.
    pub fn get_function(&self, name: &str) -> anyhow::Result<CudaFunction> {
        let (cubin_path, function_name) = self.kernels.get(name)
            .ok_or_else(|| anyhow::anyhow!("Kernel '{}' not found", name))?;
        let module = self.modules.get(cubin_path)
            .ok_or_else(|| anyhow::anyhow!("Module '{}' not loaded", cubin_path))?;
        module.load_function(function_name)
            .map_err(|e| anyhow::anyhow!("Failed to load function '{}': {:?}", function_name, e))
    }

    /// Launch a kernel with the given config and arguments.
    ///
    /// # Safety
    /// The kernel launch is inherently unsafe (incorrect grid/block dims cause undefined behavior),
    /// but we treat it as safe here because the caller controls the config.
    ///
    /// # Arguments
    /// * `name` - Kernel function name
    /// * `stream` - CUDA stream to enqueue on
    /// * `config` - Grid/block/shared memory config
    pub fn launch(
        &self,
        name: &str,
        stream: &CudaStream,
        config: LaunchConfig,
    ) -> anyhow::Result<()> {
        let func = self.get_function(name)?;
        unsafe {
            let _ = stream.launch_builder(&func).launch(config)
                .map_err(|e| anyhow::anyhow!("Kernel launch '{}' failed: {:?}", name, e))?;
        }
        Ok(())
    }
}
