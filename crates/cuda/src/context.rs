//! CUDA device context management.

use std::sync::Arc;

pub use cudarc::driver::CudaContext;

/// Runtime managing CUDA device contexts and streams.
///
/// Holds the primary cudarc context for each GPU and provides
/// a shared interface for context management.
pub struct CudaRuntime {
    /// cudarc contexts for each GPU device.
    pub devices: Vec<Arc<CudaContext>>,
    /// Number of available GPU devices.
    pub num_devices: usize,
}

impl CudaRuntime {
    /// Create a new runtime, enumerating all available GPU devices.
    pub fn new() -> anyhow::Result<Self> {
        let num_devices = CudaContext::device_count()? as usize;
        anyhow::ensure!(num_devices >= 1, "No CUDA devices found");

        let mut devices = Vec::with_capacity(num_devices);
        for ordinal in 0..num_devices {
            let ctx = CudaContext::new(ordinal)?;
            devices.push(ctx);
        }

        tracing::info!("CudaRuntime initialized with {} device(s)", num_devices);
        Ok(Self { devices, num_devices })
    }

    /// Get the primary context for device `ordinal`.
    pub fn device(&self, ordinal: usize) -> anyhow::Result<&Arc<CudaContext>> {
        self.devices.get(ordinal)
            .ok_or_else(|| anyhow::anyhow!("Device ordinal {} out of range", ordinal))
    }
}
