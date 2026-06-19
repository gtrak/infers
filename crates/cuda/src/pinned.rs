//! Page-locked (pinned) host memory for fast DMA transfers to GPU.

use cudarc::driver::sys;

/// Page-locked (pinned) host memory for fast DMA transfers to GPU.
///
/// Allocates via `cuMemAllocHost_v2()` (page-locked). RAII: `Drop` calls `cuMemFreeHost()`.
/// One buffer is allocated at startup and reused for all weight uploads.
pub struct PinnedHostBuffer {
    ptr: *mut u8,
    size: usize,
}

impl PinnedHostBuffer {
    /// Allocate `size` bytes of pinned host memory.
    ///
    /// Requires a CUDA context active on the current thread (caller responsibility).
    pub fn new(size: usize) -> anyhow::Result<Self> {
        if size == 0 {
            return Ok(Self { ptr: std::ptr::null_mut(), size: 0 });
        }

        let mut ptr: *mut u8 = std::ptr::null_mut();
        unsafe {
            sys::cuMemAllocHost_v2(&mut ptr as *mut _ as *mut *mut ::core::ffi::c_void, size).result()
                .map_err(|e| anyhow::anyhow!("cuMemAllocHost failed: {:?}", e))?;
        }

        if ptr.is_null() {
            anyhow::bail!("cuMemAllocHost returned null pointer for {} bytes", size);
        }

        tracing::debug!(bytes = size, "Pinned host buffer allocated");
        Ok(Self { ptr, size })
    }

    /// Safe read access to the pinned memory.
    pub fn as_slice(&self) -> &[u8] {
        if self.size == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Safe write access to the pinned memory.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.size == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Size of the buffer in bytes.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.size > 0 {
            unsafe {
                sys::cuMemFreeHost(self.ptr as *mut ::core::ffi::c_void).result()
                    .expect("cuMemFreeHost failed");
            }
        }
    }
}

unsafe impl Send for PinnedHostBuffer {}

impl std::fmt::Debug for PinnedHostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedHostBuffer")
            .field("size", &self.size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_host_buffer_compiles() {
        // Type check only — actual allocation requires CUDA context
        let _ = std::any::type_name::<PinnedHostBuffer>();
    }
}
