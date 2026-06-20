//! Zero-copy 2D DMA copy from host (mmap) to device using cuMemcpy2D.
//!
//! Enables non-contiguous tensor sharding without intermediate CPU copies.
//! Strided data in the mmap is copied directly to contiguous GPU memory via DMA.

use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
use cudarc::driver::sys;

// @lat: [[lat#Mmap Weight Upload]]
/// Copy a 2D region from host memory to device memory using DMA.
///
/// `base_ptr` — host pointer to the first row of the source tensor (original tensor base)
/// `col_start_bytes` — byte offset within each row where shard data begins (srcXInBytes)
/// `src_pitch` — bytes between consecutive rows in source (e.g., total_cols * elem_size)
/// `width_bytes` — bytes per row to copy (shard_cols * elem_size)
/// `height` — number of rows to copy
///
/// Returns a contiguous CudaSlice<u8> on the device.
pub fn clone_htod_2d(
    stream: &Arc<CudaStream>,
    base_ptr: *const u8,
    col_start_bytes: usize,
    src_pitch: usize,
    width_bytes: usize,
    height: usize,
) -> Result<CudaSlice<u8>> {
    let dst_bytes = width_bytes * height;

    // Allocate device memory via cudarc's safe API
    let mut dst: CudaSlice<u8> = unsafe { stream.alloc(dst_bytes) }
        .map_err(|e| anyhow::anyhow!("Failed to allocate GPU memory for 2D copy: {:?}", e))?;

    // Inner scope limits the lifetime of the SyncOnDrop borrow from device_ptr_mut.
    // cuMemcpy2D is synchronous, so the copy completes before _sync drops.
    {
        let (dst_device_ptr, _sync) = dst.device_ptr_mut(stream);

        // Set up 2D copy descriptor. cuMemcpy2D copies from strided host memory
        // to contiguous device memory. The column offset (col_start_bytes) is
        // handled by srcXInBytes — no need to pre-compute pointer offsets.
        let copy = sys::CUDA_MEMCPY2D {
            srcXInBytes: col_start_bytes,
            srcY: 0,
            srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
            srcHost: base_ptr as *const _,
            srcDevice: 0,
            srcArray: std::ptr::null_mut(),
            srcPitch: src_pitch,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: dst_device_ptr,
            dstArray: std::ptr::null_mut(),
            dstPitch: 0, // contiguous destination
            WidthInBytes: width_bytes,
            Height: height,
        };

        unsafe {
            sys::cuMemcpy2D_v2(&copy)
                .result()
                .map_err(|e| anyhow::anyhow!("cuMemcpy2D failed: {:?}", e))?;
        }
    } // _sync dropped here — mutable borrow of dst released

    Ok(dst)
}
