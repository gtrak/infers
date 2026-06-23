//! GPU-resident weight cache for cached dequantized tensors.
//!
//! Holds per-GPU caches of weights in either BF16 or INT4-quantized form,
//! keyed by tensor name.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use half::{bf16, f16};
use infers_cuda::{CudaSlice, CudaStream};
use infers_cuda::memcpy2d;
use infers_cuda::PinnedHostBuffer;
use infers_model::{QuantCompanions, MmapCompanions, MmapQuantCompanions, MmapTensor, MmapWeightRegistry, WeightData, WeightDtype, WeightRegistry};

/// A weight stored on the GPU, either as raw BF16/FP16/FP32 or quantized.
pub enum CachedWeight {
    /// BF16/FP16/FP32 weight uploaded as CudaSlice<bf16>
    Bf16(CudaSlice<bf16>),
    /// INT4 quantized weight triplet: qweight (u32 packed) + scales (fp16) + qzeros (u32 packed)
    Int4(Int4GpuBuffers),
    /// NVFP4 quantized weight: packed FP4 data + per-block scale + global scale
    Nvfp4(Nvfp4GpuBuffers),
}

/// GPU buffers for an INT4 quantized weight tensor.
pub struct Int4GpuBuffers {
    pub qweight: CudaSlice<u32>,
    pub scales: CudaSlice<f16>,
    pub qzeros: CudaSlice<u32>,
    /// Original shape of the INT4 weight tensor, used by GEMM dispatch to
    /// determine transposition at call time (depends on the K dimension).
    pub shape: Vec<usize>,
}

/// GPU buffers for an NVFP4 quantized weight tensor.
pub struct Nvfp4GpuBuffers {
    /// Packed FP4 weight data [N, K/2] as bytes
    pub weight_packed: CudaSlice<u8>,
    /// Per-block scale factors (FP8 E4M3) [N, K/group_size]
    pub weight_scale: CudaSlice<u8>,
    /// Global scale for the tensor (stored as f32 scalar)
    pub weight_global_scale: f32,
    /// Input activation global scale (stored as f32 scalar)
    pub input_global_scale: f32,
    /// Original shape of the NVFP4 weight tensor
    pub shape: Vec<usize>,
}

/// Per-GPU cache of dequantized, GPU-resident weight buffers.
/// All weights for one GPU shard, keyed by tensor name.
pub struct GpuWeightCache {
    weights: HashMap<String, CachedWeight>,
}

impl GpuWeightCache {
    /// General lookup by tensor name.
    pub fn get(&self, name: &str) -> Option<&CachedWeight> {
        self.weights.get(name)
    }

    /// Lookup BF16 weight. Returns None if the cached weight is INT4 or NVFP4.
    pub fn get_bf16(&self, name: &str) -> Option<&CudaSlice<bf16>> {
        match self.weights.get(name)? {
            CachedWeight::Bf16(slice) => Some(slice),
            CachedWeight::Int4(_) | CachedWeight::Nvfp4(_) => None,
        }
    }

    /// Lookup INT4 weight. Returns None if the cached weight is BF16 or NVFP4.
    pub fn get_int4(&self, name: &str) -> Option<&Int4GpuBuffers> {
        match self.weights.get(name)? {
            CachedWeight::Bf16(_) | CachedWeight::Nvfp4(_) => None,
            CachedWeight::Int4(buffers) => Some(buffers),
        }
    }

    /// Lookup NVFP4 weight. Returns None if the cached weight is BF16 or INT4.
    pub fn get_nvfp4(&self, name: &str) -> Option<&Nvfp4GpuBuffers> {
        match self.weights.get(name)? {
            CachedWeight::Bf16(_) | CachedWeight::Int4(_) => None,
            CachedWeight::Nvfp4(buffers) => Some(buffers),
        }
    }

    /// Number of cached weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Iterator over the tensor names in this cache.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(|s| s.as_str())
    }

    /// Upload all weights from a `WeightRegistry` to GPU memory.
    ///
    /// Iterates over every tensor in the registry, classifying each as either
    /// BF16/FP16/FP32 (stored as `CachedWeight::Bf16`) or INT4 packed
    /// (stored as `CachedWeight::Int4` with shape for transposition detection).
    /// Unsupported dtypes are skipped with a warning.
    pub fn new(
        stream: &Arc<CudaStream>,
        registry: &WeightRegistry,
    ) -> Result<Self> {
        let mut weights = HashMap::new();

        // 1. Upload embedding table (if present)
        if let Some(embed) = &registry.embedding {
            upload_and_cache(stream, embed, &registry.quant_companions, &mut weights)?;
        }

        // 2. Upload LM head (if present, may be tied with embedding)
        if let Some(lm_head) = &registry.lm_head {
            upload_and_cache(stream, lm_head, &registry.quant_companions, &mut weights)?;
        }

        // 3. Upload final norm (if present)
        if let Some(norm) = &registry.norm {
            upload_and_cache(stream, norm, &registry.quant_companions, &mut weights)?;
        }

        // 4. Upload MTP head weights (if present)
        if let Some(mtp) = &registry.mtp {
            upload_and_cache(stream, &mtp.pre_fc_norm_embedding, &registry.quant_companions, &mut weights)?;
            upload_and_cache(stream, &mtp.pre_fc_norm_hidden, &registry.quant_companions, &mut weights)?;
            upload_and_cache(stream, &mtp.fc, &registry.quant_companions, &mut weights)?;
            if let Some(embed) = &mtp.embed_tokens {
                upload_and_cache(stream, embed, &registry.quant_companions, &mut weights)?;
            }
            for layer in &mtp.layers {
                upload_and_cache(stream, &layer.norm1, &registry.quant_companions, &mut weights)?;
                upload_and_cache(stream, &layer.norm2, &registry.quant_companions, &mut weights)?;

                if let Some(attn) = &layer.attn {
                    upload_and_cache(stream, &attn.q_proj, &registry.quant_companions, &mut weights)?;
                    upload_and_cache(stream, &attn.k_proj, &registry.quant_companions, &mut weights)?;
                    upload_and_cache(stream, &attn.v_proj, &registry.quant_companions, &mut weights)?;
                    upload_and_cache(stream, &attn.o_proj, &registry.quant_companions, &mut weights)?;
                    if let Some(q_norm) = &attn.q_norm {
                        upload_and_cache(stream, q_norm, &registry.quant_companions, &mut weights)?;
                    }
                    if let Some(k_norm) = &attn.k_norm {
                        upload_and_cache(stream, k_norm, &registry.quant_companions, &mut weights)?;
                    }
                }

                if let Some(gdn) = &layer.gdn {
                    upload_and_cache(stream, &gdn.in_proj_a, &registry.quant_companions, &mut weights)?;
                    upload_and_cache(stream, &gdn.in_proj_b, &registry.quant_companions, &mut weights)?;
                    upload_and_cache(stream, &gdn.out_proj_weight, &registry.quant_companions, &mut weights)?;
                    if let Some(x_proj) = &gdn.x_proj_weight {
                        upload_and_cache(stream, x_proj, &registry.quant_companions, &mut weights)?;
                    }
                    if let Some(dt_proj) = &gdn.dt_proj_weight {
                        upload_and_cache(stream, dt_proj, &registry.quant_companions, &mut weights)?;
                    }
                    if let Some(z) = &gdn.in_proj_z {
                        upload_and_cache(stream, z, &registry.quant_companions, &mut weights)?;
                    }
                    if let Some(qkv) = &gdn.in_proj_qkv {
                        upload_and_cache(stream, qkv, &registry.quant_companions, &mut weights)?;
                    }
                    if let Some(a_log) = &gdn.a_log {
                        upload_and_cache(stream, a_log, &registry.quant_companions, &mut weights)?;
                    }
                    if let Some(dt_bias) = &gdn.dt_bias {
                        upload_and_cache(stream, dt_bias, &registry.quant_companions, &mut weights)?;
                    }
                    if let Some(norm) = &gdn.norm {
                        upload_and_cache(stream, norm, &registry.quant_companions, &mut weights)?;
                    }
                    upload_and_cache(stream, &gdn.conv1d_weight, &registry.quant_companions, &mut weights)?;
                }

                upload_and_cache(stream, &layer.mlp.gate_proj, &registry.quant_companions, &mut weights)?;
                upload_and_cache(stream, &layer.mlp.up_proj, &registry.quant_companions, &mut weights)?;
                upload_and_cache(stream, &layer.mlp.down_proj, &registry.quant_companions, &mut weights)?;
            }
            upload_and_cache(stream, &mtp.norm, &registry.quant_companions, &mut weights)?;
        }

        // 5. Upload all layer weights
        for layer in &registry.layers {
            // Norm1 and Norm2 (always present)
            upload_and_cache(stream, &layer.norm1, &registry.quant_companions, &mut weights)?;
            upload_and_cache(stream, &layer.norm2, &registry.quant_companions, &mut weights)?;

            // Attention weights (if present)
            if let Some(attn) = &layer.attn {
                upload_and_cache(stream, &attn.q_proj, &registry.quant_companions, &mut weights)?;
                upload_and_cache(stream, &attn.k_proj, &registry.quant_companions, &mut weights)?;
                upload_and_cache(stream, &attn.v_proj, &registry.quant_companions, &mut weights)?;
                upload_and_cache(stream, &attn.o_proj, &registry.quant_companions, &mut weights)?;
                if let Some(q_norm) = &attn.q_norm {
                    upload_and_cache(stream, q_norm, &registry.quant_companions, &mut weights)?;
                }
                if let Some(k_norm) = &attn.k_norm {
                    upload_and_cache(stream, k_norm, &registry.quant_companions, &mut weights)?;
                }
            }

            // GDN weights (if present)
            if let Some(gdn) = &layer.gdn {
                upload_and_cache(stream, &gdn.in_proj_a, &registry.quant_companions, &mut weights)?;
                upload_and_cache(stream, &gdn.in_proj_b, &registry.quant_companions, &mut weights)?;
                upload_and_cache(stream, &gdn.out_proj_weight, &registry.quant_companions, &mut weights)?;
                if let Some(x_proj) = &gdn.x_proj_weight {
                    upload_and_cache(stream, x_proj, &registry.quant_companions, &mut weights)?;
                }
                if let Some(dt_proj) = &gdn.dt_proj_weight {
                    upload_and_cache(stream, dt_proj, &registry.quant_companions, &mut weights)?;
                }
                if let Some(z) = &gdn.in_proj_z {
                    upload_and_cache(stream, z, &registry.quant_companions, &mut weights)?;
                }
                if let Some(qkv) = &gdn.in_proj_qkv {
                    upload_and_cache(stream, qkv, &registry.quant_companions, &mut weights)?;
                }
                // SSM parameters (a_log, dt_bias) — always BF16, small
                if let Some(a_log) = &gdn.a_log {
                    upload_and_cache(stream, a_log, &registry.quant_companions, &mut weights)?;
                }
                if let Some(dt_bias) = &gdn.dt_bias {
                    upload_and_cache(stream, dt_bias, &registry.quant_companions, &mut weights)?;
                }
                if let Some(norm) = &gdn.norm {
                    upload_and_cache(stream, norm, &registry.quant_companions, &mut weights)?;
                }
                // conv1d_weight is uploaded but currently not used in forward pass
                upload_and_cache(stream, &gdn.conv1d_weight, &registry.quant_companions, &mut weights)?;
            }

            // MLP weights (always present)
            upload_and_cache(stream, &layer.mlp.gate_proj, &registry.quant_companions, &mut weights)?;
            upload_and_cache(stream, &layer.mlp.up_proj, &registry.quant_companions, &mut weights)?;
            upload_and_cache(stream, &layer.mlp.down_proj, &registry.quant_companions, &mut weights)?;
        }

        Ok(Self { weights })
    }

    /// @lat: [[lat.md/lat#GpuWeightCache#GPU Buffer Download]]
    /// Download a BF16 weight from GPU to CPU for debugging.
    pub fn download_bf16(&self, name: &str, stream: &Arc<CudaStream>) -> Option<Vec<bf16>> {
        match self.weights.get(name)? {
            CachedWeight::Bf16(slice) => {
                stream.clone_dtoh(slice).ok()
            }
            _ => None,
        }
    }

    /// Download an INT4 qweight from GPU to CPU for debugging.
    pub fn download_int4_qweight(&self, name: &str, stream: &Arc<CudaStream>) -> Option<Vec<u32>> {
        match self.weights.get(name)? {
            CachedWeight::Int4(bufs) => {
                // The qweight CudaSlice<u32> was transmuted from CudaSlice<u8>, so cudarc's
                // copy operations compute wrong sizes (len field stores byte count, not element count).
                // 
                // Workaround: try clone_dtoh first. If it fails, fall back to raw FFI with small test.
                // First attempt: cudarc's clone_dtoh (works for properly-typed slices)
                match stream.clone_dtoh(&bufs.qweight) {
                    Ok(data) => Some(data),
                    Err(e) => {
                        // Second attempt: raw cuMemcpyDtoH_v2 with correct byte count
                        use cudarc::driver::{DevicePtr, sys::*};
                        let (dev_ptr, _sync) = <CudaSlice<u32> as DevicePtr<u32>>::device_ptr(&bufs.qweight, stream);
                        
                        // Test small copy first to verify pointer is valid
                        let test_size = 64usize;
                        let mut test_buf = vec![0u8; test_size];
                        unsafe {
                            if cuMemcpyDtoH_v2(test_buf.as_mut_ptr() as *mut ::std::os::raw::c_void, dev_ptr, test_size) != CUresult::CUDA_SUCCESS {
                                return None;
                            }
                        }

                        // If small copy works, try full download
                        let byte_count = bufs.qweight.len() * std::mem::size_of::<u32>();
                        let mut host_data: Vec<u8> = vec![0u8; byte_count];
                        unsafe {
                            if cuMemcpyDtoH_v2(
                                host_data.as_mut_ptr() as *mut ::std::os::raw::c_void,
                                dev_ptr,
                                byte_count,
                            ) != CUresult::CUDA_SUCCESS {
                                return None;
                            }
                        }
                        
                        let u32_data: Vec<u32> = host_data.chunks_exact(std::mem::size_of::<u32>())
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        Some(u32_data)
                    }
                }
            }
            _ => None,
        }
    }

    /// @lat: [[lat.md/lat#GpuWeightCache#GPU Buffer Download]]
    /// Download INT4 scales from GPU to CPU for debugging.
    pub fn download_int4_scales(&self, name: &str, stream: &Arc<CudaStream>) -> Option<Vec<f16>> {
        match self.weights.get(name)? {
            CachedWeight::Int4(bufs) => {
                stream.clone_dtoh(&bufs.scales).ok()
            }
            _ => None,
        }
    }

    /// @lat: [[lat.md/lat#GpuWeightCache#GPU Buffer Download]]
    /// Download INT4 qzeros from GPU to CPU for debugging.
    pub fn download_int4_qzeros(&self, name: &str, stream: &Arc<CudaStream>) -> Option<Vec<u32>> {
        match self.weights.get(name)? {
            CachedWeight::Int4(bufs) => {
                stream.clone_dtoh(&bufs.qzeros).ok()
            }
            _ => None,
        }
    }

    /// Upload all weights from a memory-mapped `MmapWeightRegistry` to GPU memory.
    ///
    /// Iterates over every tensor in the registry, dispatching by dtype:
    /// - **BF16**: zero-copy reinterpret as bf16 slice, direct upload.
    /// - **FP16**: copy into pinned buffer, convert f16→bf16 in-place, upload.
    /// - **FP32**: copy into pinned buffer, convert f32→bf16, upload.
    /// - **INT4Packed**: zero-copy reinterpret as u32/u16 slices, upload triplet.
    pub fn new_from_mmap(
        stream: &Arc<CudaStream>,
        registry: &MmapWeightRegistry,
        pinned: &mut PinnedHostBuffer,
    ) -> Result<Self> {
        let mut weights = HashMap::new();

   // Single pass: upload all tensors from the registry
        // Use the registry key (original name) for companion lookup, since sliced mmap tensors
        // store a suffixed name internally (e.g. "qweight_gpu0") but are keyed by the original name.


        for (key, tensor) in &registry.tensors {
            // Skip companion tensors — they are handled as part of the parent .qweight upload.
            if key.ends_with(".qzeros") || key.ends_with(".scales") { continue; }
            upload_mmap_tensor(stream, tensor, key, &registry.quant_companions, pinned, &mut weights)?;
        }

        Ok(Self { weights })
    }

}

/// Synchronize a CUDA stream with a descriptive reason on error.
fn sync_stream(stream: &CudaStream, reason: &str) -> Result<()> {
    stream.synchronize()
        .map_err(|e| anyhow::anyhow!("Stream sync failed ({reason}): {e}"))
}

/// Upload a single weight to GPU and cache it in the `weights` map.
///
/// Classifies the weight by dtype:
/// - **Int4Packed**: look up companions from `int4_companions`, call
///   `upload_int4_weight`, store as `CachedWeight::Int4` with shape info.
/// - **Nvfp4**: upload packed weight, scales, and global scale scalar,
///   store as `CachedWeight::Nvfp4`.
/// - **Bf16 / Fp16 / Fp32**: call `upload_weight`, store as `CachedWeight::Bf16`.
/// - **Other**: skip with a warning log.
fn upload_and_cache(
    stream: &Arc<CudaStream>,
    weight: &WeightData,
    quant_companions: &HashMap<String, QuantCompanions>,
    weights: &mut HashMap<String, CachedWeight>,
) -> Result<()> {
    match weight.dtype {
        WeightDtype::Int4Packed => {
            let companions = quant_companions
                .get(&weight.name)
                .ok_or_else(|| anyhow::anyhow!("INT4 companions not found for weight '{}'", weight.name))?;

            let int4_companions = match companions {
                QuantCompanions::Int4(c) => c,
                _ => return Err(anyhow::anyhow!(
                    "Weight '{}' is INT4Packed but has non-INT4 companions",
                    weight.name
                )),
            };

            let (qweight_gpu, scales_gpu, qzeros_gpu) = crate::upload::upload_int4_weight(
                stream,
                weight,
                &int4_companions.scales,
                &int4_companions.qzeros,
            )?;

            weights.insert(
                weight.name.clone(),
                CachedWeight::Int4(Int4GpuBuffers {
                    qweight: qweight_gpu,
                    scales: scales_gpu,
                    qzeros: qzeros_gpu,
                    shape: weight.shape.clone(),
                }),
            );
        }
        WeightDtype::Nvfp4 => {
            let companions = quant_companions
                .get(&weight.name)
                .ok_or_else(|| anyhow::anyhow!("NVFP4 companions not found for weight '{}'", weight.name))?;

            let nvfp4_companions = match companions {
                QuantCompanions::Nvfp4(c) => c,
                _ => return Err(anyhow::anyhow!(
                    "Weight '{}' is Nvfp4 but has non-NVFP4 companions",
                    weight.name
                )),
            };

            // Upload packed FP4 weight data
            let weight_packed_gpu = stream.clone_htod(&weight.data[..])?;
            sync_stream(stream.as_ref(), &format!("upload_{}_packed", weight.name))?;

            // Upload per-block scale (fp8 e4m3)
            let weight_scale_gpu = stream.clone_htod(&nvfp4_companions.weight_scale.data[..])?;
            sync_stream(stream.as_ref(), &format!("upload_{}_scale", weight.name))?;

            // Read global scale as f32 scalar (stored as float32 in safetensors)
            let weight_global_scale: f32 = unsafe {
                debug_assert!(nvfp4_companions.weight_global_scale.data.len() >= std::mem::size_of::<f32>());
                *(nvfp4_companions.weight_global_scale.data.as_ptr() as *const f32)
            };

            // Read input global scale as f32 scalar (stored as float32 in safetensors)
            let input_global_scale: f32 = unsafe {
                debug_assert!(nvfp4_companions.input_global_scale.data.len() >= std::mem::size_of::<f32>());
                *(nvfp4_companions.input_global_scale.data.as_ptr() as *const f32)
            };

            weights.insert(
                weight.name.clone(),
                CachedWeight::Nvfp4(Nvfp4GpuBuffers {
                    weight_packed: weight_packed_gpu,
                    weight_scale: weight_scale_gpu,
                    weight_global_scale,
                    input_global_scale,
                    shape: weight.shape.clone(),
                }),
            );
        }
        WeightDtype::Bf16 | WeightDtype::Fp16 | WeightDtype::Fp32 => {
            let gpu_slice = crate::upload::upload_weight(stream, weight)?;
            weights.insert(weight.name.clone(), CachedWeight::Bf16(gpu_slice));
        }
        _ => {
            tracing::warn!(
                "Skipping weight '{}' with unsupported dtype {:?} during GPU cache build",
                weight.name,
                weight.dtype
            );
        }
    }
    Ok(())
}

/// Upload a single memory-mapped tensor to GPU and cache it.

/// Dispatches by dtype:
/// - **BF16**: zero-copy reinterpret as bf16 slice, direct upload via clone_htod.
/// - **FP16**: copy into pinned buffer, convert f16→bf16 in-place, upload.
/// - **FP32**: copy into pinned buffer, convert f32→bf16, upload.
/// - **INT4Packed**: zero-copy reinterpret qweight/scales/qzeros as u32/f16 slices,
///   upload each component with sync between uploads.
/// - **Other**: skip with a warning log.
fn upload_mmap_tensor(
    stream: &Arc<CudaStream>,
    tensor: &MmapTensor,
    key: &str,
    quant_companions: &HashMap<String, MmapQuantCompanions>,
    pinned: &mut PinnedHostBuffer,
    weights: &mut HashMap<String, CachedWeight>,
) -> Result<()> {
    if tensor.is_strided() {
        upload_strided_mmap_tensor(stream, tensor, key, quant_companions, weights)?;
    } else {
        upload_contiguous_mmap_tensor(stream, tensor, key, quant_companions, pinned, weights)?;
    }
    Ok(())
}

/// Upload a strided (non-contiguous) tensor using cuMemcpy2D DMA.
fn upload_strided_mmap_tensor(
    stream: &Arc<CudaStream>,
    tensor: &MmapTensor,
    key: &str,
    quant_companions: &HashMap<String, MmapQuantCompanions>,
    weights: &mut HashMap<String, CachedWeight>,
) -> Result<()> {

    match tensor.dtype() {
        WeightDtype::Bf16 => {

            // 2D DMA copy from strided mmap to contiguous GPU buffer (zero CPU copy)
            let gpu_bytes = memcpy2d::clone_htod_2d(
                stream,
                tensor.base_ptr(),
                tensor.col_start_bytes(),
                tensor.src_pitch(),
                tensor.strided_width(),
                tensor.strided_rows(),
            )?;

            // Sync after upload
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            // Cast CudaSlice<u8> to CudaSlice<bf16> — same layout, just different element type.
            let bf16_slice = unsafe { std::mem::transmute::<CudaSlice<u8>, CudaSlice<bf16>>(gpu_bytes) };
            weights.insert(key.to_string(), CachedWeight::Bf16(bf16_slice));
        }
        WeightDtype::Int4Packed => {
            let companions = match quant_companions.get(key) {
                Some(MmapQuantCompanions::Int4(c)) => c,
                _ => anyhow::bail!("INT4 companions not found for weight '{}'", key),
            };

            // TEMPORARY: Use contiguous copy instead of cuMemcpy2D for debugging
            // Copy strided data to a contiguous buffer, then upload normally
            let src = tensor.data();  // full tensor data (contiguous in mmap)
            let col_start = tensor.col_start_bytes();
            let src_pitch = tensor.src_pitch();
            let width = tensor.strided_width();
            let height = tensor.strided_rows();

            // Compute the strided slice as a contiguous buffer
            let mut qweight_bytes = Vec::with_capacity(width * height);
            for row in 0..height {
                let row_start = col_start + row * src_pitch;
                let row_end = row_start + width;
                qweight_bytes.extend_from_slice(&src[row_start..row_end]);
            }
            let qweight_u32: Vec<u32> = qweight_bytes.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let qweight_gpu = stream.clone_htod(&qweight_u32)?;
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            // Scales: same approach
            let scales_src = companions.scales.data();
            let scales_col = companions.scales.col_start_bytes();
            let scales_pitch = companions.scales.src_pitch();
            let scales_width = companions.scales.strided_width();
            let scales_height = companions.scales.strided_rows();
            let mut scales_bytes = Vec::with_capacity(scales_width * scales_height);
            for row in 0..scales_height {
                let row_start = scales_col + row * scales_pitch;
                let row_end = row_start + scales_width;
                scales_bytes.extend_from_slice(&scales_src[row_start..row_end]);
            }
            let scales_f16: Vec<f16> = scales_bytes.chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            let scales_gpu = stream.clone_htod(&scales_f16)?;
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            // Qzeros: same approach
            let qzeros_src = companions.qzeros.data();
            let qzeros_col = companions.qzeros.col_start_bytes();
            let qzeros_pitch = companions.qzeros.src_pitch();
            let qzeros_width = companions.qzeros.strided_width();
            let qzeros_height = companions.qzeros.strided_rows();
            let mut qzeros_bytes = Vec::with_capacity(qzeros_width * qzeros_height);
            for row in 0..qzeros_height {
                let row_start = qzeros_col + row * qzeros_pitch;
                let row_end = row_start + qzeros_width;
                qzeros_bytes.extend_from_slice(&qzeros_src[row_start..row_end]);
            }
            let qzeros_u32: Vec<u32> = qzeros_bytes.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let qzeros_gpu = stream.clone_htod(&qzeros_u32)?;
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            weights.insert(
                key.to_string(),
                CachedWeight::Int4(Int4GpuBuffers {
                    qweight: qweight_gpu,
                    scales: scales_gpu,
                    qzeros: qzeros_gpu,
                    shape: tensor.shape().to_vec(),
                }),
            );
        }
        _ if quant_companions.get(key).is_some_and(|c| matches!(c, MmapQuantCompanions::Nvfp4(_))) => {
            // NVFP4 strided upload
            let companions = match quant_companions.get(key) {
                Some(MmapQuantCompanions::Nvfp4(c)) => c,
                _ => anyhow::bail!("NVFP4 companions not found for weight '{}'", key),
            };

            // Copy strided weight_packed data to contiguous buffer
            let src = tensor.data();
            let col_start = tensor.col_start_bytes();
            let src_pitch = tensor.src_pitch();
            let width = tensor.strided_width();
            let height = tensor.strided_rows();
            let mut weight_packed_bytes = Vec::with_capacity(width * height);
            for row in 0..height {
                let row_start = col_start + row * src_pitch;
                let row_end = row_start + width;
                weight_packed_bytes.extend_from_slice(&src[row_start..row_end]);
            }

            // Upload weight_packed as u8 bytes
            let weight_packed_gpu = stream.clone_htod(&weight_packed_bytes)?;
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            // Copy strided weight_scale data to contiguous buffer
            let ws_src = companions.weight_scale.data();
            let ws_col = companions.weight_scale.col_start_bytes();
            let ws_pitch = companions.weight_scale.src_pitch();
            let ws_width = companions.weight_scale.strided_width();
            let ws_height = companions.weight_scale.strided_rows();
            let mut weight_scale_bytes = Vec::with_capacity(ws_width * ws_height);
            for row in 0..ws_height {
                let row_start = ws_col + row * ws_pitch;
                let row_end = row_start + ws_width;
                weight_scale_bytes.extend_from_slice(&ws_src[row_start..row_end]);
            }

            // Upload weight_scale as u8 bytes (fp8 e4m3)
            let weight_scale_gpu = stream.clone_htod(&weight_scale_bytes)?;
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            // Read global scale as f32 scalar (stored as float32 in safetensors)
            let wgs_data = companions.weight_global_scale.data();
            debug_assert!(wgs_data.len() >= std::mem::size_of::<f32>());
            let weight_global_scale: f32 = unsafe {
                *(wgs_data.as_ptr() as *const f32)
            };

            // Read input global scale as f32 scalar (stored as float32 in safetensors)
            let igs_data = companions.input_global_scale.data();
            debug_assert!(igs_data.len() >= std::mem::size_of::<f32>());
            let input_global_scale: f32 = unsafe {
                *(igs_data.as_ptr() as *const f32)
            };

            weights.insert(
                key.to_string(),
                CachedWeight::Nvfp4(Nvfp4GpuBuffers {
                    weight_packed: weight_packed_gpu,
                    weight_scale: weight_scale_gpu,
                    weight_global_scale,
                    input_global_scale,
                    shape: tensor.shape().to_vec(),
                }),
            );
        }
        _ => anyhow::bail!(
            "Unsupported dtype {:?} for strided upload of '{}'",
            tensor.dtype(),
            key
        ),
    }
    Ok(())
}

/// Upload a contiguous (non-strided) tensor — the original upload path.
fn upload_contiguous_mmap_tensor(
    stream: &Arc<CudaStream>,
    tensor: &MmapTensor,
    key: &str,
    quant_companions: &HashMap<String, MmapQuantCompanions>,
    pinned: &mut PinnedHostBuffer,
    weights: &mut HashMap<String, CachedWeight>,
) -> Result<()> {
    match tensor.dtype() {
        WeightDtype::Int4Packed => {

            let companions = match quant_companions.get(key) {
                Some(MmapQuantCompanions::Int4(c)) => c,
                _ => anyhow::bail!("INT4 companions not found for weight '{}'", key),
            };


            // Upload qweight: reinterpret u8 as u32, upload directly
            let data = tensor.data();
            let u32_slice = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u32,
                    data.len() / 4,
                )
            };
            let qweight_gpu = stream.clone_htod(u32_slice)
                .map_err(|e| anyhow::anyhow!("Failed to upload qweight '{}': {}", key, e))?;

            // Sync after qweight upload
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            // Upload scales: reinterpret u8 as f16, upload directly (scales are FP16)
            let scales_data = companions.scales.data();
            let f16_slice = unsafe {
                std::slice::from_raw_parts(
                    scales_data.as_ptr() as *const f16,
                    scales_data.len() / 2,
                )
            };
            let scales_gpu = stream.clone_htod(f16_slice)
                .map_err(|e| anyhow::anyhow!("Failed to upload scales '{}': {}", key, e))?;

            // Sync after scales upload
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            // Upload qzeros: reinterpret u8 as u32, upload directly
            let qzeros_data = companions.qzeros.data();
            let qzeros_u32_slice = unsafe {
                std::slice::from_raw_parts(
                    qzeros_data.as_ptr() as *const u32,
                    qzeros_data.len() / 4,
                )
            };
            let qzeros_gpu = stream.clone_htod(qzeros_u32_slice)
                .map_err(|e| anyhow::anyhow!("Failed to upload qzeros '{}': {}", key, e))?;

            // Sync after qzeros upload
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            weights.insert(
                key.to_string(),
                CachedWeight::Int4(Int4GpuBuffers {
                    qweight: qweight_gpu,
                    scales: scales_gpu,
                    qzeros: qzeros_gpu,
                    shape: tensor.shape().to_vec(),
                }),
            );
        }
        _ if quant_companions.get(key).is_some_and(|c| matches!(c, MmapQuantCompanions::Nvfp4(_))) => {
            // NVFP4 contiguous upload
            let companions = match quant_companions.get(key) {
                Some(MmapQuantCompanions::Nvfp4(c)) => c,
                _ => anyhow::bail!("NVFP4 companions not found for weight '{}'", key),
            };

            // Upload weight_packed as u8 bytes directly from mmap
            let data = tensor.data();
            let weight_packed_gpu = stream.clone_htod(data)
                .map_err(|e| anyhow::anyhow!("Failed to upload weight_packed '{}': {}", key, e))?;
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            // Upload weight_scale as u8 bytes (fp8 e4m3) directly from mmap
            let ws_data = companions.weight_scale.data();
            let weight_scale_gpu = stream.clone_htod(ws_data)
                .map_err(|e| anyhow::anyhow!("Failed to upload weight_scale '{}': {}", key, e))?;
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            // Read global scale as f32 scalar (stored as float32 in safetensors)
            let wgs_data = companions.weight_global_scale.data();
            debug_assert!(wgs_data.len() >= std::mem::size_of::<f32>());
            let weight_global_scale: f32 = unsafe {
                *(wgs_data.as_ptr() as *const f32)
            };

            // Read input global scale as f32 scalar (stored as float32 in safetensors)
            let igs_data = companions.input_global_scale.data();
            debug_assert!(igs_data.len() >= std::mem::size_of::<f32>());
            let input_global_scale: f32 = unsafe {
                *(igs_data.as_ptr() as *const f32)
            };

            weights.insert(
                key.to_string(),
                CachedWeight::Nvfp4(Nvfp4GpuBuffers {
                    weight_packed: weight_packed_gpu,
                    weight_scale: weight_scale_gpu,
                    weight_global_scale,
                    input_global_scale,
                    shape: tensor.shape().to_vec(),
                }),
            );
        }
        WeightDtype::Bf16 => {
            // Reinterpret u8 as bf16, upload directly (zero-copy from mmap)
            let data = tensor.data();
            let bf16_slice = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const bf16,
                    data.len() / 2,
                )
            };
            let gpu_slice = stream.clone_htod(bf16_slice)
                .map_err(|e| anyhow::anyhow!("Failed to upload weight '{}': {}", key, e))?;

           // Sync after upload
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            weights.insert(key.to_string(), CachedWeight::Bf16(gpu_slice));
        }
        WeightDtype::Fp16 => {

            // Copy into pinned buffer, convert f16→bf16 in-place, upload
            let data = tensor.data();
            let count = data.len() / 2;

            pinned.as_mut_slice()[..data.len()].copy_from_slice(data);

            // In-place conversion: read as f16, write as bf16 (both 2 bytes each)
            for i in 0..count {
                unsafe {
                    let f16_ptr = pinned.as_slice().as_ptr().add(i * 2) as *const f16;
                    let bf16_ptr = pinned.as_mut_slice().as_mut_ptr().add(i * 2) as *mut bf16;
                    let f16_val = *f16_ptr;
                    let bf16_val = bf16::from_f32(f16_val.to_f32());
                    *bf16_ptr = bf16_val;
                }
            }

            let bf16_slice = unsafe {
                std::slice::from_raw_parts(
                    pinned.as_slice().as_ptr() as *const bf16,
                    count,
                )
            };
            let gpu_slice = stream.clone_htod(bf16_slice)
                .map_err(|e| anyhow::anyhow!("Failed to upload weight '{}': {}", key, e))?;

            // Sync after upload
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            weights.insert(key.to_string(), CachedWeight::Bf16(gpu_slice));
        }
        WeightDtype::Fp32 => {
            // Copy into pinned buffer, convert f32→bf16, upload
            let data = tensor.data();
            let count = data.len() / 4;

            pinned.as_mut_slice()[..data.len()].copy_from_slice(data);

            // Read f32 values from pinned buffer
            let f32_slice = unsafe {
                std::slice::from_raw_parts(
                    pinned.as_slice().as_ptr() as *const f32,
                    count,
                )
            };

            // Convert to bf16 (sizes differ: 4→2 bytes, need separate storage)
            let mut bf16_result = Vec::with_capacity(count);
            for &f in f32_slice.iter() {
                bf16_result.push(bf16::from_f32(f));
            }

            let gpu_slice = stream.clone_htod(&bf16_result)
                .map_err(|e| anyhow::anyhow!("Failed to upload weight '{}': {}", key, e))?;

            // Sync after upload
            sync_stream(stream.as_ref(), &format!("upload_{}", key))?;

            weights.insert(key.to_string(), CachedWeight::Bf16(gpu_slice));
        }
        _ => {
            tracing::warn!(
                "Skipping weight '{}' with unsupported dtype {:?} during GPU cache build (mmap path)",
                tensor.name(),
                tensor.dtype()
            );
        }
    }
    Ok(())
}
