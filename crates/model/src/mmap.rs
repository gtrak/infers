//! Zero-copy memory-mapped tensor access for safetensors files.
//!
//! Wraps `memmap2::Mmap` in Arc-backed references so tensors can be shared
//! across threads without copying the underlying weight data.

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;

  use super::weights::{Int4Companions, Nvfp4Companions, QuantCompanions, ShardIndex, WeightData, WeightDtype, WeightRegistry};

// @lat: [[lat#Weight Registry and Tensors#MmapTensor and MmapWeightRegistry#DataOwner]]
/// Owns the backing data for an MmapTensor — either a memory-mapped file (zero-copy)
/// or an owned CPU buffer (for non-contiguous sharded results like fused QKV).
#[derive(Clone)]
pub enum DataOwner {
    /// Backing data is a memory-mapped file (zero-copy).
    Mmap(Arc<Mmap>),
    /// Backing data is an owned CPU buffer (contiguous, for sharded copies).
    Owned(Arc<Vec<u8>>),
}

impl DataOwner {
    fn new_mmap(mmap: Arc<Mmap>) -> Self {
        Self::Mmap(mmap)
    }

    fn new_owned(data: Vec<u8>) -> Self {
        Self::Owned(Arc::new(data))
    }
}

/// Zero-copy reference to a tensor stored in a memory-mapped safetensors file.
#[derive(Clone)]
pub struct MmapTensor {
    owner: DataOwner,             // keeps the data alive (mmap)
    data_ptr: *const u8,          // for contiguous: start of shard data; for strided: base of original tensor
    data_len: usize,              // for contiguous: shard data length; for strided: total bytes of original tensor
    shape: Vec<usize>,
    dtype: WeightDtype,
    name: String,
    /// Strided metadata (for non-contiguous shards via cuMemcpy2D).
    /// When src_pitch == 0 (default), data is contiguous — use data_ptr/data_len.
    /// When src_pitch > 0, data_ptr is the BASE of the original tensor,
    /// and the shard data starts at col_start_bytes on each row.
    src_pitch: usize,              // bytes between consecutive rows in source (0 = contiguous)
    col_start_bytes: usize,       // byte offset within each row where shard data begins
    strided_width: usize,         // bytes per row to copy (shard_cols * elem_size)
    strided_rows: usize,          // number of rows to copy
}

// SAFETY: MmapTensor does not allow mutable access to the underlying bytes.
// The pointer is valid for the lifetime of Arc<Mmap>. Multiple threads can
// read simultaneously without data races. However, *const u8 itself is !Sync,
// so we must implement Sync manually.
unsafe impl Send for MmapTensor {}

impl Deref for MmapTensor {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.data()
    }
}

impl std::fmt::Debug for MmapTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapTensor")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("data_len", &self.data_len)
            .finish_non_exhaustive()
    }
}

impl MmapTensor {
    /// Create a new zero-copy tensor reference backed by a memory-mapped file.
    ///
    /// # Safety
    /// The `data_ptr` must point into the region covered by `mmap`, and
    /// `data_len` bytes starting at that pointer must be valid while the
    /// `Arc<Mmap>` is alive. Callers must guarantee these invariants.
    pub fn new(
        mmap: Arc<Mmap>,
        data_ptr: *const u8,
        data_len: usize,
        shape: Vec<usize>,
        dtype: WeightDtype,
        name: String,
    ) -> Self {
        Self {
            owner: DataOwner::new_mmap(mmap),
            data_ptr,
            data_len,
            shape,
            dtype,
            name,
            src_pitch: 0,
            col_start_bytes: 0,
            strided_width: 0,
            strided_rows: 0,
        }
    }

    /// Create a contiguous tensor reference backed by an owned CPU buffer.
    /// Used for non-contiguous sharded results (e.g., fused QKV segment-aware split).
    pub fn from_owned(
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: WeightDtype,
        name: String,
    ) -> Self {
        let ptr = data.as_ptr();
        let len = data.len();
        Self {
            owner: DataOwner::new_owned(data),
            data_ptr: ptr,
            data_len: len,
            shape,
            dtype,
            name,
            src_pitch: 0,
            col_start_bytes: 0,
            strided_width: 0,
            strided_rows: 0,
        }
    }

    /// Create a strided tensor reference backed by a memory-mapped file.
    ///
    /// Used for non-contiguous sharding — the shard data is a column slice
    /// across rows of the original tensor. Data is transferred to GPU via
    /// cuMemcpy2D (zero-copy on CPU side).
    ///
    /// # Safety
    /// The `base_ptr` must point into the region covered by `mmap`.
    pub fn new_strided(
        mmap: Arc<Mmap>,
        base_ptr: *const u8,
        src_pitch: usize,
        col_start_bytes: usize,
        strided_width: usize,
        strided_rows: usize,
        shape: Vec<usize>,
        dtype: WeightDtype,
        name: String,
    ) -> Self {
        let data_len = src_pitch * strided_rows; // approximate bounding box
        Self {
            owner: DataOwner::new_mmap(mmap),
            data_ptr: base_ptr,
            data_len,
            shape,
            dtype,
            name,
            src_pitch,
            col_start_bytes,
            strided_width,
            strided_rows,
        }
    }

    /// Whether this tensor has strided (non-contiguous) memory layout.
    pub fn is_strided(&self) -> bool {
        self.src_pitch > 0
    }

    /// Base pointer to the original tensor data (for strided access).
    pub fn base_ptr(&self) -> *const u8 {
        self.data_ptr
    }

    /// Bytes between consecutive rows in source.
    pub fn src_pitch(&self) -> usize {
        self.src_pitch
    }

    /// Byte offset within each row where shard data begins.
    pub fn col_start_bytes(&self) -> usize {
        self.col_start_bytes
    }

    /// Bytes per row to copy (shard width).
    pub fn strided_width(&self) -> usize {
        self.strided_width
    }

    /// Number of rows to copy.
    pub fn strided_rows(&self) -> usize {
        self.strided_rows
    }

    /// Safe access to the tensor's raw bytes.
    pub fn data(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.data_len) }
    }

    /// Tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Data type of the tensor.
    pub fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    /// Name of the tensor in the safetensors file.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get a clone of the underlying Arc<Mmap>, if this tensor is mmap-backed.
    pub fn mmap_arc(&self) -> Option<Arc<Mmap>> {
        match &self.owner {
            DataOwner::Mmap(m) => Some(m.clone()),
            DataOwner::Owned(_) => None,
        }
    }

    /// Clone the data owner — used by sharding to create sub-views sharing the same backing.
    pub fn clone_owner(&self) -> DataOwner {
        self.owner.clone()
    }
}
// @lat: [[lat#Weight Registry and Tensors#MmapTensor and MmapWeightRegistry#MmapCompanions]]
/// Companion tensors for a zero-copy INT4 quantized weight.
#[derive(Debug, Clone)]
pub struct MmapCompanions {
    pub qzeros: MmapTensor,
    pub scales: MmapTensor,
}

// @lat: [[lat#Weight Registry and Tensors#MmapTensor and MmapWeightRegistry#MmapNvfp4Companions]]
/// Companion tensors for a zero-copy NVFP4 quantized weight (PrismaSCOUT).
#[derive(Debug, Clone)]
pub struct MmapNvfp4Companions {
    /// Per-block scale factor (FP8 E4M3).
    pub weight_scale: MmapTensor,
    /// Global scale for the tensor (BF16 scalar).
    pub weight_global_scale: MmapTensor,
    /// Input activation global scale (BF16 scalar).
    pub input_global_scale: MmapTensor,
}

// @lat: [[lat#Weight Registry and Tensors#MmapTensor and MmapWeightRegistry#MmapQuantCompanions]]
/// Unified companion tensors for any quantized weight format in the mmap path.
#[derive(Debug, Clone)]
pub enum MmapQuantCompanions {
    /// AutoRound INT4 companions.
    Int4(MmapCompanions),
    /// PrismaSCOUT NVFP4 companions.
    Nvfp4(MmapNvfp4Companions),
}

/// Complete model weight registry backed by memory-mapped files.
#[derive(Clone)]
pub struct MmapWeightRegistry {
    pub tensors: HashMap<String, MmapTensor>,
    pub quant_companions: HashMap<String, MmapQuantCompanions>,
}

impl MmapWeightRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            quant_companions: HashMap::new(),
        }
    }

    /// Total number of parameter tensors in the registry.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Total bytes of all weight data.
    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|t| {
            if t.is_strided() {
                // For strided tensors, the actual shard data is strided_width * strided_rows
                t.strided_width() * t.strided_rows()
            } else {
                t.data_len
            }
        }).sum()
    }

    /// Drop heap-owned weight data from all tensors, keeping mmap-backed data alive.
    ///
    /// After GPU upload, the CPU-side copies of owned (non-mmap) tensor data are
    /// no longer needed. This method replaces owned data with empty slices,
    /// freeing ~2 GB of heap residency for the Qwen3.6-27B model.
    ///
    /// Mmap-backed tensors are left untouched — their Arc<Mmap> references
    /// must stay alive to prevent the kernel from unmapping the files.
    pub fn clear_owned_data(&mut self) {
        let empty = Arc::new(Vec::new());
        for tensor in self.tensors.values_mut() {
            if let DataOwner::Owned(_) = &tensor.owner {
                tensor.owner = DataOwner::Owned(empty.clone());
                tensor.data_ptr = empty.as_ptr();
                tensor.data_len = 0;
            }
        }
        for companions in self.quant_companions.values_mut() {
            match companions {
                MmapQuantCompanions::Int4(c) => {
                    if let DataOwner::Owned(_) = &c.qzeros.owner {
                        c.qzeros.owner = DataOwner::Owned(empty.clone());
                        c.qzeros.data_ptr = empty.as_ptr();
                        c.qzeros.data_len = 0;
                    }
                    if let DataOwner::Owned(_) = &c.scales.owner {
                        c.scales.owner = DataOwner::Owned(empty.clone());
                        c.scales.data_ptr = empty.as_ptr();
                        c.scales.data_len = 0;
                    }
                }
                MmapQuantCompanions::Nvfp4(c) => {
                    if let DataOwner::Owned(_) = &c.weight_scale.owner {
                        c.weight_scale.owner = DataOwner::Owned(empty.clone());
                        c.weight_scale.data_ptr = empty.as_ptr();
                        c.weight_scale.data_len = 0;
                    }
                    if let DataOwner::Owned(_) = &c.weight_global_scale.owner {
                        c.weight_global_scale.owner = DataOwner::Owned(empty.clone());
                        c.weight_global_scale.data_ptr = empty.as_ptr();
                        c.weight_global_scale.data_len = 0;
                    }
                    if let DataOwner::Owned(_) = &c.input_global_scale.owner {
                        c.input_global_scale.owner = DataOwner::Owned(empty.clone());
                        c.input_global_scale.data_ptr = empty.as_ptr();
                        c.input_global_scale.data_len = 0;
                    }
                }
            }
        }
    }
}

impl Default for MmapWeightRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Shard assignment for a single GPU in tensor parallelism (mmap version).
#[derive(Clone)]
pub struct MmapWeightShard {
    pub gpu_id: usize,
    pub registry: MmapWeightRegistry,
}

/// Slice an MmapTensor along dimension 0 for a specific GPU.
/// This is a contiguous zero-copy slice — the new tensor points into the same backing store.
fn mmap_slice_dim0(tensor: &MmapTensor, gpu_id: usize, num_gpus: usize) -> Result<MmapTensor> {
    let shape = tensor.shape();
    anyhow::ensure!(
!shape.is_empty(),
        "Weight {} has no dimensions",
        tensor.name()
    );
    let dim0 = shape[0];
    let shard_size = dim0 / num_gpus;
    let start = gpu_id * shard_size;

    let bytes_per_row = tensor.data_len / dim0;
    let start_byte = start * bytes_per_row;
    let end_byte = (start + shard_size) * bytes_per_row;

    let mut new_shape = shape.to_vec();
    new_shape[0] = shard_size;

    Ok(MmapTensor {
        owner: tensor.clone_owner(),
        data_ptr: unsafe { tensor.data().as_ptr().add(start_byte) },
        data_len: end_byte - start_byte,
        shape: new_shape,
        dtype: tensor.dtype(),
        name: tensor.name().to_string(),
        src_pitch: 0,
        col_start_bytes: 0,
        strided_width: 0,
        strided_rows: 0,
    })
}

/// Slice an MmapTensor along the last dimension for a specific GPU.
/// This is a non-contiguous slice — creates a strided tensor that transfers data via cuMemcpy2D.
fn mmap_slice_last_dim(tensor: &MmapTensor, gpu_id: usize, num_gpus: usize) -> Result<MmapTensor> {
    let shape = tensor.shape();
    anyhow::ensure!(
!shape.is_empty(),
        "Weight {} has no dimensions",
        tensor.name()
    );
    anyhow::ensure!(
        shape.len() == 2,
        "mmap last-dim slicing only supports 2D weights, got {}D",
        shape.len()
    );

    let last_dim = shape[1];
    let shard_size = last_dim / num_gpus;
    let start = gpu_id * shard_size;

    let bytes_per_element = tensor.data_len / (shape.iter().product::<usize>());
    let cols = last_dim;
    let num_rows = shape[0];

    // Strided access parameters for cuMemcpy2D:
    let src_pitch = cols * bytes_per_element;        // bytes between rows in original tensor
    let col_start_bytes = start * bytes_per_element; // byte offset within each row where shard begins
    let strided_width = shard_size * bytes_per_element; // bytes per row to copy
    let strided_rows = num_rows;

    let mut new_shape = shape.to_vec();
    new_shape[1] = shard_size;

    Ok(MmapTensor::new_strided(
        tensor.mmap_arc().expect("mmap_slice_last_dim requires mmap-backed tensor"),
        tensor.data().as_ptr(), // base pointer to original mmap data
        src_pitch,
        col_start_bytes,
        strided_width,
        strided_rows,
        new_shape,
        tensor.dtype(),
        format!("{}_gpu{}", tensor.name(), gpu_id),
    ))
}

/// Shard a fused QKV projection along the column dimension with per-segment splitting.
///
/// Unlike generic column-parallel which splits the entire last dimension evenly,
/// this function splits each Q/K/V segment independently. Since segments have
/// different sizes (Q=2048, K=2048, V=6144 for example), a naive split gives
/// wrong results — GPU 0 would get columns [0,5120) instead of the correct
/// [0,1024) + [2048,3072) + [4096,7168).
///
/// The result is a contiguous owned buffer containing the correctly sharded data,
/// because the per-segment columns are non-contiguous in the source.
fn mmap_shard_fused_projection_columns(
    tensor: &MmapTensor,
    gpu_id: usize,
    num_gpus: usize,
    segments: &[(usize, usize)],
) -> Result<MmapTensor> {
    anyhow::ensure!(tensor.shape().len() >= 2, "Fused projection weight must be at least 2D");

    let shape = tensor.shape();
    let rows = shape[0];
    let full_n = shape[1];
    let bytes_per_element = tensor.data_len / (rows * full_n);

    // Total shard columns = sum of each segment's shard portion
    let shard_n: usize = segments.iter().map(|&(s, e)| (e - s) / num_gpus).sum();

    // Copy the correctly sharded data into a contiguous buffer
    let mut shard_data = Vec::with_capacity(rows * shard_n * bytes_per_element);

    for row in 0..rows {
        let row_offset = row * full_n * bytes_per_element;

        for &(start, end) in segments {
            let seg_len = end - start;
            let shard_size = seg_len / num_gpus;
            let shard_start = start + gpu_id * shard_size;
            let shard_end = shard_start + shard_size;

            shard_data.extend_from_slice(
                &tensor.data()[row_offset + shard_start * bytes_per_element
                    ..row_offset + shard_end * bytes_per_element],
            );
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[1] = shard_n;

    Ok(MmapTensor::from_owned(
        shard_data,
        new_shape,
        tensor.dtype(),
        format!("{}_gpu{}", tensor.name(), gpu_id),
    ))
}
/// Shard a fused QKV weight along dim 0 (rows) with per-segment splitting.
///
/// Unlike `mmap_slice_dim0` which naively splits dim 0 evenly, this splits
/// each Q/K/V segment independently. Used for `conv1d.weight` which has
/// shape [conv_dim, 1, kernel_size] where conv_dim = key_dim*2 + value_dim.
fn mmap_shard_fused_projection_rows(
    tensor: &MmapTensor,
    gpu_id: usize,
    num_gpus: usize,
    segments: &[(usize, usize)],
) -> Result<MmapTensor> {
    anyhow::ensure!(tensor.shape().len() >= 2, "Fused projection weight must be at least 2D");

    let shape = tensor.shape();
    let full_n = shape[0]; // conv_dim (e.g., 10240)
    let bytes_per_row = tensor.data_len / full_n; // e.g., 1*4*2 = 8 bytes for [N,1,4] BF16

    // Total shard rows = sum of each segment's shard portion
    let shard_n: usize = segments.iter().map(|&(s, e)| (e - s) / num_gpus).sum();

    // Copy the correctly sharded row data into a contiguous buffer
    let mut shard_data = Vec::with_capacity(shard_n * bytes_per_row);

    for &(start, end) in segments {
        let seg_len = end - start;
        let shard_size = seg_len / num_gpus;
        let shard_start = start + gpu_id * shard_size;
        let shard_end = shard_start + shard_size;

        let start_byte = shard_start * bytes_per_row;
        let end_byte = shard_end * bytes_per_row;
        shard_data.extend_from_slice(&tensor.data()[start_byte..end_byte]);
    }

    let mut new_shape = shape.to_vec();
    new_shape[0] = shard_n;

    Ok(MmapTensor::from_owned(
        shard_data,
        new_shape,
        tensor.dtype(),
        format!("{}_gpu{}", tensor.name(), gpu_id),
    ))
}

/// Shard model weights across `num_gpus` devices for tensor parallelism (mmap version).
/// operates on MmapTensor references. Contiguous splits (BF16 dim-0, INT4 dim-0) are zero-copy.
/// Non-contiguous splits (BF16 last-dim, INT4 last-dim) produce strided tensors uploaded via cuMemcpy2D DMA.
/// Fused QKV projections use per-segment column splitting matching the heap path.
// @lat: [[lat#Weight Registry and Tensors#MmapTensor and MmapWeightRegistry#shard_weights_tp_mmap]]
pub fn shard_weights_tp_mmap(
    registry: &MmapWeightRegistry,
    config: &super::config::ModelConfig,
    num_gpus: usize,
) -> Result<Vec<MmapWeightShard>> {
    anyhow::ensure!(num_gpus >= 1, "num_gpus must be >= 1");

    if num_gpus == 1 {
        // No sharding needed for single GPU — cheap Arc clones.
        let mut shard_registry = MmapWeightRegistry {
            tensors: registry.tensors.clone(),
            quant_companions: HashMap::new(),
        };
        for name in shard_registry.tensors.keys() {
            // Handle INT4 companions
            if name.ends_with(".qweight") {
                let base = name.strip_suffix(".qweight").unwrap_or(name.as_str());
                let scales_name = format!("{}.scales", base);
                let qzeros_name = format!("{}.qzeros", base);

                if let Some(scales) = shard_registry.tensors.get(&scales_name)
                    && let Some(qzeros) = shard_registry.tensors.get(&qzeros_name)
                {
                    shard_registry.quant_companions.insert(
                        name.clone(),
                        MmapQuantCompanions::Int4(MmapCompanions { scales: scales.clone(), qzeros: qzeros.clone() }),
                    );
                }
            }

            // Handle NVFP4 companions
            if name.ends_with(".weight_packed") {
                let base = name.strip_suffix(".weight_packed").unwrap_or(name.as_str());
                let weight_scale_name = format!("{}.weight_scale", base);
                let weight_global_scale_name = format!("{}.weight_global_scale", base);
                let input_global_scale_name = format!("{}.input_global_scale", base);

                if let Some(weight_scale) = shard_registry.tensors.get(&weight_scale_name)
                    && let Some(weight_global_scale) = shard_registry.tensors.get(&weight_global_scale_name)
                    && let Some(input_global_scale) = shard_registry.tensors.get(&input_global_scale_name)
                {
                    shard_registry.quant_companions.insert(
                        name.clone(),
                        MmapQuantCompanions::Nvfp4(MmapNvfp4Companions {
                            weight_scale: weight_scale.clone(),
                            weight_global_scale: weight_global_scale.clone(),
                            input_global_scale: input_global_scale.clone(),
                        }),
                    );
                }
            }
        }
        return Ok(vec![MmapWeightShard { gpu_id: 0, registry: shard_registry }]);
    }

    let mut shards: Vec<MmapWeightShard> = (0..num_gpus)
        .map(|gpu_id| MmapWeightShard {
            gpu_id,
            registry: MmapWeightRegistry::new(),
        })
        .collect();

     // Pre-scan: companion tensors are skipped during sharding since they are
    // processed together with their parent quantized weight.
    let mut companion_skip: HashSet<String> = HashSet::new();
    for name in registry.tensors.keys() {
        if name.ends_with(".scales") || name.ends_with(".qzeros") {
            companion_skip.insert(name.clone());
        }
        // NVFP4 companions
        if name.ends_with(".weight_scale") || name.ends_with(".weight_global_scale") || name.ends_with(".input_global_scale") {
            companion_skip.insert(name.clone());
        }
    }

    for (name, tensor) in &registry.tensors {
        if companion_skip.contains(name) {
            continue;
        }

        let is_int4 = matches!(tensor.dtype(), WeightDtype::Int4Packed);
        let is_nvfp4 = matches!(tensor.dtype(), WeightDtype::Nvfp4)
            || name.ends_with(".weight_packed");

        // Check if this is a fused QKV projection that needs per-projection sharding.
        // Shard each sub-projection (Q/K/V) independently rather than splitting the
        // entire fused tensor evenly, because segments have different sizes.
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let qkv_segments: &[(usize, usize)] = &[
            (0, key_dim),                // Q
            (key_dim, 2 * key_dim),      // K
            (2 * key_dim, conv_dim),     // V
        ];

        if name.contains("in_proj_qkv") {
            // The fused output dimension is the last segment endpoint.
            let fused_dim = qkv_segments.last().map(|(_, e)| *e).unwrap_or(0);
            // Detect which axis of the weight tensor holds the fused output dimension.
            // INT4 qweight: shape (K/8, N=fused_dim) → split dim1 → columns fn
            // NVFP4 weight_packed: shape (N=fused_dim, K/2) → split dim0 → rows fn
            let use_rows = tensor.shape()[0] == fused_dim;

            // Scaled segments for qzeros (conv_dim/8 instead of conv_dim)
            let qzeros_segments: Vec<(usize, usize)> =
                qkv_segments.iter().map(|&(s, e)| (s / 8, e / 8)).collect();

            for (gpu_id, shard) in shards.iter_mut().enumerate() {
                let sliced = if use_rows {
                    mmap_shard_fused_projection_rows(tensor, gpu_id, num_gpus, qkv_segments)
                } else {
                    mmap_shard_fused_projection_columns(tensor, gpu_id, num_gpus, qkv_segments)
                }
                .context(format!("Failed to shard fused QKV projection: {}", name))?;
                shard.registry.tensors.insert(name.clone(), sliced);

                 // Shard INT4 companion weights with the same segment structure
                if is_int4 && name.ends_with(".qweight") {
                    let base = name.strip_suffix(".qweight").unwrap_or(name.as_str());
                    let scales_name = format!("{}.scales", base);
                    let qzeros_name = format!("{}.qzeros", base);

                    if let Some(scales) = registry.tensors.get(&scales_name)
                        && let Some(qzeros) = registry.tensors.get(&qzeros_name) {

                        // Debug: trace companion lookup
                        tracing::debug!(
                            "Fused QKV shard {}: found companions scales={} qzeros={}",
                            gpu_id, scales.shape().len(), qzeros.shape().len()
                        );

                        let sliced_scales = mmap_shard_fused_projection_columns(
                            scales, gpu_id, num_gpus, qkv_segments,
                        )
                        .context(format!("Failed to shard INT4 scales: {}", scales_name))?;
                        // qzeros has last_dim = conv_dim/8, so segments must be scaled by 1/8
                        let sliced_qzeros = mmap_shard_fused_projection_columns(
                            qzeros, gpu_id, num_gpus, &qzeros_segments,
                        )
                        .context(format!("Failed to shard INT4 qzeros: {}", qzeros_name))?;
                        shard.registry.quant_companions.insert(
                            name.clone(),
                            MmapQuantCompanions::Int4(MmapCompanions { scales: sliced_scales, qzeros: sliced_qzeros }),
                        );
                    }
                }

                // Shard NVFP4 companion weights — replicate ALL companions to all GPUs.
                // Fused QKV: weight_scale has a different second dim than weight_packed
                // (groups vs elements), so segment-based sharding doesn't apply.
                if name.ends_with(".weight_packed") {
                    let base = name.strip_suffix(".weight_packed").unwrap_or(name.as_str());
                    let weight_scale_name = format!("{}.weight_scale", base);
                    let weight_global_scale_name = format!("{}.weight_global_scale", base);
                    let input_global_scale_name = format!("{}.input_global_scale", base);

                    if let Some(weight_scale) = registry.tensors.get(&weight_scale_name)
                        && let Some(weight_global_scale) = registry.tensors.get(&weight_global_scale_name)
                        && let Some(input_global_scale) = registry.tensors.get(&input_global_scale_name) {
                        companion_skip.insert(weight_scale_name.clone());
                        companion_skip.insert(weight_global_scale_name.clone());
                        companion_skip.insert(input_global_scale_name.clone());
                        shard.registry.quant_companions.insert(
                            name.clone(),
                            MmapQuantCompanions::Nvfp4(MmapNvfp4Companions {
                                weight_scale: weight_scale.clone(),
                                weight_global_scale: weight_global_scale.clone(),
                                input_global_scale: input_global_scale.clone(),
                            }),
                        );
                    }
                }
            }
            continue;
        }

        // conv1d.weight: fused QKV weight needing per-segment row splitting (same as heap path)
        if name.contains("conv1d.weight") {
            // The fused output dimension is the last segment endpoint.
            let fused_dim = qkv_segments.last().map(|(_, e)| *e).unwrap_or(0);
            // Detect which axis of the weight tensor holds the fused output dimension.
            // BF16: shape (N=fused_dim, K...) → split dim0 → rows fn
            // NVFP4 might differ — detect generically from shape vs segments.
            let use_rows = tensor.shape()[0] == fused_dim;

            for (gpu_id, shard) in shards.iter_mut().enumerate() {
                let sliced = if use_rows {
                    mmap_shard_fused_projection_rows(tensor, gpu_id, num_gpus, qkv_segments)
                } else {
                    mmap_shard_fused_projection_columns(tensor, gpu_id, num_gpus, qkv_segments)
                }
                    .context(format!("Failed to shard conv1d weight: {}", name))?;
                shard.registry.tensors.insert(name.clone(), sliced);
            }
            continue;
        }
        let shard_type = super::sharding::determine_shard_type(name);

        match shard_type {
            super::sharding::ShardType::ColumnParallel => {
                if is_int4 {
                    // INT4 column-parallel: split last dim — non-contiguous, copies data.
                    for (gpu_id, shard) in shards.iter_mut().enumerate() {
                        let sliced = mmap_slice_last_dim(tensor, gpu_id, num_gpus)
                            .context(format!("Failed to shard column-parallel weight: {}", name))?;
                        shard.registry.tensors.insert(name.clone(), sliced);

                        // Shard INT4 companion weights — scales and qzeros split on last dim.
                        if name.ends_with(".qweight") {
                            let base = name.strip_suffix(".qweight").unwrap_or(name.as_str());
                            let scales_name = format!("{}.scales", base);
                            let qzeros_name = format!("{}.qzeros", base);

                            if let Some(scales) = registry.tensors.get(&scales_name)
                                && let Some(qzeros) = registry.tensors.get(&qzeros_name)
                            {
                                let sliced_scales = mmap_slice_last_dim(scales, gpu_id, num_gpus)
                                    .context(format!("Failed to shard INT4 scales: {}", scales_name))?;
                                let sliced_qzeros = mmap_slice_last_dim(qzeros, gpu_id, num_gpus)
                                    .context(format!("Failed to shard INT4 qzeros: {}", qzeros_name))?;
                                shard.registry.quant_companions.insert(
                                    name.clone(),
                                    MmapQuantCompanions::Int4(MmapCompanions { scales: sliced_scales, qzeros: sliced_qzeros }),
                                );
                            }
                        }
                    }
                } else if is_nvfp4 {
                    // NVFP4 column-parallel: split dim 0 — N is on dim 0.
                    for (gpu_id, shard) in shards.iter_mut().enumerate() {
                        let sliced = mmap_slice_dim0(tensor, gpu_id, num_gpus)
                            .context(format!("Failed to shard column-parallel weight: {}", name))?;
                        shard.registry.tensors.insert(name.clone(), sliced);

                        // Shard NVFP4 companion weights — weight_scale split on dim0 (N) to match weight_packed.
                        if name.ends_with(".weight_packed") {
                            let base = name.strip_suffix(".weight_packed").unwrap_or(name.as_str());
                            let weight_scale_name = format!("{}.weight_scale", base);
                            let weight_global_scale_name = format!("{}.weight_global_scale", base);
                            let input_global_scale_name = format!("{}.input_global_scale", base);

                            if let Some(weight_scale) = registry.tensors.get(&weight_scale_name)
                                && let Some(weight_global_scale) = registry.tensors.get(&weight_global_scale_name)
                                && let Some(input_global_scale) = registry.tensors.get(&input_global_scale_name) {
                                let sliced_ws = mmap_slice_dim0(weight_scale, gpu_id, num_gpus)
                                    .context(format!("Failed to shard NVFP4 weight_scale: {}", weight_scale_name))?;
                                // weight_global_scale and input_global_scale are 1D scalars — replicate to all GPUs
                                shard.registry.quant_companions.insert(
                                    name.clone(),
                                    MmapQuantCompanions::Nvfp4(MmapNvfp4Companions {
                                        weight_scale: sliced_ws,
                                        weight_global_scale: weight_global_scale.clone(),
                                        input_global_scale: input_global_scale.clone(),
                                    }),
                                );
                            }
                        }
                    }
                } else {
                    // BF16 column-parallel: split dim 0 — contiguous, zero-copy.
                    for (gpu_id, shard) in shards.iter_mut().enumerate() {
                        let sliced = mmap_slice_dim0(tensor, gpu_id, num_gpus)
                            .context(format!("Failed to shard column-parallel weight: {}", name))?;
                        shard.registry.tensors.insert(name.clone(), sliced);
                    }
                }
            }
            super::sharding::ShardType::RowParallel => {
                if is_int4 {
                    // INT4 row-parallel: split dim 0 — contiguous, zero-copy.
                    for (gpu_id, shard) in shards.iter_mut().enumerate() {
                        let sliced = mmap_slice_dim0(tensor, gpu_id, num_gpus)
                            .context(format!("Failed to shard INT4 row-parallel weight: {}", name))?;
                        shard.registry.tensors.insert(name.clone(), sliced);

                        // Shard INT4 companion weights — scales and qzeros split on dim 0.
                        if name.ends_with(".qweight") {
                            let base = name.strip_suffix(".qweight").unwrap_or(name.as_str());
                            let scales_name = format!("{}.scales", base);
                            let qzeros_name = format!("{}.qzeros", base);

                            if let Some(scales) = registry.tensors.get(&scales_name)
                                && let Some(qzeros) = registry.tensors.get(&qzeros_name)
                            {
                                let sliced_scales = mmap_slice_dim0(scales, gpu_id, num_gpus)
                                    .context(format!("Failed to shard INT4 scales: {}", scales_name))?;
                                let sliced_qzeros = mmap_slice_dim0(qzeros, gpu_id, num_gpus)
                                    .context(format!("Failed to shard INT4 qzeros: {}", qzeros_name))?;
                                shard.registry.quant_companions.insert(
                                    name.clone(),
                                    MmapQuantCompanions::Int4(MmapCompanions { scales: sliced_scales, qzeros: sliced_qzeros }),
                                );
                            }
                        }
                    }
                } else if is_nvfp4 {
                    // NVFP4 row-parallel: split last dim (K/2 on dim1) — non-contiguous, copies data.
                    for (gpu_id, shard) in shards.iter_mut().enumerate() {
                        let sliced = mmap_slice_last_dim(tensor, gpu_id, num_gpus)
                            .context(format!("Failed to shard NVFP4 row-parallel weight: {}", name))?;
                        shard.registry.tensors.insert(name.clone(), sliced);

                        // Shard NVFP4 companion weights — weight_scale split on last dim (K/group_size on dim1).
                        if name.ends_with(".weight_packed") {
                            let base = name.strip_suffix(".weight_packed").unwrap_or(name.as_str());
                            let weight_scale_name = format!("{}.weight_scale", base);
                            let weight_global_scale_name = format!("{}.weight_global_scale", base);
                            let input_global_scale_name = format!("{}.input_global_scale", base);

                            if let Some(weight_scale) = registry.tensors.get(&weight_scale_name)
                                && let Some(weight_global_scale) = registry.tensors.get(&weight_global_scale_name)
                                && let Some(input_global_scale) = registry.tensors.get(&input_global_scale_name) {
                                let sliced_ws = mmap_slice_last_dim(weight_scale, gpu_id, num_gpus)
                                    .context(format!("Failed to shard NVFP4 weight_scale: {}", weight_scale_name))?;
                                // weight_global_scale and input_global_scale are 1D scalars — replicate to all GPUs
                                shard.registry.quant_companions.insert(
                                    name.clone(),
                                    MmapQuantCompanions::Nvfp4(MmapNvfp4Companions {
                                        weight_scale: sliced_ws,
                                        weight_global_scale: weight_global_scale.clone(),
                                        input_global_scale: input_global_scale.clone(),
                                    }),
                                );
                            }
                        }
                    }
                } else {
                    // BF16 row-parallel: split last dim — non-contiguous, copies data.
                    for (gpu_id, shard) in shards.iter_mut().enumerate() {
                        let sliced = mmap_slice_last_dim(tensor, gpu_id, num_gpus)
                            .context(format!("Failed to shard row-parallel weight: {}", name))?;
                        shard.registry.tensors.insert(name.clone(), sliced);
                    }
                }
            }
            super::sharding::ShardType::Replicated => {
                // Replicate on all GPUs — cheap Arc clone.
                for shard in shards.iter_mut() {
                    shard.registry.tensors.insert(name.clone(), tensor.clone());

                    // Replicate INT4 companion weights.
                    if is_int4 && name.ends_with(".qweight") {
                        let base = name.strip_suffix(".qweight").unwrap_or(name.as_str());
                        let scales_name = format!("{}.scales", base);
                        let qzeros_name = format!("{}.qzeros", base);

                        if let Some(scales) = registry.tensors.get(&scales_name)
                            && let Some(qzeros) = registry.tensors.get(&qzeros_name)
                        {
                            shard.registry.quant_companions.insert(
                                name.clone(),
                                MmapQuantCompanions::Int4(MmapCompanions { scales: scales.clone(), qzeros: qzeros.clone() }),
                            );
                        }
                    }

                    // Replicate NVFP4 companion weights.
                    if name.ends_with(".weight_packed") {
                        let base = name.strip_suffix(".weight_packed").unwrap_or(name.as_str());
                        let weight_scale_name = format!("{}.weight_scale", base);
                        let weight_global_scale_name = format!("{}.weight_global_scale", base);
                        let input_global_scale_name = format!("{}.input_global_scale", base);

                        if let Some(weight_scale) = registry.tensors.get(&weight_scale_name)
                            && let Some(weight_global_scale) = registry.tensors.get(&weight_global_scale_name)
                            && let Some(input_global_scale) = registry.tensors.get(&input_global_scale_name) {
                            shard.registry.quant_companions.insert(
                                name.clone(),
                                MmapQuantCompanions::Nvfp4(MmapNvfp4Companions {
                                    weight_scale: weight_scale.clone(),
                                    weight_global_scale: weight_global_scale.clone(),
                                    input_global_scale: input_global_scale.clone(),
                                }),
                            );
                        }
                    }
                }
            }
        }
    }

    Ok(shards)
}


/// Map safetensors dtype to our WeightDtype.
fn map_safetensor_dtype_mmap(dtype: safetensors::Dtype) -> WeightDtype {
    match dtype {
        safetensors::Dtype::BF16 => WeightDtype::Bf16,
        safetensors::Dtype::F16 => WeightDtype::Fp16,
        safetensors::Dtype::F32 => WeightDtype::Fp32,
        safetensors::Dtype::U32 | safetensors::Dtype::I32 => WeightDtype::Int4Packed,
        _ => WeightDtype::Other,
    }
}

/// Load safetensors using zero-copy memory-mapped access.
///
/// Handles both single-file (`model.safetensors`) and sharded
/// (`model.safetensors.index.json` + multiple shard files) formats.
/// Unlike [`load_safetensors`](super::loader::load_safetensors), this stores
/// raw pointers into the mmap region instead of copying data via
/// `Bytes::copy_from_slice()`.
pub fn load_safetensors_mmap(model_dir: &Path) -> Result<MmapWeightRegistry> {
    let index_path = model_dir.join("model.safetensors.index.json");

    if index_path.exists() {
        load_sharded_mmap(model_dir, &index_path)
    } else {
        let single_path = model_dir.join("model.safetensors");
        if single_path.exists() {
            load_single_mmap(&single_path)
        } else {
            anyhow::bail!(
                "No safetensors files found in {:?}. Expected model.safetensors or model.safetensors.index.json",
                model_dir
            )
        }
    }
}

fn load_single_mmap(path: &Path) -> Result<MmapWeightRegistry> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open safetensors file: {:?}", path))?;
    // SAFETY: The file is opened read-only, the file handle is verified to exist
    // before mapping, and the mapping is read-only weight data.
    let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file)? });
    let st = SafeTensors::deserialize(&mmap)?;

    let mut tensors = HashMap::new();
    for name in st.names() {
        let tensor = st.tensor(name)?;
        let shape: Vec<usize> = tensor.shape().to_vec();
        let dtype = map_safetensor_dtype_mmap(tensor.dtype());
        let data_ptr = tensor.data().as_ptr();
        let data_len = tensor.data().len();

        tensors.insert(
            name.to_string(),
            MmapTensor::new(mmap.clone(), data_ptr, data_len, shape, dtype, name.to_string()),
        );
    }

    let total_bytes: usize = tensors.values().map(|t| t.data_len).sum();
    tracing::debug!(
        "Loaded {} tensors ({:.2} GB) from {:?} (zero-copy mmap)",
        tensors.len(),
        total_bytes as f64 / 1e9,
        path
    );

    let mut registry = MmapWeightRegistry::new();
    registry.tensors = tensors;
    Ok(registry)
}

fn load_sharded_mmap(model_dir: &Path, index_path: &Path) -> Result<MmapWeightRegistry> {
    let index_content = std::fs::read_to_string(index_path)?;
    let index: ShardIndex = serde_json::from_str(&index_content)?;

    // Collect unique shard filenames
    let shards: std::collections::HashSet<String> = index.weight_map.values().cloned().collect();
    let mut all_tensors = HashMap::new();
    let mut mmaps = Vec::new();

    for shard_name in &shards {
        let shard_path = model_dir.join(shard_name);
        tracing::debug!("Loading shard (mmap): {:?}", shard_path);

        let file = std::fs::File::open(&shard_path)
            .with_context(|| format!("Failed to open shard: {:?}", shard_path))?;
        // SAFETY: The shard file is opened read-only, verified to exist before mapping.
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file)? });
        mmaps.push(mmap.clone());
        let st = SafeTensors::deserialize(&mmap)?;

        for name in st.names() {
            let tensor = st.tensor(name)?;
            let shape: Vec<usize> = tensor.shape().to_vec();
            let dtype = map_safetensor_dtype_mmap(tensor.dtype());
            let data_ptr = tensor.data().as_ptr();
            let data_len = tensor.data().len();

            all_tensors.insert(
                name.to_string(),
                MmapTensor::new(mmap.clone(), data_ptr, data_len, shape, dtype, name.to_string()),
            );
        }
    }

    let total_bytes: usize = all_tensors.values().map(|t| t.data_len).sum();
    tracing::info!(
        "Loaded {} shards, {} tensors ({:.2} GB) (zero-copy mmap)",
        shards.len(),
        all_tensors.len(),
        total_bytes as f64 / 1e9,
    );

    // mmaps are kept alive by DataOwner::Mmap in each tensor.
    drop(mmaps);

    let mut registry = MmapWeightRegistry::new();
    registry.tensors = all_tensors;
    Ok(registry)
}

/// Strip `model.language_model.` prefix from tensor names and remove vision tensors.
///
/// Tensors starting with `model.language_model.` get that prefix stripped, so
/// `model.language_model.layers.0.input_layernorm.weight` becomes
/// `layers.0.input_layernorm.weight`. Tensors starting with `model.visual.`
/// are removed entirely. Tensors starting with `mtp.` and all other tensors
/// are kept as-is.
pub fn strip_language_model_prefix_mmap(registry: &mut MmapWeightRegistry) {
    let lang_prefix = "model.language_model.";
    let vis_prefix = "model.visual.";
    let mut to_remove = Vec::new();
    let mut to_rename = Vec::new();
    for key in registry.tensors.keys() {
        if key.starts_with(vis_prefix) {
            to_remove.push(key.clone());
        } else if key.starts_with(lang_prefix) {
            let new_key = key.strip_prefix(lang_prefix).unwrap().to_string();
            if new_key != *key {
                to_rename.push((key.clone(), new_key));
            }
        }
    }
    for key in to_remove {
        registry.tensors.remove(&key);
    }
    for (old_key, new_key) in to_rename {
        if let Some(mut tensor) = registry.tensors.remove(&old_key) {
            // Also update the internal name field so companion lookups match
            if tensor.name == old_key || tensor.name.starts_with("model.language_model.") {
                tensor.name = new_key.clone();
            }
            registry.tensors.insert(new_key, tensor);
        }
    }
}

// @lat: [[lat#Weight Registry and Tensors#MmapTensor and MmapWeightRegistry#build_metadata_registry]]
/// Build a WeightRegistry with metadata (names/shapes only) from a MmapWeightRegistry.
/// The WeightData entries have empty Bytes — they are used only for name lookups
/// during inference, not for data access.
pub fn build_metadata_registry(mmap_reg: &MmapWeightRegistry) -> WeightRegistry {
    use bytes::Bytes;

    let mut registry = WeightRegistry::new();
    for (name, tensor) in &mmap_reg.tensors {
        registry.tensors.insert(name.clone(), WeightData {
            data: Bytes::new(), // empty — metadata only
            shape: tensor.shape().to_vec(),
            dtype: tensor.dtype(),
            name: name.clone(),
        });
    }
    // Copy quantized companion metadata (names only) — INT4 and NVFP4
    for (name, companions) in &mmap_reg.quant_companions {
        match companions {
            MmapQuantCompanions::Int4(c) => {
                registry.quant_companions.insert(name.clone(), QuantCompanions::Int4(Int4Companions {
                    qzeros: WeightData {
                        data: Bytes::new(),
                        shape: c.qzeros.shape().to_vec(),
                        dtype: c.qzeros.dtype(),
                        name: c.qzeros.name().to_string(),
                    },
                    scales: WeightData {
                        data: Bytes::new(),
                        shape: c.scales.shape().to_vec(),
                        dtype: c.scales.dtype(),
                        name: c.scales.name().to_string(),
                    },
                }));
            }
            MmapQuantCompanions::Nvfp4(c) => {
                registry.quant_companions.insert(name.clone(), QuantCompanions::Nvfp4(Nvfp4Companions {
                    weight_scale: WeightData {
                        data: Bytes::new(),
                        shape: c.weight_scale.shape().to_vec(),
                        dtype: c.weight_scale.dtype(),
                        name: c.weight_scale.name().to_string(),
                    },
                    weight_global_scale: WeightData {
                        data: Bytes::new(),
                        shape: c.weight_global_scale.shape().to_vec(),
                        dtype: c.weight_global_scale.dtype(),
                        name: c.weight_global_scale.name().to_string(),
                    },
                    input_global_scale: WeightData {
                        data: Bytes::new(),
                        shape: c.input_global_scale.shape().to_vec(),
                        dtype: c.input_global_scale.dtype(),
                        name: c.input_global_scale.name().to_string(),
                    },
                }));
            }
        }
    }
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmap_tensor_data_returns_correct_slice() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = (0..16).collect();
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let tensor = MmapTensor::new(
            mmap,
            unsafe { ptr.add(4) }, // start at offset 4
            8,          // 8 bytes: [4,5,6,7,8,9,10,11]
            vec![2, 4],
            WeightDtype::Bf16,
            "test_tensor".to_string(),
        );

        let slice = tensor.data();
        assert_eq!(slice.len(), 8);
        assert_eq!(slice, &[4, 5, 6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn mmap_tensor_deref_works() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = (0..32).collect();
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let tensor = MmapTensor::new(
            mmap.clone(),
            ptr,
            16,
            vec![4, 4],
            WeightDtype::Fp32,
            "deref_test".to_string(),
        );

        // Deref allows using &tensor as &[u8]
        let slice: &[u8] = &*tensor;
        assert_eq!(slice, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        // Also test method access through deref
        let sum: u8 = tensor.iter().sum();
        let expected: u8 = (0..16).sum();
        assert_eq!(sum, expected);
    }

    #[test]
    fn mmap_weight_registry_new_is_empty() {
        let registry = MmapWeightRegistry::new();
        assert_eq!(registry.num_tensors(), 0);
        assert_eq!(registry.total_bytes(), 0);
    }

    #[test]
    fn mmap_weight_registry_total_bytes() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = (0..100).collect();
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let mut registry = MmapWeightRegistry::new();
        registry.tensors.insert(
            "tensor_a".to_string(),
            MmapTensor::new(mmap.clone(), ptr, 50, vec![25], WeightDtype::Bf16, "tensor_a".into()),
        );
        registry.tensors.insert(
            "tensor_b".to_string(),
            MmapTensor::new(mmap.clone(), unsafe { ptr.add(50) }, 50, vec![25], WeightDtype::Fp32, "tensor_b".into()),
        );

        assert_eq!(registry.num_tensors(), 2);
        assert_eq!(registry.total_bytes(), 100);
    }

    #[test]
    fn mmap_tensor_clone_is_shallow() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = (0..16).collect();
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let tensor = MmapTensor::new(
            mmap.clone(),
            ptr,
            16,
            vec![4, 4],
            WeightDtype::Bf16,
            "clone_test".to_string(),
        );

        let cloned = tensor.clone();
        assert_eq!(cloned.data(), tensor.data());
    }

    #[test]
    fn test_load_safetensors_mmap_single() {
        use safetensors::{serialize, tensor::TensorView, Dtype};

        // Create a synthetic safetensors file with two tensors
        let data_a: Vec<u8> = vec![1u8; 64]; // 32 bf16 values
        let data_b: Vec<u8> = vec![2u8; 32]; // 16 bf16 values

        let view_a = TensorView::new(Dtype::BF16, vec![4, 8], &data_a).unwrap();
        let view_b = TensorView::new(Dtype::BF16, vec![2, 8], &data_b).unwrap();

        let serialized = serialize(
            [
                ("layer.0.weight", view_a),
                ("norm.weight", view_b),
            ],
            None,
        ).unwrap();

        // Write to a temp directory as model.safetensors
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), &serialized).unwrap();

        // Load via mmap
        let registry = load_safetensors_mmap(dir.path()).unwrap();

        assert_eq!(registry.num_tensors(), 2);
        assert!(registry.tensors.contains_key("layer.0.weight"));
        assert!(registry.tensors.contains_key("norm.weight"));
    // Verify tensors are mmap-backed (mmap_arc() returns Some(Arc<Mmap>))
        let layer_tensor = registry.tensors.get("layer.0.weight").unwrap();
        let _arc: Arc<Mmap> = layer_tensor.mmap_arc().expect("should be mmap-backed");

        // Verify tensor shapes and data
        let layer_tensor = registry.tensors.get("layer.0.weight").unwrap();
        assert_eq!(layer_tensor.shape(), &[4, 8]);
        assert_eq!(layer_tensor.dtype(), WeightDtype::Bf16);
        assert_eq!(layer_tensor.data_len, 64);
        assert_eq!(layer_tensor.data(), &data_a[..]);

        let norm_tensor = registry.tensors.get("norm.weight").unwrap();
        assert_eq!(norm_tensor.shape(), &[2, 8]);
        assert_eq!(norm_tensor.dtype(), WeightDtype::Bf16);
        assert_eq!(norm_tensor.data_len, 32);
        assert_eq!(norm_tensor.data(), &data_b[..]);

        // Total bytes should match
        assert_eq!(registry.total_bytes(), 96);
    }

    #[test]
    fn test_strip_language_model_prefix_mmap() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = vec![0u8; 256];
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let mut registry = MmapWeightRegistry::new();

        // Add a language model tensor (should be stripped)
        registry.tensors.insert(
            "model.language_model.layers.0.input_layernorm.weight".to_string(),
            MmapTensor::new(
                mmap.clone(),
                ptr,
                32,
                vec![4, 8],
                WeightDtype::Bf16,
                "model.language_model.layers.0.input_layernorm.weight".to_string(),
            ),
        );

        // Add a visual tensor (should be removed)
        registry.tensors.insert(
            "model.visual.patch_embed.proj.weight".to_string(),
            MmapTensor::new(
                mmap.clone(),
                unsafe { ptr.add(32) },
                32,
                vec![4, 8],
                WeightDtype::Bf16,
                "model.visual.patch_embed.proj.weight".to_string(),
            ),
        );

        // Add an MTP tensor (should be kept as-is)
        registry.tensors.insert(
            "mtp.layers.0.self_attn.q_proj.weight".to_string(),
            MmapTensor::new(
                mmap.clone(),
                unsafe { ptr.add(64) },
                32,
                vec![4, 8],
                WeightDtype::Bf16,
                "mtp.layers.0.self_attn.q_proj.weight".to_string(),
            ),
        );

        // Add a bare tensor (should be kept as-is)
        registry.tensors.insert(
            "embed_tokens.weight".to_string(),
            MmapTensor::new(
                mmap.clone(),
                unsafe { ptr.add(96) },
                32,
                vec![10, 8],
                WeightDtype::Bf16,
                "embed_tokens.weight".to_string(),
            ),
        );

        assert_eq!(registry.num_tensors(), 4);

        strip_language_model_prefix_mmap(&mut registry);

        // Language model prefix should be stripped
        assert!(registry.tensors.contains_key("layers.0.input_layernorm.weight"));
        let stripped = registry.tensors.get("layers.0.input_layernorm.weight").unwrap();
        assert_eq!(stripped.name(), "layers.0.input_layernorm.weight");
        assert!(!registry.tensors.contains_key("model.language_model.layers.0.input_layernorm.weight"));

        // Visual tensors should be removed
        assert!(!registry.tensors.contains_key("model.visual.patch_embed.proj.weight"));

        // MTP and bare tensors should remain unchanged
        assert!(registry.tensors.contains_key("mtp.layers.0.self_attn.q_proj.weight"));
        assert!(registry.tensors.contains_key("embed_tokens.weight"));

        assert_eq!(registry.num_tensors(), 3);
    }

    #[test]
    fn shard_weights_tp_mmap_single_gpu_clones() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = (0..64).collect();
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let mut registry = MmapWeightRegistry::new();
        registry.tensors.insert(
            "test.weight".to_string(),
            MmapTensor::new(mmap.clone(), ptr, 64, vec![4, 8], WeightDtype::Bf16, "test.weight".into()),
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_num_value_heads":4,"linear_value_head_dim":16}"#;
        let config: crate::config::ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp_mmap(&registry, &config, 1).unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].gpu_id, 0);
        assert_eq!(shards[0].registry.num_tensors(), 1);
        // Verify the tensor is a shallow clone (same data)
        let orig = registry.tensors.get("test.weight").unwrap();
        let cloned = shards[0].registry.tensors.get("test.weight").unwrap();
        assert_eq!(cloned.data(), orig.data());
    }

    #[test]
    fn shard_weights_tp_mmap_bf16_column_parallel_zero_copy() {
        // BF16 column-parallel: split dim 0 (rows) — contiguous, zero-copy.
        let file = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = (0..64).collect(); // 4 rows * 8 cols * 2 bytes
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let mut registry = MmapWeightRegistry::new();
        registry.tensors.insert(
            "layers.0.self_attn.q_proj.weight".to_string(),
            MmapTensor::new(mmap.clone(), ptr, 64, vec![4, 8], WeightDtype::Bf16, "layers.0.self_attn.q_proj.weight".into()),
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_num_value_heads":4,"linear_value_head_dim":16}"#;
        let config: crate::config::ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp_mmap(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        // GPU 0 gets rows 0-1 (shape [2, 8]), GPU 1 gets rows 2-3 (shape [2, 8])
        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("layers.0.self_attn.q_proj.weight").unwrap();
            assert_eq!(w.shape(), &[2, 8], "GPU {} shape should be [2, 8]", gpu_id);
            assert_eq!(w.data_len, 32, "GPU {} data_len should be 32", gpu_id);
        }

        // Verify zero-copy: GPU 0 data starts at offset 0, GPU 1 at offset 32
        let g0_data = shards[0].registry.tensors.get("layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(g0_data.data(), &bytes[0..32]);
        let g1_data = shards[1].registry.tensors.get("layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(g1_data.data(), &bytes[32..64]);
    }

    #[test]
    fn shard_weights_tp_mmap_int4_row_parallel_zero_copy() {
        // INT4 row-parallel: split dim 0 (K/8) — contiguous, zero-copy.
        let file = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = (0..128).collect(); // 4 rows * 8 cols * 4 bytes
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let mut registry = MmapWeightRegistry::new();
        registry.tensors.insert(
            "layers.0.self_attn.o_proj.qweight".to_string(),
            MmapTensor::new(mmap.clone(), ptr, 128, vec![4, 8], WeightDtype::Int4Packed, "layers.0.self_attn.o_proj.qweight".into()),
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_num_value_heads":4,"linear_value_head_dim":16}"#;
        let config: crate::config::ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp_mmap(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        // GPU 0 gets rows 0-1 (shape [2, 8]), GPU 1 gets rows 2-3 (shape [2, 8])
        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("layers.0.self_attn.o_proj.qweight").unwrap();
            assert_eq!(w.shape(), &[2, 8], "GPU {} shape should be [2, 8]", gpu_id);
            assert_eq!(w.data_len, 64, "GPU {} data_len should be 64", gpu_id);
        }

        // Verify zero-copy: GPU 0 data starts at offset 0, GPU 1 at offset 64
        let g0_data = shards[0].registry.tensors.get("layers.0.self_attn.o_proj.qweight").unwrap();
        assert_eq!(g0_data.data(), &bytes[0..64]);
        let g1_data = shards[1].registry.tensors.get("layers.0.self_attn.o_proj.qweight").unwrap();
        assert_eq!(g1_data.data(), &bytes[64..128]);
    }

    #[test]
    fn shard_weights_tp_mmap_replicated_tensor() {
        // Norm weights should be replicated on all GPUs.
        let file = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = (0..16).collect();
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let mut registry = MmapWeightRegistry::new();
        registry.tensors.insert(
            "layers.0.input_layernorm.weight".to_string(),
            MmapTensor::new(mmap.clone(), ptr, 16, vec![8], WeightDtype::Bf16, "layers.0.input_layernorm.weight".into()),
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_num_value_heads":4,"linear_value_head_dim":16}"#;
        let config: crate::config::ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp_mmap(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        // Both GPUs should have the same full tensor (replicated)
        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("layers.0.input_layernorm.weight").unwrap();
            assert_eq!(w.shape(), &[8], "GPU {} shape should be [8]", gpu_id);
            assert_eq!(w.data_len, 16, "GPU {} data_len should be 16", gpu_id);
            assert_eq!(w.data(), &bytes[..]); // same data as original
        }
    }

    #[test]
    fn shard_weights_tp_mmap_rejects_zero_gpus() {
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_num_value_heads":4,"linear_value_head_dim":16}"#;
        let config: crate::config::ModelConfig = serde_json::from_str(config_json).unwrap();

        let result = shard_weights_tp_mmap(&MmapWeightRegistry::new(), &config, 0);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("num_gpus must be >= 1"));
        }
    }

    #[test]
    fn shard_weights_tp_mmap_int4_column_parallel_owned() {
        // INT4 column-parallel (e.g. q_proj.qweight with shape [K/8, N]): split last dim — non-contiguous.
        let file = tempfile::NamedTempFile::new().unwrap();
        // 4 rows * 8 cols * 2 bytes per element = 64 bytes
        let bytes: Vec<u8> = (0..64).collect();
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let mut registry = MmapWeightRegistry::new();
        registry.tensors.insert(
            "layers.0.self_attn.q_proj.qweight".to_string(),
            MmapTensor::new(mmap.clone(), ptr, 64, vec![4, 8], WeightDtype::Int4Packed, "layers.0.self_attn.q_proj.qweight".into()),
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_num_value_heads":4,"linear_value_head_dim":16}"#;
        let config: crate::config::ModelConfig = serde_json::from_str(config_json).unwrap();

        // This should succeed (not bail) — non-contiguous INT4 column-parallel is now supported.
        let shards = shard_weights_tp_mmap(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("layers.0.self_attn.q_proj.qweight").unwrap();
            // Shape should have last dim halved: [4, 8] -> [4, 4]
            assert_eq!(w.shape(), &[4, 4], "GPU {} shape should be [4, 4]", gpu_id);
            // Strided tensor — is_strided should be true
            assert!(w.is_strided(), "GPU {} tensor should be strided", gpu_id);
            // Still has mmap backing (zero-copy from original)
            let _arc: Arc<Mmap> = w.mmap_arc().expect("should be mmap-backed");
        }

        // Verify strided access parameters are correct.
        // Original tensor: [4, 8], elem_size=2 (BF16-sized for Int4Packed).
        // src_pitch = 8 * 2 = 16 bytes per row in original tensor
        // col_start_bytes for GPU 0 = 0, for GPU 1 = 4 * 2 = 8
        let g0_data = shards[0].registry.tensors.get("layers.0.self_attn.q_proj.qweight").unwrap();
        assert_eq!(g0_data.src_pitch(), 16);
        assert_eq!(g0_data.col_start_bytes(), 0);
        assert_eq!(g0_data.strided_width(), 8); // 4 cols * 2 bytes
        assert_eq!(g0_data.strided_rows(), 4);

        let g1_data = shards[1].registry.tensors.get("layers.0.self_attn.q_proj.qweight").unwrap();
        assert_eq!(g1_data.src_pitch(), 16);
        assert_eq!(g1_data.col_start_bytes(), 8); // columns 4-7, offset by 4*2=8 bytes
        assert_eq!(g1_data.strided_width(), 8);
        assert_eq!(g1_data.strided_rows(), 4);
    }

    #[test]
    fn shard_weights_tp_mmap_bf16_row_parallel_owned() {
        // BF16 row-parallel (e.g. o_proj.weight with shape [N, K]): split last dim — non-contiguous.
        let file = tempfile::NamedTempFile::new().unwrap();
        // 4 rows * 8 cols * 2 bytes per element = 64 bytes
        let bytes: Vec<u8> = (0..64).collect();
        std::fs::write(&file, &bytes).unwrap();

        let mmap = Arc::new(unsafe { Mmap::map(&file).unwrap() });
        let ptr = mmap.as_ptr();

        let mut registry = MmapWeightRegistry::new();
        registry.tensors.insert(
            "layers.0.self_attn.o_proj.weight".to_string(),
            MmapTensor::new(mmap.clone(), ptr, 64, vec![4, 8], WeightDtype::Bf16, "layers.0.self_attn.o_proj.weight".into()),
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_num_value_heads":4,"linear_value_head_dim":16}"#;
        let config: crate::config::ModelConfig = serde_json::from_str(config_json).unwrap();

        // This should succeed (not bail) — non-contiguous BF16 row-parallel is now supported.
        let shards = shard_weights_tp_mmap(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("layers.0.self_attn.o_proj.weight").unwrap();
            assert_eq!(w.shape(), &[4, 4], "GPU {} shape should be [4, 4]", gpu_id);
            // Strided tensor — is_strided should be true
            assert!(w.is_strided(), "GPU {} tensor should be strided", gpu_id);
            // Still has mmap backing (zero-copy from original)
            let _arc: Arc<Mmap> = w.mmap_arc().expect("should be mmap-backed");
        }
    }

    #[test]
    fn shard_weights_tp_mmap_fused_qkv_segment_aware() {
        // Fused QKV: each segment (Q/K/V) should be split independently.
        // With key_dim=32, value_dim=64, conv_dim=128:
        //   Q=[0,32), K=[32,64), V=[64,128)
        // For TP=2, GPU 0 gets first half of each segment:
        //   [0,16) from Q + [32,48) from K + [64,96) from V = 64 cols total
        // This is NOT the same as taking columns [0,64) (which would include
        // all of Q and K but only half of V — wrong!).

        // Fill with predictable data: byte at position (row * 128 + col) = col % 256
        let mut qweight_bytes: Vec<u8> = Vec::with_capacity(4 * 128);
        for _row in 0..4 {
            for col in 0..128 {
                qweight_bytes.push(col as u8);
            }
        }

        // Scales: shape [4, 128] (same as qweight for this test)
        let mut scales_bytes: Vec<u8> = Vec::with_capacity(4 * 128);
        for _row in 0..4 {
            for col in 0..128 {
                scales_bytes.push(col as u8 + 1);
            }
        }

        // Qzeros: shape [4, 16] (conv_dim/8 = 128/8 = 16)
        let mut qzeros_bytes: Vec<u8> = Vec::with_capacity(4 * 16);
        for _row in 0..4 {
            for col in 0..16 {
                qzeros_bytes.push(col as u8 + 2);
            }
        }

        // Write each to a separate temp file so they don't overwrite each other
        let file1 = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(&file1, &qweight_bytes).unwrap();
        let mmap1 = Arc::new(unsafe { Mmap::map(&file1).unwrap() });

        let file2 = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(&file2, &scales_bytes).unwrap();
        let mmap2 = Arc::new(unsafe { Mmap::map(&file2).unwrap() });

        let file3 = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(&file3, &qzeros_bytes).unwrap();
        let mmap3 = Arc::new(unsafe { Mmap::map(&file3).unwrap() });

        let mut registry = MmapWeightRegistry::new();
        registry.tensors.insert(
            "layers.0.linear_attn.in_proj_qkv.qweight".to_string(),
            MmapTensor::new(mmap1.clone(), unsafe { mmap1.as_ptr() }, 512, vec![4, 128], WeightDtype::Int4Packed, "layers.0.linear_attn.in_proj_qkv.qweight".into()),
        );

        registry.tensors.insert(
            "layers.0.linear_attn.in_proj_qkv.scales".to_string(),
            MmapTensor::new(mmap2.clone(), unsafe { mmap2.as_ptr() }, 512, vec![4, 128], WeightDtype::Int4Packed, "layers.0.linear_attn.in_proj_qkv.scales".into()),
        );

        registry.tensors.insert(
            "layers.0.linear_attn.in_proj_qkv.qzeros".to_string(),
            MmapTensor::new(mmap3.clone(), unsafe { mmap3.as_ptr() }, 64, vec![4, 16], WeightDtype::Int4Packed, "layers.0.linear_attn.in_proj_qkv.qzeros".into()),
        );

        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10],"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_num_value_heads":4,"linear_value_head_dim":16}"#;
        let config: crate::config::ModelConfig = serde_json::from_str(config_json).unwrap();

        let shards = shard_weights_tp_mmap(&registry, &config, 2).unwrap();
        assert_eq!(shards.len(), 2);

        for gpu_id in 0..2 {
            let w = shards[gpu_id].registry.tensors.get("layers.0.linear_attn.in_proj_qkv.qweight").unwrap();
            // Shape should be [4, 64] (64 = 16 from Q + 16 from K + 32 from V)
            assert_eq!(w.shape(), &[4, 64], "GPU {} shape should be [4, 64]", gpu_id);

            // Debug: verify data is correctly owned (not mmap-backed)
            assert!(w.mmap_arc().is_none(), "GPU {} fused QKV qweight should be owned", gpu_id);
            // NOT strided — owned data for contiguous shard buffer
            assert!(!w.is_strided(), "GPU {} fused QKV should not be strided (owned data)", gpu_id);

            // Verify the data is correctly sharded per segment
            // For row 0: we expect bytes from columns [0,16) + [32,48) + [64,96) for GPU 0
            // or [16,32) + [48,64) + [96,128) for GPU 1
            let row0_data = &w.data()[0..64];

            if gpu_id == 0 {
                // GPU 0: first half of Q [0,16), first half of K [32,48), first half of V [64,96)
                let expected: Vec<u8> = (0..16).chain(32..48).chain(64..96).collect();
                assert_eq!(row0_data, &expected[..], "GPU 0 row data mismatch");
            } else {
                // GPU 1: second half of Q [16,32), second half of K [48,64), second half of V [96,128)
                let expected: Vec<u8> = (16..32).chain(48..64).chain(96..128).collect();
                assert_eq!(row0_data, &expected[..], "GPU 1 row data mismatch");
            }

          // Verify companion tensors were also correctly sharded
            let companions = match shards[gpu_id].registry.quant_companions.get("layers.0.linear_attn.in_proj_qkv.qweight")
                .expect("companion should exist for qweight") {
                    MmapQuantCompanions::Int4(c) => c,
                    _ => panic!("Expected MmapQuantCompanions::Int4"),
                };
            assert_eq!(companions.scales.shape(), &[4, 64], "GPU {} scales shape", gpu_id);
            // Qzeros segments scaled by 1/8: Q=[0,4), K=[4,8), V=[8,16) → shard 2+2+4=8 cols per GPU
            assert_eq!(companions.qzeros.shape(), &[4, 8], "GPU {} qzeros shape", gpu_id);
        }
    }
}
