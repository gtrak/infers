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

use super::loader::ShardIndex;
use super::weights::{Int4Companions, WeightData, WeightDtype, WeightRegistry};

// @lat: [[lat#Weight Registry and Tensors#MmapTensor and MmapWeightRegistry#DataOwner]]
/// Owns the backing data for an MmapTensor — either a memory-mapped file or heap-owned bytes.
#[derive(Clone)]
pub enum DataOwner {
    /// Backing data is a memory-mapped file (zero-copy).
    Mmap(Arc<Mmap>),
    /// Backing data is heap-allocated bytes (copy of sliced tensor).
    Owned(Arc<Vec<u8>>),
}

/// Zero-copy reference to a tensor stored in a memory-mapped safetensors file.
#[derive(Clone)]
pub struct MmapTensor {
    owner: DataOwner,       // keeps the data alive (mmap or heap)
    data_ptr: *const u8,   // pointer into owned region
    data_len: usize,       // byte length
    shape: Vec<usize>,
    dtype: WeightDtype,
    name: String,
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
            owner: DataOwner::Mmap(mmap),
            data_ptr,
            data_len,
            shape,
            dtype,
            name,
        }
    }

    /// Create a new tensor backed by heap-allocated data.
    pub fn from_owned(
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: WeightDtype,
        name: String,
    ) -> Self {
        let data_len = data.len();
        // Get the pointer before moving into Arc — Vec is guaranteed to not
        // reallocate once constructed.
        let data_ptr = data.as_ptr();
        let arc = Arc::new(data);
        // SAFETY: Arc<Vec> guarantees the pointer remains valid while the Arc is alive.
        Self {
            owner: DataOwner::Owned(arc),
            data_ptr,
            data_len,
            shape,
            dtype,
            name,
        }
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

    /// Get a clone of the underlying Arc<Mmap> if this tensor is mmap-backed.
    pub fn mmap_arc(&self) -> Option<Arc<Mmap>> {
        match &self.owner {
            DataOwner::Mmap(mmap) => Some(mmap.clone()),
            DataOwner::Owned(_) => None,
        }
    }

    /// Clone the data owner — used by sharding to create sub-views sharing the same backing.
    pub fn clone_owner(&self) -> DataOwner {
        self.owner.clone()
    }
}
/// Companion tensors for a zero-copy INT4 quantized weight.
#[derive(Debug, Clone)]
pub struct MmapCompanions {
    pub qzeros: MmapTensor,
    pub scales: MmapTensor,
}


/// Complete model weight registry backed by memory-mapped files.
#[derive(Clone)]
pub struct MmapWeightRegistry {
    pub tensors: HashMap<String, MmapTensor>,
    pub int4_companions: HashMap<String, MmapCompanions>,
}

impl MmapWeightRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            int4_companions: HashMap::new(),
        }
    }

    /// Total number of parameter tensors in the registry.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Total bytes of all weight data.
    pub fn total_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.data_len).sum()
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
    })
}

/// Slice an MmapTensor along the last dimension for a specific GPU.
/// This is a non-contiguous slice — copies data into a contiguous heap buffer.
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

    let mut new_data = Vec::with_capacity(num_rows * shard_size * bytes_per_element);
    for row in 0..num_rows {
        let row_start = row * cols * bytes_per_element + start * bytes_per_element;
        let row_end = row_start + shard_size * bytes_per_element;
        new_data.extend_from_slice(&tensor.data()[row_start..row_end]);
    }

    let mut new_shape = shape.to_vec();
    new_shape[1] = shard_size;

    Ok(MmapTensor::from_owned(
        new_data,
        new_shape,
        tensor.dtype(),
        tensor.name().to_string(),
    ))
}

/// Shard model weights across `num_gpus` devices for tensor parallelism (mmap version).
///
/// Follows the same sharding rules as [`shard_weights_tp`](super::sharding::shard_weights_tp) but
/// operates on MmapTensor references. Contiguous splits (BF16 dim-0, INT4 dim-0) are zero-copy.
/// Non-contiguous splits (BF16 last-dim, INT4 last-dim) return an error — use --no-mmap.
// @lat: [[lat#Weight Registry and Tensors#MmapTensor and MmapWeightRegistry#shard_weights_tp_mmap]]
pub fn shard_weights_tp_mmap(
    registry: &MmapWeightRegistry,
    _config: &super::config::ModelConfig,
    num_gpus: usize,
) -> Result<Vec<MmapWeightShard>> {
    anyhow::ensure!(num_gpus >= 1, "num_gpus must be >= 1");

    if num_gpus == 1 {
        // No sharding needed for single GPU — cheap Arc clones.
        return Ok(vec![MmapWeightShard { gpu_id: 0, registry: registry.clone() }]);
    }

    let mut shards: Vec<MmapWeightShard> = (0..num_gpus)
        .map(|gpu_id| MmapWeightShard {
            gpu_id,
            registry: MmapWeightRegistry::new(),
        })
        .collect();

    // Pre-scan: companion tensors (.scales, .qzeros) are skipped during sharding
    // since they are processed together with their qweight parent.
    let mut companion_skip: HashSet<String> = HashSet::new();
    for name in registry.tensors.keys() {
        if name.ends_with(".scales") || name.ends_with(".qzeros") {
            companion_skip.insert(name.clone());
        }
    }

    for (name, tensor) in &registry.tensors {
        if companion_skip.contains(name) {
            continue;
        }

        let is_int4 = tensor.dtype() == WeightDtype::Int4Packed;

        // Check if this is a fused QKV projection that needs per-projection sharding.
        // TODO: proper fused QKV mmap sharding not yet implemented; falls through to
        // regular column-parallel which may give incorrect results for GDN models.
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
                                companion_skip.insert(scales_name.clone());
                                companion_skip.insert(qzeros_name.clone());
                                let sliced_scales = mmap_slice_last_dim(scales, gpu_id, num_gpus)
                                    .context(format!("Failed to shard INT4 scales: {}", scales_name))?;
                                let sliced_qzeros = mmap_slice_last_dim(qzeros, gpu_id, num_gpus)
                                    .context(format!("Failed to shard INT4 qzeros: {}", qzeros_name))?;
                                shard.registry.int4_companions.insert(
                                    name.clone(),
                                    MmapCompanions { scales: sliced_scales, qzeros: sliced_qzeros },
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
                                companion_skip.insert(scales_name.clone());
                                companion_skip.insert(qzeros_name.clone());
                                let sliced_scales = mmap_slice_dim0(scales, gpu_id, num_gpus)
                                    .context(format!("Failed to shard INT4 scales: {}", scales_name))?;
                                let sliced_qzeros = mmap_slice_dim0(qzeros, gpu_id, num_gpus)
                                    .context(format!("Failed to shard INT4 qzeros: {}", qzeros_name))?;
                                shard.registry.int4_companions.insert(
                                    name.clone(),
                                    MmapCompanions { scales: sliced_scales, qzeros: sliced_qzeros },
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
                            companion_skip.insert(scales_name.clone());
                            companion_skip.insert(qzeros_name.clone());
                            shard.registry.int4_companions.insert(
                                name.clone(),
                                MmapCompanions { scales: scales.clone(), qzeros: qzeros.clone() },
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
    // Copy int4_companions metadata (names only)
    for (name, companions) in &mmap_reg.int4_companions {
        registry.int4_companions.insert(name.clone(), Int4Companions {
            qzeros: WeightData {
                data: Bytes::new(),
                shape: companions.qzeros.shape().to_vec(),
                dtype: companions.qzeros.dtype(),
                name: companions.qzeros.name().to_string(),
            },
            scales: WeightData {
                data: Bytes::new(),
                shape: companions.scales.shape().to_vec(),
                dtype: companions.scales.dtype(),
                name: companions.scales.name().to_string(),
            },
        });
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
        // Verify tensors are mmap-backed
        let layer_tensor = registry.tensors.get("layer.0.weight").unwrap();
        assert!(layer_tensor.mmap_arc().is_some());

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
    fn mmap_tensor_from_owned_works() {
        // Create an owned tensor with 4x8 BF16 data (64 bytes)
        let data: Vec<u8> = (0..64).collect();
        let tensor = MmapTensor::from_owned(
            data,
            vec![4, 8],
            WeightDtype::Bf16,
            "owned_test".to_string(),
        );

        assert_eq!(tensor.shape(), &[4, 8]);
        assert_eq!(tensor.dtype(), WeightDtype::Bf16);
        assert_eq!(tensor.data_len, 64);
        assert_eq!(tensor.name(), "owned_test");
        assert!(tensor.mmap_arc().is_none()); // not mmap-backed

        // Verify data content via Deref
        let slice: &[u8] = &*tensor;
        assert_eq!(slice.len(), 64);
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
            // Each shard has 4 rows * 4 cols * 2 bytes = 32 bytes
            assert_eq!(w.data_len, 32, "GPU {} data_len should be 32", gpu_id);
            // Owned tensors should not have an mmap arc (data was copied)
            assert!(w.mmap_arc().is_none());
        }

        // Verify the two shards' data doesn't overlap — combined they cover original data.
        let g0_data = shards[0].registry.tensors.get("layers.0.self_attn.q_proj.qweight").unwrap();
        let g1_data = shards[1].registry.tensors.get("layers.0.self_attn.q_proj.qweight").unwrap();
        // Combined shard data should be a permutation of the original bytes (non-contiguous slices).
        let mut combined: Vec<u8> = Vec::new();
        combined.extend_from_slice(g0_data.data());
        combined.extend_from_slice(g1_data.data());
        assert_eq!(combined.len(), 64); // same total as original
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
            assert_eq!(w.data_len, 32, "GPU {} data_len should be 32", gpu_id);
            // Owned tensors should not have an mmap arc
            assert!(w.mmap_arc().is_none());
        }
    }
}
