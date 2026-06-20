//! Weight registry and layer weight structures.
//!
//! Stores model weights as raw byte data with shape metadata,
//! ready for GPU upload when the CUDA runtime is available.
//! Supports both direct BF16/FP16 weights and INT4 quantized triplets
//! (qweight, scales, qzeros) for AutoRound/GPTQ models.

use std::collections::HashMap;
use bytes::Bytes;
use super::config::LayerType;

/// Companion tensors for an INT4 quantized weight.
///
/// When a projection uses INT4 quantization, the qweight is stored as the
/// main `WeightData` in layer structs (e.g., `MlpWeights.gate_proj`), and
/// the scales/qzeros companions are stored in `WeightRegistry.int4_companions`
/// keyed by the qweight's tensor name.
#[derive(Debug, Clone)]
pub struct Int4Companions {
    /// Packed zero points (u32, 8 per u32).
    pub qzeros: WeightData,
    /// BF16 group scales.
    pub scales: WeightData,
}

/// Raw tensor data with shape metadata, stored as bytes until GPU upload.
// @lat: [[lat#Weight Registry and Tensors#WeightData]]
#[derive(Debug, Clone)]
pub struct WeightData {
    /// Raw tensor bytes (BF16, FP16, INT4 packed, or NVFP4 packed).
    pub data: Bytes,
    /// Tensor shape, e.g. [5120, 13888] for a gate projection.
    pub shape: Vec<usize>,
    /// Data type of the stored tensor.
    pub dtype: WeightDtype,
    /// Name of the tensor in the safetensors file.
    pub name: String,
}

/// Data type of a weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    /// Brain float 16 (2 bytes per element).
    Bf16,
    /// Float 16 / half (2 bytes per element).
    Fp16,
    /// Float 32 (4 bytes per element).
    Fp32,
    /// Packed INT4 (8 weights per u32).
    Int4Packed,
    /// NVFP4 (Blackwell 4-bit float, packed).
    Nvfp4,
    /// Unknown / other format.
    Other,
}

impl WeightDtype {
    /// Bytes per element for this data type, or None for packed formats.
    pub fn bytes_per_element(&self) -> Option<usize> {
        match self {
            Self::Bf16 | Self::Fp16 => Some(2),
            Self::Fp32 => Some(4),
            Self::Int4Packed => None, // 8 weights per u32
            Self::Nvfp4 => None,       // packed format
            Self::Other => None,
        }
    }
}

/// GDN (Gated DeltaNet) layer weights.
#[derive(Debug, Clone)]
pub struct GdnWeights {
    /// Linear attention projection A.
    pub in_proj_a: WeightData,
    /// Linear attention projection B.
    pub in_proj_b: WeightData,
    /// 1D convolution weight.
    pub conv1d_weight: WeightData,
    /// State projection weight (optional in Qwen3.6).
    pub x_proj_weight: Option<WeightData>,
    /// Delta projection weight (optional in Qwen3.6).
    pub dt_proj_weight: Option<WeightData>,
    /// Output projection weight.
    pub out_proj_weight: WeightData,
    /// Combined QKV projection (INT4 in real model).
    pub in_proj_qkv: Option<WeightData>,
    /// Output gate projection (INT4 in real model).
    pub in_proj_z: Option<WeightData>,
    /// SSM state transition log (BF16).
    pub a_log: Option<WeightData>,
    /// SSM timescale bias (BF16).
    pub dt_bias: Option<WeightData>,
    /// GDN normalization weight (BF16).
    pub norm: Option<WeightData>,
}

/// Standard attention layer weights.
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Query projection.
    pub q_proj: WeightData,
    /// Key projection.
    pub k_proj: WeightData,
    /// Value projection.
    pub v_proj: WeightData,
    /// Output projection.
    pub o_proj: WeightData,
    /// Q-norm weight for full attention layers.
    pub q_norm: Option<WeightData>,
    /// K-norm weight for full attention layers.
    pub k_norm: Option<WeightData>,
}

/// MLP layer weights.
#[derive(Debug, Clone)]
pub struct MlpWeights {
    /// Gate projection (swiglu gate).
    pub gate_proj: WeightData,
    /// Up projection (swiglu up).
    pub up_proj: WeightData,
    /// Down projection.
    pub down_proj: WeightData,
}

/// A single transformer layer's weights.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Type of this layer (GDN or full attention).
    pub layer_type: LayerType,
    /// Layer index in the model (0-63).
    pub layer_idx: usize,
    /// GDN weights (present only for GatedDeltaNet layers).
    pub gdn: Option<GdnWeights>,
    /// Attention weights (present only for full attention layers).
    pub attn: Option<AttentionWeights>,
    /// MLP weights (present for all layers).
    pub mlp: MlpWeights,
    /// Pre-attention/SSM layer norm.
    pub norm1: WeightData,
    /// Pre-MLP layer norm.
    pub norm2: WeightData,
}

/// Multi-Token Prediction head weights.
///
/// MTP adds one or more full transformer layers that predict future tokens
/// from the main model's hidden state. The MTP head:
/// 1. Normalizes the input embedding and the main model's hidden state
/// 2. Concatenates and projects them through an FC layer
/// 3. Passes through full transformer decoder layers (attention + MLP)
/// 4. Applies final norm
/// 5. Projects to logits via the shared LM head
///
/// When `mtp_use_dedicated_embeddings: false` (typical), the MTP head reuses
/// the main model's embedding table. When true, it has its own `embed_tokens`.
// @lat: [[lat#Weight Registry and Tensors#MtpWeights]]
#[derive(Debug, Clone)]
pub struct MtpWeights {
    /// Norm applied to the input token embedding before FC projection.
    pub pre_fc_norm_embedding: WeightData,
    /// Norm applied to the main model's hidden state before FC projection.
    pub pre_fc_norm_hidden: WeightData,
    /// FC projection: concatenates [embed, hidden] → hidden_size.
    pub fc: WeightData,
    /// The MTP transformer layer(s). Each is a full decoder layer with
    /// attention + MLP, identical in structure to main model layers.
    pub layers: Vec<LayerWeights>,
    /// Final post-layer norm.
    pub norm: WeightData,
    /// Dedicated MTP embeddings (only present if
    /// `mtp_use_dedicated_embeddings: true`).
    pub embed_tokens: Option<WeightData>,
}

/// Complete model weight registry.
// @lat: [[lat#Weight Registry and Tensors#WeightRegistry]]
#[derive(Debug, Clone)]
pub struct WeightRegistry {
    /// Token embedding weights.
    pub embedding: Option<WeightData>,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
    /// MTP head weights (present if model has MTP).
    pub mtp: Option<MtpWeights>,
    /// LM head (output projection).
    pub lm_head: Option<WeightData>,
    /// Final layer norm.
    pub norm: Option<WeightData>,
    /// All tensors by name, for lookup and sharding.
    pub tensors: HashMap<String, WeightData>,
    /// Companion tensors for INT4 weights (qzeros, scales) keyed by the
    /// qweight tensor name. Populated during `build_main_layers` when
    /// INT4 quantized projections are detected.
    pub int4_companions: HashMap<String, Int4Companions>,
}

impl WeightRegistry {
    /// Create a new empty weight registry.
    pub fn new() -> Self {
        Self {
            embedding: None,
            layers: Vec::new(),
            mtp: None,
            lm_head: None,
            norm: None,
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
        self.tensors.values().map(|t| t.data.len()).sum()
    }
}


impl GdnWeights {
    /// Drop all weight data (Bytes) from this GDN layer.
    pub fn clear_data(&mut self) {
        self.in_proj_a.data = Bytes::new();
        self.in_proj_b.data = Bytes::new();
        self.conv1d_weight.data = Bytes::new();
        if let Some(ref mut w) = self.x_proj_weight { w.data = Bytes::new(); }
        if let Some(ref mut w) = self.dt_proj_weight { w.data = Bytes::new(); }
        self.out_proj_weight.data = Bytes::new();
        if let Some(ref mut w) = self.in_proj_qkv { w.data = Bytes::new(); }
        if let Some(ref mut w) = self.in_proj_z { w.data = Bytes::new(); }
        if let Some(ref mut w) = self.a_log { w.data = Bytes::new(); }
        if let Some(ref mut w) = self.dt_bias { w.data = Bytes::new(); }
        if let Some(ref mut w) = self.norm { w.data = Bytes::new(); }
    }
}

impl AttentionWeights {
    /// Drop all weight data (Bytes) from this attention layer.
    pub fn clear_data(&mut self) {
        self.q_proj.data = Bytes::new();
        self.k_proj.data = Bytes::new();
        self.v_proj.data = Bytes::new();
        self.o_proj.data = Bytes::new();
        if let Some(ref mut w) = self.q_norm { w.data = Bytes::new(); }
        if let Some(ref mut w) = self.k_norm { w.data = Bytes::new(); }
    }
}

impl MlpWeights {
    /// Drop all weight data (Bytes) from this MLP layer.
    pub fn clear_data(&mut self) {
        self.gate_proj.data = Bytes::new();
        self.up_proj.data = Bytes::new();
        self.down_proj.data = Bytes::new();
    }
}

impl LayerWeights {
    /// Drop all weight data (Bytes) from this layer.
    pub fn clear_data(&mut self) {
        if let Some(ref mut gdn) = self.gdn {
            gdn.clear_data();
        }
        if let Some(ref mut attn) = self.attn {
            attn.clear_data();
        }
        self.mlp.clear_data();
        self.norm1.data = Bytes::new();
        self.norm2.data = Bytes::new();
    }
}

impl MtpWeights {
    /// Drop all weight data (Bytes) from this MTP head.
    pub fn clear_data(&mut self) {
        self.pre_fc_norm_embedding.data = Bytes::new();
        self.pre_fc_norm_hidden.data = Bytes::new();
        self.fc.data = Bytes::new();
        for layer in &mut self.layers {
            layer.clear_data();
        }
        self.norm.data = Bytes::new();
        if let Some(ref mut w) = self.embed_tokens { w.data = Bytes::new(); }
    }
}

impl WeightRegistry {
    /// Drop weight data (Bytes) from all WeightData entries, keeping names/shapes/dtypes.

    /// After GPU upload, the CPU-side weight data is no longer needed — the GPU
    /// weight cache holds the actual buffers. This method frees ~5 GB per GPU
    /// of persistent heap residency for the Qwen3.6-27B model.
    pub fn clear_data(&mut self) {
        // Clear embedding, norm, lm_head data
        if let Some(ref mut embed) = self.embedding {
            embed.data = Bytes::new();
        }
        if let Some(ref mut norm) = self.norm {
            norm.data = Bytes::new();
        }
        if let Some(ref mut lm_head) = self.lm_head {
            lm_head.data = Bytes::new();
        }

        // Clear MTP data if present
        if let Some(ref mut mtp) = self.mtp {
            mtp.clear_data();
        }

        // Clear per-layer data
        for layer in &mut self.layers {
            layer.clear_data();
        }

        // Clear remaining flat tensors
        for (_name, weight) in self.tensors.iter_mut() {
            weight.data = Bytes::new();
        }

        // Clear INT4 companions
        for (_name, companions) in self.int4_companions.iter_mut() {
            companions.qzeros.data = Bytes::new();
            companions.scales.data = Bytes::new();
        }
    }
}
impl Default for WeightRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Index file for sharded safetensors models.
/// Maps tensor names to shard filenames.
#[derive(Debug, serde::Deserialize)]
pub struct ShardIndex {
    /// Map from tensor name to shard filename.
    pub weight_map: HashMap<String, String>,
    /// Metadata (model name, etc). Uses Value to handle mixed types
    /// (some models store integers like `total_shards: 10`).
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Shard assignment for a single GPU in tensor parallelism.
#[derive(Debug, Clone)]
pub struct WeightShard {
    /// Which GPU this shard is for (0 or 1 for TP=2).
    pub gpu_id: usize,
    /// Sharded weight registry.
    pub registry: WeightRegistry,
}
#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    #[test]
    fn weight_registry_default() {
        let registry = WeightRegistry::default();
        assert_eq!(registry.num_tensors(), 0);
        assert_eq!(registry.total_bytes(), 0);
        assert!(registry.layers.is_empty());
        assert!(registry.mtp.is_none());
    }

    #[test]
    fn weight_dtype_bytes_per_element() {
        assert_eq!(WeightDtype::Bf16.bytes_per_element(), Some(2));
        assert_eq!(WeightDtype::Fp16.bytes_per_element(), Some(2));
        assert_eq!(WeightDtype::Fp32.bytes_per_element(), Some(4));
        assert_eq!(WeightDtype::Int4Packed.bytes_per_element(), None);
        assert_eq!(WeightDtype::Nvfp4.bytes_per_element(), None);
        assert_eq!(WeightDtype::Other.bytes_per_element(), None);
    }

    #[test]
    fn weight_registry_with_tensors() {
        let mut registry = WeightRegistry::new();
        registry.tensors.insert(
            "test.weight".to_string(),
            WeightData {
                data: Bytes::from(vec![0u8; 100]),
                shape: vec![10, 5],
                dtype: WeightDtype::Bf16,
                name: "test.weight".to_string(),
            },
        );
        assert_eq!(registry.num_tensors(), 1);
        assert_eq!(registry.total_bytes(), 100);
    }
}
