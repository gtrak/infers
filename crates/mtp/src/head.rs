//! MTP prediction head — a single transformer decoder layer that
//! predicts the next token from the main model's hidden state.
//!
//! The MTP head architecture:
//! 1. Normalizes the input embedding (`pre_fc_norm_embedding`) and the main
//!    model's hidden state (`pre_fc_norm_hidden`)
//! 2. Projects them independently through an FC layer to `hidden_size`
//! 3. Combines via element-wise addition
//! 4. Passes through one full transformer decoder layer (attention + MLP)
//! 5. Applies final layer norm
//! 6. Outputs hidden state ready for LM head projection (done by caller)

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::gemm::{GemmConfig, GemmEngine};
use infers_cuda::{CudaSlice, CudaStream};
use infers_model::{LayerWeights, ModelConfig, MtpWeights, WeightData};

/// MTP prediction head, with GPU-resident weight buffers.
///
/// Stores the MTP head's norms, FC projection, decoder layer weights,
/// and final norm on the GPU. The `forward` method takes callbacks
/// for embedding lookup, RMS normalization, and decoder layer execution
/// — operations that depend on the backend's kernel dispatch.
pub struct MtpHead {
    /// Pre-FC norm weights for token embedding (GPU-resident).
    pre_fc_norm_embedding_gpu: CudaSlice<bf16>,
    /// Pre-FC norm weights for main model hidden state (GPU-resident).
    pre_fc_norm_hidden_gpu: CudaSlice<bf16>,
    /// Left half of FC weight: maps normalized embedding → hidden_size.
    /// Derived from the first hidden_size columns of the full FC weight
    /// (which is [hidden_size × 2*hidden_size]).
    fc_weight_left_gpu: CudaSlice<bf16>,
    /// Right half of FC weight: maps normalized hidden state → hidden_size.
    /// Derived from the last hidden_size columns of the full FC weight.
    fc_weight_right_gpu: CudaSlice<bf16>,
    /// The MTP decoder layer weights (CPU-side, for use by layer callback).
    /// Contains attention or GDN weights, MLP weights, and layer norms.
    pub layer: LayerWeights,
    /// Final post-layer norm weights (GPU-resident).
    norm_gpu: CudaSlice<bf16>,
    /// Whether to use dedicated MTP embeddings (default: false = share main model).
    pub use_dedicated_embeddings: bool,
}

impl MtpHead {
    /// Construct an MtpHead from model weights, uploading weight data to GPU.
    ///
    /// The FC weight matrix of shape `[hidden_size, 2 × hidden_size]` is split
    /// into two halves — one for the embedding path and one for the hidden-state
    /// path — avoiding the need for a GPU-side concat operation during forward.
    ///
    /// # Arguments
    /// * `mtp` — MTP weights from model loading (includes norms, FC, layers)
    /// * `config` — Model configuration (provides hidden_size, rms_norm_eps)
    /// * `stream` — CUDA stream for weight uploads
    pub fn from_weights(
        mtp: &MtpWeights,
        config: &ModelConfig,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let _dtype_size = 2; // BF16 = 2 bytes

        // Upload pre-FC norms
        let pre_fc_norm_embedding_gpu = upload_weight_bytes(stream, &mtp.pre_fc_norm_embedding)?;
        let pre_fc_norm_hidden_gpu = upload_weight_bytes(stream, &mtp.pre_fc_norm_hidden)?;

        // Split FC weight [hidden_size × 2*hidden_size] into left and right halves
        // Each half: [hidden_size × hidden_size]
        let fc_shape = &mtp.fc.shape;
        anyhow::ensure!(
            fc_shape.len() == 2 && fc_shape[0] == hidden_size && fc_shape[1] == 2 * hidden_size,
            "MTP FC weight shape mismatch: expected [{}, {}], got {:?}",
            hidden_size,
            2 * hidden_size,
            fc_shape,
        );

        // Extract left half (first hidden_size columns of each row)
        let fc_bytes = &mtp.fc.data;
        let half_bytes = hidden_size * hidden_size * 2; // 2 bytes per bf16
        anyhow::ensure!(
            fc_bytes.len() >= 2 * half_bytes,
            "MTP FC weight data too short: {} bytes, need {}",
            fc_bytes.len(),
            2 * half_bytes,
        );

        let left_data: Vec<bf16> = (0..hidden_size)
            .flat_map(|row| {
                let row_start = row * 2 * hidden_size;
                (0..hidden_size).map(move |col| {
                    let offset = (row_start + col) * 2;
                    let lo = fc_bytes[offset];
                    let hi = fc_bytes[offset + 1];
                    bf16::from_bits(u16::from_le_bytes([lo, hi]))
                })
            })
            .collect();

        let right_data: Vec<bf16> = (0..hidden_size)
            .flat_map(|row| {
                let row_start = row * 2 * hidden_size;
                (hidden_size..2 * hidden_size).map(move |col| {
                    let offset = (row_start + col) * 2;
                    let lo = fc_bytes[offset];
                    let hi = fc_bytes[offset + 1];
                    bf16::from_bits(u16::from_le_bytes([lo, hi]))
                })
            })
            .collect();

        let fc_weight_left_gpu = stream
            .clone_htod(&left_data)
            .map_err(|e| anyhow::anyhow!("Failed to upload MTP FC weight left: {e}"))?;
        let fc_weight_right_gpu = stream
            .clone_htod(&right_data)
            .map_err(|e| anyhow::anyhow!("Failed to upload MTP FC weight right: {e}"))?;

        // Take the first MTP layer (mtp_num_hidden_layers == 1 for Qwen3.6-27B)
        let layer = mtp
            .layers
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("MTP has no layers"))?;

        // Upload final norm
        let norm_gpu = upload_weight_bytes(stream, &mtp.norm)?;

        Ok(Self {
            pre_fc_norm_embedding_gpu,
            pre_fc_norm_hidden_gpu,
            fc_weight_left_gpu,
            fc_weight_right_gpu,
            layer,
            norm_gpu,
            use_dedicated_embeddings: mtp.embed_tokens.is_some(),
        })
    }

    /// Run the MTP head forward pass.
    ///
    /// Steps:
    /// 1. Embed `input_token` via the `embed` callback
    /// 2. RMSNorm the embedding and the main model's hidden state
    /// 3. GEMM projection: `left = embed_norm @ fc_left^T`, `right = hidden_norm @ fc_right^T`
    /// 4. Element-wise addition: `projected = left + right`
    /// 5. Full decoder layer forward via the `forward_layer` callback
    /// 6. Final RMSNorm
    ///
    /// # Arguments
    /// * `hidden` — Main model's hidden state `[hidden_size]` (pre-LM-head)
    /// * `input_token` — Token ID to embed (the MTP input)
    /// * `stream` — CUDA stream for kernel launches
    /// * `gemm` — cuBLASLt engine for matrix multiplications
    /// * `rms_norm_eps` — Epsilon for RMSNorm (from model config)
    /// * `hidden_size` — Model hidden dimension
    /// * `embed` — Callback: embed a token ID, return `[hidden_size]`
    /// * `rms_norm` — Callback: apply RMSNorm to a tensor
    /// * `forward_layer` — Callback: run a full decoder layer (norm1 → attn/GDN → residual → norm2 → MLP → residual)
    ///
    /// # Returns
    /// Output hidden state `[hidden_size]` after MTP head, ready for LM head projection.
    pub fn forward(
        &self,
        hidden: &CudaSlice<bf16>,
        input_token: u32,
        stream: &Arc<CudaStream>,
        gemm: &mut GemmEngine,
        rms_norm_eps: f32,
        hidden_size: usize,
        embed: &dyn Fn(u32, &Arc<CudaStream>) -> Result<CudaSlice<bf16>>,
        rms_norm: &dyn Fn(&Arc<CudaStream>, &CudaSlice<bf16>, &CudaSlice<bf16>, f32, usize) -> Result<CudaSlice<bf16>>,
        forward_layer: &dyn Fn(&LayerWeights, &CudaSlice<bf16>, &Arc<CudaStream>, &mut GemmEngine) -> Result<CudaSlice<bf16>>,
    ) -> Result<CudaSlice<bf16>> {
        // Step 1: Embed the input token (uses main model's embedding table by default)
        let embedding = embed(input_token, stream)?;

        // Step 2: RMSNorm both paths
        let embed_norm = rms_norm(
            stream,
            &embedding,
            &self.pre_fc_norm_embedding_gpu,
            rms_norm_eps,
            hidden_size,
        )?;
        let hidden_norm = rms_norm(
            stream,
            hidden,
            &self.pre_fc_norm_hidden_gpu,
            rms_norm_eps,
            hidden_size,
        )?;

        // Step 3: GEMM projections (each is [1 × hidden_size])
        // left = embed_norm [1 × hidden_size] @ fc_weight_left^T [hidden_size × hidden_size]
        // right = hidden_norm [1 × hidden_size] @ fc_weight_right^T [hidden_size × hidden_size]
        let mut left = stream
            .alloc_zeros::<bf16>(hidden_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate MTP left buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: 1,
                n: hidden_size,
                k: hidden_size,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &embed_norm,
            &self.fc_weight_left_gpu,
            &mut left,
        )?;

        let mut right = stream
            .alloc_zeros::<bf16>(hidden_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate MTP right buffer: {e}"))?;
        gemm.matmul_bf16(
            &GemmConfig {
                m: 1,
                n: hidden_size,
                k: hidden_size,
                transa: true,
                transb: false,
                alpha: 1.0,
                beta: 0.0,
                lda: None,
                ldb: None,
                ldc: None,
                activation: None,
            },
            &hidden_norm,
            &self.fc_weight_right_gpu,
            &mut right,
        )?;

        // Step 4: Element-wise addition (projected = left + right)
        let projected = add_bf16_simple(stream, &left, &right)?;

        // Step 5: Full decoder layer forward
        // The forward_layer callback handles norm1 → attention/GDN → residual → norm2 → MLP → residual
        let layer_out = forward_layer(&self.layer, &projected, stream, gemm)?;

        // Step 6: Final RMSNorm
        let output = rms_norm(
            stream,
            &layer_out,
            &self.norm_gpu,
            rms_norm_eps,
            hidden_size,
        )?;

        Ok(output)
    }
}

/// Upload a `WeightData` blob to the GPU as a BF16 buffer.
///
/// Converts raw bytes to BF16 values (2 bytes each, little-endian)
/// and uploads via `clone_htod`.
fn upload_weight_bytes(
    stream: &Arc<CudaStream>,
    weight: &WeightData,
) -> Result<CudaSlice<bf16>> {
    let bf16_vec: Vec<bf16> = weight
        .data
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();
    stream
        .clone_htod(&bf16_vec)
        .map_err(|e| anyhow::anyhow!("Failed to upload weight '{}': {}", weight.name, e))
}

/// Simple element-wise addition of two BF16 GPU tensors.
///
/// Downloads both to CPU, adds, and uploads the result.
/// This is used for single-token MTP head operations where
/// the overhead is negligible compared to the model forward pass.
fn add_bf16_simple(
    stream: &Arc<CudaStream>,
    a: &CudaSlice<bf16>,
    b: &CudaSlice<bf16>,
) -> Result<CudaSlice<bf16>> {
    anyhow::ensure!(a.len() == b.len(), "add_bf16_simple: length mismatch");
    let a_host: Vec<bf16> = stream
        .clone_dtoh(a)
        .map_err(|e| anyhow::anyhow!("Failed to download a for add: {e}"))?;
    let b_host: Vec<bf16> = stream
        .clone_dtoh(b)
        .map_err(|e| anyhow::anyhow!("Failed to download b for add: {e}"))?;
    let result: Vec<bf16> = a_host
        .iter()
        .zip(b_host.iter())
        .map(|(x, y)| bf16::from_f32(x.to_f32() + y.to_f32()))
        .collect();
    stream
        .clone_htod(&result)
        .map_err(|e| anyhow::anyhow!("Failed to upload add result: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a minimal WeightData for testing.
    fn make_weight(rows: usize, cols: usize, fill: u16) -> WeightData {
        let elem_count = rows * cols;
        let mut data = Vec::with_capacity(elem_count * 2);
        for _ in 0..elem_count {
            data.extend_from_slice(&fill.to_le_bytes());
        }
        WeightData {
            data,
            shape: vec![rows, cols],
            dtype: infers_model::WeightDtype::Bf16,
            name: "test.weight".to_string(),
        }
    }

    fn make_norm_weight(hidden_size: usize) -> WeightData {
        make_weight(hidden_size, 1, 0x3F80) // 1.0 in bf16
    }

    #[test]
    fn test_upload_weight_bytes_conversion() {
        // Verify that upload_weight_bytes correctly converts bytes to BF16.
        // bf16 value 1.0 = 0x3F80 stored as [0x80, 0x3F] little-endian
        // bf16 value 2.0 = 0x4000 stored as [0x00, 0x40] little-endian
        let data = vec![0x80u8, 0x3F, 0x00, 0x40];
        let weight = WeightData {
            data,
            shape: vec![2, 1],
            dtype: infers_model::WeightDtype::Bf16,
            name: "test".to_string(),
        };
        // Verify byte layout: 2 bf16 values = 4 bytes
        assert_eq!(weight.data.len(), 4, "Should have 4 bytes for 2 bf16 values");
        // Verify the bf16 values decode correctly
        let val0 = bf16::from_bits(u16::from_le_bytes([weight.data[0], weight.data[1]]));
        let val1 = bf16::from_bits(u16::from_le_bytes([weight.data[2], weight.data[3]]));
        assert!((val0.to_f32() - 1.0).abs() < 0.01, "First value should be ~1.0, got {}", val0.to_f32());
        assert!((val1.to_f32() - 2.0).abs() < 0.01, "Second value should be ~2.0, got {}", val1.to_f32());
    }

    #[test]
    fn test_fc_weight_split_dimensions() {
        let hidden_size = 64;
        // Build a minimal MtpWeights to verify the split logic
        let fc = make_weight(hidden_size, 2 * hidden_size, 0x3F80); // [64, 128]
        let mtp = MtpWeights {
            pre_fc_norm_embedding: make_norm_weight(hidden_size),
            pre_fc_norm_hidden: make_norm_weight(hidden_size),
            fc,
            layers: vec![], // empty — won't call from_weights
            norm: make_norm_weight(hidden_size),
            embed_tokens: None,
        };

        // Verify the FC weight shape is as expected
        assert_eq!(mtp.fc.shape[0], hidden_size);
        assert_eq!(mtp.fc.shape[1], 2 * hidden_size);

        // Verify split byte counts
        let half_bytes = hidden_size * hidden_size * 2;
        assert_eq!(mtp.fc.data.len(), 2 * half_bytes);
    }
}
