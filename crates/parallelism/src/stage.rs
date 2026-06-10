//! Pipeline stage data structures.
//!
//! Defines `PipelineStage` (stage identity, weights, communication) and
//! `StageState` (per-stage KV cache and GDN state management) for use
//! by the pipeline engine.

use std::collections::HashMap;
use std::sync::Arc;

use infers_cuda::nccl::NcclCommunicator;
use infers_kv::manager::{PagedKvManager, SequenceId};
use infers_model::config::{LayerType, ModelConfig};

use crate::comm::StageComm;

/// A single pipeline stage holding a subset of layers on one GPU.
///
/// PP=2 splits 64 layers into two stages of 32 layers each. Each stage
/// runs on a separate GPU and communicates hidden states via NCCL.
#[derive(Debug)]
pub struct PipelineStage {
    /// Stage index (0 for first half, 1 for second half).
    pub stage_id: usize,
    /// GPU device ID this stage runs on.
    pub gpu_id: usize,
    /// First layer index (inclusive) managed by this stage.
    pub start_layer: usize,
    /// Last layer index (exclusive) managed by this stage.
    pub end_layer: usize,
    /// Sharded weights for this stage's layers.
    pub weights: infers_model::weights::WeightRegistry,
    /// P2P communicator for sending/receiving hidden states.
    pub comm: StageComm,
}

impl PipelineStage {
    /// Create a new pipeline stage.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stage_id: usize,
        gpu_id: usize,
        start_layer: usize,
        end_layer: usize,
        weights: infers_model::weights::WeightRegistry,
        nccl: Arc<NcclCommunicator>,
        rank: usize,
        peer_rank: usize,
    ) -> Self {
        Self {
            stage_id,
            gpu_id,
            start_layer,
            end_layer,
            weights,
            comm: StageComm::new(nccl, rank, peer_rank),
        }
    }

    /// Number of layers in this stage.
    pub fn num_layers(&self) -> usize {
        self.end_layer - self.start_layer
    }

    /// Whether this stage manages the given layer index.
    pub fn contains_layer(&self, layer_idx: usize) -> bool {
        (self.start_layer..self.end_layer).contains(&layer_idx)
    }
}

/// Lightweight GDN state descriptor used for state tracking.
///
/// Stores the hidden size for allocation; actual GPU buffer allocation
/// is handled by the backend's forward pass implementation.
#[derive(Debug, Clone)]
pub struct GdnStateRef {
    /// Hidden size this state was allocated for.
    pub hidden_size: usize,
    /// Whether the state has been initialized on GPU.
    pub is_initialized: bool,
}

impl GdnStateRef {
    /// Create a new uninitialized GDN state reference.
    pub fn new() -> Self {
        Self {
            hidden_size: 0,
            is_initialized: false,
        }
    }

    /// Create a GDN state reference with a known hidden size.
    pub fn with_hidden_size(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            is_initialized: false,
        }
    }

    /// Mark this state as initialized.
    pub fn mark_initialized(&mut self) {
        self.is_initialized = true;
    }
}

impl Default for GdnStateRef {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-stage state management: paged KV for attention, GDN states for
/// recurrent layers.
///
/// Each stage manages its own subset of layers. Full-attention layers use
/// the paged KV system (`PagedKvManager`). GDN layers use recurrent state
/// vectors (`GdnStateRef`).
#[derive(Debug)]
pub struct StageState {
    /// Paged KV manager for full-attention layers in this stage's range.
    pub kv_manager: PagedKvManager,
    /// Per-session GDN recurrent states for layers in this stage's range.
    /// Key: `(session_id, layer_idx)` → `GdnStateRef`.
    pub gdn_states: HashMap<(usize, usize), GdnStateRef>,
}

impl StageState {
    /// Create a new stage state with the given KV cache parameters.
    pub fn new(
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_cache_bytes: usize,
    ) -> Self {
        Self {
            kv_manager: PagedKvManager::new(
                num_pages,
                page_size,
                num_kv_heads,
                head_dim,
                max_cache_bytes,
                num_pages * page_size * num_kv_heads * head_dim * 2,
            ),
            gdn_states: HashMap::new(),
        }
    }

    /// Allocate KV pages for a new session's attention layers in this stage.
    pub fn create_session(&mut self) -> SequenceId {
        self.kv_manager.create_sequence()
    }

    /// Ensure GDN states exist for all GDN layers in this stage's range.
    pub fn ensure_gdn_states(
        &mut self,
        session_id: usize,
        config: &ModelConfig,
        start_layer: usize,
        end_layer: usize,
    ) {
        for layer_idx in start_layer..end_layer {
            if config.get_layer_type(layer_idx) == LayerType::GatedDeltaNet {
                self.gdn_states
                    .entry((session_id, layer_idx))
                    .or_insert_with(|| GdnStateRef::with_hidden_size(config.hidden_size));
            }
        }
    }

    /// Get a mutable reference to a GDN state for a session and layer.
    pub fn get_gdn_state(
        &mut self,
        session_id: usize,
        layer_idx: usize,
    ) -> Option<&mut GdnStateRef> {
        self.gdn_states.get_mut(&(session_id, layer_idx))
    }

    /// Free all resources for a session.
    pub fn free_session(&mut self, session_id: SequenceId) {
        let _ = self.kv_manager.delete_sequence(session_id);
        self.gdn_states
            .retain(|(sid, _), _| *sid != session_id);
    }

    /// Number of active sessions tracked by this stage.
    pub fn num_sessions(&self) -> usize {
        self.kv_manager.num_sequences()
    }

    /// Number of active GDN states tracked by this stage.
    pub fn num_gdn_states(&self) -> usize {
        self.gdn_states.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdn_state_ref_default() {
        let state = GdnStateRef::new();
        assert_eq!(state.hidden_size, 0);
        assert!(!state.is_initialized);
    }

    #[test]
    fn test_gdn_state_ref_with_hidden_size() {
        let mut state = GdnStateRef::with_hidden_size(5120);
        assert_eq!(state.hidden_size, 5120);
        assert!(!state.is_initialized);

        state.mark_initialized();
        assert!(state.is_initialized);
    }

    #[test]
    fn test_stage_state_creation() {
        let state = StageState::new(1000, 16, 4, 256, 1024 * 1024 * 1024);
        assert_eq!(state.num_sessions(), 0);
        assert_eq!(state.num_gdn_states(), 0);
    }

    #[test]
    fn test_create_and_free_session() {
        let mut state = StageState::new(1000, 16, 4, 256, 1024 * 1024 * 1024);

        let seq_id = state.create_session();
        assert_eq!(state.num_sessions(), 1);

        state.free_session(seq_id);
        assert_eq!(state.num_sessions(), 0);
    }

    #[test]
    fn test_stage_state_ensure_gdn_states() {
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":8,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let mut state = StageState::new(1000, 16, 4, 256, 1024 * 1024 * 1024);
        state.ensure_gdn_states(0, &config, 0, 8);

        // With 8 layers and default pattern (every 4th is full attention):
        // Layers 0,1,2 are GDN, layer 3 is full attention
        // Layers 4,5,6 are GDN, layer 7 is full attention
        // So 6 GDN states for 8 layers
        assert_eq!(state.num_gdn_states(), 6);

        // Access a specific GDN state
        let gdn = state.get_gdn_state(0, 0);
        assert!(gdn.is_some());
        assert_eq!(gdn.unwrap().hidden_size, 5120);
    }

    #[test]
    fn test_free_session_removes_gdn_states() {
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":8,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let mut state = StageState::new(1000, 16, 4, 256, 1024 * 1024 * 1024);
        state.ensure_gdn_states(0, &config, 0, 8);
        assert_eq!(state.num_gdn_states(), 6);

        state.free_session(0);
        assert_eq!(state.num_gdn_states(), 0);
    }

    #[test]
    fn test_gdn_states_across_multiple_sessions() {
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":4,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let mut state = StageState::new(1000, 16, 4, 256, 1024 * 1024 * 1024);
        state.ensure_gdn_states(0, &config, 0, 4);
        state.ensure_gdn_states(1, &config, 0, 4);

        // 2 sessions × 3 GDN layers each
        assert_eq!(state.num_gdn_states(), 6);

        // Free one session
        state.free_session(0);
        assert_eq!(state.num_gdn_states(), 3);
    }
}
