//! Pipeline parallelism engine with microbatching.
//!
//! Orchestrates PP=2 across two GPUs using stage partitioning, NCCL P2P
//! communication, and microbatch scheduling to hide pipeline bubbles.
//!
//! # Pipeline Flow
//!
//! 1. Split batch into microbatches
//! 2. Stage 0 processes each microbatch (layers 0-31)
//! 3. Hidden states sent from GPU0 → GPU1 via NCCL P2P
//! 4. Stage 1 processes hidden states (layers 32-63)
//! 5. Logits → token sampling per microbatch

use std::sync::Arc;

use anyhow::{Context, Result};
use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::CudaStream;
use infers_model::config::ModelConfig;
use infers_model::sharding::{shard_weights_for_stage, split_layers_pp};
use infers_model::weights::WeightRegistry;

use crate::microbatch::{Microbatch, MicrobatchScheduler, Request};
use crate::stage::{PipelineStage, StageState};

/// Timing information for a single pipeline forward pass.
#[derive(Debug, Clone, Default)]
pub struct PipelineTiming {
    /// Total wall-clock time for the batch (nanoseconds).
    pub total_time_ns: u64,
    /// Time GPU 0 was actively computing (nanoseconds).
    pub gpu0_active_ns: u64,
    /// Time GPU 1 was actively computing (nanoseconds).
    pub gpu1_active_ns: u64,
    /// Time spent in NCCL communication (nanoseconds).
    pub comm_time_ns: u64,
}

impl PipelineTiming {
    /// Calculate bubble fraction: fraction of time GPUs are idle.
    ///
    /// Bubble fraction = 1 - (gpu0_active + gpu1_active) / (2 * total_time)
    pub fn bubble_fraction(&self) -> f64 {
        if self.total_time_ns == 0 {
            return 0.0;
        }
        1.0 - (self.gpu0_active_ns + self.gpu1_active_ns) as f64
            / (2.0 * self.total_time_ns as f64)
    }
}

/// Result of a pipeline forward pass.
#[derive(Debug)]
pub struct PipelineOutput {
    /// Sampled token IDs for each request.
    pub tokens: Vec<Vec<u32>>,
    /// Timing information for performance analysis.
    pub timing: PipelineTiming,
}

/// Pipeline parallelism engine orchestrating two stages with microbatching.
///
/// Splits the model into two pipeline stages (PP=2), each on its own GPU.
/// Batches are divided into microbatches to keep both GPUs busy and
/// minimize the pipeline bubble.
pub struct PipelineEngine {
    /// The two pipeline stages (stage 0 = GPU0, stage 1 = GPU1).
    pub stages: Vec<PipelineStage>,
    /// Number of requests per microbatch.
    pub microbatch_size: usize,
    /// Model configuration (shared reference).
    pub config: Arc<ModelConfig>,
    /// Per-stage state (KV cache + GDN states).
    pub stage_states: Vec<StageState>,
}

impl PipelineEngine {
    /// Create a new pipeline engine for PP=2.
    ///
    /// # Arguments
    ///
    /// * `config` — Model configuration (layer count, hidden size, etc.)
    /// * `weights` — Full weight registry (will be split across stages)
    /// * `microbatch_size` — Requests per microbatch
    /// * `gpu0_stream` — CUDA stream for GPU 0 (stage 0)
    /// * `gpu1_stream` — CUDA stream for GPU 1 (stage 1)
    /// * `num_pages` — Total KV cache pages
    /// * `page_size` — Tokens per KV page
    /// * `max_cache_bytes` — Maximum KV cache memory budget
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Arc<ModelConfig>,
        weights: WeightRegistry,
        microbatch_size: usize,
        gpu0_stream: Arc<CudaStream>,
        gpu1_stream: Arc<CudaStream>,
        num_pages: usize,
        page_size: usize,
        max_cache_bytes: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            config.num_hidden_layers >= 2,
            "Need at least 2 layers for PP=2, got {}",
            config.num_hidden_layers
        );

        let num_stages = 2;
        let stage_ranges = split_layers_pp(&config, num_stages);

        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        // Wrap nccl_comms in Arc so both stages can share it
        let nccl_comms = Arc::new(NcclCommunicator::new(vec![gpu0_stream, gpu1_stream])
            .context("Failed to create NCCL communicator for PP")?);

        // Build stage 0 (GPU 0, layers 0..mid)
        let stage0_weights = shard_weights_for_stage(&weights, &stage_ranges[0]);
        let stage0 = PipelineStage::new(
            0,
            0,
            stage_ranges[0].start,
            stage_ranges[0].end,
            stage0_weights,
            nccl_comms.clone(),
            0, // rank
            1, // peer_rank,
        );

        let stage0_state = StageState::new(
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            max_cache_bytes,
        );

        // Build stage 1 (GPU 1, layers mid..end)
        let stage1_weights = shard_weights_for_stage(&weights, &stage_ranges[1]);
        let stage1 = PipelineStage::new(
            1,
            1,
            stage_ranges[1].start,
            stage_ranges[1].end,
            stage1_weights,
            nccl_comms.clone(),
            1, // rank
            0, // peer_rank,
        );

        let stage1_state = StageState::new(
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            max_cache_bytes,
        );

        Ok(Self {
            stages: vec![stage0, stage1],
            microbatch_size,
            config,
            stage_states: vec![stage0_state, stage1_state],
        })
    }

    /// Number of pipeline stages (always 2 for PP=2).
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Run a full batch through the pipeline with microbatching.
    ///
    /// Splits requests into microbatches, processes them through stage 0,
    /// transfers hidden states via NCCL, processes through stage 1,
    /// and samples output tokens.
    ///
    /// Returns output tokens and timing information.
    pub fn forward_batch(&mut self, requests: Vec<Request>) -> Result<PipelineOutput> {
        let mut scheduler = MicrobatchScheduler::new(self.microbatch_size);
        for req in requests {
            scheduler.add_request(req);
        }

        let mut all_tokens: Vec<Vec<u32>> = Vec::new();
        let mut timing = PipelineTiming::default();

        // Pipeline loop: process microbatches sequentially
        // For PP=2, the pattern is:
        //   Step 1: Stage 0 processes microbatch → send to stage 1
        //   Step 2: Stage 1 recv → processes → sample tokens
        // With multiple microbatches, steps overlap to keep GPUs busy

        let mut microbatch_count = 0;

        while scheduler.is_busy() {
            // --- Stage 0: Form and process next microbatch ---
            if let Some(mut microbatch) = scheduler.next_microbatch() {
                let hidden = self
                    .forward_stage0(&microbatch)
                    .context("Stage 0 forward failed")?;
                microbatch.hidden_states = Some(hidden);

                // Placeholder: NCCL send from stage 0 to stage 1
                // In production: self.stages[0].comm.send_hidden(hidden_gpu_slice)
                // Currently hidden_states is Vec<u8> placeholder — actual GPU buffer
                // allocation requires CudaSlice<bf16> from the CUDA stream context.

                scheduler.in_flight.push(microbatch);
            }

            // --- Stage 1: Process received microbatches ---
            // Collect tokens from completed microbatches
            let completed: Vec<usize> = scheduler
                .in_flight
                .iter()
                .enumerate()
                .filter(|(_, mb)| mb.stage == 1)
                .map(|(i, _)| i)
                .collect();

            for idx in completed.into_iter().rev() {
                let microbatch = &scheduler.in_flight[idx];

                // Allocate receive buffer for hidden states
                let hidden_size = self.config.hidden_size;
                let batch_tokens = microbatch.num_tokens();
                let _num_elements = batch_tokens * hidden_size;

                // The actual GPU buffer allocation would happen here.
                // For now, this is a placeholder since GPU allocation
                // depends on the CUDA stream context of stage 1.
                //
                // In production:
                //   let mut hidden = stage1_stream.alloc_zeros::<bf16>(num_elements)?;
                //   self.stages[1].comm.recv_hidden(&mut hidden)?;
                //   let output = self.forward_stage1(microbatch)?;
                //   let tokens = self.sample(&output, microbatch)?;

                // Placeholder: produce dummy tokens matching batch size
                for req in &microbatch.requests {
                    all_tokens.push(vec![req.tokens[0]]); // Echo first input token
                }
            }

            scheduler.advance_pipeline(self.num_stages());
            microbatch_count += 1;
        }

        timing.total_time_ns = microbatch_count as u64 * 1000; // Placeholder timing

        Ok(PipelineOutput {
            tokens: all_tokens,
            timing,
        })
    }

    /// Forward pass through stage 0 (layers 0 to mid).
    ///
    /// Embeds input tokens, runs through all GDN and attention layers
    /// assigned to stage 0, and returns hidden states for transfer to
    /// stage 1.
    ///
    /// NOTE: This is a placeholder implementation. The actual forward
    /// pass logic depends on Phase 4 (single-GPU forward pass) which
    /// provides embed_batch, gdn_forward, attention_forward, etc.
    fn forward_stage0(&self, _microbatch: &Microbatch) -> Result<Vec<u8>> {
        // Placeholder: return empty vec
        // In production, this would:
        //   1. Embed input tokens → hidden states on GPU 0
        //   2. For each layer in stage 0 range:
        //      a. GDN layer: gdn_forward(layer_idx, hidden, microbatch, &self.stages[0])
        //      b. Attention layer: attention_forward(layer_idx, hidden, microbatch, &self.stages[0])
        //   3. Return final hidden state tensor
        Ok(Vec::new())
    }

    /// Create sessions in both stages for each request.
    pub fn create_sessions(&mut self, microbatch: &Microbatch) -> Result<Vec<usize>> {
        let mut session_ids = Vec::new();
        for _req in &microbatch.requests {
            let seq_id = self.stage_states[0].create_session();
            let _seq_id1 = self.stage_states[1].create_session();
            session_ids.push(seq_id);
        }
        Ok(session_ids)
    }

    /// Free sessions in both stages.
    pub fn free_sessions(&mut self, session_ids: &[usize]) {
        for &sid in session_ids {
            self.stage_states[0].free_session(sid);
            self.stage_states[1].free_session(sid);
        }
    }
}

/// Compute the bubble fraction for a given number of microbatches with PP=2.
///
/// Bubble fraction = (num_stages - 1) / (num_microbatches + num_stages - 1)
///
/// For PP=2: bubble = 1 / (num_microbatches + 1)
pub fn compute_bubble_fraction(num_microbatches: usize) -> f64 {
    if num_microbatches == 0 {
        return 1.0;
    }
    1.0 / (num_microbatches as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bubble_fraction_formula() {
        // For PP=2 with 1 microbatch: bubble = 1/2 = 0.5
        assert!((compute_bubble_fraction(1) - 0.5).abs() < 1e-10);

        // For PP=2 with 3 microbatches: bubble = 1/4 = 0.25
        assert!((compute_bubble_fraction(3) - 0.25).abs() < 1e-10);

        // For PP=2 with 7 microbatches: bubble = 1/8 = 0.125
        assert!((compute_bubble_fraction(7) - 0.125).abs() < 1e-10);

        // For PP=2 with 15 microbatches: bubble = 1/16 = 0.0625
        assert!((compute_bubble_fraction(15) - 0.0625).abs() < 1e-10);

        // Edge case: 0 microbatches → 1.0 (100% bubble, no work done)
        assert!((compute_bubble_fraction(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_timing_default() {
        let timing = PipelineTiming::default();
        assert_eq!(timing.total_time_ns, 0);
        assert_eq!(timing.bubble_fraction(), 0.0);
    }

    #[test]
    fn test_pipeline_timing_bubble_fraction() {
        // 50% bubble: GPU0 active 50ns, GPU1 active 50ns, total 100ns
        let timing = PipelineTiming {
            total_time_ns: 100,
            gpu0_active_ns: 50,
            gpu1_active_ns: 50,
            comm_time_ns: 0,
        };
        assert!((timing.bubble_fraction() - 0.5).abs() < 1e-10);

        // 25% bubble: GPU0 active 75ns, GPU1 active 75ns, total 100ns
        let timing = PipelineTiming {
            total_time_ns: 100,
            gpu0_active_ns: 75,
            gpu1_active_ns: 75,
            comm_time_ns: 0,
        };
        assert!((timing.bubble_fraction() - 0.25).abs() < 1e-10);

        // 0% bubble (perfect utilization)
        let timing = PipelineTiming {
            total_time_ns: 100,
            gpu0_active_ns: 100,
            gpu1_active_ns: 100,
            comm_time_ns: 0,
        };
        assert!((timing.bubble_fraction() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ensure_session_states() {
        // Unit test for the ensure_session_states logic
        // This doesn't create a full PipelineEngine (needs GPUs)
        // but tests the concept through StageState directly
        let config_json = r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":8,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#;
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        let mut stage_state = StageState::new(1000, 16, 4, 256, 1024 * 1024);
        stage_state.ensure_gdn_states(0, &config, 0, 8);
        assert_eq!(stage_state.num_gdn_states(), 6);
    }

    #[test]
    fn test_pipeline_output_creation() {
        let output = PipelineOutput {
            tokens: vec![vec![1, 2, 3], vec![4, 5]],
            timing: PipelineTiming::default(),
        };
        assert_eq!(output.tokens.len(), 2);
        assert_eq!(output.tokens[0], vec![1, 2, 3]);
    }
}