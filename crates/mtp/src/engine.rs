//! MTP engine orchestrating draft generation, verification, and acceptance.
//!
//! The `MtpEngine` wraps an `MtpHead` and manages the speculative decoding
//! lifecycle: generating draft tokens from the MTP head, verifying them
//! against the main model, accepting the longest valid prefix, and adapting
//! the number of draft tokens based on recent acceptance rates.

use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::gemm::GemmEngine;
use infers_cuda::{CudaSlice, CudaStream};
use infers_model::LayerWeights;

use crate::head::MtpHead;
use crate::verify::VerificationResult;

/// Collection of operations the MTP engine needs from the main model.
///
/// This struct bundles callbacks for all GPU operations that the MTP engine
/// requires. By passing these as callbacks, the MTP crate avoids depending
/// on the backend crate for kernel dispatch and CUDA resource management.
pub struct MtpOperations<'a> {
    /// Embed a single token ID, returning `[hidden_size]` hidden state.
    pub embed: &'a dyn Fn(u32, &Arc<CudaStream>) -> Result<CudaSlice<bf16>>,
    /// Apply RMSNorm to a tensor.
    pub rms_norm: &'a dyn Fn(
        &Arc<CudaStream>,
        &CudaSlice<bf16>,
        &CudaSlice<bf16>,
        f32,
        usize,
    ) -> Result<CudaSlice<bf16>>,
    /// Run a full transformer decoder layer forward (norm1 → attention/GDN → residual → norm2 → MLP → residual).
    pub forward_layer: &'a dyn Fn(
        &LayerWeights,
        &CudaSlice<bf16>,
        &Arc<CudaStream>,
        &mut GemmEngine,
    ) -> Result<CudaSlice<bf16>>,
    /// Project hidden state through the LM head, returning logits `[vocab_size]`.
    pub lm_head: &'a dyn Fn(
        &CudaSlice<bf16>,
        &Arc<CudaStream>,
        &mut GemmEngine,
    ) -> Result<CudaSlice<bf16>>,
    /// Greedy sample: return the argmax token from BF16 logits.
    /// The sample implementation handles BF16→FP32 conversion internally.
    pub sample: &'a dyn Fn(&CudaSlice<bf16>, &Arc<CudaStream>) -> Result<u32>,
    /// Run the full model forward pass for a single token: embed → all layers → final norm → LM head,
    /// returning logits `[vocab_size]`. Used for verification.
    pub full_forward: &'a dyn Fn(u32, &Arc<CudaStream>, &mut GemmEngine) -> Result<CudaSlice<bf16>>,
}

impl<'a> MtpOperations<'a> {
    /// Create a new `MtpOperations` from individual callbacks.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embed: &'a dyn Fn(u32, &Arc<CudaStream>) -> Result<CudaSlice<bf16>>,
        rms_norm: &'a dyn Fn(
            &Arc<CudaStream>,
            &CudaSlice<bf16>,
            &CudaSlice<bf16>,
            f32,
            usize,
        ) -> Result<CudaSlice<bf16>>,
        forward_layer: &'a dyn Fn(
            &LayerWeights,
            &CudaSlice<bf16>,
            &Arc<CudaStream>,
            &mut GemmEngine,
        ) -> Result<CudaSlice<bf16>>,
        lm_head: &'a dyn Fn(
            &CudaSlice<bf16>,
            &Arc<CudaStream>,
            &mut GemmEngine,
        ) -> Result<CudaSlice<bf16>>,
        sample: &'a dyn Fn(&CudaSlice<bf16>, &Arc<CudaStream>) -> Result<u32>,
        full_forward: &'a dyn Fn(u32, &Arc<CudaStream>, &mut GemmEngine) -> Result<CudaSlice<bf16>>,
    ) -> Self {
        Self {
            embed,
            rms_norm,
            forward_layer,
            lm_head,
            sample,
            full_forward,
        }
    }
}

/// Engine for MTP speculative decoding.
///
/// Coordinates the MTP head, draft generation, verification against the
/// main model, and acceptance logic. Tracks acceptance history for
/// adaptive draft token count.
pub struct MtpEngine {
    /// The MTP prediction head (GPU-resident weights + forward method).
    pub mtp_head: MtpHead,
    /// Base number of draft tokens to generate (1-4, 2 recommended).
    pub num_draft_tokens: usize,
    /// History of recent acceptance results (true = all drafts accepted).
    pub acceptance_history: Vec<bool>,
    /// Epsilon for RMSNorm (from model config).
    rms_norm_eps: f32,
    /// Model hidden dimension.
    hidden_size: usize,
}

impl MtpEngine {
    /// Construct a new MTP engine from model weights and config.
    ///
    /// # Arguments
    /// * `mtp_weights` — MTP weights from model loading
    /// * `config` — Model configuration
    /// * `num_draft_tokens` — Number of draft tokens per speculative step (1-4)
    /// * `stream` — CUDA stream for weight uploads
    pub fn new(
        mtp_weights: &infers_model::MtpWeights,
        config: &infers_model::ModelConfig,
        num_draft_tokens: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        let mtp_head = MtpHead::from_weights(mtp_weights, config, stream)?;
        Ok(Self {
            mtp_head,
            num_draft_tokens: num_draft_tokens.clamp(1, 4),
            acceptance_history: Vec::new(),
            rms_norm_eps: config.rms_norm_eps,
            hidden_size: config.hidden_size,
        })
    }

    /// Generate draft tokens from the MTP head.
    ///
    /// Iteratively runs the MTP head: for each step, runs `mtp_head.forward()`
    /// to get a hidden state, projects through the LM head, and samples greedily.
    /// Each draft step uses the previous draft token as input.
    ///
    /// # Arguments
    /// * `hidden` — Main model's hidden state `[hidden_size]` (pre-LM-head)
    /// * `last_token` — The last token generated by the main model
    /// * `num_drafts` — Number of draft tokens to generate
    /// * `stream` — CUDA stream for kernel launches
    /// * `gemm` — cuBLASLt engine
    /// * `ops` — Main model operations (embed, norms, layers, LM head, sampling)
    ///
    /// # Returns
    /// List of draft token IDs, one per MTP step.
    pub fn generate_drafts(
        &self,
        hidden: &CudaSlice<bf16>,
        last_token: u32,
        num_drafts: usize,
        stream: &Arc<CudaStream>,
        gemm: &mut GemmEngine,
        ops: &MtpOperations,
    ) -> Result<Vec<u32>> {
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_hidden = hidden.clone();
        let mut current_token = last_token;

        for _ in 0..num_drafts {
            // Step 1: Forward through MTP head → hidden state (pre-LM-head)
            let mtp_hidden = self.mtp_head.forward(
                &current_hidden,
                current_token,
                stream,
                gemm,
                self.rms_norm_eps,
                self.hidden_size,
                ops.embed,
                ops.rms_norm,
                ops.forward_layer,
            )?;

            // Step 2: Project through shared LM head → logits
            let logits = (ops.lm_head)(&mtp_hidden, stream, gemm)?;

            // Step 3: Greedy sample
            let token = (ops.sample)(&logits, stream)?;
            drafts.push(token);

            // Prepare for next iteration
            current_token = token;
            current_hidden = mtp_hidden;
        }

        Ok(drafts)
    }

    /// Verify draft tokens against the main model.
    ///
    /// For each draft token, runs the full model forward pass (embed → all
    /// layers → LM head) and compares the main model's greedy prediction
    /// against the draft token.
    ///
    /// The longest prefix of matching tokens is accepted. At the first
    /// mismatch, the main model's prediction replaces the draft token.
    ///
    /// # Arguments
    /// * `draft_tokens` — Draft tokens to verify
    /// * `stream` — CUDA stream
    /// * `gemm` — cuBLASLt engine
    /// * `ops` — Main model operations
    ///
    /// # Returns
    /// `VerificationResult` with accepted tokens and optional correction.
    pub fn verify_drafts(
        &self,
        draft_tokens: &[u32],
        stream: &Arc<CudaStream>,
        gemm: &mut GemmEngine,
        ops: &MtpOperations,
    ) -> Result<VerificationResult> {
        if draft_tokens.is_empty() {
            return Ok(VerificationResult::new(vec![], None, 1.0));
        }

        let num_drafts = draft_tokens.len();
        let mut accepted = 0usize;
        let mut rejected_token = None;

        for &draft_token in draft_tokens.iter() {
            // Run full forward pass for this draft token
            let logits = (ops.full_forward)(draft_token, stream, gemm)?;

            // Greedy sample from main model's logits
            let main_token = (ops.sample)(&logits, stream)?;

            if main_token == draft_token {
                accepted += 1;
            } else {
                // First mismatch: record the corrected token and stop
                rejected_token = Some(main_token);
                break;
            }
        }

        let accepted_tokens = draft_tokens[..accepted].to_vec();
        let acceptance_rate = if num_drafts > 0 {
            accepted as f32 / num_drafts as f32
        } else {
            1.0
        };

        Ok(VerificationResult::new(accepted_tokens, rejected_token, acceptance_rate))
    }

    /// Accept the longest valid prefix from a verification result.
    ///
    /// Returns all accepted tokens plus the correction token (if any).
    /// Updates acceptance history for adaptive draft count.
    ///
    /// # Arguments
    /// * `result` — Verification result from `verify_drafts`
    ///
    /// # Returns
    /// The sequence of tokens to append to the output:
    /// all accepted tokens (if any) followed by the correction (if any).
    pub fn accept_prefix(&mut self, result: &VerificationResult) -> Vec<u32> {
        let mut output_tokens = Vec::new();

        // Add all accepted tokens
        output_tokens.extend_from_slice(&result.accepted_tokens);

        // Add corrected token for first rejection (if any)
        if let Some(token) = result.rejected_token {
            output_tokens.push(token);
        }

        // Update acceptance history
        self.acceptance_history.push(result.all_accepted());

        output_tokens
    }

    /// Compute adaptive draft token count based on recent acceptance rate.
    ///
    /// If the last 10 steps had >80% full acceptance, increase draft count
    /// (up to max 4). If <30%, decrease (min 1). Otherwise keep current.
    ///
    /// Uses at most the last 10 history entries.
    pub fn adaptive_num_drafts(&self) -> usize {
        let window = 10;
        let history_len = self.acceptance_history.len();

        if history_len < window {
            return self.num_draft_tokens;
        }

        let recent: f32 = self.acceptance_history
            .iter()
            .rev()
            .take(window)
            .filter(|&&x| x)
            .count() as f32
            / window as f32;

        if recent > 0.8 {
            (self.num_draft_tokens + 1).min(4)
        } else if recent < 0.3 {
            self.num_draft_tokens.saturating_sub(1).max(1)
        } else {
            self.num_draft_tokens
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // MtpEngine unit tests: test non-GPU logic only.
    // We test the algorithmic logic via free functions to avoid constructing
    // an MtpEngine (which requires a GPU-initialized MtpHead).
    // ---------------------------------------------------------------------------

    /// Test adaptive_num_drafts logic in isolation.
    fn test_adaptive(history: &[bool], num_draft_tokens: usize) -> usize {
        // Inline the adaptive logic from MtpEngine::adaptive_num_drafts
        let window = 10;
        if history.len() < window {
            return num_draft_tokens;
        }
        let recent: f32 = history
            .iter()
            .rev()
            .take(window)
            .filter(|&&x| x)
            .count() as f32
            / window as f32;
        if recent > 0.8 {
            (num_draft_tokens + 1).min(4)
        } else if recent < 0.3 {
            num_draft_tokens.saturating_sub(1).max(1)
        } else {
            num_draft_tokens
        }
    }

    /// Test accept_prefix output logic in isolation.
    fn test_accept(result: &VerificationResult) -> (Vec<u32>, bool) {
        let mut output = result.accepted_tokens.clone();
        let all_accepted = result.rejected_token.is_none();
        if let Some(t) = result.rejected_token {
            output.push(t);
        }
        (output, all_accepted)
    }

    #[test]
    fn adaptive_below_window_returns_default() {
        let history = vec![true; 5]; // fewer than 10 entries
        assert_eq!(test_adaptive(&history, 2), 2);
    }

    #[test]
    fn adaptive_high_acceptance_increases() {
        let history = vec![true; 10]; // rate = 1.0 > 0.8
        assert_eq!(test_adaptive(&history, 2), 3);
    }

    #[test]
    fn adaptive_low_acceptance_decreases() {
        let history = vec![false; 10]; // rate = 0.0 < 0.3
        assert_eq!(test_adaptive(&history, 2), 1);
    }

    #[test]
    fn adaptive_stays_at_one() {
        let history = vec![false; 10];
        assert_eq!(test_adaptive(&history, 1), 1); // can't go below 1
    }

    #[test]
    fn adaptive_stays_at_four() {
        let history = vec![true; 10];
        assert_eq!(test_adaptive(&history, 4), 4); // can't go above 4
    }

    #[test]
    fn adaptive_mid_acceptance_stays() {
        let history: Vec<bool> = (0..10).map(|i| i % 2 == 0).collect(); // rate = 0.5
        assert_eq!(test_adaptive(&history, 2), 2);
    }

    #[test]
    fn accept_with_rejected_token() {
        let result = VerificationResult::new(vec![101, 102], Some(103), 0.66);
        let (tokens, all_ok) = test_accept(&result);
        assert_eq!(tokens, vec![101, 102, 103]);
        assert!(!all_ok);
    }

    #[test]
    fn accept_all_accepted_no_rejected() {
        let result = VerificationResult::new(vec![101, 102, 103], None, 1.0);
        let (tokens, all_ok) = test_accept(&result);
        assert_eq!(tokens, vec![101, 102, 103]);
        assert!(all_ok);
    }

    #[test]
    fn accept_empty_accepted_with_rejection() {
        let result = VerificationResult::new(vec![], Some(101), 0.0);
        let (tokens, _) = test_accept(&result);
        assert_eq!(tokens, vec![101]);
    }
}
