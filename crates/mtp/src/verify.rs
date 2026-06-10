//! Verification result for MTP speculative decoding.
//!
//! After the main model verifies draft tokens from the MTP head,
//! a `VerificationResult` describes which tokens were accepted
//! and which (if any) should be regenerated.

/// Result of verifying MTP draft tokens against the main model.
///
/// The verification process compares each draft token against the
/// main model's greedy prediction at the same position. The longest
/// prefix of matching tokens is accepted; the first mismatch (if any)
/// produces a corrected token from the main model.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Draft tokens that matched the main model's prediction.
    /// Consecutive from the start — this is the longest valid prefix.
    pub accepted_tokens: Vec<u32>,

    /// Corrected token from the main model at the first mismatch position.
    /// `None` if ALL draft tokens were accepted (no mismatch).
    pub rejected_token: Option<u32>,

    /// Fraction of draft tokens that were accepted (0.0 to 1.0).
    pub acceptance_rate: f32,
}

impl VerificationResult {
    /// Create a new verification result.
    pub fn new(
        accepted_tokens: Vec<u32>,
        rejected_token: Option<u32>,
        acceptance_rate: f32,
    ) -> Self {
        Self {
            accepted_tokens,
            rejected_token,
            acceptance_rate,
        }
    }

    /// Number of accepted tokens.
    pub fn num_accepted(&self) -> usize {
        self.accepted_tokens.len()
    }

    /// Total number of draft tokens that were verified.
    pub fn num_drafts(&self) -> usize {
        self.accepted_tokens.len() + if self.rejected_token.is_some() { 1 } else { 0 }
    }

    /// Whether all draft tokens were accepted (no correction needed).
    pub fn all_accepted(&self) -> bool {
        self.rejected_token.is_none()
    }

    /// Whether any tokens were accepted at all.
    pub fn any_accepted(&self) -> bool {
        !self.accepted_tokens.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_result_new() {
        let result = VerificationResult::new(vec![101, 102], Some(103), 0.66);
        assert_eq!(result.accepted_tokens, vec![101, 102]);
        assert_eq!(result.rejected_token, Some(103));
        assert!((result.acceptance_rate - 0.66).abs() < 0.01);
    }

    #[test]
    fn num_accepted_counts_accepted() {
        let result = VerificationResult::new(vec![101, 102], Some(103), 0.66);
        assert_eq!(result.num_accepted(), 2);
    }

    #[test]
    fn num_drafts_with_rejection() {
        let result = VerificationResult::new(vec![101, 102], Some(103), 0.66);
        assert_eq!(result.num_drafts(), 3);
    }

    #[test]
    fn num_drafts_all_accepted() {
        let result = VerificationResult::new(vec![101, 102, 103], None, 1.0);
        assert_eq!(result.num_drafts(), 3);
    }

    #[test]
    fn all_accepted_true_when_no_rejection() {
        let result = VerificationResult::new(vec![101, 102], None, 1.0);
        assert!(result.all_accepted());
    }

    #[test]
    fn all_accepted_false_when_rejection() {
        let result = VerificationResult::new(vec![101], Some(102), 0.5);
        assert!(!result.all_accepted());
    }

    #[test]
    fn any_accepted_true() {
        let result = VerificationResult::new(vec![101], None, 1.0);
        assert!(result.any_accepted());
    }

    #[test]
    fn any_accepted_false_when_empty() {
        let result = VerificationResult::new(vec![], Some(101), 0.0);
        assert!(!result.any_accepted());
    }
}
