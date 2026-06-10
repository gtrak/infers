//! MTP metrics tracking for speculative decoding performance.
//!
//! Provides lightweight metrics collection for MTP acceptance rate,
//! tokens saved, and speedup estimation. These metrics can be exposed
//! via Prometheus or logged for analysis.

/// Tracks MTP speculative decoding performance metrics.
///
/// Collects running statistics on draft token acceptance rate,
/// tokens saved via speculative decoding, and provides helpers
/// for speedup estimation.
#[derive(Debug, Clone)]
pub struct MtpMetrics {
    /// Total number of draft tokens generated.
    total_drafts: u64,
    /// Total number of draft tokens accepted by the main model.
    total_accepted: u64,
    /// Running count of verification steps performed.
    verification_steps: u64,
    /// Running sum of acceptance rates (for rolling average).
    rate_sum: f64,
}

impl Default for MtpMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl MtpMetrics {
    /// Create a new MTP metrics tracker with zeroed counters.
    pub fn new() -> Self {
        Self {
            total_drafts: 0,
            total_accepted: 0,
            verification_steps: 0,
            rate_sum: 0.0,
        }
    }

    /// Record a single verification step result.
    ///
    /// # Arguments
    /// * `accepted` — Number of accepted tokens in this step.
    /// * `total` — Total number of draft tokens in this step.
    pub fn record_step(&mut self, accepted: usize, total: usize) {
        if total == 0 {
            return;
        }
        self.total_drafts += total as u64;
        self.total_accepted += accepted as u64;
        self.verification_steps += 1;
        self.rate_sum += accepted as f64 / total as f64;
    }

    /// Overall acceptance rate across all steps (0.0 to 1.0).
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafts == 0 {
            return 0.0;
        }
        self.total_accepted as f32 / self.total_drafts as f32
    }

    /// Average acceptance rate per verification step.
    pub fn average_step_rate(&self) -> f32 {
        if self.verification_steps == 0 {
            return 0.0;
        }
        (self.rate_sum / self.verification_steps as f64) as f32
    }

    /// Total number of draft tokens generated.
    pub fn total_drafts(&self) -> u64 {
        self.total_drafts
    }

    /// Total number of draft tokens accepted.
    pub fn total_accepted(&self) -> u64 {
        self.total_accepted
    }

    /// Number of tokens saved by speculative decoding.
    ///
    /// Each accepted draft token that is beyond the first one represents
    /// a token that was generated without a full model forward pass.
    /// The first accepted token corresponds to the main model's own output,
    /// so only `accepted - 1` tokens are saved per step.
    ///
    /// However, in practice we count all accepted tokens as savings
    /// because the main model still ran to verify them. The true savings
    /// come from generating multiple draft tokens in one MTP forward pass
    /// vs. multiple main model decode steps.
    pub fn tokens_saved(&self) -> u64 {
        self.total_accepted
    }

    /// Estimated speedup factor relative to non-speculative decoding.
    ///
    /// A rough estimate: if acceptance rate is `r` and draft count is `k`,
    /// speedup ≈ 1 / (1 - r + r/k). For k=2, r=0.8 → ~1.67x.
    ///
    /// Returns `None` if no steps have been recorded.
    pub fn estimated_speedup(&self, num_drafts: usize) -> Option<f32> {
        if self.verification_steps == 0 || num_drafts == 0 {
            return None;
        }
        let rate = self.acceptance_rate() as f64;
        if rate >= 1.0 {
            return Some(num_drafts as f32);
        }
        let k = num_drafts as f64;
        let speedup = 1.0 / (1.0 - rate + rate / k);
        Some(speedup as f32)
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        self.total_drafts = 0;
        self.total_accepted = 0;
        self.verification_steps = 0;
        self.rate_sum = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_metrics_are_zero() {
        let m = MtpMetrics::new();
        assert_eq!(m.total_drafts(), 0);
        assert_eq!(m.total_accepted(), 0);
        assert_eq!(m.acceptance_rate(), 0.0);
        assert_eq!(m.average_step_rate(), 0.0);
        assert!(m.estimated_speedup(2).is_none());
    }

    #[test]
    fn record_step_updates_counts() {
        let mut m = MtpMetrics::new();
        m.record_step(2, 3); // 2 accepted out of 3
        assert_eq!(m.total_drafts(), 3);
        assert_eq!(m.total_accepted(), 2);
    }

    #[test]
    fn acceptance_rate_multiple_steps() {
        let mut m = MtpMetrics::new();
        m.record_step(2, 3); // 2/3 = 0.666
        m.record_step(1, 2); // 1/2 = 0.5
        assert!((m.acceptance_rate() - 0.6).abs() < 0.01); // (2+1)/(3+2) = 3/5 = 0.6
    }

    #[test]
    fn average_step_rate() {
        let mut m = MtpMetrics::new();
        m.record_step(3, 4); // rate = 0.75
        m.record_step(1, 4); // rate = 0.25
        assert!((m.average_step_rate() - 0.5).abs() < 0.01); // (0.75 + 0.25) / 2 = 0.5
    }

    #[test]
    fn estimated_speedup_with_high_acceptance() {
        let mut m = MtpMetrics::new();
        for _ in 0..10 {
            m.record_step(4, 5); // 80% acceptance
        }
        let speedup = m.estimated_speedup(2).unwrap();
        // k=2, r=0.8: speedup = 1/(1-0.8+0.8/2) = 1/0.6 = 1.67
        assert!((speedup - 1.67).abs() < 0.1, "Expected ~1.67, got {}", speedup);
    }

    #[test]
    fn record_step_zero_total_is_noop() {
        let mut m = MtpMetrics::new();
        m.record_step(0, 0);
        assert_eq!(m.total_drafts(), 0);
        assert_eq!(m.verification_steps(), 0);
    }

    #[test]
    fn reset_clears_all() {
        let mut m = MtpMetrics::new();
        m.record_step(3, 4);
        m.reset();
        assert_eq!(m.total_drafts(), 0);
        assert_eq!(m.total_accepted(), 0);
        assert_eq!(m.acceptance_rate(), 0.0);
    }

    // Helper needed for test
    impl MtpMetrics {
        fn verification_steps(&self) -> u64 { self.verification_steps }
    }
}
