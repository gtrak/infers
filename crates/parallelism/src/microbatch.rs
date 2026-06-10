//! Microbatch scheduler for pipeline parallelism.
//!
//! Splits incoming requests into microbatches and tracks each microbatch's
//! progress through pipeline stages. Keeps both GPUs busy by interleaving
//! microbatches across stages.
//!
//! For PP=2, a batch of 8 requests with microbatch_size=2 produces 4
//! microbatches. The scheduler tracks which pipeline stage each microbatch
//! is in and advances them through stage 0 → stage 1 → complete.

/// A single inference request (tokenized input).
///
/// This is a simplified request type for pipeline parallelism. In production,
/// requests come from the scheduler crate's session management (Phase 6).
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request identifier.
    pub id: usize,
    /// Input token IDs (already tokenized).
    pub tokens: Vec<u32>,
    /// Session identifier for KV cache lookup.
    pub session_id: usize,
}

impl Request {
    /// Create a new request.
    pub fn new(id: usize, tokens: Vec<u32>, session_id: usize) -> Self {
        Self { id, tokens, session_id }
    }
}

/// A group of requests processed together as a pipeline unit.
///
/// Microbatches flow through pipeline stages sequentially:
/// 1. Stage 0 processes the microbatch and produces hidden states
/// 2. Hidden states are sent via NCCL to stage 1
/// 3. Stage 1 processes the hidden states and produces logits
/// 4. Tokens are sampled and the microbatch is completed
#[derive(Debug)]
pub struct Microbatch {
    /// Unique microbatch identifier.
    pub id: usize,
    /// Requests in this microbatch.
    pub requests: Vec<Request>,
    /// Current pipeline stage index (0 or 1 for PP=2).
    pub stage: usize,
    /// Hidden states after stage 0 processing (GPU buffer).
    /// Set by `PipelineEngine::forward_stage0`.
    pub hidden_states: Option<Vec<u8>>,  // Placeholder — real GPU CudaSlice<bf16>
}

impl Microbatch {
    /// Create a new microbatch.
    pub fn new(id: usize, requests: Vec<Request>) -> Self {
        Self {
            id,
            requests,
            stage: 0,
            hidden_states: None,
        }
    }

    /// Number of tokens in this microbatch (sum of all request lengths).
    pub fn num_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.tokens.len()).sum()
    }

    /// Number of requests in this microbatch.
    pub fn batch_size(&self) -> usize {
        self.requests.len()
    }
}

/// Scheduler that splits requests into microbatches for pipeline parallelism.
///
/// Maintains a queue of pending requests and a list of in-flight microbatches.
/// Each microbatch advances through pipeline stages until complete.
///
/// # Example
///
/// ```ignore
/// let mut scheduler = MicrobatchScheduler::new(4);
/// scheduler.add_request(Request::new(0, vec![1, 2, 3], 0));
/// scheduler.add_request(Request::new(1, vec![4, 5], 1));
///
/// while let Some(mut mb) = scheduler.next_microbatch() {
///     // Process through stage 0, send to stage 1, etc.
///     scheduler.advance_pipeline();
/// }
/// ```
pub struct MicrobatchScheduler {
    /// Maximum requests per microbatch.
    pub microbatch_size: usize,
    /// Requests waiting to be formed into microbatches.
    pub pending_requests: Vec<Request>,
    /// Microbatches currently being processed through the pipeline.
    pub in_flight: Vec<Microbatch>,
    /// Counter for microbatch IDs.
    next_id: usize,
}

impl MicrobatchScheduler {
    /// Create a new scheduler with the given microbatch size.
    pub fn new(microbatch_size: usize) -> Self {
        assert!(microbatch_size > 0, "Microbatch size must be positive");
        Self {
            microbatch_size,
            pending_requests: Vec::new(),
            in_flight: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a request to the pending queue.
    pub fn add_request(&mut self, request: Request) {
        self.pending_requests.push(request);
    }

    /// Add multiple requests to the pending queue.
    pub fn add_requests(&mut self, requests: Vec<Request>) {
        self.pending_requests.extend(requests);
    }

    /// Form the next microbatch from pending requests.
    ///
    /// Returns `None` if no pending requests remain.
    pub fn next_microbatch(&mut self) -> Option<Microbatch> {
        if self.pending_requests.is_empty() {
            return None;
        }

        let take = self.microbatch_size.min(self.pending_requests.len());
        let requests: Vec<_> = self.pending_requests.drain(..take).collect();
        let id = self.next_id;
        self.next_id += 1;

        Some(Microbatch::new(id, requests))
    }

    /// Whether there are pending requests or in-flight microbatches.
    pub fn is_busy(&self) -> bool {
        !self.pending_requests.is_empty() || !self.in_flight.is_empty()
    }

    /// Whether all work is done (no pending, no in-flight).
    pub fn is_done(&self) -> bool {
        self.pending_requests.is_empty() && self.in_flight.is_empty()
    }

    /// Number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }

    /// Number of in-flight microbatches.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Advance all in-flight microbatches to the next pipeline stage.
    ///
    /// Microbatches that reach stage 2 (past the last stage) are removed
    /// as completed.
    pub fn advance_pipeline(&mut self, num_stages: usize) {
        for microbatch in &mut self.in_flight {
            microbatch.stage += 1;
        }

        // Remove completed microbatches (past the last stage)
        self.in_flight.retain(|mb| mb.stage < num_stages);
    }

    /// Clear all state (pending requests and in-flight microbatches).
    pub fn reset(&mut self) {
        self.pending_requests.clear();
        self.in_flight.clear();
        self.next_id = 0;
    }
}

/// Statistics for pipeline scheduling.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total number of microbatches processed.
    pub total_microbatches: usize,
    /// Total number of requests processed.
    pub total_requests: usize,
    /// Peak number of in-flight microbatches.
    pub peak_in_flight: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_request() {
        let req = Request::new(0, vec![1, 2, 3], 0);
        assert_eq!(req.id, 0);
        assert_eq!(req.tokens, vec![1, 2, 3]);
        assert_eq!(req.session_id, 0);
    }

    #[test]
    fn test_microbatch_creation() {
        let reqs = vec![
            Request::new(0, vec![1, 2], 0),
            Request::new(1, vec![3, 4, 5], 1),
        ];
        let mb = Microbatch::new(0, reqs);
        assert_eq!(mb.id, 0);
        assert_eq!(mb.stage, 0);
        assert!(mb.hidden_states.is_none());
        assert_eq!(mb.batch_size(), 2);
        assert_eq!(mb.num_tokens(), 5);
    }

    #[test]
    fn test_scheduler_empty_initially() {
        let mut sched = MicrobatchScheduler::new(4);
        assert!(sched.is_done());
        assert!(!sched.is_busy());
        assert_eq!(sched.pending_count(), 0);
        assert_eq!(sched.in_flight_count(), 0);
        assert!(sched.next_microbatch().is_none());
    }

    #[test]
    fn test_scheduler_one_microbatch_exact_fit() {
        let mut sched = MicrobatchScheduler::new(2);
        sched.add_request(Request::new(0, vec![1], 0));
        sched.add_request(Request::new(1, vec![2], 0));

        let mb = sched.next_microbatch().unwrap();
        assert_eq!(mb.batch_size(), 2);
        assert_eq!(mb.id, 0);
        assert!(sched.next_microbatch().is_none());
    }

    #[test]
    fn test_scheduler_splits_into_multiple_microbatches() {
        let mut sched = MicrobatchScheduler::new(3);
        for i in 0..10 {
            sched.add_request(Request::new(i, vec![i as u32], 0));
        }

        // 10 requests with microbatch_size=3 → 4 microbatches (3+3+3+1)
        let mb1 = sched.next_microbatch().unwrap();
        assert_eq!(mb1.batch_size(), 3);
        assert_eq!(mb1.id, 0);

        let mb2 = sched.next_microbatch().unwrap();
        assert_eq!(mb2.batch_size(), 3);
        assert_eq!(mb2.id, 1);

        let mb3 = sched.next_microbatch().unwrap();
        assert_eq!(mb3.batch_size(), 3);
        assert_eq!(mb3.id, 2);

        let mb4 = sched.next_microbatch().unwrap();
        assert_eq!(mb4.batch_size(), 1);
        assert_eq!(mb4.id, 3);

        assert!(sched.next_microbatch().is_none());
    }

    #[test]
    fn test_pipeline_advance() {
        let mut sched = MicrobatchScheduler::new(4);
        sched.add_request(Request::new(0, vec![1], 0));

        let mb = sched.next_microbatch().unwrap();
        assert_eq!(mb.stage, 0);
        sched.in_flight.push(mb);

        // Advance: stage 0 → 1
        sched.advance_pipeline(2);
        assert_eq!(sched.in_flight[0].stage, 1);

        // Advance: stage 1 → 2 (complete, gets removed)
        sched.advance_pipeline(2);
        assert!(sched.in_flight.is_empty());
    }

    #[test]
    fn test_add_requests() {
        let mut sched = MicrobatchScheduler::new(2);
        let requests = vec![
            Request::new(0, vec![1], 0),
            Request::new(1, vec![2], 0),
            Request::new(2, vec![3], 0),
        ];
        sched.add_requests(requests);
        assert_eq!(sched.pending_count(), 3);
    }

    #[test]
    fn test_reset() {
        let mut sched = MicrobatchScheduler::new(2);
        sched.add_request(Request::new(0, vec![1], 0));

        let mb = sched.next_microbatch().unwrap();
        sched.in_flight.push(mb);
        assert!(sched.is_busy());

        sched.reset();
        assert!(sched.is_done());
        assert_eq!(sched.next_id, 0);
    }

    #[test]
    #[should_panic(expected = "Microbatch size must be positive")]
    fn test_zero_microbatch_size() {
        let _sched = MicrobatchScheduler::new(0);
    }

    #[test]
    fn test_is_busy_with_pending() {
        let mut sched = MicrobatchScheduler::new(2);
        assert!(!sched.is_busy());
        sched.add_request(Request::new(0, vec![1], 0));
        assert!(sched.is_busy());
    }

    #[test]
    fn test_is_busy_with_in_flight() {
        let mut sched = MicrobatchScheduler::new(2);
        sched.add_request(Request::new(0, vec![1], 0));
        let mb = sched.next_microbatch().unwrap();
        sched.in_flight.push(mb);
        assert!(sched.is_busy());
    }

    #[test]
    fn test_microbatch_ids_are_unique() {
        let mut sched = MicrobatchScheduler::new(1);
        sched.add_request(Request::new(0, vec![1], 0));
        sched.add_request(Request::new(1, vec![2], 0));
        sched.add_request(Request::new(2, vec![3], 0));

        let ids: Vec<_> = std::iter::from_fn(|| sched.next_microbatch())
            .map(|mb| mb.id)
            .collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_advance_multiple_microbatches() {
        let mut sched = MicrobatchScheduler::new(2);
        for i in 0..4 {
            sched.add_request(Request::new(i, vec![i as u32], 0));
        }

        // Form 2 microbatches
        let mb0 = sched.next_microbatch().unwrap();
        let mb1 = sched.next_microbatch().unwrap();
        sched.in_flight.push(mb0);
        sched.in_flight.push(mb1);

        assert_eq!(sched.in_flight_count(), 2);

        // Both advance to stage 1
        sched.advance_pipeline(2);
        assert_eq!(sched.in_flight_count(), 2);
        assert_eq!(sched.in_flight[0].stage, 1);
        assert_eq!(sched.in_flight[1].stage, 1);

        // Both complete (advance past stage 1 → stage 2, removed)
        sched.advance_pipeline(2);
        assert_eq!(sched.in_flight_count(), 0);
    }
}
