//! Request types and priority queue for inference scheduling.

use std::collections::VecDeque;

use infers_kv::SequenceId;

// @lat: [[lat.md/lat#Scheduler#Sampling Strategy]]
/// Sampling strategy selection for token generation.
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Pure greedy: always pick the token with highest logit.
    Greedy,
    /// Temperature-scaled softmax sampling.
    Temperature { temp: f32 },
    /// Top-k sampling with temperature scaling.
    TopK { k: usize, temp: f32 },
    /// Top-p (nucleus) sampling with temperature scaling.
    TopP { p: f64, temp: f32 },
}

// @lat: [[lat.md/lat#Scheduler#SamplingConfig]]
/// Sampling configuration for token generation.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Strategy for selecting the next token.
    pub strategy: SamplingStrategy,
    /// Maximum number of tokens to generate per request.
    pub max_tokens: usize,
    /// Sequences that, if generated, stop further generation.
    pub stop_sequences: Vec<String>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::Greedy,
            max_tokens: 512,
            stop_sequences: Vec::new(),
        }
    }
}

// @lat: [[lat.md/lat#Scheduler#Request]]
/// A tokenized inference request waiting to be scheduled.
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request identifier.
    pub id: usize,
    /// Input token IDs (already tokenized).
    pub tokens: Vec<u32>,
    /// Session ID to use for KV cache lookup (0 if not yet assigned).
    pub session_id: SequenceId,
    /// Sampling configuration.
    pub config: SamplingConfig,
    /// Scheduling priority (higher = more important).
    pub priority: i32,
}

impl Request {
    /// Create a new request.
    pub fn new(id: usize, tokens: Vec<u32>, config: SamplingConfig) -> Self {
        Self {
            id,
            tokens,
            session_id: 0,
            config,
            priority: 0,
        }
    }
}

// @lat: [[lat.md/lat#Scheduler#RequestQueue]]
/// Priority-ordered request queue.
///
/// Requests with higher priority values are dequeued first.
/// Within the same priority level, FIFO order is preserved.
#[derive(Debug, Clone)]
pub struct RequestQueue {
    /// Internal deque of requests ordered by priority (highest first).
    queue: VecDeque<Request>,
}

impl RequestQueue {
    /// Create an empty request queue.
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    /// Add a request to the queue in priority order.
    ///
    /// Higher-priority requests are inserted ahead of lower-priority ones.
    /// Requests with equal priority maintain FIFO order (appended after existing same-priority requests).
    pub fn enqueue(&mut self, request: Request) {
        // Find the insertion point: first request with strictly lower priority
        let insert_idx = self
            .queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(self.queue.len());
        self.queue.insert(insert_idx, request);
    }

    /// Remove and return the highest-priority request.
    pub fn dequeue(&mut self) -> Option<Request> {
        self.queue.pop_front()
    }

    /// Return a reference to the highest-priority request without removing it.
    pub fn peek(&self) -> Option<&Request> {
        self.queue.front()
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the number of requests in the queue.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Remove all requests from the queue.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Drain all requests from the queue, highest priority first.
    pub fn drain(&mut self) -> Vec<Request> {
        self.queue.drain(..).collect()
    }
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_creation() {
        let config = SamplingConfig::default();
        let req = Request::new(0, vec![1, 2, 3], config);
        assert_eq!(req.id, 0);
        assert_eq!(req.tokens, vec![1, 2, 3]);
        assert_eq!(req.session_id, 0);
    }

    #[test]
    fn test_queue_empty_initially() {
        let mut queue = RequestQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert!(queue.dequeue().is_none());
        assert!(queue.peek().is_none());
    }

    #[test]
    fn test_queue_fifo_within_priority() {
        let mut queue = RequestQueue::new();
        let config = SamplingConfig::default();
        queue.enqueue(Request { id: 0, tokens: vec![], session_id: 0, config: config.clone(), priority: 0 });
        queue.enqueue(Request { id: 1, tokens: vec![], session_id: 0, config: config.clone(), priority: 0 });
        queue.enqueue(Request { id: 2, tokens: vec![], session_id: 0, config, priority: 0 });

        assert_eq!(queue.dequeue().unwrap().id, 0);
        assert_eq!(queue.dequeue().unwrap().id, 1);
        assert_eq!(queue.dequeue().unwrap().id, 2);
    }

    #[test]
    fn test_queue_priority_ordering() {
        let mut queue = RequestQueue::new();
        let config = SamplingConfig::default();
        queue.enqueue(Request { id: 0, tokens: vec![], session_id: 0, config: config.clone(), priority: 1 });
        queue.enqueue(Request { id: 1, tokens: vec![], session_id: 0, config: config.clone(), priority: 3 });
        queue.enqueue(Request { id: 2, tokens: vec![], session_id: 0, config, priority: 2 });

        // Should be ordered by priority: 3, 2, 1
        assert_eq!(queue.dequeue().unwrap().id, 1); // priority 3
        assert_eq!(queue.dequeue().unwrap().id, 2); // priority 2
        assert_eq!(queue.dequeue().unwrap().id, 0); // priority 1
    }

    #[test]
    fn test_queue_peek() {
        let mut queue = RequestQueue::new();
        let config = SamplingConfig::default();
        queue.enqueue(Request::new(42, vec![1, 2], config));
        assert_eq!(queue.peek().unwrap().id, 42);
        assert_eq!(queue.len(), 1); // peek doesn't remove
    }

    #[test]
    fn test_queue_clear() {
        let mut queue = RequestQueue::new();
        let config = SamplingConfig::default();
        queue.enqueue(Request::new(0, vec![], config));
        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_queue_drain() {
        let mut queue = RequestQueue::new();
        let config = SamplingConfig::default();
        queue.enqueue(Request::new(0, vec![], SamplingConfig { strategy: SamplingStrategy::Greedy, ..config.clone() }));
        queue.enqueue(Request::new(1, vec![], config));
        let drained = queue.drain();
        assert_eq!(drained.len(), 2);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert!(matches!(config.strategy, SamplingStrategy::Greedy));
        assert_eq!(config.max_tokens, 512);
        assert!(config.stop_sequences.is_empty());
    }

    #[test]
    fn test_sampling_strategy_variants() {
        let greedy = SamplingStrategy::Greedy;
        let temp = SamplingStrategy::Temperature { temp: 0.8 };
        let topk = SamplingStrategy::TopK { k: 50, temp: 0.7 };
        let topp = SamplingStrategy::TopP { p: 0.9, temp: 1.0 };

        assert!(matches!(greedy, SamplingStrategy::Greedy));
        assert!(matches!(temp, SamplingStrategy::Temperature { .. }));
        assert!(matches!(topk, SamplingStrategy::TopK { .. }));
        assert!(matches!(topp, SamplingStrategy::TopP { .. }));
    }
}
