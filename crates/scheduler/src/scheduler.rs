//! Round-robin scheduler for continuous batching.
//!
//! Admits requests from the queue, manages session lifecycle, and builds
//! interleaved decode/prefill batches for the inference engine.

use std::time::Instant;

use anyhow::{anyhow, Result};
use infers_kv::{PagedKvManager, SequencePageTable};

use crate::batch::{BatchBuilder, DecodeBatch};
use crate::lifecycle;
use crate::pressure::{self, PressureConfig};
use crate::queue::{Request, RequestQueue};
use crate::session::{Session, SessionState};

/// The output of a single scheduling iteration.
#[derive(Debug)]
pub struct ScheduledWork {
    /// Decode batch: active sessions needing single-token generation.
    pub decode_batch: DecodeBatch,
    /// Optional prefill batch: a new session being prefilled.
    pub prefill_batch: Option<DecodeBatch>,
    /// Session evicted due to memory pressure (if any). The backend
    /// should copy its GPU page data and call `kv_manager.evict_sequence()`.
    pub evicted_session: Option<usize>,
}

/// Round-robin scheduler for continuous batching.
///
/// Manages the lifecycle of inference sessions:
/// 1. Admits new requests from the queue (up to `max_concurrent_sessions`)
/// 2. Removes completed sessions and frees their KV resources
/// 3. Builds a decode batch from active sessions
/// 4. If the decode batch has room, prefills a new session
#[derive(Debug)]
pub struct RoundRobinScheduler {
    /// Incoming request queue (priority-ordered).
    pub request_queue: RequestQueue,
    /// Currently active sessions.
    pub active_sessions: Vec<Session>,
    /// Maximum number of concurrent sessions.
    pub max_concurrent_sessions: usize,
    /// Batch builder for constructing decode/prefill batches.
    pub batch_builder: BatchBuilder,
    /// Paged KV manager for block allocation and page table access.
    pub kv_manager: PagedKvManager,
    /// Memory pressure configuration for eviction policy.
    pub pressure_config: PressureConfig,
}

impl RoundRobinScheduler {
   /// Create a new round-robin scheduler.
    pub fn new(
        max_concurrent_sessions: usize,
        max_batch_size: usize,
        kv_manager: PagedKvManager,
    ) -> Self {
        Self {
            request_queue: RequestQueue::new(),
            active_sessions: Vec::new(),
            max_concurrent_sessions,
            batch_builder: BatchBuilder::new(max_batch_size),
            kv_manager,
            pressure_config: PressureConfig::default(),
        }
    }

    /// Enqueue a request for scheduling.
    pub fn enqueue_request(&mut self, request: Request) {
        self.request_queue.enqueue(request);
    }

    /// Run one scheduling iteration.
    ///
    /// 1. Admit new requests from the queue up to `max_concurrent_sessions`.
    /// 2. Remove completed sessions and free their KV resources.
    /// 3. Handle memory pressure — evict idle sessions if pool utilization is high.
    /// 4. Build a decode batch from active sessions.
    /// 5. If the decode batch is small, try to prefill a new session.
    pub fn schedule(&mut self) -> Result<ScheduledWork> {
        // Step 1: Admit new requests from the queue
        self.admit_new_requests();

        // Step 2: Remove completed sessions
        self.cleanup_completed_sessions();

        // Step 3: Handle memory pressure — evict idle sessions if pool utilization is high
        let evicted_session = self.handle_memory_pressure()?;

        // Step 3: Build decode batch
        let decode_batch = self
            .batch_builder
            .build_decode_batch(&self.active_sessions, &self.kv_manager)?;

        // Step 4: If decode batch is small, try to prefill a new session
        let prefill_batch = if decode_batch.sessions.len() < self.batch_builder.max_batch_size / 2 {
            self.batch_builder
                .build_prefill_batch(&mut self.active_sessions, &self.kv_manager)
                .ok()
        } else {
            None
        };

        Ok(ScheduledWork {
            decode_batch,
            prefill_batch,
            evicted_session,
        })
    }

    /// Handle memory pressure by evicting the oldest idle session if needed.
    ///
    /// Checks if the KV page pool utilization exceeds the configured threshold.
    /// If so, selects the LRU evictable session and transitions it to Evicted.
    /// Returns the evicted session ID if one was evicted.
    pub fn handle_memory_pressure(&mut self) -> Result<Option<usize>> {
        if !pressure::is_under_pressure(&self.kv_manager, &self.pressure_config) {
            return Ok(None);
        }

        let candidate_idx =
            pressure::select_lru_eviction_candidate(&self.active_sessions);

        match candidate_idx {
            Some(idx) => self.evict_session_at(idx),
            None => Ok(None),
        }
    }

    /// Evict the session at the given index.
    ///
    /// Transitions the session to `Evicted` state and removes it from
    /// the active sessions list. Returns the session ID.
    fn evict_session_at(&mut self, idx: usize) -> Result<Option<usize>> {
        let session_id = self.active_sessions[idx].id;
        lifecycle::transition(&mut self.active_sessions[idx], SessionState::Evicted)
            .map_err(|e| anyhow!("Failed to evict session {}: {:?}", session_id, e))?;
        self.active_sessions.remove(idx);
        Ok(Some(session_id))
    }

    /// Admit new requests from the queue up to capacity.
    fn admit_new_requests(&mut self) {
        while self.active_sessions.len() < self.max_concurrent_sessions {
            let request = match self.request_queue.dequeue() {
                Some(r) => r,
                None => break,
            };

            let request_id = request.id;
            match self.create_session(request) {
                Ok(session) => self.active_sessions.push(session),
                Err(e) => {
                    eprintln!("Failed to create session for request {request_id}: {e:?}");
                }
            }
        }
    }

    /// Remove completed sessions and free their KV cache resources.
    fn cleanup_completed_sessions(&mut self) {
        let mut i = 0;
        while i < self.active_sessions.len() {
            if self.active_sessions[i].state == SessionState::Completed {
                let session = self.active_sessions.swap_remove(i);
                // Free KV cache resources
                let _ = self.kv_manager.delete_sequence(session.id);
            } else {
                i += 1;
            }
        }
    }

    /// Create a new session from a request.
    ///
    /// Allocates KV pages for the prompt tokens and initializes the session.
    pub fn create_session(&mut self, request: Request) -> Result<Session> {
        let num_prompt_tokens = request.tokens.len();

        // Create KV sequence and allocate pages for prompt tokens
        let seq_id = self.kv_manager.create_sequence();
        let page_size = self.kv_manager.page_size();
        let num_pages = num_prompt_tokens.div_ceil(page_size);
        for _ in 0..num_pages {
            self.kv_manager
                .append_page(seq_id)
                .map_err(|e| anyhow!("Failed to allocate KV pages: {:?}", e))?;
        }

        // Add token count to the sequence for page tracking
        for _ in 0..num_prompt_tokens {
            self.kv_manager
                .add_token(seq_id)
                .map_err(|e| anyhow!("Failed to add tokens to sequence: {:?}", e))?;
        }

        Ok(Session {
            id: seq_id,
            state: SessionState::Created,
            tokens: request.tokens,
            num_prompt_tokens,
            num_generated_tokens: 0,
            max_tokens: request.config.max_tokens,
            page_table: SequencePageTable::new(page_size),
            created_at: Instant::now(),
            last_activity: Instant::now(),
            priority: request.priority,
            routing_id: request.routing_id,
            sampling_config: request.config.clone(),
        })
    }

    /// Returns the number of active sessions.
    pub fn active_count(&self) -> usize {
        self.active_sessions.len()
    }

    /// Returns the number of pending requests in the queue.
    pub fn pending_count(&self) -> usize {
        self.request_queue.len()
    }

    /// Returns `true` if there is any work (active sessions or pending requests).
    pub fn is_busy(&self) -> bool {
        !self.active_sessions.is_empty() || !self.request_queue.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::SamplingConfig;

    fn make_kv_manager() -> PagedKvManager {
        PagedKvManager::new(200, 16, 4, 256, 1024 * 1024, 1024 * 1024)
    }

    fn make_request(id: usize, tokens: Vec<u32>) -> Request {
        Request::new(id, tokens, SamplingConfig::default())
    }

    #[test]
    fn test_scheduler_empty_initially() {
        let kv = make_kv_manager();
        let sched = RoundRobinScheduler::new(4, 4, kv);
        assert_eq!(sched.active_count(), 0);
        assert_eq!(sched.pending_count(), 0);
        assert!(!sched.is_busy());
    }

    #[test]
    fn test_scheduler_enqueue_and_admit() {
        let kv = make_kv_manager();
        let mut sched = RoundRobinScheduler::new(4, 4, kv);
        sched.enqueue_request(make_request(0, vec![1, 2, 3]));
        sched.enqueue_request(make_request(1, vec![4, 5]));
        assert_eq!(sched.pending_count(), 2);

        let work = sched.schedule().unwrap();
        // Should have admitted both requests and created sessions
        assert_eq!(sched.active_count(), 2);
        // Decode batch should be empty (sessions are Created, not active)
        assert!(work.decode_batch.sessions.is_empty());
        // Prefill batch should exist (one created session)
        assert!(work.prefill_batch.is_some());
    }

    #[test]
    fn test_scheduler_respects_max_concurrent() {
        let kv = make_kv_manager();
       let mut sched = RoundRobinScheduler::new(2, 4, kv);
        for i in 0..5 {
            sched.enqueue_request(make_request(i, vec![i as u32]));
        }

        let _ = sched.schedule().unwrap();
        assert_eq!(sched.active_count(), 2);
        assert_eq!(sched.pending_count(), 3); // 3 remain in queue
    }

    #[test]
    fn test_create_session_allocates_pages() {
        let kv = make_kv_manager();
        let free_before = kv.num_free_pages();
        let mut sched = RoundRobinScheduler::new(4, 4, kv);
        sched.enqueue_request(make_request(0, vec![1, 2, 3]));
        let _ = sched.schedule().unwrap();
        // Session was admitted and pages allocated
        let free_after = sched.kv_manager.num_free_pages();
        assert!(free_after < free_before, "Free pages should decrease after session creation");
    }

    #[test]
    fn test_scheduler_cleanup_completed() {
        let kv = make_kv_manager();
        let mut sched = RoundRobinScheduler::new(4, 4, kv);
        sched.enqueue_request(make_request(0, vec![1, 2, 3]));
        sched.enqueue_request(make_request(1, vec![4, 5]));

        // First schedule: admit both, prefill first
        let _ = sched.schedule().unwrap();
        assert_eq!(sched.active_count(), 2);

        // Second schedule: prefill second session
        let _ = sched.schedule().unwrap();

        // Transition both to Decoding (simulating prefill completion)
        for session in &mut sched.active_sessions {
            let _ = crate::lifecycle::finish_prefill(session);
        }

        // Now complete both
        for session in &mut sched.active_sessions {
            let _ = crate::lifecycle::complete_session(session);
        }

        // Next schedule should clean them up
        let _ = sched.schedule().unwrap();
        assert_eq!(sched.active_count(), 0);
    }

    #[test]
    fn test_schedule_decode_and_prefill_interleaving() {
        let kv = make_kv_manager();
        let mut sched = RoundRobinScheduler::new(4, 4, kv);
        sched.enqueue_request(make_request(0, vec![1, 2, 3]));

        // First schedule: admit and prefill
        let work1 = sched.schedule().unwrap();
        assert_eq!(sched.active_count(), 1);
        assert!(work1.prefill_batch.is_some());

        // The session is now in Prefilling state (from build_prefill_batch)
        // Move it to Decoding to simulate prefill completion
        if let Some(session) = sched.active_sessions.first_mut() {
            let _ = crate::lifecycle::finish_prefill(session);
        }

        // Second schedule: should build decode batch
        let work2 = sched.schedule().unwrap();
        assert_eq!(work2.decode_batch.sessions.len(), 1);
    }

    #[test]
    fn test_is_busy_with_pending() {
        let kv = make_kv_manager();
        let mut sched = RoundRobinScheduler::new(4, 4, kv);
        assert!(!sched.is_busy());
        sched.enqueue_request(make_request(0, vec![1]));
        assert!(sched.is_busy());
    }

 }
