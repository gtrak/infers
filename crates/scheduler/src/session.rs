//! Session lifecycle states and session data structures.

use std::time::{Duration, Instant};

use infers_kv::{SequenceId, SequencePageTable};

use crate::queue::SamplingConfig;

// @lat: [[lat.md/lat#Scheduler#Session State]]
/// The lifecycle state of an inference session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Just allocated, not yet running.
    Created,
    /// Processing prompt tokens.
    Prefilling,
    /// Generating response tokens.
    Decoding,
    /// Temporarily stopped (e.g., waiting for client).
    Paused,
    /// KV cache moved to CPU/SSD (reserved for future use).
    Evicted,
    /// Finished generating.
    Completed,
}

// @lat: [[lat.md/lat#Scheduler#Session]]
/// A single inference session tracking generation state, tokens, and
/// paged KV cache page table.
#[derive(Debug, Clone)]
pub struct Session {
    /// Sequence ID used to look up the KV cache pages in PagedKvManager.
    pub id: SequenceId,
    /// Current lifecycle state.
    pub state: SessionState,
    /// All token IDs for this session (prompt + generated).
    pub tokens: Vec<u32>,
    /// Number of prompt tokens (excludes generated tokens).
    pub num_prompt_tokens: usize,
    /// Number of tokens generated so far.
    pub num_generated_tokens: usize,
    /// Maximum tokens to generate for this session.
    pub max_tokens: usize,
    /// Page table mapping this session's tokens to physical pages.
    /// Mirrors the state in PagedKvManager for local query convenience.
    pub page_table: SequencePageTable,
    /// Wall-clock time when the session was created.
    pub created_at: Instant,
    /// Wall-clock time of the most recent activity.
    pub last_activity: Instant,
    /// Scheduling priority (higher = more important).
    pub priority: i32,
    /// Routing ID from the original request, used to correlate with response channel.
    pub routing_id: Option<usize>,
    /// Sampling configuration for this session (from the original request).
    pub sampling_config: SamplingConfig,
}

impl Session {
    /// Returns `true` if the session is actively processing (prefilling or decoding).
    pub fn is_active(&self) -> bool {
        matches!(self.state, SessionState::Prefilling | SessionState::Decoding)
    }

    /// Returns `true` if the session can be evicted to CPU/SSD.
    ///
    /// GDN state (H×H matrix per layer) stays on GPU regardless.
    /// Only idle sessions with no recent activity are candidates.
    pub fn is_evictable(&self) -> bool {
        self.last_activity.elapsed() > Duration::from_secs(30)
    }

    /// Returns `true` if the session has reached its max_tokens limit.
    pub fn is_complete(&self) -> bool {
        self.num_generated_tokens >= self.max_tokens
    }

    /// Returns the total number of tokens (prompt + generated).
    pub fn total_tokens(&self) -> usize {
        self.num_prompt_tokens + self.num_generated_tokens
    }
}
