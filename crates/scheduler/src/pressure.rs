//! Memory pressure monitoring and eviction policy for the scheduler.
//!
//! Provides functions to check whether the KV page pool is under memory
//! pressure and to select the best session to evict (LRU among inactive
//! sessions). The actual GPU→CPU data copy is performed by the backend;
//! this module selects *which* session to evict.

use crate::session::Session;

/// Configuration for memory pressure monitoring.
#[derive(Debug, Clone)]
pub struct PressureConfig {
    /// Pool utilization threshold (0.0–1.0). When `pool_utilization()`
    /// exceeds this value, eviction candidates are sought.
    pub eviction_threshold: f64,
}

impl Default for PressureConfig {
    fn default() -> Self {
        Self { eviction_threshold: 0.90 }
    }
}

/// Check whether the page pool is under memory pressure.
pub fn is_under_pressure(
    kv_manager: &infers_kv::PagedKvManager,
    config: &PressureConfig,
) -> bool {
    kv_manager.pool_utilization() >= config.eviction_threshold
}

/// Select the best session to evict using LRU among evictable sessions.
///
/// Returns the index in `active_sessions` of the session with the oldest
/// `last_activity` that is evictable. Returns `None` if no session is
/// evictable.
pub fn select_lru_eviction_candidate(active_sessions: &[Session]) -> Option<usize> {
    active_sessions
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_evictable())
        .min_by_key(|(_, s)| s.last_activity)
        .map(|(idx, _)| idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionState;
    use std::time::{Duration, Instant};

    fn make_session(
        id: usize,
        state: SessionState,
        last_activity: Instant,
    ) -> Session {
        Session {
            id,
            state,
            tokens: Vec::new(),
            num_prompt_tokens: 0,
            num_generated_tokens: 10,
            max_tokens: 100,
            page_table: infers_kv::SequencePageTable::new(16),
            created_at: Instant::now(),
            last_activity,
            priority: 0,
            routing_id: None,
            sampling_config: crate::queue::SamplingConfig::default(),
        }
    }

    #[test]
    fn test_no_pressure_when_pool_empty() {
        let kv = infers_kv::PagedKvManager::new(100, 16, 4, 256, 1024, 65536);
        let config = PressureConfig::default();
        assert!(!is_under_pressure(&kv, &config));
    }

    #[test]
    fn test_select_lru_among_evictable() {
        let now = Instant::now();
        let sessions = vec![
            make_session(0, SessionState::Decoding, now),
            make_session(1, SessionState::Decoding, now - Duration::from_secs(60)),
            make_session(2, SessionState::Decoding, now - Duration::from_secs(120)),
        ];

        let candidate = select_lru_eviction_candidate(&sessions);
        assert_eq!(candidate, Some(2)); // oldest
    }

    #[test]
    fn test_select_lru_skips_recent() {
        let now = Instant::now();
        let sessions = vec![
            make_session(0, SessionState::Decoding, now), // not evictable (< 30s)
        ];

        let candidate = select_lru_eviction_candidate(&sessions);
        assert_eq!(candidate, None);
    }

    #[test]
    fn test_select_lru_empty_sessions() {
        let sessions: Vec<Session> = vec![];
        let candidate = select_lru_eviction_candidate(&sessions);
        assert_eq!(candidate, None);
    }

    #[test]
    fn test_pressure_config_default() {
        let config = PressureConfig::default();
        assert_eq!(config.eviction_threshold, 0.90);
    }
}
