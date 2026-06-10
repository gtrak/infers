//! Batch construction for decode and prefill scheduling.
//!
//! Builds batches of sessions for the inference engine to execute.

use anyhow::{anyhow, Result};
use infers_kv::{PageId, PagedKvManager, SequenceId};

use crate::session::{Session, SessionState};

/// A batch of sessions ready for decode execution on GPU.
///
/// Contains the input tokens (one per session — the latest generated token),
/// the block tables for paged KV cache access, and the session identifiers
/// for output routing.
#[derive(Debug, Clone)]
pub struct DecodeBatch {
    /// Session IDs in this batch, in order.
    pub sessions: Vec<SequenceId>,
    /// Input token IDs — one per session (the latest token to continue from).
    pub input_tokens: Vec<u32>,
    /// GPU block table slices for each session's paged KV cache.
    /// Each entry is the block table for the corresponding session.
    pub block_tables: Vec<Vec<PageId>>,
}

/// Constructs decode and prefill batches from session state.
#[derive(Debug)]
pub struct BatchBuilder {
    /// Maximum number of sessions in a single batch.
    pub max_batch_size: usize,
    /// Maximum number of tokens across all sessions in a single batch.
    pub max_tokens_per_batch: usize,
}

impl BatchBuilder {
    /// Create a new batch builder with the given limits.
    pub fn new(max_batch_size: usize, max_tokens_per_batch: usize) -> Self {
        Self {
            max_batch_size,
            max_tokens_per_batch,
        }
    }

    /// Build a decode batch from active sessions.
    ///
    /// Iterates over `active_sessions`, collecting sessions in `Decoding` or
    /// `Prefilling` state up to `max_batch_size`. For each session, fetches
    /// the block table from `kv_manager` and the latest token.
    pub fn build_decode_batch(
        &self,
        active_sessions: &[Session],
        kv_manager: &PagedKvManager,
    ) -> Result<DecodeBatch> {
        let mut batch = DecodeBatch {
            sessions: Vec::new(),
            input_tokens: Vec::new(),
            block_tables: Vec::new(),
        };

        for session in active_sessions {
            if !session.is_active() {
                continue;
            }

            if batch.sessions.len() >= self.max_batch_size {
                break;
            }

            let block_table = kv_manager
                .block_table(session.id)
                .map_err(|e| anyhow!("Failed to get block table for session {}: {:?}", session.id, e))?
                .to_vec();

            let latest_token = session
                .tokens
                .last()
                .copied()
                .ok_or_else(|| anyhow!("Session {} has no tokens", session.id))?;

            batch.sessions.push(session.id);
            batch.input_tokens.push(latest_token);
            batch.block_tables.push(block_table);
        }

        Ok(batch)
    }

    /// Build a prefill batch from pending sessions.
    ///
    /// Takes the first session in `pending_sessions` that is in `Created` state,
    /// transitions it to `Prefilling`, and builds a batch with all its prompt tokens.
    /// Only one session is prefilled at a time (prefill is memory-intensive).
    pub fn build_prefill_batch(
        &self,
        pending_sessions: &mut [Session],
        kv_manager: &PagedKvManager,
    ) -> Result<DecodeBatch> {
        let session = pending_sessions
            .iter_mut()
            .find(|s| s.state == SessionState::Created)
            .ok_or_else(|| anyhow!("No pending sessions to prefill"))?;

        session.state = SessionState::Prefilling;

        let prompt_tokens = session.tokens.clone();
        let block_table = kv_manager
            .block_table(session.id)
            .map(|bt| bt.to_vec())
            .unwrap_or_default();

        Ok(DecodeBatch {
            sessions: vec![session.id],
            input_tokens: prompt_tokens,
            block_tables: vec![block_table],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use infers_kv::{SequencePageTable};

    fn make_session(
        id: SequenceId,
        state: SessionState,
        tokens: Vec<u32>,
        num_prompt: usize,
        num_gen: usize,
        max_tokens: usize,
    ) -> Session {
        Session {
            id,
            state,
            tokens,
            num_prompt_tokens: num_prompt,
            num_generated_tokens: num_gen,
            max_tokens,
            page_table: SequencePageTable::new(16),
            created_at: Instant::now(),
            last_activity: Instant::now(),
            priority: 0,
        }
    }

    #[test]
    fn test_decode_batch_empty_for_inactive_sessions() {
        let kv = PagedKvManager::new(100, 16, 4, 256, 1024, 65536);
        let builder = BatchBuilder::new(4, 128);
        let sessions = vec![make_session(0, SessionState::Created, vec![1], 1, 0, 100)];
        let batch = builder.build_decode_batch(&sessions, &kv).unwrap();
        assert!(batch.sessions.is_empty());
        assert!(batch.input_tokens.is_empty());
        assert!(batch.block_tables.is_empty());
    }

    #[test]
    fn test_decode_batch_collects_active_sessions() {
        let mut kv = PagedKvManager::new(100, 16, 4, 256, 1024, 65536);
        let seq1 = kv.create_sequence();
        let seq2 = kv.create_sequence();
        kv.append_page(seq1).unwrap();
        kv.append_page(seq2).unwrap();

        let builder = BatchBuilder::new(2, 128);
        let sessions = vec![
            make_session(seq1, SessionState::Decoding, vec![1, 2, 3], 2, 1, 100),
            make_session(seq2, SessionState::Decoding, vec![4, 5], 1, 1, 100),
        ];

        let batch = builder.build_decode_batch(&sessions, &kv).unwrap();
        assert_eq!(batch.sessions.len(), 2);
        assert_eq!(batch.input_tokens, vec![3, 5]);
        assert_eq!(batch.block_tables.len(), 2);
    }

    #[test]
    fn test_decode_batch_respects_max_batch_size() {
        let mut kv = PagedKvManager::new(100, 16, 4, 256, 1024, 65536);
        let ids: Vec<_> = (0..5).map(|_| kv.create_sequence()).collect();
        for &id in &ids {
            kv.append_page(id).unwrap();
        }

        let builder = BatchBuilder::new(3, 128);
        let sessions: Vec<_> = ids
            .into_iter()
            .map(|id| make_session(id, SessionState::Decoding, vec![id as u32], 1, 1, 100))
            .collect();

        let batch = builder.build_decode_batch(&sessions, &kv).unwrap();
        assert_eq!(batch.sessions.len(), 3);
        assert_eq!(batch.input_tokens, vec![0, 1, 2]);
    }

    #[test]
    fn test_build_prefill_batch() {
        let mut kv = PagedKvManager::new(100, 16, 4, 256, 1024, 65536);
        let seq_id = kv.create_sequence();
        for _ in 0..2 {
            kv.append_page(seq_id).unwrap();
        }

        let builder = BatchBuilder::new(4, 128);
        let mut sessions = vec![make_session(seq_id, SessionState::Created, vec![10, 20, 30], 3, 0, 100)];

        let batch = builder.build_prefill_batch(&mut sessions, &kv).unwrap();
        assert_eq!(batch.sessions.len(), 1);
        assert_eq!(batch.sessions[0], seq_id);
        assert_eq!(batch.input_tokens, vec![10, 20, 30]);
        assert_eq!(sessions[0].state, SessionState::Prefilling);
    }

    #[test]
    fn test_build_prefill_batch_no_pending() {
        let kv = PagedKvManager::new(100, 16, 4, 256, 1024, 65536);
        let builder = BatchBuilder::new(4, 128);
        let mut sessions = vec![];
        let result = builder.build_prefill_batch(&mut sessions, &kv);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_prefill_batch_skips_non_created() {
        let mut kv = PagedKvManager::new(100, 16, 4, 256, 1024, 65536);
        let seq_id = kv.create_sequence();
        let builder = BatchBuilder::new(4, 128);
        let mut sessions = vec![make_session(seq_id, SessionState::Decoding, vec![1], 1, 1, 100)];
        let result = builder.build_prefill_batch(&mut sessions, &kv);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_builder_new() {
        let builder = BatchBuilder::new(8, 512);
        assert_eq!(builder.max_batch_size, 8);
        assert_eq!(builder.max_tokens_per_batch, 512);
    }
}
