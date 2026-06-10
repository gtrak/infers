//! Session lifecycle state transitions.
//!
//! Valid transitions:
//!   Created → Prefilling  (start processing prompt)
//!   Prefilling → Decoding  (prompt done, start generating)
//!   Decoding → Completed   (finished generating)
//!   Decoding → Paused      (temporarily stopped)
//!   Paused → Decoding      (resumed)
//!   Prefilling → Completed (empty prompt or error)
//!   Decoding → Evicted     (moved to CPU/SSD, reserved)
//!   Evicted → Prefilling   (restored from CPU/SSD, reserved)

use std::time::Instant;

use crate::session::{Session, SessionState};

/// Error returned when an invalid state transition is attempted.
#[derive(Debug, Clone)]
pub struct TransitionError {
    /// The current state of the session.
    pub from: SessionState,
    /// The attempted target state.
    pub to: SessionState,
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid state transition: {:?} → {:?}", self.from, self.to)
    }
}

impl std::error::Error for TransitionError {}

/// Attempt to transition a session to a new state.
///
/// Returns `Ok(())` if the transition is valid, `Err(TransitionError)` otherwise.
/// On success, updates `session.state` and `session.last_activity`.
pub fn transition(session: &mut Session, to: SessionState) -> Result<(), TransitionError> {
    let from = session.state;
    let valid = matches!(
        (from, to),
        (SessionState::Created, SessionState::Prefilling)
            | (SessionState::Prefilling, SessionState::Decoding)
            | (SessionState::Decoding, SessionState::Completed)
            | (SessionState::Decoding, SessionState::Paused)
            | (SessionState::Paused, SessionState::Decoding)
            | (SessionState::Prefilling, SessionState::Completed)
            | (SessionState::Decoding, SessionState::Evicted)
            | (SessionState::Evicted, SessionState::Prefilling)
    );

    if !valid {
        return Err(TransitionError { from, to });
    }

    session.state = to;
    session.last_activity = Instant::now();
    Ok(())
}

/// Transition a session to Completed, recording the final token count.
pub fn complete_session(session: &mut Session) -> Result<(), TransitionError> {
    transition(session, SessionState::Completed)
}

/// Transition a session to Paused (temporarily stop processing).
pub fn pause_session(session: &mut Session) -> Result<(), TransitionError> {
    transition(session, SessionState::Paused)
}

/// Transition a paused session back to Decoding (resume generation).
pub fn resume_session(session: &mut Session) -> Result<(), TransitionError> {
    transition(session, SessionState::Decoding)
}

/// Transition a session from Created to Prefilling (begin prompt processing).
pub fn start_prefill(session: &mut Session) -> Result<(), TransitionError> {
    transition(session, SessionState::Prefilling)
}

/// Transition a session from Prefilling to Decoding (prompt processed, begin generation).
pub fn finish_prefill(session: &mut Session) -> Result<(), TransitionError> {
    transition(session, SessionState::Decoding)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(state: SessionState) -> Session {
        Session {
            id: 0,
            state,
            tokens: Vec::new(),
            num_prompt_tokens: 0,
            num_generated_tokens: 0,
            max_tokens: 100,
            page_table: infers_kv::SequencePageTable::new(16),
            created_at: Instant::now(),
            last_activity: Instant::now(),
            priority: 0,
            routing_id: None,
        }
    }

    #[test]
    fn test_valid_created_to_prefilling() {
        let mut s = make_session(SessionState::Created);
        assert!(start_prefill(&mut s).is_ok());
        assert_eq!(s.state, SessionState::Prefilling);
    }

    #[test]
    fn test_valid_prefilling_to_decoding() {
        let mut s = make_session(SessionState::Prefilling);
        assert!(finish_prefill(&mut s).is_ok());
        assert_eq!(s.state, SessionState::Decoding);
    }

    #[test]
    fn test_valid_decoding_to_completed() {
        let mut s = make_session(SessionState::Decoding);
        assert!(complete_session(&mut s).is_ok());
        assert_eq!(s.state, SessionState::Completed);
    }

    #[test]
    fn test_valid_decoding_to_paused_and_back() {
        let mut s = make_session(SessionState::Decoding);
        assert!(pause_session(&mut s).is_ok());
        assert_eq!(s.state, SessionState::Paused);
        assert!(resume_session(&mut s).is_ok());
        assert_eq!(s.state, SessionState::Decoding);
    }

    #[test]
    fn test_invalid_transition() {
        let mut s = make_session(SessionState::Created);
        let result = transition(&mut s, SessionState::Completed);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().from, SessionState::Created);
        assert_eq!(s.state, SessionState::Created); // unchanged
    }

    #[test]
    fn test_invalid_direct_to_decoding() {
        let mut s = make_session(SessionState::Created);
        let result = transition(&mut s, SessionState::Decoding);
        assert!(result.is_err());
        assert_eq!(s.state, SessionState::Created);
    }

    #[test]
    fn test_complete_from_prefilling() {
        let mut s = make_session(SessionState::Prefilling);
        assert!(transition(&mut s, SessionState::Completed).is_ok());
        assert_eq!(s.state, SessionState::Completed);
    }

    #[test]
    fn test_evict_and_restore() {
        let mut s = make_session(SessionState::Decoding);
        assert!(transition(&mut s, SessionState::Evicted).is_ok());
        assert_eq!(s.state, SessionState::Evicted);
        assert!(transition(&mut s, SessionState::Prefilling).is_ok());
        assert_eq!(s.state, SessionState::Prefilling);
    }
}
