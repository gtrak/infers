//! Multi-token prediction for speculative decoding.
//!
//! Implements native MTP (Multi-Token Prediction) speculative decoding
//! for Qwen3.6-27B. The MTP head is a single transformer decoder layer
//! that predicts future tokens from the main model's hidden state.
//!
//! Architecture:
//! 1. `head::MtpHead` — GPU-resident MTP prediction head with forward pass
//! 2. `engine::MtpEngine` — Orchestrates draft generation, verification, acceptance
//! 3. `verify::VerificationResult` — Outcome of draft token verification
//! 4. `metrics::MtpMetrics` — Performance tracking (acceptance rate, speedup)

pub mod engine;
pub mod head;
pub mod metrics;
pub mod verify;

pub use engine::{MtpEngine, MtpOperations};
pub use head::MtpHead;
pub use metrics::MtpMetrics;
pub use verify::VerificationResult;
