//! Session lifecycle management, batch construction, and scheduling
//! for continuous batching inference.

pub mod batch;
pub mod lifecycle;
pub mod queue;
pub mod scheduler;
pub mod session;

// Re-exports for convenience
pub use batch::{BatchBuilder, DecodeBatch};
pub use lifecycle::TransitionError;
pub use queue::{Request, RequestQueue, SamplingConfig, SamplingStrategy};
pub use scheduler::{RoundRobinScheduler, ScheduledWork};
pub use session::{Session, SessionState};
