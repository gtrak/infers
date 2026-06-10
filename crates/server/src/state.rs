use std::sync::Arc;
use tokio::sync::Mutex;

use crate::orchestrator::InferenceOrchestrator;
use infers_tokenizer::Tokenizer;

/// Shared application state passed to all handlers.
pub struct AppState {
    /// Model name served by this instance.
    pub model_name: String,
    /// Inference orchestrator coordinating scheduler, engine, and response channels.
    pub orchestrator: Arc<Mutex<InferenceOrchestrator>>,
    /// Tokenizer for encoding prompts and decoding token IDs.
    pub tokenizer: Tokenizer,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("model_name", &self.model_name)
            .finish_non_exhaustive()
    }
}

impl Clone for AppState {
    fn clone(&self) -> Self {
        Self {
            model_name: self.model_name.clone(),
            orchestrator: self.orchestrator.clone(),
            tokenizer: self.tokenizer.clone(),
        }
    }
}

/// Convenience type alias for shared state.
pub type SharedState = Arc<AppState>;
