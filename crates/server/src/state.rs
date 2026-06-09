use std::sync::Arc;

/// Shared application state passed to all handlers.
#[derive(Debug, Clone)]
pub struct AppState {
    /// Model name served by this instance.
    pub model_name: String,
}

/// Convenience type alias for shared state.
pub type SharedState = Arc<AppState>;
