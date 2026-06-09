use axum::{routing::{get, post}, Router};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::handlers::{chat, health, models};
use crate::state::SharedState;

/// Build the axum router with all routes and middleware.
pub fn build_router(state: SharedState) -> Router {
    Router::new()
        .route("/health", get(health::health_check))
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/metrics", get(infers_metrics::metrics_handler))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}
