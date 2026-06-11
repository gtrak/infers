use std::sync::Arc;
use std::time::Duration;

use axum::{routing::{get, post}, Router};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::handlers::{chat, health, models};
use crate::orchestrator::InferenceOrchestrator;
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

/// Spawn the background scheduler loop that continuously calls
/// `orchestrator.step()` to drive inference forward.
pub fn spawn_scheduler_loop(orchestrator: Arc<Mutex<InferenceOrchestrator>>) {
    tokio::spawn(async move {
        tracing::info!("Background scheduler loop started");
        loop {
            {
                let mut guard = orchestrator.lock().await;
                if let Err(e) = guard.step() {
                    tracing::error!("Scheduler step failed: {:?}", e);
                }
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    });
}
