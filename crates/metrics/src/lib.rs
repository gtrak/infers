/// Prometheus metrics crate.
pub mod registry;

pub use registry::*;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use prometheus::Encoder;

// @lat: [[lat.md/lat#Metrics#Metrics HTTP Endpoint]]
pub async fn metrics_handler() -> impl IntoResponse {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    let body = String::from_utf8(buffer).unwrap_or_default();
    (StatusCode::OK, [("content-type", "text/plain; version=0.0.4; charset=utf-8")], body)
}
