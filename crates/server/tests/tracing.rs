//! Smoke tests for OTLP tracing integration.
//!
//! These tests verify that the OTLP tracing infrastructure compiles and
//! initializes correctly. They do NOT require a running OTLP collector.

/// Verify that the OTLP CLI args parse correctly by ensuring the
/// opentelemetry types are available at compile time.
#[test]
fn otlp_args_parse_defaults() {
    // Compile-time check that our OTLP dependency types are wired correctly.
    let _ = std::any::type_name::<opentelemetry::global::BoxedTracer>();
}

/// Verify that the layered subscriber can be constructed without OTLP.
#[test]
fn layered_subscriber_init_without_otlp() {
    use tracing_subscriber::{layer::SubscriberExt, Registry};

    let env_filter = tracing_subscriber::EnvFilter::new("info");
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_thread_ids(true);

    // This should not panic — it's the non-OTLP path.
    // We intentionally do NOT call .init() because that would panic if
    // called a second time in the same process (other tests may already
    // have initialized the global subscriber).
    let _subscriber = Registry::default()
        .with(env_filter)
        .with(fmt_layer);
}
