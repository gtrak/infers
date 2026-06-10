/// Prometheus registry and metric definitions for inference server monitoring.
use std::sync::LazyLock;
use prometheus::{
    Histogram, HistogramOpts, Opts, Registry, Counter, Gauge,
};

// @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Counters#Tokens Generated]]
pub static TOKENS_GENERATED: LazyLock<Counter> = LazyLock::new(|| {
    Counter::with_opts(
        Opts::new("infers_tokens_generated_total", "Total tokens generated")
    ).expect("failed to create TOKENS_GENERATED counter: duplicate metric name?")
});

// @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Gauges#Active Sessions]]
pub static ACTIVE_SESSIONS: LazyLock<Gauge> = LazyLock::new(|| {
    Gauge::with_opts(
        Opts::new("infers_active_sessions", "Number of active inference sessions")
    ).expect("failed to create ACTIVE_SESSIONS gauge: duplicate metric name?")
});

// @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Gauges#KV Cache Usage Bytes]]
pub static KV_CACHE_USAGE_BYTES: LazyLock<Gauge> = LazyLock::new(|| {
    Gauge::with_opts(
        Opts::new("infers_kv_cache_usage_bytes", "KV cache memory usage in bytes")
    ).expect("failed to create KV_CACHE_USAGE_BYTES gauge: duplicate metric name?")
});

// @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Gauges#Batch Size]]
pub static BATCH_SIZE: LazyLock<Gauge> = LazyLock::new(|| {
    Gauge::with_opts(
        Opts::new("infers_batch_size", "Current batch size")
    ).expect("failed to create BATCH_SIZE gauge: duplicate metric name?")
});

// @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Gauges#MTP Acceptance Rate]]
pub static MTP_ACCEPTANCE_RATE: LazyLock<Gauge> = LazyLock::new(|| {
    Gauge::with_opts(
        Opts::new("infers_mtp_acceptance_rate", "MTP draft token acceptance rate")
    ).expect("failed to create MTP_ACCEPTANCE_RATE gauge: duplicate metric name?")
});

// @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Histograms#Request Latency]]
pub static REQUEST_LATENCY: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new("infers_request_latency_seconds", "Request latency in seconds")
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    ).expect("failed to create REQUEST_LATENCY histogram: duplicate metric name?")
});

// @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Gauges#GPU Memory Usage Bytes]]
pub static GPU_MEMORY_USAGE_BYTES: LazyLock<Gauge> = LazyLock::new(|| {
    Gauge::with_opts(
        Opts::new("infers_gpu_memory_usage_bytes", "GPU memory usage in bytes")
    ).expect("failed to create GPU_MEMORY_USAGE_BYTES gauge: duplicate metric name?")
});

// @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions]]
pub static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let registry = Registry::new();
    registry.register(Box::new(TOKENS_GENERATED.clone())).expect("failed to register TOKENS_GENERATED");
    registry.register(Box::new(ACTIVE_SESSIONS.clone())).expect("failed to register ACTIVE_SESSIONS");
    registry.register(Box::new(KV_CACHE_USAGE_BYTES.clone())).expect("failed to register KV_CACHE_USAGE_BYTES");
    registry.register(Box::new(BATCH_SIZE.clone())).expect("failed to register BATCH_SIZE");
    registry.register(Box::new(MTP_ACCEPTANCE_RATE.clone())).expect("failed to register MTP_ACCEPTANCE_RATE");
    registry.register(Box::new(REQUEST_LATENCY.clone())).expect("failed to register REQUEST_LATENCY");
    registry.register(Box::new(GPU_MEMORY_USAGE_BYTES.clone())).expect("failed to register GPU_MEMORY_USAGE_BYTES");
    registry
});
