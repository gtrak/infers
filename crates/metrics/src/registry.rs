/// Prometheus registry and metric definitions for inference server monitoring.
use lazy_static::lazy_static;
use prometheus::{
    Histogram, HistogramOpts, Opts, Registry, Counter, Gauge,
};

lazy_static! {
    // @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Counters#Tokens Generated]]
    pub static ref TOKENS_GENERATED: Counter = Counter::with_opts(
        Opts::new("infers_tokens_generated_total", "Total tokens generated")
    ).unwrap();

    // @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Gauges#Active Sessions]]
    pub static ref ACTIVE_SESSIONS: Gauge = Gauge::with_opts(
        Opts::new("infers_active_sessions", "Number of active inference sessions")
    ).unwrap();

    // @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Gauges#KV Cache Usage Bytes]]
    pub static ref KV_CACHE_USAGE_BYTES: Gauge = Gauge::with_opts(
        Opts::new("infers_kv_cache_usage_bytes", "KV cache memory usage in bytes")
    ).unwrap();

    // @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Gauges#Batch Size]]
    pub static ref BATCH_SIZE: Gauge = Gauge::with_opts(
        Opts::new("infers_batch_size", "Current batch size")
    ).unwrap();

    // @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Gauges#MTP Acceptance Rate]]
    pub static ref MTP_ACCEPTANCE_RATE: Gauge = Gauge::with_opts(
        Opts::new("infers_mtp_acceptance_rate", "MTP draft token acceptance rate")
    ).unwrap();

    // @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Histograms#Request Latency]]
    pub static ref REQUEST_LATENCY: Histogram = Histogram::with_opts(
        HistogramOpts::new("infers_request_latency_seconds", "Request latency in seconds")
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    ).unwrap();

    // @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions#Gauges#GPU Memory Usage Bytes]]
    pub static ref GPU_MEMORY_USAGE_BYTES: Gauge = Gauge::with_opts(
        Opts::new("infers_gpu_memory_usage_bytes", "GPU memory usage in bytes")
    ).unwrap();

    // @lat: [[lat.md/lat#Metrics#Registry and Metric Definitions]]
    pub static ref REGISTRY: Registry = {
        let registry = Registry::new();
        registry.register(Box::new(TOKENS_GENERATED.clone())).unwrap();
        registry.register(Box::new(ACTIVE_SESSIONS.clone())).unwrap();
        registry.register(Box::new(KV_CACHE_USAGE_BYTES.clone())).unwrap();
        registry.register(Box::new(BATCH_SIZE.clone())).unwrap();
        registry.register(Box::new(MTP_ACCEPTANCE_RATE.clone())).unwrap();
        registry.register(Box::new(REQUEST_LATENCY.clone())).unwrap();
        registry.register(Box::new(GPU_MEMORY_USAGE_BYTES.clone())).unwrap();
        registry
    };
}
