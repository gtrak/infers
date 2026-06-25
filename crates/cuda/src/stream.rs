//! CUDA stream management.

pub use cudarc::driver::CudaStream;
use cudarc::driver::CudaContext;

/// A pool of CUDA streams for async execution.
pub struct StreamPool {
    streams: Vec<std::sync::Arc<CudaStream>>,
}

impl StreamPool {
    /// Create a stream pool with one default (null) stream per context.
    ///
    /// Uses default streams (not non-blocking) so that cuda-core's `cc_stream`
    /// (also the null stream) shares the same stream and operations are ordered
    /// FIFO. Non-blocking streams would NOT synchronize with the null stream,
    /// causing race conditions between cudarc and cuda-core kernel launches.
    pub fn new(contexts: &[std::sync::Arc<CudaContext>]) -> anyhow::Result<Self> {
        let mut streams = Vec::with_capacity(contexts.len());
        for ctx in contexts {
            let stream = ctx.default_stream();
            streams.push(stream);
        }
        Ok(Self { streams })
    }

    /// Get a stream by index.
    pub fn get(&self, index: usize) -> Option<&std::sync::Arc<CudaStream>> {
        self.streams.get(index)
    }

    /// Number of streams in the pool.
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}
