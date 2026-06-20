//! CPU-side storage for evicted KV cache page data.
//!
//! The `CpuPagePool` stores page data as `Vec<u8>` blobs that have been
//! evicted from GPU memory. The actual GPU→CPU copy is performed by the
//! backend crate (which owns the GPU buffers and CUDA streams); this pool
//! provides the CPU-side storage and memory tracking.
//!
//! # Layering
//!
//! ```ignore
//! backend (owns GPU buffer, CUDA stream)
//!   │  cudaMemcpyAsync page_pool[...] → pinned_cpu_buffer
//!   │  CpuPagePool::store(page_id, cpu_buffer)
//!   ▼
//! infers-kv (CPU-side metadata + storage)
//!   CpuPagePool: page_id → Vec<u8>
//!   EvictedSequence: original page order for restoration
//! ```

use super::page::PageId;

/// Errors produced by [`CpuPagePool`] operations.
#[derive(Debug, Clone)]
pub enum EvictionError {
    /// The page has already been evicted.
    AlreadyEvicted(PageId),
    /// The page was not found in the eviction pool.
    NotEvicted(PageId),
    /// The eviction pool has exceeded its memory budget.
    BudgetExceeded { budget: usize, needed: usize },
    /// The page data size does not match the expected page size.
    SizeMismatch { expected: usize, actual: usize },
    /// The sequence has no pages to evict.
    EmptySequence,
}

impl std::fmt::Display for EvictionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyEvicted(id) => write!(f, "Page {id} is already evicted"),
            Self::NotEvicted(id) => write!(f, "Page {id} is not evicted"),
            Self::BudgetExceeded { budget, needed } => {
                write!(f, "Eviction budget exceeded: {budget} bytes budget, {needed} bytes needed")
            }
            Self::SizeMismatch { expected, actual } => {
                write!(f, "Page data size mismatch: expected {expected} bytes, got {actual} bytes")
            }
            Self::EmptySequence => write!(f, "Sequence has no pages to evict"),
        }
    }
}

impl std::error::Error for EvictionError {}

/// A snapshot of a sequence's page table at the time of eviction.
///
/// Stores the original page IDs in order so restoration can re-allocate
/// pages in the correct layout.
#[derive(Debug, Clone)]
pub struct EvictedSequence {
    /// The original sequence ID.
    pub seq_id: usize,
    /// Page IDs in order, as they were at eviction time.
    pub page_ids: Vec<PageId>,
    /// Number of tokens that had been written.
    pub num_tokens: usize,
    /// Page size (tokens per page) at eviction time.
    pub page_size: usize,
    /// Wall-clock priority for LRU ordering (set by the scheduler).
    pub last_access: std::time::Instant,
}

/// CPU-side storage for evicted KV cache page data.
///
/// Stores page data as `Vec<u8>` blobs, indexed by `PageId`. Tracks
/// memory usage against a configurable budget. Pages are stored using
/// `store()` and retrieved using `retrieve()`.
///
/// This pool only manages CPU-side storage. The actual GPU→CPU data
/// copy is performed by the backend (which calls `store()` after
/// `cudaMemcpyAsync`).
#[derive(Debug)]
pub struct CpuPagePool {
    /// Page data indexed by page_id. `None` = page not evicted.
    storage: Vec<Option<Vec<u8>>>,
    /// Total bytes currently stored.
    used_bytes: usize,
    /// Maximum bytes allowed for evicted data.
    max_bytes: usize,
    /// Size of each page's data in bytes.
    page_bytes: usize,
}

impl CpuPagePool {
    /// Create a new CPU page pool with a memory budget.
    ///
    /// # Arguments
    /// * `total_pages` — Maximum number of pages that can be tracked.
    /// * `page_bytes` — Size of each page's data in bytes.
    /// * `max_bytes` — Maximum total bytes for evicted data.
    pub fn new(total_pages: usize, page_bytes: usize, max_bytes: usize) -> Self {
        Self {
            storage: (0..total_pages).map(|_| None).collect(),
            used_bytes: 0,
            max_bytes,
            page_bytes,
        }
    }

    /// Store evicted page data for a given page ID.
    ///
    /// Returns an error if:
    /// - The page is already stored.
    /// - Storing the data would exceed the memory budget.
    /// - The data size does not match `page_bytes`.
    ///
    /// Once stored, the data can be retrieved later with `retrieve()`.
    pub fn store(&mut self, page_id: PageId, data: Vec<u8>) -> Result<(), EvictionError> {
        let idx = page_id as usize;
        if idx >= self.storage.len() {
            return Err(EvictionError::NotEvicted(page_id));
        }
        if self.storage[idx].is_some() {
            return Err(EvictionError::AlreadyEvicted(page_id));
        }
        if data.len() != self.page_bytes {
            return Err(EvictionError::SizeMismatch {
                expected: self.page_bytes,
                actual: data.len(),
            });
        }
        if self.used_bytes + self.page_bytes > self.max_bytes {
            return Err(EvictionError::BudgetExceeded {
                budget: self.max_bytes,
                needed: self.used_bytes + self.page_bytes,
            });
        }

        self.used_bytes += self.page_bytes;
        self.storage[idx] = Some(data);
        Ok(())
    }

    /// Retrieve evicted page data for a given page ID.
    ///
    /// Returns `None` if the page is not currently evicted.
    /// The data is removed from the pool (caller must re-store if
    /// re-evicting later).
    pub fn retrieve(&mut self, page_id: PageId) -> Result<Vec<u8>, EvictionError> {
        let idx = page_id as usize;
        if idx >= self.storage.len() {
            return Err(EvictionError::NotEvicted(page_id));
        }
        self.storage[idx]
            .take()
            .inspect(|_| {
                self.used_bytes -= self.page_bytes;
            })
            .ok_or(EvictionError::NotEvicted(page_id))
    }

    /// Remove evicted page data without retrieving it (e.g., on sequence deletion).
    pub fn remove(&mut self, page_id: PageId) {
        let idx = page_id as usize;
        if idx < self.storage.len() && self.storage[idx].is_some() {
            self.storage[idx] = None;
            self.used_bytes -= self.page_bytes;
        }
    }

    /// Check if a page has evicted data stored.
    pub fn is_evicted(&self, page_id: PageId) -> bool {
        let idx = page_id as usize;
        idx < self.storage.len() && self.storage[idx].is_some()
    }

    /// Number of pages currently evicted.
    pub fn num_evicted(&self) -> usize {
        self.storage.iter().filter(|s| s.is_some()).count()
    }

    /// Total bytes currently used by evicted data.
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Maximum bytes allowed for evicted data.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Whether the pool has reached its memory budget.
    pub fn is_full(&self) -> bool {
        self.used_bytes >= self.max_bytes
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_empty_initially() {
        let pool = CpuPagePool::new(10, 1024, 10_240);
        assert_eq!(pool.num_evicted(), 0);
        assert_eq!(pool.used_bytes(), 0);
        assert!(!pool.is_full());
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut pool = CpuPagePool::new(10, 1024, 10_240);
        let data = vec![0u8; 1024];

        pool.store(0, data).unwrap();
        assert_eq!(pool.num_evicted(), 1);
        assert_eq!(pool.used_bytes(), 1024);
        assert!(pool.is_evicted(0));

        let retrieved = pool.retrieve(0).unwrap();
        assert_eq!(retrieved.len(), 1024);
        assert_eq!(pool.num_evicted(), 0);
        assert!(!pool.is_evicted(0));
    }

    #[test]
    fn test_double_store_fails() {
        let mut pool = CpuPagePool::new(10, 1024, 10_240);
        pool.store(0, vec![0u8; 1024]).unwrap();
        let result = pool.store(0, vec![1u8; 1024]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EvictionError::AlreadyEvicted(0)));
    }

    #[test]
    fn test_retrieve_nonexistent_fails() {
        let mut pool = CpuPagePool::new(10, 1024, 10_240);
        let result = pool.retrieve(99);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EvictionError::NotEvicted(99)));
    }

    #[test]
    fn test_budget_exceeded() {
        let mut pool = CpuPagePool::new(10, 2048, 4096); // 2 pages max
        pool.store(0, vec![0u8; 2048]).unwrap();
        pool.store(1, vec![0u8; 2048]).unwrap();
        assert!(pool.is_full());

        let result = pool.store(2, vec![0u8; 2048]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EvictionError::BudgetExceeded { .. }));
    }

    #[test]
    fn test_size_mismatch() {
        let mut pool = CpuPagePool::new(10, 1024, 10_240);
        let result = pool.store(0, vec![0u8; 512]); // wrong size
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EvictionError::SizeMismatch { .. }));
    }

    #[test]
    fn test_remove() {
        let mut pool = CpuPagePool::new(10, 1024, 10_240);
        pool.store(0, vec![0u8; 1024]).unwrap();
        pool.remove(0);
        assert!(!pool.is_evicted(0));
        assert_eq!(pool.used_bytes(), 0);
    }

    #[test]
    fn test_evicted_sequence() {
        let evicted = EvictedSequence {
            seq_id: 42,
            page_ids: vec![0, 1, 2],
            num_tokens: 48,
            page_size: 16,
            last_access: std::time::Instant::now(),
        };
        assert_eq!(evicted.seq_id, 42);
        assert_eq!(evicted.page_ids.len(), 3);
        assert_eq!(evicted.num_tokens, 48);
    }
}
