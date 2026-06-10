//! GPU memory allocation with block pool semantics.

/// Block pool-based GPU memory allocator.
///
/// Pre-allocates GPU memory in fixed-size blocks and recycles them
/// to avoid repeated allocation/deallocation overhead.
#[derive(Debug, Clone)]
pub struct GpuAllocator {
    /// Size of each block in bytes.
    block_size: usize,
    /// Total bytes allocated so far.
    total_allocated: usize,
    /// Maximum bytes allowed.
    max_bytes: usize,
    /// Number of free blocks available for reuse.
    free_count: usize,
}

impl GpuAllocator {
    /// Create a new allocator with the given block size and memory budget.
    pub fn new(block_size: usize, max_bytes: usize) -> Self {
        Self {
            block_size,
            total_allocated: 0,
            max_bytes,
            free_count: 0,
        }
    }

    /// Check if a new block can be allocated within budget.
    pub fn can_allocate(&self) -> bool {
        self.total_allocated + self.block_size <= self.max_bytes
    }

    /// Mark a block as allocated (bookkeeping only — actual GPU allocation happens separately).
    /// Returns Ok(()) if within budget, Err if out of memory.
    pub fn allocate(&mut self) -> anyhow::Result<AllocInfo> {
        if self.free_count > 0 {
            self.free_count -= 1;
            Ok(AllocInfo {
                size: self.block_size,
                reused: true,
            })
        } else if self.can_allocate() {
            self.total_allocated += self.block_size;
            Ok(AllocInfo {
                size: self.block_size,
                reused: false,
            })
        } else {
            anyhow::bail!(
                "Out of GPU memory: requested {} bytes, budget {}/{} used",
                self.block_size,
                self.total_allocated,
                self.max_bytes
            )
        }
    }

    /// Return a block to the pool.
    pub fn free(&mut self) {
        // Guard against free_count overflow: if free_count exceeds total_blocks,
        // something is wrong (e.g., freeing without allocating). Cap it to prevent
        // silent integer overflow in 32/64-bit usize arithmetic.
        let total_blocks = (self.total_allocated / self.block_size) + self.free_count;
        if self.free_count < total_blocks {
            self.free_count += 1;
        }
    }

    /// Size of each block in bytes.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Total bytes allocated so far.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Maximum bytes allowed.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Number of free blocks available for reuse.
    pub fn free_count(&self) -> usize {
        self.free_count
    }

    /// Total number of blocks (allocated + free).
    pub fn total_blocks(&self) -> usize {
        (self.total_allocated / self.block_size) + self.free_count
    }

    /// Memory utilization as a fraction of max budget.
    pub fn utilization(&self) -> f64 {
        self.total_allocated as f64 / self.max_bytes as f64
    }
}

/// Information about an allocation.
#[derive(Debug, Clone)]
pub struct AllocInfo {
    /// Size of the allocated block in bytes.
    pub size: usize,
    /// Whether this allocation reused a freed block.
    pub reused: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_new() {
        let alloc = GpuAllocator::new(16 * 1024 * 1024, 1024 * 1024 * 1024);
        assert_eq!(alloc.block_size(), 16 * 1024 * 1024);
        assert_eq!(alloc.total_allocated(), 0);
        assert_eq!(alloc.free_count(), 0);
        assert_eq!(alloc.max_bytes(), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_allocator_allocate() {
        let mut alloc = GpuAllocator::new(1024, 4096);
        assert!(alloc.can_allocate());

        let info = alloc.allocate().unwrap();
        assert_eq!(info.size, 1024);
        assert!(!info.reused);
        assert_eq!(alloc.total_allocated(), 1024);
    }

    #[test]
    fn test_allocator_free_reuse() {
        let mut alloc = GpuAllocator::new(1024, 4096);

        let info1 = alloc.allocate().unwrap();
        assert!(!info1.reused);

        alloc.free();
        assert_eq!(alloc.free_count(), 1);

        let info2 = alloc.allocate().unwrap();
        assert!(info2.reused);
    }

    #[test]
    fn test_allocator_out_of_memory() {
        let mut alloc = GpuAllocator::new(2048, 4096);

        alloc.allocate().unwrap(); // 2048
        alloc.allocate().unwrap(); // 4096 total
        assert!(!alloc.can_allocate());

        assert!(alloc.allocate().is_err());
    }

    #[test]
    fn test_allocator_utilization() {
        let mut alloc = GpuAllocator::new(1024, 4096);
        assert!((alloc.utilization() - 0.0).abs() < 1e-6);

        alloc.allocate().unwrap();
        assert!((alloc.utilization() - 0.25).abs() < 1e-6);
    }
}
