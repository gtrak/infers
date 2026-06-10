//! Pre-allocated pool of physical pages for KV cache storage.
//!
//! Pages are allocated from a fixed-size pool at engine init time. The pool
//! manages a free list for O(1) allocation and deallocation.
//!
//! This is CPU-side bookkeeping only — actual GPU buffer management lives
//! at the integration layer (attention.rs / ForwardEngine).

use super::page::{PageId, PageLocation, PageState, PhysicalPage};
use thiserror::Error;

/// Errors produced by the [`PagePool`].
#[derive(Debug, Error)]
pub enum PagePoolError {
    /// The pool has no free pages remaining.
    #[error("Page pool exhausted: no free pages available")]
    PoolExhausted,
    /// The requested page ID does not exist in the pool.
    #[error("Invalid page ID: {0}")]
    InvalidPageId(PageId),
}

/// Pre-allocated pool of physical pages for KV cache storage.
///
/// Pages are allocated from a fixed-size pool at engine init time. The pool
/// manages a free list for O(1) allocation and deallocation.
///
/// This is CPU-side bookkeeping only — actual GPU buffer management lives
/// at the integration layer (attention.rs / ForwardEngine).
#[derive(Debug)]
pub struct PagePool {
    /// Pre-allocated physical pages.
    pages: Vec<PhysicalPage>,
    /// Free list: stack of available page IDs for O(1) alloc/free.
    free_list: Vec<PageId>,
    /// Number of tokens per page.
    page_size: usize,
    /// Number of KV heads per page (for size calculation).
    num_kv_heads: usize,
    /// Head dimension per page (for size calculation).
    head_dim: usize,
}

impl PagePool {
    /// Create a new page pool with `total_pages` pre-allocated physical pages.
    ///
    /// All pages start in `Mutable` state, on the GPU, with refcount 1
    /// (owned by the pool). The free list is populated with every page ID.
    ///
    /// # Arguments
    /// * `total_pages` — Total number of pages in the pool.
    /// * `page_size` — Number of tokens each page can hold.
    /// * `num_kv_heads` — Number of KV heads in the model.
    /// * `head_dim` — Dimension of each head.
    pub fn new(
        total_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let pages: Vec<PhysicalPage> = (0..total_pages)
            .map(|id| {
                let mut page = PhysicalPage::new(id as PageId);
                page.state = PageState::Mutable;
                page.location = PageLocation::Gpu;
                page
            })
            .collect();

        let free_list: Vec<PageId> = (0..total_pages as PageId).collect();

        Self {
            pages,
            free_list,
            page_size,
            num_kv_heads,
            head_dim,
        }
    }

    /// Allocate a single page from the pool.
    ///
    /// Pops from the free list for O(1) allocation. The allocated page
    /// remains in `Mutable` state — the caller is responsible for writing
    /// KV data and eventually sealing it.
    ///
    /// # Errors
    /// Returns [`PagePoolError::PoolExhausted`] if no pages are available.
    pub fn allocate(&mut self) -> Result<PageId, PagePoolError> {
        self.free_list
            .pop()
            .ok_or(PagePoolError::PoolExhausted)
    }

    /// Return a page to the pool's free list.
    ///
    /// Resets the page state to `Mutable` so it can be reused.
    pub fn free(&mut self, page_id: PageId) {
        if let Some(page) = self.pages.get_mut(page_id as usize) {
            page.state = PageState::Mutable;
        }
        self.free_list.push(page_id);
    }

    /// Check whether the pool is full (no free pages).
    pub fn is_full(&self) -> bool {
        self.free_list.is_empty()
    }

    /// Check whether all pages are free.
    pub fn is_empty(&self) -> bool {
        self.free_list.len() == self.pages.len()
    }

    /// Number of pages currently available for allocation.
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of pages in the pool.
    pub fn num_total(&self) -> usize {
        self.pages.len()
    }

    /// Look up a physical page by its ID.
    ///
    /// Returns `None` if the page ID is out of range.
    pub fn get(&self, page_id: PageId) -> Option<&PhysicalPage> {
        self.pages.get(page_id as usize)
    }

    /// Look up a physical page by its ID with a mutable reference.
    ///
    /// Returns `None` if the page ID is out of range.
    pub fn get_mut(&mut self, page_id: PageId) -> Option<&mut PhysicalPage> {
        self.pages.get_mut(page_id as usize)
    }

    /// Number of tokens each page can hold.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Size of a single page in bytes.
    ///
    /// Calculated as `page_size * num_kv_heads * head_dim * 2`
    /// (2 bytes per BF16 element).
    pub fn page_bytes(&self) -> usize {
        self.page_size * self.num_kv_heads * self.head_dim * 2
    }

    /// Seal a page, marking it as immutable.
    ///
    /// A sealed page is full and ready for prefix caching. Its content
    /// will not be modified.
    pub fn seal(&mut self, page_id: PageId) {
        if let Some(page) = self.pages.get_mut(page_id as usize) {
            page.state = PageState::Sealed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_allocate_free() {
        let mut pool = PagePool::new(4, 16, 8, 128);
        assert_eq!(pool.num_free(), 4);
        assert_eq!(pool.num_total(), 4);
        assert!(!pool.is_full());
        assert!(pool.is_empty()); // Fresh pool: all pages free

        // Allocate all pages
        let p0 = pool.allocate().unwrap();
        let p1 = pool.allocate().unwrap();
        let p2 = pool.allocate().unwrap();
        let p3 = pool.allocate().unwrap();
        assert_eq!(pool.num_free(), 0);
        assert!(pool.is_full());

        // Free them all
        pool.free(p0);
        pool.free(p1);
        pool.free(p2);
        pool.free(p3);
        assert_eq!(pool.num_free(), 4);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_pool_exhaustion() {
        let mut pool = PagePool::new(2, 16, 8, 128);
        pool.allocate().unwrap();
        pool.allocate().unwrap();
        let result = pool.allocate();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PagePoolError::PoolExhausted));
    }

    #[test]
    fn test_pool_seal() {
        let mut pool = PagePool::new(2, 16, 8, 128);
        let pid = pool.allocate().unwrap();
        assert_eq!(pool.get(pid).unwrap().state, PageState::Mutable);

        pool.seal(pid);
        assert_eq!(pool.get(pid).unwrap().state, PageState::Sealed);
    }

    #[test]
    fn test_pool_page_bytes() {
        // page_size=16, num_kv_heads=8, head_dim=128
        // bytes = 16 * 8 * 128 * 2 = 32768
        let pool = PagePool::new(4, 16, 8, 128);
        assert_eq!(pool.page_bytes(), 32_768);
    }

    #[test]
    fn test_pool_double_free() {
        let mut pool = PagePool::new(2, 16, 8, 128);

        // Allocate a page
        let pid = pool.allocate().unwrap();
        assert_eq!(pool.num_free(), 1);

        // Free it
        pool.free(pid);
        assert_eq!(pool.num_free(), 2);

        // Allocate again — should succeed and return a valid page
        let pid2 = pool.allocate().unwrap();
        assert_eq!(pool.num_free(), 1);

        // The page should be Mutable after free+reallocate
        assert_eq!(pool.get(pid2).unwrap().state, PageState::Mutable);
    }
}
