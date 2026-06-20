//! Physical page management for the paged KV cache pool.

use std::sync::atomic::AtomicU32;

/// Unique identifier for a physical page in the pool.
pub type PageId = u32;

/// Sentinel value indicating no page is assigned.
pub const INVALID_PAGE_ID: PageId = u32::MAX;

/// Whether a page is still being written to or is sealed (immutable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageState {
    /// Page is still being filled. New tokens can be written.
    Mutable,
    /// Page is full and sealed. Its content hash can be computed for prefix caching.
    Sealed,
}

/// Where the page data physically resides. CPU offload deferred to future phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageLocation {
    Gpu,
}
/// A physical page in the KV cache page pool.
///
/// Each page holds `page_size` token slots worth of K/V data for a single layer.
/// Pages are stored in a flat pool and referenced by sequences via page tables.
#[derive(Debug)]
pub struct PhysicalPage {
    /// Unique page identifier (index in pool).
    pub page_id: PageId,
    /// Reference count: how many sequence page tables point to this page.
    /// When this drops to 0, the page can be returned to the free list.
    pub refcount: AtomicU32,
    /// Whether the page is mutable (being written) or sealed (immutable).
    pub state: PageState,
    /// Where the page data physically resides.
    pub location: PageLocation,
}

impl PhysicalPage {
    /// Create a new physical page with the given ID.
    ///
    /// New pages start in the `Mutable` state on the GPU with a refcount of 1.
    pub fn new(page_id: PageId) -> Self {
        Self {
            page_id,
            refcount: AtomicU32::new(1),
            state: PageState::Mutable,
            location: PageLocation::Gpu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_page_starts_mutable() {
        let page = PhysicalPage::new(0);
        assert_eq!(page.page_id, 0);
        assert_eq!(page.state, PageState::Mutable);
        assert_eq!(page.location, PageLocation::Gpu);
        assert_eq!(page.refcount.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn invalid_page_id_is_max() {
        assert_eq!(INVALID_PAGE_ID, u32::MAX);
    }
}
