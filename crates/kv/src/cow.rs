//! Copy-on-Write (COW) logic for shared KV cache pages.
//!
//! When a sequence appends a token to a page shared between multiple sequences
//! (refcount > 1), a private copy must be made before writing. This module
//! handles the bookkeeping: deciding when COW is needed, allocating a new page,
//! updating refcounts, and replacing page IDs in the sequence table.
//!
//! The actual GPU memory copy is performed by the attention kernel layer, not
//! this bookkeeping crate.

use std::sync::atomic::Ordering::SeqCst;

use super::page::{PageId, PageState, INVALID_PAGE_ID};
use super::pool::PagePool;
use super::table::SequencePageTable;
use thiserror::Error;

// @lat: [[lat.md/lat#Paged KV Types#CowResult]]
/// Result of a copy-on-write check.
///
/// Returned by [`ensure_mutable_page`] to indicate whether the caller can write
/// directly to the existing tail page or must first copy data to a new page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CowResult {
    /// No COW was needed — the page is exclusively owned (refcount == 1) and
    /// mutable. The sequence can write directly to this page.
    NoCowNeeded {
        /// The page ID to write to (same as the existing tail page, or a freshly
        /// allocated page if the table was empty/full).
        page_id: PageId,
    },
    /// COW was performed — a new page was allocated and the original page's
    /// refcount was decremented. The caller must copy GPU data from the
    /// original page to the new page before writing the new token.
    CowPerformed {
        /// The newly allocated page ID.
        new_page_id: PageId,
        /// The original page ID that was shared (or sealed).
        /// The caller may need this for GPU-side memcpy.
        original_page_id: PageId,
    },
}

// @lat: [[lat.md/lat#Paged KV Types#CowError]]
/// Errors produced by COW operations.
#[derive(Debug, Error)]
pub enum CowError {
    /// The page pool has no free pages for COW allocation.
    #[error("Page pool exhausted: cannot allocate COW page")]
    PoolExhausted,
    /// The requested page ID is invalid (out of range or `INVALID_PAGE_ID`).
    #[error("Invalid page ID: {0}")]
    InvalidPageId(PageId),
}

// @lat: [[lat.md/lat#Paged KV Types#ensure_mutable_page]]
/// Ensure that the tail page of a sequence is exclusively owned and mutable.
///
/// Checks the current tail page's refcount and state:
/// - If the table is empty or the tail page is full, allocates a new page and
///   appends it to the table. Returns `NoCowNeeded`.
/// - If the tail page is exclusively owned (refcount == 1) and mutable,
///   returns `NoCowNeeded` — write in place.
/// - If the tail page is shared (refcount > 1), performs COW: allocates a new
///   page, decrements the original refcount, and replaces the table entry.
///   Returns `CowPerformed`.
/// - If the tail page is exclusively owned but sealed, allocates a new page
///   and replaces the table entry. Returns `CowPerformed`.
///
/// # Errors
/// Returns [`CowError::PoolExhausted`] if the pool has no free pages.
/// Returns [`CowError::InvalidPageId`] if the tail page ID lookup fails.
pub fn ensure_mutable_page(
    pool: &mut PagePool,
    table: &mut SequencePageTable,
) -> Result<CowResult, CowError> {
    // Case 1: Table is empty or tail page is full — need a fresh page
    if table.is_tail_page_full() {
        let new_page_id = pool.allocate().map_err(|_| CowError::PoolExhausted)?;
        table.push_page(new_page_id);
        return Ok(CowResult::NoCowNeeded {
            page_id: new_page_id,
        });
    }

    // Case 2: Tail page exists and has space
    let tail_page_id = table
        .tail_page_id()
        .ok_or(CowError::InvalidPageId(INVALID_PAGE_ID))?;

    // Read refcount and state without holding the borrow across the allocate call
    let (refcount, state) = {
        let page = pool
            .get(tail_page_id)
            .ok_or(CowError::InvalidPageId(tail_page_id))?;
        (page.refcount.load(SeqCst), page.state)
    };

    if refcount == 1 && state == PageState::Mutable {
        // Exclusively owned and mutable — write in place
        Ok(CowResult::NoCowNeeded {
            page_id: tail_page_id,
        })
    } else if refcount > 1 {
        // Shared page — COW required
        let new_page_id = pool.allocate().map_err(|_| CowError::PoolExhausted)?;
        // Decrement refcount on the original page
        let page = pool
            .get(tail_page_id)
            .expect("page must exist");
        page.refcount.fetch_sub(1, SeqCst);
        // Replace the tail page in the sequence table
        let last_idx = table.page_ids.len() - 1;
        table.page_ids[last_idx] = new_page_id;
        Ok(CowResult::CowPerformed {
            new_page_id,
            original_page_id: tail_page_id,
        })
    } else {
        // refcount == 1 but Sealed — need a new page (sealed is immutable)
        let new_page_id = pool.allocate().map_err(|_| CowError::PoolExhausted)?;
        let last_idx = table.page_ids.len() - 1;
        table.page_ids[last_idx] = new_page_id;
        Ok(CowResult::CowPerformed {
            new_page_id,
            original_page_id: tail_page_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ensure_mutable_empty_table() {
        let mut pool = PagePool::new(4, 16, 8, 128);
        let mut table = SequencePageTable::new(16);

        let result = ensure_mutable_page(&mut pool, &mut table).unwrap();

        assert!(matches!(&result, CowResult::NoCowNeeded { .. }));
        if let CowResult::NoCowNeeded { page_id } = result {
            assert_eq!(pool.get(page_id).unwrap().state, PageState::Mutable);
        }
        assert_eq!(table.num_pages(), 1);
        // Page was allocated — 3 free pages left
        assert_eq!(pool.num_free(), 3);
    }

    #[test]
    fn test_ensure_mutable_exclusive() {
        let mut pool = PagePool::new(4, 16, 8, 128);
        let mut table = SequencePageTable::new(16);

        // Allocate a page and push to table, simulating a partially filled page
        let page_id = pool.allocate().unwrap();
        table.push_page(page_id);
        table.add_token(); // 1 token — not full

        // Page has refcount 1, state is Mutable
        assert_eq!(pool.get(page_id).unwrap().refcount.load(SeqCst), 1);

        let result = ensure_mutable_page(&mut pool, &mut table).unwrap();

        assert!(matches!(&result, CowResult::NoCowNeeded { .. }));
        if let CowResult::NoCowNeeded { page_id: pid } = result {
            assert_eq!(pid, page_id);
        }
        // No extra pages allocated
        assert_eq!(pool.num_free(), 3);
        assert_eq!(table.num_pages(), 1);
    }

    #[test]
    fn test_ensure_mutable_shared_cow() {
        let mut pool = PagePool::new(4, 16, 8, 128);
        let mut table = SequencePageTable::new(16);

        // Allocate a page, manually raise refcount to simulate sharing
        let page_id = pool.allocate().unwrap();
        table.push_page(page_id);
        table.add_token();

        // Simulate sharing: bump refcount
        pool.get(page_id)
            .unwrap()
            .refcount
            .fetch_add(1, SeqCst);
        assert_eq!(pool.get(page_id).unwrap().refcount.load(SeqCst), 2);

        let result = ensure_mutable_page(&mut pool, &mut table).unwrap();

        assert!(matches!(&result, CowResult::CowPerformed { .. }));
        if let CowResult::CowPerformed {
            new_page_id,
            original_page_id,
        } = result
        {
            assert_eq!(original_page_id, page_id);
            assert_ne!(new_page_id, page_id);

            // Original page refcount decremented back to 1
            assert_eq!(pool.get(page_id).unwrap().refcount.load(SeqCst), 1);

            // New page is mutable with refcount 1
            assert_eq!(pool.get(new_page_id).unwrap().state, PageState::Mutable);
            assert_eq!(pool.get(new_page_id).unwrap().refcount.load(SeqCst), 1);

            // Table now points to the new page
            assert_eq!(table.tail_page_id(), Some(new_page_id));
        }

        // Two pages allocated, 2 free pages left
        assert_eq!(pool.num_free(), 2);
    }

    #[test]
    fn test_ensure_mutable_full_page() {
        let mut pool = PagePool::new(4, 16, 8, 128);
        let mut table = SequencePageTable::new(16);

        // Fill the first page to capacity
        let page_id = pool.allocate().unwrap();
        table.push_page(page_id);
        for _ in 0..16 {
            table.add_token();
        }
        assert!(table.is_tail_page_full());

        let result = ensure_mutable_page(&mut pool, &mut table).unwrap();

        assert!(matches!(&result, CowResult::NoCowNeeded { .. }));
        if let CowResult::NoCowNeeded { page_id: new_pid } = result {
            assert_ne!(new_pid, page_id);
        }
        assert_eq!(table.num_pages(), 2);
    }

    #[test]
    fn test_cow_decrements_original_refcount() {
        let mut pool = PagePool::new(4, 16, 8, 128);
        let mut table = SequencePageTable::new(16);

        let page_id = pool.allocate().unwrap();
        table.push_page(page_id);
        table.add_token();

        // Simulate sharing: refcount = 3
        pool.get(page_id)
            .unwrap()
            .refcount
            .fetch_add(2, SeqCst);
        assert_eq!(pool.get(page_id).unwrap().refcount.load(SeqCst), 3);

        let _ = ensure_mutable_page(&mut pool, &mut table).unwrap();

        // After COW, original page refcount should be 2 (decremented by 1)
        assert_eq!(pool.get(page_id).unwrap().refcount.load(SeqCst), 2);
    }

    #[test]
    fn test_cow_replaces_table_entry() {
        let mut pool = PagePool::new(4, 16, 8, 128);
        let mut table = SequencePageTable::new(16);

        let page_id = pool.allocate().unwrap();
        table.push_page(page_id);
        table.add_token();

        // Simulate sharing
        pool.get(page_id)
            .unwrap()
            .refcount
            .fetch_add(1, SeqCst);

        let result = ensure_mutable_page(&mut pool, &mut table).unwrap();

        if let CowResult::CowPerformed { new_page_id, .. } = result {
            // Table's last entry should now be the new page
            assert_eq!(table.page_ids[table.page_ids.len() - 1], new_page_id);
            // Original page ID is no longer in the table
            assert!(!table.page_ids.contains(&page_id));
        } else {
            panic!("Expected CowPerformed");
        }
    }

    #[test]
    fn test_cow_pool_exhausted() {
        let mut pool = PagePool::new(1, 16, 8, 128);
        let mut table = SequencePageTable::new(16);

        // Allocate the only page
        pool.allocate().unwrap();
        assert!(pool.is_full());

        // Try to get a mutable page — should fail
        let result = ensure_mutable_page(&mut pool, &mut table);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CowError::PoolExhausted));
    }

    #[test]
    fn test_ensure_mutable_sealed_exclusive() {
        let mut pool = PagePool::new(4, 16, 8, 128);
        let mut table = SequencePageTable::new(16);

        let page_id = pool.allocate().unwrap();
        table.push_page(page_id);
        // Partially filled page but manually sealed
        table.add_token();
        pool.seal(page_id);

        let result = ensure_mutable_page(&mut pool, &mut table).unwrap();

        // Sealed page is immutable — needs COW even with refcount 1
        assert!(matches!(&result, CowResult::CowPerformed { .. }));
        if let CowResult::CowPerformed {
            new_page_id,
            original_page_id,
        } = result
        {
            assert_eq!(original_page_id, page_id);
            assert_ne!(new_page_id, page_id);
            // Original still sealed, refcount unchanged (sole owner)
            assert_eq!(pool.get(page_id).unwrap().state, PageState::Sealed);
            assert_eq!(pool.get(page_id).unwrap().refcount.load(SeqCst), 1);
            // New page is mutable
            assert_eq!(pool.get(new_page_id).unwrap().state, PageState::Mutable);
        }
    }

    /// Verify that after COW on a shared page, the original page remains
    /// completely unchanged: state, refcount, and identity are preserved.
    // @lat: [[lat.md/lat#Paged KV Types#ensure_mutable_page]]
    #[test]
    fn test_cow_immutable_original() {
        let mut pool = PagePool::new(8, 16, 8, 128);
        let mut table = SequencePageTable::new(16);

        let original_page_id = pool.allocate().unwrap();
        table.push_page(original_page_id);
        table.add_token();

        // Simulate sharing: refcount = 2
        pool.get(original_page_id)
            .unwrap()
            .refcount
            .fetch_add(1, SeqCst);

        // Record original state before COW
        let original_state_before = pool.get(original_page_id).unwrap().state;
        let original_refcount_before = pool.get(original_page_id).unwrap().refcount.load(SeqCst);
        assert_eq!(original_refcount_before, 2);

        // Trigger COW
        let result = ensure_mutable_page(&mut pool, &mut table).unwrap();

        assert!(matches!(&result, CowResult::CowPerformed { .. }));
        let (cow_new_page_id, cow_original_page_id) = if let CowResult::CowPerformed {
            new_page_id,
            original_page_id: orig,
        } = result
        {
            (new_page_id, orig)
        } else {
            unreachable!();
        };

        assert_eq!(cow_original_page_id, original_page_id);

        // Verify original page is completely unchanged:
        // 1. State preserved
        assert_eq!(
            pool.get(original_page_id).unwrap().state,
            original_state_before,
            "Original page state must not change after COW"
        );
        // 2. Refcount decremented by exactly 1 (from 2 to 1)
        let original_refcount_after =
            pool.get(original_page_id).unwrap().refcount.load(SeqCst);
        assert_eq!(
            original_refcount_after,
            1,
            "Original page refcount should be decremented by exactly 1 after COW"
        );
        // 3. New page is distinct and mutable
        assert_ne!(cow_new_page_id, original_page_id);
        assert_eq!(
            pool.get(cow_new_page_id).unwrap().state,
            PageState::Mutable
        );
        assert_eq!(pool.get(cow_new_page_id).unwrap().refcount.load(SeqCst), 1);
    }
}
