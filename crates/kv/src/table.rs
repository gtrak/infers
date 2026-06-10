//! Sequence page tables mapping logical token positions to physical pages.

use super::page::PageId;

/// Page table mapping logical token positions to physical page IDs.
///
/// A sequence does not own pages directly — it holds a page table pointing
/// into the shared `PagePool`. The table maps from logical token position
/// to physical page, enabling paged attention kernels to traverse KV data
/// without contiguous allocation.
#[derive(Debug, Clone)]
pub struct SequencePageTable {
    /// Ordered list of physical page IDs for this sequence.
    /// page_ids\[i\] is the physical page holding tokens \[i\*page_size .. (i\+1)\*page_size\).
    pub page_ids: Vec<PageId>,
    /// Total number of tokens currently stored across all pages.
    pub num_tokens: usize,
    /// Page size (number of tokens per page).
    pub page_size: usize,
}

impl SequencePageTable {
    /// Create an empty page table with the given page size.
    pub fn new(page_size: usize) -> Self {
        assert!(page_size > 0, "page_size must be > 0");
        Self {
            page_ids: Vec::new(),
            num_tokens: 0,
            page_size,
        }
    }

    /// Number of pages currently assigned to this sequence.
    pub fn num_pages(&self) -> usize {
        self.page_ids.len()
    }

    /// Returns the ID of the tail (last) page, or `None` if the table is empty.
    pub fn tail_page_id(&self) -> Option<PageId> {
        self.page_ids.last().copied()
    }

    /// Returns `true` if the tail page is full (needs a new page for the next token).
    ///
    /// A page is full when `num_tokens` is an exact multiple of `page_size`.
    /// An empty table (0 tokens) also returns `true` since we need a new page
    /// to write the first token.
    pub fn is_tail_page_full(&self) -> bool {
        if self.num_tokens == 0 {
            return true;
        }
        self.num_tokens % self.page_size == 0
    }

    /// Append a new page to the sequence's page table.
    pub fn push_page(&mut self, page_id: PageId) {
        self.page_ids.push(page_id);
    }

    /// Increment the token count by one.
    pub fn add_token(&mut self) {
        self.num_tokens += 1;
    }

    /// Remove and return the last page ID from the table.
    ///
    /// Returns `None` if the table has no pages.
    pub fn remove_last_page(&mut self) -> Option<PageId> {
        if self.page_ids.is_empty() {
            None
        } else {
            Some(self.page_ids.pop().unwrap())
        }
    }

    /// Returns the page IDs slice for CUDA kernel consumption.
    ///
    /// The returned slice can be uploaded to the GPU as a block table
    /// for paged attention kernel dispatch.
    pub fn block_table(&self) -> &[PageId] {
        &self.page_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_table_is_empty() {
        let table = SequencePageTable::new(16);
        assert_eq!(table.num_pages(), 0);
        assert_eq!(table.num_tokens, 0);
        assert_eq!(table.page_size, 16);
        assert_eq!(table.tail_page_id(), None);
    }

    #[test]
    fn push_page_and_count() {
        let mut table = SequencePageTable::new(16);
        table.push_page(5);
        assert_eq!(table.num_pages(), 1);
        assert_eq!(table.tail_page_id(), Some(5));
    }

    #[test]
    fn is_tail_page_full_when_empty() {
        let table = SequencePageTable::new(16);
        assert!(table.is_tail_page_full());
    }

    #[test]
    fn is_tail_page_full_when_exactly_full() {
        let mut table = SequencePageTable::new(16);
        table.push_page(0);
        for _ in 0..16 {
            table.add_token();
        }
        assert!(table.is_tail_page_full());
    }

    #[test]
    fn is_tail_page_not_full_partial() {
        let mut table = SequencePageTable::new(16);
        table.push_page(0);
        for _ in 0..15 {
            table.add_token();
        }
        assert!(!table.is_tail_page_full());
    }

    #[test]
    fn remove_last_page() {
        let mut table = SequencePageTable::new(16);
        table.push_page(10);
        table.push_page(20);
        let id = table.remove_last_page().unwrap();
        assert_eq!(id, 20);
        assert_eq!(table.num_pages(), 1);
        assert_eq!(table.tail_page_id(), Some(10));
    }

    #[test]
    fn remove_last_page_empty() {
        let mut table = SequencePageTable::new(16);
        assert!(table.remove_last_page().is_none());
    }

    #[test]
    fn block_table_returns_slice() {
        let mut table = SequencePageTable::new(16);
        table.push_page(1);
        table.push_page(2);
        table.push_page(3);
        let bt = table.block_table();
        assert_eq!(bt, &[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "page_size must be > 0")]
    fn zero_page_size_panics() {
        SequencePageTable::new(0);
    }
}
