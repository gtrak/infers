//! Orchestrator for paged KV cache management.
//!
//! `PagedKvManager` ties together the page pool, prefix cache, and
//! copy-on-write logic into a single coordinator that manages multiple
//! concurrent sequences.

use std::sync::{Arc, Mutex};

use super::cow::{ensure_mutable_page, CowResult};
use super::page::PageId;
use super::pool::PagePool;
use super::prefix::{hash_page, PrefixCache};
use super::table::SequencePageTable;
use thiserror::Error;

// @lat: [[lat.md/lat#Phase 4.6 Deliverables#Paged KV Types#SequenceId]]
/// Identifier for a sequence in the manager.
pub type SequenceId = usize;

// @lat: [[lat.md/lat#Phase 4.6 Deliverables#Paged KV Types#ManagerError]]
/// Errors produced by the [`PagedKvManager`].
#[derive(Debug, Error)]
pub enum ManagerError {
    /// The requested sequence ID is not found or has already been deleted.
    #[error("Sequence {0} not found or already deleted")]
    InvalidSequence(SequenceId),
    /// The page pool has no free pages remaining.
    #[error("Page pool exhausted")]
    PoolExhausted,
}

// @lat: [[lat.md/lat#Phase 4.6 Deliverables#Paged KV Types#PagedKvManager]]
/// Orchestrates paged KV cache management.
///
/// Manages sequences, page allocation, prefix caching, and copy-on-write
/// logic. The pool and cache are shared via `Arc<Mutex<>>` to allow
/// safe access from multiple contexts.
pub struct PagedKvManager {
    /// Shared page pool for allocation and deallocation.
    page_pool: Arc<Mutex<PagePool>>,
    /// Shared prefix cache for page content hashing and sharing.
    prefix_cache: Arc<Mutex<PrefixCache>>,
    /// Number of tokens per page.
    page_size: usize,
    /// Number of KV heads in the model.
    num_kv_heads: usize,
    /// Dimension of each head.
    head_dim: usize,
    /// Sequence page tables indexed by sequence ID.
    sequences: Vec<Option<SequencePageTable>>,
    /// Pool of free sequence IDs for reuse.
    free_sequence_ids: Vec<SequenceId>,
}

impl PagedKvManager {
    /// Create a new PagedKvManager with the given configuration.
    ///
    /// # Arguments
    /// * `total_pages` — Total number of physical pages in the pool.
    /// * `page_size` — Number of tokens per page.
    /// * `num_kv_heads` — Number of KV heads in the model.
    /// * `head_dim` — Dimension of each head.
    /// * `max_cache_bytes` — Memory budget for the prefix cache.
    pub fn new(
        total_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_cache_bytes: usize,
    ) -> Self {
        let page_pool = PagePool::new(total_pages, page_size, num_kv_heads, head_dim);
        let page_bytes = page_size * num_kv_heads * head_dim * 2;
        let prefix_cache = PrefixCache::new(max_cache_bytes, page_bytes);

        Self {
            page_pool: Arc::new(Mutex::new(page_pool)),
            prefix_cache: Arc::new(Mutex::new(prefix_cache)),
            page_size,
            num_kv_heads,
            head_dim,
            sequences: Vec::new(),
            free_sequence_ids: Vec::new(),
        }
    }

    /// Create a new sequence and return its ID.
    ///
    /// Allocates a sequence ID from the free list if available,
    /// otherwise appends a new ID. Creates an empty page table.
    pub fn create_sequence(&mut self) -> SequenceId {
        let seq_id = self
            .free_sequence_ids
            .pop()
            .unwrap_or_else(|| self.sequences.len());

        let table = SequencePageTable::new(self.page_size);
        if seq_id < self.sequences.len() {
            self.sequences[seq_id] = Some(table);
        } else {
            self.sequences.push(Some(table));
        }

        seq_id
    }

    /// Delete a sequence, freeing all its pages back to the pool.
    ///
    /// # Errors
    /// Returns [`ManagerError::InvalidSequence`] if the sequence ID is
    /// invalid or already deleted.
    pub fn delete_sequence(&mut self, seq_id: SequenceId) -> Result<(), ManagerError> {
        let table = self
            .sequences
            .get_mut(seq_id)
            .and_then(|opt| opt.take())
            .ok_or(ManagerError::InvalidSequence(seq_id))?;

        let page_ids = table.page_ids;
        {
            let mut pool = self.page_pool.lock().unwrap();
            for page_id in page_ids {
                pool.free(page_id);
            }
        }

        self.free_sequence_ids.push(seq_id);
        Ok(())
    }

    /// Allocate a new page from the pool and append it to the sequence's page table.
    ///
    /// # Errors
    /// Returns [`ManagerError::InvalidSequence`] if the sequence does not exist.
    /// Returns [`ManagerError::PoolExhausted`] if no pages are available.
    pub fn append_page(&mut self, seq_id: SequenceId) -> Result<PageId, ManagerError> {
        let page_id = {
            let mut pool = self.page_pool.lock().unwrap();
            pool.allocate().map_err(|_| ManagerError::PoolExhausted)?
        };

        let table = self
            .sequences
            .get_mut(seq_id)
            .and_then(|opt| opt.as_mut())
            .ok_or(ManagerError::InvalidSequence(seq_id))?;
        table.push_page(page_id);

        Ok(page_id)
    }

    /// Ensure the tail page of the sequence is writable.
    ///
    /// Delegates to [`ensure_mutable_page`] which handles copy-on-write
    /// if the tail page is shared or sealed.
    ///
    /// # Errors
    /// Returns [`ManagerError::InvalidSequence`] if the sequence does not exist.
    /// Returns [`ManagerError::PoolExhausted`] if no pages are available for COW.
    pub fn ensure_writable(&mut self, seq_id: SequenceId) -> Result<CowResult, ManagerError> {
        let result = {
            let mut pool = self.page_pool.lock().unwrap();
            let table = self
                .sequences
                .get_mut(seq_id)
                .and_then(|opt| opt.as_mut())
                .ok_or(ManagerError::InvalidSequence(seq_id))?;
            ensure_mutable_page(&mut pool, table).map_err(|e| match e {
                super::cow::CowError::PoolExhausted => ManagerError::PoolExhausted,
                super::cow::CowError::InvalidPageId(_) => {
                    ManagerError::InvalidSequence(seq_id)
                }
            })?
        };

        Ok(result)
    }

    /// Increment the token count for a sequence.
    ///
    /// If the token count reaches a page boundary (multiple of `page_size`),
    /// the current tail page is sealed in the pool.
    ///
    /// # Errors
    /// Returns [`ManagerError::InvalidSequence`] if the sequence does not exist.
    pub fn add_token(&mut self, seq_id: SequenceId) -> Result<(), ManagerError> {
        let mut pool = self.page_pool.lock().unwrap();

        let table = self
            .sequences
            .get_mut(seq_id)
            .and_then(|opt| opt.as_mut())
            .ok_or(ManagerError::InvalidSequence(seq_id))?;

        table.num_tokens += 1;

        if table.num_tokens > 0 && table.num_tokens % self.page_size == 0 {
            if let Some(tail_page_id) = table.tail_page_id() {
                pool.seal(tail_page_id);
            }
        }

        Ok(())
    }

    /// Seal the tail page and insert it into the prefix cache.
    ///
    /// Computes a content hash from the provided K/V data and model info,
    /// then inserts the page into the prefix cache for future sharing.
    ///
    /// # Arguments
    /// * `seq_id` — The sequence whose tail page to seal.
    /// * `layer_idx` — The transformer layer index.
    /// * `model_id` — The model identifier string.
    /// * `k_data` — Raw K data bytes for the page.
    /// * `v_data` — Raw V data bytes for the page.
    ///
    /// # Returns
    /// * `Ok(Some(page_id))` if the page was sealed and cached.
    /// * `Ok(None)` if the sequence had no tail page to seal.
    ///
    /// # Errors
    /// Returns [`ManagerError::InvalidSequence`] if the sequence does not exist.
    pub fn seal_and_cache(
        &mut self,
        seq_id: SequenceId,
        layer_idx: usize,
        model_id: &str,
        k_data: &[u8],
        v_data: &[u8],
    ) -> Result<Option<PageId>, ManagerError> {
        let table = self
            .sequences
            .get(seq_id)
            .and_then(|opt| opt.as_ref())
            .ok_or(ManagerError::InvalidSequence(seq_id))?;

        let tail_page_id = table
            .tail_page_id()
            .ok_or(ManagerError::InvalidSequence(seq_id))?;

        {
            let mut pool = self.page_pool.lock().unwrap();
            pool.seal(tail_page_id);
        }

        let hash = hash_page(k_data, v_data, model_id, layer_idx);

        {
            let mut cache = self.prefix_cache.lock().unwrap();
            cache.insert(hash, tail_page_id);
        }

        Ok(Some(tail_page_id))
    }

    /// Get a reference to the sequence's page table.
    ///
    /// # Errors
    /// Returns [`ManagerError::InvalidSequence`] if the sequence does not exist.
    pub fn get_page_table(&self, seq_id: SequenceId) -> Result<&SequencePageTable, ManagerError> {
        self.sequences
            .get(seq_id)
            .and_then(|opt| opt.as_ref())
            .ok_or(ManagerError::InvalidSequence(seq_id))
    }

    /// Get the block table (page ID slice) for CUDA kernel consumption.
    ///
    /// # Errors
    /// Returns [`ManagerError::InvalidSequence`] if the sequence does not exist.
    pub fn block_table(&self, seq_id: SequenceId) -> Result<&[PageId], ManagerError> {
        self.get_page_table(seq_id)
            .map(|table| table.block_table())
    }

    /// Get the number of pages assigned to a sequence.
    ///
    /// # Errors
    /// Returns [`ManagerError::InvalidSequence`] if the sequence does not exist.
    pub fn num_pages(&self, seq_id: SequenceId) -> Result<usize, ManagerError> {
        self.get_page_table(seq_id).map(|table| table.num_pages())
    }

    /// Get the number of tokens in a sequence.
    ///
    /// # Errors
    /// Returns [`ManagerError::InvalidSequence`] if the sequence does not exist.
    pub fn num_tokens(&self, seq_id: SequenceId) -> Result<usize, ManagerError> {
        self.get_page_table(seq_id).map(|table| table.num_tokens)
    }

    /// Get the number of free pages in the pool.
    pub fn num_free_pages(&self) -> usize {
        let pool = self.page_pool.lock().unwrap();
        pool.num_free()
    }

    /// Get the pool utilization ratio (0.0 to 1.0).
    pub fn pool_utilization(&self) -> f64 {
        let pool = self.page_pool.lock().unwrap();
        let total = pool.num_total();
        let free = pool.num_free();
        if total == 0 {
            return 0.0;
        }
        (total - free) as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::page::PageState;

    #[test]
    fn test_create_delete_sequence() {
        let mut manager = PagedKvManager::new(8, 16, 8, 128, 1024);

        // Create a sequence
        let seq_id = manager.create_sequence();
        assert!(manager.get_page_table(seq_id).is_ok());
        assert_eq!(manager.num_pages(seq_id).unwrap(), 0);
        assert_eq!(manager.num_tokens(seq_id).unwrap(), 0);

        // Delete the sequence
        manager.delete_sequence(seq_id).unwrap();

        // Sequence should be gone
        assert!(manager.get_page_table(seq_id).is_err());
    }

    #[test]
    fn test_append_page() {
        let mut manager = PagedKvManager::new(8, 16, 8, 128, 1024);
        let seq_id = manager.create_sequence();

        // Append a page
        let page_id = manager.append_page(seq_id).unwrap();
        assert!(page_id < 8); // Must be within pool range

        // Verify page table
        assert_eq!(manager.num_pages(seq_id).unwrap(), 1);
        let table = manager.get_page_table(seq_id).unwrap();
        assert_eq!(table.page_ids, vec![page_id]);

        // Verify pool has fewer free pages
        assert_eq!(manager.num_free_pages(), 7);
    }

    #[test]
    fn test_add_token_seals_page() {
        let mut manager = PagedKvManager::new(8, 16, 8, 128, 1024);
        let seq_id = manager.create_sequence();

        // Append a page
        manager.append_page(seq_id).unwrap();

        // Add tokens up to page_size
        for _ in 0..16 {
            manager.add_token(seq_id).unwrap();
        }

        // Verify page was sealed
        assert_eq!(manager.num_tokens(seq_id).unwrap(), 16);
        let page_id = manager.block_table(seq_id).unwrap()[0];
        {
            let pool = manager.page_pool.lock().unwrap();
            let page = pool.get(page_id).unwrap();
            assert_eq!(page.state, PageState::Sealed);
        }
    }

    #[test]
    fn test_multiple_sequences() {
        let mut manager = PagedKvManager::new(16, 16, 8, 128, 1024);

        // Create two sequences
        let seq1 = manager.create_sequence();
        let seq2 = manager.create_sequence();
        assert_ne!(seq1, seq2);

        // Append pages to each independently
        let page1 = manager.append_page(seq1).unwrap();
        let page2 = manager.append_page(seq2).unwrap();
        assert_ne!(page1, page2);

        // Verify independent page tables
        assert_eq!(manager.num_pages(seq1).unwrap(), 1);
        assert_eq!(manager.num_pages(seq2).unwrap(), 1);
        assert_eq!(manager.block_table(seq1).unwrap(), &[page1]);
        assert_eq!(manager.block_table(seq2).unwrap(), &[page2]);
    }

    #[test]
    fn test_delete_frees_pages() {
        let mut manager = PagedKvManager::new(4, 16, 8, 128, 1024);

        // Create a sequence and allocate pages
        let seq_id = manager.create_sequence();
        manager.append_page(seq_id).unwrap();
        manager.append_page(seq_id).unwrap();
        assert_eq!(manager.num_free_pages(), 2);

        // Delete the sequence — pages should be freed
        manager.delete_sequence(seq_id).unwrap();
        assert_eq!(manager.num_free_pages(), 4);

        // Create a new sequence — pages should be available
        let seq2 = manager.create_sequence();
        let page = manager.append_page(seq2).unwrap();
        assert!(page < 4);
        assert_eq!(manager.num_free_pages(), 3);
    }

    #[test]
    fn test_ensure_writable_cow() {
        let mut manager = PagedKvManager::new(8, 16, 8, 128, 1024);

        // Create two sequences and allocate pages
        let seq1 = manager.create_sequence();
        let seq2 = manager.create_sequence();
        manager.append_page(seq1).unwrap();
        manager.append_page(seq2).unwrap();

        // ensure_writable on both sequences should succeed
        let result1 = manager.ensure_writable(seq1).unwrap();
        let result2 = manager.ensure_writable(seq2).unwrap();

        // Both should return valid page IDs
        match &result1 {
            CowResult::NoCowNeeded { page_id } => {
                assert!(*page_id < 8);
            }
            CowResult::CowPerformed { new_page_id, .. } => {
                assert!(*new_page_id < 8);
            }
        }
        match &result2 {
            CowResult::NoCowNeeded { page_id } => {
                assert!(*page_id < 8);
            }
            CowResult::CowPerformed { new_page_id, .. } => {
                assert!(*new_page_id < 8);
            }
        }
    }

    #[test]
    fn test_block_table_for_kernel() {
        let mut manager = PagedKvManager::new(8, 16, 8, 128, 1024);
        let seq_id = manager.create_sequence();

        // Append multiple pages
        let p1 = manager.append_page(seq_id).unwrap();
        let p2 = manager.append_page(seq_id).unwrap();
        let p3 = manager.append_page(seq_id).unwrap();

        // Verify block_table returns correct slice
        let block_table = manager.block_table(seq_id).unwrap();
        assert_eq!(block_table, &[p1, p2, p3]);
        assert_eq!(block_table.len(), 3);
    }
}
