//! LRU-based prefix cache for sharing sealed KV cache pages across sequences.
//!
//! When a sequence writes tokens into its tail page and the page becomes full
//! (sealed), its content can be hashed. If another sequence produces the same
//! content hash, the existing cached page is reused instead of allocating a new one.
//!
//! The cache has a memory budget. When the budget is exceeded, the least recently
//! used entries are evicted.

use std::collections::HashMap;

use super::page::PageId;

/// A 256-bit content hash identifying the contents of a sealed page.
/// Used to look up identical prefix pages across sequences.
pub type PageHash = [u8; 32];

/// Entry in the prefix cache mapping a content hash to a physical page.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The physical page ID holding this content.
    pub page_id: PageId,
    /// Number of sequences currently referencing this cached entry.
    pub refcount: u32,
}

/// LRU-based prefix cache for sharing sealed KV cache pages across sequences.
///
/// When a sequence writes tokens into its tail page and the page becomes full
/// (sealed), its content can be hashed. If another sequence produces the same
/// content hash, the existing cached page is reused instead of allocating a new one.
///
/// The cache has a memory budget. When the budget is exceeded, the least recently
/// used entries are evicted.
pub struct PrefixCache {
    /// Map from content hash → cache entry.
    map: HashMap<PageHash, CacheEntry>,
    /// Ordered list of hashes for LRU eviction (most recent at the back).
    lru_order: Vec<PageHash>,
    /// Maximum memory budget in bytes.
    max_memory_bytes: usize,
    /// Current memory usage in bytes (sum of all cached pages).
    current_memory_bytes: usize,
    /// Bytes per page (for budget tracking).
    page_bytes: usize,
}

impl PrefixCache {
    /// Create a new prefix cache with the given memory budget.
    ///
    /// # Arguments
    /// * `max_memory_bytes` — Maximum memory budget in bytes.
    /// * `page_bytes` — Size of a single page in bytes (for budget tracking).
    pub fn new(max_memory_bytes: usize, page_bytes: usize) -> Self {
        assert!(page_bytes > 0, "page_bytes must be > 0");
        Self {
            map: HashMap::new(),
            lru_order: Vec::new(),
            max_memory_bytes,
            current_memory_bytes: 0,
            page_bytes,
        }
    }

    /// Insert a hash → page mapping into the prefix cache.
    ///
    /// Returns `true` if this is a new entry, `false` if the hash already exists
    /// (in which case the existing entry's refcount is incremented).
    ///
    /// Updates LRU order: if the hash already exists, it is moved to the
    /// most-recent position. For a new entry, it is appended as most-recent.
    ///
    /// Does **not** check the memory budget — call [`evict_if_needed`][Self::evict_if_needed] separately.
    pub fn insert(&mut self, hash: PageHash, page_id: PageId) -> bool {
        if let Some(entry) = self.map.get_mut(&hash) {
            // Hash already exists — increment refcount and update LRU order
            entry.refcount += 1;
            self.touch_internal(&hash);
            return false;
        }

        // New entry — add to map and LRU order
        self.map.insert(hash, CacheEntry { page_id, refcount: 1 });
        self.lru_order.push(hash);
        self.current_memory_bytes += self.page_bytes;
        true
    }

    /// Look up a hash and return the page ID if found.
    ///
    /// Does **not** update LRU order — use [`touch`][Self::touch] for that.
    pub fn lookup(&self, hash: &PageHash) -> Option<PageId> {
        self.map.get(hash).map(|entry| entry.page_id)
    }

    /// Touch a hash, moving it to the most-recent position in LRU order.
    ///
    /// Used when a sequence reads from a cached page to mark it as recently used.
    /// Has no effect if the hash is not in the cache.
    pub fn touch(&mut self, hash: &PageHash) {
        if self.map.contains_key(hash) {
            self.touch_internal(hash);
        }
    }

    /// Release a reference to a cached entry.
    ///
    /// Decrements the refcount for the hash. If the refcount drops to 0, returns
    /// the page_id so the caller knows the page is no longer referenced.
    /// The entry stays in the cache until [`evict_if_needed`][Self::evict_if_needed]
    /// removes it during memory pressure.
    ///
    /// Returns `None` if the hash is not found or the refcount is still > 0 after decrement.
    pub fn release(&mut self, hash: &PageHash) -> Option<PageId> {
        let entry = self.map.get_mut(hash)?;
        entry.refcount = entry.refcount.saturating_sub(1);
        if entry.refcount == 0 {
            Some(entry.page_id)
        } else {
            None
        }
    }

    /// Evict least-recently-used entries until the memory budget is satisfied.
    ///
    /// Skips entries with refcount > 0 (in-use pages cannot be evicted).
    ///
    /// Returns the list of page IDs that were evicted so the caller can free
    /// them from the page pool.
    pub fn evict_if_needed(&mut self) -> Vec<PageId> {
        let mut evicted = Vec::new();

        while self.current_memory_bytes > self.max_memory_bytes {
            // Find the oldest entry with refcount == 0 (safe to evict)
            let mut found_index = None;
            for (idx, hash) in self.lru_order.iter().enumerate() {
                if let Some(entry) = self.map.get(hash)
                    && entry.refcount == 0
                {
                    found_index = Some(idx);
                    break;
                }
            }

            match found_index {
                Some(idx) => {
                    let hash = self.lru_order.remove(idx);
                    let page_id = self.map.remove(&hash).unwrap().page_id;
                    self.current_memory_bytes = self.current_memory_bytes.saturating_sub(self.page_bytes);
                    evicted.push(page_id);
                }
                None => {
                    // No entries with refcount == 0 available — stop evicting
                    break;
                }
            }
        }

        evicted
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Current memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.current_memory_bytes
    }

    /// Returns `true` if the cache has no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Internal helper: move a hash to the most-recent position in LRU order.
    fn touch_internal(&mut self, hash: &PageHash) {
        if let Some(pos) = self.lru_order.iter().position(|h| h == hash) {
            self.lru_order.remove(pos);
            self.lru_order.push(*hash);
        }
    }
}

/// Compute a Blake3 content hash for a sealed page.
///
/// The hash uniquely identifies page content across models and layers.
/// Derives a 32-byte key from `model_id` using `blake3::hash`, then uses
/// `blake3::Hasher::new_keyed` to hash `k_data || v_data || layer_idx.to_le_bytes()`.
///
/// NOTE: Since this crate doesn't have GPU access, the data passed in is
/// raw bytes (the caller copies page data from GPU to host before hashing).
/// This is a CPU-side operation.
///
/// # Arguments
/// * `k_data` — Raw K data bytes for the page.
/// * `v_data` — Raw V data bytes for the page.
/// * `model_id` — Model identifier string (used to derive the Blake3 key).
/// * `layer_idx` — Transformer layer index.
pub fn hash_page(k_data: &[u8], v_data: &[u8], model_id: &str, layer_idx: usize) -> PageHash {
    // Derive a 32-byte key from model_id using a standard blake3 hash
    let key_bytes: [u8; 32] = *blake3::hash(model_id.as_bytes()).as_bytes();
    let mut hasher = blake3::Hasher::new_keyed(&key_bytes);
    hasher.update(k_data);
    hasher.update(v_data);
    hasher.update(&layer_idx.to_le_bytes());
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(result.as_bytes());
    hash
}
#[cfg(test)]
mod tests {
    use super::*;

    fn make_hash(i: u8) -> PageHash {
        let mut h = [0u8; 32];
        h[0] = i;
        h
    }

    #[test]
    fn test_insert_new() {
        let mut cache = PrefixCache::new(1024, 256);
        let hash = make_hash(1);
        assert!(cache.insert(hash, 42));
        assert_eq!(cache.lookup(&hash), Some(42));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.memory_usage(), 256);
    }

    #[test]
    fn test_insert_duplicate_increments_refcount() {
        let mut cache = PrefixCache::new(1024, 256);
        let hash = make_hash(1);
        assert!(cache.insert(hash, 42));
        assert!(!cache.insert(hash, 99)); // Duplicate — should not replace page_id
        assert_eq!(cache.lookup(&hash), Some(42)); // Original page_id preserved
        assert_eq!(cache.len(), 1); // Still one entry
        assert_eq!(cache.memory_usage(), 256); // Memory unchanged
        // Verify refcount is 2
        assert_eq!(cache.map[&hash].refcount, 2);
    }

    #[test]
    fn test_lookup_miss() {
        let cache = PrefixCache::new(1024, 256);
        let hash = make_hash(99);
        assert_eq!(cache.lookup(&hash), None);
    }

    #[test]
    fn test_touch_updates_lru() {
        // Budget: 256 bytes — only 1 page fits, forcing eviction
        let mut cache = PrefixCache::new(256, 256);
        let h1 = make_hash(1);
        let h2 = make_hash(2);
        cache.insert(h1, 10);
        cache.insert(h2, 20); // usage = 512 > budget
        // LRU order is [h1, h2] — h1 is oldest
        cache.touch(&h1);
        // LRU order is now [h2, h1] — h2 is oldest
        // Verify by releasing h2 (refcount 0), then evicting
        cache.release(&h2); // h2 refcount = 0, still in map
        cache.evict_if_needed(); // Should evict h2 (oldest with refcount 0)
        assert_eq!(cache.lookup(&h2), None); // h2 evicted
        assert_eq!(cache.lookup(&h1), Some(10)); // h1 still cached (refcount 1)
    }

    #[test]
    fn test_touch_then_evict() {
        // Budget allows only 1 page (256 bytes)
        let mut cache = PrefixCache::new(256, 256);
        let h1 = make_hash(1);
        let h2 = make_hash(2);
        let h3 = make_hash(3);
        cache.insert(h1, 10); // usage = 256, within budget
        cache.insert(h2, 20); // usage = 512, over budget
        // Touch h1 so it's most recent. LRU order: [h2, h1]
        cache.touch(&h1);
        // Insert h3 — usage = 768, still over budget
        cache.insert(h3, 30);
        // Release h2 so its refcount drops to 0 (safe to evict)
        cache.release(&h2);
        // Evict — should evict h2 (oldest with refcount 0)
        let evicted = cache.evict_if_needed();
        assert_eq!(evicted, vec![20]);
        assert_eq!(cache.memory_usage(), 512); // h1 + h3 remain
    }

    #[test]
    fn test_release_decrements() {
        let mut cache = PrefixCache::new(1024, 256);
        let hash = make_hash(1);
        cache.insert(hash, 42);
        cache.insert(hash, 42); // refcount = 2
        // Release once — refcount drops to 1, entry stays cached
        let result = cache.release(&hash);
        assert_eq!(result, None); // Not removed (refcount still 1)
        assert_eq!(cache.lookup(&hash), Some(42));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_release_returns_page_id_when_zero() {
        // Budget: 0 bytes — forces eviction on any insert
        let mut cache = PrefixCache::new(0, 256);
        let hash = make_hash(1);
        cache.insert(hash, 42); // refcount = 1, usage = 256 > budget
        let result = cache.release(&hash);
        assert_eq!(result, Some(42)); // page_id returned
        // Entry still in cache with refcount 0, awaiting eviction
        assert_eq!(cache.lookup(&hash), Some(42));
        assert_eq!(cache.map[&hash].refcount, 0);
        // Evict cleans up the entry
        cache.evict_if_needed();
        assert_eq!(cache.lookup(&hash), None);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn test_evict_removes_lru() {
        // Budget: 256 bytes (fits exactly 1 page)
        let mut cache = PrefixCache::new(256, 256);
        let h1 = make_hash(1);
        let h2 = make_hash(2);
        let h3 = make_hash(3);

        cache.insert(h1, 10); // usage = 256, OK
        // Release h1 so it's eligible for eviction
        cache.release(&h1); // refcount = 0, entry still in map

        cache.insert(h2, 20); // usage = 512, over budget
        cache.insert(h3, 30); // usage = 768, over budget
        // Release h2 and h3 so they're eligible
        cache.release(&h2);
        cache.release(&h3);

        // Evict — should remove h1 (oldest, refcount 0) then h2 (next oldest)
        let evicted = cache.evict_if_needed();
        assert_eq!(evicted, vec![10, 20]);
        assert_eq!(cache.memory_usage(), 256); // Only h3 remains
        assert_eq!(cache.lookup(&h3), Some(30));
    }

    #[test]
    fn test_evict_skips_in_use() {
        // Budget: 256 bytes (fits exactly 1 page)
        let mut cache = PrefixCache::new(256, 256);
        let h1 = make_hash(1);
        let h2 = make_hash(2);

        // Insert h1, keep refcount = 1 (in use — cannot be evicted)
        cache.insert(h1, 10);
        // Insert h2, release immediately so refcount = 0
        cache.insert(h2, 20);
        cache.release(&h2);

        // Evict — should evict h2 (refcount 0), skip h1 (refcount 1)
        let evicted = cache.evict_if_needed();
        assert_eq!(evicted, vec![20]);
        assert_eq!(cache.lookup(&h1), Some(10)); // h1 still cached
        assert_eq!(cache.lookup(&h2), None); // h2 removed
        assert_eq!(cache.memory_usage(), 256);
    }

    #[test]
    fn test_hash_page_deterministic() {
        let k = [1u8; 16];
        let v = [2u8; 16];
        let h1 = hash_page(&k, &v, "model-a", 0);
        let h2 = hash_page(&k, &v, "model-a", 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_page_different_layers() {
        let k = [1u8; 16];
        let v = [2u8; 16];
        let h0 = hash_page(&k, &v, "model-a", 0);
        let h1 = hash_page(&k, &v, "model-a", 1);
        assert_ne!(h0, h1);
    }

    #[test]
    fn test_is_empty() {
        let cache = PrefixCache::new(1024, 256);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_release_unknown_hash() {
        let mut cache = PrefixCache::new(1024, 256);
        let hash = make_hash(99);
        assert_eq!(cache.release(&hash), None);
    }
}
