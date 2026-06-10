//! Per-layer evicted page data store for the GPU backend.
//!
//! The `CpuPagePool` in `infers-kv` stores one blob per `PageId`, but each
//! page's data is actually per-layer (each full-attention layer has its own
//! K/V values for the same page). The `BackendEvictionStore` manages this
//! multi-layer aspect, storing `Vec<u8>` per (layer, page_id) pair.

use std::collections::HashMap;

use half::bf16;
use infers_kv::PageId;

/// Per-layer storage for evicted KV page data.
///
/// Each layer has its own `HashMap<PageId, Vec<u8>>` mapping page IDs to
/// their raw K/V data bytes. Pages from all layers for the same token
/// range are stored separately.
// @lat: [[lat.md/lat#Phase 4.6 Deliverables#Paged Attention Implementation#Backend Eviction Store]]
#[derive(Debug)]
pub struct BackendEvictionStore {
    /// One map per layer: PageId → raw page data bytes.
    layers: Vec<HashMap<PageId, Vec<u8>>>,
}

impl BackendEvictionStore {
    /// Create a new eviction store for the given number of layers.
    ///
    /// # Arguments
    /// * `num_layers` — Number of full-attention layers (NOT total hidden layers).
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| HashMap::new()).collect(),
        }
    }

    /// Store page data for a specific layer.
    pub fn store(&mut self, layer: usize, page_id: PageId, data: Vec<u8>) {
        debug_assert!(layer < self.layers.len(), "Layer {layer} out of range (max {})", self.layers.len());
        if let Some(map) = self.layers.get_mut(layer) {
            map.insert(page_id, data);
        }
    }

    /// Retrieve and remove page data for a specific layer.
    pub fn retrieve(&mut self, layer: usize, page_id: PageId) -> Option<Vec<u8>> {
        self.layers.get_mut(layer).and_then(|map| map.remove(&page_id))
    }

    /// Check if page data exists for a specific layer.
    pub fn contains(&self, layer: usize, page_id: PageId) -> bool {
        self.layers.get(layer).is_some_and(|map| map.contains_key(&page_id))
    }

    /// Number of layers being tracked.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total number of evicted pages across all layers.
    pub fn total_pages(&self) -> usize {
        self.layers.iter().map(|m| m.len()).sum()
    }

    /// Remove all page data for a given page_id across all layers.
    pub fn remove_page(&mut self, page_id: PageId) {
        for map in &mut self.layers {
            map.remove(&page_id);
        }
    }

    /// Clear all evicted data.
    pub fn clear(&mut self) {
        for map in &mut self.layers {
            map.clear();
        }
    }

    /// Convert a slice of bf16 GPU data to raw bytes.
    pub fn bf16_slice_to_bytes(data: &[bf16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &val in data {
            bytes.extend_from_slice(&val.to_bits().to_le_bytes());
        }
        bytes
    }

    /// Convert raw bytes back to a Vec of bf16 values.
    pub fn bytes_to_bf16_slice(data: &[u8]) -> Vec<bf16> {
        data.chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> BackendEvictionStore {
        BackendEvictionStore::new(3) // 3 layers
    }

    #[test]
    fn test_store_empty_initially() {
        let store = make_store();
        assert_eq!(store.num_layers(), 3);
        assert_eq!(store.total_pages(), 0);
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut store = make_store();
        let data = vec![42u8; 1024];

        store.store(0, 5, data.clone());
        assert!(store.contains(0, 5));
        assert_eq!(store.total_pages(), 1);

        let retrieved = store.retrieve(0, 5).unwrap();
        assert_eq!(retrieved.len(), 1024);
        assert_eq!(retrieved[0], 42);
        assert!(!store.contains(0, 5)); // removed after retrieve
    }

    #[test]
    fn test_per_layer_isolation() {
        let mut store = make_store();
        store.store(0, 1, vec![0u8; 1024]);
        store.store(1, 1, vec![1u8; 1024]);
        store.store(2, 1, vec![2u8; 1024]);

        assert_eq!(store.total_pages(), 3);
        assert_eq!(store.retrieve(1, 1).unwrap()[0], 1);
        assert_eq!(store.total_pages(), 2);
    }

    #[test]
    fn test_retrieve_nonexistent() {
        let mut store = make_store();
        assert!(store.retrieve(0, 999).is_none());
    }

    #[test]
    fn test_store_overwrites() {
        let mut store = make_store();
        store.store(0, 1, vec![0u8; 1024]);
        store.store(0, 1, vec![1u8; 1024]); // overwrite
        assert_eq!(store.total_pages(), 1);
        assert_eq!(store.retrieve(0, 1).unwrap()[0], 1);
    }

    #[test]
    fn test_remove_page() {
        let mut store = make_store();
        store.store(0, 1, vec![0u8; 1024]);
        store.store(1, 1, vec![1u8; 1024]);
        assert_eq!(store.total_pages(), 2);

        store.remove_page(1);
        assert_eq!(store.total_pages(), 0);
    }

    #[test]
    fn test_clear() {
        let mut store = make_store();
        store.store(0, 1, vec![0u8; 1024]);
        store.store(1, 2, vec![1u8; 1024]);
        store.store(2, 3, vec![2u8; 1024]);
        assert_eq!(store.total_pages(), 3);

        store.clear();
        assert_eq!(store.total_pages(), 0);
    }

    #[test]
    fn test_bf16_bytes_round_trip() {
        use half::bf16;

        let original = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(-2.5),
            bf16::from_f32(3.1415),
            bf16::ZERO,
            bf16::NEG_ONE,
        ];

        let bytes = BackendEvictionStore::bf16_slice_to_bytes(&original);
        assert_eq!(bytes.len(), original.len() * 2);

        let restored = BackendEvictionStore::bytes_to_bf16_slice(&bytes);
        assert_eq!(restored.len(), original.len());

        for (a, b) in original.iter().zip(restored.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }


}
