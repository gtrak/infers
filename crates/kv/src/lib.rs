//! Paged KV cache subsystem for attention inference.
//!
//! Implements a vLLM-inspired paged attention design with:
//! - Physical page pool with O(1) allocate/free
//! - Sequence page tables mapping logical positions to physical pages
//! - Prefix caching with Blake3 content hashing and LRU eviction
//! - Copy-on-write page sharing for branching prompts

pub mod cow;
pub mod eviction;
pub mod manager;
pub mod page;
pub mod pool;
pub mod prefix;
pub mod quant;
pub use quant::{KvCacheDtype, QuantizedKvCache};
pub mod table;

pub use cow::{ensure_mutable_page, CowError, CowResult};
pub use manager::{ManagerError, PagedKvManager, SequenceId};
pub use page::{INVALID_PAGE_ID, PageId, PageLocation, PageState, PhysicalPage};
pub use pool::{PagePool, PagePoolError};
pub use prefix::{CacheEntry, PageHash, PrefixCache};
pub use eviction::{CpuPagePool, EvictedSequence, EvictionError};
pub use table::SequencePageTable;
