# Phase 6: Continuous Batching

**Duration:** 3 weeks  
**Goal:** Implement continuous batching with dynamic join/leave, round-robin scheduling, and the hybrid KV manager.

## Deliverables

1. Hybrid KV state manager (Mamba + paged)
2. Block allocator with free lists
3. Session lifecycle management
4. Batch builder with dynamic join/leave
5. Round-robin scheduler
6. Request queue with priority
7. Prefill/decode interleaving
8. Session eviction to CPU/SSD (for paged KV only)
9. Memory pressure handling

## Technical Details

### Session Lifecycle

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionState {
    Created,     // Just allocated, not yet running
    Prefilling,  // Processing prompt tokens
    Decoding,    // Generating response tokens
    Paused,      // Temporarily stopped (e.g., waiting for client)
    Evicted,     // KV cache moved to CPU/SSD
    Completed,   // Finished generating
}

pub struct Session {
    pub id: SequenceId,
    pub state: SessionState,
    pub tokens: Vec<u32>,
    pub num_prompt_tokens: usize,
    pub num_generated_tokens: usize,
    pub max_tokens: usize,
    pub sampling_config: SamplingConfig,
    /// Page table mapping this session's tokens to physical pages.
    /// Managed by the global PagedKvManager.
    pub page_table: SequencePageTable,
    pub created_at: Instant,
    pub last_activity: Instant,
    pub priority: i32,  // Higher = more important
}

impl Session {
    pub fn is_active(&self) -> bool {
        matches!(self.state, SessionState::Prefilling | SessionState::Decoding)
    }

    pub fn is_evictable(&self) -> bool {
        // GDN state is small (HxH matrix per layer) — always kept on GPU.
        // Only evict if session has been inactive for a while.
        self.last_activity.elapsed() > Duration::from_secs(30)
    }
}
```

### Paged KV Manager

The paged KV system lives in `infers-kv` and is already built (Phase 4.6).
Key types:

```rust
use infers_kv::{PagedKvManager, SequencePageTable, SequenceId};

// PagedKvManager: orchestration layer wrapping PagePool + PrefixCache + COW
// Already implemented — see crates/kv/src/manager.rs
// API:
//   create_sequence() → SequenceId
//   delete_sequence(seq_id)
//   append_page(seq_id) → PageId
//   ensure_writable(seq_id) → CowResult
//   add_token(seq_id)
//   seal_and_cache(seq_id, layer_idx, model_id, k_data, v_data) → Option<PageId>
//   block_table(seq_id) → &[PageId]
//   num_tokens(seq_id) → usize
//   num_free_pages() → usize

// SequencePageTable: page table for one sequence
// Already implemented — see crates/kv/src/table.rs
// pub struct SequencePageTable {
//     pub page_ids: Vec<PageId>,
//     pub num_tokens: usize,
//     pub page_size: usize,
// }

// PhysicalPage: metadata for one page in the pool
// Already implemented — see crates/kv/src/page.rs
// pub struct PhysicalPage {
//     pub page_id: PageId,
//     pub refcount: AtomicU32,
//     pub state: PageState,
//     pub location: PageLocation,
// }

// The page pool uses a SINGLE interleaved GPU buffer:
//   page_pool[page_id * page_stride + ...]
// with per-page layout: [K tokens | V tokens]
// page_stride = 2 * page_size * kv_dim

// GDN state is managed separately in the native backend:
//   See crates/backends/native/src/gdn.rs → GdnState
// pub struct GdnState {
//     pub state: Option<CudaSlice<bf16>>,  // HxH recurrent state matrix
// }
```

### Block Allocator

Block allocation is handled by `infers_kv::PagedKvManager` (fully implemented
in Phase 4.6). The scheduler wraps it for session management:

```rust
use infers_kv::{PagedKvManager, PageId, SequenceId};
use infers_kv::Sequencer; // Session → sequence binding for continuous batching

impl PagedKvManager {
    /// Allocate blocks for a new session with the given number of prompt tokens.
    /// Returns the session's SequenceId.
    pub fn allocate_session(&mut self, num_prompt_tokens: usize) -> Result<SequenceId> {
        let seq_id = self.create_sequence();

        // Allocate enough pages for the prompt
        let num_pages = (num_prompt_tokens + self.page_size() - 1) / self.page_size();
        for _ in 0..num_pages {
            self.append_page(seq_id)?;
        }

        Ok(seq_id)
    }

    /// Append a token to a session — ensures writable page, increments count.
    /// Returns the block table slice for GPU kernel consumption.
    pub fn append_token_to_session(
        &mut self,
        seq_id: SequenceId,
    ) -> Result<&[PageId]> {
        // Ensure tail page is writable (handles COW if page is shared)
        self.ensure_writable(seq_id)?;

        // Increment token count (may trigger page sealing at boundaries)
        self.add_token(seq_id);

        // Return block table for kernel dispatch
        self.block_table(seq_id)
    }

    /// Free a session's resources — returns pages to pool.
    pub fn free_session_blocks(&mut self, seq_id: SequenceId) {
        let _ = self.delete_sequence(seq_id);
    }
}
```
        
        Ok(())
    }
}
```

### Eviction to CPU/SSD

CPU eviction is **not yet implemented**. The `infers_kv::PageLocation::Cpu`
variant is reserved for this but requires:

1. GPU-side page copy kernels (`cudaMemcpyAsync` from page pool → pinned CPU buffer)
2. CPU page pool management (allocating pinned host memory for evicted pages)
3. LRU session tracking (which sessions haven't been accessed recently)
4. Restoration logic (copy evicted pages back to GPU on cache hit)

The `PagedKvManager::PrefixCache` already handles LRU eviction of **shared
sealed pages** (when prefix cache memory budget is exceeded). Session-level
eviction is a future addition:

```rust
// Conceptual approach (not yet implemented):
// Evict LRU sessions by copying their page data to CPU pinned memory.
// GDN states (H×H matrices) stay on GPU — only paged KV pages are evicted.

// 1. Select LRU session using last_activity timestamps
// 2. For each page in the session's page table:
//    a. Launch cudaMemcpyAsync to copy page_pool[...] → pinned_cpu_buffer
//    b. Record page → CPU address mapping in eviction table
// 3. Free the GPU pages via PagedKvManager::delete_sequence
// 4. Mark session as SessionState::Evicted
//
// Restoration happens on cache hit: reverse the copy and re-register
// pages with the PagedKvManager.
```

### Batch Builder

```rust
pub struct BatchBuilder {
    pub max_batch_size: usize,
    pub max_tokens_per_batch: usize,
}

pub struct DecodeBatch {
    pub sessions: Vec<SequenceId>,
    pub input_tokens: Vec<u32>,
    /// GPU block table slices for each session's paged KV.
    /// A single contiguous slice per session for paged kernel dispatch.
    pub block_tables: Vec<Vec<PageId>>,
}

impl BatchBuilder {
    pub fn build_decode_batch(
        &self,
        active_sessions: &mut Vec<Session>,
        kv_manager: &PagedKvManager,
    ) -> Result<DecodeBatch> {
        let mut batch = DecodeBatch {
            sessions: Vec::new(),
            input_tokens: Vec::new(),
            block_tables: Vec::new(),
        };

        for session in active_sessions.iter_mut() {
            if !session.is_active() {
                continue;
            }

            if batch.sessions.len() >= self.max_batch_size {
                break;
            }

            // Get the block table from PagedKvManager for this sequence
            let block_table = kv_manager.block_table(session.id)?.to_vec();
            let latest_token = session.tokens.last()
                .copied()
                .ok_or_else(|| anyhow!("Session has no tokens"))?;

            batch.sessions.push(session.id);
            batch.input_tokens.push(latest_token);
            batch.block_tables.push(block_table);
        }

        Ok(batch)
    }

    pub fn build_prefill_batch(
        &self,
        pending_sessions: &mut Vec<Session>,
        kv_manager: &PagedKvManager,
    ) -> Result<DecodeBatch> {
        if let Some(session) = pending_sessions.first_mut() {
            if session.state == SessionState::Created {
                session.state = SessionState::Prefilling;

                let prompt_tokens = session.tokens.clone();
                let block_table = kv_manager.block_table(session.id)
                    .map(|bt| bt.to_vec())
                    .unwrap_or_default();

                Ok(DecodeBatch {
                    sessions: vec![session.id],
                    input_tokens: prompt_tokens,
                    block_tables: vec![block_table],
                })
            } else {
                Err(anyhow!("No sessions to prefill"))
            }
        } else {
            Err(anyhow!("No pending sessions"))
        }
    }
}
```

### Round-Robin Scheduler

```rust
pub struct RoundRobinScheduler {
    pub request_queue: VecDeque<Request>,
    pub active_sessions: Vec<Session>,
    pub max_concurrent_sessions: usize,
    pub batch_builder: BatchBuilder,
    pub kv_manager: PagedKvManager,
}

impl RoundRobinScheduler {
    pub fn schedule(&mut self) -> Result<ScheduledWork> {
        // 1. Admit new requests
        while self.active_sessions.len() < self.max_concurrent_sessions {
            if let Some(request) = self.request_queue.pop_front() {
                let session = self.create_session(request)?;
                self.active_sessions.push(session);
            } else {
                break;
            }
        }

        // 2. Check for completed sessions
        self.active_sessions.retain(|s| {
            if s.state == SessionState::Completed {
                self.kv_manager.delete_sequence(s.id).ok();
                false
            } else {
                true
            }
        });

        // 3. Build decode batch (highest priority)
        let decode_batch = self.batch_builder
            .build_decode_batch(&mut self.active_sessions, &self.kv_manager)?;

        // 4. If decode batch is small, try to prefill new sessions
        let prefill_batch = if decode_batch.sessions.len() < self.batch_builder.max_batch_size / 2 {
            self.batch_builder
                .build_prefill_batch(&mut self.active_sessions, &self.kv_manager)
                .ok()
        } else {
            None
        };

        Ok(ScheduledWork {
            decode_batch,
            prefill_batch,
        })
    }

    pub fn create_session(&mut self, request: Request) -> Result<Session> {
        let id = SequenceId::default();
        let tokens = self.tokenizer.encode(&request.prompt)?;

        // Allocate KV pages via PagedKvManager
        let _ = self.kv_manager.allocate_session(tokens.len())?;

        Ok(Session {
            id,
            state: SessionState::Created,
            tokens,
            num_prompt_tokens: tokens.len(),
            num_generated_tokens: 0,
            max_tokens: request.max_tokens.unwrap_or(512),
            sampling_config: request.sampling_config,
            page_table: SequencePageTable::new(self.page_size),
            created_at: Instant::now(),
            last_activity: Instant::now(),
            priority: request.priority,
        })
    }
}
```

## File Structure

```
crates/scheduler/          # NEW: session lifecycle, batch builder, scheduler
  Cargo.toml
  src/
    lib.rs
    session.rs           # Session, SessionState, SequenceId
    lifecycle.rs         # Session lifecycle transitions
    batch.rs             # BatchBuilder, DecodeBatch
    scheduler.rs         # RoundRobinScheduler
    queue.rs             # RequestQueue

crates/kv/                # ALREADY EXISTS (Phase 4.6)
  Cargo.toml
  src/
    lib.rs
    page.rs              # PhysicalPage, PageId, PageState, PageLocation
    pool.rs              # PagePool with O(1) free-list alloc/free
    table.rs             # SequencePageTable
    prefix.rs            # PrefixCache with Blake3 hashing, LRU eviction
    cow.rs               # Copy-on-write page sharing
    manager.rs           # PagedKvManager — orchestration layer

# Note: GDN state is in crates/backends/native/src/gdn.rs (GdnState),
# not in crates/kv. Eviction to CPU/SSD is not yet implemented — see
# the "Eviction" section above for the deferred design.
```

## Testing

### Batch Builder Test

```rust
#[test]
fn test_batch_builder() {
    let mut kv_manager = PagedKvManager::new(
        500,    // total_pages
        16,     // page_size (tokens per page)
        4,      // num_kv_heads
        256,    // head_dim
        1024 * 1024 * 1024, // max_cache_bytes
    );

    let mut sessions = vec![
        Session {
            id: kv_manager.create_sequence(),
            state: SessionState::Decoding,
            tokens: vec![1, 2, 3, 4, 5],
            num_prompt_tokens: 5,
            num_generated_tokens: 0,
            max_tokens: 100,
            sampling_config: SamplingConfig::default(),
            page_table: SequencePageTable::new(16),
            created_at: Instant::now(),
            last_activity: Instant::now(),
            priority: 0,
        },
        // ... second session ...
    ];

    let builder = BatchBuilder {
        max_batch_size: 2,
        max_tokens_per_batch: 100,
    };

    let batch = builder.build_decode_batch(&mut sessions, &kv_manager).unwrap();
    assert_eq!(batch.sessions.len(), 2);
}
```

### PagedKvManager Integration Test

**Note:** CPU eviction is not yet implemented (see "Eviction to CPU/SSD" section).
Test the PagedKvManager's existing page lifecycle and prefix cache instead:

```rust
#[test]
fn test_page_lifecycle_with_sessions() {
    let mut kv = PagedKvManager::new(
        100,    // total_pages
        16,     // page_size
        4,      // num_kv_heads
        256,    // head_dim
        1024 * 1024 * 1024, // max_cache_bytes
    );

    // Create 3 sequences
    let id1 = kv.create_sequence();
    let id2 = kv.create_sequence();
    let id3 = kv.create_sequence();

    // Allocate pages for id1
    for _ in 0..3 {
        kv.append_page(id1).unwrap();
    }
    for _ in 0..2 {
        kv.append_page(id2).unwrap();
    }

    // Verify usage
    assert_eq!(kv.num_pages(id1).unwrap(), 3);
    assert_eq!(kv.num_pages(id2).unwrap(), 2);
    assert!(kv.num_pages(id3).unwrap() == 0);

    // Verify pool reflects allocated pages
    let free_before = kv.num_free_pages();
    kv.delete_sequence(id1).unwrap();
    let free_after = kv.num_free_pages();
    assert_eq!(free_after, free_before + 3);
}
```
```

## Dependencies

### Phase 6 → Phase 4

Uses forward pass to execute batches.

### Phase 6 → Phase 5

Works with both TP and PP modes.

### Phase 6 → Phase 3

Uses memory budget from model loading.

## Success Criteria

1. Can handle 3+ concurrent sessions with 262K context each
2. Sessions can join and leave batches dynamically
3. KV cache blocks are reused efficiently
4. Eviction to CPU works for paged KV (not Mamba)
5. Round-robin gives fair token distribution
6. Memory usage stays within budget
7. No memory leaks over 24-hour test

## Cross-References

- **Research:** See `../research/architecture.md` for GDN vs KV cache details
- **Phase 3:** Memory budget calculator informs max concurrent sessions
- **Phase 4:** Forward pass executes built batches
- **Phase 5:** Scheduler works with both TP and PP

## Open Questions

1. Should we implement priority scheduling (e.g., interactive > background)?
2. How to handle preemption (pause a session to free KV cache)?
3. Should we batch prefill and decode together (mixed batch)?
