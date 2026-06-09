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
    pub id: SessionId,
    pub state: SessionState,
    pub tokens: Vec<u32>,
    pub num_prompt_tokens: usize,
    pub num_generated_tokens: usize,
    pub max_tokens: usize,
    pub sampling_config: SamplingConfig,
    pub kv_cache: SessionKvCache,
    pub created_at: Instant,
    pub last_activity: Instant,
    pub priority: i32,  // Higher = more important
}

impl Session {
    pub fn is_active(&self) -> bool {
        matches!(self.state, SessionState::Prefilling | SessionState::Decoding)
    }
    
    pub fn is_evictable(&self) -> bool {
        // Mamba state cannot be evicted easily
        // Only evict if session has been inactive for a while
        self.last_activity.elapsed() > Duration::from_secs(30)
    }
}
```

### Hybrid KV Manager

```rust
pub struct HybridKvManager {
    // Configuration
    pub block_size: usize,           // 16 tokens per block
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    
    // Paged KV cache (for full attention layers)
    pub paged_k: DeviceBuffer<half>,   // [num_pages, page_size, num_kv_heads, head_dim]
    pub paged_v: DeviceBuffer<half>,   // same shape
    pub block_table: Vec<BlockTable>,  // per-session
    pub free_gpu_blocks: Vec<u32>,
    pub free_cpu_blocks: Vec<u32>,
    pub cpu_buffer: PinnedBuffer<half>, // CPU staging for eviction
    
    // Mamba states (for GDN layers)
    pub mamba_states: HashMap<SessionId, Vec<MambaState>>, // 48 layers per session
    
    // Memory tracking
    pub total_gpu_pages: usize,
    pub used_gpu_pages: usize,
    pub total_cpu_pages: usize,
    pub used_cpu_pages: usize,
}

pub struct BlockTable {
    pub session_id: SessionId,
    pub physical_blocks: Vec<u32>,  // Maps logical block idx → physical block id
    pub num_tokens: usize,          // How many tokens valid in last block
}

pub struct MambaState {
    pub conv_state: DeviceBuffer<f32>,
    pub ssm_state: DeviceBuffer<f32>,
}
```

### Block Allocator

```rust
impl HybridKvManager {
    pub fn allocate_blocks(
        &mut self,
        session_id: SessionId,
        num_tokens: usize,
    ) -> Result<Vec<u32>> {
        let num_blocks = (num_tokens + self.block_size - 1) / self.block_size;
        
        if self.free_gpu_blocks.len() < num_blocks {
            // Try to evict least recently used sessions
            self.evict_lru_sessions(num_blocks - self.free_gpu_blocks.len())?;
        }
        
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(self.free_gpu_blocks.pop()
                .ok_or_else(|| anyhow!("Out of GPU KV cache memory"))?);
        }
        
        self.block_table.push(BlockTable {
            session_id,
            physical_blocks: blocks.clone(),
            num_tokens,
        });
        
        self.used_gpu_pages += num_blocks;
        
        Ok(blocks)
    }
    
    pub fn append_token(&mut self, session_id: SessionId) -> Result<u32> {
        let table = self.block_table
            .iter_mut()
            .find(|t| t.session_id == session_id)
            .ok_or_else(|| anyhow!("Session not found"))?;
        
        table.num_tokens += 1;
        
        // Check if we need a new block
        if table.num_tokens > table.physical_blocks.len() * self.block_size {
            let new_block = self.free_gpu_blocks.pop()
                .ok_or_else(|| anyhow!("Out of blocks"))?;
            table.physical_blocks.push(new_block);
            self.used_gpu_pages += 1;
        }
        
        Ok(*table.physical_blocks.last().unwrap())
    }
    
    pub fn free_session(&mut self, session_id: SessionId) -> Result<()> {
        let idx = self.block_table
            .iter()
            .position(|t| t.session_id == session_id)
            .ok_or_else(|| anyhow!("Session not found"))?;
        
        let table = self.block_table.remove(idx);
        
        // Return blocks to free list
        for block in &table.physical_blocks {
            self.free_gpu_blocks.push(*block);
        }
        
        self.used_gpu_pages -= table.physical_blocks.len();
        
        // Free Mamba states
        self.mamba_states.remove(&session_id);
        
        Ok(())
    }
}
```

### Eviction to CPU/SSD

```rust
impl HybridKvManager {
    pub fn evict_lru_sessions(&mut self, num_blocks_needed: usize) -> Result<()> {
        let mut lru_sessions: Vec<_> = self.block_table
            .iter()
            .filter(|t| t.session_id != SessionId::current())  // Don't evict active
            .map(|t| (t.session_id, self.get_last_access(t.session_id)))
            .collect();
        
        lru_sessions.sort_by(|a, b| a.1.cmp(&b.1));
        
        let mut freed = 0;
        for (session_id, _) in lru_sessions {
            if freed >= num_blocks_needed {
                break;
            }
            
            freed += self.evict_session_to_cpu(session_id)?;
        }
        
        Ok(())
    }
    
    fn evict_session_to_cpu(&mut self, session_id: SessionId) -> Result<usize> {
        let table = self.block_table
            .iter_mut()
            .find(|t| t.session_id == session_id)
            .ok_or_else(|| anyhow!("Session not found"))?;
        
        let num_blocks = table.physical_blocks.len();
        
        // Copy KV blocks to CPU pinned memory
        for (i, block_id) in table.physical_blocks.iter().enumerate() {
            let gpu_offset = *block_id as usize * self.block_size * self.num_kv_heads * self.head_dim;
            let cpu_offset = i * self.block_size * self.num_kv_heads * self.head_dim;
            
            // Copy K
            self.paged_k.copy_to_cpu(
                gpu_offset..gpu_offset + self.block_size * self.num_kv_heads * self.head_dim,
                &mut self.cpu_buffer,
                cpu_offset,
            )?;
            
            // Copy V
            self.paged_v.copy_to_cpu(
                gpu_offset..gpu_offset + self.block_size * self.num_kv_heads * self.head_dim,
                &mut self.cpu_buffer,
                cpu_offset + self.total_cpu_pages * self.block_size * self.num_kv_heads * self.head_dim,
            )?;
        }
        
        // Mark blocks as free (but keep table for restoration)
        for block_id in &table.physical_blocks {
            self.free_gpu_blocks.push(*block_id);
        }
        
        self.used_gpu_pages -= num_blocks;
        self.used_cpu_pages += num_blocks;
        
        // Note: Mamba states are NOT evicted (too complex)
        // For now, sessions with Mamba state cannot be evicted
        
        Ok(num_blocks)
    }
}
```

### Batch Builder

```rust
pub struct BatchBuilder {
    pub max_batch_size: usize,
    pub max_tokens_per_batch: usize,
}

pub struct DecodeBatch {
    pub sessions: Vec<SessionId>,
    pub input_tokens: Vec<u32>,
    pub block_tables: Vec<BlockTable>,
    pub mamba_states: Vec<Vec<MambaState>>,
}

impl BatchBuilder {
    pub fn build_decode_batch(
        &self,
        active_sessions: &mut Vec<Session>,
    ) -> Result<DecodeBatch> {
        let mut batch = DecodeBatch {
            sessions: Vec::new(),
            input_tokens: Vec::new(),
            block_tables: Vec::new(),
            mamba_states: Vec::new(),
        };
        
        let mut total_tokens = 0;
        
        for session in active_sessions.iter_mut() {
            if !session.is_active() {
                continue;
            }
            
            if batch.sessions.len() >= self.max_batch_size {
                break;
            }
            
            if total_tokens + session.num_generated_tokens >= self.max_tokens_per_batch {
                break;
            }
            
            // Get next token from previous generation
            let next_token = session.get_last_generated_token()
                .ok_or_else(|| anyhow!("No generated token"))?;
            
            batch.sessions.push(session.id);
            batch.input_tokens.push(next_token);
            batch.block_tables.push(session.kv_cache.block_table.clone());
            batch.mamba_states.push(session.kv_cache.mamba_states.clone());
            
            total_tokens += 1;
        }
        
        Ok(batch)
    }
    
    pub fn build_prefill_batch(
        &self,
        pending_sessions: &mut Vec<Session>,
    ) -> Result<DecodeBatch> {
        // Similar to decode batch, but with prompt tokens
        // For now, prefill one session at a time (simpler)
        
        if let Some(session) = pending_sessions.first_mut() {
            if session.state == SessionState::Created {
                session.state = SessionState::Prefilling;
                
                let prompt_tokens = session.tokens.clone();
                
                Ok(DecodeBatch {
                    sessions: vec![session.id],
                    input_tokens: prompt_tokens,
                    block_tables: vec![session.kv_cache.block_table.clone()],
                    mamba_states: vec![session.kv_cache.mamba_states.clone()],
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
    pub kv_manager: HybridKvManager,
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
                self.kv_manager.free_session(s.id).ok();
                false
            } else {
                true
            }
        });
        
        // 3. Build decode batch (highest priority)
        let decode_batch = self.batch_builder.build_decode_batch(&mut self.active_sessions)?;
        
        // 4. If decode batch is small, try to prefill new sessions
        let prefill_batch = if decode_batch.sessions.len() < self.batch_builder.max_batch_size / 2 {
            self.batch_builder.build_prefill_batch(&mut self.active_sessions).ok()
        } else {
            None
        };
        
        Ok(ScheduledWork {
            decode_batch,
            prefill_batch,
        })
    }
    
    pub fn create_session(&mut self, request: Request) -> Result<Session> {
        let id = SessionId::new();
        let tokens = self.tokenizer.encode(&request.prompt)?;
        
        // Allocate KV cache
        let kv_cache = self.kv_manager.allocate_for_session(id, tokens.len())?;
        
        Ok(Session {
            id,
            state: SessionState::Created,
            tokens,
            num_prompt_tokens: tokens.len(),
            num_generated_tokens: 0,
            max_tokens: request.max_tokens.unwrap_or(512),
            sampling_config: request.sampling_config,
            kv_cache,
            created_at: Instant::now(),
            last_activity: Instant::now(),
            priority: request.priority,
        })
    }
}
```

## File Structure

```
crates/scheduler/
  Cargo.toml
  src/
    lib.rs
    session.rs           # Session, SessionState, SessionId
    lifecycle.rs         # Session lifecycle transitions
    batch.rs             # BatchBuilder, DecodeBatch, PrefillBatch
    scheduler.rs         # RoundRobinScheduler
    queue.rs             # RequestQueue
    
crates/kv/
  Cargo.toml
  src/
    lib.rs
    manager.rs           # HybridKvManager
    paged.rs             # PagedKvCache, BlockTable, BlockAllocator
    mamba.rs             # MambaState
    eviction.rs          # LRU eviction to CPU/SSD
```

## Testing

### Batch Builder Test

```rust
#[test]
fn test_batch_builder() {
    let mut sessions = vec![
        Session::new("Hello", 10),
        Session::new("World", 10),
        Session::new("Test", 10),
    ];
    
    let builder = BatchBuilder {
        max_batch_size: 2,
        max_tokens_per_batch: 100,
    };
    
    let batch = builder.build_decode_batch(&mut sessions).unwrap();
    assert_eq!(batch.sessions.len(), 2);
}
```

### Eviction Test

```rust
#[test]
fn test_kv_eviction() {
    let mut kv = HybridKvManager::new(100, 16, 64, 4, 256);
    
    // Allocate for 3 sessions
    let id1 = SessionId::new();
    let blocks1 = kv.allocate_blocks(id1, 1000).unwrap();
    
    let id2 = SessionId::new();
    let blocks2 = kv.allocate_blocks(id2, 1000).unwrap();
    
    let id3 = SessionId::new();
    let blocks3 = kv.allocate_blocks(id3, 1000).unwrap();
    
    // Simulate activity
    // id1 is active, id2 and id3 are inactive
    
    // Try to allocate for id4 (should evict id2)
    let id4 = SessionId::new();
    let blocks4 = kv.allocate_blocks(id4, 500).unwrap();
    
    // Verify id2 was evicted
    assert!(kv.block_table.iter().any(|t| t.session_id == id4));
    // id2 might still be in CPU cache, not GPU
}
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
