//! Integration tests for the scheduler crate.
//!
//! Tests end-to-end flows: request admission, session lifecycle,
//! batch construction, and scheduling iterations.

use std::time::{Duration, Instant};

use infers_kv::{PagedKvManager, SequencePageTable};

use infers_scheduler::{
    BatchBuilder, Request, RequestQueue, RoundRobinScheduler,
    SamplingConfig, SamplingStrategy, Session, SessionState,
};

fn make_kv() -> PagedKvManager {
    PagedKvManager::new(200, 16, 4, 256, 1024 * 1024, 1024 * 1024)
}

fn make_request(id: usize, tokens: Vec<u32>) -> Request {
    Request::new(id, tokens, SamplingConfig::default())
}

/// End-to-end: enqueue requests → schedule → prefill → decode → complete → cleanup
#[test]
fn test_full_session_lifecycle() {
    let kv = make_kv();
  let mut sched = RoundRobinScheduler::new(4, 4, kv);

    // Enqueue 3 requests
    sched.enqueue_request(make_request(0, vec![1, 2, 3, 4, 5]));
    sched.enqueue_request(make_request(1, vec![10, 20]));
    sched.enqueue_request(make_request(2, vec![100]));

    // Round 1: admit all 3, prefill one
    let work1 = sched.schedule().unwrap();
    assert_eq!(sched.active_count(), 3);
    assert!(work1.prefill_batch.is_some());
    assert_eq!(work1.prefill_batch.as_ref().unwrap().sessions.len(), 1);
    assert!(work1.decode_batch.sessions.is_empty()); // all sessions are still Created/Prefilling

    // Simulate prefill completion → transition to Decoding
    for session in &mut sched.active_sessions {
        if session.state == SessionState::Prefilling {
            let _ = infers_scheduler::lifecycle::finish_prefill(session);
        }
    }

    // Round 2: should build decode batch (one Decoding session)
    let work2 = sched.schedule().unwrap();
    assert!(work2.decode_batch.sessions.len() >= 1);
    assert_eq!(work2.decode_batch.input_tokens.len(), work2.decode_batch.sessions.len());
    assert_eq!(work2.decode_batch.block_tables.len(), work2.decode_batch.sessions.len());

    // Transition any newly prefilled session to Decoding
    for session in &mut sched.active_sessions {
        if session.state == SessionState::Prefilling {
            let _ = infers_scheduler::lifecycle::finish_prefill(session);
        }
    }

    // Transition all sessions to Completed (Created sessions need full lifecycle)
    for session in &mut sched.active_sessions {
        match session.state {
            SessionState::Created => {
                let _ = infers_scheduler::lifecycle::start_prefill(session);
                let _ = infers_scheduler::lifecycle::finish_prefill(session);
                let _ = infers_scheduler::lifecycle::complete_session(session);
            }
            SessionState::Decoding => {
                let _ = infers_scheduler::lifecycle::complete_session(session);
            }
            _ => {}
        }
    }

    // Round 3: cleanup completed sessions
    let _ = sched.schedule().unwrap();
    assert_eq!(sched.active_count(), 0);
    assert!(!sched.is_busy());
}

/// Verify BatchBuilder constructs correct decode batch from sessions
/// with varying page tables via PagedKvManager.
#[test]
fn test_batch_builder_with_real_kv_manager() {
    let mut kv = make_kv();
    let builder = BatchBuilder::new(4);

    // Create sessions with real KV sequences
    let mut sessions = Vec::new();

    // Session 1: 2 pages, 20 tokens
    let seq1 = kv.create_sequence();
    kv.append_page(seq1).unwrap();
    kv.append_page(seq1).unwrap();
    for _ in 0..20 {
        kv.add_token(seq1).unwrap();
    }
      sessions.push(Session {
        id: seq1,
        state: SessionState::Decoding,
        tokens: (0..20).collect(),
        num_prompt_tokens: 20,
        num_generated_tokens: 5,
        max_tokens: 100,
        page_table: SequencePageTable::new(16),
        created_at: Instant::now(),
        last_activity: Instant::now(),
        priority: 0,
        routing_id: None,
        sampling_config: SamplingConfig::default(),
    });

    // Session 2: 1 page, 10 tokens
    let seq2 = kv.create_sequence();
    kv.append_page(seq2).unwrap();
    for _ in 0..10 {
        kv.add_token(seq2).unwrap();
    }
    sessions.push(Session {
        id: seq2,
        state: SessionState::Decoding,
        tokens: (0..10).collect(),
        num_prompt_tokens: 10,
        num_generated_tokens: 3,
        max_tokens: 100,
        page_table: SequencePageTable::new(16),
        created_at: Instant::now(),
        last_activity: Instant::now(),
        priority: 0,
        routing_id: None,
        sampling_config: SamplingConfig::default(),
    });

    let batch = builder.build_decode_batch(&sessions, &kv).unwrap();
    assert_eq!(batch.sessions.len(), 2);
    assert_eq!(batch.sessions[0], seq1);
    assert_eq!(batch.sessions[1], seq2);
    assert_eq!(batch.input_tokens, vec![19, 9]); // latest tokens
    assert_eq!(batch.block_tables[0].len(), 2); // seq1 has 2 pages
    assert_eq!(batch.block_tables[1].len(), 1); // seq2 has 1 page
}

/// Verify the PagedKvManager integration test from the Phase 6 plan.
/// Tests page lifecycle with multiple sessions.
#[test]
fn test_page_lifecycle_with_sessions() {
    let mut kv = PagedKvManager::new(
        100,                // total_pages
        16,                 // page_size
        4,                  // num_kv_heads
        256,                // head_dim
        1024 * 1024 * 1024, // max_cache_bytes
        1024 * 1024,        // eviction_max_bytes
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
    assert_eq!(kv.num_pages(id3).unwrap(), 0);

    // Verify pool reflects allocated pages
    let free_before = kv.num_free_pages();
    kv.delete_sequence(id1).unwrap();
    let free_after = kv.num_free_pages();
    assert_eq!(free_after, free_before + 3);
}

/// Verify scheduler memory management — sessions that complete
/// return their pages to the pool.
#[test]
fn test_scheduler_page_reclamation() {
    let kv = make_kv();
   let mut sched = RoundRobinScheduler::new(4, 4, kv);

    // Get initial free page count
    let initial_free = sched.kv_manager.num_free_pages();

    // Enqueue and admit requests with enough tokens to use pages
    sched.enqueue_request(make_request(0, (0..32).collect())); // 2 pages
    sched.enqueue_request(make_request(1, (0..48).collect())); // 3 pages
    let _ = sched.schedule().unwrap();

    let free_after_admit = sched.kv_manager.num_free_pages();
    assert!(free_after_admit < initial_free, "Pages should be consumed by sessions");

    // Complete all sessions through proper lifecycle transitions
    for session in &mut sched.active_sessions {
        match session.state {
            SessionState::Created => {
                let _ = infers_scheduler::lifecycle::start_prefill(session);
                let _ = infers_scheduler::lifecycle::finish_prefill(session);
                let _ = infers_scheduler::lifecycle::complete_session(session);
            }
            SessionState::Prefilling => {
                let _ = infers_scheduler::lifecycle::finish_prefill(session);
                let _ = infers_scheduler::lifecycle::complete_session(session);
            }
            SessionState::Decoding => {
                let _ = infers_scheduler::lifecycle::complete_session(session);
            }
            _ => {}
        }
    }

    // Schedule cleanup
    let _ = sched.schedule().unwrap();

    // Pages should be returned
    let free_after_cleanup = sched.kv_manager.num_free_pages();
    assert_eq!(free_after_cleanup, initial_free, "All pages should be freed after all sessions complete");
}

/// Verify RequestQueue priority ordering with multiple priority levels.
#[test]
fn test_priority_queue_integration() {
    let mut queue = RequestQueue::new();
    let config = SamplingConfig::default();

    // Mix priorities: interactive (high=10), background (low=0)
    queue.enqueue(Request { id: 0, tokens: vec![1], config: config.clone(), priority: 0, routing_id: None });
    queue.enqueue(Request { id: 1, tokens: vec![2], config: config.clone(), priority: 10, routing_id: None });
    queue.enqueue(Request { id: 2, tokens: vec![3], config: config.clone(), priority: 5, routing_id: None });
    queue.enqueue(Request { id: 3, tokens: vec![4], config, priority: 10, routing_id: None });

    // Order should be: id1(10), id3(10), id2(5), id0(0)
    assert_eq!(queue.dequeue().unwrap().id, 1);
    assert_eq!(queue.dequeue().unwrap().id, 3);
    assert_eq!(queue.dequeue().unwrap().id, 2);
    assert_eq!(queue.dequeue().unwrap().id, 0);
}

/// Verify Session is_evictable behavior.
#[test]
fn test_session_eviction_timing() {
    let mut session = Session {
        id: 0,
        state: SessionState::Decoding,
        tokens: vec![1, 2, 3],
        num_prompt_tokens: 3,
        num_generated_tokens: 5,
        max_tokens: 100,
        page_table: SequencePageTable::new(16),
        created_at: Instant::now(),
        last_activity: Instant::now(),
        priority: 0,
        routing_id: None,
        sampling_config: SamplingConfig::default(),
    };

    // Just created — not evictable yet
    assert!(!session.is_evictable());

    // Simulate passage of time
    session.last_activity = Instant::now() - Duration::from_secs(31);
    assert!(session.is_evictable());
}

 /// Verify SamplingStrategy and SamplingConfig types are re-exportable and usable.
#[test]
fn test_sampling_config_reexport() {
    let config = SamplingConfig {
        strategy: SamplingStrategy::Temperature { temp: 0.8 },
        max_tokens: 256,
        stop_sequences: vec!["<eos>".to_string()],
        ..Default::default()
    };

    assert!(matches!(config.strategy, SamplingStrategy::Temperature { .. }));
    assert_eq!(config.max_tokens, 256);
    assert_eq!(config.stop_sequences.len(), 1);
}

/// Verify that memory pressure detection works end-to-end with a full pool.
#[test]
fn test_memory_pressure_triggers_eviction() {
    use infers_scheduler::pressure::{is_under_pressure, PressureConfig};

    // Create a tiny pool (4 pages) so we can fill it easily
    let mut kv = infers_kv::PagedKvManager::new(4, 16, 4, 256, 1024, 65536);
    let config = PressureConfig::default();

    // Fill the pool completely
    let seq = kv.create_sequence();
    for _ in 0..4 {
        kv.append_page(seq).unwrap();
    }
    assert_eq!(kv.pool_utilization(), 1.0);

    // Should be under pressure
    assert!(is_under_pressure(&kv, &config));
}

/// Verify evict→restore round-trip through PagedKvManager with data integrity.
#[test]
fn test_evict_restore_round_trip_integration() {
    use infers_kv::PagedKvManager;

    let page_size = 16;
    let num_kv_heads = 4;
    let head_dim = 256;
    let page_bytes = page_size * num_kv_heads * head_dim * 2; // 32768

    let mut kv = PagedKvManager::new(10, page_size, num_kv_heads, head_dim, 1024, 65536);

    // Create sequence and allocate pages
    let seq1 = kv.create_sequence();
    kv.append_page(seq1).unwrap();
    kv.append_page(seq1).unwrap();

    // Create distinguishable page data
    let page_data = vec![
        vec![0xABu8; page_bytes], // page 0
        vec![0xCDu8; page_bytes], // page 1
    ];

    let free_before = kv.num_free_pages();

    // Evict
    let evicted = kv.evict_sequence(seq1, page_data).unwrap();
    assert_eq!(evicted.page_ids.len(), 2);

    // Pages returned to pool
    assert_eq!(kv.num_free_pages(), free_before + 2);
    assert_eq!(kv.num_evicted_pages(), 2);

    // Verify sequence is gone
    assert!(kv.get_page_table(seq1).is_err());

    // Restore
    let (new_seq, restored_data) = kv.restore_sequence(evicted).unwrap();
    assert_eq!(restored_data.len(), 2);
    assert_eq!(restored_data[0][0], 0xAB);
    assert_eq!(restored_data[1][0], 0xCD);
    assert_eq!(kv.num_pages(new_seq).unwrap(), 2);
    assert_eq!(kv.num_evicted_pages(), 0);
}

/// Verify that LRU eviction selection works with scheduler sessions.
#[test]
fn test_lru_eviction_candidate_selection() {
    use std::time::{Duration, Instant};
    use infers_scheduler::session::{Session, SessionState};
    use infers_scheduler::pressure::select_lru_eviction_candidate;
    use infers_kv::SequencePageTable;

    let now = Instant::now();

    let sessions = vec![
        Session {
            id: 0, state: SessionState::Decoding, tokens: vec![1],
            num_prompt_tokens: 1, num_generated_tokens: 5, max_tokens: 100,
            page_table: SequencePageTable::new(16),
            created_at: now, last_activity: now - Duration::from_secs(100),
            priority: 0,
            routing_id: None,
            sampling_config: SamplingConfig::default(),
        },
        Session {
            id: 1, state: SessionState::Decoding, tokens: vec![2],
            num_prompt_tokens: 1, num_generated_tokens: 5, max_tokens: 100,
            page_table: SequencePageTable::new(16),
            created_at: now, last_activity: now - Duration::from_secs(200),
            priority: 0,
            routing_id: None,
            sampling_config: SamplingConfig::default(),
        },
        Session {
            id: 2, state: SessionState::Prefilling, tokens: vec![3],
            num_prompt_tokens: 1, num_generated_tokens: 0, max_tokens: 100,
            page_table: SequencePageTable::new(16),
            created_at: now, last_activity: now, // too recent
            priority: 0,
            routing_id: None,
            sampling_config: SamplingConfig::default(),
        },
    ];

    // Session 1 has oldest last_activity (200s ago) — should be selected
    let candidate = select_lru_eviction_candidate(&sessions);
    assert_eq!(candidate, Some(1));
}

/// Verify CpuPagePool budget enforcement across multiple stores.
#[test]
fn test_cpu_page_pool_budget_integration() {
    use infers_kv::CpuPagePool;

    // Budget: 2 pages worth
    let page_bytes = 1024;
    let mut pool = CpuPagePool::new(10, page_bytes, 2 * page_bytes);

    pool.store(0, vec![0u8; page_bytes]).unwrap();
    pool.store(1, vec![0u8; page_bytes]).unwrap();
    assert!(pool.is_full());
    assert_eq!(pool.used_bytes(), 2 * page_bytes);

    // Retrieve one page — budget freed
    let _ = pool.retrieve(0).unwrap();
    assert_eq!(pool.used_bytes(), page_bytes);
    assert!(!pool.is_full());

    // Can store again
    pool.store(5, vec![0u8; page_bytes]).unwrap();
    assert!(pool.is_full());
}
