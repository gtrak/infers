# Phase 10: Server Orchestration — Wire Everything Together

---
**Status**: PARTIAL
**Last Updated**: 2026-06-11
**Rationale**: Server binary exists with axum routes. BUT: NOT wired to real inference engine. Still returns mock responses.
**Actual Deliverables**:
- [x] Server binary exists with axum routes
- [x] HTTP server starts and responds
- [ ] `InferenceOrchestrator` wired to real engine
- [ ] Background scheduler loop
- [ ] Token streaming through `mpsc` channels
- [ ] Chat handler with real tokenization + streaming
- [ ] Two concurrent requests produce interleaved tokens
- [ ] Session cleanup
---

**Duration:** 1.5 weeks  
**Goal:** Connect the HTTP server, scheduler, and inference backend into an end-to-end pipeline. Replace mock handlers with real generation.

## Problem

All the pieces exist but aren't wired together:

```
HTTP request → [nothing] → scheduler → [nothing] → ForwardEngine → [nothing] → SSE response
```

The server crate has mock handlers. The scheduler has no background loop. The ForwardEngine has no caller. The tokenizer exists but isn't connected.

## Architecture

### Orchestrator (new: `crates/orchestrator/` or inline in server)

A new struct that owns the scheduler, backend engine, and eviction store, running a continuous schedule→dispatch loop:

```
                     schedule()
                         │
    enqueue(request) ──►▼──────► decode/prefill batches
                         │              │
                         │         ForwardEngine
                         │         (prefill_paged /
                         │          decode_paged)
                         │              │
                         │         sampled tokens
                         │              │
                         ▼              ▼
                    evict idle     send tokens to
                    (mark_evicted +  response streams
                     BackendEvictionStore)
```

### Response Routing

Each session needs a channel back to the HTTP response. Use `tokio::sync::mpsc`:
- On enqueue: create an `mpsc::sender` → store in `HashMap<SequenceId, Sender<u32>>`
- On token generated: send token through the channel
- On complete: close channel

## Deliverables

### 1. [ ] InferenceOrchestrator (new struct)

Owns `RoundRobinScheduler`, `ForwardEngine`, `BackendEvictionStore`, and response channels.

```rust
pub struct InferenceOrchestrator {
    scheduler: RoundRobinScheduler,
    engine: ForwardEngine,
    eviction_store: BackendEvictionStore,
    stream: Arc<CudaStream>,
    response_tx: HashMap<SequenceId, mpsc::Sender<u32>>,
}
```

Methods:
- `enqueue_request(prompt_tokens, config) -> SequenceId` — creates Request, enqueues, returns seq_id
- `register_response_channel(seq_id, tx)` — stores sender for token streaming
- `step() -> Result<()>` — one schedule→execute iteration
- `process_batch(decode_batch) -> Result<Vec<u32>>` — dispatches decode batch to ForwardEngine
- `cleanup_session(seq_id)` — marks complete, sends final token

### 2. [ ] Background Scheduler Loop

A tokio task that calls `orchestrator.step()` in a loop with a small delay (or triggered by events):

```rust
pub async fn run_scheduler_loop(orchestrator: Arc<Mutex<InferenceOrchestrator>>) {
    loop {
        let mut guard = orchestrator.lock().await;
        if let Err(e) = guard.step() {
            tracing::error!("Scheduler step failed: {e:?}");
        }
        drop(guard);
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}
```

### 3. [ ] Token Streaming

Each generate-session token is sent through an `mpsc` channel to the HTTP handler:

```rust
// In orchestrator.step():
for token in tokens {
    if let Some(tx) = self.response_tx.get(&seq_id) {
        let _ = tx.send(token).await;
    }
}
```

The HTTP handler reads from the `mpsc::Receiver` and emits SSE events:

```rust
let rx = orchestrator.register_listener(seq_id);
let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
    .map(|token| Ok(Event::default().data(serde_json::to_string(&chunk).unwrap())));
```

### 4. [~] AppState Update (struct exists but not wired to real engine)

```rust
pub struct AppState {
    pub model_name: String,
    pub orchestrator: Arc<Mutex<InferenceOrchestrator>>,
    pub tokenizer: Tokenizer,  // from infers-tokenizer
}
```

### 5. [ ] Chat Handler Rewrite

Replace mock streaming with real pipeline:

1. Tokenize prompt using tokenizer
2. Enqueue request in orchestrator
3. Register response channel
4. Return SSE stream that yields tokens as they're generated
5. On stream end, report usage stats

## File Changes

```
crates/server/Cargo.toml             # ADD: infers-scheduler, infers-backend-native, infers-tokenizer, infers-kv, tokio-stream
crates/server/src/
  orchestrator.rs                    # NEW: InferenceOrchestrator
  state.rs                           # MODIFY: AppState holds orchestrator + tokenizer
  handlers/chat.rs                   # REWRITE: real tokenization + streaming
  server.rs                          # MODIFY: initialize orchestrator, spawn loop
  main.rs                            # MODIFY: pass CLI args to orchestrator init

Optional:
crates/orchestrator/                 # Separate crate if orchestrator logic is large
  Cargo.toml
  src/lib.rs
```

## Deferred

- Full paged pipeline integration (prefill_paged / decode_paged are stubs — the flat-cache prefill/decode are used instead)
- Multi-GPU orchestration (single GPU for now)
- Metrics wiring (existing metrics crate needs to be connected)

## Success Criteria

1. `curl POST /v1/chat/completions -d '{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true}'` returns SSE stream of real generated tokens
2. Two concurrent requests produce interleaved tokens (continuous batching)
3. Session cleanup works: completed sessions free resources
4. Error handling: invalid requests return proper API errors
