# Phase 5: PP=2 with Microbatching

**Duration:** 3 weeks  
**Goal:** Implement Pipeline Parallelism with microbatching across 2 GPUs.

## Deliverables

1. Stage partitioning (layers 0-31 on GPU0, 32-63 on GPU1)
2. P2P communication between stages (send/recv hidden states)
3. Microbatch scheduler
4. Pipeline bubble minimization
5. KV cache per-stage management
6. Switch between TP and PP at load time
7. Performance parity with TP for small batches

## Technical Details

### Stage Partitioning

```rust
use infers_model::{ModelConfig, WeightRegistry};
use infers_kv::PagedKvManager;
use infers_cuda::nccl::NcclCommunicator;

/// A single pipeline stage holding a subset of layers on one GPU.
///
/// PP=2 splits 64 layers into two stages of 32 layers each. Each stage
/// runs on a separate GPU and communicates hidden states via NCCL.
pub struct PipelineStage {
    pub stage_id: usize,
    pub gpu_id: usize,
    pub start_layer: usize,
    pub end_layer: usize,
    /// Sharded weights for this stage's layers.
    pub weights: WeightRegistry,
    /// Paged KV manager for full-attention layers in this stage.
    pub kv_manager: PagedKvManager,
}

/// Pipeline engine orchestrating two stages with microbatching.
pub struct PipelineEngine {
    pub stages: Vec<PipelineStage>,
    pub microbatch_size: usize,
    /// NCCL communicator for stage-to-stage hidden state transfer.
    pub nccl: NcclCommunicator,
}

impl PipelineEngine {
    pub fn new(
        config: &ModelConfig,
        weights: WeightRegistry,
        microbatch_size: usize,
        num_pages: usize,
        page_size: usize,
        max_cache_bytes: usize,
    ) -> Result<Self> {
        let num_layers = config.num_hidden_layers;
        let layers_per_stage = num_layers / 2; // PP=2

        // Use existing split_layers_pp() from infers-model
        let stage_ranges = split_layers_pp(config, 2);

        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let stage0 = PipelineStage {
            stage_id: 0,
            gpu_id: 0,
            start_layer: stage_ranges[0].start,
            end_layer: stage_ranges[0].end,
            weights: shard_weights_for_stage(&weights, &stage_ranges[0]),
            kv_manager: PagedKvManager::new(
                num_pages, page_size, num_kv_heads, head_dim, max_cache_bytes,
            ),
        };

        let stage1 = PipelineStage {
            stage_id: 1,
            gpu_id: 1,
            start_layer: stage_ranges[1].start,
            end_layer: stage_ranges[1].end,
            weights: shard_weights_for_stage(&weights, &stage_ranges[1]),
            kv_manager: PagedKvManager::new(
                num_pages, page_size, num_kv_heads, head_dim, max_cache_bytes,
            ),
        };

        // NCCL communicator for P2P between stages
        let nccl = NcclCommunicator::new(vec![
            get_stage_stream(0).clone(),
            get_stage_stream(1).clone(),
        ])?;

        Ok(Self {
            stages: vec![stage0, stage1],
            microbatch_size,
            nccl,
        })
    }
}
```

### P2P Communication

PP uses NCCL for stage-to-stage hidden state transfer. Each pipeline stage
runs on a separate GPU. The output hidden states from stage N are sent to
stage N+1 via NCCL `send`/`recv` operations on dedicated peer streams.

```rust
use infers_cuda::nccl::NcclCommunicator;
use infers_cuda::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Stage-to-stage hidden state transfer via NCCL P2P.
///
/// Each `PipelineStage` holds its own `NcclCommunicator` initialized with
/// the stage's CUDA stream and the peer stage's stream. Hidden states are
/// BF16 tensors of shape `[microbatch_size × seq_len × hidden_size]`.
///
/// For PP=2 with a single microbatch, stage 0 sends and stage 1 receives.
/// For multiple microbatches, sends and receives are interleaved to keep
/// both GPUs busy.
pub struct StageComm {
    /// NCCL communicator initialized with this stage and peer streams.
    pub nccl: NcclCommunicator,
    /// Rank within the NCCL communicator (0 for stage 0, 1 for stage 1).
    pub rank: usize,
    /// Peer rank.
    pub peer_rank: usize,
}

impl StageComm {
    /// Send hidden states to the next stage.
    pub fn send_hidden(
        &self,
        hidden: &CudaSlice<bf16>,
    ) -> Result<()> {
        self.nccl.send(hidden, self.peer_rank)
            .map_err(|e| anyhow!("NCCL send failed: {e}"))
    }

    /// Receive hidden states from the previous stage.
    pub fn recv_hidden(
        &self,
        hidden: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        self.nccl.recv(hidden, self.peer_rank)
            .map_err(|e| anyhow!("NCCL recv failed: {e}"))
    }
}
```

**Key design decisions:**
- NCCL handles P2P memory copies transparently (peer access or host staging)
- No raw CUDA runtime calls — all GPU communication goes through `cudarc::nccl`
- Each stage's `CudaStream` is passed to `NcclCommunicator::new()` at init
- For multi-GPU nodes, NCCL automatically uses NVLink/PCIe as available

### Microbatch Scheduler

```rust
pub struct MicrobatchScheduler {
    pub microbatch_size: usize,
    pub pending_requests: Vec<Request>,
    pub in_flight: Vec<Microbatch>,
}

#[derive(Debug)]
pub struct Microbatch {
    pub id: usize,
    pub requests: Vec<Request>,
    pub stage: usize,  // Current pipeline stage
    pub hidden_states: Option<CudaSlice<bf16>>,
}

impl MicrobatchScheduler {
    pub fn new(microbatch_size: usize) -> Self {
        Self {
            microbatch_size,
            pending_requests: Vec::new(),
            in_flight: Vec::new(),
        }
    }
    
    pub fn add_request(&mut self, request: Request) {
        self.pending_requests.push(request);
    }
    
    pub fn next_microbatch(&mut self) -> Option<Microbatch> {
        if self.pending_requests.is_empty() {
            return None;
        }
        
        let take = self.microbatch_size.min(self.pending_requests.len());
        let requests: Vec<_> = self.pending_requests.drain(..take).collect();
        
        Some(Microbatch {
            id: self.in_flight.len(),
            requests,
            stage: 0,
            hidden_states: None,
        })
    }
    
    pub fn advance_pipeline(&mut self) {
        // Move microbatches forward through pipeline stages
        for microbatch in &mut self.in_flight {
            microbatch.stage += 1;
        }
        
        // Remove completed microbatches
        self.in_flight.retain(|mb| mb.stage < 2);
    }
}
```

### Pipeline Forward Pass

```rust
impl PipelineEngine {
    pub fn forward_batch(
        &mut self,
        requests: Vec<Request>,
    ) -> Result<Vec<TokenBatch>> {
        // 1. Split into microbatches
        let mut scheduler = MicrobatchScheduler::new(self.microbatch_size);
        for req in requests {
            scheduler.add_request(req);
        }

        let mut results = Vec::new();
        let mut microbatch_id = 0;

        // 2. Pipeline loop
        loop {
            // Stage 0: Process new microbatches
            if let Some(mut microbatch) = scheduler.next_microbatch() {
                let hidden = self.forward_stage0(&microbatch)?;
                microbatch.hidden_states = Some(hidden);

                // Send to stage 1 via NCCL
                self.stages[0].comm.send_hidden(
                    microbatch.hidden_states.as_ref().unwrap(),
                )?;

                scheduler.in_flight.push(microbatch);
            }

            // Stage 1: Process received microbatches
            for microbatch in &mut scheduler.in_flight {
                if microbatch.stage == 1 {
                    let hidden_size = microbatch.hidden_states.as_ref().unwrap().len();
                    let mut hidden = self.stages[1].stream
                        .alloc_zeros::<bf16>(hidden_size)?;

                    self.stages[1].comm.recv_hidden(&mut hidden)?;

                    let output = self.forward_stage1(&hidden, microbatch)?;

                    // Sample tokens
                    let tokens = self.sample_batch(&output, &microbatch.requests)?;
                    results.push(tokens);

                    microbatch.stage = 2; // Complete
                }
            }

            scheduler.advance_pipeline();

            // Check if done
            if scheduler.pending_requests.is_empty()
                && scheduler.in_flight.is_empty() {
                break;
            }

            microbatch_id += 1;
        }

        Ok(results)
    }
                    microbatch.stage = 2; // Complete
                }
            }
            
            scheduler.advance_pipeline();
            
            // Check if done
            if scheduler.pending_requests.is_empty() 
                && scheduler.in_flight.is_empty() {
                break;
            }
            
            microbatch_id += 1;
        }
        
        Ok(results)
    }
    
    fn forward_stage0(
        &self,
        microbatch: &Microbatch,
    ) -> Result<CudaSlice<bf16>> {
        let stage = &self.stages[0];
        let mut hidden = self.embed_batch(&microbatch.requests, stage.gpu_id)?;
        
        for layer_idx in stage.start_layer..stage.end_layer {
            let layer_type = self.config.get_layer_type(layer_idx);
            
            match layer_type {
                LayerType::GatedDeltaNet => {
                    hidden = self.gdn_forward(
                        layer_idx, &hidden, microbatch, stage
                    )?;
                }
                LayerType::FullAttention => {
                    hidden = self.attention_forward(
                        layer_idx, &hidden, microbatch, stage
                    )?;
                }
            }
        }
        
        Ok(hidden)
    }
    
    fn forward_stage1(
        &self,
        hidden: &CudaSlice<bf16>,
        microbatch: &Microbatch,
    ) -> Result<CudaSlice<bf16>> {
        let stage = &self.stages[1];
        let mut hidden = hidden.clone();  // Or use P2P buffer directly
        
        for layer_idx in stage.start_layer..stage.end_layer {
            let layer_type = self.config.get_layer_type(layer_idx);
            
            match layer_type {
                LayerType::GatedDeltaNet => {
                    hidden = self.gdn_forward(
                        layer_idx, &hidden, microbatch, stage
                    )?;
                }
                LayerType::FullAttention => {
                    hidden = self.attention_forward(
                        layer_idx, &hidden, microbatch, stage
                    )?;
                }
            }
        }
        
        // Final norm + LM head
        let logits = self.lm_head(&hidden, stage.gpu_id)?;
        
        Ok(logits)
    }
}
```

### Stage KV Cache Management

Each stage manages its own subset of layers. Full-attention layers use the
paged KV system (`infers_kv::PagedKvManager`). GDN layers use recurrent
state vectors (`GdnState`).

```rust
use infers_kv::{PagedKvManager, SequenceId};
use crate::gdn::GdnState;
use std::collections::HashMap;

/// Per-stage state management: paged KV for attention, GDN states for recurrent layers.
pub struct StageState {
    /// Paged KV manager for full-attention layers in this stage's range.
    pub kv_manager: PagedKvManager,
    /// Per-session GDN recurrent states for layers in this stage's range.
    /// Key: (session_id, layer_idx) → GdnState.
    pub gdn_states: HashMap<(usize, usize), GdnState>,
}

impl StageState {
    pub fn new(
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_cache_bytes: usize,
    ) -> Self {
        Self {
            kv_manager: PagedKvManager::new(
                num_pages, page_size, num_kv_heads, head_dim, max_cache_bytes,
            ),
            gdn_states: HashMap::new(),
        }
    }

    /// Allocate KV pages for a new session's attention layers in this stage.
    pub fn create_session(&mut self) -> SequenceId {
        self.kv_manager.create_sequence()
    }

    /// Ensure GDN states exist for all GDN layers in this stage.
    pub fn ensure_gdn_states(
        &mut self,
        session_id: usize,
        config: &ModelConfig,
        start_layer: usize,
        end_layer: usize,
    ) {
        for layer_idx in start_layer..end_layer {
            if config.get_layer_type(layer_idx) == LayerType::GatedDeltaNet {
                self.gdn_states
                    .entry((session_id, layer_idx))
                    .or_insert_with(GdnState::new);
            }
        }
    }

    /// Free all resources for a session.
    pub fn free_session(&mut self, session_id: SequenceId) {
        let _ = self.kv_manager.delete_sequence(session_id);
        self.gdn_states
            .retain(|(sid, _), _| *sid != session_id as usize);
    }
}
```

## Bubble Analysis

### Bubble Calculation

```
For PP=2 with microbatching:

Bubble fraction = (num_stages - 1) / (num_microbatches + num_stages - 1)

Examples:
- Batch=8, microbatch=1: bubble = 1 / 9 = 11%
- Batch=8, microbatch=2: bubble = 1 / 5 = 20%
- Batch=8, microbatch=4: bubble = 1 / 3 = 33%
- Batch=16, microbatch=2: bubble = 1 / 9 = 11%
- Batch=16, microbatch=4: bubble = 1 / 5 = 20%
```

### Optimization: Interleaved Scheduling

To reduce bubbles, we can interleave forward passes:

```
Time 0: GPU0 forward [0,1] → send
Time 1: GPU0 forward [2,3] → send
        GPU1 recv [0,1] → forward → send
Time 2: GPU0 forward [4,5] → send
        GPU1 recv [2,3] → forward → send
Time 3: GPU0 forward [6,7] → send
        GPU1 recv [4,5] → forward → send
Time 4:        GPU1 recv [6,7] → forward
```

This keeps both GPUs busy after the initial fill.

## Memory Distribution

### PP=2 Memory Layout

```
GPU0:
  Weights (layers 0-31): ~10 GB (PrismaSCOUT)
  KV cache (layers 0-31): ~4.3 GB (NVFP4, 262K context)
  Workspace: ~4 GB
  Total: ~18.3 GB (fits in 32 GB)

GPU1:
  Weights (layers 32-63): ~10 GB (PrismaSCOUT)
  KV cache (layers 32-63): ~4.3 GB (NVFP4, 262K context)
  Workspace: ~4 GB
  Total: ~18.3 GB (fits in 32 GB)
```

### Comparison with TP=2

| Metric | TP=2 | PP=2 |
|---|---|---|
| Weights per GPU | ~10 GB | ~20 GB |
| KV per GPU | ~8.5 GB | ~4.3 GB |
| Total memory | ~19 GB | ~18.3 GB |
| Communication | All-reduce (every layer) | P2P (once per microbatch) |
| Bubble | None | 11-33% |
| Best batch size | Large (5+) | Small (1-4) |

## File Structure

```
crates/parallelism/
  Cargo.toml
  src/
    lib.rs              # TP=2 all-reduce, PP=2 stage orchestration
    tp.rs               # TensorParallel: NCCL all-reduce after attention/MLP
    pp.rs               # PipelineParallel: stage partitioning, microbatch scheduling
    microbatch.rs       # MicrobatchScheduler
    stage.rs            # PipelineStage, StageState
    comm.rs             # StageComm: NCCL send/recv between stages
    engine.rs           # Unified ParallelEngine (TP or PP dispatch)
```

## Testing

### Bubble Test

```rust
#[test]
fn test_pp_bubble_fraction() {
    let engine = PipelineEngine::new(
        &config, weights, 2, 1000, 16, 1024 * 1024 * 1024,
    )?;

    let batch = vec![request; 8];
    let (_, timing) = engine.forward_batch(batch)?;

    // Measure GPU utilization
    let gpu0_active = timing.gpu0_active_time;
    let gpu1_active = timing.gpu1_active_time;
    let total_time = timing.total_time;

    let bubble = 1.0 - (gpu0_active + gpu1_active) / (2.0 * total_time);
    assert!(bubble < 0.25, "Bubble too high: {}", bubble);
}
```

### Correctness Test

```rust
#[test]
fn test_pp_correctness() {
    // PP should produce same results as TP
    let tp_engine = TensorParallelEngine::new(&config, weights.clone())?;
    let pp_engine = PipelineEngine::new(
        &config, weights, 1, 1000, 16, 1024 * 1024 * 1024,
    )?;
    
    let prompt = "Hello, world!";
    let tokens = tokenizer.encode(prompt)?;
    
    let tp_result = tp_engine.prefill(&mut session, &tokens)?;
    let pp_result = pp_engine.prefill(&mut session, &tokens)?;
    
    assert_eq!(tp_result, pp_result);
}
```

## Dependencies

### Phase 5 → Phase 4

Uses forward pass logic from Phase 4, but split across stages.

### Phase 5 → Phase 2

Uses P2P communication primitives.

### Phase 5 → Phase 6

Continuous batching will need to work with both TP and PP modes.

## Success Criteria

1. PP=2 produces identical results to TP=2
2. Microbatch scheduler keeps both GPUs busy
3. Bubble fraction < 25% for batch size 4+
4. P2P communication works without memory copies (if peer access enabled)
5. Stage KV cache manages memory independently
6. Switching between TP and PP works at load time
7. Performance: comparable to TP for batch sizes 1-4

## Cross-References

- **Research:** See `../research/parallelism.md` for PP theory and memory calculations
- **Phase 2:** P2P communication uses CUDA runtime
- **Phase 4:** Forward pass logic is split across stages
- **Phase 6:** Continuous batching integrates with both modes

## Open Questions

1. Should we support dynamic switching between TP and PP at runtime?
2. How to handle pipeline flush (when no more requests)?
3. Should we implement pipeline parallelism for prefill (harder than decode)?
