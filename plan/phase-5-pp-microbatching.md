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
pub struct PipelineStage {
    pub stage_id: usize,
    pub gpu_id: usize,
    pub start_layer: usize,
    pub end_layer: usize,
    pub weights: WeightRegistry,
    pub kv_manager: StageKvManager,
}

pub struct PipelineEngine {
    pub stages: Vec<PipelineStage>,
    pub microbatch_size: usize,
    pub p2p_comm: P2PCommunicator,
}

impl PipelineEngine {
    pub fn new(
        config: &ModelConfig,
        weights: WeightRegistry,
        microbatch_size: usize,
    ) -> Result<Self> {
        let num_layers = config.num_hidden_layers;
        let layers_per_stage = num_layers / 2;  // PP=2
        
        let stage0 = PipelineStage {
            stage_id: 0,
            gpu_id: 0,
            start_layer: 0,
            end_layer: layers_per_stage,
            weights: weights.slice_layers(0..layers_per_stage),
            kv_manager: StageKvManager::new(),
        };
        
        let stage1 = PipelineStage {
            stage_id: 1,
            gpu_id: 1,
            start_layer: layers_per_stage,
            end_layer: num_layers,
            weights: weights.slice_layers(layers_per_stage..num_layers),
            kv_manager: StageKvManager::new(),
        };
        
        let p2p = P2PCommunicator::new(0, 1)?;
        
        Ok(Self {
            stages: vec![stage0, stage1],
            microbatch_size,
            p2p_comm: p2p,
        })
    }
}
```

### P2P Communication

```rust
pub struct P2PCommunicator {
    pub src_device: usize,
    pub dst_device: usize,
    pub can_access_peer: bool,
}

impl P2PCommunicator {
    pub fn new(src: usize, dst: usize) -> Result<Self> {
        let can_access = unsafe {
            let mut can_access = 0i32;
            cudaDeviceCanAccessPeer(&mut can_access, src as i32, dst as i32);
            can_access != 0
        };
        
        if can_access {
            unsafe {
                cudaDeviceEnablePeerAccess(dst as i32, 0);
            }
        }
        
        Ok(Self {
            src_device: src,
            dst_device: dst,
            can_access_peer: can_access,
        })
    }
    
    pub fn send(
        &self,
        buffer: &DeviceBuffer<half>,
        stream: &CudaStream,
    ) -> Result<()> {
        if self.can_access_peer {
            // Direct P2P copy
            stream.memcpy_peer(
                buffer,
                self.dst_device,
            )?;
        } else {
            // Fallback: copy through CPU pinned memory
            let pinned = stream.alloc_pinned(buffer.len())?;
            stream.memcpy_dtoh(buffer, &pinned)?;
            // ... send to other GPU ...
        }
        
        Ok(())
    }
    
    pub fn recv(
        &self,
        buffer: &mut DeviceBuffer<half>,
        stream: &CudaStream,
    ) -> Result<()> {
        if self.can_access_peer {
            stream.memcpy_peer_from(
                buffer,
                self.src_device,
            )?;
        } else {
            // Fallback: copy through CPU pinned memory
        }
        
        Ok(())
    }
}
```

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
    pub hidden_states: Option<DeviceBuffer<half>>,
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
                
                // Send to stage 1
                self.p2p_comm.send(
                    microbatch.hidden_states.as_ref().unwrap(),
                    &self.stages[0].stream,
                )?;
                
                scheduler.in_flight.push(microbatch);
            }
            
            // Stage 1: Process received microbatches
            for microbatch in &mut scheduler.in_flight {
                if microbatch.stage == 1 {
                    let mut hidden = DeviceBuffer::alloc(
                        microbatch.hidden_states.as_ref().unwrap().len()
                    )?;
                    
                    self.p2p_comm.recv(&mut hidden, &self.stages[1].stream)?;
                    
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
    
    fn forward_stage0(
        &self,
        microbatch: &Microbatch,
    ) -> Result<DeviceBuffer<half>> {
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
        hidden: &DeviceBuffer<half>,
        microbatch: &Microbatch,
    ) -> Result<DeviceBuffer<half>> {
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

```rust
pub struct StageKvManager {
    // For full attention layers in this stage
    pub paged_kv: PagedKvCache,
    
    // For GDN layers in this stage
    pub mamba_states: HashMap<SessionId, Vec<MambaState>>,
}

impl StageKvManager {
    pub fn new(max_pages: usize, page_size: usize) -> Self {
        Self {
            paged_kv: PagedKvCache::new(max_pages, page_size),
            mamba_states: HashMap::new(),
        }
    }
    
    pub fn allocate_for_microbatch(
        &mut self,
        microbatch: &Microbatch,
        stage: &PipelineStage,
    ) -> Result<()> {
        for request in &microbatch.requests {
            // Allocate Mamba states for GDN layers in this stage
            for layer_idx in stage.start_layer..stage.end_layer {
                if stage.config.get_layer_type(layer_idx) == LayerType::GatedDeltaNet {
                    self.mamba_states
                        .entry(request.id)
                        .or_insert_with(Vec::new)
                        .push(MambaState::new()?);
                }
            }
            
            // Allocate KV blocks for full attention layers
            for layer_idx in stage.start_layer..stage.end_layer {
                if stage.config.get_layer_type(layer_idx) == LayerType::FullAttention {
                    let num_blocks = (request.num_tokens + stage.page_size - 1) / stage.page_size;
                    let blocks = self.paged_kv.allocate(num_blocks)?;
                    request.set_kv_blocks(layer_idx, blocks)?;
                }
            }
        }
        
        Ok(())
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
    lib.rs
    tp.rs               # TensorParallel (from Phase 4)
    pp.rs               # PipelineParallel
    microbatch.rs       # MicrobatchScheduler
    stage.rs            # PipelineStage
    p2p.rs              # P2PCommunicator
    engine.rs           # Unified ParallelEngine (TP or PP)
```

## Testing

### Bubble Test

```rust
#[test]
fn test_pp_bubble_fraction() {
    let engine = PipelineEngine::new(&config, weights, 2)?;
    
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
    let tp_engine = TensorParallel::new(&config, weights.clone())?;
    let pp_engine = PipelineParallel::new(&config, weights, 1)?;
    
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
