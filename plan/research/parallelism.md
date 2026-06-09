# Parallel Processing Research

## Supported Schemes

### 1. Tensor Parallelism (TP=2)

**Startup:** `--parallelism tp`

**Weight distribution:**
- Each GPU holds half of every layer's weight matrices
- Column-parallel for Q/K/V projections, gate/up projections
- Row-parallel for O projection, down projection

**Communication pattern:**
```
Layer 0 (GDN):
  GPU0: q_proj[0:1280], k_proj[0:1280], v_proj[0:1280]
  GPU1: q_proj[1280:2560], k_proj[1280:2560], v_proj[1280:2560]
  
  After GDN forward:
    NCCL all-reduce(output)  ← synchronize partial results

Layer 0 (MLP):
  GPU0: gate_proj[0:8704], up_proj[0:8704]
  GPU1: gate_proj[8704:17408], up_proj[8704:17408]
  
  After MLP forward:
    NCCL all-reduce(output)  ← synchronize partial results
```

**KV cache distribution:**
- Each GPU holds KV for its shard
- All-gather before attention if needed (or use local KV only)

**Best for:** Large batch sizes (continuous batching)

### 2. Pipeline Parallelism (PP=2) with Microbatching

**Startup:** `--parallelism pp --pp-microbatch-size N`

**Stage partitioning:**
```
GPU0 (Stage 0): Layers 0-31
  - 24 GDN layers (0-23)
  - 8 full attention layers (0, 4, 8, 12, 16, 20, 24, 28)
  - Owns Mamba states for layers 0-31
  - Owns KV cache for full attention layers 0-31

GPU1 (Stage 1): Layers 32-63
  - 24 GDN layers (32-55)
  - 8 full attention layers (32, 36, 40, 44, 48, 52, 56, 60)
  - Owns Mamba states for layers 32-63
  - Owns KV cache for full attention layers 32-63
```

**Communication:**
- P2P send/recv of hidden states between stages
- No all-reduce needed
- Uses `cudaMemcpyPeerAsync` or NCCL P2P

**Microbatching example (batch=8, microbatch=2):**
```
Time 0: GPU0 processing request batch [0,1] (layers 0-31)
Time 1: GPU0 processing request batch [2,3] (layers 0-31)
        GPU1 processing request batch [0,1] (layers 32-63)
Time 2: GPU0 processing request batch [4,5] (layers 0-31)
        GPU1 processing request batch [2,3] (layers 32-63)
Time 3: GPU0 processing request batch [6,7] (layers 0-31)
        GPU1 processing request batch [4,5] (layers 32-63)
Time 4: GPU1 processing request batch [6,7] (layers 32-63)
```

**Bubble calculation:**
```
Bubble fraction = (num_stages - 1) / (num_microbatches + num_stages - 1)
                = 1 / (4 + 2 - 1) = 1/5 = 20% for batch=8, microbatch=2
                
With microbatch=1: bubble = 1/8 = 12.5%
With microbatch=4: bubble = 1/4 = 25%
```

**Best for:** Small-to-medium batch sizes, latency-sensitive

## Mode Selection Criteria

| Scenario | Recommended Mode | Microbatch |
|---|---|---|
| Single request | PP | 1 |
| 2-4 concurrent requests | PP | 1-2 |
| 5+ concurrent requests | TP | N/A |
| Max throughput | TP | N/A |
| Min latency (single user) | PP | 1 |

## Memory Distribution

### TP=2

| Component | Per GPU | Total |
|---|---|---|
| Weights (PrismaSCOUT) | ~10 GB | ~20 GB |
| Weights (AutoRound) | ~9 GB | ~18 GB |
| Weights (BF16) | ~27 GB | ~54 GB |
| KV cache (FP8, 262K) | ~17 GB | ~34 GB |
| KV cache (NVFP4, 262K) | ~8.5 GB | ~17 GB |
| Workspace | ~4 GB | ~8 GB |

### PP=2

| Component | Per GPU | Total |
|---|---|---|
| Weights (PrismaSCOUT) | ~20 GB | ~40 GB |
| Weights (AutoRound) | ~18 GB | ~36 GB |
| Weights (BF16) | ~54 GB | ~108 GB → OOM |
| KV cache (FP8, 262K) | ~8.5 GB | ~17 GB |
| KV cache (NVFP4, 262K) | ~4.3 GB | ~8.6 GB |
| Workspace | ~4 GB | ~8 GB |

**Note:** PP=2 with BF16 weights OOMs. Only PrismaSCOUT and AutoRound are viable.

## NCCL Requirements

### TP=2

**Collectives needed:**
- `ncclAllReduce` — after attention and MLP
- `ncclAllGather` — for KV cache if needed

**Comm setup:**
```rust
use cudarc::nccl::{Comm, Id};

let nccl_id = Id::new()?;
let comm = Comm::from_rank(
    &stream, rank, world_size, nccl_id
)?;

// After attention
comm.all_reduce(
    &input_slice,
    &mut output_slice,
    &ReduceOp::Sum,
)?;
```

### PP=2

**Collectives needed:**
- `ncclSend` / `ncclRecv` — between stages
- Or: `cudaMemcpyPeerAsync` (if P2P enabled)

**Comm setup:**
```rust
// Stage 0 → Stage 1
if rank == 0 {
    comm.send(&output_buffer, 1, tag)?;
} else {
    comm.recv(&mut input_buffer, 0, tag)?;
}
```

## P2P Memory Access

For PP=2 on 2× RTX 5060 Ti (PCIe, no NVLink):

```rust
// Check P2P capability
let can_access = cuda_device_can_access_peer(0, 1)?;

if can_access {
    // Enable P2P
    cuda_device_enable_peer_access(0, 1)?;
    
    // Use cudaMemcpyPeerAsync (faster than through CPU)
    stream.memcpy_peer(&dst, &src, peer_device)?;
} else {
    // Fallback: copy through CPU pinned memory
    let pinned = stream.alloc_pinned(size)?;
    stream.memcpy_dtoh(&src, &pinned)?;
    stream.memcpy_htod(&pinned, &dst)?;
}
```

**Note:** Consumer GPUs (RTX series) typically support P2P over PCIe but not NVLink. Bandwidth is limited to PCIe speed (~16-32 GB/s for PCIe 4.0 x16).

## Microbatching Implementation

```rust
struct PipelineEngine {
    stage: usize,  // 0 or 1
    num_stages: usize,
    microbatch_size: usize,
    
    // For stage 0
    input_queue: Vec<RequestBatch>,
    
    // For stage 1
    output_queue: Vec<RequestBatch>,
}

impl PipelineEngine {
    fn forward(&mut self, requests: Vec<Request>) {
        // Split into microbatches
        let microbatches: Vec<_> = requests
            .chunks(self.microbatch_size)
            .collect();
        
        for (i, microbatch) in microbatches.iter().enumerate() {
            if self.stage == 0 {
                // Process layers 0-31
                let hidden = self.forward_stage0(microbatch);
                
                // Send to stage 1
                self.send_to_next_stage(&hidden, i);
            } else {
                // Receive from stage 0
                let hidden = self.recv_from_prev_stage(i);
                
                // Process layers 32-63
                let output = self.forward_stage1(&hidden);
                
                // Emit tokens
                self.emit_output(&output, microbatch);
            }
        }
    }
}
```

## References

1. NCCL Docs: https://docs.nvidia.com/deeplearning/nccl/
2. vLLM TP Implementation: `../vllm/vllm/distributed/parallel_state.py`
3. vLLM PP Implementation: `../vllm/vllm/pipeline_parallel/`
4. Megatron-LM TP/PP: https://github.com/NVIDIA/Megatron-LM

## Cross-References

- See `architecture.md` for layer distribution details
- See `quantization.md` for memory calculations
- See Phase 4 (TP Forward) for TP implementation
- See Phase 5 (PP Microbatching) for PP implementation
- See Phase 6 (Continuous Batching) for batching across modes
