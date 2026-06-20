# Parallelism Crate

Pipeline parallelism and tensor parallelism implementations for multi-GPU inference.

## Stage Communication

P2P hidden state transfer between pipeline stages via NCCL. See [[crates/parallelism/src/comm.rs#StageComm]].

### StageComm

P2P hidden state transfer between pipeline stages via NCCL. `send_hidden()` and `recv_hidden()` delegate to `NcclCommunicator` P2P methods. See [[crates/parallelism/src/comm.rs#StageComm]].

For PP=2, stage 0 (rank 0) sends to peer rank 1, and stage 1 (rank 1) receives from peer rank 0.

### NcclCommunicator P2P Methods

P2P data transfer between NCCL ranks for pipeline parallelism hidden state exchange. See [[crates/cuda/src/nccl.rs#NcclCommunicator]].

NcclCommunicator implements `Send` and `Sync` (unsafe) so it can be wrapped in `Arc` and shared across threads — a requirement for the PP engine which shares the communicator between stages.

`send(rank, data, peer)` sends a `CudaSlice` to the peer rank. `recv(rank, data, peer)` receives into a mutable buffer. Both lookup the comm by rank index and delegate to cudarc `Comm::send`/`recv`.

## Pipeline Stage Data Structures

Per-stage types for pipeline parallelism: stage identity, weights, KV cache, and GDN state management. See [[crates/parallelism/src/stage.rs]].

### PipelineStage

Holds a stage's ID, GPU assignment, layer range, sharded weights, and P2P communicator. See [[crates/parallelism/src/stage.rs#PipelineStage]].

`new()` takes stage index, GPU ID, layer boundaries, weight registry, and NCCL communicator. `num_layers()` returns layer count; `contains_layer()` checks layer membership. For PP=2, stage 0 covers layers 0-31 and stage 1 covers layers 32-63.

### GdnStateRef

Lightweight GDN state descriptor for tracking recurrent state allocation. `new()` creates an uninitialized state; `with_hidden_size()` sets the hidden dimension. `mark_initialized()` flags the state as GPU-initialized. See [[crates/parallelism/src/stage.rs#GdnStateRef]].

### StageState

Per-stage state management combining paged KV cache and GDN recurrent states. See [[crates/parallelism/src/stage.rs#StageState]].

`new()` initializes the `PagedKvManager` with cache parameters. `create_session()` allocates a new sequence ID. `ensure_gdn_states()` populates GDN state entries for all GDN layers in the stage's range. `free_session()` releases both KV and GDN resources. `num_sessions()` and `num_gdn_states()` report active counts.

## Microbatch Scheduler

Microbatch scheduler for pipeline parallelism. Splits incoming requests into microbatches and tracks progress through pipeline stages, keeping both GPUs busy by interleaving microbatches across stages. See [[crates/parallelism/src/microbatch.rs#MicrobatchScheduler]].

### Request

Simplified request type for pipeline parallelism holding an ID, token IDs, and session ID. See [[crates/parallelism/src/microbatch.rs#Request]].

### Microbatch

Group of requests processed together as a pipeline unit. Flows through stages sequentially — stage 0 produces hidden states, hidden states sent to stage 1, stage 1 produces logits, tokens sampled and microbatch completes. See [[crates/parallelism/src/microbatch.rs#Microbatch]].

### MicrobatchScheduler

Splits pending requests into microbatches of configured size and advances them through pipeline stages until completion. See [[crates/parallelism/src/microbatch.rs#MicrobatchScheduler]].

Methods: `new(microbatch_size)`, `add_request()`, `add_requests()`, `next_microbatch()`, `is_busy()`, `is_done()`, `pending_count()`, `in_flight_count()`, `advance_pipeline(num_stages)`, `reset()`.

## Pipeline Engine

Main orchestration module for PP=2 with microbatching. Assembles pipeline stages, manages the pipeline forward loop, and coordinates NCCL P2P send/recv between stages. See [[crates/parallelism/src/pp.rs#PipelineEngine]].

### PipelineEngine

Orchestrates PP=2 across two GPUs using stage partitioning, NCCL P2P communication, and microbatch scheduling to hide pipeline bubbles. See [[crates/parallelism/src/pp.rs#PipelineEngine]].

`new()` creates the engine: splits the model into two pipeline stages via `split_layers_pp`, wraps the NCCL communicator in `Arc` for sharing between stages, creates `PipelineStage` instances for each GPU, and initializes per-stage `StageState` for KV cache and GDN state management. `forward_batch()` splits requests into microbatches, processes them through the pipeline loop, and returns sampled tokens with timing. `create_sessions()` and `free_sessions()` manage lifecycle across both stages.

### PipelineTiming

Timing information for a single pipeline forward pass. Tracks total wall-clock time, per-GPU active compute time, and NCCL communication time. See [[crates/parallelism/src/pp.rs#PipelineTiming]].

`bubble_fraction()` computes idle fraction: `1 - (gpu0_active + gpu1_active) / (2 * total_time)`. A fraction of 0 means perfect GPU utilization; 1 means complete bubble.

### PipelineOutput

Result of a pipeline forward pass containing sampled token IDs for each request and timing information. See [[crates/parallelism/src/pp.rs#PipelineOutput]].

### compute_bubble_fraction

Theoretical bubble fraction for PP=2: `1 / (num_microbatches + 1)`. One microbatch gives 50% bubble; more microbatches reduce the fraction. See [[crates/parallelism/src/pp.rs#compute_bubble_fraction]].

## Tensor Parallelism Engine

TP=2 engine that shards weight tensors across GPUs and synchronizes activations via NCCL all-reduce. See [[crates/parallelism/src/tp.rs]].

### TensorParallelEngine

Manages NCCL all-reduce for tensor parallelism. See [[crates/parallelism/src/tp.rs#TensorParallelEngine]].

`new()` creates the engine with NCCL communicator from GPU streams. `all_reduce_attention()`, `all_reduce_mlp()`, and `all_reduce_gdn()` delegate to `NcclCommunicator::all_reduce` with sum operation. `all_reduce_in_place()` overwrites the input buffer with the reduced result.

### All-Reduce Operations

All-reduce after attention/GDN and MLP layers. See `all_reduce_attention()`, `all_reduce_mlp()`, `all_reduce_gdn()`, and `all_reduce_in_place()`.

## Unified Engine Dispatch

Single entry point for selecting between TP and PP parallelism strategies at load time. See [[crates/parallelism/src/engine.rs]].

### ParallelEngine

Enum wrapping either `TensorParallelEngine` or `PipelineEngine`. `select()` constructs the appropriate engine based on `ParallelismMode`. `forward_batch()` dispatches to the underlying engine's forward pass. See [[crates/parallelism/src/engine.rs#ParallelEngine]].

### ParallelismMode

Enum specifying parallelism strategy: `TensorParallel(n)` or `PipelineParallel(n)`. Default is TP=2. `is_tp()`, `is_pp()`, and `parallelism_degree()` provide inspection methods. See [[crates/parallelism/src/engine.rs#ParallelismMode]].
