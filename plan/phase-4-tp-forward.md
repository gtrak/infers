# Phase 4: TP=2 Forward Pass

---
**Status**: PARTIAL
**Last Updated**: 2026-06-11
**Rationale**: Forward pass works end-to-end, produces tokens. BUT: Performance is 200× off target (0.1 vs 20 tok/s). No reference comparison. Only greedy sampling.
**Actual Deliverables**:
- [x] GDN prefill kernel integration
- [x] GDN decode kernel integration
- [x] Standard attention prefill/decode
- [x] Layer dispatch based on `layer_type`
- [x] GEMM via cuBLASLt
- [x] NCCL all-reduce after attention/MLP
- [x] RMSNorm + SiLU activation
- [x] RoPE position embedding
- [x] Prefill path: tokenize → allocate → forward → sample
- [x] Decode path: single token → update KV → forward → sample
- [x] KV cache allocation and distribution
- [ ] Performance target (>20 tok/s) — NOT MET (~0.1 tok/s)
- [ ] Reference comparison against HF
- [ ] Single-GPU parity verification
- [~] GDN works but not verified against reference
- [~] Sampling: only greedy (no temperature/top_p/top_k)
---

**Duration:** 3 weeks  
**Goal:** Implement the model forward pass with Tensor Parallelism across 2 GPUs.

## Deliverables

- [x] GDN prefill kernel integration (custom CUDA kernels)
- [x] GDN decode kernel integration (custom CUDA kernels)
- [x] Standard attention prefill/decode (custom CUDA kernels)
- [x] Layer dispatch based on `layer_type`
- [x] GEMM via cuBLASLt (NVFP4, FP16, BF16)
- [x] NCCL all-reduce after attention/MLP
- [x] RMSNorm + SiLU activation
- [x] RoPE position embedding
- [x] Prefill path: tokenize → allocate → forward → sample
- [x] Decode path: single token → update KV → forward → sample
- [x] KV cache allocation and distribution
- [ ] Performance target (>20 tok/s) — NOT MET (~0.1 tok/s)
- [ ] Reference comparison against HF
- [ ] Single-GPU parity verification
- [~] GDN works but not verified against reference
- [~] Sampling: only greedy (no temperature/top_p/top_k)

## Technical Details

### Forward Pass Architecture

```rust
pub struct ForwardEngine {
    pub config: Arc<ModelConfig>,
    pub weights: Vec<WeightRegistry>,  // One per GPU
    pub kernels: KernelRegistry,
    pub gemm: GemmEngine,
    pub nccl: NcclCommunicator,
    pub kv_manager: HybridKvManager,
    pub tokenizer: Tokenizer,
}

impl ForwardEngine {
    pub fn prefill(
        &self,
        session: &mut Session,
        prompt_tokens: &[u32],
    ) -> Result<u32> {
        // 1. Token embeddings
        let mut hidden = self.embed(session, prompt_tokens)?;
        
        // 2. Forward through all layers
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer_type = self.config.get_layer_type(layer_idx);
            
            match layer_type {
                LayerType::GatedDeltaNet => {
                    hidden = self.gdn_forward(
                        layer_idx, &hidden, session, ForwardMode::Prefill
                    )?;
                }
                LayerType::FullAttention => {
                    hidden = self.attention_forward(
                        layer_idx, &hidden, session, ForwardMode::Prefill
                    )?;
                }
            }
        }
        
        // 3. Final norm + LM head
        let logits = self.lm_head(&hidden)?;
        
        // 4. Sample first token
        let token = self.sample(&logits, &session.sampling_config)?;
        
        Ok(token)
    }
    
    pub fn decode(
        &self,
        session: &mut Session,
        input_token: u32,
    ) -> Result<u32> {
        // 1. Token embedding
        let mut hidden = self.embed_single(session, input_token)?;
        
        // 2. Forward through all layers
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer_type = self.config.get_layer_type(layer_idx);
            
            match layer_type {
                LayerType::GatedDeltaNet => {
                    hidden = self.gdn_forward(
                        layer_idx, &hidden, session, ForwardMode::Decode
                    )?;
                }
                LayerType::FullAttention => {
                    hidden = self.attention_forward(
                        layer_idx, &hidden, session, ForwardMode::Decode
                    )?
                }
            }
        }
        
        // 3. Final norm + LM head
        let logits = self.lm_head(&hidden)?;
        
        // 4. Sample next token
        let token = self.sample(&logits, &session.sampling_config)?;
        
        Ok(token)
    }
}
```

### GDN Forward

```rust
fn gdn_forward(
    &self,
    layer_idx: usize,
    hidden: &DeviceBuffer<half>,
    session: &mut Session,
    mode: ForwardMode,
) -> Result<DeviceBuffer<half>> {
    let gpu_id = self.nccl.rank;
    let weights = &self.weights[gpu_id].layers[layer_idx].gdn
        .as_ref()
        .ok_or_else(|| anyhow!("GDN weights not found for layer {}", layer_idx))?;
    
    // 1. RMSNorm
    let norm1_out = self.rms_norm(hidden, &weights.norm1)?;
    
    // 2. GDN attention
    let gdn_out = match mode {
        ForwardMode::Prefill => {
            self.gdn_prefill(&norm1_out, weights, session, layer_idx)?
        }
        ForwardMode::Decode => {
            self.gdn_decode(&norm1_out, weights, session, layer_idx)?
        }
    };
    
    // 3. Residual connection
    let attn_out = self.add(hidden, &gdn_out)?;
    
    // 4. RMSNorm
    let norm2_out = self.rms_norm(&attn_out, &weights.norm2)?;
    
    // 5. MLP
    let mlp_out = self.mlp_forward(&norm2_out, &self.weights[gpu_id].layers[layer_idx].mlp)?;
    
    // 6. Residual
    let output = self.add(&attn_out, &mlp_out)?;
    
    Ok(output)
}
```

### GDN Prefill (Custom CUDA Kernels)

```rust
fn gdn_prefill(
    &self,
    hidden: &DeviceBuffer<half>,
    weights: &GdnWeights,
    session: &mut Session,
    layer_idx: usize,
) -> Result<DeviceBuffer<half>> {
    let batch_size = 1;  // Single session for now
    let seq_len = session.num_tokens();
    
    // Project to q, k, v
    let q = self.gemm(&hidden, &weights.in_proj_a)?;
    let k = self.gemm(&hidden, &weights.in_proj_b)?;
    let v = self.gemm(&hidden, &weights.in_proj_b)?;  // Shared projection
    
    // Load or create Mamba state
    let mamba_state = session.get_or_create_mamba_state(layer_idx)?;
    
    // Launch GDN prefill kernel
    let mut output = self.device.alloc(hidden.len())?;
    
    self.kernels.gdn_prefill.launch(
        &self.stream,
        dim3((seq_len + 127) / 128, 1, 1),
        dim3(128, 1, 1),
        0,
        &(
            &mut output,
            &q, &k, &v,
            &mamba_state.conv_state,
            &mamba_state.ssm_state,
            batch_size,
            seq_len,
        ),
    )?;
    
    // Update Mamba state
    session.update_mamba_state(layer_idx, mamba_state)?;
    
    Ok(output)
}
```

### Standard Attention Forward

```rust
fn attention_forward(
    &self,
    layer_idx: usize,
    hidden: &DeviceBuffer<half>,
    session: &mut Session,
    mode: ForwardMode,
) -> Result<DeviceBuffer<half>> {
    let gpu_id = self.nccl.rank;
    let weights = &self.weights[gpu_id].layers[layer_idx].attn
        .as_ref()
        .ok_or_else(|| anyhow!("Attention weights not found"))?;
    
    // 1. RMSNorm
    let norm_out = self.rms_norm(hidden, &self.weights[gpu_id].layers[layer_idx].norm1)?;
    
    // 2. QKV projection
    let q = self.gemm(&norm_out, &weights.q_proj)?;
    let k = self.gemm(&norm_out, &weights.k_proj)?;
    let v = self.gemm(&norm_out, &weights.v_proj)?;
    
    // 3. Apply RoPE
    let (q_rot, k_rot) = self.apply_rope(&q, &k, session.positions())?;
    
    // 4. KV cache management
    match mode {
        ForwardMode::Prefill => {
            // Allocate KV blocks for all prompt tokens
            let kv_blocks = session.allocate_kv_blocks(layer_idx, session.num_tokens())?;
            
            // Write KV to paged cache
            self.write_kv_cache(&k_rot, &v_rot, &kv_blocks, layer_idx)?;
            
            // Custom CUDA kernel prefill
            let attn_out = self.flashinfer_prefill(&q_rot, &kv_blocks)?;
            
            // Update session KV metadata
            session.set_kv_blocks(layer_idx, kv_blocks)?;
            
            attn_out
        }
        ForwardMode::Decode => {
            // Append KV for single token
            let kv_block = session.get_kv_block(layer_idx, session.current_position())?;
            self.append_kv_cache(&k_rot, &v_rot, &kv_block, layer_idx)?;
            
            // Custom CUDA kernel decode
            let attn_out = self.flashinfer_decode(&q_rot, &kv_block)?;
            
            attn_out
        }
    }
}
```

### TP Synchronization

```rust
fn sync_attention_output(
    &self,
    output: &mut DeviceBuffer<half>,
) -> Result<()> {
    // All-reduce attention output across GPUs
    self.nccl.all_reduce(
        output.as_slice(),
        output.as_mut_slice(),
        &ReduceOp::Sum,
    )?;
    Ok(())
}

fn sync_mlp_output(
    &self,
    output: &mut DeviceBuffer<half>,
) -> Result<()> {
    // All-reduce MLP output across GPUs
    self.nccl.all_reduce(
        output.as_slice(),
        output.as_mut_slice(),
        &ReduceOp::Sum,
    )?;
    Ok(())
}
```

### Sampling

```rust
fn sample(
    &self,
    logits: &DeviceBuffer<half>,
    config: &SamplingConfig,
) -> Result<u32> {
    match config.strategy {
        SamplingStrategy::Greedy => {
            self.greedy_sample(logits)
        }
        SamplingStrategy::Temperature { temp } => {
            self.temperature_sample(logits, temp)
        }
        SamplingStrategy::TopK { k, temp } => {
            self.topk_sample(logits, k, temp)
        }
        SamplingStrategy::TopP { p, temp } => {
            self.topp_sample(logits, p, temp)
        }
    }
}
```

## File Structure

```
crates/backends/native/
  Cargo.toml
  src/
    lib.rs
    engine.rs           # ForwardEngine
    prefill.rs          # Prefill path
    decode.rs           # Decode path
    gdn.rs              # GDN forward (prefill + decode)
    attention.rs        # Standard attention forward
    mlp.rs              # MLP/GEMM forward
    norm.rs             # RMSNorm
    rope.rs             # RoPE application
    sample.rs           # Sampling strategies
    embedding.rs        # Token embedding
    sync.rs             # NCCL synchronization for TP
```

## Testing

### Correctness Tests

```rust
#[test]
fn test_forward_consistency() {
    // Compare our output with reference (vLLM or PyTorch)
    let prompt = "The capital of France is";
    let tokens = tokenizer.encode(prompt).unwrap();
    
    let our_output = engine.prefill(&mut session, &tokens).unwrap();
    let ref_output = reference_model.forward(&tokens).unwrap();
    
    // Allow small floating point differences
    assert!(tensor_allclose(&our_output, &ref_output, 1e-2));
}

#[test]
fn test_tp_correctness() {
    // Verify TP produces same results as single GPU
    let tp_output = tp_engine.prefill(&mut session, &tokens).unwrap();
    let single_output = single_engine.prefill(&mut session, &tokens).unwrap();
    
    assert!(tensor_allclose(&tp_output, &single_output, 1e-3));
}
```

## Dependencies

### Phase 4 → Phase 2

Uses `KernelRegistry`, `GemmEngine`, `NcclCommunicator`, `CudaStream`.

### Phase 4 → Phase 3

Uses `WeightRegistry`, `ModelConfig`, `LayerWeights`.

### Phase 4 → Phase 6

Continuous batching will call `prefill` and `decode` in loops.

## Success Criteria

1. Single forward pass completes without errors
2. Prefill produces correct first token (vs reference)
3. Decode produces correct next token
4. TP=2 produces identical results to single GPU
5. GDN layers update Mamba state correctly
6. Full attention layers write/read KV cache correctly
7. Sampling respects temperature/top_k/top_p
8. Performance: >20 tok/s for single request decode

## Cross-References

- **Research:** See `../research/kernels.md` for kernel compilation strategy (deprecated FlashInfer notes)
- **Research:** See `../research/parallelism.md` for TP synchronization details
- **Phase 2:** Uses compiled kernels and CUDA runtime
- **Phase 3:** Uses loaded weights and config
- **Phase 6:** Continuous batching will integrate this forward pass

## Open Questions

1. Should we fuse RMSNorm + projection into single kernel?
2. Should we use CUDA graphs for decode (latency optimization)?
3. How to handle different batch sizes in TP (all GPUs must have same batch)?
