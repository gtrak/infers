# Phase 7: MTP (Multi-Token Prediction)

**Duration:** 2 weeks  
**Goal:** Implement native MTP speculative decoding for Qwen3.6-27B.

## Deliverables

1. MTP head weight loading
2. MTP forward pass (reuse standard attention kernel)
3. Draft token generation (greedy sampling from MTP head)
4. Verification: main model checks draft tokens in single forward pass
5. Acceptance logic: accept longest valid prefix
6. `speculative-config` API parameter
7. MTP metrics (acceptance rate, tokens saved)

## Technical Details

### MTP Architecture

Qwen3.6 has `mtp_num_hidden_layers: 1` — the MTP head is a full transformer
layer that predicts future tokens from the main model's hidden state:

1. Normalizes the input embedding (`pre_fc_norm_embedding`) and the main model's
   hidden state (`pre_fc_norm_hidden`)
2. Concatenates them and projects through an FC layer to `hidden_size`
3. Passes through one full transformer decoder layer (self-attention + MLP)
4. Applies final layer norm
5. Projects to logits via the shared LM head (same weights as main model)

```rust
use crate::forward::LayerWeights;

/// MTP head: full transformer layer that predicts the next token from
/// the main model's hidden state + the MTP input embedding.
///
/// Architecture (from MtpWeights):
/// 1. pre_fc_norm_embedding → normalize input embedding
/// 2. pre_fc_norm_hidden → normalize main model hidden state
/// 3. fc → concat and project [embed, hidden] → hidden_size
/// 4. Full decoder layer (attention + MLP)
/// 5. norm → final layer norm
/// 6. LM head → logits (shared with main model)
pub struct MtpHead {
    /// Pre-FC norm for the token embedding.
    pub pre_fc_norm_embedding: WeightData,
    /// Pre-FC norm for the main model's hidden state.
    pub pre_fc_norm_hidden: WeightData,
    /// FC projection: concat([embed_norm, hidden_norm]) → hidden_size.
    pub fc_weight: CudaSlice<bf16>,
    /// Full transformer decoder layer (attention + MLP/MoE).
    pub layer: LayerWeights,
    /// Final post-layer norm.
    pub norm: WeightData,
    /// Whether to use dedicated MTP embeddings (default: false = share main model).
    pub use_dedicated_embeddings: bool,
}

impl MtpHead {
    pub fn from_weights(weights: &MtpWeights, stream: &Arc<CudaStream>) -> Result<Self> {
        let fc_weight = upload_weight(stream, &weights.fc)?;
        let layer = LayerWeights::from_mtp_weights(weights, stream)?;
        Ok(Self {
            pre_fc_norm_embedding: weights.pre_fc_norm_embedding.clone(),
            pre_fc_norm_hidden: weights.pre_fc_norm_hidden.clone(),
            fc_weight,
            layer,
            norm: weights.norm.clone(),
            use_dedicated_embeddings: weights.embed_tokens.is_some(),
        })
    }

    /// Forward MTP head: produce logits for the next token.
    ///
    /// # Arguments
    /// * `hidden` — Main model's hidden state `[hidden_size]`
    /// * `input_token` — Token ID to embed (input to MTP)
    /// * `embed_fn` — Function to embed token IDs (main model's embedding)
    /// * `stream` — CUDA stream for kernel launches
    pub fn forward(
        &self,
        hidden: &CudaSlice<bf16>,
        input_token: u32,
        embed_fn: impl Fn(u32, &CudaStream) -> Result<CudaSlice<bf16>>,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<bf16>> {
        // 1. Embed the input token (shared with main model)
        let embedding = embed_fn(input_token, stream)?;

        // 2. Norms
        let embed_norm = rms_norm(stream, &embedding, &self.pre_fc_norm_embedding)?;
        let hidden_norm = rms_norm(stream, hidden, &self.pre_fc_norm_hidden)?;

        // 3. Concat and FC project: [embed_norm | hidden_norm] @ fc_weight
        let concat = concat_bf16(stream, &[&embed_norm, &hidden_norm])?;
        let mut projected = stream.alloc_zeros::<bf16>(hidden.len())?;
        gemm::matmul_bf16(
            &GemmConfig { m: 1, n: hidden.len(), k: 2 * hidden.len(), ... },
            &concat, &self.fc_weight, &mut projected,
        )?;

        // 4. Full decoder layer (attention + MLP)
        let mut layer_out = self.layer.forward(&projected, stream)?;

        // 5. Final norm → logits via shared LM head (done by caller)
        let output = rms_norm(stream, &layer_out, &self.norm)?;
        Ok(output)
    }
}
```

### MTP Engine

```rust
pub struct MtpEngine {
    pub mtp_head: MtpHead,
    pub num_draft_tokens: usize,  // 1-4 (2 recommended)
    pub acceptance_history: Vec<bool>,
}

impl MtpEngine {
    pub fn new(
        mtp_weights: &MtpWeights,
        num_draft_tokens: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<Self> {
        Ok(Self {
            mtp_head: MtpHead::from_weights(mtp_weights, stream)?,
            num_draft_tokens,
            acceptance_history: Vec::new(),
        })
    }

    /// Generate draft tokens from the MTP head.
    ///
    /// Iteratively runs the MTP head: embed token → MTP forward → sample.
    /// The LM head projection uses the main model's LM head (shared weights).
    pub fn generate_drafts(
        &self,
        hidden: &CudaSlice<bf16>,
        num_drafts: usize,
        main_model: &ForwardEngine,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<u32>> {
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_hidden = hidden.clone();
        let mut current_token = main_model.last_token(); // last generated token

        for _ in 0..num_drafts {
            // Forward MTP head: produces hidden state
            let mtp_hidden = self.mtp_head.forward(
                &current_hidden,
                current_token,
                |token, s| main_model.embed(token, s),
                stream,
            )?;

            // LM head projection (shared with main model)
            let logits = main_model.lm_head_projection(&mtp_hidden, stream)?;

            // Greedy sample
            let token = sample::greedy_sample(stream, main_model.argmax_kernel(), &logits)?;
            drafts.push(token);

            current_token = token;
            current_hidden = mtp_hidden;
        }

        Ok(drafts)
    }
    
    /// Verify draft tokens against main model.
    ///
    /// **Prerequisite:** ForwardEngine must expose a method that returns
    /// per-token hidden states (not just sampled tokens). Currently
    /// `ForwardEngine::decode()` returns only `u32` — the sampled token.
    ///
    /// This requires adding a method like:
    ///   `ForwardEngine::decode_with_hidden(token, position, seq) -> (u32, CudaSlice<bf16>)`
    /// which returns both the sampled token and the final hidden state.
    ///
    /// Until that exists, MTP verification must run the main model's forward
    /// pass separately for each draft position, embedding → layer loop → LM head.
    pub fn verify_drafts(
        &self,
        main_model: &ForwardEngine,
        draft_position: usize,
        draft_tokens: &[u32],
        hidden_state: &CudaSlice<bf16>,
        stream: &Arc<CudaStream>,
    ) -> Result<VerificationResult> {
        let mut all_logits = Vec::new();
        let mut current_hidden = hidden_state.clone();

        for &draft_token in draft_tokens {
            // Embed the draft token (uses main model's embedding table)
            let embedded = main_model.embed_single(draft_token, stream)?;

            // Forward through all layers: RMSNorm → GDN/Attention → MLP → residual
            // Returns the hidden state at the final layer (pre LM head)
            let hidden = main_model.forward_layer_loop(&embedded, stream)?;

            // Project to logits via shared LM head
            let logits = main_model.lm_head_projection(&hidden, stream)?;
            all_logits.push(logits);

            current_hidden = hidden;
        }

        // Check which draft tokens match main model's prediction
        let mut accepted = 0;
        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let main_token = sample::greedy_sample(stream, main_model.argmax_kernel(), &all_logits[i])?;

            if main_token == draft_token {
                accepted += 1;
            } else {
                break;
            }
        }

        // Accept accepted tokens, reject the rest
        let accepted_tokens = draft_tokens[..accepted].to_vec();
        let rejected_token = if accepted < draft_tokens.len() {
            let logits = &all_logits[accepted];
            Some(sample::greedy_sample(stream, main_model.argmax_kernel(), logits)?)
        } else {
            None
        };

        Ok(VerificationResult {
            accepted_tokens,
            rejected_token,
            acceptance_rate: accepted as f32 / draft_tokens.len() as f32,
        })
    }
}
```

### Acceptance Logic

```rust
pub struct VerificationResult {
    pub accepted_tokens: Vec<u32>,
    pub rejected_token: Option<u32>,
    pub acceptance_rate: f32,
}

impl MtpEngine {
    /// Accept longest valid prefix
    pub fn accept_prefix(
        &self,
        result: &VerificationResult,
        session: &mut Session,
    ) -> Result<Vec<u32>> {
        let mut output_tokens = Vec::new();
        
        // Add all accepted tokens
        for &token in &result.accepted_tokens {
            session.append_token(token);
            output_tokens.push(token);
        }
        
        // Add corrected token for first rejection
        if let Some(token) = result.rejected_token {
            session.append_token(token);
            output_tokens.push(token);
        }
        
        // Update metrics
        self.acceptance_history.push(result.acceptance_rate > 0.0);
        
        Ok(output_tokens)
    }
    
    /// Dynamic draft token count based on recent acceptance rate
    pub fn adaptive_num_drafts(&self) -> usize {
        if self.acceptance_history.len() < 10 {
            return self.num_draft_tokens;
        }
        
        let recent_rate: f32 = self.acceptance_history
            .iter()
            .rev()
            .take(10)
            .filter(|&&x| x)
            .count() as f32 / 10.0;
        
        if recent_rate > 0.8 {
            (self.num_draft_tokens + 1).min(4)
        } else if recent_rate < 0.3 {
            (self.num_draft_tokens.saturating_sub(1)).max(1)
        } else {
            self.num_draft_tokens
        }
    }
}
```

### Integration with Decode Loop

**Prerequisite:** ForwardEngine needs to expose hidden states. Currently
`decode()` returns only a `u32` token ID. For MTP, we need a method like:

```rust
impl ForwardEngine {
    /// Decode a single token and return both the next token and the
    /// final hidden state (pre-LM-head) for MTP drafting.
    pub fn decode_with_hidden(
        &mut self,
        stream: &Arc<CudaStream>,
        token_id: u32,
        position: u32,
        seq_id: SequenceId,
    ) -> Result<(u32, CudaSlice<bf16>)> {
        // Same as decode(), but return the final hidden state
        // before LM head projection
    }
}
```

Once that exists, the MTP decode loop looks like:

```rust
impl ForwardEngine {
    pub fn decode_with_mtp(
        &mut self,
        stream: &Arc<CudaStream>,
        token_id: u32,
        position: u32,
        seq_id: SequenceId,
        mtp: &MtpEngine,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut output_tokens = Vec::new();
        let mut current_token = token_id;
        let mut current_pos = position;

        while output_tokens.len() < max_tokens {
            // 1. Get current hidden state from main model
            let (sampled_token, hidden_state) = self.decode_with_hidden(
                stream, current_token, current_pos, seq_id,
            )?;
            output_tokens.push(sampled_token);
            current_pos += 1;

            // 2. Generate draft tokens from MTP
            let num_drafts = mtp.adaptive_num_drafts();
            let drafts = mtp.generate_drafts(
                &hidden_state, num_drafts, self, stream,
            )?;

            // 3. Verify drafts with main model
            let verification = mtp.verify_drafts(
                self, current_pos, &drafts, &hidden_state, stream,
            )?;

            // 4. Accept/reject
            let accepted = mtp.accept_prefix(&verification);
            output_tokens.extend(accepted);
            current_pos += accepted.len() as u32;

            // Check for stop
            if let Some(&token) = accepted.last() {
                if token == self.tokenizer_eos_id() {
                    break;
                }
            }
        }

        Ok(output_tokens)
    }
}
            
            // 3. Verify drafts with main model
            let verification = mtp.verify_drafts(self, session, &drafts)?;
            
            // 4. Accept/reject
            let accepted = mtp.accept_prefix(&verification, session)?;
            output_tokens.extend(accepted);
            
            // Check for stop condition
            if let Some(stop_token) = accepted.last() {
                if *stop_token == self.tokenizer.eos_token_id() {
                    break;
                }
            }
        }
        
        Ok(output_tokens)
    }
}
```

### API Integration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    pub method: String,  // "mtp", "eagle", "medusa"
    pub num_speculative_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    // ... other fields ...
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative_config: Option<SpeculativeConfig>,
}
```

**vLLM equivalent:**
```bash
vllm serve Qwen/Qwen3.6-27B \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

## Performance Expectations

Based on Lorbus AutoRound model card:
- **With MTP (k=1):** ~58 tok/s (single request)
- **Without MTP:** ~32 tok/s
- **Speedup:** ~1.8x
- **Acceptance rate:** 80-90% (typical)

With `num_speculative_tokens=2`:
- If acceptance rate = 85%: speedup ≈ 1.7x
- If acceptance rate = 50%: speedup ≈ 1.2x
- If acceptance rate = 30%: speedup ≈ 1.0x (no benefit)

## File Structure

```
crates/mtp/
  Cargo.toml
  src/
    lib.rs
    head.rs             # MtpHead, single transformer layer
    engine.rs           # MtpEngine (draft + verify)
    verify.rs           # Verification logic
    metrics.rs          # Acceptance rate tracking
```

## Testing

### Acceptance Rate Test

```rust
#[test]
fn test_mtp_acceptance_rate() {
    let mtp = MtpEngine::new(&weights, 2)?;
    let mut session = Session::new("Hello", 100);
    
    // Run 100 decode steps
    let mut accepted = 0;
    let mut total = 0;
    
    for _ in 0..100 {
        let drafts = mtp.generate_drafts(&session.hidden_state, 2)?;
        let verification = mtp.verify_drafts(&main_model, &mut session, &drafts)?;
        
        accepted += verification.accepted_tokens.len();
        total += drafts.len();
    }
    
    let rate = accepted as f32 / total as f32;
    assert!(rate > 0.5, "Acceptance rate too low: {}", rate);
}
```

### Speedup Test

```rust
#[test]
fn test_mtp_speedup() {
    let prompt = "Explain quantum computing";
    let tokens = tokenizer.encode(prompt)?;
    
    // Without MTP
    let start = Instant::now();
    let without_mtp = engine.decode_without_mtp(&mut session, &tokens, 100)?;
    let time_without = start.elapsed();
    
    // With MTP
    let start = Instant::now();
    let with_mtp = engine.decode_with_mtp(&mut session, &tokens, 100)?;
    let time_with = start.elapsed();
    
    let speedup = time_without.as_secs_f32() / time_with.as_secs_f32();
    assert!(speedup > 1.2, "MTP speedup too low: {}", speedup);
}
```

## Dependencies

### Phase 7 → Phase 4

Uses main model forward pass for verification.

### Phase 7 → Phase 3

Uses MTP weights from model loading.

## Success Criteria

1. MTP generates draft tokens correctly
2. Verification matches main model's output
3. Acceptance rate > 50% on typical prompts
4. Speedup > 1.2x with num_drafts=2
5. API accepts `speculative_config` parameter
6. Metrics track acceptance rate and tokens saved

## Cross-References

- **Research:** See `../research/architecture.md` for MTP config details
- **Phase 3:** MTP weights loaded by model loader
- **Phase 4:** Main model forward pass used for verification
- **Phase 9:** Tool calls work with speculative decoding

## Open Questions

1. Should we implement Eagle/Medusa as alternatives to native MTP?
2. How to handle MTP with continuous batching (multiple sessions)?
3. Should we cache MTP hidden states for efficiency?
