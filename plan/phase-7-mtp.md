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

Qwen3.6 has `mtp_num_hidden_layers: 1` — a single MTP layer that:
- Takes the hidden state from the main model
- Projects to a new set of logits
- Predicts future tokens greedily
- Acts as a draft for speculative decoding

```rust
pub struct MtpHead {
    pub layer: TransformerLayer,  // Single full attention layer
    pub lm_head: DeviceBuffer<half>,
}

impl MtpHead {
    pub fn from_weights(weights: &MtpWeights) -> Result<Self> {
        Ok(Self {
            layer: TransformerLayer::new(weights.layer)?,
            lm_head: weights.lm_head.clone(),
        })
    }
    
    pub fn forward(
        &self,
        hidden: &DeviceBuffer<half>,
        position: usize,
    ) -> Result<DeviceBuffer<half>> {
        // Single transformer layer
        let output = self.layer.forward(hidden, position)?;
        
        // LM head projection
        let logits = matmul(&output, &self.lm_head)?;
        
        Ok(logits)
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
    ) -> Result<Self> {
        Ok(Self {
            mtp_head: MtpHead::from_weights(mtp_weights)?,
            num_draft_tokens,
            acceptance_history: Vec::new(),
        })
    }
    
    /// Generate draft tokens from the MTP head
    pub fn generate_drafts(
        &self,
        hidden: &DeviceBuffer<half>,
        num_drafts: usize,
    ) -> Result<Vec<u32>> {
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut current_hidden = hidden.clone();
        
        for _ in 0..num_drafts {
            // Forward MTP head
            let logits = self.mtp_head.forward(&current_hidden, 0)?;
            
            // Greedy sample
            let token = self.greedy_sample(&logits)?;
            drafts.push(token);
            
            // Get token embedding for next draft
            current_hidden = self.embed_token(token)?;
        }
        
        Ok(drafts)
    }
    
    /// Verify draft tokens against main model
    pub fn verify_drafts(
        &self,
        main_model: &ForwardEngine,
        session: &mut Session,
        draft_tokens: &[u32],
    ) -> Result<VerificationResult> {
        // Prepare input: [confirmed_token, draft_token_1, draft_token_2, ...]
        let mut verification_tokens = vec![session.get_last_token()];
        verification_tokens.extend_from_slice(draft_tokens);
        
        // Run main model forward on all draft positions
        let mut all_logits = Vec::new();
        let mut current_hidden = session.get_last_hidden_state()?.clone();
        
        for (i, &token) in verification_tokens.iter().enumerate().skip(1) {
            // Embed token
            let embedded = main_model.embed_single(token)?;
            
            // Forward through main model (all layers)
            let hidden = main_model.forward_decode(&embedded, session)?;
            
            // Get logits
            let logits = main_model.lm_head(&hidden)?;
            all_logits.push(logits);
            
            current_hidden = hidden;
        }
        
        // Check which draft tokens match main model's prediction
        let mut accepted = 0;
        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let main_token = self.greedy_sample(&all_logits[i])?;
            
            if main_token == draft_token {
                accepted += 1;
            } else {
                break;
            }
        }
        
        // Accept accepted tokens, reject the rest
        let accepted_tokens = draft_tokens[..accepted].to_vec();
        let rejected_token = if accepted < draft_tokens.len() {
            Some(self.sample_from_logits(&all_logits[accepted])?)
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

```rust
impl ForwardEngine {
    pub fn decode_with_mtp(
        &self,
        session: &mut Session,
        mtp: &MtpEngine,
    ) -> Result<Vec<u32>> {
        let mut output_tokens = Vec::new();
        
        while session.num_generated_tokens < session.max_tokens {
            // 1. Get current hidden state from last token
            let hidden = session.get_last_hidden_state()?;
            
            // 2. Generate draft tokens from MTP
            let num_drafts = mtp.adaptive_num_drafts();
            let drafts = mtp.generate_drafts(&hidden, num_drafts)?;
            
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
