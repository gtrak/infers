# Qwen3.6-27B Architecture Research

## Model Confirmed

**Model:** Qwen/Qwen3.6-27B  
**Release:** April 2026  
**License:** Apache-2.0  
**Downloads:** 5.2M/month  

## Architecture Parameters

From `config.json` (`https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json`):

| Parameter | Value |
|-----------|-------|
| `architectures` | `Qwen3_5ForConditionalGeneration` |
| `model_type` | `qwen3_5` |
| `num_hidden_layers` | 64 |
| `hidden_size` | 5120 |
| `intermediate_size` | 17408 |
| `vocab_size` | 248320 |
| `num_attention_heads` | 24 (query) |
| `num_key_value_heads` | 4 |
| `head_dim` | 256 |
| `max_position_embeddings` | 262144 |
| `sliding_window` | null |
| `use_sliding_window` | false |
| `rms_norm_eps` | 1e-06 |
| `hidden_act` | `silu` |
| `tie_word_embeddings` | false |
| `rope_theta` | 10000000.0 |
| `partial_rotary_factor` | 0.25 |
| `mrope_interleaved` | true |
| `mrope_section` | [11, 11, 10] |
| `mtp_num_hidden_layers` | 1 |
| `mtp_use_dedicated_embeddings` | false |

## Hybrid Attention Architecture

Qwen3.6 uses **two distinct attention mechanisms** in a repeating pattern:

### Layer Pattern

```python
layer_types = [
    "linear_attention" if bool((i + 1) % 4) else "full_attention"
    for i in range(64)
]
```

Result: 48 GDN layers, 16 full attention layers.

### Gated DeltaNet (GDN) — 48 layers

**Type:** Linear attention (not softmax-based)  
**Heads:** 48 for V, 16 for QK  
**Head dim:** 128  
**Convolution kernel dim:** 4  

**Key characteristics:**
- Uses recurrent state (`conv_state` + `ssm_state`) instead of traditional KV cache
- Cannot use standard PagedAttention
- State must be updated incrementally per token
- Not easily evictable to CPU/SSD

**vLLM implementation:** `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`

### Full Attention (Gated Attention) — 16 layers

**Type:** Standard softmax attention  
**Heads:** 24 Q, 4 KV  
**Head dim:** 256  
**RoPE dim:** 64  

**Key characteristics:**
- Standard transformer attention
- Uses Paged KV cache
- Supports custom CUDA kernels
- Evictable to CPU/SSD

**vLLM implementation:** `vllm/model_executor/models/qwen3_next.py`

## Custom CUDA Kernel Support

The project uses custom CUDA kernels compiled and loaded via the infers kernel pipeline.

### GDN Kernels

**Prefill:** `infers_gdn_prefill_bf16` (custom CUDA kernel)  
**Decode:** `infers_gdn_update_bf16` (custom CUDA kernel)

### Standard Attention

Custom attention kernels are implemented for prefill/decode paths using per-head weight slicing and online softmax.

### Sampling

Custom argmax sampling kernel (`infers_argmax_f32`) with greedy strategy.  

## KV Cache Requirements

### Two State Types Required

| Layer Type | State | Paged? | Evictable? |
|---|---|---|---|
| GDN (48 layers) | `conv_state` + `ssm_state` | No | No |
| Full Attention (16 layers) | Key + Value tensors | Yes | Yes |

### State Sizes (per sequence, per layer)

**GDN state:**
- `conv_state`: [batch, d_conv] — small, ~KB per layer
- `ssm_state`: [batch, d_state] — small, ~KB per layer

**Full attention KV:**
- Key: [num_tokens, num_kv_heads, head_dim] — grows with sequence length
- Value: [num_tokens, num_kv_heads, head_dim] — grows with sequence length

### Hybrid KV Manager Design

```rust
struct HybridKvManager {
    // Per-session Mamba states (for GDN layers)
    mamba_states: HashMap<SessionId, Vec<MambaState>>, // 48 layers per session
    
    // Paged KV cache (for full attention layers)
    paged_kv: PagedKvCache, // 16 layers, paged across all sessions
    
    // Memory pools
    gpu_free_blocks: Vec<BlockId>,
    cpu_free_blocks: Vec<BlockId>,
}
```

## Context Length

- **Native:** 262,144 tokens
- **Extensible:** 1,010,000 tokens via YaRN (not supported in Phase 1)
- **Default context:** 32,768 tokens for typical usage
- **Max concurrent sessions (262K context, TP=2, NVFP4 KV):** 2-3

## RoPE Configuration

```json
{
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10]
}
```

- Uses multi-dimensional RoPE (mRoPE)
- Position IDs have 3 dimensions (matching `mrope_section`)
- Must be applied differently for text vs vision tokens

## MTP (Multi-Token Prediction)

**Config:**
- `mtp_num_hidden_layers`: 1
- `mtp_use_dedicated_embeddings`: false

**Architecture:**
- Single MTP layer with full attention
- Shares embeddings with main model
- Generates draft tokens for speculative decoding
- vLLM command: `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`

## Chat Template

**Thinking mode (default):**
- Wraps reasoning in `<thinking>...</thinking>` blocks
- Uses special tokens: `<|im_start|>`, `<|im_end|>`

**Tool calls:**
- Supports XML-style tool calling
- Parser: `qwen3_xml` or `qwen3_coder`
- Format: `<tool_call>...</tool_call>`

**vLLM flags:**
- `--reasoning-parser qwen3`
- `--enable-auto-tool-choice`
- `--tool-call-parser qwen3_coder`

## References

1. Qwen3.6 Model Card: https://huggingface.co/Qwen/Qwen3.6-27B
2. vLLM Qwen3.6 Implementation: `../vllm/vllm/model_executor/models/qwen3_next.py`
3. vLLM GDN Implementation: `../vllm/vllm/model_executor/layers/mamba/gdn/`
4. vLLM Attention Backends: `../vllm/vllm/v1/attention/backends/`

## Cross-References

- See `quantization.md` for weight quantization formats
- See `kernels.md` for kernel compilation strategy
- See `parallelism.md` for TP/PP distribution of this architecture
- See `api.md` for OpenAI API tool call streaming format
