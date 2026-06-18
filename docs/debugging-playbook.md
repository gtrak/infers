# GDN Output Debugging Playbook

## Current Status

The engine produces garbage after 64 layers. "Paris" token ranks #11 instead of #1.
All individual GDN stages are correct (cos ≥ 0.99) but magnitude errors accumulate.

Layer 0 output: cos=0.991, **ratio=1.10** (10% too large).
Layer 1 input (from Layer 0 output): cos=0.991, ratio=1.10.
This 10% compounds: 1.10^64 ≈ 340× magnitude error after 64 layers.

## Tooling

### Generate oracle dumps (HF ground truth)
```bash
cd /home/gary/dev/infers
python3 -m tests.compare.oracle \
  --model-dir /home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4 \
  --token-ids 248045,846,198,3710,369,279,6511,314,9338,30,248046,198,248045,74455,198,3710,9338 \
  --output-dir /tmp/hf_oracle_v3
```

### Generate engine dumps
```bash
# Set these env vars before running the engine:
INFERS_DUMP_DIR=/tmp/dump_v3
INFERS_DUMP_LAYERS=0,1,2,3
INFERS_DUMP_STAGES=gdn,mlp,residual
```

### Compare engine vs oracle
```bash
python3 -m tests.compare.hf_compare \
  --oracle-dir /tmp/hf_oracle_v3 \
  --engine-dir /tmp/dump_v3 \
  --phase prefill \
  --threshold 0.98
```

### Manual per-stage comparison (Python)
```python
import torch, json
from tests.compare.io import load_raw_bf16

def load_engine(path, meta):
    with open(meta) as f:
        shape = tuple(json.load(f)['shape'])
    with open(path, 'rb') as f:
        return torch.frombuffer(f.read(), dtype=torch.bfloat16).reshape(shape).float().clone()

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

# Example: compare layer 0 norm1_input
eng = load_engine('/tmp/dump_v3/layer_0/prefill/gdn.norm1_input_gpu0.raw',
                  '/tmp/dump_v3/layer_0/prefill/gdn.norm1_input_gpu0.meta')
hf = torch.load('/tmp/hf_oracle_v3/layer_0/prefill/norm1_input.pt', map_location='cpu').float().squeeze(0)
print(f"cos={cos(eng, hf):.6f}  ratio={eng.norm().item()/hf.norm().item():.4f}")
```

## Known Facts

### Engine correctness per stage (Layer 0, TP=2)
| Stage | cos | ratio | Notes |
|-------|-----|-------|-------|
| norm1_input | 1.000 | 1.000 | Perfect |
| norm1_output | 1.000 | 1.000 | Perfect |
| mixed_qkv | 0.994 | 0.980 | INT4 quantization noise |
| conv_out | 0.9996 | 0.965 | Correct |
| query/key/value | 0.995-1.000 | 0.92-0.97 | INT4 noise |
| core_attn_out (GPU0) | 0.999 | ~1.0 | Perfect per-GPU |
| core_attn_out (GPU1) | 0.931 | ~0.93 | GPU1 worse — INT4 + recurrence |
| a_proj/b_proj | 1.000 | 1.000 | BF16 weights, perfect |
| z_gate | 0.989 | 0.99 | INT4 noise |
| norm_output | 0.991 | 1.02 | Correct |
| o_proj (SUM) | 0.989 | 1.02 | Row-parallel sum |
| after_ar | 0.989 | 1.02 | Post all-reduce |
| norm2 (unweighted) | 0.988 | — | Weighted cos=0.708 is ARTIFACT, not bug |
| **Layer 0 output** | **0.991** | **1.10** | **10% magnitude error** |

### Key architecture facts
- Qwen3_5RMSNorm weight = `1.0 + stored_weight` (stored weight is zero-initialized residual)
- GDN fused QKV uses segment sharding: per-GPU layout = [Q_part, K_part, V_part]
- Comparison must use segment-aware reconstruction (not naive cat)
- core_attn_out oracle shape = [seq_len*num_v_heads, head_v_dim] → reshape to [seq_len, num_v_heads*head_v_dim]

## Investigation Plan: Find the Magnitude Source

The 10% magnitude error per layer is the critical bug. Find WHERE it comes from.

### Step 1: Isolate which component adds the 10%

The layer output = residual + MLP_output. The 10% must come from either:
- A: The residual (which is input + GDN_output) already has 10% error
- B: The MLP_output is 10% too large

Check which by computing:
```python
# Layer 0 components
residual_attn = load_engine('.../residual.attn_gpu0.raw', ...)  # = input + gdn_output
oracle_residual = torch.load('.../norm2_input.pt', ...).squeeze(0)

# residual.attn should = norm1_input + gdn_after_ar
# cos should be ~0.989, ratio should be ~1.02
# But the LAYER output ratio is 1.10

# Check: is the 10% from the residual or the MLP?
mlp_output_eng = load_engine('.../mlp.down_ar_gpu0.raw', ...) + load_engine('.../mlp.down_ar_gpu1.raw', ...)
mlp_output_hf = torch.load('.../mlp_output.pt', ...).squeeze(0)

# Key metric: what's the ratio of mlp_output?
# If mlp_output ratio >> 1.10, the MLP is amplifying the error
# If mlp_output ratio ≈ 1.10, the error is already in the residual
```

### Step 2: Check if error is additive or multiplicative

If each layer multiplies the error by 1.10, then after N layers the ratio = 1.10^N.
If each layer ADDS 10% of the residual norm, the ratio grows linearly.

Run comparison for layers 0, 1, 2, 3 to check:
```bash
# Dump layers 0-3
INFERS_DUMP_LAYERS=0,1,2,3

# Then compare ratio for each layer's output
python3 -c "
import torch
from tests.compare.io import load_raw_bf16
for layer in range(4):
    eng = load_engine(f'/tmp/dump_v3/layer_{layer}/prefill/gdn.norm1_input_gpu0.raw',
                      f'/tmp/dump_v3/layer_{layer}/prefill/gdn.norm1_input_gpu0.meta')
    hf = torch.load(f'/tmp/hf_oracle_v3/layer_{layer}/prefill/norm1_input.pt', map_location='cpu').float().squeeze(0)
    cos = torch.nn.functional.cosine_similarity(eng.flatten().unsqueeze(0), hf.flatten().unsqueeze(0)).item()
    ratio = eng.norm().item() / hf.norm().item()
    print(f'Layer {layer} input: cos={cos:.4f} ratio={ratio:.4f}')
"
```

Expected pattern:
- Multiplicative: ratio grows exponentially (1.10, 1.21, 1.33, 1.46)
- Additive: ratio grows linearly

### Step 3: Check if INT4 dequantization is the root cause

The mixed_qkv GEMM has ratio=0.98 (engine 2% smaller than oracle). But the layer OUTPUT has ratio=1.10 (10% larger). Something is AMPLIFYING the error.

The key suspect: the **GDN recurrent state** accumulates errors across tokens. Each token's attention output depends on all previous tokens' states. If the state drifts by even 1% per token, after 19 tokens the drift is (1.01)^19 ≈ 1.21.

Check by comparing per-token error growth:
```python
# For each token t, compare engine vs oracle at core_attn_out
for t in range(19):
    eng_t = eng_core_attn[t*tokens_per_token:(t+1)*tokens_per_token]
    hf_t = hf_core_attn[t*tokens_per_token:(t+1)*tokens_per_token]
    cos_t = cos(eng_t, hf_t)
    ratio_t = eng_t.norm() / hf_t.norm()
    print(f'token {t}: cos={cos_t:.4f} ratio={ratio_t:.4f}')
```

If the ratio grows with token position, the recurrent state is drifting.

### Step 4: Check the RMSNorm weight loading

The engine must load norm weights as `1.0 + stored_weight`. Verify:
```bash
# Check what the engine actually loads for norm2 weight
# Look in crates/model/src/loader.rs for how norm weights are loaded
grep -n "post_attention_layernorm\|norm_weight\|rms_norm" crates/model/src/loader.rs
```

If the engine loads the raw safetensors value (≈0, range -1 to +0.2) instead of 1.0+weight, the norm output will be nearly zero — which matches the low RMS we saw in oracle output.

Actually, we already verified the engine IS correct (cos=0.999999 vs manual). So the weight loading is fine.

### Step 5: Check the MLP gate/up weight loading

The MLP uses INT4 weights for gate_proj and up_proj. If these are loaded with wrong scales or zero-points, the MLP output will have magnitude errors.

Compare engine gate_proj/up_proj against HF dequantized weights:
```python
# Load INT4 weight from engine (via cache or dump)
# Compare against HF dequantized weight
# Check scale and zero-point values
```

### Step 6: If magnitude source is found, fix it

Most likely fixes:
- **Wrong INT4 scale/zero-point**: Fix weight loading in `crates/model/src/loader.rs`
- **Wrong RMSNorm formula**: Already verified correct
- **Accumulating recurrent state drift**: May need higher-precision state (fp32 vs bf16)
- **MLP weight dequantization bug**: Fix INT4 GEMM for MLP weights

## Quick Diagnostic Commands

```bash
# Full layer 0 comparison
python3 -m tests.compare.hf_compare \
  --oracle-dir /tmp/hf_oracle_layer1_v2 \
  --engine-dir /tmp/dump_layer0_v2 \
  --phase prefill --threshold 0.98

# Check magnitude ratio per layer (need dumps for layers 0-3)
for layer in 0 1 2 3; do
  python3 -c "
import torch
from tests.compare.io import load_raw_bf16
eng = load_raw_bf16('/tmp/dump_v3/layer_${layer}/prefill/gdn.norm1_input_gpu0.raw',
                     '/tmp/dump_v3/layer_${layer}/prefill/gdn.norm1_input_gpu0.meta')
hf = torch.load('/tmp/hf_oracle_v3/layer_${layer}/prefill/norm1_input.pt', map_location='cpu').float().squeeze(0)
print(f'Layer ${layer}: cos={torch.nn.functional.cosine_similarity(eng.flatten().unsqueeze(0), hf.flatten().unsqueeze(0)).item():.4f} ratio={eng.norm().item()/hf.norm().item():.4f}')
"
done
```

## Important Notes

1. **Always use `--release`** for Rust builds and tests. Debug builds are ~100x slower.
2. **The norm2 cos=0.708 is NOT a bug.** It's a weighted cosine similarity artifact. The true norm2 accuracy is cos=0.988 (unweighted normalized vectors).
3. **The mixed_qkv segment-aware reconstruction is critical.** Naive `cat([GPU0, GPU1])` gives wrong column ordering for fused QKV projections.
4. **The Qwen3_5RMSNorm weight formula is `1.0 + stored_weight`**, NOT just `stored_weight`.
5. **`core_attn_out` oracle shape is `[seq_len*num_v_heads, head_v_dim]`**, not `[seq_len, num_v_heads*head_v_dim]`. Must reshape before comparison.
