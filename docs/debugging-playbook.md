# GDN Output Debugging Playbook

## Current Status

The engine produces garbage after 64 layers. "Paris" token ranks #11 instead of #1.
All individual GDN stages are correct (cos ≥ 0.99) but magnitude errors accumulate.

Layer 0 output: cos=0.991, **ratio=1.10** (10% too large).
Layer 1 input (from Layer 0 output): cos=0.991, ratio=1.10.
This 10% compounds: 1.10^64 ≈ 340× magnitude error after 64 layers.

## Investigation Results (Steps 1-5 COMPLETED)

### Root Cause: RMSNorm weight-direction amplification

**Every engine kernel is correct.** The norm2 kernel matches manual computation at cos=0.999999 per-token. The problem is a mathematical amplification effect:

1. INT4 quantization introduces small directional errors (cos≈0.99 per GEMM)
2. RMSNorm normalizes to unit-RMS, preserving these directional differences
3. The non-uniform RMSNorm weight (range 0.5-1.0) amplifies directional differences into L2 magnitude differences
4. When engine's normed vector aligns better with high-weight dimensions than oracle's, `||normed_eng * w|| >> ||normed_hf * w||`

### Per-token magnitude chain (Layer 0)

| Stage | Typical ratio | Worst ratio | Notes |
|-------|--------------|-------------|-------|
| norm1_input | 1.000 | 1.000 | Perfect |
| core_attn_out | 0.94-1.00 | 1.002 | No drift over tokens |
| after_ar | 0.95-1.10 | 1.104 | Oscillates, no monotonic growth |
| **norm2_output** | **1.16-1.99** | **3.68** | **Weight-direction amplification** |
| mlp_output | 1.15-2.93 | 3.68 | MLP amplifies norm2 error |
| layer_output | 1.01-1.19 | 1.19 | Residual partially dilutes |

### Key finding: norm2 is the amplification bottleneck

The after_ar ratio is only 0.95-1.10, but norm2 output ratio jumps to 1.16-3.68. This is NOT a norm2 kernel bug — manual RMSNorm computation produces identical ratios. The non-uniform weight vector (effective range 0.5-1.0, std=0.038) creates a weighted inner product that amplifies small directional differences.

### Not recurrent state drift

core_attn_out per-token ratio stays bounded (0.94-1.00) and does NOT grow monotonically over 19 tokens.

### Not INT4 systematic bias

mixed_qkv per-segment ratios: Q≈0.98, K≈1.00, V≈0.98. No systematic over/under-estimation.

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
| norm2 (unweighted) | 0.988 | 1.02-3.68* | *Weighted cos=0.708 is ARTIFACT |
| **Layer 0 output** | **0.991** | **1.10** | **10% magnitude error** |

### Key architecture facts
- Qwen3_5RMSNorm weight = `1.0 + stored_weight` (stored weight is zero-initialized residual)
- GDN fused QKV uses segment sharding: per-GPU layout = [Q_part, K_part, V_part]
- Comparison must use segment-aware reconstruction (not naive cat)
- core_attn_out oracle shape = [seq_len*num_v_heads, head_v_dim] → reshape to [seq_len, num_v_heads*head_v_dim]
- `down_ar` on each GPU already contains the all-reduce SUM. Don't add down_ar_gpu0 + down_ar_gpu1 (that double-counts).

## Next Steps: Fix the Magnitude Error

Since the root cause is INT4 quantization noise amplified by non-uniform RMSNorm weights, the fix must either:
1. Reduce the quantization noise (smaller directional error → less amplification)
2. Reduce the amplification (modify the computation path)

### Option A: Dequantize INT4 weights to BF16 before GEMM (RECOMMENDED)

Dequantize the INT4 weights to BF16 at load time, then use BF16 GEMM instead of INT4 GEMM. This eliminates quantization noise entirely for the affected layers.

**Pros**: Eliminates the root cause. BF16 GEMM is well-tested.
**Cons**: Increases memory usage (BF16 weights are 4-8x larger than INT4).
**Impact**: mixed_qkv cos would go from 0.994 → ~1.000, eliminating the downstream amplification.

**Implementation**: In `crates/model/src/loader.rs`, add an option to dequantize INT4 weights to BF16 after loading. In `crates/backends/native/src/gdn.rs`, use BF16 GEMM for the dequantized weights.

### Option B: Use FP32 accumulation in INT4 GEMM

The INT4 GEMM currently accumulates in FP32 but the output is BF16. The quantization noise comes from the weight dequantization, not the accumulation. This option wouldn't help.

### Option C: Increase quantization precision (group_size=32)

Currently group_size=128. Reducing to 32 would cut quantization error by ~2x.
**Pros**: Still uses INT4 GEMM, just with smaller groups.
**Cons**: Increases memory for scales/zeros by 4x. May not be enough.

### Option D: Use the chunked parallel GDN attention

Replace the per-token sequential recurrent step with the chunked parallel algorithm (chunk_size=64). The chunked algorithm processes all tokens in a chunk simultaneously, which may produce more consistent per-token magnitudes.

**Pros**: The HF model uses this path and it works correctly.
**Cons**: Significant kernel development effort. The chunked kernel is complex.

### Option E: Normalize the RMSNorm weight to reduce amplification

If the weight variance is the amplification mechanism, we could pre-scale the weight to have lower variance. But this changes the model's behavior.

**Pros**: Simple change.
**Cons**: Changes model output — not a valid fix.

## Quick Diagnostic Commands

```bash
# Full layer 0 comparison
python3 -m tests.compare.hf_compare \
  --oracle-dir /tmp/hf_oracle_layer1_v2 \
  --engine-dir /tmp/dump_layer0_v2 \
  --phase prefill --threshold 0.98

# Per-token magnitude chain
python3 << 'EOF'
import torch, json, os
from tests.compare.io import load_raw_bf16

def load_engine(path, meta):
    with open(meta) as f: shape = tuple(json.load(f)['shape'])
    with open(path, 'rb') as f: return torch.frombuffer(f.read(), dtype=torch.bfloat16).reshape(shape).float().clone()

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

base = "/tmp/dump_layer0_v2/layer_0/prefill"
hf_base = "/tmp/hf_oracle_layer1_v2/layer_0/prefill"

stages = [
    ("gdn.norm1_input_gpu0", "norm1_input", "gpu0"),
    ("gdn.core_attn_out_gpu0", "core_attn_out", "reshape"),
    ("gdn.after_ar_gpu0", "output", "gpu0"),
    ("mlp.norm2_gpu0", "norm2_output", "gpu0"),
    ("residual.mlp_gpu0", "layer_output", "gpu0"),
]

for eng_name, hf_name, recon in stages:
    eng = load_engine(f"{base}/{eng_name}.raw", f"{base}/{eng_name}.meta")
    hf = torch.load(f"{hf_base}/{hf_name}.pt", map_location='cpu').float()
    if hf.dim() == 3: hf = hf.squeeze(0)
    if recon == "reshape" and hf_name == "core_attn_out":
        hf = hf.reshape(19, 48, 128)[:, :24, :].reshape(19, 3072)
    r = eng.norm().item() / hf.norm().item()
    print(f"{eng_name}: ratio={r:.4f}")
EOF
```

## Important Notes

1. **Always use `--release`** for Rust builds and tests. Debug builds are ~100x slower.
2. **The norm2 cos=0.708 is NOT a bug.** It's a weighted cosine similarity artifact. The true norm2 accuracy is cos=0.988 (unweighted normalized vectors). The engine kernel matches manual computation at cos=0.999999.
3. **The mixed_qkv segment-aware reconstruction is critical.** Naive `cat([GPU0, GPU1])` gives wrong column ordering for fused QKV projections.
4. **The Qwen3_5RMSNorm weight formula is `1.0 + stored_weight`**, NOT just `stored_weight`.
5. **`core_attn_out` oracle shape is `[seq_len*num_v_heads, head_v_dim]`**, not `[seq_len, num_v_heads*head_v_dim]`. Must reshape before comparison.
6. **`down_ar` already contains the all-reduce sum.** Don't add `down_ar_gpu0 + down_ar_gpu1` — that double-counts, giving 2.67x instead of the true 1.33x ratio.
