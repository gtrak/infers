# Quantization Research

## Supported Formats

### 1. PrismaSCOUT (Primary)

**Model:** `rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`  
**Size:** 20.17 GB  
**License:** Apache-2.0

**Format:** Mixed precision — NVFP4 for most layers + BF16 for selected layers

**Quantization strategy:**
- End-to-end KL divergence optimization
- Per-layer format assignment
- Three-stage cost cascade (L1/L2/L3)
- Pareto frontier selection (size vs quality)

**Config structure:**
```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "config_groups": {
      "group_0": {
        "weights": {"num_bits": 4, "type": "float", "strategy": "tensor"},
        "input_activations": {"num_bits": 16, "type": "float", "strategy": "tensor"},
        "output_activations": {"num_bits": 16, "type": "float", "strategy": "tensor"}
      }
    }
  }
}
```

**Hardware requirement:** Blackwell (SM100+) for NVFP4 tensor cores  
**Dequantization:** Hardware-native in tensor cores (no software overhead)

**vLLM serve command:**
```bash
vllm serve rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm \
  --quantization compressed-tensors \
  --trust-remote-code \
  --max-model-len 32768
```

### 2. AutoRound INT4 (Secondary)

**Model:** `Lorbus/Qwen3.6-27B-int4-AutoRound`  
**Size:** 17.7 GB (19 GB with all files)  
**License:** Apache-2.0

**Format:** W4A16 (INT4 weights, FP16 activations), group_size=128, symmetric

**Quantization config:**
```json
{
  "quantization_config": {
    "quant_method": "auto-round",
    "bits": 4,
    "data_type": "int",
    "group_size": 128,
    "sym": true,
    "packing_format": "auto_round:auto_gptq",
    "block_name_to_quantize": "model.language_model.layers"
  }
}
```

**Key characteristics:**
- Uniform INT4 across all quantizable layers
- Group size: 128 (128 weights share one scale)
- Symmetric quantization
- GPTQ-compatible packing
- Unquantized layers: `linear_attn.in_proj_a/b`, `mtp.fc`, norms, routers
- MTP head preserved in BF16

**Dequantization:** On-the-fly in custom GEMM kernels (weights stay packed in GPU memory, never expanded)

**Packed format:**
- INT4 weights stored in `uint32_t` arrays (8 weights per uint32)
- Scales: FP16, one per group of 128 weights
- Zeros: INT4 packed, one per group of 128 weights

**GEMM strategy:**
```
For each output element:
  1. Load 8 INT4 weights from uint32_t
  2. Unpack to INT8 in registers
  3. Subtract zero point, multiply by scale
  4. Cast to FP16
  5. Multiply with FP16 activation
  6. Accumulate in FP32
  7. Cast result to FP16
```

**VRAM savings:** ~75% vs BF16 (4 bits per weight instead of 16)

**vLLM serve command:**
```bash
vllm serve Lorbus/Qwen3.6-27B-int4-AutoRound \
  --dtype half \
  --max-model-len 262144 \
  --trust-remote-code \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

### 3. GGUF (Tertiary)

**Format:** llama.cpp native format  
**Quantization levels:** Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0

**Key characteristics:**
- On-the-fly dequantization in CUDA kernels (no expansion)
- Block-wise quantization (typically 256 or 128 elements per block)
- Mixed formats within model (attention vs MLP may have different quants)
- GGUF stores both weights and metadata (vocab, tensors, etc.)

**Memory usage (27B model):**
| Format | Size | Quality |
|---|---|---|
| Q2_K | ~7 GB | Low |
| Q4_K_M | ~15 GB | Good |
| Q5_K_M | ~18 GB | Very Good |
| Q8_0 | ~27 GB | Near-lossless |

**Dequantization:** In llama.cpp's ggml CUDA kernels, on-the-fly during GEMM

### 4. BF16 (Baseline)

**Model:** `Qwen/Qwen3.6-27B`  
**Size:** ~54 GB  
**No quantization config**

## Format Detection

The server auto-detects format by inspecting model directory:

```rust
pub enum QuantizationFormat {
    Gguf,           // .gguf files present
    PrismaScout,    // compressed-tensors config
    AutoRound,      // auto-round config
    Bf16,           // No quantization config
}

impl QuantizationFormat {
    fn detect(model_dir: &Path) -> Self {
        if has_gguf_files(model_dir) {
            return Self::Gguf;
        }
        if let Some(config) = read_quantization_config(model_dir) {
            match config.quant_method.as_str() {
                "compressed-tensors" => Self::PrismaScout,
                "auto-round" => Self::AutoRound,
                _ => Self::Bf16,
            }
        } else {
            Self::Bf16
        }
    }
}
```

## Dequantization Strategy by Format

| Format | When | How | Kernel | VRAM |
|---|---|---|---|---|
| PrismaSCOUT | Runtime (GEMM) | Hardware (tensor core) | cuBLASLt (NVFP4 input) | ~20 GB |
| AutoRound | Runtime (GEMM) | Software (unpack + scale in registers) | Custom INT4 GEMM | ~18 GB |
| GGUF | Runtime (GEMM) | Hardware/Software | llama.cpp ggml CUDA | varies |
| BF16 | N/A | N/A | Standard cuBLASLt | ~54 GB |

## KV Cache Quantization

### Supported Formats

| Format | Bits | Hardware | VRAM Savings |
|---|---|---|---|
| BF16 | 16 | Any | Baseline |
| FP8 (E4M3) | 8 | Ampere+ | 50% |
| FP8 (E5M2) | 8 | Ampere+ | 50% |
| NVFP4 | 4 | Blackwell only | 75% |

### KV Cache Size Calculation

For 262K context, 64 layers, 4 KV heads, 256 head_dim:

```
KV per layer = 2 * num_kv_heads * head_dim * num_tokens * bytes_per_element
             = 2 * 4 * 256 * 262144 * 2  (for BF16)
             = ~4.3 GB per layer
             = ~275 GB total (all layers)
```

**With 16 full attention layers only (GDN doesn't use standard KV):**

```
KV total = 16 * 4.3 GB = ~69 GB (BF16)
         = 16 * 2.1 GB = ~34 GB (FP8)
         = 16 * 1.1 GB = ~17 GB (NVFP4)
```

**With TP=2:** Split across 2 GPUs:
- BF16: ~34 GB per GPU
- FP8: ~17 GB per GPU
- NVFP4: ~8.5 GB per GPU

**With PP=2:** Each GPU manages its own stage's KV:
- 8 full attention layers per stage
- BF16: ~17 GB per GPU
- FP8: ~8.5 GB per GPU
- NVFP4: ~4.3 GB per GPU

## Memory Budget (2× RTX 5060 Ti, 32GB each)

### TP=2

| Component | PrismaSCOUT | AutoRound | BF16 |
|---|---|---|---|
| Weights (per GPU) | ~10 GB | ~9 GB | ~27 GB |
| KV cache (per GPU, FP8) | ~17 GB | ~17 GB | ~34 GB |
| Workspace | ~4 GB | ~4 GB | ~4 GB |
| **Total** | **~31 GB** | **~30 GB** | **~65 GB → OOM** |

### PP=2

| Component | PrismaSCOUT | AutoRound | BF16 |
|---|---|---|---|
| Weights (per GPU) | ~20 GB | ~18 GB | ~54 GB |
| KV cache (per GPU, FP8) | ~8.5 GB | ~8.5 GB | ~17 GB |
| Workspace | ~4 GB | ~4 GB | ~4 GB |
| **Total** | **~32.5 GB** | **~30.5 GB** | **~75 GB → OOM** |

**Note:** PP=2 with PrismaSCOUT is tight (~32.5 GB). NVFP4 KV cache is recommended for PP mode.

## References

1. PrismaSCOUT Model Card: https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm
2. AutoRound Model Card: https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound
3. AutoRound Paper: arXiv:2309.05516
4. NVFP4 Specification: NVIDIA Blackwell architecture docs
5. GGUF Format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

## Cross-References

- See `architecture.md` for how quantization interacts with GDN vs full attention layers
- See `kernels.md` for kernel requirements per quantization format
- See `parallelism.md` for memory distribution in TP vs PP
- See Phase 3 (Model Loading) for loader implementation
- See Phase 8 (Quantization) for end-to-end integration
