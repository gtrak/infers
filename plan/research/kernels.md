# Kernel Strategy Research

## Kernel Sources

### 1. FlashInfer Gated DeltaNet (GDN)

**Source:** vLLM FlashInfer submodule (`../vllm/`)  
**Files to extract:**
- `flashinfer/gdn_prefill/` — GDN prefill kernels
- `flashinfer/gdn_decode/` — GDN decode kernels
- `include/flashinfer/attention/gdn.cuh` — GDN attention headers

**Key functions:**
```cpp
// Prefill
void chunk_gated_delta_rule(
    float* output,
    const float* q, const float* k, const float* v,
    const float* g, const float* beta,
    int batch_size, int seq_len, int num_heads, int head_dim
);

// Decode
void fused_sigmoid_gating_delta_rule_update(
    float* conv_state, float* ssm_state,
    const float* input, const float* weight,
    int batch_size, int d_state, int d_conv
);
```

**Compilation:** nvcc → `.cubin` → load via cuda-oxide

### 2. FlashInfer Standard Attention

**Source:** FlashInfer repo or vLLM submodule  
**Files to extract:**
- `csrc/batch_prefill.cu` — Prefill kernel
- `csrc/batch_decode.cu` — Decode kernel
- `include/flashinfer/attention/prefill.cuh` — Prefill headers
- `include/flashinfer/attention/decode.cuh` — Decode headers

**Key functions:**
```cpp
// Prefill with paged KV cache
template<typename T, typename IdType>
void BatchPrefillWithPagedKVCache(
    T* q, T* k, T* v, T* o,
    IdType* block_table, IdType* page_indices,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int page_size, cudaStream_t stream
);

// Decode with paged KV cache
template<typename T, typename IdType>
void BatchDecodeWithPagedKVCache(
    T* q, T* k, T* v, T* o,
    IdType* block_table, IdType* page_indices,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int page_size, cudaStream_t stream
);
```

### 3. FlashInfer Sampling

**Source:** `include/flashinfer/sampling.cuh`  
**Kernels:**
- `top_k_sampling_from_probs`
- `top_p_sampling_from_probs`
- `temperature_sampling`
- `greedy_sampling`

### 4. cuBLASLt GEMM

**Source:** cudarc `cublaslt` module  
**Usage:**
```rust
use cudarc::cublaslt::safe::CudaBlasLt;

let cublaslt = CudaBlasLt::new(stream)?;
// For NVFP4: use cublasLtMatmul with CUBLAS_COMPUTE_32F, A=NVFP4, B=NVFP4
// For FP16: standard matmul
// For BF16: standard matmul
```

**Note:** cudarc v0.19.7 supports cuBLASLt with NVFP4 on Blackwell.

### 5. Custom Kernels (if needed)

**INT4 GEMM (AutoRound on-the-fly):**

Since AutoRound weights stay in packed INT4 format in GPU memory, the GEMM kernel must unpack and scale on-the-fly.

```cpp
__global__ void int4_gemm_onthefly_kernel(
    half* output,
    const uint32_t* qweight,  // Packed INT4 weights
    const half* scales,        // FP16 per-group scales
    const uint32_t* qzeros,    // Packed INT4 zero points
    const half* input,
    int M, int N, int K,
    int group_size
) {
    // Unpack 8 INT4 weights from each uint32_t
    // Apply (w - zero) * scale per group of 128
    // Multiply with FP16 input, accumulate
}
```

**Key:** Weights never expand to FP16. Output is FP16, fed to next layer.

**RMSNorm:**
```cpp
__global__ void rms_norm_kernel(
    half* output, const half* input,
    const half* weight, float eps,
    int num_tokens, int hidden_size
);
```

**RoPE (multi-dimensional):**
```cpp
__global__ void apply_rotary_pos_emb_kernel(
    half* q, half* k,
    const int* pos_ids,  // [batch, 3] for mRoPE
    float theta, float scaling_factor,
    int num_tokens, int num_heads, int head_dim
);
```

## Kernel Compilation Pipeline

```bash
# Step 1: Extract .cu/.cuh files from vLLM
mkdir -p kernels/flashinfer
cp -r ../vllm/vllm/attention/ops/flashinfer/* kernels/flashinfer/

# Step 2: Compile to .cubin
nvcc -cubin -arch=sm_100 \
  -I kernels/flashinfer/include \
  kernels/flashinfer/csrc/batch_prefill.cu \
  -o kernels/flashinfer/batch_prefill.cubin

nvcc -cubin -arch=sm_100 \
  -I kernels/flashinfer/include \
  kernels/flashinfer/csrc/batch_decode.cu \
  -o kernels/flashinfer/batch_decode.cubin

# Step 3: Embed .cubin in Rust binary
# Use include_bytes!() or load at runtime
```

**Rust loading:**
```rust
use cuda_host::module::Module;

let kernel_bytes = include_bytes!("../kernels/flashinfer/batch_prefill.cubin");
let module = Module::load(&ctx, kernel_bytes)?;
let kernel = module.get_function("BatchPrefillWithPagedKVCache")?;
```

## Kernel Dispatch Strategy

### Per-Layer Dispatch

```rust
enum LayerType {
    GatedDeltaNet,  // 48 layers
    FullAttention,  // 16 layers
}

fn forward_layer(
    layer_type: LayerType,
    layer_idx: usize,
    hidden_states: &DeviceBuffer<half>,
    kv_manager: &HybridKvManager,
    stream: &CudaStream,
) -> Result<DeviceBuffer<half>> {
    match layer_type {
        LayerType::GatedDeltaNet => {
            // Update Mamba state
            let mamba_state = kv_manager.get_mamba_state(session_id, layer_idx);
            gdn_forward(hidden_states, mamba_state, weights, stream)
        }
        LayerType::FullAttention => {
            // Use paged KV cache
            let kv_cache = kv_manager.get_kv_blocks(session_id, layer_idx);
            attention_forward(hidden_states, kv_cache, weights, stream)
        }
    }
}
```

### Prefill vs Decode Dispatch

```rust
enum ForwardMode {
    Prefill,  // Process prompt tokens
    Decode,   // Process single token
}

fn attention_forward(
    mode: ForwardMode,
    q: &DeviceBuffer<half>,
    kv_cache: &PagedKvCache,
    stream: &CudaStream,
) -> Result<DeviceBuffer<half>> {
    match mode {
        ForwardMode::Prefill => {
            flashinfer_prefill(q, kv_cache, stream)
        }
        ForwardMode::Decode => {
            flashinfer_decode(q, kv_cache, stream)
        }
    }
}
```

## Performance Considerations

### Kernel Fusion Opportunities

1. **RMSNorm + SiLU:** Can fuse into single kernel
2. **QKV projection + RoPE:** Can fuse for full attention layers
3. **Gate projection + Up projection:** Standard MLP fusion
4. **Attention + MLP residual:** Cannot fuse due to Mamba state dependency

### Memory Bandwidth

- GDN layers: Memory-bound (recurrent state access)
- Full attention: Compute-bound (FlashInfer optimized)
- MLP: Compute-bound (GEMM dominant)

### Recommended Optimizations

1. **CUDA Graphs:** For decode phase, capture kernel launches (adds ~2 weeks)
2. **Async kernel launch:** All kernels launched asynchronously on streams
3. **Pinned memory:** For CPU↔GPU transfers
4. **Zero-copy:** Where possible, avoid intermediate copies

## References

1. FlashInfer GitHub: https://github.com/flashinfer-ai/flashinfer
2. FlashInfer Docs: https://docs.flashinfer.ai
3. vLLM FlashInfer Integration: `../vllm/vllm/v1/attention/backends/flashinfer.py`
4. vLLM GDN Backend: `../vllm/vllm/v1/attention/backends/gdn_attn.py`
5. cudarc cuBLASLt: https://docs.rs/cudarc/latest/cudarc/cublaslt/index.html

## Cross-References

- See `architecture.md` for which layers use which kernels
- See `quantization.md` for dequantization kernel requirements
- See `parallelism.md` for how kernels are distributed in TP/PP
- See Phase 2 (CUDA Backend) for kernel compilation pipeline
- See Phase 4 (TP Forward) for single-GPU kernel integration
