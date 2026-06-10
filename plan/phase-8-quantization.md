# Phase 8: Quantization Polish

**Duration:** 3 weeks  
**Goal:** Complete AutoRound INT4 and GGUF support, optimize quantization kernels.

## Deliverables

1. AutoRound INT4 end-to-end test
2. GGUF parser and loader
3. llama.cpp backend integration
4. NVFP4 KV cache implementation
5. FP8 KV cache implementation
6. Cross-format benchmarking
7. Backend routing (auto-detect → native or llama.cpp)

## Technical Details

### AutoRound INT4 End-to-End

**Not yet implemented.** The format detection exists (`QuantizationFormat::detect`
returns `AutoRound` for `quantization_config.json`), but the weight loading and
custom INT4 GEMM are not yet implemented.

AutoRound weights are in packed INT4 format. The planned approach:

```rust
pub struct AutoRoundEngine {
    pub weight_registry: WeightRegistry, // INT4 packed weights (stays compressed in GPU)
    pub scales: HashMap<String, CudaSlice<bf16>>,  // FP16 group scales
    pub zeros: HashMap<String, CudaSlice<u32>>,    // Packed INT4 zeros
}
```

**Key design decisions:**
- Weights stay in INT4-packed format in GPU memory (~75% VRAM savings)
- Dequantization happens in registers during custom GEMM kernel
- No FP16 weight copy ever exists
- Output is BF16, fed directly into next layer (RMSNorm, activation, etc.)
- Group size: 128 (fixed by AutoRound format)

**Performance note:** The INT4 GEMM kernel is memory-bandwidth-bound, not
compute-bound. The savings from INT4 storage (less memory bandwidth) roughly
offset the cost of unpacking.

### Custom INT4 GEMM Kernel

```cuda
__global__ void int4_gemm_kernel(
    half* __restrict__ output,          // [M, N] output
    const uint32_t* __restrict__ weight,  // [N, K/8] packed INT4
    const half* __restrict__ scales,      // [N, K/group_size] FP16
    const uint32_t* __restrict__ zeros,   // [N, K/group_size/8] packed INT4
    const half* __restrict__ input,       // [M, K] FP16 activation
    int M, int N, int K,
    int group_size
) {
    // Each thread computes one output element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float acc = 0.0f;
    
    for (int k = 0; k < K; k += group_size) {
        // Load scale and zero for this group
        int group_idx = k / group_size;
        half scale = scales[col * (K / group_size) + group_idx];
        
        // Unpack zero point
        int zero_packed_idx = (col * (K / group_size) + group_idx) / 8;
        int zero_shift = ((col * (K / group_size) + group_idx) % 8) * 4;
        uint32_t zero_packed = zeros[zero_packed_idx];
        int8_t zero = (int8_t)((zero_packed >> zero_shift) & 0xF);
        
        for (int kk = 0; kk < group_size; kk += 8) {
            // Load 8 INT4 weights from one uint32_t
            int weight_idx = (col * K + k + kk) / 8;
            uint32_t packed = weight[weight_idx];
            
            // Unpack each of 8 weights
            #pragma unroll
            for (int w = 0; w < 8; w++) {
                int shift = w * 4;
                int8_t w_int4 = (int8_t)((packed >> shift) & 0xF);
                
                // Dequantize: (w - zero) * scale
                float w_fp32 = ((float)(w_int4 - zero)) * __half2float(scale);
                
                // Load activation
                half a = input[row * K + k + kk + w];
                
                // Multiply and accumulate
                acc += w_fp32 * __half2float(a);
            }
        }
    }
    
    // Write output
    output[row * N + col] = __float2half(acc);
}
```

**Key design decisions:**
- Weights stay in INT4-packed format in GPU memory (~75% VRAM savings)
- Dequantization happens in registers during GEMM
- No FP16 weight copy ever exists
- Output is FP16, fed directly into next layer (RMSNorm, activation, etc.)
- Group size: 128 (fixed by AutoRound format)

**Performance note:** This kernel is memory-bandwidth-bound, not compute-bound. The savings from INT4 storage (less memory bandwidth) roughly offset the cost of unpacking.

### GGUF Parser

```rust
pub struct GgufParser {
    pub metadata: GgufMetadata,
    pub tensor_infos: Vec<TensorInfo>,
    pub tensor_data_offset: u64,
}

impl GgufParser {
    pub fn parse(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Read header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        assert_eq!(&magic, b"GGUF");
        
        let version = reader.read_u32_le()?;
        let tensor_count = reader.read_u64_le()?;
        let metadata_kv_count = reader.read_u64_le()?;
        
        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut reader)?;
            let value_type = reader.read_u32_le()?;
            let value = read_value(&mut reader, value_type)?;
            metadata.insert(key, value);
        }
        
        // Read tensor info
        let mut tensor_infos = Vec::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut reader)?;
            let num_dims = reader.read_u32_le()?;
            let mut dims = Vec::new();
            for _ in 0..num_dims {
                dims.push(reader.read_u64_le()?);
            }
            let dtype = reader.read_u32_le()?;
            let offset = reader.read_u64_le()?;
            
            tensor_infos.push(TensorInfo {
                name, dims, dtype, offset,
            });
        }
        
        let tensor_data_offset = reader.stream_position()?;
        
        Ok(Self {
            metadata: GgufMetadata(metadata),
            tensor_infos,
            tensor_data_offset,
        })
    }
}
```

### llama.cpp Backend

**Deferred.** The `infers-backend-gguf` crate exists as a stub (`crates/backends/gguf/`).
Integration with llama.cpp requires:

1. Adding llama.cpp as a git submodule (or system dependency)
2. Building `libllama.a` via `build.rs`
3. Generating Rust FFI bindings via `bindgen` for `llama.h`
4. Implementing `LlamaCppBackend` wrapping `llama_context`/`llama_model`:
   - `load(path, n_gpu_layers)` → model loading with GPU offload
   - `decode(tokens)` → batch encode + sample next token
   - Compatible with `KvCacheDtype` for KV cache quantization
5. Adding a `Backend` enum and router for format-based dispatch

This is a self-contained integration that can be done independently of
the native backend. The `infers-backend-gguf` crate is already set up
with the correct dependency structure.

```rust
// Conceptual approach (not yet implemented):
pub struct LlamaCppBackend {
    pub ctx: *mut llama_context,
    pub model: *mut llama_model,
}

impl LlamaCppBackend {
    pub fn load(path: &Path, n_gpu_layers: i32) -> Result<Self>;
    pub fn decode(&self, tokens: &[u32]) -> Result<u32>;
}
```

### Backend Router

### Backend Router (Deferred)

A `Backend` enum and `BackendRouter` do not yet exist. Once llama.cpp
integration is complete, the router will auto-detect format at load time:

```rust
pub enum Backend {
    Native(NativeEngine),      // Our implementation (PrismaSCOUT, AutoRound, BF16)
    // LlamaCpp(LlamaCppBackend), // Future: llama.cpp GGUF support
}

impl Backend {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let format = QuantizationFormat::detect(model_dir)?;
        match format {
            QuantizationFormat::Gguf => {
                anyhow::bail!("GGUF backend not yet implemented — see crates/backends/gguf/");
            }
            _ => {
                let engine = NativeEngine::load(model_dir)?;
                Ok(Self::Native(engine))
            }
        }
    }
}
```

### KV Cache Quantization

KV cache quantization must work with the existing `PagedKvCache` interleaved
page layout (`[K tokens | V tokens]` per page, `page_stride = 2 * page_size * kv_dim`).

```rust
pub enum KvCacheDtype {
    Bf16,
    Fp8E4M3,
    Fp8E5M2,
    Nvfp4,
}

impl KvCacheDtype {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8E4M3 | Self::Fp8E5M2 => 1,
            Self::Nvfp4 => 1,  // Packed with block scales
        }
    }
}

/// Quantized paged KV cache using interleaved K+V per page layout.
///
/// Matches the PagedKvCache layout but stores quantized values:
/// page_pool[page_id * page_stride + side * page_size * kv_dim + ...]
/// where side=0 for K, side=1 for V.
pub struct QuantizedKvCache {
    pub dtype: KvCacheDtype,
    /// Interleaved page pool (K then V per page), quantized to dtype.
    pub page_pool: CudaSlice<u8>,
    /// Number of pages in the pool.
    pub num_pages: usize,
    /// Page size (tokens per page).
    pub page_size: usize,
    /// KV dimension.
    pub kv_dim: usize,
    /// Block scales for NVFP4 (one per 128-element block).
    pub scales: Option<CudaSlice<bf16>>,
}

impl QuantizedKvCache {
    pub fn allocate(
        stream: &CudaStream,
        num_pages: usize,
        page_size: usize,
        kv_dim: usize,
        dtype: KvCacheDtype,
    ) -> Result<Self> {
        let bytes_per_elem = dtype.bytes_per_element();
        let page_bytes = 2 * page_size * kv_dim * bytes_per_elem; // K+V per page

        let page_pool = stream
            .alloc_zeros::<u8>(num_pages * page_bytes)?;

        let scales = if matches!(dtype, KvCacheDtype::Nvfp4) {
            let num_blocks = (num_pages * 2 * page_size * kv_dim) / 128;
            Some(stream.alloc_zeros::<bf16>(num_blocks * 2)?)
        } else {
            None
        };

        Ok(Self {
            dtype,
            page_pool,
            num_pages,
            page_size,
            kv_dim,
            scales,
        })
    }
}
            scales,
        })
    }
}
```

### FP8 KV Cache

```rust
impl QuantizedKvCache {
    pub fn write_fp8(
        &mut self,
        page_id: usize,
        page_offset: usize,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        stream: &CudaStream,
    ) -> Result<()> {
        // Quantize BF16 → FP8 (E4M3)
        let k_fp8 = quantize_fp8_e4m3(k, stream)?;
        let v_fp8 = quantize_fp8_e4m3(v, stream)?;
        
        // Write to cache
        let offset = page_id * self.page_size + page_offset;
        self.k_cache.copy_from(&k_fp8, offset)?;
        self.v_cache.copy_from(&v_fp8, offset)?;
        
        Ok(())
    }
    
    pub fn read_fp8(
        &self,
        page_id: usize,
        page_offset: usize,
        len: usize,
        stream: &CudaStream,
    ) -> Result<(CudaSlice<bf16>, CudaSlice<bf16>)> {
        let offset = page_id * self.page_size + page_offset;
        
        let k_fp8 = self.k_cache.slice(offset, offset + len)?;
        let v_fp8 = self.v_cache.slice(offset, offset + len)?;
        
        // Dequantize FP8 → BF16
        let k = dequantize_fp8_e4m3(&k_fp8, stream)?;
        let v = dequantize_fp8_e4m3(&v_fp8, stream)?;
        
        Ok((k, v))
    }
}
```

## File Structure

```
crates/backends/
  native/
    src/
      quant.rs            # NEW: Quantization helpers (FP8, NVFP4), INT4 GEMM kernel dispatch

  gguf/                   # EXISTS AS STUB — needs implementation
    Cargo.toml
    build.rs              # FUTURE: Build llama.cpp submodule
    src/
      lib.rs
      backend.rs          # FUTURE: LlamaCppBackend
      parser.rs           # FUTURE: GGUF parser
      ffi.rs              # FUTURE: llama.cpp FFI bindings

crates/cuda/
  src/
    gemm.rs               # UPDATE: Add INT4 GEMM engine
  kernels/infers/
    int4_gemm.cu          # NEW: Custom INT4 GEMM kernel (future)

crates/kv/
  src/
    quant.rs              # NEW: KvCacheDtype, QuantizedKvCache
```

## Testing

### Format Detection

```rust
#[test]
fn test_autoround_format_detection() {
    // Format detection works, loading doesn't yet (INT4 GEMM not implemented)
    let dir = tempfile::tempdir().unwrap();

    // Create quantization_config.json for AutoRound
    std::fs::write(
        dir.path().join("quantization_config.json"),
        r#"{"quantization_method":"auto-round","group_size":128}"#,
    ).unwrap();

    let format = QuantizationFormat::detect(dir.path()).unwrap();
    assert_eq!(format, QuantizationFormat::AutoRound);
}
```

### KV Cache Quantization (Unit Test)

```rust
#[test]
fn test_quantized_kv_cache_alloc() {
    let stream = CudaStream::new(0).unwrap();
    let cache = QuantizedKvCache::allocate(
        &stream, 100, // num_pages
        16,           // page_size
        1024,         // kv_dim
        KvCacheDtype::Fp8E4M3,
    ).unwrap();

    assert_eq!(cache.num_pages, 100);
    // page_pool = 100 * 2 * 16 * 1024 * 1 byte = 3,276,800 bytes
    assert_eq!(cache.page_pool.len(), 3_276_800);
}
```

### GGUF Parser (Deferred)

```rust
// GGUF parsing and llama.cpp backend tests will be added once
// crates/backends/gguf/ is implemented. See that crate for details.
```

## Dependencies

### Phase 8 → Phase 1-7

Integrates all previous work.

### External Dependencies

- llama.cpp (git submodule)
- bindgen (for FFI generation)

## Success Criteria

1. AutoRound produces similar output to BF16 (>80% token match)
2. GGUF models load and generate tokens
3. NVFP4 KV cache works on Blackwell
4. FP8 KV cache works on any Ampere+
5. Backend router auto-selects correct backend
6. All formats pass basic correctness tests
7. Benchmark results documented

## Cross-References

- **Research:** See `../research/quantization.md` for format details
- **Phase 3:** Model loader detects formats
- **Phase 4:** Native backend handles PrismaSCOUT and AutoRound
- **Phase 6:** KV cache quantization applies to continuous batching

## Open Questions

1. Should we support dynamic quantization switching at runtime?
2. How to handle GGUF model updates (new format versions)?
3. Should we quantize KV cache on-the-fly or pre-allocate quantized?
