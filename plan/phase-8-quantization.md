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

```rust
pub struct AutoRoundEngine {
    pub dequantized_weights: WeightRegistry,  // FP16 after load-time dequant
    pub gemm_engine: GemmEngine,
}

impl AutoRoundEngine {
    pub fn from_model(
        model_dir: &Path,
        device: &CudaDevice,
        stream: &CudaStream,
    ) -> Result<Self> {
        // 1. Detect format
        let format = QuantizationFormat::detect(model_dir)?;
        assert_eq!(format, QuantizationFormat::AutoRound);
        
        // 2. Load packed weights
        let loader = AutoRoundLoader::new(model_dir)?;
        let packed = loader.load_safetensors()?;
        
        // 3. Dequantize to FP16
        let dequantized = loader.dequantize_all(&packed, device, stream)?;
        
        // 4. Shard for TP/PP
        let sharded = shard_weights(dequantized, num_gpus)?;
        
        Ok(Self {
            dequantized_weights: sharded,
            gemm_engine: GemmEngine::new(stream)?,
        })
    }
}
```

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

```rust
pub struct LlamaCppBackend {
    pub ctx: *mut llama_context,
    pub model: *mut llama_model,
    pub params: llama_model_params,
}

impl LlamaCppBackend {
    pub fn load(path: &Path, n_gpu_layers: i32) -> Result<Self> {
        let model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers;
        
        let model = unsafe {
            llama_load_model_from_file(
                path.as_ptr(),
                model_params,
            )
        };
        
        if model.is_null() {
            return Err(anyhow!("Failed to load GGUF model"));
        }
        
        let ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 262144;
        
        let ctx = unsafe {
            llama_new_context_with_model(model, ctx_params)
        };
        
        if ctx.is_null() {
            unsafe { llama_free_model(model) };
            return Err(anyhow!("Failed to create context"));
        }
        
        Ok(Self {
            ctx,
            model,
            params: model_params,
        })
    }
    
    pub fn decode(&self, tokens: &[u32]) -> Result<u32> {
        let batch = unsafe {
            llama_batch_init(tokens.len(), 0, 1)
        };
        
        // Add tokens to batch
        for (i, &token) in tokens.iter().enumerate() {
            unsafe {
                llama_batch_add(batch, token as i32, i as i32, &[0], i == tokens.len() - 1);
            }
        }
        
        // Decode
        let result = unsafe {
            llama_decode(self.ctx, batch)
        };
        
        if result != 0 {
            return Err(anyhow!("llama_decode failed"));
        }
        
        // Sample next token
        let logits = unsafe {
            llama_get_logits_ith(self.ctx, tokens.len() - 1)
        };
        
        let token = self.sample(logits)?;
        
        unsafe { llama_batch_free(batch); }
        
        Ok(token)
    }
}
```

### Backend Router

```rust
pub enum Backend {
    Native(NativeEngine),      // Our implementation (PrismaSCOUT, AutoRound, BF16)
    LlamaCpp(LlamaCppBackend), // llama.cpp (GGUF)
}

pub struct BackendRouter {
    pub backend: Backend,
}

impl BackendRouter {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let format = QuantizationFormat::detect(model_dir)?;
        
        match format {
            QuantizationFormat::Gguf => {
                let backend = LlamaCppBackend::load(model_dir, 999)?; // All layers on GPU
                Ok(Self { backend: Backend::LlamaCpp(backend) })
            }
            _ => {
                let backend = NativeEngine::load(model_dir)?;
                Ok(Self { backend: Backend::Native(backend) })
            }
        }
    }
    
    pub fn chat_completions(&self, req: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        match &self.backend {
            Backend::Native(engine) => engine.chat_completions(req),
            Backend::LlamaCpp(engine) => engine.chat_completions(req),
        }
    }
}
```

### KV Cache Quantization

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
            Self::Nvfp4 => 1,  // Packed with scales
        }
    }
}

pub struct QuantizedKvCache {
    pub dtype: KvCacheDtype,
    pub k_cache: DeviceBuffer<u8>,  // Raw bytes
    pub v_cache: DeviceBuffer<u8>,
    pub scales: Option<DeviceBuffer<half>>,  // For NVFP4
}

impl QuantizedKvCache {
    pub fn allocate(
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: KvCacheDtype,
    ) -> Result<Self> {
        let bytes_per_elem = dtype.bytes_per_element();
        let page_bytes = page_size * num_kv_heads * head_dim * bytes_per_elem;
        
        let k_cache = DeviceBuffer::alloc(num_pages * page_bytes)?;
        let v_cache = DeviceBuffer::alloc(num_pages * page_bytes)?;
        
        let scales = if matches!(dtype, KvCacheDtype::Nvfp4) {
            // NVFP4 requires FP8 block scales
            let num_blocks = (num_pages * page_size * num_kv_heads * head_dim) / 128;
            Some(DeviceBuffer::alloc(num_blocks * 2)?)
        } else {
            None
        };
        
        Ok(Self {
            dtype,
            k_cache,
            v_cache,
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
        k: &DeviceBuffer<half>,
        v: &DeviceBuffer<half>,
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
    ) -> Result<(DeviceBuffer<half>, DeviceBuffer<half>)> {
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
      quant.rs            # Quantization helpers (FP8, NVFP4)
      
  gguf/
    Cargo.toml
    build.rs              # Build llama.cpp submodule
    src/
      lib.rs
      backend.rs          # LlamaCppBackend
      parser.rs           # GGUF parser
      ffi.rs              # llama.cpp FFI bindings
      
crates/kv/
  src/
    quant.rs              # KvCacheDtype, QuantizedKvCache
```

## Testing

### AutoRound Correctness

```rust
#[test]
fn test_autoround_vs_bf16() {
    let prompt = "The capital of France is";
    
    let bf16_engine = NativeEngine::load("/models/Qwen3.6-27B-BF16").unwrap();
    let int4_engine = NativeEngine::load("/models/Qwen3.6-27B-AutoRound").unwrap();
    
    let bf16_tokens = bf16_engine.generate(prompt, 10).unwrap();
    let int4_tokens = int4_engine.generate(prompt, 10).unwrap();
    
    // Allow some divergence
    let match_rate = bf16_tokens.iter()
        .zip(int4_tokens.iter())
        .filter(|(a, b)| a == b)
        .count() as f32 / bf16_tokens.len() as f32;
    
    assert!(match_rate > 0.8, "AutoRound divergence too high: {}", match_rate);
}
```

### GGUF Loading

```rust
#[test]
fn test_gguf_load() {
    let parser = GgufParser::parse("/models/Qwen3.6-27B-Q4_K_M.gguf").unwrap();
    
    assert!(parser.metadata.contains_key("general.architecture"));
    assert!(!parser.tensor_infos.is_empty());
    
    let backend = LlamaCppBackend::load("/models/Qwen3.6-27B-Q4_K_M.gguf", 999).unwrap();
    
    let tokens = vec![1, 2, 3];  // dummy
    let output = backend.decode(&tokens).unwrap();
    
    assert!(output > 0);
}
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
