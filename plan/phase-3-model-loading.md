# Phase 3: Model Loading

**Duration:** 3 weeks  
**Goal:** Implement multi-format model loader with auto-detection, weight sharding, and memory budgeting.

## Deliverables

1. Safetensors parallel reader
2. `config.json` parser for Qwen3.6
3. Quantization format auto-detection
4. PrismaSCOUT loader
5. AutoRound loader
6. Weight registry with tensor lookup
7. Weight sharding for TP=2
8. Memory budget calculator
9. Model validation (checksums, shape verification)

## Technical Details

### Config Parser

**Qwen3.6-specific fields:**

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub hidden_act: String,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub partial_rotary_factor: f32,
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>,
    
    // MTP
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
    
    // Quantization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_config: Option<QuantizationConfig>,
    
    // Layer types for hybrid architecture
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
}

impl ModelConfig {
    pub fn get_layer_type(&self, layer_idx: usize) -> LayerType {
        if let Some(types) = &self.layer_types {
            match types[layer_idx].as_str() {
                "linear_attention" => LayerType::GatedDeltaNet,
                "full_attention" => LayerType::FullAttention,
                _ => panic!("Unknown layer type: {}", types[layer_idx]),
            }
        } else {
            // Default pattern: every 4th layer is full attention
            if (layer_idx + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::GatedDeltaNet
            }
        }
    }
    
    pub fn has_mtp(&self) -> bool {
        self.mtp_num_hidden_layers > 0
    }
}
```

### Quantization Format Detection

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationFormat {
    Bf16,        // No quantization
    PrismaScout, // compressed-tensors, NVFP4+BF16
    AutoRound,   // auto_round:auto_gptq, INT4
    Gguf,        // llama.cpp GGUF format
}

impl QuantizationFormat {
    pub fn detect(model_dir: &Path) -> Result<Self> {
        // Check for GGUF files
        if has_gguf_files(model_dir) {
            return Ok(Self::Gguf);
        }
        
        // Check for quantization_config.json
        let config_path = model_dir.join("quantization_config.json");
        if config_path.exists() {
            let config: QuantizationConfig = read_json(&config_path)?;
            match config.quant_method.as_str() {
                "compressed-tensors" => return Ok(Self::PrismaScout),
                "auto-round" => return Ok(Self::AutoRound),
                _ => {}
            }
        }
        
        // Check config.json for quantization_config field
        let model_config_path = model_dir.join("config.json");
        if model_config_path.exists() {
            let config: ModelConfig = read_json(&model_config_path)?;
            if let Some(qc) = config.quantization_config {
                match qc.quant_method.as_str() {
                    "compressed-tensors" => return Ok(Self::PrismaScout),
                    "auto-round" => return Ok(Self::AutoRound),
                    _ => {}
                }
            }
        }
        
        Ok(Self::Bf16)
    }
}
```

### Safetensors Loader

```rust
pub struct SafetensorsLoader {
    pub tensors: HashMap<String, TensorView>,
    pub metadata: Metadata,
}

impl SafetensorsLoader {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        
        if index_path.exists() {
            // Sharded model
            let index: ShardIndex = read_json(&index_path)?;
            Self::load_sharded(model_dir, &index)
        } else {
            // Single file
            let path = model_dir.join("model.safetensors");
            Self::load_single(&path)
        }
    }
    
    fn load_sharded(model_dir: &Path, index: &ShardIndex) -> Result<Self> {
        let mut tensors = HashMap::new();
        
        // Parallel loading
        let shards: Vec<_> = index
            .weight_map
            .values()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        
        let loaded: Vec<_> = shards
            .into_par_iter()
            .map(|shard_name| {
                let path = model_dir.join(shard_name);
                safetensors::load(&path)
            })
            .collect::<Result<Vec<_>>>()?;
        
        for shard in loaded {
            for (name, tensor) in shard.tensors() {
                tensors.insert(name.to_string(), tensor);
            }
        }
        
        Ok(Self { tensors, metadata: index.metadata.clone() })
    }
}
```

### Weight Registry

```rust
pub struct WeightRegistry {
    pub embedding: DeviceBuffer<half>,
    pub layers: Vec<LayerWeights>,
    pub mtp: Option<MtpWeights>,
    pub lm_head: DeviceBuffer<half>,
    pub norm: DeviceBuffer<half>,
}

pub struct LayerWeights {
    pub layer_type: LayerType,
    pub layer_idx: usize,
    
    // For GDN layers
    pub gdn: Option<GdnWeights>,
    
    // For full attention layers
    pub attn: Option<AttentionWeights>,
    
    // Common
    pub mlp: MlpWeights,
    pub norm1: DeviceBuffer<half>,
    pub norm2: DeviceBuffer<half>,
}

pub struct GdnWeights {
    pub in_proj_a: DeviceBuffer<half>,  // Linear attention projection A
    pub in_proj_b: DeviceBuffer<half>,  // Linear attention projection B
    pub conv1d_weight: DeviceBuffer<half>,
    pub x_proj_weight: DeviceBuffer<half>,
    pub dt_proj_weight: DeviceBuffer<half>,
    pub out_proj_weight: DeviceBuffer<half>,
}

pub struct AttentionWeights {
    pub q_proj: DeviceBuffer<half>,
    pub k_proj: DeviceBuffer<half>,
    pub v_proj: DeviceBuffer<half>,
    pub o_proj: DeviceBuffer<half>,
}

pub struct MlpWeights {
    pub gate_proj: DeviceBuffer<half>,
    pub up_proj: DeviceBuffer<half>,
    pub down_proj: DeviceBuffer<half>,
}
```

### PrismaSCOUT Loader

```rust
pub struct PrismaScoutLoader {
    pub config: CompressedTensorsConfig,
}

impl PrismaScoutLoader {
    pub fn load_weights(
        &self,
        tensors: &HashMap<String, TensorView>,
        device: &CudaDevice,
        stream: &CudaStream,
    ) -> Result<WeightRegistry> {
        let mut registry = WeightRegistry::new();
        
        for (name, tensor) in tensors {
            // Check quantization config for this tensor
            let quant_config = self.config.get_tensor_config(name);
            
            match quant_config.format {
                TensorFormat::Nvfp4 => {
                    // Load as NVFP4 (packed format)
                    let packed = device.alloc(tensor.len())?;
                    stream.memcpy_htod(&tensor.data(), &packed)?;
                    
                    // Store with metadata for dequantization
                    registry.insert_nvfp4(name, packed, quant_config)?;
                }
                TensorFormat::Bf16 => {
                    // Load as BF16 directly
                    let buffer = device.alloc(tensor.len() * 2)?;
                    stream.memcpy_htod(&tensor.data(), &buffer)?;
                    registry.insert_bf16(name, buffer)?;
                }
                _ => panic!("Unsupported format: {:?}", quant_config.format),
            }
        }
        
        Ok(registry)
    }
}
```

### AutoRound Loader

```rust
pub struct AutoRoundLoader {
    pub config: AutoRoundConfig,
}

impl AutoRoundLoader {
    pub fn load_weights(
        &self,
        tensors: &HashMap<String, TensorView>,
        device: &CudaDevice,
        stream: &CudaStream,
    ) -> Result<WeightRegistry> {
        let mut registry = WeightRegistry::new();
        
        for (name, tensor) in tensors {
            if name.ends_with(".qweight") {
                // Packed INT4 weights
                let qweight = device.alloc(tensor.len())?;
                stream.memcpy_htod(&tensor.data(), &qweight)?;
                
                // Find corresponding scales and zeros
                let base_name = name.trim_end_matches(".qweight");
                let scales = tensors.get(&format!("{}.scales", base_name))
                    .ok_or_else(|| anyhow!("Missing scales for {}", name))?;
                let qzeros = tensors.get(&format!("{}.qzeros", base_name))
                    .ok_or_else(|| anyhow!("Missing qzeros for {}", name))?;
                
                // Dequantize to FP16 on GPU
                let dequantized = self.dequantize_int4(
                    &qweight, scales, qzeros,
                    device, stream,
                )?;
                
                registry.insert_fp16(base_name, dequantized)?;
            } else if !name.ends_with(".scales") && !name.ends_with(".qzeros") {
                // Non-quantized tensor (norms, biases, etc.)
                let buffer = device.alloc(tensor.len() * 2)?;
                stream.memcpy_htod(&tensor.data(), &buffer)?;
                registry.insert_bf16(name, buffer)?;
            }
        }
        
        Ok(registry)
    }
    
    fn dequantize_int4(
        &self,
        qweight: &DeviceBuffer<u32>,
        scales: &TensorView,
        qzeros: &TensorView,
        device: &CudaDevice,
        stream: &CudaStream,
    ) -> Result<DeviceBuffer<half>> {
        // Launch INT4 dequantization kernel
        let group_size = self.config.group_size;
        let (out_features, in_features) = self.get_dimensions(qweight)?;
        
        let output = device.alloc(out_features * in_features * 2)?;
        
        let kernel = get_kernel("int4_dequantize")?;
        stream.launch(
            &kernel,
            dim3((out_features + 255) / 256, 1, 1),
            dim3(256, 1, 1),
            0,
            &(
                &output,
                qweight,
                scales,
                qzeros,
                out_features,
                in_features,
                group_size,
            ),
        )?;
        
        Ok(output)
    }
}
```

### Weight Sharding for TP=2

```rust
pub fn shard_weights_tp(
    registry: &mut WeightRegistry,
    num_gpus: usize,
) -> Result<Vec<WeightRegistry>> {
    let mut shards = vec![WeightRegistry::new(); num_gpus];
    
    for layer in &registry.layers {
        let shard_size = layer.hidden_size() / num_gpus;
        
        for gpu_id in 0..num_gpus {
            let start = gpu_id * shard_size;
            let end = start + shard_size;
            
            // Shard column-parallel weights
            if let Some(attn) = &layer.attn {
                shards[gpu_id].layers[layer.layer_idx].attn = Some(AttentionWeights {
                    q_proj: attn.q_proj.slice(start..end)?,
                    k_proj: attn.k_proj.slice(start..end)?,
                    v_proj: attn.v_proj.slice(start..end)?,
                    o_proj: attn.o_proj.slice(start..end)?,
                });
            }
            
            // MLP: gate/up are column-parallel, down is row-parallel
            shards[gpu_id].layers[layer.layer_idx].mlp = MlpWeights {
                gate_proj: layer.mlp.gate_proj.slice(start..end)?,
                up_proj: layer.mlp.up_proj.slice(start..end)?,
                down_proj: layer.mlp.down_proj,  // Row-parallel: full tensor on each GPU
            };
        }
    }
    
    Ok(shards)
}
```

### Memory Budget Calculator

```rust
pub struct MemoryBudget {
    pub total_vram: usize,
    pub weight_bytes: usize,
    pub kv_cache_bytes: usize,
    pub workspace_bytes: usize,
    pub available_kv: usize,
}

impl MemoryBudget {
    pub fn calculate(
        config: &ModelConfig,
        quant_format: QuantizationFormat,
        gpu_count: usize,
        gpu_memory: usize,
        gpu_utilization: f32,
    ) -> Result<Self> {
        let total_vram = (gpu_memory as f32 * gpu_utilization) as usize;
        
        // Calculate weight size
        let weight_bytes = match quant_format {
            QuantizationFormat::Bf16 => config.estimate_weight_bytes_bf16(),
            QuantizationFormat::PrismaScout => config.estimate_weight_bytes_nvfp4(),
            QuantizationFormat::AutoRound => config.estimate_weight_bytes_int4(),
            QuantizationFormat::Gguf => panic!("GGUF memory budget not applicable"),
        };
        
        let weight_bytes_per_gpu = weight_bytes / gpu_count;
        
        // Calculate KV cache size
        let num_attention_layers = config.num_full_attention_layers();
        let kv_bytes_per_token = 2 * config.num_key_value_heads * config.head_dim * 2; // K + V, BF16
        let max_kv_bytes = num_attention_layers * kv_bytes_per_token * config.max_position_embeddings;
        let max_kv_per_gpu = max_kv_bytes / gpu_count;
        
        // Reserve workspace (activations, temp buffers)
        let workspace_bytes = 4 * 1024 * 1024 * 1024; // 4 GB
        
        let available_kv = total_vram.saturating_sub(weight_bytes_per_gpu + workspace_bytes);
        
        Ok(Self {
            total_vram,
            weight_bytes: weight_bytes_per_gpu,
            kv_cache_bytes: max_kv_per_gpu,
            workspace_bytes,
            available_kv,
        })
    }
    
    pub fn max_concurrent_sessions(&self, avg_context_len: usize) -> usize {
        let kv_per_session = self.kv_cache_bytes / self.max_position_embeddings * avg_context_len;
        self.available_kv / kv_per_session
    }
}
```

## File Structure

```
crates/model/
  Cargo.toml
  src/
    lib.rs
    config.rs           # ModelConfig, layer type parsing
    loader.rs           # Multi-format loader, format detection
    formats.rs          # QuantizationFormat enum
    weights.rs          # WeightRegistry, LayerWeights
    sharding.rs         # TP/PP weight sharding
    budget.rs           # MemoryBudget calculator
    
    loaders/
      bf16.rs           # BF16 loader (no quant)
      prisma_scout.rs   # PrismaSCOUT loader
      auto_round.rs     # AutoRound loader
      gguf.rs           # GGUF loader (delegates to llama.cpp)
```

## Dependencies

### Phase 3 → Phase 2

Uses `GpuAllocator`, `CudaDevice`, `CudaStream` from Phase 2.

### Phase 3 → Phase 4

Weight registry will be consumed by the forward pass.

### External Dependencies

- safetensors v0.5
- memmap2 v0.9 (for memory-mapped file reading)
- serde_json v1.0

## Success Criteria

1. Can load BF16 base model in < 30 seconds
2. Can detect PrismaSCOUT format automatically
3. Can detect AutoRound format automatically
4. Can detect GGUF format automatically
5. PrismaSCOUT weights load with correct mixed precision
6. AutoRound weights load in packed INT4 format (not dequantized)
7. Weight sharding produces correct splits for TP=2
8. Memory budget calculator returns reasonable values
9. All tensor shapes match config expectations

## Cross-References

- **Research:** See `../research/quantization.md` for format details
- **Research:** See `../research/architecture.md` for Qwen3.6 config fields
- **Phase 2:** Uses CUDA runtime for device allocation
- **Phase 4:** Forward pass consumes `WeightRegistry`
- **Phase 5:** PP sharding uses `sharding.rs`

## Open Questions

1. Should we support loading from HuggingFace Hub (download on-the-fly)?
2. How to handle partial model loading (e.g., text-only, no vision)?
3. Should we cache dequantized weights to disk (for faster restart)?
