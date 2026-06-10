//! Memory budget calculator for different quantization formats and parallelism modes.

use super::config::ModelConfig;
use super::formats::QuantizationFormat;

/// Default workspace size (4 GB) for activation and temporary buffers.
const DEFAULT_WORKSPACE_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Result of paged KV cache estimation.
#[derive(Debug, Clone)]
pub struct PagedKvEstimate {
    /// Total number of pages that fit in available KV memory per GPU.
    pub total_pages_per_gpu: usize,
    /// Bytes per page (2 * page_size * num_kv_heads * head_dim * bytes_per_element * num_attn_layers).
    pub page_bytes: usize,
    /// Total KV cache bytes used per GPU.
    pub kv_cache_bytes_per_gpu: usize,
    /// Maximum concurrent sessions given max context length.
    pub max_concurrent_sessions: usize,
    /// Number of attention layers using KV cache.
    pub num_attn_layers: usize,
}

/// Memory budget estimate for a model configuration.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Total VRAM available per GPU (bytes).
    pub total_vram_per_gpu: usize,
    /// Weight memory per GPU (bytes).
    pub weight_bytes_per_gpu: usize,
    /// KV cache memory per GPU (bytes, for full context).
    pub kv_cache_bytes_per_gpu: usize,
    /// Workspace memory per GPU (bytes).
    pub workspace_bytes_per_gpu: usize,
    /// Available memory for KV cache per GPU (bytes).
    pub available_for_kv: usize,
    /// Maximum position embeddings from model config.
    pub max_position_embeddings: usize,
}

impl MemoryBudget {
    /// Calculate memory budget for a given model, quant format, and GPU config.
    // @lat: [[lat#Memory Budget#Budget Calculation]]
    pub fn calculate(
        config: &ModelConfig,
        quant_format: QuantizationFormat,
        num_gpus: usize,
        vram_per_gpu: usize,
        gpu_utilization: f32,
    ) -> Self {
        let total_vram_per_gpu = (vram_per_gpu as f32 * gpu_utilization) as usize;

        // Calculate total weight bytes
        let total_weight_bytes = Self::estimate_weight_bytes(config, quant_format);
        let weight_bytes_per_gpu = total_weight_bytes / num_gpus;

        // Calculate KV cache size per GPU
        let kv_cache_bytes_per_gpu = Self::estimate_kv_cache_bytes(config, quant_format, num_gpus);

        // Workspace (activations, temp buffers)
        let workspace_bytes_per_gpu = DEFAULT_WORKSPACE_BYTES;

        let available_for_kv = total_vram_per_gpu
            .saturating_sub(weight_bytes_per_gpu)
            .saturating_sub(workspace_bytes_per_gpu);

        Self {
            total_vram_per_gpu,
            weight_bytes_per_gpu,
            kv_cache_bytes_per_gpu,
            workspace_bytes_per_gpu,
            available_for_kv,
            max_position_embeddings: config.max_position_embeddings,
        }
    }

    /// Estimate total weight size in bytes for the given quant format.
    // @lat: [[lat#Memory Budget#Weight Size Estimation]]
    pub fn estimate_weight_bytes(config: &ModelConfig, format: QuantizationFormat) -> usize {
        // Rough estimate based on parameter count and quantization
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        // Embedding: vocab_size * hidden_size * 2 (BF16)
        let embedding = vocab_size * hidden_size;

        // Per-layer: attention + MLP + norms
        // Attention: (num_heads + 2 * num_kv_heads) * head_dim * hidden_size * 2 (Q+K+V+O projections)
        let attn = (num_heads + 2 * num_kv_heads) * head_dim * hidden_size;
        // MLP: 3 * hidden_size * intermediate_size (gate + up + down)
        let mlp = 3 * hidden_size * intermediate_size;
        // Norms: ~2 * hidden_size (RMSNorm)
        let norms = 2 * hidden_size;

        let per_layer = attn + mlp + norms;
        let total_params = embedding + num_layers * per_layer + hidden_size; // + final norm

        // LM head (may be tied)
        let lm_head = if !config.tie_word_embeddings {
            vocab_size * hidden_size
        } else {
            0
        };
        let total_params = total_params + lm_head;

        // Multiply by bytes per parameter based on format
        let bytes_per_param: usize = match format {
            QuantizationFormat::Bf16 => 2,
            QuantizationFormat::PrismaScout => 1, // ~50% NVFP4 + ~50% BF16, average ~1 byte
            QuantizationFormat::AutoRound => 1,    // INT4 = 0.5 bytes, but scales/zeros add overhead
            QuantizationFormat::Gguf => 1,        // Varies by quant level, ~1 byte average
        };

        total_params * bytes_per_param
    }

    /// Estimate KV cache bytes per GPU for full context.
    // @lat: [[lat#Memory Budget#KV Cache Estimation]]
    fn estimate_kv_cache_bytes(
        config: &ModelConfig,
        format: QuantizationFormat,
        num_gpus: usize,
    ) -> usize {
        // Only full attention layers use paged KV cache
        let num_attn_layers = config.num_full_attention_layers();

        // KV per token per attention layer:
        // 2 (K+V) * num_kv_heads * head_dim * bytes_per_kv_element
        let bytes_per_kv_element = match format {
            QuantizationFormat::PrismaScout | QuantizationFormat::AutoRound => 2, // FP8 for KV in quantized models
            _ => 2, // BF16
        };

        let kv_per_token_per_layer = 2 * config.num_key_value_heads * config.head_dim * bytes_per_kv_element;

        // Total KV across all attention layers for full context
        let total_kv = num_attn_layers * kv_per_token_per_layer * config.max_position_embeddings;

        // Split across GPUs
        total_kv / num_gpus
    }

    /// Estimate maximum concurrent sessions given average context length.
    // @lat: [[lat#Memory Budget#Concurrent Session Planning]]
    pub fn max_concurrent_sessions(&self, avg_context_len: usize) -> usize {
        if self.kv_cache_bytes_per_gpu == 0 || avg_context_len == 0 {
            return 0;
        }
        // KV cache scales linearly with context length
        let kv_per_session = self.kv_cache_bytes_per_gpu * avg_context_len / self.max_position_embeddings;
        self.available_for_kv / kv_per_session.max(1)
    }

    /// Estimate paged KV cache bytes per GPU using block-based allocation.
    ///
    /// Unlike the flat model which allocates max_seq_len for every sequence,
    /// the paged model allocates pages on demand. Total page pool size is:
    /// `num_attn_layers * total_pages_per_gpu * page_size * 2 * kv_per_token_per_layer`
    ///
    /// where total_pages_per_gpu = available_kv_memory / page_bytes
    // @lat: [[lat#Memory Budget#KV Cache Estimation]]
    pub fn estimate_paged_kv_cache_bytes(
        config: &ModelConfig,
        format: QuantizationFormat,
        _num_gpus: usize,
        page_size: usize,
        available_kv_bytes: usize,
    ) -> PagedKvEstimate {
        let num_attn_layers = config.num_full_attention_layers();

        let bytes_per_kv_element = match format {
            QuantizationFormat::PrismaScout | QuantizationFormat::AutoRound => 2,
            _ => 2,
        };

        let kv_per_token_per_layer = 2 * config.num_key_value_heads * config.head_dim * bytes_per_kv_element;

        // Bytes per page per layer
        let page_bytes_per_layer = page_size * kv_per_token_per_layer;
        // Bytes per page across ALL attention layers (shared pool)
        let page_bytes = page_bytes_per_layer * num_attn_layers;

        // Total pages that fit in available memory
        let total_pages_per_gpu = available_kv_bytes / page_bytes.max(1);

        // Actual memory used
        let kv_cache_bytes_per_gpu = total_pages_per_gpu * page_bytes;

        // Pages needed per session for average context
        // Each session needs ceil(avg_context / page_size) pages per attention layer
        // Max concurrent = total_pages / (pages_per_session * num_attn_layers)
        let pages_per_session_per_layer = (config.max_position_embeddings + page_size - 1) / page_size;
        let max_concurrent_sessions = if pages_per_session_per_layer > 0 && num_attn_layers > 0 {
            total_pages_per_gpu / (pages_per_session_per_layer * num_attn_layers)
        } else {
            0
        };

        PagedKvEstimate {
            total_pages_per_gpu,
            page_bytes,
            kv_cache_bytes_per_gpu,
            max_concurrent_sessions,
            num_attn_layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn qwen3_6_config() -> ModelConfig {
        serde_json::from_str(
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","num_hidden_layers":64,"hidden_size":5120,"intermediate_size":17408,"vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"max_position_embeddings":262144,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":false,"rope_theta":10000000.0,"partial_rotary_factor":0.25,"mrope_interleaved":true,"mrope_section":[11,11,10]}"#,
        )
        .unwrap()
    }

    #[test]
    fn test_memory_budget_bf16() {
        let config = qwen3_6_config();
        let budget = MemoryBudget::calculate(
            &config,
            QuantizationFormat::Bf16,
            2, // TP=2
            32 * 1024 * 1024 * 1024, // 32 GB per GPU
            0.85,
        );

        // BF16 weights should be very large
        assert!(
            budget.weight_bytes_per_gpu > 10_000_000_000,
            "BF16 weights should be > 10 GB per GPU, got {}",
            budget.weight_bytes_per_gpu
        );
        assert!(
            budget.weight_bytes_per_gpu < 30_000_000_000,
            "BF16 weights should be < 30 GB per GPU (TP=2), got {}",
            budget.weight_bytes_per_gpu
        );
    }

    #[test]
    fn test_memory_budget_prisma_scout() {
        let config = qwen3_6_config();
        let budget = MemoryBudget::calculate(
            &config,
            QuantizationFormat::PrismaScout,
            2, // TP=2
            32 * 1024 * 1024 * 1024, // 32 GB per GPU
            0.85,
        );

        // PrismaSCOUT should be smaller than BF16
        assert!(
            budget.weight_bytes_per_gpu < 15_000_000_000,
            "PrismaSCOUT weights should be < 15 GB per GPU, got {}",
            budget.weight_bytes_per_gpu
        );
    }

    #[test]
    fn test_memory_budget_auto_round() {
        let config = qwen3_6_config();
        let budget = MemoryBudget::calculate(
            &config,
            QuantizationFormat::AutoRound,
            2, // TP=2
            32 * 1024 * 1024 * 1024, // 32 GB per GPU
            0.85,
        );

        // AutoRound INT4 should be smaller than PrismaSCOUT
        assert!(
            budget.weight_bytes_per_gpu < 12_000_000_000,
            "AutoRound weights should be < 12 GB per GPU, got {}",
            budget.weight_bytes_per_gpu
        );
    }

    #[test]
    fn test_kv_cache_size() {
        let config = qwen3_6_config();
        let budget = MemoryBudget::calculate(
            &config,
            QuantizationFormat::Bf16,
            2, // TP=2
            32 * 1024 * 1024 * 1024,
            0.85,
        );

        // KV cache should be substantial for 262K context
        // 16 full attention layers * 2 * 4 KV heads * 256 head_dim * 2 bytes * 262144 tokens / 2 GPUs
        assert!(
            budget.kv_cache_bytes_per_gpu > 1_000_000_000,
            "KV cache should be > 1 GB per GPU, got {}",
            budget.kv_cache_bytes_per_gpu
        );
    }

    #[test]
    fn test_available_for_kv() {
        let config = qwen3_6_config();
        let budget = MemoryBudget::calculate(
            &config,
            QuantizationFormat::PrismaScout,
            2,
            32 * 1024 * 1024 * 1024,
            0.85,
        );

        // Should have some available KV memory
        assert!(
            budget.available_for_kv > 0,
            "Should have available KV memory"
        );
    }

    #[test]
    fn test_estimate_weight_bytes_gguf() {
        let config = qwen3_6_config();
        let bytes = MemoryBudget::estimate_weight_bytes(&config, QuantizationFormat::Gguf);
        // GGUF should be roughly half of BF16
        assert!(bytes < 50_000_000_000, "GGUF weights should be reasonable, got {}", bytes);
        assert!(bytes > 10_000_000_000, "GGUF weights should be substantial, got {}", bytes);
    }

    #[test]
    fn test_max_concurrent_sessions() {
        let config = qwen3_6_config();
        let budget = MemoryBudget::calculate(
            &config,
            QuantizationFormat::PrismaScout,
            2,
            32 * 1024 * 1024 * 1024,
            0.85,
        );

        // With PrismaSCOUT, there should be KV room for at least a few sessions
        let sessions = budget.max_concurrent_sessions(8192);
        assert!(sessions > 0, "Should support at least 1 concurrent session");
    }

    #[test]
    fn test_max_concurrent_sessions_zero_context() {
        let config = qwen3_6_config();
        let budget = MemoryBudget::calculate(
            &config,
            QuantizationFormat::Bf16,
            2,
            32 * 1024 * 1024 * 1024,
            0.85,
        );

        // Zero context length should return 0
        assert_eq!(budget.max_concurrent_sessions(0), 0);
    }

    #[test]
    fn test_paged_kv_estimate_basic() {
        let config = qwen3_6_config();
        let page_size = 16;
        // 2 * 4 KV heads * 256 head_dim * 2 bytes = 4096 bytes per token per layer
        let kv_per_token_per_layer = 2 * config.num_key_value_heads * config.head_dim * 2;
        // page_bytes = page_size * kv_per_token_per_layer * num_attn_layers
        let expected_page_bytes = page_size * kv_per_token_per_layer * config.num_full_attention_layers();

        let estimate = MemoryBudget::estimate_paged_kv_cache_bytes(
            &config,
            QuantizationFormat::Bf16,
            2,
            page_size,
            4 * 1024 * 1024 * 1024, // 4 GB available
        );

        assert_eq!(estimate.page_bytes, expected_page_bytes, "page_bytes mismatch");
        assert_eq!(estimate.num_attn_layers, config.num_full_attention_layers());
    }

    #[test]
    fn test_paged_kv_estimate_with_budget() {
        let config = qwen3_6_config();
        let available = 2 * 1024 * 1024 * 1024; // 2 GB

        let estimate = MemoryBudget::estimate_paged_kv_cache_bytes(
            &config,
            QuantizationFormat::Bf16,
            2,
            16,
            available,
        );

        // total_pages * page_bytes should not exceed available
        assert!(
            estimate.kv_cache_bytes_per_gpu <= available,
            "KV cache bytes {} exceeds available {}",
            estimate.kv_cache_bytes_per_gpu,
            available
        );
        // total_pages * page_bytes should equal kv_cache_bytes_per_gpu
        assert_eq!(
            estimate.total_pages_per_gpu * estimate.page_bytes,
            estimate.kv_cache_bytes_per_gpu,
            "KV cache bytes mismatch"
        );
    }

    #[test]
    fn test_paged_kv_estimate_concurrent_sessions() {
        let config = qwen3_6_config();
        // page_bytes = 16 * 4096 * 16 = 1,048,576
        // pages_per_session_per_layer = ceil(262144 / 16) = 16384
        // One session needs 16384 * 16 = 262,144 pages = 256 GB
        // Use enough memory for 2 sessions
        let available = 512 * 1024 * 1024 * 1024; // 512 GB

        let estimate = MemoryBudget::estimate_paged_kv_cache_bytes(
            &config,
            QuantizationFormat::Bf16,
            2,
            16,
            available,
        );

        // Verify max_concurrent_sessions formula
        let pages_per_session = (config.max_position_embeddings + 16 - 1) / 16;
        let expected_max = estimate.total_pages_per_gpu / (pages_per_session * estimate.num_attn_layers);
        assert_eq!(
            estimate.max_concurrent_sessions, expected_max,
            "max_concurrent_sessions mismatch"
        );
        // Should be at least 2 sessions with 512 GB
        assert!(
            estimate.max_concurrent_sessions >= 2,
            "Should support at least 2 sessions with 512 GB, got {}",
            estimate.max_concurrent_sessions
        );
    }

    #[test]
    fn test_paged_kv_estimate_page_size_16() {
        let config = qwen3_6_config();
        let available = 4 * 1024 * 1024 * 1024; // 4 GB

        let estimate_16 = MemoryBudget::estimate_paged_kv_cache_bytes(
            &config,
            QuantizationFormat::Bf16,
            2,
            16,
            available,
        );

        // page_bytes with size 16
        let kv_per_token_per_layer = 2 * config.num_key_value_heads * config.head_dim * 2;
        let expected_page_bytes_16 = 16 * kv_per_token_per_layer * config.num_full_attention_layers();
        assert_eq!(estimate_16.page_bytes, expected_page_bytes_16);

        // Verify page_bytes is positive
        assert!(estimate_16.page_bytes > 0, "page_bytes should be positive");
        // Verify total_pages is positive
        assert!(estimate_16.total_pages_per_gpu > 0, "total_pages should be positive");
    }

    #[test]
    fn test_paged_kv_estimate_page_size_32() {
        let config = qwen3_6_config();
        let available = 4 * 1024 * 1024 * 1024; // 4 GB

        let estimate_32 = MemoryBudget::estimate_paged_kv_cache_bytes(
            &config,
            QuantizationFormat::Bf16,
            2,
            32,
            available,
        );

        // page_bytes with size 32
        let kv_per_token_per_layer = 2 * config.num_key_value_heads * config.head_dim * 2;
        let expected_page_bytes_32 = 32 * kv_per_token_per_layer * config.num_full_attention_layers();
        assert_eq!(estimate_32.page_bytes, expected_page_bytes_32);

        // With larger page_size, fewer total pages but each page is bigger
        let estimate_16 = MemoryBudget::estimate_paged_kv_cache_bytes(
            &config,
            QuantizationFormat::Bf16,
            2,
            16,
            available,
        );

        assert!(
            estimate_32.page_bytes > estimate_16.page_bytes,
            "page_bytes for size 32 ({}) should be > size 16 ({})",
            estimate_32.page_bytes,
            estimate_16.page_bytes
        );
    }
}
