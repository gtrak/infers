//! Sampling strategies for token generation.
//!
//! Supports greedy sampling (argmax), temperature scaling, top-k filtering,
//! top-p nucleus sampling, and penalty application — all operating on CPU
//! `&mut [f32]` logits downloaded from GPU when needed.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use half::bf16;
use infers_cuda::{CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};
use infers_scheduler::{SamplingConfig, SamplingStrategy};

/// Xoshiro256++ pseudo-random number generator.
/// Fast, high-quality, seedable. No external dependency.
#[derive(Debug, Clone)]
pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    pub fn from_seed(seed: u64) -> Self {
        // Use SplitMix64 to expand the seed into 4 state words
        let mut sm = SplitMix64 { state: seed };
        Self {
            s: [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()],
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Generate a float in [0, 1) using the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        let x = self.next_u64();
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

/// Create a random seed from system entropy (e.g., for unseeded sessions).
pub fn random_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Apply repetition, presence, and frequency penalties to logits.
///
/// - Repetition penalty: `logit *= repetition_penalty` if logit > 0, else `logit /= repetition_penalty`
///   Applied to ALL tokens that appear in token_history.
/// - Presence penalty: subtract `presence_penalty` from logit for any token in token_history.
/// - Frequency penalty: subtract `count_in_history * frequency_penalty` for each token.
///
/// Per the OpenAI API spec: presence and frequency penalties only apply to GENERATED tokens,
/// not prompt tokens. `num_prompt_tokens` is the count of prompt tokens at the start of
/// token_history; tokens after that index are generated.
pub fn apply_penalties(
    logits: &mut [f32],
    token_history: &[u32],
    num_prompt_tokens: usize,
    config: &SamplingConfig,
) {
    if config.repetition_penalty == 1.0
        && config.presence_penalty == 0.0
        && config.frequency_penalty == 0.0
    {
        return; // fast path — no penalties
    }

    let vocab_size = logits.len();
    let mut token_counts: Vec<u32> = vec![0u32; vocab_size];
    for &token in &token_history[num_prompt_tokens..] {
        if (token as usize) < vocab_size {
            token_counts[token as usize] += 1;
        }
    }

    let mut prompt_seen: Vec<bool> = vec![false; vocab_size];
    for &token in &token_history[..num_prompt_tokens.min(token_history.len())] {
        if (token as usize) < vocab_size {
            prompt_seen[token as usize] = true;
        }
    }

    for (i, logit) in logits.iter_mut().enumerate() {
        let in_prompt = prompt_seen[i];
        let gen_count = token_counts[i];

        // Repetition penalty applies to ANY token in the full history
        if in_prompt || gen_count > 0 {
            if config.repetition_penalty != 1.0 {
                if *logit > 0.0 {
                    *logit *= config.repetition_penalty;
                } else {
                    *logit /= config.repetition_penalty;
                }
            }
        }

        // Presence and frequency penalties only apply to GENERATED tokens
        if gen_count > 0 {
            *logit -= config.presence_penalty;
            *logit -= gen_count as f32 * config.frequency_penalty;
        }
    }
}

/// Divide all logits by temperature. Clamps temperature to minimum 1e-8
/// to avoid division by zero (effectively greedy).
pub fn temperature_scale(logits: &mut [f32], temp: f32) {
    if temp <= 0.0 {
        // Temperature 0 = greedy: no scaling, argmax will dominate
        return;
    }
    let t = temp.max(1e-8);
    for logit in logits.iter_mut() {
        *logit /= t;
    }
}

/// Zero out (set to -f32::INFINITY) all logits except the top-k.
pub fn top_k_filter(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return; // no filtering
    }

    let mut values: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    values.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // After select_nth_unstable_by, values[..k] contains the k largest (unordered)
    let top_k_set: HashSet<usize> = values[..k].iter().map(|&(i, _)| i).collect();
    for (i, logit) in logits.iter_mut().enumerate() {
        if !top_k_set.contains(&i) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Numerically stable in-place softmax: subtract max, exponentiate, normalize.
pub fn softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    // Find max for numerical stability
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Subtract max and exponentiate
    let mut sum = 0.0f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max_val).exp();
        sum += *logit;
    }
    // Normalize
    if sum > 0.0 {
        for logit in logits.iter_mut() {
            *logit /= sum;
        }
    }
}

/// Sample from the cumulative probability distribution using top-p filtering.
///
/// Sorts probabilities descending, accumulates until cumulative > p,
/// then does a weighted random sample from the surviving tokens.
///
/// Returns the token index.
pub fn top_p_sample(probs: &[f32], p: f64, rng: &mut Xoshiro256PlusPlus) -> usize {
    if p >= 1.0 {
        // No filtering — weighted sample from full distribution
        return weighted_sample(probs, rng);
    }

    // Sort indices by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Accumulate until cumulative > p
    let mut cumulative = 0.0f64;
    let mut cutoff = indexed.len(); // default: all tokens
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cumulative += prob as f64;
        if cumulative > p {
            cutoff = i + 1;
            break;
        }
    }

    // Weighted sample from the top-p tokens
    let top_probs: Vec<f32> = indexed[..cutoff].iter().map(|&(_, p)| p).collect();
    let local_idx = weighted_sample(&top_probs, rng);
    indexed[local_idx].0
}

/// Weighted random sample from a probability distribution.
fn weighted_sample(probs: &[f32], rng: &mut Xoshiro256PlusPlus) -> usize {
    let r = rng.next_f64() as f32;
    let mut cumulative = 0.0f32;
    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if cumulative > r {
            return i;
        }
    }
    // Fallback: last token (handles floating-point rounding)
    probs.len() - 1
}

/// Main sampling dispatch: given raw BF16 logits on GPU, apply the full
/// sampling pipeline based on SamplingConfig.strategy, return sampled token ID.
///
/// Greedy path (no penalties, no temperature scaling): stays on GPU via
/// `greedy_sample_bf16()` — no logits download.
///
/// Non-greedy: download logits → BF16→F32 → penalties → temperature → top_k →
/// softmax → top_p or weighted sample → token ID.
pub fn sample_with_config(
    stream: &Arc<CudaStream>,
    gpu_logits: &CudaView<'_, bf16>,
    argmax_kernel: &CudaFunction,
    config: &SamplingConfig,
    token_history: &[u32],
    num_prompt_tokens: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<u32> {
    // Fast path: pure greedy with no penalties → GPU argmax (no download)
    if matches!(config.strategy, SamplingStrategy::Greedy)
        && config.repetition_penalty == 1.0
        && config.presence_penalty == 0.0
        && config.frequency_penalty == 0.0
    {
        return greedy_sample_bf16(stream, argmax_kernel, gpu_logits);
    }

    // Slow path: download logits and sample on CPU
    let logits_bf16: Vec<bf16> = stream.clone_dtoh(gpu_logits)?;
    let mut logits: Vec<f32> = logits_bf16.iter().map(|&v| v.to_f32()).collect();

    // Apply penalties
    apply_penalties(&mut logits, token_history, num_prompt_tokens, config);

    match &config.strategy {
        SamplingStrategy::Greedy => {
            // Greedy with penalties: just argmax on the penalized logits
            let max_idx = logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            Ok(max_idx as u32)
        }
        SamplingStrategy::Temperature { temp } => {
            temperature_scale(&mut logits, *temp);
            softmax(&mut logits);
            Ok(weighted_sample(&logits, rng) as u32)
        }
        SamplingStrategy::TopK { k, temp } => {
            temperature_scale(&mut logits, *temp);
            top_k_filter(&mut logits, *k);
            softmax(&mut logits);
            Ok(weighted_sample(&logits, rng) as u32)
        }
        SamplingStrategy::TopP { p, temp } => {
            temperature_scale(&mut logits, *temp);
            softmax(&mut logits);
            Ok(top_p_sample(&logits, *p, rng) as u32)
        }
    }
}

/// Check if a sampled token should stop generation.
/// Returns true if token matches eos_token_id or any stop_token_ids.
pub fn should_stop(token: u32, config: &SamplingConfig) -> bool {
    if let Some(eos) = config.eos_token_id {
        if token == eos {
            return true;
        }
    }
    config.stop_token_ids.contains(&token)
}

/// Greedy sampling: find the token ID with the highest logit.
///
/// Uses the `infers_argmax_f32` kernel to compute argmax on GPU,
/// then copies the result back to host memory.
///
/// # Arguments
/// * `stream` — CUDA stream for kernel launch
/// * `kernel` — Loaded function handle for `infers_argmax_f32`
/// * `logits` — FP32 logit vector `[vocab_size]`
///
/// # Returns
/// The token ID with the highest logit
pub fn greedy_sample(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    logits: &CudaSlice<f32>,
) -> Result<u32> {
    let vocab_size = logits.len();
    anyhow::ensure!(vocab_size > 0, "Logit vector must not be empty");

    // Allocate result buffer on device (a single i32 for the argmax index)
    let mut result_gpu = stream
        .alloc_zeros::<i32>(1)
        .map_err(|e| anyhow::anyhow!("Failed to allocate argmax result: {e}"))?;

    let vocab_size_i32 = vocab_size as i32;
    let batch_size_i32 = 1i32;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1), // single block for argmax
        block_dim: (256, 1, 1),
        shared_mem_bytes: (256 * 8) as u32, // shared mem for reduction (2 values per thread: val + idx)
    };

    unsafe {
        stream
            .launch_builder(kernel)
            .arg(logits)
            .arg(&mut result_gpu)
            .arg(&batch_size_i32)
            .arg(&vocab_size_i32)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("Argmax kernel launch failed: {e}"))?;
    }

    // Copy result back to host
    let result_host = stream
        .clone_dtoh(&result_gpu)
        .map_err(|e| anyhow::anyhow!("Failed to copy argmax result from device: {e}"))?;

    Ok(result_host[0] as u32)
}

/// Greedy sampling directly on BF16 logits (no CPU round-trip).
///
/// Uses the `infers_argmax_bf16` kernel to compute argmax on GPU,
/// avoiding the download→convert→upload cycle of the F32 path.
///
/// # Arguments
/// * `stream` — CUDA stream for kernel launch
/// * `kernel` — Loaded function handle for `infers_argmax_bf16`
/// * `logits` — BF16 logit view `[vocab_size]`
///
/// # Returns
/// The token ID with the highest logit
pub fn greedy_sample_bf16(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    logits: &CudaView<'_, bf16>,
) -> Result<u32> {
    let vocab_size = logits.len();
    anyhow::ensure!(vocab_size > 0, "Logit vector must not be empty");

    // Allocate result buffer on device (a single i32 for the argmax index)
    let mut result_gpu = stream
        .alloc_zeros::<i32>(1)
        .map_err(|e| anyhow::anyhow!("Failed to allocate argmax result: {e}"))?;

    let vocab_size_i32 = vocab_size as i32;
    let batch_size_i32 = 1i32;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1), // single block for argmax
        block_dim: (256, 1, 1),
        shared_mem_bytes: (256 * 8) as u32, // shared mem for reduction (2 values per thread: val + idx)
    };

    unsafe {
        stream
            .launch_builder(kernel)
            .arg(logits)
            .arg(&mut result_gpu)
            .arg(&batch_size_i32)
            .arg(&vocab_size_i32)
            .launch(config)
            .map_err(|e| anyhow::anyhow!("Argmax BF16 kernel launch failed: {e}"))?;
    }

    // Copy result back to host (single i32 — minimal transfer)
    let result_host = stream
        .clone_dtoh(&result_gpu)
        .map_err(|e| anyhow::anyhow!("Failed to copy argmax result from device: {e}"))?;

    Ok(result_host[0] as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(strategy: SamplingStrategy) -> SamplingConfig {
        SamplingConfig {
            strategy,
            ..Default::default()
        }
    }

    #[test]
    fn test_xoshiro_deterministic() {
        let mut rng1 = Xoshiro256PlusPlus::from_seed(42);
        let mut rng2 = Xoshiro256PlusPlus::from_seed(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_xoshiro_different_seeds() {
        let mut rng1 = Xoshiro256PlusPlus::from_seed(42);
        let mut rng2 = Xoshiro256PlusPlus::from_seed(43);
        assert_ne!(rng1.next_u64(), rng2.next_u64());
    }

    #[test]
    fn test_apply_penalties_repetition() {
        let mut logits = vec![1.0f32, 2.0, 3.0, -1.0, -2.0];
        let token_history = vec![0u32, 1, 2, 3, 4]; // tokens 0-4 all in history
        let num_prompt = 5; // all are prompt tokens, so no gen penalties
        let config = SamplingConfig {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        apply_penalties(&mut logits, &token_history, num_prompt, &config);
        // Positive logits get multiplied by 2.0
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 4.0).abs() < 1e-6);
        assert!((logits[2] - 6.0).abs() < 1e-6);
        // Negative logits get divided by 2.0
        assert!((logits[3] - (-0.5)).abs() < 1e-6);
        assert!((logits[4] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_apply_penalties_presence_frequency() {
        let mut logits = vec![5.0f32, 3.0, 1.0, 0.0];
        let token_history = vec![10u32, 0, 0, 0]; // token 0 appears 3x generated, token 10 is prompt
        let num_prompt = 1; // only token 10 is prompt
        let config = SamplingConfig {
            presence_penalty: 0.5,
            frequency_penalty: 0.3,
            ..Default::default()
        };
        apply_penalties(&mut logits, &token_history, num_prompt, &config);
        // Token 0: 3 generated occurrences → presence -0.5, frequency -3*0.3=-0.9, total -1.4
        assert!((logits[0] - (5.0 - 0.5 - 0.9)).abs() < 1e-5);
        // Other tokens: not in generated history, unchanged
        assert!((logits[1] - 3.0).abs() < 1e-6);
        assert!((logits[2] - 1.0).abs() < 1e-6);
        assert!((logits[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_penalties_noop() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let token_history = vec![0u32];
        let config = SamplingConfig::default(); // all penalties at default
        apply_penalties(&mut logits, &token_history, 0, &config);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_scale() {
        let mut logits = vec![2.0f32, 1.0, 0.0];
        temperature_scale(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 0.5).abs() < 1e-6);
        assert!((logits[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_scale_zero_is_greedy() {
        let mut logits = vec![2.0f32, 1.0, 0.0];
        temperature_scale(&mut logits, 0.0);
        // Temperature 0 = no change (argmax dominates)
        assert!((logits[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_filter() {
        let mut logits = vec![1.0f32, 3.0, 2.0, 0.5, 4.0];
        top_k_filter(&mut logits, 2);
        // Top 2 are indices 1 (3.0) and 4 (4.0) — they survive
        // Others should be -inf
        assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
        assert!((logits[1] - 3.0).abs() < 1e-6);
        assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
        assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
        assert!((logits[4] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Largest input should get largest probability
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }

    #[test]
    fn test_softmax_stable_with_large_inputs() {
        let mut logits = vec![1000.0f32, 1001.0, 1002.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(logits.iter().all(|v| v.is_normal()));
    }

    #[test]
    fn test_should_stop_eos() {
        let config = SamplingConfig {
            eos_token_id: Some(2),
            ..Default::default()
        };
        assert!(should_stop(2, &config));
        assert!(!should_stop(1, &config));
    }

    #[test]
    fn test_should_stop_stop_tokens() {
        let config = SamplingConfig {
            stop_token_ids: vec![100, 200],
            ..Default::default()
        };
        assert!(should_stop(100, &config));
        assert!(should_stop(200, &config));
        assert!(!should_stop(1, &config));
    }

    #[test]
    fn test_weighted_sample_distribution() {
        // With a clear max, weighted sample should prefer it
        let probs = vec![0.01f32, 0.01, 0.98];
        let mut rng = Xoshiro256PlusPlus::from_seed(42);
        let mut counts = [0usize; 3];
        for _ in 0..1000 {
            let idx = weighted_sample(&probs, &mut rng);
            counts[idx] += 1;
        }
        // Token 2 should dominate
        assert!(counts[2] > 900);
    }

    #[test]
    fn test_top_p_sample() {
        // With top_p=0.7 and a distribution [0.5, 0.3, 0.2],
        // cumulative after first token = 0.5 (not > 0.7),
        // after second = 0.8 (> 0.7), so cutoff = 2 (tokens 0 and 1 only)
        let probs = vec![0.5f32, 0.3, 0.2];
        let mut rng = Xoshiro256PlusPlus::from_seed(42);
        let mut counts = [0usize; 3];
        for _ in 0..1000 {
            let idx = top_p_sample(&probs, 0.7, &mut rng);
            counts[idx] += 1;
        }
        // With a single seeded RNG, tokens 0 and 1 dominate.
        // Token 0 gets ~5/8 of samples, token 1 gets ~3/8, token 2 near 0.
        assert!(counts[0] > 0);
        assert!(counts[1] > 0);
        // Token 2 should be rare (cut off by top-p=0.7)
        assert!(counts[2] < 50);
    }

    #[test]
    fn test_make_config_helper() {
        let config = make_config(SamplingStrategy::Temperature { temp: 0.8 });
        assert!(matches!(config.strategy, SamplingStrategy::Temperature { .. }));
        assert!((config.repetition_penalty - 1.0).abs() < 1e-6);
    }
}
