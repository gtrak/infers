//! General-purpose instrumentation probe for the forward pass.
//!
//! Dumps BF16 GPU tensors to disk during inference so they can be compared
//! against a reference implementation (e.g. Python / PyTorch).  Controlled by
//! environment variables so that production runs stay clean.

use std::path::PathBuf;
use std::sync::Arc;

use half::bf16;
use infers_cuda::{CudaSlice, CudaStream};
use infers_model::ModelConfig;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Probe configuration parsed from environment variables.
pub struct ProbeConfig {
    /// Master on/off switch.  When `false` the probe is a no-op.
    enabled: bool,

    /// Which layers to dump.  `None` means all layers.
    pub layers: Option<Vec<usize>>,

    /// Which stage prefixes to dump.  `None` means all stages.
    pub stages: Option<Vec<String>>,

    /// Output directory for dumped files.  `None` disables file I/O.
    pub dir: Option<String>,

    /// Print min/max/mean_abs stats to stderr.
    pub stats: bool,
}

impl ProbeConfig {
    /// Construct a disabled config (no dumping, no stats).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            layers: None,
            stages: None,
            dir: None,
            stats: false,
        }
    }

    /// Read probe configuration from environment variables.
    ///
    /// | Variable              | Description                                        | Default          |
    /// |-----------------------|----------------------------------------------------|------------------|
    /// | `INFERS_DUMP_DIR`     | Output directory (must be set for any dumping)      | *(disabled)*     |
    /// | `INFERS_DUMP_LAYERS`  | Comma-separated layer indices, or `all`             | all layers       |
    /// | `INFERS_DUMP_STAGES`  | Comma-separated stage prefixes                     | all stages       |
    /// | `INFERS_DUMP_STATS`   | `1` to print stats                                 | off              |
    pub fn from_env() -> Self {
        let dir = std::env::var("INFERS_DUMP_DIR").ok();
        let enabled = dir.is_some();

        let layers = if enabled {
            match std::env::var("INFERS_DUMP_LAYERS") {
                Ok(v) if v == "all" => None,
                Ok(v) => {
                    let indices: Vec<usize> = v
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if indices.is_empty() {
                        None
                    } else {
                        Some(indices)
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        };

        let stages = if enabled {
            match std::env::var("INFERS_DUMP_STAGES") {
                Ok(v) if v == "all" || v.is_empty() => None,
                Ok(v) => {
                    let parts: Vec<String> = v.split(',').map(|s| s.trim().to_string()).collect();
                    if parts.is_empty() {
                        None
                    } else {
                        Some(parts)
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        };

        let stats = if enabled {
            match std::env::var("INFERS_DUMP_STATS") {
                Ok(v) if v == "1" => true,
                _ => false,
            }
        } else {
            false
        };

        Self {
            enabled,
            layers,
            stages,
            dir,
            stats,
        }
    }

    /// Returns `true` if the given layer index is allowed by this config.
    fn layer_allowed(&self, layer: usize) -> bool {
        match &self.layers {
            Some(indices) => indices.contains(&layer),
            None => true,
        }
    }

    /// Returns `true` if the given stage prefix matches this config.
    ///
    /// When `stages` is `None`, all stages match.  Otherwise checks whether
    /// any filter prefix appears at the start of the stage string.
    fn stage_allowed(&self, stage: &str) -> bool {
        match &self.stages {
            Some(prefixes) => prefixes.iter().any(|p| stage.starts_with(p)),
            None => true,
        }
    }

    /// Returns `true` if a dump at the given layer+stage should happen.
    pub fn should_dump(&self, layer: usize, stage: &str) -> bool {
        self.enabled && self.dir.is_some() && self.layer_allowed(layer) && self.stage_allowed(stage)
    }

    /// Returns `true` if stats printing is active for the given layer+stage.
    pub fn should_stats(&self, layer: usize, stage: &str) -> bool {
        self.enabled && self.stats && self.layer_allowed(layer) && self.stage_allowed(stage)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn layer_dir(dir: &str, layer: usize) -> PathBuf {
    PathBuf::from(dir).join(format!("layer_{layer}"))
}

/// Compute (min, max, mean_abs) over a BF16 slice.
fn bf16_stats(cpu: &[bf16]) -> (f32, f32, f64) {
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut sum_abs = 0.0f64;

    for v in cpu {
        let f = v.to_f32();
        if f < min_val {
            min_val = f;
        }
        if f > max_val {
            max_val = f;
        }
        sum_abs += (f as f64).abs();
    }

    let mean_abs = sum_abs / cpu.len() as f64;
    (min_val, max_val, mean_abs)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Dump a BF16 GPU tensor to disk under `{dir}/layer_{layer}/{stage}_gpu{gpu}.raw`.
///
/// Also writes a `.meta` JSON sidecar file.  When `stats` is enabled, prints
/// min/max/mean_abs to stderr.
pub fn dump(
    stream: &Arc<CudaStream>,
    config: &ProbeConfig,
    layer: usize,
    gpu: usize,
    stage: &str,
    tensor: &CudaSlice<bf16>,
    shape: &[usize],
) {
    if !config.should_dump(layer, stage) {
        return;
    }

    let cpu: Vec<bf16> = match stream.clone_dtoh(tensor) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("WARN probe dump clone_dtoh failed for layer_{layer}/{stage}_gpu{gpu}: {e}");
            return;
        }
    };

    let dir = config.dir.as_ref().unwrap(); // safe because should_dump checks
    let ldir = layer_dir(dir, layer);
    if let Err(e) = std::fs::create_dir_all(&ldir) {
        eprintln!("WARN probe create_dir failed: {e}");
        return;
    }

    let name = format!("{stage}_gpu{gpu}");

    // Write raw BF16 bytes (little-endian)
    let bytes: Vec<u8> = cpu.iter().flat_map(|v| v.to_le_bytes()).collect();
    if let Err(e) = std::fs::write(ldir.join(format!("{name}.raw")), &bytes) {
        eprintln!("WARN probe write .raw failed for layer_{layer}/{stage}_gpu{gpu}: {e}");
    }

    // Write metadata JSON
    let meta = serde_json::json!({
        "name": name,
        "layer": layer,
        "gpu": gpu,
        "shape": shape,
        "dtype": "bf16",
        "stage": stage,
    });
    if let Err(e) = std::fs::write(
        ldir.join(format!("{name}.meta")),
        serde_json::to_string_pretty(&meta).unwrap_or_default().as_bytes(),
    ) {
        eprintln!("WARN probe write .meta failed for layer_{layer}/{stage}_gpu{gpu}: {e}");
    }

    // Stats to stderr
    if config.stats {
        let (min_val, max_val, mean_abs) = bf16_stats(&cpu);
        eprintln!(
            "probe stats layer={layer} gpu={gpu} stage={} shape=[{}] min={:.4} max={:.4} mean_abs={:.6}",
            stage,
            shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
            min_val,
            max_val,
            mean_abs,
        );
    }
}

/// Print stats for a BF16 GPU tensor to stderr (no file I/O).
pub fn stats(
    stream: &Arc<CudaStream>,
    config: &ProbeConfig,
    layer: usize,
    gpu: usize,
    stage: &str,
    tensor: &CudaSlice<bf16>,
) {
    if !config.should_stats(layer, stage) {
        return;
    }

    let cpu: Vec<bf16> = match stream.clone_dtoh(tensor) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("WARN probe stats clone_dtoh failed for layer_{layer}/{stage}_gpu{gpu}: {e}");
            return;
        }
    };

    let (min_val, max_val, mean_abs) = bf16_stats(&cpu);
    let shape_str = "N/A"; // we don't have shape info here; could add it if needed
    eprintln!(
        "probe stats layer={layer} gpu={gpu} stage={} shape=[{}] min={:.4} max={:.4} mean_abs={:.6}",
        stage, shape_str, min_val, max_val, mean_abs,
    );
}

/// Write a `config.json` to the dump directory with model parameters needed by
/// the Python comparison framework.
pub fn dump_config(model_config: &ModelConfig, num_gpus: usize, group_size: usize) {
    let dir = match std::env::var("INFERS_DUMP_DIR") {
        Ok(d) => d,
        Err(_) => return, // no dump dir configured
    };

    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("WARN probe create_dir failed for config.json: {e}");
        return;
    }

    let layer_types: Vec<String> = (0..model_config.num_hidden_layers)
        .map(|i| {
            match model_config.get_layer_type(i) {
                infers_model::LayerType::GatedDeltaNet => "gdn".to_string(),
                infers_model::LayerType::FullAttention => "full_attention".to_string(),
            }
        })
        .collect();

    let cfg = serde_json::json!({
        "hidden_size": model_config.hidden_size,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": model_config.num_key_value_heads,
        "head_dim": model_config.head_dim,
        "intermediate_size": model_config.intermediate_size,
        "num_hidden_layers": model_config.num_hidden_layers,
        "layer_types": layer_types,
        "vocab_size": model_config.vocab_size,
        "num_gpus": num_gpus,
        "group_size": group_size,
        "attn_output_gate": model_config.attn_output_gate,
        "rms_norm_eps": model_config.rms_norm_eps,
        "rope_theta": model_config.rope_theta,
        "partial_rotary_factor": model_config.partial_rotary_factor,
    });

    if let Err(e) = std::fs::write(
        PathBuf::from(dir).join("config.json"),
        serde_json::to_string_pretty(&cfg).unwrap_or_default().as_bytes(),
    ) {
        eprintln!("WARN probe write config.json failed: {e}");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn set_env(key: &str, val: &str) {
        unsafe { std::env::set_var(key, val) };
    }

    fn unset_env(key: &str) {
        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn from_env_disabled_when_no_dir() {
        // Unset all env vars to ensure clean state
        unset_env("INFERS_DUMP_DIR");
        unset_env("INFERS_DUMP_LAYERS");
        unset_env("INFERS_DUMP_STAGES");
        unset_env("INFERS_DUMP_STATS");

        let config = ProbeConfig::from_env();
        assert!(!config.enabled);
        assert!(config.dir.is_none());
    }

    #[test]
    fn from_env_enabled_with_all_defaults() {
        set_env("INFERS_DUMP_DIR", "/tmp/dump");
        unset_env("INFERS_DUMP_LAYERS");
        unset_env("INFERS_DUMP_STAGES");
        unset_env("INFERS_DUMP_STATS");

        let config = ProbeConfig::from_env();
        assert!(config.enabled);
        assert_eq!(config.dir, Some("/tmp/dump".to_string()));
        assert!(config.layers.is_none()); // all layers
        assert!(config.stages.is_none()); // all stages
        assert!(!config.stats);

        unset_env("INFERS_DUMP_DIR");
    }

    #[test]
    fn from_env_layers_filter() {
        set_env("INFERS_DUMP_DIR", "/tmp/dump");
        set_env("INFERS_DUMP_LAYERS", "0,3,10");

        let config = ProbeConfig::from_env();
        assert_eq!(config.layers, Some(vec![0, 3, 10]));

        // Cleanup
        unset_env("INFERS_DUMP_DIR");
        unset_env("INFERS_DUMP_LAYERS");
    }

    #[test]
    fn from_env_stages_filter() {
        set_env("INFERS_DUMP_DIR", "/tmp/dump");
        set_env("INFERS_DUMP_STAGES", "attn,mlp");

        let config = ProbeConfig::from_env();
        assert_eq!(config.stages, Some(vec!["attn".to_string(), "mlp".to_string()]));

        unset_env("INFERS_DUMP_DIR");
        unset_env("INFERS_DUMP_STAGES");
    }

    #[test]
    fn from_env_stats_enabled() {
        set_env("INFERS_DUMP_DIR", "/tmp/dump");
        set_env("INFERS_DUMP_STATS", "1");

        let config = ProbeConfig::from_env();
        assert!(config.stats);

        unset_env("INFERS_DUMP_DIR");
        unset_env("INFERS_DUMP_STATS");
    }

    #[test]
    fn should_dump_allows_all_when_filters_none() {
        let config = ProbeConfig {
            enabled: true,
            layers: None,
            stages: None,
            dir: Some("/tmp/dump".to_string()),
            stats: false,
        };

        assert!(config.should_dump(0, "attn.norm1"));
        assert!(config.should_dump(42, "mlp.up_proj"));
    }

    #[test]
    fn should_dump_rejects_when_layer_filtered() {
        let config = ProbeConfig {
            enabled: true,
            layers: Some(vec![0, 3]),
            stages: None,
            dir: Some("/tmp/dump".to_string()),
            stats: false,
        };

        assert!(config.should_dump(0, "attn.q_proj"));
        assert!(!config.should_dump(1, "attn.q_proj"));
        assert!(config.should_dump(3, "mlp.gate"));
    }

    #[test]
    fn should_dump_matches_stage_prefix() {
        let config = ProbeConfig {
            enabled: true,
            layers: None,
            stages: Some(vec!["attn".to_string()]),
            dir: Some("/tmp/dump".to_string()),
            stats: false,
        };

        assert!(config.should_dump(5, "attn.norm1"));
        assert!(config.should_dump(5, "attn.q_proj"));
        assert!(!config.should_dump(5, "mlp.up_proj"));
    }

    #[test]
    fn should_dump_disabled_without_dir() {
        let config = ProbeConfig {
            enabled: true,
            layers: None,
            stages: None,
            dir: None,
            stats: false,
        };

        assert!(!config.should_dump(0, "attn.q_proj"));
    }

    #[test]
    fn bf16_stats_basic() {
        let vals = vec![bf16::from_f32(-1.5), bf16::from_f32(0.0), bf16::from_f32(2.5)];
        let (min_val, max_val, mean_abs) = bf16_stats(&vals);

        assert_eq!(min_val, -1.5);
        assert_eq!(max_val, 2.5);
        // (1.5 + 0.0 + 2.5) / 3 = 4.0 / 3
        assert!((mean_abs - 1.333333).abs() < 1e-4);
    }

    #[test]
    fn disabled_config_never_dumps_or_stats() {
        let config = ProbeConfig::disabled();
        assert!(!config.should_dump(0, "attn"));
        assert!(!config.should_stats(0, "attn"));
    }
}
