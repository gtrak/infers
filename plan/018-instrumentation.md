# Phase 13: General-Purpose Instrumentation

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-17
**Blocks**: Current bug investigation (garbage output), future model debugging
**Blocked by**: Nothing
**Rationale**: The engine produces garbage output and the only way to find where it diverges is layer-by-layer reference comparison. The current debug instrumentation is ad-hoc copy-paste spread across engine.rs, attention.rs, and gdn.rs with three separate Python scripts that duplicate each other's code. A general-purpose system will serve this bug hunt *and* every future model integration.
---

## Goals

1. **Find the current bug** — layer-by-layer cosine comparison against a known-correct PyTorch reference will pinpoint exactly where the forward pass diverges.
2. **Make debugging reproducible** — one command to dump all intermediates, one command to compare, zero code changes.
3. **Make it model-agnostic** — adding a new architecture means adding a stage module, not copy-pasting another script.
4. **Make it zero-cost when disabled** — no GPU→CPU transfers, no allocations, no overhead in production.

## Current State

The engine has **three independent, ad-hoc debug systems**:

| System | Location | What it covers | Mechanism |
|--------|----------|----------------|-----------|
| Engine dumps | `engine.rs` lines 548–870 | Per-layer: hidden_input, norm1, attn_raw, attn_ar, residual_attn, norm2, MLP stages, residual_mlp | `if let Ok(ref dl) = std::env::var("INFERS_DEBUG_LAYER")` then `dump_bf16_tensor()` |
| Attention dumps | `attention.rs` lines 1275–1857 | Per-head: K/V/Q, scores, softmax, combined, gated, O-proj (layer 3 only) | `if debug_attn` (`INFERS_DEBUG_LAYER3`) + `INFERS_DUMP_LAYER_DIR` |
| GDN dumps | `gdn.rs` lines 295–667 | All GDN sub-stages | `INFERS_DUMP_GDN_LAYER` + `INFERS_DUMP_GDN_DIR` |

**Python scripts** (all in `tests/`):

| Script | What it compares | Shares code with |
|--------|------------------|-------------------|
| `ref_intermediates.py` | MLP stages only | Nothing (own WeightLoader, dequant, io) |
| `ref_layer_compare.py` | Per-layer hidden state stats | Nothing |
| `gdn_layer_compare.py` | GDN stages | Own WeightLoader (duplicated) |
| `gdn_ref_intermediates.py` | GDN reference capture | Nothing |

**Problems:**

- No reference comparison for **full-attention sub-operations** (Q/K/V projections, Q-norm, K-norm, RoPE, attention scores, softmax, attention×V, gate, O-proj)
- No reference comparison for **final norm + LM head**
- No metadata with dumps — shapes, layer type, GPU index must be guessed
- Three different env var schemes (`INFERS_DEBUG_LAYER`, `INFERS_DEBUG_LAYER3`, `INFERS_DUMP_LAYER_DIR`, `INFERS_DUMP_GDN_LAYER`, `INFERS_DUMP_GDN_DIR`, `INFERS_DUMP_HIDDEN`)
- Adding a new dump point requires copy-pasting 6 lines of conditional boilerplate
- Python scripts duplicate WeightLoader, dequantize, io, cos_sim

## Architecture

### Part 1: Rust `probe` Module

New file: `crates/backends/native/src/probe.rs`

```rust
/// Named checkpoint in the forward pass.
/// Names follow a dot-separated convention: <layer_type>.<substage>
/// Examples: "attn.norm1", "attn.q_proj", "gdn.conv_out", "mlp.gate_proj", "final.logits"
```

**Environment variables** (replacing all existing ones):

| Variable | Purpose | Example |
|----------|---------|---------|
| `INFERS_DUMP_LAYERS` | Which layers to dump. `all` or comma-separated indices | `3`, `0,3,15`, `all` |
| `INFERS_DUMP_STAGES` | Stage prefix filter. Comma-separated. Default: all stages | `attn`, `attn,mlp`, `gdn` |
| `INFERS_DUMP_DIR` | Output directory (required for any dumping) | `/tmp/dump` |
| `INFERS_DUMP_STATS` | Also print min/max/mean_abs to stderr | `1` |

**These four replace**: `INFERS_DEBUG_LAYER`, `INFERS_DEBUG_LAYER3`, `INFERS_DUMP_LAYER_DIR`, `INFERS_DUMP_GDN_LAYER`, `INFERS_DUMP_GDN_DIR`, `INFERS_DUMP_HIDDEN`.

**Output structure:**

```
/tmp/dump/
  config.json                    # Model config + dump metadata
  layer_0/
    hidden_input_gpu0.raw        # bf16 binary
    hidden_input_gpu0.meta       # {"name":"hidden_input","layer":0,"gpu":0,"shape":[14,5120],"dtype":"bf16","stage":"embed"}
    hidden_input_gpu1.raw
    hidden_input_gpu1.meta
    gdn.norm1_gpu0.raw
    gdn.norm1_gpu0.meta
    gdn.mixed_qkv_gpu0.raw
    gdn.mixed_qkv_gpu0.meta
    ...
  layer_3/
    attn.norm1_gpu0.raw
    attn.norm1_gpu0.meta
    attn.q_proj_raw_gpu0.raw
    attn.q_proj_raw_gpu0.meta
    ...
  final/
    norm_output_gpu0.raw
    logits_gpu0.raw
    ...
```

**Metadata file** (`foo.meta`):

```json
{
  "name": "attn.q_proj_raw",
  "layer": 3,
  "gpu": 0,
  "shape": [14, 6144],
  "dtype": "bf16",
  "stage": "attn",
  "model_layer_type": "full_attention"
}
```

**ProbeConfig struct:**

```rust
pub struct ProbeConfig {
    pub enabled: bool,
    pub layers: Option<Vec<usize>>,    // None = all, Some(set) = filter
    pub stages: Option<Vec<String>>,    // None = all, Some(prefixes) = filter
    pub dir: Option<String>,            // None = disabled
    pub stats: bool,                    // print to stderr
}
```

Reads env vars once on construction. `should_dump(layer, stage)` does the filtering. `dump()` does the GPU→CPU transfer, writes `.raw` + `.meta`, optionally prints stats. When `enabled == false`, `dump()` returns immediately — zero cost.

**Stage naming convention:**

| Engine location | Stage name | Shape | Notes |
|-----------------|-----------|-------|-------|
| **Embedding** | | | |
| `embed_tokens` output | `embed.output` | [seq_len, hidden_size] | Same on all GPUs |
| **Attention (full_attention layers)** | | | |
| norm1 input | `attn.norm1_input` | [seq_len, hidden_size] | Pre-norm |
| norm1 output | `attn.norm1` | [seq_len, hidden_size] | |
| Q projection (raw, before split) | `attn.q_proj_raw` | [seq_len, q_out_dim] | Includes gate when attn_output_gate |
| Q after norm (Q-norm output) | `attn.q_norm` | [seq_len, per_gpu_head_dim] | |
| Gate (from Q-proj split) | `attn.gate` | [seq_len, per_gpu_head_dim] | Only when attn_output_gate |
| K projection (raw) | `attn.k_proj` | [seq_len, kv_dim] | |
| K after norm | `attn.k_norm` | [seq_len, kv_dim] | |
| V projection | `attn.v_proj` | [seq_len, kv_dim] | |
| Q after RoPE (head 0) | `attn.q_rope_h0` | [seq_len, head_dim] | |
| K after RoPE (head 0) | `attn.k_rope_h0` | [seq_len, head_dim] | |
| Attention scores (head 0) | `attn.scores_h0` | [seq_len, seq_len] | |
| Softmax output (head 0) | `attn.softmax_h0` | [seq_len, seq_len] | |
| Attention output combined | `attn.combined` | [seq_len, per_gpu_head_dim] | Before gate |
| Gated attention output | `attn.gated` | [seq_len, per_gpu_head_dim] | After sigmoid gate |
| O-projection (before AR) | `attn.o_proj` | [seq_len, hidden_size] | |
| After all-reduce | `attn.after_ar` | [seq_len, hidden_size] | |
| Residual add | `residual.attn` | [seq_len, hidden_size] | |
| **GDN (linear_attention layers)** | | | |
| norm1 input | `gdn.norm1_input` | [seq_len, hidden_size] | |
| norm1 output | `gdn.norm1` | [seq_len, hidden_size] | |
| mixed_qkv | `gdn.mixed_qkv` | [seq_len, conv_dim] | |
| conv1d output | `gdn.conv_out` | [seq_len, conv_dim] | |
| query / key / value | `gdn.query`, `gdn.key`, `gdn.value` | per-dim | |
| query_expanded / key_expanded | `gdn.query_expanded`, `gdn.key_expanded` | per-dim | |
| a_proj / b_proj | `gdn.a_proj`, `gdn.b_proj` | per-dim | |
| core_attn_out | `gdn.core_attn_out` | per-dim | |
| z_gate | `gdn.z_gate` | per-dim | |
| norm_output | `gdn.norm_output` | per-dim | |
| output (before AR) | `gdn.output` | [seq_len, hidden_size] | |
| after AR | `gdn.after_ar` | [seq_len, hidden_size] | |
| residual | `residual.attn` | [seq_len, hidden_size] | Same name as full_attention |
| **MLP (all layers)** | | | |
| norm2 input | `mlp.norm2_input` | [seq_len, hidden_size] | |
| norm2 output | `mlp.norm2` | [seq_len, hidden_size] | |
| gate_proj | `mlp.gate_proj` | [seq_len, sharded_intermediate] | |
| up_proj | `mlp.up_proj` | [seq_len, sharded_intermediate] | |
| SiLU(gate) * up | `mlp.silu` | [seq_len, sharded_intermediate] | |
| down_proj (before AR) | `mlp.down_raw` | [seq_len, hidden_size] | |
| down_proj (after AR) | `mlp.down_ar` | [seq_len, hidden_size] | |
| residual | `residual.mlp` | [seq_len, hidden_size] | |
| **Final** | | | |
| norm output | `final.norm` | [1, hidden_size] | Decode: [1, hidden_size] |
| logits | `final.logits` | [1, vocab_size] | |

**Head-level dumps** (optional, controlled by `INFERS_DUMP_STAGES=attn.heads`):

When enabled, dump per-head Q, K, V, scores, softmax for head 0 only. These are expensive (O(seq_len²) for scores) so they're opt-in.

### Part 2: Python `infers_compare` Framework

New directory: `tests/compare/`

```
tests/compare/
  __init__.py
  io.py                 # load_raw_bf16, save_raw_bf16, load_meta, discover_dumps
  dequant.py             # INT4 dequantization (AutoRound format)
  weight_loader.py        # TP-aware safetensors weight loader
  cos.py                 # cos_sim, l2_error, element_stats
  config.py              # DiscoveryConfig: reads dump's config.json
  compare.py              # CLI entry point
  stages/
    __init__.py
    base.py              # Stage base class (compute + compare)
    mlp.py                # MLP stages
    attention.py           # Full-attention stages (new!)
    gdn.py                 # GDN stages
    final.py               # Final norm + LM head (new!)
```

**Key design decisions:**

1. **Self-discovering** — `compare.py` reads `.meta` files from the dump directory to know what stages and shapes exist. No hardcoding.

2. **Two comparison modes** (from `gdn_layer_compare.py`):
   - **Full pipeline**: computes reference from `hidden_input` using dequantized weights. Catches INT4 dequant bugs.
   - **Kernel-only**: uses engine's own intermediates as inputs to each stage. Isolates individual kernel/GEMM bugs.

3. **Model config from dump** — Engine writes `config.json` to dump dir with all model params. Python reads this instead of re-deriving from the model's config.json.

4. **TP-aware by default** — `WeightLoader` takes `tp_size` and `gpu_idx`, shards weights identically to the engine.

5. **Per-stage thresholds** — Each stage module declares its own cosine threshold (INT4 GEMM: 0.99, BF16 kernel: 0.999, elementwise: 0.9999).

**`io.py`** — shared I/O (no duplication):

```python
def load_raw_bf16(path, shape) -> torch.Tensor: ...
def save_raw_bf16(path, tensor) -> None: ...
def load_meta(path) -> dict: ...
def discover_dumps(dump_dir) -> dict[int, list[dict]]: ...  # layer -> list of metas
```

**`weight_loader.py`** — single WeightLoader replacing 3 copies:

```python
class WeightLoader:
    def __init__(self, model_dir, tp_size=2): ...
    def load_dequant(self, name, gpu_idx=0) -> torch.Tensor: ...
    def load_bf16(self, name) -> torch.Tensor: ...
    # Convenience methods for common weights:
    def load_norm1(self, layer_idx) -> torch.Tensor: ...
    def load_norm2(self, layer_idx) -> torch.Tensor: ...
    def load_q_proj_dequant(self, layer_idx, gpu_idx=0) -> torch.Tensor: ...
    def load_k_proj_dequant(self, layer_idx, gpu_idx=0) -> torch.Tensor: ...
    # ... etc
```

**`stages/attention.py`** — the new critical piece:

```python
class AttentionStages:
    """Reference computation for full-attention layer sub-operations."""
    
    # Q/K/V projections (dequantized INT4)
    def compute_q_proj_raw(self, norm1_out, layer_idx, gpu_idx) -> torch.Tensor: ...
    def compute_q_norm(self, q_proj_raw, layer_idx) -> torch.Tensor: ...
    def compute_gate(self, q_proj_raw, layer_idx) -> torch.Tensor: ...
    def compute_k_proj(self, norm1_out, layer_idx, gpu_idx) -> torch.Tensor: ...
    def compute_k_norm(self, k_proj, layer_idx) -> torch.Tensor: ...
    def compute_v_proj(self, norm1_out, layer_idx, gpu_idx) -> torch.Tensor: ...
    
    # RoPE (requires position info)
    def compute_rope(self, q, k, positions, layer_idx) -> tuple: ...
    
    # Attention computation (full softmax)
    def compute_attention(self, q, k, v, num_heads, num_kv_heads, head_dim) -> torch.Tensor: ...
    
    # Gate application
    def compute_attn_gated(self, attn_combined, gate) -> torch.Tensor: ...
    
    # O-projection
    def compute_o_proj(self, gated_attn, layer_idx, gpu_idx) -> torch.Tensor: ...
    
    # All-reduce (sum across GPUs)
    def compute_all_reduce(self, *tensors) -> torch.Tensor: ...
```

**`stages/final.py`** — another new critical piece:

```python
class FinalStages:
    def compute_final_norm(self, hidden_states, layer_idx) -> torch.Tensor: ...
    def compute_logits(self, norm_out, layer_idx) -> torch.Tensor: ...
```

**`compare.py`** — CLI entry point:

```bash
# Dump a specific layer from the engine
INFERS_DUMP_LAYERS=3 INFERS_DUMP_DIR=/tmp/dump \
  cargo test --package infers-backend-native --test smoke_test smoke_test_real_model -- --ignored --nocapture

# Compare all dumped stages against reference
python -m tests.compare.compare --dump-dir /tmp/dump --model-dir /path/to/model

# Compare only attention stages for layer 3
python -m tests.compare.compare --dump-dir /tmp/dump --model-dir /path/to/model --stages attn

# Kernel-only comparison (uses engine intermediates as inputs)
python -m tests.compare.compare --dump-dir /tmp/dump --model-dir /path/to/model --kernel-only

# Verbose output with per-stage statistics
python -m tests.compare.compare --dump-dir /tmp/dump --model-dir /path/to/model -v
```

## Task Breakdown

### Phase 13.0: Remove legacy decode codepath

**Files**: `crates/backends/native/src/attention.rs`, `crates/backends/native/src/decode.rs`, `crates/backends/native/src/engine.rs`

| # | Task | Detail |
|---|------|--------|
| 13.0.1 | Remove `decode_forward()` from attention.rs | Delete the non-paged decode path (lines ~830–1200) |
| 13.0.2 | Remove `decode()` and `decode_with_hidden()` from engine.rs | Delete legacy decode methods that run on GPU 0 only |
| 13.0.3 | Remove unused `DecodeKernels` struct | Clean up kernel loading for the removed path |
| 13.0.4 | Verify `decode_paged` still compiles and passes smoke test | `cargo test --release -p infers-backend-native --test smoke_test -- --ignored` |

### Phase 13.1: Rust `probe` module

**Files**: new `crates/backends/native/src/probe.rs`, modified `crates/backends/native/src/lib.rs`

| # | Task | Detail |
|---|------|--------|
| 13.1.1 | Create `probe.rs` with `ProbeConfig`, `should_dump()`, `dump()` | As designed above. Includes: read 4 env vars, filter by layer+stage, write `.raw` + `.meta`, optional stats print |
| 13.1.2 | Add `dump_config()` — writes `config.json` to dump dir | Model config: hidden_size, num_heads, head_dim, num_kv_heads, intermediate_size, num_layers, layer_types, vocab_size, num_gpus, group_size, attn_output_gate, rms_norm_eps, rope_theta, partial_rotary_factor |
| 13.1.3 | Register `probe` module in `lib.rs` | `pub mod probe;` |
| 13.1.4 | Replace `debug_hidden_stats()` in engine.rs with probe stats | Delete inline fn, call `probe::stats()` when probe config says to |
| 13.1.5 | Replace `dump_bf16_tensor()` in engine.rs with probe::dump() | Delete inline fn |
| 13.1.6 | Replace `debug_hidden_stats()` in attention.rs with probe::stats() | Delete inline fn |
| 13.1.7 | Replace `dump_bf16_tensor()` in attention.rs with probe::dump() | Delete inline fn |
| 13.1.8 | Replace `debug_*()` and `dump_gdn_intermediate()` in gdn.rs with probe calls | Delete both inline fns |
| 13.1.9 | Remove all old env var handling | Delete references to `INFERS_DEBUG_LAYER`, `INFERS_DEBUG_LAYER3`, `INFERS_DUMP_LAYER_DIR`, `INFERS_DUMP_GDN_LAYER`, `INFERS_DUMP_GDN_DIR`, `INFERS_DUMP_HIDDEN` |

### Phase 13.2: Wire probe into attention forward_paged

**Files**: `crates/backends/native/src/attention.rs`

This is the biggest wiring task. Every intermediate that could be interesting gets a `probe::dump()` call.

| # | Task | Stage name | Detail |
|---|------|-----------|--------|
| 13.2.1 | Add probe param to `forward_paged` signature | — | `probe: &ProbeConfig` |
| 13.2.2 | Dump norm1 input | `attn.norm1_input` | Already available as `input` param |
| 13.2.3 | Dump K projection raw | `attn.k_proj` | After K GEMM, before K-norm |
| 13.2.4 | Dump K after norm | `attn.k_norm` | After K-norm |
| 13.2.5 | Dump V projection | `attn.v_proj` | After V GEMM |
| 13.2.6 | Dump Q projection raw | `attn.q_proj_raw` | After Q GEMM, before Q-norm/split |
| 13.2.7 | Dump Q after norm | `attn.q_norm` | After Q-norm (extracted Q portion) |
| 13.2.8 | Dump gate | `attn.gate` | After gate extraction from Q-proj |
| 13.2.9 | Dump Q after RoPE (head 0) | `attn.q_rope_h0` | Only when `stages` includes `attn.heads` |
| 13.2.10 | Dump K after RoPE (head 0) | `attn.k_rope_h0` | Only when `stages` includes `attn.heads` |
| 13.2.11 | Dump attention scores (head 0) | `attn.scores_h0` | Only when `stages` includes `attn.heads` |
| 13.2.12 | Dump softmax (head 0) | `attn.softmax_h0` | Only when `stages` includes `attn.heads` |
| 13.2.13 | Dump attention combined | `attn.combined` | All heads accumulated |
| 13.2.14 | Dump gated attention | `attn.gated` | After sigmoid gate |
| 13.2.15 | Dump O-projection | `attn.o_proj` | Before all-reduce |
| 13.2.16 | Remove all old debug_attn and `INFERS_DEBUG_LAYER` blocks | — | Delete the scattered conditional eprintln+dump blocks |

### Phase 13.3: Wire probe into attention decode_forward_paged

**Files**: `crates/backends/native/src/attention.rs`

| # | Task | Stage name | Detail |
|---|------|-----------|--------|
| 13.3.1 | Add probe param to `decode_forward_paged` signature | — | `probe: &ProbeConfig` |
| 13.3.2 | Mirror all forward_paged dump points | Same names | Same stages, seq_len=1 shapes |
| 13.3.3 | Remove old debug blocks | — | Delete scattered debug code |

### Phase 13.4: Wire probe into engine layer loop

**Files**: `crates/backends/native/src/engine.rs`

| # | Task | Detail |
|---|------|--------|
| 13.4.1 | Create `ProbeConfig` at start of `prefill_paged()` | `ProbeConfig::from_env()` |
| 13.4.2 | Write dump config.json | `probe::dump_config(&probe, &config)?` |
| 13.4.3 | Dump embedding output | `embed.output` |
| 13.4.4 | Dump norm1 | `attn.norm1` or `gdn.norm1` per layer type |
| 13.4.5 | Dump attention/GDN output (before AR) | `attn.o_proj` / `gdn.output` |
| 13.4.6 | Dump after all-reduce | `attn.after_ar` / `gdn.after_ar` |
| 13.4.7 | Dump residual (attn) | `residual.attn` |
| 13.4.8 | Dump norm2 | `mlp.norm2` |
| 13.4.9 | Dump MLP gate/up/silu/down | `mlp.gate_proj`, `mlp.up_proj`, `mlp.silu`, `mlp.down_raw` |
| 13.4.10 | Dump MLP after all-reduce | `mlp.down_ar` |
| 13.4.11 | Dump residual (MLP) | `residual.mlp` |
| 13.4.12 | Dump final norm output | `final.norm` |
| 13.4.13 | Dump logits | `final.logits` |
| 13.4.14 | Pass probe reference to attention::forward_paged | Thread `&probe` through |
| 13.4.15 | Pass probe reference to gdn::forward | Thread `&probe` through |
| 13.4.16 | Mirror all dumps in `decode_paged()` | Same stages, seq_len=1 |
| 13.4.17 | Remove all old `if let Ok(ref dl) ...` blocks | Delete ~200 lines of boilerplate |

### Phase 13.5: Wire probe into GDN forward/decode

**Files**: `crates/backends/native/src/gdn.rs`

| # | Task | Stage name | Detail |
|---|------|-----------|--------|
| 13.5.1 | Add probe param to `forward()` and `decode_forward()` | — | `probe: &ProbeConfig` |
| 13.5.2 | Replace all `dump_gdn_intermediate()` calls with `probe::dump()` | Per existing names | Same stages, uses dot naming: `gdn.mixed_qkv`, `gdn.conv_out`, etc. |
| 13.5.3 | Remove `dump_gdn_intermediate()` function | — | |
| 13.5.4 | Remove `debug_*()` stats functions | — | Replaced by `probe::stats()` |

### Phase 13.6: Python `infers_compare` framework

**Files**: new `tests/compare/` directory

| # | Task | Detail |
|---|------|--------|
| 13.6.1 | Create `tests/compare/__init__.py` | Package init |
| 13.6.2 | Create `tests/compare/io.py` | `load_raw_bf16()`, `save_raw_bf16()`, `load_meta()`, `discover_dumps()`, `write_meta()`. Port from `ref_intermediates.py` lines 72–130. |
| 13.6.3 | Create `tests/compare/dequant.py` | `dequantize_int4_autogptq()`, `unpack_int4()`. Port from `ref_intermediates.py` lines 136–207. |
| 13.6.4 | Create `tests/compare/weight_loader.py` | `WeightLoader` class with `tp_size`, `gpu_idx`, `load_dequant()`, `load_bf16()`, convenience methods per projection. Port from `ref_intermediates.py` lines 214–355 and `gdn_layer_compare.py`. Single unified version. |
| 13.6.5 | Create `tests/compare/cos.py` | `cos_sim()`, `l2_error()`, `element_stats()`. Port from `ref_intermediates.py` lines 101–130. |
| 13.6.6 | Create `tests/compare/config.py` | `DumpConfig` dataclass: reads `config.json` from dump dir, provides `hidden_size`, `num_heads`, `head_dim`, `num_kv_heads`, etc. |
| 13.6.7 | Create `tests/compare/stages/base.py` | `Stage` ABC: `name`, `threshold`, `compute(engine_inputs, weights) -> torch.Tensor`, `compare(engine_dir, ref_results) -> dict` |
| 13.6.8 | Create `tests/compare/stages/mlp.py` | Port from `ref_intermediates.py` MlpReference. Stages: norm1, norm2, gate_proj, up_proj, silu, down_raw, down_ar, residual. |
| 13.6.9 | Create `tests/compare/stages/gdn.py` | Port from `gdn_layer_compare.py`. Stages: mixed_qkv, conv_out, query, key, value, a_proj, b_proj, core_attn_out, z_gate, norm_output, output. |
| 13.6.10 | Create `tests/compare/stages/attention.py` | **NEW**. Stages: norm1, q_proj_raw, q_norm, gate, k_proj, k_norm, v_proj, scores_h0, softmax_h0, combined, gated, o_proj, after_ar, residual. |
| 13.6.11 | Create `tests/compare/stages/final.py` | **NEW**. Stages: final_norm, logits. |
| 13.6.12 | Create `tests/compare/compare.py` | CLI entry point. Discovers dumps, runs matching stage comparisons, prints pass/fail with cosine similarity. `--kernel-only` flag. `--stages` filter. `--verbose` per-stage stats. |

### Phase 13.7: Use instrumentation to find current bug

This is the payoff. Run the instrumented engine on layer 3 (first full-attention layer), compare against reference.

| # | Task | Detail |
|---|------|--------|
| 13.7.1 | Dump layer 3 intermediates | `INFERS_DUMP_LAYERS=3 INFERS_DUMP_DIR=/tmp/dump cargo test ...` |
| 13.7.2 | Run attention reference comparison | `python -m tests.compare.compare --dump-dir /tmp/dump --model-dir ... --stages attn` |
| 13.7.3 | Identify first divergent stage | First stage with cos < 0.99 is the bug location |
| 13.7.4 | Fix the bug | Depends on what 13.7.3 reveals |
| 13.7.5 | Dump layer 0 GDN intermediates | `INFERS_DUMP_LAYERS=0 INFERS_DUMP_DIR=/tmp/dump cargo test ...` |
| 13.7.6 | Run GDN reference comparison | Verify GDN stages still pass |
| 13.7.7 | Dump final layer + logits | `INFERS_DUMP_LAYERS=63 INFERS_DUMP_STAGES=final INFERS_DUMP_DIR=/tmp/dump cargo test ...` |
| 13.7.8 | Run final norm + LM head comparison | `python -m tests.compare.compare --dump-dir /tmp/dump --stages final` |
| 13.7.9 | If bug is in attention, dump multiple layers | `INFERS_DUMP_LAYERS=0,3,7,15,31,63 INFERS_DUMP_DIR=/tmp/dump` |
| 13.7.10 | Re-run comparison across layers | Check for error accumulation pattern |

### Phase 13.8: Clean up old scripts

| # | Task | Detail |
|---|------|--------|
| 13.8.1 | Delete `tests/ref_intermediates.py` | Replaced by `tests/compare/` |
| 13.8.2 | Delete `tests/ref_layer_compare.py` | Replaced by `tests/compare/` |
| 13.8.3 | Delete `tests/gdn_layer_compare.py` | Replaced by `tests/compare/` |
| 13.8.4 | Delete `tests/gdn_ref_intermediates.py` | Replaced by `tests/compare/` |
| 13.8.5 | Delete `tests/gdn_compare.py` | Replaced by `tests/compare/` |
| 13.8.6 | Update any references in lat.md | Replace old script references with new compare framework |

## Key Design Constraints

1. **Zero cost when disabled** — `ProbeConfig::from_env()` reads env vars once. When `enabled == false`, every `probe::dump()` returns in <1ns (no GPU transfer). No conditional compilation needed.

2. **No mutex or runtime alloc in hot path** — `ProbeConfig` is constructed once and passed by reference. `dump()` allocates a `Vec<u8>` on the heap only when active.

3. **Self-describing dumps** — `.meta` sidecar files mean the Python framework never needs to hardcode shapes. It discovers what's available and compares against matching reference stages.

4. **Model-agnostic stage names** — `attn.*`, `gdn.*`, `mlp.*`, `residual.*`, `final.*`, `embed.*` work for any transformer model. Adding a new architecture means adding a stage module, not changing the core framework.

5. **Head-level dumps are opt-in** — `INFERS_DUMP_STAGES=attn.heads` enables per-head Q/K/V/scores/softmax dumps. These are O(seq_len²) and very slow for long sequences. Off by default.

6. **Decode dumps share names with prefill** — The same stage names (`attn.q_proj_raw`, `attn.gated`, etc.) are used for both prefill and decode. Shapes differ (seq_len vs 1) but the Python framework reads shape from `.meta`.

7. **Backward compatibility** — The old env vars are removed. The new system is strictly better (4 env vars vs 6, consistent naming vs ad-hoc, metadata vs magic). No migration path needed since the old scripts are deleted in 13.8.

## Testing Strategy

- **Unit**: `ProbeConfig::from_env()` parsing, `should_dump()` filtering, `.meta` JSON roundtrip.
- **Integration**: Run the engine with `INFERS_DUMP_LAYERS=0,3 INFERS_DUMP_DIR=/tmp/test_dump`, verify that `layer_0/` and `layer_3/` directories contain `.raw` + `.meta` files with correct shapes.
- **Python**: `tests/compare/compare.py --dump-dir /tmp/test_dump --model-dir ...` completes with specific cosine thresholds per stage.
- **Smoke**: Full pipeline: dump → compare → identify divergence stage → fix → verify cos > 0.99 for all stages.

## Success Criteria

- [ ] `probe::dump()` compiles and has zero runtime cost when `INFERS_DUMP_DIR` is unset
- [ ] Running with `INFERS_DUMP_LAYERS=3 INFERS_DUMP_DIR=/tmp/dump` produces `layer_3/` with 20+ `.raw` + `.meta` files for a full-attention layer
- [ ] Running with `INFERS_DUMP_LAYERS=0 INFERS_DUMP_DIR=/tmp/dump` produces `layer_0/` with GDN-stage dumps
- [ ] `python -m tests.compare.compare --dump-dir /tmp/dump --model-dir ... --stages attn` reports cosine similarity per attention sub-stage
- [ ] **The current garbage output bug is found and fixed** — first divergent stage identified, root cause confirmed, fix applied, cos > 0.99 for all stages
- [ ] Old debug scripts deleted, old env vars removed from Rust code
- [ ] `lat.md` updated with probe naming convention and Python framework documentation