# Phase 26: PrismaSCOUT NVFP4 Model Support

---
**Status**: DONE
**Last Updated**: 2026-06-22
**Blocks**: Loading Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm model
**Blocked by**: Nothing
**Rationale**: The NVFP4 PrismaSCOUT model at `~/opt/vllm/models/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` cannot be loaded because the weight loader only handles BF16 and INT4 (AutoRound) tensor naming patterns. NVFP4 uses a 4-tensor companion structure (`weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale`) that must be parsed from the `quantization_config` metadata embedded in `config.json`.
---

## Goal

Enable loading the PrismaSCOUT NVFP4 model by:

1. Parsing the `quantization_config.config_groups` targets/ignore regex lists from `config.json` into a `QuantTargetMap` that tells us definitively which tensors are NVFP4 vs BF16 passthrough
2. Adding `Nvfp4Companions` and a unified `QuantCompanions` enum to replace the INT4-only `Int4Companions` + `int4_companions` HashMap
3. Making the weight loader metadata-driven: use `QuantTargetMap` to decide which naming pattern to look up (BF16 `.weight`, INT4 `.qweight`, NVFP4 `.weight_packed`), instead of blind fallback chains
4. Propagating NVFP4 companion support through the sharding (heap + mmap) paths and server wiring

## Target Model

`~/opt/vllm/models/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`

### NVFP4 Tensor Structure

Each NVFP4-quantized projection (e.g., `layers.0.linear_attn.in_proj_qkv`) produces 4 tensors:
- `{base}.weight_packed` — packed NVFP4 weight data
- `{base}.weight_scale` — per-block scale factor (FP8 E4M3)
- `{base}.weight_global_scale` — global scale for the tensor (BF16)
- `{base}.input_global_scale` — input activation global scale (BF16)

BF16 passthrough tensors still use the single `{base}.weight` pattern.

### Mixed Quantization Within Layers

The model is mixed-precision per-projection. Example for layer 3 (full attention):
- `self_attn.q_proj.weight` — BF16 (not in NVFP4 targets)
- `self_attn.o_proj.weight` — BF16 (not in NVFP4 targets)
- `mlp.gate_proj.weight_packed` + `.weight_scale` + `.weight_global_scale` + `.input_global_scale` — NVFP4

Layer 27 (also full attention):
- `self_attn.q_proj.weight_packed` + companions — NVFP4 (in targets)
- `self_attn.o_proj.weight_packed` + companions — NVFP4 (in targets)
- `mlp.*.weight_packed` + companions — NVFP4

GDN layers (e.g., layer 0):
- `linear_attn.in_proj_a.weight` — BF16 (always passthrough for GDN SSM weights)
- `linear_attn.in_proj_b.weight` — BF16 (always passthrough for GDN SSM weights)
- `linear_attn.in_proj_qkv.weight_packed` + companions — NVFP4
- `linear_attn.in_proj_z.weight_packed` + companions — NVFP4
- `linear_attn.out_proj.weight_packed` + companions — NVFP4
- `mlp.*.weight_packed` + companions — NVFP4

### Metadata Source

The `config.json` contains a `quantization_config` with:
- `config_groups.group_0.targets` — regex list of tensor base names that ARE NVFP4 quantized (e.g., `"re:^language_model[.]model[.]layers[.]0[.]linear_attn[.]in_proj_qkv$"`)
- `ignore` — regex list of tensor base names that are intentionally BF16 passthrough

After `strip_language_model_prefix`, the targets match against stripped names (e.g., `layers.0.linear_attn.in_proj_qkv`).

## Architecture

### QuantTargetMap

```rust
/// Resolved per-tensor quantization assignment from config metadata.
pub struct QuantTargetMap {
    /// Compiled regex patterns for NVFP4-quantized tensor base names.
    nvfp4_targets: Vec<regex::Regex>,
    /// Compiled regex patterns for INT4-quantized tensor base names (future-proof).
    int4_targets: Vec<regex::Regex>,
    /// Compiled regex patterns for BF16-passthrough (ignored) tensor base names.
    ignore: Vec<regex::Regex>,
}

impl QuantTargetMap {
    /// Build from a QuantizationConfig's config_groups and ignore list.
    pub fn from_config(config: &QuantizationConfig) -> anyhow::Result<Self>;

    /// Returns the quantization format for a tensor base name, or None if BF16 passthrough.
    pub fn resolve(&self, tensor_base: &str) -> Option<QuantizationFormat>;
}
```

### Nvfp4Companions + QuantCompanions

```rust
/// Companion tensors for an NVFP4 quantized weight.
pub struct Nvfp4Companions {
    pub weight_scale: WeightData,
    pub weight_global_scale: WeightData,
    pub input_global_scale: WeightData,
}

/// Unified companion tensors for any quantized weight format.
pub enum QuantCompanions {
    Int4(Int4Companions),
    Nvfp4(Nvfp4Companions),
}
```

`WeightRegistry.int4_companions` → `WeightRegistry.quant_companions: HashMap<String, QuantCompanions>`.

### Metadata-Driven Weight Loading

Replace `get_weight_or_int4(registry, bf16_name, int4_base)` with:

```rust
fn get_weight_with_quant(
    registry: &mut WeightRegistry,
    bf16_name: &str,       // e.g., "layers.0.mlp.gate_proj.weight"
    quant_base: &str,      // e.g., "layers.0.mlp.gate_proj"
    quant_map: &QuantTargetMap,
) -> Result<WeightData>
```

Logic:
1. `quant_map.resolve(quant_base)` → determines format
2. If `PrismaScout` → extract `{quant_base}.weight_packed` + 3 companions → store in `quant_companions`
3. If `AutoRound` → extract `{quant_base}.qweight` + `qzeros` + `scales` → store in `quant_companions`
4. If `None` (BF16 passthrough) → extract `{bf16_name}` directly

### Mmap Path Mirror

Same changes mirrored in mmap.rs:
- `MmapNvfp4Companions` with `weight_scale`, `weight_global_scale`, `input_global_scale` (all `MmapTensor`)
- `MmapQuantCompanions` enum: `Int4(MmapCompanions) | Nvfp4(MmapNvfp4Companions)`
- `MmapWeightRegistry.quant_companions: HashMap<String, MmapQuantCompanions>` (replaces `int4_companions`)
- `build_metadata_registry` copies NVFP4 companion metadata

## Implementation Phases

### Phase 1: Data Structures (weights.rs)

**Scope**: Add `Nvfp4Companions`, `QuantCompanions` enum, replace `int4_companions` with `quant_companions` in `WeightRegistry`.

**Files**: `crates/model/src/weights.rs`

**Changes**:
- Add `Nvfp4Companions` struct
- Add `QuantCompanions` enum with `Int4(Int4Companions)` and `Nvfp4(Nvfp4Companions)` variants
- Replace `int4_companions: HashMap<String, Int4Companions>` with `quant_companions: HashMap<String, QuantCompanions>` in `WeightRegistry`
- Update `WeightRegistry::new()`, `clear_data()`, and `Default` impl
- Add `clear_data()` methods on `Nvfp4Companions` and `QuantCompanions`
- Update existing tests that reference `int4_companions`

**Acceptance Criteria**:
- All existing tests pass
- `QuantCompanions::Int4` variant preserves existing INT4 behavior
- `Nvfp4Companions` has all 3 companion fields
- `cargo check -p infers-model` succeeds

**Complexity**: S
**Timebox**: 30 minutes

---

### Phase 2: QuantTargetMap (formats.rs)

**Scope**: Parse `quantization_config.config_groups` targets/ignore into a `QuantTargetMap` with compiled regexes.

**Files**: `crates/model/src/formats.rs`

**Changes**:
- Add `regex` to `infers-model` Cargo.toml dependencies
- Add `QuantTargetMap` struct with `nvfp4_targets`, `int4_targets`, `ignore` fields
- Add `QuantTargetMap::from_config(config: &QuantizationConfig)` that parses `config_groups.group_0.targets` and `ignore` regex patterns
- Add `QuantTargetMap::resolve(&self, tensor_base: &str) -> Option<QuantizationFormat>` method
- Handle the `format` field in `config_groups.group_0` to determine NVFP4 vs INT4 (currently only `"nvfp4-pack-quantized"` exists)
- Add unit tests with real PrismaSCOUT config patterns

**Acceptance Criteria**:
- `QuantTargetMap::from_config` correctly parses the PrismaSCOUT `config.json` quantization_config
- `resolve("layers.0.linear_attn.in_proj_qkv")` returns `Some(PrismaScout)` for a GDN layer target
- `resolve("layers.0.linear_attn.in_proj_a")` returns `None` for a BF16 passthrough weight (in ignore list)
- `resolve("layers.3.self_attn.q_proj")` returns `None` for BF16 (not in targets, not in ignore)
- `cargo test -p infers-model --release` passes

**Complexity**: S
**Timebox**: 45 minutes

---

### Phase 3: Metadata-Driven Loader (loader.rs)

**Scope**: Replace `get_weight_or_int4` with `get_weight_with_quant` that uses `QuantTargetMap`. Update `build_main_layers` and `build_mtp_weights` signatures.

**Files**: `crates/model/src/loader.rs`, `crates/model/src/lib.rs`

**Changes**:
- Add `get_weight_with_quant(registry, bf16_name, quant_base, quant_map)` — tries NVFP4 or INT4 based on `quant_map.resolve(quant_base)`, falls back to BF16
- Add `get_weight_with_quant_optional` variant for optional projections
- Update `build_main_layer` to accept `&QuantTargetMap` and call `get_weight_with_quant`
- Update `build_mtp_layer` similarly
- Update `build_main_layers` and `build_mtp_weights` signatures to accept `&QuantTargetMap`
- Update `lib.rs` re-exports if signature changes
- Update all existing tests in `loader.rs` to pass a default/empty `QuantTargetMap`

**Acceptance Criteria**:
- All existing tests pass with `QuantTargetMap::empty()` (all BF16)
- `build_main_layer` for a GDN layer correctly extracts NVFP4 companions for `in_proj_qkv` when the target map marks it as PrismaScout
- `build_main_layer` for a full-attention layer with BF16 Q/K/V still works
- `cargo test -p infers-model --release` passes

**Complexity**: M
**Timebox**: 60 minutes

---

### Phase 4: Heap Loader Sharding (model-loader-heap/src/lib.rs)

**Scope**: Update `shard_weights_tp` to handle `QuantCompanions::Nvfp4` in the companion_skip logic and sharding.

**Files**: `crates/model-loader-heap/src/lib.rs`

**Changes**:
- Expand `companion_skip` to recognize `.weight_scale`, `.weight_global_scale`, `.input_global_scale` suffixes
- Handle `QuantCompanions::Nvfp4` in all sharding branches (ColumnParallel, RowParallel, Replicated, fused projections)
- NVFP4 packed weights shard identically to INT4 packed weights (same dim layout)
- NVFP4 companions (weight_scale, weight_global_scale, input_global_scale) shard with the same pattern as INT4 scales/qzeros
- Update `map_safetensor_dtype` — add any new dtype mapping needed for NVFP4 packed data

**Acceptance Criteria**:
- `shard_weights_tp` produces correct shards for a model with `QuantCompanions::Nvfp4`
- Companion skip prevents double-processing of NVFP4 companion tensors
- All existing heap loader tests pass
- `cargo test -p infers-model-loader-heap --release` passes

**Complexity**: M
**Timebox**: 60 minutes

---

### Phase 5: Mmap Path (mmap.rs)

**Scope**: Mirror Phase 1 + Phase 4 changes in the mmap path. Add `MmapNvfp4Companions`, `MmapQuantCompanions`, update sharding, update `build_metadata_registry`.

**Files**: `crates/model/src/mmap.rs`

**Changes**:
- Add `MmapNvfp4Companions` struct (weight_scale, weight_global_scale, input_global_scale — all `MmapTensor`)
- Add `MmapQuantCompanions` enum: `Int4(MmapCompanions) | Nvfp4(MmapNvfp4Companions)`
- Replace `int4_companions: HashMap<String, MmapCompanions>` with `quant_companions: HashMap<String, MmapQuantCompanions>` in `MmapWeightRegistry`
- Update `shard_weights_tp_mmap` to handle NVFP4 companion sharding (same pattern as INT4)
- Update `companion_skip` logic in mmap sharding to recognize NVFP4 suffixes
- Update `build_metadata_registry` to copy NVFP4 companion metadata into `WeightRegistry.quant_companions`
- Update `lib.rs` re-exports for new types

**Acceptance Criteria**:
- `build_metadata_registry` correctly copies NVFP4 companion metadata
- Mmap sharding handles NVFP4 companions correctly
- All existing mmap tests pass
- `cargo test -p infers-model --release` passes

**Complexity**: M
**Timebox**: 60 minutes

---

### Phase 6: Server Wiring (main.rs)

**Scope**: Build `QuantTargetMap` from loaded config's `quantization_config` and pass it through to `build_main_layers`/`build_mtp_weights`.

**Files**: `crates/server/src/main.rs`

**Changes**:
- After `ModelConfig::load`, build `QuantTargetMap` from `config.quantization_config`
- Pass `&QuantTargetMap` to `build_main_layers` and `build_mtp_weights` calls
- Handle the case where `quantization_config` is `None` (use `QuantTargetMap::empty()`)

**Acceptance Criteria**:
- Server compiles and starts with a valid model path
- `cargo build --release` succeeds
- No behavioral change for non-PrismaSCOUT models

**Complexity**: S
**Timebox**: 30 minutes

---

### Phase 7: Documentation (lat.md/)

**Scope**: Document the NVFP4 companion structure, metadata-driven quantization resolution, and loading flow.

**Files**: `lat.md/` (appropriate sections)

**Changes**:
- Add/update section on NVFP4 companion tensor structure
- Document `QuantTargetMap` and metadata-driven resolution
- Add `@lat:` refs in new code
- Run `lat check` to validate

**Acceptance Criteria**:
- `lat check` passes
- New code has `@lat:` references

**Complexity**: S
**Timebox**: 30 minutes

---

### Phase 8: Integration Verification

**Scope**: Run all tests, verify compilation, run `lat check`.

**Acceptance Criteria**:
- `cargo test --release` passes for all crates
- `cargo build --release` succeeds
- `lat check` passes
- No regressions in existing INT4 or BF16 model loading

**Complexity**: XS
**Timebox**: 15 minutes

## Cross-Phase Dependencies

```
Phase 1 (weights.rs) ──────────────────┐
                                        ├─→ Phase 3 (loader.rs) ─→ Phase 6 (main.rs)
Phase 2 (formats.rs) ─────────────────┘
                                        ├─→ Phase 4 (model-loader-heap)
Phase 1 ──────────────────────────────┘   Phase 5 (mmap.rs)

Phase 3 ─→ Phase 7 (documentation) ─→ Phase 8 (verification)
```

Phases 1 and 2 can run in parallel. Phase 3 depends on both. Phases 4 and 5 depend on Phase 1 and can run in parallel after it. Phase 6 depends on Phase 3.

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Regex crate bloat (binary size) | Low | Low | `regex` already in transitive deps via other crates. Verify. |
| QuantTargetMap regex compilation slow on huge target lists | Low | Medium | ~680 regex patterns is fine; compile once at load time |
| NVFP4 packed weight layout differs from INT4 | Medium | High | Need to verify shape/layout of `weight_packed` tensors before implementing sharding |
| Missing `regex` dependency | Low | Low | Add to `Cargo.toml`; it's likely already a transitive dep |
| `int4_companions` → `quant_companions` rename breaks downstream crates | Medium | Medium | Greenfield project — breaking changes permitted per AGENTS.md |

## References

- [[lat#Model Config and Format Detection#Quantization Format Detection]]
- [[lat#Weight Registry and Tensors]]
- [[013-quantization.md]] — existing quantization phase
- Model config: `~/opt/vllm/models/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm/config.json`
- Safetensors index: `~/opt/vllm/models/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm/model.safetensors.index.json`
