# Phase 14: Decode Pipeline — Sampling, Stopping, Autoregressive Loop

---
**Status**: IN PROGRESS
**Last Updated**: 2026-06-19
**Blocks**: Server usability (currently hardcoded greedy, no stop tokens)
**Blocked by**: Nothing
**Rationale**: The engine produces correct tokens (prefill cos=0.999, decode generates "Paris is..."), but the decode loop is incomplete: greedy-only sampling causes repetition loops, no EOS/stop-token detection, `prefill_paged()` discards its sampled token, and the API's temperature/top-p/top-k/penalty parameters are ignored. A proper decode pipeline is needed before the server can serve coherent responses.
---

## Current State

| Component | What exists | What's missing |
|-----------|-------------|----------------|
| CUDA kernels | `infers_argmax_f32/bf16` only | No softmax, top-k, top-p, RNG kernels |
| Engine (`decode_paged`) | Always does GPU argmax, returns `u32` | No `SamplingConfig` param, no logits download path |
| Engine (`prefill_paged`) | Computes sampled token but discards it | Returns only `usize` (pages used); first decode step re-embeds last prompt token |
| `sample.rs` | 4 `SamplingStrategy` variants + `SamplingConfig` defined | All 3 non-greedy strategies `bail!("not yet implemented")` |
| `scheduler/queue.rs` | Duplicate `SamplingStrategy`/`SamplingConfig` | No penalty/EOS/stop/seed fields; not shared with backend |
| Orchestrator | Uses non-paged `prefill()`; hardcoded `SamplingStrategy::Greedy` | No wiring of API sampling params; no stop-token detection |
| API (`ChatCompletionRequest`) | Has temperature, top_p, top_k, repetition/presence/frequency_penalty, stop, seed | None reach the engine |
| MTP | `sample` callback: `Fn(&CudaSlice<bf16>, &CudaStream>) -> Result<u32>` | Greedy only; no config param |

## Architecture Decision: CPU Sampling

**Choice: CPU-side sampling (download logits → CPU softmax + RNG)**

Why not GPU:
- 152K vocab → 300KB BF16 logits per decode step
- Download: ~0.3ms (negligible vs 155ms decode step)
- GPU sampling kernels are complex (RNG, top-k sorting, prefix-sum for top-p)
- CPU sampling is debuggable, correctable, and fast enough
- MTP compatibility: MTP's `sample` callback signature `Fn(&CudaSlice<bf16>, &Arc<CudaStream>) -> Result<u32>` works with either approach — the callback internally decides GPU argmax or CPU download+sample

MTP impact: None. MTP verification currently uses greedy argmax internally. With stochastic sampling, MTP verification would use rejection sampling — but that's a **verification-logic change** independent of sampling location. Production MTP typically uses greedy verification regardless; stochastic sampling only applies to the final output token.

Future GPU kernels: If sampling becomes a bottleneck (unlikely at <0.2% of step time), a GPU softmax+argmax kernel can be added later without architectural changes. The `sample` callback abstracts the choice.

## Plan (6 commits)

### Commit 1: Fix `prefill_paged` return type

**Problem**: `prefill_paged()` computes a sampled token internally but discards it, returning only `usize` (pages used). The `infer` binary must embed `token_ids.last()` for decode step 0, which re-runs a forward pass the prefill already did. The orchestrator works around this by using the non-paged `prefill()` (which returns `u32`).

**Changes**:

| File | Change |
|------|--------|
| `crates/backends/native/src/engine.rs` | `prefill_paged()` return type: `Result<usize>` → `Result<(usize, u32)>`. The sampled token is already computed at line ~773-802 — just return it instead of discarding. |
| `crates/backends/native/src/bin/infer.rs` | Use returned token as `current_token` for decode step 0 instead of `token_ids.last()`. Adjust for tuple return. |
| `crates/server/src/orchestrator.rs` | Switch prefill path to `prefill_paged()` (already used by scheduler, returns token now). Use returned token for session state. |
| `crates/backends/native/tests/smoke_test.rs` | Update `prefill_paged()` call to handle tuple return. |

**Acceptance**: `infer` binary generates "Paris is..." with decode step 0 using the prefill-sampled token. No functional change in output (the token was already correct), but eliminates the wasted re-embedding.

**Complexity**: S
**Timebox**: 30 min

---

### Commit 2: Unify `SamplingConfig` / `SamplingStrategy`

**Problem**: Duplicate types in `scheduler/queue.rs` and `backends/native/sample.rs`. Missing fields for penalties, EOS, stop tokens, seed. Backend crate can't reference scheduler types (no dependency).

**Canonical location**: `crates/scheduler/src/queue.rs` — already the single source for scheduling types. The server and orchestrator already import from here.

**Changes**:

| File | Change |
|------|--------|
| `crates/scheduler/src/queue.rs` | Extend `SamplingConfig` with: `repetition_penalty: f32` (default 1.0), `presence_penalty: f32` (default 0.0), `frequency_penalty: f32` (default 0.0), `eos_token_id: Option<u32>`, `stop_token_ids: Vec<u32>`, `seed: Option<u64>`. Add `Default` impl with sensible values. |
| `crates/backends/native/Cargo.toml` | Add `infers-scheduler = { path = "../../scheduler" }` dependency |
| `crates/backends/native/src/sample.rs` | Delete duplicate `SamplingStrategy` and `SamplingConfig` enums. Replace with `use infers_scheduler::{SamplingConfig, SamplingStrategy};`. Update all internal references. |
| `crates/backends/native/src/lib.rs` | Update `pub mod sample;` — ensure re-exports still work |
| `crates/backends/native/src/engine.rs` | Update any `use crate::sample::SamplingStrategy` references |

**Acceptance**: `cargo build --release` succeeds. No duplicate type definitions. `SamplingConfig` has all new fields with correct defaults. All existing tests pass.

**Complexity**: S
**Timebox**: 30 min

---

### Commit 3: CPU Sampling Implementation

**New functions in `crates/backends/native/src/sample.rs`** (all operate on `&mut [f32]` logits downloaded from GPU):

**RNG**: Xoshiro256++ (inline implementation, no external dependency). `SeedableRng` trait with `from_seed(u64)`. SplitMix64 for seed expansion. Per-session instance held by orchestrator.

**Pipeline functions**:

```rust
/// Apply repetition, presence, and frequency penalties to logits.
/// token_history contains all previous tokens (prompt + generated).
fn apply_penalties(logits: &mut [f32], token_history: &[u32], config: &SamplingConfig)

/// Divide all logits by temperature. Temperature <= 0.0 is clamped to minimal epsilon
/// to avoid division by zero (effectively greedy).
fn temperature_scale(logits: &mut [f32], temp: f32)

/// Zero out (set to -f32::INFINITY) all logits except the top-k.
/// Preserves relative order of surviving logits.
fn top_k_filter(logits: &mut [f32], k: usize)

/// Numerically stable in-place softmax: subtract max, exponentiate, normalize.
fn softmax(logits: &mut [f32])

/// Sample from the cumulative probability distribution using top-p (nucleus) filtering.
/// Sorts descending by probability, accumulates until cumulative > p, then weighted
/// random sample from the surviving tokens.
fn top_p_sample(probs: &[f32], p: f64, rng: &mut impl Rng) -> usize

/// Main dispatch: given raw BF16 logits on GPU, download to CPU, apply full sampling
/// pipeline based on SamplingConfig.strategy, return sampled token ID.
/// 
/// Greedy path (no penalties, no temperature scaling): stays on GPU via
/// greedy_sample_bf16() — no logits download.
/// 
/// Non-greedy: download logits → BF16→F32 → penalties → temperature → top_k →
/// softmax → top_p or weighted sample → token ID.
fn sample_with_config(
    stream: &Arc<CudaStream>,
    gpu_logits: &CudaView<'_, bf16>,
    argmax_kernel: &CudaFunction,
    config: &SamplingConfig,
    token_history: &[u32],
    rng: &mut impl Rng,
) -> Result<u32>

/// Check if a sampled token should stop generation.
/// Returns true if token matches eos_token_id or any stop_token_ids.
fn should_stop(token: u32, config: &SamplingConfig) -> bool
```

**Strategy dispatch table**:

| Strategy | Pipeline |
|----------|----------|
| `Greedy` | GPU argmax (existing fast path) if no penalties; otherwise: penalties → argmax on CPU |
| `Temperature { temp }` | penalties → temperature_scale → softmax → weighted sample |
| `TopK { k, temp }` | penalties → temperature_scale → top_k_filter → softmax → weighted sample |
| `TopP { p, temp }` | penalties → temperature_scale → softmax → top_p_sample |

**Unit tests** (pure CPU, no GPU needed):

| Test | What it verifies |
|------|------------------|
| `test_apply_penalties_repetition` | Token appearing 3x in history has logit reduced by `3 * (repetition_penalty - 1)` |
| `test_apply_penalties_presence` | Any token in history gets flat `presence_penalty` deduction |
| `test_apply_penalties_frequency` | Token frequency in history multiplies `frequency_penalty` deduction |
| `test_temperature_scale` | Logits divided by temp; temp<epsilon clamps to greedy |
| `test_top_k_filter` | Only top-k survive; rest are -inf |
| `test_softmax` | Sums to 1.0; numerically stable with large inputs |
| `test_top_p_sample_deterministic` | With seed, produces same token |
| `test_should_stop_eos` | EOS token triggers stop |
| `test_should_stop_stop_tokens` | Configured stop tokens trigger stop |
| `test_sample_greedy_on_gpu` | Greedy strategy calls GPU argmax, not CPU path |

**Acceptance**: All unit tests pass. `sample_with_config()` handles all 4 strategies. Greedy path stays on GPU. Non-greedy path downloads logits and runs CPU pipeline.

**Complexity**: M
**Timebox**: 2 hours

---

### Commit 4: Wire `SamplingConfig` Through Engine

**Problem**: `decode_paged()` and `prefill_paged()` always do GPU argmax. No way to pass sampling configuration.

**Changes**:

| File | Change |
|------|--------|
| `crates/backends/native/src/engine.rs` | Add params to `decode_paged()`: `sampling_config: &SamplingConfig`, `token_history: &[u32]`, `rng: &mut impl Rng`. Replace hardcoded `greedy_sample_bf16()` call with `sample::sample_with_config()`. Same for `prefill_paged()` final sample. |
| `crates/backends/native/src/engine.rs` | Add `use infers_scheduler::SamplingConfig;` import. Trait bound on `rng`: use a concrete `Xoshiro256PlusPlus` type defined in `sample.rs` to avoid generic propagation. |
| `crates/backends/native/src/lib.rs` | Re-export `sample::Xoshiro256PlusPlus` if needed by orchestrator |

**Dispatch logic** (inside `sample_with_config`):
```
if matches!(strategy, Greedy) && config.repetition_penalty == 1.0
    && config.presence_penalty == 0.0 && config.frequency_penalty == 0.0 {
    // GPU argmax — no logits download, existing fast path
    greedy_sample_bf16(stream, argmax_kernel, logits)
} else {
    // Download logits → CPU → full pipeline
    download_logits_bf16_to_f32(stream, logits) → apply_penalties → 
    temperature → top_k → softmax → sample
}
```

**MTP compatibility**: The `MtpOperations.sample` callback currently has signature `Fn(&CudaSlice<bf16>, &Arc<CudaStream>) -> Result<u32>`. With `sample_with_config`, the orchestrator constructs this closure capturing the session's `SamplingConfig` and `Rng`. No signature change needed to `MtpOperations` — the closure hides the complexity.

**Acceptance**: `decode_paged()` accepts and uses `SamplingConfig`. Greedy path still produces same output as before. Temperature=0.7 + top_p=0.9 produces different (non-repetitive) output vs greedy. `cargo test --release` passes. `cargo run --release --bin infer` still works.

**Complexity**: M
**Timebox**: 1.5 hours

---

### Commit 5: Wire API → SamplingConfig + Stop Token Detection

**Problem**: `ChatCompletionRequest` has temperature/top_p/top_k/penalties/stop/seed but they're all ignored. No stop-token detection in the decode loop.

**Changes**:

| File | Change |
|------|--------|
| `crates/server/src/handlers/chat.rs` | Build `SamplingConfig` from `ChatCompletionRequest`: map temperature/top_p/top_k to `SamplingStrategy::TopP`/`TopK`/`Temperature`, set penalty fields, tokenize stop sequences to `stop_token_ids`, set `eos_token_id` from tokenizer config, set `seed`, set `max_tokens`. |
| `crates/server/src/orchestrator.rs` | Create per-session RNG (seeded from `config.seed` or `OsRng`). Pass `SamplingConfig` + `token_history` to `prefill_paged()` / `decode_paged()`. After each decode: check `should_stop(sampled, config)` → mark session complete + close response channel. |
| `crates/scheduler/src/session.rs` | Add `sampling_config: SamplingConfig` field to `Session`. Add `rng_seed: u64` field (actual RNG state lives in orchestrator). |

**RNG state management**: Per-session `Xoshiro256PlusPlus` held in `HashMap<SequenceId, Xoshiro256PlusPlus>` in the orchestrator. Created on session admission (seed from config or random). Passed by `&mut` reference to engine calls. Dropped on session completion.

**Stop token detection**:
```rust
// After decode step:
if sample::should_stop(sampled_token, &session.sampling_config) {
    lifecycle::complete_session(session);
    self.response_tx.remove(&seq_id);
    // Also remove RNG state
    self.session_rngs.remove(&seq_id);
}
```

**Acceptance**: `curl` to server with `temperature=0.7, top_p=0.9` produces stochastic output. `stop=["\n"]` terminates generation at newline. EOS token terminates generation. `max_tokens` limit works. `seed=42` produces deterministic output across runs. Default (no params) still uses greedy.

**Complexity**: M
**Timebox**: 1.5 hours

---

### Commit 6: Update `infer` Binary

**Problem**: `infer` binary has no CLI flags for sampling parameters. Always uses greedy. No EOS stop. Has debug top-5 logit dump.

**Changes**:

| File | Change |
|------|--------|
| `crates/backends/native/src/bin/infer.rs` | Add CLI flags: `--temperature <f32>`, `--top-k <usize>`, `--top-p <f64>`, `--repetition-penalty <f32>`, `--presence-penalty <f32>`, `--frequency-penalty <f32>`, `--seed <u64>`. Build `SamplingConfig` from CLI args. Create RNG from seed. Pass through to engine calls. Break decode loop on EOS. Remove DEBUG top-5 logit dump from prefill output. |

**New CLI args**:

```
--temperature <f32>      Temperature for sampling (default: 1.0, 0.0 = greedy)
--top-k <usize>         Top-k sampling (default: disabled)
--top-p <f64>           Top-p (nucleus) sampling (default: 1.0)
--repetition-penalty <f32>  Repetition penalty (default: 1.0)
--presence-penalty <f32>    Presence penalty (default: 0.0)
--frequency-penalty <f32>   Frequency penalty (default: 0.0)
--seed <u64>            RNG seed for deterministic sampling
```

**Acceptance**: `--temperature 0.7` produces non-repetitive output. `--repetition-penalty 1.2` reduces loops. `--seed 42` is deterministic. Default (no flags) still greedy. EOS terminates generation.

**Complexity**: S
**Timebox**: 45 min

---

## Crate Dependency Graph (after changes)

```
infers-api (ChatCompletionRequest: temperature, top_p, etc.)
    ↓
infers-server (chat.rs: API → SamplingConfig; orchestrator: session RNG, stop detection)
    ↓ depends on
infers-scheduler (SamplingConfig, SamplingStrategy, Session)  ← CANONICAL types
    ↓ depends on
infers-kv (SequenceId)

infers-backend-native (sample_with_config, Xoshiro256PlusPlus, engine dispatch)
    ↓ depends on
infers-scheduler (imports SamplingConfig, SamplingStrategy)
infers-cuda (CudaSlice, CudaStream, kernels)
infers-model (ModelConfig)

infers-mtp (MtpOperations.sample callback — no changes needed)
    ↓ depends on
infers-cuda, infers-model
```

**New dependency**: `infers-backend-native → infers-scheduler` (was not present before).

## MTP Forward Compatibility

The `MtpOperations.sample` callback is `Fn(&CudaSlice<bf16>, &Arc<CudaStream>) -> Result<u32>`. The orchestrator constructs this closure, which can internally:
1. Call `greedy_sample_bf16()` for MTP draft generation (greedy is correct for drafting)
2. Call `sample_with_config()` for the main model's final output with user's sampling params

MTP **verification** always uses greedy argmax internally (compare `main_token == draft_token`). This is standard practice — stochastic sampling is for the output, not for verification. If we later want probabilistic verification (rejection sampling), that's a change to `verify_drafts()` logic, not to the sampling callback.

## Testing Strategy

| Test Type | What | Where |
|-----------|------|-------|
| Unit tests | `apply_penalties`, `temperature_scale`, `top_k_filter`, `softmax`, `top_p_sample`, `should_stop` | `crates/backends/native/src/sample.rs` |
| Integration | `infer` binary with `--temperature 0.7` produces non-repetitive output | Manual (`cargo run --release --bin infer`) |
| Integration | `infer` binary with `--seed 42` is deterministic across runs | Manual |
| Integration | Server `curl` with sampling params produces stochastic output | Manual (after server wiring) |
| Regression | Greedy output unchanged from current (same "Paris is..." output) | Manual |

## Success Criteria

- [x] `prefill_paged()` returns the first sampled token
- [ ] `SamplingConfig` unified in scheduler crate with all fields
- [ ] All 4 sampling strategies implemented and tested
- [ ] Repetition/presence/frequency penalties working
- [ ] Stop on EOS token
- [ ] Stop on configured stop sequences
- [ ] Deterministic output with seed
- [ ] API params wired through to engine
- [ ] `infer` binary has CLI sampling flags
- [ ] No regression in greedy output quality
- [ ] MTP `sample` callback compatible with new sampling

## Files Modified (summary)

| File | Action | Commit |
|------|--------|-------|
| `crates/backends/native/src/engine.rs` | Modify: return type, sampling params | 1, 4 |
| `crates/backends/native/src/bin/infer.rs` | Modify: CLI flags, sampling usage | 1, 6 |
| `crates/server/src/orchestrator.rs` | Modify: paged prefill, session RNG, stop detection | 1, 5 |
| `crates/backends/native/tests/smoke_test.rs` | Modify: tuple return | 1 |
| `crates/scheduler/src/queue.rs` | Modify: extend SamplingConfig | 2 |
| `crates/backends/native/Cargo.toml` | Modify: add infers-scheduler dep | 2 |
| `crates/backends/native/src/sample.rs` | Rewrite: delete duplicates, add CPU sampling, RNG | 2, 3 |
| `crates/backends/native/src/lib.rs` | Modify: re-exports | 2 |
| `crates/server/src/handlers/chat.rs` | Modify: build SamplingConfig from API | 5 |
| `crates/scheduler/src/session.rs` | Modify: add sampling_config field | 5 |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU softmax overflow for large logits | Wrong token | Numerically stable softmax (subtract max first) |
| Top-p sorting O(V log V) for 152K vocab | Slow (~1ms) | Acceptable at 155ms/step; use `select_nth_unstable` for O(V) partial sort |
| Penalty application double-counts prompt tokens | Over-penalized output | `token_history` includes prompt; spec says "only generated tokens" for some APIs. Use `prompt_token_count` offset to skip prompt tokens for frequency/presence penalties. |
| RNG state lost on session eviction | Non-deterministic after restore | Store RNG seed in Session; re-seed on restore. Low priority — eviction is rare. |
| `infers-backend-native → infers-scheduler` circular dep | Won't compile | No: scheduler only depends on `infers-kv`. Backend depends on scheduler. One-directional. |
