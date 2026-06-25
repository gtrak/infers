# Phase 046: Split engine.rs and attention.rs

---
**Status**: NOT STARTED
**Last Updated**: 2026-06-25
**Blocks**: None (cleanup, no functional change)
**Blocked by**: None
**Rationale**: engine.rs (1569 lines) and attention.rs (1629 lines) are too large for maintainable development. Both contain multiple logical sections that should be separate files. engine.rs has decode, prefill, sampling, and engine init. attention.rs has paged decode, flat-cache forward, paged forward, and helper functions.
---

## Goal

Split large files into logical modules without changing any functionality.

## engine.rs (1569 lines) → Split into:

| New file | Source lines | Contents |
|---|---|---|
| `engine.rs` | ~200 | Struct definition, new(), load_kernels(), public API |
| `decode.rs` | ~500 | decode_paged() and helpers |
| `prefill.rs` | ~300 | prefill_paged() and helpers |
| `sample.rs` | (already exists, ~400 lines) | Already separate |
| `norm.rs` | (already exists) | Already separate |
| `add.rs` | (already exists) | Already separate |

## attention.rs (1629 lines) → Split into:

| New file | Source lines | Contents |
|---|---|---|
| `attention.rs` | ~100 | Struct definition, type enums, public API |
| `paged_decode.rs` | ~700 | decode_forward_paged() and helpers |
| `flat_forward.rs` | ~400 | forward() (legacy flat-cache prefill) |
| `paged_forward.rs` | ~400 | forward_paged() (paged prefill) |

## oxide_bridge.rs (1674 lines) → Consider splitting:

| New file | Source lines | Contents |
|---|---|---|
| `oxide_bridge.rs` | ~200 | OxideKernels struct, new(), CudaSliceView |
| `int4_bridge.rs` | ~400 | All INT4 launch methods |
| `attention_bridge.rs` | ~200 | All attention launch methods |
| `gdn_bridge.rs` | ~200 | All GDN launch methods |
| `common_bridge.rs` | ~400 | Common, norm, activation, bf16 launch methods |

Or: Keep as one file since it's purely boilerplate (1:1 method-per-kernel wrappers) with no complex logic.

## Implementation Plan

1. Create new module files
2. Move functions to new files
3. Update `mod.rs` / `lib.rs` exports
4. Verify compilation: `cargo build --release`
5. Verify correctness: smoke test passes at 0.036s/step

No functional changes. Pure code movement.
