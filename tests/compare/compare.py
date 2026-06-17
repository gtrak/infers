#!/usr/bin/env python3
"""Compare engine dumps against PyTorch reference.

Usage:
    # Dump engine intermediates
    INFERS_DUMP_LAYERS=3 INFERS_DUMP_DIR=/tmp/dump \
        cargo test --package infers-backend-native --test smoke_test -- --ignored --nocapture

    # Compare against reference (full attention + MLP stages)
    python -m tests.compare.compare --dump-dir /tmp/dump/layer_3 --model-dir /path/to/model

    # Compare only attention stages
    python -m tests.compare.compare --dump-dir /tmp/dump/layer_3 --model-dir /path/to/model --stages attn

    # Verbose per-stage stats
    python -m tests.compare.compare --dump-dir /tmp/dump/layer_3 --model-dir /path/to/model -v
"""

import argparse
import os
import sys

# Allow running as a script from the tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path
from typing import Dict, List, Optional

import torch

from tests.compare.config import DumpConfig
from tests.compare.weight_loader import WeightLoader
from tests.compare.stages.base import Stage


# =========================================================================
# Stage registries — organized by category
# =========================================================================

def _get_attention_stages() -> List[Stage]:
    """Return the sequence of attention reference stages."""
    from tests.compare.stages.attention import (
        AfterArStage,
        GatedStage,
        KNormStage,
        KProjStage,
        Norm1InputStage,
        Norm1Stage,
        OProjStage,
        QNormStage,
        QProjRawStage,
        VProjStage,
        AttentionCombinedStage,
        GateStage,
    )
    return [
        Norm1InputStage(),
        Norm1Stage(),
        QProjRawStage(),
        QNormStage(),
        GateStage(),
        KProjStage(),
        KNormStage(),
        VProjStage(),
        AttentionCombinedStage(),
        GatedStage(),
        OProjStage(),
        AfterArStage(),
    ]


def _get_mlp_stages() -> List[Stage]:
    """Return the sequence of MLP reference stages."""
    from tests.compare.stages.mlp import (
        DownArStage,
        DownRawStage,
        GateProjStage,
        Norm2Stage,
        ResidualAttnStage,
        ResidualMlpStage,
        SiluStage,
        UpProjStage,
    )
    return [
        Norm2Stage(),
        GateProjStage(),
        UpProjStage(),
        SiluStage(),
        DownRawStage(),
        DownArStage(),
        ResidualAttnStage(),
        ResidualMlpStage(),
    ]


def _get_gdn_stages() -> List[Stage]:
    """Return the sequence of GDN reference stages."""
    from tests.compare.stages.gdn import (
        GdnCoreAttnOutStage,
        GdnConvOutStage,
        GdnMixedQkvStage,
        GdnNormOutputStage,
        GdnOutputStage,
    )
    return [
        GdnMixedQkvStage(),
        GdnConvOutStage(),
        GdnCoreAttnOutStage(),
        GdnNormOutputStage(),
        GdnOutputStage(),
    ]


def _get_final_stages() -> List[Stage]:
    """Return the final norm + LM head stages."""
    from tests.compare.stages.final import FinalNormStage, LogitsStage
    return [
        FinalNormStage(),
        LogitsStage(),
    ]

# Mapping from category name to stage getter function
_STAGE_CATEGORIES = {
    "attn": _get_attention_stages,
    "mlp": _get_mlp_stages,
    "gdn": _get_gdn_stages,
    "final": _get_final_stages,
}

# Mapping from engine dump file stem (with dot prefix) to reference stage name
_DUMP_TO_STAGE = {
    # Attention stages
    "attn.norm1_input": "hidden_input",
    "attn.norm1": "norm1",
    "attn.q_proj_raw": "q_proj_raw",
    "attn.q_norm": "q_norm",
    "attn.gate": "gate",
    "attn.k_proj": "k_proj",
    "attn.k_norm": "k_norm",
    "attn.v_proj": "v_proj",
    "attn.combined": "attn.combined",
    "attn.gated": "attn.gated",
    "attn.o_proj": "attn.o_proj",
    # MLP stages
    "mlp.norm1": "norm1",
    "mlp.norm2": "norm2",
    "mlp.gate_proj": "gate_proj_raw",
    "mlp.up_proj": "up_proj_raw",
    "mlp.silu": "silu",
    "mlp.down_raw": "down_raw",
    "mlp.down_ar": "down_ar",
    # Residual stages
    "residual.attn": "residual_attn",
    "residual.mlp": "residual_mlp",
    # Final stages
    "final_norm": "final_norm",
    "logits": "logits",
}


# =========================================================================
# Layer-type-aware stage resolution
# =========================================================================

def _resolve_stages_for_layer(
    layer_idx: int,
    config: DumpConfig,
    categories: Optional[List[str]] = None,
) -> List[Stage]:
    """Resolve the list of stages for a given layer based on its type.

    Full-attention layers have attention + MLP stages.
    GDN layers have GDN stages only (no self-attn + mlp split).
    Final layer has final_norm + logits.
    """
    if categories is None:
        categories = list(_STAGE_CATEGORIES.keys())

    layer_type = config.get_layer_type(layer_idx)

    # Filter out incompatible stages based on layer type
    filtered_categories = []
    for cat in categories:
        if cat == "gdn" and layer_type != "gdn":
            continue  # GDN stages only for GDN layers
        if cat == "attn" and layer_type == "gdn":
            continue  # Attention stages only for full-attention layers
        filtered_categories.append(cat)

    stages = []
    for cat in filtered_categories:
        getter = _STAGE_CATEGORIES.get(cat)
        if getter:
            stages.extend(getter())
    return stages


# =========================================================================
# Core comparison logic
# =========================================================================

def _load_hidden_input(dump_dir: str, hidden_size: int) -> Optional[torch.Tensor]:
    """Load hidden_input from the dump directory.

    Tries multiple possible filenames in order:
      1. hidden_input.raw (legacy naming)
      2. attn.norm1_input_gpu0.raw (attention layer naming)
      3. gdn.norm1_input_gpu0.raw (GDN layer naming)

    Determines seq_len from file size. Returns None if not found.
    """
    from tests.compare import io

    candidates = [
        "hidden_input.raw",
        "attn.norm1_input_gpu0.raw",
        "gdn.norm1_input_gpu0.raw",
    ]
    raw_path: Optional[Path] = None
    for name in candidates:
        candidate = Path(dump_dir) / name
        if candidate.exists():
            raw_path = candidate
            break

    if raw_path is None:
        return None

    n_bf16 = os.path.getsize(raw_path) // 2
    seq_len = n_bf16 // hidden_size
    return io.load_raw_bf16(str(raw_path), (seq_len, hidden_size))


def _compute_and_compare_layer(
    layer_idx: int,
    dump_dir: str,
    weights: WeightLoader,
    config: DumpConfig,
    stages: List[Stage],
    verbose: bool = False,
) -> Optional[Dict[str, dict]]:
    """Compute reference outputs and compare against engine dumps for a single layer.

    Stages are computed in order, with each stage's output stored in the `inputs`
    dict under its name (with GPU suffix if applicable). Subsequent stages can
    reference previous outputs by name.

    Args:
        layer_idx: Target layer index.
        dump_dir: Path to engine dump directory for this layer.
        weights: WeightLoader instance.
        config: DumpConfig with model parameters.
        stages: Ordered list of Stage instances.
        verbose: If True, include extra stats in results dict.

    Returns:
        Dict mapping stage_name → comparison result dict, or None on failure.
    """
    # Load hidden input
    hidden_input = _load_hidden_input(dump_dir, config.hidden_size)
    if hidden_input is None:
        return None

    tp_size = config.num_gpus
    results: Dict[str, dict] = {}
    inputs: Dict[str, torch.Tensor] = {"hidden_input": hidden_input}

    for stage in stages:
        # Try each GPU if the stage supports TP sharding
        computed_any = False
        for gpu_idx in range(tp_size):
            try:
                ref_tensor = stage.compute(inputs, weights, config, layer_idx, gpu_idx)

                # Store with GPU suffix so subsequent stages can reference it
                key = f"{stage.name}_gpu{gpu_idx}" if gpu_idx > 0 else stage.name
                inputs[key] = ref_tensor

                # Compare against engine dump
                cmp_result = stage.compare(dump_dir, ref_tensor, layer_idx, gpu_idx)
                result_key = f"{stage.name}_gpu{gpu_idx}" if gpu_idx > 0 else stage.name
                results[result_key] = cmp_result
                computed_any = True

            except KeyError as e:
                # Missing dependency — skip this GPU for this stage
                if not verbose:
                    continue
                print(f"  [SKIP] {stage.name}_gpu{gpu_idx}: missing dependency {e}")
            except ValueError as e:
                # Stage doesn't support this GPU (e.g. DownArStage on GPU 1)
                if not verbose:
                    continue
                print(f"  [SKIP] {stage.name}_gpu{gpu_idx}: {e}")
            except NotImplementedError:
                # Stub stage — skip
                continue

        if not computed_any and not verbose:
            pass  # Stage skipped silently (e.g. GDN stubs)

    return results


def _print_results(results: dict, verbose: bool = False):
    """Print comparison results grouped by pass/fail."""
    passed = {k: v for k, v in results.items() if v.get("passed")}
    failed = {k: v for k, v in results.items() if not v.get("passed")}

    if passed:
        print(f"\n[PASS] {len(passed)} stages:")
        for name, r in sorted(passed.items()):
            extra = ""
            if verbose:
                extra = f"  l2_err={r['l2_err']:.6f}  max_diff={r['max_diff']:.6f}"
            print(f"  {name:35s}  cos={r['cos']:.6f}{extra}")

    if failed:
        print(f"\n[FAIL] {len(failed)} stages:")
        for name, r in sorted(failed.items()):
            if "error" in r:
                print(f"  {name:35s}  ERROR: {r['error']}")
            else:
                extra = ""
                if verbose:
                    extra = f"  l2_err={r['l2_err']:.6f}  max_diff={r['max_diff']:.6f}"
                print(f"  {name:35s}  cos={r['cos']:.6f}{extra}")

    return len(failed) == 0


# =========================================================================
# CLI
# =========================================================================

DEFAULT_MODEL_PATH = "/home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4"

def main():
    parser = argparse.ArgumentParser(
        description="Compare engine dumps against PyTorch reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dump-dir", type=str, required=True,
        help="Engine dump directory for a single layer (e.g. /tmp/dump/layer_3)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Model weights directory with safetensors [default: {DEFAULT_MODEL_PATH}]",
    )
    parser.add_argument(
        "--stages", type=str, default=None, nargs="*",
        help="Filter to specific stage categories: attn, mlp, gdn, final. "
             "If omitted, runs all applicable stages for the layer type.",
    )
    parser.add_argument(
        "--layer-idx", type=int, default=None,
        help="Override the inferred layer index (useful when dump-dir is not named layer_N)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output with per-stage L2 and max_diff statistics",
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    if not dump_dir.exists():
        print(f"[ERROR] Dump directory does not exist: {dump_dir}")
        sys.exit(1)

    # Infer layer index from directory name or config
    try:
        layer_idx = int(dump_dir.name.split("_", 1)[1])
    except (ValueError, IndexError):
        if args.layer_idx is not None:
            layer_idx = args.layer_idx
        else:
            print(f"[ERROR] Cannot infer layer index from '{dump_dir.name}' — use --layer-idx")
            sys.exit(1)

    # Load dump config (from root dump dir's config.json)
    # The dump_dir may be a subdirectory like /tmp/dump/layer_3, so look for config.json
    # in the parent directory first, then fall back to the dump_dir itself.
    config_path = dump_dir / "config.json"
    if not config_path.exists():
        config_path = dump_dir.parent / "config.json"
    if not config_path.exists():
        print(f"[ERROR] No config.json found in {dump_dir} or {dump_dir.parent}")
        sys.exit(1)

    # Find the root dump directory (where config.json lives)
    root_dump_dir = str(config_path.parent)

    print("=" * 70)
    print(f"Compare — Layer {layer_idx}, Model: {args.model_dir}")
    print(f"  Dump dir: {dump_dir}")
    print(f"  Config:   {config_path}")
    print("=" * 70)

    # Load config and weights
    try:
        config = DumpConfig.from_dir(root_dump_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load dump config: {e}")
        sys.exit(1)

    try:
        weights = WeightLoader(args.model_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n  hidden_size:          {config.hidden_size}")
    print(f"  num_attention_heads:   {config.num_attention_heads}")
    print(f"  num_key_value_heads:   {config.num_key_value_heads}")
    print(f"  head_dim:              {config.head_dim}")
    print(f"  intermediate_size:     {config.intermediate_size}")
    print(f"  layer_type:            {config.get_layer_type(layer_idx)}")
    print(f"  num_gpus:              {config.num_gpus}")

    # Resolve stages for this layer type
    categories = args.stages if args.stages else None
    stages = _resolve_stages_for_layer(layer_idx, config, categories)

    if not stages:
        print(f"\n[WARN] No stages available for layer {layer_idx} with categories {categories}")
        sys.exit(0)

    print(f"\n  Running {len(stages)} stages...")

    # Compute and compare
    results = _compute_and_compare_layer(layer_idx, str(dump_dir), weights, config, stages, args.verbose)

    if results is None:
        print("[ERROR] missing hidden_input.raw")
        sys.exit(1)

    all_passed = _print_results(results, verbose=args.verbose)

    print(f"\n{'=' * 70}")
    if all_passed:
        print("ALL STAGES PASSED")
    else:
        print("SOME STAGES FAILED — check which suboperation shows divergence")
    print(f"{'=' * 70}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
