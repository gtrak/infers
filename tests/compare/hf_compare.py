#!/usr/bin/env python3
"""Compare HF oracle dumps against engine dumps.

Loads raw tensors from the HF oracle (saved as .pt files) and engine dumps
(saved as .raw bf16 files), computes cosine similarity, L2 error, and max
absolute difference, and prints a per-layer comparison table.

Usage:
    # First generate oracle dumps
    python -m tests.compare.oracle --model-dir /path/to/model --token-ids 1,2,3 --output-dir /tmp/oracle

    # Then compare against engine dumps
    python -m tests.compare.hf_compare --oracle-dir /tmp/oracle --engine-dir /tmp/dump --phase prefill
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Allow running as a script from the tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.compare.io import load_raw_bf16, load_meta
from tests.compare.cos import cos_sim, l2_error, element_stats
from collections import defaultdict


# --- TP-aware reconstruction types for GDN tensors ---
GPU0_ONLY = "gpu0_only"                # Post-all-gather: use GPU0 directly
HEAD_SHARD_CAT = "head_shard_cat"      # Cat along head dim across GPUs
SEGMENT_AWARE = "segment_aware"        # Fused QKV with interleaved segments
ROW_PARALLEL_SUM = "row_parallel_sum"  # Sum partial sums across GPUs

# GDN tensor registry: stripped_name → (oracle_name, reconstruction_type)
# Each entry maps the engine's stripped tensor name to the oracle tensor
# filename and the TP-aware reconstruction strategy needed.
GDN_RECONSTRUCT = {
    # GPU0_ONLY — post all-gather, both GPUs have identical full tensors
    "norm1_input": ("norm1_input", GPU0_ONLY),
    "norm1": ("norm1_output", GPU0_ONLY),

    # HEAD_SHARD_CAT — each GPU has half the heads; cat along head dim
    "query": ("query", HEAD_SHARD_CAT),
    "key": ("key", HEAD_SHARD_CAT),
    "value": ("value", HEAD_SHARD_CAT),
    "query_expanded": ("query_expanded", HEAD_SHARD_CAT),
    "key_expanded": ("key_expanded", HEAD_SHARD_CAT),
    "a_proj": ("a_proj", HEAD_SHARD_CAT),
    "b_proj": ("b_proj", HEAD_SHARD_CAT),
    "core_attn_out": ("core_attn_out", HEAD_SHARD_CAT),
    "z_gate": ("z_gate", HEAD_SHARD_CAT),
    "norm_output": ("norm_output", HEAD_SHARD_CAT),

    # SEGMENT_AWARE — fused QKV with [Q_part, K_part, V_part] per GPU
    "mixed_qkv": ("mixed_qkv", SEGMENT_AWARE),
    "conv_out": ("conv_out", SEGMENT_AWARE),

    # ROW_PARALLEL_SUM — sum partial sums across GPUs
    "o_proj": ("o_proj", ROW_PARALLEL_SUM),

    # GPU0_ONLY — post all-reduce, both GPUs identical
    "after_ar": ("output", GPU0_ONLY),

    # Residual tensors (post-all-gather, identical on both GPUs)
    "residual.attn": ("norm2_input", GPU0_ONLY),
}

# GDN tensors to skip comparison (no full oracle baseline available)
GDN_SKIP = {"output"}  # row-parallel partial sum before all-reduce

# Tensor name mapping: stripped engine tensor name → oracle tensor name
# Engine dumps use names like "gdn.norm1_gpu0.raw" or "attn.attn_output_gpu0.raw"
# We strip the prefix (gdn., attn., mlp.) and GPU suffix (_gpu0) to get the key,
# then look up the corresponding oracle tensor name.
ENGINE_TO_ORACLE = {
    # GDN layer tensors (oracle name mapping only — reconstruction via GDN_RECONSTRUCT):
    "norm1_input": "norm1_input",
    "norm1": "norm1_output",
    "mixed_qkv": "mixed_qkv",
    "conv_out": "conv_out",
    "query": "query",
    "key": "key",
    "value": "value",
    "query_expanded": "query_expanded",
    "key_expanded": "key_expanded",
    "a_proj": "a_proj",
    "b_proj": "b_proj",
    "core_attn_out": "core_attn_out",
    "z_gate": "z_gate",
    "norm_output": "norm_output",
    "o_proj": "o_proj",
    # "output" is skipped — row-parallel partial sum before all-reduce
    # Full attention layer tensors:
    "q_proj_raw": "q_proj_raw",
    "q_norm": "q_norm",
    "gate": "gate",
    "k_proj": "k_proj",
    "k_norm": "k_norm",
    "v_proj": "v_proj",
    "k_before_rope": "k_before_rope",
    "k_after_rope": "k_after_rope",
    "k_cached": "k_cached",
    "v_cached": "v_cached",
    "combined": "combined",
    "gated": "gated",
    "after_ar": "output",
    # Residual tensors (post-all-gather):
    "residual.attn": "norm2_input",
    # MLP layer tensors:
    "norm2": "norm2_output",
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "silu": "silu",
    "down_raw": "down_raw",
    "down_ar": "down_ar",
}
# Legacy tensor names that the oracle may save but have no direct engine dump
# Legacy tensor names that the oracle may save but have no direct engine dump
# (e.g. older oracle versions without GDN intermediates).
# These are checked against _ORACLE_NAMES to find oracle-only tensors.
# GDN-relevant names are excluded since they're covered by GDN_RECONSTRUCT.
_ORACLE_NAMES = [
    "attn_output",      # full_attention only — missing for GDN layers (expected)
    "norm2_input",      # pre-attention norm for full_attention layers
    "mlp_output",       # full_attention MLP output
    "layer_output",     # full_attention layer output
]


def _strip_engine_prefix(stem: str) -> Tuple[Optional[str], Optional[int]]:
    """Strip engine dump prefix and GPU suffix to get (stripped_name, gpu_idx).

    Returns None for stripped_name if the tensor is unknown.
    Examples:
        \"attn.norm1_input_gpu0\" → (\"norm1_input\", 0)
        \"gdn.mixed_qkv_gpu1\"   → (\"mixed_qkv\", 1)
    """
    # Remove GPU suffix and extract index
    gpu_idx = None
    for i in range(8):
        suffix = f"_gpu{i}"
        if stem.endswith(suffix):
            gpu_idx = i
            stem = stem[: -len(suffix)]
            break

    # Remove known prefixes (NOT "residual." — residual.attn must keep the prefix)
    for prefix in ["attn.", "gdn.", "mlp.", "final_"]:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break

    return stem, gpu_idx


def _get_oracle_info(stripped_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Look up oracle tensor name and reconstruction type for a stripped engine name.

    Returns (oracle_name, recon_type) or (None, None) if not found.
    """
    if stripped_name in GDN_RECONSTRUCT:
        oracle_name, recon_type = GDN_RECONSTRUCT[stripped_name]
        return oracle_name, recon_type

    oracle_name = ENGINE_TO_ORACLE.get(stripped_name)
    if oracle_name is not None:
        return oracle_name, GPU0_ONLY  # default for non-GDN

    return None, None


def _discover_engine_dumps(
    engine_dir: Path, layer_idx: int, phase: str
) -> Dict[str, List[Tuple[int, Path, Tuple]]]:
    """Find all .raw dumps for a given layer and phase.

    Returns dict mapping stripped_name → [(gpu_idx, raw_path, shape), ...].
    The shape is read from the .meta sidecar file; if unavailable, it's None.
    """
    layer_dir = engine_dir / f"layer_{layer_idx}" / phase
    if not layer_dir.exists():
        return {}

    dumps: Dict[str, List[Tuple[int, Path, Tuple]]] = defaultdict(list)
    for raw_path in sorted(layer_dir.glob("*.raw")):
        stem = raw_path.stem  # e.g., "gdn.mixed_qkv_gpu0"
        stripped_name, gpu_idx = _strip_engine_prefix(stem)

        if stripped_name is None or gpu_idx is None:
            continue

        # Check if this tensor has a known oracle mapping
        oracle_name, _ = _get_oracle_info(stripped_name)
        if oracle_name is None:
            continue  # unknown tensor, skip

        # Skip GDN tensors that should not be compared
        if stripped_name in GDN_SKIP:
            continue

        # Read per-GPU shape from .meta sidecar
        meta_path = raw_path.with_suffix(".meta")
        shape = None
        if meta_path.exists():
            try:
                meta = load_meta(str(meta_path))
                shape = tuple(meta["shape"])
            except Exception:
                pass  # meta file unavailable or malformed, use None

        dumps[stripped_name].append((gpu_idx, raw_path, shape))

    return dumps


def _load_oracle_tensor(
    oracle_dir: Path, layer_idx: int, phase: str, tensor_name: str
) -> Optional[torch.Tensor]:
    """Load an oracle .pt tensor."""
    pt_path = oracle_dir / f"layer_{layer_idx}" / phase / f"{tensor_name}.pt"
    if not pt_path.exists():
        return None
    return torch.load(pt_path, weights_only=True)


def _reconstruct_engine_tensor(
    gpu_tensors: List[Tuple[int, torch.Tensor]],
    recon_type: str,
    dump_config=None,
) -> Optional[torch.Tensor]:
    """Reconstruct a full tensor from per-GPU engine dumps.

    Args:
        gpu_tensors: [(gpu_idx, tensor), ...] — must be sorted by gpu_idx
        recon_type: one of the GPU0_ONLY / HEAD_SHARD_CAT / SEGMENT_AWARE / ROW_PARALLEL_SUM constants
        dump_config: DumpConfig instance (required for SEGMENT_AWARE)
    """
    if not gpu_tensors:
        return None

    if recon_type == GPU0_ONLY:
        # Post-all-gather: both GPUs have identical full tensors; use GPU0
        return sorted(gpu_tensors, key=lambda x: x[0])[0][1].float()

    elif recon_type == HEAD_SHARD_CAT:
        # Each GPU has half the heads; cat along head dimension (dim 1)
        sorted_t = sorted(gpu_tensors, key=lambda x: x[0])
        return torch.cat([t for _, t in sorted_t], dim=1).float()

    elif recon_type == ROW_PARALLEL_SUM:
        # Sum partial sums across GPUs
        return sum(t.float() for _, t in gpu_tensors)

    elif recon_type == SEGMENT_AWARE:
        if dump_config is None:
            print("  [WARN] SEGMENT_AWARE reconstruction requires DumpConfig")
            return None

        num_gpus = dump_config.num_gpus
        per_gpu_t = gpu_tensors[0][1]
        conv_dim_per_gpu = per_gpu_t.size(1)

        # Derive segment boundaries from per-GPU shape:
        # Per-GPU layout: [Q_part(key_dim_per_gpu), K_part(key_dim_per_gpu), V_part(value_dim_per_gpu)]
        # We need key_dim_per_gpu. Compute it as conv_dim_per_gpu - value_dim_per_gpu,
        # where value_dim_per_gpu = conv_dim_full / num_gpus - 2 * key_dim_per_gpu
        # This is underdetermined, so we use the oracle query tensor shape to derive it.
        # However, if we can't get that, fall back: try common patterns.
        #
        # For GDN models: conv_dim_per_gpu = key_dim_full + value_dim_full/num_gpus
        # and key_dim_full is typically smaller than hidden_size.
        # We compute key_dim_per_gpu from the query oracle shape.
        return None  # Will be handled in compare_layers with oracle context

    print(f"  [WARN] Unknown reconstruction type: {recon_type}")
    return None


def _reshape_oracle_tensor(
    oracle_t: torch.Tensor, target_shape: Tuple[int, int]
) -> Optional[torch.Tensor]:
    """Reshape an oracle tensor to match the reconstructed engine shape.

    Tries transformations in this order:
    1. Already 2D and matches → return as-is
    2. Squeeze batch dim [1, seq, feat] → [seq, feat]
    3. Flatten heads [1, seq, h, hd] → [seq, h*hd]
    4. Reshape [A, B] → [seq, feat] if total elements match
    """
    # Already matches
    if oracle_t.dim() == 2 and oracle_t.shape == target_shape:
        return oracle_t.float()

    # Squeeze batch dim: [1, seq, feat] → [seq, feat]
    if oracle_t.dim() == 3 and oracle_t.size(0) == 1:
        squeezed = oracle_t.squeeze(0)
        if squeezed.shape == target_shape:
            return squeezed.float()

    # Flatten heads: [1, seq, h, hd] → [seq, h*hd]
    if oracle_t.dim() == 4 and oracle_t.size(0) == 1:
        t = oracle_t.squeeze(0)
        flat = t.flatten(1)
        if flat.shape == target_shape:
            return flat.float()

    # Reshape: [A, B] → [seq_len, feature_dim] if element count matches
    if oracle_t.dim() == 2 and oracle_t.numel() == target_shape[0] * target_shape[1]:
        try:
            return oracle_t.reshape(target_shape).float()
        except RuntimeError as e:
            print(f"  [WARN] reshape failed: {e}")

    return None


def _reconstruct_segment_aware(
    gpu_tensors: List[Tuple[int, torch.Tensor]],
    dump_config,
    oracle_dir: Path,
    layer_idx: int,
    phase: str,
) -> Optional[torch.Tensor]:
    """Reconstruct a fused QKV tensor from TP-sharded per-GPU dumps.

    Derives segment boundaries (Q/K/V split) from the oracle's query tensor shape.
    Per-GPU layout: [Q_part(key_dim_per_gpu), K_part(key_dim_per_gpu), V_part(value_dim_per_gpu)]
    Full layout:     [Q(key_dim_full), K(key_dim_full), V(value_dim_full)]
    """
    if dump_config is None:
        print("  [WARN] SEGMENT_AWARE reconstruction requires DumpConfig")
        return None

    num_gpus = dump_config.num_gpus
    per_gpu_t = gpu_tensors[0][1]
    seq_len = per_gpu_t.size(0)
    conv_dim_per_gpu = per_gpu_t.size(1)

    # Derive key_dim_full from oracle query shape [1, seq_len, num_k_heads, head_k_dim]
    query_oracle = _load_oracle_tensor(oracle_dir, layer_idx, phase, "query")
    if query_oracle is None or query_oracle.dim() != 4:
        print(f"  [WARN] Could not derive key_dim from query oracle (shape={getattr(query_oracle, 'shape', None)})")
        return None

    # key_dim_full = num_k_heads * head_k_dim
    key_dim_full = query_oracle.size(2) * query_oracle.size(3)
    key_dim_per_gpu = key_dim_full // num_gpus
    value_dim_per_gpu = conv_dim_per_gpu - 2 * key_dim_per_gpu
    value_dim_full = value_dim_per_gpu * num_gpus
    conv_dim_full = 2 * key_dim_full + value_dim_full

    if value_dim_per_gpu <= 0:
        print(f"  [WARN] Invalid segment dimensions: key={key_dim_per_gpu}, val={value_dim_per_gpu}")
        return None

    full = torch.zeros(seq_len, conv_dim_full)
    sorted_t = sorted(gpu_tensors, key=lambda x: x[0])
    for gpu_idx, t in sorted_t:
        # Q: interleave GPU shards along [0:key_dim_full]
        start_q = gpu_idx * key_dim_per_gpu
        full[:, start_q : start_q + key_dim_per_gpu] = t[:, 0:key_dim_per_gpu].float()

        # K: interleave GPU shards along [key_dim_full:2*key_dim_full]
        start_k = key_dim_full + gpu_idx * key_dim_per_gpu
        full[:, start_k : start_k + key_dim_per_gpu] = t[
            :, key_dim_per_gpu : 2 * key_dim_per_gpu
        ].float()

        # V: interleave GPU shards along [2*key_dim_full:conv_dim_full]
        start_v = 2 * key_dim_full + gpu_idx * value_dim_per_gpu
        full[:, start_v : start_v + value_dim_per_gpu] = t[
            :, 2 * key_dim_per_gpu : 2 * key_dim_per_gpu + value_dim_per_gpu
        ].float()

    return full.float()




def compare_layers(
    oracle_dir: Path,
    engine_dir: Path,
    phase: str,
    num_layers: int,
    threshold: float = 0.99,
) -> Tuple[bool, List[dict], Optional[int], Optional[str]]:
    """Compare all layers between oracle and engine dumps.

    Uses TP-aware reconstruction for GDN tensors before comparison.
    Returns:
        (all_passed, results, first_fail_layer, first_fail_tensor)
    """
    # Load dump config for model dimensions
    try:
        from tests.compare.config import DumpConfig
        dump_config = DumpConfig.from_dir(str(engine_dir))
    except Exception as e:
        print(f"  [WARN] Could not load DumpConfig: {e}")
        dump_config = None

    results = []
    first_fail_layer = None
    first_fail_tensor = None

    for i in range(num_layers):
        engine_dumps = _discover_engine_dumps(engine_dir, i, phase)

        # Discover oracle tensors: from engine-mapped names + legacy oracle names
        all_oracle_names = set(engine_dumps.keys()) | set(_ORACLE_NAMES)
        oracle_tensors = {}
        for stripped_name in sorted(all_oracle_names):
            oracle_name, _ = _get_oracle_info(stripped_name)
            actual_name = oracle_name or stripped_name  # fallback to stripped name
            t = _load_oracle_tensor(oracle_dir, i, phase, actual_name)
            if t is not None:
                oracle_tensors[stripped_name] = (actual_name, t)

        # Collect all tensor names to compare (union of engine and oracle)
        all_tensor_names = sorted(set(oracle_tensors.keys()) | set(engine_dumps.keys()))

        for stripped_name in all_tensor_names:
            # Skip if neither oracle nor engine has this tensor
            if stripped_name not in oracle_tensors and stripped_name not in engine_dumps:
                continue

            # Oracle-only: report as missing_engine_dump
            if stripped_name not in engine_dumps:
                results.append({
                    "layer": i,
                    "tensor": stripped_name,
                    "cos": 0.0,
                    "l2_err": 1.0,
                    "max_diff": -1.0,
                    "passed": False,
                    "error": "missing_engine_dump",
                })
                if first_fail_layer is None:
                    first_fail_layer = i
                    first_fail_tensor = stripped_name
                continue

            # Engine-only: oracle doesn't have this tensor (skip, no baseline to compare)
            if stripped_name not in oracle_tensors:
                continue

            oracle_name, oracle_t_raw = oracle_tensors[stripped_name]

            # Get reconstruction type
            _, recon_type = _get_oracle_info(stripped_name)
            if recon_type is None:
                recon_type = GPU0_ONLY  # default fallback

            # Load all GPU tensors for this engine dump
            gpu_tensor_list = []
            load_failed = False
            for gpu_idx, raw_path, gshape in engine_dumps[stripped_name]:
                try:
                    if gshape is not None:
                        t = load_raw_bf16(str(raw_path), gshape).float()
                    else:
                        # Fallback: load as flat and use oracle shape for reshaping
                        t = load_raw_bf16(str(raw_path), (-1,)).float()
                    gpu_tensor_list.append((gpu_idx, t))
                except Exception as e:
                    print(f"  [WARN] Failed to load {raw_path}: {e}")
                    load_failed = True
                    break

            if load_failed or not gpu_tensor_list:
                results.append({
                    "layer": i,
                    "tensor": stripped_name,
                    "cos": 0.0,
                    "l2_err": 1.0,
                    "max_diff": -1.0,
                    "passed": False,
                    "error": "load_failed",
                })
                if first_fail_layer is None:
                    first_fail_layer = i
                    first_fail_tensor = stripped_name
                continue

            # Reconstruct full engine tensor from GPU shards
            if recon_type == SEGMENT_AWARE:
                # SEGMENT_AWARE needs oracle query shape to derive segment boundaries
                engine_t = _reconstruct_segment_aware(
                    gpu_tensor_list, dump_config, oracle_dir, i, phase
                )
            else:
                engine_t = _reconstruct_engine_tensor(
                    gpu_tensor_list, recon_type, dump_config
                )
            if engine_t is None:
                results.append({
                    "layer": i,
                    "tensor": stripped_name,
                    "cos": 0.0,
                    "l2_err": 1.0,
                    "max_diff": -1.0,
                    "passed": False,
                    "error": "reconstruct_failed",
                })
                if first_fail_layer is None:
                    first_fail_layer = i
                    first_fail_tensor = stripped_name
                continue

            # Reshape oracle tensor to match reconstructed engine shape
            target_shape = (engine_t.size(0), engine_t.size(1))
            oracle_t = _reshape_oracle_tensor(oracle_t_raw, target_shape)

            if oracle_t is None:
                results.append({
                    "layer": i,
                    "tensor": stripped_name,
                    "cos": 0.0,
                    "l2_err": 1.0,
                    "max_diff": -1.0,
                    "passed": False,
                    "error": f"oracle_reshape_failed target={target_shape} oracle={oracle_t_raw.shape}",
                })
                if first_fail_layer is None:
                    first_fail_layer = i
                    first_fail_tensor = stripped_name
                continue

            if engine_t.numel() != oracle_t.numel():
                results.append({
                    "layer": i,
                    "tensor": stripped_name,
                    "cos": 0.0,
                    "l2_err": 1.0,
                    "max_diff": -1.0,
                    "passed": False,
                    "error": f"size_mismatch engine={engine_t.numel()} oracle={oracle_t.numel()}",
                })
                if first_fail_layer is None:
                    first_fail_layer = i
                    first_fail_tensor = stripped_name
                continue

            cos = cos_sim(engine_t, oracle_t)
            l2 = l2_error(engine_t, oracle_t)
            stats = element_stats(engine_t, oracle_t)
            passed = cos >= threshold

            results.append({
                "layer": i,
                "tensor": stripped_name,
                "cos": cos,
                "l2_err": l2,
                "max_diff": stats["max"],
                "passed": passed,
            })

            if not passed and first_fail_layer is None:
                first_fail_layer = i
                first_fail_tensor = stripped_name

    all_passed = all(r.get("passed", False) for r in results)
    return all_passed, results, first_fail_layer, first_fail_tensor


def print_results(results: List[dict], first_fail_layer, first_fail_tensor):
    """Print comparison results grouped by layer."""
    layers = defaultdict(list)
    for r in results:
        layers[r["layer"]].append(r)

    print(f"\n{'Layer':>5}  {'Tensor':<18}  {'cos':>10}  {'l2_err':>10}  {'max_diff':>10}  {'Status':>6}")
    print("-" * 75)

    for layer in sorted(layers):
        print(f"\n  === Layer {layer} ===")
        for r in layers[layer]:
            tensor = r["tensor"]
            cos = r.get("cos", 0.0)
            l2 = r.get("l2_err", 0.0)
            mx = r.get("max_diff", 0.0)
            error = r.get("error")
            passed = r.get("passed", False)

            if error:
                status = "ERR"
                print(f"{layer:5d}  {tensor:<18}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {status:>6}  ({error})")
            else:
                status = "OK" if passed else "FAIL"
                print(f"{layer:5d}  {tensor:<18}  {cos:10.6f}  {l2:10.3f}  {mx:10.3f}  {status:>6}")

    if first_fail_layer is not None:
        print(f"\nFirst divergence: layer {first_fail_layer}, tensor '{first_fail_tensor}'")


def main():
    parser = argparse.ArgumentParser(
        description="Compare HF oracle dumps against engine dumps",
    )
    parser.add_argument(
        "--oracle-dir", type=str, required=True,
        help="Directory with HF oracle dumps (from oracle.py)",
    )
    parser.add_argument(
        "--engine-dir", type=str, required=True,
        help="Directory with engine dumps",
    )
    parser.add_argument(
        "--phase", type=str, default="prefill", choices=["prefill", "decode"],
        help="Phase to compare [default: prefill]",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.99,
        help="Cosine similarity threshold for PASS/FAIL [default: 0.99]",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    oracle_path = Path(args.oracle_dir)
    engine_path = Path(args.engine_dir)

    if not oracle_path.exists():
        print(f"[ERROR] Oracle directory does not exist: {oracle_path}")
        sys.exit(1)
    if not engine_path.exists():
        print(f"[ERROR] Engine directory does not exist: {engine_path}")
        sys.exit(1)

    # Load summary to get num_layers
    summary_path = oracle_path / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        num_layers = summary["num_layers"]
    else:
        # Count layer directories
        num_layers = sum(1 for d in oracle_path.iterdir() if d.is_dir() and d.name.startswith("layer_"))

    print(f"Comparing oracle vs engine — {num_layers} layers, phase={args.phase}")
    print(f"  Oracle: {oracle_path}")
    print(f"  Engine: {engine_path}")
    print(f"  Threshold: {args.threshold}")

    all_passed, results, first_fail_layer, first_fail_tensor = compare_layers(
        oracle_path, engine_path, args.phase, num_layers, args.threshold,
    )

    print_results(results, first_fail_layer, first_fail_tensor)

    passed_count = sum(1 for r in results if r.get("passed", False))
    total = len(results)
    print(f"\n{'=' * 70}")
    print(f"  {passed_count}/{total} tensors passed")
    if all_passed:
        print("  ALL TENSORS PASSED")
    else:
        print("  SOME TENSORS FAILED")
    print(f"{'=' * 70}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
