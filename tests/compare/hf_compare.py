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


# Tensor name mapping: stripped engine tensor name → oracle tensor name
# Engine dumps use names like "gdn.norm1_gpu0.raw" or "attn.attn_output_gpu0.raw"
# We strip the prefix (gdn., attn., mlp.) and GPU suffix (_gpu0) to get the key,
# then look up the corresponding oracle tensor name.
ENGINE_TO_ORACLE = {
    # GDN layer tensors:
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
    "core_attn_out": "attn_output",
    "z_gate": "z_gate",
    "norm_output": "norm2_output",
    "o_proj": "mlp_output",
    "output": "layer_output",
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
    "after_ar": "after_ar",
    # MLP layer tensors:
    "norm2": "norm2_output",
    "gate_proj": "gate_proj",
    "up_proj": "up_proj",
    "silu": "silu",
    "down_raw": "down_raw",
    "down_ar": "down_ar",
}

# Legacy tensor names that the oracle may save but that have no direct engine dump
# (e.g. older oracle versions without GDN intermediates)
_ORACLE_NAMES = [
    "norm1_input",
    "norm1_output",
    "attn_output",
    "norm2_input",
    "norm2_output",
    "mlp_output",
    "layer_output",
]


def _strip_engine_prefix(stem: str) -> Optional[str]:
    """Strip engine dump prefix and GPU suffix to get oracle tensor name.

    Examples:
        "attn.norm1_input_gpu0" → "norm1_input"
        "gdn.norm1_gpu0" → "norm1_output"
        "gdn.core_attn_out_gpu0" → "attn_output"
        "mlp.mlp_output_gpu0" → "mlp_output"
    """
    # Remove GPU suffix
    for suffix in ["_gpu0", "_gpu1", "_gpu2", "_gpu3"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    # Remove known prefixes
    for prefix in ["attn.", "gdn.", "mlp.", "residual.", "final_"]:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break

    return ENGINE_TO_ORACLE.get(stem)


def _discover_engine_dumps(engine_dir: Path, layer_idx: int, phase: str) -> Dict[str, Path]:
    """Find all .raw dumps for a given layer and phase.

    Returns dict mapping oracle tensor name → path to .raw file.
    """
    layer_dir = engine_dir / f"layer_{layer_idx}" / phase
    if not layer_dir.exists():
        return {}

    dumps = {}
    for raw_path in sorted(layer_dir.glob("*.raw")):
        stem = raw_path.stem
        oracle_name = _strip_engine_prefix(stem)
        if oracle_name is not None:
            # Prefer GPU 0 dumps for comparison (TP shards are partial)
            if oracle_name not in dumps or "gpu0" in stem:
                dumps[oracle_name] = raw_path
    return dumps


def _load_oracle_tensor(oracle_dir: Path, layer_idx: int, phase: str, tensor_name: str) -> Optional[torch.Tensor]:
    """Load an oracle .pt tensor."""
    pt_path = oracle_dir / f"layer_{layer_idx}" / phase / f"{tensor_name}.pt"
    if not pt_path.exists():
        return None
    return torch.load(pt_path, weights_only=True)


def _load_engine_tensor(raw_path: Path, oracle_shape: tuple) -> Optional[torch.Tensor]:
    """Load an engine .raw bf16 tensor and reshape to match oracle shape."""
    try:
        flat = load_raw_bf16(str(raw_path), (-1,))
        return flat.reshape(oracle_shape).float()
    except Exception as e:
        print(f"  [WARN] Failed to load {raw_path}: {e}")
        return None


def compare_layers(
    oracle_dir: Path,
    engine_dir: Path,
    phase: str,
    num_layers: int,
    threshold: float = 0.99,
) -> Tuple[bool, List[dict], Optional[int], Optional[str]]:
    """Compare all layers between oracle and engine dumps.

    Returns:
        (all_passed, results, first_fail_layer, first_fail_tensor)
    """
    results = []
    first_fail_layer = None
    first_fail_tensor = None

    for i in range(num_layers):
        engine_dumps = _discover_engine_dumps(engine_dir, i, phase)

        # Discover oracle tensors: from engine-mapped names + legacy oracle names
        all_oracle_names = set(engine_dumps.keys()) | set(_ORACLE_NAMES)
        oracle_tensors = {}
        for name in sorted(all_oracle_names):
            t = _load_oracle_tensor(oracle_dir, i, phase, name)
            if t is not None:
                oracle_tensors[name] = t

        # Collect all tensor names to compare (union of engine and oracle)
        all_tensor_names = sorted(set(oracle_tensors.keys()) | set(engine_dumps.keys()))

        for name in all_tensor_names:
            # Skip if neither oracle nor engine has this tensor
            if name not in oracle_tensors and name not in engine_dumps:
                continue
            # Oracle-only: report as missing_engine_dump
            if name not in engine_dumps:
                results.append({
                    "layer": i,
                    "tensor": name,
                    "cos": 0.0,
                    "l2_err": 1.0,
                    "max_diff": -1.0,
                    "passed": False,
                    "error": "missing_engine_dump",
                })
                if first_fail_layer is None:
                    first_fail_layer = i
                    first_fail_tensor = name
                continue
            # Engine-only: oracle doesn't have this tensor (skip, no baseline to compare)
            if name not in oracle_tensors:
                continue

            oracle_t = oracle_tensors[name]
            engine_t = _load_engine_tensor(engine_dumps[name], oracle_t.shape)

            if engine_t is None:
                results.append({
                    "layer": i,
                    "tensor": name,
                    "cos": 0.0,
                    "l2_err": 1.0,
                    "max_diff": -1.0,
                    "passed": False,
                    "error": "load_failed",
                })
                if first_fail_layer is None:
                    first_fail_layer = i
                    first_fail_tensor = name
                continue

            if engine_t.numel() != oracle_t.numel():
                results.append({
                    "layer": i,
                    "tensor": name,
                    "cos": 0.0,
                    "l2_err": 1.0,
                    "max_diff": -1.0,
                    "passed": False,
                    "error": f"size_mismatch engine={engine_t.numel()} oracle={oracle_t.numel()}",
                })
                if first_fail_layer is None:
                    first_fail_layer = i
                    first_fail_tensor = name
                continue

            cos = cos_sim(engine_t, oracle_t)
            l2 = l2_error(engine_t, oracle_t)
            stats = element_stats(engine_t, oracle_t)
            passed = cos >= threshold

            results.append({
                "layer": i,
                "tensor": name,
                "cos": cos,
                "l2_err": l2,
                "max_diff": stats["max"],
                "passed": passed,
            })

            if not passed and first_fail_layer is None:
                first_fail_layer = i
                first_fail_tensor = name

    all_passed = all(r.get("passed", False) for r in results)
    return all_passed, results, first_fail_layer, first_fail_tensor


def print_results(results: List[dict], first_fail_layer, first_fail_tensor):
    """Print comparison results grouped by layer."""
    # Group results by layer
    from collections import defaultdict

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
