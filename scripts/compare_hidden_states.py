#!/usr/bin/env python3
"""Compare per-layer hidden states between PyTorch oracle dumps and Rust inference engine probes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def read_bf16(path: str) -> np.ndarray:
    """Read raw BF16 bytes from disk, converting to float32.

    Uses PyTorch for correct BF16→FP32 conversion (numpy lacks native BF16).
    """
    raw = np.fromfile(path, dtype=np.uint8)
    t = torch.frombuffer(raw.tobytes(), dtype=torch.bfloat16).to(torch.float32)
    return t.numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_flat = a.ravel()
    b_flat = b.ravel()
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_metrics(oracle_arr: np.ndarray, infer_arr: np.ndarray) -> dict:
    """Compute comparison metrics between oracle and infer arrays."""
    diff = np.abs(oracle_arr - infer_arr)
    max_abs_err = float(np.max(diff))
    mean_abs_err = float(np.mean(diff))
    max_oracle = float(np.max(np.abs(oracle_arr)))
    rel_err = max_abs_err / max_oracle if max_oracle > 0 else float("inf")
    cos = cosine_similarity(oracle_arr, infer_arr)

    return {
        "cosine": cos,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "rel_err": rel_err,
    }


def status_label(cosine: float, threshold: float) -> str:
    """Determine status based on cosine similarity and threshold."""
    if cosine >= threshold:
        return "PASS"
    elif cosine >= 0.95:
        return "WARN"
    else:
        return "FAIL"


def find_infer_norm1_input(infer_dir: Path, layer: int, layer_type: str) -> Path | None:
    """Find the norm1 input file for a given layer in the infer dump.

    The filename uses either 'attn' or 'gdn' based on layer type.
    Returns the path if found, None otherwise.
    """
    # Try both naming conventions
    for suffix in ("attn", "gdn"):
        candidate = infer_dir / f"layer_{layer}" / "prefill" / f"{suffix}.norm1_input_gpu0.raw"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare per-layer hidden states between oracle and infer dumps"
    )
    parser.add_argument("--oracle-dir", required=True, help="Directory containing oracle dumps")
    parser.add_argument("--infer-dir", required=True, help="Directory containing infer probe dumps")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Cosine similarity threshold for PASS (default: 0.99)",
    )
    args = parser.parse_args()

    oracle_dir = Path(args.oracle_dir)
    infer_dir = Path(args.infer_dir)
    threshold = args.threshold

    # ------------------------------------------------------------------
    # 1. Load oracle config
    # ------------------------------------------------------------------
    if not (oracle_dir / "oracle_config.json").exists():
        print(f"ERROR: {oracle_dir}/oracle_config.json not found", file=sys.stderr)
        sys.exit(1)

    with open(oracle_dir / "oracle_config.json") as f:
        oracle_config = json.load(f)

    num_layers = oracle_config["num_layers"]
    layer_types = oracle_config.get("layer_types", {})
    print(f"Loaded oracle config: {num_layers} layers")

    # ------------------------------------------------------------------
    # 2. Comparison results storage
    # ------------------------------------------------------------------
    results: list[dict] = []
    pass_count = 0
    warn_count = 0
    fail_count = 0
    missing_count = 0
    worst_cosine = 1.0
    worst_layer = "N/A"

    def add_result(
        layer_label: str,
        layer_type: str,
        cosine: float | None,
        max_abs_err: float | None,
        mean_abs_err: float | None,
        status: str,
    ) -> None:
        """Record a comparison result and update counters."""
        results.append(
            {
                "layer": layer_label,
                "type": layer_type,
                "cosine": cosine,
                "max_abs_err": max_abs_err,
                "mean_abs_err": mean_abs_err,
                "status": status,
            }
        )

    # ------------------------------------------------------------------
    # 3. Compare embedding output (layer 0)
    # ------------------------------------------------------------------
    embed_oracle_path = oracle_dir / "layer_0" / "oracle" / "embed_output.raw"
    embed_infer_path = infer_dir / "layer_0" / "prefill" / "embed.output_gpu0.raw"

    if embed_oracle_path.exists() and embed_infer_path.exists():
        embed_oracle = read_bf16(str(embed_oracle_path))
        embed_infer = read_bf16(str(embed_infer_path))

        if embed_oracle.shape == embed_infer.shape:
            metrics = compute_metrics(embed_oracle, embed_infer)
            cos = metrics["cosine"]
            status = status_label(cos, threshold)

            if status == "PASS":
                pass_count += 1
            elif status == "WARN":
                warn_count += 1
            else:
                fail_count += 1

            if cos < worst_cosine:
                worst_cosine = cos
                worst_layer = "embed"

            add_result(
                layer_label="embed",
                layer_type="-",
                cosine=cos,
                max_abs_err=metrics["max_abs_err"],
                mean_abs_err=metrics["mean_abs_err"],
                status=status,
            )
        else:
            add_result("embed", "-", None, None, None, "SHAPE MISMATCH")
    elif not embed_oracle_path.exists():
        add_result("embed", "-", None, None, None, "MISSING (oracle)")
        missing_count += 1
    elif not embed_infer_path.exists():
        add_result("embed", "-", None, None, None, "MISSING (infer)")
        missing_count += 1

    # ------------------------------------------------------------------
    # 4. Compare per-layer hidden states
    # ------------------------------------------------------------------
    for layer_idx in range(num_layers):
        lt = layer_types.get(str(layer_idx), "?")

        # Oracle hidden states (input to this layer)
        oracle_path = (
            oracle_dir / f"layer_{layer_idx}" / "oracle" / "hidden_states.raw"
        )
        # Infer norm1 input (input to this layer's attention/MLP block)
        infer_path = find_infer_norm1_input(infer_dir, layer_idx, lt)

        if oracle_path.exists() and infer_path is not None and infer_path.exists():
            oracle_arr = read_bf16(str(oracle_path))
            infer_arr = read_bf16(str(infer_path))

            if oracle_arr.shape == infer_arr.shape:
                metrics = compute_metrics(oracle_arr, infer_arr)
                cos = metrics["cosine"]
                status = status_label(cos, threshold)

                if status == "PASS":
                    pass_count += 1
                elif status == "WARN":
                    warn_count += 1
                else:
                    fail_count += 1

                if cos < worst_cosine:
                    worst_cosine = cos
                    worst_layer = str(layer_idx)

                add_result(
                    layer_label=str(layer_idx),
                    layer_type=lt,
                    cosine=cos,
                    max_abs_err=metrics["max_abs_err"],
                    mean_abs_err=metrics["mean_abs_err"],
                    status=status,
                )
            else:
                add_result(str(layer_idx), lt, None, None, None, "SHAPE MISMATCH")
        elif not oracle_path.exists():
            add_result(str(layer_idx), lt, None, None, None, "MISSING (oracle)")
            missing_count += 1
        elif infer_path is None or not infer_path.exists():
            add_result(str(layer_idx), lt, None, None, None, "MISSING (infer)")
            missing_count += 1

    # ------------------------------------------------------------------
    # 5. Compare final logits
    # ------------------------------------------------------------------
    logits_oracle_path = oracle_dir / "final" / "oracle" / "logits.raw"
    # Infer logits might be at the last layer or in a separate path
    logits_infer_path = infer_dir / f"layer_{num_layers - 1}" / "prefill" / "final.logits_gpu0.raw"

    if not logits_infer_path.exists():
        # Try alternative location
        alt_path = infer_dir / "final" / "prefill" / "logits_gpu0.raw"
        if alt_path.exists():
            logits_infer_path = alt_path

    if logits_oracle_path.exists() and logits_infer_path is not None and logits_infer_path.exists():
        logits_oracle = read_bf16(str(logits_oracle_path))
        logits_infer = read_bf16(str(logits_infer_path))

        if logits_oracle.shape == logits_infer.shape:
            metrics = compute_metrics(logits_oracle, logits_infer)
            cos = metrics["cosine"]
            status = status_label(cos, threshold)

            if status == "PASS":
                pass_count += 1
            elif status == "WARN":
                warn_count += 1
            else:
                fail_count += 1

            if cos < worst_cosine:
                worst_cosine = cos
                worst_layer = "final"

            add_result(
                layer_label="final",
                layer_type="logits",
                cosine=cos,
                max_abs_err=metrics["max_abs_err"],
                mean_abs_err=metrics["mean_abs_err"],
                status=status,
            )
        else:
            add_result("final", "logits", None, None, None, "SHAPE MISMATCH")
    elif not logits_oracle_path.exists():
        add_result("final", "logits", None, None, None, "MISSING (oracle)")
        missing_count += 1
    else:
        add_result("final", "logits", None, None, None, "MISSING (infer)")
        missing_count += 1

    # ------------------------------------------------------------------
    # 6. Print results table
    # ------------------------------------------------------------------
    print()
    print(f"{'Layer':>7} | {'Type':>4} | {'Cosine':>12} | {'MaxAbsErr':>12} | {'MeanAbsErr':>12} | Status")
    print("-" * 80)

    for r in results:
        cosine_str = f"{r['cosine']:.5f}" if r["cosine"] is not None else "N/A"
        max_err_str = f"{r['max_abs_err']:.6g}" if r["max_abs_err"] is not None else "N/A"
        mean_err_str = (
            f"{r['mean_abs_err']:.6g}" if r["mean_abs_err"] is not None else "N/A"
        )
        print(
            f"{r['layer']:>7} | {r['type']:>4} | {cosine_str:>12} | "
            f"{max_err_str:>12} | {mean_err_str:>12} | {r['status']}"
        )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    total_compared = pass_count + warn_count + fail_count
    print()
    print(
        f"Summary: {pass_count}/{total_compared} layers PASS, "
        f"{warn_count} WARN, {fail_count} FAIL"
    )
    if missing_count > 0:
        print(f"         {missing_count} MISSING")
    print(f"Worst layer: {worst_layer} (cos={worst_cosine:.5f})")


if __name__ == "__main__":
    main()
