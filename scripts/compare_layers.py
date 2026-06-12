#!/usr/bin/env python3
"""Compare our engine's per-layer hidden states against HuggingFace reference.

Reads:
  /tmp/ref_hidden/layer_{i}_lastpos.pt  - reference hidden states (PyTorch tensor, last token pos)
  /tmp/our_hidden/layer_{i}_gpu0.raw    - our engine's hidden states (raw bf16, full sequence)

Outputs:
  Per-layer cosine similarity, MSE, and max absolute error
  Identifies first layer where divergence exceeds threshold.

Usage:
  uv run python3 scripts/compare_layers.py [--ref-dir /tmp/ref_hidden] [--our-dir /tmp/our_hidden]
"""

import torch
import numpy as np
import os
import struct
import argparse
from pathlib import Path

# half::bf16 in Rust is stored as bf16 (1 sign, 8 exponent, 7 mantissa)
# PyTorch bfloat16 is the same format

def parse_bf16_le(bytes_data):
    """Convert little-endian bf16 bytes to numpy float32."""
    arr = np.frombuffer(bytes_data, dtype=np.uint16)
    # bf16 -> float32: shift left by 16 bits
    arr_f32 = arr.astype(np.uint32) << 16
    return arr_f32.view(np.float32)

def load_our_layer(filepath, seq_len=15, hidden_size=5120):
    """Load our engine's raw bf16 layer dump."""
    data = Path(filepath).read_bytes()
    expected = seq_len * hidden_size * 2
    assert len(data) == expected, f"Expected {expected} bytes, got {len(data)}"
    # Parse as bf16, reshape to [seq_len, hidden_size]
    arr = parse_bf16_le(data).reshape(seq_len, hidden_size)
    # Convert to torch float32 for comparison
    return torch.from_numpy(arr).float()

def main():
    parser = argparse.ArgumentParser(description="Compare per-layer hidden states")
    parser.add_argument("--ref-dir", default="/tmp/ref_hidden", help="Reference hidden states directory")
    parser.add_argument("--our-dir", default="/tmp/our_hidden", help="Our engine hidden states directory")
    parser.add_argument("--threshold", type=float, default=0.01, help="Max allowed relative error")
    parser.add_argument("--seq-len", type=int, default=15, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=5120, help="Hidden dimension size")
    parser.add_argument("--num-layers", type=int, default=64, help="Number of transformer layers")
    parser.add_argument("--print-topk", type=int, default=5, help="Print top-K values for comparison")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)
    our_dir = Path(args.our_dir)
    seq_len = args.seq_len
    hidden_size = args.hidden_size
    num_layers = args.num_layers

    # Input verification
    ref_input = torch.load(ref_dir / "input_ids.pt")
    print(f"Reference input IDs ({len(ref_input)} tokens): {ref_input.tolist()}")
    print(f"Hidden state reference: layer i = after layer (i-1) in HF convention")
    print(f"Our engine: layer i = after layer i (matching convention)")
    print()

    # Compare layer by layer
    # Reference: layer_{i}_lastpos.pt where i=0 is embedding, i=1 is after layer 0
    # Our engine: layer_{i}_gpu0.raw where i=0 is after layer 0
    # So reference layer_{i+1} should match our layer_{i}
    diverged_at = None
    similarities = []
    mses = []
    max_errs = []

    for layer_idx in range(num_layers + 1):
        ref_key = layer_idx  # ref: 0=embed, 1=layer0, ..., 64=layer63
        our_key = layer_idx  # our: 0=layer0, 1=layer1, ..., 63=layer63

        # Load reference (last position only)
        ref_file = ref_dir / f"layer_{ref_key}_lastpos.pt"
        if not ref_file.exists():
            print(f"Ref file {ref_file} not found, skipping...")
            continue
        ref_hidden = torch.load(ref_file).float()  # [hidden_size]

        # Load our engine's hidden state
        if ref_key == 0:
            # Embedding output - our engine doesn't dump this
            # Skip comparison for embedding
            print(f"Layer [embed]: reference only (our engine doesn't dump embedding)")

            # Embedding is usually very close since it's just a lookup
            # But we can load the ref and print norms
            print(f"  ref norm={ref_hidden.norm().item():.6f}")
            print(f"  ref first5={ref_hidden[:5].tolist()}")
            continue

        our_file = our_dir / f"layer_{our_key - 1}_gpu0.raw"
        if not our_file.exists():
            print(f"Our file {our_file} not found, skipping...")
            continue

        our_full = load_our_layer(our_file, seq_len, hidden_size)  # [seq, hidden]
        our_hidden = our_full[-1, :]  # Last token position

        # Compare
        diff = (our_hidden - ref_hidden).abs()
        mse = (diff ** 2).mean().item()
        max_err = diff.max().item()
        ref_norm = ref_hidden.norm().item()
        our_norm = our_hidden.norm().item()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            our_hidden.unsqueeze(0), ref_hidden.unsqueeze(0)
        ).item()

        # Relative error
        ref_nz = ref_hidden.abs() > 1e-8 if ref_norm > 1e-8 else torch.ones_like(ref_hidden, dtype=torch.bool)
        rel_err = (diff / (ref_hidden.abs() + 1e-10))[ref_nz]
        mean_rel = rel_err.mean().item() if len(rel_err) > 0 else 0.0
        max_rel = rel_err.max().item() if len(rel_err) > 0 else 0.0

        similarities.append(cos_sim)
        mses.append(mse)
        max_errs.append(max_err)

        # Determine if this layer is a full_attention or gdn layer
        # Every 4th layer (3, 7, 11, ...) is full_attention (0-indexed)
        layer_type = "FULL-ATTN" if (layer_idx - 1) % 4 == 3 else "GDN"
        if (layer_idx - 1) == 0:
            layer_type = "GDN"

        # Divergence check
        is_diverged = cos_sim < 0.999 or max_err > args.threshold
        if is_diverged and diverged_at is None:
            diverged_at = layer_idx - 1  # 0-indexed layer number

        status = "*** DIVERGED ***" if is_diverged else "OK"
        print(f"Layer {layer_idx - 1:3d} [{layer_type:>10s}]: "
              f"cos_sim={cos_sim:.8f}  "
              f"MSE={mse:.8e}  "
              f"max_err={max_err:.6f}  "
              f"mean_rel={mean_rel:.6f}  "
              f"max_rel={max_rel:.6f}  "
              f"{status}")

        # For diverged layers, show some values
        if is_diverged and layer_idx - 1 < 3:
            topk = args.print_topk
            _, ref_top_idx = ref_hidden.abs().topk(topk)
            print(f"  Ref top{topk} values:  {ref_hidden[ref_top_idx].tolist()}")
            print(f"  Our top{topk} values:  {our_hidden[ref_top_idx].tolist()}")

    print()
    if diverged_at is not None:
        print(f"FIRST DIVERGENCE at Layer {diverged_at}")
    else:
        print("All layers match within threshold!")

    # Summary stats
    if similarities:
        print(f"\nSimilarity stats over all layers:")
        print(f"  cos_sim: min={min(similarities):.8f} mean={sum(similarities)/len(similarities):.8f}")
        print(f"  MSE:     min={min(mses):.8e} max={max(mses):.8e} mean={sum(mses)/len(mses):.8e}")


if __name__ == "__main__":
    main()
