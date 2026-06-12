#!/usr/bin/env python3
"""Compare GDN intermediate tensors between Rust CUDA dump and HuggingFace reference.

Loads .raw BF16 files from our Rust engine and .npy files from the HF reference,
then computes per-tensor comparison metrics (cosine similarity, MSE, max error, etc.).
"""

import argparse
import os
import numpy as np


def bf16_raw_to_f32(path: str) -> np.ndarray:
    """Read a raw BF16 file and convert to float32."""
    data = np.frombuffer(open(path, "rb").read(), dtype=np.uint16)
    return (data.astype(np.uint32) << 16).view(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened tensors."""
    flat_a = a.ravel().astype(np.float64)
    flat_b = b.ravel().astype(np.float64)
    dot = np.dot(flat_a, flat_b)
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return float(dot / (norm_a * norm_b))


def mean_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean |a-b| / |b| where |b| > eps."""
    denom = np.abs(b.ravel())
    mask = denom > 1e-8
    if not mask.any():
        return float("nan")
    diff = np.abs(a.ravel() - b.ravel())
    return float(np.mean(diff[mask] / denom[mask]))


def compare_tensors(our_dir: str, ref_dir: str) -> list[dict]:
    """Compare all matching tensors between our dump and the reference."""
    # Tensor name -> expected shape (for our raw dumps, need reshape; ref uses .npy shape)
    # We'll discover shapes from the reference .npy files.
    
    results = []
    
    # Get our tensor names (from .raw filenames)
    our_files = {}
    for f in os.listdir(our_dir):
        if f.endswith(".raw"):
            name = f[:-4]  # strip .raw
            our_files[name] = os.path.join(our_dir, f)
    
    # Get reference tensor names (from .npy filenames)
    ref_files = {}
    for f in os.listdir(ref_dir):
        if f.endswith(".npy"):
            name = f[:-4]  # strip .npy
            ref_files[name] = os.path.join(ref_dir, f)
    
    # Compare matching tensors
    common_names = sorted(set(our_files.keys()) & set(ref_files.keys()))
    
    for name in common_names:
        our_path = our_files[name]
        ref_path = ref_files[name]
        
        try:
            # Load reference tensor
            ref_tensor = np.load(ref_path)
            
            # Load our tensor as BF16 -> f32
            our_f32 = bf16_raw_to_f32(our_path)
            
            # Reshape to match reference shape
            if ref_tensor.ndim == 1:
                our_tensor = our_f32[:ref_tensor.shape[0]]
            else:
                our_tensor = our_f32[:np.prod(ref_tensor.shape)].reshape(ref_tensor.shape)
            
            # Compute metrics
            a_flat = our_tensor.ravel()
            b_flat = ref_tensor.ravel().astype(np.float32)
            
            cos_sim = cosine_similarity(our_tensor, ref_tensor)
            mse = float(np.mean((a_flat - b_flat) ** 2))
            max_err = float(np.max(np.abs(a_flat - b_flat)))
            mae = float(np.mean(np.abs(a_flat - b_flat)))
            mre = mean_relative_error(our_tensor, ref_tensor)
            
            # Check for divergence
            diverges = cos_sim < 0.99 or max_err > 0.1
            
            results.append({
                "name": name,
                "shape": tuple(ref_tensor.shape),
                "cos_sim": cos_sim,
                "mse": mse,
                "max_err": max_err,
                "mae": mae,
                "mre": mre,
                "diverges": diverges,
            })
            
        except Exception as e:
            results.append({
                "name": name,
                "shape": "?",
                "cos_sim": float("nan"),
                "mse": float("nan"),
                "max_err": float("nan"),
                "mae": float("nan"),
                "mre": float("nan"),
                "diverges": True,
                "error": str(e),
            })
    
    return results


def print_results(results: list[dict]) -> None:
    """Print comparison results as a formatted table."""
    print("=" * 90)
    print(f"{'Tensor':<20} {'Shape':>15} {'Cos Sim':>8} {'MSE':>10} {'Max Err':>10} {'MAE':>10} {'MRE':>10} {'Flag':>6}")
    print("=" * 90)
    
    for r in results:
        shape_str = str(r["shape"]) if len(str(r["shape"])) <= 15 else str(r["shape"])[:14] + "."
        
        cos_sim = f"{r['cos_sim']:.6f}" if not np.isnan(r["cos_sim"]) else "   NaN"
        mse = f"{r['mse']:.8e}" if not np.isnan(r["mse"]) else "    NaN"
        max_err = f"{r['max_err']:.8e}" if not np.isnan(r["max_err"]) else "    NaN"
        mae = f"{r['mae']:.8e}" if not np.isnan(r["mae"]) else "    NaN"
        mre = f"{r['mre']:.6f}" if not np.isnan(r["mre"]) else "   NaN"
        
        flag = "**FAIL**" if r["diverges"] else "  OK  "
        
        print(f"{r['name']:<20} {shape_str:>15} {cos_sim:>8} {mse:>10} {max_err:>10} {mae:>10} {mre:>10} {flag:>6}")
    
    print("=" * 90)
    
    # Summary
    failed = [r for r in results if r.get("diverges")]
    ok = [r for r in results if not r.get("diverges", False)]
    errors = [r for r in results if "error" in r]
    
    print(f"\nSummary: {len(ok)} OK, {len(failed)} divergent, {len(errors)} errors")
    
    if failed:
        print("\nDivergent tensors (cos_sim < 0.99 or max_err > 0.1):")
        for r in failed:
            reasons = []
            if not np.isnan(r["cos_sim"]) and r["cos_sim"] < 0.99:
                reasons.append(f"cos_sim={r['cos_sim']:.6f}")
            if not np.isnan(r["max_err"]) and r["max_err"] > 0.1:
                reasons.append(f"max_err={r['max_err']:.8e}")
            print(f"  - {r['name']} ({', '.join(reasons)})")


def main():
    parser = argparse.ArgumentParser(description="Compare GDN intermediate tensors")
    parser.add_argument("--our-dir", default="/tmp/our_gdn", help="Directory with our .raw BF16 dumps")
    parser.add_argument("--ref-dir", default="/tmp/ref_gdn", help="Directory with reference .npy files")
    parser.add_argument("--seq-len", type=int, default=7, help="Sequence length (for shape verification)")
    parser.add_argument("--num-v-heads", type=int, default=48, help="Number of value heads")
    parser.add_argument("--head-v-dim", type=int, default=128, help="Head V dimension")
    args = parser.parse_args()
    
    if not os.path.isdir(args.our_dir):
        print(f"Error: our directory '{args.our_dir}' does not exist")
        return 1
    
    if not os.path.isdir(args.ref_dir):
        print(f"Error: reference directory '{args.ref_dir}' does not exist")
        return 1
    
    results = compare_tensors(args.our_dir, args.ref_dir)
    
    if not results:
        print("No matching tensors found between our dump and reference.")
        our_files = set(os.listdir(args.our_dir)) - {".gitkeep"}
        ref_files = set(os.listdir(args.ref_dir)) - {".gitkeep"}
        print(f"  Our files: {[f for f in our_files if f.endswith('.raw')]}")
        print(f"  Ref files: {[f for f in ref_files if f.endswith('.npy')]}")
        return 1
    
    print_results(results)
    return 0


if __name__ == "__main__":
    exit(main())
