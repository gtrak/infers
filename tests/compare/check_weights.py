#!/usr/bin/env python3
"""Check oracle model's actual decompressed weights vs our reference dequant.

Loads the model via transformers (like the oracle), extracts the weight of
in_proj_qkv (which should be decompressed by compressed-tensors during loading),
and compares against our manual dequant from safetensors.

Usage:
    python -m tests.compare.check_weights [--model-dir /path/to/model]
"""

import argparse
import os
import torch
import numpy as np
from safetensors import safe_open


def manual_dequant(packed, scale, global_scale):
    """NVFP4 dequant matching our Rust kernel implementation."""
    m, n_half = packed.shape  # [N, K/2]
    n = n_half * 2            # [N, K]
    group_size = n // scale.shape[1]  # K // groups
    
    # Unpack FP4 nibbles: compressed-tensors format [low, high] per byte
    packed_np = packed.cpu().numpy().astype(np.uint8)
    
    # FP4 E2M1 lookup table
    fp4_lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    
    # Convert scale to float32 numpy
    scale_np = scale.cpu().float().numpy()
    gs = float(global_scale)
    
    result = np.zeros((m, n), dtype=np.float32)
    
    for row in range(m):
        for g in range(scale.shape[1]):
            lo_nibble = np.uint8(0x0F)  # low nibble mask
            hi_nibble = np.uint8(0xF0)  # high nibble mask
            s = scale_np[row, g]
            
            for i in range(group_size // 2):
                byte_val = packed_np[row, g * (group_size // 2) + i]
                lo = byte_val & lo_nibble  # bits 3:0
                hi = (byte_val & hi_nibble) >> 4  # bits 7:4
                
                # LOW nibble first (position 0), HIGH nibble second (position 1)
                out_idx = g * group_size + i * 2
                result[row, out_idx] = fp4_lut[lo] * s / gs
                result[row, out_idx + 1] = fp4_lut[hi] * s / gs
    
    return torch.from_numpy(result)


def load_safetensors_weight(model_dir, tensor_name):
    """Load a weight tensor from safetensors file."""
    model_dir = os.path.expanduser(model_dir)
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    safetensors_idx = os.path.join(model_dir, "model.safetensors.index.json")
    
    if os.path.exists(safetensors_idx):
        import json
        with open(safetensors_idx) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        
        # Find which shard contains this tensor
        if tensor_name not in weight_map:
            # Search for alternatives
            for key, shard in weight_map.items():
                if tensor_name in key or key.endswith(tensor_name):
                    return load_safetensors_weight(model_dir, key)
            raise KeyError(f"Tensor {tensor_name} not found in index")
        
        shard_path = os.path.join(model_dir, weight_map[tensor_name])
        with safe_open(shard_path, framework="pt") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
    else:
        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt") as f:
                return f.get_tensor(tensor_name)
    
    raise KeyError(f"Tensor {tensor_name} not found")


def find_quant_tensors(model_dir, tensor_prefix="model.layers.0"):
    """Find all NVFP4-related tensors for a given layer."""
    import json
    
    safetensors_idx = os.path.join(model_dir, "model.safetensors.index.json")
    with open(safetensors_idx) as f:
        index = json.load(f)
    
    weight_map = index.get("weight_map", {})
    
    packed_tensors = []
    scale_tensors = []
    global_scale_tensors = []
    
    for key in weight_map:
        if key.startswith(tensor_prefix) and "weight_packed" in key:
            packed_key = key
            scale_key = key.replace("weight_packed", "weight_scale")
            gs_key = key.replace("weight_packed", "weight_global_scale")
            
            packed_tensors.append(packed_key)
            scale_tensors.append(scale_key if scale_key in weight_map else None)
            global_scale_tensors.append(gs_key if gs_key in weight_map else None)
    
    return packed_tensors, scale_tensors, global_scale_tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, 
                        default=os.path.expanduser("~/opt/vllm/models/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm"))
    parser.add_argument("--layer", type=str, default="model.layers.0")
    parser.add_argument("--output-dir", type=str, default="/tmp/weight_check")
    args = parser.parse_args()
    
    model_dir = os.path.expanduser(args.model_dir)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load the model via transformers (like the oracle)
    print(f"Loading model from {model_dir}...")
    import warnings
    warnings.filterwarnings("ignore")
    
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  Device: {model.device}")
    
    # Step 2: Extract the decompressed weight from the oracle model
    # Navigate the model structure to find in_proj_qkv
    decoder_layer = model.model.layers[0]
    lin_attn = decoder_layer.linear_attn
    in_proj_qkv = lin_attn.in_proj_qkv
    
    print(f"\nin_proj_qkv weight shape: {in_proj_qkv.weight.shape}")
    print(f"in_proj_qkv weight dtype: {in_proj_qkv.weight.dtype}")
    print(f"in_proj_qkv weight device: {in_proj_qkv.weight.device}")
    
    # Save oracle's decompressed weight
    oracle_weight = in_proj_qkv.weight.detach().cpu().float()
    torch.save(oracle_weight, os.path.join(output_dir, "oracle_weight.pt"))
    print(f"\nSaved oracle weight: {oracle_weight.shape}")
    print(f"  First 10 elements: {oracle_weight[0, :10].tolist()}")
    print(f"  Mean: {oracle_weight.mean().item():.6f}")
    print(f"  Std: {oracle_weight.std().item():.6f}")
    print(f"  Min: {oracle_weight.min().item():.6f}")
    print(f"  Max: {oracle_weight.max().item():.6f}")
    
    # Step 3: Load the raw packed weights from safetensors
    qkv_packed_name = f"{args.layer}.linear_attn.in_proj_qkv.weight_packed"
    qkv_scale_name = f"{args.layer}.linear_attn.in_proj_qkv.weight_scale"
    qkv_gs_name = f"{args.layer}.linear_attn.in_proj_qkv.weight_global_scale"
    
    print(f"\nLoading safetensors: {qkv_packed_name}")
    packed = load_safetensors_weight(model_dir, qkv_packed_name)
    scale = load_safetensors_weight(model_dir, qkv_scale_name)
    global_scale = load_safetensors_weight(model_dir, qkv_gs_name)
    
    print(f"  packed: {packed.shape} {packed.dtype}")
    print(f"  scale: {scale.shape} {scale.dtype}")
    print(f"  global_scale: {global_scale} {global_scale.dtype}")
    
    # Step 4: Manual dequant
    print(f"\nRunning manual dequant...")
    dequantized = manual_dequant(packed, scale, global_scale.item())
    torch.save(dequantized, os.path.join(output_dir, "dequantized_weight.pt"))
    print(f"  Result: {dequantized.shape}")
    print(f"  First 10 elements: {dequantized[0, :10].tolist()}")
    print(f"  Mean: {dequantized.mean().item():.6f}")
    print(f"  Std: {dequantized.std().item():.6f}")
    print(f"  Min: {dequantized.min().item():.6f}")
    print(f"  Max: {dequantized.max().item():.6f}")
    
    # Step 5: Compare
    print(f"\n=== COMPARISON ===")
    
    # Ensure both on CPU
    oracle_cpu = oracle_weight.cpu()
    
    # Check if shapes match
    if oracle_cpu.shape != dequantized.shape:
        print(f"Shape MISMATCH: oracle {oracle_cpu.shape} vs dequant {dequantized.shape}")
        # May need to transpose
        if oracle_cpu.shape[0] == dequantized.shape[1] and oracle_cpu.shape[1] == dequantized.shape[0]:
            print("  Shapes are transposed - trying transpose")
            dequantized = dequantized.T
    
    # Align shapes
    if oracle_cpu.shape != dequantized.shape:
        print(f"  Still mismatch. Trying to align...")
        return
    
    # Compute metrics
    cos = torch.nn.functional.cosine_similarity(
        oracle_cpu.flatten().unsqueeze(0),
        dequantized.flatten().unsqueeze(0)
    ).item()
    
    diff = (oracle_cpu - dequantized).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    l2_err = torch.sqrt((diff ** 2).mean()).item()
    
    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  L2 error: {l2_err:.6f}")
    
    # Also compare first row
    print(f"\n  Oracle first 20: {oracle_cpu[0, :20].tolist()}")
    print(f"  Dequant first 20: {dequantized[0, :20].tolist()}")
    
    # Check elements that differ most
    max_diff_idx = diff.argmax().item()
    row = max_diff_idx // oracle_cpu.shape[1]
    col = max_diff_idx % oracle_cpu.shape[1]
    print(f"\n  Max diff at [{row}, {col}]:")
    print(f"    Oracle: {oracle_cpu[row, col].item():.6f}")
    print(f"    Dequant: {dequantized[row, col].item():.6f}")
    
    # Try alternative: maybe oracle weight is BF16 and we need to cast
    print(f"\n=== Trying BF16 comparison ===")
    dequant_bf16 = dequantized.bfloat16().float()
    cos_bf16 = torch.nn.functional.cosine_similarity(
        oracle_cpu.flatten().unsqueeze(0),
        dequant_bf16.flatten().unsqueeze(0)
    ).item()
    max_diff_bf16 = (oracle_cpu - dequant_bf16).abs().max().item()
    print(f"  BF16 cosine: {cos_bf16:.6f}")
    print(f"  BF16 max diff: {max_diff_bf16:.6f}")
    
    # Also check if the results correlate at all
    print(f"\n=== Detailed Analysis ===")
    print(f"  Oracle: std={oracle_cpu.std():.6f}, mean={oracle_cpu.mean():.6f}")
    print(f"  Dequant: std={dequantized.std():.6f}, mean={dequantized.mean():.6f}")
    
    # Sample a group and check
    print(f"\n  === First group (elements 0-15) row 0 ===")
    print(f"  Oracle: {oracle_cpu[0, :16].tolist()}")
    print(f"  Dequant: {dequantized[0, :16].tolist()}")
    
    # Check if there's a scale factor difference
    ratio = (oracle_cpu / (dequantized + 1e-10)).mean()
    print(f"\n  oracle/dequant mean ratio: {ratio:.6f}")
    
    # Try scaling: maybe global_scale is not a reciprocal?
    print(f"\n  === Trying without global_scale division ===")
    result_no_gs = np.zeros((m, n), dtype=np.float32)
    for row in range(m):
        for g in range(scale.shape[1]):
            s = scale_np[row, g]
            for i in range(group_size // 2):
                byte_val = packed_np[row, g * (group_size // 2) + i]
                lo = byte_val & np.uint8(0x0F)
                hi = (byte_val & np.uint8(0xF0)) >> 4
                out_idx = g * group_size + i * 2
                result_no_gs[row, out_idx] = fp4_lut[lo] * s
                result_no_gs[row, out_idx + 1] = fp4_lut[hi] * s
    
    dequant_no_gs = torch.from_numpy(result_no_gs)
    cos_no_gs = torch.nn.functional.cosine_similarity(
        oracle_cpu.flatten().unsqueeze(0),
        dequant_no_gs.flatten().unsqueeze(0)
    ).item()
    print(f"  Without GS: cosine={cos_no_gs:.6f}, std={dequant_no_gs.std():.6f}")
    
    # Try with multiple scale factor (e.g., use sqrt)
    print(f"\n  === Trying sqrt(global_scale) ===")
    gs_sqrt = np.sqrt(float(global_scale))
    result_gs_sqrt = np.zeros((m, n), dtype=np.float32)
    for row in range(m):
        for g in range(scale.shape[1]):
            s = scale_np[row, g] / gs_sqrt
            for i in range(group_size // 2):
                byte_val = packed_np[row, g * (group_size // 2) + i]
                lo = byte_val & np.uint8(0x0F)
                hi = (byte_val & np.uint8(0xF0)) >> 4
                out_idx = g * group_size + i * 2
                result_gs_sqrt[row, out_idx] = fp4_lut[lo] * s
                result_gs_sqrt[row, out_idx + 1] = fp4_lut[hi] * s
    
    dequant_gs_sqrt = torch.from_numpy(result_gs_sqrt)
    cos_gs_sqrt = torch.nn.functional.cosine_similarity(
        oracle_cpu.flatten().unsqueeze(0),
        dequant_gs_sqrt.flatten().unsqueeze(0)
    ).item()
    print(f"  sqrt GS: cosine={cos_gs_sqrt:.6f}, std={dequant_gs_sqrt.std():.6f}")

    # Try with multiply instead of divide
    result_mul = np.zeros((m, n), dtype=np.float32)
    for row in range(m):
        for g in range(scale.shape[1]):
            s = scale_np[row, g] * gs
            for i in range(group_size // 2):
                byte_val = packed_np[row, g * (group_size // 2) + i]
                lo = byte_val & np.uint8(0x0F)
                hi = (byte_val & np.uint8(0xF0)) >> 4
                out_idx = g * group_size + i * 2
                result_mul[row, out_idx] = fp4_lut[lo] * s
                result_mul[row, out_idx + 1] = fp4_lut[hi] * s
    
    dequant_mul = torch.from_numpy(result_mul)
    cos_mul = torch.nn.functional.cosine_similarity(
        oracle_cpu.flatten().unsqueeze(0),
        dequant_mul.flatten().unsqueeze(0)
    ).item()
    print(f"  Multiply GS: cosine={cos_mul:.6f}, std={dequant_mul.std():.6f}")

    # What if the FP4 LUT is different?
    print(f"\n  === Trying signed FP4 LUT ===")
    fp4_signed = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
    result_signed = np.zeros((m, n), dtype=np.float32)
    for row in range(m):
        for g in range(scale.shape[1]):
            s = scale_np[row, g] / gs
            for i in range(group_size // 2):
                byte_val = packed_np[row, g * (group_size // 2) + i]
                lo = byte_val & np.uint8(0x0F)
                hi = (byte_val & np.uint8(0xF0)) >> 4
                out_idx = g * group_size + i * 2
                result_signed[row, out_idx] = fp4_signed[lo] * s
                result_signed[row, out_idx + 1] = fp4_signed[hi] * s
    
    dequant_signed = torch.from_numpy(result_signed)
    cos_signed = torch.nn.functional.cosine_similarity(
        oracle_cpu.flatten().unsqueeze(0),
        dequant_signed.flatten().unsqueeze(0)
    ).item()
    print(f"  Signed LUT: cosine={cos_signed:.6f}, std={dequant_signed.std():.6f}")
    
    # --- Re-do analyses with proper scoping ---
    
    m, n = packed.shape
    n_full = n * 2
    group_size = n_full // scale.shape[1]
    scale_np2 = scale.cpu().float().numpy()
    gs = float(global_scale)
    packed_np2 = packed.cpu().numpy().astype(np.uint8)
    fp4_lut_full = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                             0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    
    # Without GS div
    r_no_gs = np.zeros((m, n_full), dtype=np.float32)
    for row in range(m):
        for g in range(scale.shape[1]):
            s = scale_np2[row, g]
            for i in range(group_size // 2):
                bv = packed_np2[row, g * (group_size // 2) + i]
                lo = bv & np.uint8(0x0F)
                hi = (bv & np.uint8(0xF0)) >> 4
                idx = g * group_size + i * 2
                r_no_gs[row, idx] = fp4_lut_full[lo] * s
                r_no_gs[row, idx+1] = fp4_lut_full[hi] * s
    d_no_gs = torch.from_numpy(r_no_gs)
    c_no_gs = torch.nn.functional.cosine_similarity(oracle_cpu.flatten().unsqueeze(0), d_no_gs.flatten().unsqueeze(0)).item()
    print(f"  Without GS: cos={c_no_gs:.6f}, std={d_no_gs.std():.6f}")
    
    # Multiply by GS instead of divide
    r_mul = np.zeros((m, n_full), dtype=np.float32)
    for row in range(m):
        for g in range(scale.shape[1]):
            s = scale_np2[row, g] * gs
            for i in range(group_size // 2):
                bv = packed_np2[row, g * (group_size // 2) + i]
                lo = bv & np.uint8(0x0F)
                hi = (bv & np.uint8(0xF0)) >> 4
                idx = g * group_size + i * 2
                r_mul[row, idx] = fp4_lut_full[lo] * s
                r_mul[row, idx+1] = fp4_lut_full[hi] * s
    d_mul = torch.from_numpy(r_mul)
    c_mul = torch.nn.functional.cosine_similarity(oracle_cpu.flatten().unsqueeze(0), d_mul.flatten().unsqueeze(0)).item()
    print(f"  Multiply GS: cos={c_mul:.6f}, std={d_mul.std():.6f}")
    
    # Signed LUT
    fp4_s = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
    r_s = np.zeros((m, n_full), dtype=np.float32)
    for row in range(m):
        for g in range(scale.shape[1]):
            s = scale_np2[row, g] / gs
            for i in range(group_size // 2):
                bv = packed_np2[row, g * (group_size // 2) + i]
                lo = bv & np.uint8(0x0F)
                hi = (bv & np.uint8(0xF0)) >> 4
                idx = g * group_size + i * 2
                r_s[row, idx] = fp4_s[lo] * s
                r_s[row, idx+1] = fp4_s[hi] * s
    d_s = torch.from_numpy(r_s)
    c_s = torch.nn.functional.cosine_similarity(oracle_cpu.flatten().unsqueeze(0), d_s.flatten().unsqueeze(0)).item()
    print(f"  Signed LUT: cos={c_s:.6f}, std={d_s.std():.6f}")


if __name__ == "__main__":
    main()
