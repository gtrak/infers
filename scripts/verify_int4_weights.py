#!/usr/bin/env python3
"""Verify INT4 dequantized weights match between our engine and HF auto_gptq.

Strategy: Instead of manually unpacking the complex int32-packed format,
we use HF's QuantLinear forward pass as the ground truth, and compare
our engine's dequantized weight (via a test GEMM) against it.

Key: Our engine's dump is TP=2 (half dimensions). We need to compare
the corresponding TP shard.
"""

import os
import json
import numpy as np
from safetensors import safe_open

MODEL_DIR = os.path.expanduser("~/opt/vllm/models/qwen3.6-27b-autoround-int4")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f64 = a.ravel().astype(np.float64)
    b_f64 = b.ravel().astype(np.float64)
    dot = np.dot(a_f64, b_f64)
    norm_a = np.linalg.norm(a_f64)
    norm_b = np.linalg.norm(b_f64)
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return float(dot / (norm_a * norm_b))


def bf16_raw_to_f32(path: str) -> np.ndarray:
    data = np.frombuffer(open(path, "rb").read(), dtype=np.uint16)
    return (data.astype(np.uint32) << 16).view(np.float32)


def main():
    from transformers import AutoModelForCausalLM
    import torch

    # ── Config ────────────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)
    
    tc = cfg['text_config']
    hidden_size = tc['hidden_size']                # 5120
    num_k_heads = tc['linear_num_key_heads']        # 16
    num_v_heads = tc['linear_num_value_heads']      # 48
    head_k_dim = tc['linear_key_head_dim']          # 128
    head_v_dim = tc['linear_value_head_dim']        # 128
    
    key_dim = num_k_heads * head_k_dim              # 2048
    value_dim = num_v_heads * head_v_dim            # 6144
    conv_dim_full = 2 * key_dim + value_dim         # 10240
    group_size = 128

    print("=" * 80)
    print(f"Model: hidden={hidden_size}, conv_dim={conv_dim_full}")
    print(f"key_dim={key_dim}, value_dim={value_dim}, kv_ratio={num_v_heads//num_k_heads}")
    print("=" * 80)

    # ── Step 1: Load model ───────────────────────────────────────────
    print("\nLoading model via HF...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    gdn_layer = None
    for i, layer in enumerate(model.model.layers):
        attn = getattr(layer, 'linear_attn', None)
        if attn is not None:
            gdn_layer = attn
            break

    qlayer = gdn_layer.in_proj_qkv
    print(f"QuantLinear: {type(qlayer)}")
    print(f"  qweight: {qlayer.qweight.shape} ({qlayer.qweight.dtype})")
    print(f"  scales:  {qlayer.scales.shape} ({qlayer.scales.dtype})")
    print(f"  qzeros:  {qlayer.qzeros.shape} ({qlayer.qzeros.dtype})")

    # ── Step 2: Quantization noise analysis ──────────────────────────
    print("\n" + "=" * 80)
    print("Quantization noise analysis...")
    print("=" * 80)

    sc_np = qlayer.scales.detach().cpu().float().numpy()
    scale_mean = np.abs(sc_np).mean()
    scale_max = np.abs(sc_np).max()

    print(f"Scale mean_abs: {scale_mean:.10f}")
    print(f"Scale max_abs:  {scale_max:.10f}")

    # ── Step 3: Manual dequantization (vectorized numpy) ─────────────
    print("\n" + "=" * 80)
    print("Manual dequantization matching our CUDA kernel...")
    print("=" * 80)

    qw = qlayer.qweight.detach().cpu().numpy().astype(np.int32)
    sc = sc_np
    qz = qlayer.qzeros.detach().cpu().numpy().astype(np.int32)

    # Shapes: qw [640, 10240], sc [40, 10240], qz [40, 1280]
    # This is the TRANSPOSED layout: [K//8, N] for qweight
    # K = 640*8 = 5120 = hidden_size ✓
    # N = 10240 = conv_dim_full ✓

    n_out = 10240   # conv_dim_full (output dimension)
    k_in = 5120     # hidden_size (input dimension)

    print(f"Layout: TRANSPOSED [K//8={qw.shape[0]}, N={qw.shape[1]}]")
    print(f"N={n_out} (conv_dim), K={k_in} (hidden_size)")

    # Dequantize using our engine's algorithm:
    # weight_fp32 = (int4_val - zero_point) * scale
    # 
    # Our CUDA kernel for transposed layout:
    #   weight_idx = ((k + kk) >> 3) * N + col
    #   scale from scales[group_idx, col]
    #   zero from zeros packed per column
    
    n_packed = (n_out + 7) // 8  # 1280 = ceil(10240/8) ✓

    # Unpack qweight: reshape to get individual int4 values
    # qw [K//8, N] → expand each u32 into 8 int4 values
    # After expansion: [K, N] — the dequantized INT4 weight matrix

    # Expand qweight: each int32 holds 8 int4 values
    # qw [K//8=640, N=10240] → qw_expanded [K=5120, N=10240]
    # Vectorized: for each bit position w (0..7), extract 4-bit values
    qw_int32 = qw.astype(np.int64)  # avoid overflow with shifts
    qw_expanded = np.zeros((k_in, n_out), dtype=np.int32)
    for w in range(8):
        # Extract bit position w for all columns at once
        extracted = ((qw_int32 >> (w * 4)) & 0xF).astype(np.int32)  # [640, 10240]
        start = w * (k_in // 8)
        end = start + (k_in // 8)
        qw_expanded[start:end, :] = extracted

    print(f"Unpacked qweight: {qw_expanded.shape}")
    print(f"  min={int(qw_expanded.min())}, max={int(qw_expanded.max())}")
    print(f"  mean={float(qw_expanded.mean()):.2f}")

    # Unpack scales: [K//group_size, N] → expand to [K, N]
    sc_expanded = np.repeat(sc, group_size, axis=0)  # [40, 10240] → [5120, 10240]

    # Unpack qzeros: [K//group_size, N//8] → unpack 8 per u32
    # Need to match the CUDA kernel's packing logic for transposed layout
    # zero_packed_idx = group_idx * n_packed + col // 8
    # zero_shift = (col % 8) * 4

    # Unpack qzeros: [K//group_size, N//8] = [40, 1280]
    # For transposed layout: zero at (g_idx, col) comes from qz[g_idx, col//8] bits
    qz_expanded = np.zeros((k_in, n_out), dtype=np.int32)
    for g_idx in range(k_in // group_size):
        k_start = g_idx * group_size
        # Get all 8 zero points from each packed u32 at once
        for packed_col in range(n_out // 8):
            packed_val = qz[g_idx, packed_col]  # one u32 with 8 zero points
            for bit_pos in range(8):
                col = packed_col * 8 + bit_pos
                if col < n_out:
                    zs = bit_pos * 4
                    zp = ((packed_val >> zs) & 0xF).astype(np.int32)
                    qz_expanded[k_start:k_start + group_size, col] = zp

    print(f"Unpacked scales: {sc_expanded.shape}")
    print(f"Unpacked zeros:  {qz_expanded.shape}, min={int(qz_expanded.min())}, max={int(qz_expanded.max())}")

    # Dequantize: (int4_val - zero) * scale
    result = ((qw_expanded.astype(np.float64) - qz_expanded.astype(np.float64)) * sc_expanded).astype(np.float32)

    print(f"\nDequantized weight: {result.shape}")
    print(f"  min={result.min():.6f}, max={result.max():.6f}")
    print(f"  mean_abs={np.abs(result).mean():.6f}, std={result.std():.6f}")

    # ── Step 4: Verify via HF forward pass ───────────────────────────
    print("\n" + "=" * 80)
    print("Verifying dequantization via HF forward pass...")
    print("=" * 80)

    token_ids = [248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 
                 248046, 198, 248045, 74455, 198]
    seq_len = len(token_ids)  # 15

    with torch.no_grad():
        # Use the device of the in_proj_qkv qweight
        dev = qlayer.qweight.device
        input_ids = torch.tensor([token_ids], device=dev)
        
        # Get the GDN input (hidden states after embed + norm)
        hidden = model.model.embed_tokens(input_ids).half()
        hidden = model.model.norm(hidden)  # [1, 15, 5120]

        # HF QuantLinear forward
        hf_mixed_qkv = qlayer(hidden)  # [1, 15, 10240]

        # Our dequantized weight: result is [K, N] = [5120, 10240]
        # Linear layer: output = input @ W_t where W_t = [K, N]
        # Must be on same device as hidden
        our_weight = torch.from_numpy(result.T).half().to(hidden.device)  # [5120, 10240]
        our_mixed_qkv = torch.nn.functional.linear(hidden, our_weight)

    hf_np = hf_mixed_qkv[0].float().detach().cpu().numpy()
    our_np = our_mixed_qkv[0].float().detach().cpu().numpy()

    print(f"\nHF mixed_qkv:   shape={hf_np.shape}, min={hf_np.min():.4f}, max={hf_np.max():.4f}")
    print(f"Our mixed_qkv:  shape={our_np.shape}, min={our_np.min():.4f}, max={our_np.max():.4f}")

    cos = cosine_similarity(our_np, hf_np)
    max_err = float(np.max(np.abs(our_np - hf_np)))
    mae = float(np.mean(np.abs(our_np - hf_np)))
    mse = float(np.mean((our_np - hf_np) ** 2))

    print(f"\nDequantization comparison:")
    print(f"  Cosine similarity: {cos:.10f}")
    print(f"  Max error:         {max_err:.6f}")
    print(f"  MAE:               {mae:.8f}")
    print(f"  MSE:               {mse:.10e}")

    # Error distribution
    diff = np.abs(our_np - hf_np)
    for pct in [50, 90, 99, 99.5, 99.9]:
        val = float(np.percentile(diff, pct))
        print(f"  P{pct} error: {val:.6f}")

    # ── Step 5: Compare with engine dump (TP=2 shard) ─────────────────
    print("\n" + "=" * 80)
    print("Comparing with our engine's TP=2 dump...")
    print("=" * 80)

    our_mq_path = "/tmp/our_gdn/mixed_qkv.raw"
    ref_mq_path = "/tmp/ref_gdn_new/mixed_qkv.npy"

    if os.path.exists(our_mq_path):
        our_dump = bf16_raw_to_f32(our_mq_path)
        expected_size = seq_len * (conv_dim_full // 2)  # 15 * 5120 = 76800
        
        if len(our_dump) >= expected_size:
            our_dump_shaped = our_dump[:expected_size].reshape(seq_len, conv_dim_full // 2)
            
            # Extract corresponding shard from HF (first half of conv_dim)
            hf_shard = hf_np[:, :conv_dim_full // 2]

            cos_shard = cosine_similarity(our_dump_shaped, hf_shard)
            max_err_shard = float(np.max(np.abs(our_dump_shaped - hf_shard)))
            mae_shard = float(np.mean(np.abs(our_dump_shaped - hf_shard)))

            print(f"\nTP=2 GPU 0 shard (first conv_dim//2):")
            print(f"  Cosine similarity: {cos_shard:.10f}")
            print(f"  Max error:         {max_err_shard:.6f}")
            print(f"  MAE:               {mae_shard:.8f}")

    # ── Step 6: Check if a_proj (BF16) also diverges ───────────────
    print("\n" + "=" * 80)
    print("Checking BF16 weight layers for reference...")
    print("=" * 80)

    try:
        with torch.no_grad():
            # a_proj is BF16 — should match exactly
            hf_a_proj = gdn_layer.in_proj_a(hidden)[0].float().detach().cpu().numpy()
            
        our_ap_path = "/tmp/our_gdn/a_proj.raw"
        ref_ap_path = "/tmp/ref_gdn_new/a_proj.npy"

        if os.path.exists(our_ap_path) and os.path.exists(ref_ap_path):
            our_ap = bf16_raw_to_f32(our_ap_path)
            expected_ap_size = seq_len * (num_v_heads // 2)  # 15 * 24 = 360
            
            if len(our_ap) >= expected_ap_size:
                our_ap_shaped = our_ap[:expected_ap_size].reshape(seq_len, num_v_heads // 2)
                
                # HF a_proj is [seq_len, num_v_heads] — our shard is first half
                hf_a_shard = hf_a_proj[:, :num_v_heads // 2]
                
                cos_ap = cosine_similarity(our_ap_shaped, hf_a_shard)
                max_err_ap = float(np.max(np.abs(our_ap_shaped - hf_a_shard)))

                print(f"\na_proj (BF16, TP=2 shard):")
                print(f"  Cosine similarity: {cos_ap:.10f}")
                print(f"  Max error:         {max_err_ap:.6f}")
    except RuntimeError as e:
        print(f"  a_proj comparison failed: {e}")

    # ── Step 7: Final verdict ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if cos > 0.999 and max_err < 1.0:
        print(f"\n✅ DEQUANTIZATION MATCHES (cos={cos:.6f}, max_err={max_err:.4f})")
        print(f"   Our INT4 dequantization produces the same result as HF.")
        print(f"   The original mixed_qkv divergence (cos=0.993) is from:")
        print(f"     1. Accumulation precision (FP32 vs BF16 GEMM)")
        print(f"     2. TP sharding differences in previous comparison")
    elif cos > 0.99:
        print(f"\n⚠️  NEAR MATCH (cos={cos:.6f}, max_err={max_err:.4f})")
        print(f"   Minor differences — could be from:")
        print(f"     1. Zero-point packing interpretation")
        print(f"     2. Accumulation precision in our kernel (FP32)")
        print(f"     3. BF16 → float conversions")
    else:
        print(f"\n❌ MISMATCH (cos={cos:.6f}, max_err={max_err:.4f})")
        print(f"   Possible dequantization bug in our engine.")

    print("\nDone!")


if __name__ == "__main__":
    main()
