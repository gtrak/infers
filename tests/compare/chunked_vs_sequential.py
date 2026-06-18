#!/usr/bin/env python3
"""
Compare sequential vs chunked GDN attention algorithms against HF reference.

Loads oracle intermediates from /tmp/ref_gdn_new/ and model weights for layer 0,
runs both algorithms with identical inputs, and reports numerical differences.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# ── Load data ────────────────────────────────────────────────────────────────

ORACLE_DIR = "/tmp/ref_gdn_new/"
MODEL_PATH = "/home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4/"

query_expanded = torch.from_numpy(np.load(f"{ORACLE_DIR}query_expanded.npy")).float()  # [S, H_v, K]
key_expanded   = torch.from_numpy(np.load(f"{ORACLE_DIR}key_expanded.npy")).float()    # [S, H_v, K]
value_raw      = torch.from_numpy(np.load(f"{ORACLE_DIR}value.npy")).float()           # [S, H_v*V]
a_proj         = torch.from_numpy(np.load(f"{ORACLE_DIR}a_proj.npy")).float()          # [S, H_v]
b_proj         = torch.from_numpy(np.load(f"{ORACLE_DIR}b_proj.npy")).float()          # [S, H_v]
core_attn_ref  = torch.from_numpy(np.load(f"{ORACLE_DIR}core_attn_out.npy")).float()   # [S, H_v, V]

# Reshape value from flat to head structure
seq_len, num_v_heads, head_k_dim = query_expanded.shape
head_v_dim = core_attn_ref.shape[-1]
value = value_raw.reshape(seq_len, num_v_heads, head_v_dim)  # [S, H_v, V]

# Load A_log and dt_bias for layer 0 (BF16 quantized → cast to float32)
layer0_shard = load_file(f"{MODEL_PATH}model-00001-of-00010.safetensors")
A_log   = layer0_shard["model.language_model.layers.0.linear_attn.A_log"].float()     # [H_v]
dt_bias = layer0_shard["model.language_model.layers.0.linear_attn.dt_bias"].float()    # [H_v]

print("=" * 80)
print("GDN Attention: Sequential vs Chunked Numerical Comparison")
print("=" * 80)
print(f"  seq_len        = {seq_len}")
print(f"  num_v_heads    = {num_v_heads}")
print(f"  head_k_dim     = {head_k_dim}")
print(f"  head_v_dim     = {head_v_dim}")
print(f"  A_log range    = [{A_log.min():.6f}, {A_log.max():.6f}]")
print(f"  dt_bias range  = [{dt_bias.min():.6f}, {dt_bias.max():.6f}]")
print("=" * 80)

# ── Sequential GDN (engine kernel equivalent) ───────────────────────────────

def sequential_gated_delta_rule(
    query, key, value, a_proj, b_proj, A_log, dt_bias
):
    """
    Per-token sequential recurrence matching gdn_recurrent_step.cu.

    All inputs in float32.  query/key are L2-normalised; query already scaled
    by 1/sqrt(K) at this point but we re-apply the normalisation+scaling to
    be safe and symmetric with the chunked version.
    """
    g    = -A_log.exp() * F.softplus(a_proj + dt_bias)   # [S, H]
    beta = b_proj.sigmoid()                               # [S, H]

    K = head_k_dim
    query_scaled = F.normalize(query.float(), dim=-1) / math.sqrt(K)
    key_normed   = F.normalize(key.float(), dim=-1)

    S = torch.zeros(num_v_heads, head_k_dim, head_v_dim, dtype=torch.float32)
    output = torch.zeros(seq_len, num_v_heads, head_v_dim, dtype=torch.float32)

    for t in range(seq_len):
        g_t     = g[t]      # [H]
        beta_t  = beta[t]   # [H]
        decay   = g_t.exp() # [H]

        for h in range(num_v_heads):
            S[h] *= decay[h]
            kv_mem = S[h] @ key_normed[t, h]             # [V]
            delta  = beta_t[h] * (value[t, h].float() - kv_mem)
            S[h]  += torch.outer(key_normed[t, h], delta)
            output[t, h] = query_scaled[t, h] @ S[h]

    return output, S

# ── Chunked GDN (HF reference equivalent) ───────────────────────────────────

def chunked_gated_delta_rule(
    query, key, value, a_proj, b_proj, A_log, dt_bias, chunk_size=64
):
    """
    Chunked parallel GDN algorithm matching HF's torch_chunk_gated_delta_rule.
    """
    g    = -A_log.exp() * F.softplus(a_proj + dt_bias)   # [S, H]
    beta = b_proj.sigmoid()                               # [S, H]

    K = head_k_dim
    query_scaled = F.normalize(query.float(), dim=-1) / math.sqrt(K)
    key_normed   = F.normalize(key.float(), dim=-1)

    # Transpose to [H, S, ...] layout
    q      = query_scaled.transpose(0, 1).contiguous().float()   # [H, S, K]
    k      = key_normed.transpose(0, 1).contiguous().float()     # [H, S, K]
    v      = value.transpose(0, 1).contiguous().float()          # [H, S, V]
    g_t    = g.transpose(0, 1).contiguous().float()              # [H, S]
    beta_t = beta.transpose(0, 1).contiguous().float()           # [H, S]

    # Pad to chunk_size multiple
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size > 0:
        q      = F.pad(q, (0, 0, 0, pad_size))
        k      = F.pad(k, (0, 0, 0, pad_size))
        v      = F.pad(v, (0, 0, 0, pad_size))
        beta_t = F.pad(beta_t, (0, pad_size))
        g_t    = F.pad(g_t, (0, pad_size))

    total_len   = seq_len + pad_size
    num_chunks  = total_len // chunk_size

    # Reshape into chunks [H, N, C, ...]
    q      = q.reshape(num_v_heads, num_chunks, chunk_size, K)
    k      = k.reshape(num_v_heads, num_chunks, chunk_size, K)
    v      = v.reshape(num_v_heads, num_chunks, chunk_size, head_v_dim)
    beta_t = beta_t.reshape(num_v_heads, num_chunks, chunk_size)
    g_t    = g_t.reshape(num_v_heads, num_chunks, chunk_size)

    # Chunk cumulative decay
    g_cumsum = g_t.cumsum(dim=-1)  # [H, N, C]

    # Lower-triangular decay mask (strict: zero on diagonal and above)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    g_diff = (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).tril().exp()
    g_diff = g_diff.tril()

    # Weighted key/value
    k_beta = k * beta_t.unsqueeze(-1)   # [H, N, C, K]
    v_beta = v * beta_t.unsqueeze(-1)   # [H, N, C, V]

    # --- Intra-chunk attention matrix with forward substitution ---
    attn = -(k_beta @ k.transpose(-1, -2)) * g_diff
    attn = attn.masked_fill(mask, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)

    # Intra-chunk corrected values and cumulative decay keys
    v_new      = attn @ v_beta                                       # [H, N, C, V]
    k_cumdecay = attn @ (k_beta * g_cumsum.unsqueeze(-1).exp())       # [H, N, C, K]

    # --- Inter-chunk recurrence ---
    S = torch.zeros(num_v_heads, K, head_v_dim, dtype=torch.float32)
    core_attn_out = torch.zeros(
        num_v_heads, num_chunks, chunk_size, head_v_dim, dtype=torch.float32
    )

    intra_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)

    for i in range(num_chunks):
        q_i  = q[:, i, :]                       # [H, C, K]
        k_i  = k[:, i, :]                       # [H, C, K]
        v_i  = v_new[:, i, :]                   # [H, C, V]

        # Intra-chunk q-k attention (strictly upper triangular)
        attn_qk = (q_i @ k_i.transpose(-1, -2)) * g_diff[:, i, :, :]
        attn_qk = attn_qk.masked_fill(intra_mask, 0)

        # Inter-chunk contribution via accumulated state S
        v_prime       = k_cumdecay[:, i, :] @ S                          # [H, C, V]
        v_new_correct = v_i - v_prime                                    # [H, C, V]

        attn_inter    = (q_i * g_cumsum[:, i, :, None].exp()) @ S     # [H, C, V]
        core_attn_out[:, i, :] = attn_inter + attn_qk @ v_new_correct
        # Update state for next chunk
        exp_diff = (g_cumsum[:, i, -1, None] - g_cumsum[:, i]).exp()  # [H, C]
        S = (S * g_cumsum[:, i, -1, None, None].exp()
             + (k_i * exp_diff[..., None]).transpose(-1, -2) @ v_new_correct)

    # Un-pad and transpose back to [S, H, V]
    core_attn_out = core_attn_out.reshape(num_v_heads, total_len, head_v_dim)
    core_attn_out = core_attn_out[:, :seq_len]
    core_attn_out = core_attn_out.transpose(0, 1).contiguous()

    return core_attn_out, S

# ── Run both algorithms ─────────────────────────────────────────────────────

print("\n[1/3] Running sequential GDN ...")
out_seq, _ = sequential_gated_delta_rule(query_expanded, key_expanded, value, a_proj, b_proj, A_log, dt_bias)

print("[2/3] Running chunked GDN ...")
out_chunk, _ = chunked_gated_delta_rule(query_expanded, key_expanded, value, a_proj, b_proj, A_log, dt_bias)

# ── Comparison against HF reference ─────────────────────────────────────────

print("[3/3] Computing metrics ...")
print()

def cosine_sim(a, b):
    """Cosine similarity between two 1-D vectors (per-token per-head)."""
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))

# Per-token, all-heads-averaged metrics
cos_seq_all = cosine_sim(out_seq, core_attn_ref).mean(dim=1).cpu().numpy()     # [S]
cos_chunk_all = cosine_sim(out_chunk, core_attn_ref).mean(dim=1).cpu().numpy() # [S]

ratio_seq_all = out_seq.norm(dim=-1).mean(dim=1) / core_attn_ref.norm(dim=-1).mean(dim=1)
ratio_seq_all = ratio_seq_all.cpu().numpy()                                     # [S]

ratio_chunk_all = out_chunk.norm(dim=-1).mean(dim=1) / core_attn_ref.norm(dim=-1).mean(dim=1)
ratio_chunk_all = ratio_chunk_all.cpu().numpy()                                 # [S]

# Also compute direct L2 norm between the outputs themselves
# Per-token, per-head L2: ||out[t,h] - ref[t,h]||_2
l2_seq     = (out_seq - core_attn_ref).norm(dim=-1)        # [S, H]
l2_chunk   = (out_chunk - core_attn_ref).norm(dim=-1)       # [S, H]

# Average across heads per token
l2_seq_h_avg     = l2_seq.mean(dim=1).cpu().numpy()         # [S]
l2_chunk_h_avg   = l2_chunk.mean(dim=1).cpu().numpy()       # [S]

# ── Print table ──────────────────────────────────────────────────────────────

print("=" * 80)
print(f"{'Token':>5} | {'Cos Sim Seq':>11} | {'Cos Sim Chk':>11} | "
      f"{'Ratio Seq':>11} | {'Ratio Chk':>11} | {'L2 Seq':>10} | {'L2 Chk':>10}")
print("-" * 80)

for t in range(seq_len):
    print(f"  {t:3d} | {cos_seq_all[t]:.6f} | {cos_chunk_all[t]:.6f} | "
          f"{ratio_seq_all[t]:.6f} | {ratio_chunk_all[t]:.6f} | "
          f"{l2_seq_h_avg[t]:.6f} | {l2_chunk_h_avg[t]:.6f}")

print("=" * 80)

# ── Summary statistics ───────────────────────────────────────────────────────

print()
print("Summary Statistics:")
print("-" * 80)

for label, cos_arr, ratio_arr, l2_arr in [
    ("Sequential", cos_seq_all, ratio_seq_all, l2_seq_h_avg),
    ("Chunked", cos_chunk_all, ratio_chunk_all, l2_chunk_h_avg),
]:
    print(f"\n  {label}:")
    print(f"    Mean Cosine Similarity : {cos_arr.mean():.8f}")
    print(f"    Min  Cosine Similarity : {cos_arr.min():.8f}")
    print(f"    Max  Cosine Similarity : {cos_arr.max():.8f}")
    print(f"    Mean Norm Ratio        : {ratio_arr.mean():.8f}")
    print(f"    Max |Ratio - 1|       : {abs(ratio_arr - 1).max():.8f}")
    print(f"    Mean L2 (per head)     : {l2_arr.mean():.6f}")
    print(f"    Max  L2 (per head)     : {l2_arr.max():.6f}")

# ── Direct comparison: sequential vs chunked ─────────────────────────────────

cos_seq_vs_chunk = cosine_sim(out_seq, out_chunk).mean(dim=1).cpu().numpy()
print(f"\n  Cosine similarity (Sequential vs Chunked):")
print(f"    Mean : {cos_seq_vs_chunk.mean():.8f}")
print(f"    Min  : {cos_seq_vs_chunk.min():.8f}")

# ── Conclusion ───────────────────────────────────────────────────────────────

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)

ref_cos_sequential = cos_seq_all.mean()
ref_cos_chunked    = cos_chunk_all.mean()

if ref_cos_chunked > ref_cos_sequential:
    print(f"  Chunked GDN is CLOSER to HF reference (mean cos {ref_cos_chunked:.6f} vs {ref_cos_sequential:.6f}).")
elif ref_cos_sequential > ref_cos_chunked:
    print(f"  Sequential GDN is CLOSER to HF reference (mean cos {ref_cos_sequential:.6f} vs {ref_cos_chunked:.6f}).")
else:
    print(f"  Both are essentially identical (cos = {ref_cos_chunked:.6f}).")

diff = abs(ref_cos_chunked - ref_cos_sequential)
if diff > 1e-3:
    print("  The difference is SIGNIFICANT — switching algorithms would matter.")
elif diff > 1e-6:
    print("  The difference is small but measurable.")
else:
    print("  The difference is negligible in FP32.")
