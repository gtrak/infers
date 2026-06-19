#!/usr/bin/env python3
"""
Debug the chunked GDN CUDA kernel by simulating it step-by-step in Python
and comparing each phase against both the HF reference and the CUDA output.

Focuses on head=0, chunk=0 (the only chunk since seq_len=15 < chunk_size=64).
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import json
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DUMP_DIR = "/tmp/dump_chunked/layer_0/prefill/"
MODEL_PATH = "/home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4/"
ORACLE_DIR = "/tmp/ref_gdn_new/"

# ── Helper: load tensor from engine dump ──────────────────────────────────────

def load_dump(name, gpu=0):
    """Load a bf16 tensor from the engine dump."""
    meta_path = f"{DUMP_DIR}gdn.{name}_gpu{gpu}.meta"
    raw_path  = f"{DUMP_DIR}gdn.{name}_gpu{gpu}.raw"
    meta = json.load(open(meta_path))
    raw = np.fromfile(raw_path, dtype=np.float16)
    tensor = torch.from_numpy(raw.reshape(meta["shape"])).bfloat16()
    return tensor

# ── Load engine dumps (GPU0: 24 heads) ───────────────────────────────────────

query_expanded = load_dump("query_expanded", gpu=0).float()  # [S, H*K] → reshape later
key_expanded   = load_dump("key_expanded", gpu=0).float()
value          = load_dump("value", gpu=0).float()
a_proj         = load_dump("a_proj", gpu=0).float()
b_proj         = load_dump("b_proj", gpu=0).float()
cuda_output    = load_dump("core_attn_out", gpu=0).float()   # CUDA kernel output

# ── Load model weights (A_log, dt_bias) for layer 0 ─────────────────────────

layer0_shard = load_file(f"{MODEL_PATH}model-00001-of-00010.safetensors")
A_log_full   = layer0_shard["model.language_model.layers.0.linear_attn.A_log"].float()     # [48]
dt_bias_full = layer0_shard["model.language_model.layers.0.linear_attn.dt_bias"].float()    # [48]

# GPU0 gets first 24 heads (TP=2)
A_log   = A_log_full[:24]
dt_bias = dt_bias_full[:24]

# ── Load HF reference output ────────────────────────────────────────────────

if os.path.exists(ORACLE_DIR):
    core_attn_ref = torch.from_numpy(np.load(f"{ORACLE_DIR}core_attn_out.npy")).float()  # [S, H_full, V]
    # Only GPU0 heads (first 24 of the full 48)
    core_attn_ref_gpu0 = core_attn_ref[:, :24, :]  # [S, 24, V]
else:
    core_attn_ref_gpu0 = None

# ── Shape parameters ────────────────────────────────────────────────────────

seq_len = 15
num_v_heads = 24
head_k_dim = 128
head_v_dim = 128
chunk_size = 64

# Reshape to [S, H, K/V]
Q = query_expanded.reshape(seq_len, num_v_heads, head_k_dim)  # [15, 24, 128]
K_tensor = key_expanded.reshape(seq_len, num_v_heads, head_k_dim)
V = value.reshape(seq_len, num_v_heads, head_v_dim)

# ── Print config ────────────────────────────────────────────────────────────

print("=" * 80)
print("Chunked GDN CUDA Kernel Debug — Step-by-Step Comparison")
print("=" * 80)
print(f"  seq_len      = {seq_len}")
print(f"  num_v_heads  = {num_v_heads}  (GPU0 of TP=2)")
print(f"  head_k_dim   = {head_k_dim}")
print(f"  head_v_dim   = {head_v_dim}")
print(f"  chunk_size   = {chunk_size}")
print(f"  num_chunks   = {(seq_len + chunk_size - 1) // chunk_size}")
print()

# ── Target head ─────────────────────────────────────────────────────────────

HEAD = 0
print(f"Analyzing HEAD={HEAD}\n")

# ═════════════════════════════════════════════════════════════════════════════
# REFERENCE: Run the HF chunked algorithm (from chunked_vs_sequential.py)
# ═════════════════════════════════════════════════════════════════════════════

def hf_chunked_gdn(Q, K_tensor, V, a_proj, b_proj, A_log, dt_bias, chunk_size=64):
    """HF reference chunked GDN for GPU0 tensors."""
    S = Q.shape[0]
    H = Q.shape[1]
    g    = -A_log.exp() * F.softplus(a_proj + dt_bias)   # [S, H]
    beta = b_proj.sigmoid()                               # [S, H]

    query_scaled = F.normalize(Q.float(), dim=-1) / math.sqrt(head_k_dim)
    key_normed   = F.normalize(K_tensor.float(), dim=-1)

    q      = query_scaled.transpose(0, 1).contiguous().float()   # [H, S, K]
    k      = key_normed.transpose(0, 1).contiguous().float()     # [H, S, K]
    v      = V.transpose(0, 1).contiguous().float()              # [H, S, V]
    g_t    = g.transpose(0, 1).contiguous().float()              # [H, S]
    beta_t = beta.transpose(0, 1).contiguous().float()           # [H, S]

    pad_size = (chunk_size - S % chunk_size) % chunk_size
    if pad_size > 0:
        q      = F.pad(q, (0, 0, 0, pad_size))
        k      = F.pad(k, (0, 0, 0, pad_size))
        v      = F.pad(v, (0, 0, 0, pad_size))
        beta_t = F.pad(beta_t, (0, pad_size))
        g_t    = F.pad(g_t, (0, pad_size))

    total_len   = S + pad_size
    num_chunks  = total_len // chunk_size

    q      = q.reshape(H, num_chunks, chunk_size, head_k_dim)
    k      = k.reshape(H, num_chunks, chunk_size, head_k_dim)
    v      = v.reshape(H, num_chunks, chunk_size, head_v_dim)
    beta_t = beta_t.reshape(H, num_chunks, chunk_size)
    g_t    = g_t.reshape(H, num_chunks, chunk_size)

    g_cumsum = g_t.cumsum(dim=-1)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    g_diff = (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).tril().exp()
    g_diff = g_diff.tril()

    k_beta = k * beta_t.unsqueeze(-1)

    attn = -(k_beta @ k.transpose(-1, -2)) * g_diff
    attn = attn.masked_fill(mask, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=q.device)

    v_new      = attn @ (v * beta_t.unsqueeze(-1))
    k_cumdecay = attn @ (k_beta * g_cumsum.unsqueeze(-1).exp())

    S_state = torch.zeros(H, head_k_dim, head_v_dim, dtype=torch.float32)
    core_attn_out = torch.zeros(
        H, num_chunks, chunk_size, head_v_dim, dtype=torch.float32
    )

    intra_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)

    for i in range(num_chunks):
        q_i  = q[:, i, :]
        k_i  = k[:, i, :]

        attn_qk = (q_i @ k_i.transpose(-1, -2)) * g_diff[:, i, :, :]
        attn_qk = attn_qk.masked_fill(intra_mask, 0)

        v_i             = v_new[:, i, :]
        v_prime         = k_cumdecay[:, i, :] @ S_state
        v_new_correct   = v_i - v_prime

        attn_inter      = (q_i * g_cumsum[:, i, :, None].exp()) @ S_state
        core_attn_out[:, i, :] = attn_inter + attn_qk @ v_new_correct

        exp_diff = (g_cumsum[:, i, -1, None] - g_cumsum[:, i]).exp()
        S_state = (S_state * g_cumsum[:, i, -1, None, None].exp()
                  + (k_i * exp_diff[..., None]).transpose(-1, -2) @ v_new_correct)

    core_attn_out = core_attn_out.reshape(H, total_len, head_v_dim)
    core_attn_out = core_attn_out[:, :S]
    core_attn_out = core_attn_out.transpose(0, 1).contiguous()

    return core_attn_out, g_cumsum[HEAD, 0], attn[HEAD, 0], g_diff[HEAD, 0]

print("-" * 80)
print("Phase: Running HF reference chunked GDN ...")
print("-" * 80)
ref_output, ref_g_cs, ref_attn, ref_g_diff = hf_chunked_gdn(
    Q, K_tensor, V, a_proj, b_proj, A_log, dt_bias, chunk_size
)

# ═════════════════════════════════════════════════════════════════════════════
# CUDA KERNEL SIMULATION — step by step for head=HEAD, chunk=0
# ═════════════════════════════════════════════════════════════════════════════

C = chunk_size
K = head_k_dim
V_dim = head_v_dim
H = num_v_heads

print("=" * 80)
print("CUDA Kernel Simulation (head={}, chunk=0)")
print("=" * 80)
print()

# ─── PHASE 1: g, beta, g_cumsum, L2-norm key, k_beta ────────────────────────

print("-" * 70)
print("PHASE 1: g, beta, g_cumsum, L2-normalize key, k_beta")
print("-" * 70)

# CUDA: decay_rate = expf(A_log[h])
decay_rate_cuda = math.exp(A_log[HEAD].item())
# HF reference does the same: -A_log.exp() * softplus(...)
decay_rate_hf   = A_log[HEAD].exp().item()

print(f"  A_log[{HEAD}]        = {A_log[HEAD]:.8f}")
print(f"  decay_rate (CUDA)    = exp(A_log[{HEAD}]) = {decay_rate_cuda:.8f}")
print(f"  decay_rate (HF)      = A_log[{HEAD}].exp() = {decay_rate_hf:.8f}")
print(f"  Match: {'YES' if abs(decay_rate_cuda - decay_rate_hf) < 1e-6 else 'NO'}")
print()

# g computation — CUDA style (with softplus thresholds)
g_cuda = torch.zeros(seq_len, dtype=torch.float32)
for i in range(seq_len):
    a_val = a_proj[i, HEAD].item()
    sp_val = a_val + dt_bias[HEAD].item()
    if sp_val > 20.0:
        softplus = sp_val
    elif sp_val < -20.0:
        softplus = 0.0
    else:
        softplus = math.log(1.0 + math.exp(sp_val))
    g_cuda[i] = -decay_rate_cuda * softplus

# HF reference g (no thresholding in PyTorch softplus)
g_hf = (-A_log[HEAD].exp() * F.softplus(a_proj[:, HEAD] + dt_bias[HEAD]))

print(f"  --- g values (first 5 tokens) ---")
for i in range(min(5, seq_len)):
    print(f"    g[{i}] CUDA={g_cuda[i]:12.8f}  HF={g_hf[i]:12.8f}  "
          f"diff={abs(g_cuda[i] - g_hf[i]):.2e}")

print(f"  --- g statistics ---")
print(f"    |g_cuda - g_hf| max = {abs(g_cuda - g_hf).max():.2e}")
print(f"    |g_cuda - g_hf| mean= {abs(g_cuda - g_hf).mean():.2e}")

# beta computation
beta_cuda = torch.zeros(seq_len, dtype=torch.float32)
for i in range(seq_len):
    b_val = b_proj[i, HEAD].item()
    beta_cuda[i] = 1.0 / (1.0 + math.exp(-b_val))

beta_hf = b_proj[:, HEAD].sigmoid()

print(f"  --- beta values (first 5 tokens) ---")
for i in range(min(5, seq_len)):
    print(f"    beta[{i}] CUDA={beta_cuda[i]:12.8f}  HF={beta_hf[i]:12.8f}  "
          f"diff={abs(beta_cuda[i] - beta_hf[i]):.2e}")

print(f"  |beta_cuda - beta_hf| max = {abs(beta_cuda - beta_hf).max():.2e}")
print()

# g_cumsum — CUDA style (sequential running sum, with zeroing for padded positions)
g_cs_cuda = torch.zeros(C, dtype=torch.float32)
for i in range(seq_len):
    g_cs_cuda[i] = g_cuda[i]
# Padded positions remain 0 (actual_len=seq_len since seq_len < C)

running = 0.0
g_cs_cumsum_cuda = torch.zeros(C, dtype=torch.float32)
for i in range(C):
    running += g_cs_cuda[i]
    g_cs_cumsum_cuda[i] = running

# HF reference (cumsum over chunk, with padding zeros)
g_hf_expanded = torch.zeros(C, dtype=torch.float32)
g_hf_expanded[:seq_len] = g_hf
g_cs_cumsum_hf = g_hf_expanded.cumsum(dim=0)

print(f"  --- g_cumsum (first 5 positions) ---")
for i in range(min(5, seq_len)):
    print(f"    g_cs[{i}] CUDA={g_cs_cumsum_cuda[i]:12.8f}  HF={g_cs_cumsum_hf[i]:12.8f}  "
          f"diff={abs(g_cs_cumsum_cuda[i] - g_cs_cumsum_hf[i]):.2e}")

print(f"  |g_cs_cuda - g_cs_hf| max = {abs(g_cs_cumsum_cuda[:seq_len] - g_cs_cumsum_hf[:seq_len]).max():.2e}")
print()

# L2-normalize key — CUDA style (per-row normalization)
key_raw = K_tensor[:, HEAD, :].clone()  # [S, K] fp32
key_norm_cuda = torch.zeros_like(key_raw)
for i in range(seq_len):
    sum_sq = (key_raw[i] ** 2).sum().item()
    rcp_norm = 1.0 / math.sqrt(sum_sq + 1e-6)
    key_norm_cuda[i] = key_raw[i] * rcp_norm

# HF reference
key_norm_hf = F.normalize(K_tensor[:, HEAD, :].float(), dim=-1)

print(f"  --- L2 norm of key (first 5 positions) ---")
for i in range(min(5, seq_len)):
    cuda_norm = key_norm_cuda[i].norm().item()
    hf_norm   = key_norm_hf[i].norm().item()
    print(f"    ||key[{i}]||_2 CUDA={cuda_norm:.8f}  HF={hf_norm:.8f}")

print(f"  |k_norm_cuda - k_norm_hf| max = {abs(key_norm_cuda - key_norm_hf).max():.2e}")
print()

# k_beta = k_normed * beta — CUDA style (only for valid positions)
k_beta_cuda = torch.zeros_like(key_norm_cuda)
for i in range(seq_len):
    k_beta_cuda[i] = key_norm_cuda[i] * beta_cuda[i]

k_beta_hf = key_norm_hf * beta_hf.unsqueeze(1)  # [S, K]

print(f"  --- |k_beta_cuda - k_beta_hf| max = {abs(k_beta_cuda - k_beta_hf).max():.2e}")
print()

# ─── PHASE 2: attn = -(k_beta @ k_normed^T) * decay_mask ────────────────────

print("-" * 70)
print("PHASE 2: Intra-chunk GEMM — attn = -(k_beta @ k_normed^T) * decay_mask")
print("-" * 70)

# CUDA style: compute full GEMM then apply mask
attn_raw_cuda = key_norm_cuda[:C, :] @ key_norm_cuda[:C, :].T  # [C, C] — wait, this is wrong!
# Actually: attn = -(k_beta @ k_normed^T) so it's k_beta @ k_normed^T

k_norm_full = torch.zeros(C, K, dtype=torch.float32)
k_norm_full[:seq_len] = key_norm_cuda
k_beta_full = torch.zeros(C, K, dtype=torch.float32)
k_beta_full[:seq_len] = k_beta_cuda

attn_gemm_cuda = -(k_beta_full @ k_norm_full.T)  # [C, C]

# Decay mask: exp(g_cs[row] - g_cs[col]) for row >= col, else 0
decay_mask_cuda = torch.zeros(C, C, dtype=torch.float32)
for row in range(C):
    for col in range(C):
        if row >= col:
            decay_mask_cuda[row, col] = math.exp(g_cs_cumsum_cuda[row].item() - g_cs_cumsum_cuda[col].item())

attn_cuda = attn_gemm_cuda * decay_mask_cuda

# HF reference (already computed above)
print(f"  --- attn matrix stats ---")
print(f"    CUDA: min={attn_cuda[:seq_len, :seq_len].min():.8f}, max={attn_cuda[:seq_len, :seq_len].max():.8f}")
print(f"    HF  : min={ref_attn[:seq_len, :seq_len].min():.8f}, max={ref_attn[:seq_len, :seq_len].max():.8f}")

# Compare specific entries
print(f"  --- attn[3,:4] comparison ---")
for j in range(min(4, seq_len)):
    print(f"    [{3},{j}] CUDA={attn_cuda[3,j]:12.8f}  HF={ref_attn[3,j]:12.8f}  "
          f"diff={abs(attn_cuda[3,j] - ref_attn[3,j]):.2e}")

print(f"  |attn_cuda - attn_hf| max (valid region) = {abs(attn_cuda[:seq_len,:seq_len] - ref_attn[:seq_len,:seq_len]).max():.2e}")
print(f"  |attn_cuda - attn_hf| mean (valid region)= {abs(attn_cuda[:seq_len,:seq_len] - ref_attn[:seq_len,:seq_len]).mean():.2e}")

# Also compare g_diff
print(f"  --- g_diff comparison ---")
g_diff_cuda = torch.zeros(C, C, dtype=torch.float32)
for row in range(C):
    for col in range(row + 1):
        g_diff_cuda[row, col] = math.exp(g_cs_cumsum_cuda[row].item() - g_cs_cumsum_cuda[col].item())

print(f"  |g_diff_cuda - g_diff_hf| max (valid region) = {abs(g_diff_cuda[:seq_len,:seq_len] - ref_g_diff[:seq_len,:seq_len]).max():.2e}")
print()

# ─── PHASE 3: Forward substitution + identity addition ──────────────────────

print("-" * 70)
print("PHASE 3: Forward substitution + identity")
print("-" * 70)

attn_cuda_copy = attn_cuda[:seq_len, :seq_len].clone()
actual_len = seq_len

# CUDA forward substitution (exactly as in kernel)
for i in range(1, actual_len):
    row_buf = attn_cuda_copy[i, :i].clone()
    for j in range(i):
        accum = 0.0
        for m in range(i):
            accum += row_buf[m].item() * attn_cuda_copy[m, j].item()
        attn_cuda_copy[i, j] = row_buf[j] + accum

# Add identity
for i in range(actual_len):
    attn_cuda_copy[i, i] += 1.0

attn_cuda_final = attn_cuda_copy

print(f"  --- attn after forward sub (first 4x4) ---")
for r in range(min(4, seq_len)):
    vals_cuda = [f"{attn_cuda_final[r,c]:8.4f}" for c in range(min(4, seq_len))]
    vals_hf   = [f"{ref_attn[r,c]:8.4f}" for c in range(min(4, seq_len))]
    print(f"    row {r}: CUDA={'  '.join(vals_cuda)}")
    print(f"           HF  ={'  '.join(vals_hf)}")

diff_attn = abs(attn_cuda_final - ref_attn[:seq_len, :seq_len])
print(f"  |attn_cuda - attn_hf| max (forward sub)   = {diff_attn.max():.2e}")
print(f"  |attn_cuda - attn_hf| mean (forward sub)  = {diff_attn.mean():.2e}")
print()

# ─── PHASE 4: Output computation ────────────────────────────────────────────

print("-" * 70)
print("PHASE 4: Output computation")
print("-" * 70)

# State S is zero-initialized for chunk 0 (since there are no previous chunks)
S = torch.zeros(K, V_dim, dtype=torch.float32)

# Per-row query L2 normalization (CUDA style)
q_rcp_sqrt_k = 1.0 / math.sqrt(K)

# Pre-compute reused intermediates
v_beta_all = V[:seq_len, HEAD, :] * beta_cuda[:seq_len].unsqueeze(1)  # [S, V]
k_beta_full_arr = torch.zeros(seq_len, K, dtype=torch.float32)
k_beta_full_arr[:seq_len] = k_beta_cuda
g_cs_exp_arr = torch.exp(g_cs_cumsum_cuda[:seq_len])  # [S]

output_cuda_sim = torch.zeros(seq_len, V_dim, dtype=torch.float32)

# Also track per-row contributions for debugging
row_contributions = {}  # {row: output_from_qk}

for row in range(seq_len):
    # Load and L2-normalize query for this row
    q_raw = Q[row, HEAD, :].clone()  # [K] fp32
    q_l2_sq = (q_raw ** 2).sum().item()
    q_rcp = (1.0 / math.sqrt(q_l2_sq + 1e-6)) * q_rcp_sqrt_k

    exp_g_row = math.exp(g_cs_cumsum_cuda[row].item())

    # attn_inter = (q_scaled @ S)[row] — but S=0 for chunk 0
    q_scl = q_raw * q_rcp * exp_g_row
    attn_inter_vec = q_scl @ S  # [K] @ [K,V] = [V]

    # output_from_qk — accumulate over j (vectorized over V)
    output_from_qk = torch.zeros(V_dim, dtype=torch.float32)

    for j in range(row + 1):  # j <= row (lower triangle including diagonal)
        # qk_dot_j = (q_raw * q_rcp) · k_normed[j]
        qk_dot_j = (q_raw * q_rcp) @ key_norm_cuda[j]

        g_diff_val = math.exp(g_cs_cumsum_cuda[row].item() - g_cs_cumsum_cuda[j].item())
        attn_qk_val = qk_dot_j.item() * g_diff_val

        # v_new_j[col_v] = sum_ii attn_sm[j][ii] * (value[ii] * beta[ii])
        v_new_j = attn_cuda_final[j, :seq_len] @ v_beta_all  # [S] @ [S,V] = [V]

        # k_cumdecay[j][d] = sum_ii attn_sm[j][ii] * k_beta[ii][d] * exp(g_cs[ii])
        # Vectorized: (attn_row_j * g_cs_exp) @ k_beta  → [K]
        weighted_attn = attn_cuda_final[j, :seq_len] * g_cs_exp_arr  # [S]
        k_cd_j = weighted_attn @ k_beta_full_arr  # [S] @ [S,K] = [K]

        v_prime_j = k_cd_j @ S  # [K] @ [K,V] = [V]

        v_nc_j = v_new_j - v_prime_j  # [V]

        output_from_qk += attn_qk_val * v_nc_j  # scalar * [V] → accumulate

    out_val = attn_inter_vec + output_from_qk
    output_cuda_sim[row] = out_val
    row_contributions[row] = output_from_qk.clone()

# ── Diagnostic: what does the HF reference compute for Phase 4? ─────────────

print("\n  --- Phase 4 intermediate diagnostics (row=3 as example) ---")
row_debug = 3

# Re-run just row 3 in detail
q_raw_d = Q[row_debug, HEAD, :].clone()
q_l2_sq_d = (q_raw_d ** 2).sum().item()
q_rcp_d = (1.0 / math.sqrt(q_l2_sq_d + 1e-6)) * q_rcp_sqrt_k
exp_g_row_d = math.exp(g_cs_cumsum_cuda[row_debug].item())

print(f"    row={row_debug}: q_norm_sq={q_l2_sq_d:.8f}, q_rcp={q_rcp_d:.10f}, exp_g={exp_g_row_d:.8f}")
print(f"    output_sim[{row_debug}] range: [{output_cuda_sim[row_debug].min():.6f}, {output_cuda_sim[row_debug].max():.6f}]")

cuda_out_head = cuda_output[:, HEAD * V_dim:(HEAD + 1) * V_dim].float()
print(f"    CUDA_kernel[{row_debug}] range: [{cuda_out_head[row_debug].min():.6f}, {cuda_out_head[row_debug].max():.6f}]")

if core_attn_ref_gpu0 is not None:
    ref_out_head = core_attn_ref_gpu0[:, HEAD, :].float()
    print(f"    HF_ref[{row_debug}] range: [{ref_out_head[row_debug].min():.6f}, {ref_out_head[row_debug].max():.6f}]")

# Show the diagonal values for context
print("\n  --- Diagonal comparison (attn before forward sub) ---")
for i in range(min(8, seq_len)):
    diag_cuda = attn_cuda[i, i].item()
    diag_hf   = ref_attn[i, i].item()
    diag_after_fwd_cuda = attn_cuda_final[i, i].item()
    diag_after_fwd_hf   = ref_attn[i, i].item()  # after forward sub + identity in HF
    print(f"    i={i}: CUDA_pre={diag_cuda:.6f}, HF_pre={diag_hf:.6f}, "
          f"CUDA_post={diag_after_fwd_cuda:.6f}, HF_post={diag_after_fwd_hf:.6f}")

# Show contribution per j for row 3
print(f"\n  --- Phase 4 contributions per j (row={row_debug}) ---")
for j in range(row_debug + 1):
    qk_dot_j_d = (q_raw_d * q_rcp_d) @ key_norm_cuda[j]
    g_diff_d = math.exp(g_cs_cumsum_cuda[row_debug].item() - g_cs_cumsum_cuda[j].item())
    attn_qk_val_d = qk_dot_j_d.item() * g_diff_d
    
    v_new_j_d = attn_cuda_final[j, :seq_len] @ v_beta_all
    
    contribution = attn_qk_val_d * v_new_j_d  # since S=0, v_prime=0
    print(f"    j={j}: attn_qk={attn_qk_val_d:+8.6f}, "
          f"|v_new|_max={v_new_j_d.abs().max():.6f}, "
          f"|contrib|_max={contribution.abs().max():.6f}")

print(f"\n  Total output_from_qk[{row_debug}] range: [{row_contributions[row_debug].min():.6f}, {row_contributions[row_debug].max():.6f}]")

# ═══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION: Run same algorithm as CUDA kernel but with PyTorch ops
# If this matches the CUDA kernel output, the bug is elsewhere (not in our sim)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CROSS-VALIDATION: PyTorch implementation of same algorithm as CUDA kernel")
print("=" * 70)

def chunked_gdn_cuda_style(Q, K_tensor, V_tens, a_proj_h, b_proj_h, A_log_h, dt_bias_h):
    """PyTorch implementation matching CUDA kernel's Phase 2 (no diagonal mask)."""
    S = Q.shape[0]
    
    g    = -A_log_h.exp() * F.softplus(a_proj_h + dt_bias_h)   # [S, H]
    beta = b_proj_h.sigmoid()                                   # [S, H]
    
    key_normed = F.normalize(K_tensor.float(), dim=-1)          # [S, H, K]
    
    k_beta = key_normed * beta.unsqueeze(-1)                    # [S, H, K]
    
    g_cs_full = torch.zeros(C, num_v_heads)
    g_cs_full[:S] = g.cumsum(dim=0)
    
    h = HEAD
    
    k_beta_h = k_beta[:, h, :]   # [S, K]
    k_normed_h = key_normed[:, h, :]  # [S, K]
    g_cs_h = g_cs_full[:S, h]     # [S]
    
    attn_gemm_h = -(k_beta_h @ k_normed_h.T)  # [S, S]
    
    # Decay mask (lower triangular including diagonal) — NO masking of diagonal!
    decay_mask_h = torch.zeros(S, S)
    for r in range(S):
        for c in range(r + 1):
            decay_mask_h[r, c] = math.exp(g_cs_h[r].item() - g_cs_h[c].item())
    
    attn_pre_h = attn_gemm_h * decay_mask_h
    
    # Phase 3: Forward substitution + identity (same as CUDA kernel)
    for i in range(1, S):
        row_buf = attn_pre_h[i, :i].clone()
        for j in range(i):
            accum = (row_buf[:i] * attn_pre_h[:i, j]).sum().item()
            attn_pre_h[i, j] = row_buf[j] + accum
    attn_pre_h[torch.arange(S), torch.arange(S)] += 1.0
    
    # Phase 4: Output computation
    v_beta_all_h = V_tens[:S, h, :] * beta[:, h].unsqueeze(1)  # [S, V]
    
    output_final = torch.zeros(S, V_dim)
    q_rcp_sqrt_k = 1.0 / math.sqrt(K)
    
    for row in range(S):
        q_raw = Q[row, h, :].float()
        q_l2_sq = (q_raw ** 2).sum().item()
        q_rcp = (1.0 / math.sqrt(q_l2_sq + 1e-6)) * q_rcp_sqrt_k
        
        output_from_qk = torch.zeros(V_dim)
        g_cs_exp = torch.exp(g_cs_h[:S])
        
        for j in range(row + 1):
            qk_dot_j = (q_raw * q_rcp) @ k_normed_h[j]
            g_diff_val = math.exp(g_cs_h[row].item() - g_cs_h[j].item())
            attn_qk_val = qk_dot_j.item() * g_diff_val
            
            v_new_j = attn_pre_h[j, :S] @ v_beta_all_h
            
            weighted_attn = attn_pre_h[j, :S] * g_cs_exp
            k_cd_j = weighted_attn @ k_beta_h[:S]
            
            v_prime_j = torch.zeros(V_dim)  # S=0 for chunk 0, so v_prime is all zeros
            v_nc_j = v_new_j - v_prime_j
            
            output_from_qk += attn_qk_val * v_nc_j
        
        output_final[row] = output_from_qk
    
    return output_final

def cosine_sim(a, b):
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))

output_cuda_style = chunked_gdn_cuda_style(Q, K_tensor, V, a_proj[:, :num_v_heads], b_proj[:, :num_v_heads], A_log, dt_bias)

cuda_out_head_verify = cuda_output[:, HEAD * V_dim:(HEAD + 1) * V_dim].float()

print(f"  |output_pt - CUDA_kernel| max = {abs(output_cuda_style - cuda_out_head_verify).max():.2e}")
print(f"  |output_pt - CUDA_kernel| mean= {abs(output_cuda_style - cuda_out_head_verify).mean():.2e}")
print(f"  output_pt range: [{output_cuda_style.min():.6f}, {output_cuda_style.max():.6f}]")

if core_attn_ref_gpu0 is not None:
    ref_out_head = core_attn_ref_gpu0[:, HEAD, :].float()
    cos_nv_hf = cosine_sim(output_cuda_style, ref_out_head)
    cos_nv_cuda = cosine_sim(output_cuda_style, cuda_out_head_verify)
    print(f"  Per-token cos(pt_vs CUDA): min={cos_nv_cuda.min():.6f}, mean={cos_nv_cuda.mean():.6f}")
    print(f"  Per-token cos(pt_vs HF):   min={cos_nv_hf.min():.6f}, mean={cos_nv_hf.mean():.6f}")

# Also check: does output_pt match my simulation (output_cuda_sim)?
print(f"  |output_pt - output_sim| max = {abs(output_cuda_style - output_cuda_sim).max():.2e}")

# ═══════════════════════════════════════════════════════════════════════════
# DEEP DIVE: Element-wise analysis of CUDA kernel vs simulation
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("DEEP DIVE: Element-wise ratio and pattern analysis")
print("=" * 70)

ratio_cuda_sim = cuda_out_head_verify / (output_cuda_sim.abs() + 1e-10)
print(f"  |CUDA/sim| ratio: min={ratio_cuda_sim.min():.4f}, max={ratio_cuda_sim.max():.4f}")
print(f"  Mean |CUDA/sim| ratio = {ratio_cuda_sim.mean():.4f}")

# Per-row analysis
for r in range(seq_len):
    sim_row = output_cuda_sim[r]
    cuda_row = cuda_out_head_verify[r]
    diff_row = (cuda_row - sim_row).abs()
    
    # Check if there's a constant offset pattern
    mean_diff = (cuda_row - sim_row).mean().item()
    mean_ratio = ((cuda_row.abs() / (sim_row.abs() + 1e-10)).mean()).item()
    
    print(f"  Row {r}: |CUDA-sim| max={diff_row.max():.6f}, "
          f"|CUDA/sim| ratio_mean={mean_ratio:.4f}, "
          f"(CUDA-sim) mean offset={mean_diff:.6f}")

# Check bf16 precision of input tensors
print("\n" + "=" * 70)
print("DEEP DIVE: BF16 quantization check on inputs")
print("=" * 70)

# For head 0, check how much value differs after bf16 roundtrip
val_raw = V[:seq_len, HEAD, :].bfloat16().float()  # bf16 -> fp32
val_orig = V[:seq_len, HEAD, :]
val_quant_err = (val_raw - val_orig).abs()
print(f"  Value bf16 quantization error: max={val_quant_err.max():.6f}, mean={val_quant_err.mean():.8f}")

q_raw_bf = Q[:seq_len, HEAD, :].bfloat16().float()
q_orig = Q[:seq_len, HEAD, :]
q_quant_err = (q_raw_bf - q_orig).abs()
print(f"  Query bf16 quantization error: max={q_quant_err.max():.6f}, mean={q_quant_err.mean():.8f}")

# Check if CUDA kernel values look like they could be from wrong head
print("\n" + "=" * 70)
print("DEEP DIVE: Could the CUDA kernel output match a different head?")
print("=" * 70)

for h in range(min(8, num_v_heads)):
    cuda_h = cuda_output[:, h * V_dim:(h + 1) * V_dim].float()
    diff_sim = abs(output_cuda_sim - cuda_h).max().item()
    cos_val = cosine_sim(output_cuda_sim, cuda_h).mean().item()
    if h == HEAD:
        print(f"  Head {h} (target): max_diff={diff_sim:.4f}, cos_mean={cos_val:.6f}")
    elif abs(cos_val - 0.936) < 0.1 or diff_sim < 2.0:
        print(f"  Head {h}:       max_diff={diff_sim:.4f}, cos_mean={cos_val:.6f} ← interesting?")

# Check: is the CUDA kernel output consistent across col_v dimensions?
print("\n" + "=" * 70)
print("DEEP DIVE: Per-dimension variance of CUDA kernel output")
print("=" * 70)

for r in range(min(3, seq_len)):
    cuda_r = cuda_out_head_verify[r]
    sim_r = output_cuda_sim[r]
    
    # Look for specific col_v where the difference is largest
    max_diff_dim = abs(cuda_r - sim_r).argmax().item()
    print(f"  Row {r}: worst col_v={max_diff_dim}, "
          f"CUDA[{max_diff_dim}]={cuda_r[max_diff_dim]:.6f}, "
          f"sim[{max_diff_dim}]={sim_r[max_diff_dim]:.6f}, "
          f"diff={abs(cuda_r[max_diff_dim] - sim_r[max_diff_dim]):.6f}")

# Check: is there a constant offset pattern (cuda - sim ≈ c)?
print("\n" + "=" * 70)
print("DEEP DIVE: Constant offset analysis")
print("=" * 70)

# For each row, compute cuda - sim and see if it's approximately constant
for r in range(min(5, seq_len)):
    diff = (cuda_out_head_verify[r] - output_cuda_sim[r])
    print(f"  Row {r}: |CUDA-sim| mean={diff.abs().mean():.6f}, std={diff.std():.6f}, "
          f"min={diff.min():.6f}, max={diff.max():.6f}")

# If the difference is roughly constant per row, it suggests an additive bug
# Check: does cuda - sim correlate with the value tensor?
print("\n  Checking if (CUDA-sim) correlates with value[...]")
for r in range(min(3, seq_len)):
    diff = (cuda_out_head_verify[r] - output_cuda_sim[r])
    val_r = V[r, HEAD, :].float()
    
    # Correlation between diff and value
    corr = (diff * val_r).sum().item() / (diff.std() * val_r.std()).item() if diff.std() > 0 and val_r.std() > 0 else float('nan')
    
    print(f"  Row {r}: corr(CUDA-sim, value) = {corr:.6f}")

# Check: does the CUDA kernel output correlate with k_beta?
print("\n  Checking if (CUDA-sim) correlates with query")
for r in range(min(3, seq_len)):
    diff = (cuda_out_head_verify[r] - output_cuda_sim[r])
    q_r = Q[r, HEAD, :].float()
    
    corr = (diff * q_r).sum().item() / (diff.std() * q_r.std()).item() if diff.std() > 0 and q_r.std() > 0 else float('nan')
    
    print(f"  Row {r}: corr(CUDA-sim, query) = {corr:.6f}")

# Check: is there a specific col_v range that shows more deviation?
print("\n  Deviation by col_v groups (first 5 rows)")
for r in range(min(5, seq_len)):
    diff = abs(cuda_out_head_verify[r] - output_cuda_sim[r])
    for group_start in [0, 32, 64, 96]:
        group_end = min(group_start + 32, V_dim)
        mean_diff = diff[group_start:group_end].mean().item()
        print(f"    Row {r}, col_v[{group_start}:{group_end}]: mean_diff={mean_diff:.6f}")

# Check: what if the key was not normalized? Would that produce the large output?
print("\n" + "=" * 70)
print("DEEP DIVE: What if key normalization is wrong?")
print("=" * 70)

key_raw_h = K_tensor[:seq_len, HEAD, :].float()  # [S, K] raw key (unnormalized)
key_norm_h = F.normalize(key_raw_h, dim=-1)

attn_gemm_unnorm = -(key_raw_h * beta_cuda.unsqueeze(1) @ key_raw_h.T)
attn_gemm_normed = -(key_norm_h * beta_cuda.unsqueeze(1) @ key_norm_h.T)

print(f"  GEMM (unnormalized key): max abs value = {attn_gemm_unnorm.abs().max():.6f}")
print(f"  GEMM (normalized key):   max abs value = {abs(attn_gemm_normed).max():.6f}")
print(f"  Key L2 norm squared (avg) = {(key_raw_h**2).sum(dim=-1).mean():.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# CRITICAL: Check if beta is not applied to value in Phase 4
# The correlation with value suggests the bug might be in v_nc computation
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HYPOTHESIS: What if beta is NOT applied to value in Phase 4?")
print("=" * 70)

output_no_beta = torch.zeros(seq_len, V_dim)
v_raw_all = V[:seq_len, HEAD, :].float()  # No beta!

for row in range(seq_len):
    q_raw = Q[row, HEAD, :].float()
    q_l2_sq = (q_raw**2).sum().item()
    q_rcp = (1.0 / math.sqrt(q_l2_sq + 1e-6)) * (1.0 / math.sqrt(K))

    output_from_qk = torch.zeros(V_dim)
    g_cs_exp = torch.exp(g_cs_cumsum_cuda[:seq_len])

    for j in range(row + 1):
        qk_dot_j = (q_raw * q_rcp) @ key_norm_cuda[j]
        g_diff_val = math.exp(g_cs_cumsum_cuda[row].item() - g_cs_cumsum_cuda[j].item())
        attn_qk_val = qk_dot_j.item() * g_diff_val

        # Use raw value WITHOUT beta!
        v_new_j = attn_cuda_final[j, :seq_len] @ v_raw_all

        weighted_attn = attn_cuda_final[j, :seq_len] * g_cs_exp
        k_cd_j = weighted_attn @ k_beta_full_arr  # k_beta still uses normalized key + beta
        v_prime_j = torch.zeros(V_dim)  # S=0 for chunk 0

        v_nc_j = v_new_j - v_prime_j
        output_from_qk += attn_qk_val * v_nc_j

    output_no_beta[row] = output_from_qk

print(f"  |output_no_beta - CUDA_kernel| max = {abs(output_no_beta - cuda_out_head_verify).max():.2e}")
print(f"  |output_no_beta - CUDA_kernel| mean= {abs(output_no_beta - cuda_out_head_verify).mean():.2e}")
print(f"  output_no_beta range: [{output_no_beta.min():.6f}, {output_no_beta.max():.6f}]")

if core_attn_ref_gpu0 is not None:
    ref_out_head = core_attn_ref_gpu0[:, HEAD, :].float()
    cos_nb_cuda = cosine_sim(output_no_beta, cuda_out_head_verify)
    print(f"  Per-token cos(no_beta vs CUDA): min={cos_nb_cuda.min():.6f}, mean={cos_nb_cuda.mean():.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# Check: what if the query L2 norm is not applied in Phase 4?
# If q_rcp were 1 instead of ~0.006, output would be ~160x larger
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HYPOTHESIS: What if query normalization is wrong (q_rcp ≈ 1)?")
print("=" * 70)

output_no_qnorm = torch.zeros(seq_len, V_dim)

for row in range(seq_len):
    q_raw = Q[row, HEAD, :].float()

    # Use q_rcp = 1 instead of the correct value
    q_rcp_fake = 1.0  # WRONG: no normalization!

    output_from_qk = torch.zeros(V_dim)
    g_cs_exp = torch.exp(g_cs_cumsum_cuda[:seq_len])

    for j in range(row + 1):
        qk_dot_j = (q_raw * q_rcp_fake) @ key_norm_cuda[j]
        g_diff_val = math.exp(g_cs_cumsum_cuda[row].item() - g_cs_cumsum_cuda[j].item())
        attn_qk_val = qk_dot_j.item() * g_diff_val

        v_new_j = attn_cuda_final[j, :seq_len] @ v_beta_all

        weighted_attn = attn_cuda_final[j, :seq_len] * g_cs_exp
        k_cd_j = weighted_attn @ k_beta_full_arr
        v_prime_j = torch.zeros(V_dim)

        v_nc_j = v_new_j - v_prime_j
        output_from_qk += attn_qk_val * v_nc_j

    output_no_qnorm[row] = output_from_qk

print(f"  |output_no_qnorm - CUDA_kernel| max = {abs(output_no_qnorm - cuda_out_head_verify).max():.2e}")
print(f"  |output_no_qnorm - CUDA_kernel| mean= {abs(output_no_qnorm - cuda_out_head_verify).mean():.2e}")
print(f"  output_no_qnorm range: [{output_no_qnorm.min():.6f}, {output_no_qnorm.max():.6f}]")

if core_attn_ref_gpu0 is not None:
    cos_nq_cuda = cosine_sim(output_no_qnorm, cuda_out_head_verify)
    print(f"  Per-token cos(no_qnorm vs CUDA): min={cos_nq_cuda.min():.6f}, mean={cos_nq_cuda.mean():.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# Combined hypothesis: no beta AND partially wrong query norm
# Let's try q_rcp without L2 normalization but with 1/sqrt(K) scaling
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HYPOTHESIS: q_rcp = 1/sqrt(K) only (no L2 norm), beta NOT applied")
print("=" * 70)

output_qrcp_k_only_no_beta = torch.zeros(seq_len, V_dim)

for row in range(seq_len):
    q_raw = Q[row, HEAD, :].float()
    # q_rcp = only 1/sqrt(K), no L2 norm division
    q_rcp_partial = 1.0 / math.sqrt(K)  # ~0.0884 instead of ~0.006

    output_from_qk = torch.zeros(V_dim)
    g_cs_exp = torch.exp(g_cs_cumsum_cuda[:seq_len])

    for j in range(row + 1):
        qk_dot_j = (q_raw * q_rcp_partial) @ key_norm_cuda[j]
        g_diff_val = math.exp(g_cs_cumsum_cuda[row].item() - g_cs_cumsum_cuda[j].item())
        attn_qk_val = qk_dot_j.item() * g_diff_val

        # No beta on value!
        v_new_j = attn_cuda_final[j, :seq_len] @ v_raw_all

        weighted_attn = attn_cuda_final[j, :seq_len] * g_cs_exp
        k_cd_j = weighted_attn @ k_beta_full_arr
        v_prime_j = torch.zeros(V_dim)

        v_nc_j = v_new_j - v_prime_j
        output_from_qk += attn_qk_val * v_nc_j

    output_qrcp_k_only_no_beta[row] = output_from_qk

print(f"  |output - CUDA_kernel| max = {abs(output_qrcp_k_only_no_beta - cuda_out_head_verify).max():.2e}")
print(f"  |output - CUDA_kernel| mean= {abs(output_qrcp_k_only_no_beta - cuda_out_head_verify).mean():.2e}")
print(f"  output range: [{output_qrcp_k_only_no_beta.min():.6f}, {output_qrcp_k_only_no_beta.max():.6f}]")

if core_attn_ref_gpu0 is not None:
    cos_comb_cuda = cosine_sim(output_qrcp_k_only_no_beta, cuda_out_head_verify)
    print(f"  Per-token cos(combined vs CUDA): min={cos_comb_cuda.min():.6f}, mean={cos_comb_cuda.mean():.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# What if the key in k_beta uses UNNORMALIZED key? 
# (k_normed might be wrong but k_beta is correct, or vice versa)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HYPOTHESIS: k_beta uses unnormalized key (but GEMM uses normalized)")
print("=" * 70)

# If k_beta used raw key instead of normalized key, the values would be larger
k_beta_unnorm = key_raw_h * beta_cuda.unsqueeze(1)  # [S, K]

output_kbeta_unnorm = torch.zeros(seq_len, V_dim)

for row in range(seq_len):
    q_raw = Q[row, HEAD, :].float()
    q_l2_sq = (q_raw**2).sum().item()
    q_rcp = (1.0 / math.sqrt(q_l2_sq + 1e-6)) * (1.0 / math.sqrt(K))

    output_from_qk = torch.zeros(V_dim)
    g_cs_exp = torch.exp(g_cs_cumsum_cuda[:seq_len])

    for j in range(row + 1):
        qk_dot_j = (q_raw * q_rcp) @ key_norm_cuda[j]
        g_diff_val = math.exp(g_cs_cumsum_cuda[row].item() - g_cs_cumsum_cuda[j].item())
        attn_qk_val = qk_dot_j.item() * g_diff_val

        v_new_j = attn_cuda_final[j, :seq_len] @ v_beta_all

        # Use unnormalized k_beta!
        weighted_attn = attn_cuda_final[j, :seq_len] * g_cs_exp
        k_cd_j = weighted_attn @ k_beta_unnorm  # DIFFERENT: unnormalized key in k_beta
        v_prime_j = torch.zeros(V_dim)

        v_nc_j = v_new_j - v_prime_j
        output_from_qk += attn_qk_val * v_nc_j

    output_kbeta_unnorm[row] = output_from_qk

print(f"  |output - CUDA_kernel| max = {abs(output_kbeta_unnorm - cuda_out_head_verify).max():.2e}")
print(f"  |output - CUDA_kernel| mean= {abs(output_kbeta_unnorm - cuda_out_head_verify).mean():.2e}")
print(f"  output range: [{output_kbeta_unnorm.min():.6f}, {output_kbeta_unnorm.max():.6f}]")

if core_attn_ref_gpu0 is not None:
    cos_kb_cuda = cosine_sim(output_kbeta_unnorm, cuda_out_head_verify)
    print(f"  Per-token cos(k_beta_unnorm vs CUDA): min={cos_kb_cuda.min():.6f}, mean={cos_kb_cuda.mean():.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# What if the key used for GEMM (Phase 2) is unnormalized?
# But k_beta and Phase 4 still use normalized keys
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HYPOTHESIS: GEMM in Phase 2 uses unnormalized key (attn wrong)")
print("=" * 70)

# Recompute attn with unnormalized keys for the GEMM
attn_gemm_unnorm_full = -(key_raw_h * beta_cuda.unsqueeze(1)) @ key_norm_h.T

attn_pre_unnorm = attn_gemm_unnorm_full[:seq_len, :seq_len] * decay_mask_cuda[:seq_len, :seq_len]

# Forward substitution + identity on the unnormalized-GEMM result
for i in range(1, seq_len):
    row_buf = attn_pre_unnorm[i, :i].clone()
    for j in range(i):
        accum = (row_buf[:i] * attn_pre_unnorm[:i, j]).sum().item()
        attn_pre_unnorm[i, j] = row_buf[j] + accum

attn_pre_unnorm[torch.arange(seq_len), torch.arange(seq_len)] += 1.0

# Recompute Phase 4 with the unnormalized-GEMM attn matrix
output_attn_unnorm = torch.zeros(seq_len, V_dim)

for row in range(seq_len):
    q_raw = Q[row, HEAD, :].float()
    q_l2_sq = (q_raw**2).sum().item()
    q_rcp = (1.0 / math.sqrt(q_l2_sq + 1e-6)) * (1.0 / math.sqrt(K))

    output_from_qk = torch.zeros(V_dim)
    g_cs_exp = torch.exp(g_cs_cumsum_cuda[:seq_len])

    for j in range(row + 1):
        qk_dot_j = (q_raw * q_rcp) @ key_norm_cuda[j]
        g_diff_val = math.exp(g_cs_cumsum_cuda[row].item() - g_cs_cumsum_cuda[j].item())
        attn_qk_val = qk_dot_j.item() * g_diff_val

        v_new_j = attn_pre_unnorm[j, :seq_len] @ v_beta_all

        weighted_attn = attn_pre_unnorm[j, :seq_len] * g_cs_exp
        k_cd_j = weighted_attn @ k_beta_full_arr
        v_prime_j = torch.zeros(V_dim)

        v_nc_j = v_new_j - v_prime_j
        output_from_qk += attn_qk_val * v_nc_j

    output_attn_unnorm[row] = output_from_qk

print(f"  |output - CUDA_kernel| max = {abs(output_attn_unnorm - cuda_out_head_verify).max():.2e}")
print(f"  |output - CUDA_kernel| mean= {abs(output_attn_unnorm - cuda_out_head_verify).mean():.2e}")
print(f"  output range: [{output_attn_unnorm.min():.6f}, {output_attn_unnorm.max():.6f}]")

if core_attn_ref_gpu0 is not None:
    cos_aun_cuda = cosine_sim(output_attn_unnorm, cuda_out_head_verify)
    print(f"  Per-token cos(attn_unnorm vs CUDA): min={cos_aun_cuda.min():.6f}, mean={cos_aun_cuda.mean():.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# DIRECT APPROACH: What correction transforms sim → CUDA output?
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("DIRECT: Required correction per row to match CUDA kernel")
print("=" * 70)

for r in range(min(5, seq_len)):
    sim_r = output_cuda_sim[r]
    cuda_r = cuda_out_head_verify[r]
    
    # What multiplicative factor is needed? (cuda - 0.45*mean_offset) / sim
    # Or simpler: what additive offset is needed per dimension?
    add_needed = cuda_r - sim_r
    
    # Is the additive offset proportional to v_beta_all? 
    # (If so, it suggests a bug in how value contributes)
    if sim_r.abs().sum() > 0:
        ratio_per_dim = add_needed / (sim_r.abs() + 1e-10)
        print(f"  Row {r}: mean_add={add_needed.mean():.6f}, "
              f"mean_ratio_to_sim={ratio_per_dim.mean():.2f}")
    
    # Check: is the additive offset similar to (attn_inter) from a non-zero state?
    # For chunk 0, attn_inter = q_scaled @ S where S should be zero
    # But what if S were not properly initialized?

# ═══════════════════════════════════════════════════════════════════════════
# CRITICAL INSIGHT: Check if CUDA kernel state S might not be zero for chunk 0
# What if there's leftover state from a previous run?
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CRITICAL: Could the state S buffer have garbage values for chunk 0?")
print("=" * 70)

# For chunk 0 with zero state, output should be small (~0.01)
# But CUDA kernel gives large output (~0.5). If S had non-zero values, 
# both attn_inter and v_prime would be non-trivial.

# Let's try: what if we use a non-zero state that could explain the pattern?
# For chunk 0: attn_inter = q_scaled * exp(g_cs) @ S
# This adds S-weighted query to the output

# What magnitude of S would give the observed ~0.45 offset?
# attn_inter ≈ q_rcp * exp(g_cs) * (q_raw @ S) ≈ 0.006 * 1.0 * (q_raw @ S)
# For this to contribute ~0.45: need ||S|| ≈ 0.45 / (0.006 * ||q_raw||/sqrt(K))

for r in range(min(3, seq_len)):
    q_raw = Q[r, HEAD, :].float()
    q_l2_sq = (q_raw**2).sum().item()
    q_rcp = (1.0 / math.sqrt(q_l2_sq + 1e-6)) * (1.0 / math.sqrt(K))
    exp_g_row = math.exp(g_cs_cumsum_cuda[r].item())
    
    # What ||S|| would be needed to produce the observed output?
    diff_mean = abs(cuda_out_head_verify[r] - output_cuda_sim[r]).mean().item()
    required_s_scale = diff_mean / (q_rcp * exp_g_row * q_raw.norm().item())
    
    print(f"  Row {r}: q_rcp={q_rcp:.6f}, exp_g={exp_g_row:.4f}, "
          f"|cuda-sim|_mean={diff_mean:.6f}")
    print(f"    Required ||S|| ≈ {required_s_scale:.4f} to explain offset via attn_inter")

# Compare with CUDA kernel output
cuda_out_head = cuda_output[:, HEAD * V_dim:(HEAD + 1) * V_dim].float()

print(f"  --- Output comparison for head={HEAD} ---")
print(f"    |output_sim - CUDA_kernel| max = {abs(output_cuda_sim - cuda_out_head).max():.2e}")
print(f"    |output_sim - CUDA_kernel| mean= {abs(output_cuda_sim - cuda_out_head).mean():.2e}")

if core_attn_ref_gpu0 is not None:
    ref_out_head = core_attn_ref_gpu0[:, HEAD, :].float()
    print(f"    |output_sim - HF_reference| max  = {abs(output_cuda_sim - ref_out_head).max():.2e}")
    print(f"    |output_sim - HF_reference| mean = {abs(output_cuda_sim - ref_out_head).mean():.2e}")

# Cosine similarity
def cosine_sim(a, b):
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))

cos_vs_cuda_per_row = cosine_sim(output_cuda_sim, cuda_out_head)  # [S]
print(f"\n    Cosine similarity (sim vs CUDA), per row: min={cos_vs_cuda_per_row.min():.8f}, mean={cos_vs_cuda_per_row.mean():.8f}")

if core_attn_ref_gpu0 is not None:
    cos_vs_hf_per_row = cosine_sim(output_cuda_sim, ref_out_head)  # [S]
    cos_hf_cuda_per_row = cosine_sim(cuda_out_head, ref_out_head)  # [S]
    print(f"    Cosine similarity (sim vs HF), per row:   min={cos_vs_hf_per_row.min():.8f}, mean={cos_vs_hf_per_row.mean():.8f}")
    print(f"    Cosine similarity (CUDA vs HF), per row:  min={cos_hf_cuda_per_row.min():.8f}, mean={cos_hf_cuda_per_row.mean():.8f}")

# ─── Detailed per-token comparison ──────────────────────────────────────────

print()
print("-" * 70)
print("Per-token detailed comparison (first 5 tokens)")
print("-" * 70)

for t in range(min(5, seq_len)):
    sim_t = output_cuda_sim[t]
    cuda_t = cuda_out_head[t]
    hf_t   = ref_out_head[t] if core_attn_ref_gpu0 is not None else None

    diff_sim_cuda = abs(sim_t - cuda_t).max().item()
    diff_sim_hf   = abs(sim_t - hf_t).max().item() if hf_t is not None else float('inf')
    cos_sim_cuda  = cosine_sim(sim_t.unsqueeze(0), cuda_t.unsqueeze(0)).item()

    print(f"  Token {t}: max_diff_sim_cuda={diff_sim_cuda:.4e}, "
          f"cos_sim_cuda={cos_sim_cuda:.6f}", end="")
    if hf_t is not None:
        diff_hf_cuda = abs(hf_t - cuda_t).max().item()
        print(f", max_diff_hf_cuda={diff_hf_cuda:.4e}")
    else:
        print()

# ─── Full head comparison ───────────────────────────────────────────────────

print()
print("-" * 70)
print("Full output statistics (head=0)")
print("-" * 70)
print(f"  Simulated output range: [{output_cuda_sim.min():.6f}, {output_cuda_sim.max():.6f}]")
print(f"  CUDA kernel output range: [{cuda_out_head.min():.6f}, {cuda_out_head.max():.6f}]")
if core_attn_ref_gpu0 is not None:
    print(f"  HF reference output range: [{ref_out_head.min():.6f}, {ref_out_head.max():.6f}]")

print()
cos_sim_cuda_full = cosine_sim(output_cuda_sim, cuda_out_head)
print(f"  Per-token cos(sim vs CUDA): min={cos_sim_cuda_full.min():.6f}, mean={cos_sim_cuda_full.mean():.6f}")
if core_attn_ref_gpu0 is not None:
    cos_hf_full = cosine_sim(output_cuda_sim, ref_out_head)
    cos_hf_cuda_full = cosine_sim(cuda_out_head, ref_out_head)
    print(f"  Per-token cos(sim vs HF):   min={cos_hf_full.min():.6f}, mean={cos_hf_full.mean():.6f}")
    print(f"  Per-token cos(CUDA vs HF): min={cos_hf_cuda_full.min():.6f}, mean={cos_hf_cuda_full.mean():.6f}")

print()
print("=" * 80)
print("Debug complete.")
print("=" * 80)
