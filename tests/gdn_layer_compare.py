#!/usr/bin/env python3
"""Compare engine GDN dump intermediates against PyTorch reference for Qwen3.6-27B INT4, TP=2 GPU 0.

Two comparison modes:

1) **Full pipeline** — computes all stages from hidden_input using dequantized weights.
   This verifies the entire chain including INT4 GEMM accuracy. Expect cosine < threshold
   for INT4 stages due to quantization error in weight dequantization.

2) **Kernel-only** — uses engine's own intermediates as inputs, avoiding INT4 dequantization.
   Only BF16 weights (conv1d, A_log, dt_bias, norm.weight) are needed. This isolates kernel
   correctness from weight errors.

Stages:
  hidden_input   — loaded from engine dump as reference input (no comparison)
  mixed_qkv      — input @ in_proj_qkv^T          [seq_len, conv_dim]        (INT4 GEMM)
  conv_out       — depthwise_conv1d_silu(mixed_qkv) [seq_len, conv_dim]      (BF16 kernel)
  query          — conv_out[:, :key_dim]            [seq_len, key_dim]
  key            — conv_out[:, key_dim:2*key_dim]   [seq_len, key_dim]
  value          — conv_out[:, 2*key_dim:]           [seq_len, value_dim]
  query_expanded — repeat_interleave(query, kv_ratio) [seq_len, num_v_heads, head_k_dim]
  key_expanded   — repeat_interleave(key, kv_ratio)  [seq_len, num_v_heads, head_k_dim]
  a_proj         — input @ in_proj_a^T              [seq_len, num_v_heads]  (BF16 GEMM)
  b_proj         — input @ in_proj_b^T              [seq_len, num_v_heads]  (BF16 GEMM)
  core_attn_out  — GDN_recurrent_step(...)          [seq_len, num_v_heads, head_v_dim]  (BF16 kernel)
  z_gate         — input @ in_proj_z^T              [seq_len, num_v_heads, head_v_dim] (INT4 GEMM)
  norm_output    — RMSNormGated(core, z, weight)     [seq_len, num_v_heads, head_v_dim] (BF16 kernel)
  output         — norm_output @ out_proj^T          [seq_len, hidden_size]  (INT4 GEMM)

Usage:
    python tests/gdn_layer_compare.py
    python tests/gdn_layer_compare.py --layer 5
    python tests/gdn_layer_compare.py --dump-dir /tmp/gdn_dump/layer_18 --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


# =========================================================================
# Configuration
# =========================================================================

DEFAULT_MODEL_PATH = "/home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4"
GROUP_SIZE = 128

STAGE_THRESHOLDS = {
    "mixed_qkv": 0.990,
    "conv_out": 0.999,
    "query": 0.990,
    "key": 0.990,
    "value": 0.990,
    "query_expanded": 0.990,
    "key_expanded": 0.990,
    "a_proj": 0.999,
    "b_proj": 0.999,
    "core_attn_out": 0.999,
    "z_gate": 0.990,
    "norm_output": 0.990,
    "output": 0.990,
}

# =========================================================================
# IO helpers (matching ref_intermediates.py)
# =========================================================================

def load_raw_bf16(path, shape):
    """Load a .raw bf16 file into a float32 torch tensor with given shape."""
    data = open(path, "rb").read()
    if len(data) == 0:
        raise ValueError(f"Empty file: {path}")
    n = len(data) // 2
    arr = np.frombuffer(data, dtype=np.uint16)
    f32_bits = arr.astype(np.uint32) << 16
    return torch.from_numpy(f32_bits.view(np.float32).reshape(shape))


def cos_sim(a, b):
    """Cosine similarity between two flattened tensors."""
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    dot = (a_f * b_f).sum()
    na = a_f.norm()
    nb = b_f.norm()
    if na.item() == 0 or nb.item() == 0:
        return 0.0
    return (dot / (na * nb)).item()


def l2_error(a, b):
    """Normalized L2 error ||a-b|| / ||a||."""
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    return (a_f - b_f).norm().item() / (a_f.norm().item() + 1e-30)


# =========================================================================
# INT4 Dequantization (matching ref_intermediates.py)
# =========================================================================

def unpack_int4(data: torch.Tensor) -> torch.Tensor:
    """Unpack INT4 values from int32-packed tensor.

    Input: [M, N/8] int32 where each int32 packs 8 int4 values.
    Output: [M, N] float with each element in [0, 15].
    """
    M, N_packed = data.shape
    raw_bytes = data.numpy().astype(np.int32).view(np.uint8).reshape(M, N_packed * 4)
    low = raw_bytes & 0x0F
    high = (raw_bytes >> 4) & 0x0F
    result = np.zeros((M, N_packed * 8), dtype=np.uint8)
    result[:, 0::2] = low
    result[:, 1::2] = high
    return torch.from_numpy(result.astype(np.float32))


def dequantize_int4_autogptq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = GROUP_SIZE,
) -> torch.Tensor:
    """Dequantize AutoRound/AutoGPTQ INT4 weights.

    Layout: qweight [K/8, N], qzeros [num_groups, N/8], scales [num_groups, N]
    Dequant formula: w_deq = (w_int4 - zero_point) * scale
    """
    K_packed_dim = qweight.shape[0]
    N_dim = qweight.shape[1]
    K_dim = K_packed_dim * 8

    # Unpack
    w_int4 = unpack_int4(qweight)
    z_int4 = unpack_int4(qzeros)

    num_groups = scales.shape[0]

    # Fix layout: [K_packed, N*8] -> reshape -> permute -> flatten -> [K, N]
    w_correct = w_int4.reshape(K_packed_dim, N_dim, 8).permute(0, 2, 1).reshape(K_dim, N_dim)
    z_correct = z_int4.reshape(-1, N_dim, 8).permute(0, 2, 1).reshape(-1, N_dim)

    # Per-group dequant along K axis
    w_grps = w_correct.reshape(num_groups, group_size, N_dim)
    z_grps = z_correct.reshape(num_groups, 1, N_dim)
    s_grps = scales.reshape(num_groups, 1, N_dim)

    w_f32 = (w_grps.float() - z_grps.float()) * s_grps.float()
    return w_f32.reshape(K_dim, N_dim)


# =========================================================================
# Weight loading (matching ref_intermediates.py _load_tensor pattern)
# =========================================================================

class GdnWeightLoader:
    """Load and dequantize GDN weights for a specific layer + GPU shard."""

    def __init__(self, model_dir, layer_idx, tp_size=2, gpu_idx=0):
        self.model_dir = Path(model_dir)
        self.layer_idx = layer_idx
        self.tp_size = tp_size
        self.gpu_idx = gpu_idx

        with open(self.model_dir / "config.json") as f:
            raw_config = json.load(f)
        if "text_config" in raw_config and "hidden_size" in raw_config["text_config"]:
            self.config = raw_config["text_config"]
        else:
            self.config = raw_config

        with open(self.model_dir / "model.safetensors.index.json") as f:
            self.weight_idx = json.load(f)["weight_map"]

        # Model dimensions (full, before TP sharding)
        self.hidden_size = self.config["hidden_size"]
        self.num_v_heads_full = self.config["linear_num_value_heads"]    # 48
        self.num_k_heads_full = self.config["linear_num_key_heads"]      # 16
        self.head_k_dim = self.config["linear_key_head_dim"]             # 128
        self.head_v_dim = self.config["linear_value_head_dim"]           # 128
        self.kv_ratio = self.num_v_heads_full // self.num_k_heads_full   # 3
        self.conv_kernel_size = self.config.get("linear_conv_kernel_dim", 4)

        # Per-GPU sharded dimensions
        self.num_v_heads = self.num_v_heads_full // self.tp_size   # 24
        self.num_k_heads = self.num_k_heads_full // self.tp_size   # 8
        self.key_dim = self.num_k_heads * self.head_k_dim          # 1024
        self.value_dim = self.num_v_heads * self.head_v_dim        # 3072
        self.conv_dim = 2 * self.key_dim + self.value_dim          # 5120

    def _load_tensor(self, name: str) -> torch.Tensor:
        """Load a tensor from safetensors.

        CRITICAL: Do NOT convert int32/uint32 tensors to float32 — this corrupts INT4 packed data.
        """
        fname = self.weight_idx[name]
        path = self.model_dir / fname
        with safe_open(str(path), framework="pt") as f:
            tensor = f.get_tensor(name)
        if tensor.dtype in (torch.int32, torch.int8, torch.uint8, torch.int64):
            return tensor  # keep INT4 packed data intact
        return tensor.float()

    def _weight_name(self, attr: str) -> str:
        """Build full weight name for this layer."""
        return f"model.language_model.layers.{self.layer_idx}.linear_attn.{attr}"

    def load_in_proj_qkv_dequant(self):
        """Dequantize in_proj_qkv and shard to GPU. Returns [hidden_size, conv_dim_per_gpu]."""
        qweight = self._load_tensor(self._weight_name("in_proj_qkv.qweight"))
        qzeros = self._load_tensor(self._weight_name("in_proj_qkv.qzeros"))
        scales = self._load_tensor(self._weight_name("in_proj_qkv.scales"))

        W_full = dequantize_int4_autogptq(qweight, qzeros, scales)  # [5120, 10240]

        # Column-parallel: shard along output (N) dimension
        sharded_N = self.conv_dim
        start = self.gpu_idx * sharded_N
        end = start + sharded_N
        return W_full[:, start:end]

    def load_in_proj_z_dequant(self):
        """Dequantize in_proj_z and shard to GPU. Returns [hidden_size, value_dim_per_gpu]."""
        qweight = self._load_tensor(self._weight_name("in_proj_z.qweight"))
        qzeros = self._load_tensor(self._weight_name("in_proj_z.qzeros"))
        scales = self._load_tensor(self._weight_name("in_proj_z.scales"))

        W_full = dequantize_int4_autogptq(qweight, qzeros, scales)  # [5120, 6144]

        sharded_N = self.value_dim
        start = self.gpu_idx * sharded_N
        end = start + sharded_N
        return W_full[:, start:end]

    def load_out_proj_dequant(self):
        """Dequantize out_proj and shard to GPU. Returns [value_dim_per_gpu, hidden_size]."""
        qweight = self._load_tensor(self._weight_name("out_proj.qweight"))
        qzeros = self._load_tensor(self._weight_name("out_proj.qzeros"))
        scales = self._load_tensor(self._weight_name("out_proj.scales"))

        W_full = dequantize_int4_autogptq(qweight, qzeros, scales)  # [6144, 5120]

        sharded_K = self.value_dim
        start = self.gpu_idx * sharded_K
        end = start + sharded_K
        return W_full[start:end, :]

    def load_bf16_weights(self):
        """Load BF16 weights and shard where needed."""
        result = {}

        # in_proj_a: [full_heads, hidden_size] → [per_gpu_heads, hidden_size]
        w_a = self._load_tensor(self._weight_name("in_proj_a.weight"))
        sharded_K = self.num_v_heads
        start = self.gpu_idx * sharded_K
        end = start + sharded_K
        result["a_weight"] = w_a[start:end, :]

        # in_proj_b: same sharding
        w_b = self._load_tensor(self._weight_name("in_proj_b.weight"))
        result["b_weight"] = w_b[start:end, :]

        # conv1d.weight: [conv_dim_full=10240, 1, kernel_size] → [:conv_dim_per_gpu, 1, K]
        conv_w = self._load_tensor(self._weight_name("conv1d.weight"))
        result["conv_weight"] = conv_w[:self.conv_dim, :, :]

        # A_log: per-head → shard along head dim
        a_log = self._load_tensor(self._weight_name("A_log"))
        result["a_log"] = a_log[start:end]

        # dt_bias: per-head → shard along head dim
        dt_bias = self._load_tensor(self._weight_name("dt_bias"))
        result["dt_bias"] = dt_bias[start:end]

        # norm.weight: [head_v_dim=128], not sharded
        result["norm_weight"] = self._load_tensor(self._weight_name("norm.weight"))

        return result


# =========================================================================
# GDN Reference Computation
# =========================================================================

class GdnReference:
    """Compute full GDN forward pipeline using PyTorch on CPU."""

    def __init__(self, weight_loader: GdnWeightLoader):
        self.wl = weight_loader

        # Load all weights upfront
        self.W_qkv = weight_loader.load_in_proj_qkv_dequant()   # [hidden_size, conv_dim]
        self.W_z = weight_loader.load_in_proj_z_dequant()       # [hidden_size, value_dim]
        self.W_out = weight_loader.load_out_proj_dequant()      # [value_dim, hidden_size]

        bf16 = weight_loader.load_bf16_weights()
        self.W_a = bf16["a_weight"]                             # [num_v_heads, hidden_size]
        self.W_b = bf16["b_weight"]                             # [num_v_heads, hidden_size]
        self.conv_weight = bf16["conv_weight"]                  # [conv_dim, 1, K]
        self.a_log = bf16["a_log"].float()                      # [num_v_heads]
        self.dt_bias = bf16["dt_bias"].float()                  # [num_v_heads]
        self.norm_weight = bf16["norm_weight"].float()          # [head_v_dim]

    def forward(self, hidden_input: torch.Tensor) -> dict:
        """Compute all GDN intermediates from hidden input.

        Args:
            hidden_input: [seq_len, hidden_size] float32 tensor

        Returns:
            Dict of stage_name -> torch.Tensor (float32).
        """
        S = hidden_input.shape[0]  # seq_len
        K_head = self.wl.key_dim   # key_dim_per_gpu
        V_head = self.wl.value_dim # value_dim_per_gpu

        out = {}

        # ── Stage 1: mixed_qkv = hidden_input @ W_qkv ──
        mixed_qkv = hidden_input.float() @ self.W_qkv.float()
        out["mixed_qkv"] = mixed_qkv

        # ── Stage 2: conv_out = depthwise_conv1d_silu(mixed_qkv) ──
        x = mixed_qkv.unsqueeze(0).transpose(1, 2).float()  # [1, conv_dim, S]
        raw_conv = F.conv1d(
            x, self.conv_weight.float(),
            padding=self.wl.conv_kernel_size - 1,
            groups=self.wl.conv_dim,
        )
        conv_out = F.silu(raw_conv[:, :, :S]).transpose(1, 2).squeeze(0)
        out["conv_out"] = conv_out

        # ── Stage 3: split into query, key, value ──
        out["query"] = conv_out[:, :K_head]
        out["key"] = conv_out[:, K_head : 2 * K_head]
        out["value"] = conv_out[:, 2 * K_head :]

        # ── Stage 4: expand query/key to num_v_heads ──
        kv_ratio = self.wl.kv_ratio
        q_shape = (S, self.wl.num_k_heads, self.wl.head_k_dim)
        k_shape = (S, self.wl.num_k_heads, self.wl.head_k_dim)

        query_expanded = out["query"].reshape(q_shape).repeat_interleave(kv_ratio, dim=1)
        key_expanded = out["key"].reshape(k_shape).repeat_interleave(kv_ratio, dim=1)
        out["query_expanded"] = query_expanded
        out["key_expanded"] = key_expanded

        # ── Stage 5: a_proj and b_proj (bf16 GEMM) ──
        out["a_proj"] = hidden_input.float() @ self.W_a.float().T
        out["b_proj"] = hidden_input.float() @ self.W_b.float().T

        # ── Stage 6: GDN recurrent step ──
        value_per_head = out["value"].reshape(S, self.wl.num_v_heads, self.wl.head_v_dim)
        core_attn = self._gdn_recurrent_step(
            query_expanded, key_expanded, value_per_head,
            out["a_proj"], out["b_proj"],
            self.a_log, self.dt_bias,
        )
        out["core_attn_out"] = core_attn

        # ── Stage 7: z_gate (INT4 GEMM) ──
        z_flat = hidden_input.float() @ self.W_z.float()
        out["z_gate"] = z_flat.reshape(S, self.wl.num_v_heads, self.wl.head_v_dim)

        # ── Stage 8: RMSNormGated ──
        core_flat = core_attn.reshape(-1, self.wl.head_v_dim)
        z_flat2 = out["z_gate"].reshape(-1, self.wl.head_v_dim)
        norm_out = self._rms_norm_gated(core_flat, z_flat2, self.norm_weight)
        out["norm_output"] = norm_out.reshape(S, self.wl.num_v_heads, self.wl.head_v_dim)

        # ── Stage 9: output = norm_output @ W_out (row-parallel partial output) ──
        out_final = out["norm_output"].reshape(S, self.wl.value_dim) @ self.W_out.float()
        out["output"] = out_final

        return out

    def forward_kernel_only(
        self, engine_dir: Path
    ) -> dict:
        """Compute kernel-only stages using ENGINE intermediates as inputs.

        This avoids INT4 dequantization entirely by using the engine's own
        dumped intermediates as inputs to each stage. Only BF16 weights are needed.

        Stages verified:
          - conv_out: from engine's mixed_qkv + BF16 conv weight
          - core_attn_out: from engine's query_expanded, key_expanded, value, a_proj, b_proj
              + BF16 A_log, dt_bias
          - norm_output: from engine's core_attn_out, z_gate + BF16 norm weight

        Args:
            engine_dir: Path to engine dump directory

        Returns:
            Dict of stage_name -> torch.Tensor (float32)
        """
        S = self._infer_seq_len(engine_dir)
        K_head = self.wl.key_dim
        V_head = self.wl.value_dim

        out = {}

        # ── Stage 2: conv_out from engine's mixed_qkv ──
        # Use engine's conv1d weight (not model's — they differ in how weights are stored/processed)
        mixed_qkv_path = engine_dir / "mixed_qkv.raw"
        conv_weight_path = engine_dir / "conv1d_weight.raw"
        if mixed_qkv_path.exists() and conv_weight_path.exists():
            mixed_qkv = load_raw_bf16(str(mixed_qkv_path), (S, self.wl.conv_dim))
            # Engine dumps conv weight as [conv_dim, kernel_size] flat
            engine_conv_w = load_raw_bf16(
                str(conv_weight_path), (self.wl.conv_dim, self.wl.conv_kernel_size)
            )
            x = mixed_qkv.unsqueeze(0).transpose(1, 2).float()
            raw_conv = F.conv1d(
                x, engine_conv_w.unsqueeze(1).float(),
                padding=self.wl.conv_kernel_size - 1,
                groups=self.wl.conv_dim,
            )
            conv_out = F.silu(raw_conv[:, :, :S]).transpose(1, 2).squeeze(0)
            out["conv_out"] = conv_out

        # ── Stage 6: core_attn_out from engine's intermediates ──
        qk_paths = {
            "query_expanded": engine_dir / "query_expanded.raw",
            "key_expanded": engine_dir / "key_expanded.raw",
            "value": engine_dir / "value.raw",
            "a_proj": engine_dir / "a_proj.raw",
            "b_proj": engine_dir / "b_proj.raw",
        }

        all_gdn_inputs_ok = True
        gdn_vars = {}
        for name, path in qk_paths.items():
            if path.exists():
                if name == "value":
                    gdn_vars[name] = load_raw_bf16(str(path), (S, V_head))
                elif name in ("a_proj", "b_proj"):
                    gdn_vars[name] = load_raw_bf16(str(path), (S, self.wl.num_v_heads))
                else:
                    shape = (S, self.wl.num_v_heads, self.wl.head_k_dim)
                    gdn_vars[name] = load_raw_bf16(str(path), shape)
            else:
                all_gdn_inputs_ok = False

        if all_gdn_inputs_ok:
            value_per_head = gdn_vars["value"].reshape(
                S, self.wl.num_v_heads, self.wl.head_v_dim
            )
            core_attn = self._gdn_recurrent_step(
                gdn_vars["query_expanded"],
                gdn_vars["key_expanded"],
                value_per_head,
                gdn_vars["a_proj"],
                gdn_vars["b_proj"],
                self.a_log,
                self.dt_bias,
            )
            out["core_attn_out"] = core_attn

        # ── Stage 8: norm_output from engine's core_attn_out + z_gate ──
        core_path = engine_dir / "core_attn_out.raw"
        zgate_path = engine_dir / "z_gate.raw"
        if core_path.exists() and zgate_path.exists():
            core_t = load_raw_bf16(str(core_path), (S, self.wl.num_v_heads, self.wl.head_v_dim))
            z_t = load_raw_bf16(str(zgate_path), (S, self.wl.num_v_heads, self.wl.head_v_dim))
            core_flat = core_t.reshape(-1, self.wl.head_v_dim)
            z_flat = z_t.reshape(-1, self.wl.head_v_dim)
            norm_out = self._rms_norm_gated(core_flat, z_flat, self.norm_weight)
            out["norm_output"] = norm_out.reshape(S, self.wl.num_v_heads, self.wl.head_v_dim)

        return out

    def _infer_seq_len(self, engine_dir: Path) -> int:
        """Infer sequence length from hidden_input dump size."""
        hi_path = engine_dir / "hidden_input.raw"
        n_bf16 = os.path.getsize(hi_path) // 2
        return n_bf16 // self.wl.hidden_size

    def _gdn_recurrent_step(
        self,
        query: torch.Tensor,       # [S, num_v_heads, head_k_dim]
        key: torch.Tensor,         # [S, num_v_heads, head_k_dim]
        value: torch.Tensor,       # [S, num_v_heads, head_v_dim]
        a_proj: torch.Tensor,      # [S, num_v_heads]
        b_proj: torch.Tensor,      # [S, num_v_heads]
        A_log: torch.Tensor,       # [num_v_heads]
        dt_bias: torch.Tensor,     # [num_v_heads]
    ) -> torch.Tensor:
        """GDN recurrence matching the CUDA kernel exactly (gdn_recurrent_step.cu).

        For each token t and head h:
          1. L2-normalize q[h] and k[h], scale q by 1/sqrt(K)
          2. g = -exp(A_log[h]) * softplus(a_proj[t,h] + dt_bias[h])
          3. decay = exp(g), beta = sigmoid(b_proj[t,h])
          4. S[h] *= decay (state decay)
          5. kv_mem = S[h]^T @ k_normed_h
          6. delta = beta * (v[h] - kv_mem)
          7. S[h] += outer(k_normed_h, delta)
          8. output[t,h] = S[h]^T @ q_scaled_h
        """
        S_len, num_heads, K = query.shape
        V = value.shape[2]
        rcp_sqrt_k = 1.0 / (K ** 0.5)

        state = torch.zeros(num_heads, K, V, dtype=torch.float32)
        outputs = torch.zeros(S_len, num_heads, V, dtype=torch.float32)

        for t in range(S_len):
            q_t = query[t]
            k_t = key[t]
            v_t = value[t]
            a_t = a_proj[t]
            b_t = b_proj[t]

            for h in range(num_heads):
                q_h = q_t[h].float()
                k_h = k_t[h].float()
                v_h = v_t[h].float()
                a_h = a_t[h].float()
                b_h = b_t[h].float()

                # L2 normalize query and key, scale query by 1/sqrt(K)
                k_l2 = (k_h.pow(2).sum() + 1e-6).sqrt()
                q_l2 = (q_h.pow(2).sum() + 1e-6).sqrt()
                k_normed = k_h / k_l2
                q_scaled = (q_h / q_l2) * rcp_sqrt_k

                # Compute g and decay
                A_val = torch.exp(A_log[h].float())
                sp_input = a_h + dt_bias[h]

                if sp_input > 20.0:
                    sp_val = sp_input
                elif sp_input < -20.0:
                    sp_val = torch.tensor(0.0)
                else:
                    sp_val = torch.log(torch.tensor(1.0) + torch.exp(sp_input))

                g_val = -A_val * sp_val
                decay = torch.exp(g_val)

                # Beta
                beta = 1.0 / (1.0 + torch.exp(-b_h))

                # State update
                state[h] *= decay
                kv_mem = (state[h].T @ k_normed)
                delta = beta * (v_h - kv_mem)
                state[h] += torch.outer(k_normed, delta)
                outputs[t, h] = state[h].T @ q_scaled

        return outputs

    def _rms_norm_gated(
        self,
        x: torch.Tensor,      # [N, D] where N=seq_len*num_v_heads, D=head_v_dim
        gate: torch.Tensor,   # [N, D]
        weight: torch.Tensor, # [D]
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """RMSNormGated matching Qwen3_5RMSNormGated.

        Formula (from HF source):
          1. variance = x^2.mean(-1)
          2. x_normed = x * rsqrt(variance + eps)
          3. x_weighted = weight * x_normed   (multiplicative, NOT 1+weight)
          4. return x_weighted * silu(gate)    (SiLU, NOT sigmoid)
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        x_weighted = weight.float() * x_normed
        return x_weighted * F.silu(gate.to(torch.float32))


# =========================================================================
# Comparison
# =========================================================================

def compare_stage(engine_dir: Path, stage_name: str, ref_tensor: torch.Tensor) -> dict:
    """Compare a single stage's engine dump against reference."""
    raw_path = engine_dir / f"{stage_name}.raw"
    if not raw_path.exists():
        return {
            "cos": 0.0, "l2_err": 1.0, "max_diff": -1.0,
            "passed": False, "error": f"missing_engine_dump ({stage_name}.raw)",
        }

    ref_shape = ref_tensor.shape
    engine_flat = load_raw_bf16(str(raw_path), (-1,))

    if engine_flat.numel() != ref_tensor.numel():
        return {
            "cos": 0.0, "l2_err": 1.0, "max_diff": -1.0,
            "passed": False,
            "error": f"size_mismatch engine={engine_flat.numel()} ref={ref_tensor.numel()}",
        }

    engine_t = engine_flat.reshape(ref_shape).float()
    cos = cos_sim(engine_t, ref_tensor)
    l2 = l2_error(engine_t, ref_tensor)
    max_diff = (engine_t.float() - ref_tensor.float()).abs().max().item()

    threshold = STAGE_THRESHOLDS.get(stage_name, 0.99)
    passed = cos >= threshold

    return {"cos": cos, "l2_err": l2, "max_diff": max_diff, "passed": passed}


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare engine GDN dumps against PyTorch reference for layer N"
    )
    parser.add_argument("--dump-dir", type=str, default="/tmp/gdn_dump/layer_18")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--gpu-idx", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    engine_dir = Path(args.dump_dir)
    if not engine_dir.exists():
        print(f"[ERROR] Engine dump directory does not exist: {engine_dir}")
        sys.exit(1)

    # ── Print header ──
    print("=" * 70)
    print(f"GDN Layer Compare — Layer {args.layer}, GPU {args.gpu_idx} (TP={args.tp_size})")
    print(f"  Engine dump:   {engine_dir}")
    print(f"  Model path:    {args.model_dir}")
    print("=" * 70)

    # ── Load weights ──
    print("\nLoading model weights...")
    wl = GdnWeightLoader(args.model_dir, args.layer, tp_size=args.tp_size, gpu_idx=args.gpu_idx)
    print(f"  hidden_size:     {wl.hidden_size}")
    print(f"  num_v_heads:     {wl.num_v_heads} (full={wl.num_v_heads_full})")
    print(f"  num_k_heads:     {wl.num_k_heads} (full={wl.num_k_heads_full})")
    print(f"  head_k_dim:      {wl.head_k_dim}")
    print(f"  head_v_dim:      {wl.head_v_dim}")
    print(f"  key_dim:         {wl.key_dim}")
    print(f"  value_dim:       {wl.value_dim}")
    print(f"  conv_dim:        {wl.conv_dim}")
    print(f"  kv_ratio:        {wl.kv_ratio}")
    print(f"  kernel_size:     {wl.conv_kernel_size}")

    # ── Load hidden_input as reference input ──
    hidden_input_path = engine_dir / "hidden_input.raw"
    if not hidden_input_path.exists():
        print(f"[ERROR] hidden_input.raw not found at {hidden_input_path}")
        sys.exit(1)

    n_bf16 = os.path.getsize(hidden_input_path) // 2
    seq_len = n_bf16 // wl.hidden_size
    hidden_input = load_raw_bf16(str(hidden_input_path), (seq_len, wl.hidden_size))
    print(f"\n  Sequence length: {seq_len}")
    print(f"  Hidden input:    shape={list(hidden_input.shape)} mean_abs={hidden_input.abs().mean():.4f}")

    # ── Mode 1: Full pipeline from scratch ──
    print("\n" + "=" * 70)
    print("Mode 1: Full Pipeline (from scratch with dequantized INT4 weights)")
    print("=" * 70)
    print("\nComputing GDN reference intermediates...")
    gdn_ref = GdnReference(wl)
    ref_results = gdn_ref.forward(hidden_input)

    for name, tensor in ref_results.items():
        print(f"  {name:20s}: shape={list(tensor.shape)} dtype={tensor.dtype}")

    print("\nComparing engine vs full-pipeline reference...")
    compare_stages = list(STAGE_THRESHOLDS.keys())
    results_full = {}
    for stage_name in compare_stages:
        if stage_name not in ref_results:
            continue
        result = compare_stage(engine_dir, stage_name, ref_results[stage_name])
        results_full[stage_name] = result

        threshold = STAGE_THRESHOLDS.get(stage_name, 0.99)
        status = "PASS" if result["passed"] else "FAIL"
        extra = ""
        if args.verbose:
            extra = f"  l2_err={result['l2_err']:.6f}  max_diff={result['max_diff']:.6f}"

        print(f"[STAGE] {stage_name:20s}: cos={result['cos']:.6f} l2={result['l2_err']:.6f} "
              f"max_diff={result['max_diff']:.6f} shape=({', '.join(str(s) for s in ref_results[stage_name].shape)}) {status}{extra}")

    # ── Mode 2: Kernel-only (using engine intermediates as inputs) ──
    print("\n" + "=" * 70)
    print("Mode 2: Kernel-Only (using engine intermediates, no INT4 dequant)")
    print("=" * 70)
    print("\nComputing kernel-only reference...")
    ref_kernel = gdn_ref.forward_kernel_only(engine_dir)

    for name, tensor in ref_kernel.items():
        print(f"  {name:20s}: shape={list(tensor.shape)} dtype={tensor.dtype}")

    print("\nComparing engine vs kernel-only reference...")
    results_kernel = {}
    for stage_name in ref_kernel:
        if stage_name not in STAGE_THRESHOLDS:
            continue
        result = compare_stage(engine_dir, stage_name, ref_kernel[stage_name])
        results_kernel[stage_name] = result

        threshold = STAGE_THRESHOLDS.get(stage_name, 0.99)
        status = "PASS" if result["passed"] else "FAIL"
        extra = ""
        if args.verbose:
            extra = f"  l2_err={result['l2_err']:.6f}  max_diff={result['max_diff']:.6f}"

        print(f"[STAGE] {stage_name:20s}: cos={result['cos']:.6f} l2={result['l2_err']:.6f} "
              f"max_diff={result['max_diff']:.6f} shape=({', '.join(str(s) for s in ref_kernel[stage_name].shape)}) {status}{extra}")

    # ── Overall summary ──
    all_passed_full = all(v["passed"] for v in results_full.values()) if results_full else True
    all_passed_kernel = all(v["passed"] for v in results_kernel.values()) if results_kernel else True

    print(f"\n{'='*70}")
    full_pass = sum(1 for v in results_full.values() if v["passed"])
    full_fail = len(results_full) - full_pass
    kern_pass = sum(1 for v in results_kernel.values() if v["passed"])
    kern_fail = len(results_kernel) - kern_pass
    print(f"Full pipeline:  {full_pass} passed, {full_fail} failed out of {len(results_full)} stages")
    print(f"Kernel-only:    {kern_pass} passed, {kern_fail} failed out of {len(results_kernel)} stages")
    if kern_fail > 0:
        failed = [k for k, v in results_kernel.items() if not v["passed"]]
        print(f"  Failed: {', '.join(failed)}")
    print(f"{'='*70}")

    return all_passed_kernel


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
