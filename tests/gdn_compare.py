#!/usr/bin/env python3
"""Compare engine GDN layer dumps against PyTorch reference.

Usage:
    # Compare existing engine dumps against reference:
    python tests/gdn_compare.py --engine-dir /tmp/gdn_debug_L0

    # Generate reference from model weights + engine input, then compare:
    python tests/gdn_compare.py --engine-dir /tmp/gdn_debug_L0 --model-dir ~/opt/vllm/models/qwen3.6-27b-autoround-int4/

    # Compare all layers from a multi-layer dump:
    python tests/gdn_compare.py --dump-dir /tmp/engine_dump/ --model-dir ~/opt/vllm/models/qwen3.6-27b-autoround-int4/
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


# =========================================================================
# IO helpers
# =========================================================================

def load_raw_bf16(path, shape):
    """Load a .raw bf16 file into a float32 torch tensor with given shape."""
    data = open(path, 'rb').read()
    if len(data) == 0:
        raise ValueError(f"Empty file: {path}")
    n = len(data) // 2
    vals = struct.unpack(f'<{n}H', data)
    arr = np.array(
        [struct.unpack('f', struct.pack('I', v << 16))[0] for v in vals],
        dtype=np.float32,
    )
    return torch.from_numpy(arr.reshape(shape))


def load_npy(path):
    """Load a .npy file as float32 torch tensor."""
    return torch.from_numpy(np.load(path)).float()


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
    diff = (a_f - b_f).norm().item()
    norm = a_f.norm().item()
    return diff / (norm + 1e-30)


def element_stats(a, b):
    """Element-wise diff statistics."""
    diff = (a.float() - b.float()).abs()
    return {
        "max": diff.max().item(),
        "mean": diff.mean().item(),
        "median": diff.median().item(),
    }


def stats_str(t, name="tensor"):
    """Short stats string for a tensor."""
    return f"{name}: shape={list(t.shape)} min={t.min():.4f} max={t.max():.4f} mean_abs={t.abs().mean():.4f}"


# =========================================================================
# INT4 Dequantization (AutoRound/AutoGPTQ format)
# =========================================================================

def unpack_int4(data: torch.Tensor) -> torch.Tensor:
    """Unpack INT4 values from int32-packed tensor.

    Input: [M, N/8] int32 where each int32 packs 8 int4 values.
    Output: [M, N] int8 with each element in [0, 15].
    """
    M, N_packed = data.shape
    N = N_packed * 8
    # Unpack: each int32 holds 8 int4 values, low 4 bytes = 2 int4s per byte
    # Storage: byte 0 = col[1]:col[0] (high:low), byte 1 = col[3]:col[2], etc.
    # So each byte stores 2 int4 values.
    data_u8 = data.view(torch.uint8).reshape(M, N_packed * 4)  # int32 -> 4 x uint8
    # Low 4 bits = first value, high 4 bits = second value
    low = data_u8 & 0x0F
    high = (data_u8 >> 4) & 0x0F
    # Interleave: each original byte contributes [low, high]
    result = torch.zeros(M, N_packed * 8, dtype=torch.uint8)
    result[:, 0::2] = low
    result[:, 1::2] = high
    return result


def dequantize_int4_autogptq(qweight, qzeros, scales, group_size=128):
    """Dequantize AutoRound/AutoGPTQ INT4 weights.

    The format is:
        w_deq = (w_int4 - (zero - 1)) * scale

    where:
        qweight: [out_features, in_features / 8] int32 packed
        qzeros:  [out_features / group_size * out_features_blocks, in_features / 8] int32 packed
        scales:  [out_features / group_size * out_features_blocks or similar, in_features] fp16

    AutoRound stores: zero_point in qzeros is stored as (16 - zero_point).
    So the dequant formula is: w = (q_unpacked - (16 - zero_unpacked)) * scale
    which simplifies to: w = (q_unpacked + zero_unpacked - 16) * scale

    Actually, different quantizers use different conventions. The verified
    engine formula is: w_deq = (w_int4 - (zero + 1)) * scale
    where zero is the unpacked zero point.
    """
    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * 8
    num_groups = out_features // group_size

    # Unpack weights
    w_int4 = unpack_int4(qweight)  # [out_features, in_features]
    z_int4 = unpack_int4(qzeros)   # [out_features // group_size, in_features]

    # Reshape for group-wise dequant
    w_grps = w_int4.reshape(num_groups, group_size, in_features)
    z_grps = z_int4.reshape(num_groups, group_size // group_size, in_features)  # typically just [num_groups, 1, in_features]
    s_grps = scales.reshape(num_groups, group_size // group_size, in_features)  # typically [num_groups, 1, in_features]

    # Broadcast zero/scale to match weight groups
    # z_grps and s_grps have shape [num_groups, 1, in_features]
    # w_grps has shape [num_groups, group_size, in_features]

    # Verified engine formula: w_deq = (w_int4 - (zero_point - 1)) * scale
    # Where zero_point is the stored zero value.
    # AutoRound stores zero_point in qzeros as (16 - zero_point),
    # so actual_zero = 16 - stored_zero, and formula becomes:
    # w_deq = (w_int4 - (16 - stored_zero - 1)) * scale
    #       = (w_int4 + stored_zero - 15) * scale

    # Based on engine verification: w_deq = (w_int4 - (zero_point_raw - 1)) * scale
    w_f32 = (w_grps.float() - (z_grps.float() - 1.0)) * s_grps.float()
    return w_f32.reshape(out_features, in_features)


# =========================================================================
# Reference computation
# =========================================================================

class GdnReference:
    """Compute reference GDN layer outputs using PyTorch."""

    def __init__(self, model_dir, layer_idx, tp_size=2, gpu_idx=0):
        self.model_dir = Path(model_dir)
        self.layer_idx = layer_idx
        self.tp_size = tp_size
        self.gpu_idx = gpu_idx

        # Load model config (handle nested text_config for Qwen3.5)
        with open(self.model_dir / "config.json") as f:
            raw_config = json.load(f)
        # Some models nest the text config under "text_config"
        if "text_config" in raw_config and "hidden_size" in raw_config["text_config"]:
            self.config = raw_config["text_config"]
        else:
            self.config = raw_config

        # Load weight index
        with open(self.model_dir / "model.safetensors.index.json") as f:
            self.weight_idx = json.load(f)["weight_map"]

        # Compute sharded dimensions
        self.hidden_size = self.config["hidden_size"]
        self.num_v_heads_full = self.config["linear_num_value_heads"]  # 48
        self.num_k_heads_full = self.config["linear_num_key_heads"]    # 16
        self.head_k_dim = self.config["linear_key_head_dim"]           # 128
        self.head_v_dim = self.config["linear_value_head_dim"]         # 128
        self.kv_ratio = self.num_v_heads_full // self.num_k_heads_full  # 3

        # Per-GPU sharded dimensions at TP=2
        self.num_v_heads = self.num_v_heads_full // self.tp_size       # 24
        self.num_k_heads = self.num_k_heads_full // self.tp_size       # 8
        self.key_dim = self.num_k_heads * self.head_k_dim              # 1024
        self.value_dim = self.num_v_heads * self.head_v_dim            # 3072
        self.conv_dim = 2 * self.key_dim + self.value_dim              # 5120

    def _load_tensor(self, name):
        """Load a tensor from safetensors, returning float32 torch tensor."""
        fname = self.weight_idx[name]
        path = self.model_dir / fname
        with safe_open(str(path), framework="pt") as f:
            tensor = f.get_tensor(name)
        return tensor.float()

    def _get_weight_name(self, attr):
        """Build full weight name for this layer."""
        return f"model.language_model.layers.{self.layer_idx}.linear_attn.{attr}"

    def _shard_headwise(self, tensor, head_dim=-1):
        """Shard a [full_heads, ...] tensor to [per_gpu_heads, ...]."""
        full_heads = tensor.shape[0]
        per_gpu = full_heads // self.tp_size
        start = self.gpu_idx * per_gpu
        end = start + per_gpu
        if head_dim >= 0:
            # [full_heads, dim] -> [per_gpu, dim]
            return tensor[start:end]
        else:
            # For multi-dim weights, slice first dim
            return tensor[start:end]

    def load_weights(self):
        """Load and return all weights needed for GDN reference."""
        w = {}

        # in_proj_qkv: INT4 quantized
        w["qkv_qweight"] = self._load_tensor(self._get_weight_name("in_proj_qkv.qweight"))
        w["qkv_qzeros"] = self._load_tensor(self._get_weight_name("in_proj_qkv.qzeros"))
        w["qkv_scales"] = self._load_tensor(self._get_weight_name("in_proj_qkv.scales"))

        # in_proj_z: INT4 quantized
        w["z_qweight"] = self._load_tensor(self._get_weight_name("in_proj_z.qweight"))
        w["z_qzeros"] = self._load_tensor(self._get_weight_name("in_proj_z.qzeros"))
        w["z_scales"] = self._load_tensor(self._get_weight_name("in_proj_z.scales"))

        # out_proj: INT4 quantized
        w["out_qweight"] = self._load_tensor(self._get_weight_name("out_proj.qweight"))
        w["out_qzeros"] = self._load_tensor(self._get_weight_name("out_proj.qzeros"))
        w["out_scales"] = self._load_tensor(self._get_weight_name("out_proj.scales"))

        # in_proj_a, in_proj_b: bf16 (not quantized)
        w["a_weight"] = self._load_tensor(self._get_weight_name("in_proj_a.weight"))
        w["b_weight"] = self._load_tensor(self._get_weight_name("in_proj_b.weight"))

        # conv1d: bf16, shape [conv_dim, 1, kernel_size]
        w["conv1d_weight"] = self._load_tensor(self._get_weight_name("conv1d.weight"))
        w["conv1d_bias"] = None  # No conv1d bias in this model

        # A_log and dt_bias: bf16 per-head
        w["a_log"] = self._load_tensor(self._get_weight_name("A_log"))       # [48]
        w["dt_bias"] = self._load_tensor(self._get_weight_name("dt_bias"))   # [48]

        # Norm weight
        w["norm_weight"] = self._load_tensor(self._get_weight_name("norm.weight"))  # [128]

        return w

    def compute_reference_kernels(self, engine_dir):
        """Compute reference for kernel-only stages using ENGINE inputs.

        This avoids INT4 dequantization entirely by using the engine's own
        intermediates as inputs to subsequent stages:

          mixed_qkv (engine) → conv1d ref → conv_out (compare)
          engine q/k/v + a_proj + b_proj + a_log + dt_bias → GDN ref → core_attn_out (compare)
          engine core_attn_out + z_gate + norm_weight → RMSNormGated ref → norm_output (compare)

        Args:
            engine_dir: Path to engine dump directory

        Returns:
            dict of stage_name → reference tensor
        """
        engine_dir = Path(engine_dir)
        ref = {}

        # ---- Load weights (non-GEMM: conv1d, norm, a_log, dt_bias) ----
        w = self.load_weights()
        seq_len = verify_shapes(engine_dir)
        if seq_len is None:
            seq_len = 15

        # conv1d
        conv1d_weight_tp = w["conv1d_weight"][:self.conv_dim]  # [5120, 1, 4]

        # ---- A_log, dt_bias ----
        a_log_tp = self._shard_headwise(w["a_log"])      # [24]
        dt_bias_tp = self._shard_headwise(w["dt_bias"])  # [24]

        # ---- Norm weight ----
        norm_weight = w["norm_weight"]  # [128]

        # =============================================================
        # Stage 1: Load engine's mixed_qkv, run conv1d → compare with engine conv_out
        # =============================================================
        mixed_qkv_path = engine_dir / "mixed_qkv.raw"
        if mixed_qkv_path.exists():
            mixed_qkv = load_raw_bf16(str(mixed_qkv_path), (seq_len, self.conv_dim))
            conv_out = self._conv1d_reference(mixed_qkv, conv1d_weight_tp)
            ref["conv_out"] = conv_out
            ref["_mixed_qkv_source"] = mixed_qkv  # keep for reference

        # =============================================================
        # Stage 2: Load engine's q/k/v, run GDN → compare with engine core_attn_out
        # =============================================================
        # We need: query_expanded, key_expanded, value_flat, a_proj, b_proj
        names_2d = {
            "query_expanded": (seq_len, self.num_v_heads, self.head_k_dim),
            "key_expanded": (seq_len, self.num_v_heads, self.head_k_dim),
            "value_flat": (seq_len, self.num_v_heads, self.head_v_dim),  # actually value.raw
            "a_proj": (seq_len, self.num_v_heads),
            "b_proj": (seq_len, self.num_v_heads),
        }
        engine_vars = {}
        all_gdn_inputs_ok = True
        for name, shape in names_2d.items():
            fname = "value" if name == "value_flat" else name
            p = engine_dir / f"{fname}.raw"
            if p.exists():
                engine_vars[name] = load_raw_bf16(str(p), shape).float()
            else:
                all_gdn_inputs_ok = False

        if all_gdn_inputs_ok:
            core_attn = self._gdn_recurrence(
                engine_vars["query_expanded"], engine_vars["key_expanded"],
                engine_vars["value_flat"],
                engine_vars["a_proj"], engine_vars["b_proj"],
                a_log_tp, dt_bias_tp,
            )
            ref["core_attn_out"] = core_attn

        # =============================================================
        # Stage 3: Load engine's core_attn_out + z_gate, run RMSNormGated
        # =============================================================
        # Note: z_gate from engine dump was computed via INT4 GEMM.
        # RMSNormGated uses z_gate and core_attn_out, both from dump.
        # So this verifies the RMSNormGated kernel itself.
        gdn_path = engine_dir / "core_attn_out.raw"
        zgate_path = engine_dir / "z_gate.raw"
        if gdn_path.exists() and zgate_path.exists():
            core_t = load_raw_bf16(str(gdn_path), (seq_len, self.num_v_heads, self.head_v_dim))
            z_t = load_raw_bf16(str(zgate_path), (seq_len, self.num_v_heads, self.head_v_dim))
            # RMSNormGated operates on [seq_len * num_v_heads, head_v_dim]
            core_flat = core_t.reshape(-1, self.head_v_dim)
            z_flat = z_t.reshape(-1, self.head_v_dim)
            norm_out = self._rms_norm_gated(core_flat, z_flat, norm_weight)
            ref["norm_output"] = norm_out.reshape(seq_len, self.num_v_heads, self.head_v_dim)
            ref["norm_weight"] = norm_weight

        return ref

    def _conv1d_reference(self, mixed_qkv, conv1d_weight):
        """Compute conv1d reference using PyTorch F.conv1d."""
        seq_len, conv_dim = mixed_qkv.shape
        x = mixed_qkv.unsqueeze(0).transpose(1, 2).float()  # [1, conv_dim, seq_len]
        # conv1d_weight: [conv_dim, 1, kernel_size] — already 3D depthwise
        w = conv1d_weight.float()
        raw = F.conv1d(x, w, bias=None,
                       padding=3, groups=conv_dim)
        # Trim padding
        conv = F.silu(raw[:, :, :seq_len])
        return conv.transpose(1, 2).squeeze(0)  # [seq_len, conv_dim]

    def _gdn_recurrence(self, query, key, value, a_proj, b_proj, a_log, dt_bias):
        """GDN recurrence matching the CUDA kernel."""
        seq_len, num_v_heads, head_k_dim = query.shape
        head_v_dim = value.shape[2]
        K = head_k_dim
        V = head_v_dim
        rcp_sqrt_k = 1.0 / (K ** 0.5)

        S = torch.zeros(num_v_heads, K, V, dtype=torch.float32)
        outputs = torch.zeros(seq_len, num_v_heads, V, dtype=torch.float32)

        for t in range(seq_len):
            q = query[t].float()
            k = key[t].float()
            v = value[t].float()
            a = a_proj[t].float()
            b = b_proj[t].float()

            for h in range(num_v_heads):
                q_h, k_h = q[h], k[h]
                v_h, a_h, b_h = v[h], a[h], b[h]

                # L2 normalize
                k_norm = k_h / ((k_h.pow(2).sum() + 1e-6) ** 0.5)
                q_norm = q_h / ((q_h.pow(2).sum() + 1e-6) ** 0.5)
                q_scaled = q_norm * rcp_sqrt_k

                # Decay = exp(-exp(A_log[h]) * softplus(a_proj[h] + dt_bias[h]))
                decay_rate = torch.exp(a_log[h])
                sp = a_h + dt_bias[h]
                if sp > 20.0:
                    sp_val = sp
                elif sp < -20.0:
                    sp_val = 0.0
                else:
                    sp_val = torch.log(1.0 + torch.exp(sp))
                g_val = -decay_rate * sp_val
                decay = torch.exp(g_val)

                # Beta = sigmoid(b_proj[h])
                beta = 1.0 / (1.0 + torch.exp(-b_h))

                # State
                S[h] *= decay
                kv_mem = S[h].T @ k_norm
                delta = beta * (v_h - kv_mem)
                S[h] += torch.outer(k_norm, delta)
                outputs[t, h] = S[h].T @ q_scaled

        return outputs

    def _rms_norm_gated(self, x, gate, weight, eps=1e-6):
        """RMSNormGated: output = rmsnorm(x) * sigmoid(gate) * (1 + weight)."""
        # x: [N, D] or [S*H, V]
        # gate: [N, D]
        # weight: [D]
        rms = (x.float().pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
        x_normed = x.float() / rms
        gate_sigmoid = torch.sigmoid(gate.float())
        return (x_normed * gate_sigmoid * (1.0 + weight.float())).to(x.dtype)


# =========================================================================
# Comparison
# =========================================================================

STAGE_THRESHOLDS = {
    "mixed_qkv": {"cos": 0.99},
    "conv_out": {"cos": 0.999},
    "query": {"cos": 0.99},
    "key": {"cos": 0.99},
    "value": {"cos": 0.99},
    "query_expanded": {"cos": 0.99},
    "key_expanded": {"cos": 0.99},
    "a_proj": {"cos": 0.999},
    "b_proj": {"cos": 0.999},
    "core_attn_out": {"cos": 0.999},
    "z_gate": {"cos": 0.99},
    "norm_output": {"cos": 0.99},
    "output": {"cos": 0.99},
}

# Optional stages that may not always be present
OPTIONAL_STAGES = {"norm_weight"}


def compare_layer(engine_dir, ref_results):
    """Compare engine dumps at engine_dir against reference results.

    Args:
        engine_dir: Path to engine dump directory (e.g. /tmp/gdn_debug_L0)
        ref_results: dict from GdnReference.compute_reference()

    Returns:
        dict of stage_name -> {cos, l2_err, max_diff, passed}
    """
    engine_dir = Path(engine_dir)
    results = {}

    # All stage names in order
    stage_names = [
        "hidden_input", "mixed_qkv", "conv_out",
        "query", "key", "value",
        "query_expanded", "key_expanded",
        "a_proj", "b_proj",
        "core_attn_out",
        "z_gate", "norm_weight", "norm_output",
        "output",
    ]

    for stage_name in stage_names:
        raw_path = engine_dir / f"{stage_name}.raw"
        if not raw_path.exists():
            if stage_name in OPTIONAL_STAGES:
                print(f"  [SKIP] {stage_name}: no engine dump")
                continue
            else:
                print(f"  [MISS] {stage_name}: no engine dump at {raw_path}")
                results[stage_name] = {"cos": 0.0, "l2_err": 1.0, "max_diff": -1,
                                        "passed": False, "error": "missing_engine_dump"}
                continue

        if stage_name not in ref_results:
            results[stage_name] = {"cos": 0.0, "l2_err": 1.0, "max_diff": -1,
                                    "passed": False, "error": "missing_reference"}
            continue

        # Get reference (provides shape and dtype)
        ref_t = ref_results[stage_name].float()
        ref_shape = ref_t.shape

        # Load engine dump — reshape to ref shape (bf16 dump files are flat)
        engine_flat = load_raw_bf16(str(raw_path), (-1,))
        if engine_flat.numel() != ref_t.numel():
            print(f"  [SIZE] {stage_name}: engine={engine_flat.numel()} ref={ref_t.numel()} — SKIP")
            results[stage_name] = {"cos": 0.0, "l2_err": 1.0, "max_diff": -1,
                                    "passed": False, "error": "size_mismatch"}
            continue
        engine_t = engine_flat.reshape(ref_shape).float()

        # Compute metrics
        cos = cos_sim(engine_t, ref_t)
        l2 = l2_error(engine_t, ref_t)
        stats = element_stats(engine_t, ref_t)

        # Check against threshold
        threshold = STAGE_THRESHOLDS.get(stage_name, {"cos": 0.99})
        passed = cos >= threshold["cos"]

        results[stage_name] = {
            "cos": cos,
            "l2_err": l2,
            "max_diff": stats["max"],
            "mean_diff": stats["mean"],
            "passed": passed,
        }

    return results


def print_tiered_results(results, indent=2, verbose=False):
    """Print comparison results grouped by pass/fail."""
    prefix = " " * indent

    passed = {k: v for k, v in results.items() if v.get("passed")}
    failed = {k: v for k, v in results.items() if not v.get("passed")}

    if passed:
        print(f"\n{prefix}[PASS] {len(passed)} stages passed:")
        for name, r in sorted(passed.items()):
            extra = ""
            if verbose:
                extra = f"  l2_err={r['l2_err']:.6f}  max_diff={r['max_diff']:.6f}"
            print(f"{prefix}  {name:20s}  cos={r['cos']:.4f}{extra}")

    if failed:
        print(f"\n{prefix}[FAIL] {len(failed)} stages failed:")
        for name, r in sorted(failed.items()):
            if "error" in r:
                print(f"{prefix}  {name:20s}  ERROR: {r['error']}")
            else:
                extra = f"  l2_err={r['l2_err']:.6f}  max_diff={r['max_diff']:.6f}"
                print(f"{prefix}  {name:20s}  cos={r['cos']:.4f}  (threshold={STAGE_THRESHOLDS.get(name, {}).get('cos', 0.99):.4f}){extra}")

    return len(failed) == 0


# =========================================================================
# Main
# =========================================================================

def get_tensor_shape(file_path, conv_dim=5120):
    """Determine tensor shape from file size and name heuristic."""
    size = os.path.getsize(file_path)
    n_bf16 = size // 2
    fname = Path(file_path).stem

    # Known shapes from engine dump code
    shapes = {
        "hidden_input": lambda n: (n // 5120, 5120),
    }
    if fname in shapes:
        return shapes[fname](n_bf16)
    return (n_bf16,)


def verify_shapes(engine_dir, hidden_size=5120):
    """Load hidden_input to determine seq_len."""
    hidden_path = Path(engine_dir) / "hidden_input.raw"
    if hidden_path.exists():
        n_bf16 = os.path.getsize(hidden_path) // 2
        seq_len = n_bf16 // hidden_size
        return seq_len
    return None


def run_single_layer(engine_dir, model_dir, layer_idx=None, verbose=False):
    """Run full comparison for a single layer.

    Returns True if all stages pass.
    """
    engine_dir = Path(engine_dir)
    if layer_idx is None:
        layer_idx = int(engine_dir.name.split("_")[1]) if "layer_" in engine_dir.name else 0

    print(f"\n{'='*60}")
    print(f"Layer {layer_idx}  ({engine_dir})")
    print(f"{'='*60}")

    # Determine seq_len
    seq_len = verify_shapes(engine_dir)
    if seq_len is None:
        print("  [ERROR] No hidden_input.raw found — cannot determine seq_len")
        return False
    print(f"  Sequence length: {seq_len}")

    # Load hidden input
    hidden = load_raw_bf16(str(engine_dir / "hidden_input.raw"), (seq_len, 5120))
    print(f"  Hidden input: {hidden.shape}  mean_abs={hidden.abs().mean():.4f}")

    # Compute reference using engine intermediates (no INT4 dequantization needed)
    print(f"\n  Loading model weights and computing reference ...")
    try:
        ref = GdnReference(model_dir, layer_idx, tp_size=2, gpu_idx=0)
        ref_results = ref.compute_reference_kernels(engine_dir)
    except Exception as e:
        print(f"  [ERROR] Reference computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare
    print(f"\n  Comparing engine vs reference ...")
    results = compare_layer(engine_dir, ref_results)
    passed = print_tiered_results(results, verbose=verbose)

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Compare engine GDN dumps against PyTorch reference"
    )
    parser.add_argument("--engine-dir", type=str,
                        help="Single layer engine dump (e.g. /tmp/gdn_debug_L0)")
    parser.add_argument("--dump-dir", type=str,
                        help="Multi-layer dump root (contains layer_0/, layer_1/, ...)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index (for --engine-dir without layer_N prefix)")
    parser.add_argument("--model-dir", type=str,
                        default=os.path.expanduser("~/opt/vllm/models/qwen3.6-27b-autoround-int4/"),
                        help="Model weights directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Detailed per-stage output")
    args = parser.parse_args()

    if not args.engine_dir and not args.dump_dir:
        parser.print_help()
        return

    all_passed = True

    if args.engine_dir:
        passed = run_single_layer(args.engine_dir, args.model_dir, args.layer, args.verbose)
        all_passed = all_passed and passed

    if args.dump_dir:
        dump_root = Path(args.dump_dir)
        layer_dirs = sorted(dump_root.glob("layer_*/"))
        if not layer_dirs:
            print(f"No layer directories found in {dump_root}")
            sys.exit(1)
        print(f"Found {len(layer_dirs)} layer directories")
        for ld in layer_dirs:
            passed = run_single_layer(ld, args.model_dir, args.verbose)
            all_passed = all_passed and passed

    if all_passed:
        print(f"\n{'='*60}")
        print("ALL LAYERS PASSED")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("SOME LAYERS FAILED")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
