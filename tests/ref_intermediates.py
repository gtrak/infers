#!/usr/bin/env python3
"""Capture per-suboperation intermediates from PyTorch for a target layer, and save them as raw binary files for cosine comparison with engine dumps.

For each MLP suboperation in the target layer:
1. Load engine's intermediate (from INFERS_DUMP_LAYER_DIR)
2. Compute PyTorch reference using safetensors weights + manual INT4 dequant
3. Compare via cosine similarity
4. Save reference as raw binary (.raw bf16 format, same as engine dumps)

The script avoids loading the full model on GPU (which would compete for VRAM with
the engine and require Triton/GDN CUDA kernels). Instead it loads weights from
safetensors and computes each suboperation manually on CPU.

Usage:
    # Compare against engine dumps at layer 3:
    python tests/ref_intermediates.py --dump-dir /tmp/dump --layer 3

    # Save reference intermediates only (no comparison):
    python tests/ref_intermediates.py --dump-dir /tmp/dump --layer 3 --save-only

    # Specify custom output directory for reference dumps:
    python tests/ref_intermediates.py --dump-dir /tmp/dump --layer 3 --ref-dir /tmp/ref_layer3

Model path is configured via the MODEL_PATH constant or --model-dir argument.
# @lat: [[lat.md/lat#Phase 4 Deliverables#Forward Engine#Prefill Path]]
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
# Configuration
# =========================================================================

DEFAULT_MODEL_PATH = "/home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4"
REF_DIR_SUFFIX = "_ref"  # appended to dump dir name, e.g. /tmp/dump -> /tmp/dump_ref

# Stage thresholds for cosine similarity pass/fail
STAGE_THRESHOLDS = {
    "norm1": {"cos": 0.99},
    "attn_raw": {"cos": 0.99},
    "residual_attn": {"cos": 0.99},
    "norm2": {"cos": 0.99},
    "mlp_gate": {"cos": 0.995},   # INT4 GEMM — allow slightly more error
    "mlp_up": {"cos": 0.995},     # INT4 GEMM — allow slightly more error
    "mlp_silu": {"cos": 0.999},   # elementwise only, should be very close
    "mlp_down": {"cos": 0.99},    # INT4 GEMM + all-reduce
    "residual_mlp": {"cos": 0.99},
}

# Optional stages that may not always have engine dumps
OPTIONAL_STAGES = {
    "attn_raw",      # only for full-attention layers, not GDN layers
    "norm1",         # may be absent depending on dump config
}


# =========================================================================
# IO helpers (same format as engine dumps)
# =========================================================================

def load_raw_bf16(path, shape):
    """Load a .raw bf16 file into a float32 torch tensor with given shape."""
    data = open(path, "rb").read()
    if len(data) == 0:
        raise ValueError(f"Empty file: {path}")
    n = len(data) // 2
    vals = struct.unpack(f"<{n}H", data)
    # Convert bf16 bits to float32 via reinterpret cast
    arr = np.array(
        [struct.unpack("f", struct.pack("I", v << 16))[0] for v in vals],
        dtype=np.float32,
    )
    return torch.Tensor(arr.reshape(shape))


def save_raw_bf16(path, tensor):
    """Save a float32 torch tensor as .raw bf16 file (same format as engine dumps).

    Flattens the tensor and writes little-endian bf16 values.
    """
    t = tensor.float().flatten()
    # Convert f32 to bf16: reinterpret via int32 bit pattern, shift right 16 bits
    f32_arr = t.numpy()
    i32_bits = f32_arr.view(np.int32)
    bf16_bits = (i32_bits >> 16).astype(np.uint16)
    with open(path, "wb") as f:
        f.write(bf16_bits.tobytes())


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
    """Element-wise absolute diff statistics."""
    diff = (a.float() - b.float()).abs()
    return {
        "max": diff.max().item(),
        "mean": diff.mean().item(),
        "median": diff.median().item(),
    }


# =========================================================================
# INT4 Dequantization (AutoRound / GPTQ format)
# =========================================================================

def unpack_int4(data: torch.Tensor) -> torch.Tensor:
    """Unpack INT4 values from int32-packed tensor.

    Input: [M, N/8] int32 where each int32 packs 8 int4 values.
    Output: [M, N] uint8 with each element in [0, 15].
    """
    # Convert to numpy for byte-level manipulation, then back to torch
    M, N_packed = data.shape
    raw_bytes = data.numpy().astype(np.int32).view(np.uint8).reshape(M, N_packed * 4)
    low = raw_bytes & 0x0F
    high = (raw_bytes >> 4) & 0x0F
    result = np.zeros((M, N_packed * 8), dtype=np.uint8)
    result[:, 0::2] = low
    result[:, 1::2] = high
    return torch.Tensor(result.astype(np.float32))


def dequantize_int4_autogptq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize AutoRound/AutoGPTQ INT4 weights.

    The safetensors store transposed weight layout:
        qweight: [K/8, N] for gate_proj/up_proj (transposed from [N, K])
        qzeros:  [num_groups, N/8]
        scales:  [num_groups, N]

    The weight data layout is:
        qweight[k_packed][n] = uint32 packing 8 INT4 values for
        K positions k_packed*8..k_packed*8+7 at output feature n.

    After unpack_int4, the shape is [K_packed, N*8]. We must reshape
    to [K_packed, N, 8] then permute to [K_packed, 8, N] to get the
    correct [K, N] layout where element [k][n] = weight[k][n].

    The dequant formula is: w_deq = (w_int4 - zero_point) * scale
    where zero_point is the stored value in qzeros.

    For gate_proj with hidden_size=5120, intermediate_size=17408:
        qweight shape: [640, 17408] = [K/8, N]
        After correct unpack + permute: [5120, 17408] = [K, N]
    """
    K_packed_dim = qweight.shape[0]
    N_dim = qweight.shape[1]
    K_dim = K_packed_dim * 8

    # Unpack weights: [K/8, N] int32 -> [K_packed, N*8] float
    w_int4 = unpack_int4(qweight)
    z_int4 = unpack_int4(qzeros)

    num_groups = scales.shape[0]
    if K_dim % group_size != 0:
        raise ValueError(f"K={K_dim} not divisible by group_size={group_size}")

    # CRITICAL: Fix the layout from [K_packed, N*8] to [K, N].
    # Each uint32 packed 8 values along K, interleaved with N.
    # Correct: reshape -> permute -> flatten
    #   [K_packed, N, 8] -> [K_packed, 8, N] -> [K, N]
    w_correct = w_int4.reshape(K_packed_dim, N_dim, 8).permute(0, 2, 1).reshape(K_dim, N_dim)
    z_correct = z_int4.reshape(-1, N_dim, 8).permute(0, 2, 1).reshape(-1, N_dim)

    # Reshape for per-group dequant along K axis
    w_grps = w_correct.reshape(num_groups, group_size, N_dim)
    z_grps = z_correct.reshape(num_groups, 1, N_dim)
    s_grps = scales.reshape(num_groups, 1, N_dim)

    # Dequant: (w_int4 - zero_point) * scale
    w_f32 = (w_grps.float() - z_grps.float()) * s_grps.float()
    return w_f32.reshape(K_dim, N_dim)


# =========================================================================
# Weight loading from safetensors
# =========================================================================

class WeightLoader:
    """Load and dequantize model weights from safetensors."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

        with open(self.model_dir / "config.json") as f:
            raw_config = json.load(f)
        # Handle nested text_config for Qwen3.5 models
        if "text_config" in raw_config and "hidden_size" in raw_config["text_config"]:
            self.config = raw_config["text_config"]
        else:
            self.config = raw_config

        with open(self.model_dir / "model.safetensors.index.json") as f:
            self.weight_idx = json.load(f)["weight_map"]

        self.hidden_size = self.config["hidden_size"]
        self.intermediate_size = self.config["intermediate_size"]
        self.rms_norm_eps = float(self.config.get("rms_norm_eps", 1e-6))
        # Check if this is a quantized model
        self.quantized = True

    def _load_tensor(self, name: str) -> torch.Tensor:
        """Load a tensor from safetensors as float32.

        CRITICAL: INT4-packed tensors (qweight, qzeros) have int32 dtype.
        Converting int32 to float32 LOSES LOW BITS for values > 2^24,
        which corrupts the packed INT4 data (99.5% of values affected).
        Keep integer tensors in their original dtype.
        """
        fname = self.weight_idx[name]
        path = self.model_dir / fname
        with safe_open(str(path), framework="pt") as f:
            tensor = f.get_tensor(name)
        # Keep INT4-packed tensors (qweight, qzeros) as-is to avoid precision loss
        if tensor.dtype in (torch.int32, torch.int8, torch.uint8, torch.int64):
            return tensor
        return tensor.float()

    def _get_weight_name(self, layer_idx: int, attr: str) -> str:
        """Build full weight name for a given layer."""
        return f"model.language_model.layers.{layer_idx}.{attr}"

    # --- MLP weights (INT4 quantized) ---

    def load_gate_proj_dequant(self, layer_idx: int, tp_size: int = 2, gpu_idx: int = 0):
        """Dequantize gate_proj weight for a specific TP shard.

        Returns the dequantized weight as [sharded_intermediate, hidden_size].

        For column-parallel at TP=2, each GPU gets half the output features:
            GPU 0: intermediate_size // tp_size rows (first half)
            GPU 1: intermediate_size // tp_size rows (second half)
        """
        group_size = 128
        qweight = self._load_tensor(
            self._get_weight_name(layer_idx, "mlp.gate_proj.qweight")
        )
        qzeros = self._load_tensor(
            self._get_weight_name(layer_idx, "mlp.gate_proj.qzeros")
        )
        scales = self._load_tensor(
            self._get_weight_name(layer_idx, "mlp.gate_proj.scales")
        )

        # Full dequantized: [K=hidden_size, N=intermediate_size]
        W_full = dequantize_int4_autogptq(qweight, qzeros, scales, group_size)

        # Column-parallel sharding: split output (N) dimension by TP
        # W_full is [K=hidden_size, N=intermediate_size], output splits along N
        sharded_intermediate = self.intermediate_size // tp_size
        start = gpu_idx * sharded_intermediate
        end = start + sharded_intermediate
        return W_full[:, start:end]  # [hidden_size, sharded_intermediate] — for norm2 @ W.T

    def load_up_proj_dequant(self, layer_idx: int, tp_size: int = 2, gpu_idx: int = 0):
        """Dequantize up_proj weight for a specific TP shard."""
        group_size = 128
        qweight = self._load_tensor(
            self._get_weight_name(layer_idx, "mlp.up_proj.qweight")
        )
        qzeros = self._load_tensor(
            self._get_weight_name(layer_idx, "mlp.up_proj.qzeros")
        )
        scales = self._load_tensor(
            self._get_weight_name(layer_idx, "mlp.up_proj.scales")
        )

        W_full = dequantize_int4_autogptq(qweight, qzeros, scales, group_size)

        sharded_intermediate = self.intermediate_size // tp_size
        start = gpu_idx * sharded_intermediate
        end = start + sharded_intermediate
        return W_full[:, start:end]

    def load_down_proj_dequant(self, layer_idx: int, tp_size: int = 2, gpu_idx: int = 0):
        """Dequantize down_proj weight for a specific TP shard.

        Returns the dequantized weight as [sharded_intermediate, hidden_size].

        For row-parallel at TP=2, each GPU gets half the input (K) features:
            GPU 0: first sharded_intermediate rows of W
            GPU 1: second sharded_intermediate rows of W
        """
        group_size = 128
        qweight = self._load_tensor(
            self._get_weight_name(layer_idx, "mlp.down_proj.qweight")
        )
        qzeros = self._load_tensor(
            self._get_weight_name(layer_idx, "mlp.down_proj.qzeros")
        )
        scales = self._load_tensor(
            self._get_weight_name(layer_idx, "mlp.down_proj.scales")
        )

        # Full dequantized: [K=intermediate_size, N=hidden_size]
        W_full = dequantize_int4_autogptq(qweight, qzeros, scales, group_size)

        # Row-parallel sharding: split K dimension by TP
        sharded_intermediate = self.intermediate_size // tp_size
        start = gpu_idx * sharded_intermediate
        end = start + sharded_intermediate
        return W_full[start:end]  # [sharded_intermediate, hidden_size]

    def load_norm1_weight(self, layer_idx: int) -> torch.Tensor:
        """Load input_layernorm weight for a layer. [hidden_size]."""
        return self._load_tensor(
            self._get_weight_name(layer_idx, "input_layernorm.weight")
        )

    def load_norm2_weight(self, layer_idx: int) -> torch.Tensor:
        """Load post_attention_layernorm weight for a layer. [hidden_size]."""
        return self._load_tensor(
            self._get_weight_name(layer_idx, "post_attention_layernorm.weight")
        )

    # --- Helper to get sharded dimensions ---

    def sharded_intermediate(self, tp_size: int = 2) -> int:
        return self.intermediate_size // tp_size


# =========================================================================
# Reference MLP computation (CPU)
# =========================================================================

class MlpReference:
    """Compute reference MLP suboperation outputs on CPU."""

    def __init__(self, layer_idx: int, weight_loader: WeightLoader, tp_size: int = 2, gpu_idx: int = 0):
        self.layer_idx = layer_idx
        self.wl = weight_loader
        self.tp_size = tp_size
        self.gpu_idx = gpu_idx
        self.sharded_int = weight_loader.sharded_intermediate(tp_size)

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """RMSNorm with additive weight (Qwen style: output = x * rsqrt(...) * (1 + weight))."""
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
        return x / rms * (1.0 + weight.unsqueeze(0))

    def compute_all(self, engine_dir: Path, gpu_idx: int = 0) -> dict:
        """Compute reference outputs for all MLP suboperations.

        Uses engine dump intermediates as inputs where possible to avoid
        requiring the full forward pass or GDN kernel compatibility.

        Returns dict mapping stage_name -> torch.Tensor (float32).
        """
        ref = {}
        hidden_size = self.wl.hidden_size
        sharded_int = self.sharded_int
        suffix = f"_gpu{gpu_idx}"

        # -------------------------------------------------------------------------
        # 1. Load engine's hidden_input to determine seq_len
        # -------------------------------------------------------------------------
        hidden_input_path = engine_dir / f"hidden_input{suffix}.raw"
        if not hidden_input_path.exists():
            print(f"  [WARN] No hidden_input{suffix}.raw — cannot determine seq_len")
            return ref
        n_bf16 = os.path.getsize(hidden_input_path) // 2
        seq_len = n_bf16 // hidden_size
        print(f"  Sequence length: {seq_len}")

        # -------------------------------------------------------------------------
        # 2. norm1 (from engine's hidden_input via reference RMSNorm)
        # -------------------------------------------------------------------------
        try:
            hidden_input = load_raw_bf16(str(hidden_input_path), (seq_len, hidden_size))
            norm1_weight = self.wl.load_norm1_weight(self.layer_idx)
            ref["norm1"] = self.rms_norm(hidden_input, norm1_weight, self.wl.rms_norm_eps)
            save_raw_bf16(engine_dir / "ref_norm1.raw", ref["norm1"])
        except Exception as e:
            print(f"  [WARN] norm1 reference failed: {e}")

        # -------------------------------------------------------------------------
        # 3. Use engine's residual_attn as input for norm2 (avoids attn path)
        # -------------------------------------------------------------------------
        residual_attn_path = engine_dir / f"residual_attn{suffix}.raw"
        if not residual_attn_path.exists():
            print(f"  [WARN] No residual_attn{suffix}.raw — skipping norm2+")
            return ref

        try:
            residual_attn = load_raw_bf16(str(residual_attn_path), (seq_len, hidden_size))
            norm2_weight = self.wl.load_norm2_weight(self.layer_idx)
            ref["norm2"] = self.rms_norm(residual_attn, norm2_weight, self.wl.rms_norm_eps)
            save_raw_bf16(engine_dir / "ref_norm2.raw", ref["norm2"])
        except Exception as e:
            print(f"  [WARN] norm2 reference failed: {e}")
            return ref

        # -------------------------------------------------------------------------
        # 4. gate_proj: norm2 @ W_gate^T -> [S, sharded_intermediate]
        # -------------------------------------------------------------------------
        try:
            W_gate = self.wl.load_gate_proj_dequant(self.layer_idx, self.tp_size, gpu_idx)
            # W_gate is [hidden_size, sharded_intermediate], compute norm2 @ W_gate
            ref["mlp_gate"] = ref["norm2"] @ W_gate  # [S, hidden] @ [hidden, sharded_int] -> [S, sharded_int]
            save_raw_bf16(engine_dir / "ref_mlp_gate.raw", ref["mlp_gate"])
        except Exception as e:
            print(f"  [WARN] mlp_gate reference failed: {e}")

        # -------------------------------------------------------------------------
        # 5. up_proj: norm2 @ W_up -> [S, sharded_intermediate]
        # -------------------------------------------------------------------------
        try:
            W_up = self.wl.load_up_proj_dequant(self.layer_idx, self.tp_size, gpu_idx)
            ref["mlp_up"] = ref["norm2"] @ W_up  # [S, hidden] @ [hidden, sharded_int] -> [S, sharded_int]
            save_raw_bf16(engine_dir / "ref_mlp_up.raw", ref["mlp_up"])
        except Exception as e:
            print(f"  [WARN] mlp_up reference failed: {e}")

        # -------------------------------------------------------------------------
        # 6. SiLU(gate) * up -> [S, sharded_intermediate]
        # -------------------------------------------------------------------------
        if "mlp_gate" in ref and "mlp_up" in ref:
            try:
                ref["mlp_silu"] = F.silu(ref["mlp_gate"]) * ref["mlp_up"]
                save_raw_bf16(engine_dir / "ref_mlp_silu.raw", ref["mlp_silu"])
            except Exception as e:
                print(f"  [WARN] mlp_silu reference failed: {e}")

        # -------------------------------------------------------------------------
        # 7. down_proj (GPU-local, before all-reduce): silu_out @ W_down
        #    -> [S, hidden_size]
        # -------------------------------------------------------------------------
        if "mlp_silu" in ref:
            try:
                W_down = self.wl.load_down_proj_dequant(self.layer_idx, self.tp_size, gpu_idx)
                # W_down is [sharded_intermediate, hidden_size], compute silu_out @ W_down
                ref["mlp_down_raw"] = ref["mlp_silu"] @ W_down  # [S, sharded_int] @ [sharded_int, hidden] -> [S, hidden]
                save_raw_bf16(engine_dir / "ref_mlp_down_raw.raw", ref["mlp_down_raw"])
            except Exception as e:
                print(f"  [WARN] mlp_down_raw reference failed: {e}")

        # -------------------------------------------------------------------------
        # 8. down_proj (after all-reduce): sum both GPUs' raw outputs
        # -------------------------------------------------------------------------
        if "mlp_down_raw" in ref and gpu_idx == 0:
            try:
                # Load GPU 1's raw down projection from engine dump
                mlp_down_gpu1_path = engine_dir / f"mlp_down_raw_gpu1.raw"
                if mlp_down_gpu1_path.exists():
                    gpu1_raw = load_raw_bf16(str(mlp_down_gpu1_path), (seq_len, hidden_size))
                    ref["mlp_down"] = ref["mlp_down_raw"] + gpu1_raw  # all-reduce = sum
                    save_raw_bf16(engine_dir / "ref_mlp_down.raw", ref["mlp_down"])
                else:
                    print(f"  [WARN] No mlp_down_raw_gpu1.raw — skipping mlp_down (post-AR)")
            except Exception as e:
                print(f"  [WARN] mlp_down reference failed: {e}")

        # -------------------------------------------------------------------------
        # 9. residual_mlp: residual_attn + mlp_down_ar
        # -------------------------------------------------------------------------
        if "mlp_down" in ref:
            try:
                ref["residual_mlp"] = residual_attn + ref["mlp_down"]
                save_raw_bf16(engine_dir / "ref_residual_mlp.raw", ref["residual_mlp"])
            except Exception as e:
                print(f"  [WARN] residual_mlp reference failed: {e}")

        return ref


# =========================================================================
# Comparison
# =========================================================================

def compare_stages(engine_dir: Path, ref_results: dict, tp_size: int = 2) -> dict:
    """Compare engine dumps against reference results.

    Returns dict mapping stage_name -> comparison_result.
    """
    results = {}
    hidden_size_key = "hidden_size"  # resolved later from ref tensor shape

    for gpu_idx in range(tp_size):
        suffix = f"_gpu{gpu_idx}"
        stage_map = {
            "norm1": (f"attn_norm1_gpu{gpu_idx}", f"attn_norm1_gpu{gpu_idx}"),
            # "attn_raw": mapped below per layer type
            "residual_attn": (f"residual_attn_gpu{gpu_idx}", f"residual_attn_gpu{gpu_idx}"),
            "norm2": (f"mlp_norm2_gpu{gpu_idx}", f"mlp_norm2_gpu{gpu_idx}"),
            "mlp_gate": (f"mlp_gate_gpu{gpu_idx}", f"mlp_gate_gpu{gpu_idx}"),
            "mlp_up": (f"mlp_up_gpu{gpu_idx}", f"mlp_up_gpu{gpu_idx}"),
            "mlp_silu": (f"mlp_silu_gpu{gpu_idx}", f"mlp_silu_gpu{gpu_idx}"),
            "mlp_down_raw": (f"mlp_down_raw_gpu{gpu_idx}", f"mlp_down_raw_gpu{gpu_idx}"),
        }

        if gpu_idx == 0:
            # Post-all-reduce stages only on GPU 0
            stage_map["mlp_down"] = ("mlp_down_ar_gpu0", "mlp_down_ar_gpu0")
            stage_map["residual_mlp"] = ("mlp_residual_gpu0", "mlp_residual_gpu0")

        for ref_key, (engine_name, _alias) in stage_map.items():
            if ref_key not in ref_results:
                continue

            engine_path = engine_dir / f"{engine_name}.raw"
            if not engine_path.exists():
                if ref_key in OPTIONAL_STAGES:
                    results[f"{ref_key}_gpu{gpu_idx}"] = {
                        "cos": 0.0, "l2_err": 1.0, "max_diff": -1,
                        "passed": False, "error": "missing_engine_dump",
                    }
                else:
                    results[f"{ref_key}_gpu{gpu_idx}"] = {
                        "cos": 0.0, "l2_err": 1.0, "max_diff": -1,
                        "passed": False, "error": f"missing_engine_dump ({engine_name}.raw)",
                    }
                continue

            # Load engine dump and reshape to ref shape
            ref_t = ref_results[ref_key].float()
            engine_flat = load_raw_bf16(str(engine_path), (-1,))

            if engine_flat.numel() != ref_t.numel():
                results[f"{ref_key}_gpu{gpu_idx}"] = {
                    "cos": 0.0, "l2_err": 1.0, "max_diff": -1,
                    "passed": False,
                    "error": f"size_mismatch engine={engine_flat.numel()} ref={ref_t.numel()}",
                }
                continue

            engine_t = engine_flat.reshape(ref_t.shape).float()

            cos = cos_sim(engine_t, ref_t)
            l2 = l2_error(engine_t, ref_t)
            stats = element_stats(engine_t, ref_t)
            threshold = STAGE_THRESHOLDS.get(ref_key, {"cos": 0.99})
            passed = cos >= threshold["cos"]

            results[f"{ref_key}_gpu{gpu_idx}"] = {
                "cos": cos,
                "l2_err": l2,
                "max_diff": stats["max"],
                "mean_diff": stats["mean"],
                "passed": passed,
            }

    return results


def print_results(results: dict, verbose: bool = False):
    """Print comparison results grouped by pass/fail."""
    passed = {k: v for k, v in results.items() if v.get("passed")}
    failed = {k: v for k, v in results.items() if not v.get("passed")}

    if passed:
        print(f"\n[PASS] {len(passed)} stages passed:")
        for name, r in sorted(passed.items()):
            extra = ""
            if verbose:
                extra = f"  l2_err={r['l2_err']:.6f}  max_diff={r['max_diff']:.6f}"
            print(f"  {name:30s}  cos={r['cos']:.6f}{extra}")

    if failed:
        print(f"\n[FAIL] {len(failed)} stages failed:")
        for name, r in sorted(failed.items()):
            if "error" in r:
                print(f"  {name:30s}  ERROR: {r['error']}")
            else:
                extra = f"  l2_err={r['l2_err']:.6f}  max_diff={r['max_diff']:.6f}"
                threshold = STAGE_THRESHOLDS.get(name.split("_gpu")[0], {}).get("cos", 0.99)
                print(f"  {name:30s}  cos={r['cos']:.6f}  (threshold={threshold:.4f}){extra}")

    return len(failed) == 0


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare engine MLP intermediates with PyTorch reference"
    )
    parser.add_argument("--dump-dir", type=str, required=True,
                        help="Engine dump directory containing layer_N/ subdirs")
    parser.add_argument("--layer", type=int, required=True,
                        help="Target layer index (e.g. 3 for first full attention layer)")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_PATH,
                        help="Model weights directory with safetensors")
    parser.add_argument("--tp-size", type=int, default=2,
                        help="Tensor parallel size (default: 2)")
    parser.add_argument("--gpu-idx", type=int, default=0,
                        help="GPU index for reference computation (default: 0)")
    parser.add_argument("--ref-dir", type=str, default=None,
                        help="Directory to save reference intermediates (default: dump_dir + _ref suffix)")
    parser.add_argument("--save-only", action="store_true",
                        help="Only compute and save reference intermediates, skip comparison")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output with per-stage statistics")
    args = parser.parse_args()

    engine_dir = Path(args.dump_dir) / f"layer_{args.layer}"
    if not engine_dir.exists():
        print(f"[ERROR] Engine dump directory does not exist: {engine_dir}")
        sys.exit(1)

    # Set up reference output directory
    if args.ref_dir:
        ref_dir = Path(args.ref_dir)
    else:
        ref_dir = Path(args.dump_dir).parent / (Path(args.dump_dir).name + REF_DIR_SUFFIX) / f"layer_{args.layer}"

    print("=" * 70)
    print(f"Ref Intermediates — Layer {args.layer}, GPU {args.gpu_idx}")
    print(f"  Engine dump:   {engine_dir}")
    print(f"  Reference dir: {ref_dir}")
    print(f"  Model path:    {args.model_dir}")
    print("=" * 70)

    # Load weights
    print("\nLoading model weights...")
    try:
        wl = WeightLoader(args.model_dir)
        print(f"  hidden_size:     {wl.hidden_size}")
        print(f"  intermediate_size: {wl.intermediate_size}")
        print(f"  rms_norm_eps:    {wl.rms_norm_eps}")
        print(f"  sharded_intermediate (TP={args.tp_size}): {wl.sharded_intermediate(args.tp_size)}")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Compute reference intermediates
    print("\nComputing reference MLP intermediates...")
    mlp_ref = MlpReference(args.layer, wl, tp_size=args.tp_size, gpu_idx=args.gpu_idx)

    try:
        ref_results = mlp_ref.compute_all(engine_dir, gpu_idx=args.gpu_idx)
    except Exception as e:
        print(f"[ERROR] Reference computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nComputed {len(ref_results)} reference intermediates:")
    for name, tensor in ref_results.items():
        print(f"  {name:20s}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    # Save only mode
    if args.save_only:
        print("\n[INFO] Save-only mode — skipping comparison")
        return

    # Compare against engine dumps
    print("\nComparing engine vs reference...")
    results = compare_stages(engine_dir, ref_results, tp_size=args.tp_size)

    all_passed = print_results(results, verbose=args.verbose)

    if all_passed:
        print(f"\n{'='*70}")
        print("ALL STAGES PASSED")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("SOME STAGES FAILED — check which suboperation shows divergence")
        print(f"{'='*70}")
        sys.exit(1)


if __name__ == "__main__":
    main()
