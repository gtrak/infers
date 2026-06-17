"""MLP reference stages ported from ref_intermediates.py."""

import torch
import torch.nn.functional as F

from tests.compare.config import DumpConfig
from tests.compare.weight_loader import WeightLoader
from tests.compare.stages.base import Stage, _get_input


# Thresholds from ref_intermediates.py STAGE_THRESHOLDS
_MLP_THRESHOLDS = {
    "mlp.norm1": 0.99,
    "mlp.norm2": 0.99,
    "mlp.gate_proj_raw": 0.995,
    "mlp.up_proj_raw": 0.995,
    "mlp.silu": 0.999,
    "mlp.down_raw": 0.99,
    "mlp.down_ar": 0.99,
    "residual.attn": 0.99,
    "residual.mlp": 0.99,
}


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with multiplicative weight (Qwen style: output = x * rsqrt(rms^2 + eps) * weight)."""
    rms = (x.float().pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    return (x / rms) * weight.float().unsqueeze(0)


class Norm1Stage(Stage):
    """RMSNorm of hidden_input with norm1_weight (MLP path)."""

    name = "mlp.norm1"
    threshold = _MLP_THRESHOLDS["mlp.norm1"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        hidden_input = inputs["hidden_input"]
        norm1_w = weights.load_norm1(layer_idx)
        return _rms_norm(hidden_input, norm1_w, config.rms_norm_eps)


class Norm2Stage(Stage):
    """RMSNorm of attn.after_ar with norm2_weight."""

    name = "mlp.norm2"
    threshold = _MLP_THRESHOLDS["mlp.norm2"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        residual_attn = _get_input(inputs, "attn.after_ar", 0)
        norm2_w = weights.load_norm2(layer_idx)
        return _rms_norm(residual_attn, norm2_w, config.rms_norm_eps)


class GateProjStage(Stage):
    """mlp.norm2 @ gate_proj_dequant (TP-sharded column-parallel)."""

    name = "mlp.gate_proj"
    threshold = _MLP_THRESHOLDS["mlp.gate_proj_raw"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        norm2_out = _get_input(inputs, "mlp.norm2", gpu_idx)
        W_gate = weights.load_gate_proj_dequant(layer_idx, config.num_gpus, gpu_idx)
        return norm2_out @ W_gate.float()  # [S, hidden] @ [hidden, sharded_int]


class UpProjStage(Stage):
    """mlp.norm2 @ up_proj_dequant (TP-sharded column-parallel)."""

    name = "mlp.up_proj"
    threshold = _MLP_THRESHOLDS["mlp.up_proj_raw"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        norm2_out = _get_input(inputs, "mlp.norm2", gpu_idx)
        W_up = weights.load_up_proj_dequant(layer_idx, config.num_gpus, gpu_idx)
        return norm2_out @ W_up.float()


class SiluStage(Stage):
    """silu(gate) * up."""

    name = "mlp.silu"
    threshold = _MLP_THRESHOLDS["mlp.silu"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        gate = _get_input(inputs, "mlp.gate_proj", gpu_idx)
        up = _get_input(inputs, "mlp.up_proj", gpu_idx)
        return F.silu(gate.float()) * up.float()


class DownRawStage(Stage):
    """mlp.silu @ down_proj_dequant — pre-AR, single GPU."""

    name = "mlp.down_raw"
    threshold = _MLP_THRESHOLDS["mlp.down_raw"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        silu_out = _get_input(inputs, "mlp.silu", gpu_idx)
        W_down = weights.load_down_proj_dequant(layer_idx, config.num_gpus, gpu_idx)
        return silu_out @ W_down.float()  # [S, sharded_int] @ [sharded_int, hidden]


class DownArStage(Stage):
    """Sum of both GPUs' mlp.down_raw — post-AR (only computed on GPU 0)."""

    name = "mlp.down_ar"
    threshold = _MLP_THRESHOLDS["mlp.down_ar"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        if gpu_idx != 0:
            raise ValueError("DownArStage should only be computed on GPU 0")
        # Sum all GPUs' mlp.down_raw outputs
        result = None
        for g in range(config.num_gpus):
            key = f"mlp.down_raw_gpu{g}"
            if key in inputs:
                if result is None:
                    result = inputs[key]
                else:
                    result = result + inputs[key]
        if result is None:
            raise ValueError("No down_raw inputs found for all-reduce")
        return result


class ResidualAttnStage(Stage):
    """hidden_input + attn.after_ar — combines hidden_input with attention output."""

    name = "residual.attn"
    threshold = _MLP_THRESHOLDS["residual.attn"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        # This stage is the sum of hidden_input and attention output.
        # attn.after_ar comes from the AfterArStage (all-reduce of attn.o_proj).
        hidden_input = inputs["hidden_input"]
        attn_ar = _get_input(inputs, "attn.after_ar", 0)  # GPU 0 only after AR
        if attn_ar is None:
            raise ValueError("attn.after_ar not found for residual_attn computation")
        return hidden_input + attn_ar


class ResidualMlpStage(Stage):
    """residual.attn + mlp.down_ar — final MLP residual."""

    name = "residual.mlp"
    threshold = _MLP_THRESHOLDS["residual.mlp"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        residual_attn = _get_input(inputs, "residual.attn", 0)
        mlp_down = _get_input(inputs, "mlp.down_ar", 0)
        return residual_attn + mlp_down
