"""MLP reference stages ported from ref_intermediates.py."""

import torch
import torch.nn.functional as F

from tests.compare.config import DumpConfig
from tests.compare.weight_loader import WeightLoader
from tests.compare.stages.base import Stage


# Thresholds from ref_intermediates.py STAGE_THRESHOLDS
_MLP_THRESHOLDS = {
    "norm1": 0.99,
    "norm2": 0.99,
    "gate_proj_raw": 0.995,
    "up_proj_raw": 0.995,
    "silu": 0.999,
    "down_raw": 0.99,
    "down_ar": 0.99,
    "residual_attn": 0.99,
    "residual_mlp": 0.99,
}


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with additive weight (Qwen style: output = x * rsqrt(rms^2 + eps) * (1 + weight))."""
    rms = (x.float().pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    return (x / rms) * (1.0 + weight.float().unsqueeze(0))


class Norm1Stage(Stage):
    """RMSNorm of hidden_input with norm1_weight."""

    name = "norm1"
    threshold = _MLP_THRESHOLDS["norm1"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        hidden_input = inputs["hidden_input"]
        norm1_w = weights.load_norm1(layer_idx)
        return _rms_norm(hidden_input, norm1_w, config.rms_norm_eps)


class Norm2Stage(Stage):
    """RMSNorm of residual_attn with norm2_weight."""

    name = "norm2"
    threshold = _MLP_THRESHOLDS["norm2"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        residual_attn = inputs["residual_attn"]
        norm2_w = weights.load_norm2(layer_idx)
        return _rms_norm(residual_attn, norm2_w, config.rms_norm_eps)


class GateProjStage(Stage):
    """norm2 @ gate_proj_dequant (TP-sharded column-parallel)."""

    name = "gate_proj_raw"
    threshold = _MLP_THRESHOLDS["gate_proj_raw"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        norm2_out = inputs["norm2"]
        W_gate = weights.load_gate_proj_dequant(layer_idx, config.num_gpus, gpu_idx)
        return norm2_out @ W_gate.float()  # [S, hidden] @ [hidden, sharded_int]


class UpProjStage(Stage):
    """norm2 @ up_proj_dequant (TP-sharded column-parallel)."""

    name = "up_proj_raw"
    threshold = _MLP_THRESHOLDS["up_proj_raw"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        norm2_out = inputs["norm2"]
        W_up = weights.load_up_proj_dequant(layer_idx, config.num_gpus, gpu_idx)
        return norm2_out @ W_up.float()


class SiluStage(Stage):
    """silu(gate) * up."""

    name = "silu"
    threshold = _MLP_THRESHOLDS["silu"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        gate = inputs["gate_proj_raw"]
        up = inputs["up_proj_raw"]
        return F.silu(gate.float()) * up.float()


class DownRawStage(Stage):
    """silu_out @ down_proj_dequant — pre-AR, single GPU."""

    name = "down_raw"
    threshold = _MLP_THRESHOLDS["down_raw"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        silu_out = inputs["silu"]
        W_down = weights.load_down_proj_dequant(layer_idx, config.num_gpus, gpu_idx)
        return silu_out @ W_down.float()  # [S, sharded_int] @ [sharded_int, hidden]


class DownArStage(Stage):
    """Sum of both GPUs' down_raw — post-AR (only computed on GPU 0)."""

    name = "down_ar"
    threshold = _MLP_THRESHOLDS["down_ar"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        if gpu_idx != 0:
            raise ValueError("DownArStage should only be computed on GPU 0")
        # Sum all GPUs' down_raw outputs
        result = None
        for g in range(config.num_gpus):
            key = f"down_raw_gpu{g}"
            if key in inputs:
                if result is None:
                    result = inputs[key]
                else:
                    result = result + inputs[key]
        if result is None:
            raise ValueError("No down_raw inputs found for all-reduce")
        return result


class ResidualAttnStage(Stage):
    """input + attn_ar — combines hidden_input with attention output."""

    name = "residual_attn"
    threshold = _MLP_THRESHOLDS["residual_attn"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        # This stage is the sum of hidden_input and attention output.
        # We expect both to be available from previous stages.
        hidden_input = inputs["hidden_input"]
        attn_ar = inputs.get("attn_o_proj")  # After all-reduce from attention path
        if attn_ar is None:
            raise ValueError("attn_o_proj not found for residual_attn computation")
        return hidden_input + attn_ar


class ResidualMlpStage(Stage):
    """residual_attn + mlp_ar — final MLP residual."""

    name = "residual_mlp"
    threshold = _MLP_THRESHOLDS["residual_mlp"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        residual_attn = inputs["residual_attn"]
        mlp_down = inputs["down_ar"]
        return residual_attn + mlp_down
