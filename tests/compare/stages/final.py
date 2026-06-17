"""Final norm and LM head reference stages."""

import torch

from tests.compare.config import DumpConfig
from tests.compare.weight_loader import WeightLoader
from tests.compare.stages.base import Stage


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with additive weight (Qwen style: output = x * rsqrt(rms^2 + eps) * (1 + weight))."""
    rms = (x.float().pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    return (x / rms) * (1.0 + weight.float().unsqueeze(0))


class FinalNormStage(Stage):
    """RMSNorm of final hidden state with final_norm_weight."""

    name = "final_norm"
    threshold = 0.99  # RMSNorm should be very close

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        final_hidden = inputs["residual_mlp"]  # output from last layer's residual_mlp
        final_norm_w = weights.load_final_norm()
        return _rms_norm(final_hidden, final_norm_w, config.rms_norm_eps)


class LogitsStage(Stage):
    """norm_out @ lm_head_dequant (full vocab, TP-sharded column-parallel).

    LM head is replicated across GPUs in practice, but we compare per-GPU
    shard against the engine's sharded dump.
    """

    name = "logits"
    threshold = 0.99  # INT4 GEMM allows some error

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        norm_out = inputs["final_norm"]
        W_lm = weights.load_lm_head_dequant(config.num_gpus, gpu_idx)
        return norm_out @ W_lm.float()  # [S, hidden] @ [hidden, vocab//tp] -> [S, vocab//tp]
