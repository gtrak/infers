"""Full-attention reference stages for Qwen3.5 attention forward.

Port of the HuggingFace Qwen3.5 attention forward (modeling_qwen3_5.py lines 657-714).

Forward pass summary:
    1. Q projection: norm1_out @ q_proj → [S, num_heads * head_dim * 2] (doubled for gate)
    2. Split Q+gate from interleaved layout [Q_h0, G_h0, Q_h1, G_h1, ...] → query_states + gate
    3. Q-norm: per-head RMSNorm(query_states)
    4. K projection: norm1_out @ k_proj → [S, num_kv_heads * head_dim]
    5. K-norm: per-head RMSNorm(key_states)
    6. V projection: norm1_out @ v_proj → [S, num_kv_heads * head_dim]
    7. RoPE on Q and K (GQA — repeat KV heads to match query heads)
    8. Attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
    9. Gate: attn_output * sigmoid(gate)
    10. O-projection: gated_attn @ o_proj → [S, hidden_size]
"""

import math

import torch
import torch.nn.functional as F

from tests.compare.config import DumpConfig
from tests.compare.weight_loader import WeightLoader
from tests.compare.stages.base import Stage

# Attention thresholds — tighter for elementwise/norm stages, looser for INT4 GEMM
_ATTN_THRESHOLDS = {
    "attn.norm1": 0.99,
    "attn.q_proj_raw": 0.995,
    "attn.q_norm": 0.999,
    "attn.gate": 0.995,
    "attn.k_proj": 0.995,
    "attn.k_norm": 0.999,
    "attn.v_proj": 0.995,
    "attn.combined": 0.99,     # INT4 GEMM through attention kernel
    "attn.gated": 0.99,
    "attn.o_proj": 0.99,
}


def _per_head_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm over the last dimension (head_dim).

    Expects x shape [S, num_heads, head_dim] and weight shape [head_dim].
    Uses multiplicative weight style: output = x * rsqrt(rms^2 + eps) * weight.
    """
    rms = (x.float().pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    return (x / rms) * weight.float().unsqueeze(0).unsqueeze(0)


def _apply_rope(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    partial_rotary_factor: float,
    head_dim: int,
) -> torch.Tensor:
    """Apply rotary position embedding.

    Expects:
        x: [S, num_heads, head_dim]
        cos_cache: [1, 1, S, head_dim] — precomputed from rope_theta
        sin_cache: [1, 1, S, head_dim]

    Only the first partial_rotary_factor * head_dim dimensions get rotated.
    """
    dim = int(head_dim * partial_rotary_factor)
    x_rot = x[..., :dim]
    x_pass = x[..., dim:]

    # cos_cache and sin_cache have shape [1, 1, S, head_dim]
    # We need to broadcast against [S, num_heads, dim]
    # Squeeze batch dims [1,1,S,dim] → [S,dim], then add head broadcast dim → [S,1,dim]
    cos = cos_cache.squeeze(0).squeeze(0)[:, :dim].unsqueeze(1).expand_as(x_rot)
    sin = sin_cache.squeeze(0).squeeze(0)[:, :dim].unsqueeze(1).expand_as(x_rot)

    # Rotate: x_rot_out = rotate_half(x_rot) * sin + x_rot * cos
    x1, x2 = x_rot.chunk(2, dim=-1)
    x_rot_out = torch.cat([-x2, x1], dim=-1) * sin + x_rot * cos

    return torch.cat([x_rot_out, x_pass], dim=-1)


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    rope_theta: float,
    partial_rotary_factor: float,
) -> tuple:
    """Build rotary position embedding cache.

    Returns (cos_cache, sin_cache) each with shape [1, 1, S, head_dim].
    """
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))
    # Positional IDs for prefill: [0, 1, ..., seq_len-1]
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = positions[:, None] @ inv_freq[None, :]  # [S, dim/2]

    emb = torch.cat([freqs, freqs], dim=-1)  # [S, dim]
    cos_cache = emb.cos().unsqueeze(0).unsqueeze(0)   # [1, 1, S, dim]
    sin_cache = emb.sin().unsqueeze(0).unsqueeze(0)   # [1, 1, S, dim]

    # Pad to full head_dim (non-rotated part gets cos=1, sin=0 implicitly via concat)
    if dim < head_dim:
        pad_size = head_dim - dim
        cos_pad = torch.ones(1, 1, seq_len, pad_size, dtype=torch.float32)
        sin_pad = torch.zeros(1, 1, seq_len, pad_size, dtype=torch.float32)
        cos_cache = torch.cat([cos_cache, cos_pad], dim=-1)
        sin_cache = torch.cat([sin_cache, sin_pad], dim=-1)

    return cos_cache, sin_cache


def _gqa_repeat_kv(
    x: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> torch.Tensor:
    """Repeat KV heads to match query head count (GQA).

    Expects x shape [S, num_key_value_heads, head_dim].
    Returns shape [S, num_attention_heads, head_dim].
    """
    # Interleave: each KV head is repeated for the corresponding group of query heads
    if num_attention_heads == num_key_value_heads:
        return x

    repeat_factor = num_attention_heads // num_key_value_heads
    x_unsqueezed = x.unsqueeze(2)  # [S, kv_heads, 1, head_dim]
    x_expanded = x_unsqueezed.expand(-1, -1, repeat_factor, -1)  # [S, kv_heads, repeat, head_dim]
    return x_expanded.reshape(-1, num_attention_heads, x.shape[-1])  # [S, attn_heads, head_dim]


def _scaled_dot_product_attention(
    q: torch.Tensor,   # [S, num_heads, head_dim]
    k: torch.Tensor,   # [S, num_heads, head_dim]
    v: torch.Tensor,   # [S, num_heads, head_dim]
    head_dim: int,
) -> torch.Tensor:
    """Scaled dot-product attention for prefill (no sliding window mask needed).

    softmax(Q @ K^T / sqrt(head_dim)) @ V

    Returns [S, num_heads, head_dim].
    """
    scale = 1.0 / math.sqrt(head_dim)
    # Q @ K^T: [S, H, D] @ [S, H, D]^T -> [S, H, H]
    scores = torch.einsum("sah,sbh->sab", q.float(), k.float()) * scale
    # Softmax over key dimension (last dim)
    attn_weights = F.softmax(scores, dim=-1)
    # @V: [S, H, H] @ [S, H, D] -> [S, H, D]
    return torch.einsum("sab,sbd->sad", attn_weights.float(), v.float())


class Norm1InputStage(Stage):
    """Load the engine's hidden_input as reference input for attention stages."""

    name = "attn.norm1_input"
    threshold = _ATTN_THRESHOLDS["attn.norm1"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        # Already loaded by the CLI — just pass through
        return inputs["hidden_input"]


class Norm1Stage(Stage):
    """RMSNorm of hidden_input with norm1_weight (attention path)."""

    name = "attn.norm1"
    threshold = _ATTN_THRESHOLDS["attn.norm1"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        from tests.compare.stages.mlp import _rms_norm
        hidden_input = inputs["hidden_input"]
        norm1_w = weights.load_norm1(layer_idx)
        return _rms_norm(hidden_input, norm1_w, config.rms_norm_eps)


class QProjRawStage(Stage):
    """attn.norm1 @ q_proj_dequant (TP-sharded).

    Full output includes both Q and gate dimensions:
        [S, num_heads * head_dim * 2] split as [S, per_gpu_heads * head_dim * 2].
    """

    name = "attn.q_proj_raw"
    threshold = _ATTN_THRESHOLDS["attn.q_proj_raw"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        norm1_out = inputs["attn.norm1"]
        W_q = weights.load_q_proj_dequant(layer_idx, config.num_gpus, gpu_idx)
        return norm1_out @ W_q.float()


class QNormStage(Stage):
    """Extract Q portion from attn.q_proj_raw, apply per-head RMSNorm."""

    name = "attn.q_norm"
    threshold = _ATTN_THRESHOLDS["attn.q_norm"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        q_proj_raw = inputs["attn.q_proj_raw"]
        num_heads_per_gpu = config.num_attention_heads // config.num_gpus
        head_dim = config.head_dim

        # Extract Q from interleaved layout: [Q_h0, G_h0, Q_h1, G_h1, ...]
        # Each head's Q block is at offset h * (head_dim * 2) within the row.
        query_flat = torch.zeros(
            q_proj_raw.shape[0], num_heads_per_gpu * head_dim, device=q_proj_raw.device
        )
        for h in range(num_heads_per_gpu):
            src_start = h * (head_dim * 2)
            dst_start = h * head_dim
            query_flat[..., dst_start:dst_start + head_dim] = q_proj_raw[..., src_start:src_start + head_dim]

        # Reshape for per-head norm: [S, num_heads_per_gpu, head_dim]
        query = query_flat.view(-1, num_heads_per_gpu, head_dim)

        q_norm_w = weights.load_q_norm(layer_idx)
        return _per_head_rms_norm(query, q_norm_w, config.rms_norm_eps)


class GateStage(Stage):
    """Extract gate portion from attn.q_proj_raw."""

    name = "attn.gate"
    threshold = _ATTN_THRESHOLDS["attn.gate"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        q_proj_raw = inputs["attn.q_proj_raw"]
        num_heads_per_gpu = config.num_attention_heads // config.num_gpus
        head_dim = config.head_dim

        # Extract gate from interleaved layout: [Q_h0, G_h0, Q_h1, G_h1, ...]
        # Each head's gate block is at offset h * (head_dim * 2) + head_dim.
        gate_flat = torch.zeros(
            q_proj_raw.shape[0], num_heads_per_gpu * head_dim, device=q_proj_raw.device
        )
        for h in range(num_heads_per_gpu):
            src_start = h * (head_dim * 2) + head_dim
            dst_start = h * head_dim
            gate_flat[..., dst_start:dst_start + head_dim] = q_proj_raw[..., src_start:src_start + head_dim]

        return gate_flat


class KProjStage(Stage):
    """attn.norm1 @ k_proj_dequant (TP-sharded)."""

    name = "attn.k_proj"
    threshold = _ATTN_THRESHOLDS["attn.k_proj"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        norm1_out = inputs["attn.norm1"]
        W_k = weights.load_k_proj_dequant(layer_idx, config.num_gpus, gpu_idx)
        return norm1_out @ W_k.float()


class KNormStage(Stage):
    """Apply per-head RMSNorm to K."""

    name = "attn.k_norm"
    threshold = _ATTN_THRESHOLDS["attn.k_norm"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        k_proj_raw = inputs["attn.k_proj"]
        num_kv_heads_per_gpu = config.num_key_value_heads // config.num_gpus
        head_dim = config.head_dim

        key_flat = k_proj_raw.view(-1, num_kv_heads_per_gpu, head_dim)
        k_norm_w = weights.load_k_norm(layer_idx)
        return _per_head_rms_norm(key_flat, k_norm_w, config.rms_norm_eps)


class VProjStage(Stage):
    """attn.norm1 @ v_proj_dequant (TP-sharded)."""

    name = "attn.v_proj"
    threshold = _ATTN_THRESHOLDS["attn.v_proj"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        norm1_out = inputs["attn.norm1"]
        W_v = weights.load_v_proj_dequant(layer_idx, config.num_gpus, gpu_idx)
        return norm1_out @ W_v.float()


class AttentionCombinedStage(Stage):
    """Full attention computation (Q@K^T/sqrt(d), softmax, @V).

    Compares against engine's attn.combined dump. This is a GPU-local stage
    that does NOT yet all-reduce across GPUs — each GPU computes attention
    only over its own shard of heads.
    """

    name = "attn.combined"
    threshold = _ATTN_THRESHOLDS["attn.combined"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        seq_len = inputs["hidden_input"].shape[0]
        q_normed = inputs["attn.q_norm"]
        k_normed = inputs["attn.k_norm"]
        v_proj_raw = inputs["attn.v_proj"]

        num_heads_per_gpu = config.num_attention_heads // config.num_gpus
        num_kv_heads_per_gpu = config.num_key_value_heads // config.num_gpus
        head_dim = config.head_dim

        # V needs reshape: [S, num_kv_heads * head_dim] -> [S, num_kv_heads, head_dim]
        v = v_proj_raw.view(-1, num_kv_heads_per_gpu, head_dim)

        # Build RoPE cache
        cos_cache, sin_cache = _build_rope_cache(
            seq_len, head_dim, config.rope_theta, config.partial_rotary_factor
        )

        # Apply RoPE to Q and K (per-GPU shard)
        q_rope = _apply_rope(q_normed, cos_cache, sin_cache, config.partial_rotary_factor, head_dim)
        k_rope = _apply_rope(k_normed, cos_cache, sin_cache, config.partial_rotary_factor, head_dim)

        # GQA: repeat KV heads to match query heads (per-GPU)
        k_expanded = _gqa_repeat_kv(k_rope, num_heads_per_gpu, num_kv_heads_per_gpu)
        v_expanded = _gqa_repeat_kv(v, num_heads_per_gpu, num_kv_heads_per_gpu)

        # Scaled dot-product attention
        combined = _scaled_dot_product_attention(q_rope, k_expanded, v_expanded, head_dim)
        return combined


class GatedStage(Stage):
    """combined * sigmoid(gate) — gated attention output.

    The gate was extracted from the Q projection and reshaped to match the
    attention output shape: [S, num_heads_per_gpu, head_dim].
    """

    name = "attn.gated"
    threshold = _ATTN_THRESHOLDS["attn.gated"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        combined = inputs["attn.combined"]
        gate_flat = inputs["attn.gate"]

        # Reshape gate to match attention output: [S, num_heads_per_gpu, head_dim]
        num_heads_per_gpu = config.num_attention_heads // config.num_gpus
        head_dim = config.head_dim
        gate = gate_flat.view(-1, num_heads_per_gpu, head_dim)

        return combined * torch.sigmoid(gate.float())


class OProjStage(Stage):
    """gated @ o_proj_dequant (TP-sharded row-parallel).

    Compares against engine's attn.o_proj dump.
    """

    name = "attn.o_proj"
    threshold = _ATTN_THRESHOLDS["attn.o_proj"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        gated = inputs["attn.gated"]
        num_heads_per_gpu = config.num_attention_heads // config.num_gpus
        head_dim = config.head_dim
        per_gpu_attn_dim = num_heads_per_gpu * head_dim

        # Flatten: [S, num_heads_per_gpu, head_dim] -> [S, per_gpu_attn_dim]
        gated_flat = gated.view(-1, per_gpu_attn_dim)

        W_o = weights.load_o_proj_dequant(layer_idx, config.num_gpus, gpu_idx)
        return gated_flat @ W_o.float()  # [S, hidden_size] — row-parallel partial output

class AfterArStage(Stage):
    """Sum of both GPUs' attn.o_proj — post all-reduce (only computed on GPU 0)."""
    name = "attn.after_ar"
    threshold = _ATTN_THRESHOLDS["attn.o_proj"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        if gpu_idx != 0:
            raise ValueError("AfterArStage should only be computed on GPU 0")
        result = None
        for g in range(config.num_gpus):
            key = f"attn.o_proj_gpu{g}" if g > 0 else "attn.o_proj"
            if key in inputs:
                if result is None:
                    result = inputs[key]
                else:
                    result = result + inputs[key]
        if result is None:
            raise ValueError("No attn.o_proj inputs found for all-reduce")
        return result
