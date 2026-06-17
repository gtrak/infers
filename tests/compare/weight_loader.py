"""TP-aware weight loader for loading and dequantizing model weights from safetensors."""

import json
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open

from tests.compare.dequant import dequantize_int4_autogptq


class WeightLoader:
    """Load and dequantize model weights from safetensors.

    Handles both full-attention layers (self_attn.*) and MLP layers (mlp.*),
    with TP-aware sharding for column-parallel and row-parallel projections.
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

        with open(self.model_dir / "config.json") as f:
            raw_config = json.load(f)
        # Handle nested text_config for Qwen3.x models
        if "text_config" in raw_config and "hidden_size" in raw_config["text_config"]:
            self.config: Dict = raw_config["text_config"]
        else:
            self.config = raw_config

        with open(self.model_dir / "model.safetensors.index.json") as f:
            self.weight_idx: Dict[str, str] = json.load(f)["weight_map"]

        self.hidden_size = self.config["hidden_size"]
        self.intermediate_size = self.config["intermediate_size"]
        self.rms_norm_eps = float(self.config.get("rms_norm_eps", 1e-6))

    def _load_tensor(self, name: str) -> torch.Tensor:
        """Load a tensor from safetensors.

        CRITICAL: INT4-packed tensors (qweight, qzeros) have int32 dtype.
        Converting int32 to float32 LOSES LOW BITS for values > 2^24,
        which corrupts the packed INT4 data. Keep integer dtypes as-is.
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
        """Build full weight name for a given layer.

        Returns e.g. 'model.language_model.layers.3.mlp.gate_proj.qweight'
        """
        return f"model.language_model.layers.{layer_idx}.{attr}"

    def _dequant_weight(
        self, layer_idx: int, attr: str, tp_size: int = 2, gpu_idx: int = 0
    ) -> torch.Tensor:
        """Dequantize an INT4 weight and apply column-parallel TP sharding.

        The dequantized full weight has shape [K, N] where K is the input
        dimension (hidden_size for MLP projections) and N is the output
        dimension (intermediate_size for gate/up). Column-parallel splits
        N along TP.

        Args:
            layer_idx: Target layer index.
            attr: Weight attribute path (e.g. 'mlp.gate_proj.qweight').
            tp_size: Tensor parallel size.
            gpu_idx: GPU index within the TP group.

        Returns:
            Sharded dequantized weight of shape [K, N // tp_size].
        """
        qweight = self._load_tensor(self._get_weight_name(layer_idx, f"{attr}.qweight"))
        qzeros = self._load_tensor(self._get_weight_name(layer_idx, f"{attr}.qzeros"))
        scales = self._load_tensor(self._get_weight_name(layer_idx, f"{attr}.scales"))

        W_full = dequantize_int4_autogptq(qweight, qzeros, scales)

        # Column-parallel sharding: split output (N) dimension by TP
        N_full = W_full.shape[1]
        sharded_N = N_full // tp_size
        start = gpu_idx * sharded_N
        end = start + sharded_N
        return W_full[:, start:end]

    # ------------------------------------------------------------------
    # Norm weight convenience methods
    # ------------------------------------------------------------------

    def load_norm1(self, layer_idx: int) -> torch.Tensor:
        """Load input_layernorm weight. Returns [hidden_size]."""
        return self._load_tensor(
            self._get_weight_name(layer_idx, "input_layernorm.weight")
        )

    def load_norm2(self, layer_idx: int) -> torch.Tensor:
        """Load post_attention_layernorm weight. Returns [hidden_size]."""
        return self._load_tensor(
            self._get_weight_name(layer_idx, "post_attention_layernorm.weight")
        )

    def load_final_norm(self) -> torch.Tensor:
        """Load the final RMSNorm weight (model.norm.weight). Returns [hidden_size]."""
        return self._load_tensor("model.norm.weight")

    # ------------------------------------------------------------------
    # Attention projection convenience methods (full attention layers)
    # ------------------------------------------------------------------

    def load_q_proj_dequant(
        self, layer_idx: int, tp_size: int = 2, gpu_idx: int = 0
    ) -> torch.Tensor:
        """Dequantize q_proj for a specific TP shard.

        Returns [hidden_size, per_gpu_attention_heads * head_dim].
        """
        return self._dequant_weight(layer_idx, "self_attn.q_proj", tp_size, gpu_idx)

    def load_k_proj_dequant(
        self, layer_idx: int, tp_size: int = 2, gpu_idx: int = 0
    ) -> torch.Tensor:
        """Dequantize k_proj for a specific TP shard.

        Returns [hidden_size, per_gpu_kv_heads * head_dim].
        """
        return self._dequant_weight(layer_idx, "self_attn.k_proj", tp_size, gpu_idx)

    def load_v_proj_dequant(
        self, layer_idx: int, tp_size: int = 2, gpu_idx: int = 0
    ) -> torch.Tensor:
        """Dequantize v_proj for a specific TP shard.

        Returns [hidden_size, per_gpu_kv_heads * head_dim].
        """
        return self._dequant_weight(layer_idx, "self_attn.v_proj", tp_size, gpu_idx)

    def load_o_proj_dequant(
        self, layer_idx: int, tp_size: int = 2, gpu_idx: int = 0
    ) -> torch.Tensor:
        """Dequantize o_proj for a specific TP shard (row-parallel).

        Row-parallel splits the input (K) dimension by TP.

        Returns [per_gpu_attention_heads * head_dim, hidden_size].
        """
        qweight = self._load_tensor(
            self._get_weight_name(layer_idx, "self_attn.o_proj.qweight")
        )
        qzeros = self._load_tensor(
            self._get_weight_name(layer_idx, "self_attn.o_proj.qzeros")
        )
        scales = self._load_tensor(
            self._get_weight_name(layer_idx, "self_attn.o_proj.scales")
        )

        W_full = dequantize_int4_autogptq(qweight, qzeros, scales)

        # Row-parallel sharding: split K dimension by TP
        K_full = W_full.shape[0]
        sharded_K = K_full // tp_size
        start = gpu_idx * sharded_K
        end = start + sharded_K
        return W_full[start:end, :]

    def load_q_norm(self, layer_idx: int) -> torch.Tensor:
        """Load q_norm weight (self_attn.q_norm.weight). Returns [hidden_size]."""
        return self._load_tensor(
            self._get_weight_name(layer_idx, "self_attn.q_norm.weight")
        )

    def load_k_norm(self, layer_idx: int) -> torch.Tensor:
        """Load k_norm weight (self_attn.k_norm.weight). Returns [hidden_size]."""
        return self._load_tensor(
            self._get_weight_name(layer_idx, "self_attn.k_norm.weight")
        )

    # ------------------------------------------------------------------
    # MLP projection convenience methods
    # ------------------------------------------------------------------

    def load_gate_proj_dequant(
        self, layer_idx: int, tp_size: int = 2, gpu_idx: int = 0
    ) -> torch.Tensor:
        """Dequantize gate_proj for a specific TP shard.

        Column-parallel: each GPU gets half the output (N) features.

        Returns [hidden_size, intermediate_size // tp_size].
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

        W_full = dequantize_int4_autogptq(qweight, qzeros, scales, group_size)

        sharded_intermediate = self.intermediate_size // tp_size
        start = gpu_idx * sharded_intermediate
        end = start + sharded_intermediate
        return W_full[:, start:end]

    def load_up_proj_dequant(
        self, layer_idx: int, tp_size: int = 2, gpu_idx: int = 0
    ) -> torch.Tensor:
        """Dequantize up_proj for a specific TP shard.

        Column-parallel: each GPU gets half the output (N) features.

        Returns [hidden_size, intermediate_size // tp_size].
        """
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

    def load_down_proj_dequant(
        self, layer_idx: int, tp_size: int = 2, gpu_idx: int = 0
    ) -> torch.Tensor:
        """Dequantize down_proj for a specific TP shard.

        Row-parallel: each GPU gets half the input (K) features.

        Returns [intermediate_size // tp_size, hidden_size].
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

        W_full = dequantize_int4_autogptq(qweight, qzeros, scales, group_size)

        # Row-parallel sharding: split K dimension by TP
        sharded_intermediate = self.intermediate_size // tp_size
        start = gpu_idx * sharded_intermediate
        end = start + sharded_intermediate
        return W_full[start:end]

    # ------------------------------------------------------------------
    # LM head and embedding
    # ------------------------------------------------------------------

    def load_lm_head_dequant(
        self, tp_size: int = 2, gpu_idx: int = 0
    ) -> torch.Tensor:
        """Dequantize lm_head for a specific TP shard (column-parallel).

        Returns [hidden_size, vocab_size // tp_size].
        """
        qweight = self._load_tensor("model.lm_head.qweight")
        qzeros = self._load_tensor("model.lm_head.qzeros")
        scales = self._load_tensor("model.lm_head.scales")

        W_full = dequantize_int4_autogptq(qweight, qzeros, scales)

        # Column-parallel sharding: split vocab (N) dimension by TP
        N_full = W_full.shape[1]
        sharded_N = N_full // tp_size
        start = gpu_idx * sharded_N
        end = start + sharded_N
        return W_full[:, start:end]

    def load_embedding(self) -> torch.Tensor:
        """Load the token embedding table.

        Returns [vocab_size, hidden_size].
        """
        return self._load_tensor("model.embed_tokens.weight")
