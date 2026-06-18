"""Dump config reader — parses config.json written by probe::dump_config()."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DumpConfig:
    """Configuration for engine dump comparisons.

    Read from the config.json file written by the Rust probe::dump_config()
    function at runtime. Contains model parameters needed to set up
    reference computations and comparison thresholds.
    """
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    layer_types: List[str]
    vocab_size: int
    num_gpus: int
    group_size: int
    attn_output_gate: bool
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    linear_num_key_heads: int = 1
    linear_num_value_heads: int = 1
    linear_key_head_dim: int = 1
    linear_value_head_dim: int = 1
    linear_conv_kernel_dim: int = 4

    @classmethod
    def from_dir(cls, dump_dir: str) -> "DumpConfig":
        """Load from config.json in the given dump directory.

        Args:
            dump_dir: Path to the root dump directory containing config.json.

        Returns:
            DumpConfig with all fields populated.
        """
        config_path = Path(dump_dir) / "config.json"
        with open(config_path) as f:
            data = json.load(f)
        return cls(
            hidden_size=data["hidden_size"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            layer_types=data["layer_types"],
            vocab_size=data["vocab_size"],
            num_gpus=data["num_gpus"],
            group_size=data["group_size"],
            attn_output_gate=data["attn_output_gate"],
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data["rope_theta"]),
            partial_rotary_factor=float(data["partial_rotary_factor"]),
            linear_num_key_heads=data.get("linear_num_key_heads", 1),
            linear_num_value_heads=data.get("linear_num_value_heads", 1),
            linear_key_head_dim=data.get("linear_key_head_dim", 1),
            linear_value_head_dim=data.get("linear_value_head_dim", 1),
            linear_conv_kernel_dim=data.get("linear_conv_kernel_dim", 4),
        )

    def get_layer_type(self, layer_idx: int) -> str:
        """Return the layer type for a given index.

        Args:
            layer_idx: Layer index (0-based).

        Returns:
            'full_attention' or 'gdn'.
        """
        return self.layer_types[layer_idx]
