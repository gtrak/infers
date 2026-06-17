"""GDN reference stages — stub pending full implementation.

GDN stages are already verified to have cos≈1.0 in gdn_layer_compare.py,
so we keep this minimal as a placeholder for the comparison framework.
"""

from tests.compare.stages.base import Stage


# Thresholds from gdn_layer_compare.py STAGE_THRESHOLDS
_GDN_THRESHOLDS = {
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


class GdnMixedQkvStage(Stage):
    """hidden_input @ in_proj_qkv_dequant (INT4 GEMM)."""

    name = "gdn.mixed_qkv"
    threshold = _GDN_THRESHOLDS["mixed_qkv"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        raise NotImplementedError("GDN stages not yet implemented")


class GdnConvOutStage(Stage):
    """depthwise_conv1d_silu(mixed_qkv)."""

    name = "gdn.conv_out"
    threshold = _GDN_THRESHOLDS["conv_out"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        raise NotImplementedError("GDN stages not yet implemented")


class GdnCoreAttnOutStage(Stage):
    """GDN recurrent step (query_expanded, key_expanded, value, a_proj, b_proj)."""

    name = "gdn.core_attn_out"
    threshold = _GDN_THRESHOLDS["core_attn_out"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        raise NotImplementedError("GDN stages not yet implemented")


class GdnNormOutputStage(Stage):
    """RMSNormGated(core_attn_out, z_gate, norm_weight)."""

    name = "gdn.norm_output"
    threshold = _GDN_THRESHOLDS["norm_output"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        raise NotImplementedError("GDN stages not yet implemented")


class GdnOutputStage(Stage):
    """norm_output @ out_proj_dequant (INT4 GEMM)."""

    name = "gdn.output"
    threshold = _GDN_THRESHOLDS["output"]

    def compute(self, inputs, weights, config, layer_idx, gpu_idx):
        raise NotImplementedError("GDN stages not yet implemented")
