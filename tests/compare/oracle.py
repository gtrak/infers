#!/usr/bin/env python3
"""HuggingFace oracle — runs the HF model and captures intermediate tensors via hooks.

This module provides ground-truth reference values by executing the actual HuggingFace
model and recording intermediate activations at each layer. It replaces the custom
Python reimplementations that had bugs (e.g., wrong RMSNorm formula).

Usage:
    python -m tests.compare.oracle --model-dir /path/to/model --token-ids 1,2,3 --output-dir /tmp/oracle

The oracle captures per-layer tensors for both prefill and decode phases, saved as
.pt files that can be compared against engine dumps via hf_compare.py.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _capture_layer(module, layer_idx, captures, phase):
    """Register forward hooks on a decoder layer to capture intermediate tensors.

    Hooks capture:
    - norm1_input: hidden state entering the layer
    - norm1_output: output of input_layernorm
    - attn_output: output of self_attn or linear_attn
    - norm2_input: hidden state after attention residual
    - norm2_output: output of post_attention_layernorm
    - mlp_output: output of mlp
    - layer_output: hidden state leaving the layer
    """
    layer_captures = {}

    # Hook for input_layernorm
    if hasattr(module, "input_layernorm"):
        def make_norm1_hook(name):
            def hook_fn(m, inp, out):
                layer_captures[name] = out[0] if isinstance(out, tuple) else out
            return hook_fn
        module.input_layernorm.register_forward_hook(make_norm1_hook("norm1_output"))

    # Hook for self_attn (full attention layers)
    attn_module = getattr(module, "self_attn", None)
    if attn_module is not None:
        def attn_hook(m, inp, out):
            layer_captures["attn_output"] = out[0] if isinstance(out, tuple) else out
        attn_module.register_forward_hook(attn_hook)

    # Hook for linear_attn (GDN layers) — captures GDN-internal intermediates
    lin_attn_module = getattr(module, "linear_attn", None)
    if lin_attn_module is not None:
        def lin_attn_hook(m, inp, out):
            layer_captures["attn_output"] = out[0] if isinstance(out, tuple) else out
            layer_captures["output"] = out[0] if isinstance(out, tuple) else out
        lin_attn_module.register_forward_hook(lin_attn_hook)

        # Hook submodules that exist on the HF model's linear_attn
        # in_proj_qkv → mixed_qkv (QKV projection output)
        if hasattr(lin_attn_module, "in_proj_qkv"):
            def make_proj_hook(name, key):
                def hook_fn(m, inp, out):
                    layer_captures[key] = out[0] if isinstance(out, tuple) else out
                return hook_fn
            lin_attn_module.in_proj_qkv.register_forward_hook(
                make_proj_hook("in_proj_qkv", "mixed_qkv"))

        # conv1d → conv_out (temporal mixing output, after activation and transpose)
        # We DON'T hook conv1d directly because its raw output includes padding.
        # Instead, we compute conv_out from mixed_qkv in _compute_gdn_internals.

        # in_proj_a → a_proj
        if hasattr(lin_attn_module, "in_proj_a"):
            def make_proj_hook(name, key):
                def hook_fn(m, inp, out):
                    layer_captures[key] = out[0] if isinstance(out, tuple) else out
                return hook_fn
            lin_attn_module.in_proj_a.register_forward_hook(
                make_proj_hook("in_proj_a", "a_proj"))

        # in_proj_b → b_proj
        if hasattr(lin_attn_module, "in_proj_b"):
            def make_proj_hook(name, key):
                def hook_fn(m, inp, out):
                    layer_captures[key] = out[0] if isinstance(out, tuple) else out
                return hook_fn
            lin_attn_module.in_proj_b.register_forward_hook(
                make_proj_hook("in_proj_b", "b_proj"))

        # in_proj_z → z_gate
        if hasattr(lin_attn_module, "in_proj_z"):
            def make_proj_hook(name, key):
                def hook_fn(m, inp, out):
                    layer_captures[key] = out[0] if isinstance(out, tuple) else out
                return hook_fn
            lin_attn_module.in_proj_z.register_forward_hook(
                make_proj_hook("in_proj_z", "z_gate"))

        # norm (FusedRMSNormGated) pre-hook → core_attn_out (before norm, i.e. recurrent step output)
        if hasattr(lin_attn_module, "norm"):
            def norm_pre_hook(m, inp):
                # inp is (core_attn_out, z) before reshaping
                core_out = inp[0]
                if isinstance(core_out, tuple):
                    core_out = core_out[0]
                layer_captures["core_attn_out"] = core_out
            lin_attn_module.norm.register_forward_pre_hook(norm_pre_hook)

            def norm_hook(m, inp, out):
                layer_captures["norm_output"] = out[0] if isinstance(out, tuple) else out
            lin_attn_module.norm.register_forward_hook(norm_hook)

        # out_proj → o_proj
        if hasattr(lin_attn_module, "out_proj"):
            def make_proj_hook(name, key):
                def hook_fn(m, inp, out):
                    layer_captures[key] = out[0] if isinstance(out, tuple) else out
                return hook_fn
            lin_attn_module.out_proj.register_forward_hook(
                make_proj_hook("out_proj", "o_proj"))

        # Store GDN dimensions and module reference for post-processing
        layer_captures["gdn_linear_attn"] = lin_attn_module
        layer_captures["gdn_key_dim"] = lin_attn_module.key_dim
        layer_captures["gdn_value_dim"] = lin_attn_module.value_dim
        layer_captures["gdn_num_k_heads"] = lin_attn_module.num_k_heads
        layer_captures["gdn_num_v_heads"] = lin_attn_module.num_v_heads
        layer_captures["gdn_head_k_dim"] = lin_attn_module.head_k_dim
        layer_captures["gdn_head_v_dim"] = lin_attn_module.head_v_dim

    # Hook for post_attention_layernorm
    if hasattr(module, "post_attention_layernorm"):
        def make_norm2_hook(name):
            def hook_fn(m, inp, out):
                layer_captures[name] = out[0] if isinstance(out, tuple) else out
            return hook_fn
        module.post_attention_layernorm.register_forward_hook(make_norm2_hook("norm2_output"))

    # Hook for mlp
    if hasattr(module, "mlp"):
        def mlp_hook(m, inp, out):
            layer_captures["mlp_output"] = out[0] if isinstance(out, tuple) else out
        module.mlp.register_forward_hook(mlp_hook)

    # Hook for full layer forward — captures input and output
    def layer_input_hook(m, inp, out):
        # The first positional arg is the hidden state input
        layer_captures["norm1_input"] = inp[0] if isinstance(inp, tuple) else inp
        layer_captures["layer_output"] = out[0] if isinstance(out, tuple) else out
    module.register_forward_hook(layer_input_hook)

    captures[layer_idx] = layer_captures


def _compute_gdn_internals(captures, layer_idx):
    """Compute GDN-internal tensors from hooked outputs.

    Runs the conv pipeline on mixed_qkv to produce conv_out, then splits into
    query, key, value and applies repeat_interleave for GQA.
    These are computed in post-processing since they're inline ops, not hookable modules.
    """
    import torch.nn.functional as F

    lc = captures[layer_idx]
    if "mixed_qkv" not in lc:
        return

    # GDN module reference and dimensions stored during hook registration
    if "gdn_key_dim" not in lc or "gdn_linear_attn" not in lc:
        return

    lin_attn = lc["gdn_linear_attn"]
    key_dim = lc["gdn_key_dim"]
    value_dim = lc["gdn_value_dim"]
    num_k_heads = lc["gdn_num_k_heads"]
    num_v_heads = lc["gdn_num_v_heads"]
    head_k_dim = lc["gdn_head_k_dim"]
    head_v_dim = lc["gdn_head_v_dim"]

    mixed_qkv = lc["mixed_qkv"]  # [batch, seq, hidden] from in_proj_qkv

    # Run conv pipeline: transpose → conv1d → slice → activation → transpose back
    # This mirrors the HF forward method's conv processing
    mixed_qkv_t = mixed_qkv.transpose(1, 2)  # [batch, hidden, seq]
    conv_out_raw = lin_attn.conv1d(mixed_qkv_t)  # [batch, hidden, seq+pad]
    conv_out_sliced = conv_out_raw[:, :, :mixed_qkv_t.shape[-1]]  # [batch, hidden, seq]
    conv_out_activated = F.silu(conv_out_sliced)  # [batch, hidden, seq]
    conv_out = conv_out_activated.transpose(1, 2)  # [batch, seq, hidden]

    lc["conv_out"] = conv_out

    # Split conv_out into query, key, value
    query, key, value = torch.split(conv_out, [key_dim, key_dim, value_dim], dim=-1)

    # Reshape to [batch, seq, num_heads, head_dim]
    query = query.reshape(*query.shape[:-1], num_k_heads, head_k_dim)
    key = key.reshape(*key.shape[:-1], num_k_heads, head_k_dim)
    value = value.reshape(*value.shape[:-1], num_v_heads, head_v_dim)

    lc["query"] = query
    lc["key"] = key
    lc["value"] = value

    # GQA expansion: repeat_interleave if num_v_heads > num_k_heads
    if num_v_heads // num_k_heads > 1:
        query_expanded = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        key_expanded = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        lc["query_expanded"] = query_expanded
        lc["key_expanded"] = key_expanded


def _compute_norm2_input(captures, layer_idx):
    """Compute norm2_input = norm1_input + attn_output (residual connection).

    This must be done after the full forward pass since it depends on
    both norm1_input and attn_output being captured.
    """
    lc = captures[layer_idx]
    if "norm1_input" in lc and "attn_output" in lc:
        # Move to same device before adding (model may be sharded across GPUs)
        a = lc["norm1_input"].cpu() if lc["norm1_input"].is_cuda else lc["norm1_input"]
        b = lc["attn_output"].cpu() if lc["attn_output"].is_cuda else lc["attn_output"]
        lc["norm2_input"] = a + b


def run_oracle(model_dir: str, token_ids: List[int], output_dir: str) -> dict:
    """Run the HF model and capture intermediate tensors.

    Args:
        model_dir: Path to HuggingFace model directory.
        token_ids: Input token IDs (not a string prompt).
        output_dir: Directory to save captured tensors.

    Returns:
        Summary dict with shapes and metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded. Device: {model.device}")

    # Get the decoder layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError("Cannot find decoder layers in model architecture")

    num_layers = len(layers)
    print(f"  Found {num_layers} decoder layers")

    # Check layer types from config
    config = model.config
    tc = getattr(config, "text_config", config)
    layer_types = getattr(tc, "layer_types", None)

    # Determine layer types
    if layer_types is None:
        layer_types = ["full_attention"] * num_layers
    print(f"  Layer types: {set(layer_types)}")

    # Prepare input
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    seq_len = len(token_ids)
    print(f"  Input: {seq_len} tokens")

    # ---- Register hooks ----
    captures: Dict[int, dict] = {}
    for i, layer in enumerate(layers):
        _capture_layer(layer, i, captures, "prefill")

    # ---- Run prefill ----
    print("\nRunning prefill...")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        logits_prefill = outputs.logits[0]  # [seq_len, vocab_size]

    # Compute norm2_input and GDN internals for each layer
    for i in range(num_layers):
        _compute_norm2_input(captures, i)
        _compute_gdn_internals(captures, i)

    # Save prefill tensors
    print("Saving prefill tensors...")
    gdn_tensor_names = [
        "mixed_qkv", "conv_out", "query", "key", "value",
        "query_expanded", "key_expanded", "a_proj", "b_proj",
        "core_attn_out", "z_gate", "norm_output", "o_proj", "output",
    ]
    for i in range(num_layers):
        layer_dir = output_path / f"layer_{i}" / "prefill"
        layer_dir.mkdir(parents=True, exist_ok=True)
        lc = captures[i]
        for name in ["norm1_input", "norm1_output", "attn_output",
                      "norm2_input", "norm2_output", "mlp_output", "layer_output"] + gdn_tensor_names:
            if name in lc:
                tensor = lc[name]
                if tensor is not None:
                    torch.save(tensor.cpu().float(), layer_dir / f"{name}.pt")

    # Save prefill logits
    torch.save(logits_prefill.cpu().float(), output_path / "logits_prefill.pt")
    print(f"  Saved logits: {logits_prefill.shape}")

    # ---- Run decode (1 step) ----
    print("\nRunning decode (1 step)...")
    # Remove old hooks and re-register for decode
    for i, layer in enumerate(layers):
        # We need fresh captures for decode
        captures[i] = {}
        _capture_layer(layer, i, captures, "decode")

    # Decode input: last token from prefill output
    decode_input = input_ids[:, -1:]  # [1, 1]
    print(f"  Decode input shape: {decode_input.shape}")

    with torch.no_grad():
        # Use the model's forward with use_cache=True to get KV cache behavior
        # For simplicity, just feed the last token and capture intermediates
        outputs_decode = model(decode_input, output_hidden_states=True)
        logits_decode = outputs_decode.logits[0]  # [1, vocab_size]

    # Compute norm2_input and GDN internals for decode
    for i in range(num_layers):
        _compute_norm2_input(captures, i)
        _compute_gdn_internals(captures, i)

    # Save decode tensors
    print("Saving decode tensors...")
    for i in range(num_layers):
        layer_dir = output_path / f"layer_{i}" / "decode"
        layer_dir.mkdir(parents=True, exist_ok=True)
        lc = captures[i]
        for name in ["norm1_input", "norm1_output", "attn_output",
                      "norm2_input", "norm2_output", "mlp_output", "layer_output"] + gdn_tensor_names:
            if name in lc:
                tensor = lc[name]
                if tensor is not None:
                    torch.save(tensor.cpu().float(), layer_dir / f"{name}.pt")

    # Save decode logits
    torch.save(logits_decode.cpu().float(), output_path / "logits_decode.pt")
    print(f"  Saved logits: {logits_decode.shape}")

    # ---- Build summary ----
    summary = {
        "model_dir": model_dir,
        "token_ids": token_ids,
        "num_layers": num_layers,
        "layer_types": layer_types,
        "hidden_size": tc.hidden_size if hasattr(tc, "hidden_size") else None,
        "prefill_seq_len": seq_len,
        "tensors": {},
    }

    for i in range(num_layers):
        layer_tensors = {}
        for phase in ["prefill", "decode"]:
            layer_dir = output_path / f"layer_{i}" / phase
            phase_tensors = {}
            for name in ["norm1_input", "norm1_output", "attn_output",
                          "norm2_input", "norm2_output", "mlp_output", "layer_output"] + gdn_tensor_names:
                pt_path = layer_dir / f"{name}.pt"
                if pt_path.exists():
                    t = torch.load(pt_path, weights_only=True)
                    phase_tensors[name] = list(t.shape)
            layer_tensors[phase] = phase_tensors
        summary["tensors"][f"layer_{i}"] = layer_tensors

    summary["tensors"]["logits_prefill"] = list(logits_prefill.shape)
    summary["tensors"]["logits_decode"] = list(logits_decode.shape)

    # Save summary
    import json
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOracle dump saved to {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run HuggingFace oracle and capture intermediate tensors",
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--token-ids", type=str, required=True,
        help="Comma-separated token IDs (e.g. 1,2,3,4)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save oracle dumps",
    )
    args = parser.parse_args()

    token_ids = [int(x.strip()) for x in args.token_ids.split(",")]
    summary = run_oracle(args.model_dir, token_ids, args.output_dir)

    print("\nSummary:")
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
