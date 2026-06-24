#!/usr/bin/env python3
"""Dump per-layer hidden states from a Qwen3.5 model as oracle ground truth."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from compressed_tensors import NVFP4PackedCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def patch_nvfp4_config(config) -> None:
    """Patch vLLM-namespaced quantization config to transformers namespace.

    NVFP4 models exported for vLLM use `language_model.model.layers` in their
    compressed_tensors target regexes.  The transformers library expects
    `model.layers`.  This function rewrites those prefixes in-place so that
    the weight decompression logic can match modules correctly.
    """
    if not hasattr(config, "quantization_config"):
        return

    qconfig = config.quantization_config
    if "config_groups" not in qconfig:
        return

    # Check whether any target uses the vLLM namespace
    for group in qconfig["config_groups"].values():
        targets = group.get("targets", [])
        for t in targets:
            if "language_model.model.layers" in t:
                # vLLM namespace detected — patch all groups and ignore list
                for g in qconfig["config_groups"].values():
                    new_targets = []
                    for tgt in g.get("targets", []):
                        patched = tgt.replace("language_model.model.layers", "model.layers")
                        patched = patched.replace(
                            "visual.blocks", "model.vision_tower.visual.blocks"
                        )
                        new_targets.append(patched)
                    g["targets"] = new_targets

                if "ignore" in qconfig:
                    qconfig["ignore"] = [
                        ig.replace("language_model.model.layers", "model.layers")
                        .replace("visual.blocks", "model.vision_tower.visual.blocks")
                        for ig in qconfig["ignore"]
                    ]
                break  # Only need to patch once


def load_nvfp4_weights_manual(model, model_path: str) -> None:
    """Manually decompress and load NVFP4-quantized weights from safetensors.

    When compressed_tensors fails to match the vLLM-namespaced target regexes,
    quantized layers get randomly initialized weights.  This function walks all
    safetensors shards, decompresses each ``weight_packed`` / ``weight_scale`` /
    ``weight_global_scale`` triplet via ``NVFP4PackedCompressor``, and writes the
    resulting BF16 tensor into the model's corresponding ``.weight`` parameter.

    It also loads any remaining BF16-only tensors (``A_log``, ``dt_bias``,
    ``conv1d.weight``, ``norm.weight``, ``in_proj_a.weight``, ``in_proj_b.weight``,
    and unquantized ``.weight``) so that every tensor in the checkpoint is loaded,
    regardless of whether transformers managed to pick it up.

    The safetensors key prefix ``model.language_model.layers.*`` is mapped to the
    model's internal module path ``model.layers.*``.
    """

    def _st_key_to_model_key(st_key: str) -> str:
        """Map a safetensors key to the model's get_submodule path.

        Mapping rules:
        - ``model.language_model.layers.X.yyy.zzz`` → ``model.layers.X.yyy.zzz``
        - ``language_model.layers.X.yyy.zzz``      → ``model.layers.X.yyy.zzz``
        - Other keys are returned unchanged (e.g. ``lm_head.weight``).
        """
        if st_key.startswith("model.language_model.layers"):
            return "model." + st_key[len("model.language_model."):]
        elif st_key.startswith("language_model.layers"):
            return "model." + st_key[len("language_model."):]
        return st_key

    safetensors_dir = Path(model_path)
    st_files = sorted(safetensors_dir.glob("*.safetensors"))
    if not st_files:
        print("[NVFP4] No safetensors files found, skipping manual load")
        return

    # ---- Build a key→tensor map for quantization metadata only (small) ----
    quant_key_map: dict[str, torch.Tensor] = {}
    for st_file in st_files:
        with safe_open(str(st_file), framework="pt") as f:
            for key in f.keys():
                if key.endswith(".weight_packed") or key.endswith(".weight_scale") or key.endswith(
                    ".weight_global_scale"
                ):
                    quant_key_map[key] = f.get_tensor(key)

    quant_scheme = _build_nvfp4_quant_scheme()
    decompressed_count = 0
    bf16_loaded = 0
    errors = 0

    # Suffixes that indicate quantization metadata (skip these for BF16 loading)
    quant_meta_suffixes = {
        ".weight_packed", ".weight_scale", ".weight_global_scale",
        ".input_global_scale",
    }

    # ---- Phase 1: Decompress NVFP4 quantized weights ----
    packed_keys = [k for k in quant_key_map if k.endswith(".weight_packed")]

    for pk in packed_keys:
        scale_key = pk.replace(".weight_packed", ".weight_scale")
        global_key = pk.replace(".weight_packed", ".weight_global_scale")

        # Derive submodule path: convert safetensors key to model key,
        # then strip the .weight attribute suffix
        model_full_key = _st_key_to_model_key(pk)
        if model_full_key.endswith(".weight_packed"):
            model_sub_key = model_full_key[:-len(".weight_packed")]
        else:
            continue

        try:
            state_dict = {
                "weight_packed": quant_key_map[pk],
                "weight_scale": quant_key_map[scale_key],
                "weight_global_scale": quant_key_map[global_key],
            }

            decompressed = NVFP4PackedCompressor.decompress(
                state_dict, quant_scheme
            )
            weight_bf16 = decompressed["weight"]

            submodule = model.get_submodule(model_sub_key)
            if not hasattr(submodule, "weight"):
                raise ValueError(f"Module {model_sub_key} has no .weight attr")
            submodule.weight.data.copy_(
                weight_bf16.to(submodule.weight.device)
            )
            decompressed_count += 1
        except Exception as e:
            print(f"[NVFP4] Error loading {pk}: {e}")
            errors += 1

    # ---- Phase 2: Load all non-quantization-metadata BF16 tensors (file-by-file) ----
    for st_file in st_files:
        with safe_open(str(st_file), framework="pt") as f:
            for sk in f.keys():
                is_quant_meta = any(sk.endswith(s) for s in quant_meta_suffixes)
                if is_quant_meta:
                    continue

                tensor = f.get_tensor(sk)
                dtype = tensor.dtype
                # Skip float32 scalar scales (input_global_scale etc.)
                if dtype == torch.float32 and tensor.numel() <= 1:
                    continue

                # Convert safetensors key to model key, then split into
                # (submodule_path, attr_name) at the last "."
                model_full_key = _st_key_to_model_key(sk)
                parts = model_full_key.rsplit(".", 1)
                if len(parts) != 2:
                    continue
                model_sub_key, attr_name = parts

                try:
                    submodule = model.get_submodule(model_sub_key)
                    target_tensor = getattr(submodule, attr_name)
                    target_tensor.data.copy_(tensor.to(target_tensor.device))
                    bf16_loaded += 1
                except Exception as e:
                    print(f"[NVFP4] Error loading BF16 tensor {sk}: {e}")
                    errors += 1

    print(
        f"[NVFP4] Manual load complete: "
        f"{decompressed_count} NVFP4 weights decompressed, "
        f"{bf16_loaded} BF16 tensors loaded, "
        f"{errors} errors"
    )


def _build_nvfp4_quant_scheme() -> QuantizationScheme:
    """Build the QuantizationScheme matching this model's NVFP4 config."""
    from compressed_tensors.quantization import QuantizationType, QuantizationStrategy

    weights_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        group_size=16,
        symmetric=True,
        dynamic=False,
        scale_dtype=torch.float8_e4m3fn,
        zp_dtype=torch.float8_e4m3fn,
    )
    return QuantizationScheme(weights=weights_config, targets=[".*"])


def write_raw(tensor: torch.Tensor, path: str) -> None:
    """Write raw little-endian BF16 bytes to disk.

    Uses an int16 view of the bfloat16 buffer so that we bypass numpy's
    (sometimes missing) native BF16 support while preserving the exact
    bit-pattern.
    """
    bf16 = tensor.to(torch.bfloat16).cpu().contiguous()
    with open(path, "wb") as f:
        # view as int16 → numpy → tobytes keeps the raw 2-byte BF16 encoding
        f.write(bf16.view(torch.int16).numpy().tobytes())


def write_meta(
    name: str,
    layer: int | None,
    shape: list[int],
    dtype: str = "bf16",
    stage: str = "hidden_states",
    phase: str = "oracle",
    path: str | None = None,
) -> None:
    """Write JSON metadata alongside a .raw file."""
    meta = {
        "name": name,
        "layer": layer,
        "shape": shape,
        "dtype": dtype,
        "stage": stage,
        "phase": phase,
    }
    if path is None:
        # Derive from context — caller must set this explicitly
        raise ValueError("path must be provided")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dump per-layer hidden states as oracle ground truth"
    )
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--prompt", required=True, help="Prompt string")
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    parser.add_argument(
        "--dtype", default="bf16", choices=["bf16", "float16", "float32"],
        help="Dtype for the forward pass (default: bf16)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bf16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    load_dtype = dtype_map[args.dtype]

    # ------------------------------------------------------------------
    # 1. Load model and tokenizer
    # ------------------------------------------------------------------
    print(f"Loading model from {args.model} ...")

    # Patch config in-memory for NVFP4 models that have vLLM-namespaced
    # quantization target regexes (language_model.model.layers → model.layers)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    patch_nvfp4_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=load_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Manual NVFP4 weight decompression (needed when compressed_tensors
    # fails to match the vLLM-namespaced target regexes)
    load_nvfp4_weights_manual(model, args.model)

    # ------------------------------------------------------------------
    # 2. Determine layer container (handles both Qwen3_5ForCausalLM and
    #    Qwen3_5ForConditionalGeneration where layers live at .model.layers)
    # ------------------------------------------------------------------
    if hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise RuntimeError(
            f"Cannot find layer list on {type(model).__name__}. "
            "Expected model.layers or model.model.layers."
        )

    num_layers = len(layers)
    print(f"Found {num_layers} decoder layers in {type(model).__name__}")

    # ------------------------------------------------------------------
    # 3. Build layer-type map from config
    # ------------------------------------------------------------------
    config = model.config
    # Some configs expose layer_types directly; others use full_attention_interval
    if hasattr(config, "text_config"):
        text_cfg = config.text_config
    else:
        text_cfg = config

    layer_types_map: dict[str, str] = {}
    if hasattr(text_cfg, "layer_types") and text_cfg.layer_types is not None:
        for idx, lt in enumerate(text_cfg.layer_types):
            label = "gdn" if "linear" in lt.lower() or "delta" in lt.lower() else "attn"
            layer_types_map[str(idx)] = label
    else:
        interval = getattr(text_cfg, "full_attention_interval", 4)
        for idx in range(num_layers):
            is_full_attn = (idx + 1) % interval == 0
            layer_types_map[str(idx)] = "attn" if is_full_attn else "gdn"

    hidden_size = text_cfg.hidden_size if hasattr(text_cfg, "hidden_size") else config.hidden_size
    vocab_size = text_cfg.vocab_size if hasattr(text_cfg, "vocab_size") else config.vocab_size
    print(f"hidden_size={hidden_size}, vocab_size={vocab_size}")

    # ------------------------------------------------------------------
    # 4. Tokenize prompt
    # ------------------------------------------------------------------
    tokenized = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokenized.input_ids.to(model.device)
    ids = input_ids.tolist()[0]
    print(f"Token IDs: {ids}")

    seq_len = len(ids)

    # ------------------------------------------------------------------
    # 5. Register forward hooks on every decoder layer
    # ------------------------------------------------------------------
    captured: dict[int, torch.Tensor] = {}  # layer_idx -> hidden_states

    def make_hook(layer_idx: int):
        def hook(module, input_, output):
            hidden = input_[0].to(torch.bfloat16).detach().clone()
            captured[layer_idx] = hidden
        return hook

    handles = []
    for idx in range(num_layers):
        h = layers[idx].register_forward_hook(make_hook(idx))
        handles.append(h)

    # ------------------------------------------------------------------
    # 6. Embedding output: capture by running embedding layer manually
    # ------------------------------------------------------------------
    embed_output: torch.Tensor | None = None
    if hasattr(model.model, "embed_tokens"):
        embed = model.model.embed_tokens(input_ids)
        embed_output = embed.to(torch.bfloat16).detach().clone()
    elif hasattr(model, "embed_tokens"):
        embed = model.embed_tokens(input_ids)
        embed_output = embed.to(torch.bfloat16).detach().clone()
    else:
        raise RuntimeError("Cannot find embed_tokens on model")

    # ------------------------------------------------------------------
    # 7. Forward pass (single forward, NOT generate)
    # ------------------------------------------------------------------
    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits.to(torch.bfloat16).detach().clone()
    print(f"Logits shape: {logits.shape}")

    # Verify next-token prediction
    next_token_id = logits[0, -1].argmax(dim=-1).item()
    predicted_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
    print(f"Next token ID: {next_token_id} -> '{predicted_text}'")

    # ------------------------------------------------------------------
    # 8. Remove hooks
    # ------------------------------------------------------------------
    for h in handles:
        h.remove()

    # ------------------------------------------------------------------
    # 9. Dump files
    # ------------------------------------------------------------------
    # Embedding output (before layer 0)
    (output_dir / "layer_0" / "oracle").mkdir(parents=True, exist_ok=True)
    write_raw(embed_output, str(output_dir / "layer_0" / "oracle" / "embed_output.raw"))
    def shape_no_batch(t: torch.Tensor) -> list[int]:
        """Return shape with leading batch dimension (if size==1) stripped."""
        s = list(t.shape)
        if len(s) >= 1 and s[0] == 1:
            return s[1:]
        return s

    write_meta(
        name="embed_output",
        layer=0,
        shape=shape_no_batch(embed_output),
        path=str(output_dir / "layer_0" / "oracle" / "embed_output.meta"),
    )

    # Per-layer hidden states
    for layer_idx in range(num_layers):
        if layer_idx not in captured:
            print(f"WARNING: No capture for layer {layer_idx}")
            continue

        ts = captured[layer_idx]
        layer_dir = output_dir / f"layer_{layer_idx}" / "oracle"
        layer_dir.mkdir(parents=True, exist_ok=True)

        raw_path = layer_dir / "hidden_states.raw"
        meta_path = layer_dir / "hidden_states.meta"

        write_raw(ts, str(raw_path))
        write_meta(
            name="hidden_states",
            layer=layer_idx,
            shape=shape_no_batch(ts),
            path=str(meta_path),
        )

    # Final logits
    final_dir = output_dir / "final" / "oracle"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_raw(logits, str(final_dir / "logits.raw"))
    write_meta(
        name="logits",
        layer=None,
        shape=shape_no_batch(logits),
        stage="logits",
        path=str(final_dir / "logits.meta"),
    )

    # Top-level config
    oracle_config = {
        "model_path": args.model,
        "prompt": args.prompt,
        "token_ids": ids,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "layer_types": layer_types_map,
    }
    with open(output_dir / "oracle_config.json", "w") as f:
        json.dump(oracle_config, f, indent=2)

    # ------------------------------------------------------------------
    # 10. Verify file sizes
    # ------------------------------------------------------------------
    bf16_elem = 2  # bytes per BF16 element
    expected_hidden = seq_len * hidden_size * bf16_elem
    expected_logits = seq_len * vocab_size * bf16_elem

    print("\n--- File size verification ---")
    for layer_idx in range(num_layers):
        raw = output_dir / f"layer_{layer_idx}" / "oracle" / "hidden_states.raw"
        actual = raw.stat().st_size
        if actual != expected_hidden:
            print(f"  MISMATCH layer {layer_idx}: got {actual}, expected {expected_hidden}")
        else:
            print(f"  OK     layer {layer_idx}: {actual} bytes")

    logit_raw = output_dir / "final" / "oracle" / "logits.raw"
    actual_logits = logit_raw.stat().st_size
    if actual_logits != expected_logits:
        print(f"  MISMATCH logits: got {actual_logits}, expected {expected_logits}")
    else:
        print(f"  OK     logits:   {actual_logits} bytes")

    # Embed output size
    embed_raw = output_dir / "layer_0" / "oracle" / "embed_output.raw"
    actual_embed = embed_raw.stat().st_size
    expected_embed = seq_len * hidden_size * bf16_elem
    if actual_embed != expected_embed:
        print(f"  MISMATCH embed: got {actual_embed}, expected {expected_embed}")
    else:
        print(f"  OK     embed:    {actual_embed} bytes")

    print("\nDone. Oracle dump written to", str(output_dir))


if __name__ == "__main__":
    main()
