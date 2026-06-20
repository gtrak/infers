#!/usr/bin/env python3
"""
Dump GDN intermediate tensors from HF Qwen3 model at layer 0,
using the same prompt as the smoke test.

Output: float32 .npy files in /tmp/ref_gdn_new/
Matching file names and shapes against our engine's TP=2 dump format.
# @lat: [[testing#Re-exports#GDN Reference Tests]]
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = "/home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4/"
OUTPUT_DIR = "/tmp/ref_gdn_new"
# Same prompt as crates/backends/native/tests/smoke_test.rs line 138
PROMPT = "▁user\nWhat is the capital of France?\n\n▁assistant\n"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(name: str, tensor: torch.Tensor):
    """Save a tensor as float32 .npy file."""
    cpu_arr = tensor.detach().cpu().float().numpy()
    path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(path, cpu_arr)
    print(f"  {name}: shape={cpu_arr.shape} dtype={cpu_arr.dtype}")

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    # ── Use engine's exact token IDs (not HF tokenizer) ─────────────────
    # Our Rust engine (tokenizers crate) produces 15 tokens for this prompt.
    # HF AutoTokenizer produces 16 tokens — the mismatch makes GDN comparisons invalid.
    # Engine's token IDs from crates/backends/native/tests/smoke_test.rs:
    token_ids = [248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198]
    print(f"Prompt: {PROMPT!r}")
    print(f"Token IDs ({len(token_ids)}): {token_ids}")

    # ── Load model with bf16 and quantization ───────────────────────────
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── Extract config (loaded model has Qwen3_5TextConfig directly) ────
    tc = model.config
    hidden_size = tc.hidden_size
    num_k_heads = tc.linear_num_key_heads
    num_v_heads = tc.linear_num_value_heads
    head_k_dim = tc.linear_key_head_dim
    head_v_dim = tc.linear_value_head_dim
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim
    kv_ratio = num_v_heads // num_k_heads

    print(f"\nConfig: hidden={hidden_size}, k_heads={num_k_heads}, v_heads={num_v_heads}")
    print(f"  head_k={head_k_dim}, head_v={head_v_dim}")
    print(f"  key_dim={key_dim}, value_dim={value_dim}, conv_dim={conv_dim}")
    print(f"  kv_ratio={kv_ratio}")

    # Save model config info
    with open(os.path.join(OUTPUT_DIR, "model_config.json"), "w") as f:
        json.dump({
            "model_path": MODEL_DIR,
            "model_class": "Qwen3_5ForConditionalGeneration",
            "prompt": PROMPT,
            "input_ids": token_ids,
            "hidden_size": hidden_size,
            "linear_num_key_heads": num_k_heads,
            "linear_num_value_heads": num_v_heads,
            "linear_key_head_dim": head_k_dim,
            "linear_value_head_dim": head_v_dim,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "conv_dim": conv_dim,
            "kv_ratio": kv_ratio,
        }, f, indent=2)

    # ── Find GDN layer 0 (called linear_attn in loaded model) ───────────
    gdn_layer = None
    for i, layer in enumerate(model.model.layers):
        attn = getattr(layer, 'linear_attn', None)
        if attn is not None and isinstance(attn, Qwen3_5GatedDeltaNet):
            gdn_layer = attn
            print(f"Found GDN at layer {i}")
            break

    if gdn_layer is None:
        # Try self_attn as fallback
        for i, layer in enumerate(model.model.layers[:5]):
            attn = getattr(layer, 'self_attn', None)
            if attn is not None and isinstance(attn, Qwen3_5GatedDeltaNet):
                gdn_layer = attn
                print(f"Found GDN (self_attn) at layer {i}")
                break

    if gdn_layer is None:
        raise RuntimeError("Could not find GDN layer in model")

    # ── Monkey-patch GDN forward to dump intermediates ──────────────────
    def patched_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        batch_size, seq_len = hidden_states.shape[:2]

        # Phase 1: in_proj_qkv → mixed_qkv [batch, seq_len, conv_dim]
        mixed_qkv = self.in_proj_qkv(hidden_states)
        save("mixed_qkv", mixed_qkv[0].float())

        # Transpose for conv1d: [batch, conv_dim, seq_len]
        mixed_qkv_t = mixed_qkv.transpose(1, 2)

        # Phase 2: in_proj_z (gate for RMSNormGated)
        z = self.in_proj_z(hidden_states)
        z_reshaped = z.reshape(batch_size, seq_len, -1, head_v_dim)

        # Phase 3: a and b projections [batch, seq_len, num_v_heads]
        b_proj_raw = self.in_proj_b(hidden_states)
        a_proj_raw = self.in_proj_a(hidden_states)
        save("a_proj", a_proj_raw[0].float())
        save("b_proj", b_proj_raw[0].float())

        # Phase 4: conv1d + SiLU
        mixed_qkv_t = F.silu(self.conv1d(mixed_qkv_t)[:, :, :mixed_qkv_t.shape[-1]])
        save("conv_out", mixed_qkv_t.transpose(1, 2)[0].float())

        # Phase 5: split into q, k, v BEFORE repeat_interleave
        mixed_qkv_back = mixed_qkv_t.transpose(1, 2)
        query_raw, key_raw, value_raw = torch.split(
            mixed_qkv_back, [key_dim, key_dim, value_dim], dim=-1
        )
        save("query", query_raw[0].float())
        save("key", key_raw[0].float())
        save("value", value_raw[0].float())

        # Reshape to per-head form
        query = query_raw.reshape(batch_size, seq_len, -1, head_k_dim)
        key = key_raw.reshape(batch_size, seq_len, -1, head_k_dim)
        value = value_raw.reshape(batch_size, seq_len, -1, head_v_dim)

        # Phase 6: repeat_interleave if needed
        if kv_ratio > 1:
            query_expanded = query.repeat_interleave(kv_ratio, dim=2)
            key_expanded = key.repeat_interleave(kv_ratio, dim=2)
        else:
            query_expanded = query
            key_expanded = key

        save("query_expanded", query_expanded[0].float())
        save("key_expanded", key_expanded[0].float())

        # Phase 7: compute g and beta
        beta = b_proj_raw.sigmoid()
        A_log_f32 = self.A_log.float()
        a_f32 = a_proj_raw.float()
        dt_bias_f32 = self.dt_bias.float()
        g = -A_log_f32.exp() * F.softplus(a_f32 + dt_bias_f32)

        # Phase 8: GDN recurrence (chunk_gated_delta_rule for multi-token)
        core_attn_out, _ = self.chunk_gated_delta_rule(
            query_expanded, key_expanded, value,
            g=g, beta=beta, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True,
        )
        save("core_attn_out", core_attn_out[0].float())

        # Phase 9: RMSNormGated
        batch_seq = batch_size * seq_len
        core_flat = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z_reshaped.reshape(-1, self.head_v_dim)

        save("z_gate", z_flat.float())

        normed = self.norm(core_flat, z_flat)
        save("norm_output", normed.float())

        # Phase 10: out_proj
        norm_reshaped = normed.reshape(batch_size, seq_len, -1)
        output = self.out_proj(norm_reshaped)
        save("output", output[0].float())

        return output

    gdn_layer.forward = lambda *args, **kwargs: patched_forward(gdn_layer, *args, **kwargs)

    # ── Run forward pass ────────────────────────────────────────────────
    print("\nRunning forward pass on layer 0 (GDN)...")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device='cuda')

    with torch.no_grad():
        model.eval()
        outputs = model(input_ids=input_ids, use_cache=False)

    # Save input_ids separately
    save("input_ids", torch.tensor(token_ids, dtype=torch.int64))

    print("\nDone! Files saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
