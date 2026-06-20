#!/usr/bin/env python3
"""Dump full-attention intermediate tensors from HF Qwen3 model at layer 3,
using the same 15 token IDs as our engine.

Output: float32 .npy files in /tmp/ref_attn_l3/
# @lat: [[testing#Re-exports#Full-Attention Reference Tests (Layer 3)]]
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = "/home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4/"
OUTPUT_DIR = "/tmp/ref_attn_l3"
TOKEN_IDS = [248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30,
             248046, 198, 248045, 74455, 198]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(name, tensor):
    """Save a tensor as float32 .npy file, printing stats."""
    cpu_arr = tensor.detach().cpu().float().numpy()
    path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(path, cpu_arr)
    print(f"  {name}: shape={list(cpu_arr.shape)} mean_abs={np.abs(cpu_arr).mean():.6f}")


def main():
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    # ── Load model ───────────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── Locate layer 3 full-attention self_attn ──────────────────
    attn = model.model.layers[3].self_attn
    print(f"Layer 3 attn type: {type(attn).__name__}")
    print(f"  head_dim={attn.head_dim}")
    print(f"  scaling={attn.scaling}")

    num_heads = model.config.num_attention_heads       # 24
    num_kv_heads = model.config.num_key_value_heads    # 4
    head_dim = attn.head_dim                           # 256
    print(f"  num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    # ── Monkey-patch self_attn.forward to capture intermediates ───
    original_forward = attn.forward
    captured = {}

    def patched_forward(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]          # (B, seq_len)
        batch, seq_len = input_shape[0], input_shape[1]
        hidden_shape = (*input_shape, -1, head_dim)     # (B, seq_len, H, head_dim)

        # ── Q projection (doubled: query + gate per head) ───────
        q_full = attn.q_proj(hidden_states)             # [B, seq_len, num_heads*head_dim*2]
        save("q_full", q_full[0].float())               # engine ATTN-Q-FULL

        # View and split into query + gate (interleaved per head)
        viewed = q_full.view(*input_shape, -1, head_dim * 2)  # [B, seq_len, 24, 512]
        query_states, gate = torch.chunk(viewed, 2, dim=-1)   # [B, S, 24, 256] each

        # Gate reshape: gate.reshape(*input_shape, -1) → [B, seq_len, num_heads*head_dim]
        gate = gate.reshape(*input_shape, -1)           # [B, seq_len, 6144]
        save("gate_reshaped", gate[0].float())          # engine ATTN-GATE-HEADS

        # ── Q norm (per-head RMSNorm on head_dim dimension) ─────
        query_states = attn.q_norm(query_states.view(hidden_shape))  # [B, S, 24, 256]
        save("query_pre_rope", query_states[0].float())  # per head, before rope

        # Transpose: [B, seq_len, num_heads, head_dim] → [B, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)     # [B, 24, S, 256]
        save("query_heads", query_states[0].float())  # [24, 15, 256] all heads after norm, before rope
        save("q_h0_pre_rope", query_states[0, 0].float())  # engine ATTN-Q-H0: head 0 before rope
        # ── K projection + norm ─────────────────────────────────
        key_raw = attn.k_proj(hidden_states)            # [B, seq_len, 1024]
        key_normed = attn.k_norm(key_raw.view(hidden_shape))  # [B, S, 4, 256]
        save("key_pre_rope", key_normed[0].float())
        key_states = key_normed.transpose(1, 2)         # [B, 4, S, 256]

        # ── V projection ────────────────────────────────────────
        value_raw = attn.v_proj(hidden_states)          # [B, seq_len, 1024]
        save("value_full", value_raw[0].float())        # engine ATTN-V-FULL
        value_states = value_raw.view(hidden_shape).transpose(1, 2)  # [B, 4, S, 256]
        save("value_heads", value_states[0].float())

        # ── RoPE (partial: only first 64 dims of 256) ───────────
        cos, sin = position_embeddings                   # [B, seq_len, 64] each
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        save("query_rope_all", query_states[0].float())
        save("key_rope_all", key_states[0].float())

        # ── Compute per-head attention for head 0 ───────────────
        n_rep = num_heads // num_kv_heads               # 6

        # Expand K, V via GQA: repeat_kv(key, n_rep)
        # [B, num_kv_heads, S, head_dim] → [B, num_heads, S, head_dim]
        key_expanded = key_states.repeat_interleave(n_rep, dim=1)
        value_expanded = value_states.repeat_interleave(n_rep, dim=1)

        # Head 0 extraction
        q_h0 = query_states[0, 0]          # [S, head_dim] = [15, 256]
        k_h0 = key_expanded[0, 0]          # [S, head_dim]
        v_h0 = value_expanded[0, 0]        # [S, head_dim]

        save("q_h0_rope", q_h0.float())    # engine ATTN-Q-ROPE-H0
        save("k_h0_rope", k_h0.float())    # engine ATTN-K-H0
        save("v_h0", v_h0.float())         # engine ATTN-V-H0

        # ── Attention scores for head 0 ────────────────────────
        # Scaled dot-product: Q @ K^T / sqrt(head_dim)
        scores = (q_h0.unsqueeze(0) @ k_h0.unsqueeze(0).transpose(-2, -1)) * attn.scaling
        # [1, S, S] = [1, 15, 15]

        # Causal mask: attention to future tokens is -inf
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        save("scores_h0", scores[0].float())

        # Softmax (cast to float32 for numerical stability)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        save("softmax_h0", probs[0].float())  # engine ATTN-SOFTMAX-H0

        # ── Attention output for head 0 ────────────────────────
        attn_out_h0 = probs @ v_h0.unsqueeze(0)  # [1, S, head_dim]
        save("attn_out_h0", attn_out_h0[0].float())  # engine ATTN-OUT-H0

        # ── Continue with actual model forward for full output ──
        result = original_forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

        # Result is (attn_output, attn_weights) where attn_output includes gating + o_proj
        attn_output = result[0] if isinstance(result, tuple) else result
        save("attn_output_gated", attn_output[0].float())  # engine ATTN-GATED
        save("o_proj_output", attn_output[0].float())      # engine ATTN-O-PROJ (same buffer after o_proj)

        return result

    # Apply the monkey-patch
    attn.forward = lambda *a, **kw: patched_forward(*a, **kw)

    # ── Run forward pass with engine's 15 token IDs ─────────────
    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long, device="cuda")
    print(f"\nRunning forward with {len(TOKEN_IDS)} tokens...")
    print(f"Token IDs: {TOKEN_IDS}")

    with torch.no_grad():
        model.eval()
        outputs = model(input_ids=input_ids, use_cache=False)

    # ── Save and display logits ─────────────────────────────────
    last_logits = outputs.logits[0, -1, :].float()
    save("logits_last_token", last_logits.float())

    top5 = torch.topk(last_logits, 5)
    print(f"\nTop 5 logits at last token position:")
    for i in range(5):
        print(f"  token {top5.indices[i].item()}: logit {top5.values[i].item():.4f}")

    # Reference token 248068 (if it exists in vocab)
    ref_token = 248068
    vocab_size = last_logits.shape[0]
    if ref_token < vocab_size:
        ref_logit = last_logits[ref_token].item()
        ref_rank = int((last_logits > ref_logit).sum().item())
        print(f"  Token {ref_token}: logit={ref_logit:.4f} rank={ref_rank}")
    else:
        print(f"  Token {ref_token} is out of vocab size {vocab_size}")

    # ── Print comparison summary ────────────────────────────────
    print(f"\n=== Reference Layer 3 Full Attention Stats ===")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".npy"):
            arr = np.load(os.path.join(OUTPUT_DIR, f))
            name = f[:-4]
            print(f"  REF {name}: shape={list(arr.shape)} mean_abs={np.abs(arr).mean():.6f}")

    print()
    print("=== Engine Layer 3 Stats (from smoke test) ===")
    engine_stats = {
        "q_full": 1.710352,           # ATTN-Q-FULL: after gate extract + Q-norm on first half
        "value_full": 0.609187,       # ATTN-V-FULL: after v_proj
        "key_rope_all": 0.920381,     # ATTN-K-FULL: after k_norm + RoPE (not pre-rope!)
        "q_h0_pre_rope": 1.000198,    # ATTN-Q-H0: head 0 after norm, before rope
        "k_h0_rope": 0.917587,       # ATTN-K-H0: head 0 (KV head 0) after RoPE
        "v_h0": 0.562760,            # ATTN-V-H0: head 0 value
        "softmax_h0": 0.066662,      # ATTN-SOFTMAX-H0: softmax weights head 0
        "attn_out_h0": 0.532119,     # ATTN-OUT-H0: attention output head 0
    }

    for name, engine_val in engine_stats.items():
        ref_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
        if os.path.exists(ref_path):
            arr = np.load(ref_path)
            ref_val = float(np.abs(arr).mean())
            ratio = engine_val / ref_val if ref_val > 0 else float("nan")
            print(f"  {name}: REF={ref_val:.6f}  ENG={engine_val:.6f}  ratio={ratio:.6f}")
        else:
            print(f"  {name}: NO REF  ENG={engine_val:.6f}")

    print(f"\nDone! Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
