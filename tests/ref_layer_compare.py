"""
Compare per-layer hidden state stats between engine and PyTorch reference.
Usage:
  python3 tests/ref_layer_compare.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import json
import numpy as np

PROMPT_TOKENS = [248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198]
MODEL_PATH = "/home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4"

def load_and_run():
    print("Loading model with auto_round...")
    from auto_round import AutoRoundConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load with the correct quantization config
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"Model loaded: {model.config.model_type}")

    # Tokenize
    tokens = torch.tensor([PROMPT_TOKENS], device="cuda:0")
    print(f"Input tokens: {tokens.shape}")

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            tokens,
            output_hidden_states=True,
            return_dict=True,
        )

    # Get hidden states for each layer
    hidden_states = outputs.hidden_states  # tuple of (embedding + each layer output)
    print(f"\nPyTorch per-layer hidden state stats:")
    print(f"{'Layer':>8} {'mean_abs':>10} {'min':>10} {'max':>10} {'nan':>6}")
    print("-" * 50)
    for i, h in enumerate(hidden_states):
        h_f32 = h[0].float().cpu().numpy()  # first token, all hidden dims
        mean_abs = np.mean(np.abs(h_f32))
        min_val = np.min(h_f32)
        max_val = np.max(h_f32)
        nan_count = np.sum(np.isnan(h_f32))
        print(f"{i-1:>8} {mean_abs:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {nan_count:>6}")

    # Logits
    logits = outputs.logits[0]  # [seq_len, vocab_size]
    # Get predicted token for last position
    last_logits = logits[-1]  # [vocab_size]
    probs = torch.softmax(last_logits, dim=-1)
    top5 = torch.topk(probs, 5)
    print(f"\nLast token top-5:")
    for i in range(5):
        print(f"  {top5.indices[i].item():>8}  {top5.values[i].item():.4f}")

if __name__ == "__main__":
    load_and_run()
