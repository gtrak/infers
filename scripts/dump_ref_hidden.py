#!/usr/bin/env python3
"""Dump per-layer hidden states from reference HuggingFace model.

Writes:
  /tmp/ref_hidden/input_ids.pt         - input token IDs
  /tmp/ref_hidden/layer_{i}_lastpos.pt - hidden state at last token position after layer i
  /tmp/ref_hidden/layer_{i}_full.pt    - full [seq_len, hidden_size] hidden state after layer i
  /tmp/ref_hidden/all_layers_lastpos.pt - stacked [n_layers+1, hidden_size] last-position states
  /tmp/ref_hidden/logits_lastpos.pt    - final logits at last position

Uses HuggingFace Transformers + auto_round for the INT4 model.
GPU-accelerated via CUDA 13.0+ (RTX 5060 Ti Blackwell).
"""

import torch
import time
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get(
    "INFERS_REF_MODEL",
    os.path.expanduser("~/opt/vllm/models/qwen3.6-27b-autoround-int4"),
)
OUTPUT_DIR = os.environ.get("REF_HIDDEN_DIR", "/tmp/ref_hidden")
PROMPT = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

# Ensure we use CUDA if available
DEVICE = os.environ.get("REF_HIDDEN_DEVICE", "auto")  # "auto", "cuda:0", or "cpu"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device mode: {DEVICE}", flush=True)
    print(f"Loading model from {MODEL_PATH}...", flush=True)

    t0 = time.time()

    if DEVICE == "cpu":
        device_map = "cpu"
    elif DEVICE == "auto":
        device_map = "auto"
    else:
        device_map = DEVICE

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device_map,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)
    if hasattr(model, "hf_device_map"):
        gpu_layers = sum(1 for v in model.hf_device_map.values() if isinstance(v, int))
        print(f"Device map: {gpu_layers} GPU layers", flush=True)
    print(f"Model device: {model.device}", flush=True)

    # Use engine's exact 15 token IDs instead of HF tokenizer
    TOKEN_IDS = [248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198]
    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long, device=model.device)
    inputs = {"input_ids": input_ids}

    print(f"Input IDs ({input_ids.shape}): {input_ids[0].tolist()}", flush=True)
    torch.save(input_ids[0].cpu(), f"{OUTPUT_DIR}/input_ids.pt")

    # Run forward pass collecting hidden states from all layers
    print(f"Running forward pass ({input_ids.shape[1]} tokens)...", flush=True)
    t0 = time.time()

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )

    total_time = time.time() - t0
    print(f"Forward pass completed in {total_time:.1f}s", flush=True)

    n_layers = len(outputs.hidden_states) - 1  # exclude embedding output
    print(f"Hidden states: {len(outputs.hidden_states)} (embed + {n_layers} layers)", flush=True)

    # hidden_states[0] = embedding output, hidden_states[i] = after layer (i-1)
    for i, hs in enumerate(outputs.hidden_states):
        lastpos = hs[0, -1, :].cpu().float()
        torch.save(lastpos, f"{OUTPUT_DIR}/layer_{i}_lastpos.pt")

        # Also save full hidden state for quick comparison
        full = hs[0].cpu().float()  # [seq_len, hidden_size]
        torch.save(full, f"{OUTPUT_DIR}/layer_{i}_full.pt")

        if i <= 3 or i == n_layers:
            print(f"  layer[{i}]: shape={hs.shape} "
                  f"first5={hs[0,-1,:5].tolist()}", flush=True)

    # Stack all last-position hidden states for easy comparison
    stacked = torch.stack([
        hs[0, -1, :].cpu().float()
        for hs in outputs.hidden_states
    ])
    torch.save(stacked, f"{OUTPUT_DIR}/all_layers_lastpos.pt")
    print(f"Saved stacked: {stacked.shape}", flush=True)

    # Logits
    torch.save(outputs.logits[0, -1, :].cpu().float(), f"{OUTPUT_DIR}/logits_lastpos.pt")
    print(f"Logits lastpos shape: {outputs.logits[0, -1, :].shape}", flush=True)

    # Final norm hidden state
    torch.save(
        outputs.hidden_states[-1][0, -1, :].cpu().float(),
        f"{OUTPUT_DIR}/final_hidden_lastpos.pt",
    )

    print(f"\nAll reference data saved to {OUTPUT_DIR}/", flush=True)
    print(f"Files: {sorted(os.listdir(OUTPUT_DIR))}", flush=True)


if __name__ == "__main__":
    main()
