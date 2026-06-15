#!/usr/bin/env python3
"""Dump full-attention reference intermediates for layer 3 with consistent 15 engine token IDs."""
import os, numpy as np, torch, torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

MODEL_DIR = "/home/gary/opt/vllm/models/qwen3.6-27b-autoround-int4/"
OUTPUT_DIR = "/tmp/ref_attn_l3"
TOKEN_IDS = [248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(name, tensor):
    cpu_arr = tensor.detach().cpu().float().numpy()
    np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), cpu_arr)
    print(f"  {name}: shape={list(cpu_arr.shape)} mean_abs={np.abs(cpu_arr).mean():.6f}")

def main():
    from transformers import AutoModelForCausalLM
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    
    attn = model.model.layers[3].self_attn
    num_heads = model.config.num_attention_heads  # 24
    num_kv_heads = model.config.num_key_value_heads  # 4
    head_dim = model.config.head_dim  # 256
    print(f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    
    # Monkey-patch to capture ALL intermediates consistently
    captured = {}
    original_forward = attn.forward
    
    def patched_forward(self, hidden_states, **kwargs):
        position_embeddings = kwargs.get('position_embeddings')
        batch, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]  # [batch, seq]
        hidden_shape = (*input_shape, -1, self.head_dim)  # [batch, seq, heads, head_dim*2]
        num_heads = model.config.num_attention_heads
        head_dim = model.config.head_dim
        num_kv_heads = model.config.num_key_value_heads
        n_rep = num_heads // num_kv_heads
        scaling = self.scaling
        
        # Save input for verification
        save('layer3_input', hidden_states[0].float())
        
        # Q projection (doubled for Q+gate) - same as model
        q_full_raw = self.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2)
        save('q_full', q_full_raw[0].reshape(seq_len, num_heads * head_dim * 2).float())  # [15, 12288]
        
        # Split Q and gate (as model does)
        query_states, gate = torch.chunk(q_full_raw, 2, dim=-1)  # each [batch, seq, heads, head_dim]
        save('query_raw', query_states[0].reshape(seq_len, num_heads * head_dim).float())  # [15, 6144]
        
        gate = gate.reshape(*input_shape, -1)  # [batch, seq, heads*head_dim] = [1, 15, 6144]
        save('gate_raw', gate[0].float())  # [15, 6144]
        
        # Print gate statistics
        gate_np = gate[0].float().cpu().numpy()
        print(f'  Gate stats: mean_abs={np.abs(gate_np).mean():.6f} mean={gate_np.mean():.6f} min={gate_np.min():.4f} max={gate_np.max():.4f}')
        print(f'  Gate sign: pos_frac={float((gate_np > 0).mean()):.3f} neg_frac={float((gate_np < 0).mean()):.3f}')
        
        # Gate per-head for GPU 0 (heads 0-11) - reshape back to heads
        gate_per_head = gate.reshape(batch, seq_len, num_heads, head_dim)
        gate_gpu0 = gate_per_head[0, :, 0:12, :]  # [15, 12, 256]
        save('gate_gpu0', gate_gpu0.reshape(15, 3072).float())
        gate_gpu0_flat = gate_gpu0.reshape(15, 3072).float().cpu().numpy()
        print(f'  Gate GPU0: mean_abs={np.abs(gate_gpu0_flat).mean():.6f} mean={gate_gpu0_flat.mean():.6f}')
        print(f'  Gate GPU0 first 10 values (last token): {gate_gpu0_flat[-1, :10].tolist()}')
        
        # Sigmoid of gate (for verification)
        sigmoid_gate = torch.sigmoid(gate)  # [1, 15, 6144]
        save('sigmoid_gate', sigmoid_gate[0].float())
        sig_np = sigmoid_gate[0].float().cpu().numpy()
        print(f'  Sigmoid gate: mean={sig_np.mean():.6f} min={sig_np.min():.6f} max={sig_np.max():.6f}')
        
        # Q norm (as model does)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)  # [batch, heads, seq, head_dim]
        save('query_normed', query_states[0].reshape(num_heads * head_dim, seq_len).T.float())  # [15, 6144]
        
        # K + norm
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [batch, kv_heads, seq, head_dim]
        save('key_raw', self.k_proj(hidden_states)[0].float())
        save('key_normed', key_states[0].reshape(num_kv_heads * head_dim, seq_len).T.float())
        
        # V
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [batch, kv_heads, seq, head_dim]
        save('value_raw', value_states[0].reshape(num_kv_heads * head_dim, seq_len).T.float())
        
        # GQA expand
        key_expanded = key_states.repeat_interleave(n_rep, dim=1)   # [batch, heads, seq, head_dim]
        value_expanded = value_states.repeat_interleave(n_rep, dim=1)
        
        # RoPE from kwargs
        cos, sin = position_embeddings
        query_states, key_expanded = apply_rotary_pos_emb(query_states, key_expanded, cos, sin)
        
        # Compute attention for head 0 manually
        q_h0 = query_states[0, 0]  # [seq, head_dim]
        k_h0 = key_expanded[0, 0]
        v_h0 = value_expanded[0, 0]
        
        save('q_h0_rope', q_h0.float())
        save('k_h0_rope', k_h0.float())
        save('v_h0', v_h0.float())
        
        # V for GPU 0 heads (0-11)
        v_gpu0 = value_expanded[0, 0:12]  # [12, seq, head_dim]
        save('v_gpu0_all', v_gpu0.reshape(seq_len, 3072).float())
        
        # Attention scores for head 0
        scores = torch.matmul(q_h0.unsqueeze(0), k_h0.unsqueeze(0).transpose(-2, -1)) * scaling
        mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        save('softmax_h0', probs[0].float())
        
        attn_out_h0 = torch.matmul(probs, v_h0.unsqueeze(0))
        save('attn_out_h0', attn_out_h0[0].float())
        
        # Call original forward for actual output
        result = original_forward(hidden_states, **kwargs)
        return result
    
    attn.forward = lambda *a, **kw: patched_forward(attn, *a, **kw)
    
    input_ids = torch.tensor([TOKEN_IDS], dtype=torch.long, device='cuda')
    print(f"Running forward with {len(TOKEN_IDS)} tokens...")
    
    with torch.no_grad():
        model.eval()
        outputs = model(input_ids=input_ids, use_cache=False)
    
    save("logits_lastpos", outputs.logits[0, -1, :].float())
    
    last_logits = outputs.logits[0, -1, :].float()
    top5 = torch.topk(last_logits, 5)
    print(f"\nTop 5 logits:")
    for i in range(5):
        print(f"  token {top5.indices[i].item()}: logit {top5.values[i].item():.4f}")
    print(f"  Token 248068: logit={last_logits[248068].item():.4f} rank={int((last_logits > last_logits[248068]).sum())}")
    
    print(f"\nDone! Files in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
