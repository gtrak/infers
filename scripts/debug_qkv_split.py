#!/usr/bin/env python3
"""Debug the QKV split by comparing raw column slices of conv_out against reference Q/K/V."""
import numpy as np
import os

OUR_DIR = "/tmp/our_gdn"
REF_DIR = "/tmp/ref_gdn_new"

def bf16_raw_to_f32(path):
    data = np.frombuffer(open(path, "rb").read(), dtype=np.uint16)
    return (data.astype(np.uint32) << 16).view(np.float32)

def cos_sim(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

# Load our conv_out: [15, 5120]
our_conv = bf16_raw_to_f32(os.path.join(OUR_DIR, "conv_out.raw"))
our_conv_2d = our_conv.reshape(15, 5120)

# Load reference conv_out, query, key, value
ref_conv = np.load(os.path.join(REF_DIR, "conv_out.npy"))  # [15, 10240]
ref_query = np.load(os.path.join(REF_DIR, "query.npy"))     # [15, 2048]
ref_key = np.load(os.path.join(REF_DIR, "key.npy"))         # [15, 2048]
ref_value = np.load(os.path.join(REF_DIR, "value.npy"))     # [15, 6144]

# Load our query, key, value dumps
our_query = bf16_raw_to_f32(os.path.join(OUR_DIR, "query.raw")).reshape(15, 1024)
our_key = bf16_raw_to_f32(os.path.join(OUR_DIR, "key.raw")).reshape(15, 1024)
our_value = bf16_raw_to_f32(os.path.join(OUR_DIR, "value.raw")).reshape(15, 3072)

print(f"Our conv_out shape: {our_conv_2d.shape}")
print(f"Ref conv_out shape: {ref_conv.shape}")
print(f"Ref query shape: {ref_query.shape}")
print(f"Ref key shape: {ref_key.shape}")
print(f"Ref value shape: {ref_value.shape}")

# Compare different column slices of our conv_out against reference
print("\n=== Comparing column slices of our conv_out against reference ===")

# Our conv_out[:, :1024] should match ref conv_out[:, :1024] or ref query[:, :1024]
print(f"\nour_conv[:, :1024] vs ref_conv[:, :1024]:  cos={cos_sim(our_conv_2d[:, :1024], ref_conv[:, :1024]):.6f}")
print(f"our_conv[:, :1024] vs ref_query[:, :1024]: cos={cos_sim(our_conv_2d[:, :1024], ref_query[:, :1024]):.6f}")

# Our conv_out[:, 1024:2048] — is this more query or start of key?
print(f"our_conv[:, 1024:2048] vs ref_query[:, 1024:2048]: cos={cos_sim(our_conv_2d[:, 1024:2048], ref_query[:, 1024:2048]):.6f}")
print(f"our_conv[:, 1024:2048] vs ref_key[:, :1024]: cos={cos_sim(our_conv_2d[:, 1024:2048], ref_key[:, :1024]):.6f}")

# Our conv_out[:, 2048:4096] — is this key or something else?
print(f"our_conv[:, 2048:3072] vs ref_key[:, :1024]: cos={cos_sim(our_conv_2d[:, 2048:3072], ref_key[:, :1024]):.6f}")
print(f"our_conv[:, 2048:4096] vs ref_key[:, :2048]: cos={cos_sim(our_conv_2d[:, 2048:4096], ref_key[:, :2048]):.6f}")

# Our conv_out[:, 4096:5120] — partial value
print(f"our_conv[:, 4096:5120] vs ref_value[:, :1024]: cos={cos_sim(our_conv_2d[:, 4096:5120], ref_value[:, :1024]):.6f}")

# Now compare our dumped query/key/value against various slices
print("\n=== Our dumped query/key/value against reference slices ===")
print(f"our_query vs ref_query[:, :1024]: cos={cos_sim(our_query, ref_query[:, :1024]):.6f}")
print(f"our_query vs ref_conv[:, :1024]: cos={cos_sim(our_query, ref_conv[:, :1024]):.6f}")
print(f"our_key vs ref_key[:, :1024]: cos={cos_sim(our_key, ref_key[:, :1024]):.6f}")
print(f"our_key vs ref_query[:, 1024:2048]: cos={cos_sim(our_key, ref_query[:, 1024:2048]):.6f}")
print(f"our_value vs ref_value[:, :3072]: cos={cos_sim(our_value, ref_value[:, :3072]):.6f}")

# Also compare our conv_out vs ref conv_out (first 5120 columns)
print(f"\nour_conv vs ref_conv[:, :5120]: cos={cos_sim(our_conv_2d, ref_conv[:, :5120]):.6f}")

# Check token 0 values to spot patterns
print("\n=== Token 0, first 5 values ===")
print(f"our_conv[0, :5]: {our_conv_2d[0, :5]}")
print(f"ref_conv[0, :5]: {ref_conv[0, :5]}")
print(f"our_query[0, :5]: {our_query[0, :5]}")
print(f"ref_query[0, :5]: {ref_query[0, :5]}")
