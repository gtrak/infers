# Oracle Hidden State Compare

End-to-end regression test comparing per-layer hidden states between the Rust CUDA engine and a PyTorch oracle (transformers).

## Oracle Dump

Script `scripts/dump_oracle_hidden.py` loads a model via transformers, hooks all 64 layers to capture residual stream input, and writes raw BF16 + JSON metadata matching the probe dump format.

NVFP4 models from vLLM have a config namespace mismatch — quantization target regexes use `language_model.model.layers` instead of `model.layers`. The script patches this in-memory via `patch_nvfp4_config()` before loading. See [[testing#Oracle Hidden State Dumps]].

## Comparison Script

Script `scripts/compare_hidden_states.py` reads oracle dumps and engine probe dumps, computes per-layer cosine similarity, max/mean absolute error, and classifies PASS (cos > 0.99), WARN (0.95–0.99), FAIL (< 0.95).

The comparison reads BF16 via `torch.frombuffer` for correct bit-pattern conversion. Engine probe dumps come from `INFERS_DUMP_DIR` with stage names `{attn|gdn}.norm1_input_gpu0` matching the oracle's layer input.

## INT4 Results (After Softmax Fix)

Prompt: "The capital of France is" (5 tokens), TP=2 vs PyTorch device_map="auto".

Final logits cosine: 0.99472 (PASS). All 65 layers PASS or WARN — zero FAIL. Worst layer cosine: 0.96451 (layer 52). The remaining WARN divergence is gradual BF16 rounding compounding between TP=2 and sequential device_map — expected and not a bug.

## Softmax Kernel Bug (Fixed)

The softmax kernel's shared memory reduction failed when `block_dim` was non-power-of-2. For `seq_len=5`, `block_size=5` caused thread 4's data to be orphaned, producing softmax values of 241 instead of ~1.0 for the last row's diagonal element.

Fix: always use `block_size = 256` in the launcher. This caused catastrophic hidden state divergence starting at layer 19: values exploded from max=153 to max=14016 because softmax amplified through V projection and o_proj GEMM.
