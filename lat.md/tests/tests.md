# Tests

Test specifications and diagnostic tools for the infers CUDA runtime.

- [[gemm_compare]] — bf16_gemm_tiled vs cuBLAS regression test
- [[nvfp4_ref_compare]] — NVFP4 dequant→GEMM pipeline vs Python reference
- [[gdn_recurrent_step_test]] — GDN recurrent step kernel vs CPU and Python references
- [[rms_norm_gated_test]] — RMSNorm + SiLU gate kernel vs CPU and Python references
- [[gdn_chunked_prefill_test]] — GDN chunked prefill kernel vs sequential CPU and Python references
- [[oracle_hidden_state_compare]] — Per-layer hidden state comparison against PyTorch oracle (INT4 + NVFP4)
- [[nvfp4_fused_compare]] — nvfp4_gemm_fused vs dequant+GEMM direct GPU comparison
