#!/usr/bin/env bash
# Extract FlashInfer kernel source files from a local vLLM checkout.
#
# Usage: ./scripts/extract-kernels.sh [path/to/vllm]
#
# This script copies the GDN (Gated DeltaNet) and standard attention
# kernel source files from vLLM's FlashInfer submodule into our kernels/
# directory. After extraction, run `cargo build` to compile them with nvcc.
#
# Expected vLLM directory structure:
#   vllm/
#     vllm/
#       model_executor/layers/mamba/   -- GDN kernels
#       v1/attention/backends/         -- attention backend dispatch
#     csrc/                             -- C++/CUDA source files
#     flashinfer/                       -- FlashInfer submodule (if present)

set -euo pipefail

VLLM_DIR="${1:-../vllm}"
KERNEL_DIR="$(dirname "$0")/../crates/cuda/kernels"

echo "Extracting FlashInfer kernels from: $VLLM_DIR"
echo "Output directory: $KERNEL_DIR"

# Create output directories
mkdir -p "$KERNEL_DIR/flashinfer-gdn"
mkdir -p "$KERNEL_DIR/flashinfer-attn"
mkdir -p "$KERNEL_DIR/compiled"

# Extract GDN (Gated DeltaNet) kernels
# These implement chunk_gated_delta_rule (prefill) and
# fused_sigmoid_gating_delta_rule_update (decode) for the
# 48 GDN layers in Qwen3.6-27B.
if [ -d "$VLLM_DIR/vllm/model_executor/layers/mamba/gdn" ]; then
    echo "Found GDN kernels..."
    cp -v "$VLLM_DIR"/vllm/model_executor/layers/mamba/gdn/*.cu "$KERNEL_DIR/flashinfer-gdn/" 2>/dev/null || true
    cp -v "$VLLM_DIR"/vllm/model_executor/layers/mamba/gdn/*.cuh "$KERNEL_DIR/flashinfer-gdn/" 2>/dev/null || true
    cp -vr "$VLLM_DIR"/vllm/model_executor/layers/mamba/gdn/include/ "$KERNEL_DIR/flashinfer-gdn/" 2>/dev/null || true
    echo "GDN kernels extracted."
else
    echo "WARNING: GDN kernel directory not found at $VLLM_DIR/vllm/model_executor/layers/mamba/gdn"
    echo "The GDN kernels are required for Qwen3.6-27B's hybrid attention architecture."
fi

# Extract standard FlashInfer attention kernels
# These implement BatchPrefillWithPagedKVCache and BatchDecodeWithPagedKVCache
# for the 16 full-attention layers in Qwen3.6-27B.
if [ -d "$VLLM_DIR/csrc" ]; then
    echo "Found standard attention kernels in csrc..."
    # Look for FlashInfer source files
    find "$VLLM_DIR/csrc" -name "*.cu" -path "*flashinfer*" -exec cp -v {} "$KERNEL_DIR/flashinfer-attn/" \; 2>/dev/null || true
    find "$VLLM_DIR/csrc" -name "*.cu" -path "*prefill*" -exec cp -v {} "$KERNEL_DIR/flashinfer-attn/" \; 2>/dev/null || true
    find "$VLLM_DIR/csrc" -name "*.cu" -path "*decode*" -exec cp -v {} "$KERNEL_DIR/flashinfer-attn/" \; 2>/dev/null || true
    find "$VLLM_DIR/csrc" -name "*.cu" -path "*sampling*" -exec cp -v {} "$KERNEL_DIR/flashinfer-attn/" \; 2>/dev/null || true
    echo "Standard attention kernels extracted."
else
    echo "WARNING: Standard attention kernel directory not found at $VLLM_DIR/csrc"
fi

# Check for FlashInfer submodule
if [ -d "$VLLM_DIR/flashinfer" ]; then
    echo "Found FlashInfer submodule..."
    # Extract headers
    find "$VLLM_DIR/flashinfer" -name "*.cuh" -path "*include*" -exec cp -v {} "$KERNEL_DIR/flashinfer-attn/" \; 2>/dev/null || true
    echo "FlashInfer headers extracted."
else
    echo "WARNING: FlashInfer submodule not found at $VLLM_DIR/flashinfer"
    echo "You may need to init submodules: cd $VLLM_DIR && git submodule update --init --recursive"
fi

echo ""
echo "Kernel extraction complete."
echo "Kernel source files are in: $KERNEL_DIR/flashinfer-gdn/ and $KERNEL_DIR/flashinfer-attn/"
echo "Compiled .cubin files will be placed in: $KERNEL_DIR/compiled/"
echo ""
echo "Next steps:"
echo "  1. Review extracted .cu files in kernels/"
echo "  2. Run build.rs to compile kernels with nvcc (requires CUDA toolkit)"
echo "  3. Kernel loading happens at runtime via KernelRegistry"
