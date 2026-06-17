#!/usr/bin/env bash
# ------------------------------------------------------------------
# GDN Pipeline Verification Script
#
# Orchestrates:
#   1. Run engine prefill with dump enabled for specified layers
#   2. Compare engine dumps against PyTorch reference
#
# Usage:
#   # Dump all layers and compare against reference
#   ./tests/run_gdn_verify.sh --layers all --model-dir ~/opt/vllm/models/qwen3.6-27b-autoround-int4/
#
#   # Dump specific layers
#   ./tests/run_gdn_verify.sh --layers 0,7,13
#
#   # Re-analyze existing dumps
#   ./tests/run_gdn_verify.sh --dump-dir /tmp/engine_dump/ --compare-only
#
#   # Run engine but skip comparison
#   ./tests/run_gdn_verify.sh --layers all --dump-only
#
# Environment:
#   INFERS_TEST_MODEL   Path to model (default: ~/opt/vllm/models/qwen3.6-27b-autoround-int4/)
#   INFERS_DUMP_DIR     Where to save dumps (default: /tmp/gdn_verify_dump/)
# ------------------------------------------------------------------
set -euo pipefail

MODEL_DIR="${INFERS_TEST_MODEL:-$HOME/opt/vllm/models/qwen3.6-27b-autoround-int4/}"
DUMP_DIR="${INFERS_DUMP_DIR:-/tmp/gdn_verify_dump}"
COMPARE_SCRIPT="$(dirname "$0")/gdn_compare.py"
RUST_PROJECT="$(dirname "$0")/.."

# Default to dump all layers
LAYERS="${INFERS_DUMP_GDN_LAYER:-all}"
COMPARE_ONLY=false
DUMP_ONLY=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layers) LAYERS="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --dump-dir) DUMP_DIR="$2"; shift 2 ;;
        --compare-only) COMPARE_ONLY=true; shift ;;
        --dump-only) DUMP_ONLY=true; shift ;;
        --verbose|-v) VERBOSE="-v"; shift ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "  --layers all|N,N,...    Layers to dump (default: all)"
            echo "  --model-dir DIR          Model weights path"
            echo "  --dump-dir DIR           Dump output directory"
            echo "  --compare-only           Skip engine run, compare existing dumps"
            echo "  --dump-only              Skip comparison, just dump"
            echo "  --verbose                Detailed comparison output"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "GDN Pipeline Verification"
echo "========================================"
echo "Model:       $MODEL_DIR"
echo "Dump dir:    $DUMP_DIR"
echo "Layers:      $LAYERS"
echo ""

# Step 1: Run the engine with dump enabled
if [[ "$COMPARE_ONLY" != "true" ]]; then
    echo "[Step 1] Running engine prefill with dump..."
    echo "  INFERS_DUMP_GDN_LAYER=$LAYERS"
    echo "  INFERS_DUMP_GDN_DIR=$DUMP_DIR"

    # Clean dump directory
    rm -rf "$DUMP_DIR"
    mkdir -p "$DUMP_DIR"

    # Run the smoke test with dump env vars
    cd "$RUST_PROJECT"
    INFERS_DUMP_GDN_LAYER="$LAYERS" \
    INFERS_DUMP_GDN_DIR="$DUMP_DIR" \
    INFERS_TEST_MODEL="$MODEL_DIR" \
        cargo test --package infers-backend-native --test smoke_test \
        smoke_test_real_model -- --ignored --nocapture 2>&1 | tee "$DUMP_DIR/engine_output.log"

    ENGINE_EXIT="${PIPESTATUS[0]}"
    if [[ "$ENGINE_EXIT" -ne 0 ]]; then
        echo "[ERROR] Engine run failed (exit code $ENGINE_EXIT)"
        echo "        Check $DUMP_DIR/engine_output.log for details"
        exit "$ENGINE_EXIT"
    fi
    echo "[Step 1] Engine dump complete: $DUMP_DIR"
else
    echo "[Step 1] Skipped (--compare-only)"
fi

# Step 2: Compare against reference
if [[ "$DUMP_ONLY" != "true" ]]; then
    echo ""
    echo "[Step 2] Comparing engine dumps against reference..."

    if [[ -d "$DUMP_DIR/layer_0" ]]; then
        # Multi-layer dump format
        python3 "$COMPARE_SCRIPT" \
            --dump-dir "$DUMP_DIR" \
            --model-dir "$MODEL_DIR" \
            ${VERBOSE:-}
    else
        # Single layer (or old format)
        python3 "$COMPARE_SCRIPT" \
            --engine-dir "$DUMP_DIR" \
            --model-dir "$MODEL_DIR" \
            ${VERBOSE:-}
    fi

    COMPARE_EXIT=$?
    if [[ "$COMPARE_EXIT" -eq 0 ]]; then
        echo ""
        echo "========================================"
        echo "VERIFICATION PASSED"
        echo "========================================"
    else
        echo ""
        echo "========================================"
        echo "VERIFICATION FAILED"
        echo "========================================"
        exit "$COMPARE_EXIT"
    fi
else
    echo "[Step 2] Skipped (--dump-only)"
fi
