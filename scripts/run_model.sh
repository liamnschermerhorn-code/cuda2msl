#!/bin/bash
#
# run_model.sh — Download a HuggingFace model and run it via metal-chat.
#
# Usage:
#   ./scripts/run_model.sh <model_id> [metal-chat args...]
#
# Examples:
#   ./scripts/run_model.sh HuggingFaceTB/SmolLM2-135M-Instruct
#   ./scripts/run_model.sh HuggingFaceTB/SmolLM2-360M-Instruct --temp 0.5
#   ./scripts/run_model.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 --greedy
#   ./scripts/run_model.sh mistralai/Mistral-7B-Instruct-v0.2
#
# The process:
#   1. Downloads the model from HuggingFace Hub to ./models/<name>/
#   2. Builds metal-chat (if not already built)
#   3. Runs inference through the CUDA-to-Metal translation layer
#
# Requirements:
#   - Python 3 with huggingface_hub installed
#   - CMake 3.20+ and a C++ compiler
#

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ID="${1:?Usage: $0 <huggingface_model_id> [args...]}"
shift
EXTRA_ARGS=("$@")

# Derive local directory name from model ID
MODEL_NAME="$(basename "$MODEL_ID")"
MODEL_DIR="./models/$MODEL_NAME"

# ── Step 1: Download model ──────────────────────────────────────────────────

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "==> Downloading $MODEL_ID ..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR')
"
    echo "==> Model downloaded to $MODEL_DIR"
else
    echo "==> Model already present at $MODEL_DIR"
fi

# ── Step 2: Build ───────────────────────────────────────────────────────────

if [ ! -f ./build/metal-chat ]; then
    echo "==> Building metal-chat ..."
    cmake -B build
    cmake --build build --target metal-chat
fi

# ── Step 3: Run ─────────────────────────────────────────────────────────────

echo "==> Starting inference ..."
echo ""
exec ./build/metal-chat "$MODEL_DIR" "${EXTRA_ARGS[@]}"
