#!/bin/bash
#
# quickstart.sh — Validate the entire CUDA-to-Metal stack in <2 minutes.
#
# Builds, tests, downloads a small model, and runs inference in both
# standard and speculative modes to verify everything works end-to-end.
#
# Usage:
#   ./scripts/quickstart.sh
#

set -euo pipefail
cd "$(dirname "$0")/.."

NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_DIR="./models/SmolLM2-135M-Instruct"

echo "=== CUDA-to-Metal Quickstart ==="
echo ""

# ── Step 1: Build ─────────────────────────────────────────────────────────
echo "[1/5] Building..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev 2>/dev/null
cmake --build build -j"$NCPU" 2>&1 | tail -1
echo "  Build complete."

# ── Step 2: Unit tests ───────────────────────────────────────────────────
echo "[2/5] Running unit tests..."
cd build
ctest -R "test_memory|test_device|test_stream|test_event|test_gemm|test_activation|test_softmax|test_secure_weight_pool" --timeout 30 2>&1 | tail -3
cd ..
echo "  Tests passed."

# ── Step 3: Download model (if needed) ───────────────────────────────────
echo "[3/5] Ensuring test model is available..."
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "  Downloading $MODEL_ID (~270MB)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR')
"
    echo "  Downloaded."
else
    echo "  Model already present."
fi

# ── Step 4: Standard inference ───────────────────────────────────────────
echo "[4/5] Standard greedy inference..."
OUTPUT=$(echo "Hello" | ./build/metal-chat "$MODEL_DIR" --greedy --max-tokens 15 2>&1)
GENERATED=$(echo "$OUTPUT" | grep "\[generated" | head -1)
# Extract text between prefill line and generated line
RESPONSE=$(echo "$OUTPUT" | sed -n '/\[prefill:/,/\[generated/p' | grep -v '^\[' | head -1)
echo "  Response: $RESPONSE"
echo "  $GENERATED"

# Sanity: check we got actual text, not garbage
if echo "$RESPONSE" | grep -qi "endoftext\|NaN\|inf"; then
    echo "  ERROR: Output looks like garbage. Something is wrong."
    exit 1
fi

# ── Step 5: Speculative inference ────────────────────────────────────────
echo "[5/5] Speculative decoding (K=3)..."
OUTPUT=$(echo "Hello" | ./build/metal-chat "$MODEL_DIR" --speculative --draft-k 3 --greedy --max-tokens 15 2>&1)
SPEC_STATS=$(echo "$OUTPUT" | grep "\[speculative:" | head -1)
RESPONSE=$(echo "$OUTPUT" | sed -n '/\[prefill:/,/\[speculative/p' | grep -v '^\[' | head -1)
echo "  Response: $RESPONSE"
echo "  $SPEC_STATS"

echo ""
echo "=== All checks passed. CUDA-to-Metal stack is operational. ==="
echo ""
echo "Next steps:"
echo "  ./build/metal-chat models/SmolLM2-135M-Instruct/       # interactive chat"
echo "  ./scripts/run_model.sh <huggingface_model_id>           # try other models"
