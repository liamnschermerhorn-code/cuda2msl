#!/bin/bash
# compile_pytorch_shaders.sh
#
# Compiles all PyTorch Metal shaders (.metal) into individual .metallib files
# and a single combined pytorch_kernels.metallib for loading via ShaderLoader.
#
# Usage:
#   ./scripts/compile_pytorch_shaders.sh [input_dir] [output_dir]
#
# Defaults:
#   input_dir  = ~/Downloads/metal_shaders
#   output_dir = build/pytorch_shaders

set -euo pipefail

INPUT_DIR="${1:-$HOME/Downloads/metal_shaders}"
OUTPUT_DIR="${2:-$(dirname "$0")/../build/pytorch_shaders}"
SDK="macosx"
METAL_STD="metal3.0"
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)

mkdir -p "$OUTPUT_DIR/air"

echo "=== PyTorch Metal Shader Compilation ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Jobs:   $JOBS"
echo ""

# Phase 1: Compile .metal -> .air (parallelized)
TOTAL=$(ls "$INPUT_DIR"/*.metal 2>/dev/null | wc -l | tr -d ' ')
echo "Phase 1: Compiling $TOTAL .metal files to .air ..."

COMPILED=0
FAILED=0
FAIL_LIST=""

compile_one() {
    local src="$1"
    local name
    name=$(basename "$src" .metal)
    local air="$OUTPUT_DIR/air/$name.air"

    if xcrun -sdk "$SDK" metal -c "$src" -o "$air" -std="$METAL_STD" -w 2>/dev/null; then
        echo "  OK: $name"
        return 0
    else
        echo "  FAIL: $name"
        return 1
    fi
}

export -f compile_one
export OUTPUT_DIR SDK METAL_STD

# Use xargs for parallel compilation
RESULTS=$(find "$INPUT_DIR" -name '*.metal' -print0 | \
    xargs -0 -P "$JOBS" -I {} bash -c 'compile_one "$@"' _ {} 2>&1) || true

echo "$RESULTS"

# Count results
COMPILED=$(echo "$RESULTS" | grep -c "^  OK:" || true)
FAILED=$(echo "$RESULTS" | grep -c "^  FAIL:" || true)

echo ""
echo "Phase 1 complete: $COMPILED compiled, $FAILED failed"

# Phase 2: Link all .air files into combined .metallib
AIR_FILES=$(find "$OUTPUT_DIR/air" -name '*.air' 2>/dev/null | sort)
AIR_COUNT=$(echo "$AIR_FILES" | wc -l | tr -d ' ')

if [ "$AIR_COUNT" -eq 0 ]; then
    echo "ERROR: No .air files produced, cannot link metallib"
    exit 1
fi

echo ""
echo "Phase 2: Linking $AIR_COUNT .air files into pytorch_kernels.metallib ..."

COMBINED="$OUTPUT_DIR/pytorch_kernels.metallib"
echo "$AIR_FILES" | xargs xcrun -sdk "$SDK" metallib -o "$COMBINED"

echo "  -> $COMBINED ($(du -h "$COMBINED" | cut -f1))"

# Phase 3: Also create per-category metallibs for selective loading
echo ""
echo "Phase 3: Creating category metallibs ..."

declare -A CATEGORIES
for air in $AIR_FILES; do
    name=$(basename "$air" .air)
    # Extract category from naming convention (e.g., ActivationElu -> Activation)
    category=$(echo "$name" | sed -E 's/([A-Z][a-z]+).*/\1/')
    CATEGORIES[$category]+=" $air"
done

for category in $(echo "${!CATEGORIES[@]}" | tr ' ' '\n' | sort); do
    airs="${CATEGORIES[$category]}"
    count=$(echo "$airs" | wc -w | tr -d ' ')
    catlib="$OUTPUT_DIR/${category,,}_kernels.metallib"
    echo "$airs" | xargs xcrun -sdk "$SDK" metallib -o "$catlib" 2>/dev/null && \
        echo "  $category: $count kernels -> $(basename "$catlib")" || \
        echo "  $category: FAILED to link"
done

echo ""
echo "=== Done ==="
echo "Combined metallib: $COMBINED"
echo "Set CUDA_METAL_SHADER_DIR=$OUTPUT_DIR to load at runtime"
