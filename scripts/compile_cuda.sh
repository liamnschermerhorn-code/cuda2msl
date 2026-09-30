#!/bin/bash
#
# compile_cuda.sh — Transpile CUDA to Metal and compile to .metallib
#
# Usage:
#   ./scripts/compile_cuda.sh input.cu [input2.cu ...] -o output.metallib
#
# The process:
#   1. cuda2msl transpiles each .cu → .metal  (CUDA kernels → Metal Shading Language)
#   2. xcrun metal compiles each .metal → .air (Apple Intermediate Representation)
#   3. xcrun metallib links all .air → .metallib (GPU binary)
#
# Drop in your CUDA code, get a .metallib that runs on Apple Silicon.
#

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Parse args ──────────────────────────────────────────────────────────────

OUTPUT=""
INPUTS=()
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o)       OUTPUT="$2"; shift 2 ;;
        -v|--verbose) VERBOSE=true; shift ;;
        -h|--help)
            echo "Usage: $0 input.cu [input2.cu ...] -o output.metallib"
            echo ""
            echo "Transpiles CUDA .cu files to Metal and compiles into a .metallib."
            echo "The resulting .metallib runs on any Apple Silicon Mac."
            exit 0
            ;;
        *)        INPUTS+=("$1"); shift ;;
    esac
done

if [[ ${#INPUTS[@]} -eq 0 ]]; then
    echo "error: no input .cu files specified"
    echo "Usage: $0 input.cu [input2.cu ...] -o output.metallib"
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    # Default: first input basename with .metallib extension
    OUTPUT="${INPUTS[0]%.cu}.metallib"
fi

# ── Ensure cuda2msl is built ────────────────────────────────────────────────

CUDA2MSL="./build/src/transpiler/cuda2msl"
if [[ ! -f "$CUDA2MSL" ]]; then
    echo "==> Building cuda2msl transpiler..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
    cmake --build build --target cuda2msl 2>&1 | tail -1
fi

# ── Process each .cu file ──────────────────────────────────────────────────

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

AIR_FILES=()

for cu_file in "${INPUTS[@]}"; do
    base=$(basename "${cu_file%.cu}")
    metal_file="$TMPDIR/${base}.metal"
    air_file="$TMPDIR/${base}.air"

    # Step 1: Transpile CUDA → Metal
    echo "  [transpile] $cu_file → ${base}.metal"
    "$CUDA2MSL" "$cu_file" -o "$metal_file" 2>&1
    if $VERBOSE; then
        echo "    $(grep -c 'kernel void' "$metal_file") kernel(s) generated"
    fi

    # Step 2: Compile Metal → AIR
    echo "  [compile]   ${base}.metal → ${base}.air"
    xcrun metal -std=metal3.0 -O2 -c "$metal_file" -o "$air_file" 2>&1

    AIR_FILES+=("$air_file")
done

# ── Link into .metallib ─────────────────────────────────────────────────────

echo "  [link]      → $OUTPUT"
xcrun metallib "${AIR_FILES[@]}" -o "$OUTPUT" 2>&1

echo ""
echo "Done. $(ls -lh "$OUTPUT" | awk '{print $5}') metallib at: $OUTPUT"
echo ""

# List kernels
echo "Kernels:"
for cu_file in "${INPUTS[@]}"; do
    base=$(basename "${cu_file%.cu}")
    grep -o 'kernel void [a-zA-Z_][a-zA-Z0-9_]*' "$TMPDIR/${base}.metal" | sed 's/kernel void /  /'
done
