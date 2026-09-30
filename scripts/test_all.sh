#!/bin/bash
# test_all.sh — Run ALL tests (must be run outside Claude Code sandbox)
#
# Usage:
#   bash scripts/test_all.sh          # Run all tests
#   bash scripts/test_all.sh --bench  # Include GPU benchmarks
#   bash scripts/test_all.sh --quick  # Skip slow integration tests
#
# This must be run from a normal terminal, NOT from within Claude Code,
# because Claude Code's sandbox blocks MTLCreateSystemDefaultDevice().

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

BENCH=false
QUICK=false
for arg in "$@"; do
    case "$arg" in
        --bench) BENCH=true ;;
        --quick) QUICK=true ;;
    esac
done

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           CUDA-to-Metal Full Test Suite                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# System info
echo -e "${CYAN}System:${NC}"
echo "  macOS:  $(sw_vers -productVersion)"
echo "  Chip:   $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
echo "  Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB", $1/1073741824}')"
echo ""

TOTAL_PASS=0
TOTAL_FAIL=0

run_section() {
    local title="$1"
    echo -e "${BOLD}━━━ $title ━━━${NC}"
}

# ─── Section 1: Build ─────────────────────────────────────────────
run_section "Build"
if [ ! -f build/Makefile ]; then
    echo "Building..."
    cmake -B build 2>&1 | tail -5
fi
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4) 2>&1 | tail -5
echo -e "${GREEN}Build OK${NC}"
echo ""

# ─── Section 2: Metal Shader Compilation ──────────────────────────
run_section "Shader Compilation (20 shaders)"
# Use validate_kernels.swift which compiles from source
swift tests/metal/validate_kernels.swift src/shaders/ 2>&1
echo ""

# ─── Section 3: GPU Kernel Execution ─────────────────────────────
run_section "GPU Kernel Execution & Numerical Validation"
BENCH_FLAG=""
if $BENCH; then BENCH_FLAG="--bench"; fi
swift tests/metal/gpu_run_all.swift $BENCH_FLAG 2>&1
echo ""

# ─── Section 4: C++ Unit Tests ────────────────────────────────────
run_section "C++ Unit Tests (ctest)"
cd build
if $QUICK; then
    ctest --output-on-failure --timeout 30 -R "test_(memory|device|stream|event|gemm|activation|softmax)" 2>&1
else
    ctest --output-on-failure --timeout 60 2>&1
fi
cd ..
echo ""

# ─── Section 5: PyTorch Tahoe Patch ──────────────────────────────
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
if [ "$MACOS_MAJOR" -ge 26 ]; then
    run_section "PyTorch Tahoe (macOS $MACOS_MAJOR.x) Patch"
    python3 scripts/patch_pytorch_tahoe.py --verify 2>&1
    echo ""
fi

# ─── Section 6: Benchmarks ───────────────────────────────────────
if $BENCH; then
    run_section "GPU Benchmarks"
    if [ -f build/tests/bench_gemm ]; then
        echo "GEMM benchmark:"
        build/tests/bench_gemm 2>&1
    fi
    echo ""
fi

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Test Suite Complete                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
