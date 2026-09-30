#!/bin/bash
# Run GPU kernel tests — execute outside of Claude's sandbox
# Usage: bash scripts/run_gpu_tests.sh [--bench]
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Metal GPU Kernel Test Suite ==="
echo "Running: swift tests/metal/gpu_run_all.swift $*"
echo ""

swift tests/metal/gpu_run_all.swift "$@"
