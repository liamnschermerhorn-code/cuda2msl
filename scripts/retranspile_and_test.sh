#!/bin/bash
# Retranspile all CUDA sources with template instantiation, then run GPU tests
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Retranspiling all CUDA sources ==="
COUNT=0
for f in models/cuda_sources/*/*.cu; do
  [ -f "$f" ] || continue
  dir=$(basename $(dirname "$f"))
  name="${dir}__$(basename "$f" .cu)"
  build/src/transpiler/cuda2msl "$f" -o "models/cuda_sources/metal_output/${name}.metal" 2>/dev/null && COUNT=$((COUNT+1))
done
echo "Transpiled $COUNT files"
echo ""

echo "=== Running GPU tests on transpiled output ==="
swift tests/metal/run_transpiled.swift
