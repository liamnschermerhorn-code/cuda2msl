#!/usr/bin/env bash
# run_tests.sh <directory-with-metal-files>
#
# Stage 1: Metal compiler syntax check (xcrun metal -c)
# Stage 2: GPU runtime test — pipeline states + numerical validation
#
# Usage:
#   ./run_tests.sh /path/to/your/metal/output

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-}"

if [[ -z "$TARGET" ]]; then
  echo "Usage: $0 <directory containing .metal files>"
  exit 1
fi

echo ""
echo "▶ Stage 1: Compile check"
bash "$SCRIPT_DIR/compile_check.sh" "$TARGET"

echo ""
echo "▶ Stage 2: GPU runtime + numerical tests"
swift "$SCRIPT_DIR/gpu_test.swift" "$TARGET"
