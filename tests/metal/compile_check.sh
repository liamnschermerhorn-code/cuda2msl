#!/usr/bin/env bash
# compile_check.sh <dir>
# Stage 1: Run every .metal file through the Metal compiler.
# Catches syntax errors, unknown intrinsics, bad types — before touching the GPU.
#
# Usage:
#   ./compile_check.sh /path/to/metal/files
#   ./compile_check.sh .                    # current directory

set -euo pipefail
DIR="${1:-.}"
PASS=0; FAIL=0; ERRORS=()

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Metal compile check  →  $DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

shopt -s nullglob
FILES=("$DIR"/*.metal "$DIR"/**/*.metal)
if [ ${#FILES[@]} -eq 0 ]; then
  echo "No .metal files found in $DIR"
  exit 1
fi

for f in "${FILES[@]}"; do
  name=$(basename "$f")
  air="$TMP/${name%.metal}.air"

  # -w suppresses warnings so we see only errors; remove -w if you want warnings too
  if xcrun -sdk macosx metal -std=metal3.0 -c "$f" -o "$air" 2>"$TMP/err.txt"; then
    printf "  ✓  %s\n" "$name"
    PASS=$((PASS+1))
  else
    printf "  ✗  %s\n" "$name"
    # indent the compiler errors
    sed 's/^/     /' "$TMP/err.txt"
    ERRORS+=("$name")
    FAIL=$((FAIL+1))
  fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " $PASS passed  ·  $FAIL failed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $FAIL -gt 0 ]; then
  echo ""
  echo "Failed files:"
  for e in "${ERRORS[@]}"; do echo "  • $e"; done
  exit 1
fi
