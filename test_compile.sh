#!/bin/bash
# Test Metal compilation of all transpiled outputs
# Usage: ./test_compile.sh [directory]
# Default: tests all frameworks in metal_output/

METAL=""
# Try xcrun first
METAL=$(xcrun -f metal 2>/dev/null)
# Try cryptex mount
if [ -z "$METAL" ]; then
    METAL=$(find /private/var/run/com.apple.security.cryptexd -name "metal" -path "*/usr/bin/metal" -not -path "*/32023/*" 2>/dev/null | head -1)
fi
# Try Xcode toolchain
if [ -z "$METAL" ]; then
    METAL=$(find /Applications/Xcode.app -name "metal" -path "*/usr/bin/metal" 2>/dev/null | head -1)
fi
# Try System Library
if [ -z "$METAL" ]; then
    METAL=$(find /System/Library -name "metal" -path "*/usr/bin/metal" 2>/dev/null | head -1)
fi
if [ -z "$METAL" ]; then
    echo "Metal compiler not found."
    echo "Run: xcodebuild -downloadComponent MetalToolchain"
    echo "Then try again immediately (the mount is temporary)."
    exit 1
fi
echo "Using: $METAL"

BASE="models/cuda_sources/metal_output"
DIR="${1:-}"

if [ -n "$DIR" ]; then
    DIRS="$BASE/$DIR"
else
    DIRS="$BASE"/*
fi

TOTAL_PASS=0
TOTAL_FAIL=0

for dir in $DIRS; do
    [ -d "$dir" ] || continue
    fw=$(basename "$dir")
    PASS=0; FAIL=0; ERRORS=""
    for f in "$dir"/*.metal; do
        [ -f "$f" ] || continue
        name=$(basename "$f")
        if $METAL -c -std=metal3.2 -Wno-unused-variable -Wno-unused-function "$f" -o /dev/null 2>/dev/null; then
            PASS=$((PASS+1))
        else
            FAIL=$((FAIL+1))
            ERRORS="$ERRORS $name"
        fi
    done
    TOTAL=$((PASS+FAIL))
    [ $TOTAL -eq 0 ] && continue
    if [ $FAIL -eq 0 ]; then
        printf "  PASS  %-20s %3d/%3d\n" "$fw" "$PASS" "$TOTAL"
    else
        printf "  FAIL  %-20s %3d/%3d  (%d failed)\n" "$fw" "$PASS" "$TOTAL" "$FAIL"
    fi
    TOTAL_PASS=$((TOTAL_PASS+PASS))
    TOTAL_FAIL=$((TOTAL_FAIL+FAIL))
done

echo ""
echo "Total: $TOTAL_PASS pass, $TOTAL_FAIL fail, $((TOTAL_PASS+TOTAL_FAIL)) files"

# Show first error from first failing file for debugging
if [ $TOTAL_FAIL -gt 0 ]; then
    echo ""
    echo "=== First error ==="
    for dir in $DIRS; do
        [ -d "$dir" ] || continue
        for f in "$dir"/*.metal; do
            [ -f "$f" ] || continue
            ERR=$($METAL -c -std=metal3.2 -Wno-unused-variable -Wno-unused-function "$f" -o /dev/null 2>&1)
            if [ $? -ne 0 ]; then
                echo "File: $(basename "$f")"
                echo "$ERR" | head -10
                exit 1
            fi
        done
    done
fi
