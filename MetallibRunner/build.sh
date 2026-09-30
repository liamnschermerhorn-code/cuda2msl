#!/bin/bash
# Build MetallibRunner.app — a native macOS app for GPU-testing .metallib files
# Usage: ./build.sh [--release]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="MetallibRunner"
BUNDLE="${APP_NAME}.app"
CONTENTS="${BUNDLE}/Contents"
MACOS="${CONTENTS}/MacOS"
RESOURCES="${CONTENTS}/Resources"

OPT="-Onone"
if [[ "${1:-}" == "--release" ]]; then
    OPT="-O -whole-module-optimization"
    echo "Building release..."
else
    echo "Building debug..."
fi

# Clean previous build
rm -rf "$BUNDLE"

# Create .app bundle structure
mkdir -p "$MACOS" "$RESOURCES"

# Copy Info.plist
cp Info.plist "$CONTENTS/Info.plist"

# Compile
echo "Compiling..."
swiftc \
    $OPT \
    -sdk "$(xcrun --show-sdk-path)" \
    -target "$(uname -m)-apple-macosx14.0" \
    -framework Metal \
    -framework SwiftUI \
    -framework UniformTypeIdentifiers \
    -parse-as-library \
    -o "${MACOS}/${APP_NAME}" \
    Sources/MetallibRunnerApp.swift

# Sign ad-hoc for local distribution
echo "Signing..."
codesign --force --sign - --entitlements /dev/stdin "$BUNDLE" <<'ENTITLEMENTS'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.device.gpu</key>
    <true/>
</dict>
</plist>
ENTITLEMENTS

echo ""
echo "Built: ${SCRIPT_DIR}/${BUNDLE}"
echo "To open: open ${BUNDLE}"
echo "To set as default for .metallib: right-click a .metallib > Get Info > Open With > Change All"
echo ""
echo "To distribute: zip -r ${APP_NAME}.zip ${BUNDLE}"
