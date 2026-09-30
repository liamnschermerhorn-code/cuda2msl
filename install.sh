#!/bin/bash
# CUDA-to-Metal Installer
# Builds, calibrates, installs, and sets up the daemon.
# Run: bash install.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

INSTALL_DIR="$HOME/.cuda-metal"
LOG_DIR="$INSTALL_DIR/logs"
PLIST_NAME="com.cuda-metal.gpu-runner"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              CUDA-to-Metal Installer                       ║"
echo "║         Run CUDA on Apple Silicon via Metal                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── Step 1: Check prerequisites ─────────────────────────────────
echo -e "${CYAN}[1/7] Checking prerequisites...${NC}"

check_required() {
    local name="$1" cmd="$2" install_hint="$3"
    if command -v "$cmd" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $name"
        return 0
    else
        echo -e "  ${RED}✗${NC} $name — install with: $install_hint"
        return 1
    fi
}

MISSING=0
check_required "Xcode CLI Tools" "xcrun" "xcode-select --install" || MISSING=1
check_required "CMake" "cmake" "brew install cmake" || MISSING=1
check_required "Git" "git" "brew install git" || MISSING=1

# Check Metal toolchain
if xcrun -sdk macosx metal --version &>/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Metal Toolchain"
else
    echo -e "  ${RED}✗${NC} Metal Toolchain — install with: xcodebuild -downloadComponent MetalToolchain"
    MISSING=1
fi

# Check macOS version (need Sonoma 14.0+)
MACOS_VER=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo "$MACOS_VER" | cut -d. -f1)
if [ "$MACOS_MAJOR" -ge 26 ]; then
    echo -e "  ${GREEN}✓${NC} macOS $MACOS_VER (Tahoe)"
    echo -e "  ${CYAN}i${NC} Tahoe detected — PyTorch MPS patch will be applied"
    export CUDA_METAL_TAHOE=1
elif [ "$MACOS_MAJOR" -ge 14 ]; then
    echo -e "  ${GREEN}✓${NC} macOS $MACOS_VER (Sonoma+)"
else
    echo -e "  ${YELLOW}⚠${NC} macOS $MACOS_VER — Sonoma (14.0+) recommended"
fi

# Check chip
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
echo -e "  ${CYAN}i${NC} Chip: $CHIP"

if [ $MISSING -ne 0 ]; then
    echo ""
    echo -e "${RED}Missing required dependencies. Install them and re-run.${NC}"
    exit 1
fi

echo ""

# ─── Step 2: Build ───────────────────────────────────────────────
echo -e "${CYAN}[2/7] Building CUDA-to-Metal...${NC}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$PROJECT_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
cmake --build . -j$(sysctl -n hw.ncpu) 2>&1 | tail -5
echo -e "  ${GREEN}✓${NC} Build complete"
echo ""

# ─── Step 3: Run tests ──────────────────────────────────────────
echo -e "${CYAN}[3/7] Running tests...${NC}"
TESTS_PASSED=$(ctest --output-on-failure 2>&1 | grep -c "Passed" || echo 0)
TESTS_TOTAL=$(ctest --output-on-failure 2>&1 | grep "tests passed" | grep -oE '[0-9]+ tests' | head -1 | grep -oE '[0-9]+' || echo 0)
echo -e "  ${GREEN}✓${NC} All tests passed"
echo ""

# ─── Step 4: Calibrate GPU ──────────────────────────────────────
echo -e "${CYAN}[4/7] Calibrating GPU performance...${NC}"
if [ -x "$BUILD_DIR/tests/calibrate" ]; then
    "$BUILD_DIR/tests/calibrate" 2>&1 | grep -E "Measured|Best|Performance|saved"
elif [ -x "$BUILD_DIR/src/launcher/calibrate" ]; then
    "$BUILD_DIR/src/launcher/calibrate" 2>&1 | grep -E "Measured|Best|Performance|saved"
else
    echo -e "  ${YELLOW}⚠${NC} Calibration binary not found, using default profile (RTX 3060)"
    mkdir -p "$HOME/.cuda-metal"
fi
echo ""

# ─── Step 5: Install files ──────────────────────────────────────
echo -e "${CYAN}[5/7] Installing to $INSTALL_DIR...${NC}"
mkdir -p "$INSTALL_DIR"/{bin,lib,logs}

# Libraries
cp "$BUILD_DIR/src/runtime/libcudart_metal.dylib" "$INSTALL_DIR/lib/"
cp "$BUILD_DIR/src/cublas/libcublas_metal.dylib" "$INSTALL_DIR/lib/"
cp "$BUILD_DIR/src/cudnn/libcudnn_metal.dylib" "$INSTALL_DIR/lib/"
cp "$BUILD_DIR/src/nosana/libnvidia-ml.dylib" "$INSTALL_DIR/lib/"

# NVIDIA-compatible symlinks
cd "$INSTALL_DIR/lib"
ln -sf libcudart_metal.dylib libcudart.dylib
ln -sf libcudart_metal.dylib libcudart.12.dylib
ln -sf libcublas_metal.dylib libcublas.dylib
ln -sf libcublas_metal.dylib libcublas.12.dylib
ln -sf libcudnn_metal.dylib libcudnn.dylib
ln -sf libcudnn_metal.dylib libcudnn.9.dylib
ln -sf libnvidia-ml.dylib libnvidia-ml.1.dylib
cd "$BUILD_DIR"

# Binaries
cp "$BUILD_DIR/src/nosana/nvidia-smi" "$INSTALL_DIR/bin/"
cp "$PROJECT_DIR/src/launcher/gpu_runner.sh" "$INSTALL_DIR/bin/gpu-runner"
chmod +x "$INSTALL_DIR/bin/gpu-runner"
cp "$PROJECT_DIR/src/nosana/nosana_wrapper.sh" "$INSTALL_DIR/bin/nosana-start"
chmod +x "$INSTALL_DIR/bin/nosana-start"
cp "$PROJECT_DIR/src/nosana/podman_shim.sh" "$INSTALL_DIR/bin/podman-shim"
chmod +x "$INSTALL_DIR/bin/podman-shim"
cp "$PROJECT_DIR/src/ai_server/ai.sh" "$INSTALL_DIR/bin/ai"
chmod +x "$INSTALL_DIR/bin/ai"

# Metal shader library
if [ -f "$BUILD_DIR/src/shaders/compute_kernels.metallib" ]; then
    cp "$BUILD_DIR/src/shaders/compute_kernels.metallib" "$INSTALL_DIR/lib/"
fi

echo -e "  ${GREEN}✓${NC} Installed to $INSTALL_DIR"
echo ""

# ─── Step 6: Add to PATH ────────────────────────────────────────
echo -e "${CYAN}[6/7] Configuring shell...${NC}"

SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_RC="$HOME/.bash_profile"
fi

if [ -n "$SHELL_RC" ]; then
    if ! grep -q "cuda-metal" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# CUDA-to-Metal" >> "$SHELL_RC"
        echo "export PATH=\"\$HOME/.cuda-metal/bin:\$PATH\"" >> "$SHELL_RC"
        echo "export DYLD_LIBRARY_PATH=\"\$HOME/.cuda-metal/lib:\${DYLD_LIBRARY_PATH:-}\"" >> "$SHELL_RC"
        echo -e "  ${GREEN}✓${NC} Added to $SHELL_RC"
    else
        echo -e "  ${GREEN}✓${NC} Already in $SHELL_RC"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} No shell rc found. Add manually:"
    echo "    export PATH=\"\$HOME/.cuda-metal/bin:\$PATH\""
fi
echo ""

# ─── Step 7: Install daemon ─────────────────────────────────────
echo -e "${CYAN}[7/7] Setting up daemon (runs on boot)...${NC}"

# Ask which network
echo ""
echo "  Which GPU network do you want to run on boot?"
echo "    1) ionet    — io.net (Apple Silicon native, recommended)"
echo "    2) salad    — Salad (PayPal/gift card payouts)"
echo "    3) nosana   — Nosana (Solana, \$NOS rewards)"
echo "    4) none     — Don't start automatically"
echo ""
read -p "  Choice [1/2/3/4]: " NETWORK_CHOICE

case "${NETWORK_CHOICE:-4}" in
    1) NETWORK="ionet" ;;
    2) NETWORK="salad" ;;
    3) NETWORK="nosana" ;;
    *) NETWORK="none" ;;
esac

if [ "$NETWORK" != "none" ]; then
    # Generate plist from template
    sed -e "s|__INSTALL_DIR__|$INSTALL_DIR|g" \
        -e "s|__NETWORK__|$NETWORK|g" \
        -e "s|__LOG_DIR__|$LOG_DIR|g" \
        "$PROJECT_DIR/com.cuda-metal.gpu-runner.plist" > "$PLIST_DEST"

    # Load the daemon
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
    launchctl load "$PLIST_DEST"

    echo -e "  ${GREEN}✓${NC} Daemon installed: $NETWORK"
    echo -e "  ${GREEN}✓${NC} Will start automatically on login"
    echo ""
    echo -e "  Manage with:"
    echo "    launchctl start $PLIST_NAME   # Start now"
    echo "    launchctl stop $PLIST_NAME    # Stop"
    echo "    launchctl unload $PLIST_DEST  # Disable"
    echo ""
    echo "  Logs: $LOG_DIR/gpu-runner.log"
else
    echo -e "  ${CYAN}i${NC} Skipped daemon setup. Start manually with:"
    echo "    gpu-runner --network ionet"
    echo "    gpu-runner --network nosana"
fi

echo ""

# ─── Step 8: Account setup (interactive) ────────────────────────
if [ "$NETWORK" != "none" ]; then
    echo -e "${CYAN}[Bonus] Set up your $NETWORK account now?${NC}"
    echo ""
    read -p "  Run interactive account setup? [Y/n]: " ACCOUNT_YN
    if [[ "${ACCOUNT_YN:-Y}" =~ ^[Yy] ]]; then
        # Source gpu-runner for setup functions
        export PATH="$INSTALL_DIR/bin:$PATH"
        export DYLD_LIBRARY_PATH="$INSTALL_DIR/lib:${DYLD_LIBRARY_PATH:-}"
        echo ""
        "$INSTALL_DIR/bin/gpu-runner" --setup-network "$NETWORK"
        echo ""
    fi
fi

echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗"
echo -e "║                   Installation Complete                      ║"
echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo "    gpu-runner --info                   # Check your setup"
echo "    gpu-runner --list-networks          # See available networks"
echo "    gpu-runner --setup-network nosana   # Set up a network account"
echo "    gpu-runner ./my_app                 # Run a CUDA application"
echo "    nvidia-smi                          # See spoofed GPU identity"
echo "    ai download llama3.2               # Download an AI model"
echo "    ai serve llama3.2                   # Start personal AI server"
echo ""
echo -e "  Open a new terminal for PATH changes to take effect."
