#!/usr/bin/env bash
# test_coral_kmod.sh — Integration test for coral-kmod privileged path
#
# Requires: root, RTX 5060 (or any Blackwell GPU), NVIDIA proprietary driver loaded
#
# Steps:
#   1. Build coral_kmod.ko
#   2. Load it
#   3. Verify /dev/coral-rm exists
#   4. Run the hotSpring benchmark suite (which uses coral-driver → coral-kmod path)
#   5. Unload
set -euo pipefail

KMOD_DIR="$(cd "$(dirname "$0")/../../../primals/coralReef/crates/coral-kmod" && pwd)"
HOTSPRING_DIR="$(cd "$(dirname "$0")/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 1. Build ────────────────────────────────────────────────────────
info "Building coral_kmod.ko..."
cd "$KMOD_DIR"
MAKE=/usr/bin/make /usr/bin/make -C /lib/modules/$(uname -r)/build M="$KMOD_DIR" modules
if [ ! -f "$KMOD_DIR/coral_kmod.ko" ]; then
    error "Build failed — coral_kmod.ko not found"
    exit 1
fi
info "Build OK"

# ── 2. Check preconditions ──────────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    error "This script must be run as root"
    exit 1
fi

if [ ! -c /dev/nvidiactl ]; then
    error "/dev/nvidiactl not found — NVIDIA driver not loaded?"
    exit 1
fi

# ── 3. Load module ──────────────────────────────────────────────────
if [ -d /sys/module/coral_kmod ]; then
    warn "coral_kmod already loaded — removing first"
    rmmod coral_kmod || true
    sleep 0.5
fi

info "Loading coral_kmod.ko..."
insmod "$KMOD_DIR/coral_kmod.ko"
sleep 0.5

if [ ! -d /sys/module/coral_kmod ]; then
    error "Module failed to load (check dmesg)"
    dmesg | tail -20
    exit 1
fi

if [ ! -c /dev/coral-rm ]; then
    error "/dev/coral-rm not created (check dmesg)"
    dmesg | tail -20
    rmmod coral_kmod
    exit 1
fi

info "/dev/coral-rm ready"
dmesg | grep "coral_kmod" | tail -5

# ── 4. Run hotSpring benchmark ──────────────────────────────────────
info "Running hotSpring compute benchmark..."
cd "$HOTSPRING_DIR"

# The benchmark will auto-detect /dev/coral-rm for Blackwell GPUs
if PATH="$HOME/.cargo/bin:$PATH" cargo test --features nvidia-drm -- --nocapture 2>&1; then
    info "Benchmark PASSED"
else
    warn "Benchmark had failures (check output above)"
fi

# ── 5. Unload ───────────────────────────────────────────────────────
info "Unloading coral_kmod..."
rmmod coral_kmod
if [ -d /sys/module/coral_kmod ]; then
    error "Module failed to unload"
    exit 1
fi
info "Module unloaded cleanly"
info "Test complete"
