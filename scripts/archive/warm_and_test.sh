#!/bin/bash
# DEPRECATED: This script uses raw sysfs writes (driver_override, bind, unbind).
# Use `coralctl swap` for driver transitions. Archived 2026-03-25 (Gap 13 safety).
# warm_and_test.sh — Warm GPU via nouveau, rebind to vfio-pci, run diagnostic matrix
#
# Usage: sudo ./scripts/warm_and_test.sh [GPU_BDF] [WAIT_SECS]
# Default GPU_BDF: 0000:4b:00.0
# Default WAIT_SECS: 5 (how long nouveau has to initialize)
#
# This script handles the full cycle:
#   1. Unbind from current driver
#   2. Bind to nouveau (warm the GPU)
#   3. Verify warm state via BAR0 PMC_ENABLE
#   4. Unbind nouveau
#   5. Bind to vfio-pci
#   6. Run the diagnostic matrix as the original user

set -euo pipefail

GPU="${1:-0000:4b:00.0}"
WAIT="${2:-5}"
AUD="${GPU%.*}.1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORALREEF_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/coralReef"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║ hotSpring VFIO Diagnostic Matrix — Warm & Test          ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ GPU:       $GPU"
echo "║ Audio:     $AUD"
echo "║ Warm wait: ${WAIT}s"
echo "║ coralReef: $CORALREEF_DIR"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Unbind ──
echo ">>> Step 1: Unbinding from current driver..."
echo "" > "/sys/bus/pci/devices/$GPU/reset_method" 2>/dev/null || true
echo "$GPU" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
echo "$AUD" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
echo "$GPU" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
echo "" > "/sys/bus/pci/devices/$GPU/driver_override"
sleep 1

# ── Step 2: Bind nouveau ──
echo ">>> Step 2: Binding nouveau (waiting ${WAIT}s)..."
echo "$GPU" > /sys/bus/pci/drivers/nouveau/bind 2>&1
sleep "$WAIT"

# ── Step 3: Verify warm ──
echo ">>> Step 3: Verifying warm state..."
WARM_CHECK=$(python3 -c "
import mmap, os, struct
try:
    fd = os.open('/sys/bus/pci/devices/${GPU}/resource0', os.O_RDONLY)
    m = mmap.mmap(fd, 0x01000000, mmap.MAP_SHARED, mmap.PROT_READ)
    boot0 = struct.unpack_from('<I', m, 0x000)[0]
    pmc = struct.unpack_from('<I', m, 0x200)[0]
    pfifo = struct.unpack_from('<I', m, 0x2200)[0]
    pbdma = struct.unpack_from('<I', m, 0x2004)[0]
    m.close(); os.close(fd)
    warm = pmc != 0x40000020 and pfifo != 0xbad0da00
    print(f'BOOT0={boot0:#010x} PMC={pmc:#010x} PFIFO={pfifo:#010x} PBDMA_MAP={pbdma:#010x} WARM={warm}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
echo "    $WARM_CHECK"

if echo "$WARM_CHECK" | grep -q "WARM=False"; then
    echo "    ⚠ GPU IS COLD — nouveau init may have failed"
    echo "    Retrying with longer wait (10s)..."
    echo "$GPU" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
    sleep 2
    echo "" > "/sys/bus/pci/devices/$GPU/driver_override"
    echo "$GPU" > /sys/bus/pci/drivers/nouveau/bind 2>&1
    sleep 10

    WARM_CHECK2=$(python3 -c "
import mmap, os, struct
fd = os.open('/sys/bus/pci/devices/${GPU}/resource0', os.O_RDONLY)
m = mmap.mmap(fd, 0x01000000, mmap.MAP_SHARED, mmap.PROT_READ)
pmc = struct.unpack_from('<I', m, 0x200)[0]
m.close(); os.close(fd)
warm = pmc != 0x40000020
print(f'PMC={pmc:#010x} WARM={warm}')
" 2>&1)
    echo "    Retry: $WARM_CHECK2"

    if echo "$WARM_CHECK2" | grep -q "WARM=False"; then
        echo "    ✗ GPU still cold after retry. Aborting."
        exit 1
    fi
fi
echo "    ✓ GPU is warm"

# ── Step 4: Unbind nouveau ──
echo ">>> Step 4: Unbinding nouveau..."
echo "$GPU" > /sys/bus/pci/drivers/nouveau/unbind 2>&1
sleep 2

# ── Step 5: Bind vfio-pci ──
echo ">>> Step 5: Binding vfio-pci..."
echo "vfio-pci" > "/sys/bus/pci/devices/$GPU/driver_override"
echo "vfio-pci" > "/sys/bus/pci/devices/$AUD/driver_override"
echo "$GPU" > /sys/bus/pci/drivers/vfio-pci/bind 2>&1
echo "$AUD" > /sys/bus/pci/drivers/vfio-pci/bind 2>&1
sleep 1
chmod 666 /dev/vfio/36 2>/dev/null || true

DRIVER=$(readlink "/sys/bus/pci/devices/$GPU/driver" 2>/dev/null || echo "none")
echo "    Driver: $DRIVER"

# ── Step 6: Run diagnostic matrix as the real user ──
echo ""
echo ">>> Step 6: Running diagnostic matrix..."
echo ""

# Find the real user (not root)
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo biomegate)}"
REAL_HOME=$(eval echo "~$REAL_USER")

cd "$CORALREEF_DIR"
RUSTUP_HOME="$REAL_HOME/.rustup" \
CARGO_HOME="$REAL_HOME/.cargo" \
CORALREEF_VFIO_BDF="$GPU" \
CORALREEF_VFIO_SM=70 \
su "$REAL_USER" -c "
    export PATH=\"$REAL_HOME/.cargo/bin:\$PATH\"
    export RUSTUP_HOME=\"$REAL_HOME/.rustup\"
    export CARGO_HOME=\"$REAL_HOME/.cargo\"
    export CORALREEF_VFIO_BDF=$GPU
    export CORALREEF_VFIO_SM=70
    cd '$CORALREEF_DIR'
    $REAL_HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo test \
        --test hw_nv_vfio --features vfio \
        -- --ignored --test-threads=1 --nocapture vfio_pfifo_diagnostic_matrix
"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║ Diagnostic matrix complete.                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
