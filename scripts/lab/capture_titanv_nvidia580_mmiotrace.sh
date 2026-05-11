#!/bin/bash
# capture_titanv_nvidia580_mmiotrace.sh
#
# Captures an mmiotrace of the nvidia-580 proprietary driver initializing
# the Titan V (GV100) at BDF 0000:02:00.0.
#
# PURPOSE: Determine the actual FECS/SEC2/ACR/PMU boot sequence used by
# the nvidia driver on Volta. This informs whether WPR is used and
# how the FalconBootSolver should branch for GV100.
#
# REQUIREMENTS:
#   - Must run as root
#   - nvidia-580 must be installed (kernel modules available)
#   - Titan V at 0000:02:00.0 (currently bound to vfio-pci)
#   - System may become unstable — save work first
#
# USAGE:
#   sudo bash capture_titanv_nvidia580_mmiotrace.sh
#
# OUTPUT:
#   wateringHole/mmiotraces/titanv_nvidia580_<timestamp>.mmiotrace

set -euo pipefail

TITANV_BDF="0000:02:00.0"
TITANV_AUDIO_BDF="0000:02:00.1"
OUTDIR="$(dirname "$0")/../../wateringHole/mmiotraces"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${OUTDIR}/titanv_nvidia580_${TIMESTAMP}.mmiotrace"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Titan V nvidia-580 mmiotrace capture                      ║"
echo "║  BDF: ${TITANV_BDF}                                           ║"
echo "║  Output: ${OUTFILE}                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Sanity checks
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Must run as root"
    exit 1
fi

if ! modinfo nvidia &>/dev/null; then
    echo "ERROR: nvidia kernel module not found"
    exit 1
fi

CURRENT_DRIVER=$(readlink /sys/bus/pci/devices/${TITANV_BDF}/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
echo "Current driver: ${CURRENT_DRIVER}"

# Step 1: Unbind from current driver
echo "[1/7] Unbinding ${TITANV_BDF} from ${CURRENT_DRIVER}..."
if [[ -e /sys/bus/pci/devices/${TITANV_BDF}/driver ]]; then
    echo "${TITANV_BDF}" > /sys/bus/pci/devices/${TITANV_BDF}/driver/unbind 2>/dev/null || true
fi
if [[ -e /sys/bus/pci/devices/${TITANV_AUDIO_BDF}/driver ]]; then
    echo "${TITANV_AUDIO_BDF}" > /sys/bus/pci/devices/${TITANV_AUDIO_BDF}/driver/unbind 2>/dev/null || true
fi
sleep 1

# Step 2: Remove driver_override
echo "[2/7] Clearing driver_override..."
echo "" > /sys/bus/pci/devices/${TITANV_BDF}/driver_override 2>/dev/null || true
echo "" > /sys/bus/pci/devices/${TITANV_AUDIO_BDF}/driver_override 2>/dev/null || true

# Step 3: Enable mmiotrace
echo "[3/7] Enabling mmiotrace..."
echo mmiotrace > /sys/kernel/tracing/current_tracer
echo "  mmiotrace active"

# Step 4: Load nvidia module (this triggers the init sequence we want to capture)
echo "[4/7] Loading nvidia module (this is the traced operation)..."
modprobe nvidia
sleep 3

# Step 5: Trigger device probe
echo "[5/7] Probing Titan V to trigger driver init..."
echo 1 > /sys/bus/pci/devices/${TITANV_BDF}/driver_probe 2>/dev/null || \
    echo "${TITANV_BDF}" > /sys/bus/pci/drivers/nvidia/bind 2>/dev/null || true
sleep 5

# Step 6: Capture trace
echo "[6/7] Capturing trace..."
mkdir -p "${OUTDIR}"
cat /sys/kernel/tracing/trace > "${OUTFILE}"
LINES=$(wc -l < "${OUTFILE}")
BYTES=$(stat -c%s "${OUTFILE}")
echo "  Captured: ${LINES} lines, ${BYTES} bytes"

# Step 7: Disable mmiotrace and cleanup
echo "[7/7] Disabling mmiotrace..."
echo nop > /sys/kernel/tracing/current_tracer

# Rebind to vfio-pci
echo "[cleanup] Rebinding ${TITANV_BDF} to vfio-pci..."
rmmod nvidia-drm nvidia-modeset nvidia-uvm nvidia 2>/dev/null || true
echo "vfio-pci" > /sys/bus/pci/devices/${TITANV_BDF}/driver_override
echo "vfio-pci" > /sys/bus/pci/devices/${TITANV_AUDIO_BDF}/driver_override
echo "${TITANV_BDF}" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
echo "${TITANV_AUDIO_BDF}" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true

echo
echo "═══ Capture complete ═══"
echo "  Trace: ${OUTFILE}"
echo "  Lines: ${LINES}"
echo
echo "Next steps:"
echo "  1. Search for SEC2/ACR registers (offset 0x840xxx)"
echo "  2. Search for FECS registers (offset 0x409xxx)"
echo "  3. Search for WPR setup (offset 0x100xxx)"
echo "  4. Determine if WPR is actually configured"
echo "  5. Update FalconBootSolver with Volta-specific path"
