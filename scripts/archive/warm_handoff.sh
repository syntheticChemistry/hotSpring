#!/bin/bash
# warm_handoff.sh — Full nouveau→livepatch→vfio warm handoff for Titan V
# Run as root or with sudo
set -euo pipefail

BDF="${1:-0000:02:00.0}"
BDF_AUDIO="${BDF%.*}.1"
BRIDGE="00:01.3"
LIVEPATCH_DIR="$(cd "$(dirname "$0")/livepatch" && pwd)"

echo "═══════════════════════════════════════════════════════════"
echo "  Warm Handoff: ${BDF}"
echo "═══════════════════════════════════════════════════════════"

# 1. Ensure livepatch is loaded
if ! grep -q livepatch_nvkm_mc_reset /proc/modules 2>/dev/null; then
    echo "[1] Loading livepatch..."
    insmod "${LIVEPATCH_DIR}/livepatch_nvkm_mc_reset.ko"
    sleep 2
else
    echo "[1] Livepatch already loaded"
fi

# 2. SBR to ensure clean GPU state
echo "[2] SBR reset..."
echo 1 > /sys/bus/pci/devices/${BDF}/remove 2>/dev/null || true
sleep 1
BRIDGE_CTL=$(setpci -s ${BRIDGE} BRIDGE_CONTROL.w)
setpci -s ${BRIDGE} BRIDGE_CONTROL.w=0x0040
sleep 0.5
setpci -s ${BRIDGE} BRIDGE_CONTROL.w=${BRIDGE_CTL}
sleep 2
echo 1 > /sys/bus/pci/rescan
sleep 3

if ! lspci -s ${BDF} >/dev/null 2>&1; then
    echo "  FAIL: GPU not found after SBR"
    exit 1
fi
echo "  GPU back: $(lspci -s ${BDF} | head -1)"

# 3. Unbind whatever driver claimed it
DRIVER=$(basename "$(readlink /sys/bus/pci/devices/${BDF}/driver 2>/dev/null)" 2>/dev/null || echo "none")
if [ "$DRIVER" != "none" ]; then
    echo "[3] Unbinding ${DRIVER}..."
    echo ${BDF} > /sys/bus/pci/drivers/${DRIVER}/unbind 2>/dev/null || true
    sleep 1
fi

# 4. Bind nouveau (livepatch patches gr_fini, pmu_fini, mc_disable)
echo "[4] Binding nouveau (livepatch active)..."
echo "nouveau" > /sys/bus/pci/devices/${BDF}/driver_override
modprobe --ignore-install nouveau 2>/dev/null || true
sleep 1
echo ${BDF} > /sys/bus/pci/drivers_probe
sleep 8

DRIVER=$(basename "$(readlink /sys/bus/pci/devices/${BDF}/driver 2>/dev/null)" 2>/dev/null || echo "none")
if [ "$DRIVER" != "nouveau" ]; then
    echo "  FAIL: nouveau did not bind (driver=${DRIVER})"
    dmesg | grep "nouveau ${BDF}" | tail -5
    exit 1
fi
echo "  PASS: nouveau initialized"

# 5. Unbind nouveau (livepatch blocks PMC/GR/PMU teardown; FIFO tears down naturally)
echo "[5] Unbinding nouveau (livepatch preserving PMC/GR/PMU)..."
echo ${BDF} > /sys/bus/pci/drivers/nouveau/unbind
sleep 1

# 6. Bind vfio-pci
echo "[6] Binding vfio-pci..."
echo "" > /sys/bus/pci/devices/${BDF}/driver_override
echo "vfio-pci" > /sys/bus/pci/devices/${BDF}/driver_override
echo ${BDF} > /sys/bus/pci/drivers_probe
sleep 1

# Also bind audio function if present
if [ -d "/sys/bus/pci/devices/${BDF_AUDIO}" ]; then
    echo "vfio-pci" > /sys/bus/pci/devices/${BDF_AUDIO}/driver_override 2>/dev/null || true
    echo ${BDF_AUDIO} > /sys/bus/pci/drivers_probe 2>/dev/null || true
fi

DRIVER=$(basename "$(readlink /sys/bus/pci/devices/${BDF}/driver 2>/dev/null)" 2>/dev/null || echo "none")
if [ "$DRIVER" = "vfio-pci" ]; then
    echo "  PASS: vfio-pci bound"
else
    echo "  WARN: expected vfio-pci, got ${DRIVER}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Warm handoff complete. Run warm pipeline:"
echo "  sudo -E RUST_LOG=info cargo run --example volta_warm_pipeline \\"
echo "    --features vfio -- ${BDF}"
echo "═══════════════════════════════════════════════════════════"
