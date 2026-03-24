#!/bin/bash
set -e

GPU="0000:4b:00.0"
AUDIO="0000:4b:00.1"
TRACE_OUT="/home/biomegate/Development/ecoPrimals/hotSpring/data/nouveau_mmiotrace.log"
KVER=$(uname -r)

mkdir -p "$(dirname "$TRACE_OUT")"

echo "=== Nouveau MMIOTRACE Capture (v2 - targeted) ==="

echo "Step 1: Unbind Titan V from vfio-pci..."
echo $GPU > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || echo "  Already unbound"
echo $AUDIO > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || echo "  Audio already unbound"
echo "" > /sys/bus/pci/devices/$GPU/driver_override
echo "" > /sys/bus/pci/devices/$AUDIO/driver_override
sleep 1

echo "Step 2: Load nouveau dependencies (without mmiotrace)..."
insmod /lib/modules/$KVER/kernel/drivers/gpu/drm/scheduler/gpu-sched.ko 2>/dev/null || echo "  gpu-sched already loaded"
modprobe drm 2>/dev/null || echo "  drm already loaded"
modprobe drm_kms_helper 2>/dev/null || echo "  drm_kms_helper already loaded"
modprobe ttm 2>/dev/null || echo "  ttm already loaded"
echo "  Dependencies loaded"

echo "Step 3: Enable mmiotrace (ONLY traces future MMIO mappings)..."
echo mmiotrace > /sys/kernel/debug/tracing/current_tracer
echo "  mmiotrace enabled"

echo "Step 4: Load nouveau (this triggers Titan V init)..."
modprobe nouveau 2>&1 || {
    echo "  modprobe failed, trying insmod..."
    NOUVEAU_KO=$(find /lib/modules/$KVER -name "nouveau.ko" -o -name "nouveau.ko.zst" | head -1)
    insmod $NOUVEAU_KO
}

echo "Step 5: Wait for init..."
sleep 5

echo "Step 6: Stop mmiotrace IMMEDIATELY..."
echo nop > /sys/kernel/debug/tracing/current_tracer
echo "  mmiotrace stopped"

echo "Step 7: Capture trace..."
cat /sys/kernel/debug/tracing/trace > "$TRACE_OUT"
LINES=$(wc -l < "$TRACE_OUT")
SIZE=$(du -h "$TRACE_OUT" | cut -f1)
echo "  Captured $LINES lines ($SIZE) to $TRACE_OUT"

echo "Step 8: Verify Titan V driver..."
DRIVER=$(readlink /sys/bus/pci/devices/$GPU/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
echo "  GPU driver: $DRIVER"

echo "Step 9: Rebind to vfio-pci..."
if [ "$DRIVER" = "nouveau" ]; then
    echo $GPU > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
fi
echo $AUDIO > /sys/bus/pci/drivers/snd_hda_intel/unbind 2>/dev/null || true
echo "vfio-pci" > /sys/bus/pci/devices/$GPU/driver_override
echo "vfio-pci" > /sys/bus/pci/devices/$AUDIO/driver_override
echo $GPU > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || echo 1 > /sys/bus/pci/rescan
echo $AUDIO > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
sleep 1

FINAL=$(readlink /sys/bus/pci/devices/$GPU/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
echo "  Final GPU driver: $FINAL"

echo "=== Done ==="
