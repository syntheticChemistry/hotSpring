#!/bin/bash
set -e

GPU="0000:4b:00.0"
AUDIO="0000:4b:00.1"
TRACE_OUT="/home/biomegate/Development/ecoPrimals/hotSpring/data/nouveau_mmiotrace.log"

mkdir -p "$(dirname "$TRACE_OUT")"

echo "=== Nouveau MMIOTRACE Capture ==="
echo "This captures EVERY BAR0 register write nouveau does to init the Titan V."

echo "Step 1: Unbind Titan V from vfio-pci..."
echo $GPU > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || echo "  GPU already unbound"
echo $AUDIO > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || echo "  Audio already unbound"
echo "" > /sys/bus/pci/devices/$GPU/driver_override
echo "" > /sys/bus/pci/devices/$AUDIO/driver_override

echo "Step 2: Unload nouveau if loaded..."
modprobe -r nouveau 2>/dev/null || echo "  nouveau not loaded or can't unload"

echo "Step 3: Enable mmiotrace..."
echo mmiotrace > /sys/kernel/debug/tracing/current_tracer
echo "  mmiotrace enabled"

echo "Step 4: Load nouveau and let it initialize Titan V..."
modprobe nouveau 2>/dev/null || {
    # Load dependencies first
    modprobe gpu-sched 2>/dev/null || true
    modprobe drm 2>/dev/null || true
    modprobe drm_kms_helper 2>/dev/null || true
    modprobe nouveau
}

echo "Step 5: Wait for nouveau init to complete..."
sleep 5

echo "Step 6: Verify Titan V is on nouveau..."
DRIVER=$(readlink /sys/bus/pci/devices/$GPU/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
echo "  GPU driver: $DRIVER"

echo "Step 7: Capture trace..."
cat /sys/kernel/debug/tracing/trace > "$TRACE_OUT"
LINES=$(wc -l < "$TRACE_OUT")
echo "  Captured $LINES lines to $TRACE_OUT"

echo "Step 8: Stop mmiotrace..."
echo nop > /sys/kernel/debug/tracing/current_tracer
echo "  mmiotrace stopped"

echo "Step 9: Unbind nouveau and rebind vfio-pci..."
echo $GPU > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || echo "  unbind failed"
echo $AUDIO > /sys/bus/pci/drivers/snd_hda_intel/unbind 2>/dev/null || true
echo "vfio-pci" > /sys/bus/pci/devices/$GPU/driver_override
echo "vfio-pci" > /sys/bus/pci/devices/$AUDIO/driver_override
echo $GPU > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || echo 1 > /sys/bus/pci/rescan
echo $AUDIO > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true

FINAL=$(readlink /sys/bus/pci/devices/$GPU/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
echo "  Final GPU driver: $FINAL"

echo "=== Done ==="
echo "Trace saved to: $TRACE_OUT"
