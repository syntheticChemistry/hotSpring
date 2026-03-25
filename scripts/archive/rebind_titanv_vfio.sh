#!/bin/bash
# DEPRECATED: Use `coralctl swap <BDF> vfio` instead. Raw sysfs writes risk D-state hangs.
set -e
GPU="0000:4b:00.0"
AUDIO="0000:4b:00.1"

echo "=== Rebinding Titan V ($GPU) to vfio-pci ==="

echo "Step 1: Unbind from current driver..."
CURR=$(readlink /sys/bus/pci/devices/$GPU/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
echo "  Current driver: $CURR"
if [ "$CURR" != "none" ]; then
    echo $GPU > /sys/bus/pci/drivers/$CURR/unbind 2>/dev/null || true
fi
CURRA=$(readlink /sys/bus/pci/devices/$AUDIO/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
if [ "$CURRA" != "none" ]; then
    echo $AUDIO > /sys/bus/pci/drivers/$CURRA/unbind 2>/dev/null || true
fi

echo "Step 2: Set driver_override to vfio-pci..."
echo "vfio-pci" > /sys/bus/pci/devices/$GPU/driver_override
echo "vfio-pci" > /sys/bus/pci/devices/$AUDIO/driver_override

echo "Step 3: Bind to vfio-pci..."
echo $GPU > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || {
    echo "  Direct bind failed, rescanning..."
    echo 1 > /sys/bus/pci/rescan
    sleep 2
}
echo $AUDIO > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true

echo "Step 4: Verify..."
DRIVER=$(readlink /sys/bus/pci/devices/$GPU/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
echo "  GPU driver: $DRIVER"
ls /dev/vfio/ 2>/dev/null

echo "=== Done ==="
