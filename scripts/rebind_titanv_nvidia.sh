#!/bin/bash
set -e
GPU="0000:4b:00.0"
AUDIO="0000:4b:00.1"

echo "=== Rebinding Titan V ($GPU) to nvidia driver ==="

echo "Step 1: Clear driver_override..."
echo "" > /sys/bus/pci/devices/$GPU/driver_override
echo "" > /sys/bus/pci/devices/$AUDIO/driver_override

echo "Step 2: Unbind from vfio-pci..."
echo $GPU > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || echo "  GPU already unbound"
echo $AUDIO > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || echo "  Audio already unbound"

echo "Step 3: Bind GPU to nvidia..."
echo $GPU > /sys/bus/pci/drivers/nvidia/bind 2>/dev/null || {
    echo "  Direct bind failed, trying PCI rescan..."
    echo 1 > /sys/bus/pci/rescan
    sleep 2
}

echo "Step 4: Verify..."
DRIVER=$(readlink /sys/bus/pci/devices/$GPU/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
echo "  GPU driver: $DRIVER"
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"

echo "=== Done ==="
