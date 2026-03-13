#!/bin/bash
echo "10de 1d81" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
echo "0000:4b:00.0" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
echo "0000:4b:00.1" > /sys/bus/pci/devices/0000:4b:00.1/driver/unbind 2>/dev/null || true
echo "10de 10f2" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
echo "0000:4b:00.1" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
chmod 0666 /dev/vfio/36 2>/dev/null || echo "NO_VFIO_36"
basename $(readlink /sys/bus/pci/devices/0000:4b:00.0/driver) 2>/dev/null || echo "GPU:NONE"
basename $(readlink /sys/bus/pci/devices/0000:4b:00.1/driver) 2>/dev/null || echo "AUDIO:NONE"
ls -la /dev/vfio/
