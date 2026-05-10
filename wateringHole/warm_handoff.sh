#!/bin/bash
# DEPRECATED — Legacy ad-hoc lab script for direct insmod/unbind/rebind.
# Violates current architecture: all GPU driver transitions must route through
# coralctl / coral-ember / coral-glowplug.  Kept as fossil record.
#
# Original purpose: Run after reboot to do full nouveau init then hand off
# Usage: sudo bash warm_handoff.sh
set -euo pipefail

LIVEPATCH_KO="/home/biomegate/Development/ecoPrimals/springs/hotSpring/scripts/livepatch/livepatch_nvkm_mc_reset.ko"
GPU="0000:02:00.0"

echo "[1] Loading livepatch..."
insmod "$LIVEPATCH_KO" || true
sleep 1
lsmod | grep livepatch_nvkm_mc_reset || { echo "WARN: livepatch not loaded!"; }

echo "[2] Unbind from vfio-pci..."
echo "$GPU" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
echo "0000:02:00.1" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
sleep 0.5

echo "[3] Set driver_override → nouveau..."
echo nouveau > /sys/bus/pci/devices/$GPU/driver_override
echo nouveau > /sys/bus/pci/devices/0000:02:00.1/driver_override

echo "[4] Bind to nouveau..."
echo "$GPU" > /sys/bus/pci/drivers/nouveau/bind &
BIND_PID=$!

echo "[5] Monitoring dmesg for init completion..."
sleep 5
dmesg | tail -30

echo "[6] Checking driver..."
ls -la /sys/bus/pci/devices/$GPU/driver 2>&1 || true

wait $BIND_PID || true
dmesg | grep -E "livepatch|nouveau.*$GPU" | tail -20
