#!/bin/bash
# DEPRECATED: Use `coralctl swap <BDF> nouveau` instead. Raw sysfs writes risk D-state hangs.
# Unbind Titan V from vfio-pci and return to nouveau.
#
# Run with: sudo ./scripts/unbind-titanv-vfio.sh

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

GPU_BDF="0000:4b:00.0"
AUDIO_BDF="0000:4b:00.1"
GPU_ID="10de 1d81"
AUDIO_ID="10de 10f2"

echo -e "${CYAN}=== Restore Titan V to nouveau ===${NC}"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Must run as root (sudo)"
    exit 1
fi

# Unbind from vfio-pci
for BDF in "$GPU_BDF" "$AUDIO_BDF"; do
    DRV=$(readlink /sys/bus/pci/devices/$BDF/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
    if [ "$DRV" = "vfio-pci" ]; then
        echo "$BDF" > /sys/bus/pci/devices/$BDF/driver/unbind
        echo -e "  ${GREEN}✓${NC} Unbound $BDF from vfio-pci"
    fi
done

# Remove new_id entries
echo "$GPU_ID" > /sys/bus/pci/drivers/vfio-pci/remove_id 2>/dev/null || true
echo "$AUDIO_ID" > /sys/bus/pci/drivers/vfio-pci/remove_id 2>/dev/null || true

# Trigger driver re-probe
echo "$GPU_BDF" > /sys/bus/pci/drivers_probe 2>/dev/null || true
echo "$AUDIO_BDF" > /sys/bus/pci/drivers_probe 2>/dev/null || true

sleep 1

GPU_NOW=$(readlink /sys/bus/pci/devices/$GPU_BDF/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
AUDIO_NOW=$(readlink /sys/bus/pci/devices/$AUDIO_BDF/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")

echo ""
echo "  GPU:   $GPU_NOW"
echo "  Audio: $AUDIO_NOW"
echo -e "${GREEN}Titan V restored.${NC}"
