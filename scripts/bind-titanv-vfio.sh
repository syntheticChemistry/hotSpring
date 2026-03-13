#!/bin/bash
# Bind Titan V (0000:4b:00.0) to vfio-pci for sovereign compute validation.
#
# Run with: sudo ./scripts/bind-titanv-vfio.sh
#
# This binds ONLY the Titan V (secondary GPU, not driving display).
# The RTX 3090 (boot_vga=1) stays on nouveau for display.
#
# To restore: sudo ./scripts/unbind-titanv-vfio.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

GPU_BDF="0000:4b:00.0"
AUDIO_BDF="0000:4b:00.1"
GPU_ID="10de 1d81"
AUDIO_ID="10de 10f2"
IOMMU_GROUP=36

echo -e "${CYAN}=== Bind Titan V to VFIO for sovereign compute ===${NC}"
echo ""

if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}ERROR: Must run as root (sudo)${NC}"
    exit 1
fi

# Verify this is not the boot VGA
BOOT_VGA=$(cat /sys/bus/pci/devices/$GPU_BDF/boot_vga 2>/dev/null)
if [ "$BOOT_VGA" = "1" ]; then
    echo -e "${RED}ABORT: $GPU_BDF is the boot VGA — do NOT unbind it${NC}"
    exit 1
fi

echo "Target: Titan V ($GPU_BDF) + audio ($AUDIO_BDF)"
echo "IOMMU group: $IOMMU_GROUP"
echo ""

# 1. Load VFIO modules
echo "1. Loading VFIO modules..."
modprobe vfio
modprobe vfio_pci
modprobe vfio_iommu_type1
echo -e "   ${GREEN}✓${NC} vfio, vfio_pci, vfio_iommu_type1 loaded"

# 2. Unbind GPU from nouveau
echo "2. Unbinding Titan V from nouveau..."
CURRENT=$(readlink /sys/bus/pci/devices/$GPU_BDF/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
if [ "$CURRENT" = "vfio-pci" ]; then
    echo -e "   ${GREEN}✓${NC} Already on vfio-pci"
elif [ "$CURRENT" != "none" ]; then
    echo "$GPU_BDF" > /sys/bus/pci/devices/$GPU_BDF/driver/unbind
    echo -e "   ${GREEN}✓${NC} Unbound from $CURRENT"
else
    echo -e "   ${GREEN}✓${NC} No driver bound"
fi

# 3. Unbind audio from snd_hda_intel
echo "3. Unbinding audio from snd_hda_intel..."
AUDIO_DRV=$(readlink /sys/bus/pci/devices/$AUDIO_BDF/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
if [ "$AUDIO_DRV" = "vfio-pci" ]; then
    echo -e "   ${GREEN}✓${NC} Already on vfio-pci"
elif [ "$AUDIO_DRV" != "none" ]; then
    echo "$AUDIO_BDF" > /sys/bus/pci/devices/$AUDIO_BDF/driver/unbind
    echo -e "   ${GREEN}✓${NC} Unbound from $AUDIO_DRV"
else
    echo -e "   ${GREEN}✓${NC} No driver bound"
fi

# 4. Bind GPU to vfio-pci
echo "4. Binding GPU to vfio-pci..."
echo "$GPU_ID" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
echo "$GPU_BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
echo -e "   ${GREEN}✓${NC} $GPU_BDF → vfio-pci"

# 5. Bind audio to vfio-pci
echo "5. Binding audio to vfio-pci..."
echo "$AUDIO_ID" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true
echo "$AUDIO_BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
echo -e "   ${GREEN}✓${NC} $AUDIO_BDF → vfio-pci"

# 6. Set permissions
echo "6. Setting VFIO group permissions..."
VFIO_DEV="/dev/vfio/$IOMMU_GROUP"
if [ -e "$VFIO_DEV" ]; then
    chmod 0666 "$VFIO_DEV"
    echo -e "   ${GREEN}✓${NC} $VFIO_DEV → mode 0666"
else
    echo -e "${RED}   ✗ $VFIO_DEV not created — check IOMMU${NC}"
    exit 1
fi

# 7. Verify
echo ""
echo -e "${CYAN}=== Verification ===${NC}"
GPU_NOW=$(readlink /sys/bus/pci/devices/$GPU_BDF/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
AUDIO_NOW=$(readlink /sys/bus/pci/devices/$AUDIO_BDF/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
echo "  GPU driver:   $GPU_NOW"
echo "  Audio driver: $AUDIO_NOW"
echo "  VFIO device:  $(ls -la $VFIO_DEV 2>/dev/null)"
echo ""

if [ "$GPU_NOW" = "vfio-pci" ]; then
    echo -e "${GREEN}Titan V is ready for sovereign compute.${NC}"
    echo ""
    echo "Run coralReef VFIO tests:"
    echo "  cd /home/biomegate/Development/ecoPrimals/coralReef"
    echo "  CORALREEF_VFIO_BDF=0000:4b:00.0 CORALREEF_VFIO_SM=70 \\"
    echo "    cargo test --test hw_nv_vfio --features vfio -- --ignored --test-threads=1"
else
    echo -e "${RED}Binding failed — GPU driver is $GPU_NOW${NC}"
    exit 1
fi
