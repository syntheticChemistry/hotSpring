#!/bin/bash
# install-boot-config.sh — Install coralReef dual Titan V boot configuration
#
# This installs modprobe.d + udev rules so that on every boot:
#   03:00.0 Titan V → nouveau (oracle, warm)
#   4a:00.0 Titan V → vfio-pci (sovereign target, cold)
#   21:00.0 RTX 5070 → nvidia (display)
#
# Usage: sudo ./scripts/boot/install-boot-config.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== coralReef Boot Config Installer ==="
echo ""

# Step 1: Install modprobe config
echo "[1] Installing modprobe config..."
cp "$SCRIPT_DIR/coralreef-dual-titanv.conf" /etc/modprobe.d/coralreef-dual-titanv.conf
echo "  → /etc/modprobe.d/coralreef-dual-titanv.conf"

# Remove old conflicting config if it only had the nouveau override
if [ -f /etc/modprobe.d/hotspring-nouveau-titanv.conf ]; then
    echo "  Backing up old hotspring-nouveau-titanv.conf → .bak"
    mv /etc/modprobe.d/hotspring-nouveau-titanv.conf /etc/modprobe.d/hotspring-nouveau-titanv.conf.bak
fi

# Step 2: Install udev rules
echo "[2] Installing udev rules..."
cp "$SCRIPT_DIR/99-coralreef-vfio.rules" /etc/udev/rules.d/99-coralreef-vfio.rules
echo "  → /etc/udev/rules.d/99-coralreef-vfio.rules"

# Step 3: Reload udev
echo "[3] Reloading udev rules..."
udevadm control --reload-rules 2>/dev/null || true
udevadm trigger 2>/dev/null || true

# Step 4: Update initramfs (so vfio-pci is available early)
echo "[4] Updating initramfs..."
if command -v update-initramfs >/dev/null 2>&1; then
    update-initramfs -u
elif command -v dracut >/dev/null 2>&1; then
    dracut --force
else
    echo "  ⚠ No initramfs tool found — you may need to update manually"
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "After reboot:"
echo "  03:00.0 → nouveau (oracle)"
echo "  4a:00.0 → vfio-pci (target, group 34)"
echo "  21:00.0 → nvidia (display)"
echo ""
echo "Run 'sudo reboot' to apply."
