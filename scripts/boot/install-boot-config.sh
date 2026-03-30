#!/bin/bash
# install-boot-config.sh — Install coralReef GPU fleet boot configuration
#
# This installs modprobe.d, udev rules, sudoers, and the sysfs write
# helper so that on every boot:
#   03:00.0 Titan V   → vfio-pci (oracle, swappable to nouveau)
#   4c:00.0 K80 die#1 → vfio-pci (sovereign target)
#   4d:00.0 K80 die#2 → vfio-pci (oracle, swappable to nouveau)
#   21:00.0 RTX 5070  → nvidia (display, locked)
#
# Usage: sudo ./scripts/boot/install-boot-config.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== coralReef Boot Config Installer ==="
echo ""

# Step 1: Install sysfs write helper (D-state-safe replacement for tee)
echo "[1] Installing coralreef-sysfs-write helper..."
cp "$SCRIPT_DIR/coralreef-sysfs-write" /usr/local/bin/coralreef-sysfs-write
chmod 755 /usr/local/bin/coralreef-sysfs-write
echo "  -> /usr/local/bin/coralreef-sysfs-write"

# Step 2: Install sudoers (uses sysfs-write helper instead of tee)
echo "[2] Installing sudoers config..."
cp "$SCRIPT_DIR/coralreef-sudoers" /etc/sudoers.d/coralreef
chmod 440 /etc/sudoers.d/coralreef
visudo -c -f /etc/sudoers.d/coralreef || {
    echo "  ERROR: sudoers syntax check failed, restoring backup" >&2
    exit 1
}
echo "  -> /etc/sudoers.d/coralreef"

# Step 3: Install modprobe config
echo "[3] Installing modprobe config..."
cp "$SCRIPT_DIR/coralreef-dual-titanv.conf" /etc/modprobe.d/coralreef-dual-titanv.conf
echo "  -> /etc/modprobe.d/coralreef-dual-titanv.conf"

if [ -f /etc/modprobe.d/hotspring-nouveau-titanv.conf ]; then
    echo "  Backing up old hotspring-nouveau-titanv.conf -> .bak"
    mv /etc/modprobe.d/hotspring-nouveau-titanv.conf /etc/modprobe.d/hotspring-nouveau-titanv.conf.bak
fi

# Step 4: Install udev rules (VFIO binding + permissions)
echo "[4] Installing udev rules..."
cp "$SCRIPT_DIR/99-coralreef-vfio.rules" /etc/udev/rules.d/99-coralreef-vfio.rules
cp "$SCRIPT_DIR/99-coralreef-permissions.rules" /etc/udev/rules.d/99-coralreef-permissions.rules
echo "  -> /etc/udev/rules.d/99-coralreef-vfio.rules"
echo "  -> /etc/udev/rules.d/99-coralreef-permissions.rules"

# Step 5: Reload udev
echo "[5] Reloading udev rules..."
udevadm control --reload-rules 2>/dev/null || true
udevadm trigger 2>/dev/null || true

# Step 6: Update kernel cmdline vfio-pci.ids (Pop!_OS kernelstub)
# K80 (10de:102d) included — both dies boot cold on vfio-pci
echo "[6] Updating kernel cmdline vfio-pci.ids..."
VFIO_IDS="10de:1d81,10de:10f2,10de:102d"
if command -v kernelstub >/dev/null 2>&1; then
    CURRENT_IDS=$(grep -oP 'vfio-pci\.ids=\K[^ ]+' /proc/cmdline 2>/dev/null || echo "")
    if [ "$CURRENT_IDS" != "$VFIO_IDS" ]; then
        if [ -n "$CURRENT_IDS" ]; then
            kernelstub -d "vfio-pci.ids=$CURRENT_IDS" 2>/dev/null || true
        fi
        kernelstub -a "vfio-pci.ids=$VFIO_IDS"
        echo "  -> vfio-pci.ids=$VFIO_IDS (was: $CURRENT_IDS)"
    else
        echo "  -> vfio-pci.ids already correct"
    fi
else
    echo "  WARNING: kernelstub not found -- update vfio-pci.ids= in bootloader manually"
    echo "  Target: vfio-pci.ids=$VFIO_IDS"
fi

# Step 7: Update initramfs (so vfio-pci is available early)
echo "[7] Updating initramfs..."
if command -v update-initramfs >/dev/null 2>&1; then
    update-initramfs -u
elif command -v dracut >/dev/null 2>&1; then
    dracut --force
else
    echo "  WARNING: No initramfs tool found -- you may need to update manually"
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "After reboot:"
echo "  03:00.0 Titan V   -> vfio-pci (oracle)"
echo "  4c:00.0 K80 die#1 -> vfio-pci (sovereign, cold)"
echo "  4d:00.0 K80 die#2 -> vfio-pci (reagent target, cold)"
echo "  21:00.0 RTX 5070  -> nvidia (display)"
echo ""
echo "K80 init via agentReagents VM (nvidia-470 reagent) or Ember recipe replay."
echo ""
echo "Run 'sudo reboot' to apply."
