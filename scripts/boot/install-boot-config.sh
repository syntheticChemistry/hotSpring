#!/bin/bash
# install-boot-config.sh — Install coralReef GPU fleet boot configuration
#
# This installs modprobe.d, udev rules, sudoers, and the sysfs write
# helper so that on every boot:
#   02:00.0 Titan V   → vfio-pci (oracle, swappable to nouveau via ember)
#   4b:00.0 K80 die#1 → vfio-pci (sovereign target, cold)
#   4c:00.0 K80 die#2 → vfio-pci (reagent target, cold)
#   21:00.0 RTX 5060  → nvidia (display + shared compute)
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

# Step 5a: Install coral-ember (immortal VFIO fd holder + fork-isolated MMIO)
echo "[5a] Installing coral-ember.service (sacrifice / crash interceptor)..."
cp "$SCRIPT_DIR/coral-ember.service" /etc/systemd/system/coral-ember.service
systemctl daemon-reload
systemctl enable coral-ember.service
echo "  -> /etc/systemd/system/coral-ember.service (enabled)"

# Step 5b: PLX keepalive — now handled by coral-ember's pcie_keepalive.rs thread
# The standalone plx-keepalive.service is DEPRECATED (replaced May 2026).
# coral-ember reads [[pcie_switch]] from glowplug.toml and generates keepalive
# traffic internally. If the old service is still enabled, disable it.
if systemctl is-enabled --quiet plx-keepalive.service 2>/dev/null; then
    echo "[5b] Disabling deprecated plx-keepalive.service (now handled by coral-ember)..."
    systemctl disable plx-keepalive.service
    systemctl stop plx-keepalive.service 2>/dev/null || true
    echo "  -> plx-keepalive.service disabled (ember handles keepalive)"
else
    echo "[5b] PLX keepalive: handled by coral-ember (plx-keepalive.service not present)"
fi

# Step 5c: Install wake-and-run helper
cp "$SCRIPT_DIR/k80-wake-and-run.sh" /usr/local/bin/k80-wake-and-run.sh
chmod 755 /usr/local/bin/k80-wake-and-run.sh
echo "  -> /usr/local/bin/k80-wake-and-run.sh"

# Step 6: Update kernel cmdline vfio-pci.ids (Pop!_OS kernelstub)
# Titan V (1d81+10f2) and K80 (102d) MUST be claimed at boot via cmdline.
# K80 is behind a PLX PEX 8747 switch — any re-probe kills the PCIe link.
# DO NOT rely on udev drivers_probe for K80; cmdline binding is required.
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
echo "  02:00.0 Titan V   -> vfio-pci (oracle, IOMMU group 69)"
echo "  4b:00.0 K80 die#1 -> vfio-pci (sovereign, IOMMU group 35)"
echo "  4c:00.0 K80 die#2 -> vfio-pci (reagent, IOMMU group 36)"
echo "  21:00.0 RTX 5060  -> nvidia (display + shared compute)"
echo ""
echo "K80 init via coral-ember warm handoff or sovereign cold-boot pipeline."
echo ""
echo "Run 'sudo reboot' to apply."
