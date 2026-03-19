#!/bin/bash
# First-time deployment of coral-ember + updated coral-glowplug.
#
# This script installs all binaries and service files WITHOUT stopping
# the running coral-glowplug. A reboot is required after installation
# so that coral-ember starts first and holds the VFIO fds before
# coral-glowplug connects.
#
# Why reboot instead of restart?
# - The current glowplug holds VFIO groups; ember can't co-open them
# - Stopping glowplug without ember triggers PM reset (GV100 no FLR)
# - PM reset on 0000:40 root complex cascades to WiFi (46:00.0)
# - After reboot, systemd ordering ensures ember grabs fds first
#
# Run: pkexec bash /path/to/deploy_ember_first_time.sh
set -euo pipefail

CORALREEF="/home/biomegate/Development/ecoPrimals/coralReef"
TARGET="${CORALREEF}/target/release"

echo "═══ First-time ember deployment (install only, no restart) ═══"
echo ""

# --- Pre-flight checks ---
for bin in coral-ember coral-glowplug; do
    if [ ! -f "${TARGET}/${bin}" ]; then
        echo "✗ ${TARGET}/${bin} not found — run 'cargo build --release -p coral-glowplug' first"
        exit 1
    fi
done

echo "▸ Pre-flight passed: both release binaries present"
echo ""

# --- Phase 1: Install coral-ember ---
echo "▸ Phase 1: Installing coral-ember"

cp "${TARGET}/coral-ember" /usr/local/bin/coral-ember
chmod 755 /usr/local/bin/coral-ember
echo "  ✓ binary → /usr/local/bin/coral-ember"

cp "${CORALREEF}/crates/coral-glowplug/coral-ember.service" /etc/systemd/system/
echo "  ✓ unit → /etc/systemd/system/coral-ember.service"

# --- Phase 2: Install updated coral-glowplug ---
echo "▸ Phase 2: Installing updated coral-glowplug"

# rm + cp avoids "Text file busy" when the old binary is running
rm -f /usr/local/bin/coral-glowplug
cp "${TARGET}/coral-glowplug" /usr/local/bin/coral-glowplug
chmod 755 /usr/local/bin/coral-glowplug
echo "  ✓ binary → /usr/local/bin/coral-glowplug (rm+cp, running process keeps old inode)"

cp "${CORALREEF}/crates/coral-glowplug/coral-glowplug.service" /etc/systemd/system/
echo "  ✓ unit → /etc/systemd/system/coral-glowplug.service (now depends on coral-ember)"

# --- Phase 3: Ensure config exists ---
echo "▸ Phase 3: Config check"
if [ ! -f /etc/coralreef/glowplug.toml ]; then
    mkdir -p /etc/coralreef
    cp "${CORALREEF}/crates/coral-glowplug/glowplug.toml" /etc/coralreef/glowplug.toml
    echo "  ✓ config installed"
else
    echo "  ✓ config already present at /etc/coralreef/glowplug.toml"
fi

# Ensure runtime directory
mkdir -p /run/coralreef
chmod 755 /run/coralreef
echo "  ✓ runtime dir /run/coralreef exists"

# --- Phase 4: Reload systemd and enable ---
echo "▸ Phase 4: Systemd reload + enable"
systemctl daemon-reload
systemctl enable coral-ember
systemctl enable coral-glowplug
echo "  ✓ daemon-reload complete"
echo "  ✓ coral-ember enabled (will start on boot)"
echo "  ✓ coral-glowplug enabled (will start after ember)"

# --- Summary ---
echo ""
echo "═══ Installation complete ═══"
echo ""
echo "  Installed:"
echo "    /usr/local/bin/coral-ember      (immortal VFIO fd holder)"
echo "    /usr/local/bin/coral-glowplug   (restartable daemon, ember-aware)"
echo "    /etc/systemd/system/coral-ember.service"
echo "    /etc/systemd/system/coral-glowplug.service"
echo ""
echo "  Boot order: coral-ember → coral-glowplug → display-manager"
echo ""
echo "  ⚠ REBOOT REQUIRED to activate the ember architecture."
echo "    The current glowplug (PID $(pgrep coral-glowplug 2>/dev/null || echo '?')) holds"
echo "    the VFIO groups — ember can't co-open them. After reboot, ember"
echo "    starts first and all subsequent glowplug restarts are safe."
echo ""
echo "  Run: sudo reboot"
