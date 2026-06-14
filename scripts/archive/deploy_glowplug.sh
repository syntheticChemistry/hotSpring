#!/bin/bash
# Deploy coral-ember + coral-glowplug binaries and restart services.
#
# SAFE DEPLOYMENT ORDER:
#   1. Build release binaries
#   2. Deploy coral-ember first (if not already running)
#   3. Stop coral-glowplug (ember keeps VFIO fds alive — no PM reset)
#   4. Copy updated binary
#   5. Restart coral-glowplug (receives fds from ember via SCM_RIGHTS)
#
# Run: pkexec /path/to/deploy_glowplug.sh
set -euo pipefail

CORALREEF="${CORALREEF:-${HOME}/Development/ecoPrimals/primals/coralReef}"
TARGET="${CORALREEF}/target/release"

echo "═══ coral-glowplug + coral-ember deployment ═══"

# Phase 1: Deploy ember if binary changed or not running
if ! systemctl is-active --quiet coral-ember 2>/dev/null; then
    echo "▸ ember not running — deploying coral-ember first"

    if [ -f "${TARGET}/coral-ember" ]; then
        cp "${TARGET}/coral-ember" /usr/local/bin/coral-ember
        echo "  ✓ binary copied"
    else
        echo "  ✗ ${TARGET}/coral-ember not found — build first"
        exit 1
    fi

    # Install service file if needed
    if [ ! -f /etc/systemd/system/coral-ember.service ]; then
        cp "${CORALREEF}/crates/coral-glowplug/coral-ember.service" /etc/systemd/system/
        systemctl daemon-reload
        systemctl enable coral-ember
        echo "  ✓ systemd unit installed and enabled"
    fi

    # Ensure config exists
    if [ ! -f /etc/coralreef/glowplug.toml ]; then
        mkdir -p /etc/coralreef
        cp "${CORALREEF}/crates/coral-glowplug/glowplug.toml" /etc/coralreef/glowplug.toml
        echo "  ✓ config installed"
    fi

    systemctl start coral-ember
    sleep 1

    if systemctl is-active --quiet coral-ember; then
        echo "  ✓ coral-ember started"
    else
        echo "  ✗ coral-ember failed to start"
        journalctl -u coral-ember --no-pager -n 10
        exit 1
    fi
else
    echo "▸ coral-ember already running — VFIO fds are safe"
fi

# Phase 2: Deploy coral-glowplug (safe with ember holding fds)
echo "▸ deploying coral-glowplug"

if [ -f "${TARGET}/coral-glowplug" ]; then
    systemctl stop coral-glowplug 2>/dev/null || true
    sleep 1
    cp "${TARGET}/coral-glowplug" /usr/local/bin/coral-glowplug
    echo "  ✓ binary copied (ember prevented PM reset during stop)"
else
    echo "  ✗ ${TARGET}/coral-glowplug not found — build first"
    exit 1
fi

# Install service file if needed
if [ ! -f /etc/systemd/system/coral-glowplug.service ]; then
    cp "${CORALREEF}/crates/coral-glowplug/coral-glowplug.service" /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable coral-glowplug
    echo "  ✓ systemd unit installed and enabled"
fi

systemctl start coral-glowplug
sleep 2

if systemctl is-active --quiet coral-glowplug; then
    echo "  ✓ coral-glowplug started"
else
    echo "  ✗ coral-glowplug failed to start"
    journalctl -u coral-glowplug --no-pager -n 10
    exit 1
fi

# Phase 3: Verify
echo ""
echo "═══ Verification ═══"
echo "▸ ember status:"
systemctl status coral-ember --no-pager -l 2>&1 | head -5
echo ""
echo "▸ glowplug device list:"
echo '{"jsonrpc":"2.0","method":"device.list","id":1}' | socat - UNIX-CONNECT:/run/coralreef/glowplug.sock 2>/dev/null || echo "  (socat not available or socket not ready)"
echo ""
echo "═══ deployment complete ═══"
