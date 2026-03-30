#!/bin/bash
# Deploy updated coral-ember + coral-glowplug binaries and restart services.
#
# Run: pkexec bash /path/to/deploy_ember_update.sh
set -euo pipefail

CORALREEF="${CORALREEF:-${HOME}/Development/ecoPrimals/primals/coralReef}"
TARGET="${CORALREEF}/target/release"

echo "=== Deploying updated coral-ember + coral-glowplug ==="

for bin in coral-ember coral-glowplug; do
    if [ ! -f "${TARGET}/${bin}" ]; then
        echo "ERROR: ${TARGET}/${bin} not found"
        exit 1
    fi
done

echo "step 1: stopping services..."
systemctl stop coral-glowplug 2>/dev/null || true
systemctl stop coral-ember 2>/dev/null || true
sleep 1

echo "step 2: installing binaries..."
rm -f /usr/local/bin/coral-ember
cp "${TARGET}/coral-ember" /usr/local/bin/coral-ember
chmod 755 /usr/local/bin/coral-ember

rm -f /usr/local/bin/coral-glowplug
cp "${TARGET}/coral-glowplug" /usr/local/bin/coral-glowplug
chmod 755 /usr/local/bin/coral-glowplug

echo "step 3: restarting services..."
systemctl daemon-reload
systemctl start coral-ember
sleep 2

if systemctl is-active --quiet coral-ember; then
    echo "  coral-ember: ACTIVE"
else
    echo "  coral-ember: FAILED"
    journalctl -u coral-ember --no-pager -n 5
    exit 1
fi

systemctl start coral-glowplug
sleep 2

if systemctl is-active --quiet coral-glowplug; then
    echo "  coral-glowplug: ACTIVE"
else
    echo "  coral-glowplug: FAILED (non-fatal, ember is the critical piece)"
    journalctl -u coral-glowplug --no-pager -n 5
fi

echo "=== Deploy complete (reset_method race fix active) ==="
