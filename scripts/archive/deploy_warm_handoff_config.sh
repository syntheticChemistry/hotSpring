#!/bin/bash
# Deploy warm-handoff modprobe config + sudoers rule.
# Does NOT touch initramfs or kernel cmdline.
# Run: pkexec /path/to/deploy_warm_handoff_config.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOOT_DIR="${SCRIPT_DIR}/boot"

echo "═══ Warm handoff config deployment ═══"

echo "▸ Installing modprobe config (NvPreserveEngines=1)..."
cp "${BOOT_DIR}/coralreef-dual-titanv.conf" /etc/modprobe.d/coralreef-dual-titanv.conf

echo "▸ Installing sudoers (adds nouveau param write rule)..."
cp "${BOOT_DIR}/coralreef-sudoers" /etc/sudoers.d/coralreef
chmod 440 /etc/sudoers.d/coralreef
visudo -c -f /etc/sudoers.d/coralreef || {
    echo "ERROR: sudoers syntax check failed" >&2
    exit 1
}

echo "═══ done — modprobe nouveau will now set NvPreserveEngines=1 ═══"
