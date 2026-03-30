#!/bin/bash
# Deploy patched nouveau.ko with NvPreserveEngines support.
# Run: pkexec /path/to/deploy_nouveau_preserve.sh
set -euo pipefail

KVER=$(uname -r)
MODULE_DIR="/lib/modules/${KVER}/kernel/drivers/gpu/drm/nouveau"
PATCHED="/tmp/nouveau-build/linux-6.17.9/drivers/gpu/drm/nouveau/nouveau.ko"

if [ ! -f "$PATCHED" ]; then
    echo "ERROR: patched module not found at $PATCHED"
    exit 1
fi

echo "═══ nouveau NvPreserveEngines deployment ═══"
echo "kernel: ${KVER}"

if [ ! -f "${MODULE_DIR}/nouveau.ko.stock" ]; then
    echo "▸ backing up stock module"
    cp "${MODULE_DIR}/nouveau.ko" "${MODULE_DIR}/nouveau.ko.stock"
fi

echo "▸ deploying patched module"
cp "$PATCHED" "${MODULE_DIR}/nouveau.ko"

echo "▸ running depmod"
depmod -a

echo "▸ verifying"
modinfo "${MODULE_DIR}/nouveau.ko" | grep -E "vermagic|NvPreserve"

echo "═══ deployment complete ═══"
echo "Load with: sudo modprobe nouveau NvPreserveEngines=1"
