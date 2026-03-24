#!/bin/bash
# build_nvidia_oracle.sh — Build a renamed nvidia kernel module for driver coexistence.
#
# Patches MODULE_BASE_NAME and NV_MAJOR_DEVICE_NUMBER in the NVIDIA open kernel
# source so the resulting .ko loads as "nvidia_oracle" with a dynamic major number,
# allowing it to coexist alongside the system nvidia.ko.
#
# Usage: sudo ./build_nvidia_oracle.sh [VERSION]
#   VERSION defaults to 580.126.18

set -euo pipefail

VERSION="${1:-580.126.18}"
SRC_DIR="/usr/src/nvidia-${VERSION}"
WORK_DIR="/tmp/nvidia_oracle_build_${VERSION}"
KVER=$(uname -r)
INSTALL_DIR="/lib/modules/${KVER}/extra"
MODULE_NAME="nvidia_oracle"
DYNAMIC_MAJOR=0

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must run as root (need to install to /lib/modules)"
    exit 1
fi

if [ ! -d "${SRC_DIR}" ]; then
    echo "ERROR: source directory not found: ${SRC_DIR}"
    echo "Install nvidia-open-kernel-source first."
    exit 1
fi

echo "=== Building ${MODULE_NAME}.ko from ${SRC_DIR} ==="
echo "    Kernel: ${KVER}"
echo "    Work:   ${WORK_DIR}"

rm -rf "${WORK_DIR}"
cp -a "${SRC_DIR}" "${WORK_DIR}"

# --- Patch 1: MODULE_BASE_NAME ---
LINUX_H="${WORK_DIR}/common/inc/nv-linux.h"
if grep -q 'MODULE_BASE_NAME.*"nvidia"' "${LINUX_H}"; then
    sed -i 's/\(#define MODULE_BASE_NAME\s*\)"nvidia"/\1"nvidia_oracle"/' "${LINUX_H}"
    echo "[PATCH] MODULE_BASE_NAME -> \"nvidia_oracle\""
else
    echo "WARNING: MODULE_BASE_NAME pattern not found in ${LINUX_H}"
    echo "         Searching for alternative locations..."
    grep -rn 'MODULE_BASE_NAME' "${WORK_DIR}" || true
fi

# --- Patch 2: NV_MAJOR_DEVICE_NUMBER -> dynamic ---
CHARDEV_H="${WORK_DIR}/common/inc/nv-chardev-numbers.h"
if [ -f "${CHARDEV_H}" ] && grep -q 'NV_MAJOR_DEVICE_NUMBER.*195' "${CHARDEV_H}"; then
    sed -i "s/\(#define NV_MAJOR_DEVICE_NUMBER\s*\)195/\1${DYNAMIC_MAJOR}/" "${CHARDEV_H}"
    echo "[PATCH] NV_MAJOR_DEVICE_NUMBER -> ${DYNAMIC_MAJOR} (dynamic)"
else
    echo "INFO: NV_MAJOR_DEVICE_NUMBER not at expected path — trying broader search"
    for f in $(grep -rl 'NV_MAJOR_DEVICE_NUMBER.*195' "${WORK_DIR}" 2>/dev/null); do
        sed -i "s/\(#define NV_MAJOR_DEVICE_NUMBER\s*\)195/\1${DYNAMIC_MAJOR}/" "$f"
        echo "[PATCH] ${f}: NV_MAJOR_DEVICE_NUMBER -> ${DYNAMIC_MAJOR}"
    done
fi

# --- Build ---
echo ""
echo "=== Compiling (this takes a few minutes) ==="
cd "${WORK_DIR}"
make -j"$(nproc)" modules SYSSRC="/lib/modules/${KVER}/build" 2>&1 | tail -20

# --- Install ---
echo ""
echo "=== Installing to ${INSTALL_DIR} ==="
mkdir -p "${INSTALL_DIR}"

for ko in $(find "${WORK_DIR}" -name '*.ko' -o -name '*.ko.xz' -o -name '*.ko.zst' 2>/dev/null); do
    base=$(basename "$ko")
    renamed="${base/nvidia/${MODULE_NAME}}"
    cp "$ko" "${INSTALL_DIR}/${renamed}"
    echo "  installed: ${INSTALL_DIR}/${renamed}"
done

depmod -a
echo ""
echo "=== Done ==="
echo "Load with: modprobe nvidia_oracle"
echo "Verify:    lsmod | grep nvidia_oracle"
echo "The module uses dynamic major number — no conflict with nvidia.ko (major 195)"
