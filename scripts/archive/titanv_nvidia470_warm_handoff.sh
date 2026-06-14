#!/bin/bash
# titanv_nvidia470_warm_handoff.sh — Full GR/FECS boot via nvidia-470 warm-handoff
#
# STATUS: DEPRECATED — Use benchScale VM isolation instead (see below).
#
# This direct host-side driver swap is fragile: loading nvidia-470 while
# nvidia-580 is driving the display DRM can lock or crash the kernel.
# The production path is:
#   1. benchScale spins a VM with Titan V VFIO passthrough
#   2. VM loads nvidia-470 internally → GR/FECS/HBM2 fully initialized
#   3. VM shuts down → host reclaims device (reset_method=none preserves warm state)
#   4. coral-ember runs sovereign.init on the warm device
#
# See: infra/agentReagents/templates/reagent-nvidia470-titanv.yaml (to be created)
#      infra/benchScale/ for VM orchestration
#
# This script is retained as a fossil record of the direct driver swap path
# and as a fallback for display-free sessions (TTY/SSH only).
#
# USAGE: sudo bash titanv_nvidia470_warm_handoff.sh
#        (from TTY/SSH, NOT from a graphical terminal)
#        NOT SAFE when DRM is actively serving a display

set -euo pipefail

TITANV_BDF="0000:02:00.0"
RTX5060_BDF="0000:21:00.0"
NVIDIA_470_DIR="/var/lib/dkms/nvidia/470.256.02/$(uname -r)/x86_64/module"
LIVEPATCH_KO="$(dirname "$0")/../livepatch/livepatch_nvkm_mc_reset.ko"

die()  { echo "FATAL: $*" >&2; exit 1; }
log()  { echo "[$(date '+%H:%M:%S')] $*"; }

[[ $EUID -eq 0 ]] || die "Must run as root"
[[ -f "${NVIDIA_470_DIR}/nvidia.ko" ]] || die "nvidia-470 DKMS build not found at ${NVIDIA_470_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Titan V nvidia-470 Warm-Handoff                           ║"
echo "║  GR/FECS + HBM2 full initialization → VFIO sovereign       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

log "[1/10] Pre-flight: checking display sessions..."
if pgrep -x "Xorg\|Xwayland\|gdm\|sddm\|lightdm" >/dev/null 2>&1; then
    echo "  WARNING: Display manager is running."
    echo "  nvidia-580 has active references and CANNOT be unloaded."
    echo "  Stop the display manager first: sudo systemctl stop gdm"
    echo "  Or run this script from TTY (Ctrl+Alt+F2) or SSH."
    read -p "  Continue anyway? (y/N) " -r
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

log "[2/10] Unbinding Titan V from current driver..."
CURRENT_DRV=$(basename "$(readlink /sys/bus/pci/devices/${TITANV_BDF}/driver 2>/dev/null)" 2>/dev/null || echo "none")
log "  Current driver: ${CURRENT_DRV}"
if [[ "$CURRENT_DRV" != "none" ]]; then
    echo "${TITANV_BDF}" > /sys/bus/pci/devices/${TITANV_BDF}/driver/unbind 2>/dev/null || true
fi
echo 0 > /sys/bus/pci/devices/${TITANV_BDF}/d3cold_allowed 2>/dev/null || true
echo on > /sys/bus/pci/devices/${TITANV_BDF}/power/control 2>/dev/null || true
echo "" > /sys/bus/pci/devices/${TITANV_BDF}/driver_override 2>/dev/null || true

log "[3/10] Unloading nvidia-580..."
rmmod nvidia_uvm 2>/dev/null || true
rmmod nvidia_drm 2>/dev/null || true
rmmod nvidia_modeset 2>/dev/null || true
rmmod nvidia 2>/dev/null || {
    log "  WARNING: nvidia-580 has active references, trying harder..."
    nvidia_refs=$(cat /sys/module/nvidia/refcnt 2>/dev/null || echo "?")
    log "  nvidia refcnt: ${nvidia_refs}"
    if [[ "$nvidia_refs" != "0" ]]; then
        die "Cannot unload nvidia-580 (refcnt=${nvidia_refs}). Stop the display manager first."
    fi
}
log "  nvidia-580 unloaded"

log "[4/10] Loading nvidia-470..."
insmod "${NVIDIA_470_DIR}/nvidia.ko" NVreg_OpenRmEnableUnsupportedGpus=1 2>&1
sleep 2
log "  nvidia-470 loaded"

log "[5/10] Binding Titan V to nvidia-470..."
echo "${TITANV_BDF}" > /sys/bus/pci/drivers/nvidia/bind 2>&1 || true
sleep 8

log "[6/10] Checking GR/FECS initialization..."
dmesg | tail -30 | grep -i "NVRM\|nvidia\|gv100\|gr\|fecs" || log "  (no specific messages)"
log "  Power: $(cat /sys/bus/pci/devices/${TITANV_BDF}/power_state 2>/dev/null)"
log "  Driver: $(basename "$(readlink /sys/bus/pci/devices/${TITANV_BDF}/driver 2>/dev/null)" 2>/dev/null)"

log "[7/10] Unbinding from nvidia-470 (preserve warm state)..."
echo "none" > /sys/bus/pci/devices/${TITANV_BDF}/reset_method 2>/dev/null || true
echo "${TITANV_BDF}" > /sys/bus/pci/devices/${TITANV_BDF}/driver/unbind 2>/dev/null || true
sleep 1
log "  Power: $(cat /sys/bus/pci/devices/${TITANV_BDF}/power_state 2>/dev/null)"

log "[8/10] Rebinding to vfio-pci (no reset)..."
echo "vfio-pci" > /sys/bus/pci/devices/${TITANV_BDF}/driver_override
echo "${TITANV_BDF}" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
sleep 1
log "  Driver: $(basename "$(readlink /sys/bus/pci/devices/${TITANV_BDF}/driver 2>/dev/null)" 2>/dev/null)"
log "  Power: $(cat /sys/bus/pci/devices/${TITANV_BDF}/power_state 2>/dev/null)"

log "[9/10] Restoring nvidia-580 for RTX 5060..."
rmmod nvidia 2>/dev/null || true
modprobe nvidia 2>&1 || log "  modprobe nvidia (580) failed"
modprobe nvidia_modeset 2>/dev/null || true
modprobe nvidia_drm 2>/dev/null || true
modprobe nvidia_uvm 2>/dev/null || true
sleep 2
log "  RTX 5060 driver: $(basename "$(readlink /sys/bus/pci/devices/${RTX5060_BDF}/driver 2>/dev/null)" 2>/dev/null)"

log "[10/10] Restarting diesel engine..."
systemctl restart coral-glowplug.service 2>/dev/null || true
sleep 4

echo
echo "═══════════════════════════════════════════════════════════════"
echo "  Warm-handoff complete."
echo "  Next: coralctl sovereign-init --bdf ${TITANV_BDF}"
echo "  Or:   echo '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ember.sovereign.init\",\"params\":{\"bdf\":\"${TITANV_BDF}\"}}' \\"
echo "        | socat -t 30 - UNIX-CONNECT:/run/coralreef/ember-titan-v.sock"
echo "═══════════════════════════════════════════════════════════════"
