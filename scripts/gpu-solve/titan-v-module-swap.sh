#!/bin/bash
# titan-v-module-swap.sh — Swap nvidia-580 → nvidia-470 for Titan V compute testing
#
# The nvidia-580 driver (Blackwell-era) does NOT support GV100 (Titan V).
# The nvidia-470 driver (DKMS-built for this kernel) DOES support GV100.
#
# This script:
#   1. Stops display manager (X11/Wayland must not hold nvidia refs)
#   2. Unloads nvidia-580 modules
#   3. Loads nvidia-470 modules from DKMS build path
#   4. Binds Titan V to nvidia-470
#   5. Runs toadstool-cylinder hw_nv_buffers tests targeting the Titan V
#   6. Restores nvidia-580 and restarts display
#
# MUST be run from: TTY (Ctrl+Alt+F2) or SSH — NOT from a graphical terminal.
# MUST be run as root (sudo).

set -euo pipefail

TITAN_V_BDF="0000:02:00.0"
TITAN_V_AUDIO_BDF="0000:02:00.1"
RTX_5060_BDF="0000:21:00.0"
NVIDIA_470_DIR="/var/lib/dkms/nvidia/470.256.02/$(uname -r)/x86_64/module"
TOADSTOOL_ROOT="${TOADSTOOL_ROOT:-$HOME/Development/ecoPrimals/primals/toadStool}"
RUSTUP="${CARGO:-$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_prerequisites() {
    if [[ $EUID -ne 0 ]]; then
        echo "ERROR: Must run as root (sudo $0)"
        exit 1
    fi

    if [[ ! -f "$NVIDIA_470_DIR/nvidia.ko" ]]; then
        echo "ERROR: nvidia-470 module not found at $NVIDIA_470_DIR/nvidia.ko"
        echo "Build it: sudo dkms build nvidia/470.256.02 -k $(uname -r)"
        exit 1
    fi

    local drm_refs
    drm_refs=$(awk '/^nvidia_drm/ {print $3}' /proc/modules)
    if [[ -n "$drm_refs" && "$drm_refs" -gt 0 ]]; then
        echo "WARNING: nvidia_drm has $drm_refs references."
        echo "This script will stop the display manager to release them."
        echo ""
    fi
}

stop_display() {
    log "Stopping display manager..."
    if systemctl is-active --quiet gdm3 2>/dev/null; then
        systemctl stop gdm3
    elif systemctl is-active --quiet lightdm 2>/dev/null; then
        systemctl stop lightdm
    elif systemctl is-active --quiet sddm 2>/dev/null; then
        systemctl stop sddm
    fi
    sleep 2

    if command -v fuser &>/dev/null; then
        fuser -k /dev/nvidia* 2>/dev/null || true
        fuser -k /dev/dri/* 2>/dev/null || true
    fi
    sleep 1
}

unload_nvidia_580() {
    log "Unbinding GPUs from nvidia-580..."
    for bdf in "$TITAN_V_BDF" "$RTX_5060_BDF"; do
        local drv
        drv=$(basename "$(readlink "/sys/bus/pci/devices/$bdf/driver" 2>/dev/null)" 2>/dev/null || echo "none")
        if [[ "$drv" == "nvidia" ]]; then
            echo "$bdf" > "/sys/bus/pci/drivers/nvidia/unbind" 2>/dev/null || true
        fi
    done

    log "Unloading nvidia-580 modules..."
    modprobe -r nvidia-uvm 2>/dev/null || true
    modprobe -r nvidia-drm 2>/dev/null || true
    modprobe -r nvidia-modeset 2>/dev/null || true
    modprobe -r nvidia-peermem 2>/dev/null || true
    modprobe -r nvidia 2>/dev/null || true

    if grep -q "^nvidia " /proc/modules 2>/dev/null; then
        log "ERROR: nvidia-580 still loaded. Remaining refs:"
        grep "^nvidia" /proc/modules
        return 1
    fi
    log "nvidia-580 modules unloaded."
}

load_nvidia_470() {
    log "Loading nvidia-470 modules from DKMS build..."
    insmod "$NVIDIA_470_DIR/nvidia.ko" NVreg_OpenRmEnableUnsupportedGpus=1 2>&1 || true
    insmod "$NVIDIA_470_DIR/nvidia-modeset.ko" 2>&1 || true
    insmod "$NVIDIA_470_DIR/nvidia-drm.ko" modeset=0 2>&1 || true
    insmod "$NVIDIA_470_DIR/nvidia-uvm.ko" 2>&1 || true

    if ! grep -q "^nvidia " /proc/modules 2>/dev/null; then
        log "ERROR: nvidia-470 failed to load."
        return 1
    fi
    log "nvidia-470 modules loaded."
}

bind_titan_v() {
    log "Binding Titan V to nvidia-470..."

    echo "" > "/sys/bus/pci/devices/$TITAN_V_BDF/driver_override" 2>/dev/null || true

    local drv
    drv=$(basename "$(readlink "/sys/bus/pci/devices/$TITAN_V_BDF/driver" 2>/dev/null)" 2>/dev/null || echo "none")
    if [[ "$drv" == "vfio-pci" ]]; then
        echo "$TITAN_V_BDF" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
    fi

    echo "nvidia" > "/sys/bus/pci/devices/$TITAN_V_BDF/driver_override"
    echo "$TITAN_V_BDF" > /sys/bus/pci/drivers_probe 2>/dev/null || true
    sleep 2

    drv=$(basename "$(readlink "/sys/bus/pci/devices/$TITAN_V_BDF/driver" 2>/dev/null)" 2>/dev/null || echo "none")
    if [[ "$drv" != "nvidia" ]]; then
        log "ERROR: Titan V not bound to nvidia (driver=$drv)"
        return 1
    fi

    nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv 2>&1 || true
    log "Titan V bound to nvidia-470."
}

run_tests() {
    log "Running toadstool-cylinder hw_nv_buffers tests on Titan V..."
    cd "$TOADSTOOL_ROOT"

    RUSTUP_TOOLCHAIN=stable "$RUSTUP" test \
        --manifest-path crates/toadstool-cylinder/Cargo.toml \
        --features nvidia-drm \
        --test hw_nv_buffers \
        -- --ignored --nocapture 2>&1 | tee /tmp/titan-v-test-results.txt

    log "Test results saved to /tmp/titan-v-test-results.txt"
}

restore_nvidia_580() {
    log "Restoring nvidia-580..."

    echo "$TITAN_V_BDF" > /sys/bus/pci/drivers/nvidia/unbind 2>/dev/null || true

    modprobe -r nvidia-uvm 2>/dev/null || true
    modprobe -r nvidia-drm 2>/dev/null || true
    modprobe -r nvidia-modeset 2>/dev/null || true
    modprobe -r nvidia-peermem 2>/dev/null || true
    modprobe -r nvidia 2>/dev/null || true
    sleep 1

    modprobe nvidia
    modprobe nvidia-modeset
    modprobe nvidia-drm modeset=1
    modprobe nvidia-uvm

    echo "vfio-pci" > "/sys/bus/pci/devices/$TITAN_V_BDF/driver_override"
    echo "$TITAN_V_BDF" > /sys/bus/pci/drivers_probe 2>/dev/null || true

    log "nvidia-580 restored."
}

restart_display() {
    log "Restarting display manager..."
    if systemctl is-enabled --quiet gdm3 2>/dev/null; then
        systemctl start gdm3
    elif systemctl is-enabled --quiet lightdm 2>/dev/null; then
        systemctl start lightdm
    elif systemctl is-enabled --quiet sddm 2>/dev/null; then
        systemctl start sddm
    fi
    log "Display manager started."
}

show_status() {
    echo "=== Module Status ==="
    grep "^nvidia" /proc/modules 2>/dev/null || echo "No nvidia modules loaded"
    echo ""
    echo "=== GPU Status ==="
    for bdf in "$TITAN_V_BDF" "$RTX_5060_BDF"; do
        local drv
        drv=$(basename "$(readlink "/sys/bus/pci/devices/$bdf/driver" 2>/dev/null)" 2>/dev/null || echo "none")
        local name
        name=$(lspci -s "${bdf#0000:}" 2>/dev/null | cut -d: -f3- | sed 's/^ //')
        echo "  $bdf: driver=$drv ($name)"
    done
}

case "${1:-test}" in
    test)
        check_prerequisites
        echo ""
        echo "=== Titan V Compute Test (nvidia-470 module swap) ==="
        echo "This will temporarily stop your display server."
        echo "Run from TTY (Ctrl+Alt+F2) or SSH."
        echo ""
        read -rp "Continue? [y/N] " confirm
        [[ "$confirm" =~ ^[Yy]$ ]] || exit 0

        trap 'log "ERROR — attempting restore..."; restore_nvidia_580; restart_display' ERR

        stop_display
        unload_nvidia_580
        load_nvidia_470
        bind_titan_v
        run_tests
        restore_nvidia_580
        restart_display

        log "=== Complete ==="
        ;;

    swap-only)
        check_prerequisites
        stop_display
        unload_nvidia_580
        load_nvidia_470
        bind_titan_v
        log "Titan V on nvidia-470. Run tests manually, then: $0 restore"
        ;;

    restore)
        restore_nvidia_580
        restart_display
        log "Restored to nvidia-580."
        ;;

    status)
        show_status
        ;;

    *)
        echo "Usage: $0 {test|swap-only|restore|status}"
        echo ""
        echo "  test       Full cycle: swap to 470, test, restore 580"
        echo "  swap-only  Swap to 470, leave for manual testing"
        echo "  restore    Restore nvidia-580 and restart display"
        echo "  status     Show current driver/module state"
        ;;
esac
