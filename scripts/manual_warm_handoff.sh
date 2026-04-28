#!/usr/bin/env bash
# manual_warm_handoff.sh — Standalone warm handoff without ember/glowplug.
#
# Performs nouveau → vfio-pci warm handoff on Volta (GV100/Titan V) by:
#   1. Cleaning any stale livepatch/nouveau state
#   2. Unbinding GPU from vfio-pci
#   3. Loading nouveau to init HBM2 + PLLs + FECS/GPCCS
#   4. Loading livepatch to prevent teardown of falcon state
#   5. Swapping back to vfio-pci via PCI remove+rescan (no FLR)
#
# CRITICAL ORDERING:
#   - Livepatch MUST NOT be loaded before nouveau (causes module deadlock)
#   - Livepatch MUST be loaded AFTER nouveau finishes probing the GPU
#   - Never load both simultaneously (spinlock contention in init_module)
#
# Usage:
#   sudo ./manual_warm_handoff.sh 0000:03:00.0
#   sudo SETTLE_SECS=15 ./manual_warm_handoff.sh 0000:03:00.0

set -euo pipefail

BDF="${1:?Usage: $0 <PCI_BDF>  (e.g. 0000:03:00.0)}"
SETTLE_SECS="${SETTLE_SECS:-10}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIVEPATCH_DIR="$SCRIPT_DIR/livepatch"
LIVEPATCH_KO="$LIVEPATCH_DIR/livepatch_nvkm_mc_reset.ko"

SYSFS="/sys/bus/pci/devices/$BDF"
LP_SYSFS="/sys/kernel/livepatch/livepatch_nvkm_mc_reset/enabled"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[warm]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

SYSFS_TIMEOUT=30
CORAL_PROBE="${CORAL_PROBE:-coral-probe}"

gpu_read() {
    if command -v "$CORAL_PROBE" &>/dev/null; then
        "$CORAL_PROBE" read "$BDF" "$1" 2>/dev/null \
            | grep -oP '0x[0-9a-fA-F]+$' || echo "0xDEADDEAD"
    else
        python3 -c "
import mmap, struct, sys
f = open('$SYSFS/resource0', 'r+b')
mm = mmap.mmap(f.fileno(), 0x1000000)
mm.seek(int(sys.argv[1], 0))
v = struct.unpack('<I', mm.read(4))[0]
print(f'{v:#010x}')
mm.close(); f.close()
" "$1" 2>/dev/null || echo "0xDEADDEAD"
    fi
}

sysfs_write() {
    local target="$1" content="$2" desc="${3:-sysfs write}"
    if ! timeout "$SYSFS_TIMEOUT" bash -c "echo '$content' > '$target'" 2>/dev/null; then
        fail "$desc TIMED OUT after ${SYSFS_TIMEOUT}s — kernel may be stuck on $target"
    fi
}

# ──────────────────────────────────────────────────────────────
# Phase 0: Preflight
# ──────────────────────────────────────────────────────────────
log "Phase 0: Preflight"

[[ $EUID -eq 0 ]] || fail "Must run as root"
[[ -d "$SYSFS" ]] || fail "PCI device $BDF not found"
[[ -f "$LIVEPATCH_KO" ]] || fail "Livepatch .ko not found at $LIVEPATCH_KO"

# Detect current driver
CURRENT_DRV="(none)"
if [[ -L "$SYSFS/driver" ]]; then
    CURRENT_DRV="$(basename "$(readlink "$SYSFS/driver")")"
fi
ok "Device $BDF currently on: $CURRENT_DRV"

# ──────────────────────────────────────────────────────────────
# Phase 1: Clean stale state (prevent module deadlock)
# ──────────────────────────────────────────────────────────────
log "Phase 1: Cleaning stale module state"

if grep -q "livepatch_nvkm_mc_reset" /proc/modules 2>/dev/null; then
    LP_STATE=$(awk '/livepatch_nvkm_mc_reset/{print $5}' /proc/modules)
    if [[ "$LP_STATE" == "Live" ]]; then
        log "Removing stale livepatch module..."
        if [[ -f "$LP_SYSFS" ]]; then
            echo 0 > "$LP_SYSFS" 2>/dev/null || true
            sleep 2
        fi
        rmmod livepatch_nvkm_mc_reset 2>/dev/null || warn "rmmod livepatch failed (may need reboot)"
    else
        fail "livepatch_nvkm_mc_reset is in '$LP_STATE' state — reboot required"
    fi
fi

if grep -q "^nouveau " /proc/modules 2>/dev/null; then
    NV_STATE=$(awk '/^nouveau /{print $5}' /proc/modules)
    if [[ "$NV_STATE" != "Live" ]]; then
        fail "nouveau is in '$NV_STATE' state — reboot required"
    fi
    log "nouveau already loaded (live)"
    NOUVEAU_PRELOADED=1
else
    NOUVEAU_PRELOADED=0
fi
ok "Module state clean"

# ──────────────────────────────────────────────────────────────
# Phase 2: Unbind from vfio-pci (if needed)
# ──────────────────────────────────────────────────────────────
log "Phase 2: Preparing GPU for nouveau"

if [[ "$CURRENT_DRV" == "vfio-pci" ]]; then
    log "Unbinding $BDF from vfio-pci..."
    sysfs_write "/sys/bus/pci/drivers/vfio-pci/unbind" "$BDF" "unbind $BDF from vfio-pci"
    ok "Unbound from vfio-pci"
fi

# Also handle the audio function in the same IOMMU group
AUDIO_BDF="${BDF%.0}.1"
if [[ -L "/sys/bus/pci/devices/$AUDIO_BDF/driver" ]]; then
    AUDIO_DRV="$(basename "$(readlink "/sys/bus/pci/devices/$AUDIO_BDF/driver")")"
    if [[ "$AUDIO_DRV" == "vfio-pci" ]]; then
        sysfs_write "/sys/bus/pci/drivers/vfio-pci/unbind" "$AUDIO_BDF" "unbind audio from vfio-pci"
    fi
fi

# Clear driver_override so nouveau can claim via PCI ID match
sysfs_write "$SYSFS/driver_override" "" "clear driver_override for $BDF"
sysfs_write "/sys/bus/pci/devices/$AUDIO_BDF/driver_override" "" "clear driver_override for audio"

# ──────────────────────────────────────────────────────────────
# Phase 3: Load nouveau → GPU initialization
# ──────────────────────────────────────────────────────────────
log "Phase 3: Loading nouveau (NvPreserveEngines=1)"

if [[ $NOUVEAU_PRELOADED -eq 0 ]]; then
    dmesg --clear 2>/dev/null || true
    modprobe --ignore-install nouveau NvPreserveEngines=1
    ok "nouveau module loaded"
fi

# nouveau may not auto-bind if no new PCI hotplug event. Trigger a probe.
if [[ ! -L "$SYSFS/driver" ]]; then
    log "Triggering PCI driver probe..."
    sysfs_write "/sys/bus/pci/drivers_probe" "$BDF" "drivers_probe $BDF (nouveau)"
fi

log "Waiting ${SETTLE_SECS}s for nouveau to initialize HBM2 + FECS..."
sleep "$SETTLE_SECS"

# Verify nouveau claimed the device
if [[ -L "$SYSFS/driver" ]]; then
    ACTUAL_DRV="$(basename "$(readlink "$SYSFS/driver")")"
    if [[ "$ACTUAL_DRV" == "nouveau" ]]; then
        ok "nouveau claimed $BDF"
    else
        fail "Unexpected driver: $ACTUAL_DRV (expected nouveau)"
    fi
else
    warn "nouveau did not claim $BDF — checking dmesg..."
    dmesg | grep -i "nouveau\|gv100\|GR\|FECS\|error\|fail" | tail -20
    fail "nouveau probe failed"
fi

# Check FECS state via BAR0
FECS_CPUCTL=$(gpu_read 0x409100)
log "FECS CPUCTL = $FECS_CPUCTL"
if [[ "$FECS_CPUCTL" == "0xDEADDEAD" || "$FECS_CPUCTL" == "0xbadf"* ]]; then
    warn "FECS appears dead or faulted — warm handoff may not preserve state"
fi

# ──────────────────────────────────────────────────────────────
# Phase 4: Load livepatch (AFTER nouveau init)
# ──────────────────────────────────────────────────────────────
log "Phase 4: Loading livepatch (prevents falcon teardown)"

if ! grep -q "livepatch_nvkm_mc_reset" /proc/modules 2>/dev/null; then
    insmod "$LIVEPATCH_KO" || fail "insmod livepatch failed"
    sleep 1
fi

if [[ -f "$LP_SYSFS" ]]; then
    LP_VAL=$(cat "$LP_SYSFS")
    ok "Livepatch active (enabled=$LP_VAL)"
else
    fail "Livepatch sysfs not found after insmod"
fi

# ──────────────────────────────────────────────────────────────
# Phase 5: Swap to vfio-pci (preserving warm state)
# ──────────────────────────────────────────────────────────────
log "Phase 5: Swapping to vfio-pci"

# Volta GPUs lack FLR (FLReset-). Prevent any reset on close.
if [[ -f "$SYSFS/reset_method" ]]; then
    sysfs_write "$SYSFS/reset_method" "" "disable reset methods for $BDF"
fi

# Set driver_override so vfio-pci claims on rescan
sysfs_write "$SYSFS/driver_override" "vfio-pci" "set driver_override to vfio-pci"
sysfs_write "/sys/bus/pci/devices/$AUDIO_BDF/driver_override" "vfio-pci" "set audio driver_override"

# PCI remove+rescan is safer than sysfs unbind for Volta (avoids D-state).
# The livepatch NOPs mc_reset, gr_fini, falcon_fini, and runl_commit(count=0)
# so FECS stays in its context-switch-ready HALT state.
BUS="${BDF%:*}"
log "Removing $BDF from PCI..."
sysfs_write "$SYSFS/remove" "1" "PCI remove $BDF"
sleep 1

log "Rescanning PCI bus..."
sysfs_write "/sys/bus/pci/rescan" "1" "PCI rescan"
sleep 2

# Verify vfio-pci claimed the device
if [[ -L "$SYSFS/driver" ]]; then
    FINAL_DRV="$(basename "$(readlink "$SYSFS/driver")")"
    if [[ "$FINAL_DRV" == "vfio-pci" ]]; then
        ok "vfio-pci claimed $BDF"
    else
        fail "Unexpected driver after rescan: $FINAL_DRV"
    fi
else
    warn "$BDF has no driver after rescan — trying explicit bind"
    echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || fail "vfio-pci bind failed"
    ok "Explicitly bound to vfio-pci"
fi

# ──────────────────────────────────────────────────────────────
# Phase 6: Verify warm state
# ──────────────────────────────────────────────────────────────
log "Phase 6: Verifying warm GPU state"

BOOT0=$(gpu_read 0x0)
PMC_EN=$(gpu_read 0x200)
FECS_CPUCTL=$(gpu_read 0x409100)
FECS_SCTL=$(gpu_read 0x409144)

log "BOOT0       = $BOOT0"
log "PMC_ENABLE  = $PMC_EN"
log "FECS CPUCTL = $FECS_CPUCTL"
log "FECS SCTL   = $FECS_SCTL"

if [[ "$BOOT0" == "0xFFFFFFFF" || "$BOOT0" == "0x00000000" ]]; then
    fail "GPU is dead (BOOT0=$BOOT0)"
fi

if [[ "$FECS_CPUCTL" == "0xbadf"* || "$FECS_CPUCTL" == "0xDEADDEAD" ]]; then
    warn "FECS is PRI-faulted — warm state NOT preserved"
    warn "This typically means nouveau teardown reset the engine."
    warn "Check: dmesg | grep livepatch"
else
    ok "GPU appears warm — falcon state preserved!"
fi

echo ""
ok "═══════════════════════════════════════════════"
ok "  Manual warm handoff complete for $BDF"
ok "  Run dispatch test with:"
ok "    CORALREEF_VFIO_BDF=$BDF cargo test ..."
ok "═══════════════════════════════════════════════"
