#!/bin/bash
# DEPRECATED: This script uses raw sysfs writes (driver_override, bind, unbind).
# Caused D-state hang requiring forced power-off (Gap 13). Archived 2026-03-25.
# Exp 089b: Warm Swap Test — Does GPCCS survive nouveau→VFIO transition?
#
# Uses GlowPlug-style swap sequence on Titan #2 (0000:4a:00.0):
# 1. Unbind from current driver
# 2. Capture bare state (no driver, BAR0 accessible)
# 3. Bind to nouveau → ACR bootstraps FECS+GPCCS
# 4. Unbind from nouveau → capture bare warm state
# 5. Bind to vfio-pci → test with coral-driver
#
# Usage: sudo bash scripts/exp089b_warm_swap_test.sh

set -euo pipefail

BDF="0000:4a:00.0"
BDF_AUDIO="0000:4a:00.1"
BAR0="/sys/bus/pci/devices/$BDF/resource0"

echo "=== Exp 089b: Warm Swap GPCCS Survival Test ==="
echo "BDF: $BDF"
echo ""

read_bar0() {
    python3 -c "
import mmap, struct, os, sys
try:
    fd = os.open('$BAR0', os.O_RDONLY | os.O_SYNC)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
    mm.seek($1)
    val = struct.unpack('<I', mm.read(4))[0]
    mm.close()
    os.close(fd)
    print(f'{val:#010x}')
except Exception as e:
    print(f'0xERROR({e.__class__.__name__})')
"
}

capture_state() {
    local label=$1
    echo "  [$label] Capturing falcon state..."
    local fecs_cpuctl=$(read_bar0 $((0x409100)))
    local fecs_pc=$(read_bar0 $((0x409030)))
    local fecs_sctl=$(read_bar0 $((0x409240)))
    local gpccs_cpuctl=$(read_bar0 $((0x41A100)))
    local gpccs_pc=$(read_bar0 $((0x41A030)))
    local gpccs_sctl=$(read_bar0 $((0x41A240)))
    local gpccs_exci=$(read_bar0 $((0x41A148)))
    local sec2_cpuctl=$(read_bar0 $((0x087100)))
    local sec2_pc=$(read_bar0 $((0x087030)))
    local sec2_mb0=$(read_bar0 $((0x087040)))
    local sec2_mb1=$(read_bar0 $((0x087044)))

    echo "  FECS:  cpuctl=$fecs_cpuctl pc=$fecs_pc sctl=$fecs_sctl"
    echo "  GPCCS: cpuctl=$gpccs_cpuctl pc=$gpccs_pc sctl=$gpccs_sctl exci=$gpccs_exci"
    echo "  SEC2:  cpuctl=$sec2_cpuctl pc=$sec2_pc mb0=$sec2_mb0 mb1=$sec2_mb1"
    echo ""
}

current_driver() {
    readlink "/sys/bus/pci/devices/$BDF/driver" 2>/dev/null | xargs basename 2>/dev/null || echo "none"
}

unbind_device() {
    local bdf=$1
    if [ -e "/sys/bus/pci/devices/$bdf/driver/unbind" ]; then
        echo "$bdf" > "/sys/bus/pci/devices/$bdf/driver/unbind" 2>/dev/null || true
    fi
}

echo "Phase 0: Initial driver = $(current_driver)"
echo ""

# --- Step 1: Get to bare state ---
echo "Phase 1: Unbinding all drivers..."
unbind_device "$BDF"
unbind_device "$BDF_AUDIO"
sleep 1
echo "  Driver: $(current_driver)"
echo ""

echo "Phase 1b: Bare cold state (no driver, BAR0 accessible)"
capture_state "BARE-COLD"

# --- Step 2: Bind nouveau, let ACR do its work ---
echo "Phase 2: Binding to nouveau..."
echo "" > /sys/bus/pci/devices/$BDF/driver_override
echo "nouveau" > /sys/bus/pci/devices/$BDF/driver_override
echo "$BDF" > /sys/bus/pci/drivers/nouveau/bind 2>/dev/null || \
    echo "$BDF" > /sys/bus/pci/drivers_probe 2>/dev/null || true
echo "  Waiting 8s for nouveau ACR bootstrap..."
sleep 8
echo "  Driver: $(current_driver)"
echo ""

# BAR0 might not be accessible while nouveau owns it
echo "Phase 3: Nouveau-warm state (may fail if nouveau holds BAR0)"
capture_state "NOUVEAU-WARM"

# --- Step 3: Unbind nouveau → bare warm state ---
echo "Phase 4: Unbinding nouveau..."
unbind_device "$BDF"
sleep 2
echo "  Driver: $(current_driver)"
echo ""

echo "Phase 4b: Bare WARM state — THE CRITICAL MEASUREMENT"
echo "  If GPCCS survived nouveau unbind, PC should be nonzero and sctl should show LS mode"
capture_state "BARE-WARM"

# --- Step 4: Rebind vfio-pci ---
echo "Phase 5: Binding to vfio-pci..."
echo "" > /sys/bus/pci/devices/$BDF/driver_override
echo "vfio-pci" > /sys/bus/pci/devices/$BDF/driver_override
echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || \
    echo "$BDF" > /sys/bus/pci/drivers_probe 2>/dev/null || true
echo "vfio-pci" > /sys/bus/pci/devices/$BDF_AUDIO/driver_override 2>/dev/null || true
echo "$BDF_AUDIO" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
sleep 1
echo "  Driver: $(current_driver)"
echo ""

# BAR0 reads will fail under vfio-pci — skip
echo "Phase 6: VFIO rebound (BAR0 not directly accessible — use coral-driver test)"
echo "  Run: CORALREEF_VFIO_BDF=0000:4a:00.0 cargo test --test hw_nv_vfio -- vfio_sec2_cmdq_probe"
echo ""

echo "=== SUMMARY ==="
echo "Compare BARE-COLD vs BARE-WARM:"
echo "  - If BARE-WARM GPCCS PC != 0x00000000 → GPCCS SURVIVED nouveau unbind"
echo "  - If BARE-WARM GPCCS PC == 0x00000000 → nouveau unbind (or FLR) killed GPCCS"
echo ""
echo "=== End Exp 089b Warm Swap ==="
