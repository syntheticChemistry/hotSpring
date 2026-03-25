#!/bin/bash
# DEPRECATED: This script uses raw sysfs writes (driver_override, bind, unbind, reset_method).
# Use `coralctl swap` for driver transitions. Archived 2026-03-25 (Gap 13 safety).
# setup_dual_titanv.sh — Configure dual Titan V for cross-testing
#
# Layout after this script:
#   03:00.0  Titan V #1 → nouveau   (oracle, mmiotrace target)
#   21:00.0  RTX 5060   → nvidia    (display, untouched)
#   4a:00.0  Titan V #2 → vfio-pci  (VFIO dispatch target)
#
# Usage: sudo ./scripts/setup_dual_titanv.sh

set -euo pipefail

ORACLE="0000:03:00.0"
ORACLE_AUD="0000:03:00.1"
VFIO_TARGET="0000:4a:00.0"
VFIO_AUD="0000:4a:00.1"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║ Dual Titan V Cross-Test Setup                           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Oracle:  $ORACLE → nouveau                    ║"
echo "║ VFIO:    $VFIO_TARGET → vfio-pci                   ║"
echo "║ Display: 0000:21:00.0 → nvidia (untouched)         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

unbind_device() {
    local dev="$1"
    local curr
    curr=$(basename "$(readlink "/sys/bus/pci/devices/$dev/driver" 2>/dev/null)" 2>/dev/null || echo "none")
    if [ "$curr" != "none" ]; then
        echo "  Unbinding $dev from $curr"
        echo "$dev" > "/sys/bus/pci/drivers/$curr/unbind" 2>/dev/null || true
    fi
    echo "" > "/sys/bus/pci/devices/$dev/driver_override" 2>/dev/null || true
}

# ── Step 1: Load vfio-pci module ──
echo ">>> Step 1: Loading vfio-pci module..."
modprobe vfio-pci 2>/dev/null || true
if [ ! -d /sys/bus/pci/drivers/vfio-pci ]; then
    echo "  ✗ vfio-pci driver not available!"
    exit 1
fi
echo "  ✓ vfio-pci loaded"

# ── Step 2: Unbind both Titan Vs from any current driver ──
echo ">>> Step 2: Unbinding Titan Vs..."
unbind_device "$ORACLE"
unbind_device "$ORACLE_AUD"
unbind_device "$VFIO_TARGET"
unbind_device "$VFIO_AUD"
sleep 1

# ── Step 3: Disable destructive bus reset on VFIO target ──
echo ">>> Step 3: Disabling bus reset on VFIO target..."
echo "" > "/sys/bus/pci/devices/$VFIO_TARGET/reset_method" 2>/dev/null || true
echo "" > "/sys/bus/pci/devices/$ORACLE/reset_method" 2>/dev/null || true
echo "  ✓ Bus reset disabled"

# ── Step 4: Bind oracle to nouveau ──
echo ">>> Step 4: Binding oracle ($ORACLE) to nouveau..."
modprobe nouveau 2>/dev/null || true
echo "" > "/sys/bus/pci/devices/$ORACLE/driver_override"
echo "$ORACLE" > /sys/bus/pci/drivers/nouveau/bind 2>&1
sleep 5

ORACLE_DRV=$(basename "$(readlink "/sys/bus/pci/devices/$ORACLE/driver" 2>/dev/null)" 2>/dev/null || echo "none")
echo "  Oracle driver: $ORACLE_DRV"
if [ "$ORACLE_DRV" != "nouveau" ]; then
    echo "  ⚠ nouveau bind failed, retrying..."
    echo 1 > /sys/bus/pci/rescan
    sleep 2
    echo "$ORACLE" > /sys/bus/pci/drivers/nouveau/bind 2>&1 || true
    sleep 5
    ORACLE_DRV=$(basename "$(readlink "/sys/bus/pci/devices/$ORACLE/driver" 2>/dev/null)" 2>/dev/null || echo "none")
    echo "  Oracle driver (retry): $ORACLE_DRV"
fi

# ── Step 5: Bind VFIO target to vfio-pci ──
echo ">>> Step 5: Binding VFIO target ($VFIO_TARGET) to vfio-pci..."
echo "vfio-pci" > "/sys/bus/pci/devices/$VFIO_TARGET/driver_override"
echo "vfio-pci" > "/sys/bus/pci/devices/$VFIO_AUD/driver_override"
echo "$VFIO_TARGET" > /sys/bus/pci/drivers/vfio-pci/bind 2>&1
echo "$VFIO_AUD" > /sys/bus/pci/drivers/vfio-pci/bind 2>&1 || true
sleep 1

# Prevent runtime PM from putting GPU into D3hot (the "glow plug")
echo "on" > "/sys/bus/pci/devices/$VFIO_TARGET/power/control" 2>/dev/null || true

VFIO_IOMMU_GROUP=$(basename "$(readlink "/sys/bus/pci/devices/$VFIO_TARGET/iommu_group")" 2>/dev/null || echo "?")
chmod 666 "/dev/vfio/$VFIO_IOMMU_GROUP" 2>/dev/null || true

VFIO_DRV=$(basename "$(readlink "/sys/bus/pci/devices/$VFIO_TARGET/driver" 2>/dev/null)" 2>/dev/null || echo "none")
VFIO_PWR=$(cat "/sys/bus/pci/devices/$VFIO_TARGET/power_state" 2>/dev/null || echo "?")
echo "  VFIO driver: $VFIO_DRV  Power: $VFIO_PWR"
echo "  IOMMU group: $VFIO_IOMMU_GROUP"

# ── Step 6: Verify warm state on oracle ──
echo ">>> Step 6: Verifying oracle warm state..."
WARM_CHECK=$(python3 -c "
import mmap, os, struct
try:
    fd = os.open('/sys/bus/pci/devices/${ORACLE}/resource0', os.O_RDONLY)
    m = mmap.mmap(fd, 0x01000000, mmap.MAP_SHARED, mmap.PROT_READ)
    boot0 = struct.unpack_from('<I', m, 0x000)[0]
    pmc = struct.unpack_from('<I', m, 0x200)[0]
    pfifo = struct.unpack_from('<I', m, 0x2200)[0]
    pbdma = struct.unpack_from('<I', m, 0x2004)[0]
    m.close(); os.close(fd)
    warm = pmc != 0x40000020 and pfifo != 0xbad0da00
    print(f'BOOT0={boot0:#010x} PMC={pmc:#010x} PFIFO={pfifo:#010x} PBDMA_MAP={pbdma:#010x} WARM={warm}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
echo "  Oracle: $WARM_CHECK"

# ── Step 7: Verify VFIO target VRAM state ──
echo ">>> Step 7: Checking VFIO target VRAM state (D0 wake)..."
# Force D0 for VRAM check
echo "on" > "/sys/bus/pci/devices/$VFIO_TARGET/power/control" 2>/dev/null || true
sleep 0.5
VFIO_CHECK=$(python3 -c "
import mmap, os, struct
try:
    fd = os.open('/sys/bus/pci/devices/${VFIO_TARGET}/resource0', os.O_RDWR | os.O_SYNC)
    m = mmap.mmap(fd, 0x01000000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    boot0 = struct.unpack_from('<I', m, 0x000)[0]
    pmc = struct.unpack_from('<I', m, 0x200)[0]
    pramin = struct.unpack_from('<I', m, 0x700000)[0]
    # VRAM R/W test
    sentinel = 0xCAFEF00D
    struct.pack_into('<I', m, 0x700000, sentinel)
    rb = struct.unpack_from('<I', m, 0x700000)[0]
    struct.pack_into('<I', m, 0x700000, pramin)  # restore
    vram_ok = rb == sentinel
    m.close(); os.close(fd)
    print(f'BOOT0={boot0:#010x} PMC={pmc:#010x} VRAM={\"ALIVE\" if vram_ok else \"DEAD\"}')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)
echo "  VFIO target: $VFIO_CHECK"

# ── Step 8: Summary ──
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║ Setup Complete                                          ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Oracle ($ORACLE):  $ORACLE_DRV"
echo "║ VFIO   ($VFIO_TARGET):  $VFIO_DRV (group $VFIO_IOMMU_GROUP)"
echo "║ Display (21:00.0):         nvidia"
echo "║ VFIO target state: $VFIO_CHECK"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ IMPORTANT: VFIO close triggers PM reset that kills HBM2 ║"
echo "║ Run all tests in ONE session with --test-threads=1       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║ Test command:                                            ║"
echo "║   CORALREEF_VFIO_BDF=$VFIO_TARGET \\                     ║"
echo "║   CORALREEF_ORACLE_TEXT=data/oracle.txt \\                ║"
echo "║     cargo test --test hw_nv_vfio --features vfio \\      ║"
echo "║     -- --ignored --test-threads=1 --nocapture            ║"
echo "╚══════════════════════════════════════════════════════════╝"
