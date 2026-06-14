#!/usr/bin/env bash
# catalyst-sentinel.sh — lockup forensics for Exp 229 catalyst handoff
#
# v2: REMOVED PCI config space reads (xxd /sys/.../config).
# PCI config reads acquire the global pci_lock in the kernel. During SBR or
# driver transitions, a concurrent config read can enter CRS retry and
# hold pci_lock indefinitely — the SAME deadlock that the keepalive exclusion
# guard prevents. The sentinel itself was a lockup vector.
#
# Now uses ONLY:
#   - BAR0 mmap (no pci_lock, direct MMIO)
#   - sysfs symlink reads (driver binding, iommu)
#   - /proc reads (lsmod, dmesg)
#
# Usage: sudo bash catalyst-sentinel.sh <BDF> &
#        SENTINEL_PID=$!
#        ... trigger handoff ...
#        kill $SENTINEL_PID 2>/dev/null

set -u
BDF="${1:-0000:49:00.0}"
OUTDIR="/var/lib/toadstool/sentinel"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/catalyst-$(date +%Y%m%d-%H%M%S)-${BDF//[:.]/_}.log"

# Flush every write immediately
exec > >(while IFS= read -r line; do echo "$line" >> "$OUTFILE"; done)
exec 2>&1

echo "=== CATALYST SENTINEL START (v2 — no config reads) ==="
echo "BDF=$BDF  PID=$$  $(date -Iseconds)"
echo "kernel=$(uname -r)"
echo ""

# One-shot: PCIe topology snapshot
echo "--- PCIE TOPOLOGY ---"
lspci -tv 2>/dev/null | head -30
echo ""
echo "--- IOMMU GROUP ---"
ls /sys/bus/pci/devices/$BDF/iommu_group/devices/ 2>/dev/null
echo ""

TICK=0
while true; do
    TS=$(date +%H:%M:%S.%3N)
    echo "--- TICK $TICK @ $TS ---"

    # Driver binding (sysfs symlink — no pci_lock)
    DRV=$(readlink /sys/bus/pci/devices/$BDF/driver 2>/dev/null | xargs basename 2>/dev/null || echo "none")
    echo "driver=$DRV"

    # BAR0 reads via mmap (direct MMIO — no pci_lock)
    if [ -r "/sys/bus/pci/devices/$BDF/resource0" ]; then
        BAR0_DIAG=$(python3 -c "
import mmap,os,struct
try:
    fd=os.open('/sys/bus/pci/devices/$BDF/resource0',os.O_RDONLY)
    m=mmap.mmap(fd,4096,mmap.MAP_SHARED,mmap.PROT_READ)
    pmc=struct.unpack_from('<I',m,0x200)[0]
    intr_en=struct.unpack_from('<I',m,0x140)[0]
    boot0=struct.unpack_from('<I',m,0x0)[0]
    m.close();os.close(fd)
    print(f'pmc=0x{pmc:08x} pop={bin(pmc).count(\"1\")} intr_en=0x{intr_en:08x} boot0=0x{boot0:08x}')
except Exception as e: print(f'BAR0_UNREADABLE: {e}')
" 2>/dev/null)
        echo "$BAR0_DIAG"
    else
        echo "bar0=NOREAD"
    fi

    # Loaded kernel modules of interest
    MODS=$(lsmod 2>/dev/null | grep -E "nvsov|no_bus" | awk '{printf "%s(%s) ",$1,$3}')
    echo "mods=${MODS:-none}"

    # Recent kernel messages (last 3 lines)
    KMSGS=$(dmesg --time-format iso 2>/dev/null | grep -iE "nvsov|nvidia.*49:00|pci.*49:00|no_bus|AER|lockup|irq.*nobody|QUENCH" | tail -3)
    if [ -n "$KMSGS" ]; then
        echo "kern: $KMSGS"
    fi

    # Sync to ensure data hits disk before next tick
    sync

    TICK=$((TICK + 1))
    sleep 1
done
