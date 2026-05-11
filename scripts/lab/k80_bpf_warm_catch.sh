#!/bin/bash
# K80 GDDR5 warm-catch using binary-patched nouveau.ko
#
# The 4 teardown functions (gf100_gr_fini, nvkm_pmu_fini, nvkm_mc_disable,
# nvkm_fifo_fini) are already NOP'd in the patched module, so no livepatch
# or BPF guard is needed — nouveau simply cannot destroy GPU state on unbind.
#
# Flow:
#   1. Unbind K80 from vfio-pci
#   2. Load patched nouveau (via modprobe — deps resolved automatically)
#   3. K80 binds during PCI rescan → nouveau trains GDDR5, inits GPCs
#   4. Snapshot warm state (PMC_ENABLE, PRAMIN)
#   5. Unbind K80 from nouveau → state preserved (teardown NOP'd)
#   6. Unload nouveau
#   7. Rebind K80 to vfio-pci
#   8. Sovereign boot via ember → warm GPU
#
# Requires: patched nouveau.ko in place (run patch_nouveau_teardown.py first)

set -euo pipefail

K80_BDF_A="0000:4b:00.0"
K80_BDF_B="0000:4c:00.0"
PLX_BDF="0000:4a:00.0"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== K80 Warm-Catch (patched nouveau, kernel $(uname -r)) ==="

log "--- PLX bridge check ---"
PLX_REV=$(setpci -s "$PLX_BDF" 08.b 2>/dev/null || echo "??")
log "  PLX $PLX_BDF rev=$PLX_REV"
if [ "$PLX_REV" = "00" ] || [ "$PLX_REV" = "ff" ]; then
    log "  ERROR: PLX not alive (rev=$PLX_REV). Power cycle required."
    exit 1
fi

log "=== Phase 1: Stop ember/glowplug ==="
systemctl stop coral-glowplug.service 2>/dev/null || true
sleep 1
pkill -f coral-ember 2>/dev/null || true
sleep 1
log "  glowplug stopped"

log "=== Phase 2: Unbind K80 from vfio-pci ==="
for BDF in "$K80_BDF_A" "$K80_BDF_B"; do
    DRIVER=$(basename "$(readlink /sys/bus/pci/devices/$BDF/driver 2>/dev/null)" 2>/dev/null || echo "NONE")
    if [ "$DRIVER" = "vfio-pci" ]; then
        echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/unbind 2>/dev/null || true
        log "  unbound $BDF from vfio-pci"
    else
        log "  $BDF driver: $DRIVER"
    fi
    echo "" > /sys/bus/pci/devices/$BDF/driver_override 2>/dev/null || true
done
sleep 1

log "=== Phase 3: Load patched nouveau ==="
rmmod nouveau 2>/dev/null || true
modprobe nouveau
log "  nouveau loaded (teardown NOP'd)"
lsmod | grep nouveau | head -3

log "=== Phase 4: Bind K80 to nouveau ==="
for BDF in "$K80_BDF_A" "$K80_BDF_B"; do
    DRIVER=$(basename "$(readlink /sys/bus/pci/devices/$BDF/driver 2>/dev/null)" 2>/dev/null || echo "NONE")
    if [ "$DRIVER" = "nouveau" ]; then
        log "  $BDF already bound to nouveau"
    elif [ "$DRIVER" = "NONE" ]; then
        echo "nouveau" > /sys/bus/pci/devices/$BDF/driver_override 2>/dev/null || true
        echo "$BDF" > /sys/bus/pci/drivers/nouveau/bind 2>/dev/null || {
            log "  WARN: explicit bind failed for $BDF, trying rescan"
            echo 1 > /sys/bus/pci/rescan
            sleep 2
        }
        log "  bound $BDF to nouveau"
    else
        log "  $BDF bound to $DRIVER — unbinding first"
        echo "$BDF" > "/sys/bus/pci/drivers/$DRIVER/unbind" 2>/dev/null || true
        echo "nouveau" > /sys/bus/pci/devices/$BDF/driver_override 2>/dev/null || true
        echo "$BDF" > /sys/bus/pci/drivers/nouveau/bind 2>/dev/null || true
        log "  rebound $BDF to nouveau"
    fi
done

log "=== Phase 5: Wait for GDDR5 training ==="
sleep 10
dmesg | grep -i "nouveau.*\[drm\]\|nouveau.*fb\|gk210\|gk110\|tesla\|k80\|GDDR5\|mem_init" | tail -20 || true
log "  training period complete"

log "=== Phase 6: Snapshot warm state ==="
for BDF in "$K80_BDF_A"; do
    PMC_ENABLE=$(python3 -c "
import mmap, os, struct
try:
    fd = os.open('/sys/bus/pci/devices/$BDF/resource0', os.O_RDONLY | os.O_SYNC)
    with open('/sys/bus/pci/devices/$BDF/resource') as rf:
        parts = rf.readline().strip().split()
        bar_size = int(parts[1], 16) - int(parts[0], 16) + 1
    mm = mmap.mmap(fd, min(bar_size, 0x2000000), mmap.MAP_SHARED, mmap.PROT_READ)
    pmc = struct.unpack_from('<I', mm, 0x200)[0]
    pramin = struct.unpack_from('<I', mm, 0x700000)[0]
    gpc_enable = struct.unpack_from('<I', mm, 0x41a10c)[0]
    mm.close()
    os.close(fd)
    popcount = bin(pmc).count('1')
    print(f'PMC=0x{pmc:08x}(pop={popcount}) PRAMIN=0x{pramin:08x} GPC=0x{gpc_enable:08x}')
except Exception as e:
    print(f'error: {e}')
" 2>/dev/null || echo "snapshot failed")
    log "  $BDF PRE-UNBIND: $PMC_ENABLE"
done

log "=== Phase 7: Unbind K80 from nouveau (teardown NOP'd) ==="
for BDF in "$K80_BDF_A" "$K80_BDF_B"; do
    DRIVER=$(basename "$(readlink /sys/bus/pci/devices/$BDF/driver 2>/dev/null)" 2>/dev/null || echo "NONE")
    if [ "$DRIVER" = "nouveau" ]; then
        echo "$BDF" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
        log "  unbound $BDF from nouveau"
    fi
done
sleep 2

log "=== Phase 8: Post-unbind state check ==="
for BDF in "$K80_BDF_A"; do
    POST_STATE=$(python3 -c "
import mmap, os, struct
try:
    fd = os.open('/sys/bus/pci/devices/$BDF/resource0', os.O_RDONLY | os.O_SYNC)
    with open('/sys/bus/pci/devices/$BDF/resource') as rf:
        parts = rf.readline().strip().split()
        bar_size = int(parts[1], 16) - int(parts[0], 16) + 1
    mm = mmap.mmap(fd, min(bar_size, 0x2000000), mmap.MAP_SHARED, mmap.PROT_READ)
    pmc = struct.unpack_from('<I', mm, 0x200)[0]
    pramin = struct.unpack_from('<I', mm, 0x700000)[0]
    gpc_enable = struct.unpack_from('<I', mm, 0x41a10c)[0]
    mm.close()
    os.close(fd)
    popcount = bin(pmc).count('1')
    warm = popcount >= 8 and (pramin & 0xffffff00) != 0xbad0ac00
    print(f'PMC=0x{pmc:08x}(pop={popcount}) PRAMIN=0x{pramin:08x} GPC=0x{gpc_enable:08x} WARM={warm}')
except Exception as e:
    print(f'error: {e}')
" 2>/dev/null || echo "snapshot failed")
    log "  $BDF POST-UNBIND: $POST_STATE"
done

log "=== Phase 9: Unload nouveau, rebind vfio-pci ==="
rmmod nouveau 2>/dev/null || true
sleep 1
for BDF in "$K80_BDF_A" "$K80_BDF_B"; do
    echo "vfio-pci" > /sys/bus/pci/devices/$BDF/driver_override 2>/dev/null || true
    echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
    log "  $BDF rebound to vfio-pci"
done
sleep 1

log ""
log "=== K80 WARM-CATCH COMPLETE ==="
log "PRE vs POST PMC_ENABLE — if unchanged, teardown NOP worked."
log "Next: coralctl sovereign-boot $K80_BDF_A to verify warm state."
