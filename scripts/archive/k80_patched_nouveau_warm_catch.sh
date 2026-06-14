#!/bin/bash
# K80 GDDR5 warm-catch using patched nouveau.ko
#
# Flow:
#   1. Stop glowplug (releases K80 VFIO FDs)
#   2. Unbind K80 from vfio-pci
#   3. Load patched nouveau (teardown NOP'd)
#   4. Bind K80 to nouveau (trains GDDR5)
#   5. Unbind K80 from nouveau (state preserved)
#   6. Rebind K80 to vfio-pci
#   7. Restart glowplug → ember verifies warm GPU

set -euo pipefail

K80_BDF_A="0000:4b:00.0"
K80_BDF_B="0000:4c:00.0"
PATCHED_MOD="/tmp/nouveau-patch/nouveau.ko"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ ! -f "$PATCHED_MOD" ]; then
    log "ERROR: Patched module not found at $PATCHED_MOD"
    log "Run build_patched_nouveau.sh first"
    exit 1
fi

log "=== Phase 1: Stop glowplug service ==="
systemctl stop coral-glowplug.service 2>/dev/null || true
sleep 2
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
insmod "$PATCHED_MOD"
log "  nouveau loaded (patched, teardown NOP'd)"
lsmod | grep nouveau || true

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
sleep 8
dmesg | grep -i "nouveau.*\[drm\]\|nouveau.*fb\|nouveau.*mem\|gk210\|gk110\|tesla" | tail -15 || true
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
    mm.close()
    os.close(fd)
    print(f'pmc_enable=0x{pmc:08x} pramin=0x{pramin:08x}')
except Exception as e:
    print(f'error: {e}')
" 2>/dev/null || echo "snapshot failed")
    log "  $BDF WARM STATE: $PMC_ENABLE"
done

log "=== Phase 7: Unbind K80 from nouveau (NOP'd teardown) ==="
for BDF in "$K80_BDF_A" "$K80_BDF_B"; do
    DRIVER=$(basename "$(readlink /sys/bus/pci/devices/$BDF/driver 2>/dev/null)" 2>/dev/null || echo "NONE")
    if [ "$DRIVER" = "nouveau" ]; then
        echo "$BDF" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
        log "  unbound $BDF from nouveau (teardown NOP'd)"
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
    mm.close()
    os.close(fd)
    print(f'pmc_enable=0x{pmc:08x} pramin=0x{pramin:08x}')
except Exception as e:
    print(f'error: {e}')
" 2>/dev/null || echo "snapshot failed")
    log "  $BDF POST-UNBIND: $POST_STATE"
done

log "=== Phase 9: Unload nouveau, rebind vfio-pci ==="
rmmod nouveau 2>/dev/null || true
for BDF in "$K80_BDF_A" "$K80_BDF_B"; do
    echo "vfio-pci" > /sys/bus/pci/devices/$BDF/driver_override 2>/dev/null || true
    echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
    log "  $BDF rebound to vfio-pci"
done
sleep 1

log "=== Phase 10: Restart glowplug ==="
systemctl start coral-glowplug.service
sleep 5
log "  glowplug restarted"

log ""
log "=== WARM-CATCH COMPLETE ==="
log "Use ember RPC to verify: sovereign.init on K80 socket"
