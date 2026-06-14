#!/bin/bash
# Titan V HBM2/FECS warm-handoff using binary-patched nouveau.ko
#
# Solves GAP-HS-073: FECS requires ACR-authenticated PMU firmware for GV100.
# nouveau handles ACR authentication natively using linux-firmware blobs.
# The binary-patched teardown functions (NOP'd) preserve the warm FECS state
# across unbind→VFIO rebind.
#
# Flow:
#   1. Stop glowplug (releases VFIO FDs)
#   2. Unbind Titan V from vfio-pci
#   3. Load patched nouveau (teardown NOP'd, deps auto-resolved)
#   4. Bind Titan V to nouveau → trains HBM2, loads FECS via ACR
#   5. Snapshot warm state (PMC, PRAMIN, FECS_MC)
#   6. Unbind Titan V from nouveau (state preserved)
#   7. Unload nouveau, rebind vfio-pci
#   8. Verify warm state survived
#
# Requires: patched nouveau.ko in place (patch_nouveau_teardown.py)

set -euo pipefail

TITANV_BDF="0000:02:00.0"
TITANV_AUDIO_BDF="0000:02:00.1"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Titan V Warm-Handoff (patched nouveau, kernel $(uname -r)) ==="

log "=== Phase 1: Stop glowplug ==="
systemctl stop coral-glowplug.service 2>/dev/null || true
sleep 1
pkill -f coral-ember 2>/dev/null || true
sleep 1
log "  glowplug stopped"

log "=== Phase 2: Unbind Titan V from vfio-pci ==="
for BDF in "$TITANV_BDF" "$TITANV_AUDIO_BDF"; do
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

log "=== Phase 4: Bind Titan V to nouveau ==="
DRIVER=$(basename "$(readlink /sys/bus/pci/devices/$TITANV_BDF/driver 2>/dev/null)" 2>/dev/null || echo "NONE")
if [ "$DRIVER" = "nouveau" ]; then
    log "  $TITANV_BDF already bound to nouveau"
elif [ "$DRIVER" = "NONE" ]; then
    echo "nouveau" > /sys/bus/pci/devices/$TITANV_BDF/driver_override 2>/dev/null || true
    echo "$TITANV_BDF" > /sys/bus/pci/drivers/nouveau/bind 2>/dev/null || {
        log "  WARN: explicit bind failed, trying rescan"
        echo 1 > /sys/bus/pci/rescan
        sleep 2
    }
    log "  bound $TITANV_BDF to nouveau"
else
    log "  $TITANV_BDF bound to $DRIVER — unbinding first"
    echo "$TITANV_BDF" > "/sys/bus/pci/drivers/$DRIVER/unbind" 2>/dev/null || true
    echo "nouveau" > /sys/bus/pci/devices/$TITANV_BDF/driver_override 2>/dev/null || true
    echo "$TITANV_BDF" > /sys/bus/pci/drivers/nouveau/bind 2>/dev/null || true
    log "  rebound $TITANV_BDF to nouveau"
fi

log "=== Phase 5: Wait for HBM2 training + FECS init ==="
sleep 12
dmesg | grep -i "nouveau.*$TITANV_BDF\|nouveau.*GV100\|nouveau.*fb:\|nouveau.*FECS\|nouveau.*acr\|nouveau.*sec2\|gv100" | tail -20 || true
log "  init period complete"

log "=== Phase 6: Snapshot warm state ==="
WARM_STATE=$(python3 -c "
import mmap, os, struct
bdf = '$TITANV_BDF'
fd = os.open(f'/sys/bus/pci/devices/{bdf}/resource0', os.O_RDONLY | os.O_SYNC)
with open(f'/sys/bus/pci/devices/{bdf}/resource') as rf:
    parts = rf.readline().strip().split()
    bar_size = int(parts[1], 16) - int(parts[0], 16) + 1
mm = mmap.mmap(fd, min(bar_size, 0x2000000), mmap.MAP_SHARED, mmap.PROT_READ)
pmc_enable = struct.unpack_from('<I', mm, 0x200)[0]
pramin = struct.unpack_from('<I', mm, 0x700000)[0]
fecs_mc = struct.unpack_from('<I', mm, 0x409604)[0]
gpc_mask = struct.unpack_from('<I', mm, 0x41a100)[0]
mm.close()
os.close(fd)
popcount = bin(pmc_enable).count('1')
warm = popcount >= 8 and (pramin & 0xffffff00) != 0xbad0ac00
print(f'PMC=0x{pmc_enable:08x}(pop={popcount}) PRAMIN=0x{pramin:08x} FECS=0x{fecs_mc:08x} GPC=0x{gpc_mask:08x} WARM={warm}')
" 2>/dev/null || echo "snapshot failed")
log "  $TITANV_BDF PRE-UNBIND: $WARM_STATE"

log "=== Phase 7: Unbind Titan V from nouveau (teardown NOP'd) ==="
for BDF in "$TITANV_BDF" "$TITANV_AUDIO_BDF"; do
    DRIVER=$(basename "$(readlink /sys/bus/pci/devices/$BDF/driver 2>/dev/null)" 2>/dev/null || echo "NONE")
    if [ "$DRIVER" = "nouveau" ]; then
        echo "$BDF" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
        log "  unbound $BDF from nouveau"
    fi
done
sleep 2

log "=== Phase 8: Post-unbind state check ==="
POST_STATE=$(python3 -c "
import mmap, os, struct
bdf = '$TITANV_BDF'
fd = os.open(f'/sys/bus/pci/devices/{bdf}/resource0', os.O_RDONLY | os.O_SYNC)
with open(f'/sys/bus/pci/devices/{bdf}/resource') as rf:
    parts = rf.readline().strip().split()
    bar_size = int(parts[1], 16) - int(parts[0], 16) + 1
mm = mmap.mmap(fd, min(bar_size, 0x2000000), mmap.MAP_SHARED, mmap.PROT_READ)
pmc_enable = struct.unpack_from('<I', mm, 0x200)[0]
pramin = struct.unpack_from('<I', mm, 0x700000)[0]
fecs_mc = struct.unpack_from('<I', mm, 0x409604)[0]
gpc_mask = struct.unpack_from('<I', mm, 0x41a100)[0]
mm.close()
os.close(fd)
popcount = bin(pmc_enable).count('1')
warm = popcount >= 8 and (pramin & 0xffffff00) != 0xbad0ac00
print(f'PMC=0x{pmc_enable:08x}(pop={popcount}) PRAMIN=0x{pramin:08x} FECS=0x{fecs_mc:08x} GPC=0x{gpc_mask:08x} WARM={warm}')
" 2>/dev/null || echo "snapshot failed")
log "  $TITANV_BDF POST-UNBIND: $POST_STATE"

log "=== Phase 9: Unload nouveau, rebind vfio-pci ==="
rmmod nouveau 2>/dev/null || true
sleep 1
for BDF in "$TITANV_BDF" "$TITANV_AUDIO_BDF"; do
    echo "vfio-pci" > /sys/bus/pci/devices/$BDF/driver_override 2>/dev/null || true
    echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
    log "  $BDF rebound to vfio-pci"
done
sleep 1

log ""
log "=== TITAN V WARM-HANDOFF COMPLETE ==="
log "Compare PRE vs POST — if PMC/FECS/GPC unchanged, warm state preserved."
log "Next: coralctl sovereign-boot $TITANV_BDF to verify sovereign dispatch."
