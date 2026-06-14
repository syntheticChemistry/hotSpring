#!/bin/bash
# Titan V HBM2 warm-handoff using patched nouveau.ko
#
# This script uses the binary-patched nouveau.ko (with NOP'd teardown
# functions) to warm-handoff the Titan V (GV100). The patched module
# trains HBM2 and when unbound, the NOP'd teardown preserves HBM2
# state and PMC_ENABLE.
#
# IMPORTANT: After the warm-handoff, do NOT restart coral-glowplug.
# The running ember process should re-probe the warm GPU via RPC.
#
# This is the host-side alternative to the benchScale VM approach.
# It requires the host DRM (nvidia-580 on RTX 5060) to be unaffected,
# which it is — nouveau binds only to the Titan V (0000:02:00.0),
# not to the RTX 5060 (0000:21:00.0 uses nvidia driver).

set -euo pipefail

TITANV_BDF="0000:02:00.0"
TITANV_AUDIO_BDF="0000:02:00.1"
PATCHED_MOD="/tmp/nouveau-patch/nouveau_patched.ko"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ ! -f "$PATCHED_MOD" ]; then
    log "ERROR: Patched module not found at $PATCHED_MOD"
    exit 1
fi

log "=== Phase 1: Unbind Titan V from vfio-pci ==="
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

log "=== Phase 2: Load patched nouveau ==="
rmmod nouveau 2>/dev/null || true
insmod "$PATCHED_MOD"
log "  patched nouveau loaded"

log "=== Phase 3: Bind Titan V to nouveau ==="
echo "nouveau" > /sys/bus/pci/devices/$TITANV_BDF/driver_override 2>/dev/null || true
echo "$TITANV_BDF" > /sys/bus/pci/drivers/nouveau/bind 2>/dev/null || {
    log "  bind failed, trying rescan"
    echo 1 > /sys/bus/pci/rescan
}
sleep 8

dmesg | grep -i "nouveau.*$TITANV_BDF\|nouveau.*fb:\|nouveau.*GV100" | tail -10 || true

log "=== Phase 4: Unbind from nouveau (NOP'd teardown) ==="
DRIVER=$(basename "$(readlink /sys/bus/pci/devices/$TITANV_BDF/driver 2>/dev/null)" 2>/dev/null || echo "NONE")
if [ "$DRIVER" = "nouveau" ]; then
    echo "$TITANV_BDF" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
    log "  unbound (teardown NOP'd)"
fi
sleep 1

log "=== Phase 5: Verify warm state ==="
python3 -c "
import mmap, os, struct
fd = os.open('/sys/bus/pci/devices/$TITANV_BDF/resource0', os.O_RDONLY | os.O_SYNC)
with open('/sys/bus/pci/devices/$TITANV_BDF/resource') as rf:
    parts = rf.readline().strip().split()
    bar_size = int(parts[1], 16) - int(parts[0], 16) + 1
mm = mmap.mmap(fd, min(bar_size, 0x2000000), mmap.MAP_SHARED, mmap.PROT_READ)
pmc = struct.unpack_from('<I', mm, 0x200)[0]
pramin = struct.unpack_from('<I', mm, 0x700000)[0]
mm.close()
os.close(fd)
popcount = bin(pmc).count('1')
bad = (pramin & 0xffffff00) == 0xbad0ac00
warm = popcount >= 8 and not bad
print(f'PMC_ENABLE=0x{pmc:08x} (pop={popcount}) PRAMIN=0x{pramin:08x} WARM={warm}')
" 2>/dev/null || echo "snapshot failed"

log "=== Phase 6: Unload nouveau, rebind vfio-pci ==="
rmmod nouveau 2>/dev/null || true
for BDF in "$TITANV_BDF" "$TITANV_AUDIO_BDF"; do
    echo "vfio-pci" > /sys/bus/pci/devices/$BDF/driver_override 2>/dev/null || true
    echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
    log "  $BDF rebound to vfio-pci"
done

log ""
log "=== WARM-HANDOFF COMPLETE ==="
log "DO NOT restart coral-glowplug.service!"
log "Use RPC to verify: ember.sovereign.init on titan-v socket"
